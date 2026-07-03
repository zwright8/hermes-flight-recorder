import json
import tempfile
import unittest
from pathlib import Path

from flightrecorder.schema_registry import check_schema_contract
from flightrecorder.state_capture import StateCaptureError, capture_state_snapshot


class StateCaptureTests(unittest.TestCase):
    def test_capture_files_json_directories_and_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "reply.txt"
            artifact.write_text("sent reply with token=state-capture-secret", encoding="utf-8")
            payload = root / "mail.json"
            payload.write_text(json.dumps({"thread_id": "email-123", "status": "sent"}), encoding="utf-8")

            snapshot = capture_state_snapshot(
                files=[("reply", artifact)],
                directories=[("workspace", root)],
                json_sources=[("mail", payload)],
                observations=[
                    ("gmail.threads.email-123.sent_replies.0.status", "sent"),
                    ("gmail.threads.email-123.sent_replies.0.message_id", "msg-email-123-001"),
                ],
                include_file_text=True,
                secret_patterns=["state-capture-secret"],
            )

            self.assertEqual(snapshot["schema_version"], "hfr.state_snapshot.v1")
            self.assertTrue(snapshot["filesystem"]["files"]["reply"]["exists"])
            self.assertEqual(snapshot["filesystem"]["files"]["reply"]["kind"], "file")
            self.assertEqual(len(snapshot["filesystem"]["files"]["reply"]["sha256"]), 64)
            self.assertIn("[REDACTED]", snapshot["filesystem"]["files"]["reply"]["text"])
            self.assertEqual(snapshot["json"]["mail"]["status"], "sent")
            self.assertEqual(snapshot["filesystem"]["directories"]["workspace"]["entry_count"], 2)
            workspace_entries = snapshot["filesystem"]["directories"]["workspace"]["entries"]
            file_entry = next(entry for entry in workspace_entries if entry["name"] == "reply.txt")
            self.assertEqual(file_entry["kind"], "file")
            self.assertEqual(len(file_entry["sha256"]), 64)
            self.assertEqual(
                snapshot["observations"]["gmail"]["threads"]["email-123"]["sent_replies"][0]["status"],
                "sent",
            )
            schema = check_schema_contract(snapshot)
            self.assertTrue(schema["passed"], schema["errors"])
            self.assertEqual(schema["schema"]["name"], "state_snapshot")
            bad_snapshot = json.loads(json.dumps(snapshot))
            bad_snapshot["filesystem"]["files"]["reply"].pop("size_bytes")
            bad_schema = check_schema_contract(bad_snapshot)
            self.assertFalse(bad_schema["passed"])
            self.assertIn("expected exactly one matching schema from oneOf, got 0", "\n".join(bad_schema["errors"]))
            bad_hash_snapshot = json.loads(json.dumps(snapshot))
            bad_hash_snapshot["filesystem"]["files"]["reply"].pop("sha256")
            bad_hash_schema = check_schema_contract(bad_hash_snapshot)
            self.assertFalse(bad_hash_schema["passed"])
            self.assertIn("expected exactly one matching schema from oneOf, got 0", "\n".join(bad_hash_schema["errors"]))
            bad_entry_snapshot = json.loads(json.dumps(snapshot))
            bad_entry = next(
                entry
                for entry in bad_entry_snapshot["filesystem"]["directories"]["workspace"]["entries"]
                if entry["name"] == "reply.txt"
            )
            bad_entry.pop("sha256")
            bad_entry_schema = check_schema_contract(bad_entry_snapshot)
            self.assertFalse(bad_entry_schema["passed"])
            self.assertIn("expected exactly one matching schema from oneOf, got 0", "\n".join(bad_entry_schema["errors"]))

            nested = root / "nested"
            nested.mkdir()
            directory_entry_snapshot = capture_state_snapshot(directories=[("workspace", root)])
            directory_entry_schema = check_schema_contract(directory_entry_snapshot)
            self.assertTrue(directory_entry_schema["passed"], directory_entry_schema["errors"])
            nested_entry = next(
                entry
                for entry in directory_entry_snapshot["filesystem"]["directories"]["workspace"]["entries"]
                if entry["name"] == "nested"
            )
            self.assertEqual(nested_entry["kind"], "directory")
            self.assertNotIn("sha256", nested_entry)
            for non_file_kind in ("directory", "other", "missing"):
                bad_non_file_entry_snapshot = json.loads(json.dumps(directory_entry_snapshot))
                bad_non_file_entry_snapshot["filesystem"]["directories"]["workspace"]["entries"] = [
                    {
                        "name": f"{non_file_kind}-entry",
                        "kind": non_file_kind,
                        "sha256": "0" * 64,
                        "size_bytes": 1,
                    }
                ]
                bad_non_file_entry_schema = check_schema_contract(bad_non_file_entry_snapshot)
                self.assertFalse(bad_non_file_entry_schema["passed"])
                self.assertIn(
                    "expected exactly one matching schema from oneOf, got 0",
                    "\n".join(bad_non_file_entry_schema["errors"]),
                )

            diagnostic_snapshot = capture_state_snapshot(
                files=[
                    ("missing_reply", root / "missing-reply.txt"),
                    ("workspace_as_file", root),
                ]
            )
            diagnostic_schema = check_schema_contract(diagnostic_snapshot)
            self.assertTrue(diagnostic_schema["passed"], diagnostic_schema["errors"])
            self.assertEqual(diagnostic_snapshot["filesystem"]["files"]["missing_reply"]["kind"], "missing")
            self.assertEqual(diagnostic_snapshot["filesystem"]["files"]["workspace_as_file"]["kind"], "directory")
            bad_diagnostic_snapshot = json.loads(json.dumps(diagnostic_snapshot))
            bad_diagnostic_snapshot["filesystem"]["files"]["workspace_as_file"]["exists"] = False
            bad_diagnostic_schema = check_schema_contract(bad_diagnostic_snapshot)
            self.assertFalse(bad_diagnostic_schema["passed"])
            self.assertIn("expected exactly one matching schema from oneOf, got 0", "\n".join(bad_diagnostic_schema["errors"]))

    def test_capture_rejects_bad_source_key(self):
        with self.assertRaisesRegex(StateCaptureError, "Snapshot source key"):
            capture_state_snapshot(files=[("bad.key", "artifact.txt")])

    def test_capture_rejects_conflicting_observation_paths(self):
        with self.assertRaisesRegex(StateCaptureError, "conflicts"):
            capture_state_snapshot(
                observations=[
                    ("gmail.threads.email-123", "already scalar"),
                    ("gmail.threads.email-123.sent_replies.0.status", "sent"),
                ]
            )
