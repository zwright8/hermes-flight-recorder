import json
import tempfile
import unittest
from pathlib import Path

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
            self.assertEqual(
                snapshot["observations"]["gmail"]["threads"]["email-123"]["sent_replies"][0]["status"],
                "sent",
            )

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
