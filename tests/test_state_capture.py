import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from flightrecorder.schema_registry import check_schema_contract
from flightrecorder.state_capture import StateCaptureError, capture_state_snapshot


class StateCaptureTests(unittest.TestCase):
    def test_capture_file_text_reads_only_limit_plus_sentinel(self):
        class ReadProbe(io.StringIO):
            def __init__(self, value: str):
                super().__init__(value)
                self.read_sizes: list[int] = []

            def read(self, size: int = -1) -> str:
                self.read_sizes.append(size)
                return super().read(size)

        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "artifact.txt"
            artifact.write_text("abcdefghi", encoding="utf-8")
            reader = ReadProbe("abcdefghi")

            with (
                patch("flightrecorder.state_capture._sha256", return_value="0" * 64),
                patch.object(Path, "open", return_value=reader),
            ):
                snapshot = capture_state_snapshot(
                    files=[("artifact", artifact)],
                    include_file_text=True,
                    max_text_chars=5,
                )

            record = snapshot["filesystem"]["files"]["artifact"]
            self.assertEqual(reader.read_sizes, [6])
            self.assertEqual(record["text"], "abcde")
            self.assertTrue(record["text_truncated"])

    def test_capture_directory_bounds_name_scans_and_records_lower_bound(self):
        class ScandirEntry:
            def __init__(self, name: str):
                self.name = name

        class ScandirProbe:
            def __init__(self, values: list[str]):
                self._values = iter(values)
                self.next_calls = 0
                self.closed = False

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.closed = True

            def __iter__(self):
                return self

            def __next__(self) -> ScandirEntry:
                self.next_calls += 1
                return ScandirEntry(next(self._values))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries = [root / name for name in ("b.txt", "a.txt", "c.txt", "d.txt")]
            for entry in entries:
                entry.write_text(entry.name, encoding="utf-8")
            iterator = ScandirProbe([entry.name for entry in entries])
            hashed_names: list[str] = []

            with (
                patch.object(os, "scandir", return_value=iterator),
                patch(
                    "flightrecorder.state_capture._sha256",
                    side_effect=lambda path: hashed_names.append(path.name) or "0" * 64,
                ),
            ):
                snapshot = capture_state_snapshot(
                    directories=[("workspace", root)],
                    max_dir_entries=2,
                )

            record = snapshot["filesystem"]["directories"]["workspace"]
            self.assertEqual(iterator.next_calls, 3)
            self.assertTrue(iterator.closed)
            self.assertEqual(record["entry_count"], 3)
            self.assertEqual(record["scanned_entry_count"], 3)
            self.assertEqual(record["scan_limit"], 3)
            self.assertTrue(record["scan_incomplete"])
            self.assertTrue(record["entry_count_is_lower_bound"])
            self.assertEqual(record["entry_selection"], "lexicographic_within_scanned_prefix")
            self.assertTrue(record["entries_truncated"])
            self.assertEqual(
                [entry["name"] for entry in record["entries"]], ["a.txt", "b.txt"]
            )
            self.assertEqual(hashed_names, ["a.txt", "b.txt"])

    def test_capture_directory_sorts_only_the_bounded_scanned_prefix(self):
        class ScandirEntry:
            def __init__(self, name: str):
                self.name = name

        class ScandirResult:
            def __init__(self, names: list[str]):
                self._entries = iter(ScandirEntry(name) for name in names)

            def __enter__(self):
                return self._entries

            def __exit__(self, *_args):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = ["z.txt", "y.txt", "b.txt", "a.txt"]
            for name in names:
                (root / name).write_text(name, encoding="utf-8")

            with patch.object(os, "scandir", return_value=ScandirResult(names)):
                snapshot = capture_state_snapshot(
                    directories=[("workspace", root)],
                    max_dir_entries=2,
                )

            record = snapshot["filesystem"]["directories"]["workspace"]
            self.assertEqual([entry["name"] for entry in record["entries"]], ["b.txt", "y.txt"])
            self.assertNotIn("a.txt", [entry["name"] for entry in record["entries"]])
            self.assertTrue(record["scan_incomplete"])
            self.assertEqual(record["entry_selection"], "lexicographic_within_scanned_prefix")

    def test_capture_directory_never_follows_or_hashes_symlink_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "workspace"
            root.mkdir()
            target_file = base / "target.txt"
            target_file.write_text("private target", encoding="utf-8")
            target_directory = base / "target-directory"
            target_directory.mkdir()
            (root / "file-link").symlink_to(target_file)
            (root / "directory-link").symlink_to(target_directory, target_is_directory=True)
            hashed_names: list[str] = []

            with patch(
                "flightrecorder.state_capture._sha256",
                side_effect=lambda path: hashed_names.append(path.name) or "0" * 64,
            ):
                snapshot = capture_state_snapshot(
                    directories=[("workspace", root)],
                    max_dir_entries=10,
                )

            entries = {
                entry["name"]: entry
                for entry in snapshot["filesystem"]["directories"]["workspace"]["entries"]
            }
            self.assertEqual(entries["file-link"], {"name": "file-link", "kind": "other"})
            self.assertEqual(entries["directory-link"], {"name": "directory-link", "kind": "other"})
            self.assertNotIn("file-link", hashed_names)
            self.assertNotIn("directory-link", hashed_names)

    def test_capture_rejects_symlink_directory_source_before_scanning_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            external_directory = base / "external"
            external_directory.mkdir()
            (external_directory / "private.txt").write_text("private target", encoding="utf-8")
            source_link = base / "workspace-link"
            source_link.symlink_to(external_directory, target_is_directory=True)

            with (
                patch.object(os, "scandir", side_effect=AssertionError("target must not be scanned")),
                self.assertRaisesRegex(StateCaptureError, "symlink component"),
            ):
                capture_state_snapshot(directories=[("workspace", source_link)])

    def test_capture_rejects_symlinked_parent_component_before_scanning_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            external_directory = base / "external"
            nested = external_directory / "nested"
            nested.mkdir(parents=True)
            parent_link = base / "workspace-link"
            parent_link.symlink_to(external_directory, target_is_directory=True)

            with (
                patch.object(os, "scandir", side_effect=AssertionError("target must not be scanned")),
                self.assertRaisesRegex(StateCaptureError, "symlink component"),
            ):
                capture_state_snapshot(directories=[("workspace", parent_link / "nested")])

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
