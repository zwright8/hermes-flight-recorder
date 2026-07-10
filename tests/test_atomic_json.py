import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import flightrecorder.atomic_json as atomic_json_module
from flightrecorder.atomic_json import (
    AtomicJsonError,
    atomic_write_json_cas,
    json_file_sha256,
)


class AtomicJsonTests(unittest.TestCase):
    @unittest.skipIf(os.name == "nt", "POSIX umask semantics")
    def test_atomic_write_can_match_normal_json_creation_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"
            previous_umask = os.umask(0o022)
            try:
                first_sha = atomic_write_json_cas(
                    path,
                    {"version": 1},
                    expected_sha256=None,
                    new_file_mode=0o666,
                )
            finally:
                os.umask(previous_umask)

            self.assertEqual(path.stat().st_mode & 0o777, 0o644)
            path.chmod(0o600)
            atomic_write_json_cas(
                path,
                {"version": 2},
                expected_sha256=first_sha,
                new_file_mode=0o666,
            )
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_atomic_write_creates_and_replaces_exact_expected_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"

            first_sha = atomic_write_json_cas(path, {"version": 1}, expected_sha256=None)
            second_sha = atomic_write_json_cas(
                path,
                {"version": 2},
                expected_sha256=first_sha,
            )

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"version": 2})
            self.assertEqual(second_sha, json_file_sha256(path))
            self.assertTrue((path.parent / ".registry.json.hfr.lock").is_file())
            self.assertEqual(list(path.parent.glob(".registry.json.*.tmp")), [])

    def test_atomic_write_rejects_stale_compare_and_swap_without_losing_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"
            path.write_text('{"version": 1}\n', encoding="utf-8")
            stale_sha = json_file_sha256(path)
            path.write_text('{"version": 2}\n', encoding="utf-8")

            with self.assertRaisesRegex(AtomicJsonError, "changed concurrently"):
                atomic_write_json_cas(path, {"version": 3}, expected_sha256=stale_sha)

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"version": 2})

    def test_atomic_write_rechecks_target_after_temporary_file_is_durable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"
            path.write_text('{"version": 1}\n', encoding="utf-8")
            expected = json_file_sha256(path)
            competing_bytes = b'{"version": 2}\n'
            actual_write_temporary_file = atomic_json_module._write_temporary_file

            def compete_after_temporary_write(target, rendered):
                replacement = actual_write_temporary_file(target, rendered)
                path.write_bytes(competing_bytes)
                return replacement

            with patch(
                "flightrecorder.atomic_json._write_temporary_file",
                side_effect=compete_after_temporary_write,
            ):
                with self.assertRaisesRegex(AtomicJsonError, "changed concurrently"):
                    atomic_write_json_cas(path, {"version": 3}, expected_sha256=expected)

            self.assertEqual(path.read_bytes(), competing_bytes)
            self.assertEqual(list(path.parent.glob(".registry.json.*.tmp")), [])

    def test_atomic_write_rejects_symlinked_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real = root / "real.json"
            link = root / "registry.json"
            real.write_text("{}\n", encoding="utf-8")
            try:
                link.symlink_to(real)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlinks unavailable: {exc}")

            with self.assertRaisesRegex(AtomicJsonError, "symlink components"):
                json_file_sha256(link)

    def test_atomic_write_rejects_symlinked_parent_before_creating_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real_parent = root / "real-parent"
            linked_parent = root / "linked-parent"
            real_parent.mkdir()
            try:
                linked_parent.symlink_to(real_parent, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlinks unavailable: {exc}")
            nested_parent = real_parent / "new" / "nested"
            target = linked_parent / "new" / "nested" / "registry.json"

            with self.assertRaisesRegex(AtomicJsonError, "symlink components"):
                atomic_write_json_cas(target, {"version": 1}, expected_sha256=None)

            self.assertFalse(nested_parent.exists())

    def test_failed_replace_preserves_original_and_cleans_temporary_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"
            path.write_text('{"version": 1}\n', encoding="utf-8")
            expected = json_file_sha256(path)

            with patch("flightrecorder.atomic_json.os.replace", side_effect=OSError("replace failed")):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    atomic_write_json_cas(path, {"version": 2}, expected_sha256=expected)

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"version": 1})
            self.assertEqual(list(path.parent.glob(".registry.json.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
