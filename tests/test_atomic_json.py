import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from flightrecorder.atomic_json import (
    AtomicJsonError,
    atomic_write_json_cas,
    json_file_sha256,
)


class AtomicJsonTests(unittest.TestCase):
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
