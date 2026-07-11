from __future__ import annotations

import ast
import errno
import hashlib
import os
import unittest
from contextlib import chdir
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import flightrecorder.path_safety as path_safety_module
from flightrecorder.path_safety import (
    DIRECTORY_CONTENT_HASH_ALGORITHM,
    DirectoryCleanupStatus,
    assert_safe_output_directory,
    attest_directory_namespace,
    fingerprint_directory_contents,
    fingerprint_directory_namespace,
    json_marker_matches_schema,
    locked_owned_output_directory,
    opened_directory_descriptor,
    path_has_symlink_component,
    remove_directory_entry_tree_if_identity,
    replace_owned_output_directory,
)


ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCRIPTS = (
    ROOT / "scripts" / "external_verification_smoke.py",
    ROOT / "scripts" / "live_verifier_smoke.py",
    ROOT / "scripts" / "live_hermes_smoke.py",
    ROOT / "scripts" / "live_openclaw_smoke.py",
    ROOT / "scripts" / "live_coven_smoke.py",
)


class PathHasSymlinkComponentTests(unittest.TestCase):
    def test_invalid_path_is_treated_as_unsafe(self) -> None:
        self.assertTrue(
            path_has_symlink_component(Path("bad\x00path"), include_leaf=True)
        )

    def test_root_symlink_exception_is_limited_to_known_macos_system_paths(self) -> None:
        with (
            patch.object(path_safety_module.sys, "platform", "darwin"),
            patch.object(Path, "resolve", return_value=Path("/private/var")),
        ):
            self.assertTrue(
                path_safety_module._is_allowed_system_root_symlink(Path("/var"))
            )
            self.assertFalse(
                path_safety_module._is_allowed_system_root_symlink(Path("/untrusted"))
            )
        with patch.object(path_safety_module.sys, "platform", "win32"):
            self.assertFalse(
                path_safety_module._is_allowed_system_root_symlink(Path("/var"))
            )

    def test_relative_parent_walk_detects_sibling_symlink(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            working_dir = root / "working"
            target_dir = root / "target"
            working_dir.mkdir()
            target_dir.mkdir()
            sibling_link = root / "sibling-link"
            try:
                sibling_link.symlink_to(target_dir, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")

            with chdir(working_dir):
                candidate = Path("../sibling-link/artifact.json")
                self.assertTrue(path_has_symlink_component(candidate, include_leaf=False))


class DirectoryFingerprintTests(unittest.TestCase):
    def test_unrelated_ancestor_metadata_changes_do_not_invalidate_bound_tree(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ancestor = root / "ancestor"
            tree = ancestor / "tree"
            tree.mkdir(parents=True)
            (tree / "data.json").write_text("{}\n", encoding="utf-8")
            actual_hash = path_safety_module._sha256_tree_file_fd
            changed = False

            def change_unrelated_sibling(*args, **kwargs):
                nonlocal changed
                if not changed:
                    (ancestor / "unrelated-spool").mkdir()
                    changed = True
                return actual_hash(*args, **kwargs)

            with patch.object(
                path_safety_module,
                "_sha256_tree_file_fd",
                side_effect=change_unrelated_sibling,
            ):
                result = fingerprint_directory_contents(tree)

            self.assertEqual(result["file_count"], 1)
            self.assertTrue(changed)

    def test_generic_tree_digest_preserves_portable_v1_hash_semantics(self) -> None:
        with TemporaryDirectory() as tmp:
            tree = Path(tmp) / "tree"
            nested = tree / "nested"
            nested.mkdir(parents=True)
            contents = {
                "alpha.txt": b"alpha\n",
                "nested/beta.txt": b"beta\n",
            }
            for relative, content in contents.items():
                (tree / relative).write_bytes(content)

            expected_digest = hashlib.sha256()
            for relative, content in sorted(contents.items()):
                expected_digest.update(relative.encode("utf-8"))
                expected_digest.update(b"\0")
                expected_digest.update(str(len(content)).encode("ascii"))
                expected_digest.update(b"\0")
                expected_digest.update(hashlib.sha256(content).hexdigest().encode("ascii"))
                expected_digest.update(b"\0")

            self.assertEqual(
                fingerprint_directory_contents(tree),
                {
                    "tree_hash_algorithm": DIRECTORY_CONTENT_HASH_ALGORITHM,
                    "sha256": expected_digest.hexdigest(),
                    "file_count": 2,
                    "size_bytes": sum(map(len, contents.values())),
                    "contains_symlinks": False,
                },
            )

    def test_local_namespace_digest_binds_entries_ignored_by_portable_digest(self) -> None:
        with TemporaryDirectory() as tmp:
            tree = Path(tmp) / "tree"
            tree.mkdir()
            (tree / "payload.txt").write_text("payload\n", encoding="utf-8")
            portable_before = fingerprint_directory_contents(tree)
            namespace_before = fingerprint_directory_namespace(tree)

            (tree / "empty").mkdir()

            self.assertEqual(fingerprint_directory_contents(tree), portable_before)
            namespace_after = fingerprint_directory_namespace(tree)
            self.assertNotEqual(namespace_after["sha256"], namespace_before["sha256"])
            self.assertEqual(namespace_after["entry_count"], 2)

    def test_portable_digest_can_return_a_selected_file_hash_from_same_pass(self) -> None:
        with TemporaryDirectory() as tmp:
            tree = Path(tmp) / "tree"
            tree.mkdir()
            manifest = b'{"schema_version":"example.v1"}\n'
            (tree / "manifest.json").write_bytes(manifest)
            (tree / "payload.jsonl").write_text("{}\n", encoding="utf-8")

            default = fingerprint_directory_contents(tree)
            selected = fingerprint_directory_contents(
                tree,
                selected_files=("manifest.json",),
            )

            self.assertEqual(
                selected["selected_file_sha256"],
                {"manifest.json": hashlib.sha256(manifest).hexdigest()},
            )
            selected_without_hashes = dict(selected)
            selected_without_hashes.pop("selected_file_sha256")
            self.assertEqual(selected_without_hashes, default)

            with self.assertRaisesRegex(ValueError, "outside the fingerprinted file set"):
                fingerprint_directory_contents(
                    tree,
                    relative_files=("payload.jsonl",),
                    selected_files=("manifest.json",),
                )

    def test_local_namespace_digest_records_symlinks_and_special_entries(self) -> None:
        with TemporaryDirectory() as tmp:
            tree = Path(tmp) / "tree"
            tree.mkdir()
            first = tree / "first.txt"
            second = tree / "second.txt"
            first.write_text("first\n", encoding="utf-8")
            second.write_text("second\n", encoding="utf-8")
            link = tree / "selected"
            try:
                link.symlink_to(first.name)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")

            first_attestation = fingerprint_directory_namespace(tree)
            self.assertTrue(first_attestation["contains_symlinks"])
            link.unlink()
            link.symlink_to(second.name)
            second_attestation = fingerprint_directory_namespace(tree)
            self.assertNotEqual(
                second_attestation["sha256"],
                first_attestation["sha256"],
            )

            if hasattr(os, "mkfifo"):
                fifo = tree / "named-pipe"
                try:
                    os.mkfifo(fifo)
                except OSError:
                    pass
                else:
                    special_attestation = fingerprint_directory_namespace(tree)
                    self.assertTrue(special_attestation["contains_special_entries"])
                    self.assertNotEqual(
                        special_attestation["sha256"],
                        second_attestation["sha256"],
                    )

    def test_declared_tree_rejects_missing_and_undeclared_entries(self) -> None:
        with TemporaryDirectory() as tmp:
            tree = Path(tmp) / "tree"
            tree.mkdir()
            (tree / "declared.txt").write_text("declared\n", encoding="utf-8")
            (tree / "extra.txt").write_text("extra\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "undeclared entries"):
                fingerprint_directory_contents(
                    tree,
                    relative_files=("declared.txt",),
                    reject_undeclared=True,
                )

            (tree / "extra.txt").unlink()
            with self.assertRaisesRegex(ValueError, "missing declared entries"):
                fingerprint_directory_contents(
                    tree,
                    relative_files=("declared.txt", "missing.txt"),
                    reject_undeclared=True,
                )

    def test_declared_tree_rejects_non_normalized_paths(self) -> None:
        with TemporaryDirectory() as tmp:
            tree = Path(tmp) / "tree"
            tree.mkdir()

            for relative in ("../outside.txt", "./inside.txt", "/absolute.txt"):
                with self.subTest(relative=relative):
                    with self.assertRaisesRegex(ValueError, "normalized and relative"):
                        fingerprint_directory_contents(
                            tree,
                            relative_files=(relative,),
                            reject_undeclared=True,
                        )

    def test_generic_tree_rejects_symlinks(self) -> None:
        with TemporaryDirectory() as tmp:
            tree = Path(tmp) / "tree"
            tree.mkdir()
            target = tree / "target.txt"
            target.write_text("target\n", encoding="utf-8")
            link = tree / "linked.txt"
            try:
                link.symlink_to(target)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")

            with self.assertRaisesRegex(ValueError, "contains a symlink"):
                fingerprint_directory_contents(tree)

    @unittest.skipIf(os.name == "nt", "backslash is a path separator on Windows")
    def test_generic_tree_rejects_non_portable_entry_names(self) -> None:
        with TemporaryDirectory() as tmp:
            tree = Path(tmp) / "tree"
            tree.mkdir()
            (tree / "not\\portable.txt").write_text("content\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "non-portable entry name"):
                fingerprint_directory_contents(tree)

    def test_generic_tree_rejects_special_files(self) -> None:
        if not hasattr(os, "mkfifo"):
            self.skipTest("FIFO creation is unavailable")
        with TemporaryDirectory() as tmp:
            tree = Path(tmp) / "tree"
            tree.mkdir()
            fifo = tree / "named-pipe"
            try:
                os.mkfifo(fifo)
            except OSError as exc:
                self.skipTest(f"FIFO creation unavailable: {exc}")

            with self.assertRaisesRegex(ValueError, "non-regular path"):
                fingerprint_directory_contents(tree)

    def test_platform_without_nofollow_support_fails_closed(self) -> None:
        with TemporaryDirectory() as tmp:
            tree = Path(tmp) / "tree"
            tree.mkdir()

            with (
                patch.object(path_safety_module.os, "O_NOFOLLOW", 0),
                self.assertRaisesRegex(ValueError, "unavailable on this platform"),
            ):
                fingerprint_directory_contents(tree)

    def test_pathname_reopen_swap_is_rejected_before_hashing(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            tree = root / "tree"
            tree.mkdir()
            payload = tree / "payload.txt"
            payload.write_text("admitted\n", encoding="utf-8")
            replacement = tree / "replacement.tmp"
            replacement.write_text("replacement\n", encoding="utf-8")
            displaced = root / "displaced.txt"
            original_open = path_safety_module._open_tree_file_fd
            swapped = False

            def swap_before_open(directory_descriptor: int, name: str) -> int:
                nonlocal swapped
                if name == "payload.txt" and not swapped:
                    swapped = True
                    payload.rename(displaced)
                    replacement.rename(payload)
                return original_open(directory_descriptor, name)

            with (
                patch.object(
                    path_safety_module,
                    "_open_tree_file_fd",
                    side_effect=swap_before_open,
                ),
                self.assertRaisesRegex(ValueError, "changed while being admitted"),
            ):
                fingerprint_directory_contents(
                    tree,
                    relative_files=("payload.txt", "replacement.tmp"),
                    reject_undeclared=True,
                )

            self.assertTrue(swapped)
            self.assertEqual(payload.read_text(encoding="utf-8"), "replacement\n")
            self.assertEqual(displaced.read_text(encoding="utf-8"), "admitted\n")

    def test_in_place_change_during_descriptor_read_is_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            tree = Path(tmp) / "tree"
            tree.mkdir()
            payload = tree / "payload.txt"
            payload.write_text("before\n", encoding="utf-8")
            original_hash = path_safety_module._sha256_descriptor

            def mutate_after_read(descriptor: int) -> str:
                result = original_hash(descriptor)
                payload.write_text("after!\n", encoding="utf-8")
                return result

            with (
                patch.object(
                    path_safety_module,
                    "_sha256_descriptor",
                    side_effect=mutate_after_read,
                ),
                self.assertRaisesRegex(ValueError, "changed while being fingerprinted"),
            ):
                fingerprint_directory_contents(tree)


class AtomicNamespacePublicationTests(unittest.TestCase):
    def test_attested_remover_preflights_complete_tree_before_any_deletion(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "approved"
            empty = target / "approved-empty"
            empty.mkdir(parents=True)
            approved = target / "approved.txt"
            approved.write_text("approved\n", encoding="utf-8")
            expected = attest_directory_namespace(target)
            target_stat = target.stat(follow_symlinks=False)
            concurrent = target / "concurrent-unowned.txt"
            concurrent.write_text("must survive\n", encoding="utf-8")

            with opened_directory_descriptor(root) as parent_descriptor:
                removed = remove_directory_entry_tree_if_identity(
                    parent_descriptor,
                    target.name,
                    (target_stat.st_dev, target_stat.st_ino),
                    expected,
                )

            self.assertFalse(removed)
            self.assertEqual(approved.read_text(encoding="utf-8"), "approved\n")
            self.assertEqual(concurrent.read_text(encoding="utf-8"), "must survive\n")
            self.assertTrue(empty.is_dir())

    def test_attested_remover_deletes_matching_files_and_empty_directories(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "approved"
            (target / "nested" / "approved-empty").mkdir(parents=True)
            (target / "approved.txt").write_text("approved\n", encoding="utf-8")
            (target / "nested" / "approved.txt").write_text(
                "nested approved\n",
                encoding="utf-8",
            )
            expected = attest_directory_namespace(target)
            target_stat = target.stat(follow_symlinks=False)

            with opened_directory_descriptor(root) as parent_descriptor:
                removed = remove_directory_entry_tree_if_identity(
                    parent_descriptor,
                    target.name,
                    (target_stat.st_dev, target_stat.st_ino),
                    expected,
                )

            self.assertTrue(removed)
            self.assertEqual(removed.status, DirectoryCleanupStatus.COMPLETE)
            self.assertFalse(target.exists())

    def test_attested_remover_reports_partial_cleanup_after_second_unlink_eio(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "approved"
            target.mkdir()
            (target / "first.txt").write_text("first\n", encoding="utf-8")
            (target / "second.txt").write_text("second\n", encoding="utf-8")
            expected = attest_directory_namespace(target)
            target_stat = target.stat(follow_symlinks=False)
            actual_unlink = os.unlink
            unlink_count = 0

            def fail_second_private_unlink(name, *, dir_fd=None):
                nonlocal unlink_count
                if str(name).startswith(".hfr-remove-"):
                    unlink_count += 1
                    if unlink_count == 2:
                        raise OSError(errno.EIO, "injected second unlink failure")
                return actual_unlink(name, dir_fd=dir_fd)

            with opened_directory_descriptor(root) as parent_descriptor:
                with patch.object(
                    path_safety_module.os,
                    "unlink",
                    side_effect=fail_second_private_unlink,
                ):
                    outcome = remove_directory_entry_tree_if_identity(
                        parent_descriptor,
                        target.name,
                        (target_stat.st_dev, target_stat.st_ino),
                        expected,
                    )

            self.assertFalse(outcome)
            self.assertEqual(outcome.status, DirectoryCleanupStatus.PARTIAL)
            self.assertEqual(outcome.removed_entry_count, 1)
            self.assertFalse(outcome.durability_confirmed)
            self.assertIn("durability", outcome.detail or "")
            self.assertEqual(outcome.recovery_entries, ("approved",))
            self.assertEqual(outcome.concurrent_entries, ())
            self.assertEqual(outcome.cleanup_artifacts, ())
            self.assertEqual(
                [entry.name for entry in target.iterdir()],
                ["second.txt"],
            )
            self.assertFalse(list(root.glob(".hfr-remove-*")))

    def test_attested_remover_marks_nested_partial_cleanup_durability_unconfirmed(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "approved"
            nested = target / "nested"
            nested.mkdir(parents=True)
            (nested / "first.txt").write_text("first\n", encoding="utf-8")
            (nested / "second.txt").write_text("second\n", encoding="utf-8")
            expected = attest_directory_namespace(target)
            target_stat = target.stat(follow_symlinks=False)
            actual_unlink = os.unlink
            unlink_count = 0

            def fail_second_private_unlink(name, *, dir_fd=None):
                nonlocal unlink_count
                if str(name).startswith(".hfr-remove-"):
                    unlink_count += 1
                    if unlink_count == 2:
                        raise OSError(errno.EIO, "injected nested unlink failure")
                return actual_unlink(name, dir_fd=dir_fd)

            with opened_directory_descriptor(root) as parent_descriptor:
                with patch.object(
                    path_safety_module.os,
                    "unlink",
                    side_effect=fail_second_private_unlink,
                ):
                    outcome = remove_directory_entry_tree_if_identity(
                        parent_descriptor,
                        target.name,
                        (target_stat.st_dev, target_stat.st_ino),
                        expected,
                    )

            self.assertEqual(outcome.status, DirectoryCleanupStatus.PARTIAL)
            self.assertFalse(outcome.durability_confirmed)
            self.assertIn("durability", outcome.detail or "")
            self.assertEqual(outcome.recovery_entries, ("approved",))
            self.assertEqual(
                [entry.name for entry in (target / "nested").iterdir()],
                ["second.txt"],
            )

    def test_attested_remover_reports_relocated_quarantine_as_recovery(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "approved"
            target.mkdir()
            (target / "approved.txt").write_text("approved\n", encoding="utf-8")
            expected = attest_directory_namespace(target)
            target_stat = target.stat(follow_symlinks=False)
            relocated_name = ".relocated-approved-entry"
            relocated = False

            def relocate_quarantine_then_fail(name, *, dir_fd=None):
                nonlocal relocated
                if str(name).startswith(".hfr-remove-") and not relocated:
                    relocated = True
                    os.rename(
                        name,
                        relocated_name,
                        src_dir_fd=dir_fd,
                        dst_dir_fd=dir_fd,
                    )
                    raise OSError(errno.EIO, "injected relocated unlink failure")
                raise AssertionError(f"unexpected unlink target: {name}")

            with opened_directory_descriptor(root) as parent_descriptor:
                with patch.object(
                    path_safety_module.os,
                    "unlink",
                    side_effect=relocate_quarantine_then_fail,
                ):
                    outcome = remove_directory_entry_tree_if_identity(
                        parent_descriptor,
                        target.name,
                        (target_stat.st_dev, target_stat.st_ino),
                        expected,
                    )

            self.assertTrue(relocated)
            self.assertEqual(outcome.status, DirectoryCleanupStatus.PARTIAL)
            self.assertFalse(outcome.durability_confirmed)
            self.assertEqual(len(outcome.cleanup_artifacts), 1)
            vault_name = outcome.cleanup_artifacts[0]
            self.assertEqual(
                set(outcome.recovery_entries),
                {"approved", f"{vault_name}/{relocated_name}"},
            )
            for relative_entry in outcome.recovery_entries:
                self.assertTrue((root / relative_entry).exists())

    def test_attested_remover_reports_complete_when_post_root_sync_is_unconfirmed(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "approved"
            target.mkdir()
            (target / "approved.txt").write_text("approved\n", encoding="utf-8")
            expected = attest_directory_namespace(target)
            target_stat = target.stat(follow_symlinks=False)
            actual_sync = path_safety_module._sync_directory_descriptor
            sync_count = 0

            def fail_first_sync(descriptor):
                nonlocal sync_count
                sync_count += 1
                if sync_count == 1:
                    raise OSError(errno.EIO, "injected post-root-removal sync failure")
                return actual_sync(descriptor)

            with opened_directory_descriptor(root) as parent_descriptor:
                with patch.object(
                    path_safety_module,
                    "_sync_directory_descriptor",
                    side_effect=fail_first_sync,
                ):
                    outcome = remove_directory_entry_tree_if_identity(
                        parent_descriptor,
                        target.name,
                        (target_stat.st_dev, target_stat.st_ino),
                        expected,
                    )

            self.assertFalse(outcome)
            self.assertEqual(
                outcome.status,
                DirectoryCleanupStatus.COMPLETE_DURABILITY_UNCONFIRMED,
            )
            self.assertFalse(outcome.durability_confirmed)
            self.assertEqual(
                outcome.removed_entry_count,
                expected.entry_count + 1,
            )
            self.assertEqual(outcome.recovery_entries, ())
            self.assertFalse(target.exists())
            self.assertFalse(list(root.glob(".hfr-remove-*")))

    def test_attested_remover_reports_public_reinsertion_during_final_sync_failure(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "approved"
            target.mkdir()
            (target / "approved.txt").write_text("approved\n", encoding="utf-8")
            expected = attest_directory_namespace(target)
            target_stat = target.stat(follow_symlinks=False)
            actual_sync = path_safety_module._sync_directory_descriptor
            sync_count = 0

            def reinsert_during_final_sync(descriptor):
                nonlocal sync_count
                sync_count += 1
                if sync_count == 2:
                    target.mkdir()
                    (target / "concurrent.txt").write_text(
                        "concurrent\n",
                        encoding="utf-8",
                    )
                    raise OSError(errno.EIO, "injected final parent sync failure")
                return actual_sync(descriptor)

            with opened_directory_descriptor(root) as parent_descriptor:
                with patch.object(
                    path_safety_module,
                    "_sync_directory_descriptor",
                    side_effect=reinsert_during_final_sync,
                ):
                    outcome = remove_directory_entry_tree_if_identity(
                        parent_descriptor,
                        target.name,
                        (target_stat.st_dev, target_stat.st_ino),
                        expected,
                    )

            self.assertEqual(
                outcome.status,
                DirectoryCleanupStatus.COMPLETE_DURABILITY_UNCONFIRMED,
            )
            self.assertFalse(outcome.durability_confirmed)
            self.assertEqual(outcome.recovery_entries, ())
            self.assertEqual(outcome.concurrent_entries, ("approved",))
            self.assertEqual(
                (target / "concurrent.txt").read_text(encoding="utf-8"),
                "concurrent\n",
            )

    def test_attested_remover_treats_public_observation_error_as_ambiguous(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "approved"
            target.mkdir()
            (target / "approved.txt").write_text("approved\n", encoding="utf-8")
            expected = attest_directory_namespace(target)
            target_stat = target.stat(follow_symlinks=False)
            actual_stat = path_safety_module.os.stat
            public_stat_count = 0

            def fail_final_public_observations(name, *args, **kwargs):
                nonlocal public_stat_count
                if name == target.name and kwargs.get("dir_fd") is not None:
                    public_stat_count += 1
                    if public_stat_count >= 4:
                        raise OSError(errno.EIO, "injected public observation failure")
                return actual_stat(name, *args, **kwargs)

            with opened_directory_descriptor(root) as parent_descriptor:
                with patch.object(
                    path_safety_module.os,
                    "stat",
                    side_effect=fail_final_public_observations,
                ):
                    outcome = remove_directory_entry_tree_if_identity(
                        parent_descriptor,
                        target.name,
                        (target_stat.st_dev, target_stat.st_ino),
                        expected,
                    )

            self.assertEqual(outcome.status, DirectoryCleanupStatus.PARTIAL)
            self.assertFalse(outcome.durability_confirmed)
            self.assertEqual(outcome.recovery_entries, ("approved",))
            self.assertIn("could not prove public recovery entry absence", outcome.detail or "")

    def test_attested_remover_reports_vault_recovery_after_public_reinsertion(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "approved"
            target.mkdir()
            (target / "first.txt").write_text("first\n", encoding="utf-8")
            (target / "second.txt").write_text("second\n", encoding="utf-8")
            expected = attest_directory_namespace(target)
            target_stat = target.stat(follow_symlinks=False)
            actual_unlink = os.unlink
            unlink_count = 0

            def reinsert_public_name_then_fail(name, *, dir_fd=None):
                nonlocal unlink_count
                if str(name).startswith(".hfr-remove-"):
                    unlink_count += 1
                    if unlink_count == 2:
                        target.mkdir()
                        (target / "concurrent.txt").write_text(
                            "concurrent\n",
                            encoding="utf-8",
                        )
                        raise OSError(errno.EIO, "injected second unlink failure")
                return actual_unlink(name, dir_fd=dir_fd)

            with opened_directory_descriptor(root) as parent_descriptor:
                with patch.object(
                    path_safety_module.os,
                    "unlink",
                    side_effect=reinsert_public_name_then_fail,
                ):
                    outcome = remove_directory_entry_tree_if_identity(
                        parent_descriptor,
                        target.name,
                        (target_stat.st_dev, target_stat.st_ino),
                        expected,
                    )

            self.assertFalse(outcome)
            self.assertEqual(outcome.status, DirectoryCleanupStatus.PARTIAL)
            self.assertEqual(outcome.removed_entry_count, 1)
            self.assertEqual(
                (target / "concurrent.txt").read_text(encoding="utf-8"),
                "concurrent\n",
            )
            self.assertEqual(len(outcome.cleanup_artifacts), 1)
            vault_name = outcome.cleanup_artifacts[0]
            self.assertEqual(outcome.recovery_entries, (f"{vault_name}/root",))
            self.assertEqual(outcome.concurrent_entries, ("approved",))
            recovery_root = root / outcome.recovery_entries[0]
            self.assertEqual(
                [entry.name for entry in recovery_root.iterdir()],
                ["second.txt"],
            )

    def test_attested_remover_never_unlinks_caller_visible_file_name(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "approved"
            target.mkdir()
            approved = target / "approved.txt"
            approved.write_text("approved\n", encoding="utf-8")
            expected = attest_directory_namespace(target)
            target_stat = target.stat(follow_symlinks=False)
            displaced = root / "displaced-approved.txt"
            actual_unlink = os.unlink
            raced = False

            def replace_public_name_before_unlink(name, *, dir_fd=None):
                nonlocal raced
                if name == "approved.txt" and not raced:
                    raced = True
                    os.rename(name, displaced, src_dir_fd=dir_fd)
                    descriptor = os.open(
                        name,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o600,
                        dir_fd=dir_fd,
                    )
                    try:
                        os.write(descriptor, b"unowned replacement\n")
                    finally:
                        os.close(descriptor)
                return actual_unlink(name, dir_fd=dir_fd)

            with opened_directory_descriptor(root) as parent_descriptor:
                with patch.object(
                    path_safety_module.os,
                    "unlink",
                    side_effect=replace_public_name_before_unlink,
                ):
                    removed = remove_directory_entry_tree_if_identity(
                        parent_descriptor,
                        target.name,
                        (target_stat.st_dev, target_stat.st_ino),
                        expected,
                    )

            self.assertTrue(removed)
            self.assertFalse(raced)
            self.assertFalse(target.exists())
            self.assertFalse(displaced.exists())

    def test_attested_remover_preserves_file_replaced_during_atomic_detach(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "approved"
            target.mkdir()
            approved = target / "approved.txt"
            approved.write_text("approved\n", encoding="utf-8")
            expected = attest_directory_namespace(target)
            target_stat = target.stat(follow_symlinks=False)
            displaced = root / "displaced-approved.txt"
            actual_rename = path_safety_module._native_rename_entry_between
            raced = False

            def replace_then_detach(
                source_parent_descriptor,
                source_name,
                target_parent_descriptor,
                target_name,
                *,
                exchange,
            ):
                nonlocal raced
                if source_name == "approved.txt" and not raced:
                    raced = True
                    os.rename(
                        source_name,
                        displaced,
                        src_dir_fd=source_parent_descriptor,
                    )
                    descriptor = os.open(
                        source_name,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o600,
                        dir_fd=source_parent_descriptor,
                    )
                    try:
                        os.write(descriptor, b"unowned replacement\n")
                    finally:
                        os.close(descriptor)
                return actual_rename(
                    source_parent_descriptor,
                    source_name,
                    target_parent_descriptor,
                    target_name,
                    exchange=exchange,
                )

            with opened_directory_descriptor(root) as parent_descriptor:
                with patch.object(
                    path_safety_module,
                    "_native_rename_entry_between",
                    side_effect=replace_then_detach,
                ):
                    removed = remove_directory_entry_tree_if_identity(
                        parent_descriptor,
                        target.name,
                        (target_stat.st_dev, target_stat.st_ino),
                        expected,
                    )

            self.assertFalse(removed)
            self.assertTrue(raced)
            self.assertEqual(
                (target / "approved.txt").read_text(encoding="utf-8"),
                "unowned replacement\n",
            )
            self.assertEqual(displaced.read_text(encoding="utf-8"), "approved\n")
            self.assertFalse(list(root.glob(".hfr-remove-*")))

    def test_attested_remover_never_rmdirs_caller_visible_directory_names(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "approved"
            approved_empty = target / "approved-empty"
            approved_empty.mkdir(parents=True)
            expected = attest_directory_namespace(target)
            target_stat = target.stat(follow_symlinks=False)
            actual_rmdir = os.rmdir
            raced_names: list[str] = []

            def replace_public_name_before_rmdir(name, *, dir_fd=None):
                if name in {"approved-empty", "approved"}:
                    raced_names.append(name)
                return actual_rmdir(name, dir_fd=dir_fd)

            with opened_directory_descriptor(root) as parent_descriptor:
                with patch.object(
                    path_safety_module.os,
                    "rmdir",
                    side_effect=replace_public_name_before_rmdir,
                ):
                    removed = remove_directory_entry_tree_if_identity(
                        parent_descriptor,
                        target.name,
                        (target_stat.st_dev, target_stat.st_ino),
                        expected,
                    )

            self.assertTrue(removed)
            self.assertEqual(raced_names, [])
            self.assertFalse(target.exists())

    def test_attested_remover_preserves_empty_directory_replaced_during_detach(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "approved"
            approved_empty = target / "approved-empty"
            approved_empty.mkdir(parents=True)
            expected = attest_directory_namespace(target)
            target_stat = target.stat(follow_symlinks=False)
            displaced = root / "displaced-approved-empty"
            actual_rename = path_safety_module._native_rename_entry_between
            raced = False

            def replace_then_detach(
                source_parent_descriptor,
                source_name,
                target_parent_descriptor,
                target_name,
                *,
                exchange,
            ):
                nonlocal raced
                if source_name == "approved-empty" and not raced:
                    raced = True
                    os.rename(
                        source_name,
                        displaced,
                        src_dir_fd=source_parent_descriptor,
                    )
                    os.mkdir(source_name, dir_fd=source_parent_descriptor)
                return actual_rename(
                    source_parent_descriptor,
                    source_name,
                    target_parent_descriptor,
                    target_name,
                    exchange=exchange,
                )

            with opened_directory_descriptor(root) as parent_descriptor:
                with patch.object(
                    path_safety_module,
                    "_native_rename_entry_between",
                    side_effect=replace_then_detach,
                ):
                    removed = remove_directory_entry_tree_if_identity(
                        parent_descriptor,
                        target.name,
                        (target_stat.st_dev, target_stat.st_ino),
                        expected,
                    )

            self.assertFalse(removed)
            self.assertTrue(raced)
            self.assertTrue(approved_empty.is_dir())
            self.assertEqual(list(approved_empty.iterdir()), [])
            self.assertTrue(displaced.is_dir())
            self.assertEqual(list(displaced.iterdir()), [])
            self.assertFalse(list(root.glob(".hfr-remove-*")))

    def test_attested_remover_preserves_root_replaced_during_vault_detach(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "approved"
            target.mkdir()
            expected = attest_directory_namespace(target)
            target_stat = target.stat(follow_symlinks=False)
            displaced = root / "displaced-approved-root"
            actual_rename = path_safety_module._native_rename_entry_between
            raced = False

            def replace_then_detach(
                source_parent_descriptor,
                source_name,
                target_parent_descriptor,
                target_name,
                *,
                exchange,
            ):
                nonlocal raced
                if source_name == "approved" and not raced:
                    raced = True
                    os.rename(
                        source_name,
                        displaced,
                        src_dir_fd=source_parent_descriptor,
                    )
                    os.mkdir(source_name, dir_fd=source_parent_descriptor)
                return actual_rename(
                    source_parent_descriptor,
                    source_name,
                    target_parent_descriptor,
                    target_name,
                    exchange=exchange,
                )

            with opened_directory_descriptor(root) as parent_descriptor:
                with patch.object(
                    path_safety_module,
                    "_native_rename_entry_between",
                    side_effect=replace_then_detach,
                ):
                    removed = remove_directory_entry_tree_if_identity(
                        parent_descriptor,
                        target.name,
                        (target_stat.st_dev, target_stat.st_ino),
                        expected,
                    )

            self.assertFalse(removed)
            self.assertTrue(raced)
            self.assertTrue(target.is_dir())
            self.assertEqual(list(target.iterdir()), [])
            self.assertTrue(displaced.is_dir())
            self.assertEqual(list(displaced.iterdir()), [])
            self.assertFalse(list(root.glob(".hfr-remove-*")))

    def test_noreplace_post_fsync_failure_reports_applied_mutation(self) -> None:
        with TemporaryDirectory() as tmp:
            parent = Path(tmp)
            source = parent / "source"
            source.mkdir()
            (source / "marker.txt").write_text("source\n", encoding="utf-8")
            descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                with (
                    patch.object(
                        path_safety_module,
                        "_sync_directory_descriptor",
                        side_effect=[None, ValueError("injected post-rename fsync failure")],
                    ),
                    self.assertRaises(
                        path_safety_module.AtomicNamespaceMutationError
                    ) as raised,
                ):
                    path_safety_module.atomic_rename_entry_noreplace(
                        descriptor,
                        "source",
                        "published",
                    )
            finally:
                os.close(descriptor)

            self.assertTrue(raised.exception.mutation_applied)
            self.assertFalse(source.exists())
            self.assertEqual(
                (parent / "published" / "marker.txt").read_text(encoding="utf-8"),
                "source\n",
            )

    def test_exchange_post_fsync_failure_reports_applied_mutation(self) -> None:
        with TemporaryDirectory() as tmp:
            parent = Path(tmp)
            left = parent / "left"
            right = parent / "right"
            left.mkdir()
            right.mkdir()
            (left / "marker.txt").write_text("left\n", encoding="utf-8")
            (right / "marker.txt").write_text("right\n", encoding="utf-8")
            descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                with (
                    patch.object(
                        path_safety_module,
                        "_sync_directory_descriptor",
                        side_effect=[None, ValueError("injected post-exchange fsync failure")],
                    ),
                    self.assertRaises(
                        path_safety_module.AtomicNamespaceMutationError
                    ) as raised,
                ):
                    path_safety_module.atomic_exchange_entries(
                        descriptor,
                        "left",
                        "right",
                    )
            finally:
                os.close(descriptor)

            self.assertTrue(raised.exception.mutation_applied)
            self.assertEqual(
                (left / "marker.txt").read_text(encoding="utf-8"),
                "right\n",
            )
            self.assertEqual(
                (right / "marker.txt").read_text(encoding="utf-8"),
                "left\n",
            )

    def test_noreplace_preflights_parent_fsync_before_namespace_mutation(self) -> None:
        with TemporaryDirectory() as tmp:
            parent = Path(tmp)
            source = parent / "source"
            source.mkdir()
            (source / "marker.txt").write_text("source\n", encoding="utf-8")
            descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                with (
                    patch.object(
                        path_safety_module.os,
                        "fsync",
                        side_effect=OSError(errno.EINVAL, "unsupported"),
                    ),
                    self.assertRaisesRegex(ValueError, "directory fsync is unsupported"),
                ):
                    path_safety_module.atomic_rename_entry_noreplace(
                        descriptor,
                        "source",
                        "published",
                    )
            finally:
                os.close(descriptor)

            self.assertEqual(
                (source / "marker.txt").read_text(encoding="utf-8"),
                "source\n",
            )
            self.assertFalse((parent / "published").exists())

    def test_exchange_preflights_parent_fsync_before_namespace_mutation(self) -> None:
        with TemporaryDirectory() as tmp:
            parent = Path(tmp)
            left = parent / "left"
            right = parent / "right"
            left.mkdir()
            right.mkdir()
            (left / "marker.txt").write_text("left\n", encoding="utf-8")
            (right / "marker.txt").write_text("right\n", encoding="utf-8")
            descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                with (
                    patch.object(
                        path_safety_module.os,
                        "fsync",
                        side_effect=OSError(errno.EINVAL, "unsupported"),
                    ),
                    self.assertRaisesRegex(ValueError, "directory fsync is unsupported"),
                ):
                    path_safety_module.atomic_exchange_entries(
                        descriptor,
                        "left",
                        "right",
                    )
            finally:
                os.close(descriptor)

            self.assertEqual(
                (left / "marker.txt").read_text(encoding="utf-8"),
                "left\n",
            )
            self.assertEqual(
                (right / "marker.txt").read_text(encoding="utf-8"),
                "right\n",
            )


class SafeOutputDirectoryTests(unittest.TestCase):
    def test_owned_replacement_refuses_unowned_nonempty_directory(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "output"
            target.mkdir()
            (target / "unrelated.txt").write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "refusing to replace unrecognized"):
                replace_owned_output_directory(
                    target,
                    repo_root=root / "repo",
                    force=True,
                    label="test output",
                    is_owned=lambda _path: False,
                )

            self.assertTrue((target / "unrelated.txt").exists())

    def test_owned_replacement_requires_force_and_deletes_only_owned_output(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "output"
            target.mkdir()
            (target / "marker.json").write_text("{}", encoding="utf-8")
            def owned(path: Path) -> bool:
                return (path / "marker.json").is_file()

            with self.assertRaisesRegex(ValueError, "pass --force"):
                replace_owned_output_directory(
                    target,
                    repo_root=root / "repo",
                    force=False,
                    label="test output",
                    is_owned=owned,
                )

            replace_owned_output_directory(
                target,
                repo_root=root / "repo",
                force=True,
                label="test output",
                is_owned=owned,
            )
            self.assertFalse(target.exists())

    def test_schema_version_only_marker_cannot_claim_output_ownership(self) -> None:
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "output"
            target.mkdir()
            (target / "harness_result.json").write_text(
                '{"schema_version":"hfr.harness_run_result.v1"}\n',
                encoding="utf-8",
            )

            self.assertFalse(
                json_marker_matches_schema(
                    target,
                    "harness_result.json",
                    "harness_run_result",
                )
            )

    def test_lock_is_held_through_complete_writer_context(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "output"

            with locked_owned_output_directory(
                target,
                repo_root=root / "repo",
                force=False,
                label="test output",
                is_owned=lambda _path: False,
            ):
                target.mkdir()
                with self.assertRaisesRegex(ValueError, "locked for publication"):
                    with locked_owned_output_directory(
                        target,
                        repo_root=root / "repo",
                        force=True,
                        label="test output",
                        is_owned=lambda _path: True,
                    ):
                        pass

    def test_mutation_during_ownership_check_is_not_deleted(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "output"
            target.mkdir()
            protected = target / "protected.txt"
            protected.write_text("keep\n", encoding="utf-8")

            def mutate_and_claim(path: Path) -> bool:
                (path / "concurrent.txt").write_text("also keep\n", encoding="utf-8")
                return True

            with self.assertRaisesRegex(ValueError, "unrecognized"):
                with locked_owned_output_directory(
                    target,
                    repo_root=root / "repo",
                    force=True,
                    label="test output",
                    is_owned=mutate_and_claim,
                ):
                    pass

            self.assertEqual(protected.read_text(encoding="utf-8"), "keep\n")
            self.assertTrue((target / "concurrent.txt").is_file())

    def test_target_swap_during_ownership_check_is_not_deleted(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "output"
            target.mkdir()
            (target / "owned.txt").write_text("old\n", encoding="utf-8")
            displaced = root / "displaced"

            def swap_and_claim(path: Path) -> bool:
                path.rename(displaced)
                path.mkdir()
                (path / "unowned.txt").write_text("new\n", encoding="utf-8")
                return True

            with self.assertRaisesRegex(ValueError, "unrecognized"):
                with locked_owned_output_directory(
                    target,
                    repo_root=root / "repo",
                    force=True,
                    label="test output",
                    is_owned=swap_and_claim,
                ):
                    pass

            self.assertTrue((target / "unowned.txt").is_file())
            self.assertTrue((displaced / "owned.txt").is_file())

    def test_target_swap_after_ownership_check_is_restored_without_deletion(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "output"
            target.mkdir()
            (target / "owned.txt").write_text("old\n", encoding="utf-8")
            displaced = root / "displaced"
            alien = root / "alien"
            alien.mkdir()
            (alien / "unowned.txt").write_text("new\n", encoding="utf-8")
            original_rename = Path.rename
            swapped = False

            def swap_at_quarantine(source: Path, destination: Path):
                nonlocal swapped
                if source == target and not swapped:
                    swapped = True
                    os.rename(source, displaced)
                    os.rename(alien, source)
                return original_rename(source, destination)

            with (
                patch.object(Path, "rename", autospec=True, side_effect=swap_at_quarantine),
                self.assertRaisesRegex(ValueError, "contents changed during replacement"),
            ):
                with locked_owned_output_directory(
                    target,
                    repo_root=root / "repo",
                    force=True,
                    label="test output",
                    is_owned=lambda _path: True,
                ):
                    pass

            self.assertTrue(swapped)
            self.assertTrue((target / "unowned.txt").is_file())
            self.assertTrue((displaced / "owned.txt").is_file())

    def test_quarantine_swap_after_descriptor_open_does_not_delete_replacement(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "output"
            target.mkdir()
            (target / "owned.txt").write_text("old\n", encoding="utf-8")
            displaced = root / "displaced-quarantine"
            alien = root / "alien"
            alien.mkdir()
            (alien / "unowned.txt").write_text("keep\n", encoding="utf-8")
            quarantine: Path | None = None
            original_rename = Path.rename
            original_remove = path_safety_module._remove_directory_contents_fd

            def capture_quarantine(source: Path, destination: Path):
                nonlocal quarantine
                result = original_rename(source, destination)
                if source == target:
                    quarantine = Path(destination)
                return result

            def swap_after_descriptor_open(
                directory_descriptor: int, directory_flags: int
            ) -> bool:
                assert quarantine is not None
                original_rename(quarantine, displaced)
                original_rename(alien, quarantine)
                return original_remove(directory_descriptor, directory_flags)

            with (
                patch.object(
                    Path,
                    "rename",
                    autospec=True,
                    side_effect=capture_quarantine,
                ),
                patch.object(
                    path_safety_module,
                    "_remove_directory_contents_fd",
                    side_effect=swap_after_descriptor_open,
                ),
                self.assertRaisesRegex(ValueError, "quarantine changed"),
            ):
                with locked_owned_output_directory(
                    target,
                    repo_root=root / "repo",
                    force=True,
                    label="test output",
                    is_owned=lambda _path: True,
                ):
                    pass

            assert quarantine is not None
            self.assertEqual(
                (quarantine / "unowned.txt").read_text(encoding="utf-8"),
                "keep\n",
            )
            self.assertTrue(displaced.is_dir())
            self.assertEqual(list(displaced.iterdir()), [])

    def test_allows_nested_relative_output_directory(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            output_dir = repo_root / "runs" / "smoke"
            output_dir.mkdir(parents=True)

            with chdir(repo_root):
                assert_safe_output_directory(Path("runs/smoke"), repo_root=repo_root)

    def test_rejects_filesystem_root_and_protected_roots_or_ancestors(self) -> None:
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo_root = base / "repo"
            cwd = base / "workspace" / "nested"
            repo_root.mkdir()
            cwd.mkdir(parents=True)

            unsafe_targets = {
                "filesystem root": Path(repo_root.anchor),
                "repository root": repo_root,
                "repository ancestor": repo_root.parent,
                "working directory": cwd,
                "working directory ancestor": cwd.parent,
            }
            for case_name, target in unsafe_targets.items():
                with self.subTest(case=case_name):
                    with self.assertRaisesRegex(ValueError, "protected|filesystem root"):
                        assert_safe_output_directory(target, repo_root=repo_root, cwd=cwd)

    def test_rejects_symlinked_parent_and_destination(self) -> None:
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo_root = base / "repo"
            cwd = base / "workspace"
            external = base / "external"
            repo_root.mkdir()
            cwd.mkdir()
            (external / "nested").mkdir(parents=True)
            linked_parent = repo_root / "linked-parent"
            linked_destination = repo_root / "linked-destination"
            try:
                linked_parent.symlink_to(external, target_is_directory=True)
                linked_destination.symlink_to(external / "nested", target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")

            for target in (linked_parent / "nested", linked_destination):
                with self.subTest(target=target):
                    with self.assertRaisesRegex(ValueError, "symlink"):
                        assert_safe_output_directory(target, repo_root=repo_root, cwd=cwd)


class SmokeScriptGuardTests(unittest.TestCase):
    def test_every_rmtree_is_immediately_preceded_by_shared_guard(self) -> None:
        for script in SMOKE_SCRIPTS:
            tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
            rmtree_count = 0
            owned_replacement_count = sum(
                1
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id
                in {
                    "replace_owned_output_directory",
                    "locked_owned_output_directory",
                }
            )
            for node in ast.walk(tree):
                for field_name in ("body", "orelse", "finalbody"):
                    statements = getattr(node, field_name, None)
                    if not isinstance(statements, list):
                        continue
                    for index, statement in enumerate(statements):
                        rmtree_call = _expression_call(statement, owner="shutil", name="rmtree")
                        if rmtree_call is None:
                            continue
                        rmtree_count += 1
                        self.assertGreater(index, 0, f"unguarded rmtree in {script}")
                        guard_call = _expression_call(
                            statements[index - 1],
                            owner=None,
                            name="assert_safe_output_directory",
                        )
                        self.assertIsNotNone(guard_call, f"unguarded rmtree in {script}")
                        assert guard_call is not None
                        self.assertEqual(
                            ast.dump(guard_call.args[0]),
                            ast.dump(rmtree_call.args[0]),
                            f"guard checks a different target in {script}",
                        )
            self.assertGreater(
                rmtree_count + owned_replacement_count,
                0,
                f"expected a guarded removal path in {script}",
            )


def _expression_call(statement: ast.stmt, *, owner: str | None, name: str) -> ast.Call | None:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return None
    call = statement.value
    if owner is None:
        return call if isinstance(call.func, ast.Name) and call.func.id == name else None
    if not isinstance(call.func, ast.Attribute) or call.func.attr != name:
        return None
    return call if isinstance(call.func.value, ast.Name) and call.func.value.id == owner else None


if __name__ == "__main__":
    unittest.main()
