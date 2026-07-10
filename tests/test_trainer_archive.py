import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import flightrecorder.path_safety as path_safety_module
import flightrecorder.trainer_archive as trainer_archive_module
from flightrecorder.preflight import build_trainer_launch_check, build_trainer_preflight
from flightrecorder.trainer_archive import TrainerArchiveError, build_trainer_archive
from flightrecorder.validation import validate_artifacts


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_snapshot(path: Path) -> dict[str, bytes | None]:
    snapshot: dict[str, bytes | None] = {".": None}
    for child in sorted(path.rglob("*")):
        relative = child.relative_to(path).as_posix()
        snapshot[relative] = child.read_bytes() if child.is_file() else None
    return snapshot


def _write_passed_evidence_bundle(path: Path) -> None:
    _write_json(
        path,
        {
            "schema_version": "hfr.evidence_bundle.v1",
            "bundle_path": path.name,
            "passed": True,
            "readiness": "ready",
            "decision": {
                "readiness": "ready",
                "recommendation": "promote_handoff",
                "summary": "Minimal archive hardening fixture is ready.",
                "blocking_check_count": 0,
                "next_actions": [],
            },
            "check_count": 0,
            "failed_check_count": 0,
            "checks": [],
            "artifacts": {},
            "metrics": {},
            "notes": [],
        },
    )


class TrainerArchiveHardeningTests(unittest.TestCase):
    def _ready_sources(
        self,
        root: Path,
        *,
        trainer_command: str = "python train.py --bundle evidence_bundle.json",
    ) -> tuple[Path, Path, Path]:
        source = root / "source"
        source.mkdir()
        bundle = source / "evidence_bundle.json"
        preflight_path = source / "trainer_preflight.json"
        launch_path = source / "trainer_launch_check.json"
        _write_passed_evidence_bundle(bundle)
        preflight = build_trainer_preflight(
            out_path=preflight_path,
            gate_paths=[bundle],
            evidence_bundle_path=bundle,
            trainer_command=trainer_command,
            allow_unvalidated_gates=True,
        )
        _write_json(preflight_path, preflight)
        validation = validate_artifacts(trainer_preflight_paths=[preflight_path])
        self.assertTrue(validation["passed"], validation)
        launch = build_trainer_launch_check(
            preflight_path=preflight_path,
            preflight=preflight,
            validation_summary=validation,
        )
        _write_json(launch_path, launch)
        self.assertTrue(launch["passed"], launch)
        return bundle, preflight_path, launch_path

    def test_cwd_relative_command_input_is_rewritten_without_publishing_absolute_path(
        self,
    ):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            bundle_path = root / "source" / "evidence_bundle.json"
            cwd_relative_bundle = os.path.relpath(bundle_path, Path.cwd())
            _, preflight, launch = self._ready_sources(
                root,
                trainer_command=f"python train.py --bundle {cwd_relative_bundle}",
            )

            archive = build_trainer_archive(
                out_dir=root / "archive",
                preflight_path=preflight,
                launch_check_path=launch,
                require_self_contained=True,
            )

            portable_argv = archive["portable_command"]["argv"]
            self.assertTrue(archive["portable_command"]["rewritten"])
            self.assertNotIn(cwd_relative_bundle, portable_argv)
            self.assertIn("artifacts/trainer_artifacts", " ".join(portable_argv))
            rewrite_paths = {item["original_path"] for item in archive["path_rewrites"]}
            self.assertIn(cwd_relative_bundle, rewrite_paths)
            self.assertNotIn(
                str(bundle_path.resolve()), json.dumps(archive, sort_keys=True)
            )
            self.assertNotIn("_source_aliases", json.dumps(archive, sort_keys=True))
            validation = validate_artifacts(trainer_archive_paths=[root / "archive"])
            self.assertTrue(validation["passed"], validation)

    def test_source_mutation_during_copy_is_blocked_and_not_archived(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle, preflight, launch = self._ready_sources(root)
            archive_root = root / "archive"
            original_copy = shutil.copy2
            changed = False

            def mutate_then_copy(source, destination, *args, **kwargs):
                nonlocal changed
                if Path(source) == bundle and not changed:
                    bundle.write_text(
                        '{"secret":"changed-after-check"}\n', encoding="utf-8"
                    )
                    changed = True
                return original_copy(source, destination, *args, **kwargs)

            with patch(
                "flightrecorder.trainer_archive.shutil.copy2",
                side_effect=mutate_then_copy,
            ):
                archive = build_trainer_archive(
                    out_dir=archive_root,
                    preflight_path=preflight,
                    launch_check_path=launch,
                    require_self_contained=True,
                )

            self.assertTrue(changed)
            self.assertFalse(archive["passed"])
            self.assertTrue(
                any(
                    "changed while being copied" in item["reason"]
                    for item in archive["missing"]
                )
            )
            archived_text = "\n".join(
                path.read_text(encoding="utf-8", errors="replace")
                for path in (archive_root / "artifacts").rglob("*")
                if path.is_file()
            )
            self.assertNotIn("changed-after-check", archived_text)

    def test_preflight_mutation_during_copy_aborts_before_archiving_changed_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, preflight, launch = self._ready_sources(root)
            archive_root = root / "archive"
            original_copy = shutil.copy2
            changed = False

            def mutate_then_copy(source, destination, *args, **kwargs):
                nonlocal changed
                if Path(source) == preflight and not changed:
                    preflight.write_text(
                        preflight.read_text(encoding="utf-8") + " \n",
                        encoding="utf-8",
                    )
                    changed = True
                return original_copy(source, destination, *args, **kwargs)

            with (
                patch(
                    "flightrecorder.trainer_archive.shutil.copy2",
                    side_effect=mutate_then_copy,
                ),
                self.assertRaisesRegex(
                    TrainerArchiveError,
                    "trainer_preflight source changed while being copied",
                ),
            ):
                build_trainer_archive(
                    out_dir=archive_root,
                    preflight_path=preflight,
                    launch_check_path=launch,
                    require_self_contained=True,
                )

            self.assertTrue(changed)
            self.assertFalse(
                (archive_root / "artifacts" / "trainer_preflight.json").exists()
            )
            self.assertFalse((archive_root / "trainer_archive.json").exists())

    @staticmethod
    def _retarget_artifact(
        preflight_path: Path, launch_path: Path, path_value: str, source: Path
    ) -> None:
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        size_bytes = source.stat().st_size
        sha256 = _sha256(source)
        preflight["artifacts"]["evidence_bundle"].update(
            {"path": path_value, "size_bytes": size_bytes, "sha256": sha256}
        )
        preflight["schema_contracts"]["evidence_bundle"].update(
            {"path": path_value, "size_bytes": size_bytes, "sha256": sha256}
        )
        _write_json(preflight_path, preflight)

        launch = json.loads(launch_path.read_text(encoding="utf-8"))
        _write_json(launch_path, launch)

    def test_ready_archive_writes_command_specific_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, preflight, launch = self._ready_sources(root)
            archive_root = root / "archive"

            archive = build_trainer_archive(
                out_dir=archive_root,
                preflight_path=preflight,
                launch_check_path=launch,
                require_self_contained=True,
            )

            self.assertTrue(archive["passed"])
            self.assertEqual(
                (archive_root / ".hfr-trainer-archive").read_text(encoding="utf-8"),
                "hfr.trainer_archive.v1\n",
            )

    def test_absolute_artifact_reference_is_blocked_without_copying_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, preflight, launch = self._ready_sources(root)
            secret = root / "outside-secret.json"
            secret.write_text('{"secret":"must-not-copy"}\n', encoding="utf-8")
            self._retarget_artifact(preflight, launch, str(secret), secret)

            archive_root = root / "archive"
            archive = build_trainer_archive(
                out_dir=archive_root,
                preflight_path=preflight,
                launch_check_path=launch,
            )

            self.assertFalse(archive["passed"])
            self.assertTrue(
                any("absolute" in item["reason"] for item in archive["missing"])
            )
            archived_text = "\n".join(
                path.read_text(encoding="utf-8", errors="replace")
                for path in (archive_root / "artifacts").rglob("*")
                if path.is_file()
            )
            self.assertNotIn("must-not-copy", archived_text)

    def test_parent_traversing_artifact_reference_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, preflight, launch = self._ready_sources(root)
            secret = root / "outside-secret.json"
            secret.write_text('{"secret":"parent-traversal"}\n', encoding="utf-8")
            self._retarget_artifact(preflight, launch, "../outside-secret.json", secret)

            archive_root = root / "archive"
            archive = build_trainer_archive(
                out_dir=archive_root,
                preflight_path=preflight,
                launch_check_path=launch,
            )

            self.assertFalse(archive["passed"])
            self.assertTrue(
                any("parent traversal" in item["reason"] for item in archive["missing"])
            )
            self.assertFalse(
                any("outside-secret" in path.name for path in archive_root.rglob("*"))
            )

    def test_symlinked_reference_component_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, preflight, launch = self._ready_sources(root)
            outside = root / "outside"
            outside.mkdir()
            secret = outside / "secret.json"
            secret.write_text('{"secret":"symlink-component"}\n', encoding="utf-8")
            link = preflight.parent / "linked"
            try:
                link.symlink_to(outside, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink creation unavailable: {exc}")
            self._retarget_artifact(preflight, launch, "linked/secret.json", secret)

            archive_root = root / "archive"
            archive = build_trainer_archive(
                out_dir=archive_root,
                preflight_path=preflight,
                launch_check_path=launch,
            )

            self.assertFalse(archive["passed"])
            self.assertTrue(
                any(
                    "symlink component" in item["reason"] for item in archive["missing"]
                )
            )
            self.assertFalse(
                any("secret.json" == path.name for path in archive_root.rglob("*"))
            )

    def test_incomplete_record_contract_is_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, preflight_path, launch = self._ready_sources(root)
            preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
            del preflight["artifacts"]["evidence_bundle"]["sha256"]
            _write_json(preflight_path, preflight)
            archive_root = root / "archive"

            with self.assertRaisesRegex(
                TrainerArchiveError, "schema validation failed"
            ):
                build_trainer_archive(
                    out_dir=archive_root,
                    preflight_path=preflight_path,
                    launch_check_path=launch,
                )

            self.assertFalse(archive_root.exists())

    def test_malformed_launch_contract_is_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, preflight, launch_path = self._ready_sources(root)
            launch = json.loads(launch_path.read_text(encoding="utf-8"))
            launch["provider_job_id"] = "must-not-be-accepted"
            _write_json(launch_path, launch)
            archive_root = root / "archive"

            with self.assertRaisesRegex(
                TrainerArchiveError, "trainer launch check schema validation failed"
            ):
                build_trainer_archive(
                    out_dir=archive_root,
                    preflight_path=preflight,
                    launch_check_path=launch_path,
                )

            self.assertFalse(archive_root.exists())

    def test_semantically_invalid_ready_source_writes_blocked_receipt_without_references(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle, preflight_path, launch = self._ready_sources(root)
            preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
            preflight["passed_gate_count"] = 0
            _write_json(preflight_path, preflight)
            archive_root = root / "archive"

            archive = build_trainer_archive(
                out_dir=archive_root,
                preflight_path=preflight_path,
                launch_check_path=launch,
            )

            self.assertFalse(archive["passed"])
            self.assertTrue((archive_root / "trainer_archive.json").is_file())
            self.assertTrue(
                any(
                    "semantic validation failed" in item["reason"]
                    for item in archive["missing"]
                )
            )
            copied_hashes = {item["sha256"] for item in archive["artifacts"]}
            self.assertNotIn(_sha256(bundle), copied_hashes)

    def test_force_requires_command_marker_even_with_forged_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, preflight, launch = self._ready_sources(root)
            archive_root = root / "archive"
            archive_root.mkdir()
            _write_json(
                archive_root / "trainer_archive.json",
                {"schema_version": "hfr.trainer_archive.v1"},
            )
            sentinel = archive_root / "keep.txt"
            sentinel.write_text("do not delete\n", encoding="utf-8")

            with self.assertRaisesRegex(TrainerArchiveError, "command marker"):
                build_trainer_archive(
                    out_dir=archive_root,
                    preflight_path=preflight,
                    launch_check_path=launch,
                    force=True,
                )

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "do not delete\n")

    def test_force_replaces_only_a_marked_schema_valid_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, preflight, launch = self._ready_sources(root)
            archive_root = root / "archive"
            first = build_trainer_archive(
                out_dir=archive_root,
                preflight_path=preflight,
                launch_check_path=launch,
                require_self_contained=True,
            )
            self.assertTrue(first["passed"])
            stale = archive_root / "stale.txt"
            stale.write_text("replace me\n", encoding="utf-8")

            second = build_trainer_archive(
                out_dir=archive_root,
                preflight_path=preflight,
                launch_check_path=launch,
                require_self_contained=True,
                force=True,
            )

            self.assertTrue(second["passed"])
            self.assertFalse(stale.exists())
            self.assertEqual(
                (archive_root / ".hfr-trainer-archive").read_text(),
                "hfr.trainer_archive.v1\n",
            )

    def test_failed_replacement_build_preserves_previous_archive_without_public_partial(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, preflight, launch = self._ready_sources(root)
            archive_root = root / "archive"
            first = build_trainer_archive(
                out_dir=archive_root,
                preflight_path=preflight,
                launch_check_path=launch,
                require_self_contained=True,
            )
            self.assertTrue(first["passed"])
            before = _tree_snapshot(archive_root)
            original_copy = trainer_archive_module._copy_file_artifact
            copy_count = 0

            def fail_after_first_copy(*args, **kwargs):
                nonlocal copy_count
                copy_count += 1
                if copy_count == 2:
                    raise TrainerArchiveError("injected staged build failure")
                return original_copy(*args, **kwargs)

            with (
                patch(
                    "flightrecorder.trainer_archive._copy_file_artifact",
                    side_effect=fail_after_first_copy,
                ),
                self.assertRaisesRegex(
                    TrainerArchiveError, "injected staged build failure"
                ),
            ):
                build_trainer_archive(
                    out_dir=archive_root,
                    preflight_path=preflight,
                    launch_check_path=launch,
                    require_self_contained=True,
                    force=True,
                )

            self.assertEqual(_tree_snapshot(archive_root), before)
            self.assertEqual(list(root.glob(".archive.hfr-stage-*")), [])
            self.assertEqual(list(root.glob(".archive.hfr-backup-*")), [])
            validation = validate_artifacts(trainer_archive_paths=[archive_root])
            self.assertTrue(validation["passed"], validation)

    def test_failed_publication_restores_previous_archive_and_cleans_private_workdirs(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, preflight, launch = self._ready_sources(root)
            archive_root = root / "archive"
            first = build_trainer_archive(
                out_dir=archive_root,
                preflight_path=preflight,
                launch_check_path=launch,
                require_self_contained=True,
            )
            self.assertTrue(first["passed"])
            before = _tree_snapshot(archive_root)
            with (
                patch(
                    "flightrecorder.trainer_archive._atomic_exchange_directories",
                    side_effect=TrainerArchiveError("injected publication failure"),
                ),
                self.assertRaisesRegex(
                    TrainerArchiveError, "injected publication failure"
                ),
            ):
                build_trainer_archive(
                    out_dir=archive_root,
                    preflight_path=preflight,
                    launch_check_path=launch,
                    require_self_contained=True,
                    force=True,
                )

            self.assertEqual(_tree_snapshot(archive_root), before)
            self.assertEqual(list(root.glob(".archive.hfr-stage-*")), [])
            self.assertEqual(list(root.glob(".archive.hfr-backup-*")), [])
            validation = validate_artifacts(trainer_archive_paths=[archive_root])
            self.assertTrue(validation["passed"], validation)

    def test_force_exchange_never_makes_existing_archive_path_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, preflight, launch = self._ready_sources(root)
            archive_root = root / "archive"
            build_trainer_archive(
                out_dir=archive_root,
                preflight_path=preflight,
                launch_check_path=launch,
                require_self_contained=True,
            )
            real_exchange = trainer_archive_module._atomic_exchange_directories

            def assert_reader_presence(staging, target):
                self.assertTrue(Path(target).is_dir())
                real_exchange(staging, target)
                self.assertTrue(Path(target).is_dir())

            with patch(
                "flightrecorder.trainer_archive._atomic_exchange_directories",
                side_effect=assert_reader_presence,
            ):
                result = build_trainer_archive(
                    out_dir=archive_root,
                    preflight_path=preflight,
                    launch_check_path=launch,
                    require_self_contained=True,
                    force=True,
                )

            self.assertTrue(result["passed"])

    def test_target_content_mutation_during_exchange_is_rolled_back_without_deletion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, preflight, launch = self._ready_sources(root)
            archive_root = root / "archive"
            build_trainer_archive(
                out_dir=archive_root,
                preflight_path=preflight,
                launch_check_path=launch,
                require_self_contained=True,
            )
            real_exchange = trainer_archive_module._atomic_exchange_directories
            exchange_count = 0

            def mutate_then_exchange(staging, target):
                nonlocal exchange_count
                exchange_count += 1
                if exchange_count == 1:
                    (Path(target) / "concurrent-writer.txt").write_text(
                        "do not delete\n", encoding="utf-8"
                    )
                real_exchange(staging, target)

            with (
                patch(
                    "flightrecorder.trainer_archive._atomic_exchange_directories",
                    side_effect=mutate_then_exchange,
                ),
                self.assertRaisesRegex(
                    TrainerArchiveError, "publication attestation failed"
                ),
            ):
                build_trainer_archive(
                    out_dir=archive_root,
                    preflight_path=preflight,
                    launch_check_path=launch,
                    require_self_contained=True,
                    force=True,
                )

            self.assertEqual(exchange_count, 2)
            self.assertEqual(
                (archive_root / "concurrent-writer.txt").read_text(encoding="utf-8"),
                "do not delete\n",
            )
            self.assertEqual(list(root.glob(".archive.hfr-stage-*")), [])

    def test_empty_directory_added_during_exchange_is_attested_and_rolled_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, preflight, launch = self._ready_sources(root)
            archive_root = root / "archive"
            build_trainer_archive(
                out_dir=archive_root,
                preflight_path=preflight,
                launch_check_path=launch,
                require_self_contained=True,
            )
            real_exchange = trainer_archive_module._atomic_exchange_directories
            exchange_count = 0

            def add_directory_then_exchange(staging, target):
                nonlocal exchange_count
                exchange_count += 1
                if exchange_count == 1:
                    (Path(target) / "concurrent-empty-directory").mkdir()
                real_exchange(staging, target)

            with (
                patch(
                    "flightrecorder.trainer_archive._atomic_exchange_directories",
                    side_effect=add_directory_then_exchange,
                ),
                self.assertRaisesRegex(
                    TrainerArchiveError, "publication attestation failed"
                ),
            ):
                build_trainer_archive(
                    out_dir=archive_root,
                    preflight_path=preflight,
                    launch_check_path=launch,
                    require_self_contained=True,
                    force=True,
                )

            self.assertEqual(exchange_count, 2)
            self.assertTrue((archive_root / "concurrent-empty-directory").is_dir())

    def test_target_identity_swap_is_retained_and_never_recursively_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, preflight, launch = self._ready_sources(root)
            archive_root = root / "archive"
            build_trainer_archive(
                out_dir=archive_root,
                preflight_path=preflight,
                launch_check_path=launch,
                require_self_contained=True,
            )
            displaced = root / "original-moved-by-writer"
            alien = root / "alien"
            alien.mkdir()
            (alien / "unowned.txt").write_text("keep\n", encoding="utf-8")
            real_exchange = trainer_archive_module._atomic_exchange_directories

            def swap_target_then_exchange(staging, target):
                Path(target).rename(displaced)
                alien.rename(target)
                real_exchange(staging, target)

            with (
                patch(
                    "flightrecorder.trainer_archive._atomic_exchange_directories",
                    side_effect=swap_target_then_exchange,
                ),
                self.assertRaisesRegex(
                    TrainerArchiveError, "publication attestation failed"
                ),
            ):
                build_trainer_archive(
                    out_dir=archive_root,
                    preflight_path=preflight,
                    launch_check_path=launch,
                    require_self_contained=True,
                    force=True,
                )

            retained = list(root.glob(".archive.hfr-stage-*"))
            self.assertEqual(len(retained), 1)
            self.assertEqual(
                (retained[0] / "unowned.txt").read_text(encoding="utf-8"),
                "keep\n",
            )
            self.assertTrue((displaced / "trainer_archive.json").is_file())

    def test_displaced_archive_swap_after_descriptor_open_retains_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, preflight, launch = self._ready_sources(root)
            archive_root = root / "archive"
            build_trainer_archive(
                out_dir=archive_root,
                preflight_path=preflight,
                launch_check_path=launch,
                require_self_contained=True,
            )
            displaced = root / "displaced-owned-archive"
            alien = root / "alien"
            alien.mkdir()
            (alien / "unowned.txt").write_text("keep\n", encoding="utf-8")
            original_remove = path_safety_module._remove_directory_contents_fd
            swapped = False

            def swap_after_descriptor_open(
                directory_descriptor: int, directory_flags: int
            ) -> bool:
                nonlocal swapped
                if not swapped:
                    swapped = True
                    staging = next(root.glob(".archive.hfr-stage-*"))
                    staging.rename(displaced)
                    alien.rename(staging)
                return original_remove(directory_descriptor, directory_flags)

            with (
                patch(
                    "flightrecorder.path_safety._remove_directory_contents_fd",
                    side_effect=swap_after_descriptor_open,
                ),
                self.assertRaisesRegex(
                    TrainerArchiveError, "displaced archive changed and was retained"
                ),
            ):
                build_trainer_archive(
                    out_dir=archive_root,
                    preflight_path=preflight,
                    launch_check_path=launch,
                    require_self_contained=True,
                    force=True,
                )

            self.assertTrue(swapped)
            retained = list(root.glob(".archive.hfr-stage-*"))
            self.assertEqual(len(retained), 1)
            self.assertEqual(
                (retained[0] / "unowned.txt").read_text(encoding="utf-8"),
                "keep\n",
            )
            self.assertTrue(displaced.is_dir())
            self.assertEqual(list(displaced.iterdir()), [])

    def test_new_target_appearing_at_publication_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, preflight, launch = self._ready_sources(root)
            archive_root = root / "archive"
            real_publish_new = trainer_archive_module._atomic_rename_new_directory

            def create_unowned_target_then_publish(staging, target):
                Path(target).mkdir()
                (Path(target) / "unowned.txt").write_text(
                    "keep\n", encoding="utf-8"
                )
                real_publish_new(staging, target)

            with (
                patch(
                    "flightrecorder.trainer_archive._atomic_rename_new_directory",
                    side_effect=create_unowned_target_then_publish,
                ),
                self.assertRaisesRegex(
                    TrainerArchiveError, "atomic directory publication failed"
                ),
            ):
                build_trainer_archive(
                    out_dir=archive_root,
                    preflight_path=preflight,
                    launch_check_path=launch,
                    require_self_contained=True,
                )

            self.assertEqual(
                (archive_root / "unowned.txt").read_text(encoding="utf-8"),
                "keep\n",
            )
            self.assertEqual(list(root.glob(".archive.hfr-stage-*")), [])

    def test_invalid_staged_content_is_rejected_before_replacing_previous_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, preflight, launch = self._ready_sources(root)
            archive_root = root / "archive"
            first = build_trainer_archive(
                out_dir=archive_root,
                preflight_path=preflight,
                launch_check_path=launch,
                require_self_contained=True,
            )
            self.assertTrue(first["passed"])
            before = _tree_snapshot(archive_root)
            original_write = trainer_archive_module._write_json

            def write_then_corrupt(path, value):
                original_write(path, value)
                manifest_path = Path(path)
                if manifest_path.name == "trainer_archive.json":
                    artifact_path = manifest_path.parent / value["artifacts"][0]["path"]
                    artifact_path.write_text("corrupt staged bytes\n", encoding="utf-8")

            with (
                patch(
                    "flightrecorder.trainer_archive._write_json",
                    side_effect=write_then_corrupt,
                ),
                self.assertRaisesRegex(
                    TrainerArchiveError,
                    "staged trainer archive failed content validation",
                ),
            ):
                build_trainer_archive(
                    out_dir=archive_root,
                    preflight_path=preflight,
                    launch_check_path=launch,
                    require_self_contained=True,
                    force=True,
                )

            self.assertEqual(_tree_snapshot(archive_root), before)
            self.assertEqual(list(root.glob(".archive.hfr-stage-*")), [])
            self.assertEqual(list(root.glob(".archive.hfr-backup-*")), [])

    def test_concurrent_writer_fails_safely_while_target_lock_is_held(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, preflight, launch = self._ready_sources(root)
            archive_root = root / "archive"

            with trainer_archive_module._archive_lock(archive_root):
                with self.assertRaisesRegex(
                    TrainerArchiveError, "publication is locked"
                ):
                    build_trainer_archive(
                        out_dir=archive_root,
                        preflight_path=preflight,
                        launch_check_path=launch,
                        require_self_contained=True,
                    )

            self.assertFalse(archive_root.exists())
            self.assertEqual(list(root.glob(".archive.hfr-stage-*")), [])
            self.assertEqual(list(root.glob(".archive.hfr-backup-*")), [])

    def test_output_path_must_not_traverse_symlink_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, preflight, launch = self._ready_sources(root)
            real_output = root / "real-output"
            real_output.mkdir()
            linked_output = root / "linked-output"
            try:
                linked_output.symlink_to(real_output, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink creation unavailable: {exc}")

            with self.assertRaisesRegex(
                TrainerArchiveError, "output must not traverse a symlink component"
            ):
                build_trainer_archive(
                    out_dir=linked_output,
                    preflight_path=preflight,
                    launch_check_path=launch,
                )

            self.assertEqual(list(real_output.iterdir()), [])

    def test_force_rejects_deleting_current_working_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, preflight, launch = self._ready_sources(root)
            _write_json(
                root / "trainer_archive.json",
                {"schema_version": "hfr.trainer_archive.v1"},
            )
            (root / ".hfr-trainer-archive").write_text(
                "hfr.trainer_archive.v1\n", encoding="utf-8"
            )
            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                with self.assertRaisesRegex(
                    TrainerArchiveError, "protected working directory"
                ):
                    build_trainer_archive(
                        out_dir=Path("."),
                        preflight_path=preflight,
                        launch_check_path=launch,
                        force=True,
                    )
            finally:
                os.chdir(previous_cwd)

            self.assertTrue((root / "trainer_archive.json").is_file())


if __name__ == "__main__":
    unittest.main()
