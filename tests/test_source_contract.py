import hashlib
import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from flightrecorder import source_contract, validation
from flightrecorder.agentic_training_loop_plan import PHASES
from flightrecorder.cloud_training_completion import (
    build_cloud_training_completion_receipt,
    write_cloud_training_completion_receipt,
)
from flightrecorder.reviewed_gate import REVIEWED_EXPORT_CONTENT_FILES
from flightrecorder.schema_registry import check_schema_contract
from flightrecorder.source_contract import (
    _DIRECTORY_MANIFESTS,
    _GATE_CONTRACT_ROLES,
    _SEMANTIC_VALIDATOR_NAMES,
    inspect_artifact_source,
)
from tests.agentic_loop_fixtures import (
    _write_candidate_training_result,
    copy_valid_loop_artifacts,
    write_cloud_completion_fixture,
)
from tests.test_review_calibration import (
    make_reviewed_export,
    read_jsonl,
    run_cli as run_review_calibration_cli,
)


ROOT = Path(__file__).resolve().parents[1]


def make_large_reviewed_export(tmp: str) -> Path:
    """Build a valid reviewed export with JSONL data above the control-plane limit."""
    root = Path(tmp)
    make_reviewed_export(tmp)
    review = root / "review"
    labels_path = root / "completed_labels.jsonl"
    rows = read_jsonl(labels_path)
    rows[0]["notes"] = "x" * (source_contract._MAX_SEMANTIC_SNAPSHOT_FILE_BYTES + 1024)
    labels_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    reviewed = root / "large_reviewed"
    code = run_review_calibration_cli(
        [
            "apply-review",
            "--review-export",
            str(review),
            "--labels",
            str(labels_path),
            "--out",
            str(reviewed),
        ]
    )
    if code != 0:
        raise AssertionError(f"large reviewed export setup failed with exit code {code}")
    return reviewed


class SourceContractTests(unittest.TestCase):
    def test_typed_training_output_uses_bounded_digest_only_attestation(self):
        self.assertEqual(source_contract.MAX_OPAQUE_TRAINING_OUTPUT_FILES, 32)
        self.assertEqual(
            source_contract.MAX_OPAQUE_TRAINING_OUTPUT_BYTES,
            8 * 1024 * 1024 * 1024,
        )
        self.assertEqual(
            source_contract.MAX_OPAQUE_TRAINING_OUTPUT_TOTAL_BYTES,
            32 * 1024 * 1024 * 1024,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = _write_candidate_training_result(root, "local/mock-candidate")
            result, adapter_path = self._enlarge_training_artifact(
                result_path,
                "adapter",
            )

            snapshot = source_contract._capture_private_semantic_snapshot(
                result_path,
                result,
            )
            self.assertIsNotNone(snapshot)
            try:
                opaque = {
                    relative: (role, attestation)
                    for relative, role, attestation in snapshot.tree.opaque_files
                }
                relative = adapter_path.relative_to(root).as_posix()
                self.assertIn(relative, opaque)
                role, attestation = opaque[relative]
                self.assertEqual(role, "adapter")
                self.assertEqual(attestation.content, b"")
                self.assertIsNone(attestation.spool_path)
                self.assertNotIn(relative, dict(snapshot.tree.files))
            finally:
                snapshot.close()

            observed: list[source_contract.OpaqueOutputAttestation | None] = []
            contract_validator = source_contract._semantic_contract_valid

            def observe_attestation(path: Path, role: str) -> bool:
                if role == "agentic_training_result":
                    observed.append(
                        source_contract.get_active_opaque_output_attestation(
                            path.parent / adapter_path.relative_to(root)
                        )
                    )
                return contract_validator(path, role)

            with patch(
                "flightrecorder.source_contract._semantic_contract_valid",
                side_effect=observe_attestation,
            ):
                inspection = inspect_artifact_source(
                    result_path,
                    "agentic_training_result",
                )
            self.assertTrue(inspection["ready"], inspection)
            self.assertTrue(observed)
            self.assertTrue(all(item is not None for item in observed))
            self.assertIsNone(
                source_contract.get_active_opaque_output_attestation(adapter_path)
            )

    def test_typed_training_output_replacement_during_validation_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = _write_candidate_training_result(root, "local/mock-candidate")
            _result, adapter_path = self._enlarge_training_artifact(
                result_path,
                "adapter",
            )
            contract_validator = source_contract._semantic_contract_valid

            def replace_output(path: Path, role: str) -> bool:
                replacement = adapter_path.with_suffix(".replacement")
                replacement.write_bytes(adapter_path.read_bytes())
                replacement.replace(adapter_path)
                return contract_validator(path, role)

            with patch(
                "flightrecorder.source_contract._semantic_contract_valid",
                side_effect=replace_output,
            ):
                inspection = inspect_artifact_source(
                    result_path,
                    "agentic_training_result",
                )

            self.assertTrue(inspection["schema_valid"])
            self.assertFalse(inspection["stable"])
            self.assertFalse(inspection["semantic_valid"])
            self.assertFalse(inspection["ready"])

    def test_unrelated_large_training_result_reference_keeps_generic_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = _write_candidate_training_result(root, "local/mock-candidate")
            self._enlarge_training_artifact(result_path, "config")

            inspection = inspect_artifact_source(
                result_path,
                "agentic_training_result",
            )

            self.assertTrue(inspection["schema_valid"])
            self.assertFalse(inspection["semantic_valid"])
            self.assertFalse(inspection["ready"])

    def test_typed_raw_provider_result_uses_existing_64_mib_opaque_cap(self):
        self.assertEqual(
            source_contract.MAX_OPAQUE_RAW_PROVIDER_RESULT_BYTES,
            64 * 1024 * 1024,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = _write_candidate_training_result(root, "local/mock-candidate")
            completion_path = write_cloud_completion_fixture(
                root,
                result_path,
                "local/mock-candidate",
            )
            cloud_root = root / "cloud_training"
            raw_path = cloud_root / "raw_provider_result.json"
            raw_path.write_bytes(
                b"opaque-provider-result\n"
                + b"x" * source_contract._MAX_SEMANTIC_SNAPSHOT_FILE_BYTES
            )
            runner_path = cloud_root / "runner_metadata.json"
            runner = json.loads(runner_path.read_text(encoding="utf-8"))
            runner["source_sha256"]["raw_provider_result"] = hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest()
            runner_path.write_text(
                json.dumps(runner, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            completion_path.unlink()
            receipt = build_cloud_training_completion_receipt(
                launch_plan_path=cloud_root / "launch_plan.json",
                launch_receipt_path=cloud_root / "launch_receipt.json",
                status_receipt_path=cloud_root / "status_receipt.json",
                runner_metadata_path=runner_path,
                raw_provider_result_path=raw_path,
                output_artifact_manifest_path=result_path,
                out_path=completion_path,
                created_at="2026-07-03T00:30:00+00:00",
            )
            write_cloud_training_completion_receipt(receipt, completion_path)

            snapshot = source_contract._capture_private_semantic_snapshot(
                completion_path,
                dict(receipt),
            )
            self.assertIsNotNone(snapshot)
            try:
                raw_attestations = [
                    attestation
                    for _relative, role, attestation in snapshot.tree.opaque_files
                    if role == "raw_provider_result"
                ]
                self.assertEqual(len(raw_attestations), 1)
                self.assertEqual(raw_attestations[0].content, b"")
                self.assertIsNone(raw_attestations[0].spool_path)
            finally:
                snapshot.close()

            inspection = inspect_artifact_source(
                completion_path,
                "cloud_training_completion_receipt",
            )
            self.assertTrue(inspection["ready"], inspection)

    def _enlarge_training_artifact(
        self,
        result_path: Path,
        role: str,
    ) -> tuple[dict[str, object], Path]:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        row = next(item for item in result["artifacts"] if item["role"] == role)
        artifact_path = result_path.parent / row["path"]
        artifact_path.write_bytes(
            b"x" * (source_contract._MAX_SEMANTIC_SNAPSHOT_FILE_BYTES + 1024)
        )
        content = artifact_path.read_bytes()
        row["sha256"] = hashlib.sha256(content).hexdigest()
        row["size_bytes"] = len(content)
        for link in result["registry_update"]["links"]:
            if link.get("path") == row["path"]:
                link["sha256"] = row["sha256"]
                link["size_bytes"] = row["size_bytes"]
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return result, artifact_path

    def test_reviewed_export_snapshot_cap_allows_large_jsonl_but_stays_bounded(self):
        cap = 16 * 1024 * 1024
        self.assertEqual(
            source_contract._MAX_REVIEWED_EXPORT_SNAPSHOT_BYTES,
            cap,
        )
        self.assertGreater(
            source_contract._MAX_REVIEWED_EXPORT_SNAPSHOT_BYTES,
            source_contract._MAX_SEMANTIC_SNAPSHOT_FILE_BYTES,
        )
        store = source_contract._ReviewedExportSpoolStore()
        try:
            self.assertIsNone(store.allocate(cap + 1))
            self.assertEqual(store.aggregate_bytes, 0)
            self.assertIsNone(store.directory)
            self.assertIsNotNone(store.allocate(cap))
            self.assertIsNone(store.allocate(1))
        finally:
            store.close()

    def test_training_export_tree_mutation_during_validation_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            export_path = Path(tmp) / "training_export"
            shutil.copytree(ROOT / "examples" / "agentic_training" / "training_export", export_path)
            untracked_path = export_path / "untracked-for-attestation.txt"
            untracked_path.write_text("before\n", encoding="utf-8")
            contract_validator = source_contract._training_export_contract_valid
            initial = inspect_artifact_source(export_path, "training_export")
            self.assertTrue(initial["stable"])
            self.assertTrue(initial["ready"])

            def mutate_tree(path: Path) -> bool:
                untracked_path.write_text("after!\n", encoding="utf-8")
                return contract_validator(path)

            with patch(
                "flightrecorder.source_contract._training_export_contract_valid",
                side_effect=mutate_tree,
            ):
                source = inspect_artifact_source(export_path, "training_export")

            self.assertTrue(source["schema_valid"])
            self.assertTrue(source["manifest"]["stable"])
            self.assertFalse(source["stable"])
            self.assertFalse(source["semantic_valid"])
            self.assertFalse(source["ready"])

    def test_promotion_archive_tree_mutation_during_validation_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "promotion_archive"
            shutil.copytree(
                ROOT / "examples" / "agentic_training" / "promotion_governance" / "promotion_archive",
                archive_path,
            )
            untracked_path = archive_path / "untracked-for-attestation"
            untracked_path.write_text("before\n", encoding="utf-8")
            contract_validator = source_contract._directory_contract_valid
            initial = inspect_artifact_source(archive_path, "promotion_archive")
            self.assertTrue(initial["stable"])
            self.assertTrue(initial["ready"])

            def mutate_tree(path: Path, role: str) -> bool:
                untracked_path.unlink()
                untracked_path.mkdir()
                return contract_validator(path, role)

            with patch(
                "flightrecorder.source_contract._directory_contract_valid",
                side_effect=mutate_tree,
            ):
                source = inspect_artifact_source(archive_path, "promotion_archive")

            self.assertTrue(source["schema_valid"])
            self.assertTrue(source["manifest"]["stable"])
            self.assertFalse(source["stable"])
            self.assertFalse(source["semantic_valid"])
            self.assertFalse(source["ready"])

    def test_json_source_rewrite_during_semantic_validation_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate_path = Path(tmp) / "reviewed_gate.json"
            shutil.copyfile(
                ROOT / "examples" / "agentic_training" / "model_grader" / "reviewed_gate.json",
                gate_path,
            )
            semantic_validator = source_contract._semantic_contract_valid

            def rewrite_source(path: Path, role: str) -> bool:
                gate_path.write_bytes(gate_path.read_bytes() + b"\n")
                return semantic_validator(path, role)

            with patch(
                "flightrecorder.source_contract._semantic_contract_valid",
                side_effect=rewrite_source,
            ):
                source = inspect_artifact_source(gate_path, "reviewed_gate")

            self.assertTrue(source["schema_valid"])
            self.assertFalse(source["stable"])
            self.assertFalse(source["semantic_valid"])
            self.assertFalse(source["ready"])

    def test_json_source_replacement_during_semantic_validation_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate_path = root / "reviewed_gate.json"
            replacement_path = root / "replacement.json"
            shutil.copyfile(
                ROOT / "examples" / "agentic_training" / "model_grader" / "reviewed_gate.json",
                gate_path,
            )
            shutil.copyfile(gate_path, replacement_path)
            semantic_validator = source_contract._semantic_contract_valid

            def replace_source(path: Path, role: str) -> bool:
                replacement_path.replace(gate_path)
                return semantic_validator(path, role)

            with patch(
                "flightrecorder.source_contract._semantic_contract_valid",
                side_effect=replace_source,
            ):
                source = inspect_artifact_source(gate_path, "reviewed_gate")

            self.assertTrue(source["schema_valid"])
            self.assertFalse(source["stable"])
            self.assertFalse(source["semantic_valid"])
            self.assertFalse(source["ready"])

    def test_parent_directory_aba_swap_cannot_validate_alternate_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            active = root / "active"
            alternate = root / "alternate"
            admitted_stash = root / "admitted-stash"
            active.mkdir()
            alternate.mkdir()
            gate_name = "reviewed_gate.json"
            valid_gate = json.loads(
                (ROOT / "examples" / "agentic_training" / "model_grader" / gate_name).read_text(
                    encoding="utf-8"
                )
            )
            invalid_gate = json.loads(json.dumps(valid_gate))
            invalid_gate["checks"][0]["passed"] = False
            (active / gate_name).write_text(
                json.dumps(invalid_gate, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (alternate / gate_name).write_text(
                json.dumps(valid_gate, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            gate_path = active / gate_name
            semantic_validator = source_contract._semantic_contract_valid
            validated_paths: list[Path] = []

            def swap_parent_and_validate(path: Path, role: str) -> bool:
                validated_paths.append(path)
                active.rename(admitted_stash)
                alternate.rename(active)
                try:
                    return semantic_validator(path, role)
                finally:
                    active.rename(alternate)
                    admitted_stash.rename(active)

            with patch(
                "flightrecorder.source_contract._semantic_contract_valid",
                side_effect=swap_parent_and_validate,
            ):
                source = inspect_artifact_source(gate_path, "reviewed_gate")

            self.assertTrue(source["schema_valid"])
            self.assertFalse(source["semantic_valid"])
            self.assertFalse(source["ready"])
            self.assertEqual(len(validated_paths), 1)
            self.assertNotEqual(validated_paths[0], gate_path)
            self.assertEqual(
                json.loads(gate_path.read_text(encoding="utf-8"))["checks"][0]["passed"],
                False,
            )

    def test_parent_directory_aba_swap_cannot_validate_alternate_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            active = root / "active"
            alternate = root / "alternate"
            admitted_stash = root / "admitted-stash"
            active.mkdir()
            alternate.mkdir()
            source_name = "model_registry_entry.json"
            source_payload = json.loads(
                (
                    ROOT
                    / "examples"
                    / "agentic_training"
                    / "promotion_governance"
                    / source_name
                ).read_text(encoding="utf-8")
            )
            dependency_before = b'{"ready": false}\n'
            dependency_during_swap = b'{"ready": true}\n'
            dependency_ref = source_payload["links"]["datasets"][0]
            dependency_ref["path"] = "dependency.json"
            dependency_ref["size_bytes"] = len(dependency_before)
            dependency_ref["sha256"] = hashlib.sha256(dependency_before).hexdigest()
            source_bytes = (json.dumps(source_payload, indent=2, sort_keys=True) + "\n").encode()
            (active / source_name).write_bytes(source_bytes)
            (alternate / source_name).write_bytes(source_bytes)
            (active / "dependency.json").write_bytes(dependency_before)
            (alternate / "dependency.json").write_bytes(dependency_during_swap)
            (active / "unrelated-sensitive.txt").write_text("must-not-be-copied\n", encoding="utf-8")
            try:
                (active / "unrelated-link").symlink_to(active / "unrelated-sensitive.txt")
            except (NotImplementedError, OSError):
                pass
            source_path = active / source_name

            snapshot = source_contract._capture_private_semantic_snapshot(source_path, source_payload)
            self.assertIsNotNone(snapshot)
            captured_paths = {path for path, _attestation in snapshot.tree.files}
            self.assertIn("dependency.json", captured_paths)
            self.assertNotIn("unrelated-sensitive.txt", captured_paths)
            self.assertNotIn("unrelated-link", captured_paths)

            def swap_parent_and_validate_dependency(path: Path, _role: str) -> bool:
                active.rename(admitted_stash)
                alternate.rename(active)
                try:
                    dependency = json.loads((path.parent / "dependency.json").read_text(encoding="utf-8"))
                    return dependency.get("ready") is True
                finally:
                    active.rename(alternate)
                    admitted_stash.rename(active)

            with patch(
                "flightrecorder.source_contract._semantic_contract_valid",
                side_effect=swap_parent_and_validate_dependency,
            ):
                source = inspect_artifact_source(source_path, "model_registry_entry")

            self.assertTrue(source["schema_valid"])
            self.assertFalse(source["semantic_valid"])
            self.assertFalse(source["ready"])
            self.assertFalse(
                json.loads((active / "dependency.json").read_text(encoding="utf-8"))["ready"]
            )

    def test_repo_relative_semantic_sources_remain_ready_in_private_snapshot(self):
        cases = (
            (
                ROOT
                / "examples"
                / "agentic_training"
                / "promotion_governance"
                / "model_registry.json",
                "model_registry",
            ),
            (
                ROOT
                / "examples"
                / "agentic_training"
                / "promotion_governance"
                / "model_registry_entry.json",
                "model_registry_entry",
            ),
            (
                ROOT
                / "examples"
                / "agentic_training"
                / "promotion_governance"
                / "promotion_decision.json",
                "promotion_decision",
            ),
            (
                ROOT / "examples" / "agentic_training" / "loop_plan.json",
                "agentic_training_loop_plan",
            ),
        )

        for path, role in cases:
            with self.subTest(path=path, role=role):
                source = inspect_artifact_source(path, role)
                self.assertTrue(source["semantic_valid"], source)
                self.assertTrue(source["stable"], source)
                self.assertTrue(source["ready"], source)
                snapshot = source_contract._capture_private_semantic_snapshot(
                    path,
                    source["payload"],
                )
                self.assertIsNotNone(snapshot)
                captured_paths = {relative for relative, _attestation in snapshot.tree.files}
                self.assertFalse(any(relative.startswith("runs/") for relative in captured_paths))
                self.assertFalse(any(relative.startswith(".omx/") for relative in captured_paths))
                self.assertFalse(any("__pycache__" in Path(relative).parts for relative in captured_paths))

    def test_fixture_nested_refs_expand_only_to_relocated_repo_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            copy_valid_loop_artifacts(root)
            fixture_root = root / "loop_fixture"
            ledger_path = fixture_root / "examples" / "agentic_training" / "loop_ledger.json"

            source = inspect_artifact_source(ledger_path, "agentic_loop_ledger")
            self.assertTrue(source["semantic_valid"], source)
            self.assertTrue(source["ready"], source)

            snapshot = source_contract._capture_private_semantic_snapshot(
                ledger_path,
                source["payload"],
            )
            self.assertIsNotNone(snapshot)
            self.assertEqual(snapshot.boundary_path, fixture_root)
            captured_paths = {relative for relative, _attestation in snapshot.tree.files}
            self.assertIn("examples/promotion_policy.demo.json", captured_paths)
            self.assertTrue(
                all(
                    relative == "pyproject.toml"
                    or relative.startswith("examples/")
                    for relative in captured_paths
                )
            )
            self.assertFalse(any(relative.startswith("../") for relative in captured_paths))

    def test_source_under_runs_can_admit_its_explicit_sibling_closure(self):
        runs_root = ROOT / "runs"
        remove_runs_root = not runs_root.exists()
        runs_root.mkdir(exist_ok=True)
        if remove_runs_root:
            self.addCleanup(runs_root.rmdir)
        with tempfile.TemporaryDirectory(dir=runs_root) as tmp:
            run_root = Path(tmp)
            reviewed = make_reviewed_export(tmp)
            calibration_path = run_root / "review_calibration.json"
            reviewed_relative = reviewed.relative_to(ROOT)
            calibration_relative = calibration_path.relative_to(ROOT)
            self.assertEqual(
                run_review_calibration_cli(
                    [
                        "review-calibration",
                        "--reviewed-export",
                        str(reviewed_relative),
                        "--out",
                        str(calibration_relative),
                        "--min-comparable-labels",
                        "2",
                        "--min-agreement-rate",
                        "1.0",
                        "--max-disagreements",
                        "0",
                        "--preserve-paths",
                    ]
                ),
                0,
            )

            source = inspect_artifact_source(calibration_path, "review_calibration")
            self.assertEqual(source["payload"]["reviewed_export"], "reviewed")
            self.assertTrue(source["semantic_valid"], source)
            self.assertTrue(source["ready"], source)

            snapshot = source_contract._capture_private_semantic_snapshot(
                calibration_path,
                source["payload"],
            )
            self.assertIsNotNone(snapshot)
            captured_paths = {relative for relative, _attestation in snapshot.tree.files}
            self.assertIn((reviewed_relative / "manifest.json").as_posix(), captured_paths)

            review_relative = (run_root / "review").relative_to(ROOT)
            rubric_relative = (run_root / "model_grader_rubric.json").relative_to(ROOT)
            dry_run_relative = (run_root / "model_grader_dry_run.json").relative_to(ROOT)
            gate_path = run_root / "model_grader_gate.json"
            gate_relative = gate_path.relative_to(ROOT)
            self.assertEqual(
                run_review_calibration_cli(
                    [
                        "model-grader",
                        "rubric",
                        "--review-export",
                        str(review_relative),
                        "--rubric-id",
                        "source-contract-rubric",
                        "--out",
                        str(rubric_relative),
                    ]
                ),
                0,
            )
            self.assertEqual(
                run_review_calibration_cli(
                    [
                        "model-grader",
                        "dry-run",
                        "--review-export",
                        str(review_relative),
                        "--rubric",
                        str(rubric_relative),
                        "--grader-id",
                        "source-contract-mock-grader",
                        "--out",
                        str(dry_run_relative),
                    ]
                ),
                0,
            )
            self.assertEqual(
                run_review_calibration_cli(
                    [
                        "model-grader",
                        "gate",
                        "--dry-run",
                        str(dry_run_relative),
                        "--rubric",
                        str(rubric_relative),
                        "--review-calibration",
                        str(calibration_relative),
                        "--min-calibration-agreement-rate",
                        "1.0",
                        "--max-disagreements",
                        "0",
                        "--out",
                        str(gate_relative),
                    ]
                ),
                0,
            )
            gate_source = inspect_artifact_source(gate_path, "model_grader_gate")
            self.assertTrue(gate_source["semantic_valid"], gate_source)
            self.assertTrue(gate_source["ready"], gate_source)

    def test_typed_reviewed_export_spools_large_jsonl_through_downstream_closure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reviewed = make_large_reviewed_export(tmp)
            large_paths = [
                path
                for path in reviewed.rglob("*.jsonl")
                if path.stat().st_size > source_contract._MAX_SEMANTIC_SNAPSHOT_FILE_BYTES
            ]
            self.assertGreaterEqual(len(large_paths), 2)

            reviewed_gate = root / "reviewed_gate.json"
            self.assertEqual(
                run_review_calibration_cli(
                    [
                        "gate-reviewed",
                        "--reviewed-export",
                        str(reviewed),
                        "--out",
                        str(reviewed_gate),
                    ]
                ),
                0,
            )
            calibration = root / "review_calibration.json"
            self.assertEqual(
                run_review_calibration_cli(
                    [
                        "review-calibration",
                        "--reviewed-export",
                        str(reviewed),
                        "--out",
                        str(calibration),
                    ]
                ),
                0,
            )
            rubric = root / "model_grader_rubric.json"
            dry_run = root / "model_grader_dry_run.json"
            model_grader_gate = root / "model_grader_gate.json"
            self.assertEqual(
                run_review_calibration_cli(
                    [
                        "model-grader",
                        "rubric",
                        "--review-export",
                        str(root / "review"),
                        "--rubric-id",
                        "large-reviewed-source-contract",
                        "--out",
                        str(rubric),
                    ]
                ),
                0,
            )
            self.assertEqual(
                run_review_calibration_cli(
                    [
                        "model-grader",
                        "dry-run",
                        "--review-export",
                        str(root / "review"),
                        "--rubric",
                        str(rubric),
                        "--grader-id",
                        "large-reviewed-mock-grader",
                        "--out",
                        str(dry_run),
                    ]
                ),
                0,
            )
            self.assertEqual(
                run_review_calibration_cli(
                    [
                        "model-grader",
                        "gate",
                        "--dry-run",
                        str(dry_run),
                        "--rubric",
                        str(rubric),
                        "--review-calibration",
                        str(calibration),
                        "--out",
                        str(model_grader_gate),
                    ]
                ),
                0,
            )

            direct_source = inspect_artifact_source(reviewed_gate, "reviewed_gate")
            downstream_source = inspect_artifact_source(model_grader_gate, "model_grader_gate")
            self.assertTrue(direct_source["ready"], direct_source)
            self.assertTrue(downstream_source["ready"], downstream_source)

            snapshot = source_contract._capture_private_semantic_snapshot(
                reviewed_gate,
                direct_source["payload"],
            )
            self.assertIsNotNone(snapshot)
            self.assertIsNotNone(snapshot.spool_store)
            self.assertIsNotNone(snapshot.spool_store.directory)
            spool_root = Path(snapshot.spool_store.directory.name)
            try:
                spooled = {
                    relative
                    for relative, attestation in snapshot.tree.files
                    if attestation.spool_path is not None
                }
                self.assertEqual(
                    spooled,
                    {
                        f"large_reviewed/{relative}"
                        for relative in REVIEWED_EXPORT_CONTENT_FILES
                        if relative.endswith(".jsonl")
                    },
                )
                self.assertEqual(len(snapshot.tree.exact_directory_entries), 2)
            finally:
                snapshot.close()
            self.assertFalse(spool_root.exists())

            forged_gate = root / "forged_reviewed_gate.json"
            forged_payload = json.loads(json.dumps(direct_source["payload"]))
            forged_payload["source_artifacts"]["reviewed_export"]["sha256"] = "0" * 64
            forged_gate.write_text(
                json.dumps(forged_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self.assertIsNone(
                source_contract._capture_private_semantic_snapshot(
                    forged_gate,
                    forged_payload,
                )
            )

            reviewed_jsonl_bytes = sum(
                path.stat().st_size for path in reviewed.rglob("*.jsonl")
            )
            self.assertLess(
                reviewed_jsonl_bytes,
                source_contract._MAX_REVIEWED_EXPORT_SNAPSHOT_BYTES,
            )
            with patch.object(
                source_contract,
                "_MAX_REVIEWED_EXPORT_SNAPSHOT_BYTES",
                reviewed_jsonl_bytes - 1,
            ):
                self.assertIsNone(
                    source_contract._capture_private_semantic_snapshot(
                        reviewed_gate,
                        direct_source["payload"],
                    )
                )

            unexpected = reviewed / "unexpected.jsonl"
            unexpected.write_text("{}\n", encoding="utf-8")
            self.assertFalse(inspect_artifact_source(reviewed_gate, "reviewed_gate")["ready"])
            unexpected.unlink()

            semantic_validator = source_contract._semantic_contract_valid

            def mutate_large_source(path: Path, role: str) -> bool:
                with large_paths[0].open("ab") as handle:
                    handle.write(b" ")
                return semantic_validator(path, role)

            with patch.object(
                source_contract,
                "_semantic_contract_valid",
                side_effect=mutate_large_source,
            ):
                mutated = inspect_artifact_source(reviewed_gate, "reviewed_gate")
            self.assertFalse(mutated["stable"], mutated)
            self.assertFalse(mutated["ready"], mutated)

    def test_untyped_large_reference_remains_subject_to_generic_file_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            large = root / "untyped.jsonl"
            with large.open("wb") as handle:
                handle.seek(source_contract._MAX_SEMANTIC_SNAPSHOT_FILE_BYTES)
                handle.write(b"x")
            source_path = root / "source.json"
            payload = {"artifact_paths": [large.name]}
            source_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

            self.assertIsNone(
                source_contract._capture_private_semantic_snapshot(source_path, payload)
            )

    def test_all_loop_readiness_roles_have_semantic_contracts(self):
        required_roles = {role for phase in PHASES for role in phase["required"]}
        downstream_roles = {
            "action_ledger",
            "agentic_loop_ledger",
            "agentic_rollout_plan",
            "agentic_rollout_receipt",
            "agentic_training_runtime_preflight",
            "improvement_ledger",
            "rejection_sampling_gate",
            "review_calibration",
            "reviewed_gate",
            "training_export",
        }
        special_roles = {
            "training_export",
            *_DIRECTORY_MANIFESTS,
            *_GATE_CONTRACT_ROLES,
        }

        self.assertEqual(
            (required_roles | downstream_roles) - set(_SEMANTIC_VALIDATOR_NAMES) - special_roles,
            set(),
        )
        for validator_name in _SEMANTIC_VALIDATOR_NAMES.values():
            self.assertTrue(callable(getattr(validation, validator_name, None)), validator_name)

    def test_runtime_preflight_dispatches_to_full_semantic_validator(self):
        with tempfile.TemporaryDirectory() as tmp:
            examples = Path(tmp) / "examples"
            shutil.copytree(ROOT / "examples", examples)
            preflight_path = examples / "agentic_training" / "runtime_preflight" / "ready.json"
            preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
            preflight["checks"][0]["passed"] = False
            preflight_path.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            schema = check_schema_contract(preflight, name_or_id="agentic_training_runtime_preflight")
            source = inspect_artifact_source(preflight_path, "agentic_training_runtime_preflight")

            self.assertTrue(schema["passed"], schema["errors"])
            self.assertTrue(source["schema_valid"])
            self.assertFalse(source["semantic_valid"])
            self.assertFalse(source["ready"])

    def test_schema_valid_reviewed_gate_with_forged_nested_check_is_not_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate_path = Path(tmp) / "reviewed_gate.json"
            gate = json.loads(
                (ROOT / "examples" / "agentic_training" / "model_grader" / "reviewed_gate.json").read_text(
                    encoding="utf-8"
                )
            )
            gate["checks"][0]["passed"] = False
            gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            schema = check_schema_contract(gate, name_or_id="reviewed_gate")
            source = inspect_artifact_source(gate_path, "reviewed_gate")

            self.assertTrue(schema["passed"], schema["errors"])
            self.assertTrue(source["schema_valid"])
            self.assertFalse(source["semantic_valid"])
            self.assertFalse(source["ready"])

    def test_schema_valid_rollout_plan_with_forged_nested_check_is_not_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rollout_root = root / "rollouts"
            shutil.copytree(ROOT / "examples" / "agentic_training" / "rollouts", rollout_root)
            plan_path = rollout_root / "rollout_plan.json"
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan["checks"][0]["passed"] = False
            plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            schema = check_schema_contract(plan, name_or_id="agentic_rollout_plan")
            source = inspect_artifact_source(plan_path, "agentic_rollout_plan")

            self.assertTrue(schema["passed"], schema["errors"])
            self.assertTrue(source["schema_valid"])
            self.assertFalse(source["semantic_valid"])
            self.assertFalse(source["ready"])

    def test_schema_valid_action_ledger_with_forged_metrics_is_not_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            examples = root / "examples"
            shutil.copytree(ROOT / "examples", examples)
            ledger_path = examples / "agentic_training" / "iteration_ledgers" / "action_ledger.json"
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            ledger["metrics"]["action_count"] = 0
            ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            schema = check_schema_contract(ledger, name_or_id="action_ledger")
            source = inspect_artifact_source(ledger_path, "action_ledger")

            self.assertTrue(schema["passed"], schema["errors"])
            self.assertTrue(source["schema_valid"])
            self.assertFalse(source["semantic_valid"])
            self.assertFalse(source["ready"])


class SnapshotResourceBudgetTests(unittest.TestCase):
    @staticmethod
    def write_reference_source(
        root: Path,
        references: list[str],
        *,
        name: str = "s.json",
    ) -> tuple[Path, dict[str, object], bytes]:
        payload: dict[str, object] = {"artifact_paths": references}
        content = (json.dumps(payload, sort_keys=True) + "\n").encode()
        path = root / name
        path.write_bytes(content)
        return path, payload, content

    def test_regular_file_budget_accepts_exact_limit_and_rejects_regular_and_sparse_plus_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exact = root / "exact.bin"
            regular_over = root / "regular-over.bin"
            sparse_over = root / "sparse-over.bin"
            exact.write_bytes(b"x" * 64)
            regular_over.write_bytes(b"x" * 65)
            with sparse_over.open("wb") as handle:
                handle.seek(64)
                handle.write(b"x")

            with patch.object(source_contract, "_MAX_SEMANTIC_SNAPSHOT_FILE_BYTES", 64):
                self.assertIsNotNone(source_contract._attest_regular_file(exact))
                self.assertIsNone(source_contract._attest_regular_file(regular_over))
                self.assertIsNone(source_contract._attest_regular_file(sparse_over))

    def test_aggregate_byte_budget_accepts_exact_limit_and_rejects_plus_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dependencies = {"one.bin": b"1" * 7, "two.bin": b"2" * 11}
            for name, content in dependencies.items():
                (root / name).write_bytes(content)
            source_path, payload, source_content = self.write_reference_source(
                root,
                list(dependencies),
            )
            # The source is read once for pathname admission and once through
            # the descriptor-bound closure; every dependency is read once.
            exact_read_bytes = 2 * len(source_content) + sum(map(len, dependencies.values()))

            with patch.object(
                source_contract,
                "_MAX_SEMANTIC_SNAPSHOT_TOTAL_BYTES",
                exact_read_bytes,
            ):
                self.assertIsNotNone(
                    source_contract._capture_private_semantic_snapshot(source_path, payload)
                )
            with patch.object(
                source_contract,
                "_MAX_SEMANTIC_SNAPSHOT_TOTAL_BYTES",
                exact_read_bytes - 1,
            ):
                self.assertIsNone(
                    source_contract._capture_private_semantic_snapshot(source_path, payload)
                )

    def test_unique_file_budget_accepts_exact_limit_and_rejects_plus_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(3):
                (root / f"dep-{index}").write_bytes(bytes([index]))
            source_path, exact_payload, _ = self.write_reference_source(
                root,
                ["dep-0", "dep-1"],
            )
            with patch.object(source_contract, "_MAX_SEMANTIC_SNAPSHOT_FILES", 3):
                self.assertIsNotNone(
                    source_contract._capture_private_semantic_snapshot(
                        source_path,
                        exact_payload,
                    )
                )
                source_path, over_payload, _ = self.write_reference_source(
                    root,
                    ["dep-0", "dep-1", "dep-2"],
                )
                self.assertIsNone(
                    source_contract._capture_private_semantic_snapshot(source_path, over_payload)
                )

    def test_directory_count_budget_accepts_exact_limit_and_rejects_plus_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tree" / "a" / "b").mkdir(parents=True)
            source_path, payload, _ = self.write_reference_source(root, ["tree"])
            with patch.object(source_contract, "_MAX_SEMANTIC_SNAPSHOT_DIRECTORIES", 4):
                self.assertIsNotNone(
                    source_contract._capture_private_semantic_snapshot(source_path, payload)
                )
                (root / "tree" / "a" / "b" / "c").mkdir()
                self.assertIsNone(
                    source_contract._capture_private_semantic_snapshot(source_path, payload)
                )

    def test_recursion_depth_budget_accepts_exact_limit_and_rejects_plus_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tree" / "a" / "b").mkdir(parents=True)
            source_path, payload, _ = self.write_reference_source(root, ["tree"])
            with patch.object(source_contract, "_MAX_SEMANTIC_SNAPSHOT_DEPTH", 3):
                self.assertIsNotNone(
                    source_contract._capture_private_semantic_snapshot(source_path, payload)
                )
                (root / "tree" / "a" / "b" / "c").mkdir()
                self.assertIsNone(
                    source_contract._capture_private_semantic_snapshot(source_path, payload)
                )

    def test_directory_entry_budget_counts_skipped_special_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tree = root / "tree"
            tree.mkdir()
            try:
                for index in range(3):
                    (tree / f"link-{index}").symlink_to("missing")
            except (NotImplementedError, OSError) as error:
                self.skipTest(f"symlinks unavailable: {error}")
            source_path, payload, _ = self.write_reference_source(root, ["tree"])

            with patch.object(
                source_contract,
                "_MAX_SEMANTIC_SNAPSHOT_DIRECTORY_ENTRIES",
                3,
            ):
                self.assertIsNotNone(
                    source_contract._capture_private_semantic_snapshot(source_path, payload)
                )
                (tree / "link-3").symlink_to("missing")
                self.assertIsNone(
                    source_contract._capture_private_semantic_snapshot(source_path, payload)
                )

    def test_reference_count_and_resolution_step_budgets_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_path, exact_payload, _ = self.write_reference_source(
                root,
                ["missing-0", "missing-1", "missing-2"],
            )
            with patch.object(source_contract, "_MAX_SEMANTIC_SNAPSHOT_REFERENCES", 3):
                self.assertIsNotNone(
                    source_contract._capture_private_semantic_snapshot(
                        source_path,
                        exact_payload,
                    )
                )
                source_path, over_payload, _ = self.write_reference_source(
                    root,
                    ["missing-0", "missing-1", "missing-2", "missing-3"],
                )
                self.assertIsNone(
                    source_contract._capture_private_semantic_snapshot(source_path, over_payload)
                )

            source_path, step_payload, _ = self.write_reference_source(root, ["a/b"])
            with patch.object(
                source_contract,
                "_MAX_SEMANTIC_SNAPSHOT_REFERENCE_RESOLUTION_STEPS",
                3,
            ):
                self.assertIsNotNone(
                    source_contract._capture_private_semantic_snapshot(source_path, step_payload)
                )
            with patch.object(
                source_contract,
                "_MAX_SEMANTIC_SNAPSHOT_REFERENCE_RESOLUTION_STEPS",
                2,
            ):
                self.assertIsNone(
                    source_contract._capture_private_semantic_snapshot(source_path, step_payload)
                )

    def test_reference_text_budgets_cover_raw_redacted_and_component_amplification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(source_contract, "_MAX_SEMANTIC_SNAPSHOT_REFERENCE_CHARS", 16):
                for reference in ("x" * 16, f"<redacted:{'x' * 16}>"):
                    source_path, payload, _ = self.write_reference_source(root, [reference], name="s")
                    self.assertIsNotNone(
                        source_contract._capture_private_semantic_snapshot(source_path, payload)
                    )
                for reference in ("x" * 17, f"<redacted:{'x' * 17}>"):
                    source_path, payload, _ = self.write_reference_source(root, [reference], name="s")
                    self.assertIsNone(
                        source_contract._capture_private_semantic_snapshot(source_path, payload)
                    )

            with patch.object(
                source_contract,
                "_MAX_SEMANTIC_SNAPSHOT_REFERENCE_COMPONENTS",
                2,
            ):
                source_path, payload, _ = self.write_reference_source(root, ["a/b"], name="s")
                self.assertIsNotNone(
                    source_contract._capture_private_semantic_snapshot(source_path, payload)
                )
                source_path, payload, _ = self.write_reference_source(root, ["a/b/c"], name="s")
                self.assertIsNone(
                    source_contract._capture_private_semantic_snapshot(source_path, payload)
                )

    def test_json_lexical_budgets_reject_depth_and_nodes_before_parsing(self):
        with patch.object(source_contract, "_MAX_SEMANTIC_SNAPSHOT_JSON_DEPTH", 3):
            self.assertTrue(source_contract._json_bytes_within_lexical_budgets(b"[[[]]]"))
            self.assertFalse(source_contract._json_bytes_within_lexical_budgets(b"[[[[]]]]"))
        with patch.object(source_contract, "_MAX_SEMANTIC_SNAPSHOT_JSON_NODES", 5):
            self.assertTrue(source_contract._json_bytes_within_lexical_budgets(b"[0,0,0,0]"))
            self.assertFalse(source_contract._json_bytes_within_lexical_budgets(b"[0,0,0,0,0]"))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            path.write_text('{"a": 0, "b": 0}\n', encoding="utf-8")
            with (
                patch.object(source_contract, "_MAX_SEMANTIC_SNAPSHOT_JSON_NODES", 2),
                patch.object(source_contract.json, "loads", wraps=json.loads) as json_loads,
            ):
                source = source_contract.inspect_json_source(
                    path,
                    "reviewed_gate",
                    require_semantics=False,
                )
            self.assertFalse(source["parse_valid"])
            json_loads.assert_not_called()

    def test_overlapping_child_then_parent_directory_references_do_not_reread_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tree" / "child").mkdir(parents=True)
            child_content = b"child-data"
            parent_content = b"parent-data"
            (root / "tree" / "child" / "child.bin").write_bytes(child_content)
            (root / "tree" / "parent.bin").write_bytes(parent_content)
            # Reference iteration is LIFO, so this deliberately captures the
            # child before its parent and exercises overlap de-duplication.
            source_path, payload, source_content = self.write_reference_source(
                root,
                ["tree", "tree/child"],
            )
            exact_read_bytes = (
                2 * len(source_content) + len(child_content) + len(parent_content)
            )
            with patch.object(
                source_contract,
                "_MAX_SEMANTIC_SNAPSHOT_TOTAL_BYTES",
                exact_read_bytes,
            ):
                snapshot = source_contract._capture_private_semantic_snapshot(
                    source_path,
                    payload,
                )

            self.assertIsNotNone(snapshot)
            self.assertEqual(
                [relative for relative, _attestation in snapshot.tree.files],
                ["s.json", "tree/child/child.bin", "tree/parent.bin"],
            )

    def test_materialization_revalidates_path_budget_and_source_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_path, payload, _ = self.write_reference_source(root, [])
            snapshot = source_contract._capture_private_semantic_snapshot(source_path, payload)
            self.assertIsNotNone(snapshot)

            escaped = replace(snapshot, source_relative_path=Path("../escape.json"))
            with self.assertRaises(ValueError):
                with source_contract._materialize_private_semantic_snapshot(escaped):
                    self.fail("unsafe snapshot path was materialized")

            mismatched = replace(
                snapshot,
                source_attestation=replace(snapshot.source_attestation, sha256="0" * 64),
            )
            with self.assertRaises(ValueError):
                with source_contract._materialize_private_semantic_snapshot(mismatched):
                    self.fail("unbound source attestation was materialized")

            with (
                patch.object(
                    source_contract,
                    "_MAX_SEMANTIC_SNAPSHOT_FILE_BYTES",
                    len(snapshot.source_attestation.content) - 1,
                ),
                self.assertRaises(ValueError),
            ):
                with source_contract._materialize_private_semantic_snapshot(snapshot):
                    self.fail("over-budget source was materialized")

    def test_outer_artifact_directories_fail_closed_when_tree_exceeds_budget(self):
        cases = (
            (
                ROOT / "examples" / "agentic_training" / "training_export",
                "training_export",
            ),
            (
                ROOT
                / "examples"
                / "agentic_training"
                / "promotion_governance"
                / "promotion_archive",
                "promotion_archive",
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            for source_path, role in cases:
                with self.subTest(role=role):
                    artifact_path = Path(tmp) / role
                    shutil.copytree(source_path, artifact_path)
                    with patch.object(source_contract, "_MAX_SEMANTIC_SNAPSHOT_FILES", 1):
                        source = inspect_artifact_source(artifact_path, role)
                    self.assertFalse(source["ready"], source)
                    self.assertFalse(source["stable"], source)

    def test_committed_control_plane_sources_remain_within_snapshot_budgets(self):
        cases = (
            (
                ROOT / "examples" / "agentic_training" / "loop_plan.json",
                "agentic_training_loop_plan",
            ),
            (
                ROOT
                / "examples"
                / "agentic_training"
                / "promotion_governance"
                / "model_registry.json",
                "model_registry",
            ),
            (
                ROOT / "examples" / "agentic_training" / "training_export",
                "training_export",
            ),
            (
                ROOT
                / "examples"
                / "agentic_training"
                / "model_grader"
                / "passing_gate.json",
                "model_grader_gate",
            ),
        )
        for path, role in cases:
            with self.subTest(role=role):
                source = inspect_artifact_source(path, role)
                self.assertTrue(source["ready"], source)
                self.assertTrue(source["stable"], source)


if __name__ == "__main__":
    unittest.main()
