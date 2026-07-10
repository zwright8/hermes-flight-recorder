import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from flightrecorder import source_contract, validation
from flightrecorder.agentic_training_loop_plan import PHASES
from flightrecorder.schema_registry import check_schema_contract
from flightrecorder.source_contract import (
    _DIRECTORY_MANIFESTS,
    _GATE_CONTRACT_ROLES,
    _SEMANTIC_VALIDATOR_NAMES,
    inspect_artifact_source,
)
from tests.agentic_loop_fixtures import copy_valid_loop_artifacts
from tests.test_review_calibration import make_reviewed_export, run_cli as run_review_calibration_cli


ROOT = Path(__file__).resolve().parents[1]


class SourceContractTests(unittest.TestCase):
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

    def test_fixture_nested_refs_expand_only_to_fixture_root(self):
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
            self.assertFalse(any(relative.startswith("../") for relative in captured_paths))

    def test_source_under_runs_can_admit_its_explicit_sibling_closure(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as tmp:
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


if __name__ == "__main__":
    unittest.main()
