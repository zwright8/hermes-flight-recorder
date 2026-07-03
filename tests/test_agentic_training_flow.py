import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from flightrecorder.agentic_training_flow import build_agentic_training_flow, write_agentic_training_flow
from flightrecorder.agentic_training_runtime import build_agentic_training_runtime_preflight, write_agentic_training_runtime_preflight
from flightrecorder.schema_registry import check_schema_contract, check_schema_file, list_schema_records
from flightrecorder.trainer_consumer_plan import build_trainer_consumer_plan
from flightrecorder.validation import validate_artifacts


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PLAN = ROOT / "examples" / "agentic_training" / "plans" / "sft_then_dpo_plan.json"


class AgenticTrainingFlowTests(unittest.TestCase):
    def test_ready_sft_then_dpo_flow_delegates_without_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = self.write_runtime_preflight(root)
            consumer = self.write_trainer_consumer_plan(root)
            out = root / "agentic_training_flow.json"

            receipt = build_agentic_training_flow(
                plan_path=EXAMPLE_PLAN,
                runtime_preflight_path=runtime,
                trainer_consumer_plan_path=consumer,
                out_path=out,
                flow_id="flow-sft-dpo",
                created_at="2026-07-03T00:00:00+00:00",
            )
            write_agentic_training_flow(out, receipt)

            self.assertTrue(receipt["passed"], receipt["blocked_reasons"])
            self.assertEqual(receipt["recommendation"], "ready_for_delegated_trainer_execution")
            self.assertEqual(receipt["delegated_flow"]["mode"], "sft_then_dpo")
            self.assertEqual(receipt["delegated_flow"]["stage_sequence"], ["sft", "dpo"])
            self.assertEqual({stage["stage_id"] for stage in receipt["delegated_flow"]["stages"]}, {"sft", "dpo"})
            self.assertFalse(receipt["execution_boundary"]["trainer_command_executed"])
            self.assertFalse(receipt["execution_boundary"]["weights_updated_by_flight_recorder"])
            schema = check_schema_file(out)
            self.assertTrue(schema["passed"], schema["errors"])
            validation = validate_artifacts(agentic_training_flow_paths=[out], strict=True)
            self.assertTrue(validation["passed"], validation)

    def test_relative_inputs_are_written_relative_to_receipt(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            runtime = self.write_runtime_preflight(root)
            consumer = self.write_trainer_consumer_plan(root)
            out = root / "flow.json"
            old_cwd = Path.cwd()
            try:
                os.chdir(ROOT)
                receipt = build_agentic_training_flow(
                    plan_path=EXAMPLE_PLAN.relative_to(ROOT),
                    runtime_preflight_path=runtime.relative_to(ROOT),
                    trainer_consumer_plan_path=consumer.relative_to(ROOT),
                    out_path=out.relative_to(ROOT),
                    created_at="2026-07-03T00:00:00+00:00",
                )
            finally:
                os.chdir(old_cwd)
            write_agentic_training_flow(out, receipt)

            sources = receipt["source_artifacts"]
            self.assertEqual(
                sources["agentic_training_plan"]["path"],
                "../examples/agentic_training/plans/sft_then_dpo_plan.json",
            )
            self.assertEqual(sources["agentic_training_runtime_preflight"]["path"], "runtime_preflight.json")
            self.assertEqual(sources["trainer_consumer_plan"]["path"], "trainer_consumer_plan.json")
            validation = validate_artifacts(agentic_training_flow_paths=[out], strict=True)
            self.assertTrue(validation["passed"], validation)
            nested = root / "nested"
            nested.mkdir()
            old_cwd = Path.cwd()
            try:
                os.chdir(nested)
                nested_validation = validate_artifacts(agentic_training_flow_paths=[out], strict=True)
            finally:
                os.chdir(old_cwd)
            self.assertTrue(nested_validation["passed"], nested_validation)

    def test_cli_writes_valid_flow_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = self.write_runtime_preflight(root)
            consumer = self.write_trainer_consumer_plan(root)
            out = root / "flow.json"

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "flightrecorder",
                    "agentic-training-flow",
                    "--plan",
                    str(EXAMPLE_PLAN),
                    "--runtime-preflight",
                    str(runtime),
                    "--trainer-consumer-plan",
                    str(consumer),
                    "--flow-id",
                    "cli-flow",
                    "--created-at",
                    "2026-07-03T00:00:00+00:00",
                    "--out",
                    str(out),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            validation = validate_artifacts(agentic_training_flow_paths=[out], strict=True)
            self.assertTrue(validation["passed"], validation)

    def test_reward_mode_plan_is_blocked_at_flow_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = json.loads(EXAMPLE_PLAN.read_text(encoding="utf-8"))
            plan["mode"] = "reward_model"
            plan["trainer_plan"]["stage_sequence"] = ["reward_model"]
            plan["trainer_plan"]["backend"] = "process_reward_wrapper"
            plan["recommendation"] = "ready_for_external_trainer_plan"
            plan_path = root / "reward_plan.json"
            plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            runtime = self.write_runtime_preflight(root, plan_path=plan_path)
            consumer = self.write_trainer_consumer_plan(root)

            receipt = build_agentic_training_flow(
                plan_path=plan_path,
                runtime_preflight_path=runtime,
                trainer_consumer_plan_path=consumer,
                out_path=root / "blocked_flow.json",
                created_at="2026-07-03T00:00:00+00:00",
            )

            self.assertFalse(receipt["passed"])
            self.assertEqual(receipt["recommendation"], "block_delegated_trainer_execution")
            failed_ids = {check["id"] for check in receipt["checks"] if not check["passed"]}
            self.assertIn("default_executable_flow_mode", failed_ids)
            self.assertIn("stage_sequence_executable", failed_ids)
            schema = check_schema_contract(receipt)
            self.assertTrue(schema["passed"], schema["errors"])

    def test_validation_rejects_execution_boundary_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = self.write_runtime_preflight(root)
            consumer = self.write_trainer_consumer_plan(root)
            out = root / "flow.json"
            receipt = build_agentic_training_flow(
                plan_path=EXAMPLE_PLAN,
                runtime_preflight_path=runtime,
                trainer_consumer_plan_path=consumer,
                out_path=out,
                created_at="2026-07-03T00:00:00+00:00",
            )
            receipt["execution_boundary"]["trainer_command_executed"] = True
            write_agentic_training_flow(out, receipt)

            validation = validate_artifacts(agentic_training_flow_paths=[out], strict=True)
            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("execution_boundary.trainer_command_executed must be false", errors)

    def test_validation_rejects_flow_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = self.write_runtime_preflight(root)
            consumer = self.write_trainer_consumer_plan(root)
            out = root / "flow.json"
            receipt = build_agentic_training_flow(
                plan_path=EXAMPLE_PLAN,
                runtime_preflight_path=runtime,
                trainer_consumer_plan_path=consumer,
                out_path=out,
                created_at="2026-07-03T00:00:00+00:00",
            )
            receipt["flow_path"] = "../../../../etc/passwd"
            write_agentic_training_flow(out, receipt)

            validation = validate_artifacts(agentic_training_flow_paths=[out], strict=True)
            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("agentic_training_flow.flow_path must be a safe relative path without traversal", errors)

    def test_validation_rejects_missing_source_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = self.write_runtime_preflight(root)
            consumer = self.write_trainer_consumer_plan(root)
            out = root / "flow.json"
            receipt = build_agentic_training_flow(
                plan_path=EXAMPLE_PLAN,
                runtime_preflight_path=runtime,
                trainer_consumer_plan_path=consumer,
                out_path=out,
                created_at="2026-07-03T00:00:00+00:00",
            )
            receipt["source_artifacts"]["trainer_consumer_plan"]["path"] = "missing_trainer_consumer_plan.json"
            write_agentic_training_flow(out, receipt)

            validation = validate_artifacts(agentic_training_flow_paths=[out], strict=True)
            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("source_artifacts.trainer_consumer_plan.path must resolve to an existing file", errors)

    def test_validation_rejects_stale_source_artifact_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = self.write_runtime_preflight(root)
            consumer = self.write_trainer_consumer_plan(root)
            out = root / "flow.json"
            receipt = build_agentic_training_flow(
                plan_path=EXAMPLE_PLAN,
                runtime_preflight_path=runtime,
                trainer_consumer_plan_path=consumer,
                out_path=out,
                created_at="2026-07-03T00:00:00+00:00",
            )
            write_agentic_training_flow(out, receipt)
            consumer.write_text(consumer.read_text(encoding="utf-8") + "\n", encoding="utf-8")

            validation = validate_artifacts(agentic_training_flow_paths=[out], strict=True)
            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("source_artifacts.trainer_consumer_plan.size_bytes does not match path", errors)
            self.assertIn("source_artifacts.trainer_consumer_plan.sha256 does not match path", errors)

    def test_validation_rejects_source_artifact_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = self.write_runtime_preflight(root)
            consumer = self.write_trainer_consumer_plan(root)
            out = root / "flow.json"
            receipt = build_agentic_training_flow(
                plan_path=EXAMPLE_PLAN,
                runtime_preflight_path=runtime,
                trainer_consumer_plan_path=consumer,
                out_path=out,
                created_at="2026-07-03T00:00:00+00:00",
            )
            receipt["source_artifacts"]["trainer_consumer_plan"]["path"] = "../../../../../../../../etc/passwd"
            write_agentic_training_flow(out, receipt)

            validation = validate_artifacts(agentic_training_flow_paths=[out], strict=True)
            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("source_artifacts.trainer_consumer_plan.path must resolve under", errors)

    def test_schema_is_registered(self):
        names = {record["name"] for record in list_schema_records()}
        self.assertIn("agentic_training_flow", names)

    def write_runtime_preflight(self, root: Path, *, plan_path: Path = EXAMPLE_PLAN) -> Path:
        out = root / "runtime_preflight.json"
        runtime = build_agentic_training_runtime_preflight(
            plan_path=plan_path,
            out_path=out,
            require_modules=["json"],
            skip_default_modules=True,
            created_at="2026-07-03T00:00:00+00:00",
        )
        write_agentic_training_runtime_preflight(out, runtime)
        return out

    def write_trainer_consumer_plan(self, root: Path) -> Path:
        trainer_file = root / "train.py"
        trainer_file.write_text("print('dry run')\n", encoding="utf-8")
        trainer_sha = sha256(trainer_file)
        input_file = root / "agentic_training_plan.json"
        input_file.write_text(EXAMPLE_PLAN.read_text(encoding="utf-8"), encoding="utf-8")
        input_sha = sha256(input_file)
        archive_check_path = root / "trainer_archive_check.json"
        archive_check = {
            "schema_version": "hfr.trainer_archive_check.v1",
            "passed": True,
            "readiness": "ready",
            "recommendation": "consumer_ready",
            "portable_command": {
                "approved": True,
                "available": True,
                "argv": ["python", "train.py", "--plan", "agentic_training_plan.json"],
            },
            "archive": {"path": "trainer_archive"},
            "external_code_root": {"path": "trainer-code"},
            "external_code_checks": [
                {
                    "index": 0,
                    "argv_index": 1,
                    "token": "train.py",
                    "path": "train.py",
                    "resolved_path": str(trainer_file),
                    "exists": True,
                    "regular_file": True,
                    "symlink": False,
                    "passed": True,
                    "reason": "ready",
                    "sha256": trainer_sha,
                    "size_bytes": trainer_file.stat().st_size,
                }
            ],
            "trainer_input_checks": [
                {
                    "index": 0,
                    "artifact_index": 0,
                    "artifact_name": "agentic_training_plan",
                    "archive_path": "agentic_training_plan.json",
                    "resolved_path": str(input_file),
                    "kind": "file",
                    "exists": True,
                    "regular_file": True,
                    "regular_directory": False,
                    "symlink": False,
                    "expected_sha256": input_sha,
                    "expected_size_bytes": input_file.stat().st_size,
                    "passed": True,
                    "reason": "ready",
                    "sha256": input_sha,
                    "size_bytes": input_file.stat().st_size,
                }
            ],
        }
        archive_check_path.write_text(json.dumps(archive_check, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        consumer = build_trainer_consumer_plan(
            out_path=root / "trainer_consumer_plan.json",
            archive_check_path=archive_check_path,
            archive_check=archive_check,
            validation_summary={
                "passed": True,
                "strict": True,
                "target_count": 1,
                "error_count": 0,
                "warning_count": 0,
                "targets": [],
            },
        )
        out = root / "trainer_consumer_plan.json"
        out.write_text(json.dumps(consumer, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return out


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    unittest.main()
