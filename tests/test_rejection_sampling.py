import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from flightrecorder.rejection_sampling import build_rejection_sampling_gate, write_rejection_sampling_gate
from flightrecorder.rollout_generation import (
    build_agentic_rollout_plan,
    build_agentic_rollout_receipt,
    write_agentic_rollout_plan,
    write_agentic_rollout_receipt,
)
from flightrecorder.schema_registry import check_schema_file, list_schema_records
from flightrecorder.validation import validate_artifacts


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenarios" / "prompt_injection_good.json"


class RejectionSamplingGateTests(unittest.TestCase):
    def test_gate_admits_calibrated_mock_rollouts_without_writing_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rollout_receipt = self.write_rollout_receipt(root)
            model_grader_gate = self.write_json(root / "model_grader_gate.json", "hfr.model_grader_gate.v1")
            review_calibration = self.write_json(root / "review_calibration.json", "hfr.review_calibration.v1")
            reviewed_gate = self.write_json(root / "reviewed_gate.json", "hfr.reviewed_gate.v1")
            out = root / "rejection_sampling_gate.json"

            gate = build_rejection_sampling_gate(
                rollout_receipt_paths=[rollout_receipt],
                model_grader_gate_paths=[model_grader_gate],
                review_calibration_paths=[review_calibration],
                reviewed_gate_paths=[reviewed_gate],
                out_path=out,
                created_at="2026-07-03T00:00:00+00:00",
            )
            write_rejection_sampling_gate(out, gate)

            schema = check_schema_file(out)
            self.assertTrue(schema["passed"], schema["errors"])
            validation = validate_artifacts(rejection_sampling_gate_paths=[out], strict=True)
            self.assertTrue(validation["passed"], validation)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(payload["passed"], payload["blocked_reasons"])
            self.assertEqual(payload["readiness"], "ready_for_dataset_curation")
            self.assertEqual(payload["rollout_summary"]["mock_rollout_count"], 1)
            self.assertFalse(payload["execution_boundary"]["dataset_rows_written"])
            self.assertFalse(payload["admission_policy"]["accepts_uncalibrated_labels"])

    def test_cli_writes_valid_rejection_sampling_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rollout_receipt = self.write_rollout_receipt(root)
            model_grader_gate = self.write_json(root / "model_grader_gate.json", "hfr.model_grader_gate.v1")
            review_calibration = self.write_json(root / "review_calibration.json", "hfr.review_calibration.v1")
            reviewed_gate = self.write_json(root / "reviewed_gate.json", "hfr.reviewed_gate.v1")
            out = root / "gate.json"

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "flightrecorder",
                    "rejection-sampling-gate",
                    "--rollout-receipt",
                    str(rollout_receipt),
                    "--model-grader-gate",
                    str(model_grader_gate),
                    "--review-calibration",
                    str(review_calibration),
                    "--reviewed-gate",
                    str(reviewed_gate),
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
            validation = validate_artifacts(rejection_sampling_gate_paths=[out], strict=True)
            self.assertTrue(validation["passed"], validation)

    def test_gate_blocks_uncalibrated_or_missing_review_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rollout_receipt = self.write_rollout_receipt(root)
            model_grader_gate = self.write_json(root / "model_grader_gate.json", "hfr.model_grader_gate.v1")
            reviewed_gate = self.write_json(root / "reviewed_gate.json", "hfr.reviewed_gate.v1")

            gate = build_rejection_sampling_gate(
                rollout_receipt_paths=[rollout_receipt],
                model_grader_gate_paths=[model_grader_gate],
                review_calibration_paths=[],
                reviewed_gate_paths=[reviewed_gate],
                created_at="2026-07-03T00:00:00+00:00",
            )

            self.assertFalse(gate["passed"])
            self.assertEqual(gate["readiness"], "blocked")
            self.assertIn("review_calibration_present_and_passing", {check["id"] for check in gate["checks"] if not check["passed"]})
            self.assertFalse(gate["execution_boundary"]["dataset_rows_written"])

    def test_validation_rejects_dataset_side_effect_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rollout_receipt = self.write_rollout_receipt(root)
            model_grader_gate = self.write_json(root / "model_grader_gate.json", "hfr.model_grader_gate.v1")
            review_calibration = self.write_json(root / "review_calibration.json", "hfr.review_calibration.v1")
            reviewed_gate = self.write_json(root / "reviewed_gate.json", "hfr.reviewed_gate.v1")
            out = root / "gate.json"
            gate = build_rejection_sampling_gate(
                rollout_receipt_paths=[rollout_receipt],
                model_grader_gate_paths=[model_grader_gate],
                review_calibration_paths=[review_calibration],
                reviewed_gate_paths=[reviewed_gate],
                out_path=out,
                created_at="2026-07-03T00:00:00+00:00",
            )
            gate["execution_boundary"]["dataset_rows_written"] = True
            write_rejection_sampling_gate(out, gate)

            validation = validate_artifacts(rejection_sampling_gate_paths=[out], strict=True)
            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("execution_boundary.dataset_rows_written must be false", errors)

    def test_validation_rejects_forged_provider_and_trainer_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rollout_receipt = self.write_rollout_receipt(root)
            model_grader_gate = self.write_json(root / "model_grader_gate.json", "hfr.model_grader_gate.v1")
            review_calibration = self.write_json(root / "review_calibration.json", "hfr.review_calibration.v1")
            reviewed_gate = self.write_json(root / "reviewed_gate.json", "hfr.reviewed_gate.v1")
            out = root / "gate.json"
            gate = build_rejection_sampling_gate(
                rollout_receipt_paths=[rollout_receipt],
                model_grader_gate_paths=[model_grader_gate],
                review_calibration_paths=[review_calibration],
                reviewed_gate_paths=[reviewed_gate],
                out_path=out,
                created_at="2026-07-03T00:00:00+00:00",
            )
            gate["provider_job_id"] = "job_live"
            gate["checks"][0]["provider_trace_id"] = "trace_live"
            gate["input_artifacts"]["model_grader_gate"][0]["signed_url"] = "https://example.invalid/model-grader-gate.json"
            gate["input_artifacts"]["agentic_rollout_receipt"][0]["provider_rollout_id"] = "rollout_live"
            gate["rollout_summary"]["provider_rollout_count"] = 1
            gate["admission_policy"]["accepts_provider_labels"] = True
            gate["execution_boundary"]["trainer_dataset_written"] = True
            write_rejection_sampling_gate(out, gate)

            schema = check_schema_file(out)
            self.assertFalse(schema["passed"], schema)
            validation = validate_artifacts(rejection_sampling_gate_paths=[out], strict=True)
            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("rejection_sampling_gate contains unknown field", errors)
            self.assertIn("rejection_sampling_gate.checks[0] contains unknown field", errors)
            self.assertIn("rejection_sampling_gate.input_artifacts.model_grader_gate[0] contains unknown field", errors)
            self.assertIn("rejection_sampling_gate.admission_policy contains unknown field", errors)

    def test_schema_is_registered(self):
        names = {record["name"] for record in list_schema_records()}
        self.assertIn("rejection_sampling_gate", names)

    def write_rollout_receipt(self, root: Path) -> Path:
        plan_path = root / "rollout_plan.json"
        receipt_path = root / "rollout_receipt.json"
        plan = build_agentic_rollout_plan(
            out_path=plan_path,
            iteration_id="reject-sample",
            scenario_paths=[SCENARIO],
            policies={"baseline": "local/base"},
            max_rollouts=1,
            created_at="2026-07-03T00:00:00+00:00",
        )
        write_agentic_rollout_plan(plan_path, plan)
        receipt = build_agentic_rollout_receipt(
            plan_path=plan_path,
            out_path=receipt_path,
            created_at="2026-07-03T00:00:00+00:00",
        )
        write_agentic_rollout_receipt(receipt_path, receipt)
        return receipt_path

    def write_json(self, path: Path, schema_version: str) -> Path:
        path.write_text(
            json.dumps({"schema_version": schema_version, "passed": True, "readiness": "ready"}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path


if __name__ == "__main__":
    unittest.main()
