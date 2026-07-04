import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from flightrecorder.rollout_generation import (
    build_agentic_rollout_plan,
    build_agentic_rollout_receipt,
    write_agentic_rollout_plan,
    write_agentic_rollout_receipt,
)
from flightrecorder.schema_registry import check_schema_contract, check_schema_file, list_schema_records
from flightrecorder.validation import validate_artifacts


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenarios" / "prompt_injection_good.json"
VERIFIER = ROOT / "examples" / "external_verification" / "sqlite_task_state.verifier.json"


class RolloutGenerationTests(unittest.TestCase):
    def test_rollout_plan_builds_policy_scenario_matrix_without_running(self):
        plan = build_agentic_rollout_plan(
            out_path="runs/rollout_plan.json",
            iteration_id="rollout-001",
            scenario_paths=[SCENARIO],
            policies={"baseline": "local/base", "candidate": "local/candidate", "teacher": "local/teacher"},
            max_rollouts=3,
            verifier_paths=[VERIFIER],
            created_at="2026-07-03T00:00:00+00:00",
        )

        self.assertTrue(plan["passed"], plan["blocked_reasons"])
        self.assertEqual(plan["budget"]["planned_rollouts"], 3)
        verifier_gate = plan["environment"]["external_state_verifier_gate"]
        self.assertEqual(verifier_gate["declared_count"], 1)
        self.assertEqual(verifier_gate["resolved_count"], 1)
        self.assertTrue(verifier_gate["all_declared_verifiers_resolved"])
        self.assertFalse(verifier_gate["verification_side_effects_started"])
        self.assertFalse(plan["execution_boundary"]["rollouts_started"])
        self.assertFalse(plan["execution_boundary"]["dataset_rows_written"])
        self.assertTrue(plan["rejection_sampling"]["requires_review_calibration_before_training"])
        schema = check_schema_contract(plan)
        self.assertTrue(schema["passed"], schema["errors"])

    def test_cli_writes_schema_checkable_validatable_rollout_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "rollout_plan.json"
            receipt_out = Path(tmp) / "rollout_receipt.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "flightrecorder",
                    "agentic-rollout-plan",
                    "--iteration-id",
                    "rollout-cli",
                    "--scenario",
                    str(SCENARIO),
                    "--policy",
                    "baseline=local/base",
                    "--policy",
                    "candidate=local/candidate",
                    "--max-rollouts",
                    "2",
                    "--verifier",
                    str(VERIFIER),
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
            schema = check_schema_file(out)
            self.assertTrue(schema["passed"], schema["errors"])
            validation = validate_artifacts(agentic_rollout_plan_paths=[out], strict=True)
            self.assertTrue(validation["passed"], validation)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["budget"]["planned_rollouts"], 2)
            self.assertTrue(payload["environment"]["external_state_verifier_gate"]["all_declared_verifiers_resolved"])

            receipt_completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "flightrecorder",
                    "agentic-rollout-receipt",
                    "--plan",
                    str(out),
                    "--created-at",
                    "2026-07-03T00:00:00+00:00",
                    "--out",
                    str(receipt_out),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(receipt_completed.returncode, 0, receipt_completed.stderr + receipt_completed.stdout)
            receipt_schema = check_schema_file(receipt_out)
            self.assertTrue(receipt_schema["passed"], receipt_schema["errors"])
            receipt_validation = validate_artifacts(agentic_rollout_receipt_paths=[receipt_out], strict=True)
            self.assertTrue(receipt_validation["passed"], receipt_validation)
            receipt = json.loads(receipt_out.read_text(encoding="utf-8"))
            self.assertEqual(receipt["mock_rollout_count"], 2)
            self.assertFalse(receipt["execution_boundary"]["model_provider_calls_started"])
            self.assertFalse(receipt["lineage"]["dataset_rows_created"])

    def test_strict_validate_warns_on_absolute_rollout_plan_refs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan_path = root / "rollout_plan.json"
            plan = build_agentic_rollout_plan(
                out_path=plan_path,
                iteration_id="rollout-preserved-plan",
                scenario_paths=[SCENARIO],
                policies={"baseline": "local/base"},
                max_rollouts=1,
                verifier_paths=[VERIFIER],
                preserve_paths=True,
                created_at="2026-07-03T00:00:00+00:00",
            )
            write_agentic_rollout_plan(plan_path, plan)

            validation = validate_artifacts(agentic_rollout_plan_paths=[plan_path])
            strict_validation = validate_artifacts(agentic_rollout_plan_paths=[plan_path], strict=True)

            self.assertTrue(validation["passed"], validation)
            self.assertFalse(strict_validation["passed"], strict_validation)
            warnings = "\n".join(warning for target in validation["targets"] for warning in target["warnings"])
            strict_warnings = "\n".join(warning for target in strict_validation["targets"] for warning in target["warnings"])
            for expected in (
                "agentic_rollout_plan.plan_path is absolute",
                "agentic_rollout_plan.scenarios[0].path is absolute",
                "agentic_rollout_plan.environment.external_state_verifiers[0].path is absolute",
            ):
                self.assertIn(expected, warnings)
                self.assertIn(expected, strict_warnings)

    def test_rollout_receipt_records_mock_rows_without_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan_path = Path(tmp) / "rollout_plan.json"
            receipt_path = Path(tmp) / "rollout_receipt.json"
            plan = build_agentic_rollout_plan(
                out_path=plan_path,
                iteration_id="rollout-receipt",
                scenario_paths=[SCENARIO],
                policies={"baseline": "local/base", "candidate": "local/candidate"},
                max_rollouts=2,
                verifier_paths=[VERIFIER],
                created_at="2026-07-03T00:00:00+00:00",
            )
            write_agentic_rollout_plan(plan_path, plan)

            receipt = build_agentic_rollout_receipt(
                plan_path=plan_path,
                out_path=receipt_path,
                created_at="2026-07-03T00:00:00+00:00",
            )
            write_agentic_rollout_receipt(receipt_path, receipt)

            schema = check_schema_file(receipt_path)
            self.assertTrue(schema["passed"], schema["errors"])
            validation = validate_artifacts(agentic_rollout_receipt_paths=[receipt_path], strict=True)
            self.assertTrue(validation["passed"], validation)
            payload = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["passed"], payload["blocked_reasons"])
            self.assertEqual(payload["source_plan"]["path"], "rollout_plan.json")
            self.assertEqual(payload["mock_rollout_count"], 2)
            self.assertTrue(all(row["status"] == "mock_recorded" for row in payload["mock_rollouts"]))
            self.assertFalse(any(row["model_provider_called"] for row in payload["mock_rollouts"]))
            self.assertFalse(any(row["dataset_row_written"] for row in payload["mock_rollouts"]))
            self.assertTrue(payload["environment"]["external_state_verifier_gate"]["all_declared_verifiers_resolved"])

    def test_strict_validate_warns_on_absolute_rollout_receipt_source_plan_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan_path = root / "rollout_plan.json"
            receipt_path = root / "rollout_receipt.json"
            plan = build_agentic_rollout_plan(
                out_path=plan_path,
                iteration_id="rollout-preserved-source",
                scenario_paths=[SCENARIO],
                policies={"baseline": "local/base"},
                max_rollouts=1,
                created_at="2026-07-03T00:00:00+00:00",
            )
            write_agentic_rollout_plan(plan_path, plan)
            receipt = build_agentic_rollout_receipt(
                plan_path=plan_path,
                out_path=receipt_path,
                preserve_paths=True,
                created_at="2026-07-03T00:00:00+00:00",
            )
            receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(agentic_rollout_receipt_paths=[receipt_path])
            strict_validation = validate_artifacts(agentic_rollout_receipt_paths=[receipt_path], strict=True)

            self.assertTrue(validation["passed"], validation)
            self.assertFalse(strict_validation["passed"], strict_validation)
            warnings = "\n".join(warning for target in validation["targets"] for warning in target["warnings"])
            strict_warnings = "\n".join(warning for target in strict_validation["targets"] for warning in target["warnings"])
            expected = "agentic_rollout_receipt.source_plan.path is absolute"
            self.assertIn(expected, warnings)
            self.assertIn(expected, strict_warnings)

    def test_rollout_receipt_validation_rejects_live_side_effect_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan_path = Path(tmp) / "rollout_plan.json"
            receipt_path = Path(tmp) / "rollout_receipt.json"
            plan = build_agentic_rollout_plan(
                out_path=plan_path,
                iteration_id="rollout-forged",
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
            receipt["execution_boundary"]["model_provider_calls_started"] = True
            receipt["environment"]["external_state_verifier_gate"]["verification_side_effects_started"] = True
            write_agentic_rollout_receipt(receipt_path, receipt)

            validation = validate_artifacts(agentic_rollout_receipt_paths=[receipt_path], strict=True)
            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("execution_boundary.model_provider_calls_started must be false", errors)
            self.assertIn("external_state_verifier_gate.verification_side_effects_started must be false", errors)

    def test_rollout_plan_validation_rejects_forged_live_provider_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan_path = Path(tmp) / "rollout_plan.json"
            plan = build_agentic_rollout_plan(
                out_path=plan_path,
                iteration_id="rollout-forged-provider",
                scenario_paths=[SCENARIO],
                policies={"baseline": "local/base", "candidate": "local/candidate"},
                max_rollouts=2,
                verifier_paths=[VERIFIER],
                created_at="2026-07-03T00:00:00+00:00",
            )
            plan["provider_job_id"] = "job_live"
            plan["checks"][0]["provider_trace_id"] = "trace_live"
            plan["budget"]["cloud_budget_usd"] = 20
            plan["environment"]["provider_region"] = "us-east-1"
            plan["environment"]["external_state_verifiers"][0]["credential_value"] = "redacted"
            plan["environment"]["external_state_verifier_gate"]["credential_secret_ref"] = "HFR_VERIFIER_TOKEN"
            plan["policies"][0]["provider_api_key_env"] = "MODEL_PROVIDER_KEY"
            plan["scenarios"][0]["signed_url"] = "https://example.invalid/scenario.json"
            plan["harness_batches"][0]["live_provider_job_id"] = "job_live"
            plan["rejection_sampling"]["provider_dataset_uri"] = "s3://example-bucket/rollouts.jsonl"
            plan["lineage"]["trace_signed_url"] = "https://example.invalid/trace.jsonl"
            plan["execution_boundary"]["live_endpoint_url"] = "https://example.invalid/run"
            write_agentic_rollout_plan(plan_path, plan)

            schema = check_schema_file(plan_path)
            self.assertFalse(schema["passed"], schema)
            validation = validate_artifacts(agentic_rollout_plan_paths=[plan_path], strict=True)
            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("agentic_rollout_plan contains unknown field", errors)
            self.assertIn("agentic_rollout_plan.checks[0] contains unknown field", errors)
            self.assertIn("agentic_rollout_plan.environment contains unknown field", errors)
            self.assertIn("agentic_rollout_plan.harness_batches[0] contains unknown field", errors)
            self.assertIn("agentic_rollout_plan.execution_boundary contains unknown field", errors)

    def test_rollout_receipt_validation_rejects_forged_live_provider_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan_path = Path(tmp) / "rollout_plan.json"
            receipt_path = Path(tmp) / "rollout_receipt.json"
            plan = build_agentic_rollout_plan(
                out_path=plan_path,
                iteration_id="rollout-receipt-forged-provider",
                scenario_paths=[SCENARIO],
                policies={"baseline": "local/base"},
                max_rollouts=1,
                verifier_paths=[VERIFIER],
                created_at="2026-07-03T00:00:00+00:00",
            )
            write_agentic_rollout_plan(plan_path, plan)
            receipt = build_agentic_rollout_receipt(
                plan_path=plan_path,
                out_path=receipt_path,
                created_at="2026-07-03T00:00:00+00:00",
            )
            receipt["provider_job_id"] = "job_live"
            receipt["checks"][0]["provider_trace_id"] = "trace_live"
            receipt["source_plan"]["signed_url"] = "https://example.invalid/rollout-plan.json"
            receipt["environment"]["provider_region"] = "us-east-1"
            receipt["environment"]["external_state_verifiers"][0]["credential_value"] = "redacted"
            receipt["environment"]["external_state_verifier_gate"]["credential_secret_ref"] = "HFR_VERIFIER_TOKEN"
            receipt["mock_rollouts"][0]["provider_completion_id"] = "completion_live"
            receipt["lineage"]["trace_path"] = "traces/live.jsonl"
            receipt["execution_boundary"]["live_endpoint_url"] = "https://example.invalid/run"
            write_agentic_rollout_receipt(receipt_path, receipt)

            schema = check_schema_file(receipt_path)
            self.assertFalse(schema["passed"], schema)
            validation = validate_artifacts(agentic_rollout_receipt_paths=[receipt_path], strict=True)
            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("agentic_rollout_receipt contains unknown field", errors)
            self.assertIn("agentic_rollout_receipt.checks[0] contains unknown field", errors)
            self.assertIn("agentic_rollout_receipt.source_plan contains unknown field", errors)
            self.assertIn("agentic_rollout_receipt.mock_rollouts[0] contains unknown field", errors)
            self.assertIn("agentic_rollout_receipt.execution_boundary contains unknown field", errors)

    def test_rollout_plan_blocks_missing_verifier_refs(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing_verifier = Path(tmp) / "missing.verifier.json"
            plan = build_agentic_rollout_plan(
                out_path=Path(tmp) / "rollout_plan.json",
                iteration_id="rollout-missing-verifier",
                scenario_paths=[SCENARIO],
                policies={"baseline": "local/base"},
                max_rollouts=1,
                verifier_paths=[missing_verifier],
                created_at="2026-07-03T00:00:00+00:00",
            )

            self.assertFalse(plan["passed"])
            self.assertEqual(plan["readiness"], "blocked")
            self.assertEqual(plan["environment"]["external_state_verifier_gate"]["declared_count"], 1)
            self.assertEqual(plan["environment"]["external_state_verifier_gate"]["resolved_count"], 0)
            self.assertFalse(plan["environment"]["external_state_verifier_gate"]["all_declared_verifiers_resolved"])
            self.assertIn("external_state_verifiers_resolved: passed=False", plan["blocked_reasons"])

    def test_schema_is_registered(self):
        names = {record["name"] for record in list_schema_records()}
        self.assertIn("agentic_rollout_plan", names)
        self.assertIn("agentic_rollout_receipt", names)


if __name__ == "__main__":
    unittest.main()
