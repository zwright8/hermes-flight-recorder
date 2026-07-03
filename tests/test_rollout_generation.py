import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from flightrecorder.rollout_generation import build_agentic_rollout_plan
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
        self.assertFalse(plan["execution_boundary"]["rollouts_started"])
        self.assertFalse(plan["execution_boundary"]["dataset_rows_written"])
        self.assertTrue(plan["rejection_sampling"]["requires_review_calibration_before_training"])
        schema = check_schema_contract(plan)
        self.assertTrue(schema["passed"], schema["errors"])

    def test_cli_writes_schema_checkable_validatable_rollout_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "rollout_plan.json"
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

    def test_schema_is_registered(self):
        names = {record["name"] for record in list_schema_records()}
        self.assertIn("agentic_rollout_plan", names)


if __name__ == "__main__":
    unittest.main()
