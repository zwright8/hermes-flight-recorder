import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main


ROOT = Path(__file__).resolve().parents[1]


def run_cli(args):
    with redirect_stdout(StringIO()):
        return main(args)


class ScenarioQualityTests(unittest.TestCase):
    def test_scenario_quality_scores_bundled_suite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "scenario_quality.json"

            code = run_cli(
                [
                    "scenario-quality",
                    "--scenarios",
                    str(ROOT / "scenarios"),
                    "--require-traces",
                    "--out",
                    str(out),
                    "--min-average-score",
                    "80",
                    "--min-scenario-score",
                    "60",
                    "--min-observable-rate",
                    "0.8",
                    "--max-weak-scenarios",
                    "0",
                    "--max-final-only-scenarios",
                    "0",
                    "--max-missing-traces",
                    "0",
                    "--require-task-family",
                    "email_reply_completion",
                    "--require-task-family",
                    "prompt_injection",
                ]
            )

            self.assertEqual(code, 0)
            quality = json.loads(out.read_text(encoding="utf-8"))
            metrics = quality["metrics"]
            self.assertEqual(quality["schema_version"], "hfr.scenario_quality.v1")
            self.assertTrue(quality["passed"])
            self.assertEqual(metrics["scenario_count"], 7)
            self.assertEqual(metrics["task_family_count"], 5)
            self.assertEqual(metrics["average_contract_score"], 93.57)
            self.assertEqual(metrics["min_contract_score"], 65)
            self.assertEqual(metrics["observable_scenario_rate"], 0.8571)
            self.assertEqual(metrics["weak_scenario_count"], 0)
            self.assertEqual(metrics["final_only_scenario_count"], 0)
            self.assertEqual(metrics["missing_trace_count"], 0)
            by_id = {row["id"]: row for row in quality["scenarios"]}
            self.assertEqual(by_id["cron_async_delegation_bad"]["quality"], "strong")
            self.assertEqual(by_id["email_reply_completion_good"]["quality"], "strong")
            self.assertEqual(by_id["budget_runaway_bad"]["quality"], "moderate")
            self.assertIn("no_observable_assertions", by_id["budget_runaway_bad"]["risks"])

            validate_code = run_cli(["validate", "--scenario-quality", str(out), "--strict"])
            self.assertEqual(validate_code, 0)

    def test_scenario_quality_fails_unmet_thresholds(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "scenario_quality.json"

            code = run_cli(
                [
                    "scenario-quality",
                    "--scenarios",
                    str(ROOT / "scenarios"),
                    "--require-traces",
                    "--out",
                    str(out),
                    "--min-scenario-score",
                    "90",
                    "--min-observable-rate",
                    "1.0",
                    "--require-task-family",
                    "missing_family",
                ]
            )

            self.assertEqual(code, 1)
            quality = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse(quality["passed"])
            failed_ids = [check["id"] for check in quality["checks"] if not check["passed"]]
            self.assertIn("min_scenario_contract_score", failed_ids)
            self.assertIn("min_observable_scenario_rate", failed_ids)
            self.assertIn("require_task_family", failed_ids)


if __name__ == "__main__":
    unittest.main()
