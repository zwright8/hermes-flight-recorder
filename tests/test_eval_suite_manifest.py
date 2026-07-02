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


class EvalSuiteManifestTests(unittest.TestCase):
    def test_eval_suite_manifest_validates_and_schema_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary_path = Path(tmp) / "validation.json"
            manifest_path = ROOT / "eval_suites" / "red_team_prompt_injection.json"

            validate_code = run_cli(
                [
                    "validate",
                    "--eval-suite-manifest",
                    str(manifest_path),
                    "--out",
                    str(summary_path),
                    "--strict",
                ]
            )
            schema_code = run_cli(["schemas", "--check", str(manifest_path)])

            self.assertEqual(validate_code, 0)
            self.assertEqual(schema_code, 0)
            validation = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(validation["targets"][0]["type"], "eval_suite_manifest")
            self.assertEqual(validation["targets"][0]["details"]["scenario_count"], 2)

    def test_eval_suite_manifest_rejects_duplicate_scenario_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "duplicate_manifest.json"
            summary_path = Path(tmp) / "validation.json"
            manifest = json.loads((ROOT / "eval_suites" / "red_team_prompt_injection.json").read_text(encoding="utf-8"))
            manifest["scenario_ids"].append("prompt_injection_bad")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            code = run_cli(["validate", "--eval-suite-manifest", str(manifest_path), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            validation = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("eval_suite_manifest.scenario_ids must not contain duplicates", errors)

    def test_run_suite_filters_scenarios_from_eval_suite_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "prompt_injection"

            code = run_cli(
                [
                    "run-suite",
                    "--scenarios",
                    str(ROOT / "scenarios"),
                    "--suite-manifest",
                    str(ROOT / "eval_suites" / "red_team_prompt_injection.json"),
                    "--out",
                    str(out),
                ]
            )
            validate_code = run_cli(["validate", "--suite-summary", str(out / "suite_summary.json"), "--strict"])

            self.assertEqual(code, 0)
            self.assertEqual(validate_code, 0)
            summary = json.loads((out / "suite_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(
                [run["scenario_id"] for run in summary["runs"]],
                ["prompt_injection_bad", "prompt_injection_good"],
            )

    def test_red_team_safety_manifest_selects_declared_scenario_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "safety"

            code = run_cli(
                [
                    "run-suite",
                    "--scenarios",
                    str(ROOT / "scenarios"),
                    "--suite-manifest",
                    str(ROOT / "eval_suites" / "red_team_safety_regressions.json"),
                    "--out",
                    str(out),
                ]
            )

            self.assertEqual(code, 0)
            summary = json.loads((out / "suite_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(
                [run["scenario_id"] for run in summary["runs"]],
                [
                    "prompt_injection_bad",
                    "budget_runaway_bad",
                    "subagent_claim_bad",
                    "cron_async_delegation_bad",
                ],
            )


if __name__ == "__main__":
    unittest.main()
