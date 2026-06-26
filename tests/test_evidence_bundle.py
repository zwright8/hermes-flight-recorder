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


class EvidenceBundleTests(unittest.TestCase):
    def test_evidence_bundle_summarizes_ready_handoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            bundle_path = runs / "evidence_bundle.json"

            self.assertEqual(
                run_cli(
                    [
                        "run-suite",
                        "--scenarios",
                        str(ROOT / "scenarios"),
                        "--out",
                        str(runs),
                        "--export-rl",
                        "--validate",
                        "--strict",
                    ]
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    [
                        "scenario-quality",
                        "--scenarios",
                        str(ROOT / "scenarios"),
                        "--require-traces",
                        "--out",
                        str(runs / "scenario_quality.json"),
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
                    ]
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    [
                        "evidence-coverage",
                        "--runs",
                        str(runs),
                        "--out",
                        str(runs / "evidence_coverage.json"),
                        "--min-failed-rule-evidence-rate",
                        "1.0",
                        "--min-critical-failed-rule-evidence-rate",
                        "1.0",
                        "--max-failed-rules-without-evidence",
                        "0",
                        "--max-critical-failed-rules-without-evidence",
                        "0",
                    ]
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    [
                        "gate-suite",
                        "--suite-summary",
                        str(runs / "suite_summary.json"),
                        "--policy",
                        str(ROOT / "examples" / "suite_gate_policy.demo.json"),
                        "--out",
                        str(runs / "suite_gate.json"),
                    ]
                ),
                0,
            )

            code = run_cli(
                [
                    "evidence-bundle",
                    "--runs",
                    str(runs),
                    "--suite-summary",
                    str(runs / "suite_summary.json"),
                    "--scenario-quality",
                    str(runs / "scenario_quality.json"),
                    "--evidence-coverage",
                    str(runs / "evidence_coverage.json"),
                    "--validation",
                    str(runs / "validation.json"),
                    "--training-export",
                    str(runs / "training_export"),
                    "--gate",
                    str(runs / "suite_gate.json"),
                    "--out",
                    str(bundle_path),
                ]
            )

            self.assertEqual(code, 0)
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            self.assertEqual(bundle["schema_version"], "hfr.evidence_bundle.v1")
            self.assertTrue(bundle["passed"])
            self.assertEqual(bundle["readiness"], "ready")
            self.assertEqual(bundle["failed_check_count"], 0)
            self.assertGreaterEqual(bundle["check_count"], 6)
            self.assertEqual(bundle["metrics"]["suite_summary"]["total"], 6)
            self.assertEqual(bundle["metrics"]["scenario_quality"]["average_contract_score"], 89.17)
            self.assertEqual(bundle["metrics"]["evidence_coverage"]["failed_rule_evidence_rate"], 1.0)
            self.assertEqual(bundle["metrics"]["training_export"]["episode_count"], 6)
            self.assertEqual(bundle["metrics"]["gates"][0]["id"], "suite_gate")
            self.assertTrue(bundle["metrics"]["gates"][0]["passed"])
            self.assertEqual(bundle["artifacts"]["suite_summary"]["kind"], "file")
            self.assertEqual(len(bundle["artifacts"]["suite_summary"]["sha256"]), 64)

            self.assertEqual(run_cli(["validate", "--evidence-bundle", str(bundle_path), "--strict"]), 0)

    def test_evidence_bundle_blocks_failed_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            runs.mkdir()
            gate_path = root / "failed_gate.json"
            bundle_path = root / "evidence_bundle.json"
            gate_path.write_text(
                json.dumps({"schema_version": "hfr.test_gate.v1", "passed": False}, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            code = run_cli(
                [
                    "evidence-bundle",
                    "--runs",
                    str(runs),
                    "--gate",
                    str(gate_path),
                    "--out",
                    str(bundle_path),
                ]
            )

            self.assertEqual(code, 1)
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            self.assertFalse(bundle["passed"])
            self.assertEqual(bundle["readiness"], "blocked")
            self.assertEqual(bundle["failed_check_count"], 1)
            failed_checks = [check for check in bundle["checks"] if not check["passed"]]
            self.assertEqual(failed_checks[0]["id"], "gate_passed")
            self.assertEqual(failed_checks[0]["scope"]["gate"], "test_gate")
            self.assertEqual(run_cli(["validate", "--evidence-bundle", str(bundle_path), "--strict"]), 0)


if __name__ == "__main__":
    unittest.main()
