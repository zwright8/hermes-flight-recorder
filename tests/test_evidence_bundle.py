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
                        "trace-observability",
                        "--runs",
                        str(runs),
                        "--out",
                        str(runs / "trace_observability.json"),
                        "--min-average-events",
                        "2",
                        "--min-event-type-count",
                        "2",
                        "--min-tool-or-api-run-rate",
                        "0.5",
                        "--max-empty-final-answers",
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
                    "--trace-observability",
                    str(runs / "trace_observability.json"),
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
            self.assertEqual(bundle["decision"]["readiness"], "ready")
            self.assertEqual(bundle["decision"]["recommendation"], "promote_handoff")
            self.assertEqual(bundle["decision"]["blocking_check_count"], 0)
            self.assertEqual(bundle["decision"]["blocking_checks"], [])
            self.assertIn("suite_summary", bundle["decision"]["evidence_artifacts"])
            self.assertEqual(bundle["decision"]["gate_count"], 1)
            self.assertEqual(bundle["decision"]["passed_gate_count"], 1)
            action_ids = {action["id"] for action in bundle["decision"]["next_actions"]}
            self.assertEqual(bundle["decision"]["next_action_count"], len(bundle["decision"]["next_actions"]))
            self.assertIn("repair_failed_scenarios", action_ids)
            self.assertIn("repair_critical_failures", action_ids)
            self.assertIn("ground_scenario_contracts", action_ids)
            self.assertIn("improve_trace_observability", action_ids)
            self.assertEqual(bundle["decision"]["key_metrics"]["suite_summary"]["total"], 6)
            self.assertEqual(bundle["decision"]["key_metrics"]["trace_observability"]["tool_or_api_run_rate"], 0.8333)
            self.assertIn("risk_counts", bundle["decision"]["key_metrics"]["trace_observability"])
            self.assertEqual(bundle["decision"]["key_metrics"]["training_export"]["episode_count"], 6)
            self.assertEqual(bundle["metrics"]["suite_summary"]["total"], 6)
            self.assertEqual(bundle["metrics"]["scenario_quality"]["average_contract_score"], 89.17)
            self.assertIn("risk_counts", bundle["metrics"]["scenario_quality"])
            self.assertEqual(bundle["metrics"]["evidence_coverage"]["failed_rule_evidence_rate"], 1.0)
            self.assertEqual(bundle["metrics"]["trace_observability"]["run_count"], 6)
            self.assertEqual(bundle["metrics"]["trace_observability"]["event_type_count"], 6)
            self.assertIn("risk_counts", bundle["metrics"]["trace_observability"])
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
            self.assertEqual(bundle["decision"]["readiness"], "blocked")
            self.assertEqual(bundle["decision"]["recommendation"], "block_handoff")
            self.assertEqual(bundle["decision"]["blocking_check_count"], 1)
            self.assertEqual(bundle["decision"]["blocking_checks"][0]["id"], "gate_passed")
            self.assertEqual(bundle["decision"]["blocking_gates"][0]["id"], "test_gate")
            self.assertIn("fix_failed_gates", {action["id"] for action in bundle["decision"]["next_actions"]})
            self.assertEqual(bundle["decision"]["gate_count"], 1)
            self.assertEqual(bundle["decision"]["passed_gate_count"], 0)
            self.assertEqual(bundle["decision"]["key_metrics"]["gates"]["failed"], 1)
            failed_checks = [check for check in bundle["checks"] if not check["passed"]]
            self.assertEqual(failed_checks[0]["id"], "gate_passed")
            self.assertEqual(failed_checks[0]["scope"]["gate"], "test_gate")
            self.assertEqual(run_cli(["validate", "--evidence-bundle", str(bundle_path), "--strict"]), 0)

    def test_validate_rejects_stale_bundle_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            runs.mkdir()
            gate_path = root / "failed_gate.json"
            bundle_path = root / "evidence_bundle.json"
            summary_path = root / "validation.json"
            gate_path.write_text(
                json.dumps({"schema_version": "hfr.test_gate.v1", "passed": False}, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            run_cli(["evidence-bundle", "--runs", str(runs), "--gate", str(gate_path), "--out", str(bundle_path)])
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            bundle["decision"]["recommendation"] = "promote_handoff"
            bundle["decision"]["next_action_count"] = 0
            bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--evidence-bundle", str(bundle_path), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("evidence_bundle.decision.recommendation", errors)
            self.assertIn("evidence_bundle.decision.next_action_count", errors)


if __name__ == "__main__":
    unittest.main()
