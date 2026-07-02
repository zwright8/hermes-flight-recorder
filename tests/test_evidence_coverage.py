import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main
from flightrecorder.schema_registry import check_schema_file


ROOT = Path(__file__).resolve().parents[1]


def run_cli(args):
    with redirect_stdout(StringIO()):
        return main(args)


class EvidenceCoverageTests(unittest.TestCase):
    def test_evidence_coverage_reports_failed_rule_attribution(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            out = Path(tmp) / "evidence_coverage.json"
            run_cli(["run-suite", "--scenarios", str(ROOT / "scenarios"), "--out", str(runs)])

            code = run_cli(
                [
                    "evidence-coverage",
                    "--runs",
                    str(runs),
                    "--out",
                    str(out),
                    "--min-failed-rule-evidence-rate",
                    "1.0",
                    "--min-critical-failed-rule-evidence-rate",
                    "1.0",
                    "--max-failed-rules-without-evidence",
                    "0",
                    "--max-critical-failed-rules-without-evidence",
                    "0",
                    "--require-rule-evidence",
                    "required_actions",
                ]
            )

            self.assertEqual(code, 0)
            coverage = json.loads(out.read_text(encoding="utf-8"))
            metrics = coverage["metrics"]
            self.assertEqual(coverage["schema_version"], "hfr.evidence_coverage.v1")
            self.assertTrue(coverage["passed"])
            self.assertEqual(metrics["run_count"], 7)
            self.assertEqual(metrics["failed_rule_count"], 14)
            self.assertEqual(metrics["failed_rule_evidence_rate"], 1.0)
            self.assertEqual(metrics["critical_failed_rule_evidence_rate"], 1.0)
            self.assertEqual(metrics["failed_rules_without_evidence"], 0)
            self.assertGreater(metrics["event_evidence_ref_count"], 0)
            self.assertNotIn("target_counts", metrics["rule_coverage"][0])
            rule_ids = {row["rule_id"] for row in metrics["rule_coverage"]}
            self.assertIn("required_actions", rule_ids)
            self.assertIn("required_state", rule_ids)
            self.assertIn("required_state_transitions", rule_ids)

            validate_code = run_cli(["validate", "--evidence-coverage", str(out), "--strict"])
            self.assertEqual(validate_code, 0)
            schema = check_schema_file(out)
            self.assertTrue(schema["passed"], schema["errors"])
            self.assertEqual(schema["schema"]["name"], "evidence_coverage")

    def test_evidence_coverage_fails_unmet_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            out = Path(tmp) / "evidence_coverage.json"
            run_cli(["run-suite", "--scenarios", str(ROOT / "scenarios"), "--out", str(runs)])

            code = run_cli(
                [
                    "evidence-coverage",
                    "--runs",
                    str(runs),
                    "--out",
                    str(out),
                    "--min-event-evidence-refs",
                    "999",
                    "--require-rule-evidence",
                    "missing_rule",
                ]
            )

            self.assertEqual(code, 1)
            coverage = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse(coverage["passed"])
            failed_ids = [check["id"] for check in coverage["checks"] if not check["passed"]]
            self.assertIn("min_event_evidence_refs", failed_ids)
            self.assertIn("require_rule_evidence", failed_ids)


if __name__ == "__main__":
    unittest.main()
