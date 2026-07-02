import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main


def run_cli(args):
    with redirect_stdout(StringIO()):
        return main(args)


class EvalSummaryTests(unittest.TestCase):
    def test_eval_summary_allows_claims_for_identical_heldout_scenarios(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = _suite_summary(root / "baseline_suite.json", ["email_reply_completion"])
            candidate = _suite_summary(root / "candidate_suite.json", ["email_reply_completion"])
            compare_export = _compare_export(root / "compare_rl", candidate_wins=["email_reply_completion"])
            out = root / "eval_summary.json"

            code = run_cli(
                [
                    "eval-summary",
                    "--suite-summary",
                    f"baseline={baseline}",
                    "--suite-summary",
                    f"candidate={candidate}",
                    "--compare-export",
                    f"candidate={compare_export}",
                    "--out",
                    str(out),
                ]
            )
            validate_code = run_cli(["validate", "--eval-summary", str(out), "--strict"])

            self.assertEqual(code, 0)
            self.assertEqual(validate_code, 0)
            summary = _read_json(out)
            comparison = summary["comparisons"][0]
            self.assertTrue(summary["passed"])
            self.assertEqual(summary["heldout_scenarios"]["status"], "identical")
            self.assertTrue(comparison["claims_allowed"])
            self.assertEqual(comparison["governance_claims"]["candidate_win_count"], 1)
            self.assertEqual(comparison["governance_claims"]["candidate_win_scenarios"], ["email_reply_completion"])
            self.assertFalse(comparison["governance_claims"]["suppressed_raw_claims"])

    def test_eval_summary_suppresses_claims_for_mismatched_heldout_scenarios(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = _suite_summary(root / "baseline_suite.json", ["email_reply_completion", "prompt_injection"])
            candidate = _suite_summary(root / "candidate_suite.json", ["email_reply_completion"])
            compare_export = _compare_export(root / "compare_rl", candidate_wins=["email_reply_completion"])
            out = root / "eval_summary.json"

            code = run_cli(
                [
                    "eval-summary",
                    "--suite-summary",
                    f"baseline={baseline}",
                    "--suite-summary",
                    f"candidate={candidate}",
                    "--compare-export",
                    f"candidate={compare_export}",
                    "--out",
                    str(out),
                ]
            )
            validate_code = run_cli(["validate", "--eval-summary", str(out), "--strict"])

            self.assertEqual(code, 1)
            self.assertEqual(validate_code, 0)
            summary = _read_json(out)
            comparison = summary["comparisons"][0]
            self.assertFalse(summary["passed"])
            self.assertEqual(summary["heldout_scenarios"]["status"], "mismatched")
            self.assertFalse(comparison["claims_allowed"])
            self.assertEqual(comparison["raw_movement"]["candidate_win_count"], 1)
            self.assertEqual(comparison["governance_claims"]["candidate_win_count"], 0)
            self.assertEqual(comparison["governance_claims"]["candidate_win_scenarios"], [])
            self.assertTrue(comparison["governance_claims"]["suppressed_raw_claims"])
            self.assertIn("heldout_scenario_set_mismatch", comparison["governance_claims"]["suppression_reasons"])

    def test_eval_summary_blocks_claims_for_contract_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = _suite_summary(root / "baseline_suite.json", ["email_reply_completion"])
            candidate = _suite_summary(root / "candidate_suite.json", ["email_reply_completion"])
            compare_export = _compare_export(
                root / "compare_rl",
                candidate_wins=["email_reply_completion"],
                contract_drift_count=1,
            )
            out = root / "eval_summary.json"

            code = run_cli(
                [
                    "eval-summary",
                    "--suite-summary",
                    f"baseline={baseline}",
                    "--suite-summary",
                    f"candidate={candidate}",
                    "--compare-export",
                    f"candidate={compare_export}",
                    "--out",
                    str(out),
                ]
            )
            validate_code = run_cli(["validate", "--eval-summary", str(out), "--strict"])

            self.assertEqual(code, 1)
            self.assertEqual(validate_code, 0)
            summary = _read_json(out)
            comparison = summary["comparisons"][0]
            self.assertEqual(summary["heldout_scenarios"]["status"], "identical")
            self.assertFalse(comparison["claims_allowed"])
            self.assertIn("contract_fingerprint_drift", comparison["blocking_reasons"])
            self.assertEqual(comparison["governance_claims"]["candidate_win_count"], 0)

    def test_validate_rejects_eval_summary_with_unsuppressed_disallowed_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = _suite_summary(root / "baseline_suite.json", ["email_reply_completion", "prompt_injection"])
            candidate = _suite_summary(root / "candidate_suite.json", ["email_reply_completion"])
            compare_export = _compare_export(root / "compare_rl", candidate_wins=["email_reply_completion"])
            out = root / "eval_summary.json"
            validation = root / "validation.json"
            run_cli(
                [
                    "eval-summary",
                    "--suite-summary",
                    f"baseline={baseline}",
                    "--suite-summary",
                    f"candidate={candidate}",
                    "--compare-export",
                    f"candidate={compare_export}",
                    "--out",
                    str(out),
                ]
            )
            summary = _read_json(out)
            summary["comparisons"][0]["governance_claims"]["candidate_win_count"] = 1
            out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--eval-summary", str(out), "--out", str(validation)])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in _read_json(validation)["targets"] for error in target["errors"])
            self.assertIn("governance_claims.candidate_win_count must be 0", errors)


def _suite_summary(path: Path, scenario_ids: list[str]) -> Path:
    runs = [
        {
            "scenario_id": scenario_id,
            "task_family": scenario_id,
            "passed": True,
            "score": 100,
            "failed_rules": [],
            "critical_failures": [],
        }
        for scenario_id in scenario_ids
    ]
    payload = {
        "schema_version": "hfr.run_suite.v1",
        "total": len(runs),
        "passed": len(runs),
        "failed": 0,
        "error_count": 0,
        "errors": [],
        "metrics": {
            "pass_rate": 1.0 if runs else 0.0,
            "average_score": 100.0 if runs else 0.0,
            "failed_rule_counts": [],
            "critical_failure_counts": [],
        },
        "runs": runs,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _compare_export(
    path: Path,
    *,
    candidate_wins: list[str],
    contract_drift_count: int = 0,
) -> Path:
    path.mkdir(parents=True)
    payload = {
        "schema_version": "hfr.compare_rl.manifest.v1",
        "pair_count": len(candidate_wins),
        "candidate_win_count": len(candidate_wins),
        "baseline_win_count": 0,
        "candidate_win_scenarios": candidate_wins,
        "baseline_win_scenarios": [],
        "task_completion_improvement_count": len(candidate_wins),
        "task_completion_regression_count": 0,
        "task_completion_improvement_scenarios": candidate_wins,
        "task_completion_regression_scenarios": [],
        "fixed_rule_counts": {},
        "regressed_rule_counts": {},
        "new_critical_failure_counts": {},
        "contract_drift_count": contract_drift_count,
        "unverified_contract_count": 0,
        "skipped_pair_count": 0,
        "missing_in_candidate": [],
        "new_in_candidate": [],
    }
    (path / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
