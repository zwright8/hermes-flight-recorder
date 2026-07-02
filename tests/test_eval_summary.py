import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main
from flightrecorder.schema_registry import check_schema_file


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
            schema_result = check_schema_file(out)

            self.assertEqual(code, 0)
            self.assertEqual(validate_code, 0)
            self.assertTrue(schema_result["passed"], schema_result["errors"])
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
            schema_result = check_schema_file(out)

            self.assertEqual(code, 1)
            self.assertEqual(validate_code, 0)
            self.assertTrue(schema_result["passed"], schema_result["errors"])
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

    def test_eval_summary_surfaces_operational_metrics_for_governance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = _suite_summary(
                root / "candidate_suite.json",
                ["email_reply_completion", "prompt_injection"],
                run_overrides=[
                    {
                        "cost_usd": 0.12,
                        "latency_ms": 1000,
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                        "task_completion": {"status": "complete", "passed": True},
                    },
                    {
                        "cost_usd": 0.08,
                        "latency_ms": 3000,
                        "input_tokens": 20,
                        "output_tokens": 7,
                        "total_tokens": 27,
                        "task_completion_status": "incomplete",
                        "task_completion_passed": False,
                    },
                ],
            )
            out = root / "eval_summary.json"

            code = run_cli(["eval-summary", "--suite-summary", f"candidate={suite}", "--out", str(out)])
            validate_code = run_cli(["validate", "--eval-summary", str(out), "--strict"])

            self.assertEqual(code, 0)
            self.assertEqual(validate_code, 0)
            operational = _read_json(out)["arms"][0]["operational_metrics"]
            self.assertEqual(operational["cost"]["source"], "run_rows")
            self.assertEqual(operational["cost"]["total_usd"], 0.2)
            self.assertEqual(operational["latency"]["average_ms"], 2000.0)
            self.assertEqual(operational["latency"]["p50_ms"], 2000.0)
            self.assertEqual(operational["latency"]["p95_ms"], 3000.0)
            self.assertEqual(operational["tokens"]["prompt_tokens"], 30)
            self.assertEqual(operational["tokens"]["completion_tokens"], 12)
            self.assertEqual(operational["tokens"]["total_tokens"], 42)
            self.assertEqual(operational["task_completion"]["configured_count"], 2)
            self.assertEqual(operational["task_completion"]["complete_count"], 1)
            self.assertEqual(operational["task_completion"]["incomplete_count"], 1)
            self.assertEqual(operational["task_completion"]["passed_count"], 1)
            self.assertEqual(operational["task_completion"]["failed_count"], 1)
            self.assertEqual(operational["task_completion"]["pass_rate"], 0.5)

    def test_eval_summary_emits_repair_curriculum_items_for_candidate_regressions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = _suite_summary(root / "baseline_suite.json", ["email_reply_completion"])
            candidate = _suite_summary(root / "candidate_suite.json", ["email_reply_completion"])
            compare_export = _compare_export(
                root / "compare_rl",
                candidate_wins=[],
                baseline_wins=["email_reply_completion"],
                task_completion_regressions=["email_reply_completion"],
                regressed_rule_counts={"required_actions": 1},
                new_critical_failure_counts={"secret_exposure": 1},
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
            repair = _read_json(out)["repair_curriculum"]
            reasons = {item["reason"] for item in repair["items"]}
            self.assertEqual(repair["work_item_count"], 4)
            self.assertEqual(repair["critical_work_item_count"], 2)
            self.assertIn("baseline_win", reasons)
            self.assertIn("task_completion_regression", reasons)
            self.assertIn("regressed_rule", reasons)
            self.assertIn("new_critical_failure", reasons)
            self.assertTrue(any(item["category"] == "curriculum" for item in repair["items"]))

    def test_eval_summary_can_require_ready_serving_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = _suite_summary(root / "candidate_suite.json", ["email_reply_completion"])
            serving_check = _serving_check(root / "serving_check.json", passed=True)
            out = root / "eval_summary.json"

            code = run_cli(
                [
                    "eval-summary",
                    "--suite-summary",
                    f"candidate={suite}",
                    "--serving-check",
                    f"candidate={serving_check}",
                    "--require-serving-preflight",
                    "--out",
                    str(out),
                ]
            )
            validate_code = run_cli(["validate", "--eval-summary", str(out), "--strict"])

            self.assertEqual(code, 0)
            self.assertEqual(validate_code, 0)
            summary = _read_json(out)
            self.assertEqual(summary["repair_curriculum"]["work_item_count"], 0)
            self.assertTrue(summary["serving_preflight"]["required"])
            self.assertEqual(summary["serving_preflight"]["attached_count"], 1)
            self.assertEqual(summary["serving_preflight"]["blocking_reasons"], [])
            self.assertTrue(summary["arms"][0]["serving_preflight"]["passed"])
            self.assertEqual(summary["arms"][0]["serving_preflight"]["model"], "hfr-mock-model")

    def test_eval_summary_blocks_missing_required_serving_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = _suite_summary(root / "candidate_suite.json", ["email_reply_completion"])
            out = root / "eval_summary.json"

            code = run_cli(
                [
                    "eval-summary",
                    "--suite-summary",
                    f"candidate={suite}",
                    "--require-serving-preflight",
                    "--out",
                    str(out),
                ]
            )
            validate_code = run_cli(["validate", "--eval-summary", str(out), "--strict"])

            self.assertEqual(code, 1)
            self.assertEqual(validate_code, 0)
            summary = _read_json(out)
            self.assertFalse(summary["passed"])
            self.assertIn("serving_preflight_missing", summary["arms"][0]["blocking_reasons"])
            self.assertIn({"source": "serving_preflight", "label": "candidate", "reason": "serving_preflight_missing"}, summary["risks"])

    def test_eval_summary_blocks_failed_serving_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = _suite_summary(root / "candidate_suite.json", ["email_reply_completion"])
            serving_check = _serving_check(root / "serving_check.json", passed=False)
            out = root / "eval_summary.json"

            code = run_cli(
                [
                    "eval-summary",
                    "--suite-summary",
                    f"candidate={suite}",
                    "--serving-check",
                    f"candidate={serving_check}",
                    "--require-serving-preflight",
                    "--out",
                    str(out),
                ]
            )
            validate_code = run_cli(["validate", "--eval-summary", str(out), "--strict"])

            self.assertEqual(code, 1)
            self.assertEqual(validate_code, 0)
            summary = _read_json(out)
            serving = summary["arms"][0]["serving_preflight"]
            self.assertFalse(serving["passed"])
            self.assertEqual(serving["failed_checks"], ["chat_completion"])
            self.assertIn("serving_preflight_blocked", serving["blocking_reasons"])

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

    def test_validate_rejects_malformed_operational_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = _suite_summary(root / "candidate_suite.json", ["email_reply_completion"])
            out = root / "eval_summary.json"
            validation = root / "validation.json"
            run_cli(["eval-summary", "--suite-summary", f"candidate={suite}", "--out", str(out)])
            summary = _read_json(out)
            operational = summary["arms"][0]["operational_metrics"]
            operational["cost"]["total_usd"] = -1
            operational["latency"]["known_run_count"] = "one"
            operational["tokens"]["total_tokens"] = 3.14
            operational["task_completion"]["pass_rate"] = 2
            out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--eval-summary", str(out), "--out", str(validation)])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in _read_json(validation)["targets"] for error in target["errors"])
            self.assertIn("operational_metrics.cost.total_usd", errors)
            self.assertIn("operational_metrics.latency.known_run_count", errors)
            self.assertIn("operational_metrics.tokens.total_tokens", errors)
            self.assertIn("operational_metrics.task_completion.pass_rate", errors)

    def test_validate_rejects_eval_summary_with_stale_repair_curriculum_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = _suite_summary(root / "baseline_suite.json", ["email_reply_completion"])
            candidate = _suite_summary(root / "candidate_suite.json", ["email_reply_completion"])
            compare_export = _compare_export(
                root / "compare_rl",
                candidate_wins=[],
                baseline_wins=["email_reply_completion"],
            )
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
            summary["repair_curriculum"]["work_item_count"] = 0
            out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--eval-summary", str(out), "--out", str(validation)])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in _read_json(validation)["targets"] for error in target["errors"])
            self.assertIn("repair_curriculum.work_item_count expected", errors)


def _suite_summary(path: Path, scenario_ids: list[str], run_overrides=None) -> Path:
    runs = [
        {
            "scenario_id": scenario_id,
            "task_family": scenario_id,
            "passed": True,
            "score": 100,
            "failed_rules": [],
            "critical_failures": [],
        }
        for index, scenario_id in enumerate(scenario_ids)
    ]
    for index, override in enumerate(run_overrides or []):
        if index < len(runs):
            runs[index].update(override)
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
    baseline_wins: list[str] | None = None,
    task_completion_regressions: list[str] | None = None,
    regressed_rule_counts: dict[str, int] | None = None,
    new_critical_failure_counts: dict[str, int] | None = None,
    contract_drift_count: int = 0,
) -> Path:
    path.mkdir(parents=True)
    baseline_wins = baseline_wins or []
    task_completion_regressions = task_completion_regressions or []
    payload = {
        "schema_version": "hfr.compare_rl.manifest.v1",
        "pair_count": len(candidate_wins) + len(baseline_wins),
        "candidate_win_count": len(candidate_wins),
        "baseline_win_count": len(baseline_wins),
        "candidate_win_scenarios": candidate_wins,
        "baseline_win_scenarios": baseline_wins,
        "task_completion_improvement_count": len(candidate_wins),
        "task_completion_regression_count": len(task_completion_regressions),
        "task_completion_improvement_scenarios": candidate_wins,
        "task_completion_regression_scenarios": task_completion_regressions,
        "fixed_rule_counts": {},
        "regressed_rule_counts": regressed_rule_counts or {},
        "new_critical_failure_counts": new_critical_failure_counts or {},
        "contract_drift_count": contract_drift_count,
        "unverified_contract_count": 0,
        "skipped_pair_count": 0,
        "missing_in_candidate": [],
        "new_in_candidate": [],
    }
    (path / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _serving_check(path: Path, *, passed: bool) -> Path:
    payload = {
        "schema_version": "hfr.serving_endpoint_check.v1",
        "generated_at": "2026-07-02T00:00:00Z",
        "passed": passed,
        "readiness": "ready" if passed else "blocked",
        "profile_id": "candidate-transformers-hfr-mock-model",
        "arm": "candidate",
        "model": "hfr-mock-model",
        "served_model_id": "hfr-mock-model",
        "base_url": "http://127.0.0.1:8000/v1",
        "checks": [{"id": "chat_completion", "passed": passed, "details": {}}],
        "failed_checks": [] if passed else ["chat_completion"],
        "artifacts": {
            "serving_profile": "serving_profile.json",
            "compatibility_report": "compatibility_report.json",
            "serving_check": "serving_check.json",
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
