import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main
from flightrecorder.schema_registry import check_schema_contract


ROOT = Path(__file__).resolve().parents[1]


def run_cli(args):
    with redirect_stdout(StringIO()):
        return main(args)


class ImprovementPlanTests(unittest.TestCase):
    def test_improvement_plan_joins_repair_curriculum_digest_and_bundle_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            plan_path = runs / "improvement_plan.json"
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
                        "--evidence-handoff",
                    ]
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    [
                        "improvement-plan",
                        "--evidence-bundle",
                        str(runs / "evidence_bundle.json"),
                        "--repair-queue",
                        str(runs / "repair_queue.json"),
                        "--training-export",
                        str(runs / "training_export"),
                        "--runs",
                        str(runs),
                        "--out",
                        str(plan_path),
                    ]
                ),
                0,
            )
            self.assertEqual(run_cli(["validate", "--improvement-plan", str(plan_path), "--strict"]), 0)
            self.assertEqual(run_cli(["schemas", "--check", str(plan_path)]), 0)

            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            repair_queue = json.loads((runs / "repair_queue.json").read_text(encoding="utf-8"))
            bundle = json.loads((runs / "evidence_bundle.json").read_text(encoding="utf-8"))
            self.assertEqual(plan["schema_version"], "hfr.improvement_plan.v1")
            self.assertEqual(plan["decision"]["recommendation"], "run_improvement_iteration")
            self.assertEqual(plan["readiness"], "ready")
            self.assertEqual(plan["metrics"]["repair_backed_count"], repair_queue["item_count"])
            self.assertEqual(plan["metrics"]["curriculum_backed_count"], repair_queue["item_count"])
            self.assertEqual(plan["metrics"]["digest_backed_count"], repair_queue["item_count"])
            self.assertEqual(plan["metrics"]["bundle_action_count"], bundle["decision"]["next_action_count"])
            self.assertGreater(plan["metrics"]["evidence_ref_count"], 0)
            self.assertIn("repair", {row["id"] for row in plan["metrics"]["category_counts"]})
            self.assertIn("bundle_action", {row["id"] for row in plan["metrics"]["category_counts"]})
            self.assertEqual(plan["work_item_count"], len(plan["work_items"]))
            self.assertTrue(all(len(item["fingerprint"]) == 64 for item in plan["work_items"]))
            self.assertEqual(plan["source_artifacts"]["evidence_bundle"]["path"], "evidence_bundle.json")
            self.assertEqual(len(plan["source_artifacts"]["evidence_bundle"]["sha256"]), 64)
            schema = check_schema_contract(plan, name_or_id="improvement_plan")
            self.assertTrue(schema["passed"], schema["errors"])
            bad_plan = json.loads(json.dumps(plan))
            bad_plan["source_artifacts"]["evidence_bundle"].pop("sha256")
            bad_schema = check_schema_contract(bad_plan, name_or_id="improvement_plan")
            self.assertFalse(bad_schema["passed"])
            self.assertIn("expected exactly one matching schema from oneOf, got 0", "\n".join(bad_schema["errors"]))
            diagnostic_plan = json.loads(json.dumps(plan))
            diagnostic_plan["source_artifacts"]["evidence_bundle"] = {
                "kind": "file",
                "path": "missing-evidence-bundle.json",
                "exists": False,
            }
            diagnostic_schema = check_schema_contract(diagnostic_plan, name_or_id="improvement_plan")
            self.assertTrue(diagnostic_schema["passed"], diagnostic_schema["errors"])
            repair_items = [item for item in plan["work_items"] if item["category"] == "repair"]
            self.assertEqual(len(repair_items), repair_queue["item_count"])
            self.assertTrue(all(item["sources"]["curriculum_priorities"] for item in repair_items))
            self.assertTrue(all(item["sources"]["run_digest"] for item in repair_items))
            self.assertTrue(any(item["scenario_id"] == "prompt_injection_bad" for item in repair_items))

    def test_improvement_plan_imports_eval_summary_repair_curriculum_items(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            plan_path = root / "improvement_plan.json"
            eval_summary = root / "eval_summary.json"
            compare_export = _compare_export(
                root / "compare_rl",
                baseline_wins=["email_reply_completion"],
                task_completion_regressions=["email_reply_completion"],
                regressed_rule_counts={"required_actions": 1},
                new_critical_failure_counts={"secret_exposure": 1},
                contract_drift_count=1,
            )
            baseline = _suite_summary(root / "baseline_suite.json", ["email_reply_completion"])
            candidate = _suite_summary(root / "candidate_suite.json", ["email_reply_completion"])
            self.assertEqual(
                run_cli(
                    [
                        "run-suite",
                        "--scenarios",
                        str(ROOT / "scenarios"),
                        "--out",
                        str(runs),
                        "--evidence-handoff",
                    ]
                ),
                0,
            )
            self.assertEqual(
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
                        str(eval_summary),
                    ]
                ),
                1,
            )
            self.assertEqual(
                run_cli(
                    [
                        "improvement-plan",
                        "--evidence-bundle",
                        str(runs / "evidence_bundle.json"),
                        "--eval-summary",
                        str(eval_summary),
                        "--out",
                        str(plan_path),
                    ]
                ),
                0,
            )
            self.assertEqual(run_cli(["validate", "--improvement-plan", str(plan_path), "--strict"]), 0)
            self.assertEqual(run_cli(["schemas", "--check", str(plan_path)]), 0)

            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            eval_items = [
                item
                for item in plan["work_items"]
                if item["sources"].get("eval_summary_items")
            ]
            reasons = {item["sources"]["eval_summary_items"][0]["reason"] for item in eval_items}
            categories = {item["category"] for item in eval_items}
            self.assertIn("eval_summary", plan["source_artifacts"])
            self.assertEqual(plan["source_artifacts"]["evidence_bundle"]["path"], "runs/evidence_bundle.json")
            self.assertEqual(plan["source_artifacts"]["eval_summary"]["path"], "eval_summary.json")
            self.assertIn("baseline_win", reasons)
            self.assertIn("task_completion_regression", reasons)
            self.assertIn("regressed_rule", reasons)
            self.assertIn("new_critical_failure", reasons)
            self.assertIn("contract_fingerprint_drift", reasons)
            self.assertIn("repair", categories)
            self.assertIn("curriculum", categories)
            self.assertIn("bundle_action", categories)

    def test_validate_rejects_improvement_plan_missing_eval_summary_items(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            eval_plan_path = root / "improvement_plan_with_eval.json"
            forged_plan_path = root / "improvement_plan.json"
            validation = root / "validation.json"
            eval_summary = root / "eval_summary.json"
            compare_export = _compare_export(
                root / "compare_rl",
                baseline_wins=["email_reply_completion"],
                task_completion_regressions=["email_reply_completion"],
                regressed_rule_counts={"required_actions": 1},
                new_critical_failure_counts={"secret_exposure": 1},
                contract_drift_count=1,
            )
            baseline = _suite_summary(root / "baseline_suite.json", ["email_reply_completion"])
            candidate = _suite_summary(root / "candidate_suite.json", ["email_reply_completion"])
            self.assertEqual(
                run_cli(
                    [
                        "run-suite",
                        "--scenarios",
                        str(ROOT / "scenarios"),
                        "--out",
                        str(runs),
                        "--evidence-handoff",
                    ]
                ),
                0,
            )
            self.assertEqual(
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
                        str(eval_summary),
                    ]
                ),
                1,
            )
            self.assertEqual(
                run_cli(
                    [
                        "improvement-plan",
                        "--evidence-bundle",
                        str(runs / "evidence_bundle.json"),
                        "--eval-summary",
                        str(eval_summary),
                        "--out",
                        str(eval_plan_path),
                    ]
                ),
                0,
            )
            self.assertEqual(run_cli(["validate", "--improvement-plan", str(eval_plan_path), "--strict"]), 0)
            self.assertEqual(
                run_cli(
                    [
                        "improvement-plan",
                        "--evidence-bundle",
                        str(runs / "evidence_bundle.json"),
                        "--out",
                        str(forged_plan_path),
                    ]
                ),
                0,
            )
            eval_plan = json.loads(eval_plan_path.read_text(encoding="utf-8"))
            forged_plan = json.loads(forged_plan_path.read_text(encoding="utf-8"))
            forged_plan["source_artifacts"]["eval_summary"] = eval_plan["source_artifacts"]["eval_summary"]
            forged_plan_path.write_text(json.dumps(forged_plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--improvement-plan", str(forged_plan_path), "--strict", "--out", str(validation)])

            self.assertEqual(code, 1)
            summary = json.loads(validation.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("improvement_plan.work_items missing", errors)
            self.assertIn("eval_summary item", errors)

    def test_validate_rejects_stale_improvement_plan_metrics_and_fingerprints(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            plan_path = runs / "improvement_plan.json"
            summary_path = runs / "validation.json"
            self.assertEqual(
                run_cli(
                    [
                        "run-suite",
                        "--scenarios",
                        str(ROOT / "scenarios"),
                        "--out",
                        str(runs),
                        "--export-rl",
                        "--evidence-handoff",
                    ]
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    [
                        "improvement-plan",
                        "--evidence-bundle",
                        str(runs / "evidence_bundle.json"),
                        "--repair-queue",
                        str(runs / "repair_queue.json"),
                        "--training-export",
                        str(runs / "training_export"),
                        "--runs",
                        str(runs),
                        "--out",
                        str(plan_path),
                    ]
                ),
                0,
            )
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan["metrics"]["work_item_count"] += 1
            plan["work_items"][0]["fingerprint"] = "0" * 64
            plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--improvement-plan", str(plan_path), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("improvement_plan.metrics.work_item_count", errors)
            self.assertIn("fingerprint does not match item content", errors)

    def test_validate_rejects_stale_improvement_plan_source_artifact_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            plan_path = runs / "improvement_plan.json"
            summary_path = runs / "validation.json"
            self.assertEqual(
                run_cli(
                    [
                        "run-suite",
                        "--scenarios",
                        str(ROOT / "scenarios"),
                        "--out",
                        str(runs),
                        "--evidence-handoff",
                    ]
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    [
                        "improvement-plan",
                        "--evidence-bundle",
                        str(runs / "evidence_bundle.json"),
                        "--out",
                        str(plan_path),
                    ]
                ),
                0,
            )
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan["source_artifacts"]["evidence_bundle"]["sha256"] = "0" * 64
            plan["source_artifacts"]["evidence_bundle"]["size_bytes"] += 1
            plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--improvement-plan", str(plan_path), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("improvement_plan.source_artifacts.evidence_bundle.sha256 does not match the current file.", errors)
            self.assertIn("improvement_plan.source_artifacts.evidence_bundle.size_bytes does not match the current file.", errors)

    def test_validate_rejects_missing_existing_improvement_plan_source_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            plan_path = runs / "improvement_plan.json"
            summary_path = runs / "validation.json"
            self.assertEqual(
                run_cli(
                    [
                        "run-suite",
                        "--scenarios",
                        str(ROOT / "scenarios"),
                        "--out",
                        str(runs),
                        "--evidence-handoff",
                    ]
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    [
                        "improvement-plan",
                        "--evidence-bundle",
                        str(runs / "evidence_bundle.json"),
                        "--out",
                        str(plan_path),
                    ]
                ),
                0,
            )
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan["source_artifacts"]["evidence_bundle"]["path"] = "missing-evidence-bundle.json"
            plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--improvement-plan", str(plan_path), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("improvement_plan.source_artifacts.evidence_bundle.path must resolve to an existing file when exists is true.", errors)

    def test_validate_rejects_symlink_existing_improvement_plan_source_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            plan_path = runs / "improvement_plan.json"
            summary_path = runs / "validation.json"
            self.assertEqual(
                run_cli(
                    [
                        "run-suite",
                        "--scenarios",
                        str(ROOT / "scenarios"),
                        "--out",
                        str(runs),
                        "--evidence-handoff",
                    ]
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    [
                        "improvement-plan",
                        "--evidence-bundle",
                        str(runs / "evidence_bundle.json"),
                        "--out",
                        str(plan_path),
                    ]
                ),
                0,
            )
            symlink_path = runs / "evidence_bundle_link.json"
            symlink_path.symlink_to(runs / "evidence_bundle.json")
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan["source_artifacts"]["evidence_bundle"]["path"] = symlink_path.name
            plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--improvement-plan", str(plan_path), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("improvement_plan.source_artifacts.evidence_bundle.path must resolve to a regular file when exists is true.", errors)

    def test_validate_rejects_present_improvement_plan_source_artifact_marked_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            plan_path = runs / "improvement_plan.json"
            summary_path = runs / "validation.json"
            self.assertEqual(
                run_cli(
                    [
                        "run-suite",
                        "--scenarios",
                        str(ROOT / "scenarios"),
                        "--out",
                        str(runs),
                        "--evidence-handoff",
                    ]
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    [
                        "improvement-plan",
                        "--evidence-bundle",
                        str(runs / "evidence_bundle.json"),
                        "--out",
                        str(plan_path),
                    ]
                ),
                0,
            )
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            artifact = plan["source_artifacts"]["evidence_bundle"]
            artifact["exists"] = False
            artifact["sha256"] = "0" * 64
            artifact["size_bytes"] += 1
            plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--improvement-plan", str(plan_path), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("improvement_plan.source_artifacts.evidence_bundle.sha256 does not match the current file.", errors)
            self.assertIn("improvement_plan.source_artifacts.evidence_bundle.size_bytes does not match the current file.", errors)


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
    baseline_wins: list[str],
    task_completion_regressions: list[str],
    regressed_rule_counts: dict[str, int],
    new_critical_failure_counts: dict[str, int],
    contract_drift_count: int,
) -> Path:
    path.mkdir(parents=True)
    payload = {
        "schema_version": "hfr.compare_rl.manifest.v1",
        "pair_count": len(baseline_wins),
        "candidate_win_count": 0,
        "baseline_win_count": len(baseline_wins),
        "candidate_win_scenarios": [],
        "baseline_win_scenarios": baseline_wins,
        "task_completion_improvement_count": 0,
        "task_completion_regression_count": len(task_completion_regressions),
        "task_completion_improvement_scenarios": [],
        "task_completion_regression_scenarios": task_completion_regressions,
        "fixed_rule_counts": {},
        "regressed_rule_counts": regressed_rule_counts,
        "new_critical_failure_counts": new_critical_failure_counts,
        "contract_drift_count": contract_drift_count,
        "unverified_contract_count": 0,
        "skipped_pair_count": 0,
        "missing_in_candidate": [],
        "new_in_candidate": [],
    }
    (path / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
