import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.compare_agentic_finetune_results import compare, write_repair_outputs
from scripts.evaluate_hermes_heldout import _build_evaluation_summary
from scripts.plan_external_eval_adapters import build_external_eval_adapters


class AgenticEvalLayerTests(unittest.TestCase):
    def test_comparison_blocks_all_claims_when_scenarios_differ(self):
        baseline = _comparison_summary(["alpha", "beta"], pass_rate=0.1, score=50, critical=5, task_rate=0.2)
        trace_only = _comparison_summary(["alpha", "beta"], pass_rate=0.2, score=55, critical=4, task_rate=0.3)
        flightrecorder = _comparison_summary(["alpha"], pass_rate=1.0, score=100, critical=0, task_rate=1.0)

        result = compare(baseline, trace_only, flightrecorder)

        self.assertFalse(result["passed"])
        self.assertEqual(result["comparison_status"], "not_comparable")
        self.assertFalse(result["governance_handoff"]["ready"])
        self.assertEqual(result["governance_handoff"]["recommendation"], "rerun_identical_scenario_eval")
        comparative_checks = [item for item in result["checks"] if item["id"] != "same_heldout_scenarios"]
        self.assertTrue(comparative_checks)
        self.assertTrue(all(not item["passed"] for item in comparative_checks))
        self.assertTrue(all(item.get("blocked_by") == "same_heldout_scenarios" for item in comparative_checks))
        self.assertIn("beta", result["scenario_set"]["missing_by_arm"]["flightrecorder"])
        self.assertTrue(result["repair_work_items"])

    def test_comparison_marks_identical_scenarios_governance_ready(self):
        scenario_ids = ["alpha", "beta"]
        baseline = _comparison_summary(scenario_ids, pass_rate=0.1, score=50, critical=5, task_rate=0.2)
        trace_only = _comparison_summary(scenario_ids, pass_rate=0.2, score=55, critical=4, task_rate=0.3)
        flightrecorder = _comparison_summary(scenario_ids, pass_rate=0.9, score=90, critical=1, task_rate=0.8)

        result = compare(baseline, trace_only, flightrecorder)

        self.assertTrue(result["passed"])
        self.assertEqual(result["comparison_status"], "comparable")
        self.assertTrue(result["scenario_set"]["identical"])
        self.assertTrue(result["governance_handoff"]["ready"])
        self.assertEqual(result["governance_handoff"]["recommendation"], "send_to_governance")
        self.assertEqual(result["repair_work_items"], [])

    def test_evaluation_summary_contains_governance_handoff_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            run_dir = out_dir / "alpha"
            run_dir.mkdir()
            (run_dir / "task_completion.json").write_text(
                json.dumps(
                    {
                        "task_evidence_configured": True,
                        "status": "complete",
                        "passed_check_count": 1,
                        "required_check_count": 2,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (out_dir / "evaluation_plan.json").write_text("{}\n", encoding="utf-8")
            suite_summary = {
                "metrics": {
                    "pass_rate": 1.0,
                    "average_score": 90.0,
                    "critical_failure_counts": [{"id": "required_evidence", "count": 1}],
                    "failed_rule_counts": [{"id": "required_evidence", "count": 1}],
                },
                "runs": [
                    {
                        "scenario_id": "alpha",
                        "run_dir": str(run_dir),
                        "cost_usd": 0.125,
                        "latency_ms": 2500,
                    }
                ],
                "total": 1,
                "passed": 1,
                "failed": 0,
                "error_count": 0,
            }
            (out_dir / "suite_summary.json").write_text(json.dumps(suite_summary) + "\n", encoding="utf-8")
            args = SimpleNamespace(
                arm="candidate",
                model="local-model",
                provider="custom",
                split="heldout",
                mock_response=None,
            )

            summary = _build_evaluation_summary(
                args=args,
                out_dir=out_dir,
                base_url="http://127.0.0.1:8000/v1",
                scenario_paths=[Path("alpha.json")],
                suite_summary=suite_summary,
                errors=[],
                serving_profile=None,
            )

            self.assertEqual(summary["scenario_ids"], ["alpha"])
            self.assertEqual(len(summary["scenario_set_fingerprint"]), 64)
            self.assertEqual(summary["critical_failure_total"], 1)
            self.assertEqual(summary["task_completion"]["check_pass_rate"], 0.5)
            self.assertEqual(summary["cost"]["total_usd"], 0.125)
            self.assertEqual(summary["latency"]["average_seconds"], 2.5)
            self.assertEqual(len(summary["artifact_hashes"]["evaluation_plan"]["sha256"]), 64)
            self.assertTrue(summary["governance_handoff"]["ready"])

    def test_external_eval_adapter_plan_fails_closed_by_default(self):
        result = build_external_eval_adapters(
            adapter_ids=["bfcl", "inspect_ai", "lm_eval_harness", "swe_bench"],
            suite_id="external",
            model="",
            base_url="",
            scenario_manifest="",
            allow_installed=False,
        )

        self.assertEqual(result["schema_version"], "hfr.external_eval_adapters.v1")
        self.assertFalse(result["ready"])
        self.assertEqual(result["adapter_count"], 4)
        self.assertEqual(result["ready_adapter_count"], 0)
        self.assertIn("adapter_disabled_until_explicitly_enabled", result["blocking_reasons"])
        self.assertIn("missing_required_inputs", result["blocking_reasons"])
        self.assertFalse(result["governance_handoff"]["ready"])
        self.assertEqual(result["governance_handoff"]["recommendation"], "keep_external_adapters_disabled")
        self.assertTrue(
            all(adapter["runner_contract"]["must_not_claim_success_without_identical_scenarios"] for adapter in result["adapters"])
        )

    def test_external_eval_adapter_plan_hashes_scenario_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "heldout.json"
            manifest.write_text('{"heldout": []}\n', encoding="utf-8")

            result = build_external_eval_adapters(
                adapter_ids=["bfcl"],
                suite_id="external",
                model="hfr-model",
                base_url="http://127.0.0.1:8000/v1",
                scenario_manifest=str(manifest),
                allow_installed=False,
            )

            self.assertEqual(result["inputs"]["scenario_manifest"]["path"], str(manifest))
            self.assertTrue(result["inputs"]["scenario_manifest"]["exists"])
            self.assertEqual(len(result["inputs"]["scenario_manifest"]["sha256"]), 64)
            self.assertIn("tool_schema_set", result["adapters"][0]["missing_inputs"])

    def test_comparison_writes_standalone_repair_and_curriculum_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline = _comparison_summary(["alpha", "beta"], pass_rate=0.1, score=50, critical=5, task_rate=0.2)
            trace_only = _comparison_summary(["alpha", "beta"], pass_rate=0.2, score=55, critical=4, task_rate=0.3)
            flightrecorder = _comparison_summary(["alpha"], pass_rate=1.0, score=100, critical=0, task_rate=1.0)
            result = compare(baseline, trace_only, flightrecorder)
            repair_path = Path(tmp) / "eval_repair_work_items.json"
            curriculum_path = Path(tmp) / "eval_curriculum_suggestions.json"

            artifacts = write_repair_outputs(result, repair_out=repair_path, curriculum_out=curriculum_path)

            self.assertEqual(len(artifacts["eval_repair_work_items"]["sha256"]), 64)
            repair = json.loads(repair_path.read_text(encoding="utf-8"))
            curriculum = json.loads(curriculum_path.read_text(encoding="utf-8"))
            self.assertEqual(repair["schema_version"], "hfr.eval.repair_work_items.v1")
            self.assertGreater(repair["item_count"], 0)
            self.assertEqual(curriculum["schema_version"], "hfr.eval.curriculum_suggestions.v1")
            self.assertGreater(curriculum["suggestion_count"], 0)
            self.assertEqual(curriculum["suggestions"][0]["curriculum_focus"], "heldout_scenario_set_discipline")


def _comparison_summary(
    scenario_ids: list[str],
    *,
    pass_rate: float,
    score: float,
    critical: int,
    task_rate: float,
) -> dict:
    return {
        "path": "suite_summary.json",
        "total": len(scenario_ids),
        "passed": int(pass_rate * len(scenario_ids)),
        "failed": len(scenario_ids) - int(pass_rate * len(scenario_ids)),
        "pass_rate": pass_rate,
        "average_score": score,
        "critical_failure_counts": {"required_evidence": critical},
        "critical_failure_total": critical,
        "failed_rule_counts": {"required_evidence": critical},
        "forbidden_action_failures": 0,
        "unsupported_claim_failures": critical,
        "scenario_ids": scenario_ids,
        "task_completion": {
            "configured": len(scenario_ids),
            "complete": int(task_rate * len(scenario_ids)),
            "incomplete": len(scenario_ids) - int(task_rate * len(scenario_ids)),
            "passed_checks": int(task_rate * 10),
            "required_checks": 10,
            "check_pass_rate": task_rate,
            "missing_files": [],
        },
        "artifact_hashes": {},
    }
