import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.agentic_training_loop_plan import build_agentic_training_loop_plan, write_agentic_training_loop_plan
from flightrecorder.cli import main
from flightrecorder.schema_registry import list_schema_records


def run_cli(args):
    with redirect_stdout(StringIO()):
        return main(args)


class AgenticLoopLedgerTests(unittest.TestCase):
    def test_agentic_loop_ledger_tracks_blocked_and_ready_iterations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blocked_plan = self.write_loop_plan(root / "plans" / "blocked.json", "loop-001", {})
            ready_artifacts = self.write_ready_artifacts(root / "ready")
            ready_plan = self.write_loop_plan(root / "ready" / "plan.json", "loop-002", ready_artifacts)
            ledger = root / "ledger.json"

            code = run_cli(["agentic-loop", "ledger", "--plan", str(blocked_plan), "--plan", str(ready_plan), "--out", str(ledger)])

            self.assertEqual(code, 0)
            payload = json.loads(ledger.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "hfr.agentic_loop_ledger.v1")
            self.assertTrue(payload["passed"])
            self.assertEqual(payload["iteration_count"], 2)
            self.assertEqual(payload["metrics"]["ready_iteration_count"], 1)
            self.assertEqual(payload["metrics"]["blocked_iteration_count"], 1)
            self.assertEqual(payload["metrics"]["latest_iteration_id"], "loop-002")
            self.assertEqual(payload["decision"]["recommendation"], "ready_for_governance_review")
            self.assertFalse(payload["execution_boundary"]["cloud_jobs_started"])
            ready_record = payload["iterations"][1]
            groups = {row["group"]: row["count"] for row in ready_record["artifact_group_counts"]}
            self.assertGreater(groups["rollouts"], 0)
            self.assertGreater(groups["review"], 0)
            self.assertGreater(groups["training"], 0)
            self.assertGreater(groups["eval"], 0)
            self.assertTrue(ready_record["governance"]["promotion_decision_present"])
            self.assertFalse(ready_record["governance"]["weights_updated_by_flight_recorder"])
            self.assertEqual(run_cli(["validate", "--agentic-loop-ledger", str(ledger), "--strict"]), 0)
            self.assertEqual(run_cli(["schemas", "--check", str(ledger)]), 0)

    def test_validate_rejects_stale_agentic_loop_ledger_source_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = self.write_loop_plan(root / "plans" / "loop.json", "loop-001", {})
            ledger = root / "ledger.json"
            summary = root / "summary.json"
            self.assertEqual(run_cli(["agentic-loop", "ledger", "--plan", str(plan), "--out", str(ledger)]), 0)
            self.assertEqual(run_cli(["validate", "--agentic-loop-ledger", str(ledger), "--strict"]), 0)
            payload = json.loads(plan.read_text(encoding="utf-8"))
            payload["objective"] = "tampered after ledger"
            plan.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--agentic-loop-ledger", str(ledger), "--out", str(summary)])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in json.loads(summary.read_text(encoding="utf-8"))["targets"] for error in target["errors"])
            self.assertIn("sha256 does not match the current file", errors)

    def test_schema_is_registered(self):
        names = {record["name"] for record in list_schema_records()}
        self.assertIn("agentic_loop_ledger", names)

    def write_loop_plan(self, path: Path, iteration_id: str, artifacts: dict[str, list[Path]]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        plan = build_agentic_training_loop_plan(
            out_path=path,
            iteration_id=iteration_id,
            objective=f"Iteration {iteration_id}",
            artifact_paths=artifacts,
            budget={"max_cloud_cost_usd": 0, "max_gpu_hours": 0},
            created_at="2026-07-03T00:00:00+00:00",
        )
        write_agentic_training_loop_plan(path, plan)
        return path

    def write_ready_artifacts(self, root: Path) -> dict[str, list[Path]]:
        root.mkdir(parents=True, exist_ok=True)
        training_export = root / "training_export"
        training_export.mkdir()
        return {
            "agentic_rollout_plan": [self.write_json(root / "agentic_rollout_plan.json", "hfr.agentic_rollout_plan.v1")],
            "agentic_rollout_receipt": [self.write_json(root / "agentic_rollout_receipt.json", "hfr.agentic_rollout_receipt.v1")],
            "harness_result": [self.write_json(root / "harness_result.json", "hfr.harness_run_result.v1")],
            "evidence_bundle": [self.write_json(root / "evidence_bundle.json", "hfr.evidence_bundle.v1")],
            "rubric_spec": [self.write_json(root / "rubric_spec.json", "hfr.rubric_spec.v1")],
            "model_grader_dry_run": [self.write_json(root / "model_grader_dry_run.json", "hfr.model_grader_dry_run.v1")],
            "model_grader_gate": [self.write_json(root / "model_grader_gate.json", "hfr.model_grader_gate.v1")],
            "review_calibration": [self.write_json(root / "review_calibration.json", "hfr.review_calibration.v1")],
            "reviewed_gate": [self.write_json(root / "reviewed_gate.json", "hfr.reviewed_gate.v1")],
            "rejection_sampling_gate": [self.write_json(root / "rejection_sampling_gate.json", "hfr.rejection_sampling_gate.v1")],
            "dataset_curation_receipt": [self.write_json(root / "dataset_curation_receipt.json", "hfr.dataset_curation_receipt.v1")],
            "training_export": [training_export],
            "agentic_training_plan": [self.write_json(root / "agentic_training_plan.json", "hfr.agentic_training_plan.v1")],
            "agentic_training_flow": [self.write_json(root / "agentic_training_flow.json", "hfr.agentic_training_flow.v1")],
            "trainer_preflight": [self.write_json(root / "trainer_preflight.json", "hfr.trainer_preflight.v1")],
            "trainer_launch_check": [self.write_json(root / "trainer_launch_check.json", "hfr.trainer_launch_check.v1")],
            "serving_lifecycle": [self.write_json(root / "serving_lifecycle.json", "hfr.serving_lifecycle.v1")],
            "heldout_manifest": [self.write_json(root / "heldout_manifest.json", "hfr.heldout_scenario_manifest.v1")],
            "external_eval_plan": [self.write_json(root / "external_eval_plan.json", "hfr.external_eval_adapters.v1")],
            "external_eval_receipt": [self.write_json(root / "external_eval_receipt.json", "hfr.external_eval_receipt.v1")],
            "eval_summary": [self.write_json(root / "eval_summary.json", "hfr.eval_summary.v1")],
            "improvement_plan": [self.write_json(root / "improvement_plan.json", "hfr.improvement_plan.v1")],
            "promotion_decision": [self.write_json(root / "promotion_decision.json", "hfr.promotion_decision.v1")],
            "promotion_ledger": [self.write_json(root / "promotion_ledger.json", "hfr.promotion_ledger.v1")],
            "next_iteration_schedule": [self.write_json(root / "next_iteration_schedule.json", "hfr.next_iteration_schedule.v1")],
            "action_ledger": [self.write_json(root / "action_ledger.json", "hfr.action_ledger.v1")],
        }

    def write_json(self, path: Path, schema_version: str) -> Path:
        path.write_text(
            json.dumps({"schema_version": schema_version, "passed": True, "readiness": "ready"}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path


if __name__ == "__main__":
    unittest.main()
