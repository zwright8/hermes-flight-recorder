import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from flightrecorder.agentic_training_loop_plan import build_agentic_training_loop_plan
from flightrecorder.schema_registry import check_schema_contract, check_schema_file, list_schema_records
from flightrecorder.validation import validate_artifacts


ROOT = Path(__file__).resolve().parents[1]


class AgenticTrainingLoopPlanTests(unittest.TestCase):
    def test_complete_receipt_set_is_ready_for_governance_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)

            plan = build_agentic_training_loop_plan(
                out_path=root / "loop.json",
                iteration_id="loop-001",
                objective="Close the held-out tool-use regression.",
                baseline="local/baseline",
                candidate="local/candidate",
                teacher="local/teacher",
                artifact_paths=artifacts,
                budget={"max_rollouts": 20, "max_cloud_cost_usd": 0, "max_gpu_hours": 0},
                provider_constraints={"providers": ["mock"], "regions": ["local"], "gpu_classes": ["none"]},
                created_at="2026-07-03T00:00:00+00:00",
            )

            self.assertTrue(plan["passed"], plan["blocked_reasons"])
            self.assertEqual(plan["readiness"], "ready_for_governance_review")
            self.assertEqual(plan["recommendation"], "approve_iteration_execution")
            self.assertFalse(plan["execution_boundary"]["cloud_jobs_started"])
            self.assertFalse(plan["handoff_contract"]["default_live_execution_allowed"])
            self.assertEqual(plan["missing_phase_inputs"], [])
            self.assertTrue(all(phase["status"] != "blocked" for phase in plan["phases"]))
            self.assertEqual(plan["artifact_count"], sum(len(paths) for paths in artifacts.values()))
            schema = check_schema_contract(plan)
            self.assertTrue(schema["passed"], schema["errors"])

    def test_missing_receipts_keep_loop_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agentic_plan = self.write_json(root / "agentic_training_plan.json", "hfr.agentic_training_plan.v1")

            plan = build_agentic_training_loop_plan(
                out_path=root / "loop.json",
                iteration_id="loop-002",
                artifact_paths={"agentic_training_plan": [agentic_plan]},
                created_at="2026-07-03T00:00:00+00:00",
            )

            self.assertFalse(plan["passed"])
            self.assertEqual(plan["readiness"], "planned_fail_closed")
            self.assertIn("agentic_rollout_receipt", plan["missing_phase_inputs"])
            self.assertIn("rejection_sampling_gate", plan["missing_phase_inputs"])
            self.assertIn("dataset_curation_receipt", plan["missing_phase_inputs"])
            self.assertIn("trainer_preflight", plan["missing_phase_inputs"])
            self.assertIn("uncalibrated_labels_block_training_data", {check["id"] for check in plan["checks"] if not check["passed"]})
            self.assertIn("rollout_receipt_required_before_review", {check["id"] for check in plan["checks"] if not check["passed"]})
            schema = check_schema_contract(plan)
            self.assertTrue(schema["passed"], schema["errors"])

    def test_cli_writes_schema_checkable_and_validatable_loop_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            out = root / "loop.json"

            command = [
                sys.executable,
                "-m",
                "flightrecorder",
                "agentic-loop",
                "plan",
                "--iteration-id",
                "loop-cli",
                "--objective",
                "CLI closed-loop smoke",
                "--baseline",
                "local/baseline",
                "--candidate",
                "local/candidate",
                "--provider",
                "mock",
                "--region",
                "local",
                "--gpu-class",
                "none",
                "--budget",
                "max_cloud_cost_usd=0",
                "--out",
                str(out),
            ]
            for option, role in (
                ("--agentic-rollout-plan", "agentic_rollout_plan"),
                ("--agentic-rollout-receipt", "agentic_rollout_receipt"),
                ("--harness-result", "harness_result"),
                ("--evidence-bundle", "evidence_bundle"),
                ("--rubric-spec", "rubric_spec"),
                ("--model-grader-dry-run", "model_grader_dry_run"),
                ("--model-grader-gate", "model_grader_gate"),
                ("--review-calibration", "review_calibration"),
                ("--reviewed-gate", "reviewed_gate"),
                ("--rejection-sampling-gate", "rejection_sampling_gate"),
                ("--dataset-curation-receipt", "dataset_curation_receipt"),
                ("--training-export", "training_export"),
                ("--agentic-training-plan", "agentic_training_plan"),
                ("--agentic-training-flow", "agentic_training_flow"),
                ("--trainer-preflight", "trainer_preflight"),
                ("--trainer-launch-check", "trainer_launch_check"),
                ("--serving-lifecycle", "serving_lifecycle"),
                ("--heldout-manifest", "heldout_manifest"),
                ("--external-eval-plan", "external_eval_plan"),
                ("--external-eval-receipt", "external_eval_receipt"),
                ("--eval-summary", "eval_summary"),
                ("--improvement-plan", "improvement_plan"),
                ("--promotion-decision", "promotion_decision"),
                ("--promotion-ledger", "promotion_ledger"),
                ("--next-iteration-schedule", "next_iteration_schedule"),
                ("--action-ledger", "action_ledger"),
            ):
                command.extend([option, str(artifacts[role][0])])

            completed = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            schema = check_schema_file(out)
            self.assertTrue(schema["passed"], schema["errors"])
            validation = validate_artifacts(agentic_training_loop_plan_paths=[out], strict=True)
            self.assertTrue(validation["passed"], validation)

            validate_completed = subprocess.run(
                [sys.executable, "-m", "flightrecorder", "validate", "--agentic-loop-plan", str(out), "--strict"],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(validate_completed.returncode, 0, validate_completed.stderr + validate_completed.stdout)

    def test_validate_rejects_stale_or_moved_loop_plan_source_refs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agentic_plan = self.write_json(root / "agentic_training_plan.json", "hfr.agentic_training_plan.v1")
            loop_plan = root / "loop.json"
            plan = build_agentic_training_loop_plan(
                out_path=loop_plan,
                iteration_id="loop-stale-source",
                artifact_paths={"agentic_training_plan": [agentic_plan]},
                created_at="2026-07-03T00:00:00+00:00",
            )
            loop_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)
            self.assertTrue(validation["passed"], validation)

            copied_plan = root / "copy" / "loop.json"
            copied_plan.parent.mkdir()
            copied_plan.write_text(loop_plan.read_text(encoding="utf-8"), encoding="utf-8")
            copied_validation = validate_artifacts(agentic_training_loop_plan_paths=[copied_plan], strict=True)
            self.assertFalse(copied_validation["passed"])
            copied_errors = "\n".join(error for target in copied_validation["targets"] for error in target["errors"])
            self.assertIn("agentic_training_loop_plan.source_artifacts.agentic_training_plan[0].path does not resolve to an existing file.", copied_errors)

            payload = json.loads(agentic_plan.read_text(encoding="utf-8"))
            payload["stale_after_plan_write"] = True
            agentic_plan.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            stale_validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)
            self.assertFalse(stale_validation["passed"])
            stale_errors = "\n".join(error for target in stale_validation["targets"] for error in target["errors"])
            self.assertIn("agentic_training_loop_plan.source_artifacts.agentic_training_plan[0].size_bytes does not match the current file.", stale_errors)
            self.assertIn("agentic_training_loop_plan.source_artifacts.agentic_training_plan[0].sha256 does not match the current file.", stale_errors)

    def test_loop_plan_refs_are_relative_to_output_directory_for_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            runs.mkdir()
            self.write_json(runs / "agentic_training_plan.json", "hfr.agentic_training_plan.v1")
            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                plan = build_agentic_training_loop_plan(
                    out_path=Path("runs/agentic_training_loop_plan.json"),
                    iteration_id="loop-documented-paths",
                    artifact_paths={"agentic_training_plan": [Path("runs/agentic_training_plan.json")]},
                    created_at="2026-07-03T00:00:00+00:00",
                )
            finally:
                os.chdir(previous_cwd)

            ref = plan["source_artifacts"]["agentic_training_plan"][0]
            self.assertEqual(ref["path"], "agentic_training_plan.json")
            loop_plan = runs / "agentic_training_loop_plan.json"
            loop_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)
            self.assertTrue(validation["passed"], validation)

    def test_schema_is_registered(self):
        names = {record["name"] for record in list_schema_records()}
        self.assertIn("agentic_training_loop_plan", names)

    def write_loop_artifacts(self, root: Path) -> dict[str, list[Path]]:
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
            json.dumps(
                {
                    "schema_version": schema_version,
                    "passed": True,
                    "readiness": "ready",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return path
