import hashlib
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
            self.assertEqual(plan["cloud_training"]["missing_artifacts"], [])
            self.assertTrue(plan["cloud_training_lineage"]["passed"])
            self.assertEqual(plan["cloud_training_lineage"]["matched_link_count"], plan["cloud_training_lineage"]["required_link_count"])
            self.assertEqual(plan["cloud_training_lineage"]["provider"]["pipeline_provider_id"], "modal")
            self.assertFalse(plan["cloud_training"]["provider_api_calls_started"])
            self.assertTrue(plan["cloud_training_receipt_state"]["fail_closed"])
            self.assertEqual(plan["cloud_training_receipt_state"]["launch_mode"], "dry_run")
            self.assertEqual(plan["cloud_training_receipt_state"]["status_provider_status"], "not_started")
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
            self.assertIn("cloud_training_preflight", plan["missing_phase_inputs"])
            self.assertIn("uncalibrated_labels_block_training_data", {check["id"] for check in plan["checks"] if not check["passed"]})
            self.assertIn("rollout_receipt_required_before_review", {check["id"] for check in plan["checks"] if not check["passed"]})
            self.assertIn("cloud_training_receipts_bound_for_provider_handoff", {check["id"] for check in plan["checks"] if not check["passed"]})
            self.assertIn("cloud_training_lineage_bound_for_provider_handoff", {check["id"] for check in plan["checks"] if not check["passed"]})
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
                ("--model-grader-override-receipt", "model_grader_override_receipt"),
                ("--model-grader-gate", "model_grader_gate"),
                ("--review-calibration", "review_calibration"),
                ("--reviewed-gate", "reviewed_gate"),
                ("--rejection-sampling-gate", "rejection_sampling_gate"),
                ("--dataset-curation-receipt", "dataset_curation_receipt"),
                ("--training-export", "training_export"),
                ("--agentic-training-plan", "agentic_training_plan"),
                ("--agentic-training-flow", "agentic_training_flow"),
                ("--cloud-training-provider-registry", "cloud_training_provider_registry"),
                ("--cloud-training-preflight", "cloud_training_preflight"),
                ("--cloud-training-artifact-manifest", "cloud_training_artifact_manifest"),
                ("--cloud-training-launch-plan", "cloud_training_launch_plan"),
                ("--cloud-training-launch-receipt", "cloud_training_launch_receipt"),
                ("--cloud-training-status-receipt", "cloud_training_status_receipt"),
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

    def test_validate_rejects_symlink_loop_plan_source_ref(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agentic_plan = self.write_json(root / "agentic_training_plan.json", "hfr.agentic_training_plan.v1")
            loop_plan = root / "loop.json"
            plan = build_agentic_training_loop_plan(
                out_path=loop_plan,
                iteration_id="loop-symlink-source",
                artifact_paths={"agentic_training_plan": [agentic_plan]},
                created_at="2026-07-03T00:00:00+00:00",
            )
            link_path = root / "agentic_training_plan_link.json"
            try:
                link_path.symlink_to(agentic_plan)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")
            plan["source_artifacts"]["agentic_training_plan"][0]["path"] = "agentic_training_plan_link.json"
            loop_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)

            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn(
                "agentic_training_loop_plan.source_artifacts.agentic_training_plan[0].path must resolve to a regular non-symlink file.",
                errors,
            )

    def test_validate_rejects_symlink_parent_loop_plan_source_ref(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agentic_plan = self.write_json(root / "agentic_training_plan.json", "hfr.agentic_training_plan.v1")
            loop_plan = root / "loop.json"
            plan = build_agentic_training_loop_plan(
                out_path=loop_plan,
                iteration_id="loop-symlink-parent-source",
                artifact_paths={"agentic_training_plan": [agentic_plan]},
                created_at="2026-07-03T00:00:00+00:00",
            )
            linked_target = root / "linked_target"
            linked_target.mkdir()
            (linked_target / "agentic_training_plan.json").write_text(agentic_plan.read_text(encoding="utf-8"), encoding="utf-8")
            linked_parent = root / "linked_artifacts"
            try:
                linked_parent.symlink_to(linked_target, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")
            plan["source_artifacts"]["agentic_training_plan"][0]["path"] = "linked_artifacts/agentic_training_plan.json"
            loop_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)

            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn(
                "agentic_training_loop_plan.source_artifacts.agentic_training_plan[0].path must resolve to a regular non-symlink file.",
                errors,
            )

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

    def test_validate_rejects_tampered_cloud_training_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            loop_plan = root / "loop.json"
            plan = build_agentic_training_loop_plan(
                out_path=loop_plan,
                iteration_id="loop-cloud-tamper",
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )
            plan["cloud_training"]["artifact_count"] = 0
            plan["cloud_training"]["provider_api_calls_started"] = True
            plan["cloud_training_receipt_state"]["fail_closed"] = False
            plan["cloud_training_lineage"]["matched_link_count"] = 0
            loop_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)

            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("agentic_training_loop_plan.cloud_training.artifact_count must match cloud training source artifacts.", errors)
            self.assertIn("agentic_training_loop_plan.cloud_training.provider_api_calls_started must match cloud training source artifacts.", errors)
            self.assertIn("agentic_training_loop_plan.cloud_training_receipt_state.fail_closed must match cloud training receipt artifacts.", errors)
            self.assertIn("agentic_training_loop_plan.cloud_training_lineage.matched_link_count must match cloud training source lineage.", errors)

    def test_launch_receipt_side_effects_keep_loop_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            receipt = artifacts["cloud_training_launch_receipt"][0]
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            payload["launch"]["cloud_job_started"] = True
            payload["launch"]["provider_api_called"] = True
            payload["launch"]["cost_incurred_usd"] = 2
            receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            plan = build_agentic_training_loop_plan(
                out_path=root / "loop.json",
                iteration_id="loop-cloud-side-effects",
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )

            self.assertFalse(plan["passed"])
            self.assertFalse(plan["cloud_training_receipt_state"]["fail_closed"])
            self.assertTrue(plan["cloud_training_receipt_state"]["cloud_jobs_started"])
            self.assertTrue(plan["cloud_training_receipt_state"]["provider_api_calls_started"])
            self.assertEqual(plan["cloud_training_receipt_state"]["cost_incurred_usd"], 2)
            self.assertIn(
                "cloud_training_receipts_are_side_effect_free",
                {check["id"] for check in plan["checks"] if not check["passed"]},
            )
            schema = check_schema_contract(plan)
            self.assertTrue(schema["passed"], schema["errors"])

    def test_unlinked_cloud_training_receipts_keep_loop_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            receipt = artifacts["cloud_training_launch_receipt"][0]
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            payload["source_artifacts"]["launch_plan"]["sha256"] = "0" * 64
            receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            plan = build_agentic_training_loop_plan(
                out_path=root / "loop.json",
                iteration_id="loop-cloud-unlinked",
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )

            self.assertFalse(plan["passed"])
            self.assertEqual(plan["readiness"], "planned_fail_closed")
            self.assertFalse(plan["cloud_training_lineage"]["passed"])
            self.assertIn("launch_receipt_links_launch_plan", plan["cloud_training_lineage"]["mismatched_links"])
            self.assertIn(
                "cloud_training_lineage_bound_for_provider_handoff",
                {check["id"] for check in plan["checks"] if not check["passed"]},
            )
            schema = check_schema_contract(plan)
            self.assertTrue(schema["passed"], schema["errors"])

    def test_duplicate_cloud_training_lineage_roles_keep_loop_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            duplicate_receipt = root / "cloud_training_launch_receipt_duplicate.json"
            duplicate_receipt.write_text(
                artifacts["cloud_training_launch_receipt"][0].read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            artifacts["cloud_training_launch_receipt"].append(duplicate_receipt)

            loop_plan = root / "loop.json"
            plan = build_agentic_training_loop_plan(
                out_path=loop_plan,
                iteration_id="loop-cloud-duplicate",
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )
            loop_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            self.assertFalse(plan["passed"])
            self.assertEqual(plan["readiness"], "planned_fail_closed")
            self.assertEqual(plan["cloud_training_lineage"]["duplicate_roles"], ["cloud_training_launch_receipt"])
            self.assertIn("launch_receipt_links_launch_plan", plan["cloud_training_lineage"]["ambiguous_links"])
            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)
            self.assertTrue(validation["passed"], validation)

            plan["cloud_training_lineage"]["duplicate_role_count"] = 0
            plan["cloud_training_lineage"]["duplicate_roles"] = []
            plan["cloud_training_lineage"]["ambiguous_link_count"] = 0
            plan["cloud_training_lineage"]["ambiguous_links"] = []
            loop_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            tampered = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)
            self.assertFalse(tampered["passed"], tampered)
            errors = "\n".join(error for target in tampered["targets"] for error in target["errors"])
            self.assertIn("cloud_training_lineage.duplicate_role_count must match cloud training source lineage", errors)
            self.assertIn("cloud_training_lineage.ambiguous_link_count must match cloud training source lineage", errors)

    def test_duplicate_cloud_receipt_side_effects_are_not_hidden_by_first_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            duplicate_receipt = root / "cloud_training_launch_receipt_duplicate.json"
            duplicate_payload = json.loads(artifacts["cloud_training_launch_receipt"][0].read_text(encoding="utf-8"))
            duplicate_payload["execution_boundary"]["provider_api_called"] = True
            duplicate_payload["execution_boundary"]["cloud_job_started"] = True
            duplicate_payload["execution_boundary"]["live_requested"] = True
            duplicate_payload["execution_boundary"]["cloud_cost_incurred_usd"] = 3
            duplicate_receipt.write_text(json.dumps(duplicate_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            artifacts["cloud_training_launch_receipt"].append(duplicate_receipt)

            loop_plan = root / "loop.json"
            plan = build_agentic_training_loop_plan(
                out_path=loop_plan,
                iteration_id="loop-cloud-duplicate-side-effects",
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )

            self.assertFalse(plan["cloud_training_receipt_state"]["fail_closed"])
            self.assertTrue(plan["cloud_training_receipt_state"]["provider_api_calls_started"])
            self.assertTrue(plan["cloud_training_receipt_state"]["cloud_jobs_started"])
            self.assertEqual(plan["cloud_training_receipt_state"]["launch_receipt_count"], 2)
            self.assertEqual(plan["cloud_training_receipt_state"]["cost_incurred_usd"], 3)
            self.assertTrue(plan["cloud_training_receipt_state"]["live_launch_requested"])
            for check in plan["checks"]:
                if check["id"] == "cloud_training_receipts_are_side_effect_free":
                    check["passed"] = True
            plan["cloud_training"]["provider_api_calls_started"] = False
            plan["cloud_training"]["cloud_jobs_started"] = False
            plan["cloud_training_receipt_state"]["provider_api_calls_started"] = False
            plan["cloud_training_receipt_state"]["cloud_jobs_started"] = False
            plan["cloud_training_receipt_state"]["cost_incurred_usd"] = 0
            plan["cloud_training_receipt_state"]["live_launch_requested"] = False
            plan["cloud_training_receipt_state"]["fail_closed"] = True
            loop_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)

            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("agentic_training_loop_plan.cloud_training.provider_api_calls_started must match cloud training source artifacts.", errors)
            self.assertIn("agentic_training_loop_plan.cloud_training.cloud_jobs_started must match cloud training source artifacts.", errors)
            self.assertIn(
                "agentic_training_loop_plan.cloud_training_receipt_state.provider_api_calls_started must match cloud training receipt artifacts.",
                errors,
            )
            self.assertIn("agentic_training_loop_plan.cloud_training_receipt_state.cost_incurred_usd must match cloud training receipt artifacts.", errors)
            self.assertIn("agentic_training_loop_plan.cloud_training_receipt_state.fail_closed must match cloud training receipt artifacts.", errors)
            self.assertIn(
                "agentic_training_loop_plan.checks.cloud_training_receipts_are_side_effect_free.passed must match receipt state.",
                errors,
            )

    def test_live_launch_request_alone_keeps_loop_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            launch_receipt = artifacts["cloud_training_launch_receipt"][0]
            launch_payload = json.loads(launch_receipt.read_text(encoding="utf-8"))
            launch_payload["launch"]["mode"] = "live"
            launch_receipt.write_text(json.dumps(launch_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            loop_plan = root / "loop.json"
            plan = build_agentic_training_loop_plan(
                out_path=loop_plan,
                iteration_id="loop-cloud-live-only",
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )

            self.assertFalse(plan["passed"])
            self.assertTrue(plan["cloud_training_receipt_state"]["live_launch_requested"])
            self.assertFalse(plan["cloud_training_receipt_state"]["provider_api_calls_started"])
            self.assertFalse(plan["cloud_training_receipt_state"]["cloud_jobs_started"])
            self.assertEqual(plan["cloud_training_receipt_state"]["cost_incurred_usd"], 0)
            self.assertFalse(plan["cloud_training_receipt_state"]["fail_closed"])
            forged_readiness = json.loads(json.dumps(plan))
            forged_readiness["readiness"] = "ready_for_governance_review"
            forged_readiness["recommendation"] = "approve_iteration_execution"
            loop_plan.write_text(json.dumps(forged_readiness, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)

            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("agentic_training_loop_plan.readiness expected 'planned_fail_closed'", errors)
            self.assertIn(
                "agentic_training_loop_plan.recommendation expected 'collect_missing_receipts_before_live_execution'",
                errors,
            )
            for check in plan["checks"]:
                if check["id"] == "cloud_training_receipts_are_side_effect_free":
                    check["passed"] = True
            plan["cloud_training_receipt_state"]["live_launch_requested"] = False
            plan["cloud_training_receipt_state"]["fail_closed"] = True
            loop_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)

            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn(
                "agentic_training_loop_plan.cloud_training_receipt_state.live_launch_requested must match cloud training receipt artifacts.",
                errors,
            )
            self.assertIn(
                "agentic_training_loop_plan.cloud_training_receipt_state.fail_closed must match cloud training receipt artifacts.",
                errors,
            )
            self.assertIn(
                "agentic_training_loop_plan.checks.cloud_training_receipts_are_side_effect_free.passed must match receipt state.",
                errors,
            )

    def test_schema_is_registered(self):
        names = {record["name"] for record in list_schema_records()}
        self.assertIn("agentic_training_loop_plan", names)

    def write_loop_artifacts(self, root: Path) -> dict[str, list[Path]]:
        training_export = root / "training_export"
        training_export.mkdir()
        agentic_training_plan = self.write_json(root / "agentic_training_plan.json", "hfr.agentic_training_plan.v1")
        trainer_preflight = self.write_json(root / "trainer_preflight.json", "hfr.trainer_preflight.v1")
        trainer_launch_check = self.write_json(root / "trainer_launch_check.json", "hfr.trainer_launch_check.v1")
        cloud_artifact_manifest = self.write_provider_json(
            root / "cloud_training_artifact_manifest.json",
            "hfr.cloud_training_artifact_manifest.v1",
        )
        cloud_preflight = self.write_provider_json(
            root / "cloud_training_preflight.json",
            "hfr.cloud_training_preflight.v1",
            {
                "agentic_training_plan": self.artifact_ref(agentic_training_plan, "agentic_training_plan"),
                "trainer_preflight": self.artifact_ref(trainer_preflight, "trainer_preflight"),
                "trainer_launch_check": self.artifact_ref(trainer_launch_check, "trainer_launch_check"),
            },
        )
        cloud_launch_plan = self.write_provider_json(
            root / "cloud_training_launch_plan.json",
            "hfr.cloud_training_launch_plan.v1",
            {
                "preflight": self.artifact_ref(cloud_preflight, "cloud_training_preflight"),
                "artifact_manifest": self.artifact_ref(cloud_artifact_manifest, "cloud_training_artifact_manifest"),
            },
        )
        cloud_launch_receipt = self.write_json(
            root / "cloud_training_launch_receipt.json",
            "hfr.cloud_training_launch_receipt.v1",
            {"launch_plan": self.artifact_ref(cloud_launch_plan, "cloud_training_launch_plan")},
        )
        cloud_status_receipt = self.write_json(
            root / "cloud_training_status_receipt.json",
            "hfr.cloud_training_status_receipt.v1",
            {"launch_receipt": self.artifact_ref(cloud_launch_receipt, "cloud_training_launch_receipt")},
        )
        return {
            "agentic_rollout_plan": [self.write_json(root / "agentic_rollout_plan.json", "hfr.agentic_rollout_plan.v1")],
            "agentic_rollout_receipt": [self.write_json(root / "agentic_rollout_receipt.json", "hfr.agentic_rollout_receipt.v1")],
            "harness_result": [self.write_json(root / "harness_result.json", "hfr.harness_run_result.v1")],
            "evidence_bundle": [self.write_json(root / "evidence_bundle.json", "hfr.evidence_bundle.v1")],
            "rubric_spec": [self.write_json(root / "rubric_spec.json", "hfr.rubric_spec.v1")],
            "model_grader_dry_run": [self.write_json(root / "model_grader_dry_run.json", "hfr.model_grader_dry_run.v1")],
            "model_grader_override_receipt": [
                self.write_json(root / "model_grader_override_receipt.json", "hfr.model_grader_override_receipt.v1")
            ],
            "model_grader_gate": [self.write_json(root / "model_grader_gate.json", "hfr.model_grader_gate.v1")],
            "review_calibration": [self.write_json(root / "review_calibration.json", "hfr.review_calibration.v1")],
            "reviewed_gate": [self.write_json(root / "reviewed_gate.json", "hfr.reviewed_gate.v1")],
            "rejection_sampling_gate": [self.write_json(root / "rejection_sampling_gate.json", "hfr.rejection_sampling_gate.v1")],
            "dataset_curation_receipt": [self.write_json(root / "dataset_curation_receipt.json", "hfr.dataset_curation_receipt.v1")],
            "training_export": [training_export],
            "agentic_training_plan": [agentic_training_plan],
            "agentic_training_flow": [self.write_json(root / "agentic_training_flow.json", "hfr.agentic_training_flow.v1")],
            "cloud_training_provider_registry": [
                self.write_provider_registry(root / "cloud_training_provider_registry.json")
            ],
            "cloud_training_preflight": [cloud_preflight],
            "cloud_training_artifact_manifest": [cloud_artifact_manifest],
            "cloud_training_launch_plan": [cloud_launch_plan],
            "cloud_training_launch_receipt": [cloud_launch_receipt],
            "cloud_training_status_receipt": [cloud_status_receipt],
            "trainer_preflight": [trainer_preflight],
            "trainer_launch_check": [trainer_launch_check],
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

    def write_json(self, path: Path, schema_version: str, source_artifacts: dict[str, dict[str, object]] | None = None) -> Path:
        payload = {
            "schema_version": schema_version,
            "passed": True,
            "readiness": "ready",
            "source_artifacts": source_artifacts or {},
        }
        if schema_version == "hfr.cloud_training_launch_receipt.v1":
            payload.update(
                {
                    "readiness": "dry_run_recorded",
                    "recommendation": "safe_to_archive_dry_run_receipt",
                    "launch": {
                        "mode": "dry_run",
                        "cloud_job_started": False,
                        "provider_job_id": None,
                        "provider_api_called": False,
                        "cost_incurred_usd": 0,
                    },
                    "execution_boundary": {
                        "live_requested": False,
                        "allow_live": False,
                        "credential_values_recorded": False,
                    },
                }
            )
        if schema_version == "hfr.cloud_training_status_receipt.v1":
            payload.update(
                {
                    "readiness": "status_recorded",
                    "recommendation": "archive_status_receipt",
                    "status": {
                        "provider_status": "not_started",
                        "terminal": True,
                        "cancel_requested": False,
                        "provider_cancel_called": False,
                        "provider_api_called": False,
                        "cost_incurred_usd": 0,
                    },
                    "execution_boundary": {"credential_values_recorded": False},
                }
            )
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def write_provider_registry(self, path: Path) -> Path:
        path.write_text(
            json.dumps(
                {
                    "schema_version": "hfr.cloud_training_provider_registry.v1",
                    "passed": True,
                    "readiness": "ready",
                    "providers": [{"id": "modal"}],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def write_provider_json(
        self,
        path: Path,
        schema_version: str,
        source_artifacts: dict[str, dict[str, object]] | None = None,
    ) -> Path:
        path.write_text(
            json.dumps(
                {
                    "schema_version": schema_version,
                    "passed": True,
                    "readiness": "ready",
                    "provider": {"id": "modal"},
                    "source_artifacts": source_artifacts or {},
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def artifact_ref(self, path: Path, role: str) -> dict[str, object]:
        return {
            "role": role,
            "path": path.name,
            "exists": True,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
