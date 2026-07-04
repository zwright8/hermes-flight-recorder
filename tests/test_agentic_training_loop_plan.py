import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from flightrecorder.agentic_training_loop_plan import build_agentic_training_loop_plan
from flightrecorder.external_eval import (
    build_external_eval_plan,
    build_external_eval_receipt,
    write_external_eval_plan,
    write_external_eval_receipt,
)
from flightrecorder.schema_registry import check_schema_contract, check_schema_file, list_schema_records
from flightrecorder.validation import validate_artifacts
from tests.agentic_loop_fixtures import write_eval_summary, write_valid_promotion_decision, write_valid_promotion_ledger


ROOT = Path(__file__).resolve().parents[1]


class AgenticTrainingLoopPlanTests(unittest.TestCase):
    def test_committed_example_loop_plan_replays_fail_closed_sources(self):
        plan_path = ROOT / "examples" / "agentic_training" / "loop_plan.json"
        rollout_plan_path = ROOT / "examples" / "agentic_training" / "rollouts" / "rollout_plan.json"
        rollout_receipt_path = ROOT / "examples" / "agentic_training" / "rollouts" / "rollout_receipt.json"
        harness_result_path = ROOT / "examples" / "agentic_training" / "evidence_handoff" / "harness_handoff" / "harness_result.json"
        evidence_bundle_path = ROOT / "examples" / "agentic_training" / "evidence_handoff" / "evidence_bundle.json"
        reviewed_gate_path = ROOT / "examples" / "agentic_training" / "model_grader" / "reviewed_gate.json"
        rejection_sampling_gate_path = ROOT / "examples" / "agentic_training" / "rejection_sampling_gate.json"
        dataset_curation_receipt_path = ROOT / "examples" / "agentic_training" / "dataset_curation_receipt.json"
        trainer_preflight_path = ROOT / "examples" / "agentic_training" / "trainer_preflight.json"
        trainer_launch_check_path = ROOT / "examples" / "agentic_training" / "trainer_launch_check.json"
        provider_registry_path = ROOT / "examples" / "agentic_training" / "cloud_training" / "provider_registry.json"
        cloud_preflight_path = ROOT / "examples" / "agentic_training" / "cloud_training" / "preflight.json"
        cloud_launch_receipt_path = ROOT / "examples" / "agentic_training" / "cloud_training" / "launch_receipt.json"
        serving_lifecycle_path = ROOT / "examples" / "agentic_training" / "serving_lifecycle" / "managed_mock" / "serving_lifecycle.json"
        heldout_manifest_path = ROOT / "examples" / "agentic_training" / "heldout_eval" / "heldout_manifest.json"
        external_eval_plan_path = ROOT / "examples" / "agentic_training" / "heldout_eval" / "external_eval_plan.json"
        external_eval_receipt_path = ROOT / "examples" / "agentic_training" / "heldout_eval" / "external_eval_receipt.json"
        eval_summary_path = ROOT / "examples" / "agentic_training" / "heldout_eval" / "eval_summary.json"
        model_grader_gate_path = ROOT / "examples" / "agentic_training" / "model_grader" / "passing_gate.json"
        action_ledger_path = ROOT / "examples" / "agentic_training" / "iteration_ledgers" / "action_ledger.json"
        improvement_ledger_path = ROOT / "examples" / "agentic_training" / "iteration_ledgers" / "improvement_ledger.json"
        promotion_decision_path = ROOT / "examples" / "agentic_training" / "promotion_governance" / "promotion_decision.json"
        promotion_ledger_path = ROOT / "examples" / "agentic_training" / "promotion_governance" / "promotion_ledger.json"
        promotion_cards_path = ROOT / "examples" / "agentic_training" / "promotion_governance" / "promotion_cards"
        promotion_alias_apply_path = ROOT / "examples" / "agentic_training" / "promotion_governance" / "promotion_alias_apply.json"
        promotion_rollback_receipt_path = (
            ROOT / "examples" / "agentic_training" / "promotion_governance" / "promotion_rollback_receipt.json"
        )
        promotion_release_record_path = (
            ROOT / "examples" / "agentic_training" / "promotion_governance" / "promotion_release_record.json"
        )
        promotion_archive_path = ROOT / "examples" / "agentic_training" / "promotion_governance" / "promotion_archive"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))

        expected_refs = {
            "agentic_rollout_plan": ("rollouts/rollout_plan.json", rollout_plan_path),
            "agentic_rollout_receipt": ("rollouts/rollout_receipt.json", rollout_receipt_path),
            "harness_result": ("evidence_handoff/harness_handoff/harness_result.json", harness_result_path),
            "evidence_bundle": ("evidence_handoff/evidence_bundle.json", evidence_bundle_path),
            "reviewed_gate": ("model_grader/reviewed_gate.json", reviewed_gate_path),
            "rejection_sampling_gate": ("rejection_sampling_gate.json", rejection_sampling_gate_path),
            "dataset_curation_receipt": ("dataset_curation_receipt.json", dataset_curation_receipt_path),
            "trainer_preflight": ("trainer_preflight.json", trainer_preflight_path),
            "trainer_launch_check": ("trainer_launch_check.json", trainer_launch_check_path),
            "cloud_training_provider_registry": ("cloud_training/provider_registry.json", provider_registry_path),
            "cloud_training_preflight": ("cloud_training/preflight.json", cloud_preflight_path),
            "cloud_training_launch_receipt": ("cloud_training/launch_receipt.json", cloud_launch_receipt_path),
            "serving_lifecycle": ("serving_lifecycle/managed_mock/serving_lifecycle.json", serving_lifecycle_path),
            "heldout_manifest": ("heldout_eval/heldout_manifest.json", heldout_manifest_path),
            "external_eval_plan": ("heldout_eval/external_eval_plan.json", external_eval_plan_path),
            "external_eval_receipt": ("heldout_eval/external_eval_receipt.json", external_eval_receipt_path),
            "eval_summary": ("heldout_eval/eval_summary.json", eval_summary_path),
            "model_grader_gate": ("model_grader/passing_gate.json", model_grader_gate_path),
            "action_ledger": ("iteration_ledgers/action_ledger.json", action_ledger_path),
            "improvement_ledger": ("iteration_ledgers/improvement_ledger.json", improvement_ledger_path),
            "promotion_decision": ("promotion_governance/promotion_decision.json", promotion_decision_path),
            "promotion_ledger": ("promotion_governance/promotion_ledger.json", promotion_ledger_path),
            "promotion_cards": ("promotion_governance/promotion_cards", promotion_cards_path),
            "promotion_alias_apply": ("promotion_governance/promotion_alias_apply.json", promotion_alias_apply_path),
            "promotion_rollback_receipt": (
                "promotion_governance/promotion_rollback_receipt.json",
                promotion_rollback_receipt_path,
            ),
            "promotion_release_record": (
                "promotion_governance/promotion_release_record.json",
                promotion_release_record_path,
            ),
            "promotion_archive": ("promotion_governance/promotion_archive", promotion_archive_path),
        }
        for role, (expected_path, source_path) in expected_refs.items():
            ref = plan["source_artifacts"][role][0]
            self.assertEqual(ref["path"], expected_path)
            if source_path.is_dir():
                self.assertEqual(ref["kind"], "directory")
                self.assertTrue(ref["exists"])
                self.assertIsNone(ref["sha256"])
            else:
                self.assertEqual(ref["size_bytes"], source_path.stat().st_size)
                self.assertEqual(ref["sha256"], hashlib.sha256(source_path.read_bytes()).hexdigest())
        self.assertTrue(plan["passed"])
        self.assertEqual(plan["readiness"], "ready_for_governance_review")
        self.assertEqual(plan["artifact_count"], 40)
        self.assertEqual(plan["missing_phase_inputs"], [])
        self.assertNotIn("agentic_rollout_plan", plan["missing_phase_inputs"])
        self.assertNotIn("agentic_rollout_receipt", plan["missing_phase_inputs"])
        self.assertNotIn("harness_result", plan["missing_phase_inputs"])
        self.assertNotIn("evidence_bundle", plan["missing_phase_inputs"])
        self.assertNotIn("reviewed_gate", plan["missing_phase_inputs"])
        self.assertNotIn("rejection_sampling_gate", plan["missing_phase_inputs"])
        self.assertNotIn("dataset_curation_receipt", plan["missing_phase_inputs"])
        self.assertNotIn("training_export", plan["missing_phase_inputs"])
        self.assertNotIn("trainer_preflight", plan["missing_phase_inputs"])
        self.assertNotIn("trainer_launch_check", plan["missing_phase_inputs"])
        self.assertNotIn("heldout_manifest", plan["missing_phase_inputs"])
        self.assertNotIn("external_eval_plan", plan["missing_phase_inputs"])
        self.assertNotIn("external_eval_receipt", plan["missing_phase_inputs"])
        self.assertNotIn("eval_summary", plan["missing_phase_inputs"])
        self.assertNotIn("serving_lifecycle", plan["missing_phase_inputs"])
        self.assertNotIn("promotion_decision", plan["missing_phase_inputs"])
        self.assertNotIn("promotion_ledger", plan["missing_phase_inputs"])
        self.assertEqual(
            {check["id"] for check in plan["checks"] if not check["passed"]},
            set(),
        )
        self.assertNotIn("rollout_receipt_required_before_review", {check["id"] for check in plan["checks"] if not check["passed"]})
        self.assertNotIn("uncalibrated_labels_block_training_data", {check["id"] for check in plan["checks"] if not check["passed"]})
        self.assertNotIn("dataset_curation_receipt_required_for_trainer_handoff", {check["id"] for check in plan["checks"] if not check["passed"]})
        self.assertNotIn("cloud_training_lineage_bound_for_provider_handoff", {check["id"] for check in plan["checks"] if not check["passed"]})
        training_export_ref = plan["source_artifacts"]["training_export"][0]
        self.assertEqual(training_export_ref["path"], "training_export")
        self.assertEqual(training_export_ref["kind"], "directory")
        self.assertTrue(training_export_ref["exists"])
        phases = {phase["id"]: phase for phase in plan["phases"]}
        self.assertEqual(
            set(phases["rollout_collection"]["present_required_artifacts"]),
            {"agentic_rollout_plan", "agentic_rollout_receipt", "harness_result"},
        )
        self.assertEqual(phases["rollout_collection"]["missing_required_artifacts"], [])
        self.assertEqual(phases["evidence_scoring"]["status"], "ready")
        self.assertEqual(phases["evidence_scoring"]["missing_required_artifacts"], [])
        self.assertEqual(phases["rubric_model_grader_review"]["status"], "ready")
        self.assertEqual(phases["rejection_sampling"]["status"], "ready")
        self.assertEqual(phases["dataset_curation"]["status"], "ready")
        self.assertEqual(
            set(phases["dataset_curation"]["present_required_artifacts"]),
            {"rejection_sampling_gate", "dataset_curation_receipt", "training_export"},
        )
        self.assertEqual(phases["dataset_curation"]["missing_required_artifacts"], [])
        self.assertEqual(phases["external_trainer_execution"]["status"], "ready")
        self.assertIn("trainer_preflight", phases["external_trainer_execution"]["present_required_artifacts"])
        self.assertIn("trainer_launch_check", phases["external_trainer_execution"]["present_required_artifacts"])
        self.assertEqual(phases["serving_checks"]["status"], "ready")
        self.assertEqual(phases["serving_checks"]["present_required_artifacts"], ["serving_lifecycle"])
        self.assertEqual(phases["serving_checks"]["missing_required_artifacts"], [])
        self.assertEqual(phases["heldout_eval"]["status"], "ready")
        self.assertEqual(phases["heldout_eval"]["missing_required_artifacts"], [])
        self.assertEqual(phases["improvement_planning"]["status"], "ready")
        self.assertEqual(phases["governance_decision"]["status"], "ready")
        self.assertEqual(phases["governance_decision"]["present_required_artifacts"], ["promotion_decision"])
        self.assertEqual(phases["promotion_or_rollback"]["status"], "ready")
        self.assertEqual(phases["promotion_or_rollback"]["present_required_artifacts"], ["promotion_ledger"])
        self.assertEqual(phases["next_iteration"]["status"], "ready")
        self.assertFalse(plan["cloud_training"]["cloud_jobs_started"])
        self.assertFalse(plan["cloud_training"]["provider_api_calls_started"])
        self.assertTrue(plan["cloud_training_lineage"]["passed"])
        self.assertEqual(plan["cloud_training_lineage"]["missing_link_count"], 0)
        self.assertTrue(plan["cloud_training_receipt_state"]["fail_closed"])
        self.assertEqual(plan["cloud_training_receipt_state"]["cost_incurred_usd"], 0)
        self.assertEqual(plan["external_eval_receipt_state"]["adapter_count"], 1)
        self.assertEqual(plan["external_eval_receipt_state"]["ready_adapter_count"], 1)
        self.assertTrue(plan["external_eval_receipt_state"]["receipts_passed"])
        self.assertTrue(plan["external_eval_receipt_state"]["fail_closed"])
        self.assertFalse(plan["external_eval_receipt_state"]["live_benchmarks_started"])
        self.assertFalse(plan["external_eval_receipt_state"]["provider_api_calls_started"])
        heldout_check = next(check for check in plan["checks"] if check["id"] == "heldout_eval_is_fail_closed")
        self.assertTrue(heldout_check["passed"])
        self.assertTrue(heldout_check["actual"]["eval_summary_valid"])
        self.assertTrue(heldout_check["actual"]["eval_summary_passed"])
        self.assertFalse(plan["execution_boundary"]["cloud_jobs_started"])
        self.assertFalse(plan["execution_boundary"]["paid_model_grader_calls_started"])
        self.assertFalse(plan["execution_boundary"]["weights_updated_by_flight_recorder"])
        schema = check_schema_file(plan_path)
        self.assertTrue(schema["passed"], schema["errors"])
        validation = validate_artifacts(agentic_training_loop_plan_paths=[plan_path], strict=True)
        self.assertTrue(validation["passed"], validation)

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
            self.assertTrue(plan["external_eval_receipt_state"]["receipts_passed"])
            self.assertTrue(plan["external_eval_receipt_state"]["fail_closed"])
            self.assertEqual(plan["external_eval_receipt_state"]["launch_mode"], "dry_run")
            self.assertEqual(plan["external_eval_receipt_state"]["adapter_count"], 1)
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

    def test_eval_summary_is_required_by_heldout_eval_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            artifacts.pop("eval_summary")

            plan = build_agentic_training_loop_plan(
                out_path=root / "loop.json",
                iteration_id="loop-missing-eval-summary",
                objective="Require eval summary before governance review.",
                artifact_paths=artifacts,
                budget={"max_cloud_cost_usd": 0, "max_gpu_hours": 0},
                provider_constraints={"providers": ["mock"], "regions": ["local"], "gpu_classes": ["none"]},
                created_at="2026-07-03T00:00:00+00:00",
            )

            self.assertFalse(plan["passed"])
            self.assertEqual(plan["readiness"], "planned_fail_closed")
            self.assertIn("eval_summary", plan["missing_phase_inputs"])
            heldout_check = next(check for check in plan["checks"] if check["id"] == "heldout_eval_is_fail_closed")
            self.assertFalse(heldout_check["passed"])
            self.assertFalse(heldout_check["actual"]["eval_summary_present"])
            self.assertTrue(heldout_check["expected"]["eval_summary_present"])
            schema = check_schema_contract(plan)
            self.assertTrue(schema["passed"], schema["errors"])
            loop_plan = root / "loop.json"
            loop_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)
            self.assertTrue(validation["passed"], validation)

    def test_blocked_external_eval_receipt_keeps_loop_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            receipt = artifacts["external_eval_receipt"][0]
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            payload["passed"] = False
            payload["readiness"] = "blocked"
            payload["recommendation"] = "keep_external_eval_claims_disabled"
            payload["ready_adapter_count"] = 0
            receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            plan = build_agentic_training_loop_plan(
                out_path=root / "loop.json",
                iteration_id="loop-blocked-external-eval",
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )

            self.assertFalse(plan["passed"])
            self.assertEqual(plan["readiness"], "planned_fail_closed")
            self.assertFalse(plan["external_eval_receipt_state"]["receipts_passed"])
            self.assertTrue(plan["external_eval_receipt_state"]["fail_closed"])
            heldout_check = next(check for check in plan["checks"] if check["id"] == "heldout_eval_is_fail_closed")
            self.assertFalse(heldout_check["passed"])
            self.assertFalse(heldout_check["actual"]["external_eval_receipts_passed"])
            schema = check_schema_contract(plan)
            self.assertTrue(schema["passed"], schema["errors"])

    def test_external_eval_receipt_side_effects_keep_loop_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            receipt = artifacts["external_eval_receipt"][0]
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            payload["launch"]["live_benchmarks_started"] = True
            payload["launch"]["provider_api_called"] = True
            payload["launch"]["model_downloads_started"] = True
            payload["launch"]["cost_incurred_usd"] = 2
            payload["execution_boundary"]["credential_values_recorded"] = True
            receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            loop_plan = root / "loop.json"
            plan = build_agentic_training_loop_plan(
                out_path=loop_plan,
                iteration_id="loop-external-eval-side-effects",
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )

            self.assertFalse(plan["passed"])
            self.assertFalse(plan["external_eval_receipt_state"]["fail_closed"])
            self.assertTrue(plan["external_eval_receipt_state"]["live_benchmarks_started"])
            self.assertTrue(plan["external_eval_receipt_state"]["provider_api_calls_started"])
            self.assertTrue(plan["external_eval_receipt_state"]["model_downloads_started"])
            self.assertTrue(plan["external_eval_receipt_state"]["credential_values_recorded"])
            self.assertEqual(plan["external_eval_receipt_state"]["cost_incurred_usd"], 2)
            self.assertIn(
                "heldout_eval_is_fail_closed",
                {check["id"] for check in plan["checks"] if not check["passed"]},
            )

            forged = json.loads(json.dumps(plan))
            forged["external_eval_receipt_state"]["fail_closed"] = True
            forged["external_eval_receipt_state"]["live_benchmarks_started"] = False
            forged["external_eval_receipt_state"]["provider_api_calls_started"] = False
            forged["external_eval_receipt_state"]["model_downloads_started"] = False
            forged["external_eval_receipt_state"]["credential_values_recorded"] = False
            forged["external_eval_receipt_state"]["cost_incurred_usd"] = 0
            for check in forged["checks"]:
                if check["id"] == "heldout_eval_is_fail_closed":
                    check["passed"] = True
            loop_plan.write_text(json.dumps(forged, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)

            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn(
                "agentic_training_loop_plan.external_eval_receipt_state.live_benchmarks_started must match external eval receipt artifacts.",
                errors,
            )
            self.assertIn(
                "agentic_training_loop_plan.external_eval_receipt_state.provider_api_calls_started must match external eval receipt artifacts.",
                errors,
            )
            self.assertIn(
                "agentic_training_loop_plan.external_eval_receipt_state.cost_incurred_usd must match external eval receipt artifacts.",
                errors,
            )
            self.assertIn(
                "agentic_training_loop_plan.checks.heldout_eval_is_fail_closed.passed must match external eval receipt and eval summary state.",
                errors,
            )

    def test_invalid_eval_summary_cannot_unlock_loop_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            eval_summary_path = artifacts["eval_summary"][0]
            eval_summary = json.loads(eval_summary_path.read_text(encoding="utf-8"))
            del eval_summary["arms"]
            eval_summary_path.write_text(json.dumps(eval_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            loop_plan = root / "loop.json"

            plan = build_agentic_training_loop_plan(
                out_path=loop_plan,
                iteration_id="loop-invalid-eval-summary",
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )
            self.assertFalse(plan["passed"])
            loop_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)

            self.assertTrue(validation["passed"], validation)
            heldout_check = next(check for check in plan["checks"] if check["id"] == "heldout_eval_is_fail_closed")
            self.assertFalse(heldout_check["passed"])
            self.assertFalse(heldout_check["actual"]["eval_summary_valid"])

    def test_failed_eval_summary_cannot_unlock_loop_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            eval_summary_path = artifacts["eval_summary"][0]
            eval_summary = json.loads(eval_summary_path.read_text(encoding="utf-8"))
            eval_summary["passed"] = False
            eval_summary["governance_ready"] = False
            eval_summary_path.write_text(json.dumps(eval_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            plan = build_agentic_training_loop_plan(
                out_path=root / "loop.json",
                iteration_id="loop-failed-eval-summary",
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )

            self.assertFalse(plan["passed"])
            heldout_check = next(check for check in plan["checks"] if check["id"] == "heldout_eval_is_fail_closed")
            self.assertFalse(heldout_check["passed"])
            self.assertTrue(heldout_check["actual"]["eval_summary_valid"])
            self.assertFalse(heldout_check["actual"]["eval_summary_passed"])

    def test_invalid_promotion_ledger_cannot_unlock_loop_governance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            ledger_path = artifacts["promotion_ledger"][0]
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            del ledger["records"]
            ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            loop_plan = root / "loop.json"

            plan = build_agentic_training_loop_plan(
                out_path=loop_plan,
                iteration_id="loop-invalid-promotion-ledger",
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )
            self.assertTrue(plan["passed"])
            loop_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)

            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn(
                "agentic_training_loop_plan.checks.governance_required_for_promotion.passed must match promotion decision and ledger validation state.",
                errors,
            )

    def test_invalid_promotion_decision_cannot_unlock_loop_governance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            decision_path = artifacts["promotion_decision"][0]
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            del decision["checks"]
            decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            loop_plan = root / "loop.json"

            plan = build_agentic_training_loop_plan(
                out_path=loop_plan,
                iteration_id="loop-invalid-promotion-decision",
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )
            self.assertTrue(plan["passed"])
            loop_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)

            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn(
                "agentic_training_loop_plan.checks.governance_required_for_promotion.passed must match promotion decision and ledger validation state.",
                errors,
            )

    def test_public_unsafe_promotion_decision_path_cannot_unlock_loop_governance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            decision_path = artifacts["promotion_decision"][0]
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            source_path = root / "promotion_decision_sources" / "evidence_bundle.json"
            decision["artifacts"]["evidence_bundle"]["path"] = str(source_path.resolve())
            decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            loop_plan = root / "loop.json"

            plan = build_agentic_training_loop_plan(
                out_path=loop_plan,
                iteration_id="loop-public-unsafe-promotion-decision",
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )
            self.assertTrue(plan["passed"])
            loop_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)

            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn(
                "agentic_training_loop_plan.checks.governance_required_for_promotion.passed must match promotion decision and ledger validation state.",
                errors,
            )

    def test_public_path_like_promotion_decision_prose_does_not_block_loop_governance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            decision_path = artifacts["promotion_decision"][0]
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["notes"].append("Operator note: expected drift is ~5%; /tmp is mentioned as prose, not an artifact path.")
            decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            loop_plan = root / "loop.json"

            plan = build_agentic_training_loop_plan(
                out_path=loop_plan,
                iteration_id="loop-path-like-prose",
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )
            self.assertTrue(plan["passed"], plan["blocked_reasons"])
            loop_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)

            self.assertTrue(validation["passed"], validation)

    def test_forged_external_eval_receipt_keeps_loop_state_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            receipt_path = artifacts["external_eval_receipt"][0]
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["source_plan"]["sha256"] = "0" * 64
            receipt["passed"] = True
            receipt["readiness"] = "dry_run_recorded"
            receipt["recommendation"] = "archive_external_eval_dry_run"
            receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            loop_plan = root / "loop.json"

            plan = build_agentic_training_loop_plan(
                out_path=loop_plan,
                iteration_id="loop-forged-external-eval",
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )
            self.assertFalse(plan["passed"])
            self.assertFalse(plan["external_eval_receipt_state"]["receipts_passed"])
            self.assertEqual(plan["external_eval_receipt_state"]["receipt_passed_count"], 0)
            loop_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)
            receipt_validation = validate_artifacts(external_eval_receipt_paths=[receipt_path], strict=True)

            self.assertTrue(validation["passed"], validation)
            self.assertFalse(receipt_validation["passed"], receipt_validation)

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
                ("--model-grader-disagreement-queue", "model_grader_disagreement_queue"),
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
                ("--promotion-cards", "promotion_cards"),
                ("--promotion-alias-apply", "promotion_alias_apply"),
                ("--promotion-rollback-receipt", "promotion_rollback_receipt"),
                ("--promotion-release-record", "promotion_release_record"),
                ("--promotion-archive", "promotion_archive"),
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

    def test_validate_rejects_forged_loop_plan_side_effect_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            loop_plan = root / "loop.json"
            plan = build_agentic_training_loop_plan(
                out_path=loop_plan,
                iteration_id="loop-forged-side-effects",
                artifact_paths=artifacts,
                budget={"max_cloud_cost_usd": 0, "max_gpu_hours": 0},
                provider_constraints={"providers": ["mock"], "regions": ["local"], "gpu_classes": ["none"]},
                created_at="2026-07-03T00:00:00+00:00",
            )
            schema = check_schema_contract(plan, name_or_id="agentic_training_loop_plan")
            self.assertTrue(schema["passed"], schema["errors"])

            forged = json.loads(json.dumps(plan))
            forged["cloud_job_url"] = "redacted-cloud-job-url"
            forged["budget"]["provider_billing_account"] = "redacted-account"
            forged["participants"]["api_key_env"] = "RED_ACTED"
            forged["provider_constraints"]["credential_value"] = "redacted-secret"
            forged["artifact_role_counts"][0]["download_url"] = "redacted-download-url"
            forged["checks"][0]["provider_call"] = "forged"
            forged["source_artifacts"]["agentic_training_plan"][0]["signed_url"] = "redacted-signed-url"
            forged["phases"][0]["live_execution_started"] = True
            forged["cloud_training"]["provider_call_receipt"] = "forged"
            forged["cloud_training_receipt_state"]["provider_console_url"] = "redacted-provider-console"
            forged["cloud_training_lineage"]["provider"]["credential_value"] = "redacted-secret"
            forged["cloud_training_lineage"]["role_counts"][0]["provider_call"] = "forged"
            forged["cloud_training_lineage"]["links"][0]["provider_trace_url"] = "redacted-trace-url"
            forged["external_eval_receipt_state"]["benchmark_job_id"] = "bench-live"
            forged["execution_boundary"]["provider_console_url"] = "redacted-provider-console"
            forged["handoff_contract"]["credential_hint"] = "redacted-secret"
            forged["next_iteration"]["auto_schedule_started"] = True

            forged_schema = check_schema_contract(forged, name_or_id="agentic_training_loop_plan")
            self.assertFalse(forged_schema["passed"])
            schema_errors = "\n".join(forged_schema["errors"])
            for field_name in (
                "cloud_job_url",
                "provider_billing_account",
                "api_key_env",
                "credential_value",
                "download_url",
                "provider_call",
                "signed_url",
                "live_execution_started",
                "provider_call_receipt",
                "provider_console_url",
                "provider_trace_url",
                "benchmark_job_id",
                "credential_hint",
                "auto_schedule_started",
            ):
                self.assertIn(field_name, schema_errors)

            loop_plan.write_text(json.dumps(forged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)

            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("agentic_training_loop_plan contains unknown field(s): ['cloud_job_url'].", errors)
            self.assertIn("agentic_training_loop_plan.budget contains unknown field(s): ['provider_billing_account'].", errors)
            self.assertIn("agentic_training_loop_plan.participants contains unknown field(s): ['api_key_env'].", errors)
            self.assertIn("agentic_training_loop_plan.provider_constraints contains unknown field(s): ['credential_value'].", errors)
            self.assertIn("agentic_training_loop_plan.artifact_role_counts[0] contains unknown field(s): ['download_url'].", errors)
            self.assertIn("agentic_training_loop_plan.checks[0] contains unknown field(s): ['provider_call'].", errors)
            self.assertIn(
                "agentic_training_loop_plan.source_artifacts.agentic_training_plan[0] contains unknown field(s): ['signed_url'].",
                errors,
            )
            self.assertIn("agentic_training_loop_plan.phases[0] contains unknown field(s): ['live_execution_started'].", errors)
            self.assertIn("agentic_training_loop_plan.cloud_training contains unknown field(s): ['provider_call_receipt'].", errors)
            self.assertIn(
                "agentic_training_loop_plan.cloud_training_receipt_state contains unknown field(s): ['provider_console_url'].",
                errors,
            )
            self.assertIn(
                "agentic_training_loop_plan.cloud_training_lineage.provider contains unknown field(s): ['credential_value'].",
                errors,
            )
            self.assertIn(
                "agentic_training_loop_plan.cloud_training_lineage.role_counts[0] contains unknown field(s): ['provider_call'].",
                errors,
            )
            self.assertIn(
                "agentic_training_loop_plan.cloud_training_lineage.links[0] contains unknown field(s): ['provider_trace_url'].",
                errors,
            )
            self.assertIn(
                "agentic_training_loop_plan.external_eval_receipt_state contains unknown field(s): ['benchmark_job_id'].",
                errors,
            )
            self.assertIn(
                "agentic_training_loop_plan.execution_boundary contains unknown field(s): ['provider_console_url'].",
                errors,
            )
            self.assertIn("agentic_training_loop_plan.handoff_contract contains unknown field(s): ['credential_hint'].", errors)
            self.assertIn("agentic_training_loop_plan.next_iteration contains unknown field(s): ['auto_schedule_started'].", errors)

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

    def test_loop_plan_source_payload_readers_skip_symlinked_parent_refs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            loop_plan = root / "loop.json"
            plan = build_agentic_training_loop_plan(
                out_path=loop_plan,
                iteration_id="loop-symlink-payload-reader",
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )
            launch_receipt = artifacts["cloud_training_launch_receipt"][0]
            linked_parent = root / "linked_artifacts"
            try:
                linked_parent.symlink_to(launch_receipt.parent, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")
            plan["source_artifacts"]["cloud_training_launch_receipt"][0]["path"] = str(
                Path("linked_artifacts") / launch_receipt.name
            )
            loop_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)

            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn(
                "agentic_training_loop_plan.source_artifacts.cloud_training_launch_receipt[0].path must resolve to a regular non-symlink file.",
                errors,
            )
            self.assertIn(
                "agentic_training_loop_plan.cloud_training_receipt_state.launch_receipt_count must match cloud training receipt artifacts.",
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
            self.assertFalse(plan["cloud_training_receipt_state"]["launch_receipt_passed"])
            self.assertFalse(plan["cloud_training_receipt_state"]["receipts_passed"])
            self.assertFalse(plan["cloud_training_lineage"]["passed"])
            self.assertIn("launch_receipt_links_launch_plan", plan["cloud_training_lineage"]["mismatched_links"])
            self.assertIn(
                "cloud_training_lineage_bound_for_provider_handoff",
                {check["id"] for check in plan["checks"] if not check["passed"]},
            )
            schema = check_schema_contract(plan)
            self.assertTrue(schema["passed"], schema["errors"])
            loop_plan = root / "loop.json"
            loop_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)
            receipt_validation = validate_artifacts(cloud_training_launch_receipt_paths=[receipt], strict=True)
            self.assertTrue(validation["passed"], validation)
            self.assertFalse(receipt_validation["passed"], receipt_validation)

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
        heldout_manifest, external_eval_plan, external_eval_receipt = self.write_external_eval_artifacts(root)
        return {
            "agentic_rollout_plan": [self.write_json(root / "agentic_rollout_plan.json", "hfr.agentic_rollout_plan.v1")],
            "agentic_rollout_receipt": [self.write_json(root / "agentic_rollout_receipt.json", "hfr.agentic_rollout_receipt.v1")],
            "harness_result": [self.write_json(root / "harness_result.json", "hfr.harness_run_result.v1")],
            "evidence_bundle": [self.write_json(root / "evidence_bundle.json", "hfr.evidence_bundle.v1")],
            "rubric_spec": [self.write_json(root / "rubric_spec.json", "hfr.rubric_spec.v1")],
            "model_grader_dry_run": [self.write_json(root / "model_grader_dry_run.json", "hfr.model_grader_dry_run.v1")],
            "model_grader_disagreement_queue": [
                self.write_json(root / "model_grader_disagreement_queue.json", "hfr.model_grader_disagreement_queue.v1")
            ],
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
            "heldout_manifest": [heldout_manifest],
            "external_eval_plan": [external_eval_plan],
            "external_eval_receipt": [external_eval_receipt],
            "eval_summary": [write_eval_summary(root)],
            "improvement_plan": [self.write_json(root / "improvement_plan.json", "hfr.improvement_plan.v1")],
            "promotion_decision": [write_valid_promotion_decision(root)],
            "promotion_ledger": [write_valid_promotion_ledger(root)],
            "promotion_cards": [self.write_json(root / "promotion_cards.json", "hfr.promotion_cards.v1")],
            "promotion_alias_apply": [
                self.write_json(root / "promotion_alias_apply.json", "hfr.promotion_alias_apply.v1")
            ],
            "promotion_rollback_receipt": [
                self.write_json(root / "promotion_rollback_receipt.json", "hfr.promotion_rollback_receipt.v1")
            ],
            "promotion_release_record": [
                self.write_json(root / "promotion_release_record.json", "hfr.promotion_release_record.v1")
            ],
            "promotion_archive": [self.write_json(root / "promotion_archive.json", "hfr.promotion_archive.v1")],
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

    def write_external_eval_artifacts(self, root: Path) -> tuple[Path, Path, Path]:
        heldout_manifest = self.write_heldout_manifest(root / "heldout_manifest.json")
        external_eval_plan = root / "external_eval_plan.json"
        external_eval_receipt = root / "external_eval_receipt.json"
        with patch(
            "flightrecorder.external_eval._dependency_status",
            return_value={"available": True, "imports": {"inspect_ai": True}, "commands": {"inspect": True}},
        ):
            plan = build_external_eval_plan(
                adapters=["inspect_ai"],
                scenario_manifest=heldout_manifest,
                model_endpoint="http://127.0.0.1:8000/v1",
                inspect_task_set="heldout-inspect",
                sandbox_policy="locked-network",
                allow_installed=True,
            )
        write_external_eval_plan(plan, external_eval_plan)
        receipt = build_external_eval_receipt(
            plan_path=external_eval_plan,
            adapters=["inspect_ai"],
            created_at="2026-07-03T00:00:00+00:00",
            output_base_dir=external_eval_receipt.parent,
        )
        write_external_eval_receipt(receipt, external_eval_receipt)
        return heldout_manifest, external_eval_plan, external_eval_receipt

    def write_heldout_manifest(self, path: Path) -> Path:
        payload = {
            "schema_version": "hfr.heldout_scenario_manifest.v1",
            "ready": True,
            "scenario_count": 1,
            "scenario_ids": ["email_reply_completion"],
        }
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
