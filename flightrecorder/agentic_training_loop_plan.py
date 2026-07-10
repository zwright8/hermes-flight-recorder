"""Closed-loop agentic training iteration contracts."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .cloud_training import build_cloud_training_launch_receipt, build_cloud_training_status_receipt
from .external_eval import ExternalEvalPlanError, build_external_eval_receipt
from .source_contract import inspect_artifact_source

AGENTIC_TRAINING_LOOP_PLAN_SCHEMA_VERSION = "hfr.agentic_training_loop_plan.v1"

PLAN_READINESS_CHECK_IDS = frozenset(
    {
        "phase_contracts_present",
        "artifact_references_are_public_safe",
        "flight_recorder_did_not_launch_external_work",
        "live_launches_require_explicit_opt_in",
        "rollout_receipt_required_before_review",
        "uncalibrated_labels_block_training_data",
        "dataset_curation_receipt_required_for_trainer_handoff",
        "external_trainer_handoff_is_preflighted",
        "cloud_training_receipts_bound_for_provider_handoff",
        "cloud_training_receipts_are_side_effect_free",
        "cloud_training_lineage_bound_for_provider_handoff",
        "external_eval_handoff_is_preflighted",
    }
)

PLAN_REQUIRED_ARTIFACT_ROLES = frozenset(
    {
        "agentic_rollout_plan",
        "agentic_rollout_receipt",
        "harness_result",
        "evidence_bundle",
        "rubric_spec",
        "model_grader_gate",
        "review_calibration",
        "reviewed_gate",
        "rejection_sampling_gate",
        "dataset_curation_receipt",
        "training_export",
        "agentic_training_plan",
        "trainer_preflight",
        "trainer_launch_check",
        "cloud_training_provider_registry",
        "cloud_training_preflight",
        "cloud_training_artifact_manifest",
        "cloud_training_launch_plan",
        "cloud_training_launch_receipt",
        "cloud_training_status_receipt",
        "heldout_manifest",
        "external_eval_plan",
        "external_eval_receipt",
    }
)

EXTERNAL_EVAL_HANDOFF_ROLES = (
    "heldout_manifest",
    "external_eval_plan",
    "external_eval_receipt",
)

CLOUD_TRAINING_LINEAGE_LINKS: tuple[dict[str, str], ...] = (
    {
        "id": "preflight_links_agentic_training_plan",
        "source_role": "cloud_training_preflight",
        "source_ref": "agentic_training_plan",
        "target_role": "agentic_training_plan",
    },
    {
        "id": "preflight_links_trainer_preflight",
        "source_role": "cloud_training_preflight",
        "source_ref": "trainer_preflight",
        "target_role": "trainer_preflight",
    },
    {
        "id": "preflight_links_trainer_launch_check",
        "source_role": "cloud_training_preflight",
        "source_ref": "trainer_launch_check",
        "target_role": "trainer_launch_check",
    },
    {
        "id": "launch_plan_links_preflight",
        "source_role": "cloud_training_launch_plan",
        "source_ref": "preflight",
        "target_role": "cloud_training_preflight",
    },
    {
        "id": "launch_plan_links_artifact_manifest",
        "source_role": "cloud_training_launch_plan",
        "source_ref": "artifact_manifest",
        "target_role": "cloud_training_artifact_manifest",
    },
    {
        "id": "launch_receipt_links_launch_plan",
        "source_role": "cloud_training_launch_receipt",
        "source_ref": "launch_plan",
        "target_role": "cloud_training_launch_plan",
    },
    {
        "id": "status_receipt_links_launch_receipt",
        "source_role": "cloud_training_status_receipt",
        "source_ref": "launch_receipt",
        "target_role": "cloud_training_launch_receipt",
    },
)
CLOUD_TRAINING_LINEAGE_ARTIFACT_ROLES: tuple[str, ...] = tuple(
    sorted(
        {"cloud_training_provider_registry"}
        | {link["source_role"] for link in CLOUD_TRAINING_LINEAGE_LINKS}
        | {link["target_role"] for link in CLOUD_TRAINING_LINEAGE_LINKS}
    )
)

PHASES: tuple[dict[str, Any], ...] = (
    {
        "id": "scenario_task_generation",
        "name": "Scenario and task generation",
        "required": (),
        "produces": ("scenario_quality", "harness_manifest"),
        "gate": "generated tasks must be versioned, replayable, and privacy reviewed before rollout.",
    },
    {
        "id": "rollout_collection",
        "name": "Rollout collection",
        "required": ("agentic_rollout_plan", "agentic_rollout_receipt", "harness_result"),
        "produces": ("trace", "scorecard", "run_digest"),
        "gate": "rollouts must stay inside declared budget and environment descriptors.",
    },
    {
        "id": "evidence_scoring",
        "name": "Evidence scoring",
        "required": ("evidence_bundle",),
        "produces": ("evidence_coverage", "trace_observability", "repair_queue"),
        "gate": "evidence must be schema-checkable and traceable before training data selection.",
    },
    {
        "id": "rubric_model_grader_review",
        "name": "Rubric and model-grader review",
        "required": ("rubric_spec", "model_grader_gate", "review_calibration"),
        "produces": (
            "review_manifest",
            "model_grader_dry_run",
            "model_grader_disagreement_queue",
            "model_grader_override_receipt",
            "reviewed_manifest",
        ),
        "gate": "model-grader labels are blocked until calibration and human override paths exist.",
    },
    {
        "id": "rejection_sampling",
        "name": "Rejection sampling",
        "required": ("reviewed_gate", "rejection_sampling_gate"),
        "produces": ("dataset_registry", "dataset_splits"),
        "gate": "uncalibrated or low-confidence labels must not enter trainer-ready datasets.",
    },
    {
        "id": "dataset_curation",
        "name": "Dataset curation",
        "required": ("rejection_sampling_gate", "dataset_curation_receipt", "training_export"),
        "produces": ("training_manifest", "dataset_registry"),
        "gate": "datasets must be redacted, licensed, hashed, and split before trainer handoff.",
    },
    {
        "id": "external_trainer_execution",
        "name": "External trainer execution",
        "required": (
            "agentic_training_plan",
            "trainer_preflight",
            "trainer_launch_check",
            "cloud_training_provider_registry",
            "cloud_training_preflight",
            "cloud_training_artifact_manifest",
            "cloud_training_launch_plan",
            "cloud_training_launch_receipt",
            "cloud_training_status_receipt",
            "agentic_training_result",
        ),
        "produces": ("agentic_training_runtime_preflight", "agentic_training_flow", "agentic_training_result"),
        "gate": "live trainer launch requires explicit opt-in, credentials, cloud-provider receipts, and a passing preflight.",
    },
    {
        "id": "serving_checks",
        "name": "Serving checks",
        "required": ("serving_lifecycle",),
        "produces": ("serving_endpoint_check", "model_serving_probe_receipt"),
        "gate": "trained outputs must pass serving compatibility before held-out evals or promotion.",
    },
    {
        "id": "heldout_eval",
        "name": "Held-out eval and external benchmarks",
        "required": (
            "heldout_manifest",
            "external_eval_plan",
            "external_eval_receipt",
            "external_eval_result",
            "eval_summary",
        ),
        "produces": ("eval_summary", "compare_gate", "decision_gate"),
        "gate": "live external benchmarks remain disabled until dependencies and held-out scenario parity pass.",
    },
    {
        "id": "improvement_planning",
        "name": "Improvement planning",
        "required": ("improvement_plan",),
        "produces": ("improvement_ledger", "action_ledger"),
        "gate": "next work must be evidence-backed, deduplicated, and linked to repair/curriculum signals.",
    },
    {
        "id": "governance_decision",
        "name": "Governance decision",
        "required": ("promotion_decision",),
        "produces": ("agentic_loop_governance_receipt", "promotion_cards", "promotion_release_record"),
        "gate": "governance must approve, reject, rollback, or request another iteration before aliases move.",
    },
    {
        "id": "promotion_or_rollback",
        "name": "Promotion or rollback",
        "required": ("promotion_ledger",),
        "produces": ("promotion_alias_apply", "promotion_rollback_receipt"),
        "gate": "registry writes are guarded by promotion decisions and rollback receipts.",
    },
    {
        "id": "next_iteration",
        "name": "Next-iteration scheduling",
        "required": ("action_ledger",),
        "produces": ("agentic_training_loop_plan", "next_iteration_schedule"),
        "gate": "new iterations are scheduled only from ledgered decisions and open repair actions.",
    },
)

ARTIFACT_ROLES: dict[str, str] = {
    "action_ledger": "action_ledger",
    "agentic_rollout_plan": "agentic_rollout_plan",
    "agentic_rollout_receipt": "agentic_rollout_receipt",
    "agentic_training_plan": "agentic_training_plan",
    "agentic_training_flow": "agentic_training_flow",
    "agentic_training_result": "agentic_training_result",
    "agentic_training_runtime_preflight": "agentic_training_runtime_preflight",
    "cloud_training_artifact_manifest": "cloud_training_artifact_manifest",
    "cloud_training_launch_plan": "cloud_training_launch_plan",
    "cloud_training_launch_receipt": "cloud_training_launch_receipt",
    "cloud_training_preflight": "cloud_training_preflight",
    "cloud_training_provider_registry": "cloud_training_provider_registry",
    "cloud_training_status_receipt": "cloud_training_status_receipt",
    "compare_gate": "compare_gate",
    "dataset_registry": "dataset_registry",
    "dataset_splits": "dataset_splits",
    "dataset_curation_receipt": "dataset_curation_receipt",
    "decision_gate": "decision_gate",
    "evidence_bundle": "evidence_bundle",
    "evidence_coverage": "evidence_coverage",
    "eval_summary": "eval_summary",
    "external_eval_plan": "external_eval_plan",
    "external_eval_receipt": "external_eval_receipt",
    "external_eval_result": "external_eval_result",
    "harness_manifest": "harness_manifest",
    "harness_result": "harness_result",
    "heldout_manifest": "heldout_manifest",
    "improvement_ledger": "improvement_ledger",
    "improvement_plan": "improvement_plan",
    "agentic_loop_governance_receipt": "agentic_loop_governance_receipt",
    "promotion_alias_apply": "promotion_alias_apply",
    "promotion_archive": "promotion_archive",
    "promotion_cards": "promotion_cards",
    "promotion_decision": "promotion_decision",
    "promotion_ledger": "promotion_ledger",
    "promotion_release_record": "promotion_release_record",
    "promotion_rollback_receipt": "promotion_rollback_receipt",
    "model_grader_dry_run": "model_grader_dry_run",
    "model_grader_disagreement_queue": "model_grader_disagreement_queue",
    "model_grader_override_receipt": "model_grader_override_receipt",
    "model_grader_gate": "model_grader_gate",
    "next_iteration_schedule": "next_iteration_schedule",
    "review_calibration": "review_calibration",
    "reviewed_gate": "reviewed_gate",
    "rejection_sampling_gate": "rejection_sampling_gate",
    "rubric_spec": "rubric_spec",
    "serving_lifecycle": "serving_lifecycle",
    "trace_observability": "trace_observability",
    "trainer_launch_check": "trainer_launch_check",
    "trainer_preflight": "trainer_preflight",
    "training_export": "training_export",
}


class AgenticTrainingLoopPlanError(ValueError):
    """Raised when a closed-loop iteration plan cannot be built."""


def build_agentic_training_loop_plan(
    *,
    out_path: str | Path,
    iteration_id: str,
    artifact_paths: dict[str, list[str | Path]] | None = None,
    objective: str | None = None,
    candidate: str | None = None,
    baseline: str | None = None,
    teacher: str | None = None,
    budget: dict[str, Any] | None = None,
    provider_constraints: dict[str, Any] | None = None,
    schedule: dict[str, Any] | None = None,
    preserve_paths: bool = False,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a side-effect-free contract for one closed-loop training iteration."""
    if not iteration_id:
        raise AgenticTrainingLoopPlanError("iteration_id is required")
    output_path = Path(out_path)
    normalized_artifact_paths = _normalized_artifact_paths(artifact_paths or {})
    refs = _artifact_refs(normalized_artifact_paths, preserve_paths, output_path)
    cloud_training_receipt_state = _cloud_training_receipt_state(normalized_artifact_paths)
    external_eval_receipt_state = _external_eval_receipt_state(normalized_artifact_paths)
    eval_summary_state = _eval_summary_state(normalized_artifact_paths)
    external_eval_handoff_binding = _external_eval_handoff_binding(normalized_artifact_paths, refs)
    external_eval_result_binding = _external_eval_result_binding(normalized_artifact_paths, refs)
    eval_summary_result_binding = _eval_summary_result_binding(normalized_artifact_paths, refs)
    promotion_governance_state = _promotion_governance_state(normalized_artifact_paths)
    cloud_training = _cloud_training_summary(refs, cloud_training_receipt_state)
    cloud_training_lineage = _cloud_training_lineage(refs, normalized_artifact_paths)
    training_result_status = _execution_result_status(
        normalized_artifact_paths,
        "agentic_training_result",
        refs,
    )
    external_eval_result_status = _execution_result_status(
        normalized_artifact_paths,
        "external_eval_result",
        refs,
    )
    execution_result_statuses = {
        "agentic_training_result": training_result_status,
        "external_eval_result": external_eval_result_status,
    }
    execution_completion = _execution_completion(execution_result_statuses)
    training_result_plan_bound = _agentic_training_result_plan_bound(normalized_artifact_paths, refs)
    phases = [_phase_row(spec, refs, execution_result_statuses) for spec in PHASES]
    checks: list[dict[str, Any]] = []
    _add_check(checks, "phase_contracts_present", len(phases) == len(PHASES), {"phase_count": len(phases)}, {"phase_count": len(PHASES)})
    _add_check(checks, "artifact_references_are_public_safe", _refs_are_safe(refs), {"unsafe_refs": _unsafe_refs(refs)}, {"unsafe_refs": []})
    _add_check(
        checks,
        "flight_recorder_did_not_launch_external_work",
        True,
        {
            "cloud_jobs_started": False,
            "paid_model_grader_calls_started": False,
            "live_benchmarks_started": False,
            "model_downloads_started": False,
            "weights_updated": False,
        },
        {
            "cloud_jobs_started": False,
            "paid_model_grader_calls_started": False,
            "live_benchmarks_started": False,
            "model_downloads_started": False,
            "weights_updated": False,
        },
    )
    _add_check(
        checks,
        "live_launches_require_explicit_opt_in",
        True,
        {"live_launch_opt_in": False, "environment_credentials_checked": False},
        {"live_launch_opt_in": True, "environment_credentials_checked": True},
    )
    _add_check(
        checks,
        "rollout_receipt_required_before_review",
        _role_ready(refs, "agentic_rollout_plan") and _role_ready(refs, "agentic_rollout_receipt"),
        {
            "agentic_rollout_plan_present": _role_ready(refs, "agentic_rollout_plan"),
            "agentic_rollout_receipt_present": _role_ready(refs, "agentic_rollout_receipt"),
        },
        {"agentic_rollout_plan_present": True, "agentic_rollout_receipt_present": True},
    )
    _add_check(
        checks,
        "uncalibrated_labels_block_training_data",
        all(
            _role_ready(refs, role)
            for role in ("rubric_spec", "model_grader_gate", "review_calibration", "reviewed_gate", "rejection_sampling_gate")
        ),
        {
            "rubric_spec_present": _role_ready(refs, "rubric_spec"),
            "model_grader_gate_present": _role_ready(refs, "model_grader_gate"),
            "review_calibration_present": _role_ready(refs, "review_calibration"),
            "reviewed_gate_present": _role_ready(refs, "reviewed_gate"),
            "rejection_sampling_gate_present": _role_ready(refs, "rejection_sampling_gate"),
        },
        {
            "rubric_spec_present": True,
            "model_grader_gate_present": True,
            "review_calibration_present": True,
            "reviewed_gate_present": True,
            "rejection_sampling_gate_present": True,
        },
    )
    _add_check(
        checks,
        "dataset_curation_receipt_required_for_trainer_handoff",
        all(_role_ready(refs, role) for role in ("rejection_sampling_gate", "dataset_curation_receipt", "training_export")),
        {
            "rejection_sampling_gate_present": _role_ready(refs, "rejection_sampling_gate"),
            "dataset_curation_receipt_present": _role_ready(refs, "dataset_curation_receipt"),
            "training_export_present": _role_ready(refs, "training_export"),
        },
        {"rejection_sampling_gate_present": True, "dataset_curation_receipt_present": True, "training_export_present": True},
    )
    _add_check(
        checks,
        "external_trainer_handoff_is_preflighted",
        all(_role_ready(refs, role) for role in ("agentic_training_plan", "trainer_preflight", "trainer_launch_check")),
        {
            "agentic_training_plan_present": _role_ready(refs, "agentic_training_plan"),
            "trainer_preflight_present": _role_ready(refs, "trainer_preflight"),
            "trainer_launch_check_present": _role_ready(refs, "trainer_launch_check"),
        },
        {
            "agentic_training_plan_present": True,
            "trainer_preflight_present": True,
            "trainer_launch_check_present": True,
        },
    )
    _add_check(
        checks,
        "external_trainer_execution_completed",
        training_result_status == "completed" and training_result_plan_bound,
        {
            "agentic_training_result_present": _role_ready(refs, "agentic_training_result"),
            "agentic_training_result_count": len(normalized_artifact_paths.get("agentic_training_result", [])),
            "training_result_status": training_result_status,
            "training_result_plan_bound": training_result_plan_bound,
        },
        {
            "agentic_training_result_present": True,
            "agentic_training_result_count": 1,
            "training_result_status": "completed",
            "training_result_plan_bound": True,
        },
    )
    _add_check(
        checks,
        "cloud_training_receipts_bound_for_provider_handoff",
        not cloud_training["missing_artifacts"],
        {"cloud_training": cloud_training},
        {"missing_artifacts": []},
    )
    _add_check(
        checks,
        "cloud_training_receipts_are_side_effect_free",
        cloud_training_receipt_state["fail_closed"],
        {"cloud_training_receipt_state": cloud_training_receipt_state},
        {
            "provider_api_calls_started": False,
            "cloud_jobs_started": False,
            "provider_cancel_called": False,
            "credential_values_recorded": False,
            "live_launch_requested": False,
            "cost_incurred_usd": 0,
        },
    )
    _add_check(
        checks,
        "cloud_training_lineage_bound_for_provider_handoff",
        cloud_training_lineage["passed"],
        {"cloud_training_lineage": cloud_training_lineage},
        {
            "passed": True,
            "missing_link_count": 0,
            "mismatched_link_count": 0,
            "ambiguous_link_count": 0,
            "duplicate_role_count": 0,
            "provider_consistent": True,
        },
    )
    _add_check(
        checks,
        "external_eval_handoff_is_preflighted",
        all(_role_ready_once(refs, role) for role in EXTERNAL_EVAL_HANDOFF_ROLES)
        and external_eval_handoff_binding["passed"]
        and external_eval_receipt_state["receipts_passed"]
        and external_eval_receipt_state["fail_closed"],
        {
            "heldout_manifest_present": _role_ready(refs, "heldout_manifest"),
            "heldout_manifest_count": len(refs.get("heldout_manifest", [])),
            "external_eval_plan_present": _role_ready(refs, "external_eval_plan"),
            "external_eval_plan_count": len(refs.get("external_eval_plan", [])),
            "external_eval_receipt_present": _role_ready(refs, "external_eval_receipt"),
            "external_eval_receipt_count": len(refs.get("external_eval_receipt", [])),
            "external_eval_receipts_passed": external_eval_receipt_state["receipts_passed"],
            "external_eval_receipts_fail_closed": external_eval_receipt_state["fail_closed"],
            "plan_heldout_manifest_bound": external_eval_handoff_binding["plan_heldout_manifest_bound"],
            "receipt_plan_bound": external_eval_handoff_binding["receipt_plan_bound"],
            "external_eval_handoff_status": external_eval_handoff_binding["status"],
        },
        {
            "heldout_manifest_present": True,
            "heldout_manifest_count": 1,
            "external_eval_plan_present": True,
            "external_eval_plan_count": 1,
            "external_eval_receipt_present": True,
            "external_eval_receipt_count": 1,
            "external_eval_receipts_passed": True,
            "external_eval_receipts_fail_closed": True,
            "plan_heldout_manifest_bound": True,
            "receipt_plan_bound": True,
            "external_eval_handoff_status": "matched",
        },
    )
    _add_check(
        checks,
        "heldout_eval_is_fail_closed",
        all(_role_ready_once(refs, role) for role in EXTERNAL_EVAL_HANDOFF_ROLES)
        and external_eval_result_status == "completed"
        and external_eval_result_binding["passed"]
        and _role_ready_once(refs, "eval_summary")
        and eval_summary_state["valid"]
        and eval_summary_state["passed"]
        and eval_summary_result_binding["passed"]
        and external_eval_receipt_state["receipts_passed"]
        and external_eval_receipt_state["fail_closed"],
        {
            "heldout_manifest_present": _role_ready(refs, "heldout_manifest"),
            "heldout_manifest_count": len(refs.get("heldout_manifest", [])),
            "external_eval_plan_present": _role_ready(refs, "external_eval_plan"),
            "external_eval_plan_count": len(refs.get("external_eval_plan", [])),
            "external_eval_receipt_present": _role_ready(refs, "external_eval_receipt"),
            "external_eval_receipt_count": len(refs.get("external_eval_receipt", [])),
            "external_eval_result_present": _role_ready(refs, "external_eval_result"),
            "external_eval_result_count": len(normalized_artifact_paths.get("external_eval_result", [])),
            "external_eval_result_expected_count": external_eval_result_binding["expected_count"],
            "external_eval_result_adapter_coverage_complete": external_eval_result_binding[
                "adapter_coverage_complete"
            ],
            "external_eval_result_completed": external_eval_result_status == "completed",
            "external_eval_result_status": external_eval_result_status,
            "external_eval_result_plan_bound": external_eval_result_binding["result_plan_bound"],
            "external_eval_result_heldout_manifest_bound": external_eval_result_binding["result_heldout_manifest_bound"],
            "external_eval_result_binding_status": external_eval_result_binding["status"],
            "eval_summary_present": _role_ready(refs, "eval_summary"),
            "eval_summary_count": len(refs.get("eval_summary", [])),
            "eval_summary_valid": eval_summary_state["valid"],
            "eval_summary_passed": eval_summary_state["passed"],
            "eval_summary_result_bound": eval_summary_result_binding["passed"],
            "eval_summary_result_evidence_bound": eval_summary_result_binding[
                "evidence_set_bound"
            ],
            "eval_summary_result_governance_ready": eval_summary_result_binding[
                "governance_ready"
            ],
            "eval_summary_result_expected_count": eval_summary_result_binding["expected_count"],
            "eval_summary_result_match_count": eval_summary_result_binding["match_count"],
            "eval_summary_result_status": eval_summary_result_binding["status"],
            "external_eval_receipts_passed": external_eval_receipt_state["receipts_passed"],
            "external_eval_receipts_fail_closed": external_eval_receipt_state["fail_closed"],
            "live_benchmark_requested": external_eval_receipt_state["live_benchmark_requested"],
            "live_benchmarks_started": external_eval_receipt_state["live_benchmarks_started"],
            "provider_api_calls_started": external_eval_receipt_state["provider_api_calls_started"],
            "model_downloads_started": external_eval_receipt_state["model_downloads_started"],
            "credential_values_recorded": external_eval_receipt_state["credential_values_recorded"],
            "cost_incurred_usd": external_eval_receipt_state["cost_incurred_usd"],
        },
        {
            "heldout_manifest_present": True,
            "heldout_manifest_count": 1,
            "external_eval_plan_present": True,
            "external_eval_plan_count": 1,
            "external_eval_receipt_present": True,
            "external_eval_receipt_count": 1,
            "external_eval_result_present": True,
            "external_eval_result_count": external_eval_result_binding["expected_count"],
            "external_eval_result_expected_count": external_eval_result_binding["expected_count"],
            "external_eval_result_adapter_coverage_complete": True,
            "external_eval_result_completed": True,
            "external_eval_result_status": "completed",
            "external_eval_result_plan_bound": True,
            "external_eval_result_heldout_manifest_bound": True,
            "external_eval_result_binding_status": "matched",
            "eval_summary_present": True,
            "eval_summary_count": 1,
            "eval_summary_valid": True,
            "eval_summary_passed": True,
            "eval_summary_result_bound": True,
            "eval_summary_result_evidence_bound": True,
            "eval_summary_result_governance_ready": True,
            "eval_summary_result_expected_count": eval_summary_result_binding["expected_count"],
            "eval_summary_result_match_count": eval_summary_result_binding["expected_count"],
            "eval_summary_result_status": "matched",
            "external_eval_receipts_passed": True,
            "external_eval_receipts_fail_closed": True,
            "live_benchmark_requested": False,
            "live_benchmarks_started": False,
            "provider_api_calls_started": False,
            "model_downloads_started": False,
            "credential_values_recorded": False,
            "cost_incurred_usd": 0,
        },
    )
    _add_check(
        checks,
        "governance_required_for_promotion",
        promotion_governance_state["passed"],
        promotion_governance_state,
        {
            "promotion_decision_present": True,
            "promotion_decision_schema_valid": True,
            "promotion_decision_passed": True,
            "promotion_ledger_present": True,
            "promotion_ledger_schema_valid": True,
        },
    )
    missing_phase_inputs = sorted({missing for phase in phases for missing in phase["missing_required_artifacts"]})
    _add_check(
        checks,
        "required_phase_inputs_present",
        not missing_phase_inputs,
        {"missing_phase_inputs": missing_phase_inputs},
        {"missing_phase_inputs": []},
    )
    failed_checks = [check for check in checks if not check["passed"]]
    plan_failed_checks = [check for check in failed_checks if check["id"] in PLAN_READINESS_CHECK_IDS]
    missing_plan_inputs = sorted(role for role in PLAN_REQUIRED_ARTIFACT_ROLES if not _role_ready(refs, role))
    plan_readiness = "ready_to_execute" if not plan_failed_checks and not missing_plan_inputs else "blocked"
    governance_readiness = (
        "ready_for_review"
        if plan_readiness == "ready_to_execute"
        and execution_completion == "completed"
        and not failed_checks
        and not missing_phase_inputs
        else "blocked"
    )
    readiness = "ready_for_governance_review" if governance_readiness == "ready_for_review" else "planned_fail_closed"
    if execution_completion == "failed":
        recommendation = "investigate_failed_execution"
    elif plan_readiness == "blocked":
        recommendation = "collect_missing_plan_evidence"
    elif execution_completion == "incomplete":
        recommendation = "execute_ready_plan"
    elif governance_readiness == "blocked":
        recommendation = "resolve_governance_blockers"
    else:
        recommendation = "submit_for_governance_review"

    return {
        "schema_version": AGENTIC_TRAINING_LOOP_PLAN_SCHEMA_VERSION,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "iteration_id": iteration_id,
        "plan_path": _display_path(output_path, preserve_paths),
        "objective": objective or "",
        "participants": {
            "baseline_policy": baseline or "",
            "candidate_policy": candidate or "",
            "teacher_policy": teacher or "",
        },
        "passed": governance_readiness == "ready_for_review",
        "plan_readiness": plan_readiness,
        "execution_completion": execution_completion,
        "governance_readiness": governance_readiness,
        "readiness": readiness,
        "recommendation": recommendation,
        "check_count": len(checks),
        "failed_check_count": len(failed_checks),
        "checks": checks,
        "blocked_reasons": [check["summary"] for check in failed_checks],
        "missing_phase_inputs": missing_phase_inputs,
        "artifact_count": sum(len(rows) for rows in refs.values()),
        "artifact_role_counts": _role_counts(refs),
        "source_artifacts": refs,
        "phases": phases,
        "budget": _budget(budget or {}),
        "provider_constraints": _provider_constraints(provider_constraints or {}),
        "cloud_training": cloud_training,
        "cloud_training_receipt_state": cloud_training_receipt_state,
        "cloud_training_lineage": cloud_training_lineage,
        "external_eval_receipt_state": external_eval_receipt_state,
        "execution_boundary": {
            "dry_run_plan_only": True,
            "cloud_jobs_started": False,
            "paid_model_grader_calls_started": False,
            "live_benchmarks_started": False,
            "model_downloads_started": False,
            "weights_updated_by_flight_recorder": False,
            "credential_values_recorded": False,
            "public_artifact_paths_redacted": not preserve_paths,
        },
        "handoff_contract": {
            "flight_recorder_controls_preflight_and_receipts": True,
            "external_trainers_own_weight_updates": True,
            "live_launch_requires_explicit_opt_in": True,
            "requires_environment_credentials_for_live": True,
            "requires_trainer_preflight": True,
            "requires_trainer_launch_check": True,
            "requires_calibrated_review_before_training_data": True,
            "requires_heldout_eval_before_promotion": True,
            "requires_governance_decision_before_alias_update": True,
            "default_live_execution_allowed": False,
        },
        "next_iteration": {
            "scheduled": False,
            "requires_governance_decision": True,
            "schedule": schedule or {},
            "recommendation": (
                "Use promotion/action/improvement ledgers to schedule the next iteration after governance decides."
            ),
        },
        "notes": [
            "This artifact is a closed-loop iteration contract; it does not call graders, trainers, cloud APIs, or benchmarks.",
            "Missing phase inputs keep the loop fail-closed while preserving a schema-checkable plan for orchestration.",
            "Dry-run trainer and external-eval receipts establish plan readiness only; they never count as completed execution.",
            "Governance review requires completed external training and external-eval result artifacts in addition to passing gates.",
            "Use dry-run/mock provider receipts first; live provider launches must be explicit, credentialed, and separately archived.",
        ],
    }


def write_agentic_training_loop_plan(path: str | Path, plan: dict[str, Any]) -> None:
    """Write a deterministic closed-loop plan artifact."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _normalized_artifact_paths(artifact_paths: dict[str, list[str | Path]]) -> dict[str, list[Path]]:
    normalized: dict[str, list[Path]] = {}
    for role in sorted(artifact_paths):
        normalized_role = ARTIFACT_ROLES.get(role, role)
        rows = [Path(path) for path in artifact_paths[role] if str(path)]
        if rows:
            normalized.setdefault(normalized_role, []).extend(rows)
    return normalized


def _artifact_refs(
    artifact_paths: dict[str, list[Path]],
    preserve_paths: bool,
    output_path: Path,
) -> dict[str, list[dict[str, Any]]]:
    refs: dict[str, list[dict[str, Any]]] = {}
    for role in sorted(artifact_paths):
        rows = [_artifact_ref(role, path, preserve_paths, output_path) for path in artifact_paths[role]]
        if rows:
            refs[role] = rows
    return refs


def _artifact_ref(role: str, path: Path, preserve_paths: bool, output_path: Path) -> dict[str, Any]:
    source = inspect_artifact_source(path, role)
    exists = source.get("ready") is True
    is_file = exists and source.get("regular_file") is True
    is_dir = source.get("regular_directory") is True
    directory_fingerprint = _directory_tree_fingerprint(path) if exists and is_dir else {}
    directory_contains_symlinks = _directory_contains_symlink(path) if exists and is_dir else None
    payload = source["payload"] if isinstance(source.get("payload"), dict) else {}
    return {
        "role": role,
        "path": _display_source_path(path, output_path, preserve_paths),
        "kind": "directory" if is_dir else "file",
        "exists": exists,
        "sha256": _sha256(path) if is_file else directory_fingerprint.get("sha256"),
        "size_bytes": path.stat().st_size if is_file else directory_fingerprint.get("size_bytes"),
        "file_count": directory_fingerprint.get("file_count") if is_dir else None,
        "contains_symlinks": directory_contains_symlinks,
        "schema_version": str(payload.get("schema_version") or "") if payload else "",
        "passed": payload.get("passed") if isinstance(payload.get("passed"), bool) else None,
        "readiness": str(payload.get("readiness") or payload.get("status") or ""),
    }


def _phase_row(
    spec: dict[str, Any],
    refs: dict[str, list[dict[str, Any]]],
    execution_result_statuses: dict[str, str],
) -> dict[str, Any]:
    required = list(spec["required"])
    present = [role for role in required if _role_ready(refs, role)]
    missing = [role for role in required if role not in present]
    non_completed = [
        role
        for role in present
        if role in execution_result_statuses and execution_result_statuses[role] != "completed"
    ]
    status = "ready" if required and not missing and not non_completed else "planned"
    if required and (missing or non_completed):
        status = "blocked"
    return {
        "id": spec["id"],
        "name": spec["name"],
        "status": status,
        "required_artifacts": required,
        "present_required_artifacts": present,
        "missing_required_artifacts": missing,
        "non_completed_required_artifacts": non_completed,
        "produces": list(spec["produces"]),
        "gate": spec["gate"],
    }


def _budget(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "max_rollouts": _non_negative_or_none(value.get("max_rollouts")),
        "max_training_examples": _non_negative_or_none(value.get("max_training_examples")),
        "max_cloud_cost_usd": _non_negative_number_or_none(value.get("max_cloud_cost_usd")),
        "max_gpu_hours": _non_negative_number_or_none(value.get("max_gpu_hours")),
        "live_spend_allowed": False,
    }


def _provider_constraints(value: dict[str, Any]) -> dict[str, Any]:
    regions = value.get("regions") if isinstance(value.get("regions"), list) else []
    gpus = value.get("gpu_classes") if isinstance(value.get("gpu_classes"), list) else []
    providers = value.get("providers") if isinstance(value.get("providers"), list) else []
    return {
        "providers": [str(item) for item in providers if str(item)],
        "regions": [str(item) for item in regions if str(item)],
        "gpu_classes": [str(item) for item in gpus if str(item)],
        "requires_cost_estimate": True,
        "requires_region_allowlist": bool(regions),
        "requires_secret_redaction": True,
    }


def _role_counts(refs: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    counter = Counter({role: len(rows) for role, rows in refs.items()})
    return [{"role": role, "count": counter[role]} for role in sorted(counter)]


def _role_ready(refs: dict[str, list[dict[str, Any]]], role: str) -> bool:
    rows = refs.get(role)
    return bool(rows) and all(isinstance(row, dict) and row.get("exists") is True for row in rows)


def _role_ready_once(refs: dict[str, list[dict[str, Any]]], role: str) -> bool:
    rows = refs.get(role)
    return isinstance(rows, list) and len(rows) == 1 and _role_ready(refs, role)


def _execution_completion(result_statuses: dict[str, str]) -> str:
    statuses = set(result_statuses.values())
    if "failed" in statuses:
        return "failed"
    if statuses == {"completed"}:
        return "completed"
    return "incomplete"


def _execution_result_status(
    artifact_paths: dict[str, list[Path]],
    role: str,
    refs: dict[str, list[dict[str, Any]]] | None = None,
) -> str:
    if role == "external_eval_result":
        return _external_eval_results_status(artifact_paths, refs)
    paths = artifact_paths.get(role, [])
    if not paths:
        return "missing"
    if len(paths) != 1:
        return "incomplete"
    statuses: list[str] = []
    for path in paths:
        source = inspect_artifact_source(path, role)
        if source.get("physical_exists") is not True:
            statuses.append("incomplete")
            continue
        if (
            source.get("parse_valid") is not True
            or source.get("schema_valid") is not True
            or source.get("semantic_valid") is not True
        ):
            statuses.append("incomplete")
            continue
        payload = source["payload"] if isinstance(source.get("payload"), dict) else {}
        if role == "agentic_training_result" and refs is not None:
            lineage = payload.get("lineage") if isinstance(payload.get("lineage"), dict) else {}
            source_plan = lineage.get("plan") if isinstance(lineage.get("plan"), dict) else {}
            if source_plan.get("sha256") != _single_ref_sha256(refs, "agentic_training_plan"):
                statuses.append("incomplete")
                continue
        statuses.append(_execution_result_payload_status(role, payload))
    if "failed" in statuses:
        return "failed"
    if statuses and all(status == "completed" for status in statuses):
        return "completed"
    return "incomplete"


def _external_eval_results_status(
    artifact_paths: dict[str, list[Path]],
    refs: dict[str, list[dict[str, Any]]] | None = None,
) -> str:
    expected_adapter_ids = _external_eval_expected_adapter_ids(artifact_paths)
    paths = artifact_paths.get("external_eval_result", [])
    if not expected_adapter_ids or not paths:
        return "missing"
    statuses_by_adapter: dict[str, str] = {}
    invalid_or_ambiguous = False
    for path in paths:
        source = inspect_artifact_source(path, "external_eval_result")
        if (
            source.get("physical_exists") is not True
            or source.get("parse_valid") is not True
            or source.get("schema_valid") is not True
            or source.get("semantic_valid") is not True
        ):
            invalid_or_ambiguous = True
            continue
        payload = source["payload"] if isinstance(source.get("payload"), dict) else {}
        identity = payload.get("identity") if isinstance(payload.get("identity"), dict) else {}
        adapter_id = identity.get("adapter_id")
        if (
            not isinstance(adapter_id, str)
            or adapter_id not in expected_adapter_ids
            or adapter_id in statuses_by_adapter
        ):
            invalid_or_ambiguous = True
            continue
        if refs is not None and (
            identity.get("plan_sha256") != _single_ref_sha256(refs, "external_eval_plan")
            or identity.get("heldout_manifest_sha256") != _single_ref_sha256(refs, "heldout_manifest")
        ):
            invalid_or_ambiguous = True
            continue
        statuses_by_adapter[adapter_id] = _execution_result_payload_status("external_eval_result", payload)
    if invalid_or_ambiguous:
        return "incomplete"
    if any(status == "failed" for status in statuses_by_adapter.values()):
        return "failed"
    if set(statuses_by_adapter) != set(expected_adapter_ids):
        return "incomplete"
    if all(status == "completed" for status in statuses_by_adapter.values()):
        return "completed"
    return "incomplete"


def _external_eval_expected_adapter_ids(artifact_paths: dict[str, list[Path]]) -> list[str]:
    plan = _single_payload(artifact_paths, "external_eval_plan")
    selected = plan.get("selected_adapters")
    if not isinstance(selected, list) or not all(isinstance(item, str) and item for item in selected):
        return []
    adapter_ids = sorted(selected)
    return adapter_ids if len(adapter_ids) == len(set(adapter_ids)) else []


def _execution_result_payload_status(role: str, payload: dict[str, Any]) -> str:
    if role == "agentic_training_result":
        training_result = payload.get("training_result")
        training_result = training_result if isinstance(training_result, dict) else {}
        status = training_result.get("status")
        if status == "completed" and payload.get("passed") is True:
            return "completed"
        if status in {"failed", "blocked", "aborted"}:
            return "failed"
        return "incomplete"
    if role == "external_eval_result":
        execution = payload.get("execution")
        execution = execution if isinstance(execution, dict) else {}
        status = execution.get("status")
        if status == "failed":
            return "failed"
        if status != "completed":
            return "incomplete"
        integrity = payload.get("integrity")
        integrity = integrity if isinstance(integrity, dict) else {}
        coverage = payload.get("coverage")
        coverage = coverage if isinstance(coverage, dict) else {}
        benchmark_outcome = payload.get("benchmark_outcome")
        benchmark_outcome = benchmark_outcome if isinstance(benchmark_outcome, dict) else {}
        governance = payload.get("governance")
        governance = governance if isinstance(governance, dict) else {}
        if (
            integrity.get("passed") is True
            and coverage.get("complete") is True
            and benchmark_outcome.get("status") in {"passed", "failed"}
        ):
            return "completed"
        return "incomplete"
    return "incomplete"


def _cloud_training_summary(refs: dict[str, list[dict[str, Any]]], receipt_state: dict[str, Any]) -> dict[str, Any]:
    required = [
        "cloud_training_provider_registry",
        "cloud_training_preflight",
        "cloud_training_artifact_manifest",
        "cloud_training_launch_plan",
        "cloud_training_launch_receipt",
        "cloud_training_status_receipt",
    ]
    present = [role for role in required if _role_ready(refs, role)]
    missing = [role for role in required if role not in present]
    return {
        "required_artifacts": required,
        "present_artifacts": present,
        "missing_artifacts": missing,
        "artifact_count": sum(len(refs.get(role, [])) for role in required),
        "provider_registry_present": "cloud_training_provider_registry" in present,
        "preflight_present": "cloud_training_preflight" in present,
        "artifact_manifest_present": "cloud_training_artifact_manifest" in present,
        "launch_plan_present": "cloud_training_launch_plan" in present,
        "launch_receipt_present": "cloud_training_launch_receipt" in present,
        "status_receipt_present": "cloud_training_status_receipt" in present,
        "provider_api_calls_started": receipt_state["provider_api_calls_started"],
        "cloud_jobs_started": receipt_state["cloud_jobs_started"],
        "credential_values_recorded": receipt_state["credential_values_recorded"],
        "live_spend_allowed": False,
    }


def _cloud_training_receipt_state(artifact_paths: dict[str, list[Path]]) -> dict[str, Any]:
    launch_records = _payload_records(artifact_paths, "cloud_training_launch_receipt")
    status_records = _payload_records(artifact_paths, "cloud_training_status_receipt")
    launch_payloads = [record["payload"] for record in launch_records]
    status_payloads = [record["payload"] for record in status_records]
    all_payloads = [*launch_payloads, *status_payloads]
    first_launch_payload = launch_payloads[0] if launch_payloads else {}
    first_status_payload = status_payloads[0] if status_payloads else {}
    first_launch = first_launch_payload.get("launch") if isinstance(first_launch_payload.get("launch"), dict) else {}
    first_status = first_status_payload.get("status") if isinstance(first_status_payload.get("status"), dict) else {}
    provider_api_calls_started = any(
        _cloud_training_launch(payload).get("provider_api_called") is True for payload in launch_payloads
    ) or any(_cloud_training_status(payload).get("provider_api_called") is True for payload in status_payloads) or any(
        _cloud_training_boundary(payload).get("provider_api_called") is True for payload in all_payloads
    )
    cloud_jobs_started = any(
        _cloud_training_launch(payload).get("cloud_job_started") is True for payload in launch_payloads
    ) or any(_cloud_training_boundary(payload).get("cloud_job_started") is True for payload in all_payloads)
    provider_cancel_called = any(_cloud_training_status(payload).get("provider_cancel_called") is True for payload in status_payloads)
    credential_values_recorded = any(
        _cloud_training_boundary(payload).get("credential_values_recorded") is True for payload in all_payloads
    )
    cost_incurred_usd = sum(
        _non_negative_number_or_zero(_cloud_training_launch(payload).get("cost_incurred_usd")) for payload in launch_payloads
    ) + sum(_non_negative_number_or_zero(_cloud_training_status(payload).get("cost_incurred_usd")) for payload in status_payloads) + sum(
        _non_negative_number_or_zero(_cloud_training_boundary(payload).get("cloud_cost_incurred_usd"))
        for payload in all_payloads
    )
    launch_mode = str(first_launch.get("mode") or "")
    status_provider_status = str(first_status.get("provider_status") or "")
    live_launch_requested = any(
        _cloud_training_launch(payload).get("mode") == "live"
        or _cloud_training_boundary(payload).get("live_requested") is True
        for payload in launch_payloads
    ) or any(
        _cloud_training_boundary(payload).get("live_requested") is True for payload in status_payloads
    )
    fail_closed = (
        provider_api_calls_started is False
        and cloud_jobs_started is False
        and provider_cancel_called is False
        and credential_values_recorded is False
        and live_launch_requested is False
        and cost_incurred_usd == 0
    )
    launch_receipt_passed = bool(launch_records) and all(
        record["ready"] is True
        and _cloud_training_launch_receipt_semantic_passed(record["path"], record["payload"])
        for record in launch_records
    )
    status_receipt_passed = bool(status_records) and all(
        record["ready"] is True
        and _cloud_training_status_receipt_semantic_passed(record["path"], record["payload"])
        for record in status_records
    )
    return {
        "launch_receipt_count": len(launch_payloads),
        "status_receipt_count": len(status_payloads),
        "launch_receipt_passed": launch_receipt_passed,
        "status_receipt_passed": status_receipt_passed,
        "receipts_passed": launch_receipt_passed and status_receipt_passed,
        "launch_mode": launch_mode,
        "launch_readiness": str(first_launch_payload.get("readiness") or ""),
        "launch_recommendation": str(first_launch_payload.get("recommendation") or ""),
        "live_launch_requested": live_launch_requested,
        "status_provider_status": status_provider_status,
        "status_terminal": bool(status_payloads) and all(_cloud_training_status(payload).get("terminal") is True for payload in status_payloads),
        "status_not_started": status_provider_status == "not_started",
        "status_readiness": str(first_status_payload.get("readiness") or ""),
        "status_recommendation": str(first_status_payload.get("recommendation") or ""),
        "provider_api_calls_started": provider_api_calls_started,
        "cloud_jobs_started": cloud_jobs_started,
        "provider_cancel_called": provider_cancel_called,
        "credential_values_recorded": credential_values_recorded,
        "cost_incurred_usd": cost_incurred_usd,
        "fail_closed": fail_closed,
    }


def _external_eval_receipt_state(artifact_paths: dict[str, list[Path]]) -> dict[str, Any]:
    records = _payload_records(artifact_paths, "external_eval_receipt")
    payloads = [record["payload"] for record in records]
    first_payload = payloads[0] if payloads else {}
    first_launch = _external_eval_launch(first_payload)
    adapter_rows = [row for payload in payloads for row in _external_eval_adapter_receipts(payload)]
    adapter_contracts = [_external_eval_adapter_contract(row) for row in adapter_rows]
    live_benchmark_requested = any(_external_eval_launch(payload).get("mode") == "live" for payload in payloads)
    live_benchmarks_started = any(
        _external_eval_launch(payload).get("live_benchmarks_started") is True
        or _external_eval_boundary(payload).get("live_benchmarks_started") is True
        for payload in payloads
    ) or any(row.get("live_benchmark_started") is True for row in adapter_rows)
    provider_api_calls_started = any(
        _external_eval_launch(payload).get("provider_api_called") is True
        or _external_eval_boundary(payload).get("provider_api_called") is True
        for payload in payloads
    ) or any(row.get("provider_api_called") is True for row in adapter_rows) or any(
        contract.get("provider_api_called_by_flight_recorder") is True for contract in adapter_contracts
    )
    model_downloads_started = any(
        _external_eval_launch(payload).get("model_downloads_started") is True
        or _external_eval_boundary(payload).get("model_downloads_started") is True
        for payload in payloads
    ) or any(row.get("model_downloads_started") is True for row in adapter_rows) or any(
        contract.get("model_downloads_started_by_flight_recorder") is True for contract in adapter_contracts
    )
    credential_values_recorded = any(
        _external_eval_boundary(payload).get("credential_values_recorded") is True for payload in payloads
    ) or any(row.get("credential_values_recorded") is True for row in adapter_rows) or any(
        contract.get("credential_values_recorded") is True for contract in adapter_contracts
    )
    weights_updated_by_flight_recorder = any(
        _external_eval_boundary(payload).get("weights_updated_by_flight_recorder") is True for payload in payloads
    )
    cost_incurred_usd = (
        sum(_non_negative_number_or_zero(_external_eval_launch(payload).get("cost_incurred_usd")) for payload in payloads)
        + sum(
            _non_negative_number_or_zero(_external_eval_boundary(payload).get("cloud_cost_incurred_usd"))
            for payload in payloads
        )
        + sum(_non_negative_number_or_zero(row.get("cost_incurred_usd")) for row in adapter_rows)
        + sum(_non_negative_number_or_zero(contract.get("cost_incurred_usd")) for contract in adapter_contracts)
    )
    dry_run_only = bool(payloads) and all(_external_eval_boundary(payload).get("dry_run_only") is not False for payload in payloads)
    receipt_passed_count = sum(
        1
        for record in records
        if record["ready"] is True
        and _external_eval_receipt_semantic_passed(record["path"], record["payload"])
    )
    fail_closed = (
        live_benchmark_requested is False
        and live_benchmarks_started is False
        and provider_api_calls_started is False
        and model_downloads_started is False
        and credential_values_recorded is False
        and weights_updated_by_flight_recorder is False
        and cost_incurred_usd == 0
        and (not payloads or dry_run_only)
    )
    return {
        "receipt_count": len(payloads),
        "receipt_passed_count": receipt_passed_count,
        "receipts_passed": bool(payloads) and receipt_passed_count == len(payloads),
        "launch_mode": str(first_launch.get("mode") or ""),
        "readiness": str(first_payload.get("readiness") or ""),
        "recommendation": str(first_payload.get("recommendation") or ""),
        "adapter_count": sum(_non_negative_int_or_zero(payload.get("adapter_count")) for payload in payloads),
        "ready_adapter_count": sum(_non_negative_int_or_zero(payload.get("ready_adapter_count")) for payload in payloads),
        "dry_run_only": dry_run_only,
        "live_benchmark_requested": live_benchmark_requested,
        "live_benchmarks_started": live_benchmarks_started,
        "provider_api_calls_started": provider_api_calls_started,
        "model_downloads_started": model_downloads_started,
        "credential_values_recorded": credential_values_recorded,
        "weights_updated_by_flight_recorder": weights_updated_by_flight_recorder,
        "cost_incurred_usd": cost_incurred_usd,
        "fail_closed": fail_closed,
    }


def _eval_summary_state(artifact_paths: dict[str, list[Path]]) -> dict[str, Any]:
    records = _payload_records(artifact_paths, "eval_summary")
    valid_count = sum(1 for record in records if record["schema_valid"] is True)
    passed_count = sum(1 for record in records if record["ready"] is True)
    return {
        "summary_count": len(records),
        "valid_count": valid_count,
        "passed_count": passed_count,
        "valid": bool(records) and valid_count == len(records),
        "passed": bool(records) and passed_count == len(records),
    }


def _external_eval_handoff_binding(
    artifact_paths: dict[str, list[Path]],
    refs: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    plan_sha256 = _single_ref_sha256(refs, "external_eval_plan")
    heldout_sha256 = _single_ref_sha256(refs, "heldout_manifest")
    plan = _single_payload(artifact_paths, "external_eval_plan")
    receipt = _single_payload(artifact_paths, "external_eval_receipt")
    inputs = plan.get("inputs") if isinstance(plan.get("inputs"), dict) else {}
    scenario_manifest = (
        inputs.get("scenario_manifest") if isinstance(inputs.get("scenario_manifest"), dict) else {}
    )
    source_plan = receipt.get("source_plan") if isinstance(receipt.get("source_plan"), dict) else {}
    plan_heldout_manifest_bound = bool(plan_sha256 and heldout_sha256) and scenario_manifest.get("sha256") == heldout_sha256
    receipt_plan_bound = bool(plan_sha256) and source_plan.get("sha256") == plan_sha256
    passed = plan_heldout_manifest_bound and receipt_plan_bound
    return {
        "passed": passed,
        "plan_heldout_manifest_bound": plan_heldout_manifest_bound,
        "receipt_plan_bound": receipt_plan_bound,
        "status": "matched" if passed else "mismatched",
    }


def _agentic_training_result_plan_bound(
    artifact_paths: dict[str, list[Path]],
    refs: dict[str, list[dict[str, Any]]],
) -> bool:
    plan_sha256 = _single_ref_sha256(refs, "agentic_training_plan")
    result = _single_payload(artifact_paths, "agentic_training_result")
    lineage = result.get("lineage") if isinstance(result.get("lineage"), dict) else {}
    source_plan = lineage.get("plan") if isinstance(lineage.get("plan"), dict) else {}
    return bool(plan_sha256) and source_plan.get("sha256") == plan_sha256


def _external_eval_result_binding(
    artifact_paths: dict[str, list[Path]],
    refs: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    plan_sha256 = _single_ref_sha256(refs, "external_eval_plan")
    heldout_sha256 = _single_ref_sha256(refs, "heldout_manifest")
    expected_adapter_ids = _external_eval_expected_adapter_ids(artifact_paths)
    result_records = _payload_records(artifact_paths, "external_eval_result")
    result_identities = [
        payload.get("identity") if isinstance(payload.get("identity"), dict) else {}
        for payload in (record["payload"] for record in result_records)
    ]
    observed_adapter_ids = [
        identity.get("adapter_id") for identity in result_identities if isinstance(identity.get("adapter_id"), str)
    ]
    adapter_coverage_complete = (
        bool(expected_adapter_ids)
        and len(observed_adapter_ids) == len(set(observed_adapter_ids))
        and set(observed_adapter_ids) == set(expected_adapter_ids)
        and len(result_records) == len(artifact_paths.get("external_eval_result", []))
    )
    result_plan_bound = bool(plan_sha256) and bool(result_identities) and all(
        identity.get("plan_sha256") == plan_sha256 for identity in result_identities
    )
    result_heldout_manifest_bound = bool(heldout_sha256) and bool(result_identities) and all(
        identity.get("heldout_manifest_sha256") == heldout_sha256 for identity in result_identities
    )
    passed = adapter_coverage_complete and result_plan_bound and result_heldout_manifest_bound
    return {
        "passed": passed,
        "expected_count": len(expected_adapter_ids),
        "actual_count": len(artifact_paths.get("external_eval_result", [])),
        "adapter_coverage_complete": adapter_coverage_complete,
        "result_plan_bound": result_plan_bound,
        "result_heldout_manifest_bound": result_heldout_manifest_bound,
        "status": "matched" if passed else "mismatched",
    }


def _single_ref_sha256(refs: dict[str, list[dict[str, Any]]], role: str) -> str:
    rows = refs.get(role)
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        return ""
    value = rows[0].get("sha256")
    return value if isinstance(value, str) else ""


def _single_payload(artifact_paths: dict[str, list[Path]], role: str) -> dict[str, Any]:
    records = _payload_records(artifact_paths, role)
    return records[0]["payload"] if len(records) == 1 else {}


def _eval_summary_result_binding(
    artifact_paths: dict[str, list[Path]],
    refs: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    result_refs = refs.get("external_eval_result", [])
    summary_records = _payload_records(artifact_paths, "eval_summary")
    expected_count = len(_external_eval_expected_adapter_ids(artifact_paths))
    if not expected_count or len(result_refs) != expected_count:
        return {
            "passed": False,
            "evidence_set_bound": False,
            "governance_ready": False,
            "expected_count": expected_count,
            "match_count": 0,
            "status": "external_eval_result_count_invalid",
        }
    if len(summary_records) != 1:
        return {
            "passed": False,
            "evidence_set_bound": False,
            "governance_ready": False,
            "expected_count": expected_count,
            "match_count": 0,
            "status": "eval_summary_count_invalid",
        }
    result_sha256s = [row.get("sha256") for row in result_refs]
    if (
        not all(isinstance(value, str) and value for value in result_sha256s)
        or len(result_sha256s) != len(set(result_sha256s))
    ):
        return {
            "passed": False,
            "evidence_set_bound": False,
            "governance_ready": False,
            "expected_count": expected_count,
            "match_count": 0,
            "status": "external_eval_result_sha256_invalid",
        }
    summary = summary_records[0]["payload"]
    result_rows = summary.get("external_adapter_results")
    rows = [row for row in result_rows if isinstance(row, dict)] if isinstance(result_rows, list) else []
    matches = [row for row in rows if row.get("sha256") in set(result_sha256s)]
    summary_sha256s = [row.get("sha256") for row in rows]
    exact_set = (
        len(rows) == expected_count
        and all(isinstance(value, str) and value for value in summary_sha256s)
        and len(summary_sha256s) == len(set(summary_sha256s))
        and set(summary_sha256s) == set(result_sha256s)
    )
    evidence_set_bound = exact_set and all(
        match.get("integrity_passed") is True
        and match.get("execution_status") == "completed"
        and match.get("benchmark_status") in {"passed", "failed"}
        and match.get("coverage_complete") is True
        for match in matches
    )
    governance_ready = evidence_set_bound and all(
        match.get("benchmark_status") == "passed"
        and match.get("governance_readiness") == "ready_for_review"
        and match.get("external_eval_claims_allowed") is True
        for match in matches
    )
    return {
        "passed": governance_ready,
        "evidence_set_bound": evidence_set_bound,
        "governance_ready": governance_ready,
        "expected_count": expected_count,
        "match_count": len(matches),
        "status": (
            "matched"
            if governance_ready
            else "matched_but_not_governance_ready"
            if evidence_set_bound
            else "matching_eval_summary_result_set_invalid"
        ),
    }


def _promotion_governance_state(artifact_paths: dict[str, list[Path]]) -> dict[str, Any]:
    decision_records = _payload_records(artifact_paths, "promotion_decision")
    ledger_records = _payload_records(artifact_paths, "promotion_ledger")
    decision_schema_valid = bool(decision_records) and all(record["schema_valid"] is True for record in decision_records)
    decision_passed = bool(decision_records) and all(record["ready"] is True for record in decision_records)
    ledger_schema_valid = bool(ledger_records) and all(record["schema_valid"] is True for record in ledger_records)
    return {
        "promotion_decision_present": bool(decision_records),
        "promotion_decision_count": len(decision_records),
        "promotion_decision_schema_valid": decision_schema_valid,
        "promotion_decision_passed": decision_passed,
        "promotion_ledger_present": bool(ledger_records),
        "promotion_ledger_count": len(ledger_records),
        "promotion_ledger_schema_valid": ledger_schema_valid,
        "passed": decision_schema_valid and decision_passed and ledger_schema_valid,
    }


def _external_eval_launch(payload: dict[str, Any]) -> dict[str, Any]:
    launch = payload.get("launch")
    return launch if isinstance(launch, dict) else {}


def _external_eval_boundary(payload: dict[str, Any]) -> dict[str, Any]:
    boundary = payload.get("execution_boundary")
    return boundary if isinstance(boundary, dict) else {}


def _external_eval_adapter_receipts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("adapter_receipts")
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _external_eval_adapter_contract(row: dict[str, Any]) -> dict[str, Any]:
    contract = row.get("adapter_contract")
    return contract if isinstance(contract, dict) else {}


def _cloud_training_lineage(refs: dict[str, list[dict[str, Any]]], artifact_paths: dict[str, list[Path]]) -> dict[str, Any]:
    provider = _cloud_training_provider_lineage(artifact_paths)
    links = [_cloud_training_lineage_link(refs, artifact_paths, spec) for spec in CLOUD_TRAINING_LINEAGE_LINKS]
    missing_links = [link["id"] for link in links if link["status"].startswith("missing_")]
    mismatched_links = [link["id"] for link in links if link["status"] == "mismatched_sha256"]
    ambiguous_links = [link["id"] for link in links if link["status"].startswith("ambiguous_")]
    role_counts = _cloud_training_lineage_role_counts(refs)
    duplicate_roles = [row["role"] for row in role_counts if row["count"] > 1]
    matched_link_count = sum(1 for link in links if link["passed"])
    passed = (
        provider["provider_consistent"]
        and provider["registry_contains_pipeline_provider"]
        and not missing_links
        and not mismatched_links
        and not ambiguous_links
        and not duplicate_roles
    )
    return {
        "passed": passed,
        "required_link_count": len(links),
        "matched_link_count": matched_link_count,
        "missing_link_count": len(missing_links),
        "mismatched_link_count": len(mismatched_links),
        "ambiguous_link_count": len(ambiguous_links),
        "duplicate_role_count": len(duplicate_roles),
        "missing_links": missing_links,
        "mismatched_links": mismatched_links,
        "ambiguous_links": ambiguous_links,
        "duplicate_roles": duplicate_roles,
        "role_counts": role_counts,
        "provider": provider,
        "links": links,
    }


def _cloud_training_provider_lineage(artifact_paths: dict[str, list[Path]]) -> dict[str, Any]:
    registry_provider_ids = sorted(
        {
            provider_id
            for path in artifact_paths.get("cloud_training_provider_registry", [])
            for provider_id in _registry_provider_ids(_read_json(path))
        }
    )
    provider_by_role = {
        role: _provider_id_from_payload(_first_payload(artifact_paths, role))
        for role in ("cloud_training_preflight", "cloud_training_artifact_manifest", "cloud_training_launch_plan")
    }
    pipeline_provider_ids = sorted({provider_id for provider_id in provider_by_role.values() if provider_id})
    pipeline_provider_id = pipeline_provider_ids[0] if len(pipeline_provider_ids) == 1 else ""
    registry_contains_pipeline_provider = bool(pipeline_provider_id) and pipeline_provider_id in registry_provider_ids
    return {
        "registry_provider_ids": registry_provider_ids,
        "pipeline_provider_ids": pipeline_provider_ids,
        "pipeline_provider_id": pipeline_provider_id,
        "provider_by_role": provider_by_role,
        "provider_consistent": len(pipeline_provider_ids) == 1,
        "registry_contains_pipeline_provider": registry_contains_pipeline_provider,
    }


def _cloud_training_lineage_link(
    refs: dict[str, list[dict[str, Any]]],
    artifact_paths: dict[str, list[Path]],
    spec: dict[str, str],
) -> dict[str, Any]:
    source_role = spec["source_role"]
    target_role = spec["target_role"]
    source_ref_name = spec["source_ref"]
    source_count = _lineage_role_count(refs, source_role)
    target_count = _lineage_role_count(refs, target_role)
    source_ref = _first_ref(refs, source_role)
    target_ref = _first_ref(refs, target_role)
    source_payload = _first_payload(artifact_paths, source_role)
    source_artifacts = source_payload.get("source_artifacts") if isinstance(source_payload.get("source_artifacts"), dict) else {}
    nested_ref = source_artifacts.get(source_ref_name) if isinstance(source_artifacts, dict) else None
    nested_ref = nested_ref if isinstance(nested_ref, dict) else {}
    nested_sha = nested_ref.get("sha256") if isinstance(nested_ref.get("sha256"), str) else ""
    target_sha = target_ref.get("sha256") if isinstance(target_ref.get("sha256"), str) else ""
    status = "matched"
    if source_count > 1:
        status = "ambiguous_source_artifacts"
    elif target_count > 1:
        status = "ambiguous_target_artifacts"
    elif not source_ref:
        status = "missing_source_artifact"
    elif not target_ref:
        status = "missing_target_artifact"
    elif not nested_ref:
        status = "missing_source_link"
    elif not nested_sha:
        status = "missing_source_link_sha256"
    elif not target_sha:
        status = "missing_target_sha256"
    elif nested_sha != target_sha:
        status = "mismatched_sha256"
    return {
        "id": spec["id"],
        "source_role": source_role,
        "source_ref": source_ref_name,
        "target_role": target_role,
        "source_artifact_count": source_count,
        "target_artifact_count": target_count,
        "source_schema_version": source_ref.get("schema_version", "") if source_ref else "",
        "target_schema_version": target_ref.get("schema_version", "") if target_ref else "",
        "source_ref_sha256": nested_sha,
        "target_sha256": target_sha,
        "passed": status == "matched",
        "status": status,
    }


def _cloud_training_lineage_role_counts(refs: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [{"role": role, "count": _lineage_role_count(refs, role)} for role in CLOUD_TRAINING_LINEAGE_ARTIFACT_ROLES]


def _lineage_role_count(refs: dict[str, list[dict[str, Any]]], role: str) -> int:
    rows = refs.get(role)
    return len(rows) if isinstance(rows, list) else 0


def _first_ref(refs: dict[str, list[dict[str, Any]]], role: str) -> dict[str, Any]:
    rows = refs.get(role)
    return rows[0] if isinstance(rows, list) and rows and isinstance(rows[0], dict) else {}


def _first_payload(artifact_paths: dict[str, list[Path]], role: str) -> dict[str, Any]:
    records = _payload_records(artifact_paths, role)
    return records[0]["payload"] if records else {}


def _payload_records(artifact_paths: dict[str, list[Path]], role: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in artifact_paths.get(role, []):
        source = inspect_artifact_source(path, role)
        payload = source["payload"] if isinstance(source.get("payload"), dict) else {}
        if source.get("parse_valid") is True:
            records.append(
                {
                    "path": path,
                    "payload": payload,
                    "schema_valid": source.get("schema_valid") is True,
                    "ready": source.get("ready") is True,
                }
            )
    return records


def _cloud_training_launch_receipt_semantic_passed(receipt_path: Path, receipt: dict[str, Any]) -> bool:
    launch_plan_path = _cloud_training_receipt_source_path(receipt_path, receipt, "launch_plan")
    if launch_plan_path is None:
        return False
    launch = _cloud_training_launch(receipt)
    mode = launch.get("mode")
    if mode not in {"dry_run", "live"}:
        return False
    try:
        expected = build_cloud_training_launch_receipt(
            launch_plan_path=launch_plan_path,
            live=mode == "live",
            output_base_dir=receipt_path.parent,
            created_at=receipt.get("created_at") if isinstance(receipt.get("created_at"), str) else None,
        )
    except (OSError, TypeError, ValueError):
        return False
    return _cloud_training_receipt_matches_replay(
        receipt,
        expected,
        (
            "passed",
            "readiness",
            "recommendation",
            "check_count",
            "failed_check_count",
            "checks",
            "blocked_reasons",
            "source_artifacts",
            "launch",
            "execution_boundary",
        ),
    )


def _cloud_training_status_receipt_semantic_passed(receipt_path: Path, receipt: dict[str, Any]) -> bool:
    launch_receipt_path = _cloud_training_receipt_source_path(receipt_path, receipt, "launch_receipt")
    if launch_receipt_path is None:
        return False
    status = _cloud_training_status(receipt)
    cancel_requested = status.get("cancel_requested")
    if not isinstance(cancel_requested, bool):
        return False
    try:
        expected = build_cloud_training_status_receipt(
            launch_receipt_path=launch_receipt_path,
            cancel_requested=cancel_requested,
            output_base_dir=receipt_path.parent,
            created_at=receipt.get("created_at") if isinstance(receipt.get("created_at"), str) else None,
        )
    except (OSError, TypeError, ValueError):
        return False
    return _cloud_training_receipt_matches_replay(
        receipt,
        expected,
        (
            "passed",
            "readiness",
            "recommendation",
            "check_count",
            "failed_check_count",
            "checks",
            "blocked_reasons",
            "source_artifacts",
            "status",
            "execution_boundary",
        ),
    )


def _cloud_training_receipt_source_path(receipt_path: Path, receipt: dict[str, Any], ref_name: str) -> Path | None:
    sources = receipt.get("source_artifacts")
    if not isinstance(sources, dict):
        return None
    ref = sources.get(ref_name)
    if not isinstance(ref, dict) or ref.get("exists") is not True:
        return None
    value = ref.get("path")
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else receipt_path.parent / path


def _cloud_training_receipt_matches_replay(receipt: dict[str, Any], expected: dict[str, Any], fields: tuple[str, ...]) -> bool:
    return all(receipt.get(field_name) == expected.get(field_name) for field_name in fields)


def _external_eval_receipt_semantic_passed(receipt_path: Path, receipt: dict[str, Any]) -> bool:
    source_plan_path = _external_eval_receipt_source_plan_path(receipt_path, receipt)
    if source_plan_path is None:
        return False
    source_plan = receipt.get("source_plan")
    if not isinstance(source_plan, dict) or not _external_eval_receipt_source_plan_matches(source_plan_path, source_plan):
        return False
    launch = _external_eval_launch(receipt)
    mode = launch.get("mode")
    if mode not in {"dry_run", "live"}:
        return False
    adapter_rows = _external_eval_adapter_receipts(receipt)
    adapter_ids = [str(row.get("id")) for row in adapter_rows if isinstance(row.get("id"), str) and row.get("id")]
    if len(adapter_ids) != len(adapter_rows) or len(adapter_ids) != len(set(adapter_ids)):
        return False
    source_plan_payload = _read_json(source_plan_path)
    selected_adapters = source_plan_payload.get("selected_adapters")
    if not isinstance(selected_adapters, list) or not all(isinstance(adapter, str) for adapter in selected_adapters):
        return False
    if sorted(adapter_ids) != sorted(selected_adapters):
        return False
    try:
        expected = build_external_eval_receipt(
            plan_path=source_plan_path,
            adapters=selected_adapters,
            live=mode == "live",
            created_at=receipt.get("created_at") if isinstance(receipt.get("created_at"), str) else None,
            output_base_dir=receipt_path.parent,
        )
    except ExternalEvalPlanError:
        return False
    for field_name in (
        "passed",
        "readiness",
        "recommendation",
        "check_count",
        "failed_check_count",
        "checks",
        "blocked_reasons",
        "adapter_count",
        "ready_adapter_count",
        "adapter_receipts",
        "launch",
        "execution_boundary",
    ):
        if receipt.get(field_name) != expected.get(field_name):
            return False
    return True


def _external_eval_receipt_source_plan_path(receipt_path: Path, receipt: dict[str, Any]) -> Path | None:
    source_plan = receipt.get("source_plan")
    if not isinstance(source_plan, dict):
        return None
    value = source_plan.get("path")
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else receipt_path.parent / path


def _external_eval_receipt_source_plan_matches(plan_path: Path, source_plan: dict[str, Any]) -> bool:
    if source_plan.get("exists") is not True or not plan_path.is_file():
        return False
    size_bytes = source_plan.get("size_bytes")
    if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
        return False
    if size_bytes != plan_path.stat().st_size:
        return False
    expected_sha = source_plan.get("sha256")
    if not isinstance(expected_sha, str) or expected_sha != _sha256(plan_path):
        return False
    plan = _read_json(plan_path)
    for field_name in ("schema_version", "ready", "adapter_count"):
        if source_plan.get(field_name) != plan.get(field_name):
            return False
    return True


def _cloud_training_launch(payload: dict[str, Any]) -> dict[str, Any]:
    launch = payload.get("launch")
    return launch if isinstance(launch, dict) else {}


def _cloud_training_status(payload: dict[str, Any]) -> dict[str, Any]:
    status = payload.get("status")
    return status if isinstance(status, dict) else {}


def _cloud_training_boundary(payload: dict[str, Any]) -> dict[str, Any]:
    boundary = payload.get("execution_boundary")
    return boundary if isinstance(boundary, dict) else {}


def _registry_provider_ids(payload: dict[str, Any]) -> list[str]:
    providers = payload.get("providers")
    if not isinstance(providers, list):
        return []
    return [str(provider.get("id")) for provider in providers if isinstance(provider, dict) and str(provider.get("id") or "")]


def _provider_id_from_payload(payload: dict[str, Any]) -> str:
    provider = payload.get("provider")
    if not isinstance(provider, dict):
        return ""
    return str(provider.get("id") or "")


def _non_negative_number_or_zero(value: Any) -> int | float:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0 else 0


def _non_negative_int_or_zero(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _refs_are_safe(refs: dict[str, list[dict[str, Any]]]) -> bool:
    return not _unsafe_refs(refs)


def _unsafe_refs(refs: dict[str, list[dict[str, Any]]]) -> list[str]:
    unsafe: list[str] = []
    for rows in refs.values():
        for row in rows:
            path = str(row.get("path") or "")
            if not path or Path(path).is_absolute() or ".." in Path(path).parts:
                unsafe.append(path)
    return unsafe


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _display_path(path: Path, preserve_paths: bool) -> str:
    if preserve_paths:
        return str(path)
    return path.name


def _display_source_path(path: Path, output_path: Path, preserve_paths: bool) -> str:
    if preserve_paths:
        return str(path)
    try:
        source = path if path.is_absolute() else Path.cwd() / path
        output_dir = output_path.parent if output_path.is_absolute() else Path.cwd() / output_path.parent
        return os.path.relpath(source.resolve(), output_dir.resolve())
    except (OSError, ValueError):
        return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_tree_fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    file_count = 0
    size_bytes = 0
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file() and not candidate.is_symlink()):
        relative = item.relative_to(path)
        size = item.stat().st_size
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(_sha256(item).encode("ascii"))
        digest.update(b"\0")
        file_count += 1
        size_bytes += size
    return {"sha256": digest.hexdigest(), "file_count": file_count, "size_bytes": size_bytes}


def _directory_contains_symlink(path: Path) -> bool:
    return any(item.is_symlink() for item in path.rglob("*"))


def _non_negative_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _non_negative_number_or_none(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and value >= 0:
        return value
    return None


def _add_check(
    checks: list[dict[str, Any]],
    check_id: str,
    passed: bool,
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    checks.append(
        {
            "id": check_id,
            "passed": bool(passed),
            "actual": actual,
            "expected": expected,
            "summary": f"{check_id}: passed={bool(passed)}",
        }
    )
