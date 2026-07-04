"""Closed-loop agentic training iteration contracts."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

AGENTIC_TRAINING_LOOP_PLAN_SCHEMA_VERSION = "hfr.agentic_training_loop_plan.v1"

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
        "produces": ("review_manifest", "model_grader_dry_run", "model_grader_override_receipt", "reviewed_manifest"),
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
        "required": ("heldout_manifest", "external_eval_plan", "external_eval_receipt", "eval_summary"),
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
        "produces": ("promotion_cards", "promotion_release_record"),
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
    "harness_manifest": "harness_manifest",
    "harness_result": "harness_result",
    "heldout_manifest": "heldout_manifest",
    "improvement_ledger": "improvement_ledger",
    "improvement_plan": "improvement_plan",
    "promotion_decision": "promotion_decision",
    "promotion_ledger": "promotion_ledger",
    "model_grader_dry_run": "model_grader_dry_run",
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
    cloud_training = _cloud_training_summary(refs, cloud_training_receipt_state)
    cloud_training_lineage = _cloud_training_lineage(refs, normalized_artifact_paths)
    phases = [_phase_row(spec, refs) for spec in PHASES]
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
        "agentic_rollout_plan" in refs and "agentic_rollout_receipt" in refs,
        {
            "agentic_rollout_plan_present": "agentic_rollout_plan" in refs,
            "agentic_rollout_receipt_present": "agentic_rollout_receipt" in refs,
        },
        {"agentic_rollout_plan_present": True, "agentic_rollout_receipt_present": True},
    )
    _add_check(
        checks,
        "uncalibrated_labels_block_training_data",
        "rubric_spec" in refs
        and "model_grader_gate" in refs
        and "review_calibration" in refs
        and "reviewed_gate" in refs
        and "rejection_sampling_gate" in refs,
        {
            "rubric_spec_present": "rubric_spec" in refs,
            "model_grader_gate_present": "model_grader_gate" in refs,
            "review_calibration_present": "review_calibration" in refs,
            "reviewed_gate_present": "reviewed_gate" in refs,
            "rejection_sampling_gate_present": "rejection_sampling_gate" in refs,
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
        "rejection_sampling_gate" in refs and "dataset_curation_receipt" in refs and "training_export" in refs,
        {
            "rejection_sampling_gate_present": "rejection_sampling_gate" in refs,
            "dataset_curation_receipt_present": "dataset_curation_receipt" in refs,
            "training_export_present": "training_export" in refs,
        },
        {"rejection_sampling_gate_present": True, "dataset_curation_receipt_present": True, "training_export_present": True},
    )
    _add_check(
        checks,
        "external_trainer_handoff_is_preflighted",
        "agentic_training_plan" in refs and "trainer_preflight" in refs and "trainer_launch_check" in refs,
        {
            "agentic_training_plan_present": "agentic_training_plan" in refs,
            "trainer_preflight_present": "trainer_preflight" in refs,
            "trainer_launch_check_present": "trainer_launch_check" in refs,
        },
        {
            "agentic_training_plan_present": True,
            "trainer_preflight_present": True,
            "trainer_launch_check_present": True,
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
        "heldout_eval_is_fail_closed",
        "heldout_manifest" in refs and "external_eval_plan" in refs and "external_eval_receipt" in refs,
        {
            "heldout_manifest_present": "heldout_manifest" in refs,
            "external_eval_plan_present": "external_eval_plan" in refs,
            "external_eval_receipt_present": "external_eval_receipt" in refs,
        },
        {"heldout_manifest_present": True, "external_eval_plan_present": True, "external_eval_receipt_present": True},
    )
    _add_check(
        checks,
        "governance_required_for_promotion",
        "promotion_decision" in refs and "promotion_ledger" in refs,
        {"promotion_decision_present": "promotion_decision" in refs, "promotion_ledger_present": "promotion_ledger" in refs},
        {"promotion_decision_present": True, "promotion_ledger_present": True},
    )
    failed_checks = [check for check in checks if not check["passed"]]
    missing_phase_inputs = sorted({missing for phase in phases for missing in phase["missing_required_artifacts"]})
    readiness = "ready_for_governance_review" if not failed_checks and not missing_phase_inputs else "planned_fail_closed"
    recommendation = "approve_iteration_execution" if readiness == "ready_for_governance_review" else "collect_missing_receipts_before_live_execution"

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
        "passed": not failed_checks,
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
    exists = path.exists()
    is_file = path.is_file()
    is_dir = path.is_dir()
    payload = _read_json(path) if is_file and path.suffix == ".json" else {}
    return {
        "role": role,
        "path": _display_source_path(path, output_path, preserve_paths),
        "kind": "directory" if is_dir else "file",
        "exists": exists,
        "sha256": _sha256(path) if is_file else None,
        "size_bytes": path.stat().st_size if is_file else None,
        "schema_version": str(payload.get("schema_version") or "") if payload else "",
        "passed": payload.get("passed") if isinstance(payload.get("passed"), bool) else None,
        "readiness": str(payload.get("readiness") or payload.get("status") or ""),
    }


def _phase_row(spec: dict[str, Any], refs: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    required = list(spec["required"])
    present = [role for role in required if role in refs and refs[role]]
    missing = [role for role in required if role not in present]
    status = "ready" if required and not missing else "planned"
    if missing and required:
        status = "blocked"
    return {
        "id": spec["id"],
        "name": spec["name"],
        "status": status,
        "required_artifacts": required,
        "present_required_artifacts": present,
        "missing_required_artifacts": missing,
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


def _cloud_training_summary(refs: dict[str, list[dict[str, Any]]], receipt_state: dict[str, Any]) -> dict[str, Any]:
    required = [
        "cloud_training_provider_registry",
        "cloud_training_preflight",
        "cloud_training_artifact_manifest",
        "cloud_training_launch_plan",
        "cloud_training_launch_receipt",
        "cloud_training_status_receipt",
    ]
    present = [role for role in required if refs.get(role)]
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
    launch_payloads = _payloads(artifact_paths, "cloud_training_launch_receipt")
    status_payloads = _payloads(artifact_paths, "cloud_training_status_receipt")
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
    return {
        "launch_receipt_count": len(launch_payloads),
        "status_receipt_count": len(status_payloads),
        "launch_receipt_passed": bool(launch_payloads) and all(payload.get("passed") is True for payload in launch_payloads),
        "status_receipt_passed": bool(status_payloads) and all(payload.get("passed") is True for payload in status_payloads),
        "receipts_passed": bool(launch_payloads)
        and bool(status_payloads)
        and all(payload.get("passed") is True for payload in all_payloads),
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
    paths = artifact_paths.get(role)
    return _read_json(paths[0]) if paths else {}


def _payloads(artifact_paths: dict[str, list[Path]], role: str) -> list[dict[str, Any]]:
    return [_read_json(path) for path in artifact_paths.get(role, [])]


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
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except (OSError, ValueError):
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
