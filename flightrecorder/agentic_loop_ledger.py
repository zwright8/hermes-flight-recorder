"""Longitudinal ledgers over closed-loop agentic training iterations."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .agentic_training_loop_plan import AGENTIC_TRAINING_LOOP_PLAN_SCHEMA_VERSION

AGENTIC_LOOP_LEDGER_SCHEMA_VERSION = "hfr.agentic_loop_ledger.v1"

ROLE_GROUPS: dict[str, tuple[str, ...]] = {
    "rollouts": ("agentic_rollout_plan", "agentic_rollout_receipt", "harness_manifest", "harness_result"),
    "evidence": ("evidence_bundle", "evidence_coverage", "trace_observability"),
    "review": (
        "rubric_spec",
        "model_grader_dry_run",
        "model_grader_override_receipt",
        "model_grader_gate",
        "review_calibration",
        "reviewed_gate",
    ),
    "datasets": ("rejection_sampling_gate", "dataset_curation_receipt", "training_export", "dataset_registry", "dataset_splits"),
    "cloud_training": (
        "cloud_training_provider_registry",
        "cloud_training_preflight",
        "cloud_training_artifact_manifest",
        "cloud_training_launch_plan",
        "cloud_training_launch_receipt",
        "cloud_training_status_receipt",
    ),
    "training": (
        "agentic_training_plan",
        "agentic_training_runtime_preflight",
        "agentic_training_flow",
        "agentic_training_result",
        "trainer_preflight",
        "trainer_launch_check",
    ),
    "serving": ("serving_lifecycle", "serving_endpoint_check", "model_serving_probe_receipt"),
    "eval": ("heldout_manifest", "external_eval_plan", "external_eval_receipt", "eval_summary"),
    "improvement": ("repair_queue", "improvement_plan", "improvement_ledger", "action_ledger"),
    "governance": (
        "agentic_loop_governance_receipt",
        "promotion_decision",
        "promotion_ledger",
        "promotion_alias_apply",
        "promotion_rollback_receipt",
    ),
    "next_iteration": ("action_ledger", "next_iteration_schedule"),
}


class AgenticLoopLedgerError(ValueError):
    """Raised when an agentic loop ledger cannot be produced."""


def build_agentic_loop_ledger(
    plan_paths: list[str | Path],
    *,
    out_path: str | Path | None = None,
    preserve_paths: bool = False,
) -> dict[str, Any]:
    """Build a deterministic ledger over closed-loop iteration plans."""
    if not plan_paths:
        raise AgenticLoopLedgerError("At least one --plan path is required.")
    output_path = Path(out_path) if out_path is not None else None
    iterations: list[dict[str, Any]] = []
    for index, raw_path in enumerate(plan_paths):
        path = Path(raw_path)
        plan = _read_loop_plan(path)
        iterations.append(_iteration_record(path, plan, index, output_path, preserve_paths))
    metrics = _metrics(iterations)
    decision = _decision(iterations, metrics)
    return {
        "schema_version": AGENTIC_LOOP_LEDGER_SCHEMA_VERSION,
        "ledger_path": _display_path(Path(out_path), preserve_paths) if out_path is not None else "",
        "passed": True,
        "iteration_count": len(iterations),
        "iterations": iterations,
        "metrics": metrics,
        "decision": decision,
        "readiness_digest": _readiness_digest(iterations, decision),
        "execution_boundary": {
            "ledger_only": True,
            "cloud_jobs_started": False,
            "paid_model_grader_calls_started": False,
            "live_benchmarks_started": False,
            "model_downloads_started": False,
            "weights_updated_by_flight_recorder": False,
            "credential_values_recorded": False,
        },
        "notes": [
            "Agentic loop ledgers summarize archived loop-plan receipts across iterations; they do not execute training or promotion.",
            "Each iteration groups rollout, review, trainer, serving, eval, governance, repair, cost, and next-action posture.",
            "Promotion, rollback, live benchmark, and cloud trainer actions remain separate governed receipts.",
        ],
    }


def write_agentic_loop_ledger(path: str | Path, ledger: dict[str, Any]) -> None:
    """Write a deterministic loop ledger artifact."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_loop_plan(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise AgenticLoopLedgerError(f"Loop plan must not be a symlink: {path}")
    if not path.exists() or not path.is_file():
        raise AgenticLoopLedgerError(f"Loop plan not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AgenticLoopLedgerError(f"Loop plan must contain a JSON object: {path}")
    if payload.get("schema_version") != AGENTIC_TRAINING_LOOP_PLAN_SCHEMA_VERSION:
        raise AgenticLoopLedgerError(f"Loop plan has unsupported schema_version at {path}: {payload.get('schema_version')!r}")
    return payload


def _iteration_record(
    path: Path,
    plan: dict[str, Any],
    index: int,
    out_path: Path | None,
    preserve_paths: bool,
) -> dict[str, Any]:
    source_artifacts = plan.get("source_artifacts") if isinstance(plan.get("source_artifacts"), dict) else {}
    budget = plan.get("budget") if isinstance(plan.get("budget"), dict) else {}
    next_iteration = plan.get("next_iteration") if isinstance(plan.get("next_iteration"), dict) else {}
    record: dict[str, Any] = {
        "index": index,
        "path": _display_path_for_output_source(path, out_path, preserve_paths),
        "exists": path.exists(),
        "schema_version": str(plan.get("schema_version") or ""),
        "iteration_id": str(plan.get("iteration_id") or ""),
        "passed": plan.get("passed") is True,
        "readiness": str(plan.get("readiness") or ""),
        "recommendation": str(plan.get("recommendation") or ""),
        "missing_phase_inputs": [str(item) for item in plan.get("missing_phase_inputs", []) if isinstance(item, str)],
        "blocked_reason_count": len(plan.get("blocked_reasons", [])) if isinstance(plan.get("blocked_reasons"), list) else 0,
        "artifact_count": _non_negative_int(plan.get("artifact_count")),
        "phase_status_counts": _phase_status_counts(plan.get("phases")),
        "artifact_group_counts": _artifact_group_counts(source_artifacts),
        "artifact_role_counts": _role_counts(source_artifacts),
        "cost_estimate": {
            "max_cloud_cost_usd": _number_or_none(budget.get("max_cloud_cost_usd")),
            "max_gpu_hours": _number_or_none(budget.get("max_gpu_hours")),
            "live_spend_allowed": budget.get("live_spend_allowed") is True,
        },
        "serving": _group_summary(source_artifacts, "serving"),
        "evals": _group_summary(source_artifacts, "eval"),
        "cloud_training": _cloud_training_summary(source_artifacts, plan),
        "cloud_training_receipt_state": _cloud_training_receipt_state(plan),
        "cloud_training_lineage": _cloud_training_lineage(plan),
        "training_outputs": _group_summary(source_artifacts, "training"),
        "governance": _governance_summary(source_artifacts, plan),
        "next_actions": {
            "scheduled": next_iteration.get("scheduled") is True,
            "requires_governance_decision": next_iteration.get("requires_governance_decision") is not False,
            "recommendation": str(next_iteration.get("recommendation") or ""),
            "schedule": next_iteration.get("schedule") if isinstance(next_iteration.get("schedule"), dict) else {},
        },
    }
    if path.exists() and path.is_file():
        record["size_bytes"] = path.stat().st_size
        record["sha256"] = _sha256(path)
    return record


def _metrics(iterations: list[dict[str, Any]]) -> dict[str, Any]:
    latest = iterations[-1] if iterations else {}
    ready_count = sum(1 for row in iterations if row.get("readiness") == "ready_for_governance_review")
    blocked_count = sum(1 for row in iterations if row.get("readiness") != "ready_for_governance_review")
    total_cost = sum(_number_or_zero(row.get("cost_estimate", {}).get("max_cloud_cost_usd")) for row in iterations)
    total_gpu_hours = sum(_number_or_zero(row.get("cost_estimate", {}).get("max_gpu_hours")) for row in iterations)
    return {
        "iteration_count": len(iterations),
        "ready_iteration_count": ready_count,
        "blocked_iteration_count": blocked_count,
        "latest_iteration_id": latest.get("iteration_id") if latest else "",
        "latest_readiness": latest.get("readiness") if latest else "",
        "latest_recommendation": latest.get("recommendation") if latest else "",
        "latest_missing_phase_input_count": len(latest.get("missing_phase_inputs", [])) if latest else 0,
        "scheduled_next_iteration_count": sum(1 for row in iterations if row.get("next_actions", {}).get("scheduled") is True),
        "promotion_ready_count": sum(1 for row in iterations if row.get("governance", {}).get("promotion_decision_present") is True),
        "rollback_receipt_count": sum(1 for row in iterations if row.get("governance", {}).get("rollback_receipt_present") is True),
        "total_max_cloud_cost_usd": total_cost,
        "total_max_gpu_hours": total_gpu_hours,
        "readiness_counts": _count_rows(row.get("readiness") for row in iterations),
        "recommendation_counts": _count_rows(row.get("recommendation") for row in iterations),
        "artifact_group_totals": _group_totals(iterations),
    }


def _decision(iterations: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    latest = iterations[-1] if iterations else {}
    ready_for_review = _latest_ready_for_governance_review(latest)
    governance_actions = _governance_actions(latest, ready_for_review)
    if ready_for_review:
        recommendation = "ready_for_governance_review"
        summary = f"Latest loop iteration {latest.get('iteration_id')} has all required phase receipts."
    else:
        recommendation = "continue_iteration"
        missing = len(latest.get("missing_phase_inputs", [])) if latest else 0
        summary = f"Latest loop iteration {latest.get('iteration_id')} is fail-closed with {missing} missing phase input(s)."
    return {
        "readiness": "ready" if ready_for_review else "blocked",
        "recommendation": recommendation,
        "recommended_governance_action": _recommended_governance_action(governance_actions),
        "governance_action_count": len(governance_actions),
        "governance_actions": governance_actions,
        "summary": summary,
        "latest_iteration_index": latest.get("index"),
        "latest_iteration_id": latest.get("iteration_id"),
        "blocked_iteration_count": metrics.get("blocked_iteration_count"),
    }


def _latest_ready_for_governance_review(latest: dict[str, Any]) -> bool:
    if latest.get("readiness") != "ready_for_governance_review":
        return False
    missing_phase_inputs = latest.get("missing_phase_inputs") if isinstance(latest.get("missing_phase_inputs"), list) else []
    if missing_phase_inputs:
        return False
    lineage = latest.get("cloud_training_lineage") if isinstance(latest.get("cloud_training_lineage"), dict) else {}
    receipt_state = latest.get("cloud_training_receipt_state") if isinstance(latest.get("cloud_training_receipt_state"), dict) else {}
    governance = latest.get("governance") if isinstance(latest.get("governance"), dict) else {}
    side_effects_started = any(
        governance.get(field_name) is True
        for field_name in ("cloud_jobs_started", "paid_model_grader_calls_started", "weights_updated_by_flight_recorder")
    )
    return lineage.get("passed") is True and receipt_state.get("fail_closed") is True and not side_effects_started


def _governance_actions(latest: dict[str, Any], ready_for_review: bool) -> list[dict[str, Any]]:
    governance = latest.get("governance") if isinstance(latest.get("governance"), dict) else {}
    next_actions = latest.get("next_actions") if isinstance(latest.get("next_actions"), dict) else {}
    approve_blockers: list[str] = []
    if not ready_for_review:
        approve_blockers.append("latest_iteration_not_ready_for_governance_review")
    if governance.get("promotion_decision_present") is not True:
        approve_blockers.append("missing_promotion_decision")
    rollback_blockers = [] if governance.get("rollback_receipt_present") is True else ["missing_rollback_receipt"]
    request_summary = "Governance can request another iteration."
    if next_actions.get("scheduled") is True:
        request_summary = "A next-iteration schedule receipt is already represented by the latest loop plan."
    return [
        _governance_action(
            "approve",
            not approve_blockers,
            approve_blockers,
            "Approve the iteration for promotion review when all readiness and promotion-decision receipts are present.",
        ),
        _governance_action(
            "reject",
            True,
            [],
            "Reject the iteration without moving aliases or weights.",
        ),
        _governance_action(
            "rollback",
            not rollback_blockers,
            rollback_blockers,
            "Rollback is available only when a rollback receipt is present in governance artifacts.",
        ),
        _governance_action(
            "request_another_iteration",
            True,
            [],
            request_summary,
        ),
    ]


def _governance_action(action: str, available: bool, blocked_reasons: list[str], summary: str) -> dict[str, Any]:
    return {
        "action": action,
        "available": available,
        "blocked_reason_count": len(blocked_reasons),
        "blocked_reasons": blocked_reasons,
        "summary": summary,
    }


def _recommended_governance_action(actions: list[dict[str, Any]]) -> str:
    availability = {str(action.get("action")): action.get("available") is True for action in actions}
    if availability.get("approve"):
        return "approve"
    if availability.get("rollback"):
        return "rollback"
    return "request_another_iteration"


def _readiness_digest(iterations: list[dict[str, Any]], decision: dict[str, Any]) -> dict[str, Any]:
    latest = iterations[-1] if iterations else {}
    missing_phase_inputs = latest.get("missing_phase_inputs") if isinstance(latest.get("missing_phase_inputs"), list) else []
    missing_phase_inputs = [str(item) for item in missing_phase_inputs if isinstance(item, str)]
    group_counts = {
        row.get("group"): _non_negative_int(row.get("count"))
        for row in latest.get("artifact_group_counts", [])
        if isinstance(row, dict) and isinstance(row.get("group"), str)
    }
    missing_artifact_groups = [group for group in sorted(ROLE_GROUPS) if group_counts.get(group, 0) == 0]
    governance = latest.get("governance") if isinstance(latest.get("governance"), dict) else {}
    cost_estimate = latest.get("cost_estimate") if isinstance(latest.get("cost_estimate"), dict) else {}
    next_actions = latest.get("next_actions") if isinstance(latest.get("next_actions"), dict) else {}
    cloud_training_lineage = latest.get("cloud_training_lineage") if isinstance(latest.get("cloud_training_lineage"), dict) else {}
    cloud_training_provider = (
        cloud_training_lineage.get("provider") if isinstance(cloud_training_lineage.get("provider"), dict) else {}
    )
    cloud_training_receipt_state = (
        latest.get("cloud_training_receipt_state") if isinstance(latest.get("cloud_training_receipt_state"), dict) else {}
    )
    side_effects_started = any(
        governance.get(field_name) is True
        for field_name in ("cloud_jobs_started", "paid_model_grader_calls_started", "weights_updated_by_flight_recorder")
    )
    lineage_bound = cloud_training_lineage.get("passed") is True
    receipt_state_fail_closed = cloud_training_receipt_state.get("fail_closed") is True
    ready = (
        latest.get("readiness") == "ready_for_governance_review"
        and not missing_phase_inputs
        and not side_effects_started
        and lineage_bound
        and receipt_state_fail_closed
    )
    if ready:
        summary = f"Latest loop iteration {latest.get('iteration_id')} is ready for governance review."
    else:
        summary = (
            f"Latest loop iteration {latest.get('iteration_id')} remains fail-closed with "
            f"{len(missing_phase_inputs)} missing phase input(s) across {len(missing_artifact_groups)} empty artifact group(s)."
        )
    return {
        "latest_iteration_index": latest.get("index"),
        "latest_iteration_id": str(latest.get("iteration_id") or ""),
        "readiness": str(latest.get("readiness") or ""),
        "recommendation": str(latest.get("recommendation") or ""),
        "decision_readiness": str(decision.get("readiness") or ""),
        "decision_recommendation": str(decision.get("recommendation") or ""),
        "recommended_governance_action": str(decision.get("recommended_governance_action") or ""),
        "ready_for_governance_review": ready,
        "missing_phase_input_count": len(missing_phase_inputs),
        "missing_phase_inputs": missing_phase_inputs,
        "missing_artifact_group_count": len(missing_artifact_groups),
        "missing_artifact_groups": missing_artifact_groups,
        "blocked_reason_count": _non_negative_int(latest.get("blocked_reason_count")),
        "next_action_scheduled": next_actions.get("scheduled") is True,
        "next_action_recommendation": str(next_actions.get("recommendation") or ""),
        "requires_governance_decision": next_actions.get("requires_governance_decision") is not False,
        "promotion_decision_present": governance.get("promotion_decision_present") is True,
        "rollback_receipt_present": governance.get("rollback_receipt_present") is True,
        "live_spend_allowed": cost_estimate.get("live_spend_allowed") is True,
        "side_effects_started": side_effects_started,
        "cloud_training_lineage_bound": lineage_bound,
        "cloud_training_receipts_fail_closed": receipt_state_fail_closed,
        "cloud_training_live_launch_requested": cloud_training_receipt_state.get("live_launch_requested") is True,
        "cloud_training_cost_incurred_usd": _number_or_zero(cloud_training_receipt_state.get("cost_incurred_usd")),
        "cloud_training_launch_mode": str(cloud_training_receipt_state.get("launch_mode") or ""),
        "cloud_training_status_provider_status": str(cloud_training_receipt_state.get("status_provider_status") or ""),
        "cloud_training_provider_id": str(cloud_training_provider.get("pipeline_provider_id") or ""),
        "cloud_training_missing_link_count": _non_negative_int(cloud_training_lineage.get("missing_link_count")),
        "cloud_training_mismatched_link_count": _non_negative_int(cloud_training_lineage.get("mismatched_link_count")),
        "cloud_training_ambiguous_link_count": _non_negative_int(cloud_training_lineage.get("ambiguous_link_count")),
        "cloud_training_duplicate_role_count": _non_negative_int(cloud_training_lineage.get("duplicate_role_count")),
        "summary": summary,
    }


def _artifact_group_counts(source_artifacts: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"group": group, "count": _group_count(source_artifacts, group)} for group in sorted(ROLE_GROUPS)]


def _group_summary(source_artifacts: dict[str, Any], group: str) -> dict[str, Any]:
    roles = ROLE_GROUPS[group]
    return {
        "group": group,
        "artifact_count": _group_count(source_artifacts, group),
        "roles_present": [role for role in roles if _role_count(source_artifacts, role) > 0],
        "roles_missing": [role for role in roles if _role_count(source_artifacts, role) == 0],
    }


def _cloud_training_summary(source_artifacts: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    plan_summary = plan.get("cloud_training") if isinstance(plan.get("cloud_training"), dict) else {}
    summary = _group_summary(source_artifacts, "cloud_training")
    summary.update(
        {
            "provider_registry_present": _role_count(source_artifacts, "cloud_training_provider_registry") > 0,
            "preflight_present": _role_count(source_artifacts, "cloud_training_preflight") > 0,
            "artifact_manifest_present": _role_count(source_artifacts, "cloud_training_artifact_manifest") > 0,
            "launch_plan_present": _role_count(source_artifacts, "cloud_training_launch_plan") > 0,
            "launch_receipt_present": _role_count(source_artifacts, "cloud_training_launch_receipt") > 0,
            "status_receipt_present": _role_count(source_artifacts, "cloud_training_status_receipt") > 0,
            "provider_api_calls_started": plan_summary.get("provider_api_calls_started") is True,
            "cloud_jobs_started": plan_summary.get("cloud_jobs_started") is True,
            "credential_values_recorded": plan_summary.get("credential_values_recorded") is True,
            "live_spend_allowed": plan_summary.get("live_spend_allowed") is True,
        }
    )
    return summary


def _cloud_training_lineage(plan: dict[str, Any]) -> dict[str, Any]:
    lineage = plan.get("cloud_training_lineage")
    return lineage if isinstance(lineage, dict) else {}


def _cloud_training_receipt_state(plan: dict[str, Any]) -> dict[str, Any]:
    state = plan.get("cloud_training_receipt_state")
    return state if isinstance(state, dict) else {}


def _governance_summary(source_artifacts: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    boundary = plan.get("execution_boundary") if isinstance(plan.get("execution_boundary"), dict) else {}
    return {
        **_group_summary(source_artifacts, "governance"),
        "promotion_decision_present": _role_count(source_artifacts, "promotion_decision") > 0,
        "promotion_ledger_present": _role_count(source_artifacts, "promotion_ledger") > 0,
        "rollback_receipt_present": _role_count(source_artifacts, "promotion_rollback_receipt") > 0,
        "cloud_jobs_started": boundary.get("cloud_jobs_started") is True,
        "paid_model_grader_calls_started": boundary.get("paid_model_grader_calls_started") is True,
        "weights_updated_by_flight_recorder": boundary.get("weights_updated_by_flight_recorder") is True,
    }


def _group_count(source_artifacts: dict[str, Any], group: str) -> int:
    return sum(_role_count(source_artifacts, role) for role in ROLE_GROUPS[group])


def _role_count(source_artifacts: dict[str, Any], role: str) -> int:
    rows = source_artifacts.get(role)
    return len(rows) if isinstance(rows, list) else 0


def _role_counts(source_artifacts: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role in sorted(source_artifacts):
        count = _role_count(source_artifacts, role)
        if count:
            rows.append({"role": role, "count": count})
    return rows


def _group_totals(iterations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals = {group: 0 for group in ROLE_GROUPS}
    for row in iterations:
        for item in row.get("artifact_group_counts", []) if isinstance(row.get("artifact_group_counts"), list) else []:
            if isinstance(item, dict) and item.get("group") in totals:
                totals[item["group"]] += _non_negative_int(item.get("count"))
    return [{"group": group, "count": totals[group]} for group in sorted(totals)]


def _phase_status_counts(phases: Any) -> list[dict[str, Any]]:
    if not isinstance(phases, list):
        return []
    return _count_rows(phase.get("status") for phase in phases if isinstance(phase, dict))


def _count_rows(values: Any) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return [{"id": key, "count": counts[key]} for key in sorted(counts)]


def _non_negative_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _number_or_none(value: Any) -> int | float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0 else None


def _number_or_zero(value: Any) -> int | float:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0 else 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path, preserve_paths: bool = False) -> str:
    raw = str(path)
    if preserve_paths:
        return raw
    if _is_windows_absolute(raw):
        return f"<redacted:{_basename(raw)}>"
    if not path.is_absolute():
        return raw
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except (OSError, ValueError):
        return f"<redacted:{path.name}>"


def _display_path_for_output_source(path: Path, output_path: Path | None, preserve_paths: bool = False) -> str:
    if preserve_paths or output_path is None:
        return _display_path(path, preserve_paths)
    raw = str(path)
    if _is_windows_absolute(raw):
        return f"<redacted:{_basename(raw)}>"
    resolved = path.resolve()
    out_dir = output_path.parent.resolve()
    return os.path.relpath(resolved, out_dir)


def _is_windows_absolute(value: str) -> bool:
    normalized = value.replace("/", "\\")
    return (len(normalized) >= 3 and normalized[1:3] == ":\\" and normalized[0].isalpha()) or normalized.startswith("\\\\")


def _basename(value: str) -> str:
    return value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] or "path"
