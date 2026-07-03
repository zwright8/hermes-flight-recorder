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
    "rollouts": ("harness_manifest", "harness_result"),
    "evidence": ("evidence_bundle", "evidence_coverage", "trace_observability"),
    "review": ("rubric_spec", "model_grader_dry_run", "model_grader_gate", "review_calibration", "reviewed_gate"),
    "datasets": ("training_export", "dataset_registry", "dataset_splits"),
    "training": (
        "agentic_training_plan",
        "agentic_training_runtime_preflight",
        "agentic_training_result",
        "trainer_preflight",
        "trainer_launch_check",
    ),
    "serving": ("serving_lifecycle", "serving_endpoint_check", "model_serving_probe_receipt"),
    "eval": ("heldout_manifest", "external_eval_plan", "eval_summary"),
    "improvement": ("repair_queue", "improvement_plan", "improvement_ledger", "action_ledger"),
    "governance": ("promotion_decision", "promotion_ledger", "promotion_alias_apply", "promotion_rollback_receipt"),
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
    return {
        "schema_version": AGENTIC_LOOP_LEDGER_SCHEMA_VERSION,
        "ledger_path": _display_path(Path(out_path), preserve_paths) if out_path is not None else "",
        "passed": True,
        "iteration_count": len(iterations),
        "iterations": iterations,
        "metrics": metrics,
        "decision": _decision(iterations, metrics),
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
    if latest.get("readiness") == "ready_for_governance_review":
        recommendation = "ready_for_governance_review"
        summary = f"Latest loop iteration {latest.get('iteration_id')} has all required phase receipts."
    else:
        recommendation = "continue_iteration"
        missing = len(latest.get("missing_phase_inputs", [])) if latest else 0
        summary = f"Latest loop iteration {latest.get('iteration_id')} is fail-closed with {missing} missing phase input(s)."
    return {
        "readiness": "ready" if latest.get("readiness") == "ready_for_governance_review" else "blocked",
        "recommendation": recommendation,
        "summary": summary,
        "latest_iteration_index": latest.get("index"),
        "latest_iteration_id": latest.get("iteration_id"),
        "blocked_iteration_count": metrics.get("blocked_iteration_count"),
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
