"""Side-effect-free next-iteration scheduling receipts."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

NEXT_ITERATION_SCHEDULE_SCHEMA_VERSION = "hfr.next_iteration_schedule.v1"


class NextIterationScheduleError(ValueError):
    """Raised when a next-iteration schedule receipt cannot be built."""


def build_next_iteration_schedule(
    *,
    loop_ledger_path: str | Path,
    action_ledger_path: str | Path,
    improvement_ledger_path: str | Path,
    next_iteration_id: str | None = None,
    objective: str | None = None,
    schedule: dict[str, Any] | None = None,
    out_path: str | Path | None = None,
    preserve_paths: bool = False,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic next-iteration schedule proposal without scheduling anything."""
    loop_ref = _artifact_ref(Path(loop_ledger_path), "agentic_loop_ledger", preserve_paths)
    action_ref = _artifact_ref(Path(action_ledger_path), "action_ledger", preserve_paths)
    improvement_ref = _artifact_ref(Path(improvement_ledger_path), "improvement_ledger", preserve_paths)
    ledger_inputs = {
        "agentic_loop_ledger": [loop_ref],
        "action_ledger": [action_ref],
        "improvement_ledger": [improvement_ref],
    }
    loop_metrics = loop_ref.get("metrics") if isinstance(loop_ref.get("metrics"), dict) else {}
    action_metrics = action_ref.get("metrics") if isinstance(action_ref.get("metrics"), dict) else {}
    improvement_metrics = improvement_ref.get("metrics") if isinstance(improvement_ref.get("metrics"), dict) else {}
    pressure = _pressure(loop_metrics, action_metrics, improvement_metrics)
    latest_iteration_id = str(loop_metrics.get("latest_iteration_id") or "")
    iteration_id = next_iteration_id or _default_next_iteration_id(latest_iteration_id)
    checks: list[dict[str, Any]] = []
    _add_check(
        checks,
        "loop_ledger_present",
        _ref_matches(loop_ref, "hfr.agentic_loop_ledger.v1"),
        {"exists": loop_ref.get("exists"), "schema_version": loop_ref.get("schema_version"), "passed": loop_ref.get("passed")},
        {"exists": True, "schema_version": "hfr.agentic_loop_ledger.v1", "passed": True},
    )
    _add_check(
        checks,
        "action_ledger_present",
        _ref_matches(action_ref, "hfr.action_ledger.v1"),
        {"exists": action_ref.get("exists"), "schema_version": action_ref.get("schema_version"), "passed": action_ref.get("passed")},
        {"exists": True, "schema_version": "hfr.action_ledger.v1", "passed": True},
    )
    _add_check(
        checks,
        "improvement_ledger_present",
        _ref_matches(improvement_ref, "hfr.improvement_ledger.v1"),
        {
            "exists": improvement_ref.get("exists"),
            "schema_version": improvement_ref.get("schema_version"),
            "passed": improvement_ref.get("passed"),
        },
        {"exists": True, "schema_version": "hfr.improvement_ledger.v1", "passed": True},
    )
    _add_check(
        checks,
        "next_iteration_id_present",
        bool(iteration_id),
        {"next_iteration_id": iteration_id},
        {"next_iteration_id": "non_empty"},
    )
    _add_check(
        checks,
        "flight_recorder_did_not_create_scheduler_side_effects",
        True,
        {
            "automations_created": False,
            "codex_threads_created": False,
            "calendar_events_created": False,
            "cloud_jobs_started": False,
            "weights_updated_by_flight_recorder": False,
        },
        {
            "automations_created": False,
            "codex_threads_created": False,
            "calendar_events_created": False,
            "cloud_jobs_started": False,
            "weights_updated_by_flight_recorder": False,
        },
    )
    failed = [check for check in checks if not check["passed"]]
    has_pressure = _non_negative_int(pressure.get("total_open_signal_count")) > 0
    recommendation = "create_next_loop_plan" if has_pressure else "monitor_or_promote_without_new_iteration"
    if failed:
        recommendation = "fix_schedule_inputs"
    return {
        "schema_version": NEXT_ITERATION_SCHEDULE_SCHEMA_VERSION,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "schedule_path": _display_path(Path(out_path), preserve_paths) if out_path else "",
        "passed": not failed,
        "readiness": "ready_to_schedule" if not failed else "blocked",
        "recommendation": recommendation,
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "checks": checks,
        "blocked_reasons": [check["summary"] for check in failed],
        "source_ledgers": ledger_inputs,
        "pressure": pressure,
        "next_iteration": {
            "iteration_id": iteration_id,
            "objective": objective or _default_objective(pressure, latest_iteration_id),
            "latest_iteration_id": latest_iteration_id,
            "scheduled": False,
            "trigger": "manual_or_external_scheduler",
            "schedule": schedule or {},
            "reason": _reason(pressure),
        },
        "execution_boundary": {
            "schedule_only": True,
            "automations_created": False,
            "codex_threads_created": False,
            "calendar_events_created": False,
            "cloud_jobs_started": False,
            "weights_updated_by_flight_recorder": False,
            "credential_values_recorded": False,
        },
        "notes": [
            "This receipt proposes the next loop iteration; it does not create automations, threads, calendar events, or cloud jobs.",
            "Use the proposed iteration id and objective as inputs to a subsequent agentic-loop plan after governance review.",
        ],
    }


def write_next_iteration_schedule(path: str | Path, schedule: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(json.dumps(schedule))
    for rows in payload.get("source_ledgers", {}).values():
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    row["path"] = _output_relative_path(row.get("path"), out_path.parent)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_ref(path: Path, role: str, preserve_paths: bool) -> dict[str, Any]:
    payload = _read_json(path)
    ref: dict[str, Any] = {
        "role": role,
        "path": _display_path(path, preserve_paths),
        "kind": "file",
        "exists": path.exists() and path.is_file(),
        "sha256": _sha256(path) if path.exists() and path.is_file() else None,
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
        "schema_version": str(payload.get("schema_version") or ""),
        "passed": payload.get("passed") if isinstance(payload.get("passed"), bool) else None,
    }
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    if metrics:
        ref["metrics"] = _compact_metrics(metrics)
    decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
    if decision:
        ref["decision"] = {
            "readiness": str(decision.get("readiness") or ""),
            "recommendation": str(decision.get("recommendation") or ""),
            "summary": str(decision.get("summary") or ""),
        }
    return ref


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "latest_iteration_id",
        "latest_readiness",
        "latest_missing_phase_input_count",
        "open_action_count",
        "new_action_count",
        "recurring_action_count",
        "open_work_item_count",
        "critical_open_work_item_count",
        "high_open_work_item_count",
        "resolved_work_item_count",
    )
    return {key: metrics.get(key) for key in keys if key in metrics}


def _pressure(loop_metrics: dict[str, Any], action_metrics: dict[str, Any], improvement_metrics: dict[str, Any]) -> dict[str, Any]:
    missing_phase_inputs = _non_negative_int(loop_metrics.get("latest_missing_phase_input_count"))
    open_actions = _non_negative_int(action_metrics.get("open_action_count"))
    recurring_actions = _non_negative_int(action_metrics.get("recurring_action_count"))
    open_work = _non_negative_int(improvement_metrics.get("open_work_item_count"))
    critical_work = _non_negative_int(improvement_metrics.get("critical_open_work_item_count"))
    high_work = _non_negative_int(improvement_metrics.get("high_open_work_item_count"))
    total = missing_phase_inputs + open_actions + open_work
    return {
        "latest_loop_readiness": str(loop_metrics.get("latest_readiness") or ""),
        "latest_missing_phase_input_count": missing_phase_inputs,
        "open_action_count": open_actions,
        "recurring_action_count": recurring_actions,
        "open_work_item_count": open_work,
        "critical_open_work_item_count": critical_work,
        "high_open_work_item_count": high_work,
        "total_open_signal_count": total,
    }


def _reason(pressure: dict[str, Any]) -> str:
    if _non_negative_int(pressure.get("critical_open_work_item_count")):
        return "critical_work_items_remain_open"
    if _non_negative_int(pressure.get("latest_missing_phase_input_count")):
        return "latest_loop_missing_phase_inputs"
    if _non_negative_int(pressure.get("open_action_count")) or _non_negative_int(pressure.get("open_work_item_count")):
        return "open_actions_or_work_items_remain"
    return "no_open_iteration_pressure"


def _default_objective(pressure: dict[str, Any], latest_iteration_id: str) -> str:
    reason = _reason(pressure).replace("_", " ")
    suffix = f" after {latest_iteration_id}" if latest_iteration_id else ""
    return f"Resolve {reason}{suffix}"


def _default_next_iteration_id(latest_iteration_id: str) -> str:
    return f"{latest_iteration_id}-next" if latest_iteration_id else "next-iteration"


def _ref_matches(ref: dict[str, Any], schema_version: str) -> bool:
    return ref.get("exists") is True and ref.get("schema_version") == schema_version and ref.get("passed") is True


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
        return str(path)


def _output_relative_path(value: Any, output_dir: Path) -> Any:
    if not isinstance(value, str) or not value:
        return value
    path = Path(value)
    if not path.is_absolute():
        if not path.exists():
            return value
        path = path.resolve()
    return os.path.relpath(path.resolve(), output_dir.resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _non_negative_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _add_check(checks: list[dict[str, Any]], check_id: str, passed: bool, actual: dict[str, Any], expected: dict[str, Any]) -> None:
    checks.append(
        {
            "id": check_id,
            "passed": bool(passed),
            "actual": actual,
            "expected": expected,
            "summary": f"{check_id}: passed={bool(passed)}",
        }
    )
