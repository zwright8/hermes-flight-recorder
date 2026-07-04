"""Side-effect-free governance receipts for closed-loop agentic iterations."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

AGENTIC_LOOP_GOVERNANCE_RECEIPT_SCHEMA_VERSION = "hfr.agentic_loop_governance_receipt.v1"

GOVERNANCE_ACTIONS = ("approve", "reject", "rollback", "request_another_iteration")
_ACTION_RECOMMENDATIONS = {
    "approve": "record_approval_for_promotion_review",
    "reject": "record_rejection",
    "rollback": "record_rollback_request",
    "request_another_iteration": "record_next_iteration_request",
}


class AgenticLoopGovernanceReceiptError(ValueError):
    """Raised when an agentic-loop governance receipt cannot be produced."""


def build_agentic_loop_governance_receipt(
    *,
    ledger_path: str | Path,
    action: str,
    reason: str | None = None,
    requested_by: str | None = None,
    out_path: str | Path | None = None,
    preserve_paths: bool = False,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Record a governance action over the latest loop-ledger decision without applying it."""
    if action not in GOVERNANCE_ACTIONS:
        raise AgenticLoopGovernanceReceiptError(f"action must be one of {sorted(GOVERNANCE_ACTIONS)}")
    output_path = Path(out_path) if out_path is not None else None
    ledger_ref = _source_ledger_ref(Path(ledger_path), preserve_paths, output_path)
    projection = _receipt_projection(ledger_ref, action)
    requested_action = dict(projection["requested_action"])
    requested_action["requested_by"] = requested_by or "unspecified"
    requested_action["reason"] = reason or _default_reason(action, requested_action)
    return {
        "schema_version": AGENTIC_LOOP_GOVERNANCE_RECEIPT_SCHEMA_VERSION,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "receipt_path": _display_path(output_path, preserve_paths=False) if output_path is not None else "",
        "passed": projection["passed"],
        "readiness": projection["readiness"],
        "recommendation": projection["recommendation"],
        "check_count": len(projection["checks"]),
        "failed_check_count": projection["failed_check_count"],
        "checks": projection["checks"],
        "source_ledger": ledger_ref,
        "requested_action": requested_action,
        "decision": projection["decision"],
        "execution_boundary": _execution_boundary(),
        "notes": [
            "This receipt records a governance choice over the latest agentic loop ledger; it does not move aliases, apply rollback, launch cloud jobs, or update weights.",
            "Approval only records readiness for promotion review. Promotion, rollback, alias movement, and release publication remain separate governed receipts.",
        ],
    }


def write_agentic_loop_governance_receipt(path: str | Path, receipt: dict[str, Any]) -> None:
    """Write a deterministic governance receipt artifact."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(json.dumps(receipt))
    source = payload.get("source_ledger")
    if isinstance(source, dict):
        source["path"] = _output_relative_path(source.get("path"), out_path.parent)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _receipt_projection(ledger_ref: dict[str, Any], action: str) -> dict[str, Any]:
    action_row = _ledger_action_row(ledger_ref, action)
    checks = _checks(ledger_ref, action, action_row)
    failed_checks = sum(1 for check in checks if check["passed"] is False)
    passed = failed_checks == 0
    recommendation = _ACTION_RECOMMENDATIONS.get(action, "fix_governance_inputs") if passed else "fix_governance_inputs"
    requested_action = _requested_action(action, action_row)
    decision = _decision(action, requested_action, checks, passed, recommendation, ledger_ref)
    return {
        "passed": passed,
        "readiness": "recorded" if passed else "blocked",
        "recommendation": recommendation,
        "failed_check_count": failed_checks,
        "checks": checks,
        "requested_action": requested_action,
        "decision": decision,
    }


def _source_ledger_ref(path: Path, preserve_paths: bool, output_path: Path | None = None) -> dict[str, Any]:
    payload = _read_json(path)
    decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
    digest = payload.get("readiness_digest") if isinstance(payload.get("readiness_digest"), dict) else {}
    boundary = payload.get("execution_boundary") if isinstance(payload.get("execution_boundary"), dict) else {}
    exists = path.exists() and path.is_file()
    display_path, replayable = _source_display_path(path, output_path, preserve_paths)
    public_exists = exists and replayable
    return {
        "role": "agentic_loop_ledger",
        "path": display_path,
        "kind": "file" if public_exists else "missing",
        "exists": public_exists,
        "sha256": _sha256(path) if public_exists else None,
        "size_bytes": path.stat().st_size if public_exists else None,
        "schema_version": str(payload.get("schema_version") or ""),
        "passed": payload.get("passed") if isinstance(payload.get("passed"), bool) else None,
        "decision": _compact_ledger_decision(decision),
        "readiness_digest": _compact_readiness_digest(digest),
        "execution_boundary": {
            "ledger_only": boundary.get("ledger_only") is True,
            "cloud_jobs_started": boundary.get("cloud_jobs_started") is True,
            "paid_model_grader_calls_started": boundary.get("paid_model_grader_calls_started") is True,
            "live_benchmarks_started": boundary.get("live_benchmarks_started") is True,
            "model_downloads_started": boundary.get("model_downloads_started") is True,
            "weights_updated_by_flight_recorder": boundary.get("weights_updated_by_flight_recorder") is True,
            "credential_values_recorded": boundary.get("credential_values_recorded") is True,
        },
    }


def _compact_ledger_decision(decision: dict[str, Any]) -> dict[str, Any]:
    actions = decision.get("governance_actions") if isinstance(decision.get("governance_actions"), list) else []
    return {
        "readiness": str(decision.get("readiness") or ""),
        "recommendation": str(decision.get("recommendation") or ""),
        "recommended_governance_action": str(decision.get("recommended_governance_action") or ""),
        "latest_iteration_id": str(decision.get("latest_iteration_id") or ""),
        "latest_iteration_index": decision.get("latest_iteration_index"),
        "summary": str(decision.get("summary") or ""),
        "governance_actions": [dict(row) for row in actions if isinstance(row, dict)],
    }


def _compact_readiness_digest(digest: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "latest_iteration_id",
        "latest_iteration_index",
        "readiness",
        "recommendation",
        "ready_for_governance_review",
        "recommended_governance_action",
        "promotion_decision_present",
        "promotion_ledger_present",
        "rollback_receipt_present",
        "side_effects_started",
        "summary",
    )
    compact = {field: digest.get(field) for field in fields if field in digest}
    compact.setdefault("promotion_ledger_present", False)
    return compact


def _ledger_action_row(ledger_ref: dict[str, Any], action: str) -> dict[str, Any] | None:
    decision = ledger_ref.get("decision") if isinstance(ledger_ref.get("decision"), dict) else {}
    actions = decision.get("governance_actions") if isinstance(decision.get("governance_actions"), list) else []
    for row in actions:
        if isinstance(row, dict) and row.get("action") == action:
            return row
    return None


def _requested_action(action: str, action_row: dict[str, Any] | None) -> dict[str, Any]:
    blocked_reasons = (
        [str(item) for item in action_row.get("blocked_reasons", []) if isinstance(item, str)]
        if isinstance(action_row, dict)
        else ["requested_action_not_listed_by_ledger"]
    )
    available = action_row.get("available") is True if isinstance(action_row, dict) else False
    return {
        "action": action,
        "available": available,
        "blocked_reason_count": len(blocked_reasons),
        "blocked_reasons": blocked_reasons,
        "summary": str(action_row.get("summary") or "") if isinstance(action_row, dict) else _missing_action_summary(action),
    }


def _missing_action_summary(action: str) -> str:
    return f"Action {action} is blocked because the source ledger did not list it."


def _checks(ledger_ref: dict[str, Any], action: str, action_row: dict[str, Any] | None) -> list[dict[str, Any]]:
    boundary = ledger_ref.get("execution_boundary") if isinstance(ledger_ref.get("execution_boundary"), dict) else {}
    fail_closed = (
        boundary.get("ledger_only") is True
        and boundary.get("cloud_jobs_started") is False
        and boundary.get("paid_model_grader_calls_started") is False
        and boundary.get("live_benchmarks_started") is False
        and boundary.get("model_downloads_started") is False
        and boundary.get("weights_updated_by_flight_recorder") is False
        and boundary.get("credential_values_recorded") is False
    )
    checks: list[dict[str, Any]] = []
    _add_check(checks, "source_ledger_present", ledger_ref.get("exists") is True, {"exists": ledger_ref.get("exists")}, {"exists": True})
    _add_check(
        checks,
        "source_ledger_schema_version",
        ledger_ref.get("schema_version") == "hfr.agentic_loop_ledger.v1",
        {"schema_version": ledger_ref.get("schema_version")},
        {"schema_version": "hfr.agentic_loop_ledger.v1"},
    )
    _add_check(checks, "source_ledger_passed", ledger_ref.get("passed") is True, {"passed": ledger_ref.get("passed")}, {"passed": True})
    _add_check(
        checks,
        "source_ledger_fail_closed",
        fail_closed,
        boundary,
        {
            "ledger_only": True,
            "cloud_jobs_started": False,
            "paid_model_grader_calls_started": False,
            "live_benchmarks_started": False,
            "model_downloads_started": False,
            "weights_updated_by_flight_recorder": False,
            "credential_values_recorded": False,
        },
    )
    _add_check(
        checks,
        "requested_action_known",
        action in GOVERNANCE_ACTIONS,
        {"action": action},
        {"actions": list(GOVERNANCE_ACTIONS)},
    )
    _add_check(
        checks,
        "requested_action_listed_by_ledger",
        isinstance(action_row, dict),
        {"action": action, "listed": isinstance(action_row, dict)},
        {"listed": True},
    )
    _add_check(
        checks,
        "requested_action_available",
        isinstance(action_row, dict) and action_row.get("available") is True,
        {
            "action": action,
            "available": action_row.get("available") if isinstance(action_row, dict) else False,
            "blocked_reasons": action_row.get("blocked_reasons") if isinstance(action_row, dict) else ["requested_action_not_listed_by_ledger"],
        },
        {"available": True, "blocked_reasons": []},
    )
    _add_check(
        checks,
        "flight_recorder_did_not_apply_governance_side_effects",
        True,
        _execution_boundary(),
        _execution_boundary(),
    )
    return checks


def _decision(
    action: str,
    requested_action: dict[str, Any],
    checks: list[dict[str, Any]],
    passed: bool,
    recommendation: str,
    ledger_ref: dict[str, Any],
) -> dict[str, Any]:
    failed = [check for check in checks if check["passed"] is False]
    ledger_decision = ledger_ref.get("decision") if isinstance(ledger_ref.get("decision"), dict) else {}
    return {
        "readiness": "ready" if passed else "blocked",
        "recommendation": recommendation,
        "summary": _decision_summary(action, requested_action, passed, failed),
        "selected_action": action,
        "ledger_recommended_governance_action": str(ledger_decision.get("recommended_governance_action") or ""),
        "latest_iteration_id": str(ledger_decision.get("latest_iteration_id") or ""),
        "blocking_check_count": len(failed),
        "blocking_checks": [
            {"id": str(check.get("id") or ""), "summary": str(check.get("summary") or ""), "scope": {}}
            for check in failed
        ],
    }


def _decision_summary(action: str, requested_action: dict[str, Any], passed: bool, failed_checks: list[dict[str, Any]]) -> str:
    if passed:
        return f"Governance action {action} recorded without applying side effects."
    if requested_action.get("blocked_reasons"):
        return f"Governance action {action} is blocked: {', '.join(requested_action['blocked_reasons'])}."
    first = failed_checks[0]["summary"] if failed_checks else "unknown blocker"
    return f"Governance action {action} is blocked by receipt input checks; first failure: {first}"


def _default_reason(action: str, requested_action: dict[str, Any]) -> str:
    if requested_action.get("available") is True:
        return f"Record {action} from the latest loop-ledger governance action set."
    return f"Attempted {action}; ledger action blockers are recorded on this receipt."


def _execution_boundary() -> dict[str, bool]:
    return {
        "receipt_only": True,
        "promotion_alias_moved": False,
        "rollback_applied": False,
        "cloud_jobs_started": False,
        "paid_model_grader_calls_started": False,
        "live_benchmarks_started": False,
        "model_downloads_started": False,
        "weights_updated_by_flight_recorder": False,
        "credential_values_recorded": False,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _display_path(path: Path | None, preserve_paths: bool) -> str:
    if path is None:
        return ""
    raw = str(path)
    if preserve_paths:
        return raw if _is_safe_public_path(raw) else f"<redacted:{_basename(raw)}>"
    if _is_windows_absolute(raw):
        return f"<redacted:{_basename(raw)}>"
    if not path.is_absolute():
        return raw if _is_safe_public_path(raw) else f"<redacted:{_basename(raw)}>"
    try:
        relative = str(path.resolve().relative_to(Path.cwd().resolve()))
    except (OSError, ValueError):
        return f"<redacted:{path.name}>"
    return relative if _is_safe_public_path(relative) else f"<redacted:{path.name}>"


def _source_display_path(path: Path, output_path: Path | None, preserve_paths: bool) -> tuple[str, bool]:
    if preserve_paths:
        raw = str(path)
        if _is_safe_public_path(raw):
            return raw, True
        return f"<redacted:{_basename(raw)}>", False
    if output_path is not None and path.exists():
        raw = str(path)
        if _is_windows_absolute(raw):
            return f"<redacted:{_basename(raw)}>", False
        try:
            relative = os.path.relpath(path.resolve(), output_path.parent.resolve())
        except OSError:
            return f"<redacted:{path.name}>", False
        if _is_safe_public_path(relative):
            return relative, True
        return f"<redacted:{path.name}>", False
    displayed = _display_path(path, preserve_paths)
    return displayed, _is_safe_public_path(displayed)


def _output_relative_path(value: Any, output_dir: Path) -> Any:
    if not isinstance(value, str) or not value:
        return value
    if value.startswith("<redacted:"):
        return value
    path = Path(value)
    windows_path = PureWindowsPath(value)
    if windows_path.is_absolute() or windows_path.drive or "\\" in value:
        return f"<redacted:{_basename(value)}>"
    if not path.is_absolute():
        if not path.exists():
            return value
        path = path.resolve()
    try:
        path.resolve().relative_to(Path.cwd().resolve())
    except (OSError, ValueError):
        return f"<redacted:{path.name}>"
    return os.path.relpath(path.resolve(), output_dir.resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_windows_absolute(value: str) -> bool:
    normalized = value.replace("/", "\\")
    return (len(normalized) >= 3 and normalized[1:3] == ":\\" and normalized[0].isalpha()) or normalized.startswith("\\\\")


def _basename(value: str) -> str:
    return value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] or "path"


def _is_safe_public_path(value: str) -> bool:
    if not value or value.startswith("<redacted:"):
        return False
    path = Path(value)
    windows_path = PureWindowsPath(value)
    return (
        not path.is_absolute()
        and not windows_path.is_absolute()
        and not windows_path.drive
        and "\\" not in value
        and "~" not in path.parts
        and ".." not in path.parts
    )


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
