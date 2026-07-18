"""Policy gates over action-ledger repair pressure."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .action_ledger import ACTION_LEDGER_SCHEMA_VERSION

ACTION_LEDGER_GATE_SCHEMA_VERSION = "hfr.action_ledger_gate.v1"
ACTION_LEDGER_GATE_POLICY_SCHEMA_VERSION = "hfr.action_ledger_gate.policy.v1"

_COUNT_POLICY_FIELDS = {
    "min_bundles",
    "max_open_actions",
    "max_new_actions",
    "max_recurring_actions",
    "min_resolved_actions",
}
_LIST_POLICY_FIELDS = {
    "forbid_open_priorities",
    "forbid_open_actions",
    "require_resolved_actions",
}
_POLICY_FIELDS = {"schema_version", "description", *_COUNT_POLICY_FIELDS, *_LIST_POLICY_FIELDS}
_PRIORITIES = {"critical", "high", "medium", "low"}


class ActionLedgerGatePolicyError(ValueError):
    """Raised when an action-ledger gate policy file is malformed."""


class ActionLedgerGateError(ValueError):
    """Raised when an action ledger cannot be evaluated by a gate."""


def load_action_ledger_gate_policy(path: str | Path) -> dict[str, Any]:
    """Load and validate a versioned action-ledger gate policy JSON file."""
    policy_path = Path(path)
    try:
        raw = json.loads(policy_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ActionLedgerGatePolicyError(f"Invalid JSON in action-ledger gate policy {policy_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ActionLedgerGatePolicyError(f"Action-ledger gate policy must be a JSON object: {policy_path}")
    version = raw.get("schema_version")
    if version != ACTION_LEDGER_GATE_POLICY_SCHEMA_VERSION:
        raise ActionLedgerGatePolicyError(
            f"action-ledger gate policy schema_version must be {ACTION_LEDGER_GATE_POLICY_SCHEMA_VERSION!r}; got {version!r}"
        )
    unknown = sorted(set(raw) - _POLICY_FIELDS)
    if unknown:
        raise ActionLedgerGatePolicyError(f"Unknown action-ledger gate policy field(s): {', '.join(unknown)}")

    policy: dict[str, Any] = {"schema_version": ACTION_LEDGER_GATE_POLICY_SCHEMA_VERSION}
    if "description" in raw:
        if not isinstance(raw["description"], str):
            raise ActionLedgerGatePolicyError("action-ledger gate policy field description must be a string")
        policy["description"] = raw["description"]
    for field in _COUNT_POLICY_FIELDS:
        if field in raw and raw[field] is not None:
            policy[field] = _policy_non_negative_int(field, raw[field])
    for field in _LIST_POLICY_FIELDS:
        if field in raw and raw[field] is not None:
            values = _policy_string_list(field, raw[field])
            if field == "forbid_open_priorities":
                unknown_priorities = sorted(set(values) - _PRIORITIES)
                if unknown_priorities:
                    raise ActionLedgerGatePolicyError(
                        f"action-ledger gate policy field {field} has invalid priority value(s): {', '.join(unknown_priorities)}"
                    )
            policy[field] = values
    return policy


def evaluate_action_ledger_gate(
    ledger: dict[str, Any],
    *,
    action_ledger_path: str | Path,
    min_bundles: int | None = None,
    max_open_actions: int | None = None,
    max_new_actions: int | None = None,
    max_recurring_actions: int | None = None,
    min_resolved_actions: int | None = None,
    forbid_open_priorities: list[str] | None = None,
    forbid_open_actions: list[str] | None = None,
    require_resolved_actions: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate longitudinal repair-pressure checks against action_ledger.json."""
    if not isinstance(ledger, dict):
        raise ActionLedgerGateError("Action-ledger gate input must be a JSON object.")
    if ledger.get("schema_version") != ACTION_LEDGER_SCHEMA_VERSION:
        raise ActionLedgerGateError(
            f"Action-ledger gate input schema_version must be {ACTION_LEDGER_SCHEMA_VERSION!r}; got {ledger.get('schema_version')!r}."
        )
    if not isinstance(ledger.get("metrics"), dict):
        raise ActionLedgerGateError("Action-ledger gate input metrics must be a JSON object.")
    if not isinstance(ledger.get("entries"), list):
        raise ActionLedgerGateError("Action-ledger gate input entries must be a JSON array.")

    metrics = ledger["metrics"]
    entries = [entry for entry in ledger["entries"] if isinstance(entry, dict)]
    open_entries = [entry for entry in entries if entry.get("open") is True]
    resolved_entries = [entry for entry in entries if entry.get("status") == "resolved"]
    checks: list[dict[str, Any]] = []

    if min_bundles is not None:
        _add_min_check(checks, "min_bundles", _int_value(ledger.get("bundle_count")), min_bundles)
    if max_open_actions is not None:
        _add_max_check(checks, "max_open_actions", _int_value(metrics.get("open_action_count")), max_open_actions)
    if max_new_actions is not None:
        _add_max_check(checks, "max_new_actions", _int_value(metrics.get("new_action_count")), max_new_actions)
    if max_recurring_actions is not None:
        _add_max_check(checks, "max_recurring_actions", _int_value(metrics.get("recurring_action_count")), max_recurring_actions)
    if min_resolved_actions is not None:
        _add_min_check(checks, "min_resolved_actions", _int_value(metrics.get("resolved_action_count")), min_resolved_actions)

    open_priority_counts = _entry_counts(open_entries, "priority")
    for priority in forbid_open_priorities or []:
        _add_absence_check(checks, "forbid_open_priority", priority, open_priority_counts.get(priority, 0), {"priority": priority})

    open_matchers = _entry_matchers(open_entries)
    for action in forbid_open_actions or []:
        _add_absence_check(checks, "forbid_open_action", action, open_matchers.get(action, 0), {"action": action})

    resolved_matchers = _entry_matchers(resolved_entries)
    for action in require_resolved_actions or []:
        _add_presence_check(checks, "require_resolved_action", action, resolved_matchers.get(action, 0), {"action": action})

    passed = all(check["passed"] for check in checks)
    gate_metrics = {
        "bundle_count": ledger.get("bundle_count"),
        "unique_action_count": ledger.get("unique_action_count"),
        "open_action_count": metrics.get("open_action_count"),
        "new_action_count": metrics.get("new_action_count"),
        "recurring_action_count": metrics.get("recurring_action_count"),
        "resolved_action_count": metrics.get("resolved_action_count"),
        "open_priority_counts": [{"id": key, "count": open_priority_counts[key]} for key in sorted(open_priority_counts)],
    }
    return {
        "schema_version": ACTION_LEDGER_GATE_SCHEMA_VERSION,
        "action_ledger": str(action_ledger_path),
        "passed": passed,
        "decision": _decision_summary(passed, checks, gate_metrics),
        "check_count": len(checks),
        "failed_check_count": sum(1 for check in checks if not check["passed"]),
        "checks": checks,
        "metrics": gate_metrics,
    }


def _add_min_check(checks: list[dict[str, Any]], check_id: str, actual: int, minimum: int) -> None:
    checks.append(
        {
            "id": check_id,
            "passed": actual >= minimum,
            "actual": actual,
            "expected": {"min": minimum},
            "summary": f"{check_id}: actual={actual}, min={minimum}",
        }
    )


def _add_max_check(checks: list[dict[str, Any]], check_id: str, actual: int, maximum: int) -> None:
    checks.append(
        {
            "id": check_id,
            "passed": actual <= maximum,
            "actual": actual,
            "expected": {"max": maximum},
            "summary": f"{check_id}: actual={actual}, max={maximum}",
        }
    )


def _add_absence_check(checks: list[dict[str, Any]], check_id: str, item: str, count: int, scope: dict[str, str]) -> None:
    checks.append(
        {
            "id": check_id,
            "passed": count == 0,
            "actual": {"id": item, "count": count},
            "expected": {"count": 0},
            "scope": scope,
            "summary": f"{check_id}: id={item}, count={count}",
        }
    )


def _add_presence_check(checks: list[dict[str, Any]], check_id: str, item: str, count: int, scope: dict[str, str]) -> None:
    checks.append(
        {
            "id": check_id,
            "passed": count > 0,
            "actual": {"id": item, "count": count},
            "expected": {"min": 1},
            "scope": scope,
            "summary": f"{check_id}: id={item}, count={count}",
        }
    )


def _decision_summary(passed: bool, checks: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    blocking_checks = [
        {
            "id": str(check.get("id") or "unknown"),
            "summary": str(check.get("summary") or ""),
            "scope": check.get("scope") if isinstance(check.get("scope"), dict) else {},
        }
        for check in checks
        if check.get("passed") is False
    ]
    readiness = "ready" if passed else "blocked"
    next_actions = _decision_next_actions("action_ledger_gate", "Action-ledger gate", blocking_checks)
    return {
        "readiness": readiness,
        "recommendation": "promote_iteration" if passed else "block_iteration",
        "summary": _decision_text(readiness, blocking_checks),
        "blocking_check_count": len(blocking_checks),
        "blocking_checks": blocking_checks,
        "failed_checks": blocking_checks,
        "next_action_count": len(next_actions),
        "next_actions": next_actions,
        "key_metrics": _decision_key_metrics(metrics),
    }


def _decision_next_actions(artifact: str, gate_label: str, blocking_checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not blocking_checks:
        return []
    return [
        {
            "id": "resolve_failed_checks",
            "priority": "critical",
            "artifact": artifact,
            "summary": f"Resolve {len(blocking_checks)} failed {gate_label} check(s) before using this gate downstream.",
            "evidence": {
                "failed_check_count": len(blocking_checks),
                "failed_checks": blocking_checks,
            },
        }
    ]


def _decision_text(readiness: str, blocking_checks: list[dict[str, Any]]) -> str:
    if readiness == "ready":
        return "Action-ledger gate is ready: improvement-loop pressure is within policy."
    if not blocking_checks:
        return "Action-ledger gate is blocked."
    first = blocking_checks[0]
    return f"Action-ledger gate is blocked by {len(blocking_checks)} check(s); first failure: {first['summary']}"


def _decision_key_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "bundle_count",
        "unique_action_count",
        "open_action_count",
        "new_action_count",
        "recurring_action_count",
        "resolved_action_count",
        "open_priority_counts",
    )
    return {field: metrics.get(field) for field in fields if field in metrics}


def _entry_counts(entries: list[dict[str, Any]], field_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        value = entry.get(field_name)
        if isinstance(value, str) and value:
            counts[value] = counts.get(value, 0) + 1
    return counts


def _entry_matchers(entries: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        for field_name in ("routing_key", "id", "action_fingerprint"):
            value = entry.get(field_name)
            if isinstance(value, str) and value:
                counts[value] = counts.get(value, 0) + 1
    return counts


def _int_value(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _policy_non_negative_int(field: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ActionLedgerGatePolicyError(f"action-ledger gate policy field {field} must be a non-negative integer")
    return value


def _policy_string_list(field: str, value: Any) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ActionLedgerGatePolicyError(f"action-ledger gate policy field {field} must be a list of non-empty strings")
    return list(dict.fromkeys(value))
