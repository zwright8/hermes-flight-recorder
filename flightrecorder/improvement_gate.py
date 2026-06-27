"""Policy gates over improvement-ledger repair pressure."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .improvement_ledger import IMPROVEMENT_LEDGER_SCHEMA_VERSION

IMPROVEMENT_LEDGER_GATE_SCHEMA_VERSION = "hfr.improvement_ledger_gate.v1"
IMPROVEMENT_LEDGER_GATE_POLICY_SCHEMA_VERSION = "hfr.improvement_ledger_gate.policy.v1"

_COUNT_POLICY_FIELDS = {
    "min_plans",
    "max_open_work_items",
    "max_new_work_items",
    "max_recurring_work_items",
    "min_resolved_work_items",
    "max_critical_open_work_items",
    "max_high_open_work_items",
}
_LIST_POLICY_FIELDS = {
    "forbid_open_priorities",
    "forbid_open_categories",
    "forbid_open_work_keys",
    "require_open_work_keys",
    "require_resolved_work_keys",
}
_POLICY_FIELDS = {"schema_version", "description", *_COUNT_POLICY_FIELDS, *_LIST_POLICY_FIELDS}
_PRIORITIES = {"critical", "high", "medium", "low"}
_CATEGORIES = {"bundle_action", "repair", "curriculum", "digest_action"}


class ImprovementLedgerGatePolicyError(ValueError):
    """Raised when an improvement-ledger gate policy file is malformed."""


class ImprovementLedgerGateError(ValueError):
    """Raised when an improvement ledger cannot be evaluated by a gate."""


def load_improvement_ledger_gate_policy(path: str | Path) -> dict[str, Any]:
    """Load and validate a versioned improvement-ledger gate policy JSON file."""
    policy_path = Path(path)
    try:
        raw = json.loads(policy_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ImprovementLedgerGatePolicyError(f"Invalid JSON in improvement-ledger gate policy {policy_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ImprovementLedgerGatePolicyError(f"Improvement-ledger gate policy must be a JSON object: {policy_path}")
    version = raw.get("schema_version")
    if version != IMPROVEMENT_LEDGER_GATE_POLICY_SCHEMA_VERSION:
        raise ImprovementLedgerGatePolicyError(
            "improvement-ledger gate policy schema_version must be "
            f"{IMPROVEMENT_LEDGER_GATE_POLICY_SCHEMA_VERSION!r}; got {version!r}"
        )
    unknown = sorted(set(raw) - _POLICY_FIELDS)
    if unknown:
        raise ImprovementLedgerGatePolicyError(f"Unknown improvement-ledger gate policy field(s): {', '.join(unknown)}")

    policy: dict[str, Any] = {"schema_version": IMPROVEMENT_LEDGER_GATE_POLICY_SCHEMA_VERSION}
    if "description" in raw:
        if not isinstance(raw["description"], str):
            raise ImprovementLedgerGatePolicyError("improvement-ledger gate policy field description must be a string")
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
                    raise ImprovementLedgerGatePolicyError(
                        f"improvement-ledger gate policy field {field} has invalid priority value(s): "
                        f"{', '.join(unknown_priorities)}"
                    )
            if field == "forbid_open_categories":
                unknown_categories = sorted(set(values) - _CATEGORIES)
                if unknown_categories:
                    raise ImprovementLedgerGatePolicyError(
                        f"improvement-ledger gate policy field {field} has invalid category value(s): "
                        f"{', '.join(unknown_categories)}"
                    )
            policy[field] = values
    return policy


def evaluate_improvement_ledger_gate(
    ledger: dict[str, Any],
    *,
    improvement_ledger_path: str | Path,
    min_plans: int | None = None,
    max_open_work_items: int | None = None,
    max_new_work_items: int | None = None,
    max_recurring_work_items: int | None = None,
    min_resolved_work_items: int | None = None,
    max_critical_open_work_items: int | None = None,
    max_high_open_work_items: int | None = None,
    forbid_open_priorities: list[str] | None = None,
    forbid_open_categories: list[str] | None = None,
    forbid_open_work_keys: list[str] | None = None,
    require_open_work_keys: list[str] | None = None,
    require_resolved_work_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate longitudinal concrete-work checks against improvement_ledger.json."""
    if not isinstance(ledger, dict):
        raise ImprovementLedgerGateError("Improvement-ledger gate input must be a JSON object.")
    if ledger.get("schema_version") != IMPROVEMENT_LEDGER_SCHEMA_VERSION:
        raise ImprovementLedgerGateError(
            "Improvement-ledger gate input schema_version must be "
            f"{IMPROVEMENT_LEDGER_SCHEMA_VERSION!r}; got {ledger.get('schema_version')!r}."
        )
    if not isinstance(ledger.get("metrics"), dict):
        raise ImprovementLedgerGateError("Improvement-ledger gate input metrics must be a JSON object.")
    if not isinstance(ledger.get("entries"), list):
        raise ImprovementLedgerGateError("Improvement-ledger gate input entries must be a JSON array.")

    metrics = ledger["metrics"]
    entries = [entry for entry in ledger["entries"] if isinstance(entry, dict)]
    open_entries = [entry for entry in entries if entry.get("open") is True]
    resolved_entries = [entry for entry in entries if entry.get("status") == "resolved"]
    checks: list[dict[str, Any]] = []

    if min_plans is not None:
        _add_min_check(checks, "min_plans", _int_value(ledger.get("plan_count")), min_plans)
    if max_open_work_items is not None:
        _add_max_check(
            checks,
            "max_open_work_items",
            _int_value(metrics.get("open_work_item_count")),
            max_open_work_items,
        )
    if max_new_work_items is not None:
        _add_max_check(checks, "max_new_work_items", _int_value(metrics.get("new_work_item_count")), max_new_work_items)
    if max_recurring_work_items is not None:
        _add_max_check(
            checks,
            "max_recurring_work_items",
            _int_value(metrics.get("recurring_work_item_count")),
            max_recurring_work_items,
        )
    if min_resolved_work_items is not None:
        _add_min_check(
            checks,
            "min_resolved_work_items",
            _int_value(metrics.get("resolved_work_item_count")),
            min_resolved_work_items,
        )
    if max_critical_open_work_items is not None:
        _add_max_check(
            checks,
            "max_critical_open_work_items",
            _int_value(metrics.get("critical_open_work_item_count")),
            max_critical_open_work_items,
        )
    if max_high_open_work_items is not None:
        _add_max_check(
            checks,
            "max_high_open_work_items",
            _int_value(metrics.get("high_open_work_item_count")),
            max_high_open_work_items,
        )

    open_priority_counts = _entry_counts(open_entries, "priority")
    for priority in forbid_open_priorities or []:
        _add_absence_check(
            checks,
            "forbid_open_priority",
            priority,
            open_priority_counts.get(priority, 0),
            {"priority": priority},
        )

    open_category_counts = _entry_counts(open_entries, "category")
    for category in forbid_open_categories or []:
        _add_absence_check(
            checks,
            "forbid_open_category",
            category,
            open_category_counts.get(category, 0),
            {"category": category},
        )

    open_matchers = _entry_matchers(open_entries)
    for work_key in forbid_open_work_keys or []:
        _add_absence_check(checks, "forbid_open_work_key", work_key, open_matchers.get(work_key, 0), {"work_key": work_key})
    for work_key in require_open_work_keys or []:
        _add_presence_check(checks, "require_open_work_key", work_key, open_matchers.get(work_key, 0), {"work_key": work_key})

    resolved_matchers = _entry_matchers(resolved_entries)
    for work_key in require_resolved_work_keys or []:
        _add_presence_check(
            checks,
            "require_resolved_work_key",
            work_key,
            resolved_matchers.get(work_key, 0),
            {"work_key": work_key},
        )

    passed = all(check["passed"] for check in checks)
    gate_metrics = {
        "plan_count": ledger.get("plan_count"),
        "unique_work_item_count": ledger.get("unique_work_item_count"),
        "open_work_item_count": metrics.get("open_work_item_count"),
        "new_work_item_count": metrics.get("new_work_item_count"),
        "recurring_work_item_count": metrics.get("recurring_work_item_count"),
        "resolved_work_item_count": metrics.get("resolved_work_item_count"),
        "critical_open_work_item_count": metrics.get("critical_open_work_item_count"),
        "high_open_work_item_count": metrics.get("high_open_work_item_count"),
        "open_priority_counts": [{"id": key, "count": open_priority_counts[key]} for key in sorted(open_priority_counts)],
        "open_category_counts": [{"id": key, "count": open_category_counts[key]} for key in sorted(open_category_counts)],
    }
    return {
        "schema_version": IMPROVEMENT_LEDGER_GATE_SCHEMA_VERSION,
        "improvement_ledger": str(improvement_ledger_path),
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
    return {
        "readiness": readiness,
        "recommendation": "promote_iteration" if passed else "block_iteration",
        "summary": _decision_text(readiness, blocking_checks),
        "blocking_check_count": len(blocking_checks),
        "blocking_checks": blocking_checks,
        "key_metrics": _decision_key_metrics(metrics),
    }


def _decision_text(readiness: str, blocking_checks: list[dict[str, Any]]) -> str:
    if readiness == "ready":
        return "Improvement-ledger gate is ready: concrete repair pressure is within policy."
    if not blocking_checks:
        return "Improvement-ledger gate is blocked."
    first = blocking_checks[0]
    return f"Improvement-ledger gate is blocked by {len(blocking_checks)} check(s); first failure: {first['summary']}"


def _decision_key_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "plan_count",
        "unique_work_item_count",
        "open_work_item_count",
        "new_work_item_count",
        "recurring_work_item_count",
        "resolved_work_item_count",
        "critical_open_work_item_count",
        "high_open_work_item_count",
        "open_priority_counts",
        "open_category_counts",
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
        for field_name in ("work_key", "latest_routing_key", "latest_item_id", "latest_fingerprint"):
            value = entry.get(field_name)
            if isinstance(value, str) and value:
                counts[value] = counts.get(value, 0) + 1
    return counts


def _int_value(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _policy_non_negative_int(field: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ImprovementLedgerGatePolicyError(
            f"improvement-ledger gate policy field {field} must be a non-negative integer"
        )
    return value


def _policy_string_list(field: str, value: Any) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ImprovementLedgerGatePolicyError(
            f"improvement-ledger gate policy field {field} must be a list of non-empty strings"
        )
    return list(dict.fromkeys(value))
