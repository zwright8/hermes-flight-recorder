"""Policy gates over run-suite metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SUITE_GATE_SCHEMA_VERSION = "hfr.suite_gate.v1"
SUITE_GATE_POLICY_SCHEMA_VERSION = "hfr.suite_gate.policy.v1"

_SCALAR_POLICY_FIELDS = {
    "min_pass_rate",
    "min_average_score",
    "max_failed",
    "max_errors",
    "max_critical_failures",
}
_LIST_POLICY_FIELDS = {"forbid_failed_rules", "forbid_critical_rules"}
_POLICY_FIELDS = {"schema_version", "description", *_SCALAR_POLICY_FIELDS, *_LIST_POLICY_FIELDS}


class SuiteGatePolicyError(ValueError):
    """Raised when a suite gate policy file is malformed."""


def load_gate_policy(path: str | Path) -> dict[str, Any]:
    """Load and validate a versioned suite gate policy JSON file."""
    policy_path = Path(path)
    try:
        raw = json.loads(policy_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SuiteGatePolicyError(f"Invalid JSON in suite gate policy {policy_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise SuiteGatePolicyError(f"Suite gate policy must be a JSON object: {policy_path}")

    version = raw.get("schema_version")
    if version != SUITE_GATE_POLICY_SCHEMA_VERSION:
        raise SuiteGatePolicyError(
            f"suite gate policy schema_version must be {SUITE_GATE_POLICY_SCHEMA_VERSION!r}; got {version!r}"
        )

    unknown = sorted(set(raw) - _POLICY_FIELDS)
    if unknown:
        raise SuiteGatePolicyError(f"Unknown suite gate policy field(s): {', '.join(unknown)}")

    policy: dict[str, Any] = {"schema_version": SUITE_GATE_POLICY_SCHEMA_VERSION}
    if "description" in raw:
        if not isinstance(raw["description"], str):
            raise SuiteGatePolicyError("suite gate policy field description must be a string")
        policy["description"] = raw["description"]

    for field in _SCALAR_POLICY_FIELDS:
        if field not in raw or raw[field] is None:
            continue
        if field == "min_pass_rate":
            policy[field] = _policy_float(field, raw[field], 0.0, 1.0)
        elif field == "min_average_score":
            policy[field] = _policy_float(field, raw[field], 0.0, 100.0)
        else:
            policy[field] = _policy_non_negative_int(field, raw[field])

    for field in _LIST_POLICY_FIELDS:
        if field not in raw or raw[field] is None:
            continue
        policy[field] = _policy_string_list(field, raw[field])

    return policy


def evaluate_suite_gate(
    suite_summary: dict[str, Any],
    *,
    suite_summary_path: str | Path,
    min_pass_rate: float | None = None,
    min_average_score: float | None = None,
    max_failed: int | None = None,
    max_errors: int | None = 0,
    max_critical_failures: int | None = None,
    forbid_failed_rules: list[str] | None = None,
    forbid_critical_rules: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate threshold checks against a run-suite summary."""
    metrics = suite_summary.get("metrics") if isinstance(suite_summary.get("metrics"), dict) else {}
    checks: list[dict[str, Any]] = []

    if max_errors is not None:
        _add_max_check(checks, "max_errors", int(suite_summary.get("error_count", 0) or 0), max_errors)
    if min_pass_rate is not None:
        _add_min_check(checks, "min_pass_rate", _number(metrics.get("pass_rate")), min_pass_rate)
    if min_average_score is not None:
        _add_min_check(checks, "min_average_score", _number(metrics.get("average_score")), min_average_score)
    if max_failed is not None:
        _add_max_check(checks, "max_failed", int(metrics.get("failed", suite_summary.get("failed", 0)) or 0), max_failed)
    if max_critical_failures is not None:
        _add_max_check(checks, "max_critical_failures", _total_count(metrics.get("critical_failure_counts")), max_critical_failures)

    failed_rule_counts = _count_rows(metrics.get("failed_rule_counts"))
    critical_rule_counts = _count_rows(metrics.get("critical_failure_counts"))
    for rule_id in forbid_failed_rules or []:
        _add_absence_check(checks, "forbid_failed_rule", rule_id, failed_rule_counts.get(rule_id, 0))
    for rule_id in forbid_critical_rules or []:
        _add_absence_check(checks, "forbid_critical_rule", rule_id, critical_rule_counts.get(rule_id, 0))

    passed = all(check["passed"] for check in checks)
    return {
        "schema_version": SUITE_GATE_SCHEMA_VERSION,
        "suite_summary": str(suite_summary_path),
        "passed": passed,
        "check_count": len(checks),
        "failed_check_count": sum(1 for check in checks if not check["passed"]),
        "checks": checks,
        "metrics": {
            "pass_rate": metrics.get("pass_rate"),
            "average_score": metrics.get("average_score"),
            "failed": metrics.get("failed", suite_summary.get("failed")),
            "error_count": suite_summary.get("error_count"),
            "critical_failure_total": _total_count(metrics.get("critical_failure_counts")),
        },
    }


def _add_min_check(checks: list[dict[str, Any]], check_id: str, actual: float, minimum: float) -> None:
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


def _add_absence_check(checks: list[dict[str, Any]], check_id: str, rule_id: str, count: int) -> None:
    checks.append(
        {
            "id": check_id,
            "passed": count == 0,
            "actual": {"rule_id": rule_id, "count": count},
            "expected": {"count": 0},
            "summary": f"{check_id}: rule_id={rule_id}, count={count}",
        }
    )


def _number(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _policy_float(field: str, value: Any, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SuiteGatePolicyError(f"suite gate policy field {field} must be a number")
    parsed = float(value)
    if not minimum <= parsed <= maximum:
        raise SuiteGatePolicyError(f"suite gate policy field {field} must be from {minimum} to {maximum}")
    return parsed


def _policy_non_negative_int(field: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SuiteGatePolicyError(f"suite gate policy field {field} must be a non-negative integer")
    if value < 0:
        raise SuiteGatePolicyError(f"suite gate policy field {field} must be a non-negative integer")
    return value


def _policy_string_list(field: str, value: Any) -> list[str]:
    if not isinstance(value, list):
        raise SuiteGatePolicyError(f"suite gate policy field {field} must be a list of strings")
    normalized: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise SuiteGatePolicyError(f"suite gate policy field {field}[{index}] must be a non-empty string")
        normalized.append(item.strip())
    return normalized


def _total_count(value: Any) -> int:
    return sum(_count_rows(value).values())


def _count_rows(value: Any) -> dict[str, int]:
    if not isinstance(value, list):
        return {}
    counts: dict[str, int] = {}
    for row in value:
        if not isinstance(row, dict):
            continue
        row_id = row.get("id")
        count = row.get("count")
        if isinstance(row_id, str) and isinstance(count, int) and not isinstance(count, bool):
            counts[row_id] = count
    return counts
