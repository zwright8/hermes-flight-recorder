"""Policy gates over run-suite metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

SUITE_GATE_SCHEMA_VERSION = "hfr.suite_gate.v1"


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
