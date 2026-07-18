"""Policy gates over run-suite metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .gate_contract import apply_gate_decision_contract

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
_TASK_FAMILY_SCALAR_POLICY_FIELDS = _SCALAR_POLICY_FIELDS - {"max_errors"}
_TASK_FAMILY_GATE_FIELDS = {"task_family", *_TASK_FAMILY_SCALAR_POLICY_FIELDS, *_LIST_POLICY_FIELDS}
_POLICY_FIELDS = {"schema_version", "description", "task_family_gates", *_SCALAR_POLICY_FIELDS, *_LIST_POLICY_FIELDS}


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

    if "task_family_gates" in raw and raw["task_family_gates"] is not None:
        policy["task_family_gates"] = _policy_task_family_gates(raw["task_family_gates"])

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
    task_family_gates: list[dict[str, Any]] | None = None,
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

    family_rows = _task_family_rows(metrics.get("task_families"), suite_summary.get("runs"))
    for gate in task_family_gates or []:
        _evaluate_task_family_gate(checks, gate, family_rows)

    passed = all(check["passed"] for check in checks)
    result = {
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
    return apply_gate_decision_contract(
        result,
        gate_id="suite_gate",
        gate_label="Suite gate",
        key_metric_fields=("pass_rate", "average_score", "failed", "error_count", "critical_failure_total"),
    )


def _evaluate_task_family_gate(checks: list[dict[str, Any]], gate: dict[str, Any], family_rows: dict[str, dict[str, Any]]) -> None:
    family = str(gate["task_family"])
    row = family_rows.get(family)
    scope = {"task_family": family}
    if row is None:
        _add_presence_check(checks, "task_family_present", False, scope)
        return
    _add_presence_check(checks, "task_family_present", True, scope)

    if gate.get("min_pass_rate") is not None:
        _add_min_check(checks, "task_family_min_pass_rate", _number(row.get("pass_rate")), gate["min_pass_rate"], scope)
    if gate.get("min_average_score") is not None:
        _add_min_check(
            checks,
            "task_family_min_average_score",
            _number(row.get("average_score")),
            gate["min_average_score"],
            scope,
        )
    if gate.get("max_failed") is not None:
        _add_max_check(checks, "task_family_max_failed", int(row.get("failed", 0) or 0), gate["max_failed"], scope)
    if gate.get("max_critical_failures") is not None:
        _add_max_check(
            checks,
            "task_family_max_critical_failures",
            _total_count(row.get("critical_failure_counts")),
            gate["max_critical_failures"],
            scope,
        )

    failed_rule_counts = _count_rows(row.get("failed_rule_counts"))
    critical_rule_counts = _count_rows(row.get("critical_failure_counts"))
    for rule_id in gate.get("forbid_failed_rules", []):
        _add_absence_check(
            checks,
            "task_family_forbid_failed_rule",
            rule_id,
            failed_rule_counts.get(rule_id, 0),
            scope,
        )
    for rule_id in gate.get("forbid_critical_rules", []):
        _add_absence_check(
            checks,
            "task_family_forbid_critical_rule",
            rule_id,
            critical_rule_counts.get(rule_id, 0),
            scope,
        )


def _add_min_check(
    checks: list[dict[str, Any]],
    check_id: str,
    actual: float,
    minimum: float,
    scope: dict[str, str] | None = None,
) -> None:
    check = {
        "id": check_id,
        "passed": actual >= minimum,
        "actual": actual,
        "expected": {"min": minimum},
        "summary": f"{check_id}: actual={actual}, min={minimum}",
    }
    _append_check(checks, check, scope)


def _add_max_check(
    checks: list[dict[str, Any]],
    check_id: str,
    actual: int,
    maximum: int,
    scope: dict[str, str] | None = None,
) -> None:
    check = {
        "id": check_id,
        "passed": actual <= maximum,
        "actual": actual,
        "expected": {"max": maximum},
        "summary": f"{check_id}: actual={actual}, max={maximum}",
    }
    _append_check(checks, check, scope)


def _add_absence_check(
    checks: list[dict[str, Any]],
    check_id: str,
    rule_id: str,
    count: int,
    scope: dict[str, str] | None = None,
) -> None:
    check = {
        "id": check_id,
        "passed": count == 0,
        "actual": {"rule_id": rule_id, "count": count},
        "expected": {"count": 0},
        "summary": f"{check_id}: rule_id={rule_id}, count={count}",
    }
    _append_check(checks, check, scope)


def _add_presence_check(checks: list[dict[str, Any]], check_id: str, present: bool, scope: dict[str, str]) -> None:
    check = {
        "id": check_id,
        "passed": present,
        "actual": {"present": present},
        "expected": {"present": True},
        "summary": f"{check_id}: present={present}",
    }
    _append_check(checks, check, scope)


def _append_check(checks: list[dict[str, Any]], check: dict[str, Any], scope: dict[str, str] | None = None) -> None:
    if scope:
        check["scope"] = scope
    checks.append(check)


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


def _policy_task_family_gates(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise SuiteGatePolicyError("suite gate policy field task_family_gates must be a list")
    gates: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise SuiteGatePolicyError(f"suite gate policy field task_family_gates[{index}] must be an object")
        unknown = sorted(set(item) - _TASK_FAMILY_GATE_FIELDS)
        if unknown:
            raise SuiteGatePolicyError(
                f"Unknown suite gate policy field(s) in task_family_gates[{index}]: {', '.join(unknown)}"
            )
        family = item.get("task_family")
        if not isinstance(family, str) or not family.strip():
            raise SuiteGatePolicyError(f"suite gate policy field task_family_gates[{index}].task_family must be a non-empty string")
        gate: dict[str, Any] = {"task_family": family.strip()}
        for field in _TASK_FAMILY_SCALAR_POLICY_FIELDS:
            if field not in item or item[field] is None:
                continue
            if field == "min_pass_rate":
                gate[field] = _policy_float(f"task_family_gates[{index}].{field}", item[field], 0.0, 1.0)
            elif field == "min_average_score":
                gate[field] = _policy_float(f"task_family_gates[{index}].{field}", item[field], 0.0, 100.0)
            else:
                gate[field] = _policy_non_negative_int(f"task_family_gates[{index}].{field}", item[field])
        for field in _LIST_POLICY_FIELDS:
            if field not in item or item[field] is None:
                continue
            gate[field] = _policy_string_list(f"task_family_gates[{index}].{field}", item[field])
        gates.append(gate)
    return gates


def _total_count(value: Any) -> int:
    return sum(_count_rows(value).values())


def _task_family_rows(value: Any, runs: Any) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if isinstance(value, list):
        for row in value:
            if not isinstance(row, dict):
                continue
            family = row.get("task_family")
            if isinstance(family, str) and family:
                rows[family] = dict(row)
    for family, fallback in _task_family_rows_from_runs(runs).items():
        row = rows.setdefault(family, {"task_family": family})
        for field, field_value in fallback.items():
            row.setdefault(field, field_value)
    return rows


def _task_family_rows_from_runs(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        return {}
    buckets: dict[str, list[dict[str, Any]]] = {}
    for run in value:
        if not isinstance(run, dict):
            continue
        family = run.get("task_family")
        if isinstance(family, str) and family:
            buckets.setdefault(family, []).append(run)

    rows: dict[str, dict[str, Any]] = {}
    for family, family_runs in buckets.items():
        scores = [_bounded_score(run.get("score")) for run in family_runs]
        passed = sum(1 for run in family_runs if run.get("passed") is True)
        rows[family] = {
            "task_family": family,
            "total": len(family_runs),
            "passed": passed,
            "failed": len(family_runs) - passed,
            "pass_rate": round(passed / len(family_runs), 4) if family_runs else 0.0,
            "average_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
            "failed_rule_counts": _count_values(rule for run in family_runs for rule in run.get("failed_rules", [])),
            "critical_failure_counts": _count_values(rule for run in family_runs for rule in run.get("critical_failures", [])),
        }
    return rows


def _count_values(values: Any) -> list[dict[str, int]]:
    counts: dict[str, int] = {}
    for value in values:
        if isinstance(value, str) and value:
            counts[value] = counts.get(value, 0) + 1
    return [{"id": key, "count": count} for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


def _bounded_score(value: Any) -> int:
    try:
        return max(0, min(100, int(value)))
    except (TypeError, ValueError):
        return 0


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
