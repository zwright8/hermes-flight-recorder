"""Policy gates over promotion-ledger history."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .promotion_ledger import PROMOTION_LEDGER_SCHEMA_VERSION

PROMOTION_LEDGER_GATE_SCHEMA_VERSION = "hfr.promotion_ledger_gate.v1"
PROMOTION_LEDGER_GATE_POLICY_SCHEMA_VERSION = "hfr.promotion_ledger_gate.policy.v1"

_COUNT_POLICY_FIELDS = {
    "min_decisions",
    "min_allowed_count",
    "max_blocked_count",
    "min_consecutive_allowed",
    "max_consecutive_blocked",
    "max_failed_decisions",
}
_RATE_POLICY_FIELDS = {"max_blocked_rate"}
_STRING_POLICY_FIELDS = {"require_latest_recommendation"}
_BOOL_POLICY_FIELDS = {"require_latest_passed"}
_LIST_POLICY_FIELDS = {"require_source_recommendations", "forbid_source_recommendations"}
_POLICY_FIELDS = {
    "schema_version",
    "description",
    *_COUNT_POLICY_FIELDS,
    *_RATE_POLICY_FIELDS,
    *_STRING_POLICY_FIELDS,
    *_BOOL_POLICY_FIELDS,
    *_LIST_POLICY_FIELDS,
}
_RECOMMENDATIONS = {"allow_promotion", "block_promotion"}


class PromotionLedgerGatePolicyError(ValueError):
    """Raised when a promotion-ledger gate policy file is malformed."""


class PromotionLedgerGateError(ValueError):
    """Raised when a promotion ledger cannot be evaluated by a gate."""


def load_promotion_ledger_gate_policy(path: str | Path) -> dict[str, Any]:
    """Load and validate a versioned promotion-ledger gate policy JSON file."""
    policy_path = Path(path)
    try:
        raw = json.loads(policy_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PromotionLedgerGatePolicyError(f"Invalid JSON in promotion-ledger gate policy {policy_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise PromotionLedgerGatePolicyError(f"Promotion-ledger gate policy must be a JSON object: {policy_path}")

    version = raw.get("schema_version")
    if version != PROMOTION_LEDGER_GATE_POLICY_SCHEMA_VERSION:
        raise PromotionLedgerGatePolicyError(
            f"promotion-ledger gate policy schema_version must be {PROMOTION_LEDGER_GATE_POLICY_SCHEMA_VERSION!r}; got {version!r}"
        )
    unknown = sorted(set(raw) - _POLICY_FIELDS)
    if unknown:
        raise PromotionLedgerGatePolicyError(f"Unknown promotion-ledger gate policy field(s): {', '.join(unknown)}")

    policy: dict[str, Any] = {"schema_version": PROMOTION_LEDGER_GATE_POLICY_SCHEMA_VERSION}
    if "description" in raw:
        if not isinstance(raw["description"], str):
            raise PromotionLedgerGatePolicyError("promotion-ledger gate policy field description must be a string")
        policy["description"] = raw["description"]
    for field in _COUNT_POLICY_FIELDS:
        if field in raw and raw[field] is not None:
            policy[field] = _policy_non_negative_int(field, raw[field])
    for field in _RATE_POLICY_FIELDS:
        if field in raw and raw[field] is not None:
            policy[field] = _policy_rate(field, raw[field])
    for field in _STRING_POLICY_FIELDS:
        if field in raw and raw[field] is not None:
            value = raw[field]
            if not isinstance(value, str) or not value:
                raise PromotionLedgerGatePolicyError(f"promotion-ledger gate policy field {field} must be a non-empty string")
            if field == "require_latest_recommendation" and value not in _RECOMMENDATIONS:
                raise PromotionLedgerGatePolicyError(
                    f"promotion-ledger gate policy field {field} must be one of {sorted(_RECOMMENDATIONS)}"
                )
            policy[field] = value
    for field in _BOOL_POLICY_FIELDS:
        if field in raw and raw[field] is not None:
            if not isinstance(raw[field], bool):
                raise PromotionLedgerGatePolicyError(f"promotion-ledger gate policy field {field} must be a boolean")
            policy[field] = raw[field]
    for field in _LIST_POLICY_FIELDS:
        if field in raw and raw[field] is not None:
            policy[field] = _policy_string_list(field, raw[field])
    return policy


def evaluate_promotion_ledger_gate(
    ledger: dict[str, Any],
    *,
    promotion_ledger_path: str | Path,
    min_decisions: int | None = None,
    min_allowed_count: int | None = None,
    max_blocked_count: int | None = None,
    max_blocked_rate: float | None = None,
    min_consecutive_allowed: int | None = None,
    max_consecutive_blocked: int | None = None,
    max_failed_decisions: int | None = None,
    require_latest_recommendation: str | None = None,
    require_latest_passed: bool = False,
    require_source_recommendations: list[str] | None = None,
    forbid_source_recommendations: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate longitudinal promotion checks against promotion_ledger.json."""
    if not isinstance(ledger, dict):
        raise PromotionLedgerGateError("Promotion-ledger gate input must be a JSON object.")
    if ledger.get("schema_version") != PROMOTION_LEDGER_SCHEMA_VERSION:
        raise PromotionLedgerGateError(
            f"Promotion-ledger gate input schema_version must be {PROMOTION_LEDGER_SCHEMA_VERSION!r}; got {ledger.get('schema_version')!r}."
        )
    if not isinstance(ledger.get("metrics"), dict):
        raise PromotionLedgerGateError("Promotion-ledger gate input metrics must be a JSON object.")
    if not isinstance(ledger.get("records"), list):
        raise PromotionLedgerGateError("Promotion-ledger gate input records must be a JSON array.")

    metrics = ledger["metrics"]
    records = [record for record in ledger["records"] if isinstance(record, dict)]
    failed_decision_count = sum(1 for record in records if _int_value(record.get("failed_check_count")) > 0)
    decision_count = _int_value(metrics.get("decision_count"))
    blocked_count = _int_value(metrics.get("blocked_count"))
    blocked_rate = round(blocked_count / decision_count, 4) if decision_count else 0.0
    source_recommendation_counts = _count_rows(metrics.get("source_recommendation_counts"))
    checks: list[dict[str, Any]] = []

    if min_decisions is not None:
        _add_min_check(checks, "min_decisions", decision_count, min_decisions)
    if min_allowed_count is not None:
        _add_min_check(checks, "min_allowed_count", _int_value(metrics.get("allowed_count")), min_allowed_count)
    if max_blocked_count is not None:
        _add_max_check(checks, "max_blocked_count", blocked_count, max_blocked_count)
    if max_blocked_rate is not None:
        _add_max_check(checks, "max_blocked_rate", blocked_rate, max_blocked_rate)
    if min_consecutive_allowed is not None:
        _add_min_check(checks, "min_consecutive_allowed", _int_value(metrics.get("consecutive_allowed_count")), min_consecutive_allowed)
    if max_consecutive_blocked is not None:
        _add_max_check(checks, "max_consecutive_blocked", _int_value(metrics.get("consecutive_blocked_count")), max_consecutive_blocked)
    if max_failed_decisions is not None:
        _add_max_check(checks, "max_failed_decisions", failed_decision_count, max_failed_decisions)
    if require_latest_recommendation is not None:
        _add_equal_check(
            checks,
            "require_latest_recommendation",
            str(metrics.get("latest_recommendation") or ""),
            require_latest_recommendation,
        )
    if require_latest_passed:
        _add_equal_check(checks, "require_latest_passed", metrics.get("latest_passed"), True)
    for recommendation in require_source_recommendations or []:
        _add_presence_check(
            checks,
            "require_source_recommendation",
            recommendation,
            source_recommendation_counts.get(recommendation, 0),
            {"recommendation": recommendation},
        )
    for recommendation in forbid_source_recommendations or []:
        _add_absence_check(
            checks,
            "forbid_source_recommendation",
            recommendation,
            source_recommendation_counts.get(recommendation, 0),
            {"recommendation": recommendation},
        )

    gate_metrics = {
        "decision_count": decision_count,
        "allowed_count": _int_value(metrics.get("allowed_count")),
        "blocked_count": blocked_count,
        "blocked_rate": blocked_rate,
        "latest_recommendation": metrics.get("latest_recommendation"),
        "latest_readiness": metrics.get("latest_readiness"),
        "latest_passed": metrics.get("latest_passed"),
        "consecutive_allowed_count": _int_value(metrics.get("consecutive_allowed_count")),
        "consecutive_blocked_count": _int_value(metrics.get("consecutive_blocked_count")),
        "failed_decision_count": failed_decision_count,
        "unique_source_artifact_count": _int_value(metrics.get("unique_source_artifact_count")),
        "source_recommendation_counts": [
            {"id": key, "count": source_recommendation_counts[key]}
            for key in sorted(source_recommendation_counts)
        ],
    }
    passed = all(check["passed"] for check in checks)
    return {
        "schema_version": PROMOTION_LEDGER_GATE_SCHEMA_VERSION,
        "promotion_ledger": str(promotion_ledger_path),
        "passed": passed,
        "decision": _decision_summary(passed, checks, gate_metrics),
        "check_count": len(checks),
        "failed_check_count": sum(1 for check in checks if not check["passed"]),
        "checks": checks,
        "metrics": gate_metrics,
    }


def _add_min_check(checks: list[dict[str, Any]], check_id: str, actual: int | float, minimum: int | float) -> None:
    checks.append(
        {
            "id": check_id,
            "passed": actual >= minimum,
            "actual": actual,
            "expected": {"min": minimum},
            "summary": f"{check_id}: actual={actual}, min={minimum}",
        }
    )


def _add_max_check(checks: list[dict[str, Any]], check_id: str, actual: int | float, maximum: int | float) -> None:
    checks.append(
        {
            "id": check_id,
            "passed": actual <= maximum,
            "actual": actual,
            "expected": {"max": maximum},
            "summary": f"{check_id}: actual={actual}, max={maximum}",
        }
    )


def _add_equal_check(checks: list[dict[str, Any]], check_id: str, actual: Any, expected: Any) -> None:
    checks.append(
        {
            "id": check_id,
            "passed": actual == expected,
            "actual": actual,
            "expected": {"value": expected},
            "summary": f"{check_id}: actual={actual!r}, expected={expected!r}",
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
        return "Promotion-ledger gate is ready: promotion history is within policy."
    if not blocking_checks:
        return "Promotion-ledger gate is blocked."
    first = blocking_checks[0]
    return f"Promotion-ledger gate is blocked by {len(blocking_checks)} check(s); first failure: {first['summary']}"


def _decision_key_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "decision_count",
        "allowed_count",
        "blocked_count",
        "blocked_rate",
        "latest_recommendation",
        "latest_passed",
        "consecutive_allowed_count",
        "consecutive_blocked_count",
        "failed_decision_count",
        "source_recommendation_counts",
    )
    return {field: metrics.get(field) for field in fields}


def _count_rows(value: Any) -> dict[str, int]:
    if not isinstance(value, list):
        return {}
    counts: dict[str, int] = {}
    for row in value:
        if not isinstance(row, dict):
            continue
        row_id = row.get("id")
        count = row.get("count")
        if isinstance(row_id, str) and row_id and isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            counts[row_id] = count
    return counts


def _int_value(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _policy_non_negative_int(field: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise PromotionLedgerGatePolicyError(f"promotion-ledger gate policy field {field} must be a non-negative integer")
    return value


def _policy_rate(field: str, value: Any) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise PromotionLedgerGatePolicyError(f"promotion-ledger gate policy field {field} must be a number")
    result = float(value)
    if result < 0.0 or result > 1.0:
        raise PromotionLedgerGatePolicyError(f"promotion-ledger gate policy field {field} must be from 0.0 to 1.0")
    return result


def _policy_string_list(field: str, value: Any) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise PromotionLedgerGatePolicyError(f"promotion-ledger gate policy field {field} must be a list of non-empty strings")
    return list(dict.fromkeys(value))
