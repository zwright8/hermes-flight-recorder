"""Shared machine-readable decision contract for gate artifacts."""

from __future__ import annotations

from typing import Any


READY_RECOMMENDATION = "promote_iteration"
BLOCK_RECOMMENDATION = "block_iteration"


def build_gate_decision(
    *,
    gate_id: str,
    gate_label: str,
    passed: bool,
    checks: list[dict[str, Any]],
    metrics: dict[str, Any],
    key_metric_fields: tuple[str, ...],
) -> dict[str, Any]:
    """Build the common gate decision block used by evidence handoffs."""
    failed_checks = _failed_checks(checks)
    readiness = "ready" if passed else "blocked"
    next_actions = _next_actions(gate_id, gate_label, failed_checks)
    return {
        "readiness": readiness,
        "recommendation": READY_RECOMMENDATION if passed else BLOCK_RECOMMENDATION,
        "summary": _decision_text(gate_label, readiness, failed_checks),
        "blocking_check_count": len(failed_checks),
        "blocking_checks": failed_checks,
        "failed_checks": failed_checks,
        "next_action_count": len(next_actions),
        "next_actions": next_actions,
        "key_metrics": {field: metrics.get(field) for field in key_metric_fields if field in metrics},
    }


def apply_gate_decision_contract(
    result: dict[str, Any],
    *,
    gate_id: str,
    gate_label: str,
    key_metric_fields: tuple[str, ...],
) -> dict[str, Any]:
    """Attach common decision fields to an existing gate result dictionary."""
    passed = result.get("passed") is True
    checks = result.get("checks") if isinstance(result.get("checks"), list) else []
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    decision = build_gate_decision(
        gate_id=gate_id,
        gate_label=gate_label,
        passed=passed,
        checks=checks,
        metrics=metrics,
        key_metric_fields=key_metric_fields,
    )
    result["readiness"] = decision["readiness"]
    result["recommendation"] = decision["recommendation"]
    result["failed_checks"] = decision["failed_checks"]
    result["next_action_count"] = decision["next_action_count"]
    result["next_actions"] = decision["next_actions"]
    result["decision"] = decision
    return result


def summarize_gate_contract(gate: dict[str, Any]) -> dict[str, Any]:
    """Return validation-oriented metadata for an arbitrary gate artifact."""
    errors: list[str] = []
    passed = gate.get("passed")
    checks = gate.get("checks")
    failed_count = _failed_check_count(checks)
    decision = gate.get("decision")
    if not isinstance(decision, dict):
        errors.append("decision must be an object")
        decision = {}

    expected_readiness = "ready" if passed is True else "blocked" if passed is False else None
    expected_recommendation = READY_RECOMMENDATION if passed is True else BLOCK_RECOMMENDATION if passed is False else None

    if passed not in {True, False}:
        errors.append("passed must be a boolean")
    if not isinstance(checks, list):
        errors.append("checks must be a list")
    if failed_count is None:
        errors.append("checks must contain boolean passed fields")
    elif passed is True and failed_count != 0:
        errors.append("passed must be false when checks fail")
    elif passed is False and failed_count == 0:
        errors.append("passed must be true when no checks fail")

    readiness = decision.get("readiness")
    recommendation = decision.get("recommendation")
    blocking_checks = decision.get("blocking_checks")
    failed_checks = decision.get("failed_checks")
    next_actions = decision.get("next_actions")

    if expected_readiness is not None and readiness != expected_readiness:
        errors.append(f"decision.readiness must be {expected_readiness}")
    if expected_recommendation is not None and recommendation != expected_recommendation:
        errors.append(f"decision.recommendation must be {expected_recommendation}")
    if failed_count not in (None, 0) and readiness == "ready":
        errors.append("decision.readiness must be blocked when checks fail")
    if failed_count == 0 and readiness == "blocked":
        errors.append("decision.readiness must be ready when no checks fail")
    if failed_count not in (None, 0) and recommendation == READY_RECOMMENDATION:
        errors.append("decision.recommendation must block when checks fail")
    if failed_count == 0 and recommendation == BLOCK_RECOMMENDATION:
        errors.append("decision.recommendation must promote when no checks fail")
    if not isinstance(decision.get("summary"), str) or not decision.get("summary"):
        errors.append("decision.summary must be a non-empty string")
    if failed_count is not None and decision.get("blocking_check_count") != failed_count:
        errors.append("decision.blocking_check_count must match failed checks")
    if not isinstance(blocking_checks, list):
        errors.append("decision.blocking_checks must be a list")
    elif failed_count is not None and len(blocking_checks) != failed_count:
        errors.append("decision.blocking_checks length must match failed checks")
    if not isinstance(failed_checks, list):
        errors.append("decision.failed_checks must be a list")
    elif failed_count is not None and len(failed_checks) != failed_count:
        errors.append("decision.failed_checks length must match failed checks")
    if not isinstance(next_actions, list):
        errors.append("decision.next_actions must be a list")
        next_action_count = None
    else:
        next_action_count = len(next_actions)
        if failed_count and not next_actions:
            errors.append("decision.next_actions must not be empty when checks fail")
    if next_action_count is not None and decision.get("next_action_count") != next_action_count:
        errors.append("decision.next_action_count must match next_actions")
    if not isinstance(decision.get("key_metrics"), dict):
        errors.append("decision.key_metrics must be an object")

    return {
        "available": isinstance(gate.get("decision"), dict),
        "valid": not errors,
        "readiness": readiness if isinstance(readiness, str) else "",
        "recommendation": recommendation if isinstance(recommendation, str) else "",
        "failed_check_count": failed_count if failed_count is not None else 0,
        "blocking_check_count": decision.get("blocking_check_count") if isinstance(decision.get("blocking_check_count"), int) else 0,
        "next_action_count": next_action_count if next_action_count is not None else 0,
        "errors": errors,
    }


def _failed_checks(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": str(check.get("id") or "unknown"),
            "summary": str(check.get("summary") or ""),
            "scope": check.get("scope") if isinstance(check.get("scope"), dict) else {},
        }
        for check in checks
        if isinstance(check, dict) and check.get("passed") is False
    ]


def _failed_check_count(checks: Any) -> int | None:
    if not isinstance(checks, list):
        return None
    count = 0
    for check in checks:
        if not isinstance(check, dict) or not isinstance(check.get("passed"), bool):
            return None
        if check["passed"] is False:
            count += 1
    return count


def _next_actions(gate_id: str, gate_label: str, failed_checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not failed_checks:
        return []
    return [
        {
            "id": "resolve_failed_checks",
            "priority": "critical",
            "artifact": gate_id,
            "summary": f"Resolve {len(failed_checks)} failed {gate_label} check(s) before using this gate downstream.",
            "evidence": {
                "failed_check_count": len(failed_checks),
                "failed_checks": failed_checks,
            },
        }
    ]


def _decision_text(gate_label: str, readiness: str, failed_checks: list[dict[str, Any]]) -> str:
    if readiness == "ready":
        return f"{gate_label} is ready: all configured checks passed."
    if not failed_checks:
        return f"{gate_label} is blocked."
    first = failed_checks[0]
    return f"{gate_label} is blocked by {len(failed_checks)} check(s); first failure: {first['summary']}"
