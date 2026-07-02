"""Shared gate-decision contract helpers."""

from __future__ import annotations

from typing import Any

READY_RECOMMENDATION = "promote_iteration"
BLOCK_RECOMMENDATION = "block_iteration"


def build_gate_decision(
    *,
    passed: bool,
    checks: list[dict[str, Any]],
    metrics: dict[str, Any],
    ready_recommendation: str = READY_RECOMMENDATION,
    block_recommendation: str = BLOCK_RECOMMENDATION,
) -> dict[str, Any]:
    """Build a compact, shared decision block for gate artifacts."""
    failed_checks = [_decision_check(check) for check in checks if check.get("passed") is False]
    readiness = "ready" if passed else "blocked"
    recommendation = ready_recommendation if passed else block_recommendation
    next_actions = [] if passed else [_next_action(failed_checks)]
    return {
        "readiness": readiness,
        "recommendation": recommendation,
        "summary": _decision_summary(readiness, failed_checks),
        "blocking_check_count": len(failed_checks),
        "blocking_checks": failed_checks,
        "failed_checks": failed_checks,
        "next_action_count": len(next_actions),
        "next_actions": next_actions,
        "key_metrics": dict(metrics),
    }


def summarize_gate_contract(gate: dict[str, Any]) -> dict[str, Any]:
    """Summarize whether a gate exposes a self-consistent decision contract."""
    errors: list[str] = []
    passed = gate.get("passed") if isinstance(gate.get("passed"), bool) else None
    checks = gate.get("checks") if isinstance(gate.get("checks"), list) else []
    failed_checks = [check for check in checks if isinstance(check, dict) and check.get("passed") is False]
    failed_check_count = len(failed_checks)
    declared_failed_count = gate.get("failed_check_count")
    if passed is None:
        errors.append("passed must be a boolean")
    elif passed and failed_check_count:
        errors.append("passed must be false when checks fail")
    if declared_failed_count != failed_check_count:
        errors.append("failed_check_count must match failed checks")

    decision = gate.get("decision")
    if not isinstance(decision, dict):
        errors.append("decision must be an object")
        return _summary(False, False, errors, "", "", failed_check_count, 0, 0)

    expected_readiness = "ready" if passed is True and failed_check_count == 0 else "blocked"
    if decision.get("readiness") != expected_readiness:
        errors.append(f"decision.readiness must be {expected_readiness}")
    recommendation = decision.get("recommendation")
    if not isinstance(recommendation, str) or not recommendation:
        errors.append("decision.recommendation must be a non-empty string")
    if not isinstance(decision.get("summary"), str) or not decision.get("summary"):
        errors.append("decision.summary must be a non-empty string")
    if decision.get("blocking_check_count") != failed_check_count:
        errors.append("decision.blocking_check_count must match failed checks")
    for field_name in ("blocking_checks", "failed_checks", "next_actions"):
        if not isinstance(decision.get(field_name), list):
            errors.append(f"decision.{field_name} must be a list")
    blocking_checks = decision.get("blocking_checks") if isinstance(decision.get("blocking_checks"), list) else []
    if len(blocking_checks) != failed_check_count:
        errors.append("decision.blocking_checks length must match failed checks")
    next_actions = decision.get("next_actions") if isinstance(decision.get("next_actions"), list) else []
    if decision.get("next_action_count") != len(next_actions):
        errors.append("decision.next_action_count must match next_actions")
    if not isinstance(decision.get("key_metrics"), dict):
        errors.append("decision.key_metrics must be an object")

    return _summary(
        True,
        not errors,
        errors,
        str(decision.get("readiness") or ""),
        str(recommendation or ""),
        failed_check_count,
        _non_negative_int(decision.get("blocking_check_count")),
        _non_negative_int(decision.get("next_action_count")),
    )


def _summary(
    available: bool,
    valid: bool,
    errors: list[str],
    readiness: str,
    recommendation: str,
    failed_check_count: int,
    blocking_check_count: int,
    next_action_count: int,
) -> dict[str, Any]:
    return {
        "available": available,
        "valid": valid,
        "errors": errors,
        "readiness": readiness,
        "recommendation": recommendation,
        "failed_check_count": failed_check_count,
        "blocking_check_count": blocking_check_count,
        "next_action_count": next_action_count,
    }


def _decision_check(check: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(check.get("id") or "unknown"),
        "summary": str(check.get("summary") or ""),
        "scope": check.get("scope") if isinstance(check.get("scope"), dict) else {},
    }


def _decision_summary(readiness: str, failed_checks: list[dict[str, Any]]) -> str:
    if readiness == "ready":
        return "Gate is ready: all checks passed."
    if not failed_checks:
        return "Gate is blocked."
    return f"Gate is blocked by {len(failed_checks)} failed check(s); first failure: {failed_checks[0]['summary']}"


def _next_action(failed_checks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": "resolve_failed_checks",
        "priority": "critical",
        "artifact": "gate",
        "summary": f"Resolve {len(failed_checks)} failed gate check(s).",
        "evidence": {"failed_check_count": len(failed_checks), "failed_checks": failed_checks[:10]},
    }


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int) and value >= 0:
        return value
    return 0
