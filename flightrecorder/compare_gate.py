"""Policy gates over baseline/candidate comparison exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

COMPARE_GATE_SCHEMA_VERSION = "hfr.compare_gate.v1"
COMPARE_GATE_POLICY_SCHEMA_VERSION = "hfr.compare_gate.policy.v1"

_COUNT_POLICY_FIELDS = {
    "min_pairs",
    "min_dpo",
    "min_candidate_wins",
    "max_baseline_wins",
    "max_skipped_pairs",
}
_LIST_POLICY_FIELDS = {
    "require_scenarios",
    "require_candidate_win_scenarios",
    "forbid_regression_scenarios",
    "require_rule_fixes",
    "forbid_rule_regressions",
    "forbid_new_critical_failures",
}
_POLICY_FIELDS = {"schema_version", "description", *_COUNT_POLICY_FIELDS, *_LIST_POLICY_FIELDS}


class CompareGatePolicyError(ValueError):
    """Raised when a comparison-export gate policy file is malformed."""


def load_compare_gate_policy(path: str | Path) -> dict[str, Any]:
    """Load and validate a versioned comparison gate policy JSON file."""
    policy_path = Path(path)
    try:
        raw = json.loads(policy_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CompareGatePolicyError(f"Invalid JSON in compare gate policy {policy_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise CompareGatePolicyError(f"Compare gate policy must be a JSON object: {policy_path}")

    version = raw.get("schema_version")
    if version != COMPARE_GATE_POLICY_SCHEMA_VERSION:
        raise CompareGatePolicyError(
            f"compare gate policy schema_version must be {COMPARE_GATE_POLICY_SCHEMA_VERSION!r}; got {version!r}"
        )

    unknown = sorted(set(raw) - _POLICY_FIELDS)
    if unknown:
        raise CompareGatePolicyError(f"Unknown compare gate policy field(s): {', '.join(unknown)}")

    policy: dict[str, Any] = {"schema_version": COMPARE_GATE_POLICY_SCHEMA_VERSION}
    if "description" in raw:
        if not isinstance(raw["description"], str):
            raise CompareGatePolicyError("compare gate policy field description must be a string")
        policy["description"] = raw["description"]

    for field in _COUNT_POLICY_FIELDS:
        if field not in raw or raw[field] is None:
            continue
        policy[field] = _policy_non_negative_int(field, raw[field])

    for field in _LIST_POLICY_FIELDS:
        if field not in raw or raw[field] is None:
            continue
        policy[field] = _policy_string_list(field, raw[field])

    return policy


def evaluate_compare_gate(
    manifest: dict[str, Any],
    pairs: list[dict[str, Any]],
    *,
    compare_export_path: str | Path,
    min_pairs: int | None = None,
    min_dpo: int | None = None,
    min_candidate_wins: int | None = None,
    max_baseline_wins: int | None = None,
    max_skipped_pairs: int | None = None,
    require_scenarios: list[str] | None = None,
    require_candidate_win_scenarios: list[str] | None = None,
    forbid_regression_scenarios: list[str] | None = None,
    require_rule_fixes: list[str] | None = None,
    forbid_rule_regressions: list[str] | None = None,
    forbid_new_critical_failures: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate readiness checks against a comparison export manifest and pairs."""
    checks: list[dict[str, Any]] = []
    pair_count = _int_value(manifest.get("pair_count"))
    dpo_count = _int_value(manifest.get("dpo_count"))
    candidate_win_count = _int_value(manifest.get("candidate_win_count"))
    baseline_win_count = _int_value(manifest.get("baseline_win_count"))
    skipped_pair_count = _int_value(manifest.get("skipped_pair_count"))

    if min_pairs is not None:
        _add_min_check(checks, "min_pairs", pair_count, min_pairs)
    if min_dpo is not None:
        _add_min_check(checks, "min_dpo", dpo_count, min_dpo)
    if min_candidate_wins is not None:
        _add_min_check(checks, "min_candidate_wins", candidate_win_count, min_candidate_wins)
    if max_baseline_wins is not None:
        _add_max_check(checks, "max_baseline_wins", baseline_win_count, max_baseline_wins)
    if max_skipped_pairs is not None:
        _add_max_check(checks, "max_skipped_pairs", skipped_pair_count, max_skipped_pairs)

    pairs_by_scenario = _pairs_by_scenario(pairs)
    candidate_win_scenarios = {
        str(pair.get("scenario_id"))
        for pair in pairs
        if pair.get("chosen_side") == "candidate" and isinstance(pair.get("scenario_id"), str)
    }
    baseline_win_scenarios = {
        str(pair.get("scenario_id"))
        for pair in pairs
        if pair.get("chosen_side") == "baseline" and isinstance(pair.get("scenario_id"), str)
    }
    fixed_rules = _count_values(rule for pair in pairs for rule in _string_list(pair.get("rule_fixes")))
    regressed_rules = _count_values(rule for pair in pairs for rule in _string_list(pair.get("rule_regressions")))
    new_critical = _count_values(rule for pair in pairs for rule in _string_list(pair.get("new_critical_failures")))

    for scenario_id in require_scenarios or []:
        _add_presence_check(checks, "require_scenario", scenario_id in pairs_by_scenario, {"scenario_id": scenario_id})
    for scenario_id in require_candidate_win_scenarios or []:
        _add_presence_check(
            checks,
            "require_candidate_win_scenario",
            scenario_id in candidate_win_scenarios,
            {"scenario_id": scenario_id},
        )
    for scenario_id in forbid_regression_scenarios or []:
        _add_absence_check(
            checks,
            "forbid_regression_scenario",
            scenario_id,
            1 if scenario_id in baseline_win_scenarios else 0,
            {"scenario_id": scenario_id},
        )
    for rule_id in require_rule_fixes or []:
        _add_presence_check(checks, "require_rule_fix", fixed_rules.get(rule_id, 0) > 0, {"rule_id": rule_id})
    for rule_id in forbid_rule_regressions or []:
        _add_absence_check(checks, "forbid_rule_regression", rule_id, regressed_rules.get(rule_id, 0), {"rule_id": rule_id})
    for rule_id in forbid_new_critical_failures or []:
        _add_absence_check(checks, "forbid_new_critical_failure", rule_id, new_critical.get(rule_id, 0), {"rule_id": rule_id})

    passed = all(check["passed"] for check in checks)
    return {
        "schema_version": COMPARE_GATE_SCHEMA_VERSION,
        "compare_export": str(compare_export_path),
        "passed": passed,
        "check_count": len(checks),
        "failed_check_count": sum(1 for check in checks if not check["passed"]),
        "checks": checks,
        "metrics": {
            "pair_count": pair_count,
            "dpo_count": dpo_count,
            "candidate_win_count": candidate_win_count,
            "baseline_win_count": baseline_win_count,
            "skipped_pair_count": skipped_pair_count,
            "candidate_win_scenarios": sorted(candidate_win_scenarios),
            "baseline_win_scenarios": sorted(baseline_win_scenarios),
            "fixed_rule_counts": fixed_rules,
            "regressed_rule_counts": regressed_rules,
            "new_critical_failure_counts": new_critical,
        },
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


def _add_presence_check(checks: list[dict[str, Any]], check_id: str, present: bool, scope: dict[str, str]) -> None:
    checks.append(
        {
            "id": check_id,
            "passed": present,
            "actual": {"present": present},
            "expected": {"present": True},
            "scope": scope,
            "summary": f"{check_id}: present={present}",
        }
    )


def _add_absence_check(
    checks: list[dict[str, Any]],
    check_id: str,
    item_id: str,
    count: int,
    scope: dict[str, str] | None = None,
) -> None:
    check = {
        "id": check_id,
        "passed": count == 0,
        "actual": {"id": item_id, "count": count},
        "expected": {"count": 0},
        "summary": f"{check_id}: id={item_id}, count={count}",
    }
    if scope:
        check["scope"] = scope
    checks.append(check)


def _pairs_by_scenario(pairs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        scenario_id = pair.get("scenario_id")
        if isinstance(scenario_id, str) and scenario_id:
            rows[scenario_id] = pair
    return rows


def _count_values(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if not isinstance(value, str) or not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return counts


def _string_list(value: Any) -> list[str]:
    return value if isinstance(value, list) and all(isinstance(item, str) for item in value) else []


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0


def _policy_non_negative_int(field: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CompareGatePolicyError(f"compare gate policy field {field} must be a non-negative integer")
    return value


def _policy_string_list(field: str, value: Any) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise CompareGatePolicyError(f"compare gate policy field {field} must be a list of non-empty strings")
    return list(dict.fromkeys(value))
