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
    "min_task_completion_improvements",
    "max_baseline_wins",
    "max_task_completion_regressions",
    "max_skipped_pairs",
    "max_contract_drifts",
    "max_unverified_contracts",
}
_LIST_POLICY_FIELDS = {
    "require_scenarios",
    "require_candidate_win_scenarios",
    "require_task_completion_improvement_scenarios",
    "forbid_regression_scenarios",
    "forbid_task_completion_regression_scenarios",
    "require_rule_fixes",
    "forbid_rule_regressions",
    "forbid_new_critical_failures",
}
_TASK_FAMILY_COUNT_POLICY_FIELDS = {
    "min_pairs",
    "min_candidate_wins",
    "min_task_completion_improvements",
    "max_baseline_wins",
    "max_task_completion_regressions",
    "max_contract_drifts",
    "max_unverified_contracts",
}
_TASK_FAMILY_LIST_POLICY_FIELDS = {
    "require_rule_fixes",
    "forbid_rule_regressions",
    "forbid_new_critical_failures",
}
_TASK_FAMILY_GATE_FIELDS = {"task_family", *_TASK_FAMILY_COUNT_POLICY_FIELDS, *_TASK_FAMILY_LIST_POLICY_FIELDS}
_POLICY_FIELDS = {"schema_version", "description", "task_family_gates", *_COUNT_POLICY_FIELDS, *_LIST_POLICY_FIELDS}


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

    if "task_family_gates" in raw and raw["task_family_gates"] is not None:
        policy["task_family_gates"] = _policy_task_family_gates(raw["task_family_gates"])

    return policy


def evaluate_compare_gate(
    manifest: dict[str, Any],
    pairs: list[dict[str, Any]],
    *,
    compare_export_path: str | Path,
    min_pairs: int | None = None,
    min_dpo: int | None = None,
    min_candidate_wins: int | None = None,
    min_task_completion_improvements: int | None = None,
    max_baseline_wins: int | None = None,
    max_task_completion_regressions: int | None = None,
    max_skipped_pairs: int | None = None,
    max_contract_drifts: int | None = None,
    max_unverified_contracts: int | None = None,
    require_scenarios: list[str] | None = None,
    require_candidate_win_scenarios: list[str] | None = None,
    require_task_completion_improvement_scenarios: list[str] | None = None,
    forbid_regression_scenarios: list[str] | None = None,
    forbid_task_completion_regression_scenarios: list[str] | None = None,
    require_rule_fixes: list[str] | None = None,
    forbid_rule_regressions: list[str] | None = None,
    forbid_new_critical_failures: list[str] | None = None,
    task_family_gates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate readiness checks against a comparison export manifest and pairs."""
    checks: list[dict[str, Any]] = []
    pair_count = _int_value(manifest.get("pair_count"))
    dpo_count = _int_value(manifest.get("dpo_count"))
    candidate_win_count = _int_value(manifest.get("candidate_win_count"))
    baseline_win_count = _int_value(manifest.get("baseline_win_count"))
    skipped_pair_count = _int_value(manifest.get("skipped_pair_count"))
    contract_drift_count = _int_value(manifest.get("contract_drift_count"))
    unverified_contract_count = _int_value(manifest.get("unverified_contract_count"))

    if min_pairs is not None:
        _add_min_check(checks, "min_pairs", pair_count, min_pairs)
    if min_dpo is not None:
        _add_min_check(checks, "min_dpo", dpo_count, min_dpo)
    if min_candidate_wins is not None:
        _add_min_check(checks, "min_candidate_wins", candidate_win_count, min_candidate_wins)
    task_completion_improvement_scenarios = _task_completion_improvement_scenarios(pairs)
    task_completion_regression_scenarios = _task_completion_regression_scenarios(pairs)
    if min_task_completion_improvements is not None:
        _add_min_check(
            checks,
            "min_task_completion_improvements",
            len(task_completion_improvement_scenarios),
            min_task_completion_improvements,
        )
    if max_baseline_wins is not None:
        _add_max_check(checks, "max_baseline_wins", baseline_win_count, max_baseline_wins)
    if max_task_completion_regressions is not None:
        _add_max_check(
            checks,
            "max_task_completion_regressions",
            len(task_completion_regression_scenarios),
            max_task_completion_regressions,
        )
    if max_skipped_pairs is not None:
        _add_max_check(checks, "max_skipped_pairs", skipped_pair_count, max_skipped_pairs)
    if max_contract_drifts is not None:
        _add_max_check(checks, "max_contract_drifts", contract_drift_count, max_contract_drifts)
    if max_unverified_contracts is not None:
        _add_max_check(checks, "max_unverified_contracts", unverified_contract_count, max_unverified_contracts)

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
    task_family_rows = _task_family_rows(pairs)

    for scenario_id in require_scenarios or []:
        _add_presence_check(checks, "require_scenario", scenario_id in pairs_by_scenario, {"scenario_id": scenario_id})
    for scenario_id in require_candidate_win_scenarios or []:
        _add_presence_check(
            checks,
            "require_candidate_win_scenario",
            scenario_id in candidate_win_scenarios,
            {"scenario_id": scenario_id},
        )
    for scenario_id in require_task_completion_improvement_scenarios or []:
        _add_presence_check(
            checks,
            "require_task_completion_improvement_scenario",
            scenario_id in task_completion_improvement_scenarios,
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
    for scenario_id in forbid_task_completion_regression_scenarios or []:
        _add_absence_check(
            checks,
            "forbid_task_completion_regression_scenario",
            scenario_id,
            1 if scenario_id in task_completion_regression_scenarios else 0,
            {"scenario_id": scenario_id},
        )
    for rule_id in require_rule_fixes or []:
        _add_presence_check(checks, "require_rule_fix", fixed_rules.get(rule_id, 0) > 0, {"rule_id": rule_id})
    for rule_id in forbid_rule_regressions or []:
        _add_absence_check(checks, "forbid_rule_regression", rule_id, regressed_rules.get(rule_id, 0), {"rule_id": rule_id})
    for rule_id in forbid_new_critical_failures or []:
        _add_absence_check(checks, "forbid_new_critical_failure", rule_id, new_critical.get(rule_id, 0), {"rule_id": rule_id})
    for gate in task_family_gates or []:
        _evaluate_task_family_gate(checks, gate, task_family_rows)

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
            "contract_drift_count": contract_drift_count,
            "unverified_contract_count": unverified_contract_count,
            "candidate_win_scenarios": sorted(candidate_win_scenarios),
            "baseline_win_scenarios": sorted(baseline_win_scenarios),
            "task_completion_improvement_count": len(task_completion_improvement_scenarios),
            "task_completion_regression_count": len(task_completion_regression_scenarios),
            "task_completion_improvement_scenarios": sorted(task_completion_improvement_scenarios),
            "task_completion_regression_scenarios": sorted(task_completion_regression_scenarios),
            "fixed_rule_counts": fixed_rules,
            "regressed_rule_counts": regressed_rules,
            "new_critical_failure_counts": new_critical,
            "task_families": [task_family_rows[family] for family in sorted(task_family_rows)],
        },
    }


def _evaluate_task_family_gate(
    checks: list[dict[str, Any]],
    gate: dict[str, Any],
    task_family_rows: dict[str, dict[str, Any]],
) -> None:
    family = str(gate["task_family"])
    row = task_family_rows.get(family)
    scope = {"task_family": family}
    if row is None:
        _add_presence_check(checks, "task_family_present", False, scope)
        return
    _add_presence_check(checks, "task_family_present", True, scope)
    min_checks = {
        "min_pairs": ("task_family_min_pairs", "pair_count"),
        "min_candidate_wins": ("task_family_min_candidate_wins", "candidate_win_count"),
        "min_task_completion_improvements": (
            "task_family_min_task_completion_improvements",
            "task_completion_improvement_count",
        ),
    }
    for policy_field, (check_id, row_field) in min_checks.items():
        if gate.get(policy_field) is not None:
            _add_min_check(checks, check_id, _int_value(row.get(row_field)), gate[policy_field], scope)
    max_checks = {
        "max_baseline_wins": ("task_family_max_baseline_wins", "baseline_win_count"),
        "max_task_completion_regressions": (
            "task_family_max_task_completion_regressions",
            "task_completion_regression_count",
        ),
        "max_contract_drifts": ("task_family_max_contract_drifts", "contract_drift_count"),
        "max_unverified_contracts": ("task_family_max_unverified_contracts", "unverified_contract_count"),
    }
    for policy_field, (check_id, row_field) in max_checks.items():
        if gate.get(policy_field) is not None:
            _add_max_check(checks, check_id, _int_value(row.get(row_field)), gate[policy_field], scope)
    fixed_rules = row.get("fixed_rule_counts") if isinstance(row.get("fixed_rule_counts"), dict) else {}
    regressed_rules = row.get("regressed_rule_counts") if isinstance(row.get("regressed_rule_counts"), dict) else {}
    new_critical = row.get("new_critical_failure_counts") if isinstance(row.get("new_critical_failure_counts"), dict) else {}
    for rule_id in _string_list(gate.get("require_rule_fixes")):
        _add_presence_check(checks, "task_family_require_rule_fix", _int_value(fixed_rules.get(rule_id)) > 0, {**scope, "rule_id": rule_id})
    for rule_id in _string_list(gate.get("forbid_rule_regressions")):
        _add_absence_check(
            checks,
            "task_family_forbid_rule_regression",
            rule_id,
            _int_value(regressed_rules.get(rule_id)),
            {**scope, "rule_id": rule_id},
        )
    for rule_id in _string_list(gate.get("forbid_new_critical_failures")):
        _add_absence_check(
            checks,
            "task_family_forbid_new_critical_failure",
            rule_id,
            _int_value(new_critical.get(rule_id)),
            {**scope, "rule_id": rule_id},
        )


def _add_min_check(
    checks: list[dict[str, Any]],
    check_id: str,
    actual: int,
    minimum: int,
    scope: dict[str, str] | None = None,
) -> None:
    check = {
        "id": check_id,
        "passed": actual >= minimum,
        "actual": actual,
        "expected": {"min": minimum},
        "summary": f"{check_id}: actual={actual}, min={minimum}",
    }
    if scope:
        check["scope"] = scope
    checks.append(check)


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
    if scope:
        check["scope"] = scope
    checks.append(check)


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


def _task_completion_improvement_scenarios(pairs: list[dict[str, Any]]) -> set[str]:
    return {
        str(pair["scenario_id"])
        for pair in pairs
        if isinstance(pair.get("scenario_id"), str)
        and _task_completion_status(pair.get("candidate")) == "complete"
        and _task_completion_status(pair.get("baseline")) != "complete"
    }


def _task_completion_regression_scenarios(pairs: list[dict[str, Any]]) -> set[str]:
    return {
        str(pair["scenario_id"])
        for pair in pairs
        if isinstance(pair.get("scenario_id"), str)
        and _task_completion_status(pair.get("baseline")) == "complete"
        and _task_completion_status(pair.get("candidate")) != "complete"
    }


def _task_completion_status(value: Any) -> str:
    if not isinstance(value, dict):
        return "unknown"
    task = value.get("task_completion")
    if not isinstance(task, dict):
        return "unknown"
    status = task.get("status")
    return str(status) if isinstance(status, str) and status else "unknown"


def _task_family_rows(pairs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        family = str(pair.get("task_family") or "unknown")
        if not family:
            family = "unknown"
        row = rows.setdefault(
            family,
            {
                "task_family": family,
                "pair_count": 0,
                "candidate_win_count": 0,
                "baseline_win_count": 0,
                "task_completion_improvement_count": 0,
                "task_completion_regression_count": 0,
                "contract_drift_count": 0,
                "unverified_contract_count": 0,
                "scenarios": set(),
                "candidate_win_scenarios": set(),
                "baseline_win_scenarios": set(),
                "task_completion_improvement_scenarios": set(),
                "task_completion_regression_scenarios": set(),
                "fixed_rule_counts": {},
                "regressed_rule_counts": {},
                "new_critical_failure_counts": {},
            },
        )
        row["pair_count"] += 1
        scenario_id = pair.get("scenario_id")
        scenario = scenario_id if isinstance(scenario_id, str) and scenario_id else None
        if scenario:
            row["scenarios"].add(scenario)
        chosen_side = pair.get("chosen_side")
        if chosen_side == "candidate":
            row["candidate_win_count"] += 1
            if scenario:
                row["candidate_win_scenarios"].add(scenario)
        elif chosen_side == "baseline":
            row["baseline_win_count"] += 1
            if scenario:
                row["baseline_win_scenarios"].add(scenario)
        if _task_completion_status(pair.get("candidate")) == "complete" and _task_completion_status(pair.get("baseline")) != "complete":
            row["task_completion_improvement_count"] += 1
            if scenario:
                row["task_completion_improvement_scenarios"].add(scenario)
        if _task_completion_status(pair.get("baseline")) == "complete" and _task_completion_status(pair.get("candidate")) != "complete":
            row["task_completion_regression_count"] += 1
            if scenario:
                row["task_completion_regression_scenarios"].add(scenario)
        if pair.get("contract_fingerprint_status") == "drifted":
            row["contract_drift_count"] += 1
        if pair.get("contract_fingerprint_status") == "unverified":
            row["unverified_contract_count"] += 1
        _increment_counts(row["fixed_rule_counts"], _string_list(pair.get("rule_fixes")))
        _increment_counts(row["regressed_rule_counts"], _string_list(pair.get("rule_regressions")))
        _increment_counts(row["new_critical_failure_counts"], _string_list(pair.get("new_critical_failures")))

    for row in rows.values():
        for field_name in (
            "scenarios",
            "candidate_win_scenarios",
            "baseline_win_scenarios",
            "task_completion_improvement_scenarios",
            "task_completion_regression_scenarios",
        ):
            row[field_name] = sorted(row[field_name])
    return rows


def _increment_counts(counts: dict[str, int], values: list[str]) -> None:
    for value in values:
        counts[value] = counts.get(value, 0) + 1


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


def _policy_task_family_gates(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise CompareGatePolicyError("compare gate policy field task_family_gates must be a list")
    gates: list[dict[str, Any]] = []
    for index, raw_gate in enumerate(value):
        if not isinstance(raw_gate, dict):
            raise CompareGatePolicyError(f"compare gate policy task_family_gates[{index}] must be an object")
        unknown = sorted(set(raw_gate) - _TASK_FAMILY_GATE_FIELDS)
        if unknown:
            raise CompareGatePolicyError(
                f"Unknown compare gate policy task_family_gates[{index}] field(s): {', '.join(unknown)}"
            )
        task_family = raw_gate.get("task_family")
        if not isinstance(task_family, str) or not task_family:
            raise CompareGatePolicyError(f"compare gate policy task_family_gates[{index}].task_family must be a non-empty string")
        gate: dict[str, Any] = {"task_family": task_family}
        for field in _TASK_FAMILY_COUNT_POLICY_FIELDS:
            if field in raw_gate and raw_gate[field] is not None:
                gate[field] = _policy_non_negative_int(f"task_family_gates[{index}].{field}", raw_gate[field])
        for field in _TASK_FAMILY_LIST_POLICY_FIELDS:
            if field in raw_gate and raw_gate[field] is not None:
                gate[field] = _policy_string_list(f"task_family_gates[{index}].{field}", raw_gate[field])
        gates.append(gate)
    return gates
