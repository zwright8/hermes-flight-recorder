"""Scenario contract quality summaries for improvement-loop readiness."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .schema import REGEX_FIELDS, ScenarioError, load_scenario, resolve_trace_path
from .scenario_check import discover_scenarios

SCENARIO_QUALITY_SCHEMA_VERSION = "hfr.scenario_quality.v1"
FAMILY_SUFFIX_RE = re.compile(r"([_-](good|bad|pass|fail|passing|failing|chosen|rejected))+$", re.IGNORECASE)


def build_scenario_quality(
    scenarios_dir: str | Path,
    *,
    pattern: str = "*.json",
    recursive: bool = False,
    require_traces: bool = False,
    preserve_paths: bool = False,
    min_average_score: float | None = None,
    min_scenario_score: float | None = None,
    min_observable_rate: float | None = None,
    max_weak_scenarios: int | None = None,
    max_final_only_scenarios: int | None = None,
    max_missing_traces: int | None = None,
    require_task_families: list[str] | None = None,
) -> dict[str, Any]:
    """Build a deterministic scenario contract-quality report."""
    root = Path(scenarios_dir)
    scenario_paths = discover_scenarios(root, pattern, recursive)
    rows = [_scenario_quality(path, require_traces, preserve_paths) for path in scenario_paths]
    metrics = _metrics(rows)
    checks = _quality_checks(
        metrics,
        min_average_score=min_average_score,
        min_scenario_score=min_scenario_score,
        min_observable_rate=min_observable_rate,
        max_weak_scenarios=max_weak_scenarios,
        max_final_only_scenarios=max_final_only_scenarios,
        max_missing_traces=max_missing_traces,
        require_task_families=require_task_families or [],
    )
    error_count = sum(len(row["errors"]) for row in rows)
    if error_count:
        checks.append(
            {
                "id": "valid_scenarios",
                "passed": False,
                "actual": error_count,
                "expected": {"max": 0},
                "summary": f"valid_scenarios: errors={error_count}",
            }
        )
    return {
        "schema_version": SCENARIO_QUALITY_SCHEMA_VERSION,
        "scenarios_dir": _display_path(root, preserve_paths),
        "pattern": pattern,
        "recursive": recursive,
        "require_traces": require_traces,
        "passed": all(check["passed"] for check in checks),
        "check_count": len(checks),
        "failed_check_count": sum(1 for check in checks if not check["passed"]),
        "checks": checks,
        "metrics": metrics,
        "scenarios": rows,
    }


def _scenario_quality(path: Path, require_traces: bool, preserve_paths: bool) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": _display_path(path, preserve_paths),
        "id": path.stem,
        "title": path.stem,
        "task_family": _task_family(path.stem),
        "contract_score": 0,
        "quality": "invalid",
        "errors": [],
        "risks": [],
    }
    try:
        scenario = load_scenario(path)
    except ScenarioError as exc:
        row["errors"].append(str(exc))
        return row

    scenario_id = str(scenario["id"])
    policy = scenario.get("policy", {})
    assertions = scenario.get("assertions", {})
    trace_info = _trace_info(scenario, require_traces, preserve_paths)
    signals = _signals(policy, assertions, trace_info, scenario)
    score = _contract_score(signals)
    risks = _risks(signals)
    row.update(
        {
            "id": scenario_id,
            "title": str(scenario.get("title") or scenario_id),
            "task_family": _task_family(scenario_id),
            "contract_score": score,
            "quality": _quality_label(score),
            "signals": signals,
            "risks": risks,
            "trace": trace_info,
        }
    )
    if require_traces and not trace_info.get("trace_exists"):
        row["errors"].append("scenario trace is required but does not resolve to an existing file.")
    return row


def _trace_info(scenario: dict[str, Any], require_traces: bool, preserve_paths: bool) -> dict[str, Any]:
    trace = scenario.get("trace")
    has_trace_path = isinstance(trace, dict) and bool(trace.get("path"))
    info: dict[str, Any] = {"has_trace_path": has_trace_path, "trace_exists": False, "trace_required": require_traces}
    if has_trace_path:
        trace_path = resolve_trace_path(scenario)
        info["trace_path"] = _display_path(trace_path, preserve_paths)
        info["trace_exists"] = trace_path.exists()
    return info


def _signals(policy: dict[str, Any], assertions: dict[str, Any], trace_info: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    regex_count = sum(len(policy.get(field, [])) for field in REGEX_FIELDS)
    budget_fields = ("max_tool_calls", "max_subagents", "max_subagent_depth", "max_api_calls")
    budget_count = sum(1 for field in budget_fields if policy.get(field) is not None)
    required_evidence_count = len(assertions.get("required_evidence", []))
    required_action_count = len(assertions.get("required_actions", []))
    sequence_count = len(assertions.get("required_action_sequences", []))
    event_count_count = len(assertions.get("required_event_counts", []))
    final_assertion_count = len(assertions.get("final_contains", [])) + len(assertions.get("final_not_contains", []))
    observable_count = required_evidence_count + required_action_count + sequence_count + event_count_count
    task_completion_count = required_action_count + sequence_count + event_count_count
    return {
        "has_trace_path": bool(trace_info.get("has_trace_path")),
        "trace_exists": bool(trace_info.get("trace_exists")),
        "regex_constraint_count": regex_count,
        "budget_constraint_count": budget_count,
        "has_secret_policy": bool(policy.get("secret_patterns")),
        "has_budget_limits": budget_count > 0,
        "required_evidence_count": required_evidence_count,
        "required_action_count": required_action_count,
        "required_action_sequence_count": sequence_count,
        "required_event_count_count": event_count_count,
        "observable_assertion_count": observable_count,
        "task_completion_assertion_count": task_completion_count,
        "final_assertion_count": final_assertion_count,
        "pass_threshold": scenario.get("scoring", {}).get("pass_threshold"),
    }


def _contract_score(signals: dict[str, Any]) -> int:
    score = 0
    if signals["has_trace_path"]:
        score += 10
    if signals["trace_exists"]:
        score += 10
    if signals["regex_constraint_count"] > 0:
        score += 15
    if signals["budget_constraint_count"] > 0:
        score += 15
    if signals["observable_assertion_count"] > 0:
        score += 25
    if signals["final_assertion_count"] > 0:
        score += 10
    if signals["task_completion_assertion_count"] > 0:
        score += 15
    if signals["required_event_count_count"] > 0:
        score += 10
    if isinstance(signals.get("pass_threshold"), int) and signals["pass_threshold"] >= 90:
        score += 5
    return min(score, 100)


def _risks(signals: dict[str, Any]) -> list[str]:
    risks: list[str] = []
    if not signals["has_trace_path"]:
        risks.append("missing_trace_path")
    elif not signals["trace_exists"]:
        risks.append("missing_trace_file")
    if signals["regex_constraint_count"] == 0 and signals["budget_constraint_count"] == 0:
        risks.append("no_policy_constraints")
    if not signals["has_secret_policy"]:
        risks.append("no_secret_policy")
    if not signals["has_budget_limits"]:
        risks.append("no_budget_limits")
    if signals["observable_assertion_count"] == 0:
        risks.append("no_observable_assertions")
    if signals["final_assertion_count"] > 0 and signals["observable_assertion_count"] == 0 and signals["budget_constraint_count"] == 0:
        risks.append("final_only_contract")
    if signals["required_action_count"] > 0 and signals["required_event_count_count"] == 0:
        risks.append("action_without_count_guard")
    if signals["task_completion_assertion_count"] > 0 and signals["required_action_sequence_count"] == 0:
        risks.append("task_completion_without_order_guard")
    return risks


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [row for row in rows if not row["errors"]]
    scores = [int(row["contract_score"]) for row in valid_rows]
    task_families = sorted({str(row["task_family"]) for row in valid_rows})
    risk_counts: dict[str, int] = {}
    for row in valid_rows:
        for risk in row["risks"]:
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
    observable_count = sum(1 for row in valid_rows if row.get("signals", {}).get("observable_assertion_count", 0) > 0)
    weak_count = sum(1 for row in valid_rows if row["quality"] == "weak")
    final_only_count = sum(1 for row in valid_rows if "final_only_contract" in row["risks"])
    missing_trace_count = sum(1 for row in valid_rows if not row.get("trace", {}).get("trace_exists"))
    return {
        "scenario_count": len(rows),
        "valid_scenario_count": len(valid_rows),
        "invalid_scenario_count": len(rows) - len(valid_rows),
        "task_family_count": len(task_families),
        "task_families": task_families,
        "average_contract_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
        "min_contract_score": min(scores) if scores else 0,
        "max_contract_score": max(scores) if scores else 0,
        "observable_scenario_count": observable_count,
        "observable_scenario_rate": _rate(observable_count, len(valid_rows)),
        "weak_scenario_count": weak_count,
        "final_only_scenario_count": final_only_count,
        "missing_trace_count": missing_trace_count,
        "risk_counts": _count_rows(risk_counts),
    }


def _quality_checks(
    metrics: dict[str, Any],
    *,
    min_average_score: float | None,
    min_scenario_score: float | None,
    min_observable_rate: float | None,
    max_weak_scenarios: int | None,
    max_final_only_scenarios: int | None,
    max_missing_traces: int | None,
    require_task_families: list[str],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    if min_average_score is not None:
        _add_min_check(checks, "min_average_contract_score", metrics["average_contract_score"], min_average_score)
    if min_scenario_score is not None:
        _add_min_check(checks, "min_scenario_contract_score", metrics["min_contract_score"], min_scenario_score)
    if min_observable_rate is not None:
        _add_min_check(checks, "min_observable_scenario_rate", metrics["observable_scenario_rate"], min_observable_rate)
    if max_weak_scenarios is not None:
        _add_max_check(checks, "max_weak_scenarios", metrics["weak_scenario_count"], max_weak_scenarios)
    if max_final_only_scenarios is not None:
        _add_max_check(checks, "max_final_only_scenarios", metrics["final_only_scenario_count"], max_final_only_scenarios)
    if max_missing_traces is not None:
        _add_max_check(checks, "max_missing_traces", metrics["missing_trace_count"], max_missing_traces)
    families = set(metrics["task_families"])
    for family in require_task_families:
        _add_presence_check(checks, "require_task_family", family in families, {"task_family": family})
    return checks


def _add_min_check(
    checks: list[dict[str, Any]],
    check_id: str,
    actual: float | int,
    minimum: float | int,
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


def _quality_label(score: int) -> str:
    if score >= 80:
        return "strong"
    if score >= 50:
        return "moderate"
    return "weak"


def _task_family(scenario_id: str) -> str:
    family = FAMILY_SUFFIX_RE.sub("", scenario_id).strip("_-")
    return family or scenario_id or "unknown"


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _count_rows(counts: dict[str, int]) -> list[dict[str, int | str]]:
    return [
        {"id": key, "count": count}
        for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _display_path(path: Path, preserve_paths: bool = False) -> str:
    raw = str(path)
    if preserve_paths:
        return raw
    if _is_windows_absolute(raw):
        return f"<redacted:{_basename(raw)}>"
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    try:
        return str(resolved.relative_to(cwd))
    except ValueError:
        return f"<redacted:{resolved.name}>"


def _is_windows_absolute(value: str) -> bool:
    normalized = value.replace("/", "\\")
    return (len(normalized) >= 3 and normalized[1:3] == ":\\" and normalized[0].isalpha()) or normalized.startswith("\\\\")


def _basename(value: str) -> str:
    return value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] or "path"
