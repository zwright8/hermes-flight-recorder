"""Scenario-suite validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .schema import REGEX_FIELDS, ScenarioError, load_scenario, resolve_trace_path
from .state import resolve_state_snapshot_path

SCENARIO_CHECK_SCHEMA_VERSION = "hfr.scenario_check.v1"


def discover_scenarios(root: str | Path, pattern: str, recursive: bool) -> list[Path]:
    """Discover scenario files under a directory."""
    scenario_root = Path(root)
    if not scenario_root.exists():
        raise FileNotFoundError(f"Scenario directory not found: {scenario_root}")
    if not scenario_root.is_dir():
        raise NotADirectoryError(f"Scenario path is not a directory: {scenario_root}")
    paths = sorted(scenario_root.rglob(pattern) if recursive else scenario_root.glob(pattern))
    scenario_paths = [path for path in paths if path.is_file()]
    if not scenario_paths:
        raise FileNotFoundError(f"No scenario files matched {pattern!r} in {scenario_root}")
    return scenario_paths


def check_scenarios(
    scenarios_dir: str | Path,
    *,
    pattern: str = "*.json",
    recursive: bool = False,
    require_traces: bool = False,
    strict: bool = False,
    preserve_paths: bool = False,
) -> dict[str, Any]:
    """Validate a directory of scenario definitions before running them."""
    root = Path(scenarios_dir)
    scenario_paths = discover_scenarios(root, pattern, recursive)
    entries: list[dict[str, Any]] = []
    duplicates: list[dict[str, str]] = []
    seen_ids: dict[str, Path] = {}

    for scenario_path in scenario_paths:
        entry: dict[str, Any] = {
            "path": _display_path(scenario_path, preserve_paths),
            "passed": True,
            "errors": [],
            "warnings": [],
        }
        try:
            scenario = load_scenario(scenario_path)
            scenario_id = str(scenario["id"])
            entry.update(
                {
                    "id": scenario_id,
                    "title": scenario.get("title", scenario_id),
                    "trace_required": require_traces,
                    "policy": _policy_summary(scenario.get("policy", {})),
                    "assertions": _assertion_summary(scenario.get("assertions", {})),
                }
            )
            if scenario_id in seen_ids:
                duplicate = {
                    "id": scenario_id,
                    "first_path": _display_path(seen_ids[scenario_id], preserve_paths),
                    "duplicate_path": _display_path(scenario_path, preserve_paths),
                }
                duplicates.append(duplicate)
                entry["errors"].append(
                    f"Duplicate scenario id {scenario_id!r}; first seen in {duplicate['first_path']}."
                )
            else:
                seen_ids[scenario_id] = scenario_path
            _check_trace(entry, scenario, require_traces, preserve_paths)
            _check_state(entry, scenario, preserve_paths)
            _check_useful_constraints(entry, scenario)
        except ScenarioError as exc:
            entry["errors"].append(str(exc))
        entry["passed"] = not entry["errors"]
        entries.append(entry)

    error_count = sum(len(entry["errors"]) for entry in entries)
    warning_count = sum(len(entry["warnings"]) for entry in entries)
    return {
        "schema_version": SCENARIO_CHECK_SCHEMA_VERSION,
        "scenarios_dir": _display_path(root, preserve_paths),
        "pattern": pattern,
        "recursive": recursive,
        "strict": strict,
        "require_traces": require_traces,
        "passed": error_count == 0 and (warning_count == 0 or not strict),
        "total": len(entries),
        "error_count": error_count,
        "warning_count": warning_count,
        "duplicate_id_count": len(duplicates),
        "duplicates": duplicates,
        "scenarios": entries,
    }


def _check_trace(entry: dict[str, Any], scenario: dict[str, Any], require_traces: bool, preserve_paths: bool) -> None:
    trace = scenario.get("trace")
    has_trace_path = isinstance(trace, dict) and bool(trace.get("path"))
    if not has_trace_path:
        message = "scenario.trace.path is missing; run-suite needs a trace path unless every run supplies an override."
        if require_traces:
            entry["errors"].append(message)
        else:
            entry["warnings"].append(message)
        entry["trace_exists"] = False
        return

    trace_path = resolve_trace_path(scenario)
    entry["trace_path"] = _display_path(trace_path, preserve_paths)
    entry["trace_exists"] = trace_path.exists()
    if not trace_path.exists():
        message = f"scenario.trace.path does not exist: {_display_path(trace_path, preserve_paths)}"
        if require_traces:
            entry["errors"].append(message)
        else:
            entry["warnings"].append(message)


def _check_state(entry: dict[str, Any], scenario: dict[str, Any], preserve_paths: bool) -> None:
    state_path = resolve_state_snapshot_path(scenario)
    required_state = scenario.get("assertions", {}).get("required_state") or []
    if state_path is None:
        entry["state_exists"] = False
        if required_state:
            entry["warnings"].append("scenario has required_state assertions but no state.path; run or score must provide --state.")
        return
    entry["state_path"] = _display_path(state_path, preserve_paths)
    entry["state_exists"] = state_path.exists()
    if not state_path.exists():
        entry["errors"].append(f"scenario.state.path does not exist: {_display_path(state_path, preserve_paths)}")


def _check_useful_constraints(entry: dict[str, Any], scenario: dict[str, Any]) -> None:
    policy = scenario.get("policy", {})
    assertions = scenario.get("assertions", {})
    has_policy = any(policy.get(field) for field in REGEX_FIELDS) or any(
        policy.get(field) is not None
        for field in ("max_tool_calls", "max_subagents", "max_subagent_depth", "max_api_calls")
    )
    has_assertions = any(
        assertions.get(field)
        for field in (
            "final_contains",
            "final_not_contains",
            "required_evidence",
            "required_actions",
            "required_action_sequences",
            "required_event_counts",
            "required_state",
        )
    )
    if not has_policy and not has_assertions:
        entry["warnings"].append("scenario has no policy constraints or assertions; scoring evidence may be uninformative.")
    elif not has_assertions:
        entry["warnings"].append("scenario has no assertions; score depends only on policy and budget rules.")


def _policy_summary(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "regex_constraint_count": sum(len(policy.get(field, [])) for field in REGEX_FIELDS),
        "budget_constraints": {
            field: policy.get(field)
            for field in ("max_tool_calls", "max_subagents", "max_subagent_depth", "max_api_calls")
            if policy.get(field) is not None
        },
    }


def _assertion_summary(assertions: dict[str, Any]) -> dict[str, int]:
    return {
        "final_contains": len(assertions.get("final_contains", [])),
        "final_not_contains": len(assertions.get("final_not_contains", [])),
        "required_evidence": len(assertions.get("required_evidence", [])),
        "required_actions": len(assertions.get("required_actions", [])),
        "required_action_sequences": len(assertions.get("required_action_sequences", [])),
        "required_event_counts": len(assertions.get("required_event_counts", [])),
        "required_state": len(assertions.get("required_state", [])),
    }


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
