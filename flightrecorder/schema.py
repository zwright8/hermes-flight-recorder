"""Scenario loading and validation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


class ScenarioError(ValueError):
    """Raised when a scenario file cannot be loaded or validated."""


DEFAULT_POLICY: dict[str, Any] = {
    "forbidden_tool_names": [],
    "forbidden_command_patterns": [],
    "forbidden_url_patterns": [],
    "secret_patterns": [],
    "max_tool_calls": None,
    "max_subagents": None,
    "max_subagent_depth": None,
    "max_api_calls": None,
}

DEFAULT_ASSERTIONS: dict[str, Any] = {
    "final_contains": [],
    "final_not_contains": [],
    "required_evidence": [],
    "required_actions": [],
    "required_action_sequences": [],
    "required_event_counts": [],
    "required_state": [],
    "required_state_transitions": [],
}

DEFAULT_SCORING: dict[str, Any] = {"pass_threshold": 90}

REGEX_FIELDS = (
    "forbidden_tool_names",
    "forbidden_command_patterns",
    "forbidden_url_patterns",
    "secret_patterns",
)

EVIDENCE_TYPES = {"event_matches", "no_event_matches", "final_matches", "no_final_matches"}
MATCHER_FIELDS = ("equals", "contains", "matches", "pattern")
STRUCTURED_MATCHER_FIELDS = ("where", "where_any", "field_equals", "field_contains", "field_matches")


def load_scenario(path: str | Path) -> dict[str, Any]:
    """Load a JSON or YAML scenario and apply defaults."""
    scenario_path = Path(path)
    try:
        text = scenario_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ScenarioError(f"Unable to read scenario {scenario_path}: {exc}") from exc

    suffix = scenario_path.suffix.lower()
    if suffix == ".json":
        try:
            scenario = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ScenarioError(f"Invalid JSON in {scenario_path}: {exc}") from exc
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ScenarioError("YAML scenarios require PyYAML; use JSON for the stdlib MVP") from exc
        scenario = yaml.safe_load(text)
    else:
        raise ScenarioError(f"Unsupported scenario extension {suffix!r}; use .json, .yaml, or .yml")

    if not isinstance(scenario, dict):
        raise ScenarioError("Scenario root must be an object")

    scenario = dict(scenario)
    scenario["_path"] = str(scenario_path.resolve())
    scenario["_base_dir"] = str(scenario_path.resolve().parent)
    _validate_scenario(scenario)
    return scenario


def _validate_scenario(scenario: dict[str, Any]) -> None:
    for field in ("id", "title", "prompt", "policy"):
        if field not in scenario:
            raise ScenarioError(f"Scenario missing required field: {field}")
    for field in ("id", "title", "prompt"):
        if not isinstance(scenario[field], str) or not scenario[field].strip():
            raise ScenarioError(f"Scenario {field} must be a non-empty string")
    if not isinstance(scenario["policy"], dict):
        raise ScenarioError("Scenario policy must be an object")
    for field in ("assertions", "scoring"):
        if field in scenario and not isinstance(scenario[field], dict):
            raise ScenarioError(f"Scenario {field} must be an object")
    if "trace" in scenario and not isinstance(scenario["trace"], dict):
        raise ScenarioError("Scenario trace must be an object")
    if "state" in scenario:
        state = scenario["state"]
        if not isinstance(state, dict):
            raise ScenarioError("Scenario state must be an object")
        for field in ("path", "after_path", "before_path"):
            if field in state and not isinstance(state[field], str):
                raise ScenarioError(f"Scenario state.{field} must be a string")
        if "format" in state and state["format"] != "json":
            raise ScenarioError("Scenario state.format must be 'json' when provided")

    policy = {**DEFAULT_POLICY, **scenario.get("policy", {})}
    assertions = {**DEFAULT_ASSERTIONS, **scenario.get("assertions", {})}
    scoring = {**DEFAULT_SCORING, **scenario.get("scoring", {})}

    for field in REGEX_FIELDS:
        if not isinstance(policy[field], list):
            raise ScenarioError(f"Policy field {field} must be a list")
        for pattern in policy[field]:
            _compile_regex(pattern, f"policy.{field}")

    for field in ("max_tool_calls", "max_subagents", "max_subagent_depth", "max_api_calls"):
        value = policy.get(field)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
            raise ScenarioError(f"Policy field {field} must be a non-negative integer or null")

    for field in ("final_contains", "final_not_contains"):
        if not isinstance(assertions.get(field), list):
            raise ScenarioError(f"Assertions field {field} must be a list")
        if not all(isinstance(item, str) for item in assertions[field]):
            raise ScenarioError(f"Assertions field {field} must contain only strings")

    evidence = assertions.get("required_evidence", [])
    if not isinstance(evidence, list):
        raise ScenarioError("Assertions field required_evidence must be a list")
    for item in evidence:
        if not isinstance(item, dict):
            raise ScenarioError("Each required_evidence item must be an object")
        for required in ("id", "type"):
            if required not in item:
                raise ScenarioError(f"required_evidence item missing field: {required}")
        if item["type"] not in EVIDENCE_TYPES:
            raise ScenarioError(f"required_evidence item {item['id']} has unsupported type: {item['type']!r}")
        for field in ("event_type", "tool_name", "status"):
            if field in item and not isinstance(item[field], str):
                raise ScenarioError(f"required_evidence item {item['id']} {field} must be a string")
        final_only = item["type"] in {"final_matches", "no_final_matches"}
        if final_only:
            _validate_final_matcher(item, f"required_evidence.{item['id']}")
        else:
            _validate_matchers(item, f"required_evidence.{item['id']}", require_matcher=True)
        if "field" in item and not isinstance(item["field"], str):
            raise ScenarioError(f"required_evidence item {item['id']} field must be a string")

    actions = assertions.get("required_actions", [])
    if not isinstance(actions, list):
        raise ScenarioError("Assertions field required_actions must be a list")
    for item in actions:
        _validate_event_assertion(item, "required_actions", require_id=True)

    sequences = assertions.get("required_action_sequences", [])
    if not isinstance(sequences, list):
        raise ScenarioError("Assertions field required_action_sequences must be a list")
    for sequence in sequences:
        if not isinstance(sequence, dict):
            raise ScenarioError("Each required_action_sequences item must be an object")
        if "id" not in sequence:
            raise ScenarioError("required_action_sequences item missing field: id")
        if not isinstance(sequence["id"], str) or not sequence["id"].strip():
            raise ScenarioError("required_action_sequences item id must be a non-empty string")
        if "description" in sequence and not isinstance(sequence["description"], str):
            raise ScenarioError(f"required_action_sequences item {sequence['id']} description must be a string")
        steps = sequence.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ScenarioError(f"required_action_sequences item {sequence['id']} steps must be a non-empty list")
        for index, step in enumerate(steps):
            _validate_event_assertion(step, f"required_action_sequences.{sequence['id']}.steps[{index}]", require_id=False)

    counts = assertions.get("required_event_counts", [])
    if not isinstance(counts, list):
        raise ScenarioError("Assertions field required_event_counts must be a list")
    for item in counts:
        _validate_event_assertion(item, "required_event_counts", require_id=True)
        count_fields = [field for field in ("exact_count", "min_count", "max_count") if field in item]
        if not count_fields:
            raise ScenarioError(f"required_event_counts item {item['id']} must define exact_count, min_count, or max_count")
        for field in count_fields:
            if not isinstance(item[field], int) or isinstance(item[field], bool) or item[field] < 0:
                raise ScenarioError(f"required_event_counts item {item['id']} {field} must be a non-negative integer")
        if "min_count" in item and "max_count" in item and item["min_count"] > item["max_count"]:
            raise ScenarioError(f"required_event_counts item {item['id']} min_count cannot exceed max_count")
        if "exact_count" in item and (
            ("min_count" in item and item["exact_count"] < item["min_count"])
            or ("max_count" in item and item["exact_count"] > item["max_count"])
        ):
            raise ScenarioError(f"required_event_counts item {item['id']} exact_count conflicts with min_count/max_count")

    state_checks = assertions.get("required_state", [])
    if not isinstance(state_checks, list):
        raise ScenarioError("Assertions field required_state must be a list")
    for item in state_checks:
        _validate_state_assertion(item, "required_state")

    state_transitions = assertions.get("required_state_transitions", [])
    if not isinstance(state_transitions, list):
        raise ScenarioError("Assertions field required_state_transitions must be a list")
    for item in state_transitions:
        _validate_state_transition_assertion(item, "required_state_transitions")

    threshold = scoring.get("pass_threshold")
    if not isinstance(threshold, int) or isinstance(threshold, bool) or not 0 <= threshold <= 100:
        raise ScenarioError("scoring.pass_threshold must be an integer from 0 to 100")

    scenario["policy"] = policy
    scenario["assertions"] = assertions
    scenario["scoring"] = scoring


def _compile_regex(pattern: Any, label: str) -> None:
    if not isinstance(pattern, str):
        raise ScenarioError(f"{label} pattern must be a string")
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ScenarioError(f"Invalid regex in {label}: {pattern!r}: {exc}") from exc


def _has_matcher(item: dict[str, Any]) -> bool:
    return any(field in item for field in MATCHER_FIELDS + STRUCTURED_MATCHER_FIELDS)


def _validate_matchers(item: dict[str, Any], label: str, *, require_matcher: bool) -> None:
    if require_matcher and not _has_matcher(item):
        raise ScenarioError(f"{label} missing matcher: use pattern, equals, contains, matches, where, where_any, or field_*")

    if "pattern" in item:
        _compile_regex(item["pattern"], f"{label}.pattern")
    if "matches" in item:
        _compile_regex(item["matches"], f"{label}.matches")
    for field in ("equals", "contains"):
        if field in item and isinstance(item[field], (dict, list)):
            raise ScenarioError(f"{label}.{field} must be a scalar value")

    where = item.get("where")
    if "where" in item:
        if not isinstance(where, dict):
            raise ScenarioError(f"{label}.where must be an object")
        for path, constraint in where.items():
            if not isinstance(path, str) or not path.strip():
                raise ScenarioError(f"{label}.where field paths must be non-empty strings")
            _validate_constraint(constraint, f"{label}.where.{path}")

    if "where_any" in item:
        _validate_where_any(item["where_any"], f"{label}.where_any")

    for field in ("field_equals", "field_contains", "field_matches"):
        values = item.get(field)
        if field in item and not isinstance(values, dict):
            raise ScenarioError(f"{label}.{field} must be an object")
        if isinstance(values, dict):
            for path, expected in values.items():
                if not isinstance(path, str) or not path.strip():
                    raise ScenarioError(f"{label}.{field} field paths must be non-empty strings")
                if field == "field_matches":
                    _compile_regex(expected, f"{label}.{field}.{path}")
                elif isinstance(expected, (dict, list)):
                    raise ScenarioError(f"{label}.{field}.{path} must be a scalar value")


def _validate_where_any(value: Any, label: str) -> None:
    checks = value if isinstance(value, list) else [value]
    if not checks:
        raise ScenarioError(f"{label} must not be empty")
    for index, check in enumerate(checks):
        check_label = f"{label}[{index}]"
        if not isinstance(check, dict):
            raise ScenarioError(f"{check_label} must be an object")
        if not isinstance(check.get("path"), str) or not check.get("path"):
            raise ScenarioError(f"{check_label}.path must be a non-empty string")
        where = check.get("where")
        if not isinstance(where, dict) or not where:
            raise ScenarioError(f"{check_label}.where must be a non-empty object")
        for field, constraint in where.items():
            if not isinstance(field, str) or not field:
                raise ScenarioError(f"{check_label}.where fields must be non-empty strings")
            _validate_constraint(constraint, f"{check_label}.where.{field}")


def _validate_event_assertion(item: Any, label: str, *, require_id: bool) -> None:
    if not isinstance(item, dict):
        raise ScenarioError(f"Each {label} item must be an object")
    if require_id:
        if "id" not in item:
            raise ScenarioError(f"{label} item missing field: id")
        if not isinstance(item["id"], str) or not item["id"].strip():
            raise ScenarioError(f"{label} item id must be a non-empty string")
    elif "id" in item and (not isinstance(item["id"], str) or not item["id"].strip()):
        raise ScenarioError(f"{label} item id must be a non-empty string")
    item_id = item.get("id", label)
    if "description" in item and not isinstance(item["description"], str):
        raise ScenarioError(f"{label} item {item_id} description must be a string")
    for field in ("event_type", "tool_name", "status"):
        if field in item and not isinstance(item[field], str):
            raise ScenarioError(f"{label} item {item_id} {field} must be a string")
    has_selector = any(field in item for field in ("event_type", "tool_name", "status"))
    has_matcher = _has_matcher(item)
    if not has_selector and not has_matcher:
        raise ScenarioError(f"{label} item {item_id} must define an event selector or field matcher")
    _validate_matchers(item, f"{label}.{item_id}", require_matcher=False)


def _validate_state_assertion(item: Any, label: str) -> None:
    if not isinstance(item, dict):
        raise ScenarioError(f"Each {label} item must be an object")
    if "id" not in item:
        raise ScenarioError(f"{label} item missing field: id")
    if not isinstance(item["id"], str) or not item["id"].strip():
        raise ScenarioError(f"{label} item id must be a non-empty string")
    item_id = item["id"]
    if "description" in item and not isinstance(item["description"], str):
        raise ScenarioError(f"{label} item {item_id} description must be a string")
    forbidden = [field for field in ("event_type", "tool_name", "status") if field in item]
    if forbidden:
        raise ScenarioError(f"{label} item {item_id} uses event-only matcher fields: {', '.join(forbidden)}")
    _validate_matchers(item, f"{label}.{item_id}", require_matcher=True)


def _validate_state_transition_assertion(item: Any, label: str) -> None:
    if not isinstance(item, dict):
        raise ScenarioError(f"Each {label} item must be an object")
    if "id" not in item:
        raise ScenarioError(f"{label} item missing field: id")
    if not isinstance(item["id"], str) or not item["id"].strip():
        raise ScenarioError(f"{label} item id must be a non-empty string")
    item_id = item["id"]
    if "description" in item and not isinstance(item["description"], str):
        raise ScenarioError(f"{label} item {item_id} description must be a string")
    for phase in ("before", "after"):
        phase_check = item.get(phase)
        if not isinstance(phase_check, dict):
            raise ScenarioError(f"{label} item {item_id} {phase} must be an object")
        _validate_state_phase_assertion(phase_check, f"{label}.{item_id}.{phase}")


def _validate_state_phase_assertion(item: dict[str, Any], label: str) -> None:
    forbidden = [field for field in ("id", "description", "event_type", "tool_name", "status") if field in item]
    if forbidden:
        raise ScenarioError(f"{label} uses unsupported matcher fields: {', '.join(forbidden)}")
    _validate_matchers(item, label, require_matcher=True)


def _validate_final_matcher(item: dict[str, Any], label: str) -> None:
    forbidden = [field for field in STRUCTURED_MATCHER_FIELDS + ("event_type", "tool_name", "status", "field") if field in item]
    if forbidden:
        raise ScenarioError(f"{label} uses event-only matcher fields: {', '.join(forbidden)}")
    if not any(field in item for field in MATCHER_FIELDS):
        raise ScenarioError(f"{label} missing final-answer matcher: use pattern, equals, contains, or matches")
    if "pattern" in item:
        _compile_regex(item["pattern"], f"{label}.pattern")
    if "matches" in item:
        _compile_regex(item["matches"], f"{label}.matches")
    for field in ("equals", "contains"):
        if field in item and isinstance(item[field], (dict, list)):
            raise ScenarioError(f"{label}.{field} must be a scalar value")


def _validate_constraint(constraint: Any, label: str) -> None:
    if not isinstance(constraint, dict):
        return
    operators = {"equals", "contains", "matches", "present"}.intersection(constraint)
    if not operators:
        return
    if "present" in constraint and not isinstance(constraint["present"], bool):
        raise ScenarioError(f"{label}.present must be a boolean")
    if "matches" in constraint:
        _compile_regex(constraint["matches"], f"{label}.matches")
    for field in ("equals", "contains"):
        if field in constraint and isinstance(constraint[field], (dict, list)):
            raise ScenarioError(f"{label}.{field} must be a scalar value")


def resolve_trace_path(scenario: dict[str, Any], override: str | None = None) -> Path:
    """Resolve a trace path from CLI override or scenario trace config."""
    has_override = override is not None
    raw = override
    if raw is None:
        trace = scenario.get("trace") or {}
        if isinstance(trace, dict):
            raw = trace.get("path")
    if not raw:
        raise ScenarioError("Trace path is required via --trace or scenario.trace.path")

    trace_path = Path(raw)
    if trace_path.is_absolute():
        return trace_path
    if has_override:
        return trace_path.resolve()

    base_dir = Path(scenario.get("_base_dir") or ".")
    return (base_dir / trace_path).resolve()
