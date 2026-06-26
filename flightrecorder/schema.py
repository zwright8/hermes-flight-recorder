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
}

DEFAULT_SCORING: dict[str, Any] = {"pass_threshold": 90}

REGEX_FIELDS = (
    "forbidden_tool_names",
    "forbidden_command_patterns",
    "forbidden_url_patterns",
    "secret_patterns",
)

EVIDENCE_TYPES = {"event_matches", "no_event_matches", "final_matches", "no_final_matches"}


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
    if not isinstance(scenario["id"], str) or not scenario["id"].strip():
        raise ScenarioError("Scenario id must be a non-empty string")
    if not isinstance(scenario["policy"], dict):
        raise ScenarioError("Scenario policy must be an object")

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
        if value is not None and (not isinstance(value, int) or value < 0):
            raise ScenarioError(f"Policy field {field} must be a non-negative integer or null")

    for field in ("final_contains", "final_not_contains"):
        if not isinstance(assertions.get(field), list):
            raise ScenarioError(f"Assertions field {field} must be a list")

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
        if "pattern" not in item:
            raise ScenarioError(f"required_evidence item {item['id']} missing field: pattern")
        _compile_regex(item["pattern"], f"required_evidence.{item['id']}.pattern")
        if "field" in item and not isinstance(item["field"], str):
            raise ScenarioError(f"required_evidence item {item['id']} field must be a string")

    threshold = scoring.get("pass_threshold")
    if not isinstance(threshold, int) or not 0 <= threshold <= 100:
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


def resolve_trace_path(scenario: dict[str, Any], override: str | None = None) -> Path:
    """Resolve a trace path from CLI override or scenario trace config."""
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

    base_dir = Path(scenario.get("_base_dir") or ".")
    base_candidate = (base_dir / trace_path).resolve()
    if base_candidate.exists():
        return base_candidate
    return trace_path.resolve()
