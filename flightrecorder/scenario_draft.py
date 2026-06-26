"""Draft scenario definitions from existing trace evidence."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from .scorers import score_trace

DRAFT_SCHEMA_VERSION = "hfr.scenario_draft.v1"
DEFAULT_SECRET_PATTERN = "(?i)(api[_-]?key|secret|token|password|authorization|bearer)"
DEFAULT_FINAL_NOT_CONTAINS = ["probably", "should be", "I think"]

_SENSITIVE_KEY_RE = re.compile(r"(?i)(api[_-]?key|secret|token|password|authorization|bearer)")
_TEXTY_KEY_RE = re.compile(r"(?i)^(body|content|contents|description|html|message|messages|summary|text)$")
_PREFERRED_KEY_RE = re.compile(r"(?i)(^id$|_id$|^status$|^state$|^path$|^url$|^name$|^file(name)?$)")
_SEQUENCE_STEP_FIELDS = {
    "event_type",
    "tool_name",
    "status",
    "where",
    "field_equals",
    "field_contains",
    "field_matches",
    "field",
    "equals",
    "contains",
    "matches",
    "pattern",
}


def draft_scenario(
    trace: dict[str, Any],
    *,
    scenario_id: str,
    title: str,
    prompt: str,
    trace_path: str | Path | None = None,
    trace_format: str = "auto",
    out_path: str | Path | None = None,
    max_actions: int = 8,
    preserve_paths: bool = False,
) -> dict[str, Any]:
    """Build a conservative scenario draft from a normalized trace."""
    events = trace.get("events") if isinstance(trace.get("events"), list) else []
    warnings: list[str] = [
        "Drafted from one observed run; review and tighten the task-specific assertions before treating it as a benchmark.",
        "Only observable trace evidence can be checked; external side effects still need tool or observer evidence.",
    ]
    required_actions, action_warnings = _required_actions(events, max_actions)
    required_action_sequences = _required_action_sequences(required_actions)
    warnings.extend(action_warnings)

    required_evidence: list[dict[str, Any]] = []
    if trace.get("final_answer"):
        required_evidence.append({"id": "final_answer_present", "type": "final_matches", "matches": "\\S"})
    elif not required_actions:
        warnings.append("No final answer or successful tool-result actions were found; the draft has weak completion evidence.")

    scenario: dict[str, Any] = {
        "id": scenario_id,
        "title": title,
        "prompt": prompt,
        "policy": {
            "secret_patterns": [DEFAULT_SECRET_PATTERN],
            "max_tool_calls": _count_events(events, "tool_call"),
            "max_subagents": _count_events(events, "subagent_start"),
            "max_subagent_depth": _max_subagent_depth(events),
        },
        "assertions": {
            "required_actions": required_actions,
            "required_action_sequences": required_action_sequences,
            "required_evidence": required_evidence,
            "final_not_contains": DEFAULT_FINAL_NOT_CONTAINS,
        },
        "scoring": {"pass_threshold": 90},
        "draft": {
            "schema_version": DRAFT_SCHEMA_VERSION,
            "source_format": trace.get("session", {}).get("source_format", "unknown"),
            "model": trace.get("session", {}).get("model", "unknown"),
            "warnings": warnings,
        },
    }

    api_calls = _api_call_count(trace, events)
    if api_calls is not None:
        scenario["policy"]["max_api_calls"] = api_calls
    if trace_path is not None:
        scenario["trace"] = {
            "format": trace_format,
            "path": _trace_ref(trace_path, out_path, preserve_paths),
        }
    return scenario


def score_draft(scenario: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    """Score a draft against its source trace."""
    return score_trace(scenario, trace)


def safe_scenario_id(value: str) -> str:
    """Return a scenario-safe identifier."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-").lower()
    return cleaned or "draft_scenario"


def title_from_id(scenario_id: str) -> str:
    return scenario_id.replace("_", " ").replace("-", " ").title()


def _required_actions(events: list[Any], max_actions: int) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    candidates = [
        (index, event)
        for index, event in enumerate(events)
        if isinstance(event, dict)
        and event.get("type") == "tool_result"
        and event.get("tool_name")
        and event.get("tool_name") != "unknown"
        and _successful_status(event.get("status"))
    ]
    if max_actions >= 0 and len(candidates) > max_actions:
        warnings.append(f"Only the first {max_actions} of {len(candidates)} tool-result events were drafted as required actions.")
        candidates = candidates[:max_actions]

    actions: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for sequence, (index, event) in enumerate(candidates, start=1):
        tool_name = str(event.get("tool_name") or "tool")
        action_id = _unique_id(f"observed_{safe_scenario_id(tool_name)}_{sequence}", used_ids)
        action: dict[str, Any] = {
            "id": action_id,
            "description": f"Observed successful {tool_name} result from source trace event #{index}",
            "event_type": "tool_result",
            "tool_name": tool_name,
        }
        if event.get("status"):
            action["status"] = str(event["status"])
        where = _safe_where_matchers(event)
        if where:
            action["where"] = where
        actions.append(action)

    if not actions:
        warnings.append("No tool-result events were available for required_actions; add task-specific evidence manually.")
    return actions, warnings


def _required_action_sequences(required_actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(required_actions) < 2:
        return []
    steps: list[dict[str, Any]] = []
    for index, action in enumerate(required_actions, start=1):
        step = {
            key: value
            for key, value in action.items()
            if key in _SEQUENCE_STEP_FIELDS
        }
        step["id"] = f"step_{index}_{safe_scenario_id(str(action.get('tool_name') or action.get('id') or 'event'))}"
        step["description"] = str(action.get("description") or step["id"])
        steps.append(step)
    return [
        {
            "id": "observed_successful_tool_result_order",
            "description": "Successful tool-result events occurred in the observed source-trace order",
            "steps": steps,
        }
    ]


def _safe_where_matchers(event: dict[str, Any]) -> dict[str, Any]:
    matchers: dict[str, Any] = {}
    preferred: list[tuple[str, Any]] = []
    fallback: list[tuple[str, Any]] = []
    for root in ("result", "args"):
        value = event.get(root)
        if isinstance(value, dict):
            _collect_scalar_paths(value, root, preferred, fallback)
    selected = preferred[:4] or fallback[:2]
    for path, value in selected:
        matchers[path] = value
    return matchers


def _collect_scalar_paths(
    value: dict[str, Any],
    prefix: str,
    preferred: list[tuple[str, Any]],
    fallback: list[tuple[str, Any]],
) -> None:
    for key, item in value.items():
        path = f"{prefix}.{key}"
        if _skip_field(key, item):
            continue
        if isinstance(item, dict):
            _collect_scalar_paths(item, path, preferred, fallback)
        elif isinstance(item, (str, int, float, bool)) or item is None:
            target = preferred if _PREFERRED_KEY_RE.search(str(key)) else fallback
            target.append((path, item))
        elif isinstance(item, list):
            for index, child in enumerate(item[:3]):
                child_key = f"{key}.{index}"
                if _skip_field(child_key, child):
                    continue
                if isinstance(child, (str, int, float, bool)) or child is None:
                    target = preferred if _PREFERRED_KEY_RE.search(str(key)) else fallback
                    target.append((f"{path}.{index}", child))


def _skip_field(key: str, value: Any) -> bool:
    if _SENSITIVE_KEY_RE.search(key) or _TEXTY_KEY_RE.search(key):
        return True
    if isinstance(value, str):
        if len(value) > 80 or "\n" in value:
            return True
        if _SENSITIVE_KEY_RE.search(value):
            return True
    return False


def _successful_status(status: Any) -> bool:
    if status is None:
        return True
    rendered = str(status).lower()
    if rendered in {"", "ok", "success", "succeeded", "complete", "completed", "done", "end"}:
        return True
    if rendered.isdigit():
        return 200 <= int(rendered) < 400
    return False


def _count_events(events: list[Any], event_type: str) -> int:
    return sum(1 for event in events if isinstance(event, dict) and event.get("type") == event_type)


def _api_call_count(trace: dict[str, Any], events: list[Any]) -> int | None:
    metadata_api_calls = trace.get("metadata", {}).get("api_calls")
    if isinstance(metadata_api_calls, int) and not isinstance(metadata_api_calls, bool):
        return metadata_api_calls
    count = _count_events(events, "api_call")
    return count if count else None


def _max_subagent_depth(events: list[Any]) -> int:
    child_to_parent: dict[str, str] = {}
    children: set[str] = set()
    for event in events:
        if not isinstance(event, dict) or event.get("type") != "subagent_start":
            continue
        child = event.get("child_session_id") or event.get("session_id")
        parent = event.get("parent_session_id")
        if child:
            children.add(str(child))
        if child and parent:
            child_to_parent[str(child)] = str(parent)

    max_depth = 0
    for child in children:
        depth = 1
        cursor = child
        seen = {cursor}
        while child_to_parent.get(cursor) and child_to_parent[cursor] not in seen:
            cursor = child_to_parent[cursor]
            seen.add(cursor)
            if cursor in children:
                depth += 1
        max_depth = max(max_depth, depth)
    return max_depth


def _trace_ref(trace_path: str | Path, out_path: str | Path | None, preserve_paths: bool) -> str:
    path = Path(trace_path)
    if preserve_paths or out_path is None:
        return str(path)
    try:
        return os.path.relpath(path.resolve(), Path(out_path).resolve().parent).replace(os.sep, "/")
    except OSError:
        return str(path)


def _unique_id(candidate: str, used: set[str]) -> str:
    if candidate not in used:
        used.add(candidate)
        return candidate
    suffix = 2
    while f"{candidate}_{suffix}" in used:
        suffix += 1
    unique = f"{candidate}_{suffix}"
    used.add(unique)
    return unique
