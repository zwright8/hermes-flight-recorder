"""Trace adapters for Hermes artifacts."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

TRACE_SCHEMA_VERSION = "hfr.trace.v1"

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class AdapterError(ValueError):
    """Raised when a trace cannot be normalized."""


def normalize_trace(path: str | Path, fmt: str = "auto") -> dict[str, Any]:
    """Normalize a trace artifact into the hfr.trace.v1 shape."""
    trace_path = Path(path)
    if not trace_path.exists():
        raise AdapterError(f"Trace file not found: {trace_path}")

    selected = fmt
    if fmt == "auto":
        selected = detect_format(trace_path)

    if selected == "trajectory_jsonl":
        return normalize_trajectory_jsonl(trace_path)
    if selected == "observer_jsonl":
        return normalize_observer_jsonl(trace_path)
    if selected == "atof_jsonl":
        return normalize_atof_jsonl(trace_path)
    if selected == "atif_json":
        return normalize_atif_json(trace_path)
    if selected == "normalized_json":
        return json.loads(trace_path.read_text(encoding="utf-8"))
    raise AdapterError(f"Unsupported trace format: {selected}")


def detect_format(path: Path) -> str:
    """Best-effort trace format detection."""
    first = _first_json(path)
    if isinstance(first, dict):
        if first.get("schema_version") == TRACE_SCHEMA_VERSION:
            return "normalized_json"
        if "conversations" in first:
            return "trajectory_jsonl"
        if first.get("schema_version", "").startswith("ATIF") or "steps" in first:
            return "atif_json"
        hook = first.get("hook") or first.get("event") or first.get("name")
        if hook in OBSERVER_HOOKS:
            return "observer_jsonl"
        if first.get("kind") in {"scope", "mark"}:
            return "atof_jsonl"
    raise AdapterError(f"Unable to detect trace format for {path}")


def normalize_trajectory_jsonl(path: Path) -> dict[str, Any]:
    entries = _read_jsonl(path)
    if not entries:
        raise AdapterError(f"No JSONL entries in {path}")

    events: list[dict[str, Any]] = []
    final_answer = ""
    model = "unknown"
    completed = None
    api_calls = None
    session_id = path.stem

    for entry_index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        model = entry.get("model") or model
        completed = entry.get("completed", completed)
        api_calls = entry.get("api_calls", api_calls)
        current_session = entry.get("session_id") or f"{path.stem}-{entry_index + 1}"
        conversations = entry.get("conversations") or []
        if not isinstance(conversations, list):
            continue
        for turn_index, turn in enumerate(conversations):
            if not isinstance(turn, dict):
                continue
            role = turn.get("from")
            value = str(turn.get("value") or "")
            if role == "human":
                events.append(_event("user_message", current_session, text=value, order=len(events)))
            elif role == "gpt":
                for call in _extract_json_blocks(value, TOOL_CALL_RE):
                    events.append(
                        _event(
                            "tool_call",
                            current_session,
                            tool_name=str(call.get("name") or "unknown"),
                            args=call.get("arguments") if isinstance(call.get("arguments"), dict) else {},
                            status="requested",
                            text="",
                            order=len(events),
                        )
                    )
                clean = _strip_assistant_markup(value).strip()
                if clean:
                    final_answer = clean
                    events.append(_event("assistant_message", current_session, text=clean, order=len(events)))
            elif role == "tool":
                for response in _extract_json_blocks(value, TOOL_RESPONSE_RE):
                    content = response.get("content")
                    events.append(
                        _event(
                            "tool_result",
                            current_session,
                            tool_name=str(response.get("name") or "unknown"),
                            tool_call_id=response.get("tool_call_id"),
                            status="ok",
                            text=_stringify(content),
                            result=content,
                            order=len(events),
                        )
                    )
            elif role == "system":
                events.append(_event("system_message", current_session, text=value, order=len(events)))

    metadata: dict[str, Any] = {"completed": completed}
    if api_calls is not None:
        metadata["api_calls"] = api_calls
    if isinstance(entries[-1], dict) and isinstance(entries[-1].get("tool_stats"), dict):
        metadata["tool_stats"] = entries[-1]["tool_stats"]

    return _trace(session_id, "trajectory_jsonl", model, events, final_answer, metadata)


OBSERVER_HOOKS = {
    "pre_tool_call",
    "post_tool_call",
    "pre_llm_call",
    "post_llm_call",
    "pre_api_request",
    "post_api_request",
    "api_request_error",
    "pre_approval_request",
    "post_approval_response",
    "subagent_start",
    "subagent_stop",
    "on_session_start",
    "on_session_end",
    "on_session_finalize",
    "on_session_reset",
}


def normalize_observer_jsonl(path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path)
    events: list[dict[str, Any]] = []
    final_answer = ""
    model = "unknown"
    session_id = path.stem

    for row in rows:
        if not isinstance(row, dict):
            continue
        hook = row.get("hook") or row.get("event") or row.get("name")
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else row
        session_id = payload.get("session_id") or payload.get("parent_session_id") or session_id
        model = payload.get("model") or model
        if hook == "pre_tool_call":
            events.append(
                _event(
                    "tool_call",
                    payload.get("session_id") or session_id,
                    tool_name=str(payload.get("tool_name") or "unknown"),
                    args=payload.get("args") if isinstance(payload.get("args"), dict) else {},
                    status="requested",
                    tool_call_id=payload.get("tool_call_id"),
                    text="",
                    order=len(events),
                )
            )
        elif hook == "post_tool_call":
            result = payload.get("result")
            events.append(
                _event(
                    "tool_result",
                    payload.get("session_id") or session_id,
                    tool_name=str(payload.get("tool_name") or "unknown"),
                    status=payload.get("status") or "unknown",
                    tool_call_id=payload.get("tool_call_id"),
                    text=_stringify(result or payload.get("error_message") or ""),
                    result=result,
                    order=len(events),
                )
            )
        elif hook == "post_llm_call":
            final_answer = str(payload.get("assistant_response") or payload.get("output") or final_answer)
            events.append(_event("assistant_message", payload.get("session_id") or session_id, text=final_answer, order=len(events)))
        elif hook == "pre_llm_call":
            events.append(_event("user_message", payload.get("session_id") or session_id, text=str(payload.get("user_message") or ""), order=len(events)))
        elif hook == "subagent_start":
            events.append(_subagent_event("subagent_start", payload, len(events)))
        elif hook == "subagent_stop":
            events.append(_subagent_event("subagent_stop", payload, len(events)))
        elif hook in {"pre_approval_request", "post_approval_response"}:
            events.append(_event("approval", payload.get("session_id") or session_id, text=_stringify(payload), status=payload.get("choice"), order=len(events)))
        elif hook in {"pre_api_request", "post_api_request", "api_request_error"}:
            events.append(_event("api_call", payload.get("session_id") or session_id, text=hook, status=payload.get("status_code") or payload.get("finish_reason"), order=len(events)))

    return _trace(session_id, "observer_jsonl", model, events, final_answer, {"completed": None})


def normalize_atof_jsonl(path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path)
    events: list[dict[str, Any]] = []
    session_id = path.stem
    for row in rows:
        if not isinstance(row, dict):
            continue
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        data = row.get("data")
        session_id = metadata.get("session_id") or session_id
        if row.get("category") == "tool":
            events.append(
                _event(
                    "tool_call" if row.get("scope_category") == "start" else "tool_result",
                    session_id,
                    tool_name=str(row.get("name") or "unknown"),
                    status=metadata.get("status") or row.get("scope_category"),
                    args=data if isinstance(data, dict) else {},
                    text=_stringify(data),
                    order=len(events),
                )
            )
        elif str(row.get("name", "")).startswith("hermes.subagent"):
            events.append(_event(str(row.get("name")), session_id, text=_stringify(row), order=len(events)))
    return _trace(session_id, "atof_jsonl", "unknown", events, "", {"completed": None})


def normalize_atif_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    events: list[dict[str, Any]] = []
    session_id = data.get("session_id") or path.stem
    final_answer = ""
    for step in data.get("steps", []):
        if not isinstance(step, dict):
            continue
        for call in step.get("tool_calls", []) or []:
            if isinstance(call, dict):
                events.append(
                    _event(
                        "tool_call",
                        session_id,
                        tool_name=str(call.get("function_name") or call.get("name") or "unknown"),
                        args=call.get("arguments") if isinstance(call.get("arguments"), dict) else {},
                        status="requested",
                        order=len(events),
                    )
                )
        if step.get("message"):
            final_answer = str(step["message"])
            events.append(_event("assistant_message", session_id, text=final_answer, order=len(events)))
        if "observation" in step:
            events.append(_event("tool_result", session_id, text=_stringify(step["observation"]), status="ok", order=len(events)))
    agent = data.get("agent") if isinstance(data.get("agent"), dict) else {}
    return _trace(session_id, "atif_json", agent.get("model_name") or "unknown", events, final_answer, {"completed": None})


def _trace(
    session_id: str,
    source_format: str,
    model: str,
    events: list[dict[str, Any]],
    final_answer: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "session": {"id": session_id, "source_format": source_format, "model": model},
        "events": events,
        "final_answer": final_answer,
        "metadata": metadata,
    }


def _event(event_type: str, session_id: str, **kwargs: Any) -> dict[str, Any]:
    event = {
        "type": event_type,
        "session_id": session_id,
        "parent_session_id": kwargs.pop("parent_session_id", None),
        "tool_name": kwargs.pop("tool_name", None),
        "args": kwargs.pop("args", {}),
        "status": kwargs.pop("status", None),
        "text": kwargs.pop("text", ""),
        "timestamp": kwargs.pop("timestamp", None),
    }
    event.update(kwargs)
    return event


def _subagent_event(event_type: str, payload: dict[str, Any], order: int) -> dict[str, Any]:
    return _event(
        event_type,
        payload.get("child_session_id") or payload.get("session_id") or payload.get("parent_session_id") or "unknown",
        parent_session_id=payload.get("parent_session_id"),
        status=payload.get("status"),
        text=str(payload.get("child_summary") or payload.get("child_goal") or ""),
        child_session_id=payload.get("child_session_id"),
        child_subagent_id=payload.get("child_subagent_id"),
        child_role=payload.get("child_role"),
        order=order,
    )


def _read_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rows.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise AdapterError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def _first_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        raise AdapterError(f"Trace file is empty: {path}")
    if path.suffix.lower() == ".json":
        return json.loads(stripped)
    for line in stripped.splitlines():
        if line.strip():
            return json.loads(line)
    raise AdapterError(f"Trace file is empty: {path}")


def _extract_json_blocks(text: str, regex: re.Pattern[str]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for match in regex.finditer(text):
        raw = match.group(1).strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                parsed = {}
        if isinstance(parsed, dict):
            blocks.append(parsed)
    return blocks


def _strip_assistant_markup(text: str) -> str:
    text = THINK_RE.sub("", text)
    text = TOOL_CALL_RE.sub("", text)
    return text.strip()


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
