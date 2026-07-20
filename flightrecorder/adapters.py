"""Trace adapters for agent run artifacts."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from .trajectory_v2 import (
    TRAJECTORY_V2_SCHEMA_VERSION,
    trajectory_v2_context_from_path,
    trajectory_v2_from_trace,
)

TRACE_SCHEMA_VERSION = "hfr.trace.v1"
OPENCLAW_EVENT_SCHEMA_VERSION = "hfr.openclaw.event.v1"
COVEN_EVENT_SCHEMA_VERSION = "hfr.coven.event.v1"

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

OPENCLAW_HOOKS = {
    "agent_end",
    "agent_turn_prepare",
    "after_tool_call",
    "before_agent_finalize",
    "before_agent_reply",
    "before_agent_run",
    "before_agent_start",
    "before_model_resolve",
    "before_prompt_build",
    "before_tool_call",
    "llm_input",
    "llm_output",
    "model_call_ended",
    "model_call_started",
    "session_end",
    "session_start",
    "subagent_ended",
    "subagent_spawned",
    "subagent_spawning",
}

COVEN_STREAM_EVENT_TYPES = {"system", "user", "assistant", "tool_result", "result"}
COVEN_EVENT_KINDS = {
    "error",
    "exit",
    "input",
    "kill",
    "metadata",
    "output",
    "patch_metadata",
    "tool_call",
    "tool_result",
}


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
    if selected == "openclaw_jsonl":
        return normalize_openclaw_jsonl(trace_path)
    if selected == "coven_jsonl":
        return normalize_coven_jsonl(trace_path)
    if selected == "atof_jsonl":
        return normalize_atof_jsonl(trace_path)
    if selected == "atif_json":
        return normalize_atif_json(trace_path)
    if selected == "normalized_json":
        return json.loads(trace_path.read_text(encoding="utf-8"))
    raise AdapterError(f"Unsupported trace format: {selected}")


def normalize_trajectory_v2(
    path: str | Path,
    fmt: str = "auto",
    *,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize any supported observable trace into ``hfr.trajectory.v2``."""

    trace_path = Path(path)
    try:
        complete = json.loads(trace_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        complete = None
    if isinstance(complete, dict) and complete.get("schema_version") == TRAJECTORY_V2_SCHEMA_VERSION:
        return complete
    selected = detect_format(trace_path) if fmt == "auto" else fmt
    normalized = normalize_trace(trace_path, selected)
    extracted = trajectory_v2_context_from_path(trace_path)
    if context:
        extracted.update(context)
    return trajectory_v2_from_trace(
        normalized,
        source_path=trace_path,
        source_format=selected,
        context=extracted,
    )


def detect_format(path: Path) -> str:
    """Best-effort trace format detection."""
    first = _first_json(path)
    if isinstance(first, dict):
        if first.get("schema_version") == TRACE_SCHEMA_VERSION:
            return "normalized_json"
        if first.get("schema_version") == OPENCLAW_EVENT_SCHEMA_VERSION:
            return "openclaw_jsonl"
        if first.get("schema_version") == COVEN_EVENT_SCHEMA_VERSION:
            return "coven_jsonl"
        if first.get("type") in COVEN_STREAM_EVENT_TYPES:
            return "coven_jsonl"
        if first.get("kind") in COVEN_EVENT_KINDS and ("payload_json" in first or "payload" in first):
            return "coven_jsonl"
        if "conversations" in first:
            return "trajectory_jsonl"
        if first.get("schema_version", "").startswith("ATIF") or "steps" in first:
            return "atif_json"
        hook = first.get("hook") or first.get("event") or first.get("name")
        if hook in OPENCLAW_HOOKS:
            return "openclaw_jsonl"
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
        declared_session = entry.get("session_id")
        current_session = str(declared_session) if declared_session else f"{path.stem}-{entry_index + 1}"
        if declared_session and session_id == path.stem:
            session_id = current_session
        conversations = entry.get("conversations") or []
        if not isinstance(conversations, list):
            continue
        for turn_index, turn in enumerate(conversations):
            if not isinstance(turn, dict):
                continue
            role = turn.get("from")
            value = str(turn.get("value") or "")
            if role == "human":
                events.append(
                    _event(
                        "user_message",
                        current_session,
                        text=value,
                        order=len(events),
                        **_observable_event_kwargs(turn),
                    )
                )
            elif role == "gpt":
                for call in _extract_json_blocks(value, TOOL_CALL_RE):
                    events.append(
                        _event(
                            "tool_call",
                            current_session,
                            tool_name=str(call.get("name") or "unknown"),
                            args=call.get("arguments") if isinstance(call.get("arguments"), dict) else {},
                            status="requested",
                            tool_call_id=call.get("tool_call_id") or call.get("id"),
                            text="",
                            order=len(events),
                            **_observable_event_kwargs(turn),
                        )
                    )
                clean = _strip_assistant_markup(value).strip()
                if clean:
                    final_answer = clean
                    events.append(
                        _event(
                            "assistant_message",
                            current_session,
                            text=clean,
                            order=len(events),
                            **_observable_event_kwargs(turn),
                        )
                    )
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
                            **_observable_event_kwargs(turn),
                        )
                    )
            elif role == "system":
                events.append(
                    _event(
                        "system_message",
                        current_session,
                        text=value,
                        order=len(events),
                        **_observable_event_kwargs(turn),
                    )
                )
            elif role == "developer":
                events.append(
                    _event(
                        "developer_message",
                        current_session,
                        text=value,
                        order=len(events),
                        **_observable_event_kwargs(turn),
                    )
                )

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


def normalize_openclaw_jsonl(path: Path) -> dict[str, Any]:
    """Normalize Flight Recorder's OpenClaw plugin JSONL into hfr.trace.v1."""
    rows = _read_jsonl(path)
    if not rows:
        raise AdapterError(f"No JSONL entries in {path}")

    events: list[dict[str, Any]] = []
    final_answer = ""
    model = "unknown"
    session_id = path.stem
    completed = None
    observed_hooks: list[str] = []

    for row in rows:
        if not isinstance(row, dict):
            continue
        hook, payload = _openclaw_hook_payload(row)
        if not hook:
            continue
        observed_hooks.append(hook)
        context = _dict_value(payload, "context") or _dict_value(payload, "ctx") or {}
        session_id = str(
            _first_present(
                payload,
                context,
                row,
                keys=("sessionId", "session_id", "sessionKey", "session_key", "runId", "run_id"),
                default=session_id,
            )
        )
        model = str(
            _first_present(
                payload,
                context,
                keys=(
                    "model",
                    "modelId",
                    "model_id",
                    "resolvedModel",
                    "resolved_model",
                    "request.model",
                    "response.model",
                ),
                default=model,
            )
        )
        timestamp = row.get("captured_at") or row.get("timestamp") or payload.get("timestamp")

        if hook in {"session_start", "session_end"}:
            if hook == "session_end":
                completed = _openclaw_completed(payload, default=True)
            events.append(
                _event(
                    hook,
                    session_id,
                    status=str(payload.get("reason") or payload.get("status") or ""),
                    text=_stringify(payload.get("reason") or ""),
                    timestamp=timestamp,
                    source_hook=hook,
                    order=len(events),
                )
            )
        elif hook in {"before_agent_run", "before_agent_start", "before_prompt_build"}:
            text = _openclaw_input_text(payload)
            events.append(
                _event(
                    "user_message",
                    session_id,
                    text=text,
                    timestamp=timestamp,
                    source_hook=hook,
                    order=len(events),
                )
            )
        elif hook in {"model_call_started", "model_call_ended"}:
            events.append(
                _event(
                    "api_call",
                    session_id,
                    args={"model": model} if model != "unknown" else {},
                    status=_openclaw_status(payload, "started" if hook == "model_call_started" else "ok"),
                    text=hook,
                    timestamp=timestamp,
                    source_hook=hook,
                    order=len(events),
                )
            )
        elif hook == "llm_input":
            events.append(
                _event(
                    "user_message",
                    session_id,
                    text=_openclaw_input_text(payload),
                    timestamp=timestamp,
                    source_hook=hook,
                    order=len(events),
                )
            )
        elif hook in {"llm_output", "before_agent_reply", "before_agent_finalize", "agent_end"}:
            answer = _openclaw_output_text(payload, fallback=False)
            if answer:
                final_answer = answer
            if hook == "agent_end":
                completed = _openclaw_completed(payload, default=True)
            events.append(
                _event(
                    "assistant_message",
                    session_id,
                    status=_openclaw_status(payload, "ok"),
                    text=answer or _stringify(payload),
                    timestamp=timestamp,
                    source_hook=hook,
                    order=len(events),
                )
            )
        elif hook == "before_tool_call":
            events.append(
                _event(
                    "tool_call",
                    session_id,
                    tool_name=_openclaw_tool_name(payload),
                    args=_openclaw_tool_args(payload),
                    status="requested",
                    tool_call_id=_first_present(payload, context, keys=("toolCallId", "tool_call_id", "id"), default=None),
                    timestamp=timestamp,
                    source_hook=hook,
                    order=len(events),
                )
            )
        elif hook == "after_tool_call":
            result = _first_present(
                payload,
                keys=("result", "output", "response", "content", "text", "error", "errorMessage", "error_message"),
                default=None,
            )
            events.append(
                _event(
                    "tool_result",
                    session_id,
                    tool_name=_openclaw_tool_name(payload),
                    result=result,
                    status=_openclaw_status(payload, "ok"),
                    text=_openclaw_output_text(payload),
                    tool_call_id=_first_present(payload, context, keys=("toolCallId", "tool_call_id", "id"), default=None),
                    timestamp=timestamp,
                    source_hook=hook,
                    order=len(events),
                )
            )
        elif hook in {"subagent_spawned", "subagent_spawning"}:
            events.append(_openclaw_subagent_event("subagent_start", payload, context, session_id, hook, timestamp, len(events)))
        elif hook == "subagent_ended":
            events.append(_openclaw_subagent_event("subagent_stop", payload, context, session_id, hook, timestamp, len(events)))
        else:
            events.append(
                _event(
                    hook,
                    session_id,
                    status=_openclaw_status(payload, ""),
                    text=_stringify(payload),
                    timestamp=timestamp,
                    source_hook=hook,
                    order=len(events),
                )
            )

    metadata = {
        "completed": completed,
        "openclaw_hook_count": len(observed_hooks),
        "openclaw_hooks": observed_hooks,
    }
    return _trace(session_id, "openclaw_jsonl", model, events, final_answer, metadata)


def normalize_coven_jsonl(path: Path) -> dict[str, Any]:
    """Normalize Coven stream-json frames or daemon/API event rows into hfr.trace.v1."""
    rows = _read_jsonl(path)
    if not rows:
        raise AdapterError(f"No JSONL entries in {path}")

    events: list[dict[str, Any]] = []
    final_answer = ""
    model = "unknown"
    session_id = path.stem
    completed = None
    stream_event_types: list[str] = []
    daemon_event_kinds: list[str] = []
    result_metadata: dict[str, Any] = {}

    for row in rows:
        if not isinstance(row, dict):
            continue
        coven_row = _unwrap_coven_row(row)
        if _is_coven_daemon_event(coven_row):
            event = _normalize_coven_daemon_event(coven_row, session_id, len(events))
            if event is None:
                continue
            daemon_event_kinds.append(str(coven_row.get("kind") or ""))
            session_id = str(coven_row.get("session_id") or event.get("session_id") or session_id)
            if event["type"] == "assistant_message" and event.get("text"):
                final_answer = str(event["text"])
            if event["type"] == "session_end":
                payload = _coven_event_payload(coven_row)
                completed = _coven_completed_from_payload(payload, default=completed)
                result_metadata.update(_coven_daemon_result_metadata(payload))
            events.append(event)
            continue

        event_type = coven_row.get("type")
        if event_type not in COVEN_STREAM_EVENT_TYPES:
            continue
        stream_event_types.append(str(event_type))
        session_id = str(coven_row.get("session_id") or session_id)
        timestamp = coven_row.get("timestamp") or coven_row.get("created_at")

        if event_type == "system":
            model = str(coven_row.get("model") or model)
            events.append(
                _event(
                    "session_start",
                    session_id,
                    args={
                        "cwd": coven_row.get("cwd"),
                        "tools": coven_row.get("tools") if isinstance(coven_row.get("tools"), list) else [],
                        "agent_mode": coven_row.get("agent_mode"),
                        "model": coven_row.get("model"),
                    },
                    status=str(coven_row.get("subtype") or "init"),
                    text=str(coven_row.get("cwd") or ""),
                    timestamp=timestamp,
                    source_event_type="system",
                    order=len(events),
                )
            )
        elif event_type == "user":
            events.append(
                _event(
                    "user_message",
                    session_id,
                    text=_coven_message_text(coven_row.get("message")),
                    tool_call_id=coven_row.get("parent_tool_use_id"),
                    timestamp=timestamp,
                    source_event_type="user",
                    order=len(events),
                )
            )
        elif event_type == "assistant":
            text = _coven_message_text(coven_row.get("message"), include_tool_use=False)
            if text:
                final_answer = text
                events.append(
                    _event(
                        "assistant_message",
                        session_id,
                        status=str(coven_row.get("stop_reason") or "ok"),
                        text=text,
                        timestamp=timestamp,
                        source_event_type="assistant",
                        order=len(events),
                    )
                )
            for tool_call in _coven_tool_use_blocks(coven_row.get("message")):
                events.append(
                    _event(
                        "tool_call",
                        session_id,
                        tool_name=str(tool_call.get("name") or "unknown"),
                        args=tool_call.get("input") if isinstance(tool_call.get("input"), dict) else {},
                        status="requested",
                        text="",
                        tool_call_id=tool_call.get("id"),
                        timestamp=timestamp,
                        source_event_type="assistant",
                        order=len(events),
                    )
                )
        elif event_type == "tool_result":
            is_error = bool(coven_row.get("is_error"))
            events.append(
                _event(
                    "tool_result",
                    session_id,
                    status="error" if is_error else "ok",
                    text=_coven_content_text(coven_row.get("content"), include_tool_use=False),
                    tool_call_id=coven_row.get("tool_use_id"),
                    result=coven_row.get("content"),
                    timestamp=timestamp,
                    source_event_type="tool_result",
                    order=len(events),
                )
            )
        elif event_type == "result":
            completed = not bool(coven_row.get("is_error")) and coven_row.get("subtype") == "success"
            result_metadata.update(
                {
                    "duration_ms": coven_row.get("duration_ms"),
                    "num_turns": coven_row.get("num_turns"),
                    "is_error": bool(coven_row.get("is_error")),
                    "error": coven_row.get("error"),
                }
            )
            events.append(
                _event(
                    "session_end",
                    session_id,
                    status=str(coven_row.get("subtype") or ""),
                    text=str(coven_row.get("error") or ""),
                    timestamp=timestamp,
                    source_event_type="result",
                    order=len(events),
                )
            )

    metadata = {
        "completed": completed,
        "coven_stream_event_count": len(stream_event_types),
        "coven_stream_event_types": stream_event_types,
        "coven_daemon_event_count": len(daemon_event_kinds),
        "coven_daemon_event_kinds": daemon_event_kinds,
    }
    metadata.update({key: value for key, value in result_metadata.items() if value is not None})
    return _trace(session_id, "coven_jsonl", model, events, final_answer, metadata)


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


def _observable_event_kwargs(source: dict[str, Any]) -> dict[str, Any]:
    """Copy only observable timing/identity fields; never hidden reasoning."""

    aliases = {
        "event_id": ("event_id", "eventId"),
        "span_id": ("span_id", "spanId"),
        "parent_span_id": ("parent_span_id", "parentSpanId"),
        "timestamp": ("timestamp", "created_at"),
        "duration_ms": ("duration_ms", "latency_ms"),
        "token_usage": ("token_usage", "usage"),
        "cost_usd": ("cost_usd",),
        "side_effect_status": ("side_effect_status",),
    }
    copied: dict[str, Any] = {}
    for target, names in aliases.items():
        for name in names:
            if name in source and source[name] is not None:
                copied[target] = source[name]
                break
    return copied


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


def _openclaw_hook_payload(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    hook = str(row.get("hook") or row.get("event") or row.get("name") or row.get("type") or "")
    payload = row.get("payload") if isinstance(row.get("payload"), dict) else row
    return hook, payload if isinstance(payload, dict) else {}


def _openclaw_tool_name(payload: dict[str, Any]) -> str:
    return str(
        _first_present(
            payload,
            keys=("toolName", "tool_name", "name", "tool.name", "call.name", "request.toolName"),
            default="unknown",
        )
    )


def _openclaw_tool_args(payload: dict[str, Any]) -> dict[str, Any]:
    for path in ("params", "args", "arguments", "input", "toolInput", "tool.input", "call.params", "request.params"):
        value = _nested_get(payload, path)
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            parsed = _parse_json_object(value)
            if parsed is not None:
                return parsed
    return {}


def _openclaw_input_text(payload: dict[str, Any]) -> str:
    for path in (
        "messages",
        "request.messages",
        "history",
        "prompt",
        "input",
        "text",
        "message",
        "userMessage",
        "systemPrompt",
    ):
        value = _nested_get(payload, path)
        text = _render_message_value(value, roles=None)
        if text:
            return text
    return _stringify(payload)


def _openclaw_output_text(payload: dict[str, Any], *, fallback: bool = True) -> str:
    for path in ("finalMessages", "outputMessages", "messages", "response.messages"):
        value = _nested_get(payload, path)
        text = _render_message_value(value, roles={"assistant", "model"})
        if text:
            return text
    for path in (
        "assistantTexts",
        "assistantText",
        "finalAnswer",
        "final_answer",
        "answer",
        "output",
        "text",
        "content",
        "message.content",
        "assistantMessage.content",
        "reply.text",
        "reply.content",
        "response.output_text",
        "response.text",
        "response.content",
        "result.output",
        "result.text",
        "result.content",
        "errorMessage",
        "error_message",
        "error",
    ):
        value = _nested_get(payload, path)
        text = _render_message_value(value, roles={"assistant", "model"})
        if text:
            return text
    return _stringify(payload) if fallback else ""


def _openclaw_status(payload: dict[str, Any], default: str) -> str:
    if payload.get("error") or payload.get("errorMessage") or payload.get("error_message"):
        return "error"
    success = payload.get("success")
    if success is False:
        return "error"
    return str(_first_present(payload, keys=("status", "state", "outcome"), default=default))


def _openclaw_completed(payload: dict[str, Any], *, default: bool) -> bool:
    value = _first_present(payload, keys=("completed", "success", "ok"), default=default)
    return bool(value)


def _openclaw_subagent_event(
    event_type: str,
    payload: dict[str, Any],
    context: dict[str, Any],
    session_id: str,
    hook: str,
    timestamp: Any,
    order: int,
) -> dict[str, Any]:
    child_session_id = str(
        _first_present(
            payload,
            context,
            keys=("childSessionId", "child_session_id", "subagentSessionId", "sessionId", "session_id"),
            default=session_id,
        )
    )
    parent_session_id = _first_present(
        payload,
        context,
        keys=("parentSessionId", "parent_session_id", "parentSessionKey", "sessionKey", "session_id"),
        default=session_id,
    )
    return _event(
        event_type,
        child_session_id,
        parent_session_id=str(parent_session_id) if parent_session_id is not None else None,
        status=_openclaw_status(payload, ""),
        text=_openclaw_output_text(payload, fallback=False) or _openclaw_input_text(payload),
        child_session_id=child_session_id,
        child_subagent_id=_first_present(payload, keys=("subagentId", "subagent_id", "agentId", "agent_id"), default=None),
        child_role=_first_present(payload, keys=("role", "agentRole", "agent_role"), default=None),
        timestamp=timestamp,
        source_hook=hook,
        order=order,
    )


def _unwrap_coven_row(row: dict[str, Any]) -> dict[str, Any]:
    event = row.get("event")
    if row.get("schema_version") == COVEN_EVENT_SCHEMA_VERSION and isinstance(event, dict):
        return event
    return row


def _is_coven_daemon_event(row: dict[str, Any]) -> bool:
    return row.get("kind") in COVEN_EVENT_KINDS and ("payload_json" in row or "payload" in row)


def _normalize_coven_daemon_event(row: dict[str, Any], default_session_id: str, order: int) -> dict[str, Any] | None:
    kind = str(row.get("kind") or "")
    payload = _coven_event_payload(row)
    session_id = str(row.get("session_id") or payload.get("session_id") or default_session_id)
    timestamp = row.get("created_at") or row.get("timestamp")
    source = {"source_event_kind": kind, "source_event_id": row.get("id"), "source_event_seq": row.get("seq")}

    if kind == "input":
        return _event(
            "user_message",
            session_id,
            text=_coven_payload_text(payload),
            timestamp=timestamp,
            order=order,
            **source,
        )
    if kind == "output":
        return _event(
            "assistant_message",
            session_id,
            status=str(payload.get("status") or "ok"),
            text=_coven_payload_text(payload),
            result=payload,
            timestamp=timestamp,
            order=order,
            **source,
        )
    if kind == "tool_call":
        args = _first_present(payload, keys=("args", "arguments", "input", "params"), default={})
        return _event(
            "tool_call",
            session_id,
            tool_name=str(_first_present(payload, keys=("tool_name", "toolName", "name"), default="unknown")),
            args=args if isinstance(args, dict) else {},
            status=str(payload.get("status") or "requested"),
            text="",
            tool_call_id=_first_present(payload, keys=("tool_call_id", "toolUseId", "tool_use_id", "id"), default=None),
            timestamp=timestamp,
            order=order,
            **source,
        )
    if kind == "tool_result":
        return _event(
            "tool_result",
            session_id,
            tool_name=str(_first_present(payload, keys=("tool_name", "toolName", "name"), default="unknown")),
            status=str(payload.get("status") or ("error" if payload.get("is_error") else "ok")),
            text=_coven_payload_text(payload),
            result=payload,
            tool_call_id=_first_present(payload, keys=("tool_call_id", "toolUseId", "tool_use_id", "id"), default=None),
            timestamp=timestamp,
            order=order,
            **source,
        )
    if kind == "exit":
        return _event(
            "session_end",
            session_id,
            status=str(payload.get("status") or payload.get("subtype") or ""),
            text=_coven_payload_text(payload),
            timestamp=timestamp,
            order=order,
            **source,
        )
    if kind == "error":
        return _event(
            "error",
            session_id,
            status="error",
            text=_coven_payload_text(payload),
            result=payload,
            timestamp=timestamp,
            order=order,
            **source,
        )
    if kind in {"metadata", "patch_metadata"}:
        return _event(
            "session_metadata",
            session_id,
            status=str(payload.get("status") or ""),
            text=_coven_payload_text(payload),
            result=payload,
            timestamp=timestamp,
            order=order,
            **source,
        )
    return _event(
        f"coven_{kind}",
        session_id,
        status=str(payload.get("status") or ""),
        text=_coven_payload_text(payload),
        result=payload,
        timestamp=timestamp,
        order=order,
        **source,
    )


def _coven_event_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("payload")
    if isinstance(payload, dict):
        return payload
    raw = row.get("payload_json")
    if isinstance(raw, str):
        parsed = _parse_json_object(raw)
        if parsed is not None:
            return parsed
        return {"data": raw}
    return {}


def _coven_completed_from_payload(payload: dict[str, Any], default: Any) -> bool | None:
    if "is_error" in payload:
        return not bool(payload.get("is_error"))
    if "exitCode" in payload:
        return payload.get("exitCode") == 0
    if "exit_code" in payload:
        return payload.get("exit_code") == 0
    status = str(payload.get("status") or payload.get("subtype") or "").lower()
    if status:
        return status in {"completed", "success", "succeeded", "ok"}
    return default if isinstance(default, bool) else None


def _coven_daemon_result_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "exit_code": _first_present(payload, keys=("exitCode", "exit_code"), default=None),
        "is_error": payload.get("is_error"),
        "error": payload.get("error") or payload.get("message"),
    }


def _coven_message_text(message: Any, *, include_tool_use: bool = True) -> str:
    if isinstance(message, dict):
        return _coven_content_text(message.get("content"), include_tool_use=include_tool_use)
    return _coven_content_text(message, include_tool_use=include_tool_use)


def _coven_content_text(content: Any, *, include_tool_use: bool = True) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        block_type = content.get("type")
        if block_type == "text":
            return str(content.get("text") or "")
        if block_type == "image":
            source = content.get("source") if isinstance(content.get("source"), dict) else {}
            return str(source.get("path") or source.get("media_type") or "") if include_tool_use else ""
        if block_type == "tool_use":
            return _stringify(content) if include_tool_use else ""
        for key in ("text", "data", "message", "content", "output"):
            text = _coven_content_text(content.get(key), include_tool_use=include_tool_use)
            if text:
                return text
        return _stringify(content) if include_tool_use else ""
    if isinstance(content, list):
        rendered = []
        for item in content:
            text = _coven_content_text(item, include_tool_use=include_tool_use)
            if text:
                rendered.append(text)
        return "\n".join(rendered)
    return _stringify(content)


def _coven_tool_use_blocks(message: Any) -> list[dict[str, Any]]:
    content = message.get("content") if isinstance(message, dict) else message
    if not isinstance(content, list):
        return []
    return [item for item in content if isinstance(item, dict) and item.get("type") == "tool_use"]


def _coven_payload_text(payload: dict[str, Any]) -> str:
    for key in ("data", "text", "message", "output", "error", "reason"):
        text = _coven_content_text(payload.get(key), include_tool_use=True)
        if text:
            return text
    return _stringify(payload)


def _dict_value(value: dict[str, Any], key: str) -> dict[str, Any] | None:
    item = value.get(key)
    return item if isinstance(item, dict) else None


def _first_present(*objects: Any, keys: tuple[str, ...], default: Any = None) -> Any:
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        for key in keys:
            value = _nested_get(obj, key)
            if value is not None and value != "":
                return value
    return default


def _nested_get(obj: dict[str, Any], path: str) -> Any:
    current: Any = obj
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        if part in current:
            current = current[part]
            continue
        lowered = _lower_key_map(current)
        lowered_part = part.lower()
        if lowered_part not in lowered:
            return None
        current = current[lowered[lowered_part]]
    return current


def _lower_key_map(obj: dict[str, Any]) -> dict[str, str]:
    return {str(key).lower(): str(key) for key in obj}


def _render_message_value(value: Any, roles: set[str] | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        role = str(value.get("role") or value.get("from") or "").lower()
        if roles is not None and role and role not in roles:
            return ""
        for key in ("content", "text", "message", "output"):
            text = _render_message_value(value.get(key), roles=None)
            if text:
                return text
        return _stringify(value)
    if isinstance(value, list):
        rendered = []
        for item in value:
            text = _render_message_value(item, roles=roles)
            if text:
                rendered.append(text)
        return "\n".join(rendered)
    return _stringify(value)


def _parse_json_object(value: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


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
