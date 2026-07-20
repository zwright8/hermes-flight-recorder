"""Lossless, training-oriented trajectory v2 normalization and validation."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .schema_registry import check_schema_contract

TRAJECTORY_V2_SCHEMA_VERSION = "hfr.trajectory.v2"
TRAJECTORY_V2_ADAPTER_VERSION = "hfr.trajectory_v2_adapter.v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_HIDDEN_REASONING_KEYS = {
    "chain_of_thought",
    "chainOfThought",
    "hidden_reasoning",
    "hiddenReasoning",
    "internal_reasoning",
    "internalReasoning",
    "reasoning_content",
    "thinking",
}
_KNOWN_EVENT_FIELDS = {
    "args",
    "content",
    "cost_usd",
    "duration_ms",
    "event_id",
    "index",
    "latency_ms",
    "order",
    "parent_session_id",
    "parent_span_id",
    "result",
    "role",
    "session_id",
    "side_effect_status",
    "span_id",
    "status",
    "text",
    "timestamp",
    "token_usage",
    "tool_call_id",
    "tool_name",
    "type",
    "usage",
}
_REQUIRED_GOVERNANCE_FIELDS = {
    "owner",
    "tenant",
    "legal_basis",
    "allowed_purposes",
    "sensitivity",
    "jurisdiction",
    "retention_expires_at",
    "license",
    "provenance",
    "deletion_subject_ids",
}


def trajectory_v2_from_trace(
    trace: dict[str, Any],
    *,
    source_path: str | Path | None = None,
    source_format: str | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a canonical v2 trajectory from an observable normalized trace.

    ``context`` carries identities that raw trace formats often expose outside
    message events. It is intentionally explicit: missing or inferred
    identities remain visible and quarantine the trajectory from action SFT.
    """

    context = copy.deepcopy(context) if isinstance(context, dict) else {}
    trace = copy.deepcopy(trace)
    source_bytes = _source_bytes(source_path, trace)
    session = trace.get("session") if isinstance(trace.get("session"), dict) else {}
    metadata = trace.get("metadata") if isinstance(trace.get("metadata"), dict) else {}
    root_session_id = str(context.get("root_session_id") or session.get("id") or "unknown-session")
    effective_source_format = str(source_format or session.get("source_format") or "unknown")

    raw_events = trace.get("events") if isinstance(trace.get("events"), list) else []
    prefix_messages = _context_message_events(context.get("messages"), root_session_id)
    events = _canonical_events(prefix_messages + list(raw_events), context)
    sessions = _canonical_sessions(context.get("sessions"), events, root_session_id)
    tools = _canonical_tools(
        context.get("tools")
        or trace.get("tools")
        or session.get("tools")
        or metadata.get("tools")
        or _session_start_tools(raw_events)
    )
    model = _identity_record(
        context.get("model"),
        fallback_name=session.get("model"),
        fields=("provider", "name", "revision"),
    )
    tokenizer = _identity_record(
        context.get("tokenizer") or session.get("tokenizer") or metadata.get("tokenizer"),
        fields=("name", "revision"),
    )
    chat_template = _identity_record(
        context.get("chat_template") or session.get("chat_template") or metadata.get("chat_template"),
        fields=("name", "revision", "sha256"),
    )
    policy = _identity_record(
        context.get("policy") or session.get("policy") or metadata.get("policy"),
        fields=("id", "version", "sha256"),
    )
    environment = _identity_record(
        context.get("environment") or session.get("environment") or metadata.get("environment"),
        fields=("id", "version", "sha256"),
    )
    payload_sha256 = hashlib.sha256(source_bytes).hexdigest()
    trajectory_id = f"hfrtraj-{_canonical_sha256({'source': payload_sha256, 'session': root_session_id})[:20]}"
    trajectory: dict[str, Any] = {
        "schema_version": TRAJECTORY_V2_SCHEMA_VERSION,
        "trajectory_id": trajectory_id,
        "root_session_id": root_session_id,
        "source": {
            "format": effective_source_format,
            "payload_sha256": payload_sha256,
            "size_bytes": len(source_bytes),
            "adapter": {
                "name": str(context.get("adapter_name") or effective_source_format),
                "version": str(context.get("adapter_version") or TRAJECTORY_V2_ADAPTER_VERSION),
            },
        },
        "model": model,
        "tokenizer": tokenizer,
        "chat_template": chat_template,
        "policy": policy,
        "environment": environment,
        "governance": copy.deepcopy(
            context.get("governance")
            or trace.get("governance")
            or session.get("governance")
            or metadata.get("governance")
            or {}
        ),
        "tools": tools,
        "sessions": sessions,
        "events": events,
        "approvals": _canonical_approvals(context.get("approvals"), events),
        "metrics": _canonical_metrics(context.get("metrics"), metadata, events),
        "final_answer": _visible_text(str(trace.get("final_answer") or "")),
        "metadata": _without_hidden_reasoning(context.get("metadata") if isinstance(context.get("metadata"), dict) else {}),
        "action_training": {"eligible": False, "quarantine_reasons": []},
    }
    reasons = _action_quarantine_reasons(trajectory)
    trajectory["action_training"] = {
        "eligible": not reasons,
        "quarantine_reasons": reasons,
    }
    return trajectory


def trajectory_v2_context_from_path(path: str | Path) -> dict[str, Any]:
    """Extract the portable ``trajectory_context`` envelope from JSON/JSONL."""

    source = Path(path)
    contexts: list[dict[str, Any]] = []
    raw_text = source.read_text(encoding="utf-8")
    try:
        complete = json.loads(raw_text)
    except json.JSONDecodeError:
        complete = None
    if isinstance(complete, dict) and isinstance(complete.get("trajectory_context"), dict):
        contexts.append(complete["trajectory_context"])
    for line in raw_text.splitlines() if complete is None else []:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("trajectory_context"), dict):
            contexts.append(value["trajectory_context"])
        if isinstance(value, dict) and value.get("schema_version") == TRAJECTORY_V2_SCHEMA_VERSION:
            return {}
    merged: dict[str, Any] = {}
    for context in contexts:
        for key, value in context.items():
            if key in {"messages", "approvals", "sessions", "event_metadata"}:
                rows = merged.setdefault(key, [])
                if isinstance(rows, list) and isinstance(value, list):
                    rows.extend(copy.deepcopy(value))
            else:
                merged[key] = copy.deepcopy(value)
    for key in _HIDDEN_REASONING_KEYS:
        merged.pop(key, None)
    if isinstance(merged.get("metadata"), dict):
        merged["metadata"] = _without_hidden_reasoning(merged["metadata"])
    return merged


def check_trajectory_v2(trajectory: Any) -> dict[str, Any]:
    """Validate schema shape, graph identity, tools, and call/result matching."""

    errors: list[str] = []
    try:
        schema = check_schema_contract(trajectory, name_or_id="trajectory_v2")
        errors.extend(schema["errors"])
    except (TypeError, ValueError) as exc:
        schema = {"passed": False, "errors": [str(exc)]}
        errors.append(str(exc))
    if not isinstance(trajectory, dict):
        return _validation_result(errors, [], schema)

    hidden_paths = _exported_hidden_reasoning_paths(trajectory)
    errors.extend(f"hidden reasoning field is not allowed: {path}" for path in hidden_paths)
    errors.extend(_integrity_errors(trajectory))
    graph_errors = _graph_errors(trajectory)
    errors.extend(graph_errors)
    reasons = _action_quarantine_reasons(trajectory)
    recorded_action = trajectory.get("action_training")
    if not isinstance(recorded_action, dict):
        errors.append("action_training must be an object")
    else:
        if recorded_action.get("quarantine_reasons") != reasons:
            errors.append("action_training.quarantine_reasons must match recomputed trainability reasons")
        if recorded_action.get("eligible") is not (not reasons):
            errors.append("action_training.eligible must match recomputed trainability")
    return _validation_result(errors, reasons, schema)


def canonical_trajectory_v2_bytes(trajectory: dict[str, Any]) -> bytes:
    """Return deterministic JSON bytes used for round-trip and content identity."""

    return (
        json.dumps(trajectory, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def write_trajectory_v2(path: str | Path, trajectory: dict[str, Any]) -> None:
    Path(path).write_bytes(canonical_trajectory_v2_bytes(trajectory))


def load_trajectory_v2(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("trajectory v2 artifact must contain a JSON object")
    return value


def _canonical_events(raw_events: list[Any], context: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metadata_rows = context.get("event_metadata") if isinstance(context.get("event_metadata"), list) else []
    pending_by_name: dict[str, list[str]] = {}
    known_calls: dict[str, str] = {}
    for index, raw in enumerate(raw_events):
        event = raw if isinstance(raw, dict) else {"type": "unknown", "text": str(raw)}
        event_type = str(event.get("type") or "unknown")
        tool_name = str(event.get("tool_name") or "")
        recorded_call_id = event.get("tool_call_id")
        call_id = str(recorded_call_id) if recorded_call_id not in (None, "") else ""
        call_id_provenance = "recorded" if call_id else "missing"
        if event_type == "tool_call" and not call_id:
            call_id = f"generated-call-{index:06d}"
            call_id_provenance = "generated"
        if event_type == "tool_call":
            known_calls[call_id] = tool_name
            pending_by_name.setdefault(tool_name, []).append(call_id)
        elif event_type == "tool_result" and not call_id and tool_name:
            pending = pending_by_name.get(tool_name, [])
            if len(pending) == 1:
                call_id = pending[0]
                call_id_provenance = "inferred"
        if event_type == "tool_result" and call_id in known_calls and not tool_name:
            tool_name = known_calls[call_id]

        overlay = _event_overlay(metadata_rows, event, index, call_id)
        source_event_id = overlay.get("event_id", event.get("event_id"))
        source_span_id = overlay.get("span_id", event.get("span_id"))
        role = str(event.get("role") or _event_role(event_type))
        content = event.get("content", event.get("text", ""))
        if role == "assistant":
            content = _visible_text(str(content or ""))
        elif not isinstance(content, str):
            content = json.dumps(content, sort_keys=True, ensure_ascii=False)
        canonical = {
            "event_id": str(source_event_id or f"event-{index:06d}"),
            "span_id": str(source_span_id or f"span-{index:06d}"),
            "parent_span_id": _optional_string(overlay.get("parent_span_id", event.get("parent_span_id"))),
            "session_id": str(event.get("session_id") or overlay.get("session_id") or "unknown-session"),
            "parent_session_id": _optional_string(event.get("parent_session_id", overlay.get("parent_session_id"))),
            "type": event_type,
            "role": role or None,
            "content": str(content or ""),
            "timestamp": _optional_string(overlay.get("timestamp", event.get("timestamp"))),
            "duration_ms": _optional_number(overlay.get("duration_ms", event.get("duration_ms", event.get("latency_ms")))),
            "token_usage": _token_usage(overlay.get("token_usage", event.get("token_usage", event.get("usage")))),
            "cost_usd": _optional_number(overlay.get("cost_usd", event.get("cost_usd"))),
            "status": _optional_string(event.get("status")),
            "side_effect_status": str(
                overlay.get("side_effect_status")
                or event.get("side_effect_status")
                or _side_effect_status(event_type, event.get("status"))
            ),
            "tool_name": tool_name or None,
            "tool_call_id": call_id or None,
            "tool_call_id_provenance": call_id_provenance if event_type in {"tool_call", "tool_result"} else "not_applicable",
            "arguments": copy.deepcopy(event.get("args")) if isinstance(event.get("args"), dict) else {},
            "result": copy.deepcopy(event.get("result")) if "result" in event else None,
            "source_order": _source_order(event, index),
            "attributes": _event_attributes(event),
        }
        rows.append(canonical)
    return rows


def _canonical_sessions(value: Any, events: list[dict[str, Any]], root_session_id: str) -> list[dict[str, Any]]:
    explicit = value if isinstance(value, list) else []
    by_id: dict[str, dict[str, Any]] = {}
    if explicit:
        for raw in explicit:
            if not isinstance(raw, dict):
                continue
            session_id = str(raw.get("session_id") or raw.get("id") or "")
            if not session_id:
                continue
            by_id[session_id] = {
                "session_id": session_id,
                "parent_session_id": _optional_string(raw.get("parent_session_id")),
                "agent_id": str(raw.get("agent_id") or "unknown"),
                "agent_role": str(raw.get("agent_role") or raw.get("role") or "unknown"),
            }
    else:
        by_id[root_session_id] = {
            "session_id": root_session_id,
            "parent_session_id": None,
            "agent_id": "root",
            "agent_role": "root",
        }
        for event in events:
            session_id = str(event.get("session_id") or "")
            if not session_id:
                continue
            parent = event.get("parent_session_id")
            row = by_id.setdefault(
                session_id,
                {
                    "session_id": session_id,
                    "parent_session_id": parent,
                    "agent_id": "unknown",
                    "agent_role": "unknown",
                },
            )
            if row.get("parent_session_id") is None and parent is not None:
                row["parent_session_id"] = parent
    return [by_id[key] for key in sorted(by_id, key=lambda item: (item != root_session_id, item))]


def _canonical_tools(value: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in value if isinstance(value, list) else []:
        if not isinstance(raw, dict):
            continue
        definition = copy.deepcopy(raw)
        function = definition.get("function") if isinstance(definition.get("function"), dict) else definition
        name = str(function.get("name") or definition.get("name") or "")
        version = str(definition.get("version") or function.get("version") or "unknown")
        parameters = function.get("parameters")
        exact = bool(name and version != "unknown" and isinstance(parameters, dict))
        rows.append(
            {
                "name": name or "unknown",
                "version": version,
                "schema_provenance": "recorded_exact" if exact else "inferred_or_incomplete",
                "definition_sha256": _canonical_sha256(definition),
                "definition": definition,
            }
        )
    return rows


def _canonical_approvals(value: Any, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(value if isinstance(value, list) else []):
        if not isinstance(raw, dict):
            continue
        rows.append(
            {
                "approval_id": str(raw.get("approval_id") or f"approval-{index:06d}"),
                "session_id": str(raw.get("session_id") or "unknown-session"),
                "event_id": _optional_string(raw.get("event_id")),
                "request": str(raw.get("request") or ""),
                "decision": str(raw.get("decision") or "unknown"),
                "decided_by": str(raw.get("decided_by") or "unknown"),
                "timestamp": _optional_string(raw.get("timestamp")),
            }
        )
    for event in events:
        if event.get("type") != "approval":
            continue
        rows.append(
            {
                "approval_id": f"approval-{event['event_id']}",
                "session_id": event["session_id"],
                "event_id": event["event_id"],
                "request": event.get("content", ""),
                "decision": str(event.get("status") or "unknown"),
                "decided_by": "recorded_runtime",
                "timestamp": event.get("timestamp"),
            }
        )
    return rows


def _canonical_metrics(value: Any, metadata: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = value if isinstance(value, dict) else {}
    timestamps = [event["timestamp"] for event in events if isinstance(event.get("timestamp"), str) and event["timestamp"]]
    usage = metrics.get("token_usage") or metadata.get("token_usage") or metadata.get("usage")
    return {
        "started_at": _optional_string(metrics.get("started_at") or (timestamps[0] if timestamps else None)),
        "ended_at": _optional_string(metrics.get("ended_at") or (timestamps[-1] if timestamps else None)),
        "latency_ms": _optional_number(metrics.get("latency_ms", metadata.get("duration_ms"))),
        "token_usage": _token_usage(usage),
        "cost_usd": _optional_number(metrics.get("cost_usd", metadata.get("cost_usd"))),
    }


def _action_quarantine_reasons(trajectory: dict[str, Any]) -> list[str]:
    reasons: set[str] = set()
    calls = [event for event in trajectory.get("events", []) if isinstance(event, dict) and event.get("type") == "tool_call"]
    results = [event for event in trajectory.get("events", []) if isinstance(event, dict) and event.get("type") == "tool_result"]
    tools = trajectory.get("tools") if isinstance(trajectory.get("tools"), list) else []
    governance = trajectory.get("governance") if isinstance(trajectory.get("governance"), dict) else {}
    if not _REQUIRED_GOVERNANCE_FIELDS.issubset(governance) or any(
        governance.get(key) in {None, ""} for key in _REQUIRED_GOVERNANCE_FIELDS - {"allowed_purposes", "provenance", "deletion_subject_ids"}
    ):
        reasons.add("missing_governance_metadata")
    if not isinstance(governance.get("allowed_purposes"), list) or not governance.get("allowed_purposes"):
        reasons.add("missing_governance_metadata")
    if not isinstance(governance.get("provenance"), dict) or not governance.get("provenance"):
        reasons.add("missing_governance_metadata")
    if not isinstance(governance.get("deletion_subject_ids"), list) or not governance.get("deletion_subject_ids"):
        reasons.add("missing_governance_metadata")
    tool_names = {str(tool.get("name")) for tool in tools if isinstance(tool, dict)}
    if not calls:
        reasons.add("no_tool_calls")
    if calls and not tools:
        reasons.add("missing_tool_schema")
    if any(tool.get("schema_provenance") != "recorded_exact" for tool in tools if isinstance(tool, dict)):
        reasons.add("inferred_or_incomplete_tool_schema")
    if any(
        isinstance(tool, dict)
        and tool.get("definition_sha256") != _canonical_sha256(tool.get("definition"))
        for tool in tools
    ):
        reasons.add("tool_definition_fingerprint_mismatch")
    call_ids = [event.get("tool_call_id") for event in calls]
    if any(not isinstance(call_id, str) or not call_id for call_id in call_ids):
        reasons.add("missing_tool_call_id")
    if len({call_id for call_id in call_ids if call_id}) != len(call_ids):
        reasons.add("duplicate_tool_call_id")
    if any(event.get("tool_call_id_provenance") != "recorded" for event in calls + results):
        reasons.add("inferred_or_generated_tool_call_identity")
    result_counts: dict[str, int] = {}
    for event in results:
        call_id = event.get("tool_call_id")
        if isinstance(call_id, str) and call_id:
            result_counts[call_id] = result_counts.get(call_id, 0) + 1
        else:
            reasons.add("unmatched_tool_result")
    for call in calls:
        call_id = call.get("tool_call_id")
        if result_counts.get(str(call_id), 0) == 0:
            reasons.add("tool_call_without_result")
        elif result_counts.get(str(call_id), 0) > 1:
            reasons.add("tool_call_with_multiple_results")
        if str(call.get("tool_name") or "") not in tool_names:
            reasons.add("tool_call_missing_exact_definition")
    call_id_set = {str(call_id) for call_id in call_ids if call_id}
    if any(str(result.get("tool_call_id") or "") not in call_id_set for result in results):
        reasons.add("unmatched_tool_result")
    for key in ("model", "tokenizer", "chat_template", "policy", "environment"):
        identity = trajectory.get(key)
        if not isinstance(identity, dict) or any(str(value or "") in {"", "unknown"} for value in identity.values()):
            reasons.add(f"missing_or_mutable_{key}_identity")
    for key in ("model", "tokenizer", "chat_template"):
        identity = trajectory.get(key) if isinstance(trajectory.get(key), dict) else {}
        if _is_mutable_revision(identity.get("revision")):
            reasons.add(f"missing_or_mutable_{key}_identity")
    chat_template = trajectory.get("chat_template") if isinstance(trajectory.get("chat_template"), dict) else {}
    if not _is_sha256(chat_template.get("sha256")):
        reasons.add("malformed_chat_template_identity")
    for key in ("policy", "environment"):
        identity = trajectory.get(key) if isinstance(trajectory.get(key), dict) else {}
        if not _is_sha256(identity.get("sha256")):
            reasons.add(f"malformed_{key}_identity")
    reasons.update(_graph_errors(trajectory))
    return sorted(reasons)


def _integrity_errors(trajectory: dict[str, Any]) -> list[str]:
    errors: set[str] = set()
    source = trajectory.get("source") if isinstance(trajectory.get("source"), dict) else {}
    expected_id = f"hfrtraj-{_canonical_sha256({'source': source.get('payload_sha256'), 'session': trajectory.get('root_session_id')})[:20]}"
    if trajectory.get("trajectory_id") != expected_id:
        errors.add("trajectory_id does not match source payload and root session identity")
    tools = trajectory.get("tools") if isinstance(trajectory.get("tools"), list) else []
    tool_names: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_names.append(str(tool.get("name") or ""))
        if tool.get("definition_sha256") != _canonical_sha256(tool.get("definition")):
            errors.add("tool_definition_fingerprint_mismatch")
    if len(set(tool_names)) != len(tool_names):
        errors.add("duplicate_tool_definition_name")
    events = trajectory.get("events") if isinstance(trajectory.get("events"), list) else []
    call_names = {
        str(event.get("tool_call_id")): str(event.get("tool_name") or "")
        for event in events
        if isinstance(event, dict) and event.get("type") == "tool_call" and event.get("tool_call_id")
    }
    for event in events:
        if not isinstance(event, dict) or event.get("type") != "tool_result":
            continue
        call_id = str(event.get("tool_call_id") or "")
        if call_id in call_names and event.get("tool_name") not in {None, "", call_names[call_id]}:
            errors.add("tool_result_name_mismatch")
    return sorted(errors)


def _graph_errors(trajectory: dict[str, Any]) -> list[str]:
    errors: set[str] = set()
    sessions = trajectory.get("sessions") if isinstance(trajectory.get("sessions"), list) else []
    session_ids = [row.get("session_id") for row in sessions if isinstance(row, dict)]
    if len(set(session_ids)) != len(session_ids):
        errors.add("duplicate_session_id")
    known_sessions = {str(value) for value in session_ids if isinstance(value, str)}
    root = str(trajectory.get("root_session_id") or "")
    if root not in known_sessions:
        errors.add("missing_root_session")
    parents: dict[str, str | None] = {}
    for row in sessions:
        if not isinstance(row, dict):
            continue
        session_id = str(row.get("session_id") or "")
        parent = row.get("parent_session_id")
        parents[session_id] = str(parent) if isinstance(parent, str) else None
        if parent is not None and str(parent) not in known_sessions:
            errors.add("unknown_parent_session")
        if session_id == root and parent is not None:
            errors.add("root_session_has_parent")
        if session_id != root and parent is None:
            errors.add("orphan_session")
    for session_id in parents:
        seen: set[str] = set()
        cursor: str | None = session_id
        while cursor is not None:
            if cursor in seen:
                errors.add("cyclic_session_graph")
                break
            seen.add(cursor)
            cursor = parents.get(cursor)

    events = trajectory.get("events") if isinstance(trajectory.get("events"), list) else []
    event_ids = [row.get("event_id") for row in events if isinstance(row, dict)]
    span_ids = [row.get("span_id") for row in events if isinstance(row, dict)]
    if len(set(event_ids)) != len(event_ids):
        errors.add("duplicate_event_id")
    if len(set(span_ids)) != len(span_ids):
        errors.add("duplicate_span_id")
    known_spans = {str(value) for value in span_ids if isinstance(value, str)}
    span_parents: dict[str, str | None] = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        event_session_id = str(event.get("session_id") or "")
        if event_session_id not in known_sessions:
            errors.add("event_unknown_session")
        event_parent_session = event.get("parent_session_id")
        if event_parent_session is not None and str(event_parent_session) not in known_sessions:
            errors.add("event_unknown_parent_session")
        if event_session_id in parents and event_parent_session is not None:
            if str(event_parent_session) != str(parents[event_session_id]):
                errors.add("event_parent_session_mismatch")
        parent_span = event.get("parent_span_id")
        if parent_span is not None and str(parent_span) not in known_spans:
            errors.add("unknown_parent_span")
        span_id = str(event.get("span_id") or "")
        span_parents[span_id] = str(parent_span) if isinstance(parent_span, str) else None
    for span_id in span_parents:
        seen: set[str] = set()
        cursor: str | None = span_id
        while cursor is not None:
            if cursor in seen:
                errors.add("cyclic_span_graph")
                break
            seen.add(cursor)
            cursor = span_parents.get(cursor)
    known_events = {str(value) for value in event_ids if isinstance(value, str)}
    approvals = trajectory.get("approvals") if isinstance(trajectory.get("approvals"), list) else []
    approval_ids: list[str] = []
    for approval in approvals:
        if not isinstance(approval, dict):
            continue
        approval_ids.append(str(approval.get("approval_id") or ""))
        if str(approval.get("session_id") or "") not in known_sessions:
            errors.add("approval_unknown_session")
        event_id = approval.get("event_id")
        if event_id is not None and str(event_id) not in known_events:
            errors.add("approval_unknown_event")
    if len(set(approval_ids)) != len(approval_ids):
        errors.add("duplicate_approval_id")
    return sorted(errors)


def _context_message_events(value: Any, root_session_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(value if isinstance(value, list) else []):
        if not isinstance(raw, dict):
            continue
        role = str(raw.get("role") or "")
        if role not in {"system", "developer"}:
            continue
        rows.append(
            {
                "type": f"{role}_message",
                "role": role,
                "text": str(raw.get("content") or ""),
                "session_id": str(raw.get("session_id") or root_session_id),
                "parent_session_id": raw.get("parent_session_id"),
                "timestamp": raw.get("timestamp"),
                "event_id": raw.get("event_id") or f"context-{role}-{index:06d}",
                "span_id": raw.get("span_id") or f"context-{role}-span-{index:06d}",
                "parent_span_id": raw.get("parent_span_id"),
                "order": -len(value) + index,
            }
        )
    return rows


def _identity_record(value: Any, *, fallback_name: Any = None, fields: tuple[str, ...]) -> dict[str, str]:
    source = value if isinstance(value, dict) else {}
    result: dict[str, str] = {}
    for field in fields:
        raw = source.get(field)
        if field == "name" and not raw:
            raw = fallback_name
        result[field] = str(raw or "unknown")
    return result


def _session_start_tools(events: list[Any]) -> list[Any]:
    for event in events:
        if not isinstance(event, dict) or event.get("type") != "session_start":
            continue
        args = event.get("args") if isinstance(event.get("args"), dict) else {}
        if isinstance(args.get("tools"), list):
            return args["tools"]
    return []


def _event_overlay(rows: list[Any], event: dict[str, Any], index: int, call_id: str) -> dict[str, Any]:
    for row in rows:
        if not isinstance(row, dict):
            continue
        match = row.get("match") if isinstance(row.get("match"), dict) else {}
        if "order" in match and match["order"] != _source_order(event, index):
            continue
        if "type" in match and match["type"] != event.get("type"):
            continue
        if "tool_call_id" in match and str(match["tool_call_id"]) != call_id:
            continue
        return {key: copy.deepcopy(value) for key, value in row.items() if key != "match"}
    return {}


def _event_attributes(event: dict[str, Any]) -> dict[str, Any]:
    attributes = {key: value for key, value in event.items() if key not in _KNOWN_EVENT_FIELDS}
    if event.get("role") == "assistant" or event.get("type") == "assistant_message":
        return _without_hidden_reasoning(attributes)
    return copy.deepcopy(attributes)


def _event_role(event_type: str) -> str:
    return {
        "system_message": "system",
        "developer_message": "developer",
        "user_message": "user",
        "assistant_message": "assistant",
        "tool_result": "tool",
    }.get(event_type, "")


def _side_effect_status(event_type: str, status: Any) -> str:
    normalized = str(status or "").lower()
    if event_type == "tool_call":
        return "requested"
    if event_type != "tool_result":
        return "none"
    if normalized in {"ok", "success", "completed", "complete"}:
        return "completed"
    if normalized in {"denied", "rejected", "blocked"}:
        return "denied"
    if normalized in {"error", "failed", "failure", "timeout", "timed_out"}:
        return "failed"
    return "unknown"


def _token_usage(value: Any) -> dict[str, int | None]:
    source = value if isinstance(value, dict) else {}
    prompt = _optional_int(source.get("input_tokens", source.get("prompt_tokens")))
    completion = _optional_int(source.get("output_tokens", source.get("completion_tokens")))
    total = _optional_int(source.get("total_tokens"))
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    return {"input_tokens": prompt, "output_tokens": completion, "total_tokens": total}


def _source_order(event: dict[str, Any], fallback: int) -> int:
    value = event.get("order", event.get("index", fallback))
    return value if isinstance(value, int) and not isinstance(value, bool) else fallback


def _source_bytes(source_path: str | Path | None, trace: dict[str, Any]) -> bytes:
    if source_path is not None:
        return Path(source_path).read_bytes()
    return json.dumps(trace, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _visible_text(value: str) -> str:
    return _THINK_RE.sub("", value).strip()


def _without_hidden_reasoning(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _without_hidden_reasoning(child)
            for key, child in value.items()
            if str(key) not in _HIDDEN_REASONING_KEYS
        }
    if isinstance(value, list):
        return [_without_hidden_reasoning(child) for child in value]
    return copy.deepcopy(value)


def _hidden_reasoning_paths(value: Any, path: str = "$") -> list[str]:
    rows: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if str(key) in _HIDDEN_REASONING_KEYS:
                rows.append(child_path)
            rows.extend(_hidden_reasoning_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            rows.extend(_hidden_reasoning_paths(child, f"{path}[{index}]"))
    return rows


def _exported_hidden_reasoning_paths(trajectory: dict[str, Any]) -> list[str]:
    """Inspect model-owned surfaces without rejecting exact tool payload fields."""

    rows = _hidden_reasoning_paths(trajectory.get("metadata"), "$.metadata")
    events = trajectory.get("events") if isinstance(trajectory.get("events"), list) else []
    for index, event in enumerate(events):
        if not isinstance(event, dict) or event.get("role") != "assistant":
            continue
        rows.extend(_hidden_reasoning_paths(event, f"$.events[{index}]"))
        if _THINK_RE.search(str(event.get("content") or "")):
            rows.append(f"$.events[{index}].content")
    if _THINK_RE.search(str(trajectory.get("final_answer") or "")):
        rows.append("$.final_answer")
    return rows


def _validation_result(errors: list[str], reasons: list[str], schema: dict[str, Any]) -> dict[str, Any]:
    unique_errors = sorted(set(errors))
    return {
        "schema_version": "hfr.trajectory_v2_validation.v1",
        "passed": not unique_errors and not reasons,
        "schema_passed": schema.get("passed") is True,
        "action_training_eligible": not reasons,
        "quarantined": bool(reasons),
        "quarantine_reasons": reasons,
        "error_count": len(unique_errors),
        "errors": unique_errors,
    }


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))


def _is_mutable_revision(value: Any) -> bool:
    return str(value or "").strip().lower() in {"", "unknown", "main", "master", "head", "latest"}


def _optional_string(value: Any) -> str | None:
    return str(value) if value not in (None, "") else None


def _optional_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None
