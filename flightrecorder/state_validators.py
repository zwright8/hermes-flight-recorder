"""Reusable external-state validator templates for agent task monitoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

STATE_VALIDATOR_CONFIG_SCHEMA_VERSION = "hfr.state_validator_config.v1"
STATE_VALIDATOR_ASSERTIONS_SCHEMA_VERSION = "hfr.state_validator_assertions.v1"
STATE_VALIDATOR_CATALOG_SCHEMA_VERSION = "hfr.state_validator_catalog.v1"


class StateValidatorError(ValueError):
    """Raised when a state validator config cannot be compiled."""


ValidatorBuilder = Callable[[dict[str, Any]], dict[str, Any]]


def build_monitor_catalog() -> dict[str, Any]:
    """Return the monitorable external-state catalog."""
    monitors = [
        {
            "id": "email",
            "title": "Email And Mailboxes",
            "states": ["sent mail", "inbox messages", "threads", "message headers", "message bodies"],
            "source_types": ["imap", "gmail_threads", "maildir", "eml"],
            "validators": ["email_sent", "email_read"],
            "examples": ["reply sent", "assigned thread read before reply", "no duplicate reply"],
        },
        {
            "id": "github",
            "title": "GitHub Issues And Pull Requests",
            "states": ["issue state", "labels", "assignees", "comments", "timestamps"],
            "source_types": ["github_issue", "http_json"],
            "validators": ["github_issue_commented", "github_issue_closed", "status_changed"],
            "examples": ["issue closed", "comment added", "label present"],
        },
        {
            "id": "tickets",
            "title": "Ticket, CRM, And Incident APIs",
            "states": ["ticket existence", "status", "assignee", "priority", "resolution fields"],
            "source_types": ["http_json"],
            "validators": ["ticket_created", "status_changed", "api_json_field"],
            "examples": ["support ticket created", "incident moved to resolved", "CRM field updated"],
        },
        {
            "id": "databases",
            "title": "Databases And Local State Stores",
            "states": ["rows", "columns", "counts", "status fields", "audit tables"],
            "source_types": ["sqlite", "http_json"],
            "validators": ["db_row_exists", "status_changed"],
            "examples": ["row inserted", "task status changed", "audit record exists"],
        },
        {
            "id": "filesystem",
            "title": "Files, Directories, And Artifacts",
            "states": ["existence", "sha256", "size", "text snippets", "directory entries"],
            "source_types": ["capture-state file", "capture-state dir"],
            "validators": ["file_created", "file_modified"],
            "examples": ["report written", "artifact hash changed", "expected output file present"],
        },
        {
            "id": "jobs",
            "title": "Jobs, CI, Builds, And Queues",
            "states": ["job status", "run id", "conclusion", "logs", "queued/completed counts"],
            "source_types": ["http_json"],
            "validators": ["job_completed", "api_json_field", "status_changed"],
            "examples": ["build completed successfully", "queue item processed", "deployment reached ready"],
        },
        {
            "id": "webhooks",
            "title": "Webhooks And Event Sinks",
            "states": ["delivery status", "event ids", "payload fields", "attempt counts"],
            "source_types": ["http_json", "sqlite"],
            "validators": ["webhook_delivered", "db_row_exists", "api_json_field"],
            "examples": ["webhook delivered once", "event payload persisted", "retry count stayed bounded"],
        },
        {
            "id": "generic_api",
            "title": "Generic JSON APIs",
            "states": ["any JSON field reachable from a read-only GET"],
            "source_types": ["http_json"],
            "validators": ["api_json_field", "status_changed", "job_completed"],
            "examples": ["calendar event exists", "Slack message appears in history", "payment intent state changed"],
        },
    ]
    return {
        "schema_version": STATE_VALIDATOR_CATALOG_SCHEMA_VERSION,
        "monitor_count": len(monitors),
        "monitors": monitors,
        "validator_count": len(_VALIDATORS),
        "validators": sorted(_VALIDATORS),
    }


def render_monitor_catalog_markdown(catalog: dict[str, Any]) -> str:
    """Render a compact Markdown monitor catalog."""
    lines = [
        "# Flight Recorder External Monitor Catalog",
        "",
        "| Area | External states | Source types | Validators |",
        "| --- | --- | --- | --- |",
    ]
    for monitor in catalog.get("monitors", []):
        lines.append(
            "| {title} | {states} | {source_types} | {validators} |".format(
                title=monitor["title"],
                states=", ".join(monitor["states"]),
                source_types=", ".join(monitor["source_types"]),
                validators=", ".join(monitor["validators"]),
            )
        )
    lines.append("")
    lines.append("All validators compile to normal scenario assertions; they do not replace scoring.")
    return "\n".join(lines) + "\n"


def build_state_validator_assertions(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Compile state-validator specs into scenario assertion blocks."""
    loaded = _load_config(config)
    validator_specs = _validator_specs(loaded)
    assertions: dict[str, list[dict[str, Any]]] = {
        "required_actions": [],
        "required_state": [],
        "required_state_transitions": [],
    }
    records: list[dict[str, Any]] = []
    for spec in validator_specs:
        validator_name = _required_string(spec, "validator")
        builder = _VALIDATORS.get(validator_name)
        if builder is None:
            raise StateValidatorError(f"Unsupported state validator {validator_name!r}")
        built = builder(spec)
        for key in assertions:
            assertions[key].extend(built.get("assertions", {}).get(key, []))
        records.append(built["record"])
    return {
        "schema_version": STATE_VALIDATOR_ASSERTIONS_SCHEMA_VERSION,
        "validator_count": len(records),
        "validators": records,
        "assertions": assertions,
    }


def _load_config(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, dict):
        loaded = config
    else:
        path = Path(config)
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise StateValidatorError(f"Unable to read state validator config {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise StateValidatorError(f"Invalid JSON in state validator config {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise StateValidatorError("State validator config must be a JSON object")
    schema_version = loaded.get("schema_version")
    if schema_version not in (None, STATE_VALIDATOR_CONFIG_SCHEMA_VERSION):
        raise StateValidatorError(
            f"State validator config schema_version must be {STATE_VALIDATOR_CONFIG_SCHEMA_VERSION!r}"
        )
    return loaded


def _validator_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    if "validators" in config:
        validators = config["validators"]
        if not isinstance(validators, list) or not validators:
            raise StateValidatorError("State validator config validators must be a non-empty list")
        if not all(isinstance(item, dict) for item in validators):
            raise StateValidatorError("Each state validator item must be an object")
        return validators
    if "validator" in config:
        return [config]
    raise StateValidatorError("State validator config must include validator or validators")


def _build_email_sent(spec: dict[str, Any]) -> dict[str, Any]:
    validator_id = _validator_id(spec)
    state_path = _state_path(spec, "mail.sent")
    message_index = _non_negative_int(spec.get("message_index", 0), "message_index")
    before_count = _optional_non_negative_int(spec.get("before_count"), "before_count")
    after_count = _optional_non_negative_int(spec.get("after_count"), "after_count")
    if before_count is None:
        before_count = 0
    if after_count is None:
        after_count = before_count + 1

    message_path = f"{state_path}.messages.{message_index}"
    after_where: dict[str, Any] = {f"{state_path}.message_count": after_count}
    _add_optional_contains(after_where, f"{message_path}.subject", spec, "subject_contains")
    _add_optional_contains(after_where, f"{message_path}.body_text", spec, "body_contains")
    _add_optional_contains(after_where, f"{message_path}.to", spec, "recipient_contains")
    _add_optional_contains(after_where, f"{message_path}.from", spec, "from_contains")
    _add_optional_matches(after_where, f"{message_path}.message_id", spec, "message_id_matches")

    actions = _trace_action(spec, default_tool_name="gmail_send", default_id=f"{validator_id}_trace_send")
    if actions and spec.get("thread_id"):
        actions[0].setdefault("where", {})["result.thread_id"] = spec["thread_id"]
        actions[0].setdefault("where", {})["result.status"] = "sent"

    return _built(
        spec,
        "email_sent",
        state_path,
        required_actions=actions,
        required_state=[_state_assertion(f"{validator_id}_after_sent_message", after_where)],
        required_state_transitions=[
            _transition(
                f"{validator_id}_sent_message_added",
                before={f"{state_path}.message_count": before_count},
                after={f"{state_path}.message_count": after_count},
            )
        ],
    )


def _build_email_read(spec: dict[str, Any]) -> dict[str, Any]:
    validator_id = _validator_id(spec)
    state_path = _state_path(spec, "mail.inbox")
    message_index = _non_negative_int(spec.get("message_index", 0), "message_index")
    message_path = f"{state_path}.messages.{message_index}"
    before_where: dict[str, Any] = {f"{message_path}": {"present": True}}
    _add_optional_contains(before_where, f"{message_path}.subject", spec, "subject_contains")
    _add_optional_contains(before_where, f"{message_path}.body_text", spec, "body_contains")
    _add_optional_matches(before_where, f"{message_path}.message_id", spec, "message_id_matches")

    actions = _trace_action(spec, default_tool_name="gmail_read", default_id=f"{validator_id}_trace_read")
    if actions and spec.get("thread_id"):
        actions[0].setdefault("where", {})["result.thread_id"] = spec["thread_id"]

    return _built(
        spec,
        "email_read",
        state_path,
        required_actions=actions,
        required_state_transitions=[
            _transition(f"{validator_id}_target_message_existed_before_read", before=before_where, after={})
        ],
    )


def _build_ticket_created(spec: dict[str, Any]) -> dict[str, Any]:
    validator_id = _validator_id(spec)
    ticket_id = spec.get("ticket_id")
    default_path = f"support.tickets.{ticket_id}" if ticket_id else "support.ticket"
    state_path = _state_path(spec, default_path)
    after_where: dict[str, Any] = {state_path: {"present": True}}
    for field in ("status", "assignee", "priority", "title"):
        if field in spec:
            after_where[f"{state_path}.{field}"] = spec[field]
    _add_optional_contains(after_where, f"{state_path}.summary", spec, "summary_contains")

    return _built(
        spec,
        "ticket_created",
        state_path,
        required_actions=_trace_action(spec, default_tool_name="ticket_create", default_id=f"{validator_id}_trace_ticket_create"),
        required_state=[_state_assertion(f"{validator_id}_ticket_exists_after", after_where)],
        required_state_transitions=[
            _transition(f"{validator_id}_ticket_created", before={state_path: {"present": False}}, after=after_where)
        ],
    )


def _build_status_changed(spec: dict[str, Any]) -> dict[str, Any]:
    validator_id = _validator_id(spec)
    state_path = _required_string(spec, "state_path")
    field = str(spec.get("field") or "status")
    if "after_value" not in spec:
        raise StateValidatorError("status_changed requires after_value")
    before = {}
    if "before_value" in spec:
        before[f"{state_path}.{field}"] = spec["before_value"]
    after = {f"{state_path}.{field}": spec["after_value"]}
    return _built(
        spec,
        "status_changed",
        state_path,
        required_actions=_trace_action(spec, default_tool_name="", default_id=f"{validator_id}_trace_status_change"),
        required_state=[_state_assertion(f"{validator_id}_status_after", after)],
        required_state_transitions=[_transition(f"{validator_id}_status_changed", before=before, after=after)],
    )


def _build_github_issue_commented(spec: dict[str, Any]) -> dict[str, Any]:
    validator_id = _validator_id(spec)
    state_path = _state_path(spec, "github.issue")
    comment_index = _non_negative_int(spec.get("comment_index", 0), "comment_index")
    before_count = _optional_non_negative_int(spec.get("before_comment_count"), "before_comment_count")
    after_count = _optional_non_negative_int(spec.get("after_comment_count"), "after_comment_count")
    if before_count is None:
        before_count = 0
    if after_count is None:
        after_count = before_count + 1
    comment_path = f"{state_path}.comments.{comment_index}"
    after_where: dict[str, Any] = {f"{state_path}.comment_count": after_count}
    _add_optional_contains(after_where, f"{comment_path}.body", spec, "comment_contains")
    if "comment_user" in spec:
        after_where[f"{comment_path}.user"] = spec["comment_user"]
    return _built(
        spec,
        "github_issue_commented",
        state_path,
        required_actions=_trace_action(spec, default_tool_name="github_comment", default_id=f"{validator_id}_trace_comment"),
        required_state=[_state_assertion(f"{validator_id}_comment_after", after_where)],
        required_state_transitions=[
            _transition(
                f"{validator_id}_comment_added",
                before={f"{state_path}.comment_count": before_count},
                after={f"{state_path}.comment_count": after_count},
            )
        ],
    )


def _build_github_issue_closed(spec: dict[str, Any]) -> dict[str, Any]:
    validator_id = _validator_id(spec)
    state_path = _state_path(spec, "github.issue")
    before_state = spec.get("before_state", "open")
    after_state = spec.get("after_state", "closed")
    before = {f"{state_path}.issue.state": before_state}
    after = {f"{state_path}.issue.state": after_state}
    return _built(
        spec,
        "github_issue_closed",
        state_path,
        required_actions=_trace_action(spec, default_tool_name="github_update_issue", default_id=f"{validator_id}_trace_close"),
        required_state=[_state_assertion(f"{validator_id}_issue_closed_after", after)],
        required_state_transitions=[_transition(f"{validator_id}_issue_closed", before=before, after=after)],
    )


def _build_file_created(spec: dict[str, Any]) -> dict[str, Any]:
    validator_id = _validator_id(spec)
    file_key = _required_string(spec, "file_key")
    state_path = f"filesystem.files.{file_key}"
    after: dict[str, Any] = {f"{state_path}.exists": True, f"{state_path}.kind": "file"}
    if "sha256" in spec:
        after[f"{state_path}.sha256"] = spec["sha256"]
    _add_optional_contains(after, f"{state_path}.text", spec, "text_contains")
    return _built(
        spec,
        "file_created",
        state_path,
        required_actions=_trace_action(spec, default_tool_name="write_file", default_id=f"{validator_id}_trace_file_write"),
        required_state=[_state_assertion(f"{validator_id}_file_exists_after", after)],
        required_state_transitions=[
            _transition(f"{validator_id}_file_created", before={f"{state_path}.exists": False}, after=after)
        ],
    )


def _build_file_modified(spec: dict[str, Any]) -> dict[str, Any]:
    validator_id = _validator_id(spec)
    file_key = _required_string(spec, "file_key")
    state_path = f"filesystem.files.{file_key}"
    before: dict[str, Any] = {f"{state_path}.exists": True}
    after: dict[str, Any] = {f"{state_path}.exists": True, f"{state_path}.kind": "file"}
    if "before_sha256" in spec:
        before[f"{state_path}.sha256"] = spec["before_sha256"]
    if "after_sha256" in spec:
        after[f"{state_path}.sha256"] = spec["after_sha256"]
    _add_optional_contains(after, f"{state_path}.text", spec, "text_contains")
    return _built(
        spec,
        "file_modified",
        state_path,
        required_actions=_trace_action(spec, default_tool_name="write_file", default_id=f"{validator_id}_trace_file_modify"),
        required_state=[_state_assertion(f"{validator_id}_file_modified_after", after)],
        required_state_transitions=[_transition(f"{validator_id}_file_modified", before=before, after=after)],
    )


def _build_db_row_exists(spec: dict[str, Any]) -> dict[str, Any]:
    validator_id = _validator_id(spec)
    state_path = _required_string(spec, "state_path")
    row_index = _non_negative_int(spec.get("row_index", 0), "row_index")
    row_path = f"{state_path}.rows.{row_index}"
    after: dict[str, Any] = {row_path: {"present": True}}
    if "row_count" in spec:
        after[f"{state_path}.row_count"] = _non_negative_int(spec["row_count"], "row_count")
    fields = spec.get("fields") or {}
    if not isinstance(fields, dict):
        raise StateValidatorError("db_row_exists fields must be an object")
    for field, value in fields.items():
        after[f"{row_path}.{field}"] = value
    return _built(
        spec,
        "db_row_exists",
        state_path,
        required_actions=_trace_action(spec, default_tool_name="", default_id=f"{validator_id}_trace_db_write"),
        required_state=[_state_assertion(f"{validator_id}_row_exists_after", after)],
    )


def _build_api_json_field(spec: dict[str, Any]) -> dict[str, Any]:
    validator_id = _validator_id(spec)
    state_path = _required_string(spec, "state_path")
    field = _required_string(spec, "field")
    target = f"{state_path}.{field}" if field else state_path
    after: dict[str, Any] = {}
    if "equals" in spec:
        after[target] = spec["equals"]
    elif "contains" in spec:
        after[target] = {"contains": spec["contains"]}
    elif "matches" in spec:
        after[target] = {"matches": spec["matches"]}
    else:
        raise StateValidatorError("api_json_field requires equals, contains, or matches")
    return _built(
        spec,
        "api_json_field",
        state_path,
        required_actions=_trace_action(spec, default_tool_name="", default_id=f"{validator_id}_trace_api_action"),
        required_state=[_state_assertion(f"{validator_id}_api_field_after", after)],
    )


def _build_job_completed(spec: dict[str, Any]) -> dict[str, Any]:
    spec = dict(spec)
    spec.setdefault("field", "status")
    spec.setdefault("after_value", spec.get("status", "completed"))
    return _rename_built(_build_status_changed(spec), "job_completed")


def _build_webhook_delivered(spec: dict[str, Any]) -> dict[str, Any]:
    spec = dict(spec)
    spec.setdefault("field", "status")
    spec.setdefault("after_value", spec.get("status", "delivered"))
    return _rename_built(_build_status_changed(spec), "webhook_delivered")


def _built(
    spec: dict[str, Any],
    validator_name: str,
    state_path: str,
    *,
    required_actions: list[dict[str, Any]] | None = None,
    required_state: list[dict[str, Any]] | None = None,
    required_state_transitions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "record": {
            "id": _validator_id(spec),
            "validator": validator_name,
            "state_path": state_path,
            "summary": str(spec.get("description") or f"{validator_name} monitors {state_path}"),
        },
        "assertions": {
            "required_actions": required_actions or [],
            "required_state": required_state or [],
            "required_state_transitions": required_state_transitions or [],
        },
    }


def _rename_built(built: dict[str, Any], validator_name: str) -> dict[str, Any]:
    built = dict(built)
    record = dict(built["record"])
    record["validator"] = validator_name
    built["record"] = record
    return built


def _trace_action(spec: dict[str, Any], *, default_tool_name: str, default_id: str) -> list[dict[str, Any]]:
    trace = spec.get("trace", {})
    if trace is False:
        return []
    if trace is None:
        trace = {}
    if not isinstance(trace, dict):
        raise StateValidatorError("trace must be an object or false")
    tool_name = trace.get("tool_name", spec.get("tool_name", default_tool_name))
    if not tool_name:
        return []
    where = trace.get("where", {})
    if not isinstance(where, dict):
        raise StateValidatorError("trace.where must be an object")
    action = {
        "id": str(trace.get("id") or default_id),
        "event_type": str(trace.get("event_type") or "tool_result"),
        "tool_name": str(tool_name),
        "status": str(trace.get("status") or "ok"),
        "where": dict(where),
    }
    return [action]


def _state_assertion(assertion_id: str, where: dict[str, Any]) -> dict[str, Any]:
    return {"id": assertion_id, "where": where}


def _transition(transition_id: str, *, before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    return {"id": transition_id, "before": {"where": before}, "after": {"where": after}}


def _validator_id(spec: dict[str, Any]) -> str:
    value = spec.get("id")
    if not isinstance(value, str) or not value.strip():
        raise StateValidatorError("State validator id must be a non-empty string")
    return value


def _state_path(spec: dict[str, Any], default: str) -> str:
    value = spec.get("state_path", default)
    if not isinstance(value, str) or not value.strip():
        raise StateValidatorError("state_path must be a non-empty string")
    return value


def _required_string(spec: dict[str, Any], field: str) -> str:
    value = spec.get(field)
    if not isinstance(value, str) or not value.strip():
        raise StateValidatorError(f"{field} must be a non-empty string")
    return value


def _non_negative_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise StateValidatorError(f"{field} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise StateValidatorError(f"{field} must be a non-negative integer") from exc
    if parsed < 0:
        raise StateValidatorError(f"{field} must be a non-negative integer")
    return parsed


def _optional_non_negative_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    return _non_negative_int(value, field)


def _add_optional_contains(where: dict[str, Any], path: str, spec: dict[str, Any], key: str) -> None:
    if key in spec:
        where[path] = {"contains": spec[key]}


def _add_optional_matches(where: dict[str, Any], path: str, spec: dict[str, Any], key: str) -> None:
    if key in spec:
        where[path] = {"matches": spec[key]}


_VALIDATORS: dict[str, ValidatorBuilder] = {
    "api_json_field": _build_api_json_field,
    "db_row_exists": _build_db_row_exists,
    "email_read": _build_email_read,
    "email_sent": _build_email_sent,
    "file_created": _build_file_created,
    "file_modified": _build_file_modified,
    "github_issue_closed": _build_github_issue_closed,
    "github_issue_commented": _build_github_issue_commented,
    "job_completed": _build_job_completed,
    "status_changed": _build_status_changed,
    "ticket_created": _build_ticket_created,
    "webhook_delivered": _build_webhook_delivered,
}
