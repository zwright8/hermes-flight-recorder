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
            "source_types": ["imap", "gmail_threads", "microsoft_graph_messages", "maildir", "eml"],
            "validators": ["email_sent", "email_read", "collection_item_exists"],
            "examples": ["reply sent", "assigned thread read before reply", "no duplicate reply"],
        },
        {
            "id": "github",
            "title": "GitHub Issues And Pull Requests",
            "states": ["issue state", "labels", "assignees", "comments", "timestamps"],
            "source_types": ["github_issue", "gitlab_issues", "linear_issues", "jira_issues", "http_json"],
            "validators": [
                "github_issue_commented",
                "github_issue_closed",
                "linear_issue_status",
                "jira_issue_status",
                "collection_item_exists",
                "status_changed",
            ],
            "examples": ["issue closed", "comment added", "label present"],
        },
        {
            "id": "tickets",
            "title": "Ticket, CRM, And Incident APIs",
            "states": ["ticket existence", "status", "assignee", "priority", "resolution fields"],
            "source_types": ["jira_issues", "linear_issues", "zendesk_tickets", "pagerduty_incidents", "http_json"],
            "validators": ["ticket_created", "status_changed", "api_json_field", "collection_item_exists"],
            "examples": ["support ticket created", "incident moved to resolved", "CRM field updated"],
        },
        {
            "id": "databases",
            "title": "Databases And Local State Stores",
            "states": ["rows", "columns", "counts", "status fields", "audit tables"],
            "source_types": ["sqlite", "http_json"],
            "validators": ["db_row_exists", "status_changed", "collection_item_exists"],
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
            "validators": ["job_completed", "api_json_field", "status_changed", "collection_count_changed"],
            "examples": ["build completed successfully", "queue item processed", "deployment reached ready"],
        },
        {
            "id": "webhooks",
            "title": "Webhooks And Event Sinks",
            "states": ["delivery status", "event ids", "payload fields", "attempt counts"],
            "source_types": ["http_json", "sqlite"],
            "validators": ["webhook_delivered", "db_row_exists", "api_json_field", "collection_item_exists"],
            "examples": ["webhook delivered once", "event payload persisted", "retry count stayed bounded"],
        },
        {
            "id": "generic_api",
            "title": "Generic JSON APIs",
            "states": ["any JSON field reachable from a read-only GET"],
            "source_types": ["http_json"],
            "validators": ["api_json_field", "status_changed", "job_completed", "collection_item_exists"],
            "examples": ["calendar event exists", "Slack message appears in history", "payment intent state changed"],
        },
        {
            "id": "chat",
            "title": "Chat And Collaboration",
            "states": ["channel messages", "DMs", "thread replies", "authors", "timestamps", "reactions"],
            "source_types": ["slack_history", "discord_messages", "http_json"],
            "validators": ["slack_message_sent", "collection_item_exists"],
            "examples": ["Slack message sent", "Teams post visible", "Discord reply exists"],
        },
        {
            "id": "calendar",
            "title": "Calendars And Scheduling",
            "states": ["events", "attendees", "start/end times", "conference links", "response status"],
            "source_types": ["google_calendar_events", "microsoft_graph_events", "http_json"],
            "validators": ["calendar_event_created", "collection_item_exists"],
            "examples": ["calendar event created", "attendee invited", "meeting time updated"],
        },
        {
            "id": "storage",
            "title": "Object Stores And Document Drives",
            "states": ["objects", "files", "keys", "mime types", "hashes", "owners", "modified times"],
            "source_types": ["google_drive_files", "s3_objects", "http_json"],
            "validators": ["drive_file_created", "s3_object_exists", "collection_item_exists"],
            "examples": ["Drive file created", "S3 object uploaded", "document renamed"],
        },
        {
            "id": "payments",
            "title": "Payments And Billing",
            "states": ["payment intents", "invoices", "subscriptions", "refunds", "settlement status"],
            "source_types": ["stripe_objects", "http_json"],
            "validators": ["payment_status", "api_json_field", "collection_item_exists"],
            "examples": ["payment intent succeeded", "invoice finalized", "subscription canceled"],
        },
        {
            "id": "infrastructure",
            "title": "Infrastructure And Runtime Control Planes",
            "states": ["deployments", "pods", "services", "health checks", "resource conditions"],
            "source_types": ["kubernetes_resources", "http_json"],
            "validators": ["k8s_resource_ready", "job_completed", "collection_item_exists"],
            "examples": ["deployment ready", "pod condition true", "service endpoint exists"],
        },
        {
            "id": "knowledge_docs",
            "title": "Knowledge Bases And Documents",
            "states": ["pages", "blocks", "titles", "last edited times", "owners"],
            "source_types": ["notion_database", "google_drive_files", "http_json"],
            "validators": ["notion_page_updated", "drive_file_created", "collection_item_exists"],
            "examples": ["Notion page updated", "document created", "wiki page contains expected text"],
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
    before_count = _optional_non_negative_int(spec.get("before_count"), "before_count")
    after_count = _optional_non_negative_int(spec.get("after_count"), "after_count")
    if before_count is None:
        before_count = 0
    if after_count is None:
        after_count = before_count + 1

    after_where: dict[str, Any] = {f"{state_path}.message_count": after_count}
    message_where: dict[str, Any] = {}
    _add_optional_contains(message_where, "subject", spec, "subject_contains")
    _add_optional_contains(message_where, "body_text", spec, "body_contains")
    _add_optional_contains(message_where, "to", spec, "recipient_contains")
    _add_optional_contains(message_where, "from", spec, "from_contains")
    _add_optional_matches(message_where, "message_id", spec, "message_id_matches")

    actions = _trace_action(spec, default_tool_name="gmail_send", default_id=f"{validator_id}_trace_send")
    if actions and spec.get("thread_id"):
        actions[0].setdefault("where", {})["result.thread_id"] = spec["thread_id"]
        actions[0].setdefault("where", {})["result.status"] = "sent"

    return _built(
        spec,
        "email_sent",
        state_path,
        required_actions=actions,
        required_state=[
            _state_assertion(
                f"{validator_id}_after_sent_message",
                after_where,
                where_any=[_where_any(f"{state_path}.messages", message_where)] if message_where else None,
            )
        ],
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
    before_where: dict[str, Any] = {}
    message_where: dict[str, Any] = {"message_id": {"present": True}}
    _add_optional_contains(message_where, "subject", spec, "subject_contains")
    _add_optional_contains(message_where, "body_text", spec, "body_contains")
    _add_optional_matches(message_where, "message_id", spec, "message_id_matches")

    actions = _trace_action(spec, default_tool_name="gmail_read", default_id=f"{validator_id}_trace_read")
    if actions and spec.get("thread_id"):
        actions[0].setdefault("where", {})["result.thread_id"] = spec["thread_id"]

    return _built(
        spec,
        "email_read",
        state_path,
        required_actions=actions,
        required_state_transitions=[
            _transition(
                f"{validator_id}_target_message_existed_before_read",
                before=before_where,
                after={},
                before_any=[_where_any(f"{state_path}.messages", message_where)],
                after_any=[_where_any(f"{state_path}.messages", message_where)],
            )
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
    transitions = [_transition(f"{validator_id}_status_changed", before=before, after=after)] if before else []
    return _built(
        spec,
        "status_changed",
        state_path,
        required_actions=_trace_action(spec, default_tool_name="", default_id=f"{validator_id}_trace_status_change"),
        required_state=[_state_assertion(f"{validator_id}_status_after", after)],
        required_state_transitions=transitions,
    )


def _build_github_issue_commented(spec: dict[str, Any]) -> dict[str, Any]:
    validator_id = _validator_id(spec)
    state_path = _state_path(spec, "github.issue")
    before_count = _optional_non_negative_int(spec.get("before_comment_count"), "before_comment_count")
    after_count = _optional_non_negative_int(spec.get("after_comment_count"), "after_comment_count")
    if before_count is None:
        before_count = 0
    if after_count is None:
        after_count = before_count + 1
    after_where: dict[str, Any] = {f"{state_path}.comment_count": after_count}
    comment_where: dict[str, Any] = {}
    _add_optional_contains(comment_where, "body", spec, "comment_contains")
    if "comment_user" in spec:
        comment_where["user"] = spec["comment_user"]
    return _built(
        spec,
        "github_issue_commented",
        state_path,
        required_actions=_trace_action(spec, default_tool_name="github_comment", default_id=f"{validator_id}_trace_comment"),
        required_state=[
            _state_assertion(
                f"{validator_id}_comment_after",
                after_where,
                where_any=[_where_any(f"{state_path}.comments", comment_where)] if comment_where else None,
            )
        ],
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
    after: dict[str, Any] = {}
    if "row_count" in spec:
        after[f"{state_path}.row_count"] = _non_negative_int(spec["row_count"], "row_count")
    fields = spec.get("fields") or {}
    if not isinstance(fields, dict):
        raise StateValidatorError("db_row_exists fields must be an object")
    if not fields:
        raise StateValidatorError("db_row_exists requires at least one field predicate")
    row_fields = {str(field): value for field, value in fields.items()}
    return _built(
        spec,
        "db_row_exists",
        state_path,
        required_actions=_trace_action(spec, default_tool_name="", default_id=f"{validator_id}_trace_db_write"),
        required_state=[
            _state_assertion(
                f"{validator_id}_row_exists_after",
                after,
                where_any=[_where_any(f"{state_path}.rows", row_fields)],
            )
        ],
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


def _build_collection_item_exists(spec: dict[str, Any]) -> dict[str, Any]:
    validator_id = _validator_id(spec)
    state_path = _required_string(spec, "state_path")
    item_where = _item_where_from_spec(spec)
    if not item_where:
        raise StateValidatorError("collection_item_exists requires fields, contains_fields, or matches_fields")
    count_where = _count_where_from_spec(spec, state_path)
    return _built(
        spec,
        "collection_item_exists",
        state_path,
        required_actions=_trace_action(spec, default_tool_name="", default_id=f"{validator_id}_trace_collection_action"),
        required_state=[
            _state_assertion(
                f"{validator_id}_item_exists_after",
                count_where,
                where_any=[_where_any(state_path, item_where)],
            )
        ],
    )


def _build_collection_count_changed(spec: dict[str, Any]) -> dict[str, Any]:
    validator_id = _validator_id(spec)
    count_path = _required_string(spec, "count_path")
    before_count = _non_negative_int(spec.get("before_count"), "before_count")
    after_count = _non_negative_int(spec.get("after_count"), "after_count")
    return _built(
        spec,
        "collection_count_changed",
        count_path,
        required_state=[_state_assertion(f"{validator_id}_count_after", {count_path: after_count})],
        required_state_transitions=[
            _transition(
                f"{validator_id}_count_changed",
                before={count_path: before_count},
                after={count_path: after_count},
            )
        ],
    )


def _build_slack_message_sent(spec: dict[str, Any]) -> dict[str, Any]:
    spec = dict(spec)
    spec.setdefault("state_path", "slack.messages")
    spec.setdefault("tool_name", "slack_post_message")
    _copy_spec_field(spec, "text_contains", "contains_fields", "text")
    _copy_spec_field(spec, "channel_id", "fields", "channel_id")
    _copy_spec_field(spec, "user", "fields", "user")
    return _rename_built(_build_collection_item_exists(spec), "slack_message_sent")


def _build_calendar_event_created(spec: dict[str, Any]) -> dict[str, Any]:
    spec = dict(spec)
    spec.setdefault("state_path", "calendar.events")
    spec.setdefault("tool_name", "calendar_create_event")
    _copy_spec_field(spec, "summary_contains", "contains_fields", "summary")
    _copy_spec_field(spec, "attendee_contains", "contains_fields", "attendees")
    _copy_spec_field(spec, "event_id", "fields", "id")
    _copy_spec_field(spec, "status", "fields", "status")
    return _rename_built(_build_collection_item_exists(spec), "calendar_event_created")


def _build_drive_file_created(spec: dict[str, Any]) -> dict[str, Any]:
    spec = dict(spec)
    spec.setdefault("state_path", "drive.files")
    spec.setdefault("tool_name", "drive_create_file")
    _copy_spec_field(spec, "name_contains", "contains_fields", "name")
    _copy_spec_field(spec, "file_id", "fields", "id")
    _copy_spec_field(spec, "mime_type", "fields", "mimeType")
    _copy_spec_field(spec, "owner_contains", "contains_fields", "owners")
    return _rename_built(_build_collection_item_exists(spec), "drive_file_created")


def _build_s3_object_exists(spec: dict[str, Any]) -> dict[str, Any]:
    spec = dict(spec)
    spec.setdefault("state_path", "s3.objects")
    spec.setdefault("tool_name", "s3_put_object")
    _copy_spec_field(spec, "key", "fields", "key")
    _copy_spec_field(spec, "bucket", "fields", "bucket")
    _copy_spec_field(spec, "etag", "fields", "etag")
    _copy_spec_field(spec, "key_contains", "contains_fields", "key")
    return _rename_built(_build_collection_item_exists(spec), "s3_object_exists")


def _build_k8s_resource_ready(spec: dict[str, Any]) -> dict[str, Any]:
    spec = dict(spec)
    spec.setdefault("state_path", "kubernetes.resources")
    _copy_spec_field(spec, "kind", "fields", "kind")
    _copy_spec_field(spec, "name", "fields", "name")
    _copy_spec_field(spec, "namespace", "fields", "namespace")
    fields = spec.setdefault("fields", {})
    if not isinstance(fields, dict):
        raise StateValidatorError("fields must be an object")
    fields.setdefault(str(spec.get("ready_field", "ready")), spec.get("ready_value", True))
    return _rename_built(_build_collection_item_exists(spec), "k8s_resource_ready")


def _build_payment_status(spec: dict[str, Any]) -> dict[str, Any]:
    spec = dict(spec)
    spec.setdefault("state_path", "payments.payment")
    spec.setdefault("field", "status")
    if "after_value" not in spec:
        spec["after_value"] = spec.get("status", "succeeded")
    return _rename_built(_build_status_changed(spec), "payment_status")


def _build_linear_issue_status(spec: dict[str, Any]) -> dict[str, Any]:
    spec = dict(spec)
    spec.setdefault("state_path", "linear.issue")
    spec.setdefault("field", "status")
    return _rename_built(_build_status_changed(spec), "linear_issue_status")


def _build_jira_issue_status(spec: dict[str, Any]) -> dict[str, Any]:
    spec = dict(spec)
    spec.setdefault("state_path", "jira.issue")
    spec.setdefault("field", "status")
    return _rename_built(_build_status_changed(spec), "jira_issue_status")


def _build_notion_page_updated(spec: dict[str, Any]) -> dict[str, Any]:
    spec = dict(spec)
    spec.setdefault("state_path", "notion.pages")
    spec.setdefault("tool_name", "notion_update_page")
    _copy_spec_field(spec, "page_id", "fields", "id")
    _copy_spec_field(spec, "title_contains", "contains_fields", "title")
    _copy_spec_field(spec, "body_contains", "contains_fields", "text")
    _copy_spec_field(spec, "last_edited_time_matches", "matches_fields", "last_edited_time")
    return _rename_built(_build_collection_item_exists(spec), "notion_page_updated")


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


def _state_assertion(
    assertion_id: str,
    where: dict[str, Any] | None = None,
    *,
    where_any: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    assertion: dict[str, Any] = {"id": assertion_id}
    if where:
        assertion["where"] = where
    if where_any:
        assertion["where_any"] = where_any
    return assertion


def _transition(
    transition_id: str,
    *,
    before: dict[str, Any],
    after: dict[str, Any],
    before_any: list[dict[str, Any]] | None = None,
    after_any: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    before_block: dict[str, Any] = {}
    after_block: dict[str, Any] = {}
    if before:
        before_block["where"] = before
    if after:
        after_block["where"] = after
    if before_any:
        before_block["where_any"] = before_any
    if after_any:
        after_block["where_any"] = after_any
    return {"id": transition_id, "before": before_block, "after": after_block}


def _where_any(path: str, where: dict[str, Any]) -> dict[str, Any]:
    return {"path": path, "where": where}


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


def _copy_spec_field(spec: dict[str, Any], source_key: str, target_map: str, target_field: str) -> None:
    if source_key not in spec:
        return
    values = spec.setdefault(target_map, {})
    if not isinstance(values, dict):
        raise StateValidatorError(f"{target_map} must be an object")
    values[target_field] = spec[source_key]


def _item_where_from_spec(spec: dict[str, Any]) -> dict[str, Any]:
    item_where: dict[str, Any] = {}
    fields = spec.get("fields") or {}
    contains_fields = spec.get("contains_fields") or {}
    matches_fields = spec.get("matches_fields") or {}
    for label, values in (
        ("fields", fields),
        ("contains_fields", contains_fields),
        ("matches_fields", matches_fields),
    ):
        if not isinstance(values, dict):
            raise StateValidatorError(f"{label} must be an object")
    for field, value in fields.items():
        item_where[str(field)] = value
    for field, value in contains_fields.items():
        item_where[str(field)] = {"contains": value}
    for field, value in matches_fields.items():
        item_where[str(field)] = {"matches": value}
    return item_where


def _count_where_from_spec(spec: dict[str, Any], state_path: str) -> dict[str, Any]:
    if "count_path" in spec and "count" in spec:
        return {_required_string(spec, "count_path"): _non_negative_int(spec["count"], "count")}
    if "count" in spec:
        return {f"{state_path}_count": _non_negative_int(spec["count"], "count")}
    return {}


_VALIDATORS: dict[str, ValidatorBuilder] = {
    "api_json_field": _build_api_json_field,
    "calendar_event_created": _build_calendar_event_created,
    "collection_count_changed": _build_collection_count_changed,
    "collection_item_exists": _build_collection_item_exists,
    "db_row_exists": _build_db_row_exists,
    "drive_file_created": _build_drive_file_created,
    "email_read": _build_email_read,
    "email_sent": _build_email_sent,
    "file_created": _build_file_created,
    "file_modified": _build_file_modified,
    "github_issue_closed": _build_github_issue_closed,
    "github_issue_commented": _build_github_issue_commented,
    "job_completed": _build_job_completed,
    "jira_issue_status": _build_jira_issue_status,
    "k8s_resource_ready": _build_k8s_resource_ready,
    "linear_issue_status": _build_linear_issue_status,
    "notion_page_updated": _build_notion_page_updated,
    "payment_status": _build_payment_status,
    "s3_object_exists": _build_s3_object_exists,
    "slack_message_sent": _build_slack_message_sent,
    "status_changed": _build_status_changed,
    "ticket_created": _build_ticket_created,
    "webhook_delivered": _build_webhook_delivered,
}
