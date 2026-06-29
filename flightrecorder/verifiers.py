"""Read-only verifier adapters for external-state evidence snapshots."""

from __future__ import annotations

import base64
import datetime as _datetime
import email.policy
import hashlib
import hmac
import imaplib
import json
import os
import re
import shlex
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from email.message import EmailMessage, Message
from email.parser import BytesParser
from pathlib import Path
from typing import Any

from .state import sanitize_state_snapshot
from .state_capture import STATE_SNAPSHOT_SCHEMA_VERSION, capture_state_snapshot

VERIFIER_CONFIG_SCHEMA_VERSION = "hfr.verifier_config.v1"
VERIFIER_SOURCES_SCHEMA_VERSION = "hfr.verifier_sources.v1"
DEFAULT_HTTP_TIMEOUT_SECONDS = 15
DEFAULT_MAX_BODY_CHARS = 4096
DEFAULT_MAX_HTTP_BYTES = 2 * 1024 * 1024
_SOURCE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_RESERVED_STATE_ROOTS = {"schema_version", "filesystem", "json_sources", "json", "observations", "verifiers"}
_DEFAULT_LINEAR_ISSUES_QUERY = """
query FlightRecorderIssues($first: Int!) {
  issues(first: $first) {
    nodes {
      id
      identifier
      title
      url
      createdAt
      updatedAt
      state { name }
      team { key name }
      assignee { name email }
    }
  }
}
""".strip()


class VerifierError(ValueError):
    """Raised when an external verifier snapshot cannot be captured."""


def capture_verified_state(
    config: str | Path | dict[str, Any],
    *,
    preserve_paths: bool = False,
    secret_patterns: list[str] | None = None,
) -> dict[str, Any]:
    """Capture a state snapshot from configured read-only verifier adapters."""
    loaded_config, base_dir = _load_config(config)
    patterns = list(loaded_config.get("secret_patterns") or [])
    patterns.extend(secret_patterns or [])

    snapshot = capture_state_snapshot(preserve_paths=preserve_paths)
    snapshot["schema_version"] = STATE_SNAPSHOT_SCHEMA_VERSION
    snapshot["verifiers"] = {
        "schema_version": VERIFIER_SOURCES_SCHEMA_VERSION,
        "source_count": 0,
        "sources": {},
    }

    sources = loaded_config.get("sources")
    if not isinstance(sources, list):
        raise VerifierError("Verifier config sources must be a list")

    for source in sources:
        if not isinstance(source, dict):
            raise VerifierError("Each verifier source must be an object")
        source_id = _required_string(source, "id")
        if not _SOURCE_ID_RE.fullmatch(source_id):
            raise VerifierError(f"Verifier source id must match {_SOURCE_ID_RE.pattern}: {source_id!r}")
        source_type = _required_string(source, "type")
        required = bool(source.get("required", True))
        try:
            result = _capture_source(source, base_dir=base_dir, preserve_paths=preserve_paths)
        except Exception as exc:
            if required:
                if isinstance(exc, VerifierError):
                    raise
                raise VerifierError(f"Verifier source {source_id!r} failed: {exc}") from exc
            result = _source_result(source_type, "error", {"error": str(exc)})

        snapshot["verifiers"]["sources"][source_id] = result
        snapshot["verifiers"]["source_count"] += 1
        state_path = source.get("state_path")
        if result.get("status") == "ok" and state_path:
            state_value = result.get("data")
            state_value_path = source.get("state_value_path")
            if state_value_path:
                state_value = _select_dot_path(state_value, str(state_value_path))
            _assign_dot_path(snapshot, str(state_path), state_value)

    return sanitize_state_snapshot(snapshot, patterns)


def _load_config(config: str | Path | dict[str, Any]) -> tuple[dict[str, Any], Path]:
    if isinstance(config, dict):
        loaded = config
        base_dir = Path.cwd()
    else:
        config_path = Path(config)
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise VerifierError(f"Unable to read verifier config {config_path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise VerifierError(f"Invalid JSON in verifier config {config_path}: {exc}") from exc
        base_dir = config_path.parent.resolve()

    if not isinstance(loaded, dict):
        raise VerifierError("Verifier config must contain a JSON object")
    schema_version = loaded.get("schema_version")
    if schema_version not in (None, VERIFIER_CONFIG_SCHEMA_VERSION):
        raise VerifierError(
            f"Verifier config schema_version must be {VERIFIER_CONFIG_SCHEMA_VERSION!r} when provided"
        )
    return loaded, base_dir


def _capture_source(source: dict[str, Any], *, base_dir: Path, preserve_paths: bool) -> dict[str, Any]:
    source_type = _required_string(source, "type")
    if source_type == "eml":
        data = _capture_eml_source(source, base_dir, preserve_paths)
    elif source_type == "maildir":
        data = _capture_maildir_source(source, base_dir, preserve_paths)
    elif source_type == "http_json":
        data = _capture_http_json_source(source)
    elif source_type == "slack_history":
        data = _capture_slack_history_source(source)
    elif source_type == "google_calendar_events":
        data = _capture_google_calendar_events_source(source)
    elif source_type == "google_drive_files":
        data = _capture_google_drive_files_source(source)
    elif source_type == "kubernetes_resources":
        data = _capture_kubernetes_resources_source(source)
    elif source_type == "stripe_objects":
        data = _capture_stripe_objects_source(source)
    elif source_type == "notion_database":
        data = _capture_notion_database_source(source)
    elif source_type == "linear_issues":
        data = _capture_linear_issues_source(source)
    elif source_type == "jira_issues":
        data = _capture_jira_issues_source(source)
    elif source_type == "s3_objects":
        data = _capture_s3_objects_source(source)
    elif source_type == "microsoft_graph_messages":
        data = _capture_microsoft_graph_messages_source(source)
    elif source_type == "microsoft_graph_events":
        data = _capture_microsoft_graph_events_source(source)
    elif source_type == "gitlab_issues":
        data = _capture_gitlab_issues_source(source)
    elif source_type == "discord_messages":
        data = _capture_discord_messages_source(source)
    elif source_type == "zendesk_tickets":
        data = _capture_zendesk_tickets_source(source)
    elif source_type == "pagerduty_incidents":
        data = _capture_pagerduty_incidents_source(source)
    elif source_type == "sqlite":
        data = _capture_sqlite_source(source, base_dir, preserve_paths)
    elif source_type == "github_issue":
        data = _capture_github_issue_source(source)
    elif source_type in {"gmail_thread", "gmail_threads"}:
        data = _capture_gmail_threads_source(source)
    elif source_type == "imap":
        data = _capture_imap_source(source)
    else:
        raise VerifierError(f"Unsupported verifier source type {source_type!r}")
    return _source_result(source_type, "ok", data)


def _source_result(source_type: str, status: str, data: Any) -> dict[str, Any]:
    return {
        "type": source_type,
        "status": status,
        "readonly": True,
        "data": data,
    }


def _capture_eml_source(source: dict[str, Any], base_dir: Path, preserve_paths: bool) -> dict[str, Any]:
    path = _source_path(source, base_dir)
    max_messages = _non_negative_int(source.get("max_messages", 100), "max_messages")
    include_body = bool(source.get("include_body", True))
    messages = _read_email_path(path, max_messages, include_body, preserve_paths)
    return {
        "path": _display_path(path, preserve_paths),
        "message_count": len(messages),
        "messages": messages,
    }


def _capture_maildir_source(source: dict[str, Any], base_dir: Path, preserve_paths: bool) -> dict[str, Any]:
    path = _source_path(source, base_dir)
    if not path.exists():
        raise VerifierError(f"Maildir path does not exist: {path}")
    if not path.is_dir():
        raise VerifierError(f"Maildir source path must be a directory: {path}")

    max_messages = _non_negative_int(source.get("max_messages", 100), "max_messages")
    include_body = bool(source.get("include_body", True))
    filters = _string_list(source.get("contains"))
    if max_messages == 0:
        return {
            "path": _display_path(path, preserve_paths),
            "message_count": 0,
            "messages": [],
        }
    candidates = _maildir_files(path)
    messages: list[dict[str, Any]] = []
    for candidate in candidates:
        parsed = _parse_email_file(candidate, include_body, preserve_paths)
        if filters and not _message_matches_all(parsed, filters):
            continue
        messages.append(parsed)
        if len(messages) >= max_messages:
            break

    return {
        "path": _display_path(path, preserve_paths),
        "message_count": len(messages),
        "messages": messages,
    }


def _capture_http_json_source(source: dict[str, Any]) -> dict[str, Any]:
    url = _required_string(source, "url")
    timeout = _positive_float(source.get("timeout_seconds", DEFAULT_HTTP_TIMEOUT_SECONDS), "timeout_seconds")
    max_bytes = _non_negative_int(source.get("max_bytes", DEFAULT_MAX_HTTP_BYTES), "max_bytes")
    headers = _headers_from_source(source)
    status_code, payload = _http_get_json(url, headers=headers, timeout=timeout, max_bytes=max_bytes)
    return {
        "url": url,
        "status_code": status_code,
        "json": payload,
    }


def _capture_slack_history_source(source: dict[str, Any]) -> dict[str, Any]:
    channel_id = _required_string(source, "channel_id")
    base_url = str(source.get("base_url") or "https://slack.com/api").rstrip("/")
    timeout = _positive_float(source.get("timeout_seconds", DEFAULT_HTTP_TIMEOUT_SECONDS), "timeout_seconds")
    max_bytes = _non_negative_int(source.get("max_bytes", DEFAULT_MAX_HTTP_BYTES), "max_bytes")
    headers = _bearer_headers(source, "SLACK_BOT_TOKEN")
    params: dict[str, Any] = {
        "channel": channel_id,
        "limit": _non_negative_int(source.get("limit", 100), "limit"),
    }
    for name in ("oldest", "latest"):
        if source.get(name) is not None:
            params[name] = str(source[name])
    if source.get("inclusive") is not None:
        params["inclusive"] = "true" if bool(source["inclusive"]) else "false"
    url = _url_with_params(f"{base_url}/conversations.history", params)
    _status_code, payload = _http_get_json(url, headers=headers, timeout=timeout, max_bytes=max_bytes)
    if not isinstance(payload, dict):
        raise VerifierError("Slack history response must be a JSON object")
    if payload.get("ok") is False:
        raise VerifierError(f"Slack history response error: {payload.get('error') or 'unknown'}")
    messages = payload.get("messages") or []
    if not isinstance(messages, list):
        raise VerifierError("Slack history response messages must be a list")
    return {
        "channel_id": channel_id,
        "message_count": len(messages),
        "messages": [_slack_message_summary(message, channel_id) for message in messages if isinstance(message, dict)],
        "has_more": bool(payload.get("has_more")),
    }


def _capture_google_calendar_events_source(source: dict[str, Any]) -> dict[str, Any]:
    calendar_id = str(source.get("calendar_id") or "primary")
    base_url = str(source.get("base_url") or "https://www.googleapis.com/calendar/v3").rstrip("/")
    timeout = _positive_float(source.get("timeout_seconds", DEFAULT_HTTP_TIMEOUT_SECONDS), "timeout_seconds")
    max_bytes = _non_negative_int(source.get("max_bytes", DEFAULT_MAX_HTTP_BYTES), "max_bytes")
    headers = _bearer_headers(source, "GOOGLE_CALENDAR_ACCESS_TOKEN")
    params: dict[str, Any] = {
        "maxResults": _non_negative_int(source.get("max_results", 100), "max_results"),
        "singleEvents": "true" if bool(source.get("single_events", True)) else "false",
    }
    for source_key, param_key in (
        ("query", "q"),
        ("time_min", "timeMin"),
        ("time_max", "timeMax"),
        ("orderby", "orderBy"),
    ):
        if source.get(source_key) is not None:
            params[param_key] = str(source[source_key])
    if source.get("order_by") is not None and "orderBy" not in params:
        params["orderBy"] = str(source["order_by"])
    url = _url_with_params(f"{base_url}/calendars/{urllib.parse.quote(calendar_id, safe='')}/events", params)
    _status_code, payload = _http_get_json(url, headers=headers, timeout=timeout, max_bytes=max_bytes)
    if not isinstance(payload, dict):
        raise VerifierError("Google Calendar events response must be a JSON object")
    items = payload.get("items") or []
    if not isinstance(items, list):
        raise VerifierError("Google Calendar events response items must be a list")
    return {
        "calendar_id": calendar_id,
        "event_count": len(items),
        "events": [_calendar_event_summary(event) for event in items if isinstance(event, dict)],
        "next_page_token": payload.get("nextPageToken"),
    }


def _capture_google_drive_files_source(source: dict[str, Any]) -> dict[str, Any]:
    base_url = str(source.get("base_url") or "https://www.googleapis.com/drive/v3").rstrip("/")
    timeout = _positive_float(source.get("timeout_seconds", DEFAULT_HTTP_TIMEOUT_SECONDS), "timeout_seconds")
    max_bytes = _non_negative_int(source.get("max_bytes", DEFAULT_MAX_HTTP_BYTES), "max_bytes")
    headers = _bearer_headers(source, "GOOGLE_DRIVE_ACCESS_TOKEN")
    params: dict[str, Any] = {
        "pageSize": _non_negative_int(source.get("page_size", 100), "page_size"),
        "fields": str(
            source.get("fields")
            or "nextPageToken, files(id,name,mimeType,webViewLink,owners(emailAddress,displayName),"
            "modifiedTime,createdTime,trashed,md5Checksum,size)"
        ),
    }
    if source.get("query") is not None:
        params["q"] = str(source["query"])
    url = _url_with_params(f"{base_url}/files", params)
    _status_code, payload = _http_get_json(url, headers=headers, timeout=timeout, max_bytes=max_bytes)
    if not isinstance(payload, dict):
        raise VerifierError("Google Drive files response must be a JSON object")
    files = payload.get("files") or []
    if not isinstance(files, list):
        raise VerifierError("Google Drive files response files must be a list")
    return {
        "file_count": len(files),
        "files": [_drive_file_summary(item) for item in files if isinstance(item, dict)],
        "next_page_token": payload.get("nextPageToken"),
    }


def _capture_kubernetes_resources_source(source: dict[str, Any]) -> dict[str, Any]:
    url = _required_string(source, "url")
    timeout = _positive_float(source.get("timeout_seconds", DEFAULT_HTTP_TIMEOUT_SECONDS), "timeout_seconds")
    max_bytes = _non_negative_int(source.get("max_bytes", DEFAULT_MAX_HTTP_BYTES), "max_bytes")
    headers = _headers_from_source(source)
    if "Authorization" not in headers and (source.get("token_env") or source.get("bearer_token_env")):
        headers = _bearer_headers(source, "KUBERNETES_BEARER_TOKEN")
    _status_code, payload = _http_get_json(url, headers=headers, timeout=timeout, max_bytes=max_bytes)
    items = _payload_items(payload, "items", "Kubernetes resources")
    return {
        "resource_count": len(items),
        "resources": [_kubernetes_resource_summary(item) for item in items if isinstance(item, dict)],
    }


def _capture_stripe_objects_source(source: dict[str, Any]) -> dict[str, Any]:
    base_url = str(source.get("base_url") or "https://api.stripe.com/v1").rstrip("/")
    resource = _required_string(source, "resource").strip("/")
    timeout = _positive_float(source.get("timeout_seconds", DEFAULT_HTTP_TIMEOUT_SECONDS), "timeout_seconds")
    max_bytes = _non_negative_int(source.get("max_bytes", DEFAULT_MAX_HTTP_BYTES), "max_bytes")
    headers = _bearer_headers(source, "STRIPE_SECRET_KEY")
    object_id = source.get("object_id")
    if object_id:
        url = f"{base_url}/{urllib.parse.quote(resource, safe='/')}/{urllib.parse.quote(str(object_id), safe='')}"
    else:
        params = {"limit": _non_negative_int(source.get("limit", 100), "limit")}
        url = _url_with_params(f"{base_url}/{urllib.parse.quote(resource, safe='/')}", params)
    _status_code, payload = _http_get_json(url, headers=headers, timeout=timeout, max_bytes=max_bytes)
    if object_id:
        if not isinstance(payload, dict):
            raise VerifierError("Stripe object response must be a JSON object")
        return _stripe_object_summary(payload)
    items = _payload_items(payload, "data", "Stripe objects")
    return {
        "resource": resource,
        "object_count": len(items),
        "objects": [_stripe_object_summary(item) for item in items if isinstance(item, dict)],
    }


def _capture_notion_database_source(source: dict[str, Any]) -> dict[str, Any]:
    database_id = _required_string(source, "database_id")
    base_url = str(source.get("base_url") or "https://api.notion.com/v1").rstrip("/")
    timeout = _positive_float(source.get("timeout_seconds", DEFAULT_HTTP_TIMEOUT_SECONDS), "timeout_seconds")
    max_bytes = _non_negative_int(source.get("max_bytes", DEFAULT_MAX_HTTP_BYTES), "max_bytes")
    headers = _bearer_headers(source, "NOTION_TOKEN")
    headers["Notion-Version"] = str(source.get("notion_version") or "2022-06-28")
    body = source.get("body") or {}
    if not isinstance(body, dict):
        raise VerifierError("Notion database body must be an object")
    if "page_size" in source and "page_size" not in body:
        body = dict(body)
        body["page_size"] = _non_negative_int(source["page_size"], "page_size")
    url = f"{base_url}/databases/{urllib.parse.quote(database_id, safe='')}/query"
    _status_code, payload = _http_post_json(url, headers=headers, timeout=timeout, max_bytes=max_bytes, payload=body)
    if not isinstance(payload, dict):
        raise VerifierError("Notion database response must be a JSON object")
    pages = payload.get("results") or []
    if not isinstance(pages, list):
        raise VerifierError("Notion database response results must be a list")
    return {
        "database_id": database_id,
        "page_count": len(pages),
        "pages": [_notion_page_summary(page) for page in pages if isinstance(page, dict)],
        "has_more": bool(payload.get("has_more")),
        "next_cursor": payload.get("next_cursor"),
    }


def _capture_linear_issues_source(source: dict[str, Any]) -> dict[str, Any]:
    base_url = str(source.get("base_url") or "https://api.linear.app/graphql").rstrip("/")
    timeout = _positive_float(source.get("timeout_seconds", DEFAULT_HTTP_TIMEOUT_SECONDS), "timeout_seconds")
    max_bytes = _non_negative_int(source.get("max_bytes", DEFAULT_MAX_HTTP_BYTES), "max_bytes")
    headers = _bearer_headers(source, "LINEAR_API_KEY")
    query = str(source.get("query") or _DEFAULT_LINEAR_ISSUES_QUERY)
    variables = source.get("variables") or {"first": _non_negative_int(source.get("first", 50), "first")}
    if not isinstance(variables, dict):
        raise VerifierError("Linear variables must be an object")
    _status_code, payload = _http_post_json(
        base_url,
        headers=headers,
        timeout=timeout,
        max_bytes=max_bytes,
        payload={"query": query, "variables": variables},
    )
    if not isinstance(payload, dict):
        raise VerifierError("Linear GraphQL response must be a JSON object")
    if payload.get("errors"):
        raise VerifierError(f"Linear GraphQL response errors: {payload['errors']}")
    nodes = _select_dot_path(payload, str(source.get("nodes_path") or "data.issues.nodes"))
    if not isinstance(nodes, list):
        raise VerifierError("Linear issue nodes must be a list")
    return {
        "issue_count": len(nodes),
        "issues": [_linear_issue_summary(issue) for issue in nodes if isinstance(issue, dict)],
    }


def _capture_jira_issues_source(source: dict[str, Any]) -> dict[str, Any]:
    base_url = _required_string(source, "base_url").rstrip("/")
    endpoint = str(source.get("endpoint") or "/rest/api/3/search").lstrip("/")
    timeout = _positive_float(source.get("timeout_seconds", DEFAULT_HTTP_TIMEOUT_SECONDS), "timeout_seconds")
    max_bytes = _non_negative_int(source.get("max_bytes", DEFAULT_MAX_HTTP_BYTES), "max_bytes")
    headers = _jira_headers(source)
    params: dict[str, Any] = {
        "jql": str(source.get("jql") or "ORDER BY updated DESC"),
        "maxResults": _non_negative_int(source.get("max_results", 50), "max_results"),
    }
    if source.get("fields") is not None:
        params["fields"] = ",".join(_string_list(source["fields"]))
    url = _url_with_params(f"{base_url}/{endpoint}", params)
    _status_code, payload = _http_get_json(url, headers=headers, timeout=timeout, max_bytes=max_bytes)
    if not isinstance(payload, dict):
        raise VerifierError("Jira search response must be a JSON object")
    issues = payload.get("issues") or []
    if not isinstance(issues, list):
        raise VerifierError("Jira search response issues must be a list")
    return {
        "issue_count": len(issues),
        "issues": [_jira_issue_summary(issue) for issue in issues if isinstance(issue, dict)],
        "total": payload.get("total"),
    }


def _capture_s3_objects_source(source: dict[str, Any]) -> dict[str, Any]:
    bucket = _required_string(source, "bucket")
    prefix = str(source.get("prefix") or "")
    region = str(source.get("region") or os.environ.get("AWS_REGION") or "us-east-1")
    timeout = _positive_float(source.get("timeout_seconds", DEFAULT_HTTP_TIMEOUT_SECONDS), "timeout_seconds")
    max_bytes = _non_negative_int(source.get("max_bytes", DEFAULT_MAX_HTTP_BYTES), "max_bytes")
    max_keys = _non_negative_int(source.get("max_keys", 1000), "max_keys")
    url = source.get("url")
    if url:
        list_url = str(url)
    else:
        endpoint = str(source.get("endpoint_url") or f"https://s3.{region}.amazonaws.com").rstrip("/")
        list_url = f"{endpoint}/{urllib.parse.quote(bucket, safe='')}"
    params: dict[str, Any] = {"list-type": "2", "max-keys": max_keys}
    if prefix:
        params["prefix"] = prefix
    list_url = _url_with_params(list_url, params)
    headers = _headers_from_source(source)
    if not bool(source.get("unsigned", False)):
        headers.update(_aws_sigv4_headers(source, "GET", list_url, region, service="s3"))
    _status_code, payload = _http_get_text(list_url, headers=headers, timeout=timeout, max_bytes=max_bytes)
    objects = _parse_s3_list_objects(payload)
    return {
        "bucket": bucket,
        "prefix": prefix,
        "object_count": len(objects),
        "objects": objects,
    }


def _capture_microsoft_graph_messages_source(source: dict[str, Any]) -> dict[str, Any]:
    base_url = str(source.get("base_url") or "https://graph.microsoft.com/v1.0").rstrip("/")
    timeout = _positive_float(source.get("timeout_seconds", DEFAULT_HTTP_TIMEOUT_SECONDS), "timeout_seconds")
    max_bytes = _non_negative_int(source.get("max_bytes", DEFAULT_MAX_HTTP_BYTES), "max_bytes")
    headers = _bearer_headers(source, "MICROSOFT_GRAPH_TOKEN")
    user_path = _microsoft_graph_user_path(source)
    folder_id = source.get("folder_id")
    if folder_id:
        url = f"{base_url}/{user_path}/mailFolders/{urllib.parse.quote(str(folder_id), safe='')}/messages"
    else:
        url = f"{base_url}/{user_path}/messages"
    params = _odata_params(source, top_field="top", default_top=50)
    _status_code, payload = _http_get_json(_url_with_params(url, params), headers=headers, timeout=timeout, max_bytes=max_bytes)
    if not isinstance(payload, dict):
        raise VerifierError("Microsoft Graph messages response must be a JSON object")
    messages = payload.get("value") or []
    if not isinstance(messages, list):
        raise VerifierError("Microsoft Graph messages response value must be a list")
    return {
        "message_count": len(messages),
        "messages": [_microsoft_graph_message_summary(message) for message in messages if isinstance(message, dict)],
        "next_link": payload.get("@odata.nextLink"),
    }


def _capture_microsoft_graph_events_source(source: dict[str, Any]) -> dict[str, Any]:
    base_url = str(source.get("base_url") or "https://graph.microsoft.com/v1.0").rstrip("/")
    timeout = _positive_float(source.get("timeout_seconds", DEFAULT_HTTP_TIMEOUT_SECONDS), "timeout_seconds")
    max_bytes = _non_negative_int(source.get("max_bytes", DEFAULT_MAX_HTTP_BYTES), "max_bytes")
    headers = _bearer_headers(source, "MICROSOFT_GRAPH_TOKEN")
    url = f"{base_url}/{_microsoft_graph_user_path(source)}/events"
    params = _odata_params(source, top_field="top", default_top=50)
    _status_code, payload = _http_get_json(_url_with_params(url, params), headers=headers, timeout=timeout, max_bytes=max_bytes)
    if not isinstance(payload, dict):
        raise VerifierError("Microsoft Graph events response must be a JSON object")
    events = payload.get("value") or []
    if not isinstance(events, list):
        raise VerifierError("Microsoft Graph events response value must be a list")
    return {
        "event_count": len(events),
        "events": [_microsoft_graph_event_summary(event) for event in events if isinstance(event, dict)],
        "next_link": payload.get("@odata.nextLink"),
    }


def _capture_gitlab_issues_source(source: dict[str, Any]) -> dict[str, Any]:
    project_id = _required_string(source, "project_id")
    base_url = str(source.get("base_url") or "https://gitlab.com/api/v4").rstrip("/")
    timeout = _positive_float(source.get("timeout_seconds", DEFAULT_HTTP_TIMEOUT_SECONDS), "timeout_seconds")
    max_bytes = _non_negative_int(source.get("max_bytes", DEFAULT_MAX_HTTP_BYTES), "max_bytes")
    headers = _headers_from_source(source)
    if "Authorization" not in headers:
        token_env = str(source.get("token_env") or "GITLAB_TOKEN")
        token = os.environ.get(token_env)
        if not token:
            raise VerifierError(f"Missing environment variable {token_env!r} for GitLab token")
        headers["PRIVATE-TOKEN"] = token
    params: dict[str, Any] = {"per_page": _non_negative_int(source.get("per_page", 50), "per_page")}
    for field in ("state", "labels", "search"):
        if source.get(field) is not None:
            params[field] = str(source[field])
    url = _url_with_params(f"{base_url}/projects/{urllib.parse.quote(project_id, safe='')}/issues", params)
    _status_code, payload = _http_get_json(url, headers=headers, timeout=timeout, max_bytes=max_bytes)
    if not isinstance(payload, list):
        raise VerifierError("GitLab issues response must be a JSON array")
    return {
        "issue_count": len(payload),
        "issues": [_gitlab_issue_summary(issue) for issue in payload if isinstance(issue, dict)],
    }


def _capture_discord_messages_source(source: dict[str, Any]) -> dict[str, Any]:
    channel_id = _required_string(source, "channel_id")
    base_url = str(source.get("base_url") or "https://discord.com/api/v10").rstrip("/")
    timeout = _positive_float(source.get("timeout_seconds", DEFAULT_HTTP_TIMEOUT_SECONDS), "timeout_seconds")
    max_bytes = _non_negative_int(source.get("max_bytes", DEFAULT_MAX_HTTP_BYTES), "max_bytes")
    headers = _headers_from_source(source)
    if "Authorization" not in headers:
        token_env = str(source.get("token_env") or "DISCORD_BOT_TOKEN")
        token = os.environ.get(token_env)
        if not token:
            raise VerifierError(f"Missing environment variable {token_env!r} for Discord bot token")
        headers["Authorization"] = f"Bot {token}"
    params = {"limit": _non_negative_int(source.get("limit", 50), "limit")}
    url = _url_with_params(f"{base_url}/channels/{urllib.parse.quote(channel_id, safe='')}/messages", params)
    _status_code, payload = _http_get_json(url, headers=headers, timeout=timeout, max_bytes=max_bytes)
    if not isinstance(payload, list):
        raise VerifierError("Discord messages response must be a JSON array")
    return {
        "channel_id": channel_id,
        "message_count": len(payload),
        "messages": [_discord_message_summary(message, channel_id) for message in payload if isinstance(message, dict)],
    }


def _capture_zendesk_tickets_source(source: dict[str, Any]) -> dict[str, Any]:
    base_url = _required_string(source, "base_url").rstrip("/")
    timeout = _positive_float(source.get("timeout_seconds", DEFAULT_HTTP_TIMEOUT_SECONDS), "timeout_seconds")
    max_bytes = _non_negative_int(source.get("max_bytes", DEFAULT_MAX_HTTP_BYTES), "max_bytes")
    headers = _zendesk_headers(source)
    ticket_id = source.get("ticket_id")
    if ticket_id:
        url = f"{base_url}/tickets/{urllib.parse.quote(str(ticket_id), safe='')}.json"
    else:
        query = str(source.get("query") or "type:ticket")
        url = _url_with_params(f"{base_url}/search.json", {"query": query})
    _status_code, payload = _http_get_json(url, headers=headers, timeout=timeout, max_bytes=max_bytes)
    tickets = _zendesk_ticket_items(payload)
    return {
        "ticket_count": len(tickets),
        "tickets": [_zendesk_ticket_summary(ticket) for ticket in tickets if isinstance(ticket, dict)],
    }


def _capture_pagerduty_incidents_source(source: dict[str, Any]) -> dict[str, Any]:
    base_url = str(source.get("base_url") or "https://api.pagerduty.com").rstrip("/")
    timeout = _positive_float(source.get("timeout_seconds", DEFAULT_HTTP_TIMEOUT_SECONDS), "timeout_seconds")
    max_bytes = _non_negative_int(source.get("max_bytes", DEFAULT_MAX_HTTP_BYTES), "max_bytes")
    headers = _bearer_headers(source, "PAGERDUTY_API_TOKEN")
    if not (isinstance(source.get("headers"), dict) and "Accept" in source["headers"]):
        headers["Accept"] = "application/vnd.pagerduty+json;version=2"
    params: dict[str, Any] = {"limit": _non_negative_int(source.get("limit", 50), "limit")}
    statuses = source.get("statuses")
    if statuses:
        params["statuses[]"] = _string_list(statuses)
    for field in ("since", "until", "service_ids[]", "team_ids[]"):
        if source.get(field) is not None:
            params[field] = source[field] if isinstance(source[field], list) else str(source[field])
    url = _url_with_params(f"{base_url}/incidents", params)
    _status_code, payload = _http_get_json(url, headers=headers, timeout=timeout, max_bytes=max_bytes)
    if not isinstance(payload, dict):
        raise VerifierError("PagerDuty incidents response must be a JSON object")
    incidents = payload.get("incidents") or []
    if not isinstance(incidents, list):
        raise VerifierError("PagerDuty incidents response incidents must be a list")
    return {
        "incident_count": len(incidents),
        "incidents": [_pagerduty_incident_summary(incident) for incident in incidents if isinstance(incident, dict)],
        "more": bool(payload.get("more")),
    }


def _capture_sqlite_source(source: dict[str, Any], base_dir: Path, preserve_paths: bool) -> dict[str, Any]:
    path = _source_path(source, base_dir)
    if not path.exists():
        raise VerifierError(f"SQLite database does not exist: {path}")
    max_rows = _non_negative_int(source.get("max_rows", 100), "max_rows")
    queries = source.get("queries")
    if queries is None:
        query = _required_string(source, "query")
        queries = {str(source.get("query_id") or "query"): query}
    if not isinstance(queries, dict) or not queries:
        raise VerifierError("SQLite verifier queries must be a non-empty object")

    uri_path = urllib.parse.quote(path.resolve().as_posix(), safe="/:")
    uri = f"file:{uri_path}?mode=ro"
    results: dict[str, Any] = {}
    try:
        connection = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        raise VerifierError(f"Unable to open SQLite database read-only: {exc}") from exc
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        for query_id, sql in queries.items():
            if not isinstance(query_id, str) or not query_id:
                raise VerifierError("SQLite query ids must be non-empty strings")
            if not isinstance(sql, str) or not sql.strip():
                raise VerifierError(f"SQLite query {query_id!r} must be a non-empty string")
            _validate_readonly_sql(sql, query_id)
            cursor = connection.execute(sql)
            rows = cursor.fetchmany(max_rows + 1)
            columns = [item[0] for item in cursor.description or []]
            truncated = len(rows) > max_rows
            rows = rows[:max_rows]
            results[query_id] = {
                "columns": columns,
                "row_count": len(rows),
                "truncated": truncated,
                "rows": [dict(row) for row in rows],
            }
    except sqlite3.Error as exc:
        raise VerifierError(f"SQLite verifier query failed: {exc}") from exc
    finally:
        connection.close()

    return {
        "path": _display_path(path, preserve_paths),
        "queries": results,
    }


def _capture_github_issue_source(source: dict[str, Any]) -> dict[str, Any]:
    owner = _required_string(source, "owner")
    repo = _required_string(source, "repo")
    issue_number = _non_negative_int(source.get("issue_number"), "issue_number")
    base_url = str(source.get("base_url") or "https://api.github.com").rstrip("/")
    timeout = _positive_float(source.get("timeout_seconds", DEFAULT_HTTP_TIMEOUT_SECONDS), "timeout_seconds")
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "hermes-flight-recorder",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = _optional_env(source.get("token_env"))
    if token:
        headers["Authorization"] = f"Bearer {token}"
    issue_url = f"{base_url}/repos/{urllib.parse.quote(owner)}/{urllib.parse.quote(repo)}/issues/{issue_number}"
    _status_code, issue = _http_get_json(issue_url, headers=headers, timeout=timeout)
    comments: list[Any] = []
    if bool(source.get("include_comments", True)):
        comments_url = f"{issue_url}/comments"
        _comment_status, comments_payload = _http_get_json(comments_url, headers=headers, timeout=timeout)
        if not isinstance(comments_payload, list):
            raise VerifierError("GitHub comments response must be a JSON array")
        comments = comments_payload
    if not isinstance(issue, dict):
        raise VerifierError("GitHub issue response must be a JSON object")
    return {
        "issue": _github_issue_summary(issue),
        "comments": [_github_comment_summary(comment) for comment in comments if isinstance(comment, dict)],
        "comment_count": len(comments),
    }


def _capture_gmail_threads_source(source: dict[str, Any]) -> dict[str, Any]:
    token_env = str(source.get("token_env") or "GMAIL_ACCESS_TOKEN")
    token = os.environ.get(token_env)
    if not token:
        raise VerifierError(f"Gmail verifier requires access token in environment variable {token_env}")

    base_url = str(source.get("base_url") or "https://gmail.googleapis.com/gmail/v1").rstrip("/")
    user_id = urllib.parse.quote(str(source.get("user_id") or "me"), safe="")
    timeout = _positive_float(source.get("timeout_seconds", DEFAULT_HTTP_TIMEOUT_SECONDS), "timeout_seconds")
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "hermes-flight-recorder",
    }
    requested_format = str(source.get("format") or "metadata")
    include_body = bool(source.get("include_body", requested_format == "full"))
    max_threads = _non_negative_int(source.get("max_threads", 10), "max_threads")
    if max_threads == 0:
        return {
            "thread_count": 0,
            "thread_order": [],
            "threads": {},
        }

    thread_ids = _gmail_requested_thread_ids(source, base_url, user_id, headers, timeout, max_threads)
    threads: dict[str, Any] = {}
    for thread_id in thread_ids:
        params: dict[str, Any] = {"format": requested_format}
        if requested_format == "metadata":
            params["metadataHeaders"] = ["Subject", "From", "To", "Cc", "Date", "Message-ID", "In-Reply-To"]
        thread_url = f"{base_url}/users/{user_id}/threads/{urllib.parse.quote(thread_id, safe='')}"
        thread_url = _url_with_params(thread_url, params)
        _status_code, payload = _http_get_json(thread_url, headers=headers, timeout=timeout)
        if not isinstance(payload, dict):
            raise VerifierError("Gmail thread response must be a JSON object")
        summary = _gmail_thread_summary(payload, include_body=include_body)
        threads[str(summary["id"])] = summary

    return {
        "thread_count": len(threads),
        "thread_order": thread_ids,
        "threads": threads,
    }


def _gmail_requested_thread_ids(
    source: dict[str, Any],
    base_url: str,
    user_id: str,
    headers: dict[str, str],
    timeout: float,
    max_threads: int,
) -> list[str]:
    thread_id = source.get("thread_id")
    if thread_id:
        return [str(thread_id)]
    thread_ids = _string_list(source.get("thread_ids"))
    if thread_ids:
        return thread_ids[:max_threads]

    params: dict[str, Any] = {"maxResults": max_threads}
    if source.get("query"):
        params["q"] = str(source["query"])
    label_ids = _string_list(source.get("label_ids"))
    if label_ids:
        params["labelIds"] = label_ids
    list_url = _url_with_params(f"{base_url}/users/{user_id}/threads", params)
    _status_code, payload = _http_get_json(list_url, headers=headers, timeout=timeout)
    if not isinstance(payload, dict):
        raise VerifierError("Gmail thread list response must be a JSON object")
    threads = payload.get("threads") or []
    if not isinstance(threads, list):
        raise VerifierError("Gmail thread list response threads must be a list")
    result: list[str] = []
    for item in threads:
        if isinstance(item, dict) and item.get("id"):
            result.append(str(item["id"]))
        if len(result) >= max_threads:
            break
    return result


def _capture_imap_source(source: dict[str, Any]) -> dict[str, Any]:
    host = _required_string(source, "host")
    port = _non_negative_int(source.get("port", 993), "port")
    username = _string_or_env(source, "username", "username_env")
    password = _string_or_env(source, "password", "password_env")
    mailbox = str(source.get("mailbox") or "INBOX")
    search = source.get("search") or "ALL"
    search_args = _imap_search_args(search)
    max_messages = _non_negative_int(source.get("max_messages", 50), "max_messages")
    include_body = bool(source.get("include_body", True))
    timeout = _positive_float(source.get("timeout_seconds", DEFAULT_HTTP_TIMEOUT_SECONDS), "timeout_seconds")

    messages: list[dict[str, Any]] = []
    if max_messages == 0:
        return {
            "host": host,
            "mailbox": mailbox,
            "search": search if isinstance(search, str) else list(search),
            "message_count": 0,
            "messages": [],
        }
    client: Any = None
    try:
        client = imaplib.IMAP4_SSL(host, port, timeout=timeout)
        _expect_imap_ok(client.login(username, password), "login")
        _expect_imap_ok(client.select(mailbox, readonly=True), "select")
        _status, data = _expect_imap_ok(client.search(None, *search_args), "search")
        message_ids = (data[0].split() if data and data[0] else [])[-max_messages:]
        for message_id in message_ids:
            _fetch_status, fetch_data = _expect_imap_ok(client.fetch(message_id, "(UID FLAGS RFC822)"), "fetch")
            parsed = _imap_fetch_message(fetch_data, include_body)
            if parsed is not None:
                messages.append(parsed)
    except imaplib.IMAP4.error as exc:
        raise VerifierError(f"IMAP verifier failed: {exc}") from exc
    finally:
        if client is not None:
            try:
                client.logout()
            except imaplib.IMAP4.error:
                pass

    return {
        "host": host,
        "mailbox": mailbox,
        "search": search if isinstance(search, str) else list(search),
        "message_count": len(messages),
        "messages": messages,
    }


def _read_email_path(path: Path, max_messages: int, include_body: bool, preserve_paths: bool) -> list[dict[str, Any]]:
    if not path.exists():
        raise VerifierError(f"Email path does not exist: {path}")
    if max_messages == 0:
        return []
    if path.is_file():
        return [_parse_email_file(path, include_body, preserve_paths)]
    if path.is_dir():
        messages = []
        for candidate in sorted((item for item in path.iterdir() if item.is_file()), key=lambda item: item.name):
            messages.append(_parse_email_file(candidate, include_body, preserve_paths))
            if len(messages) >= max_messages:
                break
        return messages
    raise VerifierError(f"Email source path must be a file or directory: {path}")


def _maildir_files(path: Path) -> list[Path]:
    files: list[Path] = []
    for folder in ("cur", "new"):
        folder_path = path / folder
        if folder_path.is_dir():
            files.extend(item for item in folder_path.iterdir() if item.is_file())
    if not files:
        files.extend(item for item in path.iterdir() if item.is_file())
    return sorted(files, key=lambda item: str(item.relative_to(path)))


def _parse_email_file(path: Path, include_body: bool, preserve_paths: bool) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise VerifierError(f"Unable to read email file {path}: {exc}") from exc
    parsed = _parse_email_bytes(raw, include_body)
    parsed["path"] = _display_path(path, preserve_paths)
    return parsed


def _parse_email_bytes(raw: bytes, include_body: bool) -> dict[str, Any]:
    message = BytesParser(policy=email.policy.default).parsebytes(raw)
    body_text = _email_body_text(message) if include_body else ""
    return {
        "message_id": _header(message, "Message-ID"),
        "in_reply_to": _header(message, "In-Reply-To"),
        "subject": _header(message, "Subject"),
        "from": _header(message, "From"),
        "to": _all_headers(message, "To"),
        "cc": _all_headers(message, "Cc"),
        "date": _header(message, "Date"),
        "body_text": body_text[:DEFAULT_MAX_BODY_CHARS],
        "body_truncated": len(body_text) > DEFAULT_MAX_BODY_CHARS,
        "attachment_filenames": _attachment_filenames(message),
        "raw_size_bytes": len(raw),
    }


def _header(message: Message, name: str) -> str | None:
    value = message.get(name)
    return str(value) if value is not None else None


def _all_headers(message: Message, name: str) -> list[str]:
    values = message.get_all(name, [])
    return [str(value) for value in values]


def _email_body_text(message: Message) -> str:
    if isinstance(message, EmailMessage):
        body = message.get_body(preferencelist=("plain",))
        if body is not None:
            content = body.get_content()
            return content if isinstance(content, str) else str(content)
    if message.is_multipart():
        parts: list[str] = []
        for part in message.walk():
            if part.get_content_maintype() == "multipart":
                continue
            if part.get_content_type() != "text/plain":
                continue
            payload = part.get_payload(decode=True)
            if payload is None:
                continue
            charset = part.get_content_charset() or "utf-8"
            parts.append(payload.decode(charset, errors="replace"))
        return "\n".join(parts)
    payload = message.get_payload(decode=True)
    if isinstance(payload, bytes):
        charset = message.get_content_charset() or "utf-8"
        return payload.decode(charset, errors="replace")
    return str(message.get_payload() or "")


def _attachment_filenames(message: Message) -> list[str]:
    filenames: list[str] = []
    for part in message.walk() if message.is_multipart() else []:
        filename = part.get_filename()
        if filename:
            filenames.append(str(filename))
    return filenames


def _message_matches_all(message: dict[str, Any], needles: list[str]) -> bool:
    blob = json.dumps(message, sort_keys=True).lower()
    return all(needle.lower() in blob for needle in needles)


def _headers_from_source(source: dict[str, Any]) -> dict[str, str]:
    headers = {"Accept": "application/json", "User-Agent": "hermes-flight-recorder"}
    raw_headers = source.get("headers") or {}
    if not isinstance(raw_headers, dict):
        raise VerifierError("HTTP headers must be an object")
    for name, value in raw_headers.items():
        headers[str(name)] = str(value)
    raw_env_headers = source.get("headers_from_env") or {}
    if not isinstance(raw_env_headers, dict):
        raise VerifierError("HTTP headers_from_env must be an object")
    for name, env_name in raw_env_headers.items():
        value = os.environ.get(str(env_name))
        if value is None:
            raise VerifierError(f"Missing environment variable {env_name!r} for HTTP header {name!r}")
        headers[str(name)] = value
    bearer_token_env = source.get("bearer_token_env")
    if bearer_token_env:
        token = os.environ.get(str(bearer_token_env))
        if not token:
            raise VerifierError(f"Missing environment variable {bearer_token_env!r} for bearer token")
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _bearer_headers(source: dict[str, Any], default_token_env: str) -> dict[str, str]:
    headers = _headers_from_source(source)
    if "Authorization" in headers:
        return headers
    token_env = str(source.get("token_env") or source.get("bearer_token_env") or default_token_env)
    token = os.environ.get(token_env)
    if not token:
        raise VerifierError(f"Missing environment variable {token_env!r} for bearer token")
    headers["Authorization"] = f"Bearer {token}"
    return headers


def _jira_headers(source: dict[str, Any]) -> dict[str, str]:
    headers = _headers_from_source(source)
    if "Authorization" in headers:
        return headers
    if source.get("token_env") or source.get("bearer_token_env"):
        return _bearer_headers(source, "JIRA_API_TOKEN")
    username = source.get("username") or source.get("email")
    username_env = source.get("username_env") or source.get("email_env")
    if username_env:
        username = os.environ.get(str(username_env))
    token_env = str(source.get("api_token_env") or source.get("password_env") or "JIRA_API_TOKEN")
    token = os.environ.get(token_env)
    if not username or not token:
        raise VerifierError("Jira verifier requires bearer token or username/email plus API token")
    raw = f"{username}:{token}".encode("utf-8")
    headers["Authorization"] = f"Basic {base64.b64encode(raw).decode('ascii')}"
    return headers


def _http_get_json(
    url: str,
    *,
    headers: dict[str, str],
    timeout: float,
    max_bytes: int = DEFAULT_MAX_HTTP_BYTES,
) -> tuple[int, Any]:
    return _http_json("GET", url, headers=headers, timeout=timeout, max_bytes=max_bytes)


def _http_post_json(
    url: str,
    *,
    headers: dict[str, str],
    timeout: float,
    payload: Any,
    max_bytes: int = DEFAULT_MAX_HTTP_BYTES,
) -> tuple[int, Any]:
    post_headers = dict(headers)
    post_headers.setdefault("Content-Type", "application/json")
    body = json.dumps(payload).encode("utf-8")
    return _http_json("POST", url, headers=post_headers, timeout=timeout, max_bytes=max_bytes, body=body)


def _http_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    timeout: float,
    max_bytes: int,
    body: bytes | None = None,
) -> tuple[int, Any]:
    status_code, response_body = _http_request(method, url, headers=headers, timeout=timeout, max_bytes=max_bytes, body=body)
    try:
        payload = json.loads(response_body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise VerifierError(f"HTTP {method} {url} did not return valid JSON: {exc}") from exc
    return status_code, payload


def _http_get_text(
    url: str,
    *,
    headers: dict[str, str],
    timeout: float,
    max_bytes: int = DEFAULT_MAX_HTTP_BYTES,
) -> tuple[int, str]:
    status_code, response_body = _http_request("GET", url, headers=headers, timeout=timeout, max_bytes=max_bytes)
    return status_code, response_body.decode("utf-8", errors="replace")


def _http_request(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    timeout: float,
    max_bytes: int,
    body: bytes | None = None,
) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(max_bytes + 1)
            status_code = int(response.status)
    except urllib.error.HTTPError as exc:
        body = exc.read(4096).decode("utf-8", errors="replace")
        raise VerifierError(f"HTTP {method} {url} failed with status {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise VerifierError(f"HTTP {method} {url} failed: {exc.reason}") from exc
    if len(body) > max_bytes:
        raise VerifierError(f"HTTP {method} {url} exceeded max_bytes={max_bytes}")
    return status_code, body


def _payload_items(payload: Any, list_key: str, label: str) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise VerifierError(f"{label} response must be a JSON object or array")
    if list_key in payload:
        items = payload.get(list_key)
        if not isinstance(items, list):
            raise VerifierError(f"{label} response {list_key} must be a list")
        return items
    return [payload]


def _slack_message_summary(message: dict[str, Any], channel_id: str) -> dict[str, Any]:
    reactions = message.get("reactions") or []
    return {
        "type": message.get("type"),
        "subtype": message.get("subtype"),
        "ts": message.get("ts"),
        "thread_ts": message.get("thread_ts"),
        "user": message.get("user"),
        "bot_id": message.get("bot_id"),
        "channel_id": channel_id,
        "text": message.get("text") or "",
        "reaction_names": [item.get("name") for item in reactions if isinstance(item, dict)],
        "reply_count": message.get("reply_count"),
    }


def _calendar_event_summary(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": event.get("id"),
        "summary": event.get("summary") or "",
        "description": event.get("description") or "",
        "status": event.get("status"),
        "htmlLink": event.get("htmlLink"),
        "created": event.get("created"),
        "updated": event.get("updated"),
        "start": _calendar_time(event.get("start")),
        "end": _calendar_time(event.get("end")),
        "attendees": [
            attendee.get("email")
            for attendee in event.get("attendees", [])
            if isinstance(attendee, dict) and attendee.get("email")
        ],
        "creator": _nested_value(event, "creator.email"),
        "organizer": _nested_value(event, "organizer.email"),
        "conference_link": event.get("hangoutLink") or _nested_value(event, "conferenceData.entryPoints.0.uri"),
    }


def _calendar_time(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    raw = value.get("dateTime") or value.get("date")
    return str(raw) if raw is not None else None


def _drive_file_summary(item: dict[str, Any]) -> dict[str, Any]:
    owners = item.get("owners") or []
    return {
        "id": item.get("id"),
        "name": item.get("name") or "",
        "mimeType": item.get("mimeType"),
        "webViewLink": item.get("webViewLink"),
        "createdTime": item.get("createdTime"),
        "modifiedTime": item.get("modifiedTime"),
        "trashed": bool(item.get("trashed")),
        "md5Checksum": item.get("md5Checksum"),
        "size": item.get("size"),
        "owners": [
            owner.get("emailAddress") or owner.get("displayName")
            for owner in owners
            if isinstance(owner, dict) and (owner.get("emailAddress") or owner.get("displayName"))
        ],
    }


def _kubernetes_resource_summary(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    status = item.get("status") if isinstance(item.get("status"), dict) else {}
    spec = item.get("spec") if isinstance(item.get("spec"), dict) else {}
    conditions = status.get("conditions") if isinstance(status.get("conditions"), list) else []
    return {
        "apiVersion": item.get("apiVersion"),
        "kind": item.get("kind"),
        "name": metadata.get("name"),
        "namespace": metadata.get("namespace"),
        "uid": metadata.get("uid"),
        "labels": metadata.get("labels") if isinstance(metadata.get("labels"), dict) else {},
        "phase": status.get("phase"),
        "ready": _kubernetes_ready(item, spec, status, conditions),
        "replicas": spec.get("replicas"),
        "ready_replicas": status.get("readyReplicas"),
        "conditions": [
            {
                "type": condition.get("type"),
                "status": condition.get("status"),
                "reason": condition.get("reason"),
            }
            for condition in conditions
            if isinstance(condition, dict)
        ],
    }


def _kubernetes_ready(item: dict[str, Any], spec: dict[str, Any], status: dict[str, Any], conditions: list[Any]) -> bool:
    kind = str(item.get("kind") or "")
    for condition in conditions:
        if not isinstance(condition, dict):
            continue
        condition_type = str(condition.get("type") or "")
        if condition_type in {"Ready", "Available"}:
            return str(condition.get("status") or "").lower() == "true"
    if kind.lower() in {"deployment", "statefulset", "replicaset"}:
        desired = spec.get("replicas") or status.get("replicas") or 1
        ready = status.get("readyReplicas") or 0
        try:
            return int(ready) >= int(desired)
        except (TypeError, ValueError):
            return False
    return str(status.get("phase") or "").lower() in {"running", "succeeded"}


def _stripe_object_summary(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item.get("id"),
        "object": item.get("object"),
        "status": item.get("status"),
        "amount": item.get("amount"),
        "currency": item.get("currency"),
        "customer": item.get("customer"),
        "created": item.get("created"),
        "metadata": item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
    }


def _notion_page_summary(page: dict[str, Any]) -> dict[str, Any]:
    properties = page.get("properties") if isinstance(page.get("properties"), dict) else {}
    title = ""
    text_parts: list[str] = []
    for prop in properties.values():
        if not isinstance(prop, dict):
            continue
        prop_type = prop.get("type")
        if prop_type == "title":
            title = _notion_rich_text(prop.get("title")) or title
        elif prop_type == "rich_text":
            text = _notion_rich_text(prop.get("rich_text"))
            if text:
                text_parts.append(text)
        elif prop_type == "status" and isinstance(prop.get("status"), dict):
            text_parts.append(str(prop["status"].get("name") or ""))
        elif prop_type == "select" and isinstance(prop.get("select"), dict):
            text_parts.append(str(prop["select"].get("name") or ""))
    return {
        "id": page.get("id"),
        "url": page.get("url"),
        "created_time": page.get("created_time"),
        "last_edited_time": page.get("last_edited_time"),
        "archived": bool(page.get("archived")),
        "in_trash": bool(page.get("in_trash")),
        "title": title,
        "text": " ".join(part for part in text_parts if part),
    }


def _notion_rich_text(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        plain = item.get("plain_text")
        if plain:
            parts.append(str(plain))
    return "".join(parts)


def _linear_issue_summary(issue: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": issue.get("id"),
        "identifier": issue.get("identifier"),
        "title": issue.get("title") or "",
        "url": issue.get("url"),
        "createdAt": issue.get("createdAt"),
        "updatedAt": issue.get("updatedAt"),
        "status": _nested_value(issue, "state.name"),
        "team": _nested_value(issue, "team.key") or _nested_value(issue, "team.name"),
        "assignee": _nested_value(issue, "assignee.email") or _nested_value(issue, "assignee.name"),
    }


def _jira_issue_summary(issue: dict[str, Any]) -> dict[str, Any]:
    fields = issue.get("fields") if isinstance(issue.get("fields"), dict) else {}
    return {
        "id": issue.get("id"),
        "key": issue.get("key"),
        "summary": fields.get("summary") or "",
        "status": _nested_value(fields, "status.name"),
        "issue_type": _nested_value(fields, "issuetype.name"),
        "assignee": _nested_value(fields, "assignee.emailAddress") or _nested_value(fields, "assignee.displayName"),
        "created": fields.get("created"),
        "updated": fields.get("updated"),
        "labels": fields.get("labels") if isinstance(fields.get("labels"), list) else [],
    }


def _microsoft_graph_user_path(source: dict[str, Any]) -> str:
    user_id = str(source.get("user_id") or "me")
    if user_id == "me":
        return "me"
    return f"users/{urllib.parse.quote(user_id, safe='')}"


def _odata_params(source: dict[str, Any], *, top_field: str, default_top: int) -> dict[str, Any]:
    params: dict[str, Any] = {"$top": _non_negative_int(source.get(top_field, default_top), top_field)}
    for source_key, param_key in (
        ("select", "$select"),
        ("filter", "$filter"),
        ("search", "$search"),
        ("orderby", "$orderby"),
    ):
        if source.get(source_key) is not None:
            params[param_key] = str(source[source_key])
    return params


def _microsoft_graph_message_summary(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": message.get("id"),
        "subject": message.get("subject") or "",
        "bodyPreview": message.get("bodyPreview") or "",
        "conversationId": message.get("conversationId"),
        "isRead": message.get("isRead"),
        "receivedDateTime": message.get("receivedDateTime"),
        "sentDateTime": message.get("sentDateTime"),
        "from": _nested_value(message, "from.emailAddress.address"),
        "to": _graph_recipients(message.get("toRecipients")),
        "cc": _graph_recipients(message.get("ccRecipients")),
        "webLink": message.get("webLink"),
    }


def _microsoft_graph_event_summary(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": event.get("id"),
        "summary": event.get("subject") or "",
        "bodyPreview": event.get("bodyPreview") or "",
        "status": event.get("showAs"),
        "webLink": event.get("webLink"),
        "createdDateTime": event.get("createdDateTime"),
        "lastModifiedDateTime": event.get("lastModifiedDateTime"),
        "start": _nested_value(event, "start.dateTime"),
        "end": _nested_value(event, "end.dateTime"),
        "attendees": _graph_attendees(event.get("attendees")),
        "organizer": _nested_value(event, "organizer.emailAddress.address"),
    }


def _graph_recipients(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    recipients: list[str] = []
    for item in value:
        address = _nested_value(item, "emailAddress.address")
        if address:
            recipients.append(str(address))
    return recipients


def _graph_attendees(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    attendees: list[str] = []
    for item in value:
        address = _nested_value(item, "emailAddress.address")
        if address:
            attendees.append(str(address))
    return attendees


def _gitlab_issue_summary(issue: dict[str, Any]) -> dict[str, Any]:
    assignees = issue.get("assignees") if isinstance(issue.get("assignees"), list) else []
    return {
        "id": issue.get("id"),
        "iid": issue.get("iid"),
        "title": issue.get("title") or "",
        "description": issue.get("description") or "",
        "state": issue.get("state"),
        "labels": issue.get("labels") if isinstance(issue.get("labels"), list) else [],
        "assignees": [
            assignee.get("username") or assignee.get("name")
            for assignee in assignees
            if isinstance(assignee, dict) and (assignee.get("username") or assignee.get("name"))
        ],
        "web_url": issue.get("web_url"),
        "created_at": issue.get("created_at"),
        "updated_at": issue.get("updated_at"),
        "closed_at": issue.get("closed_at"),
    }


def _discord_message_summary(message: dict[str, Any], channel_id: str) -> dict[str, Any]:
    author = message.get("author") if isinstance(message.get("author"), dict) else {}
    return {
        "id": message.get("id"),
        "channel_id": channel_id,
        "text": message.get("content") or "",
        "timestamp": message.get("timestamp"),
        "edited_timestamp": message.get("edited_timestamp"),
        "author_id": author.get("id"),
        "author": author.get("username") or author.get("global_name"),
        "type": message.get("type"),
    }


def _zendesk_headers(source: dict[str, Any]) -> dict[str, str]:
    headers = _headers_from_source(source)
    if "Authorization" in headers:
        return headers
    if source.get("token_env") or source.get("bearer_token_env"):
        return _bearer_headers(source, "ZENDESK_API_TOKEN")
    username = source.get("username") or source.get("email")
    token_env = str(source.get("api_token_env") or "ZENDESK_API_TOKEN")
    token = os.environ.get(token_env)
    if not username or not token:
        raise VerifierError("Zendesk verifier requires bearer token or username/email plus API token")
    raw = f"{username}/token:{token}".encode("utf-8")
    headers["Authorization"] = f"Basic {base64.b64encode(raw).decode('ascii')}"
    return headers


def _zendesk_ticket_items(payload: Any) -> list[Any]:
    if not isinstance(payload, dict):
        raise VerifierError("Zendesk ticket response must be a JSON object")
    if isinstance(payload.get("ticket"), dict):
        return [payload["ticket"]]
    for key in ("tickets", "results"):
        if key in payload:
            items = payload[key]
            if not isinstance(items, list):
                raise VerifierError(f"Zendesk response {key} must be a list")
            return items
    return []


def _zendesk_ticket_summary(ticket: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": ticket.get("id"),
        "subject": ticket.get("subject") or "",
        "description": ticket.get("description") or "",
        "status": ticket.get("status"),
        "priority": ticket.get("priority"),
        "requester_id": ticket.get("requester_id"),
        "assignee_id": ticket.get("assignee_id"),
        "created_at": ticket.get("created_at"),
        "updated_at": ticket.get("updated_at"),
        "url": ticket.get("url"),
    }


def _pagerduty_incident_summary(incident: dict[str, Any]) -> dict[str, Any]:
    assignments = incident.get("assignments") if isinstance(incident.get("assignments"), list) else []
    return {
        "id": incident.get("id"),
        "incident_number": incident.get("incident_number"),
        "title": incident.get("title") or incident.get("summary") or "",
        "status": incident.get("status"),
        "urgency": incident.get("urgency"),
        "service": _nested_value(incident, "service.summary"),
        "assignees": [
            _nested_value(assignment, "assignee.summary")
            for assignment in assignments
            if _nested_value(assignment, "assignee.summary")
        ],
        "html_url": incident.get("html_url"),
        "created_at": incident.get("created_at"),
        "updated_at": incident.get("updated_at"),
    }


def _parse_s3_list_objects(xml_text: str) -> list[dict[str, Any]]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise VerifierError(f"S3 ListObjects response was not valid XML: {exc}") from exc
    objects: list[dict[str, Any]] = []
    for item in root.iter():
        if _xml_name(item.tag) != "Contents":
            continue
        objects.append(
            {
                "key": _xml_child_text(item, "Key"),
                "last_modified": _xml_child_text(item, "LastModified"),
                "etag": (_xml_child_text(item, "ETag") or "").strip('"'),
                "size": _optional_int(_xml_child_text(item, "Size")),
                "storage_class": _xml_child_text(item, "StorageClass"),
            }
        )
    return objects


def _xml_child_text(root: ET.Element, name: str) -> str | None:
    for child in root:
        if _xml_name(child.tag) == name:
            return child.text
    return None


def _xml_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _nested_value(root: Any, path: str) -> Any:
    cursor = root
    for part in path.split("."):
        if isinstance(cursor, dict):
            cursor = cursor.get(part)
        elif isinstance(cursor, list) and part.isdigit() and int(part) < len(cursor):
            cursor = cursor[int(part)]
        else:
            return None
    return cursor


def _aws_sigv4_headers(source: dict[str, Any], method: str, url: str, region: str, *, service: str) -> dict[str, str]:
    access_key_env = str(source.get("access_key_env") or "AWS_ACCESS_KEY_ID")
    secret_key_env = str(source.get("secret_key_env") or "AWS_SECRET_ACCESS_KEY")
    access_key = os.environ.get(access_key_env)
    secret_key = os.environ.get(secret_key_env)
    if not access_key or not secret_key:
        raise VerifierError(f"S3 verifier requires {access_key_env} and {secret_key_env}, or unsigned=true")
    session_token = os.environ.get(str(source.get("session_token_env") or "AWS_SESSION_TOKEN"))
    now = _datetime.datetime.utcnow()
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")
    parsed = urllib.parse.urlsplit(url)
    payload_hash = hashlib.sha256(b"").hexdigest()
    headers = {
        "Host": parsed.netloc,
        "x-amz-content-sha256": payload_hash,
        "x-amz-date": amz_date,
    }
    if session_token:
        headers["x-amz-security-token"] = session_token
    canonical_headers = "".join(f"{name.lower()}:{headers[name]}\n" for name in sorted(headers, key=str.lower))
    signed_headers = ";".join(name.lower() for name in sorted(headers, key=str.lower))
    canonical_request = "\n".join(
        [
            method,
            urllib.parse.quote(parsed.path or "/", safe="/~"),
            _canonical_query(parsed.query),
            canonical_headers,
            signed_headers,
            payload_hash,
        ]
    )
    credential_scope = f"{date_stamp}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join(
        [
            "AWS4-HMAC-SHA256",
            amz_date,
            credential_scope,
            hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        ]
    )
    signing_key = _aws_signing_key(secret_key, date_stamp, region, service)
    signature = hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    headers["Authorization"] = (
        f"AWS4-HMAC-SHA256 Credential={access_key}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    return headers


def _aws_signing_key(secret_key: str, date_stamp: str, region: str, service: str) -> bytes:
    key = ("AWS4" + secret_key).encode("utf-8")
    for value in (date_stamp, region, service, "aws4_request"):
        key = hmac.new(key, value.encode("utf-8"), hashlib.sha256).digest()
    return key


def _canonical_query(query: str) -> str:
    pairs = urllib.parse.parse_qsl(query, keep_blank_values=True)
    encoded = [
        (
            urllib.parse.quote(key, safe="-_.~"),
            urllib.parse.quote(value, safe="-_.~"),
        )
        for key, value in pairs
    ]
    return "&".join(f"{key}={value}" for key, value in sorted(encoded))


def _github_issue_summary(issue: dict[str, Any]) -> dict[str, Any]:
    return {
        "number": issue.get("number"),
        "state": issue.get("state"),
        "state_reason": issue.get("state_reason"),
        "title": issue.get("title"),
        "body": issue.get("body"),
        "html_url": issue.get("html_url"),
        "labels": [label.get("name") for label in issue.get("labels", []) if isinstance(label, dict)],
        "assignees": [user.get("login") for user in issue.get("assignees", []) if isinstance(user, dict)],
        "updated_at": issue.get("updated_at"),
        "closed_at": issue.get("closed_at"),
        "comments": issue.get("comments"),
    }


def _github_comment_summary(comment: dict[str, Any]) -> dict[str, Any]:
    user = comment.get("user") if isinstance(comment.get("user"), dict) else {}
    return {
        "id": comment.get("id"),
        "body": comment.get("body"),
        "user": user.get("login"),
        "created_at": comment.get("created_at"),
        "updated_at": comment.get("updated_at"),
        "html_url": comment.get("html_url"),
    }


def _gmail_thread_summary(thread: dict[str, Any], *, include_body: bool) -> dict[str, Any]:
    messages = thread.get("messages") or []
    if not isinstance(messages, list):
        raise VerifierError("Gmail thread messages must be a list")
    thread_id = str(thread.get("id") or "")
    return {
        "id": thread_id,
        "history_id": thread.get("historyId"),
        "message_count": len(messages),
        "messages": [_gmail_message_summary(message, include_body=include_body) for message in messages if isinstance(message, dict)],
    }


def _gmail_message_summary(message: dict[str, Any], *, include_body: bool) -> dict[str, Any]:
    payload = message.get("payload") if isinstance(message.get("payload"), dict) else {}
    headers = _gmail_headers(payload.get("headers") if isinstance(payload, dict) else [])
    body_text = _gmail_plain_text(payload) if include_body and isinstance(payload, dict) else ""
    return {
        "id": message.get("id"),
        "thread_id": message.get("threadId"),
        "label_ids": message.get("labelIds") or [],
        "snippet": message.get("snippet"),
        "internal_date": message.get("internalDate"),
        "subject": headers.get("subject"),
        "from": headers.get("from"),
        "to": headers.get("to"),
        "cc": headers.get("cc"),
        "date": headers.get("date"),
        "message_id": headers.get("message-id"),
        "in_reply_to": headers.get("in-reply-to"),
        "body_text": body_text[:DEFAULT_MAX_BODY_CHARS],
        "body_truncated": len(body_text) > DEFAULT_MAX_BODY_CHARS,
    }


def _gmail_headers(headers: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    if not isinstance(headers, list):
        return result
    for header in headers:
        if isinstance(header, dict) and header.get("name"):
            result[str(header["name"]).lower()] = str(header.get("value") or "")
    return result


def _gmail_plain_text(payload: dict[str, Any]) -> str:
    mime_type = str(payload.get("mimeType") or "")
    body = payload.get("body") if isinstance(payload.get("body"), dict) else {}
    data = body.get("data") if isinstance(body, dict) else None
    if mime_type == "text/plain" and isinstance(data, str):
        return _decode_base64url(data)
    parts = payload.get("parts") or []
    if not isinstance(parts, list):
        return ""
    chunks: list[str] = []
    for part in parts:
        if isinstance(part, dict):
            text = _gmail_plain_text(part)
            if text:
                chunks.append(text)
    return "\n".join(chunks)


def _decode_base64url(value: str) -> str:
    padded = value + "=" * (-len(value) % 4)
    try:
        return base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8", errors="replace")
    except (ValueError, UnicodeError):
        return ""


def _imap_search_args(search: Any) -> list[str]:
    if isinstance(search, list):
        return [str(item) for item in search]
    if not isinstance(search, str):
        raise VerifierError("IMAP search must be a string or list")
    return shlex.split(search) if search.strip() else ["ALL"]


def _expect_imap_ok(result: tuple[str, Any], operation: str) -> tuple[str, Any]:
    status, data = result
    if status != "OK":
        raise VerifierError(f"IMAP {operation} failed with status {status}")
    return status, data


def _imap_fetch_message(fetch_data: Any, include_body: bool) -> dict[str, Any] | None:
    raw: bytes | None = None
    uid: str | None = None
    flags: list[str] = []
    for item in fetch_data or []:
        if not isinstance(item, tuple) or len(item) < 2:
            continue
        metadata = item[0].decode("utf-8", errors="replace") if isinstance(item[0], bytes) else str(item[0])
        uid_match = re.search(r"\bUID\s+(\d+)", metadata)
        if uid_match:
            uid = uid_match.group(1)
        flags_match = re.search(r"FLAGS\s+\(([^)]*)\)", metadata)
        if flags_match:
            flags = [part for part in flags_match.group(1).split() if part]
        if isinstance(item[1], bytes):
            raw = item[1]
    if raw is None:
        return None
    parsed = _parse_email_bytes(raw, include_body)
    parsed["uid"] = uid
    parsed["flags"] = flags
    return parsed


def _source_path(source: dict[str, Any], base_dir: Path) -> Path:
    raw_path = _required_string(source, "path")
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _assign_dot_path(root: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    if not parts or any(not part for part in parts):
        raise VerifierError(f"state_path must be dot-separated and non-empty: {path}")
    if parts[0] in _RESERVED_STATE_ROOTS:
        raise VerifierError(f"state_path cannot overwrite reserved state root {parts[0]!r}")
    cursor: Any = root
    for index, part in enumerate(parts):
        is_last = index == len(parts) - 1
        next_is_list = False if is_last else parts[index + 1].isdigit()
        if isinstance(cursor, list):
            if not part.isdigit():
                raise VerifierError(f"state_path expected numeric list index at {part!r} in {path}")
            item_index = int(part)
            while len(cursor) <= item_index:
                cursor.append(None)
            if is_last:
                cursor[item_index] = value
            else:
                if cursor[item_index] is None:
                    cursor[item_index] = [] if next_is_list else {}
                cursor = cursor[item_index]
            continue
        if not isinstance(cursor, dict):
            raise VerifierError(f"state_path conflicts with scalar value at {part!r} in {path}")
        if is_last:
            cursor[part] = value
            continue
        if part not in cursor:
            cursor[part] = [] if next_is_list else {}
        elif next_is_list and not isinstance(cursor[part], list):
            raise VerifierError(f"state_path conflicts with object value at {part!r} in {path}")
        elif not next_is_list and not isinstance(cursor[part], dict):
            raise VerifierError(f"state_path conflicts with non-object value at {part!r} in {path}")
        cursor = cursor[part]


def _select_dot_path(root: Any, path: str) -> Any:
    parts = path.split(".")
    if not parts or any(not part for part in parts):
        raise VerifierError(f"state_value_path must be dot-separated and non-empty: {path}")
    cursor = root
    for part in parts:
        if isinstance(cursor, dict):
            if part not in cursor:
                raise VerifierError(f"state_value_path {path!r} missing key {part!r}")
            cursor = cursor[part]
            continue
        if isinstance(cursor, list):
            if not part.isdigit():
                raise VerifierError(f"state_value_path {path!r} expected numeric list index at {part!r}")
            index = int(part)
            if index >= len(cursor):
                raise VerifierError(f"state_value_path {path!r} index {index} out of range")
            cursor = cursor[index]
            continue
        raise VerifierError(f"state_value_path {path!r} conflicts with scalar value at {part!r}")
    return cursor


def _required_string(source: dict[str, Any], field: str) -> str:
    value = source.get(field)
    if not isinstance(value, str) or not value:
        raise VerifierError(f"Verifier source {field} must be a non-empty string")
    return value


def _string_or_env(source: dict[str, Any], value_field: str, env_field: str) -> str:
    if source.get(value_field):
        return str(source[value_field])
    env_name = source.get(env_field)
    if not env_name:
        raise VerifierError(f"Verifier source requires either {value_field} or {env_field}")
    value = os.environ.get(str(env_name))
    if value is None:
        raise VerifierError(f"Missing environment variable {env_name!r}")
    return value


def _optional_env(env_name: Any) -> str | None:
    if not env_name:
        return None
    value = os.environ.get(str(env_name))
    if value is None:
        raise VerifierError(f"Missing environment variable {env_name!r}")
    return value


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    raise VerifierError("Expected string or list of strings")


def _non_negative_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise VerifierError(f"{field} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise VerifierError(f"{field} must be a non-negative integer") from exc
    if parsed < 0:
        raise VerifierError(f"{field} must be a non-negative integer")
    return parsed


def _positive_float(value: Any, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise VerifierError(f"{field} must be a positive number") from exc
    if parsed <= 0:
        raise VerifierError(f"{field} must be a positive number")
    return parsed


def _validate_readonly_sql(sql: str, query_id: str) -> None:
    stripped = sql.lstrip().lower()
    if not (stripped.startswith("select") or stripped.startswith("with") or stripped.startswith("pragma")):
        raise VerifierError(f"SQLite query {query_id!r} must be read-only SELECT, WITH, or PRAGMA")


def _url_with_params(url: str, params: dict[str, Any]) -> str:
    query = urllib.parse.urlencode(params, doseq=True)
    if not query:
        return url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}{query}"


def _display_path(path: Path, preserve_paths: bool) -> str:
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
