"""Read-only verifier adapters for external-state evidence snapshots."""

from __future__ import annotations

import base64
import email.policy
import imaplib
import json
import os
import re
import shlex
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
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


def _http_get_json(
    url: str,
    *,
    headers: dict[str, str],
    timeout: float,
    max_bytes: int = DEFAULT_MAX_HTTP_BYTES,
) -> tuple[int, Any]:
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(max_bytes + 1)
            status_code = int(response.status)
    except urllib.error.HTTPError as exc:
        body = exc.read(4096).decode("utf-8", errors="replace")
        raise VerifierError(f"HTTP GET {url} failed with status {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise VerifierError(f"HTTP GET {url} failed: {exc.reason}") from exc
    if len(body) > max_bytes:
        raise VerifierError(f"HTTP GET {url} exceeded max_bytes={max_bytes}")
    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise VerifierError(f"HTTP GET {url} did not return valid JSON: {exc}") from exc
    return status_code, payload


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
    return f"{url}?{query}" if query else url


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
