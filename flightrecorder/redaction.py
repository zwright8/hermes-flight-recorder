"""Redaction helpers for deployable flight-recorder artifacts."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any

REDACTED_VALUE = "[REDACTED]"
_SAFE_SECRET_KEY_SUFFIXES = (
    "_budget",
    "_available",
    "_checked",
    "_count",
    "_counts",
    "_env",
    "_environment",
    "_limit",
    "_limits",
    "_name",
    "_names",
    "_present",
    "_recorded",
    "_required",
    "_status",
    "_type",
    "_usage",
)
_SECRET_KEY_MARKERS = {"authorization", "bearer", "password", "secret", "token"}
_SECRET_KEY_TERMS = {"auth", "authentication", "credential", "credentials", "cookie"}
_SECRET_KEY_QUALIFIERS = {"access", "api", "client", "private", "secret", "signing"}
_REDACTED_VALUE_RE = re.compile(r"(?i)(?:\[REDACTED\]|<redacted:[^>\r\n]+>)")
_ASSIGNMENT_PREFIX_RE = re.compile(
    r"(?<![\w.:-])"
    r"(?P<key_quote>[\"']?)(?P<key>[\w.-]+)(?P=key_quote)"
    r"[^\S\r\n]*(?P<separator>[:=])[^\S\r\n]*"
)
_CLI_FLAG_RE = re.compile(r"(?<!\S)--(?P<key>[A-Za-z0-9][A-Za-z0-9_.-]*)")
_FOLLOWING_FLAG_RE = re.compile(
    r"(?:--[A-Za-z0-9_][A-Za-z0-9_.-]*|-[A-Za-z_][A-Za-z0-9_.-]*|--?)"
    r"(?=$|[\s=;&|])"
)
_UNQUOTED_VALUE_DELIMITERS = frozenset(",;}\r\n")
_CLI_VALUE_DELIMITERS = frozenset(" \t\r\n;&|\"'")


@dataclass(frozen=True)
class _SecretAssignment:
    raw_value: str
    value_start: int
    value_end: int


class RedactionError(ValueError):
    """Raised when caller-supplied redaction rules cannot be applied safely."""


def is_secret_key(key: Any) -> bool:
    """Return whether a mapping or assignment key is expected to hold a secret."""
    normalized = _normalize_key(key)
    if not normalized or normalized.endswith(_SAFE_SECRET_KEY_SUFFIXES):
        return False
    parts = normalized.split("_")
    has_api_key = "apikey" in normalized.replace("_", "")
    return bool(
        has_api_key
        or _SECRET_KEY_MARKERS.intersection(parts)
        or _SECRET_KEY_TERMS.intersection(parts)
        or ("key" in parts and _SECRET_KEY_QUALIFIERS.intersection(parts))
    )


def is_redacted_secret_value(value: Any) -> bool:
    """Return whether a value is an accepted explicit redaction placeholder."""
    return isinstance(value, str) and _REDACTED_VALUE_RE.fullmatch(value.strip()) is not None


def is_unredacted_secret_value(value: Any) -> bool:
    """Return whether a secret-bearing key still contains a non-empty raw value."""
    if value is None or is_redacted_secret_value(value):
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (dict, list, tuple, set)):
        return bool(value)
    return bool(value)


def contains_unredacted_secret_assignment(text: str) -> bool:
    """Return whether text contains a secret assignment or CLI argument with a raw value."""
    return any(
        is_unredacted_secret_value(_unquote(assignment.raw_value))
        for assignment in (*_secret_assignments(text), *_secret_cli_arguments(text))
    )


def redact_text(text: Any, secret_patterns: list[str] | None = None) -> str:
    """Return a string with secret-looking values redacted.

    The generic assignment and CLI-argument redactors keep key names and flags
    visible for evidence while removing values. Scenario-specific patterns are
    then applied as a final pass.
    """
    patterns = _compile_patterns(secret_patterns or [])
    return _redact_text(text, patterns)


def sanitize_trace(trace: dict[str, Any], secret_patterns: list[str] | None = None) -> dict[str, Any]:
    """Deep-copy and redact string values in a normalized trace."""
    patterns = _compile_patterns(secret_patterns or [])
    return _sanitize(copy.deepcopy(trace), patterns)


def _sanitize(value: Any, secret_patterns: list[re.Pattern[str]]) -> Any:
    if isinstance(value, str):
        return _redact_text(value, secret_patterns)
    if isinstance(value, list):
        return [_sanitize(item, secret_patterns) for item in value]
    if isinstance(value, dict):
        sanitized: dict[Any, Any] = {}
        for key, item in value.items():
            if _normalize_key(key) == "credentials" and isinstance(item, (dict, list)):
                sanitized[key] = _sanitize_credential_container(item, secret_patterns)
            elif is_secret_key(key) and is_unredacted_secret_value(item):
                sanitized[key] = REDACTED_VALUE
            else:
                sanitized[key] = _sanitize(item, secret_patterns)
        return sanitized
    return value


def _sanitize_credential_container(value: Any, secret_patterns: list[re.Pattern[str]]) -> Any:
    """Preserve container shape while redacting every non-metadata credential leaf."""
    if isinstance(value, dict):
        sanitized: dict[Any, Any] = {}
        for key, item in value.items():
            normalized = _normalize_key(key)
            if normalized.endswith(_SAFE_SECRET_KEY_SUFFIXES):
                sanitized[key] = _sanitize(item, secret_patterns)
            elif is_secret_key(key) and is_unredacted_secret_value(item):
                sanitized[key] = REDACTED_VALUE
            else:
                sanitized[key] = _sanitize_credential_container(item, secret_patterns)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_credential_container(item, secret_patterns) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_credential_container(item, secret_patterns) for item in value)
    if isinstance(value, set):
        return {_sanitize_credential_container(item, secret_patterns) for item in value}
    if is_unredacted_secret_value(value):
        return REDACTED_VALUE
    return value


def _compile_patterns(secret_patterns: list[str]) -> list[re.Pattern[str]]:
    compiled: list[re.Pattern[str]] = []
    for index, pattern in enumerate(secret_patterns):
        if not isinstance(pattern, str):
            raise RedactionError(f"Invalid redaction regex at index {index}: expected a string")
        try:
            compiled.append(re.compile(pattern))
        except re.error as exc:
            raise RedactionError(f"Invalid redaction regex at index {index}: {exc}") from exc
    return compiled


def _redact_text(text: Any, secret_patterns: list[re.Pattern[str]]) -> str:
    rendered = _redact_secret_cli_arguments(_stringify(text))
    rendered = _redact_secret_assignments(rendered)
    for pattern in secret_patterns:
        rendered = pattern.sub(REDACTED_VALUE, rendered)
    return rendered


def _redact_secret_assignments(text: str) -> str:
    parts: list[str] = []
    cursor = 0
    for assignment in _secret_assignments(text):
        raw_value = assignment.raw_value
        if not is_unredacted_secret_value(_unquote(raw_value)):
            continue
        parts.extend((text[cursor : assignment.value_start], _redacted_assignment_value(raw_value)))
        cursor = assignment.value_end
    if not parts:
        return text
    parts.append(text[cursor:])
    return "".join(parts)


def _redact_secret_cli_arguments(text: str) -> str:
    parts: list[str] = []
    cursor = 0
    for argument in _secret_cli_arguments(text):
        raw_value = argument.raw_value
        if not is_unredacted_secret_value(_unquote(raw_value)):
            continue
        parts.extend((text[cursor : argument.value_start], _redacted_assignment_value(raw_value)))
        cursor = argument.value_end
    if not parts:
        return text
    parts.append(text[cursor:])
    return "".join(parts)


def _secret_assignments(text: str) -> list[_SecretAssignment]:
    assignments: list[_SecretAssignment] = []
    search_from = 0
    while match := _ASSIGNMENT_PREFIX_RE.search(text, search_from):
        value_start = match.end()
        if not is_secret_key(match.group("key")):
            search_from = value_start
            continue
        value_end = _assignment_value_end(text, value_start)
        assignments.append(
            _SecretAssignment(
                raw_value=text[value_start:value_end],
                value_start=value_start,
                value_end=value_end,
            )
        )
        search_from = max(value_end, value_start + 1)
    return assignments


def _secret_cli_arguments(text: str) -> list[_SecretAssignment]:
    arguments: list[_SecretAssignment] = []
    search_from = 0
    while match := _CLI_FLAG_RE.search(text, search_from):
        flag_end = match.end()
        search_from = max(flag_end, match.start() + 1)
        if not is_secret_key(match.group("key")):
            continue
        if flag_end >= len(text) or text[flag_end] not in " \t":
            continue
        value_start = flag_end
        while value_start < len(text) and text[value_start] in " \t":
            value_start += 1
        if value_start >= len(text) or _starts_following_flag(text, value_start):
            continue
        value_end = _cli_value_end(text, value_start)
        arguments.append(
            _SecretAssignment(
                raw_value=text[value_start:value_end],
                value_start=value_start,
                value_end=value_end,
            )
        )
        search_from = max(value_end, value_start + 1)
    return arguments


def _assignment_value_end(text: str, value_start: int) -> int:
    if value_start >= len(text):
        return value_start
    if text[value_start] in {'"', "'"}:
        return _quoted_value_end(text, value_start)

    cursor = value_start
    while cursor < len(text):
        character = text[cursor]
        if character in _UNQUOTED_VALUE_DELIMITERS:
            break
        if character in " \t":
            next_field = cursor
            while next_field < len(text) and text[next_field] in " \t":
                next_field += 1
            if _starts_adjacent_assignment(text, next_field) or _starts_cli_flag(
                text, next_field
            ):
                break
        cursor += 1
    while cursor > value_start and text[cursor - 1] in " \t":
        cursor -= 1
    return cursor


def _starts_adjacent_assignment(text: str, start: int) -> bool:
    match = _ASSIGNMENT_PREFIX_RE.match(text, start)
    return bool(
        match
        and match.end() < len(text)
        and text[match.end()] not in "=,;}\r\n"
    )


def _starts_cli_flag(text: str, start: int) -> bool:
    return _CLI_FLAG_RE.match(text, start) is not None


def _starts_following_flag(text: str, start: int) -> bool:
    return _FOLLOWING_FLAG_RE.match(text, start) is not None


def _cli_value_end(text: str, value_start: int) -> int:
    if text[value_start] in {'"', "'"}:
        return _quoted_value_end(text, value_start)

    cursor = value_start
    while cursor < len(text):
        character = text[cursor]
        if character == "\\" and cursor + 1 < len(text):
            cursor += 2
            continue
        if character in _CLI_VALUE_DELIMITERS:
            break
        cursor += 1
    return cursor


def _quoted_value_end(text: str, value_start: int) -> int:
    quote = text[value_start]
    cursor = value_start + 1
    escaped = False
    while cursor < len(text):
        character = text[cursor]
        if character in "\r\n":
            return cursor
        if character == quote and not escaped:
            return cursor + 1
        if character == "\\" and not escaped:
            escaped = True
        else:
            escaped = False
        cursor += 1
    return cursor


def _redacted_assignment_value(raw_value: str) -> str:
    if raw_value.startswith(('"', "'")):
        quote = raw_value[0]
        closing_quote = quote if len(raw_value) > 1 and raw_value.endswith(quote) else ""
        return f"{quote}{REDACTED_VALUE}{closing_quote}"
    return REDACTED_VALUE


def _unquote(value: str) -> str:
    if value.startswith(('"', "'")):
        if len(value) >= 2 and value[0] == value[-1]:
            return value[1:-1]
        return value[1:]
    return value


def _normalize_key(key: Any) -> str:
    rendered = str(key)
    rendered = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", rendered)
    rendered = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", rendered)
    return re.sub(r"[^a-z0-9]+", "_", rendered.lower()).strip("_")


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
