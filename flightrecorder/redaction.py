"""Redaction helpers for deployable flight-recorder artifacts."""

from __future__ import annotations

import copy
import json
import re
from typing import Any

GENERIC_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(?P<key>[\w.-]*(?:api[_-]?key|secret|token|password|authorization|bearer)[\w.-]*)\b"
    r"\s*[:=]\s*[\"']?[^\"'\s,;}]+"
)


def redact_text(text: Any, secret_patterns: list[str] | None = None) -> str:
    """Return a string with secret-looking values redacted.

    The generic assignment redactor keeps the key name visible for evidence
    while removing the value. Scenario-specific patterns are then applied as a
    second pass.
    """
    rendered = _stringify(text)
    rendered = GENERIC_SECRET_ASSIGNMENT_RE.sub(lambda m: f"{m.group('key')}=[REDACTED]", rendered)
    for pattern in secret_patterns or []:
        try:
            rendered = re.sub(pattern, "[REDACTED]", rendered)
        except re.error:
            continue
    return rendered


def sanitize_trace(trace: dict[str, Any], secret_patterns: list[str] | None = None) -> dict[str, Any]:
    """Deep-copy and redact string values in a normalized trace."""
    return _sanitize(copy.deepcopy(trace), secret_patterns or [])


def _sanitize(value: Any, secret_patterns: list[str]) -> Any:
    if isinstance(value, str):
        return redact_text(value, secret_patterns)
    if isinstance(value, list):
        return [_sanitize(item, secret_patterns) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize(item, secret_patterns) for key, item in value.items()}
    return value


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
