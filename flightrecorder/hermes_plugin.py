"""Read-only Hermes observer collector.

This module is intentionally tiny and fail-open so it can be used as a Hermes
plugin adapter without changing Hermes runtime behavior.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HOOKS = (
    "on_session_start",
    "on_session_end",
    "on_session_finalize",
    "on_session_reset",
    "pre_llm_call",
    "post_llm_call",
    "pre_api_request",
    "post_api_request",
    "api_request_error",
    "pre_tool_call",
    "post_tool_call",
    "pre_approval_request",
    "post_approval_response",
    "subagent_start",
    "subagent_stop",
)
LIVE_SMOKE_SUMMARY_SCHEMA_VERSION = "hfr.live_smoke.summary.v2"

_LOCK = threading.Lock()


def register(ctx: Any) -> None:
    """Register read-only observer hooks with a Hermes plugin context."""
    for hook in HOOKS:
        try:
            ctx.register_hook(hook, _make_writer(hook))
        except Exception:
            continue


def _make_writer(hook: str):
    def _writer(**kwargs: Any) -> None:
        try:
            write_event(hook, kwargs)
        except Exception:
            return None

    return _writer


def write_event(hook: str, payload: dict[str, Any]) -> Path:
    """Append one observer event and return the output path."""
    output_dir = Path(os.environ.get("HERMES_FLIGHT_RECORDER_OUTPUT_DIR", ".hfr-events"))
    output_dir.mkdir(parents=True, exist_ok=True)
    session_id = str(payload.get("session_id") or payload.get("parent_session_id") or "session")
    path = output_dir / f"{_safe_name(session_id)}.observer.jsonl"
    event = {
        "hook": hook,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "payload": _bound(payload, _max_chars()),
    }
    line = json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
    with _LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
    return path


def _bound(value: Any, max_chars: int) -> Any:
    if isinstance(value, str):
        return value if len(value) <= max_chars else value[:max_chars] + "...[truncated]"
    if isinstance(value, list):
        return [_bound(item, max_chars) for item in value]
    if isinstance(value, tuple):
        return [_bound(item, max_chars) for item in value]
    if isinstance(value, dict):
        return {str(key): _bound(item, max_chars) for key, item in value.items()}
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def _max_chars() -> int:
    raw = os.environ.get("HERMES_FLIGHT_RECORDER_MAX_FIELD_CHARS", "12000")
    try:
        return max(100, int(raw))
    except ValueError:
        return 12000


def _safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)[:120]
    return safe.strip("._") or "session"
