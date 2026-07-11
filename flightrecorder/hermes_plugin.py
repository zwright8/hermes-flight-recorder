"""Read-only Hermes observer collector.

This module is intentionally tiny and fail-open so it can be used as a Hermes
plugin adapter without changing Hermes runtime behavior.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .redaction import REDACTED_VALUE, is_secret_key, redact_text

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

_MAX_COLLECTION_ITEMS = 200
_MAX_DEPTH = 32
_MAX_PAYLOAD_BYTES = 1024 * 1024
_BUDGET_RESERVE_BYTES = 256
_CIRCULAR_SENTINEL = "[Circular]"
_MAX_DEPTH_SENTINEL = "[Truncated: max depth]"
_COLLECTION_SENTINEL = "[Truncated: collection items]"
_AGGREGATE_SENTINEL = "[Truncated: aggregate limit]"
_COMMAND_SEQUENCE_KEYS = {"args", "arguments", "argv", "command", "command_argv"}
_PAIR_CONTAINER_KEYS = {
    "header",
    "headers",
    "parameters",
    "params",
    "query",
    "query_parameters",
    "query_params",
}

_LOCK = threading.Lock()


class _PayloadBudget:
    def __init__(self) -> None:
        self.remaining = _MAX_PAYLOAD_BYTES - _BUDGET_RESERVE_BYTES
        self.exhausted = False

    def consume(self, size: int) -> bool:
        if size <= self.remaining:
            self.remaining -= size
            return True
        self.remaining = 0
        self.exhausted = True
        return False


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
    output_dir = Path(
        os.environ.get("HERMES_FLIGHT_RECORDER_OUTPUT_DIR", ".hfr-events")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    session_id = _safe_string(
        payload.get("session_id") or payload.get("parent_session_id") or "session"
    )
    max_chars = _max_chars()
    path = (
        output_dir
        / f"{_safe_name(session_id, stem_value=_redacted_field_text(session_id, max_chars))}.observer.jsonl"
    )
    event = {
        "hook": _redacted_field_text(hook, max_chars),
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "payload": _bound(payload, max_chars),
    }
    line = (
        json.dumps(
            event,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
    with _LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
    return path


def _bound(
    value: Any,
    max_chars: int,
    *,
    _budget: _PayloadBudget | None = None,
    _command_sequence: bool = False,
    _pair_container: bool = False,
    _depth: int = 0,
    _seen: set[int] | None = None,
) -> Any:
    budget = _budget or _PayloadBudget()
    seen = _seen if _seen is not None else set()
    if budget.exhausted:
        return _AGGREGATE_SENTINEL
    if isinstance(value, str):
        return _bounded_text(value, max_chars, budget)
    if value is None or isinstance(value, (bool, int, float)):
        return _bounded_scalar(value, budget)
    if _depth >= _MAX_DEPTH:
        return _bounded_text(_MAX_DEPTH_SENTINEL, max_chars, budget, redact=False)

    identity = id(value)
    if identity in seen:
        return _bounded_text(_CIRCULAR_SENTINEL, max_chars, budget, redact=False)

    if isinstance(value, dict):
        if not budget.consume(2):
            return _AGGREGATE_SENTINEL
        seen.add(identity)
        try:
            return _bound_mapping(
                value,
                max_chars,
                budget,
                _depth,
                seen,
                pair_container=_pair_container,
            )
        finally:
            seen.discard(identity)
    if isinstance(value, (list, tuple, set, frozenset)):
        if not budget.consume(2):
            return _AGGREGATE_SENTINEL
        seen.add(identity)
        try:
            return _bound_collection(
                value,
                max_chars,
                budget,
                _depth,
                seen,
                command_sequence=_command_sequence,
                pair_container=_pair_container,
            )
        finally:
            seen.discard(identity)

    try:
        rendered = repr(value)
    except Exception:
        rendered = f"[Unrepresentable {type(value).__name__}]"
    return _bounded_text(rendered, max_chars, budget)


def _bound_mapping(
    value: dict[Any, Any],
    max_chars: int,
    budget: _PayloadBudget,
    depth: int,
    seen: set[int],
    *,
    pair_container: bool,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    redact_pair_value = pair_container and _is_secret_pair_mapping(value)
    for index, (raw_key, item) in enumerate(value.items()):
        if index >= _MAX_COLLECTION_ITEMS:
            _append_mapping_sentinel(output, _COLLECTION_SENTINEL, max_chars, budget)
            break
        key = _redacted_field_text(_safe_string(raw_key), max_chars)
        separator_size = 1 if output else 0
        key_size = _json_bytes(key) + 1 + separator_size
        if not budget.consume(key_size):
            _append_mapping_sentinel(output, _AGGREGATE_SENTINEL, max_chars, budget)
            break
        if redact_pair_value and _normalize_observer_key(raw_key) == "value":
            bounded_item = _bounded_text(
                REDACTED_VALUE, max_chars, budget, redact=False
            )
        elif is_secret_key(raw_key):
            bounded_item = _bounded_text(
                REDACTED_VALUE, max_chars, budget, redact=False
            )
        else:
            bounded_item = _bound(
                item,
                max_chars,
                _budget=budget,
                _command_sequence=(
                    isinstance(item, (list, tuple))
                    and _is_command_sequence_key(raw_key)
                ),
                _pair_container=(pair_container or _is_pair_container_key(raw_key)),
                _depth=depth + 1,
                _seen=seen,
            )
        output[key] = bounded_item
        if budget.exhausted:
            break
    return output


def _bound_collection(
    value: list[Any] | tuple[Any, ...] | set[Any] | frozenset[Any],
    max_chars: int,
    budget: _PayloadBudget,
    depth: int,
    seen: set[int],
    *,
    command_sequence: bool,
    pair_container: bool,
) -> list[Any]:
    output: list[Any] = []
    redact_next = False
    redact_pair_value = pair_container and _is_secret_pair_array(value)
    for index, item in enumerate(value):
        if index >= _MAX_COLLECTION_ITEMS:
            _append_collection_sentinel(output, _COLLECTION_SENTINEL, max_chars, budget)
            break
        if output and not budget.consume(1):
            _append_collection_sentinel(output, _AGGREGATE_SENTINEL, max_chars, budget)
            break
        if redact_pair_value and index == 1:
            output.append(
                _bounded_text(REDACTED_VALUE, max_chars, budget, redact=False)
            )
            if budget.exhausted:
                break
            continue
        if command_sequence and redact_next:
            if _is_cli_flag_token(item):
                redact_next = False
            elif _is_sequence_scalar(item):
                output.append(
                    _bounded_text(REDACTED_VALUE, max_chars, budget, redact=False)
                )
                redact_next = False
                if budget.exhausted:
                    break
                continue
            else:
                redact_next = False
        output.append(
            _bound(
                item,
                max_chars,
                _budget=budget,
                _command_sequence=(
                    command_sequence and isinstance(item, (list, tuple))
                ),
                _pair_container=pair_container,
                _depth=depth + 1,
                _seen=seen,
            )
        )
        if budget.exhausted:
            break
        if command_sequence:
            redact_next = _is_secret_cli_flag_token(item)
    return output


def _bounded_scalar(value: None | bool | int | float, budget: _PayloadBudget) -> Any:
    try:
        size = _json_bytes(value)
    except (TypeError, ValueError):
        return _bounded_text(_safe_string(value), _max_chars(), budget)
    if budget.consume(size):
        return value
    return _AGGREGATE_SENTINEL


def _bounded_text(
    value: Any,
    max_chars: int,
    budget: _PayloadBudget,
    *,
    redact: bool = True,
) -> str:
    text = _redacted_field_text(value, max_chars) if redact else _safe_string(value)
    if _json_bytes(text) <= budget.remaining:
        budget.consume(_json_bytes(text))
        return text

    budget.exhausted = True
    available = budget.remaining
    budget.remaining = 0
    low = 0
    high = len(text)
    while low < high:
        midpoint = (low + high + 1) // 2
        if _json_bytes(text[:midpoint]) <= available:
            low = midpoint
        else:
            high = midpoint - 1
    return text[:low] + _AGGREGATE_SENTINEL


def _redacted_field_text(value: Any, max_chars: int) -> str:
    text = redact_text(_safe_string(value))
    text = text.encode("utf-8", errors="replace").decode("utf-8")
    if len(text) > max_chars:
        return text[:max_chars] + "...[truncated]"
    return text


def _safe_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return str(value)
    except Exception:
        return f"[Unrepresentable {type(value).__name__}]"


def _is_command_sequence_key(key: Any) -> bool:
    return _normalize_observer_key(key) in _COMMAND_SEQUENCE_KEYS


def _is_pair_container_key(key: Any) -> bool:
    return _normalize_observer_key(key) in _PAIR_CONTAINER_KEYS


def _normalize_observer_key(key: Any) -> str:
    rendered = _safe_string(key)
    rendered = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", rendered)
    rendered = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", rendered)
    return re.sub(r"[^a-z0-9]+", "_", rendered.casefold()).strip("_")


def _is_secret_pair_array(value: Any) -> bool:
    return bool(
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[0], str)
        and is_secret_key(value[0])
    )


def _is_secret_pair_mapping(value: dict[Any, Any]) -> bool:
    return any(
        _normalize_observer_key(key) in {"key", "name"}
        and isinstance(item, str)
        and is_secret_key(item)
        for key, item in value.items()
    )


def _is_cli_flag_token(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    if value in {"-", "--"}:
        return True
    if value.startswith("--") and len(value) > 2:
        return value[2].isalnum() or value[2] == "_"
    return bool(
        value.startswith("-")
        and len(value) > 1
        and (value[1].isalpha() or value[1] == "_")
    )


def _is_secret_cli_flag_token(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("--") or len(value) <= 2:
        return False
    key = value[2:]
    if "=" in key or ":" in key:
        return False
    return all(
        character.isalnum() or character in "_.-" for character in key
    ) and is_secret_key(key)


def _is_sequence_scalar(value: Any) -> bool:
    return not isinstance(value, (dict, list, tuple, set, frozenset))


def _json_bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))


def _append_collection_sentinel(
    output: list[Any],
    sentinel: str,
    max_chars: int,
    budget: _PayloadBudget,
) -> None:
    if not budget.exhausted and output and not budget.consume(1):
        output.append(_AGGREGATE_SENTINEL)
        return
    if budget.exhausted:
        output.append(sentinel)
        return
    output.append(_bounded_text(sentinel, max_chars, budget, redact=False))


def _append_mapping_sentinel(
    output: dict[str, Any],
    sentinel: str,
    max_chars: int,
    budget: _PayloadBudget,
) -> None:
    key = "__truncated__"
    while key in output:
        key = f"_{key}"
    if budget.exhausted:
        output[key] = sentinel
        return
    key_size = _json_bytes(key) + 1 + (1 if output else 0)
    if not budget.consume(key_size):
        output[key] = _AGGREGATE_SENTINEL
        return
    output[key] = _bounded_text(sentinel, max_chars, budget, redact=False)


def _max_chars() -> int:
    raw = os.environ.get("HERMES_FLIGHT_RECORDER_MAX_FIELD_CHARS", "12000")
    try:
        return max(100, int(raw))
    except ValueError:
        return 12000


def _safe_name(value: str, *, stem_value: str | None = None) -> str:
    visible = value if stem_value is None else stem_value
    stem = "".join(
        ch if ch.isascii() and (ch.isalnum() or ch in {"-", "_", "."}) else "_"
        for ch in visible
    )[:96]
    stem = stem.strip("._") or "session"
    digest = hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()
    return f"{stem}-{digest}"
