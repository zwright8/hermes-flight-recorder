"""Deterministic before/after state diffs for task-completion evidence."""

from __future__ import annotations

import hashlib
from typing import Any

from .json_semantics import json_values_equal

STATE_DIFF_SCHEMA_VERSION = "hfr.state_diff.v1"
STATE_DIFF_DEFAULT_MAX_CHANGES = 200
STATE_DIFF_DEFAULT_MAX_DEPTH = 64
STATE_DIFF_DEFAULT_MAX_NODES = 10_000

_VALUE_MAX_DEPTH = 4
_VALUE_MAX_ITEMS = 16
_VALUE_MAX_NODES = 64
_VALUE_MAX_STRING_CHARS = 512
_PATH_MAX_CHARS = 512
_SUMMARY_KEY = "$hfr_summary"


class StateDiffError(ValueError):
    """Raised when a state diff cannot be constructed."""


class _ChangeLimitReached(Exception):
    """Stop walking as soon as the configured change limit is reached."""


class _NodeLimitReached(Exception):
    """Stop walking when the snapshot comparison budget is exhausted."""


def build_state_diff(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    max_changes: int = STATE_DIFF_DEFAULT_MAX_CHANGES,
    max_depth: int = STATE_DIFF_DEFAULT_MAX_DEPTH,
    max_nodes: int = STATE_DIFF_DEFAULT_MAX_NODES,
) -> dict[str, Any]:
    """Return a deterministic diff with bounded traversal and change values.

    ``change_count`` remains exact for fully traversed snapshots. When the
    change limit is exceeded, traversal stops after discovering the first
    omitted change and the count is an explicit lower bound. Depth and node
    limits can also make comparison incomplete; those exceptional outputs are
    annotated without changing the normal artifact shape.
    """
    _require_limit("max_changes", max_changes, minimum=0)
    _require_limit("max_depth", max_depth, minimum=0)
    _require_limit("max_nodes", max_nodes, minimum=1)

    changes: list[dict[str, Any]] = []
    change_count = 0
    nodes_visited = 0
    change_count_exact = True
    comparison_truncated = False
    truncation_reason: str | None = None

    def mark_truncated(reason: str) -> None:
        nonlocal change_count_exact, comparison_truncated, truncation_reason
        change_count_exact = False
        comparison_truncated = True
        if truncation_reason is None or reason in {"max_changes", "max_nodes"}:
            truncation_reason = reason

    def record(path: str, kind: str, before_value: Any, after_value: Any) -> None:
        nonlocal change_count
        change_count += 1
        if len(changes) >= max_changes:
            mark_truncated("max_changes")
            raise _ChangeLimitReached
        changes.append(
            {
                "path": path or "$",
                "kind": kind,
                "before": _bounded_value(before_value),
                "after": _bounded_value(after_value),
            }
        )

    def compare_at_limit(path: str, before_value: Any, after_value: Any, reason: str) -> None:
        before_summary, before_exact = _bounded_value_with_exactness(before_value)
        after_summary, after_exact = _bounded_value_with_exactness(after_value)
        if before_exact and after_exact:
            if not json_values_equal(before_summary, after_summary):
                record(path, "changed", before_summary, after_summary)
            return
        mark_truncated(reason)
        # A differing bounded preview proves a change. Matching previews leave
        # the comparison explicitly incomplete instead of inventing a change.
        if not json_values_equal(before_summary, after_summary):
            record(path, "changed", before_summary, after_summary)

    def walk(path: str, before_value: Any, after_value: Any, depth: int) -> None:
        nonlocal nodes_visited
        if nodes_visited >= max_nodes:
            mark_truncated("max_nodes")
            compare_at_limit(path, before_value, after_value, "max_nodes")
            raise _NodeLimitReached
        nodes_visited += 1

        before_is_dict = isinstance(before_value, dict)
        after_is_dict = isinstance(after_value, dict)
        if before_is_dict and after_is_dict:
            if before_value is after_value:
                return
            if depth >= max_depth:
                compare_at_limit(path, before_value, after_value, "max_depth")
                return
            # Avoid constructing an unbounded key union when the remaining
            # traversal budget cannot possibly inspect the container.
            if len(before_value) + len(after_value) > max_nodes - nodes_visited + 1:
                compare_at_limit(path, before_value, after_value, "max_nodes")
                return
            keys = sorted(set(before_value) | set(after_value), key=str)
            for key in keys:
                child_path = _join_path(path, str(key))
                if key not in before_value:
                    record(child_path, "added", None, after_value[key])
                elif key not in after_value:
                    record(child_path, "removed", before_value[key], None)
                else:
                    walk(child_path, before_value[key], after_value[key], depth + 1)
            return

        before_is_list = isinstance(before_value, list)
        after_is_list = isinstance(after_value, list)
        if before_is_list and after_is_list:
            if before_value is after_value:
                return
            if depth >= max_depth:
                compare_at_limit(path, before_value, after_value, "max_depth")
                return
            common = min(len(before_value), len(after_value))
            for index in range(common):
                walk(_join_path(path, str(index)), before_value[index], after_value[index], depth + 1)
            for index in range(common, len(after_value)):
                record(_join_path(path, str(index)), "added", None, after_value[index])
            for index in range(common, len(before_value)):
                record(_join_path(path, str(index)), "removed", before_value[index], None)
            return

        # Containers of unlike kinds cannot be equal. Avoid invoking their
        # potentially deep equality implementations before recording them.
        if before_is_dict or after_is_dict or before_is_list or after_is_list:
            record(path, "changed", before_value, after_value)
            return
        if not json_values_equal(before_value, after_value):
            record(path, "changed", before_value, after_value)

    if _snapshot_has_incomplete_capture(before) or _snapshot_has_incomplete_capture(after):
        mark_truncated("incomplete_snapshot")

    try:
        walk("", before, after, 0)
    except (_ChangeLimitReached, _NodeLimitReached):
        pass

    truncated = change_count > len(changes) or comparison_truncated
    change_status = "changed" if change_count > 0 else "unknown" if comparison_truncated else "unchanged"
    result: dict[str, Any] = {
        "schema_version": STATE_DIFF_SCHEMA_VERSION,
        "changed": change_count > 0,
        "change_count": change_count,
        "truncated": truncated,
        "comparison_complete": not comparison_truncated,
        "change_status": change_status,
        "max_changes": max_changes,
        "changes": changes,
        "summary": _summary(
            change_count,
            truncated,
            exact=change_count_exact,
            comparison_truncated=comparison_truncated,
            reason=truncation_reason,
        ),
    }
    if not change_count_exact:
        result["change_count_exact"] = False
    if comparison_truncated:
        result["comparison_truncated"] = True
        result["truncation_reason"] = truncation_reason
    return result


def _snapshot_has_incomplete_capture(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    filesystem = value.get("filesystem")
    if not isinstance(filesystem, dict):
        return False
    directories = filesystem.get("directories")
    if not isinstance(directories, dict):
        return False
    return any(_directory_capture_is_incomplete(record) for record in directories.values())


def _directory_capture_is_incomplete(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    # ``scan_incomplete`` is the current explicit marker. Older snapshots only
    # exposed a truncated entry list or a lower-bound count; both mean unseen
    # directory entries may exist and therefore an unchanged comparison cannot
    # be proven.
    return any(
        record.get(field_name) is True
        for field_name in ("scan_incomplete", "entries_truncated", "entry_count_is_lower_bound")
    )


def resolve_state_diff_semantics(state_diff: dict[str, Any]) -> tuple[bool, str]:
    """Return fail-closed comparison completeness and change status.

    Modern diffs carry ``comparison_complete`` and ``change_status`` directly.
    The fallback markers keep reports and digests safe when they consume older
    diffs or an artifact that has not yet been validated.
    """
    change_count = state_diff.get("change_count")
    proven_change = state_diff.get("changed") is True or (
        isinstance(change_count, int) and not isinstance(change_count, bool) and change_count > 0
    )
    explicit_complete = state_diff.get("comparison_complete")
    comparison_complete = explicit_complete is True
    if not isinstance(explicit_complete, bool):
        comparison_complete = not any(
            (
                state_diff.get("comparison_truncated") is True,
                state_diff.get("change_count_exact") is False,
                state_diff.get("truncated") is True,
            )
        )
    elif any(
        (
            state_diff.get("comparison_truncated") is True,
            state_diff.get("change_count_exact") is False,
        )
    ):
        comparison_complete = False

    explicit_status = state_diff.get("change_status")
    if proven_change or explicit_status == "changed":
        return comparison_complete, "changed"
    if explicit_status == "unknown" or not comparison_complete:
        return False, "unknown"
    return True, "unchanged"


def _require_limit(name: str, value: int, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if minimum == 0 else "positive"
        raise StateDiffError(f"{name} must be a {qualifier} integer")


def _bounded_value(value: Any) -> Any:
    bounded, _ = _bounded_value_with_exactness(value)
    return bounded


def _bounded_value_with_exactness(value: Any) -> tuple[Any, bool]:
    remaining_nodes = _VALUE_MAX_NODES

    def visit(current: Any, depth: int) -> tuple[Any, bool]:
        nonlocal remaining_nodes
        if remaining_nodes <= 0:
            return _value_summary(current), False
        remaining_nodes -= 1

        if current is None or isinstance(current, (bool, int, float)):
            return current, True
        if isinstance(current, str):
            if len(current) <= _VALUE_MAX_STRING_CHARS:
                return current, True
            return {
                _SUMMARY_KEY: {
                    "type": "string",
                    "length": len(current),
                    "preview": current[:_VALUE_MAX_STRING_CHARS],
                }
            }, False
        if isinstance(current, dict):
            if depth >= _VALUE_MAX_DEPTH or len(current) > _VALUE_MAX_ITEMS:
                return _container_summary(current, "object"), False
            bounded: dict[str, Any] = {}
            exact = True
            for key in sorted(current, key=str):
                bounded_key, key_exact = _bounded_key(str(key))
                bounded_item, item_exact = visit(current[key], depth + 1)
                bounded[bounded_key] = bounded_item
                exact = exact and key_exact and item_exact
            return bounded, exact
        if isinstance(current, (list, tuple)):
            if depth >= _VALUE_MAX_DEPTH or len(current) > _VALUE_MAX_ITEMS:
                return _container_summary(current, "array"), False
            bounded_items = []
            exact = isinstance(current, list)
            for item in current:
                bounded_item, item_exact = visit(item, depth + 1)
                bounded_items.append(bounded_item)
                exact = exact and item_exact
            return bounded_items, exact
        return _value_summary(current), False

    return visit(value, 0)


def _container_summary(value: Any, kind: str) -> dict[str, Any]:
    return {_SUMMARY_KEY: {"type": kind, "size": len(value), "truncated": True}}


def _value_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        kind = "object"
        size: int | None = len(value)
    elif isinstance(value, (list, tuple)):
        kind = "array"
        size = len(value)
    elif isinstance(value, str):
        kind = "string"
        size = len(value)
    else:
        value_type = type(value)
        kind = f"{value_type.__module__}.{value_type.__qualname__}"
        size = None
    summary: dict[str, Any] = {"type": kind, "truncated": True}
    if size is not None:
        summary["size"] = size
    return {_SUMMARY_KEY: summary}


def _bounded_key(key: str) -> tuple[str, bool]:
    if len(key) <= _VALUE_MAX_STRING_CHARS:
        return key, True
    digest = hashlib.sha256(key.encode("utf-8", errors="surrogatepass")).hexdigest()[:12]
    prefix_chars = _VALUE_MAX_STRING_CHARS - len(digest) - 5
    return f"{key[:prefix_chars]}...[{digest}]", False


def _join_path(parent: str, child: str) -> str:
    path = child if not parent else f"{parent}.{child}"
    if len(path) <= _PATH_MAX_CHARS:
        return path
    digest = hashlib.sha256(path.encode("utf-8", errors="surrogatepass")).hexdigest()[:12]
    prefix_chars = _PATH_MAX_CHARS - len(digest) - 5
    return f"{path[:prefix_chars]}...[{digest}]"


def _summary(
    change_count: int,
    truncated: bool,
    *,
    exact: bool,
    comparison_truncated: bool,
    reason: str | None,
) -> str:
    if not exact:
        if change_count == 0:
            return f"No state change was proven; comparison stopped at {reason}, so the result is unknown."
        return f"At least {change_count} state change(s) detected; comparison stopped at {reason}."
    suffix = " shown with truncation" if truncated else " detected"
    summary = f"{change_count} state change(s){suffix}."
    if comparison_truncated:
        summary += f" Comparison stopped at {reason}."
    return summary
