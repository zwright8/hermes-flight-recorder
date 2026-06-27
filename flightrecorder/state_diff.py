"""Deterministic before/after state diffs for task-completion evidence."""

from __future__ import annotations

from typing import Any

STATE_DIFF_SCHEMA_VERSION = "hfr.state_diff.v1"
STATE_DIFF_DEFAULT_MAX_CHANGES = 200


class StateDiffError(ValueError):
    """Raised when a state diff cannot be constructed."""


def build_state_diff(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    max_changes: int = STATE_DIFF_DEFAULT_MAX_CHANGES,
) -> dict[str, Any]:
    """Return a deterministic, bounded diff between two state snapshots."""
    if max_changes < 0:
        raise StateDiffError("max_changes must be non-negative")
    changes: list[dict[str, Any]] = []
    change_count = 0

    def record(path: str, kind: str, before_value: Any, after_value: Any) -> None:
        nonlocal change_count
        change_count += 1
        if len(changes) >= max_changes:
            return
        changes.append(
            {
                "path": path or "$",
                "kind": kind,
                "before": before_value,
                "after": after_value,
            }
        )

    def walk(path: str, before_value: Any, after_value: Any) -> None:
        if before_value == after_value:
            return
        if isinstance(before_value, dict) and isinstance(after_value, dict):
            for key in sorted(set(before_value) | set(after_value)):
                child_path = _join_path(path, str(key))
                if key not in before_value:
                    record(child_path, "added", None, after_value[key])
                elif key not in after_value:
                    record(child_path, "removed", before_value[key], None)
                else:
                    walk(child_path, before_value[key], after_value[key])
            return
        if isinstance(before_value, list) and isinstance(after_value, list):
            common = min(len(before_value), len(after_value))
            for index in range(common):
                walk(_join_path(path, str(index)), before_value[index], after_value[index])
            for index in range(common, len(after_value)):
                record(_join_path(path, str(index)), "added", None, after_value[index])
            for index in range(common, len(before_value)):
                record(_join_path(path, str(index)), "removed", before_value[index], None)
            return
        record(path, "changed", before_value, after_value)

    walk("", before, after)
    truncated = change_count > len(changes)
    return {
        "schema_version": STATE_DIFF_SCHEMA_VERSION,
        "changed": change_count > 0,
        "change_count": change_count,
        "truncated": truncated,
        "max_changes": max_changes,
        "changes": changes,
        "summary": _summary(change_count, truncated),
    }


def _join_path(parent: str, child: str) -> str:
    return child if not parent else f"{parent}.{child}"


def _summary(change_count: int, truncated: bool) -> str:
    suffix = " shown with truncation" if truncated else " detected"
    return f"{change_count} state change(s){suffix}."
