"""External-state snapshot loading for deterministic task evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .redaction import sanitize_trace


class StateSnapshotError(ValueError):
    """Raised when a state snapshot cannot be loaded."""


def resolve_state_snapshot_path(scenario: dict[str, Any], override: str | Path | None = None) -> Path | None:
    """Resolve a scenario after/post-run state snapshot path, if configured."""
    return _resolve_state_path(scenario, override, ("path", "after_path"))


def resolve_before_state_snapshot_path(scenario: dict[str, Any], override: str | Path | None = None) -> Path | None:
    """Resolve a scenario before/pre-run state snapshot path, if configured."""
    return _resolve_state_path(scenario, override, ("before_path",))


def _resolve_state_path(
    scenario: dict[str, Any],
    override: str | Path | None,
    keys: tuple[str, ...],
) -> Path | None:
    has_override = override is not None
    raw_path: str | Path | None = override
    if raw_path is None:
        state = scenario.get("state")
        if isinstance(state, dict):
            for key in keys:
                if state.get(key):
                    raw_path = str(state[key])
                    break
    if raw_path is None:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if has_override:
        return path.resolve()
    base = Path(scenario.get("_base_dir") or ".")
    return (base / path).resolve()


def load_state_snapshot(path: str | Path) -> dict[str, Any]:
    """Load a JSON object snapshot of post-run external state."""
    snapshot_path = Path(path)
    try:
        value = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise StateSnapshotError(f"Unable to read state snapshot {snapshot_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise StateSnapshotError(f"Invalid JSON in state snapshot {snapshot_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise StateSnapshotError(f"State snapshot {snapshot_path} must contain a JSON object")
    return value


def sanitize_state_snapshot(snapshot: dict[str, Any], secret_patterns: list[str] | None = None) -> dict[str, Any]:
    """Return a redacted copy of a state snapshot."""
    return sanitize_trace(snapshot, secret_patterns or [])
