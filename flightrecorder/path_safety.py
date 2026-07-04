"""Shared filesystem safety helpers."""

from __future__ import annotations

from pathlib import Path


def path_has_symlink_component(path: Path, *, include_leaf: bool) -> bool:
    """Return true when a path traverses a non-root symlink component."""
    parts = [part for part in path.parts if part not in {"", "."}]
    if not parts:
        return False
    current = Path(path.anchor) if path.is_absolute() else Path()
    walk_parts = parts[1:] if path.is_absolute() else parts
    for index, part in enumerate(walk_parts):
        if not include_leaf and index == len(walk_parts) - 1:
            break
        current = current.parent if part == ".." else current / part
        if current.is_symlink():
            if _is_root_level_path(current):
                # macOS commonly exposes /var as a root-level symlink to /private/var.
                current = current.resolve(strict=False)
                continue
            return True
    return False


def _is_root_level_path(path: Path) -> bool:
    return path.is_absolute() and path.parent == Path(path.anchor)
