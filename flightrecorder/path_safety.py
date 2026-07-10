"""Shared filesystem safety helpers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable


def path_has_symlink_component(path: Path, *, include_leaf: bool) -> bool:
    """Return true when a path traverses a non-root symlink component."""
    parts = [part for part in path.parts if part not in {"", "."}]
    if not parts:
        return False
    current = Path(path.anchor) if path.is_absolute() else Path.cwd()
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


def assert_safe_output_directory(path: Path, *, repo_root: Path, cwd: Path | None = None) -> None:
    """Reject a directory target that is unsafe to remove recursively."""
    working_directory = (cwd or Path.cwd()).resolve(strict=False)
    repository_root = repo_root.resolve(strict=False)
    lexical_path = path if path.is_absolute() else working_directory / path

    if path_has_symlink_component(lexical_path, include_leaf=True):
        raise ValueError(f"refusing to remove output directory with a symlink component: {path}")

    resolved_path = lexical_path.resolve(strict=False)
    if resolved_path == Path(resolved_path.anchor):
        raise ValueError(f"refusing to remove filesystem root as an output directory: {path}")

    protected_roots = (
        ("working directory", working_directory),
        ("repository root", repository_root),
    )
    for label, protected_root in protected_roots:
        if _is_same_or_ancestor(resolved_path, protected_root):
            raise ValueError(f"refusing to remove protected {label} or its ancestor as an output directory: {path}")


def replace_owned_output_directory(
    path: Path,
    *,
    repo_root: Path,
    force: bool,
    label: str,
    is_owned: Callable[[Path], bool],
    cwd: Path | None = None,
) -> None:
    """Remove a non-empty output only when it is safe, forced, and command-owned."""
    assert_safe_output_directory(path, repo_root=repo_root, cwd=cwd)
    if path.exists() and not path.is_dir():
        raise ValueError(f"{label} is not a directory: {path}")
    if not path.exists() or not any(path.iterdir()):
        return
    if not force:
        raise ValueError(f"{label} is not empty: {path}; pass --force to replace it")
    if not is_owned(path):
        raise ValueError(
            f"refusing to replace unrecognized {label}: {path}; "
            "choose an empty output directory or a valid prior command output"
        )
    shutil.rmtree(path)


def json_marker_has_schema_version(path: Path, marker_name: str, schema_version: str) -> bool:
    """Return whether a regular, non-symlinked JSON marker declares the expected schema."""
    marker = path / marker_name
    if not marker.is_file() or path_has_symlink_component(marker, include_leaf=True):
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("schema_version") == schema_version


def _is_root_level_path(path: Path) -> bool:
    return path.is_absolute() and path.parent == Path(path.anchor)


def _is_same_or_ancestor(candidate: Path, protected_path: Path) -> bool:
    try:
        protected_path.relative_to(candidate)
    except ValueError:
        return False
    return True
