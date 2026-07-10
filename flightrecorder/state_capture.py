"""Capture deterministic external-state evidence snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import re
from itertools import islice
from pathlib import Path
from typing import Any

from .path_safety import path_has_symlink_component
from .state import sanitize_state_snapshot

STATE_SNAPSHOT_SCHEMA_VERSION = "hfr.state_snapshot.v1"
_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class StateCaptureError(ValueError):
    """Raised when a state snapshot cannot be captured."""


def capture_state_snapshot(
    *,
    files: list[tuple[str, str | Path]] | None = None,
    directories: list[tuple[str, str | Path]] | None = None,
    json_sources: list[tuple[str, str | Path]] | None = None,
    observations: list[tuple[str, Any]] | None = None,
    include_file_text: bool = False,
    max_text_chars: int = 4096,
    max_dir_entries: int = 200,
    preserve_paths: bool = False,
    secret_patterns: list[str] | None = None,
) -> dict[str, Any]:
    """Build a redacted JSON state snapshot from local evidence sources."""
    if max_text_chars < 0:
        raise StateCaptureError("max_text_chars must be non-negative")
    if max_dir_entries < 0:
        raise StateCaptureError("max_dir_entries must be non-negative")

    snapshot: dict[str, Any] = {
        "schema_version": STATE_SNAPSHOT_SCHEMA_VERSION,
        "filesystem": {
            "files": {},
            "directories": {},
        },
        "json_sources": {},
        "json": {},
        "observations": {},
    }

    for key, raw_path in files or []:
        _validate_key(key)
        snapshot["filesystem"]["files"][key] = _file_record(
            Path(raw_path),
            preserve_paths,
            include_file_text,
            max_text_chars,
        )

    for key, raw_path in directories or []:
        _validate_key(key)
        snapshot["filesystem"]["directories"][key] = _directory_record(Path(raw_path), preserve_paths, max_dir_entries)

    for key, raw_path in json_sources or []:
        _validate_key(key)
        path = Path(raw_path)
        record = _file_record(path, preserve_paths, include_text=False, max_text_chars=0)
        snapshot["json_sources"][key] = record
        if path.exists():
            if not path.is_file():
                raise StateCaptureError(f"JSON source {path} is not a file")
            try:
                snapshot["json"][key] = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise StateCaptureError(f"Invalid JSON in source {path}: {exc}") from exc
            except OSError as exc:
                raise StateCaptureError(f"Unable to read JSON source {path}: {exc}") from exc

    for path, value in observations or []:
        _assign_observation(snapshot["observations"], path, value)

    return sanitize_state_snapshot(snapshot, secret_patterns or [])


def _file_record(path: Path, preserve_paths: bool, include_text: bool, max_text_chars: int) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": _display_path(path, preserve_paths),
        "exists": path.exists(),
    }
    if not path.exists():
        record["kind"] = "missing"
        return record
    if path.is_file():
        record["kind"] = "file"
        stat = path.stat()
        record["size_bytes"] = stat.st_size
        record["sha256"] = _sha256(path)
        if include_text:
            text = _read_text(path, max_text_chars)
            record["text"] = text[:max_text_chars]
            record["text_truncated"] = len(text) > max_text_chars
    elif path.is_dir():
        record["kind"] = "directory"
    else:
        record["kind"] = "other"
    return record


def _directory_record(path: Path, preserve_paths: bool, max_entries: int) -> dict[str, Any]:
    if path_has_symlink_component(path, include_leaf=True):
        raise StateCaptureError(f"Directory source {path} must not traverse a symlink component")
    record: dict[str, Any] = {
        "path": _display_path(path, preserve_paths),
        "exists": path.exists(),
    }
    if not path.exists():
        record["kind"] = "missing"
        return record
    if not path.is_dir():
        record["kind"] = "not_directory"
        return record

    scan_limit = max_entries + 1
    with os.scandir(path) as directory:
        scanned_entries = list(islice(directory, scan_limit))
    entries = sorted((path / entry.name for entry in scanned_entries), key=lambda item: item.name)
    scan_incomplete = len(scanned_entries) == scan_limit
    record["kind"] = "directory"
    # Bounded scans cannot promise a globally lexical selection. Sort only the
    # observed prefix and explicitly expose when the count is a lower bound.
    record["entry_count"] = len(entries)
    record["entry_count_is_lower_bound"] = scan_incomplete
    record["scanned_entry_count"] = len(scanned_entries)
    record["scan_limit"] = scan_limit
    record["scan_incomplete"] = scan_incomplete
    record["entry_selection"] = "lexicographic_within_scanned_prefix"
    record["entries_truncated"] = len(entries) > max_entries
    record["entries"] = [_directory_entry(item) for item in entries[:max_entries]]
    return record


def _directory_entry(path: Path) -> dict[str, Any]:
    entry: dict[str, Any] = {"name": path.name}
    if path.is_symlink():
        entry["kind"] = "other"
    elif path.is_file():
        stat = path.stat()
        entry.update({"kind": "file", "size_bytes": stat.st_size, "sha256": _sha256(path)})
    elif path.is_dir():
        entry["kind"] = "directory"
    elif path.exists():
        entry["kind"] = "other"
    else:
        entry["kind"] = "missing"
    return entry


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_text(path: Path, max_chars: int) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return handle.read(max_chars + 1)
    except OSError as exc:
        raise StateCaptureError(f"Unable to read text from {path}: {exc}") from exc


def _assign_observation(root: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    if not parts or any(not part for part in parts):
        raise StateCaptureError(f"Observation path must be dot-separated and non-empty: {path}")
    cursor: Any = root
    for index, part in enumerate(parts):
        is_last = index == len(parts) - 1
        if isinstance(cursor, list):
            if not part.isdigit():
                raise StateCaptureError(f"Observation path expected numeric list index at {part!r} in {path}")
            item_index = int(part)
            while len(cursor) <= item_index:
                cursor.append(None)
            if is_last:
                cursor[item_index] = value
            else:
                if cursor[item_index] is None:
                    cursor[item_index] = [] if parts[index + 1].isdigit() else {}
                cursor = cursor[item_index]
            continue

        if not isinstance(cursor, dict):
            raise StateCaptureError(f"Observation path conflicts with scalar value at {part!r} in {path}")
        if is_last:
            cursor[part] = value
        else:
            next_is_list = parts[index + 1].isdigit()
            if part not in cursor:
                cursor[part] = [] if next_is_list else {}
            elif next_is_list and not isinstance(cursor[part], list):
                raise StateCaptureError(f"Observation path conflicts with object value at {part!r} in {path}")
            elif not next_is_list and not isinstance(cursor[part], dict):
                raise StateCaptureError(
                    f"Observation path conflicts with non-object value at {part!r} in {path}"
                )
            cursor = cursor[part]


def _validate_key(key: str) -> None:
    if not _KEY_RE.fullmatch(key):
        raise StateCaptureError(f"Snapshot source key must match {_KEY_RE.pattern}: {key!r}")


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
