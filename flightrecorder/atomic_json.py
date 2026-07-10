"""Crash-safe, compare-and-swap JSON file updates."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .path_safety import path_has_symlink_component

try:  # pragma: no cover - Windows fallback is exercised on Windows only.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]
    import msvcrt
else:  # pragma: no cover - the Windows branch imports msvcrt.
    msvcrt = None  # type: ignore[assignment]


class AtomicJsonError(ValueError):
    """Raised when a guarded JSON update cannot be completed safely."""


def json_file_sha256(path: str | Path) -> str | None:
    """Return the current file digest, or ``None`` when the file is absent."""
    target = Path(path)
    _reject_symlinked_path(target)
    if not target.exists():
        return None
    if not target.is_file():
        raise AtomicJsonError(f"JSON target is not a regular file: {target}")
    return _sha256(target)


def atomic_write_json_cas(
    path: str | Path,
    value: dict[str, Any],
    *,
    expected_sha256: str | None,
) -> str:
    """Atomically write JSON only when the on-disk version is unchanged."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlinked_path(target)
    lock_path = target.with_name(f".{target.name}.hfr.lock")
    _reject_symlinked_path(lock_path)

    with lock_path.open("a+b") as lock_handle:
        _lock(lock_handle)
        try:
            _reject_symlinked_path(target)
            current_sha256 = _sha256(target) if target.is_file() else None
            if target.exists() and not target.is_file():
                raise AtomicJsonError(f"JSON target is not a regular file: {target}")
            if current_sha256 != expected_sha256:
                raise AtomicJsonError(
                    f"JSON target changed concurrently: {target}; reload it before retrying"
                )
            rendered = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            replacement = _write_temporary_file(target, rendered)
            try:
                os.replace(replacement, target)
                _fsync_directory(target.parent)
            finally:
                replacement.unlink(missing_ok=True)
            return _sha256(target)
        finally:
            _unlock(lock_handle)


def _write_temporary_file(target: Path, rendered: str) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    replacement = Path(raw_path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        if target.exists():
            os.chmod(replacement, target.stat().st_mode & 0o777)
        return replacement
    except Exception:
        replacement.unlink(missing_ok=True)
        raise


def _reject_symlinked_path(path: Path) -> None:
    if path_has_symlink_component(path, include_leaf=True):
        raise AtomicJsonError(f"JSON target must not contain symlink components: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lock(handle: Any) -> None:
    if fcntl is not None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    elif msvcrt is not None:  # pragma: no cover
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)


def _unlock(handle: Any) -> None:
    if fcntl is not None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    elif msvcrt is not None:  # pragma: no cover
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":  # pragma: no cover
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
