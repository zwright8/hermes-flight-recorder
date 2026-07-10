"""Shared filesystem safety helpers."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator


@dataclass(frozen=True)
class _DirectoryAttestation:
    identity: tuple[int, int]
    tree_sha256: str


_ACTIVE_OUTPUT_LOCKS: ContextVar[tuple[str, ...]] = ContextVar(
    "hfr_active_output_locks",
    default=(),
)
_MACOS_SYSTEM_ROOT_SYMLINKS = {
    Path("/etc"): Path("/private/etc"),
    Path("/tmp"): Path("/private/tmp"),
    Path("/var"): Path("/private/var"),
}


def path_has_symlink_component(path: Path, *, include_leaf: bool) -> bool:
    """Return true when a path traverses a non-root symlink component."""
    if "\x00" in os.fspath(path):
        return True
    parts = [part for part in path.parts if part not in {"", "."}]
    if not parts:
        return False
    current = Path(path.anchor) if path.is_absolute() else Path.cwd()
    walk_parts = parts[1:] if path.is_absolute() else parts
    try:
        for index, part in enumerate(walk_parts):
            if not include_leaf and index == len(walk_parts) - 1:
                break
            current = current.parent if part == ".." else current / part
            if current.is_symlink():
                if _is_allowed_system_root_symlink(current):
                    # macOS exposes these fixed system paths through /private.
                    current = current.resolve(strict=False)
                    continue
                return True
    except (OSError, ValueError, RuntimeError):
        return True
    return False


def assert_output_does_not_alias_sources(
    output: Path,
    sources: list[Path],
    *,
    label: str,
) -> None:
    """Reject an output that is or resolves to one of its source files."""
    working_directory = Path.cwd()
    output_path = output if output.is_absolute() else working_directory / output
    if path_has_symlink_component(output_path, include_leaf=True):
        raise ValueError(f"{label} output must not contain symlink components: {output}")
    try:
        output_resolved = output_path.resolve(strict=False)
        output_identity = _existing_file_identity(output_path)
    except (OSError, ValueError, RuntimeError) as exc:
        raise ValueError(f"could not verify {label} output identity at {output}: {exc}") from exc

    for source in sources:
        source_path = source if source.is_absolute() else working_directory / source
        try:
            source_resolved = source_path.resolve(strict=False)
            source_identity = _existing_file_identity(source_path)
        except (OSError, ValueError, RuntimeError) as exc:
            raise ValueError(f"could not verify {label} source identity at {source}: {exc}") from exc
        if output_resolved == source_resolved or (
            output_identity is not None
            and source_identity is not None
            and output_identity == source_identity
        ):
            raise ValueError(f"{label} output must not alias source file: {source}")


def _existing_file_identity(path: Path) -> tuple[int, int] | None:
    try:
        path_stat = path.stat()
    except FileNotFoundError:
        return None
    return path_stat.st_dev, path_stat.st_ino


def resolve_artifact_reference_path(value: str, source_path: Path) -> Path:
    """Resolve a local reference beside its artifact, then from its repo root."""
    path = Path(value)
    if path.is_absolute():
        return path
    source_relative = source_path.parent / path
    if source_relative.exists():
        return source_relative
    repository_root = artifact_repository_root(source_path)
    if repository_root is not None:
        repo_relative = repository_root / path
        if repo_relative.exists():
            return repo_relative
    return source_relative


def artifact_repository_root(source_path: Path) -> Path | None:
    """Return the nearest repository marker root containing an artifact."""
    for root in source_path.resolve().parents:
        if (root / ".git").exists() or (root / "pyproject.toml").exists():
            return root
    return None


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
    """Remove a command-owned output under its canonical publication lock.

    Writers that continue using the target after this call should prefer
    :func:`locked_owned_output_directory`, which holds the lock for the complete
    write instead of only the destructive replacement step.
    """
    with locked_owned_output_directory(
        path,
        repo_root=repo_root,
        force=force,
        label=label,
        is_owned=is_owned,
        cwd=cwd,
    ):
        return


@contextmanager
def locked_owned_output_directory(
    path: Path,
    *,
    repo_root: Path,
    force: bool,
    label: str,
    is_owned: Callable[[Path], bool],
    cwd: Path | None = None,
    keep_existing: bool = False,
) -> Iterator[None]:
    """Lock a target through a complete write and safely replace prior output."""
    assert_safe_output_directory(path, repo_root=repo_root, cwd=cwd)
    with _canonical_output_lock(path, label=label, cwd=cwd):
        assert_safe_output_directory(path, repo_root=repo_root, cwd=cwd)
        if path.exists() and not path.is_dir():
            raise ValueError(f"{label} is not a directory: {path}")
        if keep_existing:
            if path.exists() and any(path.iterdir()):
                if _owned_attestation(path, is_owned) is None:
                    raise ValueError(
                        f"refusing to reuse unrecognized {label}: {path}"
                    )
        elif path.exists() and any(path.iterdir()):
            if not force:
                raise ValueError(
                    f"{label} is not empty: {path}; pass --force to replace it"
                )
            owned_attestation = _owned_attestation(path, is_owned)
            if owned_attestation is None:
                raise ValueError(
                    f"refusing to replace unrecognized {label}: {path}; "
                    "choose an empty output directory or a valid prior command output"
                )
            _quarantine_and_remove(
                path, label=label, expected=owned_attestation
            )
        yield


def json_marker_matches_schema(
    path: Path,
    marker_name: str,
    schema_name: str,
) -> bool:
    """Return whether a marker satisfies its complete bundled JSON Schema."""
    marker = path / marker_name
    if not marker.is_file() or path_has_symlink_component(marker, include_leaf=True):
        return False
    try:
        from .schema_registry import SchemaRegistryError, check_schema_file

        return check_schema_file(marker, schema_name).get("passed") is True
    except (OSError, UnicodeError, json.JSONDecodeError, SchemaRegistryError):
        return False


def output_directory_lock_is_held(
    path: Path,
    *,
    cwd: Path | None = None,
) -> bool:
    """Return whether this execution context holds the canonical target lock."""
    canonical = _canonical_output_path(path, cwd=cwd)
    return os.fspath(canonical) in _ACTIVE_OUTPUT_LOCKS.get()


def _owned_attestation(
    path: Path, is_owned: Callable[[Path], bool]
) -> _DirectoryAttestation | None:
    try:
        before = _attest_directory(path)
        owned = is_owned(path)
        after = _attest_directory(path)
    except (OSError, UnicodeError, ValueError):
        return None
    return after if owned and before == after else None


def _quarantine_and_remove(
    path: Path, *, label: str, expected: _DirectoryAttestation
) -> None:
    if _attest_directory(path) != expected:
        raise ValueError(
            f"refusing to remove {label} because its identity or contents changed after ownership validation"
        )
    quarantine = Path(
        tempfile.mkdtemp(prefix=f".{path.name or 'output'}.hfr-remove-", dir=path.parent)
    )
    quarantine.rmdir()
    try:
        path.rename(quarantine)
    except OSError as exc:
        raise ValueError(f"could not isolate prior {label} for removal: {exc}") from exc
    try:
        actual = _attest_directory(quarantine)
    except (OSError, ValueError) as exc:
        _restore_quarantine(quarantine, path)
        raise ValueError(
            f"refusing to remove {label} because its identity changed: {exc}"
        ) from exc
    if actual != expected:
        _restore_quarantine(quarantine, path)
        raise ValueError(
            f"refusing to remove {label} because its contents changed during replacement"
        )
    if not remove_directory_tree_if_identity(quarantine, expected.identity):
        raise ValueError(
            f"refusing to finish removing {label} because its quarantine changed "
            "during descriptor-relative removal; the unexpected path was retained"
        )


def _restore_quarantine(quarantine: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        return
    try:
        quarantine.rename(target)
    except OSError:
        return


def remove_directory_tree_if_identity(
    path: Path, expected_identity: tuple[int, int]
) -> bool:
    """Remove one directory through held descriptors, or retain it on a race.

    The root is opened once relative to its parent and every descendant directory
    is opened with ``O_NOFOLLOW``.  Recursive cleanup therefore remains bound to
    the attested inode even if another writer replaces the pathname.  Namespace
    identity is checked again before each ``rmdir``; an unsupported platform or
    detected concurrent mutation fails closed instead of reopening and deleting
    the replacement tree.
    """
    if path.name in {"", ".", ".."}:
        return False
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    parent_descriptor: int | None = None
    root_descriptor: int | None = None
    try:
        parent_descriptor = os.open(path.parent, directory_flags)
        root_stat = os.stat(
            path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(root_stat.st_mode)
            or (root_stat.st_dev, root_stat.st_ino) != expected_identity
        ):
            return False
        root_descriptor = os.open(
            path.name,
            directory_flags,
            dir_fd=parent_descriptor,
        )
        opened_stat = os.fstat(root_descriptor)
        if (
            not stat.S_ISDIR(opened_stat.st_mode)
            or (opened_stat.st_dev, opened_stat.st_ino) != expected_identity
        ):
            return False
        if not _remove_directory_contents_fd(root_descriptor, directory_flags):
            return False
        current_stat = os.stat(
            path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(current_stat.st_mode)
            or (current_stat.st_dev, current_stat.st_ino) != expected_identity
            or _directory_has_entries(root_descriptor)
        ):
            return False
        os.rmdir(path.name, dir_fd=parent_descriptor)
        return True
    except (NotImplementedError, OSError, TypeError, ValueError):
        return False
    finally:
        if root_descriptor is not None:
            os.close(root_descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)


def _remove_directory_contents_fd(
    directory_descriptor: int,
    directory_flags: int,
) -> bool:
    """Remove regular descendants relative to an already-open directory."""
    try:
        with os.scandir(directory_descriptor) as iterator:
            entries = list(iterator)
    except (NotImplementedError, OSError, TypeError):
        return False

    for entry in entries:
        child_descriptor: int | None = None
        try:
            entry_stat = entry.stat(follow_symlinks=False)
            entry_identity = (entry_stat.st_dev, entry_stat.st_ino)
            if stat.S_ISDIR(entry_stat.st_mode):
                child_descriptor = os.open(
                    entry.name,
                    directory_flags,
                    dir_fd=directory_descriptor,
                )
                child_stat = os.fstat(child_descriptor)
                if (
                    not stat.S_ISDIR(child_stat.st_mode)
                    or (child_stat.st_dev, child_stat.st_ino) != entry_identity
                    or not _remove_directory_contents_fd(
                        child_descriptor, directory_flags
                    )
                ):
                    return False
                current_stat = os.stat(
                    entry.name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISDIR(current_stat.st_mode)
                    or (current_stat.st_dev, current_stat.st_ino) != entry_identity
                    or _directory_has_entries(child_descriptor)
                ):
                    return False
                os.rmdir(entry.name, dir_fd=directory_descriptor)
            elif stat.S_ISREG(entry_stat.st_mode):
                file_flags = (
                    os.O_RDONLY
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_NONBLOCK", 0)
                    | getattr(os, "O_CLOEXEC", 0)
                )
                child_descriptor = os.open(
                    entry.name,
                    file_flags,
                    dir_fd=directory_descriptor,
                )
                child_stat = os.fstat(child_descriptor)
                current_stat = os.stat(
                    entry.name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISREG(child_stat.st_mode)
                    or (child_stat.st_dev, child_stat.st_ino) != entry_identity
                    or (current_stat.st_dev, current_stat.st_ino) != entry_identity
                ):
                    return False
                os.unlink(entry.name, dir_fd=directory_descriptor)
            else:
                return False
        except (NotImplementedError, OSError, TypeError):
            return False
        finally:
            if child_descriptor is not None:
                os.close(child_descriptor)
    return not _directory_has_entries(directory_descriptor)


def _directory_has_entries(directory_descriptor: int) -> bool:
    try:
        with os.scandir(directory_descriptor) as iterator:
            return next(iterator, None) is not None
    except (NotImplementedError, OSError, TypeError):
        return True


def _attest_directory(path: Path) -> _DirectoryAttestation:
    path_stat = path.stat(follow_symlinks=False)
    if not stat.S_ISDIR(path_stat.st_mode):
        raise ValueError(f"output path is not a directory: {path}")
    identity = (path_stat.st_dev, path_stat.st_ino)
    digest = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        if child.is_symlink():
            raise ValueError(f"output directory contains a symlink: {child}")
        child_stat = child.stat(follow_symlinks=False)
        relative = child.relative_to(path).as_posix()
        if stat.S_ISDIR(child_stat.st_mode):
            kind = b"d"
            content_hash = ""
        elif stat.S_ISREG(child_stat.st_mode):
            kind = b"f"
            content_hash = _sha256_file(child)
            after_hash_stat = child.stat(follow_symlinks=False)
            if (
                child_stat.st_dev,
                child_stat.st_ino,
                child_stat.st_size,
                child_stat.st_mtime_ns,
            ) != (
                after_hash_stat.st_dev,
                after_hash_stat.st_ino,
                after_hash_stat.st_size,
                after_hash_stat.st_mtime_ns,
            ):
                raise ValueError(
                    f"output file changed while being checked: {child}"
                )
        else:
            raise ValueError(f"output directory contains a non-regular path: {child}")
        digest.update(kind)
        digest.update(b"\0")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(child_stat.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(content_hash.encode("ascii"))
        digest.update(b"\0")
    if _path_identity(path) != identity:
        raise ValueError(f"output directory identity changed while being checked: {path}")
    return _DirectoryAttestation(identity=identity, tree_sha256=digest.hexdigest())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_identity(path: Path) -> tuple[int, int] | None:
    try:
        path_stat = path.stat(follow_symlinks=False)
    except OSError:
        return None
    return path_stat.st_dev, path_stat.st_ino


@contextmanager
def _canonical_output_lock(
    path: Path, *, label: str, cwd: Path | None
) -> Iterator[None]:
    working_directory = (cwd or Path.cwd()).resolve(strict=False)
    lexical_path = path if path.is_absolute() else working_directory / path
    if path_has_symlink_component(lexical_path.parent, include_leaf=True):
        raise ValueError(f"{label} lock path traverses a symlink: {path}")
    lexical_path.parent.mkdir(parents=True, exist_ok=True)
    if path_has_symlink_component(lexical_path.parent, include_leaf=True):
        raise ValueError(f"{label} lock path traverses a symlink: {path}")
    canonical = _canonical_output_path(path, cwd=cwd)
    has_effective_uid = hasattr(os, "geteuid")
    effective_uid = os.geteuid() if has_effective_uid else os.getpid()
    lock_root = Path(tempfile.gettempdir()) / f"hfr-output-locks-{effective_uid}"
    if lock_root.is_symlink() or path_has_symlink_component(
        lock_root, include_leaf=True
    ):
        raise ValueError(f"{label} lock directory is unsafe: {lock_root}")
    lock_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_root_stat = lock_root.stat(follow_symlinks=False)
    if (
        not stat.S_ISDIR(lock_root_stat.st_mode)
        or (
            has_effective_uid
            and hasattr(lock_root_stat, "st_uid")
            and lock_root_stat.st_uid != effective_uid
        )
    ):
        raise ValueError(f"{label} lock directory is unsafe: {lock_root}")
    lock_root.chmod(0o700)
    lock_key = hashlib.sha256(os.fsencode(canonical)).hexdigest()
    lock_path = lock_root / f"{lock_key}.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise ValueError(f"{label} lock is unsafe or unavailable: {exc}") from exc
    flock_module = None
    fallback_owner: Path | None = None
    try:
        descriptor_stat = os.fstat(descriptor)
        lock_stat = lock_path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(descriptor_stat.st_mode)
            or (descriptor_stat.st_dev, descriptor_stat.st_ino)
            != (lock_stat.st_dev, lock_stat.st_ino)
        ):
            raise ValueError(f"{label} lock changed while being opened: {lock_path}")
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        try:
            import fcntl as flock_module
        except ImportError:  # pragma: no cover - supported Unix platforms have fcntl
            fallback_owner = lock_path.with_suffix(f"{lock_path.suffix}.owner")
            try:
                owner_fd = os.open(
                    fallback_owner,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                )
            except FileExistsError as exc:
                raise ValueError(f"{label} is locked for publication: {path}") from exc
            else:
                os.close(owner_fd)
        else:
            try:
                flock_module.flock(
                    descriptor, flock_module.LOCK_EX | flock_module.LOCK_NB
                )
            except (BlockingIOError, PermissionError) as exc:
                raise ValueError(f"{label} is locked for publication: {path}") from exc
            locked_path_stat = lock_path.stat(follow_symlinks=False)
            if (locked_path_stat.st_dev, locked_path_stat.st_ino) != (
                descriptor_stat.st_dev,
                descriptor_stat.st_ino,
            ):
                raise ValueError(f"{label} lock changed while being acquired: {lock_path}")
        active_token = _ACTIVE_OUTPUT_LOCKS.set(
            (*_ACTIVE_OUTPUT_LOCKS.get(), os.fspath(canonical))
        )
        try:
            yield
        finally:
            _ACTIVE_OUTPUT_LOCKS.reset(active_token)
    finally:
        if flock_module is not None:
            try:
                flock_module.flock(descriptor, flock_module.LOCK_UN)
            except OSError:
                pass
        os.close(descriptor)
        if fallback_owner is not None:
            fallback_owner.unlink(missing_ok=True)


def _canonical_output_path(path: Path, *, cwd: Path | None) -> Path:
    working_directory = (cwd or Path.cwd()).resolve(strict=False)
    lexical_path = path if path.is_absolute() else working_directory / path
    return lexical_path.resolve(strict=False)


def _is_allowed_system_root_symlink(path: Path) -> bool:
    if sys.platform != "darwin":
        return False
    expected = _MACOS_SYSTEM_ROOT_SYMLINKS.get(path)
    if expected is None:
        return False
    try:
        return path.resolve(strict=True) == expected
    except OSError:
        return False


def _is_same_or_ancestor(candidate: Path, protected_path: Path) -> bool:
    try:
        protected_path.relative_to(candidate)
    except ValueError:
        return False
    return True
