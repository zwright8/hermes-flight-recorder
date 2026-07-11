"""Shared filesystem safety helpers."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import secrets
import stat
import sys
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator


DIRECTORY_CONTENT_HASH_ALGORITHM = "hfr.sha256.file-tree.v1"
DIRECTORY_NAMESPACE_HASH_ALGORITHM = "hfr.sha256.namespace-tree.v1"


class AtomicNamespaceMutationError(RuntimeError):
    """Report that an atomic rename completed but durability confirmation failed.

    Callers must not handle this like a pre-mutation rename error: the namespace
    operation has already succeeded in the live filesystem and must be reconciled
    explicitly before an ordinary failure can be reported.
    """

    mutation_applied = True


@dataclass(frozen=True)
class _DirectoryAttestation:
    identity: tuple[int, int]
    tree_sha256: str


@dataclass(frozen=True)
class _OpenedDirectoryComponent:
    parent_descriptor: int | None
    descriptor: int
    name: str | None
    signature: tuple[int, ...]


@dataclass(frozen=True)
class DirectoryNamespaceAttestation:
    """Exact local namespace state used for destructive compare-and-swap.

    ``records`` retains every descendant path, entry kind, identity/metadata
    signature, and regular-file content hash (or symlink target).  The compact
    fields mirror :func:`fingerprint_directory_namespace`; callers that may
    delete a tree must retain the records instead of relying on the digest
    alone.
    """

    namespace_hash_algorithm: str
    sha256: str
    entry_count: int
    file_count: int
    size_bytes: int
    contains_symlinks: bool
    contains_special_entries: bool
    records: tuple[tuple[bytes, bytes, tuple[int, ...], bytes], ...]

    def summary(self) -> dict[str, int | str | bool]:
        return {
            "namespace_hash_algorithm": self.namespace_hash_algorithm,
            "sha256": self.sha256,
            "entry_count": self.entry_count,
            "file_count": self.file_count,
            "size_bytes": self.size_bytes,
            "contains_symlinks": self.contains_symlinks,
            "contains_special_entries": self.contains_special_entries,
        }


class DirectoryCleanupStatus(str, Enum):
    """Truthful terminal state for an attested destructive cleanup."""

    RETAINED = "retained"
    PARTIAL = "partial"
    COMPLETE = "complete"
    COMPLETE_DURABILITY_UNCONFIRMED = "complete_durability_unconfirmed"


@dataclass(frozen=True)
class DirectoryCleanupOutcome:
    """Describe exactly how far an attested cleanup progressed.

    Paths are relative to the parent descriptor supplied to
    :func:`remove_directory_entry_tree_if_identity`.  ``recovery_entries``
    names surviving approved or possibly approved data; ``concurrent_entries``
    names unrelated public entries observed during reconciliation;
    ``cleanup_artifacts`` names private vaults that still exist, including the
    parent of recovery entries.
    Truthiness is retained for existing callers and is true only for a complete,
    durability-confirmed cleanup.
    """

    status: DirectoryCleanupStatus
    recovery_entries: tuple[str, ...] = ()
    concurrent_entries: tuple[str, ...] = ()
    cleanup_artifacts: tuple[str, ...] = ()
    removed_entry_count: int = 0
    durability_confirmed: bool = True
    detail: str | None = None

    def __bool__(self) -> bool:
        return self.status is DirectoryCleanupStatus.COMPLETE


@dataclass
class _DirectoryCleanupProgress:
    removed_entry_count: int = 0
    detail: str | None = None

    def failed(self, detail: str) -> bool:
        if self.detail is None:
            self.detail = detail
        return False


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


def assert_output_outside_source_directories(
    output: Path,
    sources: list[Path],
    *,
    label: str,
) -> None:
    """Reject an output located inside any source directory."""
    working_directory = Path.cwd()
    output_path = output if output.is_absolute() else working_directory / output
    if path_has_symlink_component(output_path, include_leaf=True):
        raise ValueError(f"{label} output must not contain symlink components: {output}")
    try:
        output_resolved = output_path.resolve(strict=False)
    except (OSError, ValueError, RuntimeError) as exc:
        raise ValueError(f"could not verify {label} output location at {output}: {exc}") from exc

    for source in sources:
        source_path = source if source.is_absolute() else working_directory / source
        if path_has_symlink_component(source_path, include_leaf=True):
            raise ValueError(f"{label} source directory must not contain symlink components: {source}")
        try:
            source_resolved = source_path.resolve(strict=True)
        except (OSError, ValueError, RuntimeError) as exc:
            raise ValueError(f"could not verify {label} source directory at {source}: {exc}") from exc
        if not source_resolved.is_dir():
            raise ValueError(f"{label} source is not a directory: {source}")
        try:
            output_resolved.relative_to(source_resolved)
        except ValueError:
            continue
        raise ValueError(f"{label} output must not be inside source directory: {source}")


def fingerprint_directory_contents(
    path: Path,
    *,
    relative_files: tuple[str, ...] | None = None,
    reject_undeclared: bool = False,
    selected_files: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Return a descriptor-bound digest for a regular, symlink-free file tree."""
    _require_descriptor_tree_support()
    if path_has_symlink_component(path, include_leaf=True):
        raise ValueError(f"directory tree must not contain symlink components: {path}")
    declared_files = _normalize_declared_files(relative_files)
    selected = _normalize_declared_files(selected_files)
    with _open_directory_path_bound(path) as root_descriptor:
        try:
            root_before = os.fstat(root_descriptor)
            layout_before = _capture_directory_layout_fd(root_descriptor)
            _validate_declared_directory_layout(
                path,
                layout_before,
                declared_files,
                reject_undeclared=reject_undeclared,
            )
            files_to_hash = (
                sorted(declared_files)
                if declared_files is not None
                else sorted(
                    relative
                    for relative, signature in layout_before.items()
                    if stat.S_ISREG(signature[0])
                )
            )
            _validate_selected_fingerprint_files(
                path,
                layout_before,
                files_to_hash,
                selected,
            )

            digest = hashlib.sha256()
            size_bytes = 0
            selected_file_sha256: dict[str, str] = {}
            for relative in files_to_hash:
                expected_signature = layout_before[relative]
                content_sha256 = _sha256_tree_file_fd(
                    root_descriptor,
                    relative,
                    layout_before,
                )
                digest.update(relative.encode("utf-8"))
                digest.update(b"\0")
                digest.update(str(expected_signature[3]).encode("ascii"))
                digest.update(b"\0")
                digest.update(content_sha256.encode("ascii"))
                digest.update(b"\0")
                size_bytes += expected_signature[3]
                if selected is not None and relative in selected:
                    selected_file_sha256[relative] = content_sha256

            layout_after = _capture_directory_layout_fd(root_descriptor)
            if layout_after != layout_before:
                raise ValueError(
                    f"directory tree layout changed while being fingerprinted: {path}"
                )
            root_after = os.fstat(root_descriptor)
            if _stat_signature(root_after) != _stat_signature(root_before):
                raise ValueError(
                    f"directory tree root changed while being fingerprinted: {path}"
                )
        except (NotImplementedError, OSError, TypeError) as exc:
            raise ValueError(
                f"could not fingerprint directory tree through descriptors: {path}: {exc}"
            ) from exc

    result: dict[str, Any] = {
        "tree_hash_algorithm": DIRECTORY_CONTENT_HASH_ALGORITHM,
        "sha256": digest.hexdigest(),
        "file_count": len(files_to_hash),
        "size_bytes": size_bytes,
        "contains_symlinks": False,
    }
    if selected_files is not None:
        result["selected_file_sha256"] = selected_file_sha256
    return result


def fingerprint_directory_entry(
    parent_descriptor: int,
    name: str,
    *,
    display_path: Path,
    relative_files: tuple[str, ...] | None = None,
    reject_undeclared: bool = False,
    selected_files: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Fingerprint one directory entry relative to an already pinned parent."""
    _require_descriptor_tree_support()
    _require_leaf_name(name)
    declared_files = _normalize_declared_files(relative_files)
    selected = _normalize_declared_files(selected_files)
    try:
        namespace_before = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(namespace_before.st_mode):
            raise ValueError(f"directory tree root is not a directory: {display_path}")
        descriptor = os.open(
            name,
            _directory_descriptor_flags(),
            dir_fd=parent_descriptor,
        )
        try:
            opened_before = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(opened_before.st_mode)
                or _stat_signature(opened_before)
                != _stat_signature(namespace_before)
            ):
                raise ValueError(
                    f"directory tree entry changed while being opened: {display_path}"
                )
            result = _fingerprint_open_directory(
                descriptor,
                display_path=display_path,
                declared_files=declared_files,
                reject_undeclared=reject_undeclared,
                selected_files=selected,
                expose_selected_files=selected_files is not None,
            )
            opened_after = os.fstat(descriptor)
            namespace_after = os.stat(
                name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                _stat_signature(opened_after) != _stat_signature(opened_before)
                or _stat_signature(namespace_after)
                != _stat_signature(opened_before)
            ):
                raise ValueError(
                    f"directory tree entry changed while being fingerprinted: {display_path}"
                )
            return result
        finally:
            os.close(descriptor)
    except (NotImplementedError, OSError, TypeError) as exc:
        raise ValueError(
            f"could not fingerprint directory tree through descriptors: {display_path}: {exc}"
        ) from exc


def fingerprint_directory_namespace(
    path: Path,
) -> dict[str, int | str | bool]:
    """Attest a complete directory namespace for local compare-and-swap use.

    Unlike ``fingerprint_directory_contents``, this local attestation includes
    empty directories, filesystem entry kinds, symlink targets, special-entry
    metadata, and per-entry identity.  It is intentionally a separate contract
    so the portable ``file-tree.v1`` digest remains unchanged.
    """
    return attest_directory_namespace(path).summary()


def attest_directory_namespace(path: Path) -> DirectoryNamespaceAttestation:
    """Capture the complete descriptor-bound namespace for later safe cleanup."""
    _require_descriptor_tree_support()
    if path_has_symlink_component(path, include_leaf=True):
        raise ValueError(f"directory namespace must not contain symlink components: {path}")
    with _open_directory_path_bound(path) as root_descriptor:
        try:
            return _attest_open_directory_namespace(
                root_descriptor,
                display_path=path,
            )
        except (NotImplementedError, OSError, TypeError) as exc:
            raise ValueError(
                f"could not attest directory namespace through descriptors: {path}: {exc}"
            ) from exc


def fingerprint_directory_namespace_entry(
    parent_descriptor: int,
    name: str,
    *,
    display_path: Path,
) -> dict[str, int | str | bool]:
    """Attest one directory namespace entry relative to a pinned parent."""
    return attest_directory_namespace_entry(
        parent_descriptor,
        name,
        display_path=display_path,
    ).summary()


def attest_directory_namespace_entry(
    parent_descriptor: int,
    name: str,
    *,
    display_path: Path,
) -> DirectoryNamespaceAttestation:
    """Capture one complete namespace relative to an already pinned parent."""
    _require_descriptor_tree_support()
    _require_leaf_name(name)
    try:
        namespace_before = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(namespace_before.st_mode):
            raise ValueError(f"directory namespace root is not a directory: {display_path}")
        descriptor = os.open(
            name,
            _directory_descriptor_flags(),
            dir_fd=parent_descriptor,
        )
        try:
            opened_before = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(opened_before.st_mode)
                or _stat_signature(opened_before) != _stat_signature(namespace_before)
            ):
                raise ValueError(
                    f"directory namespace entry changed while being opened: {display_path}"
                )
            result = _attest_open_directory_namespace(
                descriptor,
                display_path=display_path,
            )
            opened_after = os.fstat(descriptor)
            namespace_after = os.stat(
                name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                _stat_signature(opened_after) != _stat_signature(opened_before)
                or _stat_signature(namespace_after) != _stat_signature(opened_before)
            ):
                raise ValueError(
                    f"directory namespace entry changed while being fingerprinted: {display_path}"
                )
            return result
        finally:
            os.close(descriptor)
    except (NotImplementedError, OSError, TypeError) as exc:
        raise ValueError(
            "could not attest directory namespace through descriptors: "
            f"{display_path}: {exc}"
        ) from exc


def sync_directory_tree(path: Path) -> None:
    """Make every regular file and directory in a private tree durable."""
    _require_descriptor_tree_support()
    with _open_directory_path_bound(path) as descriptor:
        try:
            _sync_directory_tree_fd(descriptor)
        except (NotImplementedError, OSError, TypeError) as exc:
            raise ValueError(f"could not sync directory tree: {path}: {exc}") from exc


@contextmanager
def locked_output_path(
    path: Path,
    *,
    label: str,
    cwd: Path | None = None,
) -> Iterator[None]:
    """Hold the canonical publication lock for an output pathname."""
    with _canonical_output_lock(path, label=label, cwd=cwd):
        yield


@contextmanager
def opened_directory_descriptor(path: Path) -> Iterator[int]:
    """Yield a descriptor-bound directory for relative namespace operations."""
    with _open_directory_path_bound(path) as descriptor:
        yield descriptor


def atomic_rename_entry_noreplace(
    parent_descriptor: int,
    source_name: str,
    target_name: str,
) -> None:
    """Atomically rename one sibling entry and reject an existing target."""
    _sync_directory_descriptor(parent_descriptor)
    _native_rename_entry(
        parent_descriptor,
        source_name,
        target_name,
        exchange=False,
    )
    try:
        _sync_directory_descriptor(parent_descriptor)
    except (OSError, ValueError) as exc:
        raise AtomicNamespaceMutationError(
            "atomic no-replace rename completed, but parent directory durability "
            "could not be confirmed"
        ) from exc


def atomic_exchange_entries(
    parent_descriptor: int,
    left_name: str,
    right_name: str,
) -> None:
    """Atomically exchange two sibling namespace entries without a gap."""
    _sync_directory_descriptor(parent_descriptor)
    _native_rename_entry(
        parent_descriptor,
        left_name,
        right_name,
        exchange=True,
    )
    try:
        _sync_directory_descriptor(parent_descriptor)
    except (OSError, ValueError) as exc:
        raise AtomicNamespaceMutationError(
            "atomic exchange completed, but parent directory durability could not "
            "be confirmed"
        ) from exc


def remove_directory_entry_tree_if_identity(
    parent_descriptor: int,
    name: str,
    expected_identity: tuple[int, int],
    expected_namespace: DirectoryNamespaceAttestation,
) -> DirectoryCleanupOutcome:
    """Remove only the exact descendant namespace approved by the caller.

    The complete tree is preflighted before the first destructive operation.
    Its root and every descendant are atomically detached into a fresh 0700
    vault and revalidated there before deletion.  Caller-visible replacement
    names therefore remain outside the deletion namespace.  The vault excludes
    other principals; same-UID writers must also honor the caller's publication
    lock because POSIX has no atomic unlink-if-inode-matches operation.

    The result distinguishes an intact/retained tree from a partially removed
    tree and from a fully removed tree whose durability could not be confirmed.
    This is essential because a recursive failure after one successful unlink
    cannot truthfully be reported as though the prior version were intact.
    """
    _require_leaf_name(name)
    if not isinstance(expected_namespace, DirectoryNamespaceAttestation):
        return DirectoryCleanupOutcome(
            status=DirectoryCleanupStatus.RETAINED,
            detail="the expected namespace attestation was not complete",
        )
    expected_by_parent = _namespace_records_by_parent(expected_namespace.records)
    if (
        expected_by_parent is None
        or expected_namespace.contains_symlinks
        or expected_namespace.contains_special_entries
    ):
        return DirectoryCleanupOutcome(
            status=DirectoryCleanupStatus.RETAINED,
            detail="the expected namespace is not eligible for destructive cleanup",
        )
    directory_flags = _directory_descriptor_flags()
    vault_name = _private_cleanup_entry_name()
    vault_root_name = "root"
    expected_entry_count = expected_namespace.entry_count + 1
    progress = _DirectoryCleanupProgress()
    vault_descriptor: int | None = None
    vault_identity: tuple[int, int] | None = None
    vault_created = False
    root_descriptor: int | None = None
    root_quarantined = False

    def finish_incomplete(
        detail: str,
        *,
        durability_unconfirmed: bool = False,
    ) -> DirectoryCleanupOutcome:
        """Reconcile surviving entries and report their actual current paths."""
        nonlocal root_quarantined, vault_created

        if progress.detail is not None:
            detail = f"{progress.detail}; {detail}"

        if root_descriptor is not None:
            try:
                _sync_directory_descriptor(root_descriptor)
            except (OSError, ValueError) as exc:
                detail = f"{detail}; surviving root sync failed: {exc}"
                durability_unconfirmed = True

        if vault_descriptor is not None:
            vault_root_present = not _namespace_entry_is_absent(
                vault_descriptor,
                vault_root_name,
            )
            if (
                root_quarantined
                and not vault_root_present
                and progress.removed_entry_count == expected_entry_count - 1
            ):
                # A failed rmdir may still have applied the mutation.  The
                # absence is observable, but its durability is not.
                progress.removed_entry_count += 1
                root_quarantined = False
                durability_unconfirmed = True
            elif vault_root_present and progress.removed_entry_count < expected_entry_count:
                if _restore_private_cleanup_entry_between(
                    vault_descriptor,
                    vault_root_name,
                    parent_descriptor,
                    name,
                ):
                    root_quarantined = False

            try:
                _sync_directory_descriptor(vault_descriptor)
            except (OSError, ValueError) as exc:
                detail = f"{detail}; private recovery vault sync failed: {exc}"
                durability_unconfirmed = True

            if vault_created and not _directory_has_entries(vault_descriptor):
                if (
                    vault_identity is not None
                    and _remove_empty_directory_entry_if_identity(
                        parent_descriptor,
                        vault_name,
                        vault_identity,
                    )
                ):
                    vault_created = False

        try:
            _sync_directory_descriptor(parent_descriptor)
        except (OSError, ValueError) as exc:
            detail = f"{detail}; parent directory sync failed: {exc}"
            durability_unconfirmed = True

        recovery_entries, concurrent_entries, cleanup_artifacts, observation_errors = (
            _directory_cleanup_recovery_entries(
                parent_descriptor,
                name,
                expected_identity,
                vault_name=vault_name if vault_created else None,
                vault_descriptor=vault_descriptor if vault_created else None,
                vault_identity=vault_identity if vault_created else None,
            )
        )
        if observation_errors:
            detail = f"{detail}; " + "; ".join(observation_errors)
            durability_unconfirmed = True
        if (
            progress.removed_entry_count >= expected_entry_count
            and not recovery_entries
            and not cleanup_artifacts
        ):
            return DirectoryCleanupOutcome(
                status=DirectoryCleanupStatus.COMPLETE_DURABILITY_UNCONFIRMED,
                recovery_entries=recovery_entries,
                concurrent_entries=concurrent_entries,
                cleanup_artifacts=cleanup_artifacts,
                removed_entry_count=progress.removed_entry_count,
                durability_confirmed=False,
                detail=detail,
            )
        status = (
            DirectoryCleanupStatus.PARTIAL
            if progress.removed_entry_count
            else DirectoryCleanupStatus.RETAINED
        )
        if progress.removed_entry_count:
            # fsync(root) is not recursive.  A partial recursive cleanup can
            # leave a mutated nested directory whose durability was not proven.
            durability_unconfirmed = True
        if durability_unconfirmed:
            detail = f"{detail}; durability of surviving recovery entries is unconfirmed"
        return DirectoryCleanupOutcome(
            status=status,
            recovery_entries=recovery_entries,
            concurrent_entries=concurrent_entries,
            cleanup_artifacts=cleanup_artifacts,
            removed_entry_count=progress.removed_entry_count,
            durability_confirmed=not durability_unconfirmed,
            detail=detail,
        )

    try:
        root_stat = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(root_stat.st_mode)
            or (root_stat.st_dev, root_stat.st_ino) != expected_identity
        ):
            return DirectoryCleanupOutcome(
                status=DirectoryCleanupStatus.RETAINED,
                concurrent_entries=(name,),
                detail="the public entry no longer has the approved directory identity",
            )
        os.mkdir(vault_name, mode=0o700, dir_fd=parent_descriptor)
        vault_created = True
        vault_namespace = os.stat(
            vault_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(vault_namespace.st_mode):
            return finish_incomplete("the private cleanup vault was not a directory")
        vault_identity = (vault_namespace.st_dev, vault_namespace.st_ino)
        vault_descriptor = os.open(
            vault_name,
            directory_flags,
            dir_fd=parent_descriptor,
        )
        os.fchmod(vault_descriptor, 0o700)
        vault_opened = os.fstat(vault_descriptor)
        if (
            not stat.S_ISDIR(vault_opened.st_mode)
            or (vault_opened.st_dev, vault_opened.st_ino) != vault_identity
            or vault_opened.st_uid != os.geteuid()
            or stat.S_IMODE(vault_opened.st_mode) != 0o700
            or _directory_has_entries(vault_descriptor)
        ):
            return finish_incomplete("the private cleanup vault was not exclusively owned")
        _native_rename_entry_between(
            parent_descriptor,
            name,
            vault_descriptor,
            vault_root_name,
            exchange=False,
        )
        root_quarantined = True
        quarantined_root = os.stat(
            vault_root_name,
            dir_fd=vault_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(quarantined_root.st_mode)
            or (quarantined_root.st_dev, quarantined_root.st_ino)
            != expected_identity
        ):
            _restore_private_cleanup_entry_between(
                vault_descriptor,
                vault_root_name,
                parent_descriptor,
                name,
            )
            root_quarantined = False
            return finish_incomplete(
                "the detached root no longer had the approved identity"
            )
        if not _namespace_entry_is_absent(parent_descriptor, name):
            return finish_incomplete(
                "the public cleanup name was concurrently reinserted"
            )
        root_descriptor = os.open(
            vault_root_name,
            directory_flags,
            dir_fd=vault_descriptor,
        )
        opened = os.fstat(root_descriptor)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != expected_identity
            or _attest_open_directory_namespace(
                root_descriptor,
                display_path=Path(name),
            )
            != expected_namespace
            or not _remove_attested_directory_contents_fd(
                root_descriptor,
                directory_flags,
                expected_by_parent,
                (),
                vault_descriptor,
                progress,
            )
        ):
            return finish_incomplete(
                "the detached tree could not be removed completely"
            )
        current = os.stat(
            vault_root_name,
            dir_fd=vault_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(current.st_mode)
            or (current.st_dev, current.st_ino) != expected_identity
            or _directory_has_entries(root_descriptor)
            or not _namespace_entry_is_absent(parent_descriptor, name)
        ):
            return finish_incomplete(
                "the detached root changed before its final removal"
            )
        os.rmdir(vault_root_name, dir_fd=vault_descriptor)
        progress.removed_entry_count += 1
        root_quarantined = False
        try:
            _sync_directory_descriptor(vault_descriptor)
        except (OSError, ValueError) as exc:
            return finish_incomplete(
                f"the approved tree was removed, but its vault sync failed: {exc}",
                durability_unconfirmed=True,
            )
        if _directory_has_entries(vault_descriptor):
            return finish_incomplete(
                "the approved tree was removed, but private recovery entries remain",
                durability_unconfirmed=True,
            )
        if not _remove_empty_directory_entry_if_identity(
            parent_descriptor,
            vault_name,
            vault_identity,
        ):
            return finish_incomplete(
                "the approved tree was removed, but its empty cleanup vault remains",
                durability_unconfirmed=True,
            )
        vault_created = False
        try:
            _sync_directory_descriptor(parent_descriptor)
        except (OSError, ValueError) as exc:
            return finish_incomplete(
                "the approved tree was removed, but parent directory durability "
                f"could not be confirmed: {exc}",
                durability_unconfirmed=True,
            )
        if not _namespace_entry_is_absent(parent_descriptor, name):
            return finish_incomplete(
                "the approved tree was removed, but absence of the public name "
                "could not be confirmed",
                durability_unconfirmed=True,
            )
        return DirectoryCleanupOutcome(
            status=DirectoryCleanupStatus.COMPLETE,
            removed_entry_count=progress.removed_entry_count,
        )
    except (NotImplementedError, OSError, TypeError, ValueError) as exc:
        return finish_incomplete(f"cleanup failed: {exc}")
    finally:
        if root_descriptor is not None:
            os.close(root_descriptor)
        if vault_descriptor is not None:
            os.close(vault_descriptor)


def _fingerprint_open_directory(
    root_descriptor: int,
    *,
    display_path: Path,
    declared_files: frozenset[str] | None,
    reject_undeclared: bool,
    selected_files: frozenset[str] | None = None,
    expose_selected_files: bool = False,
) -> dict[str, Any]:
    root_before = os.fstat(root_descriptor)
    layout_before = _capture_directory_layout_fd(root_descriptor)
    _validate_declared_directory_layout(
        display_path,
        layout_before,
        declared_files,
        reject_undeclared=reject_undeclared,
    )
    files_to_hash = (
        sorted(declared_files)
        if declared_files is not None
        else sorted(
            relative
            for relative, signature in layout_before.items()
            if stat.S_ISREG(signature[0])
        )
    )
    _validate_selected_fingerprint_files(
        display_path,
        layout_before,
        files_to_hash,
        selected_files,
    )

    digest = hashlib.sha256()
    size_bytes = 0
    selected_file_sha256: dict[str, str] = {}
    for relative in files_to_hash:
        expected_signature = layout_before[relative]
        content_sha256 = _sha256_tree_file_fd(
            root_descriptor,
            relative,
            layout_before,
        )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(expected_signature[3]).encode("ascii"))
        digest.update(b"\0")
        digest.update(content_sha256.encode("ascii"))
        digest.update(b"\0")
        size_bytes += expected_signature[3]
        if selected_files is not None and relative in selected_files:
            selected_file_sha256[relative] = content_sha256

    layout_after = _capture_directory_layout_fd(root_descriptor)
    if layout_after != layout_before:
        raise ValueError(
            f"directory tree layout changed while being fingerprinted: {display_path}"
        )
    root_after = os.fstat(root_descriptor)
    if _stat_signature(root_after) != _stat_signature(root_before):
        raise ValueError(
            f"directory tree root changed while being fingerprinted: {display_path}"
        )
    result: dict[str, Any] = {
        "tree_hash_algorithm": DIRECTORY_CONTENT_HASH_ALGORITHM,
        "sha256": digest.hexdigest(),
        "file_count": len(files_to_hash),
        "size_bytes": size_bytes,
        "contains_symlinks": False,
    }
    if expose_selected_files:
        result["selected_file_sha256"] = selected_file_sha256
    return result


def _fingerprint_open_directory_namespace(
    root_descriptor: int,
    *,
    display_path: Path,
) -> dict[str, int | str | bool]:
    return _attest_open_directory_namespace(
        root_descriptor,
        display_path=display_path,
    ).summary()


def _attest_open_directory_namespace(
    root_descriptor: int,
    *,
    display_path: Path,
) -> DirectoryNamespaceAttestation:
    root_before = os.fstat(root_descriptor)
    if not stat.S_ISDIR(root_before.st_mode):
        raise ValueError(f"directory namespace root is not a directory: {display_path}")
    records_before = _capture_directory_namespace_fd(root_descriptor)
    records_after = _capture_directory_namespace_fd(root_descriptor)
    root_after = os.fstat(root_descriptor)
    if records_after != records_before or _stat_signature(root_after) != _stat_signature(
        root_before
    ):
        raise ValueError(
            f"directory namespace changed while being fingerprinted: {display_path}"
        )

    digest = hashlib.sha256()
    file_count = 0
    size_bytes = 0
    contains_symlinks = False
    contains_special_entries = False
    for relative, kind, signature, detail in records_before:
        _update_length_delimited_digest(digest, relative)
        _update_length_delimited_digest(digest, kind)
        _update_length_delimited_digest(
            digest,
            ",".join(str(value) for value in signature).encode("ascii"),
        )
        _update_length_delimited_digest(digest, detail)
        if kind == b"file":
            file_count += 1
            size_bytes += signature[3]
        elif kind == b"symlink":
            contains_symlinks = True
        elif kind == b"special":
            contains_special_entries = True

    return DirectoryNamespaceAttestation(
        namespace_hash_algorithm=DIRECTORY_NAMESPACE_HASH_ALGORITHM,
        sha256=digest.hexdigest(),
        entry_count=len(records_before),
        file_count=file_count,
        size_bytes=size_bytes,
        contains_symlinks=contains_symlinks,
        contains_special_entries=contains_special_entries,
        records=records_before,
    )


def _capture_directory_namespace_fd(
    directory_descriptor: int,
    *,
    prefix: tuple[bytes, ...] = (),
) -> tuple[tuple[bytes, bytes, tuple[int, ...], bytes], ...]:
    """Capture a descriptor-bound local namespace without following links."""
    directory_before = os.fstat(directory_descriptor)
    if not stat.S_ISDIR(directory_before.st_mode):
        raise ValueError("directory namespace descriptor is not a directory")
    names_before = _namespace_names_fd(directory_descriptor)
    records: list[tuple[bytes, bytes, tuple[int, ...], bytes]] = []
    for name, name_bytes in names_before:
        entry_stat = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        signature = _namespace_stat_signature(entry_stat)
        relative_components = (*prefix, name_bytes)
        relative = b"/".join(relative_components)

        if stat.S_ISREG(entry_stat.st_mode):
            descriptor = os.open(
                name,
                _file_descriptor_flags(),
                dir_fd=directory_descriptor,
            )
            try:
                opened = os.fstat(descriptor)
                if _namespace_stat_signature(opened) != signature:
                    raise ValueError(
                        f"directory namespace file changed while being opened: {relative!r}"
                    )
                content_sha256 = _sha256_descriptor(descriptor).encode("ascii")
                opened_after = os.fstat(descriptor)
                namespace_after = os.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if (
                    _namespace_stat_signature(opened_after) != signature
                    or _namespace_stat_signature(namespace_after) != signature
                ):
                    raise ValueError(
                        "directory namespace file changed while being fingerprinted: "
                        f"{relative!r}"
                    )
            finally:
                os.close(descriptor)
            records.append((relative, b"file", signature, content_sha256))
            continue

        if stat.S_ISDIR(entry_stat.st_mode):
            descriptor = os.open(
                name,
                _directory_descriptor_flags(),
                dir_fd=directory_descriptor,
            )
            try:
                opened = os.fstat(descriptor)
                if _namespace_stat_signature(opened) != signature:
                    raise ValueError(
                        "directory namespace directory changed while being opened: "
                        f"{relative!r}"
                    )
                records.append((relative, b"directory", signature, b""))
                records.extend(
                    _capture_directory_namespace_fd(
                        descriptor,
                        prefix=relative_components,
                    )
                )
                opened_after = os.fstat(descriptor)
                namespace_after = os.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if (
                    _namespace_stat_signature(opened_after) != signature
                    or _namespace_stat_signature(namespace_after) != signature
                ):
                    raise ValueError(
                        "directory namespace directory changed while being inspected: "
                        f"{relative!r}"
                    )
            finally:
                os.close(descriptor)
            continue

        if stat.S_ISLNK(entry_stat.st_mode):
            target_before = os.fsencode(os.readlink(name, dir_fd=directory_descriptor))
            namespace_after = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            target_after = os.fsencode(os.readlink(name, dir_fd=directory_descriptor))
            if (
                _namespace_stat_signature(namespace_after) != signature
                or target_after != target_before
            ):
                raise ValueError(
                    f"directory namespace symlink changed while being inspected: {relative!r}"
                )
            records.append((relative, b"symlink", signature, target_before))
            continue

        namespace_after = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if _namespace_stat_signature(namespace_after) != signature:
            raise ValueError(
                f"directory namespace special entry changed while being inspected: {relative!r}"
            )
        records.append((relative, b"special", signature, b""))

    names_after = _namespace_names_fd(directory_descriptor)
    directory_after = os.fstat(directory_descriptor)
    if (
        names_after != names_before
        or _stat_signature(directory_after) != _stat_signature(directory_before)
    ):
        raise ValueError("directory namespace changed while being inspected")
    return tuple(records)


def _namespace_names_fd(
    directory_descriptor: int,
) -> tuple[tuple[str, bytes], ...]:
    with os.scandir(directory_descriptor) as iterator:
        names = tuple((entry.name, os.fsencode(entry.name)) for entry in iterator)
    return tuple(sorted(names, key=lambda value: value[1]))


def _namespace_stat_signature(value: os.stat_result) -> tuple[int, ...]:
    return (*_stat_signature(value), value.st_rdev)


def _update_length_delimited_digest(digest: Any, value: bytes) -> None:
    digest.update(str(len(value)).encode("ascii"))
    digest.update(b":")
    digest.update(value)
    digest.update(b";")


def _sync_directory_tree_fd(directory_descriptor: int) -> None:
    directory_before = os.fstat(directory_descriptor)
    if not stat.S_ISDIR(directory_before.st_mode):
        raise ValueError("directory tree descriptor is not a directory")
    with os.scandir(directory_descriptor) as iterator:
        entries = sorted(iterator, key=lambda entry: entry.name)
    for entry in entries:
        entry_stat = entry.stat(follow_symlinks=False)
        if stat.S_ISREG(entry_stat.st_mode):
            descriptor = os.open(
                entry.name,
                _file_descriptor_flags(),
                dir_fd=directory_descriptor,
            )
            try:
                opened = os.fstat(descriptor)
                if _stat_signature(opened) != _stat_signature(entry_stat):
                    raise ValueError(
                        f"directory tree file changed while syncing: {entry.name}"
                    )
                os.fsync(descriptor)
                namespace_after = os.stat(
                    entry.name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if _stat_signature(namespace_after) != _stat_signature(opened):
                    raise ValueError(
                        f"directory tree file changed while syncing: {entry.name}"
                    )
            finally:
                os.close(descriptor)
        elif stat.S_ISDIR(entry_stat.st_mode):
            descriptor = os.open(
                entry.name,
                _directory_descriptor_flags(),
                dir_fd=directory_descriptor,
            )
            try:
                opened = os.fstat(descriptor)
                if _stat_signature(opened) != _stat_signature(entry_stat):
                    raise ValueError(
                        f"directory tree directory changed while syncing: {entry.name}"
                    )
                _sync_directory_tree_fd(descriptor)
                namespace_after = os.stat(
                    entry.name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if _stat_signature(namespace_after) != _stat_signature(opened):
                    raise ValueError(
                        f"directory tree directory changed while syncing: {entry.name}"
                    )
            finally:
                os.close(descriptor)
        else:
            raise ValueError(
                f"directory tree contains a non-regular entry while syncing: {entry.name}"
            )
    directory_after = os.fstat(directory_descriptor)
    if _stat_signature(directory_after) != _stat_signature(directory_before):
        raise ValueError("directory tree changed while syncing")
    _sync_directory_descriptor(directory_descriptor)


def _native_rename_entry(
    parent_descriptor: int,
    source_name: str,
    target_name: str,
    *,
    exchange: bool,
) -> None:
    _native_rename_entry_between(
        parent_descriptor,
        source_name,
        parent_descriptor,
        target_name,
        exchange=exchange,
    )


def _native_rename_entry_between(
    source_parent_descriptor: int,
    source_name: str,
    target_parent_descriptor: int,
    target_name: str,
    *,
    exchange: bool,
) -> None:
    _require_leaf_name(source_name)
    _require_leaf_name(target_name)
    if (
        source_parent_descriptor == target_parent_descriptor
        and source_name == target_name
    ):
        raise ValueError("atomic rename entries must be distinct")
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source_name)
    target_bytes = os.fsencode(target_name)
    if sys.platform.startswith("linux"):
        function = getattr(libc, "renameat2", None)
        if function is None:
            raise ValueError(
                "atomic directory publication requires renameat2 support"
            )
        function.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        function.restype = ctypes.c_int
        flags = 0x2 if exchange else 0x1  # RENAME_EXCHANGE / RENAME_NOREPLACE
    elif sys.platform == "darwin":
        function = getattr(libc, "renameatx_np", None)
        if function is None:
            raise ValueError(
                "atomic directory publication requires renameatx_np support"
            )
        function.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        function.restype = ctypes.c_int
        flags = 0x2 if exchange else 0x4  # RENAME_SWAP / RENAME_EXCL
    else:  # pragma: no cover - release CI runs on Linux; macOS is tested locally.
        raise ValueError(
            "atomic directory publication is unavailable on this platform"
        )

    ctypes.set_errno(0)
    result = function(
        source_parent_descriptor,
        source_bytes,
        target_parent_descriptor,
        target_bytes,
        flags,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if not exchange and error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise ValueError("atomic publication target changed concurrently")
    if error_number in {errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EINVAL}:
        raise ValueError(
            "atomic directory publication is unsupported by this filesystem"
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        f"{source_name} -> {target_name}",
    )


def _sync_directory_descriptor(descriptor: int) -> None:
    try:
        os.fsync(descriptor)
    except OSError as exc:
        if exc.errno in {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}:
            raise ValueError(
                "directory fsync is unsupported; refusing crash-unsafe publication"
            ) from exc
        raise


def _require_leaf_name(name: str) -> None:
    if (
        not isinstance(name, str)
        or name in {"", ".", ".."}
        or "\x00" in name
        or "/" in name
        or "\\" in name
    ):
        raise ValueError(f"atomic namespace entry name is invalid: {name!r}")


def _require_descriptor_tree_support() -> None:
    required_flags = ("O_DIRECTORY", "O_NOFOLLOW")
    missing_flags = [name for name in required_flags if not getattr(os, name, 0)]
    supports_dir_fd = getattr(os, "supports_dir_fd", set())
    supports_fd = getattr(os, "supports_fd", set())
    supports_follow_symlinks = getattr(os, "supports_follow_symlinks", set())
    missing_apis = []
    if os.open not in supports_dir_fd:
        missing_apis.append("open(dir_fd=...)")
    if os.stat not in supports_dir_fd or os.stat not in supports_follow_symlinks:
        missing_apis.append("stat(dir_fd=..., follow_symlinks=False)")
    if os.scandir not in supports_fd:
        missing_apis.append("scandir(fd)")
    if missing_flags or missing_apis:
        unavailable = ", ".join((*missing_flags, *missing_apis))
        raise ValueError(
            "descriptor-bound directory fingerprinting is unavailable on this "
            f"platform: {unavailable}"
        )


def _normalize_declared_files(
    relative_files: tuple[str, ...] | None,
) -> frozenset[str] | None:
    if relative_files is None:
        return None
    normalized: set[str] = set()
    for value in relative_files:
        if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
            raise ValueError(f"declared directory file path is invalid: {value!r}")
        portable = PurePosixPath(value)
        if (
            portable.is_absolute()
            or portable.as_posix() != value
            or any(part in {"", ".", ".."} for part in value.split("/"))
        ):
            raise ValueError(
                f"declared directory file path must be normalized and relative: {value!r}"
            )
        if value in normalized:
            raise ValueError(f"declared directory file path is duplicated: {value!r}")
        normalized.add(value)

    for value in normalized:
        parents = PurePosixPath(value).parents
        if any(parent.as_posix() in normalized for parent in parents if parent != PurePosixPath(".")):
            raise ValueError(
                f"declared directory file path conflicts with a parent file: {value!r}"
            )
    return frozenset(normalized)


@contextmanager
def _open_directory_path_bound(path: Path) -> Iterator[int]:
    absolute = _descriptor_absolute_path(path)
    directory_flags = _directory_descriptor_flags()
    components: list[_OpenedDirectoryComponent] = []
    try:
        anchor = Path(absolute.anchor)
        if anchor != Path("/"):
            raise ValueError(
                f"descriptor-bound directory fingerprinting cannot anchor path: {path}"
            )
        descriptor = os.open(anchor, directory_flags)
        try:
            anchor_stat = os.fstat(descriptor)
            if not stat.S_ISDIR(anchor_stat.st_mode):
                raise ValueError(
                    f"directory tree anchor is not a directory: {anchor}"
                )
        except BaseException:
            os.close(descriptor)
            raise
        components.append(
            _OpenedDirectoryComponent(
                parent_descriptor=None,
                descriptor=descriptor,
                name=None,
                signature=_stat_identity_signature(anchor_stat),
            )
        )
        for name in absolute.parts[1:]:
            parent_descriptor = descriptor
            entry_stat = os.stat(
                name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if stat.S_ISLNK(entry_stat.st_mode):
                raise ValueError(f"directory tree contains a symlink component: {path}")
            if not stat.S_ISDIR(entry_stat.st_mode):
                raise ValueError(f"directory tree root is not a directory: {path}")
            descriptor = os.open(
                name,
                directory_flags,
                dir_fd=parent_descriptor,
            )
            try:
                opened_stat = os.fstat(descriptor)
            except BaseException:
                os.close(descriptor)
                raise
            if not stat.S_ISDIR(opened_stat.st_mode) or _stat_identity_signature(
                opened_stat
            ) != _stat_identity_signature(entry_stat):
                os.close(descriptor)
                raise ValueError(
                    f"directory tree component changed while being opened: {path}"
                )
            components.append(
                _OpenedDirectoryComponent(
                    parent_descriptor=parent_descriptor,
                    descriptor=descriptor,
                    name=name,
                    signature=_stat_identity_signature(opened_stat),
                )
            )
        yield descriptor
        _verify_open_directory_components(path, components)
    except (NotImplementedError, OSError, TypeError) as exc:
        raise ValueError(
            f"could not open directory tree through descriptors: {path}: {exc}"
        ) from exc
    finally:
        for component in reversed(components):
            os.close(component.descriptor)


def _descriptor_absolute_path(path: Path) -> Path:
    try:
        absolute = Path(os.path.abspath(os.fspath(path)))
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"could not normalize directory tree path: {path}: {exc}") from exc
    if sys.platform == "darwin":
        for exposed, target in _MACOS_SYSTEM_ROOT_SYMLINKS.items():
            try:
                relative = absolute.relative_to(exposed)
            except ValueError:
                continue
            if not _is_allowed_system_root_symlink(exposed):
                raise ValueError(
                    f"directory tree contains an unsafe system-root symlink: {exposed}"
                )
            return target / relative
    return absolute


def _directory_descriptor_flags() -> int:
    return (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def _file_descriptor_flags() -> int:
    return (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def _verify_open_directory_components(
    path: Path,
    components: list[_OpenedDirectoryComponent],
) -> None:
    for component in components:
        current = os.fstat(component.descriptor)
        if (
            not stat.S_ISDIR(current.st_mode)
            or _stat_identity_signature(current) != component.signature
        ):
            raise ValueError(
                f"directory tree component changed while being fingerprinted: {path}"
            )
        if component.parent_descriptor is None or component.name is None:
            continue
        namespace = os.stat(
            component.name,
            dir_fd=component.parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(namespace.st_mode)
            or _stat_identity_signature(namespace) != component.signature
        ):
            raise ValueError(
                f"directory tree component was replaced while being fingerprinted: {path}"
            )


def _capture_directory_layout_fd(
    directory_descriptor: int,
    *,
    prefix: str = "",
) -> dict[str, tuple[int, ...]]:
    directory_before = os.fstat(directory_descriptor)
    if not stat.S_ISDIR(directory_before.st_mode):
        raise ValueError("directory tree descriptor is not a directory")
    names_before = _scandir_names_fd(directory_descriptor)
    layout: dict[str, tuple[int, ...]] = {}
    for name in names_before:
        if not name or name in {".", ".."} or "\\" in name or "\x00" in name:
            raise ValueError(f"directory tree contains a non-portable entry name: {name!r}")
        entry_stat = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        relative = f"{prefix}/{name}" if prefix else name
        try:
            relative.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError(
                f"directory tree entry path is not portable UTF-8: {relative!r}"
            ) from exc
        signature = _stat_signature(entry_stat)
        if stat.S_ISLNK(entry_stat.st_mode):
            raise ValueError(f"directory tree contains a symlink: {relative}")
        if stat.S_ISREG(entry_stat.st_mode):
            layout[relative] = signature
            continue
        if not stat.S_ISDIR(entry_stat.st_mode):
            raise ValueError(f"directory tree contains a non-regular path: {relative}")

        child_descriptor = os.open(
            name,
            _directory_descriptor_flags(),
            dir_fd=directory_descriptor,
        )
        try:
            child_opened = os.fstat(child_descriptor)
            if (
                not stat.S_ISDIR(child_opened.st_mode)
                or _stat_signature(child_opened) != signature
            ):
                raise ValueError(
                    f"directory tree entry changed while being opened: {relative}"
                )
            layout[relative] = signature
            descendants = _capture_directory_layout_fd(
                child_descriptor,
                prefix=relative,
            )
            overlap = set(layout).intersection(descendants)
            if overlap:
                raise ValueError(
                    f"directory tree contains duplicate entries: {sorted(overlap)!r}"
                )
            layout.update(descendants)
            child_after = os.fstat(child_descriptor)
            namespace_after = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(child_after.st_mode)
                or _stat_signature(child_after) != signature
                or _stat_signature(namespace_after) != signature
            ):
                raise ValueError(
                    f"directory tree directory changed while being inspected: {relative}"
                )
        finally:
            os.close(child_descriptor)

    names_after = _scandir_names_fd(directory_descriptor)
    directory_after = os.fstat(directory_descriptor)
    if (
        names_after != names_before
        or _stat_signature(directory_after) != _stat_signature(directory_before)
    ):
        raise ValueError("directory tree layout changed while being inspected")
    return layout


def _scandir_names_fd(directory_descriptor: int) -> tuple[str, ...]:
    with os.scandir(directory_descriptor) as iterator:
        return tuple(sorted(entry.name for entry in iterator))


def _validate_declared_directory_layout(
    path: Path,
    layout: dict[str, tuple[int, ...]],
    declared_files: frozenset[str] | None,
    *,
    reject_undeclared: bool,
) -> None:
    if declared_files is None:
        return
    declared_directories = {
        parent.as_posix()
        for value in declared_files
        for parent in PurePosixPath(value).parents
        if parent != PurePosixPath(".")
    }
    missing = sorted((declared_files | declared_directories) - set(layout))
    if missing:
        raise ValueError(f"directory tree is missing declared entries: {missing!r}")
    non_files = sorted(
        value for value in declared_files if not stat.S_ISREG(layout[value][0])
    )
    if non_files:
        raise ValueError(f"directory tree declared files are not regular: {non_files!r}")
    non_directories = sorted(
        value
        for value in declared_directories
        if not stat.S_ISDIR(layout[value][0])
    )
    if non_directories:
        raise ValueError(
            f"directory tree declared directories are not directories: {non_directories!r}"
        )
    if reject_undeclared:
        expected = declared_files | declared_directories
        undeclared = sorted(set(layout) - expected)
        if undeclared:
            raise ValueError(
                f"directory tree contains undeclared entries at {path}: {undeclared!r}"
            )


def _validate_selected_fingerprint_files(
    path: Path,
    layout: dict[str, tuple[int, ...]],
    files_to_hash: list[str],
    selected_files: frozenset[str] | None,
) -> None:
    if selected_files is None:
        return
    missing = sorted(selected_files - set(layout))
    if missing:
        raise ValueError(
            f"directory tree is missing selected fingerprint files: {missing!r}"
        )
    non_regular = sorted(
        relative
        for relative in selected_files
        if not stat.S_ISREG(layout[relative][0])
    )
    if non_regular:
        raise ValueError(
            f"directory tree selected fingerprint paths are not regular files: {non_regular!r}"
        )
    excluded = sorted(selected_files - set(files_to_hash))
    if excluded:
        raise ValueError(
            "directory tree selected fingerprint files are outside the fingerprinted "
            f"file set at {path}: {excluded!r}"
        )


def _sha256_tree_file_fd(
    root_descriptor: int,
    relative: str,
    layout: dict[str, tuple[int, ...]],
) -> str:
    parts = relative.split("/")
    directory_descriptor = root_descriptor
    opened_directories: list[_OpenedDirectoryComponent] = []
    try:
        prefix_parts: list[str] = []
        for name in parts[:-1]:
            prefix_parts.append(name)
            directory_relative = "/".join(prefix_parts)
            expected = layout[directory_relative]
            entry_stat = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            child_descriptor = os.open(
                name,
                _directory_descriptor_flags(),
                dir_fd=directory_descriptor,
            )
            try:
                opened_stat = os.fstat(child_descriptor)
            except BaseException:
                os.close(child_descriptor)
                raise
            if (
                not stat.S_ISDIR(entry_stat.st_mode)
                or _stat_signature(entry_stat) != expected
                or _stat_signature(opened_stat) != expected
            ):
                os.close(child_descriptor)
                raise ValueError(
                    f"directory tree entry changed while being admitted: {directory_relative}"
                )
            opened_directories.append(
                _OpenedDirectoryComponent(
                    parent_descriptor=directory_descriptor,
                    descriptor=child_descriptor,
                    name=name,
                    signature=expected,
                )
            )
            directory_descriptor = child_descriptor

        file_name = parts[-1]
        expected_file = layout[relative]
        namespace_before = os.stat(
            file_name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        file_descriptor = _open_tree_file_fd(directory_descriptor, file_name)
        try:
            opened_file = os.fstat(file_descriptor)
            if (
                not stat.S_ISREG(namespace_before.st_mode)
                or _stat_signature(namespace_before) != expected_file
                or _stat_signature(opened_file) != expected_file
            ):
                raise ValueError(
                    f"directory tree file changed while being admitted: {relative}"
                )
            content_sha256 = _sha256_descriptor(file_descriptor)
            file_after = os.fstat(file_descriptor)
            namespace_after = os.stat(
                file_name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(file_after.st_mode)
                or _stat_signature(file_after) != expected_file
                or _stat_signature(namespace_after) != expected_file
            ):
                raise ValueError(
                    f"directory tree file changed while being fingerprinted: {relative}"
                )
        finally:
            os.close(file_descriptor)

        for component in opened_directories:
            current = os.fstat(component.descriptor)
            namespace = os.stat(
                component.name,
                dir_fd=component.parent_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(current.st_mode)
                or _stat_signature(current) != component.signature
                or _stat_signature(namespace) != component.signature
            ):
                raise ValueError(
                    f"directory tree parent changed while fingerprinting file: {relative}"
                )
        return content_sha256
    finally:
        for component in reversed(opened_directories):
            os.close(component.descriptor)


def _sha256_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    while chunk := os.read(descriptor, 1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def _open_tree_file_fd(directory_descriptor: int, name: str) -> int:
    return os.open(
        name,
        _file_descriptor_flags(),
        dir_fd=directory_descriptor,
    )


def _stat_signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_mode,
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _stat_identity_signature(value: os.stat_result) -> tuple[int, ...]:
    """Return metadata that is stable while an opened path stays the same object.

    Ancestor directory size and timestamps legitimately change when unrelated
    siblings are created (for example, temporary spool directories).  Binding
    ancestor traversal to type, device, and inode detects namespace rebinding
    without treating those unrelated metadata updates as source-tree races.
    """
    return (
        stat.S_IFMT(value.st_mode),
        value.st_dev,
        value.st_ino,
    )


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


def _namespace_records_by_parent(
    records: tuple[tuple[bytes, bytes, tuple[int, ...], bytes], ...],
) -> dict[tuple[bytes, ...], dict[bytes, tuple[bytes, tuple[int, ...], bytes]]] | None:
    """Index one previously attested namespace without trusting live names."""
    indexed: dict[
        tuple[bytes, ...],
        dict[bytes, tuple[bytes, tuple[int, ...], bytes]],
    ] = {(): {}}
    entry_kinds: dict[tuple[bytes, ...], bytes] = {}
    for relative, kind, signature, detail in records:
        parts = tuple(relative.split(b"/"))
        if (
            not parts
            or any(not part or part in {b".", b".."} for part in parts)
            or len(signature) != 7
            or kind not in {b"file", b"directory", b"symlink", b"special"}
        ):
            return None
        parent = parts[:-1]
        name = parts[-1]
        children = indexed.setdefault(parent, {})
        if name in children:
            return None
        children[name] = (kind, signature, detail)
        entry_kinds[parts] = kind
        if kind == b"directory":
            indexed.setdefault(parts, {})

    for path in indexed:
        if path and entry_kinds.get(path) != b"directory":
            return None
    return indexed


def _private_cleanup_entry_name() -> str:
    """Return an unguessable sibling name used to detach an approved entry.

    Deletion never targets the caller-visible name.  The approved entry is
    first moved with a no-replace rename and then revalidated at this private
    name, so a replacement of the public name is retained instead of unlinked.
    """
    return f".hfr-remove-{secrets.token_hex(16)}"


def _directory_cleanup_recovery_entries(
    parent_descriptor: int,
    public_name: str,
    expected_identity: tuple[int, int],
    *,
    vault_name: str | None,
    vault_descriptor: int | None,
    vault_identity: tuple[int, int] | None,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return exact, descriptor-observed paths left after cleanup reconciliation."""
    recovery_entries: list[str] = []
    concurrent_entries: list[str] = []
    observation_errors: list[str] = []
    try:
        public_entry = os.stat(
            public_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        pass
    except (NotImplementedError, OSError, TypeError, ValueError) as exc:
        recovery_entries.append(public_name)
        observation_errors.append(
            f"could not prove public recovery entry absence at {public_name}: {exc}"
        )
    else:
        if (
            stat.S_ISDIR(public_entry.st_mode)
            and (public_entry.st_dev, public_entry.st_ino) == expected_identity
        ):
            recovery_entries.append(public_name)
        else:
            concurrent_entries.append(public_name)

    cleanup_artifacts: tuple[str, ...] = ()
    if vault_name is not None and vault_identity is not None:
        try:
            vault_entry = os.stat(
                vault_name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        except (NotImplementedError, OSError, TypeError, ValueError) as exc:
            cleanup_artifacts = (vault_name,)
            recovery_entries.append(vault_name)
            observation_errors.append(
                f"could not prove private recovery vault absence at {vault_name}: {exc}"
            )
        else:
            if (
                stat.S_ISDIR(vault_entry.st_mode)
                and (vault_entry.st_dev, vault_entry.st_ino) == vault_identity
            ):
                cleanup_artifacts = (vault_name,)
                try:
                    vault_names = (
                        _namespace_names_fd(vault_descriptor)
                        if vault_descriptor is not None
                        else ()
                    )
                except (NotImplementedError, OSError, TypeError, ValueError) as exc:
                    vault_names = ()
                    observation_errors.append(
                        f"could not enumerate private recovery vault {vault_name}: {exc}"
                    )
                if vault_names:
                    recovery_entries.extend(
                        f"{vault_name}/{entry_name}"
                        for entry_name, _entry_bytes in vault_names
                    )
                elif vault_descriptor is None or _directory_has_entries(
                    vault_descriptor
                ):
                    # The vault itself is still an exact discoverable inspection
                    # point even when its children could not be enumerated.
                    recovery_entries.append(vault_name)
            else:
                concurrent_entries.append(vault_name)

    return (
        tuple(sorted(set(recovery_entries))),
        tuple(sorted(set(concurrent_entries))),
        cleanup_artifacts,
        tuple(observation_errors),
    )


def _namespace_entry_is_absent(parent_descriptor: int, name: str) -> bool:
    try:
        os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return True
    except (NotImplementedError, OSError, TypeError, ValueError):
        return False
    return False


def _restore_private_cleanup_entry_between(
    quarantine_parent_descriptor: int,
    quarantine_name: str,
    original_parent_descriptor: int,
    original_name: str,
) -> bool:
    """Restore a detached entry only while its public name remains absent.

    A concurrently inserted public entry wins.  In that case the detached
    entry is retained at its private recovery name so neither object is lost.
    """
    if not _namespace_entry_is_absent(
        original_parent_descriptor,
        original_name,
    ):
        return False
    try:
        _native_rename_entry_between(
            quarantine_parent_descriptor,
            quarantine_name,
            original_parent_descriptor,
            original_name,
            exchange=False,
        )
    except (NotImplementedError, OSError, TypeError, ValueError):
        return False
    return True


def _remove_empty_directory_entry_if_identity(
    parent_descriptor: int,
    name: str,
    expected_identity: tuple[int, int],
) -> bool:
    """Best-effort cleanup for an empty private vault with a pinned identity."""
    descriptor: int | None = None
    try:
        namespace = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(namespace.st_mode)
            or (namespace.st_dev, namespace.st_ino) != expected_identity
        ):
            return False
        descriptor = os.open(
            name,
            _directory_descriptor_flags(),
            dir_fd=parent_descriptor,
        )
        opened = os.fstat(descriptor)
        current = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != expected_identity
            or (current.st_dev, current.st_ino) != expected_identity
            or _directory_has_entries(descriptor)
        ):
            return False
        os.rmdir(name, dir_fd=parent_descriptor)
        return True
    except (FileNotFoundError, NotImplementedError, OSError, TypeError, ValueError):
        return False
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _namespace_signature_matches_after_private_rename(
    actual: tuple[int, ...],
    expected: tuple[int, ...],
) -> bool:
    """Compare an entry after rename while allowing rename-induced ctime."""
    return (
        len(actual) == 7
        and len(expected) == 7
        and actual[:5] == expected[:5]
        and actual[6:] == expected[6:]
    )


def _remove_attested_directory_contents_fd(
    directory_descriptor: int,
    directory_flags: int,
    expected_by_parent: dict[
        tuple[bytes, ...],
        dict[bytes, tuple[bytes, tuple[int, ...], bytes]],
    ]
    | None,
    prefix: tuple[bytes, ...],
    vault_descriptor: int,
    progress: _DirectoryCleanupProgress,
) -> bool:
    """Delete only descendants that still match a complete prior attestation."""
    if expected_by_parent is None:
        return progress.failed("the approved namespace index was unavailable")
    expected_children = expected_by_parent.get(prefix)
    if expected_children is None:
        return progress.failed("the approved directory was absent from the namespace index")
    try:
        current_names = _namespace_names_fd(directory_descriptor)
    except (NotImplementedError, OSError, TypeError) as exc:
        return progress.failed(f"the directory namespace could not be listed: {exc}")
    if tuple(name_bytes for _name, name_bytes in current_names) != tuple(
        sorted(expected_children)
    ):
        return progress.failed("the directory namespace changed before cleanup")

    for name, name_bytes in current_names:
        expected = expected_children.get(name_bytes)
        if expected is None:
            return progress.failed("an unapproved directory entry appeared during cleanup")
        expected_kind, expected_signature, expected_detail = expected
        quarantine_name = _private_cleanup_entry_name()
        quarantined = False
        child_descriptor: int | None = None
        try:
            entry_stat = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if _namespace_stat_signature(entry_stat) != expected_signature:
                return progress.failed(
                    f"the approved entry changed before detach: {name}"
                )
            _native_rename_entry_between(
                directory_descriptor,
                name,
                vault_descriptor,
                quarantine_name,
                exchange=False,
            )
            quarantined = True
            quarantined_stat = os.stat(
                quarantine_name,
                dir_fd=vault_descriptor,
                follow_symlinks=False,
            )
            if not _namespace_signature_matches_after_private_rename(
                _namespace_stat_signature(quarantined_stat),
                expected_signature,
            ):
                _restore_private_cleanup_entry_between(
                    vault_descriptor,
                    quarantine_name,
                    directory_descriptor,
                    name,
                )
                return progress.failed(
                    f"the approved entry changed during detach: {name}"
                )
            if not _namespace_entry_is_absent(directory_descriptor, name):
                return progress.failed(
                    f"the public child name was concurrently reinserted: {name}"
                )

            if expected_kind == b"file":
                child_descriptor = os.open(
                    quarantine_name,
                    _file_descriptor_flags(),
                    dir_fd=vault_descriptor,
                )
                opened = os.fstat(child_descriptor)
                renamed_signature = _namespace_stat_signature(quarantined_stat)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or _namespace_stat_signature(opened) != renamed_signature
                    or _sha256_descriptor(child_descriptor).encode("ascii")
                    != expected_detail
                ):
                    _restore_private_cleanup_entry_between(
                        vault_descriptor,
                        quarantine_name,
                        directory_descriptor,
                        name,
                    )
                    return progress.failed(
                        f"the detached file changed before removal: {name}"
                    )
                opened_after = os.fstat(child_descriptor)
                namespace_after = os.stat(
                    quarantine_name,
                    dir_fd=vault_descriptor,
                    follow_symlinks=False,
                )
                if (
                    _namespace_stat_signature(opened_after) != renamed_signature
                    or _namespace_stat_signature(namespace_after)
                    != renamed_signature
                    or not _namespace_entry_is_absent(directory_descriptor, name)
                ):
                    _restore_private_cleanup_entry_between(
                        vault_descriptor,
                        quarantine_name,
                        directory_descriptor,
                        name,
                    )
                    return progress.failed(
                        f"the detached file changed while being verified: {name}"
                    )
                os.unlink(quarantine_name, dir_fd=vault_descriptor)
                quarantined = False
                progress.removed_entry_count += 1
                if not _namespace_entry_is_absent(directory_descriptor, name):
                    return progress.failed(
                        f"the public child name was reinserted after removal: {name}"
                    )
                continue

            if expected_kind != b"directory":
                # Owned reviewed exports never contain links or special entries.
                _restore_private_cleanup_entry_between(
                    vault_descriptor,
                    quarantine_name,
                    directory_descriptor,
                    name,
                )
                return progress.failed(
                    f"the approved entry had an unsupported kind: {name}"
                )
            child_descriptor = os.open(
                quarantine_name,
                directory_flags,
                dir_fd=vault_descriptor,
            )
            opened = os.fstat(child_descriptor)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or not _namespace_signature_matches_after_private_rename(
                    _namespace_stat_signature(opened),
                    expected_signature,
                )
                or not _remove_attested_directory_contents_fd(
                    child_descriptor,
                    directory_flags,
                    expected_by_parent,
                    (*prefix, name_bytes),
                    vault_descriptor,
                    progress,
                )
            ):
                _restore_private_cleanup_entry_between(
                    vault_descriptor,
                    quarantine_name,
                    directory_descriptor,
                    name,
                )
                return progress.failed(
                    f"the detached directory could not be removed: {name}"
                )
            namespace_after = os.stat(
                quarantine_name,
                dir_fd=vault_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(namespace_after.st_mode)
                or (namespace_after.st_mode, namespace_after.st_dev, namespace_after.st_ino)
                != expected_signature[:3]
                or _directory_has_entries(child_descriptor)
                or not _namespace_entry_is_absent(directory_descriptor, name)
            ):
                _restore_private_cleanup_entry_between(
                    vault_descriptor,
                    quarantine_name,
                    directory_descriptor,
                    name,
                )
                return progress.failed(
                    f"the detached directory changed before removal: {name}"
                )
            os.rmdir(quarantine_name, dir_fd=vault_descriptor)
            quarantined = False
            progress.removed_entry_count += 1
            if not _namespace_entry_is_absent(directory_descriptor, name):
                return progress.failed(
                    f"the public child name was reinserted after removal: {name}"
                )
        except (NotImplementedError, OSError, TypeError, ValueError) as exc:
            if quarantined:
                if _namespace_entry_is_absent(vault_descriptor, quarantine_name):
                    # POSIX may report an I/O error after the namespace mutation
                    # was applied.  Count the observed removal once and surface
                    # the overall cleanup as partial/durability-uncertain.
                    progress.removed_entry_count += 1
                    quarantined = False
                else:
                    _restore_private_cleanup_entry_between(
                        vault_descriptor,
                        quarantine_name,
                        directory_descriptor,
                        name,
                    )
            return progress.failed(f"cleanup of {name} failed: {exc}")
        finally:
            if child_descriptor is not None:
                os.close(child_descriptor)
    if _directory_has_entries(directory_descriptor):
        return progress.failed("the directory still contained entries after cleanup")
    return True


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
