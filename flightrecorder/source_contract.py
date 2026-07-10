"""Fail-closed inspection of JSON artifacts used as readiness evidence."""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .path_safety import path_has_symlink_component
from .schema_registry import SchemaRegistryError, check_schema_contract

_SCHEMA_NAME_OVERRIDES = {
    "harness_manifest": "harness_run_manifest",
    "harness_result": "harness_run_result",
}

_DIRECTORY_MANIFESTS = {
    "promotion_archive": "promotion_archive.json",
    "promotion_cards": "promotion_cards.json",
}

_SEMANTIC_VALIDATOR_NAMES = {
    "action_ledger": "validate_action_ledger",
    "action_ledger_gate": "validate_action_ledger_gate",
    "agentic_loop_governance_receipt": "validate_agentic_loop_governance_receipt",
    "agentic_loop_ledger": "validate_agentic_loop_ledger",
    "agentic_rollout_plan": "validate_agentic_rollout_plan",
    "agentic_rollout_receipt": "validate_agentic_rollout_receipt",
    "agentic_training_flow": "validate_agentic_training_flow",
    "agentic_training_loop_plan": "validate_agentic_training_loop_plan",
    "agentic_training_plan": "validate_agentic_training_plan",
    "agentic_training_result": "validate_agentic_training_result",
    "agentic_training_runtime_preflight": "validate_agentic_training_runtime_preflight",
    "cloud_training_artifact_manifest": "validate_cloud_training_artifact_manifest",
    "cloud_training_launch_plan": "validate_cloud_training_launch_plan",
    "cloud_training_launch_receipt": "validate_cloud_training_launch_receipt",
    "cloud_training_preflight": "validate_cloud_training_preflight",
    "cloud_training_provider_registry": "validate_cloud_training_provider_registry",
    "cloud_training_status_receipt": "validate_cloud_training_status_receipt",
    "dataset_curation_receipt": "validate_dataset_curation_receipt",
    "decision_gate": "validate_decision_gate",
    "eval_summary": "validate_eval_summary",
    "evidence_bundle": "validate_evidence_bundle",
    "evidence_coverage": "validate_evidence_coverage",
    "external_eval_plan": "validate_external_eval_plan",
    "external_eval_receipt": "validate_external_eval_receipt",
    "harness_manifest": "validate_harness_run_manifest",
    "harness_result": "validate_harness_run_result",
    "heldout_manifest": "validate_heldout_manifest",
    "improvement_ledger": "validate_improvement_ledger",
    "improvement_ledger_gate": "validate_improvement_ledger_gate",
    "improvement_plan": "validate_improvement_plan",
    "live_smoke_summary": "validate_live_smoke_summary",
    "model_adapter_manifest": "validate_model_adapter_manifest",
    "model_candidate": "validate_model_candidate",
    "model_compatibility_report": "validate_model_compatibility_report",
    "model_grader_disagreement_queue": "validate_model_grader_disagreement_queue",
    "model_grader_dry_run": "validate_model_grader_dry_run",
    "model_grader_gate": "validate_model_grader_gate",
    "model_grader_override_receipt": "validate_model_grader_override_receipt",
    "model_registry": "validate_model_registry",
    "model_registry_entry": "validate_model_registry_entry",
    "model_scout_manifest": "validate_model_scout_manifest",
    "model_serving_probe_receipt": "validate_model_serving_probe_receipt",
    "next_iteration_schedule": "validate_next_iteration_schedule",
    "promotion_alias_apply": "validate_promotion_alias_apply",
    "promotion_decision": "validate_promotion_decision",
    "promotion_ledger": "validate_promotion_ledger",
    "promotion_ledger_gate": "validate_promotion_ledger_gate",
    "promotion_release_record": "validate_promotion_release_record",
    "promotion_rollback_receipt": "validate_promotion_rollback_receipt",
    "rejection_sampling_gate": "validate_rejection_sampling_gate",
    "repair_queue": "validate_repair_queue",
    "review_calibration": "validate_review_calibration",
    "rubric_spec": "validate_rubric_spec",
    "run_digest": "validate_run_digest",
    "scenario_check": "validate_scenario_check",
    "scenario_quality": "validate_scenario_quality",
    "serving_compatibility_report": "validate_serving_compatibility_report",
    "serving_demo_run": "validate_serving_demo_run",
    "serving_endpoint_check": "validate_serving_endpoint_check",
    "serving_lifecycle": "validate_serving_lifecycle",
    "serving_profile": "validate_serving_profile",
    "state_diff": "validate_state_diff",
    "state_snapshot": "validate_state_snapshot",
    "trace_observability": "validate_trace_observability",
    "trainer_archive": "validate_trainer_archive",
    "trainer_archive_check": "validate_trainer_archive_check",
    "trainer_consumer_plan": "validate_trainer_consumer_plan",
    "trainer_launch_check": "validate_trainer_launch_check",
    "trainer_preflight": "validate_trainer_preflight",
    "trainer_wrapper_dry_run": "validate_trainer_wrapper_dry_run",
}

_GATE_CONTRACT_ROLES = {"compare_gate", "reviewed_gate"}

_REQUIRED_VALUES: dict[str, dict[str, Any]] = {
    "action_ledger": {"passed": True},
    "agentic_loop_ledger": {"passed": True},
    "agentic_rollout_plan": {"passed": True, "readiness": "ready_for_harness_batch"},
    "agentic_rollout_receipt": {"passed": True, "readiness": "mock_rollouts_recorded"},
    "agentic_training_flow": {"passed": True},
    "agentic_training_plan": {"passed": True, "readiness": "ready"},
    "agentic_training_result": {"passed": True},
    "agentic_training_runtime_preflight": {"passed": True},
    "cloud_training_artifact_manifest": {"passed": True, "readiness": "ready"},
    "cloud_training_launch_plan": {"passed": True, "readiness": "ready_for_dry_run_launch"},
    "cloud_training_launch_receipt": {"passed": True, "readiness": "dry_run_recorded"},
    "cloud_training_preflight": {"passed": True, "readiness": "ready_for_dry_run_launch_plan"},
    "cloud_training_status_receipt": {"passed": True, "readiness": "status_recorded"},
    "compare_gate": {"passed": True},
    "dataset_curation_receipt": {"passed": True, "readiness": "ready_for_external_trainer_handoff"},
    "eval_summary": {"passed": True, "governance_ready": True},
    "evidence_bundle": {"passed": True, "readiness": "ready"},
    "external_eval_plan": {"ready": True},
    "external_eval_receipt": {"passed": True, "readiness": "dry_run_recorded"},
    "heldout_manifest": {"ready": True},
    "improvement_ledger": {"passed": True},
    "improvement_plan": {"passed": True},
    "model_grader_gate": {"passed": True, "readiness": "labels_calibrated_for_curated_handoff"},
    "promotion_archive": {"passed": True},
    "promotion_cards": {"passed": True, "readiness": "ready"},
    "promotion_decision": {"passed": True, "readiness": "ready"},
    "promotion_ledger": {"passed": True},
    "rejection_sampling_gate": {"passed": True, "readiness": "ready_for_dataset_curation"},
    "review_calibration": {"passed": True},
    "reviewed_gate": {"passed": True},
    "serving_lifecycle": {"passed": True, "ready": True, "readiness": "ready"},
    "trainer_launch_check": {"passed": True, "readiness": "ready"},
    "trainer_preflight": {"passed": True, "readiness": "ready"},
}


@dataclass(frozen=True)
class _FileAttestation:
    identity: tuple[int, int]
    metadata: tuple[int, int, int, int, int]
    sha256: str
    content: bytes


@dataclass(frozen=True)
class _DirectoryTreeAttestation:
    root: tuple[int, ...]
    entries: tuple[tuple[str, str, tuple[int, ...], str], ...]


@dataclass(frozen=True)
class _CapturedDirectoryTree:
    root: tuple[int, ...]
    directories: tuple[tuple[str, tuple[int, ...]], ...]
    files: tuple[tuple[str, _FileAttestation], ...]


@dataclass(frozen=True)
class _PrivateSemanticSnapshot:
    boundary_path: Path
    source_relative_path: Path
    source_attestation: _FileAttestation
    tree: _CapturedDirectoryTree
    repository_boundary: bool


class _SnapshotBoundaryExpansion(Exception):
    def __init__(self, parent_levels: int) -> None:
        super().__init__(f"semantic snapshot requires {parent_levels} more parent level(s)")
        self.parent_levels = parent_levels


_ACTIVE_SEMANTIC_SNAPSHOT_ROOT: ContextVar[Path | None] = ContextVar(
    "hfr_active_semantic_snapshot_root",
    default=None,
)

_ACTIVE_SEMANTIC_INSPECTION_CACHE: ContextVar[
    dict[tuple[str, str, bool], dict[str, Any]] | None
] = ContextVar("hfr_active_semantic_inspection_cache", default=None)

_PRIVATE_SNAPSHOT_EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".mypy_cache",
    ".omx",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
}

_PRIVATE_SNAPSHOT_EXCLUDED_REPOSITORY_ROOTS = {
    *_PRIVATE_SNAPSHOT_EXCLUDED_DIRECTORY_NAMES,
    "runs",
}


def inspect_artifact_source(
    path_value: str | Path,
    role: str,
    *,
    require_semantics: bool = True,
) -> dict[str, Any]:
    """Return a side-effect-free readiness assessment for one source artifact.

    ``ready`` means the source is a regular non-symlink artifact, conforms to
    the role's bundled schema, satisfies its success signal, and passes the
    role's full semantic validator when one exists. Physical existence alone
    is never treated as evidence readiness.
    """
    path = Path(path_value)
    active_snapshot_root = _ACTIVE_SEMANTIC_SNAPSHOT_ROOT.get()
    if active_snapshot_root is not None and not _path_is_within(path, active_snapshot_root):
        return _empty_inspection(path)
    cache = _ACTIVE_SEMANTIC_INSPECTION_CACHE.get()
    cache_key = (os.path.abspath(path), role, require_semantics)
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    if role == "training_export":
        result = _inspect_training_export(path, require_semantics=require_semantics)
    elif role in _DIRECTORY_MANIFESTS:
        result = _inspect_directory(path, role, require_semantics=require_semantics)
    else:
        schema_name = _SCHEMA_NAME_OVERRIDES.get(role, role)
        result = inspect_json_source(
            path,
            schema_name,
            role=role,
            require_semantics=require_semantics,
        )
    if cache is not None:
        cache[cache_key] = result
    return result


def inspect_json_source(
    path_value: str | Path,
    schema_name: str,
    *,
    role: str | None = None,
    require_semantics: bool = True,
) -> dict[str, Any]:
    """Inspect one stable JSON source without following symlinked components.

    Semantic validators accept paths and may resolve sibling artifacts relative
    to the source. Run those validators against a descriptor-bound private tree
    snapshot so a parent-directory swap cannot substitute an alternate source
    or dependency closure and then restore the admitted pathname.
    """
    path = Path(path_value)
    before = _attest_regular_file(path)
    physical_exists = before is not None or path.exists()
    regular_file = before is not None
    payload: dict[str, Any] = {}
    parse_valid = False
    if before is not None:
        try:
            raw = json.loads(before.content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raw = None
        if isinstance(raw, dict):
            payload = raw
            parse_valid = True

    schema_valid = False
    if parse_valid:
        try:
            schema_valid = bool(check_schema_contract(payload, name_or_id=schema_name).get("passed"))
        except (SchemaRegistryError, TypeError, ValueError):
            schema_valid = False

    semantic_valid = not require_semantics
    private_snapshot: _PrivateSemanticSnapshot | None = None
    if require_semantics:
        contract_role = role or schema_name
        semantic_valid = schema_valid and _semantic_ready(contract_role, payload)
        if semantic_valid and _requires_path_semantic_validation(contract_role):
            active_snapshot_root = _ACTIVE_SEMANTIC_SNAPSHOT_ROOT.get()
            if active_snapshot_root is None:
                private_snapshot = _capture_private_semantic_snapshot(path, payload)
                semantic_valid = (
                    private_snapshot is not None
                    and before == private_snapshot.source_attestation
                    and _validate_private_semantic_snapshot(private_snapshot, contract_role)
                )
            else:
                semantic_valid = (
                    _path_is_within(path, active_snapshot_root)
                    and _semantic_contract_valid(path, contract_role)
                )

    after = _attest_regular_file(path) if before is not None else None
    stable = before is not None and before == after
    if private_snapshot is not None:
        stable = stable and private_snapshot.tree == _recapture_private_semantic_snapshot(
            private_snapshot
        )
    semantic_valid = stable and semantic_valid
    return {
        "path": path,
        "physical_exists": physical_exists,
        "regular_file": regular_file,
        "parse_valid": parse_valid,
        "schema_valid": schema_valid,
        "semantic_valid": semantic_valid,
        "stable": stable,
        "ready": regular_file and parse_valid and schema_valid and semantic_valid and stable,
        "payload": payload,
    }


def _requires_path_semantic_validation(role: str) -> bool:
    return (
        role == "training_export"
        or role in _DIRECTORY_MANIFESTS
        or role in _GATE_CONTRACT_ROLES
        or role in _SEMANTIC_VALIDATOR_NAMES
    )


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        absolute_path = Path(os.path.abspath(path))
        absolute_root = Path(os.path.abspath(root))
        return os.path.commonpath((absolute_path, absolute_root)) == str(absolute_root)
    except (OSError, TypeError, ValueError):
        return False


def _capture_private_semantic_snapshot(
    path: Path,
    payload: dict[str, Any],
) -> _PrivateSemanticSnapshot | None:
    absolute_path = Path(os.path.abspath(path))
    parent_path = absolute_path.parent
    parent_descriptor = _open_stable_directory(parent_path)
    if parent_descriptor is None:
        return None

    boundary_descriptor: int | None = None
    try:
        source_attestation = _attest_regular_file_at(parent_descriptor, absolute_path.name)
        if source_attestation is None:
            return None

        repository_root = _nearest_repository_root(parent_path)
        allowed_root = _semantic_snapshot_allowed_root(parent_path, repository_root)
        try:
            maximum_parent_depth = len(parent_path.relative_to(allowed_root).parts)
        except ValueError:
            return None
        repository_boundary = repository_root is not None
        parent_depth = _semantic_snapshot_parent_depth(
            absolute_path,
            payload,
            repository_root=repository_root,
            repository_boundary=repository_boundary,
        )
        if parent_depth > maximum_parent_depth:
            return None
        boundary_path = parent_path
        boundary_descriptor = os.dup(parent_descriptor)
        for _ in range(parent_depth):
            next_boundary = boundary_path.parent
            if next_boundary == boundary_path:
                return None
            next_descriptor = _open_child_directory(boundary_descriptor, "..")
            if next_descriptor is None:
                return None
            os.close(boundary_descriptor)
            boundary_descriptor = next_descriptor
            boundary_path = next_boundary

        while True:
            if boundary_path == Path(boundary_path.anchor):
                return None
            if not _directory_descriptor_matches_path(boundary_descriptor, boundary_path):
                return None
            repository_boundary = repository_root is not None and boundary_path == repository_root
            source_relative_path = absolute_path.relative_to(boundary_path)
            try:
                tree = _capture_referenced_tree_fd(
                    boundary_descriptor,
                    source_relative_path,
                    repository_boundary=repository_boundary,
                    allow_parent_expansion=parent_depth < maximum_parent_depth,
                )
            except _SnapshotBoundaryExpansion as expansion:
                if (
                    expansion.parent_levels <= 0
                    or parent_depth + expansion.parent_levels > maximum_parent_depth
                ):
                    return None
                for _ in range(expansion.parent_levels):
                    next_boundary = boundary_path.parent
                    next_descriptor = _open_child_directory(boundary_descriptor, "..")
                    if next_descriptor is None:
                        return None
                    os.close(boundary_descriptor)
                    boundary_descriptor = next_descriptor
                    boundary_path = next_boundary
                parent_depth += expansion.parent_levels
                continue
            if tree is None:
                return None
            break
        captured_source = dict(tree.files).get(source_relative_path.as_posix())
        if captured_source != source_attestation:
            return None
        return _PrivateSemanticSnapshot(
            boundary_path=boundary_path,
            source_relative_path=source_relative_path,
            source_attestation=source_attestation,
            tree=tree,
            repository_boundary=repository_boundary,
        )
    except (OSError, UnicodeError, ValueError):
        return None
    finally:
        if boundary_descriptor is not None:
            os.close(boundary_descriptor)
        os.close(parent_descriptor)


def _required_snapshot_parent_depth(value: Any) -> int:
    if isinstance(value, dict):
        return max((_required_snapshot_parent_depth(item) for item in value.values()), default=0)
    if isinstance(value, list):
        return max((_required_snapshot_parent_depth(item) for item in value), default=0)
    if not isinstance(value, str) or "\\" in value or "\x00" in value:
        return 0
    parts = PurePosixPath(value).parts
    depth = 0
    for part in parts:
        if part != "..":
            break
        depth += 1
    return depth


def _semantic_snapshot_parent_depth(
    path: Path,
    payload: dict[str, Any],
    *,
    repository_root: Path | None,
    repository_boundary: bool,
) -> int:
    parent_depth = _required_snapshot_parent_depth(payload)
    if repository_root is None or not repository_boundary:
        return parent_depth
    try:
        repo_depth = len(path.parent.relative_to(repository_root).parts)
    except ValueError:
        return parent_depth
    return repo_depth


def _nearest_repository_root(start: Path) -> Path | None:
    for candidate in (start, *start.parents):
        try:
            if (candidate / ".git").exists() or (candidate / "pyproject.toml").is_file():
                return candidate
        except OSError:
            return None
    return None


def _semantic_snapshot_allowed_root(parent: Path, repository_root: Path | None) -> Path:
    if repository_root is not None:
        return repository_root
    temporary_root = Path(os.path.abspath(tempfile.gettempdir()))
    try:
        relative = parent.relative_to(temporary_root)
    except ValueError:
        return parent
    if not relative.parts:
        return parent
    return temporary_root / relative.parts[0]


@contextmanager
def _materialize_private_semantic_snapshot(
    snapshot: _PrivateSemanticSnapshot,
) -> Iterator[tuple[Path, Path]]:
    with tempfile.TemporaryDirectory(prefix="hfr-source-snapshot-") as temp_value:
        snapshot_root = Path(temp_value) / "tree"
        snapshot_root.mkdir(mode=0o700)
        directories = [snapshot_root]
        try:
            for relative, _signature in snapshot.tree.directories:
                directory = snapshot_root / relative
                directory.mkdir(mode=0o700, parents=True, exist_ok=True)
                directories.append(directory)
            for relative, file_attestation in snapshot.tree.files:
                destination = snapshot_root / relative
                destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                with destination.open("xb") as handle:
                    handle.write(file_attestation.content)
                destination.chmod(0o400)
            for directory in sorted(set(directories), key=lambda item: len(item.parts), reverse=True):
                directory.chmod(0o500)
            yield snapshot_root / snapshot.source_relative_path, snapshot_root
        finally:
            for directory in sorted(set(directories), key=lambda item: len(item.parts)):
                try:
                    directory.chmod(0o700)
                except OSError:
                    pass


def _validate_private_semantic_snapshot(snapshot: _PrivateSemanticSnapshot, role: str) -> bool:
    try:
        with _materialize_private_semantic_snapshot(snapshot) as (source_path, snapshot_root):
            token = _ACTIVE_SEMANTIC_SNAPSHOT_ROOT.set(snapshot_root)
            cache_token = _ACTIVE_SEMANTIC_INSPECTION_CACHE.set({})
            try:
                return _semantic_contract_valid(source_path, role)
            finally:
                _ACTIVE_SEMANTIC_INSPECTION_CACHE.reset(cache_token)
                _ACTIVE_SEMANTIC_SNAPSHOT_ROOT.reset(token)
    except (OSError, UnicodeError, ValueError):
        return False


def _attest_regular_file(path: Path) -> _FileAttestation | None:
    """Read and attest one pathname-bound regular file without following its leaf."""
    if path_has_symlink_component(path, include_leaf=True):
        return None
    try:
        pathname_before = path.stat(follow_symlinks=False)
    except OSError:
        return None
    if not stat.S_ISREG(pathname_before.st_mode):
        return None

    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return None

    chunks: list[bytes] = []
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            return None
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError:
        return None
    finally:
        os.close(descriptor)

    if _file_stat_signature(pathname_before) != _file_stat_signature(before):
        return None
    if _file_stat_signature(before) != _file_stat_signature(after):
        return None
    if path_has_symlink_component(path, include_leaf=True):
        return None
    try:
        pathname_stat = path.stat(follow_symlinks=False)
    except OSError:
        return None
    if _file_stat_signature(after) != _file_stat_signature(pathname_stat):
        return None

    content = b"".join(chunks)
    return _FileAttestation(
        identity=(after.st_dev, after.st_ino),
        metadata=(
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ),
        sha256=hashlib.sha256(content).hexdigest(),
        content=content,
    )


def _attest_regular_file_at(directory_descriptor: int, name: str) -> _FileAttestation | None:
    try:
        pathname_before = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    except OSError:
        return None
    if not stat.S_ISREG(pathname_before.st_mode):
        return None

    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    except OSError:
        return None

    chunks: list[bytes] = []
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            return None
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError:
        return None
    finally:
        os.close(descriptor)

    try:
        pathname_after = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    except OSError:
        return None
    if not (
        _file_stat_signature(pathname_before)
        == _file_stat_signature(before)
        == _file_stat_signature(after)
        == _file_stat_signature(pathname_after)
    ):
        return None

    content = b"".join(chunks)
    return _FileAttestation(
        identity=(after.st_dev, after.st_ino),
        metadata=(
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ),
        sha256=hashlib.sha256(content).hexdigest(),
        content=content,
    )


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )


def _open_stable_directory(path: Path) -> int | None:
    if path_has_symlink_component(path, include_leaf=True):
        return None
    try:
        pathname_before = path.stat(follow_symlinks=False)
        if not stat.S_ISDIR(pathname_before.st_mode):
            return None
        descriptor = os.open(path, _directory_open_flags())
    except OSError:
        return None
    try:
        opened = os.fstat(descriptor)
        pathname_after = path.stat(follow_symlinks=False)
    except OSError:
        os.close(descriptor)
        return None
    if not (
        _file_stat_signature(pathname_before)
        == _file_stat_signature(opened)
        == _file_stat_signature(pathname_after)
    ):
        os.close(descriptor)
        return None
    return descriptor


def _open_child_directory(directory_descriptor: int, name: str) -> int | None:
    try:
        pathname_before = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if not stat.S_ISDIR(pathname_before.st_mode):
            return None
        descriptor = os.open(name, _directory_open_flags(), dir_fd=directory_descriptor)
        opened = os.fstat(descriptor)
        pathname_after = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    except OSError:
        if "descriptor" in locals():
            os.close(descriptor)
        return None
    if not (
        _file_stat_signature(pathname_before)
        == _file_stat_signature(opened)
        == _file_stat_signature(pathname_after)
    ):
        os.close(descriptor)
        return None
    return descriptor


def _directory_descriptor_matches_path(descriptor: int, path: Path) -> bool:
    if path_has_symlink_component(path, include_leaf=True):
        return False
    try:
        pathname_stat = path.stat(follow_symlinks=False)
        descriptor_stat = os.fstat(descriptor)
    except OSError:
        return False
    return _file_stat_signature(pathname_stat) == _file_stat_signature(descriptor_stat)


def _recapture_private_semantic_snapshot(
    snapshot: _PrivateSemanticSnapshot,
) -> _CapturedDirectoryTree | None:
    descriptor = _open_stable_directory(snapshot.boundary_path)
    if descriptor is None:
        return None
    try:
        try:
            root_stat = os.fstat(descriptor)
        except OSError:
            return None
        directories: list[tuple[str, tuple[int, ...]]] = []
        for relative, expected_signature in snapshot.tree.directories:
            directory_stat = _relative_path_stat(descriptor, PurePosixPath(relative))
            if directory_stat is None or not stat.S_ISDIR(directory_stat.st_mode):
                return None
            signature = (
                _directory_identity(directory_stat)
                if len(expected_signature) == 3
                else _file_stat_signature(directory_stat)
            )
            directories.append((relative, signature))
        files: list[tuple[str, _FileAttestation]] = []
        for relative, _expected_attestation in snapshot.tree.files:
            attestation = _attest_relative_file(descriptor, PurePosixPath(relative))
            if attestation is None:
                return None
            files.append((relative, attestation))
        return _CapturedDirectoryTree(
            root=_directory_identity(root_stat),
            directories=tuple(directories),
            files=tuple(files),
        )
    finally:
        os.close(descriptor)


def _capture_referenced_tree_fd(
    root_descriptor: int,
    source_relative_path: Path,
    *,
    repository_boundary: bool,
    allow_parent_expansion: bool = False,
) -> _CapturedDirectoryTree | None:
    files: dict[str, _FileAttestation] = {}
    directories: dict[str, tuple[int, ...]] = {}
    pending_json: list[PurePosixPath] = []
    scanned_json: set[str] = set()

    def add_file(relative: PurePosixPath) -> bool:
        key = relative.as_posix()
        if key in files:
            return True
        attestation = _attest_relative_file(root_descriptor, relative)
        if attestation is None:
            return False
        files[key] = attestation
        pending_json.append(relative)
        return True

    def add_directory(relative: PurePosixPath) -> bool:
        captured = _capture_relative_directory(root_descriptor, relative)
        if captured is None:
            return False
        prefix = "" if relative == PurePosixPath(".") else relative.as_posix()
        if prefix:
            directories[prefix] = captured.root
        for child, signature in captured.directories:
            key = f"{prefix}/{child}" if prefix else child
            directories[key] = signature
        for child, attestation in captured.files:
            key = f"{prefix}/{child}" if prefix else child
            if key not in files:
                files[key] = attestation
                pending_json.append(PurePosixPath(key))
        return True

    source_relative = PurePosixPath(source_relative_path.as_posix())
    admitted_top_level = source_relative.parts[0] if source_relative.parts else ""
    if not add_file(source_relative):
        return None

    if repository_boundary:
        marker = PurePosixPath("pyproject.toml")
        if _relative_path_kind(root_descriptor, marker) == "file" and not add_file(marker):
            return None
        if _relative_path_kind(root_descriptor, PurePosixPath(".git")) == "directory":
            marker_stat = _relative_path_stat(root_descriptor, PurePosixPath(".git"))
            if marker_stat is not None:
                directories[".git"] = _directory_identity(marker_stat)

    while pending_json:
        current = pending_json.pop()
        current_key = current.as_posix()
        if current_key in scanned_json:
            continue
        scanned_json.add(current_key)
        attestation = files[current_key]
        try:
            payload = json.loads(attestation.content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("schema_version") == "hfr.rl.manifest.v1":
            split_directory = _normalized_relative_path(current.parent, "splits")
            if (
                split_directory is not None
                and _relative_path_kind(root_descriptor, split_directory) == "directory"
                and not add_directory(split_directory)
            ):
                return None
        for raw_value in _iter_referenced_path_values(payload):
            candidate = _referenced_relative_path(
                root_descriptor,
                current.parent,
                raw_value,
                repository_boundary=repository_boundary,
                allow_parent_expansion=allow_parent_expansion,
                admitted_top_level=admitted_top_level,
            )
            if candidate is None:
                continue
            kind = _relative_path_kind(root_descriptor, candidate)
            if kind == "file":
                if not add_file(candidate):
                    return None
            elif kind == "directory" and not add_directory(candidate):
                return None

    try:
        root_stat = os.fstat(root_descriptor)
    except OSError:
        return None
    return _CapturedDirectoryTree(
        root=_directory_identity(root_stat),
        directories=tuple(sorted(directories.items())),
        files=tuple(sorted(files.items())),
    )


_EXPLICIT_PATH_FIELD_NAMES = {
    "compatibility_report",
    "events",
    "home",
    "lineage",
    "manifest",
    "normalized_trace",
    "report",
    "reviewed_export",
    "reviewed_labels",
    "root",
    "scorecard",
    "summary",
    "workspace",
}

_PATH_MAPPING_FIELD_NAMES = {"artifacts", "files", "outputs"}


def _iter_referenced_path_values(value: Any, *, field_name: str = "") -> Iterator[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized_key = key.lower() if isinstance(key, str) else ""
            if isinstance(item, str) and (
                _is_path_field_name(normalized_key)
                or field_name in _PATH_MAPPING_FIELD_NAMES
            ):
                yield item
            elif isinstance(item, (dict, list)):
                yield from _iter_referenced_path_values(item, field_name=normalized_key)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                yield from _iter_referenced_path_values(item, field_name=field_name)
            elif isinstance(item, str) and _is_path_field_name(field_name):
                yield item


def _is_path_field_name(field_name: str) -> bool:
    return (
        field_name == "path"
        or field_name.endswith(
            ("_path", "_paths", "_file", "_files", "_dir", "_dirs", "_root", "_roots")
        )
        or field_name in _EXPLICIT_PATH_FIELD_NAMES
    )


def _referenced_relative_path(
    root_descriptor: int,
    source_parent: PurePosixPath,
    raw_value: str,
    *,
    repository_boundary: bool,
    allow_parent_expansion: bool,
    admitted_top_level: str,
) -> PurePosixPath | None:
    if raw_value.startswith("<redacted:") and raw_value.endswith(">"):
        raw_value = raw_value[len("<redacted:") : -1]
    if (
        not raw_value
        or raw_value in {".", ".."}
        or "\\" in raw_value
        or "\x00" in raw_value
        or "://" in raw_value
        or PurePosixPath(raw_value).is_absolute()
    ):
        return None
    raw_parts = PurePosixPath(raw_value).parts
    first_path_part = next((part for part in raw_parts if part not in {".", ".."}), "")
    normalized_source = posixpath.normpath((source_parent / raw_value).as_posix())
    escaped_parent_levels = 0
    for part in PurePosixPath(normalized_source).parts:
        if part != "..":
            break
        escaped_parent_levels += 1
    if escaped_parent_levels:
        if (
            first_path_part in _PRIVATE_SNAPSHOT_EXCLUDED_REPOSITORY_ROOTS
            and first_path_part != admitted_top_level
        ):
            return None
        if allow_parent_expansion:
            raise _SnapshotBoundaryExpansion(escaped_parent_levels)
        return None
    source_candidate = _normalized_relative_path(source_parent, raw_value)
    source_candidate_excluded = _repository_snapshot_candidate_excluded(
        source_candidate,
        repository_boundary=repository_boundary,
        admitted_top_level=admitted_top_level,
    )
    if (
        source_candidate is not None
        and not source_candidate_excluded
        and _relative_path_kind(root_descriptor, source_candidate)
    ):
        return source_candidate
    for index in range(1, len(raw_parts)):
        suffix_candidate = _normalized_relative_path(
            source_parent,
            PurePosixPath(*raw_parts[index:]).as_posix(),
        )
        if (
            suffix_candidate is not None
            and not _repository_snapshot_candidate_excluded(
                suffix_candidate,
                repository_boundary=repository_boundary,
                admitted_top_level=admitted_top_level,
            )
            and _relative_path_kind(root_descriptor, suffix_candidate)
        ):
            return suffix_candidate
    if repository_boundary:
        repo_candidate = _normalized_relative_path(PurePosixPath("."), raw_value)
        if (
            repo_candidate is not None
            and not _repository_snapshot_candidate_excluded(
                repo_candidate,
                repository_boundary=repository_boundary,
                admitted_top_level=admitted_top_level,
            )
            and _relative_path_kind(root_descriptor, repo_candidate)
        ):
            return repo_candidate
    return None


def _repository_snapshot_candidate_excluded(
    candidate: PurePosixPath | None,
    *,
    repository_boundary: bool,
    admitted_top_level: str,
) -> bool:
    if not repository_boundary or candidate is None or not candidate.parts:
        return False
    top_level = candidate.parts[0]
    if top_level not in _PRIVATE_SNAPSHOT_EXCLUDED_REPOSITORY_ROOTS:
        return False
    return top_level != admitted_top_level or len(candidate.parts) == 1


def _normalized_relative_path(base: PurePosixPath, raw_value: str) -> PurePosixPath | None:
    normalized = posixpath.normpath((base / raw_value).as_posix())
    if normalized == "." or normalized == ".." or normalized.startswith("../"):
        return None
    return PurePosixPath(normalized)


def _capture_relative_directory(
    root_descriptor: int,
    relative: PurePosixPath,
) -> _CapturedDirectoryTree | None:
    descriptor = _open_relative_directory(root_descriptor, relative)
    if descriptor is None:
        return None
    try:
        return _capture_directory_tree_fd(descriptor)
    finally:
        os.close(descriptor)


def _attest_relative_file(
    root_descriptor: int,
    relative: PurePosixPath,
) -> _FileAttestation | None:
    parent_descriptor = _open_relative_directory(root_descriptor, relative.parent)
    if parent_descriptor is None:
        return None
    try:
        return _attest_regular_file_at(parent_descriptor, relative.name)
    finally:
        os.close(parent_descriptor)


def _relative_path_kind(root_descriptor: int, relative: PurePosixPath) -> str | None:
    path_stat = _relative_path_stat(root_descriptor, relative)
    if path_stat is None:
        return None
    if stat.S_ISREG(path_stat.st_mode):
        return "file"
    if stat.S_ISDIR(path_stat.st_mode):
        return "directory"
    return None


def _relative_path_stat(
    root_descriptor: int,
    relative: PurePosixPath,
) -> os.stat_result | None:
    if relative == PurePosixPath("."):
        try:
            return os.fstat(root_descriptor)
        except OSError:
            return None
    parent_descriptor = _open_relative_directory(root_descriptor, relative.parent)
    if parent_descriptor is None:
        return None
    try:
        return os.stat(relative.name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError:
        return None
    finally:
        os.close(parent_descriptor)


def _open_relative_directory(
    root_descriptor: int,
    relative: PurePosixPath,
) -> int | None:
    try:
        descriptor = os.dup(root_descriptor)
    except OSError:
        return None
    if relative == PurePosixPath("."):
        return descriptor
    for part in relative.parts:
        if part in {"", ".", ".."}:
            os.close(descriptor)
            return None
        child_descriptor = _open_child_directory(descriptor, part)
        os.close(descriptor)
        if child_descriptor is None:
            return None
        descriptor = child_descriptor
    return descriptor


def _directory_identity(directory_stat: os.stat_result) -> tuple[int, ...]:
    return (directory_stat.st_dev, directory_stat.st_ino, directory_stat.st_mode)


def _capture_directory_tree_fd(root_descriptor: int) -> _CapturedDirectoryTree | None:
    directories: list[tuple[str, tuple[int, ...]]] = []
    files: list[tuple[str, _FileAttestation]] = []

    def capture(directory_descriptor: int, prefix: PurePosixPath) -> bool:
        try:
            directory_before = os.fstat(directory_descriptor)
            if not stat.S_ISDIR(directory_before.st_mode):
                return False
            names = sorted(os.listdir(directory_descriptor))
        except (OSError, UnicodeError):
            return False

        for name in names:
            relative = (prefix / name).as_posix()
            try:
                entry_before = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
            except OSError:
                return False
            if stat.S_ISDIR(entry_before.st_mode):
                if name in _PRIVATE_SNAPSHOT_EXCLUDED_DIRECTORY_NAMES:
                    continue
                child_descriptor = _open_child_directory(directory_descriptor, name)
                if child_descriptor is None:
                    return False
                try:
                    directories.append((relative, _file_stat_signature(entry_before)))
                    if not capture(child_descriptor, prefix / name):
                        return False
                finally:
                    os.close(child_descriptor)
            elif stat.S_ISREG(entry_before.st_mode):
                file_attestation = _attest_regular_file_at(directory_descriptor, name)
                if file_attestation is None:
                    return False
                files.append((relative, file_attestation))
            else:
                # Never reproduce links, sockets, devices, or FIFOs in the
                # private tree. A validator that references one consequently
                # sees a missing artifact and fails closed; unrelated special
                # entries do not invalidate an otherwise regular closure.
                continue
            try:
                entry_after = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
            except OSError:
                return False
            if _file_stat_signature(entry_before) != _file_stat_signature(entry_after):
                return False

        try:
            directory_after = os.fstat(directory_descriptor)
        except OSError:
            return False
        return _file_stat_signature(directory_before) == _file_stat_signature(directory_after)

    try:
        root_before = os.fstat(root_descriptor)
    except OSError:
        return None
    if not capture(root_descriptor, PurePosixPath(".")):
        return None
    try:
        root_after = os.fstat(root_descriptor)
    except OSError:
        return None
    if _file_stat_signature(root_before) != _file_stat_signature(root_after):
        return None
    return _CapturedDirectoryTree(
        root=_file_stat_signature(root_after),
        directories=tuple(directories),
        files=tuple(files),
    )


def _file_stat_signature(file_stat: os.stat_result) -> tuple[int, ...]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_mode,
        file_stat.st_nlink,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


def _attest_source_directory(path: Path) -> _DirectoryTreeAttestation | None:
    """Return a stable, type-aware snapshot of an artifact directory tree."""
    first = _attest_source_directory_once(path)
    if first is None:
        return None
    second = _attest_source_directory_once(path)
    return second if first == second else None


def _attest_source_directory_once(path: Path) -> _DirectoryTreeAttestation | None:
    if path_has_symlink_component(path, include_leaf=True):
        return None
    try:
        root_before = path.stat(follow_symlinks=False)
        if not stat.S_ISDIR(root_before.st_mode):
            return None
        children = sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix())
        entries: list[tuple[str, str, tuple[int, ...], str]] = []
        for child in children:
            relative = child.relative_to(path).as_posix()
            child_stat = child.stat(follow_symlinks=False)
            if stat.S_ISDIR(child_stat.st_mode):
                entries.append((relative, "directory", _file_stat_signature(child_stat), ""))
                continue
            if not stat.S_ISREG(child_stat.st_mode):
                return None
            file_attestation = _attest_regular_file(child)
            if file_attestation is None:
                return None
            entries.append(
                (
                    relative,
                    "file",
                    file_attestation.identity + file_attestation.metadata,
                    file_attestation.sha256,
                )
            )
        root_after = path.stat(follow_symlinks=False)
    except (OSError, UnicodeError, ValueError):
        return None
    if _file_stat_signature(root_before) != _file_stat_signature(root_after):
        return None
    return _DirectoryTreeAttestation(root=_file_stat_signature(root_after), entries=tuple(entries))


def _inspect_training_export(path: Path, *, require_semantics: bool) -> dict[str, Any]:
    before = _attest_source_directory(path)
    physical_exists = before is not None or path.exists()
    regular_directory = before is not None
    manifest_path = path / "manifest.json"
    manifest = (
        inspect_json_source(
            manifest_path,
            "training_manifest",
            role="training_export",
            require_semantics=require_semantics,
        )
        if regular_directory
        else _empty_inspection(manifest_path)
    )
    contract_valid = not require_semantics or manifest["semantic_valid"]
    after = _attest_source_directory(path) if before is not None else None
    stable = before is not None and before == after
    semantic_valid = manifest["semantic_valid"] and contract_valid and stable
    return {
        "path": path,
        "physical_exists": physical_exists,
        "regular_directory": regular_directory,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "payload": manifest["payload"],
        "schema_valid": manifest["schema_valid"],
        "semantic_valid": semantic_valid,
        "stable": stable,
        "ready": (
            regular_directory
            and manifest["regular_file"]
            and manifest["schema_valid"]
            and semantic_valid
            and stable
        ),
    }


def _inspect_directory(path: Path, role: str, *, require_semantics: bool) -> dict[str, Any]:
    before = _attest_source_directory(path)
    physical_exists = before is not None or path.exists()
    regular_directory = before is not None
    manifest_path = path / _DIRECTORY_MANIFESTS[role]
    manifest = (
        inspect_json_source(
            manifest_path,
            role,
            role=role,
            require_semantics=require_semantics,
        )
        if regular_directory
        else _empty_inspection(manifest_path)
    )
    contract_valid = manifest["schema_valid"] and (
        not require_semantics or manifest["semantic_valid"]
    )
    after = _attest_source_directory(path) if before is not None else None
    stable = before is not None and before == after
    semantic_valid = manifest["semantic_valid"] and contract_valid and stable
    return {
        "path": path,
        "physical_exists": physical_exists,
        "regular_directory": regular_directory,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "payload": manifest["payload"],
        "schema_valid": manifest["schema_valid"],
        "semantic_valid": semantic_valid,
        "stable": stable,
        "ready": (
            regular_directory
            and manifest["regular_file"]
            and manifest["schema_valid"]
            and semantic_valid
            and stable
        ),
    }


def _directory_contract_valid(path: Path, role: str) -> bool:
    try:
        from .validation import validate_promotion_archive, validate_promotion_cards

        validator = validate_promotion_archive if role == "promotion_archive" else validate_promotion_cards
        return not validator(path).errors
    except (OSError, UnicodeError, json.JSONDecodeError, SchemaRegistryError, TypeError, ValueError):
        return False


def _training_export_contract_valid(path: Path) -> bool:
    try:
        from .validation import validate_training_export

        return not validate_training_export(path).errors
    except (OSError, UnicodeError, json.JSONDecodeError, SchemaRegistryError, TypeError, ValueError):
        return False


def _semantic_contract_valid(path: Path, role: str) -> bool:
    if role == "training_export":
        return _training_export_contract_valid(path.parent)
    if role in _DIRECTORY_MANIFESTS:
        return _directory_contract_valid(path.parent, role)
    if role in _GATE_CONTRACT_ROLES:
        try:
            from .gate_contract import summarize_gate_contract

            payload = json.loads(path.read_text(encoding="utf-8"))
            summary = summarize_gate_contract(payload)
        except (OSError, TypeError, UnicodeError, ValueError, json.JSONDecodeError):
            return False
        checks = payload.get("checks") if isinstance(payload.get("checks"), list) else []
        decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
        return (
            summary.get("valid") is True
            and payload.get("check_count") == len(checks)
            and decision.get("key_metrics") == payload.get("metrics")
        )
    validator_name = _SEMANTIC_VALIDATOR_NAMES.get(role)
    if validator_name is None:
        return True
    try:
        from . import validation

        validator = getattr(validation, validator_name)
        result = validator(path)
    except (
        AttributeError,
        ImportError,
        IndexError,
        KeyError,
        OSError,
        SchemaRegistryError,
        TypeError,
        UnicodeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False
    errors = getattr(result, "errors", None)
    return isinstance(errors, list) and not errors


def _empty_inspection(path: Path) -> dict[str, Any]:
    return {
        "path": path,
        "physical_exists": False,
        "regular_file": False,
        "parse_valid": False,
        "schema_valid": False,
        "semantic_valid": False,
        "stable": False,
        "ready": False,
        "payload": {},
    }


def _semantic_ready(role: str, payload: dict[str, Any]) -> bool:
    for field_name, expected in _REQUIRED_VALUES.get(role, {}).items():
        actual = payload.get(field_name)
        if isinstance(expected, bool):
            if actual is not expected:
                return False
        elif actual != expected:
            return False

    if role == "rubric_spec":
        criteria = payload.get("criteria")
        count = payload.get("criterion_count")
        return isinstance(criteria, list) and isinstance(count, int) and not isinstance(count, bool) and count > 0 and count == len(criteria)
    if role == "harness_result":
        scorecard = payload.get("scorecard")
        return isinstance(scorecard, dict) and scorecard.get("passed") is True
    if role == "cloud_training_provider_registry":
        providers = payload.get("providers")
        count = payload.get("provider_count")
        return isinstance(providers, list) and isinstance(count, int) and not isinstance(count, bool) and count > 0 and count == len(providers)
    if role == "heldout_manifest":
        count = payload.get("scenario_count")
        return isinstance(count, int) and not isinstance(count, bool) and count > 0
    if role == "external_eval_plan":
        count = payload.get("adapter_count")
        ready_count = payload.get("ready_adapter_count")
        return (
            isinstance(count, int)
            and not isinstance(count, bool)
            and count > 0
            and ready_count == count
        )
    if role == "training_export":
        redaction = payload.get("redaction_status")
        episode_count = payload.get("episode_count")
        return (
            isinstance(redaction, dict)
            and redaction.get("passed") is True
            and isinstance(episode_count, int)
            and not isinstance(episode_count, bool)
            and episode_count > 0
        )
    return True
