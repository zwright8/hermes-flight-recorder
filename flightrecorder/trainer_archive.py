"""Portable archives for trainer handoff evidence."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import stat
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .path_safety import (
    assert_safe_output_directory,
    path_has_symlink_component,
    remove_directory_tree_if_identity,
)
from .preflight import (
    TRAINER_DIRECTORY_TREE_HASH_ALGORITHM,
    TRAINER_LAUNCH_CHECK_SCHEMA_VERSION,
    TRAINER_PREFLIGHT_SCHEMA_VERSION,
)
from .schema_registry import SchemaRegistryError, check_schema_contract

TRAINER_ARCHIVE_SCHEMA_VERSION = "hfr.trainer_archive.v1"
_ARCHIVE_MANIFEST = "trainer_archive.json"
_ARCHIVE_MARKER = ".hfr-trainer-archive"
_ARCHIVE_MARKER_CONTENT = f"{TRAINER_ARCHIVE_SCHEMA_VERSION}\n"
_TREE_HASH_ALGORITHM = TRAINER_DIRECTORY_TREE_HASH_ALGORITHM
_REPO_ROOT = Path(__file__).resolve().parents[1]


class TrainerArchiveError(ValueError):
    """Raised when a trainer handoff archive cannot be produced."""


@dataclass(frozen=True)
class _DirectoryAttestation:
    identity: tuple[int, int]
    sha256: str
    file_count: int
    size_bytes: int


def build_trainer_archive(
    *,
    out_dir: str | Path,
    preflight_path: str | Path,
    launch_check_path: str | Path | None = None,
    require_self_contained: bool = False,
    force: bool = False,
    preserve_paths: bool = False,
) -> dict[str, Any]:
    """Copy trainer-launch evidence into a portable hash-checked directory."""
    target = Path(out_dir)
    source_preflight_path = Path(preflight_path)
    preflight, preflight_attestation = _read_json_artifact(
        source_preflight_path,
        TRAINER_PREFLIGHT_SCHEMA_VERSION,
        "trainer preflight",
    )
    _require_schema(preflight, "trainer_preflight", "trainer preflight")
    if launch_check_path is None:
        raise TrainerArchiveError("trainer launch check path is required")
    source_launch_path = Path(launch_check_path)
    launch_check, launch_attestation = _read_json_artifact(
        source_launch_path,
        TRAINER_LAUNCH_CHECK_SCHEMA_VERSION,
        "trainer launch check",
    )
    _require_schema(launch_check, "trainer_launch_check", "trainer launch check")

    reference_issues = _preflight_reference_issues(
        preflight, source_preflight_path.parent
    )
    semantic_errors: list[str] = []
    if not reference_issues:
        semantic_errors.extend(
            _semantic_validation_errors(
                source_preflight_path,
                source_launch_path,
                preflight,
                launch_check,
            )
        )
    binding_errors = _source_binding_errors(
        preflight,
        launch_check,
        source_preflight_path=source_preflight_path,
        source_launch_path=source_launch_path,
    )
    readiness_errors = _readiness_errors(preflight, launch_check)
    source_controls_passed = (
        not reference_issues
        and not semantic_errors
        and not binding_errors
        and not readiness_errors
    )

    with _archive_lock(target):
        target_attestation = _prepare_archive_dir(target, force)
        staging_dir, staging_identity = _create_private_work_directory(target, "stage")
        staging_attestation: _DirectoryAttestation | None = None
        published = False
        try:
            archive = _build_archive_contents(
                staging_dir=staging_dir,
                display_target=target,
                source_preflight_path=source_preflight_path,
                source_launch_path=source_launch_path,
                preflight=preflight,
                launch_check=launch_check,
                preflight_attestation=preflight_attestation,
                launch_attestation=launch_attestation,
                reference_issues=reference_issues,
                semantic_errors=semantic_errors,
                binding_errors=binding_errors,
                readiness_errors=readiness_errors,
                source_controls_passed=source_controls_passed,
                require_self_contained=require_self_contained,
                preserve_paths=preserve_paths,
            )
            _validate_complete_archive(staging_dir, "staged trainer archive")
            _fsync_archive_tree(staging_dir)
            staging_attestation = _attest_directory(
                staging_dir, expected_identity=staging_identity
            )
            _publish_staged_archive(
                staging_dir,
                target,
                staging_attestation=staging_attestation,
                target_attestation=target_attestation,
            )
            published = True
            return archive
        finally:
            if not published:
                if staging_attestation is None:
                    _remove_owned_directory(staging_dir, staging_identity)
                else:
                    _remove_attested_directory(staging_dir, staging_attestation)


def _build_archive_contents(
    *,
    staging_dir: Path,
    display_target: Path,
    source_preflight_path: Path,
    source_launch_path: Path,
    preflight: dict[str, Any],
    launch_check: dict[str, Any],
    preflight_attestation: dict[str, Any],
    launch_attestation: dict[str, Any],
    reference_issues: dict[tuple[str, int, str], str],
    semantic_errors: list[str],
    binding_errors: list[str],
    readiness_errors: list[str],
    source_controls_passed: bool,
    require_self_contained: bool,
    preserve_paths: bool,
) -> dict[str, Any]:
    target = staging_dir
    artifacts_dir = target / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (target / _ARCHIVE_MARKER).write_text(_ARCHIVE_MARKER_CONTENT, encoding="utf-8")

    artifacts: list[dict[str, Any]] = [
        _copy_file_artifact(
            "trainer_preflight",
            "trainer_preflight",
            source_preflight_path,
            artifacts_dir / "trainer_preflight.json",
            target,
            preserve_paths,
            expected=preflight_attestation,
        )
    ]
    missing: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []

    launch_record = _copy_file_artifact(
        "trainer_launch_check",
        "trainer_launch_check",
        source_launch_path,
        artifacts_dir / "trainer_launch_check.json",
        target,
        preserve_paths,
        expected=launch_attestation,
    )
    artifacts.append(launch_record)
    if not binding_errors:
        relationships.append(
            {
                "from": "trainer_launch_check",
                "to": "trainer_preflight",
                "type": "validates",
            }
        )

    if source_controls_passed:
        artifacts.extend(
            _copy_preflight_paths(
                preflight.get("gates"),
                role="gate",
                base_dir=source_preflight_path.parent,
                archive_dir=artifacts_dir / "gates",
                archive_root=target,
                missing=missing,
                preserve_paths=preserve_paths,
            )
        )
        artifacts.extend(
            _copy_preflight_paths(
                preflight.get("validation_summaries"),
                role="validation_summary",
                base_dir=source_preflight_path.parent,
                archive_dir=artifacts_dir / "validation_summaries",
                archive_root=target,
                missing=missing,
                preserve_paths=preserve_paths,
            )
        )
        artifacts.extend(
            _copy_preflight_mapping(
                preflight.get("artifacts"),
                role="trainer_artifact",
                base_dir=source_preflight_path.parent,
                archive_dir=artifacts_dir / "trainer_artifacts",
                archive_root=target,
                missing=missing,
                preserve_paths=preserve_paths,
            )
        )
        artifacts.extend(
            _copy_preflight_mapping(
                preflight.get("schema_contracts"),
                role="schema_contract",
                base_dir=source_preflight_path.parent,
                archive_dir=artifacts_dir / "schema_contracts",
                archive_root=target,
                missing=missing,
                preserve_paths=preserve_paths,
            )
        )
    else:
        blocker = _source_control_blocker(
            semantic_errors, binding_errors, readiness_errors
        )
        missing.extend(
            _blocked_reference_records(
                preflight,
                base_dir=source_preflight_path.parent,
                reference_issues=reference_issues,
                default_reason=blocker,
            )
        )

    for index, artifact in enumerate(artifacts):
        artifact["index"] = index

    approved_command = _approved_command_record(launch_check)
    if not source_controls_passed:
        approved_command["approved"] = False
    path_rewrites = _path_rewrites(artifacts, approved_command)
    trainer_inputs = _trainer_inputs(artifacts)
    portable_command = _portable_command_record(approved_command, path_rewrites)
    consumer_contract = _consumer_contract(
        portable_command, trainer_inputs, path_rewrites
    )
    self_contained = not missing
    launch_included = True
    launch_passed = launch_check.get("passed") is True
    ready_for_training = preflight.get("passed") is True and launch_passed
    effective_require_self_contained = (
        require_self_contained or not source_controls_passed
    )
    passed = ready_for_training and (
        self_contained or not effective_require_self_contained
    )
    archive = {
        "schema_version": TRAINER_ARCHIVE_SCHEMA_VERSION,
        "archive_path": _display_path(display_target, preserve_paths),
        "manifest_path": _ARCHIVE_MANIFEST,
        "passed": passed,
        "readiness": "ready" if passed else "blocked",
        "recommendation": "handoff_ready" if passed else "block_handoff",
        "self_contained": self_contained,
        "require_self_contained": effective_require_self_contained,
        "ready_for_training": ready_for_training,
        "launch_check_included": launch_included,
        "approved_command": approved_command,
        "trainer_inputs": trainer_inputs,
        "path_rewrites": path_rewrites,
        "portable_command": portable_command,
        "consumer_contract": consumer_contract,
        "artifacts": artifacts,
        "missing": missing,
        "relationships": relationships,
        "metrics": _metrics(
            artifacts, missing, trainer_inputs, path_rewrites, consumer_contract
        ),
        "notes": [
            "Trainer archives copy trainer handoff evidence into a portable directory; they do not train models or execute the trainer command.",
            "The portable command is advisory: it rewrites known trainer-input paths to archive-local paths but does not include trainer code or execute anything.",
            "Archive validation checks copied file hashes and directory tree hashes, so original local source paths are not required after the archive is built.",
        ],
    }
    _write_json(target / _ARCHIVE_MANIFEST, archive)
    return archive


def _require_schema(value: dict[str, Any], schema_name: str, label: str) -> None:
    try:
        result = check_schema_contract(value, name_or_id=schema_name)
    except SchemaRegistryError as exc:
        raise TrainerArchiveError(f"{label} schema validation failed: {exc}") from exc
    if result.get("passed") is True:
        return
    errors = result.get("errors") if isinstance(result.get("errors"), list) else []
    detail = "; ".join(str(error) for error in errors[:8]) or "unknown schema error"
    raise TrainerArchiveError(f"{label} schema validation failed: {detail}")


def _semantic_validation_errors(
    preflight_path: Path,
    launch_check_path: Path,
    preflight: dict[str, Any],
    launch_check: dict[str, Any],
) -> list[str]:
    # Imported lazily because validation imports this module's public schema constant.
    from .validation import validate_trainer_launch_check, validate_trainer_preflight

    errors: list[str] = []
    for label, target in (
        (
            "trainer preflight",
            validate_trainer_preflight(preflight_path, payload=preflight),
        ),
        (
            "trainer launch check",
            validate_trainer_launch_check(launch_check_path, payload=launch_check),
        ),
    ):
        errors.extend(f"{label}: {error}" for error in target.errors)
    return errors


def _preflight_reference_issues(
    preflight: dict[str, Any],
    base_dir: Path,
) -> dict[tuple[str, int, str], str]:
    issues: dict[tuple[str, int, str], str] = {}
    artifacts = (
        preflight.get("artifacts")
        if isinstance(preflight.get("artifacts"), dict)
        else {}
    )
    for role, index, name, record in _iter_preflight_records(preflight):
        path_error = _recorded_path_safety_error(record.get("path"), base_dir)
        if path_error is not None:
            issues[(role, index, name)] = path_error
            continue
        if role == "schema_contract":
            artifact = artifacts.get(name)
            if not isinstance(artifact, dict) or artifact.get("path") != record.get(
                "path"
            ):
                issues[(role, index, name)] = (
                    "schema contract path is not bound to its source artifact"
                )
    return issues


def _iter_preflight_records(
    preflight: dict[str, Any],
) -> list[tuple[str, int, str, dict[str, Any]]]:
    records: list[tuple[str, int, str, dict[str, Any]]] = []
    for role, key in (
        ("gate", "gates"),
        ("validation_summary", "validation_summaries"),
    ):
        value = preflight.get(key)
        if not isinstance(value, list):
            continue
        for index, item in enumerate(value):
            if isinstance(item, dict):
                records.append((role, index, _record_name(role, item, index), item))
    for role, key in (
        ("trainer_artifact", "artifacts"),
        ("schema_contract", "schema_contracts"),
    ):
        value = preflight.get(key)
        if not isinstance(value, dict):
            continue
        for index, (raw_name, item) in enumerate(sorted(value.items())):
            if isinstance(item, dict):
                records.append((role, index, str(raw_name or f"{role}_{index}"), item))
    return records


def _recorded_path_safety_error(value: Any, base_dir: Path) -> str | None:
    if (
        not isinstance(value, str)
        or not value
        or value.startswith("<redacted:")
        or value.startswith("<missing-")
    ):
        return None
    if _is_windows_absolute(value):
        return f"absolute recorded path is not allowed: {value}"
    try:
        raw = Path(value)
    except (OSError, ValueError) as exc:
        return f"recorded path is invalid: {value}: {exc}"
    if raw.is_absolute():
        return f"absolute recorded path is not allowed: {value}"
    if ".." in raw.parts:
        return f"recorded path contains parent traversal: {value}"
    candidate = base_dir / raw
    if path_has_symlink_component(candidate, include_leaf=True):
        return f"recorded path traverses a symlink component: {value}"
    if not _path_resolves_inside(candidate, base_dir):
        return f"recorded path is not bound to the preflight root: {value}"
    return None


def _blocked_reference_records(
    preflight: dict[str, Any],
    *,
    base_dir: Path,
    reference_issues: dict[tuple[str, int, str], str],
    default_reason: str,
) -> list[dict[str, Any]]:
    blocked: list[dict[str, Any]] = []
    for role, index, name, record in _iter_preflight_records(preflight):
        reason = reference_issues.get((role, index, name))
        if reason is None:
            source_path, resolution_error = _resolve_recorded_path(
                record.get("path"), base_dir
            )
            if source_path is None:
                reason = resolution_error or default_reason
            else:
                reason = (
                    _record_integrity_error(record, source_path, role) or default_reason
                )
        blocked.append(_missing(role, index, reason, name=name))
    if not blocked:
        blocked.append(
            _missing("trainer_artifact", 0, default_reason, name="preflight_references")
        )
    return blocked


def _source_control_blocker(
    semantic_errors: list[str],
    binding_errors: list[str],
    readiness_errors: list[str],
) -> str:
    if semantic_errors:
        return f"preflight semantic validation failed: {'; '.join(semantic_errors[:8])}"
    if binding_errors:
        return f"launch check is not bound to the supplied preflight: {'; '.join(binding_errors[:8])}"
    if readiness_errors:
        return (
            f"trainer handoff readiness is blocked: {'; '.join(readiness_errors[:8])}"
        )
    return "preflight reference safety validation failed"


def _source_binding_errors(
    preflight: dict[str, Any],
    launch_check: dict[str, Any],
    *,
    source_preflight_path: Path,
    source_launch_path: Path,
) -> list[str]:
    errors: list[str] = []
    preflight_binding = _recorded_source_binding_error(
        preflight.get("preflight_path"),
        source_preflight_path,
        source_preflight_path.parent,
    )
    if preflight_binding is not None:
        errors.append(f"trainer_preflight.preflight_path {preflight_binding}")
    launch_binding = _recorded_source_binding_error(
        launch_check.get("preflight_path"),
        source_preflight_path,
        source_launch_path.parent,
    )
    if launch_binding is not None:
        errors.append(f"trainer_launch_check.preflight_path {launch_binding}")

    gates = preflight.get("gates") if isinstance(preflight.get("gates"), list) else []
    expected_gates = [
        {
            "id": gate.get("id"),
            "path": gate.get("path"),
            "schema_version": gate.get("schema_version"),
            "passed": gate.get("passed"),
        }
        for gate in gates
        if isinstance(gate, dict)
    ]
    if launch_check.get("gates") != expected_gates:
        errors.append("gates do not match the supplied preflight")

    artifacts = (
        preflight.get("artifacts")
        if isinstance(preflight.get("artifacts"), dict)
        else {}
    )
    expected_artifacts = {
        "count": len(artifacts),
        "names": sorted(str(name) for name in artifacts),
    }
    if launch_check.get("artifacts") != expected_artifacts:
        errors.append("artifact summary does not match the supplied preflight")
    if launch_check.get("dataset_selection") != preflight.get("dataset_selection"):
        errors.append("dataset selection does not match the supplied preflight")

    preflight_command = (
        preflight.get("trainer_command")
        if isinstance(preflight.get("trainer_command"), dict)
        else {}
    )
    launch_command = (
        launch_check.get("approved_command")
        if isinstance(launch_check.get("approved_command"), dict)
        else {}
    )
    for key in ("provided", "raw", "argv", "parseable"):
        expected = preflight_command.get(
            key, bool(preflight_command.get("argv")) if key == "parseable" else None
        )
        if launch_command.get(key) != expected:
            errors.append(
                f"approved_command.{key} does not match the supplied preflight"
            )

    required_versions = preflight.get("required_dataset_versions")
    launch_versions = launch_check.get("required_dataset_versions")
    if isinstance(required_versions, list) and isinstance(launch_versions, list):
        if not set(required_versions).issubset(launch_versions):
            errors.append(
                "required dataset versions do not include the preflight requirements"
            )
    return errors


def _recorded_source_binding_error(
    value: Any, source_path: Path, owner_root: Path
) -> str | None:
    if not isinstance(value, str) or not value:
        return "is missing"
    if value.startswith("<redacted:") and value.endswith(">"):
        recorded_name = value[len("<redacted:") : -1]
        return (
            None
            if recorded_name == source_path.name
            else "does not name the supplied source artifact"
        )
    path_error = _recorded_path_safety_error(value, owner_root)
    if path_error is not None:
        return path_error
    raw = Path(value)
    expected = source_path.resolve(strict=False)
    candidates = (
        (owner_root / raw).resolve(strict=False),
        (Path.cwd() / raw).resolve(strict=False),
    )
    if expected not in candidates:
        return "does not resolve to the supplied source artifact"
    return None


def _readiness_errors(
    preflight: dict[str, Any], launch_check: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    for label, value in (
        ("trainer preflight", preflight),
        ("trainer launch check", launch_check),
    ):
        if value.get("passed") is not True:
            errors.append(f"{label} did not pass")
        if value.get("readiness") != "ready":
            errors.append(f"{label} readiness is not ready")
        if value.get("recommendation") != "launch_allowed":
            errors.append(f"{label} recommendation is not launch_allowed")
    validation = (
        launch_check.get("validation")
        if isinstance(launch_check.get("validation"), dict)
        else {}
    )
    if validation.get("passed") is not True or validation.get("error_count") != 0:
        errors.append(
            "trainer launch check did not record a passing preflight validation"
        )
    command = (
        launch_check.get("approved_command")
        if isinstance(launch_check.get("approved_command"), dict)
        else {}
    )
    if not (
        command.get("approved") is True
        and command.get("provided") is True
        and command.get("parseable") is True
        and isinstance(command.get("argv"), list)
        and command.get("argv")
    ):
        errors.append(
            "trainer launch check does not contain an approved parseable command"
        )
    return errors


def _copy_preflight_paths(
    value: Any,
    *,
    role: str,
    base_dir: Path,
    archive_dir: Path,
    archive_root: Path,
    missing: list[dict[str, Any]],
    preserve_paths: bool,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    records: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            missing.append(_missing(role, index, "preflight record is not an object"))
            continue
        name = _record_name(role, item, index)
        source_path, reason = _resolve_recorded_path(item.get("path"), base_dir)
        if source_path is None:
            missing.append(
                _missing(role, index, reason or "path is unavailable", name=name)
            )
            continue
        integrity_error = _record_integrity_error(item, source_path, role)
        if integrity_error is not None:
            missing.append(_missing(role, index, integrity_error, name=name))
            continue
        try:
            copied = _copy_path_artifact(
                name,
                role,
                source_path,
                archive_dir / f"{index:03d}_{_slug(name)}",
                archive_root,
                preserve_paths,
                expected=item,
            )
            if not preserve_paths:
                aliases = _public_source_aliases(str(item["path"]), source_path)
                copied["original_path"] = aliases[0]
                copied["_source_aliases"] = aliases
            records.append(copied)
        except TrainerArchiveError as exc:
            missing.append(_missing(role, index, str(exc), name=name))
    return records


def _copy_preflight_mapping(
    value: Any,
    *,
    role: str,
    base_dir: Path,
    archive_dir: Path,
    archive_root: Path,
    missing: list[dict[str, Any]],
    preserve_paths: bool,
) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    records: list[dict[str, Any]] = []
    for index, (name, item) in enumerate(sorted(value.items())):
        clean_name = str(name or f"{role}_{index}")
        if not isinstance(item, dict):
            missing.append(
                _missing(
                    role, index, "preflight record is not an object", name=clean_name
                )
            )
            continue
        source_path, reason = _resolve_recorded_path(item.get("path"), base_dir)
        if source_path is None:
            missing.append(
                _missing(role, index, reason or "path is unavailable", name=clean_name)
            )
            continue
        integrity_error = _record_integrity_error(item, source_path, role)
        if integrity_error is not None:
            missing.append(_missing(role, index, integrity_error, name=clean_name))
            continue
        try:
            copied = _copy_path_artifact(
                clean_name,
                role,
                source_path,
                archive_dir / f"{index:03d}_{_slug(clean_name)}",
                archive_root,
                preserve_paths,
                expected=item,
            )
            if not preserve_paths:
                aliases = _public_source_aliases(str(item["path"]), source_path)
                copied["original_path"] = aliases[0]
                copied["_source_aliases"] = aliases
            records.append(copied)
        except TrainerArchiveError as exc:
            missing.append(_missing(role, index, str(exc), name=clean_name))
    return records


def _copy_path_artifact(
    name: str,
    role: str,
    source_path: Path,
    archive_path: Path,
    archive_root: Path,
    preserve_paths: bool,
    *,
    expected: dict[str, Any],
) -> dict[str, Any]:
    if source_path.is_dir() and not source_path.is_symlink():
        return _copy_directory_artifact(
            name,
            role,
            source_path,
            archive_path,
            archive_root,
            preserve_paths,
            expected=expected,
        )
    return _copy_file_artifact(
        name,
        role,
        source_path,
        archive_path.with_suffix(_suffix_for(source_path)),
        archive_root,
        preserve_paths,
        expected=expected,
    )


def _copy_file_artifact(
    name: str,
    role: str,
    source_path: Path,
    archive_path: Path,
    archive_root: Path,
    preserve_paths: bool,
    *,
    expected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_error = _copyable_file_error(source_path)
    if source_error is not None:
        raise TrainerArchiveError(f"{role} source {source_error}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, archive_path)
    size_bytes = archive_path.stat().st_size
    sha256 = _sha256(archive_path)
    if expected is not None and (
        size_bytes != expected.get("size_bytes") or sha256 != expected.get("sha256")
    ):
        archive_path.unlink(missing_ok=True)
        raise TrainerArchiveError(f"{role} source changed while being copied")
    payload = _read_json_optional(archive_path)
    return {
        "index": 0,
        "name": name,
        "role": role,
        "kind": "file",
        "path": str(archive_path.relative_to(archive_root)),
        "original_path": _display_path(source_path, preserve_paths),
        "exists": True,
        "schema_version": payload.get("schema_version")
        if isinstance(payload, dict)
        else None,
        "source_passed": payload.get("passed")
        if isinstance(payload, dict) and isinstance(payload.get("passed"), bool)
        else None,
        "size_bytes": size_bytes,
        "sha256": sha256,
    }


def _copy_directory_artifact(
    name: str,
    role: str,
    source_path: Path,
    archive_path: Path,
    archive_root: Path,
    preserve_paths: bool,
    *,
    expected: dict[str, Any],
) -> dict[str, Any]:
    source_error = _copyable_directory_error(source_path)
    if source_error is not None:
        raise TrainerArchiveError(f"{role} source {source_error}")
    if _path_resolves_inside(archive_path, source_path):
        raise TrainerArchiveError(
            f"archive path must not be inside source directory: {archive_path}"
        )
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists() or archive_path.is_symlink():
        raise TrainerArchiveError(
            f"{role} archive destination unexpectedly already exists: {archive_path}"
        )
    _copy_regular_tree(source_path, archive_path)
    tree = _tree_fingerprint(archive_path)
    entry_count = sum(1 for _ in archive_path.iterdir())
    if (
        entry_count != expected.get("entry_count")
        or tree["file_count"] != expected.get("file_count")
        or tree["size_bytes"] != expected.get("size_bytes")
        or tree["sha256"] != expected.get("sha256")
    ):
        raise TrainerArchiveError(f"{role} source changed while being copied")
    return {
        "index": 0,
        "name": name,
        "role": role,
        "kind": "directory",
        "path": str(archive_path.relative_to(archive_root)),
        "original_path": _display_path(source_path, preserve_paths),
        "exists": True,
        "schema_version": None,
        "source_passed": None,
        "size_bytes": tree["size_bytes"],
        "file_count": tree["file_count"],
        "sha256": tree["sha256"],
        "tree_hash_algorithm": _TREE_HASH_ALGORITHM,
    }


def _prepare_archive_dir(
    target: Path, force: bool
) -> _DirectoryAttestation | None:
    if path_has_symlink_component(target, include_leaf=True):
        raise TrainerArchiveError(
            f"trainer archive output must not traverse a symlink component: {target}"
        )
    if target.exists() and not target.is_dir():
        raise TrainerArchiveError(
            f"trainer archive output is not a directory: {target}"
        )
    if not target.exists():
        return None
    if not any(target.iterdir()):
        return _attest_directory(target)
    if not force:
        raise TrainerArchiveError(
            f"trainer archive output is not empty: {target}; pass --force to replace it"
        )
    try:
        assert_safe_output_directory(target, repo_root=_REPO_ROOT)
    except ValueError as exc:
        raise TrainerArchiveError(str(exc)) from exc
    if not _is_existing_trainer_archive(target):
        marker_path = target / _ARCHIVE_MARKER
        if not marker_path.is_file() or marker_path.is_symlink():
            raise TrainerArchiveError(
                f"refusing to replace directory without the trainer archive command marker {_ARCHIVE_MARKER}: {target}"
            )
        raise TrainerArchiveError(
            f"refusing to replace non-archive directory: {target}; choose an empty output directory or an existing trainer archive"
        )
    return _attest_directory(target)


def _is_existing_trainer_archive(target: Path) -> bool:
    marker_path = target / _ARCHIVE_MARKER
    manifest_path = target / _ARCHIVE_MANIFEST
    if target.is_symlink() or marker_path.is_symlink() or manifest_path.is_symlink():
        return False
    if not marker_path.is_file() or not manifest_path.is_file():
        return False
    try:
        if marker_path.read_text(encoding="utf-8") != _ARCHIVE_MARKER_CONTENT:
            return False
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        schema_check = check_schema_contract(value, name_or_id="trainer_archive")
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, SchemaRegistryError):
        return False
    schema_valid = (
        isinstance(value, dict)
        and value.get("schema_version") == TRAINER_ARCHIVE_SCHEMA_VERSION
        and schema_check.get("passed") is True
    )
    if not schema_valid:
        return False
    try:
        _validate_complete_archive(target, "existing trainer archive")
    except TrainerArchiveError:
        return False
    return True


@contextmanager
def _archive_lock(target: Path) -> Iterator[None]:
    """Hold a non-blocking lock shared by every publisher for one target."""
    if path_has_symlink_component(target, include_leaf=True):
        raise TrainerArchiveError(
            f"trainer archive output must not traverse a symlink component: {target}"
        )
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise TrainerArchiveError(
            f"trainer archive output parent could not be created: {target.parent}: {exc}"
        ) from exc
    if path_has_symlink_component(target.parent, include_leaf=True):
        raise TrainerArchiveError(
            f"trainer archive output must not traverse a symlink component: {target}"
        )

    canonical_target = target.resolve(strict=False)
    lock_name = canonical_target.name or "archive"
    lock_path = canonical_target.parent / f".{lock_name}.hfr-trainer-archive.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise TrainerArchiveError(
            f"trainer archive publication lock is unsafe or unavailable: {lock_path}: {exc}"
        ) from exc
    fallback_owner: tuple[Path, tuple[int, int]] | None = None
    flock_module: Any | None = None
    try:
        lock_stat = os.fstat(descriptor)
        if not stat.S_ISREG(lock_stat.st_mode):
            raise TrainerArchiveError(
                f"trainer archive publication lock is not a regular file: {lock_path}"
            )
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        lock_path_stat = lock_path.stat(follow_symlinks=False)
        if (lock_path_stat.st_dev, lock_path_stat.st_ino) != (
            lock_stat.st_dev,
            lock_stat.st_ino,
        ):
            raise TrainerArchiveError(
                f"trainer archive publication lock changed while being opened: {lock_path}"
            )
        try:
            import fcntl as flock_module
        except (
            ImportError
        ):  # pragma: no cover - exercised only on platforms without fcntl
            owner_path = lock_path.with_name(f"{lock_path.name}.owner")
            owner_flags = (
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                owner_descriptor = os.open(owner_path, owner_flags, 0o600)
            except FileExistsError as exc:
                raise TrainerArchiveError(
                    f"trainer archive publication is locked for target: {target}"
                ) from exc
            try:
                owner_stat = os.fstat(owner_descriptor)
                fallback_owner = (
                    owner_path,
                    (owner_stat.st_dev, owner_stat.st_ino),
                )
                os.write(owner_descriptor, f"pid={os.getpid()}\n".encode("ascii"))
                os.fsync(owner_descriptor)
            finally:
                os.close(owner_descriptor)
        else:
            try:
                flock_module.flock(
                    descriptor, flock_module.LOCK_EX | flock_module.LOCK_NB
                )
            except (BlockingIOError, PermissionError) as exc:
                raise TrainerArchiveError(
                    f"trainer archive publication is locked for target: {target}"
                ) from exc
            locked_path_stat = lock_path.stat(follow_symlinks=False)
            if (locked_path_stat.st_dev, locked_path_stat.st_ino) != (
                lock_stat.st_dev,
                lock_stat.st_ino,
            ):
                raise TrainerArchiveError(
                    f"trainer archive publication lock changed while being acquired: {lock_path}"
                )
        yield
    finally:
        if flock_module is not None:
            try:
                flock_module.flock(descriptor, flock_module.LOCK_UN)
            except OSError:
                pass
        os.close(descriptor)
        if fallback_owner is not None:
            _remove_owned_file(*fallback_owner)


def _create_private_work_directory(
    target: Path, purpose: str
) -> tuple[Path, tuple[int, int]]:
    name = target.name or "archive"
    try:
        work_dir = Path(
            tempfile.mkdtemp(prefix=f".{name}.hfr-{purpose}-", dir=target.parent)
        )
        work_dir.chmod(0o700)
        work_stat = work_dir.stat(follow_symlinks=False)
    except OSError as exc:
        raise TrainerArchiveError(
            f"could not create private trainer archive {purpose} directory: {exc}"
        ) from exc
    return work_dir, (work_stat.st_dev, work_stat.st_ino)


def _validate_complete_archive(path: Path, label: str) -> None:
    # Imported lazily because validation imports this module's public schema constant.
    from .validation import validate_trainer_archive

    marker_path = path / _ARCHIVE_MARKER
    try:
        marker_valid = (
            marker_path.is_file()
            and not marker_path.is_symlink()
            and marker_path.read_text(encoding="utf-8") == _ARCHIVE_MARKER_CONTENT
        )
    except (OSError, UnicodeDecodeError):
        marker_valid = False
    if not marker_valid:
        raise TrainerArchiveError(
            f"{label} failed content validation: {_ARCHIVE_MARKER} is missing or invalid"
        )
    validation = validate_trainer_archive(path)
    if not validation.errors:
        return
    detail = "; ".join(validation.errors[:8])
    raise TrainerArchiveError(f"{label} failed content validation: {detail}")


def _fsync_archive_tree(root: Path) -> None:
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        try:
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        except OSError as exc:
            raise TrainerArchiveError(
                f"could not sync staged trainer archive file {path}: {exc}"
            ) from exc
    directories = [root, *(item for item in root.rglob("*") if item.is_dir())]
    for directory in sorted(
        directories, key=lambda item: len(item.parts), reverse=True
    ):
        _fsync_directory(directory)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        # Some supported filesystems do not implement directory fsync.
        pass
    finally:
        os.close(descriptor)


def _publish_staged_archive(
    staging_dir: Path,
    target: Path,
    *,
    staging_attestation: _DirectoryAttestation,
    target_attestation: _DirectoryAttestation | None,
) -> None:
    """Publish a validated directory without making an existing target disappear."""
    _require_attestation(staging_dir, staging_attestation, "staged trainer archive")
    if target_attestation is None:
        if _path_identity(target) is not None:
            raise TrainerArchiveError(
                "trainer archive output appeared after the publication lock was acquired"
            )
        _atomic_rename_new_directory(staging_dir, target)
        try:
            _require_attestation(target, staging_attestation, "published trainer archive")
        except TrainerArchiveError:
            # The exclusive rename guarantees that no pre-existing target was removed.
            # Leave an unexpected publication in place for inspection rather than
            # recursively deleting a path whose contents no longer attest.
            raise
        _fsync_directory(target.parent)
        return

    _require_attestation(target, target_attestation, "existing trainer archive")
    _atomic_exchange_directories(staging_dir, target)
    swapped = True
    try:
        _require_attestation(target, staging_attestation, "published trainer archive")
        _require_attestation(
            staging_dir, target_attestation, "displaced trainer archive"
        )
    except TrainerArchiveError as exc:
        rollback_error: TrainerArchiveError | None = None
        if (
            _path_identity(target) == staging_attestation.identity
            and _path_identity(staging_dir) == target_attestation.identity
        ):
            try:
                _atomic_exchange_directories(staging_dir, target)
                swapped = False
                _fsync_directory(target.parent)
            except TrainerArchiveError as rollback_exc:
                rollback_error = rollback_exc
        detail = f"trainer archive publication attestation failed: {exc}"
        if rollback_error is not None:
            detail += f"; atomic rollback failed: {rollback_error}"
        raise TrainerArchiveError(detail) from exc

    _fsync_directory(target.parent)
    if swapped and not _remove_attested_directory(staging_dir, target_attestation):
        # Publication is already complete and reader-visible. A changed displaced
        # tree is deliberately retained instead of risking deletion of new data.
        raise TrainerArchiveError(
            f"trainer archive published, but the displaced archive changed and was retained at {staging_dir}"
        )
    _fsync_directory(target.parent)


def _attest_directory(
    path: Path, *, expected_identity: tuple[int, int] | None = None
) -> _DirectoryAttestation:
    path_error = _copyable_directory_error(path)
    if path_error is not None:
        raise TrainerArchiveError(path_error)
    path_stat = path.stat(follow_symlinks=False)
    identity = (path_stat.st_dev, path_stat.st_ino)
    if not stat.S_ISDIR(path_stat.st_mode):
        raise TrainerArchiveError(f"trainer archive path is not a directory: {path}")
    if expected_identity is not None and identity != expected_identity:
        raise TrainerArchiveError(
            f"trainer archive directory identity changed while being attested: {path}"
        )
    tree = _publication_tree_fingerprint(path)
    if _path_identity(path) != identity:
        raise TrainerArchiveError(
            f"trainer archive directory identity changed while being attested: {path}"
        )
    return _DirectoryAttestation(
        identity=identity,
        sha256=str(tree["sha256"]),
        file_count=int(tree["file_count"]),
        size_bytes=int(tree["size_bytes"]),
    )


def _publication_tree_fingerprint(root: Path) -> dict[str, Any]:
    """Fingerprint every path and type, including otherwise invisible empty dirs."""
    digest = hashlib.sha256()
    file_count = 0
    size_bytes = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        before = path.stat(follow_symlinks=False)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        if stat.S_ISDIR(before.st_mode):
            digest.update(b"directory\0")
            continue
        if not stat.S_ISREG(before.st_mode):
            raise TrainerArchiveError(
                f"trainer archive contains a non-regular path: {path}"
            )
        file_hash = _sha256(path)
        after = path.stat(follow_symlinks=False)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise TrainerArchiveError(
                f"trainer archive file changed while being attested: {path}"
            )
        digest.update(b"file\0")
        digest.update(str(after.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\0")
        file_count += 1
        size_bytes += after.st_size
    return {
        "sha256": digest.hexdigest(),
        "file_count": file_count,
        "size_bytes": size_bytes,
    }


def _require_attestation(
    path: Path, expected: _DirectoryAttestation, label: str
) -> None:
    try:
        actual = _attest_directory(path, expected_identity=expected.identity)
    except (OSError, TrainerArchiveError) as exc:
        raise TrainerArchiveError(f"{label} changed before publication: {exc}") from exc
    if actual != expected:
        raise TrainerArchiveError(f"{label} contents changed before publication")


def _remove_attested_directory(
    path: Path, expected: _DirectoryAttestation
) -> bool:
    try:
        _require_attestation(path, expected, "displaced trainer archive")
    except TrainerArchiveError:
        return False
    return _remove_owned_directory(path, expected.identity)


def _atomic_rename_new_directory(source: Path, destination: Path) -> None:
    if sys.platform == "darwin":
        _darwin_renamex_np(source, destination, 0x00000004)  # RENAME_EXCL
        return
    if sys.platform.startswith("linux"):
        _linux_renameat2(source, destination, 0x00000001)  # RENAME_NOREPLACE
        return
    raise TrainerArchiveError(
        "atomic exclusive directory publication is unavailable on this platform"
    )


def _atomic_exchange_directories(first: Path, second: Path) -> None:
    if sys.platform == "darwin":
        _darwin_renamex_np(first, second, 0x00000002)  # RENAME_SWAP
        return
    if sys.platform.startswith("linux"):
        _linux_renameat2(first, second, 0x00000002)  # RENAME_EXCHANGE
        return
    raise TrainerArchiveError(
        "atomic directory exchange is unavailable on this platform; refusing to replace an existing archive"
    )


def _darwin_renamex_np(source: Path, destination: Path, flags: int) -> None:
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    renamex_np = getattr(libc, "renamex_np", None)
    if renamex_np is None:
        raise TrainerArchiveError(
            "atomic directory publication is unavailable: renamex_np is missing"
        )
    renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    renamex_np.restype = ctypes.c_int
    result = renamex_np(os.fsencode(source), os.fsencode(destination), flags)
    if result != 0:
        error_number = ctypes.get_errno()
        raise TrainerArchiveError(
            f"atomic directory publication failed: {os.strerror(error_number)}"
        )


def _linux_renameat2(source: Path, destination: Path, flags: int) -> None:
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise TrainerArchiveError(
            "atomic directory publication is unavailable: renameat2 is missing"
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        flags,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise TrainerArchiveError(
            f"atomic directory publication failed: {os.strerror(error_number)}"
        )


def _path_identity(path: Path) -> tuple[int, int] | None:
    try:
        path_stat = path.stat(follow_symlinks=False)
    except OSError:
        return None
    return path_stat.st_dev, path_stat.st_ino


def _remove_owned_directory(path: Path, identity: tuple[int, int]) -> bool:
    try:
        path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return remove_directory_tree_if_identity(path, identity)


def _remove_owned_file(path: Path, identity: tuple[int, int]) -> bool:
    try:
        path_stat = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    if (
        not stat.S_ISREG(path_stat.st_mode)
        or (path_stat.st_dev, path_stat.st_ino) != identity
    ):
        return False
    try:
        path.unlink()
    except OSError:
        return False
    return True


def _read_json_artifact(
    path: Path,
    schema_version: str,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_error = _copyable_file_error(path)
    if source_error is not None:
        raise TrainerArchiveError(f"{label} {source_error}")
    try:
        source_bytes = path.read_bytes()
        value = json.loads(source_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainerArchiveError(
            f"{label} is not readable JSON: {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise TrainerArchiveError(f"{label} must contain a JSON object: {path}")
    if value.get("schema_version") != schema_version:
        raise TrainerArchiveError(
            f"{label} schema_version must be {schema_version!r}; got {value.get('schema_version')!r}"
        )
    return value, {
        "kind": "file",
        "size_bytes": len(source_bytes),
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
    }


def _read_json_optional(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _resolve_recorded_path(
    value: Any, base_dir: Path
) -> tuple[Path | None, str | None]:
    if (
        not isinstance(value, str)
        or not value
        or value.startswith("<redacted:")
        or value.startswith("<missing-")
    ):
        return None, "path is redacted or unavailable"
    safety_error = _recorded_path_safety_error(value, base_dir)
    if safety_error is not None:
        return None, safety_error
    raw = Path(value)
    candidate = base_dir / raw
    source_error = _copyable_path_error(candidate)
    if source_error is None:
        return candidate, None
    if candidate.exists() or candidate.is_symlink():
        return None, source_error
    return None, f"path could not be resolved: {value}"


def _record_integrity_error(
    record: dict[str, Any], source_path: Path, role: str
) -> str | None:
    if record.get("exists") is not True:
        return "preflight record does not attest that the source exists"
    if record.get("symlink") is True:
        return "preflight record marks the source as a symlink"

    expected_kind = (
        "file"
        if role in {"gate", "validation_summary", "schema_contract"}
        else record.get("kind")
    )
    if expected_kind not in {"file", "directory"}:
        return "source kind is missing or invalid in preflight record"
    if expected_kind == "file":
        if role != "gate" and record.get("regular_file") is not True:
            return "preflight record does not attest a regular file"
        if not source_path.is_file() or source_path.is_symlink():
            return "source type does not match the recorded file type"
        expected_sha = record.get("sha256")
        if not _is_sha256(expected_sha):
            return "sha256 is missing or invalid in preflight record"
        expected_size = record.get("size_bytes")
        if not _is_non_negative_int(expected_size):
            return "size_bytes is missing or invalid in preflight record"
        if _sha256(source_path) != expected_sha:
            return "sha256 does not match preflight record"
        if source_path.stat().st_size != expected_size:
            return "size_bytes does not match preflight record"
        return None

    if record.get("regular_directory") is not True:
        return "preflight record does not attest a regular directory"
    if not source_path.is_dir() or source_path.is_symlink():
        return "source type does not match the recorded directory type"
    if record.get("tree_hash_algorithm") != _TREE_HASH_ALGORITHM:
        return "tree_hash_algorithm is missing or invalid in preflight record"
    if not _is_non_negative_int(record.get("entry_count")):
        return "entry_count is missing or invalid in preflight record"
    if not _is_non_negative_int(record.get("file_count")):
        return "file_count is missing or invalid in preflight record"
    if not _is_non_negative_int(record.get("size_bytes")):
        return "size_bytes is missing or invalid in preflight record"
    if not _is_sha256(record.get("sha256")):
        return "sha256 is missing or invalid in preflight record"
    tree = _tree_fingerprint(source_path)
    if tree["sha256"] != record.get("sha256"):
        return "sha256 does not match preflight record"
    if tree["file_count"] != record.get("file_count"):
        return "file_count does not match preflight record"
    if tree["size_bytes"] != record.get("size_bytes"):
        return "size_bytes does not match preflight record"
    return None


def _copyable_path_error(path: Path) -> str | None:
    if path.is_symlink():
        return f"path is a symlink: {path}"
    if path.is_file():
        return _copyable_file_error(path)
    if path.is_dir():
        return _copyable_directory_error(path)
    return f"path not found: {path}"


def _copyable_file_error(path: Path) -> str | None:
    if path_has_symlink_component(path, include_leaf=True):
        return f"file path traverses a symlink component: {path}"
    if path.is_symlink():
        return f"file is a symlink: {path}"
    if not path.exists() or not path.is_file():
        return f"file not found: {path}"
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        return f"file could not be resolved: {path}: {exc}"
    if not resolved.is_file():
        return f"path is not a regular file: {path}"
    return None


def _copyable_directory_error(path: Path) -> str | None:
    if path_has_symlink_component(path, include_leaf=True):
        return f"directory path traverses a symlink component: {path}"
    if path.is_symlink():
        return f"directory is a symlink: {path}"
    if not path.exists() or not path.is_dir():
        return f"directory not found: {path}"
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        return f"directory could not be resolved: {path}: {exc}"
    if not resolved.is_dir():
        return f"path is not a regular directory: {path}"
    for child in path.rglob("*"):
        if child.is_symlink():
            return f"directory contains a symlink: {child}"
        if not child.is_file() and not child.is_dir():
            return f"directory contains a non-regular path: {child}"
    return None


def _copy_regular_tree(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for child in sorted(source.rglob("*")):
        relative = child.relative_to(source)
        destination = target / relative
        if child.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif child.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, destination)
        else:
            raise TrainerArchiveError(f"cannot archive non-regular path: {child}")


def _tree_fingerprint(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    file_count = 0
    size_bytes = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        file_hash = _sha256(path)
        size = path.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\0")
        file_count += 1
        size_bytes += size
    return {
        "sha256": digest.hexdigest(),
        "file_count": file_count,
        "size_bytes": size_bytes,
    }


def _record_name(role: str, item: dict[str, Any], index: int) -> str:
    for key in ("id", "schema_name", "path"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value if key != "path" else f"{role}_{Path(value).name or index}"
    return f"{role}_{index}"


def _missing(
    role: str, index: int, reason: str, *, name: str | None = None
) -> dict[str, Any]:
    item: dict[str, Any] = {"role": role, "index": index, "reason": reason}
    if name:
        item["name"] = name
    return item


def _approved_command_record(launch_check: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(launch_check, dict):
        return {
            "approved": False,
            "provided": False,
            "raw": "",
            "argv": [],
            "parseable": False,
            "shell": "",
        }
    command = launch_check.get("approved_command")
    if not isinstance(command, dict):
        return {
            "approved": False,
            "provided": False,
            "raw": "",
            "argv": [],
            "parseable": False,
            "shell": "",
        }
    argv = command.get("argv") if isinstance(command.get("argv"), list) else []
    clean_argv = [str(item) for item in argv if isinstance(item, str)]
    raw = command.get("raw") if isinstance(command.get("raw"), str) else ""
    shell = command.get("shell") if isinstance(command.get("shell"), str) else ""
    return {
        "approved": command.get("approved") is True,
        "provided": command.get("provided") is True,
        "raw": raw,
        "argv": clean_argv,
        "parseable": command.get("parseable") is True,
        "shell": shell if shell else (shlex.join(clean_argv) if clean_argv else raw),
    }


def _trainer_inputs(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    inputs: list[dict[str, Any]] = []
    for artifact in artifacts:
        if artifact.get("role") != "trainer_artifact":
            continue
        record: dict[str, Any] = {
            "artifact_index": artifact.get("index"),
            "artifact_name": artifact.get("name"),
            "kind": artifact.get("kind"),
            "original_path": artifact.get("original_path"),
            "archive_path": artifact.get("path"),
            "size_bytes": artifact.get("size_bytes"),
            "sha256": artifact.get("sha256"),
        }
        if artifact.get("kind") == "directory":
            record["file_count"] = artifact.get("file_count")
            record["tree_hash_algorithm"] = artifact.get("tree_hash_algorithm")
        inputs.append(record)
    return inputs


def _public_source_aliases(recorded_path: str, source_path: Path) -> list[str]:
    """Return portable aliases without disclosing the resolved source path."""
    aliases = [recorded_path]
    try:
        cwd_relative = os.path.relpath(source_path.resolve(), Path.cwd().resolve())
    except (OSError, ValueError):
        return aliases
    relative_path = Path(cwd_relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        return aliases
    if cwd_relative not in aliases:
        aliases.append(cwd_relative)
    return aliases


def _path_rewrites(
    artifacts: list[dict[str, Any]],
    approved_command: dict[str, Any],
) -> list[dict[str, Any]]:
    rewrites: list[dict[str, Any]] = []
    argv = (
        approved_command.get("argv")
        if isinstance(approved_command.get("argv"), list)
        else []
    )
    command_argv = [item for item in argv if isinstance(item, str)]
    for artifact in artifacts:
        aliases = artifact.pop("_source_aliases", None)
        if artifact.get("role") != "trainer_artifact":
            continue
        originals = (
            aliases if isinstance(aliases, list) else [artifact.get("original_path")]
        )
        archive_path = artifact.get("path")
        if not isinstance(archive_path, str) or not archive_path:
            continue
        for alias_index, original in enumerate(originals):
            if (
                not isinstance(original, str)
                or not original
                or original.startswith("<")
            ):
                continue
            if alias_index > 0 and not _command_references_path(command_argv, original):
                continue
            rewrites.append(
                {
                    "artifact_name": str(artifact.get("name") or ""),
                    "kind": str(artifact.get("kind") or ""),
                    "original_path": original,
                    "archive_path": archive_path,
                }
            )
    return rewrites


def _command_references_path(argv: list[str], path: str) -> bool:
    prefix = path.rstrip("/\\")
    for token in argv:
        values = [token]
        if "=" in token:
            values.append(token.split("=", 1)[1])
        for value in values:
            if (
                value == path
                or value.startswith(prefix + "/")
                or value.startswith(prefix + "\\")
            ):
                return True
    return False


def _portable_command_record(
    approved_command: dict[str, Any], path_rewrites: list[dict[str, Any]]
) -> dict[str, Any]:
    argv = (
        approved_command.get("argv")
        if isinstance(approved_command.get("argv"), list)
        else []
    )
    clean_argv = [item for item in argv if isinstance(item, str)]
    rewritten_argv, rewrite_count = _rewrite_command_argv(clean_argv, path_rewrites)
    return {
        "approved": approved_command.get("approved") is True,
        "available": bool(rewritten_argv),
        "rewritten": rewrite_count > 0,
        "path_rewrite_count": rewrite_count,
        "argv": rewritten_argv,
        "shell": shlex.join(rewritten_argv) if rewritten_argv else "",
        "notes": [
            "Archive-local command is advisory and rewrites only recognized trainer-input paths.",
            "Run it from the trainer archive root or resolve archive_path entries explicitly in your launcher.",
        ],
    }


def _consumer_contract(
    portable_command: dict[str, Any],
    trainer_inputs: list[dict[str, Any]],
    path_rewrites: list[dict[str, Any]],
) -> dict[str, Any]:
    argv = (
        portable_command.get("argv")
        if isinstance(portable_command.get("argv"), list)
        else []
    )
    external_paths = _external_command_paths(
        [item for item in argv if isinstance(item, str)], trainer_inputs
    )
    return {
        "execution_cwd": "archive_root",
        "command_kind": "advisory_portable_command",
        "portable_command_available": portable_command.get("available") is True,
        "portable_command_rewritten": portable_command.get("rewritten") is True,
        "trainer_input_count": len(trainer_inputs),
        "path_rewrite_count": len(path_rewrites),
        "external_code_required": bool(external_paths),
        "external_command_path_count": len(external_paths),
        "external_command_paths": external_paths,
        "notes": [
            "Validate this archive before consuming trainer inputs.",
            "Run portable_command from the archive root or resolve trainer_inputs.archive_path explicitly.",
            "External command paths are not copied by trainer-archive; provide trainer code separately.",
        ],
    }


def _external_command_paths(
    argv: list[str], trainer_inputs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    external: list[dict[str, Any]] = []
    archive_paths = [
        str(item.get("archive_path") or "")
        for item in trainer_inputs
        if isinstance(item.get("archive_path"), str)
    ]
    for index, token in enumerate(argv):
        if not token or token.startswith("-"):
            if "=" in token:
                key, value = token.split("=", 1)
                if _is_external_command_path(value, archive_paths):
                    external.append(
                        {
                            "argv_index": index,
                            "token": token,
                            "path": value,
                            "reason": f"{key} references a path outside archive inputs",
                        }
                    )
            continue
        if _is_external_command_path(token, archive_paths):
            external.append(
                {
                    "argv_index": index,
                    "token": token,
                    "path": token,
                    "reason": "path-like token is not one of the copied trainer inputs",
                }
            )
    return external


def _is_external_command_path(value: str, archive_paths: list[str]) -> bool:
    if not _looks_like_path(value):
        return False
    normalized = value.replace("\\", "/")
    for archive_path in archive_paths:
        clean = archive_path.replace("\\", "/").rstrip("/")
        if normalized == clean or normalized.startswith(clean + "/"):
            return False
    return True


def _looks_like_path(value: str) -> bool:
    if not value or value.startswith("-"):
        return False
    normalized = value.replace("\\", "/")
    if normalized.startswith(("./", "../", "/", "~")) or "/" in normalized:
        return True
    return Path(normalized).suffix.lower() in {
        ".py",
        ".sh",
        ".bash",
        ".js",
        ".mjs",
        ".ts",
        ".ipynb",
        ".json",
        ".jsonl",
        ".yaml",
        ".yml",
        ".toml",
        ".csv",
        ".txt",
    }


def _rewrite_command_argv(
    argv: list[str], path_rewrites: list[dict[str, Any]]
) -> tuple[list[str], int]:
    rewritten: list[str] = []
    rewrite_count = 0
    ordered = sorted(
        path_rewrites,
        key=lambda item: len(str(item.get("original_path") or "")),
        reverse=True,
    )
    for token in argv:
        new_token = _rewrite_command_token(token, ordered)
        if new_token != token:
            rewrite_count += 1
        rewritten.append(new_token)
    return rewritten, rewrite_count


def _rewrite_command_token(token: str, path_rewrites: list[dict[str, Any]]) -> str:
    for item in path_rewrites:
        original = item.get("original_path")
        archive_path = item.get("archive_path")
        if not isinstance(original, str) or not original:
            continue
        if not isinstance(archive_path, str) or not archive_path:
            continue
        replacement = _replace_path_value(token, original, archive_path)
        if replacement != token:
            return replacement
        if "=" in token:
            key, value = token.split("=", 1)
            rewritten_value = _replace_path_value(value, original, archive_path)
            if rewritten_value != value:
                return f"{key}={rewritten_value}"
    return token


def _replace_path_value(value: str, original: str, archive_path: str) -> str:
    if value == original:
        return archive_path
    prefix = original.rstrip("/") + "/"
    if value.startswith(prefix):
        return archive_path.rstrip("/") + "/" + value[len(prefix) :]
    return value


def _metrics(
    artifacts: list[dict[str, Any]],
    missing: list[dict[str, Any]],
    trainer_inputs: list[dict[str, Any]],
    path_rewrites: list[dict[str, Any]],
    consumer_contract: dict[str, Any],
) -> dict[str, Any]:
    role_counts = _count_rows(record.get("role") for record in artifacts)
    missing_role_counts = _count_rows(record.get("role") for record in missing)
    return {
        "artifact_count": len(artifacts),
        "file_artifact_count": sum(
            1 for record in artifacts if record.get("kind") == "file"
        ),
        "directory_artifact_count": sum(
            1 for record in artifacts if record.get("kind") == "directory"
        ),
        "trainer_input_count": len(trainer_inputs),
        "path_rewrite_count": len(path_rewrites),
        "external_command_path_count": _int_value(
            consumer_contract.get("external_command_path_count")
        ),
        "missing_count": len(missing),
        "total_size_bytes": sum(
            _int_value(record.get("size_bytes")) for record in artifacts
        ),
        "role_counts": role_counts,
        "missing_role_counts": missing_role_counts,
        "unique_sha256_count": len(
            {
                record.get("sha256")
                for record in artifacts
                if isinstance(record.get("sha256"), str)
            }
        ),
    }


def _count_rows(values: Any) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return [{"id": key, "count": counts[key]} for key in sorted(counts)]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdefABCDEF" for char in value)
    )


def _is_non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _display_path(path: Path, preserve_paths: bool = False) -> str:
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


def _path_resolves_inside(path: Path, root: Path) -> bool:
    try:
        root_resolved = root.resolve()
        path_resolved = path.resolve(strict=False)
    except OSError:
        return False
    return path_resolved == root_resolved or path_resolved.is_relative_to(root_resolved)


def _suffix_for(path: Path) -> str:
    suffix = path.suffix
    return suffix if suffix else ".artifact"


def _slug(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in value)
    collapsed = "_".join(part for part in cleaned.split("_") if part)
    return collapsed[:120] or "artifact"


def _int_value(value: Any) -> int:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else 0
    )


def _is_windows_absolute(value: str) -> bool:
    normalized = value.replace("/", "\\")
    return (
        len(normalized) >= 3 and normalized[1:3] == ":\\" and normalized[0].isalpha()
    ) or normalized.startswith("\\\\")


def _basename(value: str) -> str:
    return value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] or "path"
