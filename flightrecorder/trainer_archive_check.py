"""Consumer-side readiness checks for portable trainer archives."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .trainer_archive import TRAINER_ARCHIVE_SCHEMA_VERSION

TRAINER_ARCHIVE_CHECK_SCHEMA_VERSION = "hfr.trainer_archive_check.v1"


class TrainerArchiveCheckError(ValueError):
    """Raised when a trainer archive readiness check cannot be built."""


def build_trainer_archive_check(
    *,
    archive_path: str | Path,
    external_code_root: str | Path = ".",
    validation_summary: dict[str, Any] | None = None,
    preserve_paths: bool = False,
) -> dict[str, Any]:
    """Build a side-effect-free consumer readiness check for a trainer archive."""
    manifest_path = _archive_manifest_path(Path(archive_path))
    archive_root = manifest_path.parent
    archive = _read_archive_manifest(manifest_path)
    root = Path(external_code_root)
    checks: list[dict[str, Any]] = []

    validation = _validation_record(validation_summary)
    archive_record = _archive_record(archive, archive_root, manifest_path, preserve_paths)
    external_root = _external_code_root_record(root, preserve_paths)
    consumer_contract = archive.get("consumer_contract") if isinstance(archive.get("consumer_contract"), dict) else {}
    portable_command = archive.get("portable_command") if isinstance(archive.get("portable_command"), dict) else {}
    trainer_inputs = archive.get("trainer_inputs") if isinstance(archive.get("trainer_inputs"), list) else []

    _add_bool_check(checks, "archive_validation_passed", validation["passed"], {"archive": archive_record["manifest_path"]})
    _add_bool_check(checks, "archive_handoff_ready", archive.get("passed") is True, {"archive": archive_record["manifest_path"]})
    _add_bool_check(
        checks,
        "portable_command_approved",
        portable_command.get("approved") is True,
        {"archive": archive_record["manifest_path"]},
    )
    _add_bool_check(
        checks,
        "portable_command_available",
        bool(portable_command.get("available") is True and _string_list(portable_command.get("argv"))),
        {"archive": archive_record["manifest_path"]},
    )

    external_specs = consumer_contract.get("external_command_paths") if isinstance(consumer_contract, dict) else []
    if not isinstance(external_specs, list):
        external_specs = []
    relative_external_count = sum(1 for item in external_specs if _uses_external_root(item))
    if relative_external_count:
        _add_bool_check(
            checks,
            "external_code_root_usable",
            external_root["regular_directory"],
            {"external_code_root": external_root["path"]},
        )
    external_code_checks = [
        _external_code_check(index, item, root, preserve_paths) for index, item in enumerate(external_specs)
    ]
    for item in external_code_checks:
        _add_bool_check(
            checks,
            "external_command_path_available",
            item["passed"],
            {"path": item["path"], "resolved_path": item["resolved_path"], "reason": item["reason"]},
        )

    trainer_input_checks = [
        _trainer_input_check(index, item, archive_root, preserve_paths) for index, item in enumerate(trainer_inputs)
    ]
    for item in trainer_input_checks:
        _add_bool_check(
            checks,
            "trainer_input_available",
            item["passed"],
            {"archive_path": item["archive_path"], "reason": item["reason"]},
        )

    metrics = _metrics(
        validation,
        external_code_checks,
        trainer_input_checks,
        relative_external_count,
    )
    _add_bool_check(
        checks,
        "external_code_requirements_satisfied",
        metrics["missing_external_code_count"] == 0,
        {
            "external_command_path_count": str(metrics["external_command_path_count"]),
            "missing_external_code_count": str(metrics["missing_external_code_count"]),
        },
    )
    _add_bool_check(
        checks,
        "trainer_inputs_available",
        metrics["trainer_input_count"] == metrics["trainer_input_available_count"],
        {
            "trainer_input_count": str(metrics["trainer_input_count"]),
            "trainer_input_available_count": str(metrics["trainer_input_available_count"]),
        },
    )

    failed_checks = sum(1 for check in checks if check.get("passed") is False)
    passed = failed_checks == 0
    metrics["check_count"] = len(checks)
    metrics["failed_check_count"] = failed_checks
    return {
        "schema_version": TRAINER_ARCHIVE_CHECK_SCHEMA_VERSION,
        "archive_path": _display_path(Path(archive_path), preserve_paths),
        "manifest_path": archive_record["manifest_path"],
        "passed": passed,
        "readiness": "ready" if passed else "blocked",
        "recommendation": "consumer_ready" if passed else "block_consumer_launch",
        "check_count": len(checks),
        "failed_check_count": failed_checks,
        "checks": checks,
        "validation": validation,
        "archive": archive_record,
        "external_code_root": external_root,
        "portable_command": _portable_command_record(portable_command),
        "consumer_contract": _consumer_contract_record(consumer_contract),
        "external_code_checks": external_code_checks,
        "trainer_input_checks": trainer_input_checks,
        "metrics": metrics,
        "notes": [
            "Trainer archive checks validate a portable handoff and caller-provided trainer code paths; they do not execute training.",
            "External command paths remain outside the archive on purpose so Flight Recorder stays an evidence layer, not a trainer or sandbox.",
            "Run the approved portable command only from a separate trainer wrapper after this artifact reports consumer_ready.",
        ],
    }


def _archive_manifest_path(path: Path) -> Path:
    return path / "trainer_archive.json" if path.is_dir() else path


def _read_archive_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TrainerArchiveCheckError(f"trainer archive manifest not found: {path}") from exc
    if not isinstance(payload, dict):
        raise TrainerArchiveCheckError(f"trainer archive manifest must contain a JSON object: {path}")
    if payload.get("schema_version") != TRAINER_ARCHIVE_SCHEMA_VERSION:
        raise TrainerArchiveCheckError(
            f"trainer archive schema_version must be {TRAINER_ARCHIVE_SCHEMA_VERSION!r}; "
            f"got {payload.get('schema_version')!r}"
        )
    return payload


def _archive_record(archive: dict[str, Any], archive_root: Path, manifest_path: Path, preserve_paths: bool) -> dict[str, Any]:
    return {
        "path": _display_path(archive_root, preserve_paths),
        "manifest_path": _display_path(manifest_path, preserve_paths),
        "schema_version": str(archive.get("schema_version") or ""),
        "passed": archive.get("passed") is True,
        "self_contained": archive.get("self_contained") is True,
        "ready_for_training": archive.get("ready_for_training") is True,
        "trainer_input_count": _int_value(
            (archive.get("metrics") if isinstance(archive.get("metrics"), dict) else {}).get("trainer_input_count")
        ),
        "external_command_path_count": _int_value(
            (archive.get("consumer_contract") if isinstance(archive.get("consumer_contract"), dict) else {}).get(
                "external_command_path_count"
            )
        ),
    }


def _external_code_root_record(path: Path, preserve_paths: bool) -> dict[str, Any]:
    return {
        "path": _display_path(path, preserve_paths),
        "exists": path.exists(),
        "kind": "directory",
        "regular_directory": path.exists() and path.is_dir() and not path.is_symlink(),
        "symlink": path.is_symlink(),
    }


def _portable_command_record(value: Any) -> dict[str, Any]:
    command = value if isinstance(value, dict) else {}
    argv = _string_list(command.get("argv"))
    return {
        "approved": command.get("approved") is True,
        "available": command.get("available") is True,
        "rewritten": command.get("rewritten") is True,
        "path_rewrite_count": _int_value(command.get("path_rewrite_count")),
        "argv": argv,
        "shell": str(command.get("shell") or ""),
    }


def _consumer_contract_record(value: Any) -> dict[str, Any]:
    contract = value if isinstance(value, dict) else {}
    external_paths = contract.get("external_command_paths") if isinstance(contract.get("external_command_paths"), list) else []
    return {
        "execution_cwd": str(contract.get("execution_cwd") or ""),
        "command_kind": str(contract.get("command_kind") or ""),
        "portable_command_available": contract.get("portable_command_available") is True,
        "trainer_input_count": _int_value(contract.get("trainer_input_count")),
        "path_rewrite_count": _int_value(contract.get("path_rewrite_count")),
        "external_code_required": contract.get("external_code_required") is True,
        "external_command_path_count": _int_value(contract.get("external_command_path_count")),
        "external_command_paths": [
            {
                "argv_index": _int_value(item.get("argv_index")) if isinstance(item, dict) else 0,
                "token": str(item.get("token") or "") if isinstance(item, dict) else "",
                "path": str(item.get("path") or "") if isinstance(item, dict) else "",
                "reason": str(item.get("reason") or "") if isinstance(item, dict) else "",
            }
            for item in external_paths
            if isinstance(item, dict)
        ],
    }


def _validation_record(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {
            "available": False,
            "passed": False,
            "strict": False,
            "target_count": 0,
            "error_count": 1,
            "warning_count": 0,
            "errors": ["archive validation summary was not provided"],
            "warnings": [],
        }
    errors: list[str] = []
    warnings: list[str] = []
    for target in summary.get("targets", []) if isinstance(summary.get("targets"), list) else []:
        if not isinstance(target, dict):
            continue
        errors.extend(str(error) for error in target.get("errors", []) if isinstance(error, str))
        warnings.extend(str(warning) for warning in target.get("warnings", []) if isinstance(warning, str))
    return {
        "available": True,
        "passed": summary.get("passed") is True,
        "strict": summary.get("strict") is True,
        "target_count": _int_value(summary.get("target_count")),
        "error_count": _int_value(summary.get("error_count")),
        "warning_count": _int_value(summary.get("warning_count")),
        "errors": errors[:20],
        "warnings": warnings[:20],
    }


def _uses_external_root(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    value = item.get("path")
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    return not (path.is_absolute() or value.startswith("~") or _is_windows_absolute(value))


def _external_code_check(index: int, item: Any, external_code_root: Path, preserve_paths: bool) -> dict[str, Any]:
    spec = item if isinstance(item, dict) else {}
    raw_path = str(spec.get("path") or "")
    resolved_path, reason = _resolve_external_path(raw_path, external_code_root)
    exists = resolved_path.exists() if resolved_path is not None else False
    symlink = resolved_path.is_symlink() if resolved_path is not None else False
    regular_file = exists and resolved_path.is_file() and not symlink
    record: dict[str, Any] = {
        "index": index,
        "argv_index": _int_value(spec.get("argv_index")),
        "token": str(spec.get("token") or ""),
        "path": raw_path,
        "resolved_path": _display_path(resolved_path, preserve_paths) if resolved_path is not None else "",
        "exists": exists,
        "kind": "file",
        "regular_file": regular_file,
        "symlink": symlink,
        "passed": regular_file and reason is None,
        "reason": reason or ("external command path is available" if regular_file else "path is not a regular file"),
    }
    if regular_file and resolved_path is not None:
        record["size_bytes"] = resolved_path.stat().st_size
        record["sha256"] = _sha256(resolved_path)
    return record


def _resolve_external_path(value: str, external_code_root: Path) -> tuple[Path | None, str | None]:
    if not value:
        return None, "external command path is empty"
    if value.startswith("<"):
        return None, "external command path is redacted or unavailable"
    if _is_windows_absolute(value):
        return None, "windows absolute paths cannot be verified on this platform"
    raw = Path(value)
    if raw.is_absolute():
        return raw, None
    if value.startswith("~"):
        return raw.expanduser(), None
    if ".." in raw.parts:
        return None, "parent traversal is not allowed in external command paths"
    return external_code_root / raw, None


def _trainer_input_check(index: int, item: Any, archive_root: Path, preserve_paths: bool) -> dict[str, Any]:
    spec = item if isinstance(item, dict) else {}
    archive_path = str(spec.get("archive_path") or "")
    kind = str(spec.get("kind") or "")
    resolved_path, reason = _resolve_archive_path(archive_path, archive_root)
    exists = resolved_path.exists() if resolved_path is not None else False
    symlink = resolved_path.is_symlink() if resolved_path is not None else False
    regular_file = exists and resolved_path.is_file() and not symlink
    regular_directory = exists and resolved_path.is_dir() and not symlink
    passed = False
    record: dict[str, Any] = {
        "index": index,
        "artifact_index": _int_value(spec.get("artifact_index")),
        "artifact_name": str(spec.get("artifact_name") or ""),
        "archive_path": archive_path,
        "resolved_path": _display_path(resolved_path, preserve_paths) if resolved_path is not None else "",
        "kind": kind,
        "exists": exists,
        "regular_file": regular_file,
        "regular_directory": regular_directory,
        "symlink": symlink,
        "expected_sha256": str(spec.get("sha256") or ""),
        "expected_size_bytes": _int_value(spec.get("size_bytes")),
        "passed": False,
        "reason": reason or "",
    }
    if kind == "directory":
        record["expected_file_count"] = _int_value(spec.get("file_count"))
    if reason is not None:
        return record
    if kind == "file" and regular_file and resolved_path is not None:
        size = resolved_path.stat().st_size
        digest = _sha256(resolved_path)
        passed = size == _int_value(spec.get("size_bytes")) and digest == spec.get("sha256")
        record.update({"size_bytes": size, "sha256": digest})
    elif kind == "directory" and regular_directory and resolved_path is not None:
        symlink_child = next((child for child in resolved_path.rglob("*") if child.is_symlink()), None)
        if symlink_child is not None:
            record["reason"] = f"archive input directory contains a symlink: {symlink_child}"
            return record
        tree = _tree_fingerprint(resolved_path)
        passed = (
            tree["size_bytes"] == _int_value(spec.get("size_bytes"))
            and tree["file_count"] == _int_value(spec.get("file_count"))
            and tree["sha256"] == spec.get("sha256")
        )
        record.update(tree)
    else:
        record["reason"] = f"archive input is not a regular {kind or 'path'}"
    if passed:
        record["reason"] = "archive trainer input is available and hash-matched"
    elif not record["reason"]:
        record["reason"] = "archive trainer input hash or size does not match"
    record["passed"] = passed
    return record


def _resolve_archive_path(value: str, archive_root: Path) -> tuple[Path | None, str | None]:
    if not value:
        return None, "archive path is empty"
    if value.startswith("<"):
        return None, "archive path is redacted or unavailable"
    raw = Path(value)
    if raw.is_absolute() or ".." in raw.parts:
        return None, "archive path must be relative and stay inside the archive"
    resolved = archive_root / raw
    if not _path_resolves_inside(resolved, archive_root):
        return None, "archive path resolves outside the archive"
    return resolved, None


def _metrics(
    validation: dict[str, Any],
    external_code_checks: list[dict[str, Any]],
    trainer_input_checks: list[dict[str, Any]],
    relative_external_count: int,
) -> dict[str, Any]:
    external_count = len(external_code_checks)
    external_available = sum(1 for item in external_code_checks if item.get("passed") is True)
    trainer_input_count = len(trainer_input_checks)
    trainer_inputs_available = sum(1 for item in trainer_input_checks if item.get("passed") is True)
    return {
        "archive_validation_passed": validation.get("passed") is True,
        "archive_validation_error_count": _int_value(validation.get("error_count")),
        "archive_validation_warning_count": _int_value(validation.get("warning_count")),
        "external_command_path_count": external_count,
        "relative_external_command_path_count": relative_external_count,
        "external_code_file_count": external_available,
        "missing_external_code_count": external_count - external_available,
        "trainer_input_count": trainer_input_count,
        "trainer_input_available_count": trainer_inputs_available,
        "missing_trainer_input_count": trainer_input_count - trainer_inputs_available,
    }


def _add_bool_check(checks: list[dict[str, Any]], check_id: str, passed: bool, scope: dict[str, str]) -> None:
    checks.append(
        {
            "id": check_id,
            "passed": passed,
            "actual": {"passed": passed},
            "expected": {"passed": True},
            "scope": scope,
            "summary": f"{check_id}: passed={passed}",
        }
    )


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
    return {"sha256": digest.hexdigest(), "file_count": file_count, "size_bytes": size_bytes}


def _path_resolves_inside(path: Path, root: Path) -> bool:
    try:
        root_resolved = root.resolve()
        path_resolved = path.resolve(strict=False)
    except OSError:
        return False
    return path_resolved == root_resolved or path_resolved.is_relative_to(root_resolved)


def _string_list(value: Any) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _int_value(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _is_windows_absolute(value: str) -> bool:
    normalized = value.replace("/", "\\")
    return (len(normalized) >= 3 and normalized[1:3] == ":\\" and normalized[0].isalpha()) or normalized.startswith("\\\\")


def _basename(value: str) -> str:
    return value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] or "path"
