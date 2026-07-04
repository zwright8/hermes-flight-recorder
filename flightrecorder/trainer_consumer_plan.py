"""Side-effect-free trainer consumer plans over archive readiness checks."""

from __future__ import annotations

import hashlib
import shlex
from pathlib import Path
from typing import Any

from .path_safety import path_has_symlink_component as _path_has_symlink_component
from .trainer_archive_check import TRAINER_ARCHIVE_CHECK_SCHEMA_VERSION

TRAINER_CONSUMER_PLAN_SCHEMA_VERSION = "hfr.trainer_consumer_plan.v1"


class TrainerConsumerPlanError(ValueError):
    """Raised when a trainer consumer plan cannot be built."""


def build_trainer_consumer_plan(
    *,
    out_path: str | Path,
    archive_check_path: str | Path,
    archive_check: dict[str, Any],
    validation_summary: dict[str, Any] | None = None,
    preserve_paths: bool = False,
) -> dict[str, Any]:
    """Build a deterministic trainer handoff plan without executing training."""
    reject_symlinked_archive_check_input(Path(archive_check_path))
    if not isinstance(archive_check, dict):
        raise TrainerConsumerPlanError(f"trainer archive check must contain a JSON object: {archive_check_path}")
    if archive_check.get("schema_version") != TRAINER_ARCHIVE_CHECK_SCHEMA_VERSION:
        raise TrainerConsumerPlanError(
            f"trainer archive check schema_version must be {TRAINER_ARCHIVE_CHECK_SCHEMA_VERSION!r}; "
            f"got {archive_check.get('schema_version')!r}"
        )

    checks: list[dict[str, Any]] = []
    validation = _validation_record(validation_summary)
    source = _source_record(Path(archive_check_path), archive_check, preserve_paths)
    portable_command = archive_check.get("portable_command") if isinstance(archive_check.get("portable_command"), dict) else {}
    archive = archive_check.get("archive") if isinstance(archive_check.get("archive"), dict) else {}
    external_root = archive_check.get("external_code_root") if isinstance(archive_check.get("external_code_root"), dict) else {}
    external_code_files = _external_code_files(archive_check.get("external_code_checks"))
    trainer_inputs = _trainer_inputs(archive_check.get("trainer_input_checks"))

    _add_bool_check(checks, "archive_check_validation_passed", validation["passed"], {"archive_check": source["path"]})
    _add_bool_check(checks, "archive_check_passed", archive_check.get("passed") is True, {"archive_check": source["path"]})
    _add_bool_check(
        checks,
        "archive_check_consumer_ready",
        archive_check.get("recommendation") == "consumer_ready",
        {"recommendation": str(archive_check.get("recommendation") or "")},
    )
    _add_bool_check(
        checks,
        "portable_command_ready",
        bool(
            portable_command.get("approved") is True
            and portable_command.get("available") is True
            and _string_list(portable_command.get("argv"))
        ),
        {"archive_check": source["path"]},
    )
    _add_bool_check(
        checks,
        "external_code_ready",
        all(item.get("passed") is True for item in external_code_files),
        {"external_code_file_count": str(len(external_code_files))},
    )
    _add_bool_check(
        checks,
        "trainer_inputs_ready",
        bool(trainer_inputs) and all(item.get("passed") is True for item in trainer_inputs),
        {"trainer_input_count": str(len(trainer_inputs))},
    )
    _add_bool_check(checks, "no_execution_performed", True, {"boundary": "flightrecorder"})

    failed_checks = sum(1 for check in checks if check.get("passed") is False)
    passed = failed_checks == 0
    argv = _string_list(portable_command.get("argv"))
    metrics = {
        "check_count": len(checks),
        "failed_check_count": failed_checks,
        "command_arg_count": len(argv),
        "trainer_input_count": len(trainer_inputs),
        "trainer_input_ready_count": sum(1 for item in trainer_inputs if item.get("passed") is True),
        "external_code_file_count": len(external_code_files),
        "external_code_ready_count": sum(1 for item in external_code_files if item.get("passed") is True),
        "archive_check_error_count": _int_value(validation.get("error_count")),
        "archive_check_warning_count": _int_value(validation.get("warning_count")),
    }
    return {
        "schema_version": TRAINER_CONSUMER_PLAN_SCHEMA_VERSION,
        "plan_path": _display_path(Path(out_path), preserve_paths),
        "archive_check_path": source["path"],
        "passed": passed,
        "readiness": "ready" if passed else "blocked",
        "recommendation": "ready_for_external_trainer" if passed else "block_external_trainer",
        "check_count": len(checks),
        "failed_check_count": failed_checks,
        "checks": checks,
        "validation": validation,
        "source_archive_check": source,
        "execution": {
            "execution_cwd": "archive_root",
            "archive_root": _public_root_path(archive.get("path")),
            "external_code_root": _public_root_path(external_root.get("path")),
            "command_approved": portable_command.get("approved") is True,
            "command_available": portable_command.get("available") is True,
            "command_argv": argv,
            "command_shell": shlex.join(argv) if argv else "",
            "external_code_files": external_code_files,
            "trainer_inputs": trainer_inputs,
        },
        "handoff_contract": {
            "flight_recorder_executed_command": False,
            "runner_owns_execution": True,
            "runner_must_run_from": "archive_root",
            "runner_must_require_recommendation": "ready_for_external_trainer",
            "trainer_input_count": len(trainer_inputs),
            "external_code_file_count": len(external_code_files),
            "allowed_input_sets": ["execution.trainer_inputs", "execution.external_code_files"],
            "notes": [
                "This plan is a launch contract for an external trainer wrapper; Flight Recorder does not execute it.",
                "command_argv is canonical; command_shell is a derived display rendering.",
                "The wrapper should verify this plan and then resolve command_argv relative to archive_root plus external_code_root.",
            ],
        },
        "blocked_reasons": [str(check.get("summary")) for check in checks if check.get("passed") is False],
        "metrics": metrics,
        "notes": [
            "Trainer consumer plans are deterministic handoff manifests, not trainer launchers.",
            "They make the exact command, archive inputs, external code files, and non-execution boundary machine-readable.",
            "Local path integrity is rechecked when resolved paths are visible; redacted paths retain recorded fingerprint evidence.",
        ],
    }


def _source_record(path: Path, archive_check: dict[str, Any], preserve_paths: bool) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": _display_path(path, preserve_paths),
        "schema_version": str(archive_check.get("schema_version") or ""),
        "passed": archive_check.get("passed") is True,
        "readiness": str(archive_check.get("readiness") or ""),
        "recommendation": str(archive_check.get("recommendation") or ""),
    }
    if (
        path.exists()
        and path.is_file()
        and not path.is_symlink()
        and not _path_has_symlink_component(path, include_leaf=False)
    ):
        record["size_bytes"] = path.stat().st_size
        record["sha256"] = _sha256(path)
    return record


def reject_symlinked_archive_check_input(path: Path) -> None:
    if path.is_symlink() or _path_has_symlink_component(path, include_leaf=False):
        raise TrainerConsumerPlanError(f"trainer_consumer_plan.archive_check_path must not traverse symlinked components: {path}")


def _external_code_files(value: Any) -> list[dict[str, Any]]:
    rows = value if isinstance(value, list) else []
    records: list[dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        record: dict[str, Any] = {
            "index": _int_value(item.get("index")),
            "argv_index": _int_value(item.get("argv_index")),
            "token": str(item.get("token") or ""),
            "path": str(item.get("path") or ""),
            "resolved_path": str(item.get("resolved_path") or ""),
            "exists": item.get("exists") is True,
            "regular_file": item.get("regular_file") is True,
            "symlink": item.get("symlink") is True,
            "passed": item.get("passed") is True,
            "reason": str(item.get("reason") or ""),
        }
        if isinstance(item.get("size_bytes"), int) and not isinstance(item.get("size_bytes"), bool):
            record["size_bytes"] = item["size_bytes"]
        if isinstance(item.get("sha256"), str):
            record["sha256"] = item["sha256"]
        records.append(record)
    return records


def _trainer_inputs(value: Any) -> list[dict[str, Any]]:
    rows = value if isinstance(value, list) else []
    records: list[dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        record: dict[str, Any] = {
            "index": _int_value(item.get("index")),
            "artifact_index": _int_value(item.get("artifact_index")),
            "artifact_name": str(item.get("artifact_name") or ""),
            "archive_path": str(item.get("archive_path") or ""),
            "resolved_path": str(item.get("resolved_path") or ""),
            "kind": str(item.get("kind") or ""),
            "exists": item.get("exists") is True,
            "regular_file": item.get("regular_file") is True,
            "regular_directory": item.get("regular_directory") is True,
            "symlink": item.get("symlink") is True,
            "expected_sha256": str(item.get("expected_sha256") or ""),
            "passed": item.get("passed") is True,
            "reason": str(item.get("reason") or ""),
        }
        for field_name in ("size_bytes", "file_count", "expected_size_bytes", "expected_file_count"):
            if isinstance(item.get(field_name), int) and not isinstance(item.get(field_name), bool):
                record[field_name] = item[field_name]
        if isinstance(item.get("sha256"), str):
            record["sha256"] = item["sha256"]
        records.append(record)
    return records


def _validation_record(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {
            "available": False,
            "passed": False,
            "strict": False,
            "target_count": 0,
            "error_count": 1,
            "warning_count": 0,
            "errors": ["archive-check validation summary was not provided"],
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


def _public_root_path(value: Any) -> str:
    raw = str(value or "")
    if not raw:
        return ""
    if raw.startswith("<redacted:") and raw.endswith(">"):
        return raw
    if _is_windows_absolute(raw):
        return f"<redacted:{_basename(raw)}>"
    path = Path(raw)
    if path.is_absolute():
        return f"<redacted:{path.name or 'path'}>"
    return raw


def _is_windows_absolute(value: str) -> bool:
    normalized = value.replace("/", "\\")
    return (len(normalized) >= 3 and normalized[1:3] == ":\\" and normalized[0].isalpha()) or normalized.startswith("\\\\")


def _basename(value: str) -> str:
    return value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] or "path"
