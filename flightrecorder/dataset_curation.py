"""Side-effect-free dataset curation receipts for agentic training loops."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

from .source_contract import inspect_artifact_source

DATASET_CURATION_RECEIPT_SCHEMA_VERSION = "hfr.dataset_curation_receipt.v1"


class DatasetCurationReceiptError(ValueError):
    """Raised when a dataset curation receipt cannot be built."""


def build_dataset_curation_receipt(
    *,
    rejection_sampling_gate_paths: list[str | Path],
    training_export_paths: list[str | Path],
    out_path: str | Path | None = None,
    preserve_paths: bool = False,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a dataset curation handoff receipt without writing dataset rows."""
    output_dir = Path(out_path).parent if out_path else None
    refs = {
        "rejection_sampling_gate": [_artifact_ref(path, "rejection_sampling_gate", preserve_paths, output_dir) for path in rejection_sampling_gate_paths],
        "training_export": [_artifact_ref(path, "training_export", preserve_paths, output_dir) for path in training_export_paths],
    }
    checks: list[dict[str, Any]] = []
    _add_check(
        checks,
        "rejection_sampling_gate_ready",
        _all_present_ready(refs["rejection_sampling_gate"], "hfr.rejection_sampling_gate.v1", "ready_for_dataset_curation"),
        {"gate_count": len(refs["rejection_sampling_gate"]), "passing_count": _passing_ref_count(refs["rejection_sampling_gate"])},
        {"gate_count": ">=1", "all_passed": True, "readiness": "ready_for_dataset_curation"},
    )
    _add_check(
        checks,
        "training_exports_present",
        bool(refs["training_export"]) and all(ref.get("exists") is True for ref in refs["training_export"]),
        {"training_export_count": len(refs["training_export"]), "existing_count": sum(1 for ref in refs["training_export"] if ref.get("exists") is True)},
        {"training_export_count": ">=1", "all_exist": True},
    )
    _add_check(
        checks,
        "flight_recorder_did_not_write_curated_rows",
        True,
        {"curated_rows_written": 0, "accepted_rows_written": 0, "rejected_rows_written": 0, "dataset_registry_updated": False},
        {"curated_rows_written": 0, "accepted_rows_written": 0, "rejected_rows_written": 0, "dataset_registry_updated": False},
    )
    _add_check(
        checks,
        "trainer_handoff_requires_existing_training_gate",
        True,
        {"training_gate_required_before_live_training": True},
        {"training_gate_required_before_live_training": True},
    )
    failed = [check for check in checks if not check["passed"]]
    return {
        "schema_version": DATASET_CURATION_RECEIPT_SCHEMA_VERSION,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "receipt_path": _display_path(Path(out_path), preserve_paths, output_dir) if out_path else "",
        "passed": not failed,
        "readiness": "ready_for_external_trainer_handoff" if not failed else "blocked",
        "recommendation": "run_training_gate_and_trainer_preflight" if not failed else "fix_rejection_sampling_or_training_exports",
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "checks": checks,
        "blocked_reasons": [check["summary"] for check in failed],
        "input_artifacts": refs,
        "curation_summary": {
            "rejection_sampling_gate_count": len(refs["rejection_sampling_gate"]),
            "training_export_count": len(refs["training_export"]),
            "curated_rows_written": 0,
            "accepted_rows_written": 0,
            "rejected_rows_written": 0,
            "dataset_registry_updated": False,
        },
        "trainer_handoff": {
            "dataset_rows_source": "existing_training_exports",
            "allowed_dataset_roles": ["sft", "action_sft", "dpo", "reward_model"],
            "requires_rejection_sampling_gate": True,
            "requires_training_gate_before_live_training": True,
            "requires_trainer_preflight": True,
        },
        "execution_boundary": {
            "receipt_only": True,
            "dataset_rows_written": False,
            "dataset_registry_updated": False,
            "cloud_jobs_started": False,
            "weights_updated_by_flight_recorder": False,
        },
        "notes": [
            "This receipt records dataset curation readiness only; it does not write accepted or rejected rows.",
            "Existing training exports remain the source for downstream training gates and trainer preflights.",
        ],
    }


def write_dataset_curation_receipt(path: str | Path, receipt: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(json.dumps(receipt))
    if isinstance(payload.get("receipt_path"), str):
        payload["receipt_path"] = _output_relative_path(payload.get("receipt_path"), out_path.parent)
    for rows in payload.get("input_artifacts", {}).values():
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    row["path"] = _output_relative_path(row.get("path"), out_path.parent)
                    if row.get("manifest_path"):
                        row["manifest_path"] = _output_relative_path(row.get("manifest_path"), out_path.parent)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_ref(path_value: str | Path, role: str, preserve_paths: bool, output_dir: Path | None = None) -> dict[str, Any]:
    path = Path(path_value)
    manifest_path = path / "manifest.json" if path.is_dir() else path
    displayed_path = _display_path(path, preserve_paths, output_dir)
    public_path = _is_public_dataset_curation_ref_path(displayed_path)
    source = inspect_artifact_source(path, role) if public_path else {"payload": {}, "ready": False, "schema_valid": False}
    exists = public_path and source.get("ready") is True
    manifest_displayed_path = _display_path(manifest_path, preserve_paths, output_dir)
    manifest_source = source.get("manifest") if isinstance(source.get("manifest"), dict) else source
    manifest_exists = (
        _is_public_dataset_curation_ref_path(manifest_displayed_path)
        and manifest_source.get("regular_file") is True
        and manifest_source.get("schema_valid") is True
        and (role != "training_export" or manifest_source.get("semantic_valid") is True)
    )
    payload = source["payload"] if exists and isinstance(source.get("payload"), dict) else {}
    ref = {
        "role": role,
        "path": displayed_path,
        "kind": "directory" if exists and path.is_dir() else "file",
        "exists": exists,
        "sha256": _sha256(path) if exists and path.is_file() else None,
        "size_bytes": path.stat().st_size if exists and path.is_file() else None,
        "schema_version": str(payload.get("schema_version") or ""),
        "passed": payload.get("passed") if isinstance(payload.get("passed"), bool) else None,
        "readiness": str(payload.get("readiness") or ""),
    }
    if exists and path.is_dir():
        ref.update(
            {
                "manifest_path": manifest_displayed_path,
                "manifest_exists": manifest_exists,
                "manifest_sha256": _sha256(manifest_path) if manifest_exists else None,
                "manifest_size_bytes": manifest_path.stat().st_size if manifest_exists else None,
            }
        )
    return ref


def _all_present_ready(refs: list[dict[str, Any]], schema_version: str, readiness: str) -> bool:
    if not refs:
        return False
    for ref in refs:
        if ref.get("exists") is not True or ref.get("schema_version") != schema_version or ref.get("passed") is not True:
            return False
        if readiness and ref.get("readiness") != readiness:
            return False
    return True


def _passing_ref_count(refs: list[dict[str, Any]]) -> int:
    return sum(1 for ref in refs if ref.get("exists") is True and ref.get("passed") is True)


def _display_path(path: Path, preserve_paths: bool, output_dir: Path | None = None) -> str:
    raw = str(path)
    if output_dir is not None:
        try:
            relative = os.path.relpath(path.resolve(), output_dir.resolve())
        except OSError:
            return f"<redacted:{_basename(raw)}>"
        return relative if _is_public_dataset_curation_ref_path(relative) else f"<redacted:{_basename(raw)}>"
    if preserve_paths:
        return raw if _is_public_dataset_curation_ref_path(raw) else f"<redacted:{_basename(raw)}>"
    if not path.is_absolute():
        return raw if _is_public_dataset_curation_ref_path(raw) else f"<redacted:{_basename(raw)}>"
    try:
        relative = str(path.resolve().relative_to(Path.cwd().resolve()))
    except (OSError, ValueError):
        return f"<redacted:{_basename(raw)}>"
    return relative if _is_public_dataset_curation_ref_path(relative) else f"<redacted:{_basename(raw)}>"


def _output_relative_path(value: Any, output_dir: Path) -> Any:
    if not isinstance(value, str) or not value:
        return value
    if value.startswith("<redacted:"):
        return value
    path = Path(value)
    if not path.is_absolute():
        if not _is_public_dataset_curation_ref_path(value):
            return f"<redacted:{_basename(value)}>"
        if not path.exists():
            return value
        path = path.resolve()
    try:
        relative = os.path.relpath(path.resolve(), output_dir.resolve())
    except OSError:
        return f"<redacted:{_basename(value)}>"
    return relative if _is_public_dataset_curation_ref_path(relative) else f"<redacted:{_basename(value)}>"


def _is_public_dataset_curation_ref_path(value: str) -> bool:
    if not value or value.startswith("<redacted:"):
        return False
    path = Path(value)
    windows_path = PureWindowsPath(value)
    return (
        not path.is_absolute()
        and not windows_path.is_absolute()
        and not windows_path.drive
        and "\\" not in value
        and ".." not in path.parts
        and all(not part.startswith("~") for part in path.parts)
    )


def _basename(value: str) -> str:
    return value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] or "path"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _add_check(checks: list[dict[str, Any]], check_id: str, passed: bool, actual: dict[str, Any], expected: dict[str, Any]) -> None:
    checks.append(
        {
            "id": check_id,
            "passed": bool(passed),
            "actual": actual,
            "expected": expected,
            "summary": f"{check_id}: passed={bool(passed)}",
        }
    )
