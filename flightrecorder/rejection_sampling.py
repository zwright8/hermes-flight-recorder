"""Fail-closed rejection-sampling admission gates for agentic loops."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

from .source_contract import inspect_artifact_source

REJECTION_SAMPLING_GATE_SCHEMA_VERSION = "hfr.rejection_sampling_gate.v1"


class RejectionSamplingGateError(ValueError):
    """Raised when a rejection-sampling gate cannot be built."""


def build_rejection_sampling_gate(
    *,
    rollout_receipt_paths: list[str | Path],
    model_grader_gate_paths: list[str | Path],
    review_calibration_paths: list[str | Path],
    reviewed_gate_paths: list[str | Path],
    out_path: str | Path | None = None,
    preserve_paths: bool = False,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic admission gate without selecting or writing rows."""
    output_dir = Path(out_path).parent if out_path else None
    refs = {
        "agentic_rollout_receipt": [
            _artifact_ref(path, "agentic_rollout_receipt", preserve_paths, output_dir) for path in rollout_receipt_paths
        ],
        "model_grader_gate": [_artifact_ref(path, "model_grader_gate", preserve_paths, output_dir) for path in model_grader_gate_paths],
        "review_calibration": [_artifact_ref(path, "review_calibration", preserve_paths, output_dir) for path in review_calibration_paths],
        "reviewed_gate": [_artifact_ref(path, "reviewed_gate", preserve_paths, output_dir) for path in reviewed_gate_paths],
    }
    rollout_refs = refs["agentic_rollout_receipt"]
    mock_rollout_count = sum(_non_negative_int(ref.get("mock_rollout_count")) for ref in rollout_refs)
    checks: list[dict[str, Any]] = []
    _add_check(
        checks,
        "mock_rollout_receipts_present_and_passing",
        _all_present_ready(rollout_refs, "hfr.agentic_rollout_receipt.v1", "mock_rollouts_recorded"),
        {"receipt_count": len(rollout_refs), "passing_count": _passing_ref_count(rollout_refs)},
        {"receipt_count": ">=1", "all_passed": True, "readiness": "mock_rollouts_recorded"},
    )
    _add_check(
        checks,
        "mock_rollouts_available",
        mock_rollout_count > 0,
        {"mock_rollout_count": mock_rollout_count},
        {"mock_rollout_count": ">0"},
    )
    _add_check(
        checks,
        "model_grader_gate_present_and_passing",
        _all_present_ready(refs["model_grader_gate"], "hfr.model_grader_gate.v1", ""),
        {"gate_count": len(refs["model_grader_gate"]), "passing_count": _passing_ref_count(refs["model_grader_gate"])},
        {"gate_count": ">=1", "all_passed": True},
    )
    _add_check(
        checks,
        "review_calibration_present_and_passing",
        _all_present_ready(refs["review_calibration"], "hfr.review_calibration.v1", ""),
        {"calibration_count": len(refs["review_calibration"]), "passing_count": _passing_ref_count(refs["review_calibration"])},
        {"calibration_count": ">=1", "all_passed": True},
    )
    _add_check(
        checks,
        "reviewed_gate_present_and_passing",
        _all_present_ready(refs["reviewed_gate"], "hfr.reviewed_gate.v1", ""),
        {"reviewed_gate_count": len(refs["reviewed_gate"]), "passing_count": _passing_ref_count(refs["reviewed_gate"])},
        {"reviewed_gate_count": ">=1", "all_passed": True},
    )
    _add_check(
        checks,
        "flight_recorder_did_not_write_training_rows",
        True,
        {"dataset_rows_written": False, "accepted_rows_written": 0, "rejected_rows_written": 0},
        {"dataset_rows_written": False, "accepted_rows_written": 0, "rejected_rows_written": 0},
    )
    _add_check(
        checks,
        "only_mock_rollout_inputs_admitted",
        _rollout_boundaries_are_mock_only(rollout_refs),
        {"live_rollouts_started": any(ref.get("live_rollouts_started") is True for ref in rollout_refs)},
        {"live_rollouts_started": False},
    )
    failed = [check for check in checks if not check["passed"]]
    return {
        "schema_version": REJECTION_SAMPLING_GATE_SCHEMA_VERSION,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "gate_path": _display_path(Path(out_path), preserve_paths, output_dir) if out_path else "",
        "passed": not failed,
        "readiness": "ready_for_dataset_curation" if not failed else "blocked",
        "recommendation": "curate_accepted_training_rows" if not failed else "collect_calibrated_reviews_before_sampling",
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "checks": checks,
        "blocked_reasons": [check["summary"] for check in failed],
        "input_artifacts": refs,
        "rollout_summary": {
            "receipt_count": len(rollout_refs),
            "mock_rollout_count": mock_rollout_count,
            "live_rollouts_started": False,
            "dataset_rows_created": False,
        },
        "admission_policy": {
            "requires_mock_rollout_receipt": True,
            "requires_calibrated_review": True,
            "requires_model_grader_gate": True,
            "requires_reviewed_gate": True,
            "accepts_uncalibrated_labels": False,
            "accepted_dataset_roles": ["sft", "action_sft", "dpo", "reward_model"],
        },
        "execution_boundary": {
            "gate_only": True,
            "dataset_rows_written": False,
            "model_provider_calls_started": False,
            "paid_model_grader_calls_started": False,
            "weights_updated_by_flight_recorder": False,
        },
        "notes": [
            "This gate admits reviewed rollout evidence to dataset curation; it does not write dataset rows.",
            "Uncalibrated labels, missing reviewed gates, and live-rollout claims keep rejection sampling blocked.",
        ],
    }


def write_rejection_sampling_gate(path: str | Path, gate: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(json.dumps(gate))
    payload["gate_path"] = _output_relative_path(payload.get("gate_path"), out_path.parent)
    for rows in payload.get("input_artifacts", {}).values():
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    row["path"] = _output_relative_path(row.get("path"), out_path.parent)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_ref(path_value: str | Path, role: str, preserve_paths: bool, output_dir: Path | None = None) -> dict[str, Any]:
    path = Path(path_value)
    displayed_path = _display_path(path, preserve_paths, output_dir)
    public_path = _is_public_rejection_sampling_ref_path(displayed_path)
    source = inspect_artifact_source(path, role) if public_path else {"payload": {}, "ready": False}
    exists = public_path and source.get("ready") is True
    payload = source["payload"] if exists and isinstance(source.get("payload"), dict) else {}
    boundary = payload.get("execution_boundary") if isinstance(payload.get("execution_boundary"), dict) else {}
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
    if role == "agentic_rollout_receipt":
        ref.update(
            {
                "mock_rollout_count": _non_negative_int(payload.get("mock_rollout_count")),
                "mock_receipt_only": boundary.get("mock_receipt_only") is True,
                "live_rollouts_started": boundary.get("live_rollouts_started") is True,
                "dataset_rows_written": boundary.get("dataset_rows_written") is True,
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


def _rollout_boundaries_are_mock_only(refs: list[dict[str, Any]]) -> bool:
    return bool(refs) and all(
        ref.get("mock_receipt_only") is True
        and ref.get("live_rollouts_started") is not True
        and ref.get("dataset_rows_written") is not True
        for ref in refs
    )


def _display_path(path: Path, preserve_paths: bool, output_dir: Path | None = None) -> str:
    raw = str(path)
    if output_dir is not None:
        try:
            relative = os.path.relpath(path.resolve(), output_dir.resolve())
        except OSError:
            return f"<redacted:{_basename(raw)}>"
        return relative if _is_public_rejection_sampling_ref_path(relative) else f"<redacted:{_basename(raw)}>"
    if preserve_paths:
        return raw if _is_public_rejection_sampling_ref_path(raw) else f"<redacted:{_basename(raw)}>"
    if not path.is_absolute():
        return raw if _is_public_rejection_sampling_ref_path(raw) else f"<redacted:{_basename(raw)}>"
    try:
        relative = str(path.resolve().relative_to(Path.cwd().resolve()))
    except (OSError, ValueError):
        return f"<redacted:{_basename(raw)}>"
    return relative if _is_public_rejection_sampling_ref_path(relative) else f"<redacted:{_basename(raw)}>"


def _output_relative_path(value: Any, output_dir: Path) -> Any:
    if not isinstance(value, str) or not value:
        return value
    if value.startswith("<redacted:"):
        return value
    path = Path(value)
    if not path.is_absolute():
        if not _is_public_rejection_sampling_ref_path(value):
            return f"<redacted:{_basename(value)}>"
        if not path.exists():
            return value
        path = path.resolve()
    try:
        relative = os.path.relpath(path.resolve(), output_dir.resolve())
    except OSError:
        return f"<redacted:{_basename(value)}>"
    return relative if _is_public_rejection_sampling_ref_path(relative) else f"<redacted:{_basename(value)}>"


def _is_public_rejection_sampling_ref_path(value: str) -> bool:
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


def _non_negative_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


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
