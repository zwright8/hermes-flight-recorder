"""Fail-closed rubric and model-grader contracts."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .path_safety import path_has_symlink_component as _path_has_symlink_component
from .review import REVIEW_LABELS, review_item_sha256
from .schema_registry import SchemaRegistryError, check_schema_file

RUBRIC_SPEC_SCHEMA_VERSION = "hfr.rubric_spec.v1"
MODEL_GRADER_DRY_RUN_SCHEMA_VERSION = "hfr.model_grader_dry_run.v1"
MODEL_GRADER_DISAGREEMENT_QUEUE_SCHEMA_VERSION = "hfr.model_grader_disagreement_queue.v1"
MODEL_GRADER_OVERRIDE_RECEIPT_SCHEMA_VERSION = "hfr.model_grader_override_receipt.v1"
MODEL_GRADER_GATE_SCHEMA_VERSION = "hfr.model_grader_gate.v1"


class ModelGraderError(ValueError):
    """Raised when model-grader control-plane artifacts cannot be built."""


def build_rubric_spec(
    *,
    review_export_dir: str | Path,
    rubric_id: str,
    criteria: list[str] | None = None,
    created_at: str | None = None,
    preserve_paths: bool = False,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a schema-checkable rubric bound to a human-review queue."""
    review_dir = Path(review_export_dir)
    output_path = Path(out_path) if out_path is not None else None
    _require_evidence_dir(review_dir, "review export")
    items = _read_review_items(review_dir)
    criterion_rows = _criterion_rows(criteria)
    item_fingerprints = [_item_fingerprint(row) for row in items]
    return {
        "schema_version": RUBRIC_SPEC_SCHEMA_VERSION,
        "rubric_id": rubric_id,
        "created_at": created_at or _now(),
        "review_export": _review_export_ref(review_dir, preserve_paths, output_path),
        "criterion_count": len(criterion_rows),
        "criteria": criterion_rows,
        "label_options": list(REVIEW_LABELS),
        "calibration_requirements": {
            "required_before_training_admission": True,
            "min_calibration_agreement_rate": 0.8,
            "max_uncalibrated_labels_admitted": 0,
            "human_override_queue_required": True,
        },
        "review_item_count": len(item_fingerprints),
        "review_item_fingerprints": item_fingerprints,
        "execution_boundary": _boundary(),
        "notes": [
            "Rubric specs define reviewer/model-grader criteria only; they do not call a model grader.",
            "Labels from this rubric require a calibrated gate before entering trainer-ready data.",
        ],
    }


def build_model_grader_dry_run(
    *,
    review_export_dir: str | Path,
    rubric_path: str | Path,
    grader_id: str,
    provider: str = "mock",
    created_at: str | None = None,
    preserve_paths: bool = False,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic dry-run model-grader receipt without model calls."""
    review_dir = Path(review_export_dir)
    rubric_file = Path(rubric_path)
    output_path = Path(out_path) if out_path is not None else None
    _require_evidence_dir(review_dir, "review export")
    _require_evidence_file(rubric_file, "rubric spec")
    items = _read_review_items(review_dir)
    labels = [_mock_label(item, grader_id, provider) for item in items]
    label_counts = _label_counts(labels)
    disagreement_queue = [_disagreement_candidate(item, label) for item, label in zip(items, labels) if label["requires_human_review"]]
    checks: list[dict[str, Any]] = []
    rubric_ref = _json_artifact_ref("rubric_spec", rubric_file, "rubric_spec", preserve_paths, output_path)
    _add_check(checks, "rubric_spec_valid", _artifact_ready(rubric_ref), {"artifact": rubric_ref}, {"schema": "rubric_spec", "passed": True})
    _add_check(checks, "review_export_non_empty", bool(items), {"review_item_count": len(items)}, {"review_item_count_min": 1})
    _add_check(checks, "paid_model_grader_not_called", True, {"paid_model_grader_calls_started": False}, {"paid_model_grader_calls_started": False})
    _add_check(checks, "labels_not_admitted_to_training", True, {"training_labels_admitted": 0}, {"training_labels_admitted": 0})
    failed = [check for check in checks if not check["passed"]]
    return {
        "schema_version": MODEL_GRADER_DRY_RUN_SCHEMA_VERSION,
        "created_at": created_at or _now(),
        "grader": {
            "provider": provider,
            "grader_id": grader_id,
            "mode": "dry_run",
            "transport": "mock",
            "provider_api_called": False,
            "paid_model_grader_calls_started": False,
        },
        "passed": not failed,
        "readiness": "ready_for_calibration_gate" if not failed else "blocked",
        "recommendation": "gate_before_training_admission" if not failed else "fix_rubric_or_review_export",
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "checks": checks,
        "blocked_reasons": [check["summary"] for check in failed],
        "source_artifacts": {
            "review_export": _review_export_ref(review_dir, preserve_paths, output_path),
            "rubric_spec": rubric_ref,
        },
        "graded_item_count": len(labels),
        "label_counts": label_counts,
        "grader_labels": labels,
        "disagreement_queue": disagreement_queue,
        "human_review_overrides": {
            "required_before_training_admission": True,
            "override_queue_path": None,
            "override_count": 0,
            "accepted_override_count": 0,
        },
        "training_admission": {
            "labels_allowed_for_training": False,
            "labels_admitted_count": 0,
            "requires_calibrated_gate": True,
        },
        "execution_boundary": _boundary(),
        "notes": [
            "Dry-run labels are deterministic mock labels derived from existing scorecard suggestions.",
            "The receipt does not call a model provider, paid grader, or trainer.",
            "A separate model-grader gate must pass before any downstream dataset curation may consume grader labels.",
        ],
    }


def build_model_grader_disagreement_queue(
    *,
    dry_run_path: str | Path,
    created_at: str | None = None,
    preserve_paths: bool = False,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a portable human-review queue from a model-grader dry-run receipt."""
    dry_file = Path(dry_run_path)
    output_path = Path(out_path) if out_path is not None else None
    _require_evidence_file(dry_file, "model-grader dry-run receipt")
    dry_ref = _json_artifact_ref("model_grader_dry_run", dry_file, "model_grader_dry_run", preserve_paths, output_path)
    dry_run = _read_json(dry_file)
    queue = [dict(item) for item in dry_run.get("disagreement_queue", []) if isinstance(item, dict)] if isinstance(dry_run, dict) else []
    labels_requiring_review = _labels_requiring_human_review_count(dry_run)
    checks: list[dict[str, Any]] = []
    _add_check(checks, "dry_run_receipt_valid", _artifact_ready(dry_ref), {"artifact": dry_ref}, {"schema": "model_grader_dry_run", "passed": True})
    _add_check(
        checks,
        "queue_matches_dry_run_human_review_labels",
        len(queue) == labels_requiring_review,
        {"queue_count": len(queue), "labels_requiring_human_review_count": labels_requiring_review},
        {"queue_count": labels_requiring_review},
    )
    _add_check(
        checks,
        "dry_run_labels_not_previously_admitted",
        _labels_admitted_count(dry_run) == 0,
        {"labels_admitted_count": _labels_admitted_count(dry_run)},
        {"labels_admitted_count": 0},
    )
    _add_check(
        checks,
        "paid_model_grader_not_called",
        _paid_calls_started(dry_run) is False,
        {"paid_model_grader_calls_started": _paid_calls_started(dry_run)},
        {"paid_model_grader_calls_started": False},
    )
    failed = [check for check in checks if not check["passed"]]
    passed = not failed
    readiness = "blocked"
    recommendation = "fix_dry_run_receipt"
    if passed and queue:
        readiness = "ready_for_human_review"
        recommendation = "collect_human_overrides"
    elif passed:
        readiness = "queue_empty"
        recommendation = "no_human_override_required"
    return {
        "schema_version": MODEL_GRADER_DISAGREEMENT_QUEUE_SCHEMA_VERSION,
        "created_at": created_at or _now(),
        "passed": passed,
        "readiness": readiness,
        "recommendation": recommendation,
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "checks": checks,
        "blocked_reasons": [check["summary"] for check in failed],
        "source_artifacts": {"dry_run_receipt": dry_ref},
        "queue_count": len(queue),
        "required_review_item_ids": sorted(str(item.get("review_item_id") or "") for item in queue if item.get("review_item_id")),
        "queue": queue,
        "override_requirements": {
            "required_before_training_admission": True,
            "required_override_count": len(queue),
            "final_label_options": [label for label in REVIEW_LABELS if label != "needs_review"],
            "minimum_reviewer_confidence": "medium",
        },
        "training_admission": {
            "labels_allowed_for_training": False,
            "labels_admitted_count": 0,
            "requires_model_grader_gate": True,
        },
        "execution_boundary": _boundary(),
        "notes": [
            "Disagreement queues route dry-run model-grader labels to human review.",
            "This artifact is derived from a dry-run receipt and does not call a provider or admit labels to training.",
            "A model-grader override receipt and gate must resolve this queue before trainer-facing curation can consume labels.",
        ],
    }


def build_model_grader_gate(
    *,
    dry_run_path: str | Path,
    rubric_path: str | Path,
    calibration_path: str | Path | None = None,
    override_receipt_path: str | Path | None = None,
    min_calibration_agreement_rate: float = 0.8,
    max_disagreements: int | None = None,
    created_at: str | None = None,
    preserve_paths: bool = False,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    """Gate model-grader labels before trainer-facing data admission."""
    output_path = Path(out_path) if out_path is not None else None
    dry_file = Path(dry_run_path)
    rubric_file = Path(rubric_path)
    calibration_file = Path(calibration_path) if calibration_path else None
    override_receipt_file = Path(override_receipt_path) if override_receipt_path else None
    _require_evidence_file(dry_file, "model-grader dry-run receipt")
    _require_evidence_file(rubric_file, "rubric spec")
    if calibration_file is not None:
        _require_evidence_file(calibration_file, "review calibration")
    if override_receipt_file is not None:
        _require_evidence_file(override_receipt_file, "model-grader override receipt")
    dry_ref = _json_artifact_ref(
        "model_grader_dry_run",
        dry_file,
        "model_grader_dry_run",
        preserve_paths,
        output_path,
    )
    rubric_ref = _json_artifact_ref("rubric_spec", rubric_file, "rubric_spec", preserve_paths, output_path)
    override_ref = (
        _json_artifact_ref(
            "model_grader_override_receipt",
            override_receipt_file,
            "model_grader_override_receipt",
            preserve_paths,
            output_path,
        )
        if override_receipt_path
        else _missing_ref("model_grader_override_receipt")
    )
    calibration_ref = (
        _json_artifact_ref(
            "review_calibration",
            calibration_file,
            "review_calibration",
            preserve_paths,
            output_path,
        )
        if calibration_path
        else _missing_ref("review_calibration")
    )
    dry_run = _read_json(dry_file)
    override_receipt = _read_json(override_receipt_file) if override_receipt_file is not None else {}
    calibration = _read_json(calibration_file) if calibration_file is not None else {}
    labels_requiring_human_review = _labels_requiring_human_review_count(dry_run)
    dry_run_disagreement_queue_count = _dry_run_disagreement_queue_count(dry_run)
    overrides_required = labels_requiring_human_review > 0 or dry_run_disagreement_queue_count > 0
    override_ready = (not overrides_required) or (_artifact_ready(override_ref) and override_receipt.get("passed") is True)
    override_resolved_count = _override_resolved_queue_count(override_receipt)
    override_unresolved_count = _override_unresolved_queue_count(override_receipt)
    human_review_queue_resolved = (not overrides_required) or (
        override_ready
        and override_unresolved_count == 0
        and override_resolved_count >= max(labels_requiring_human_review, dry_run_disagreement_queue_count)
    )
    checks: list[dict[str, Any]] = []
    _add_check(checks, "dry_run_receipt_valid", _artifact_ready(dry_ref), {"artifact": dry_ref}, {"schema": "model_grader_dry_run", "passed": True})
    _add_check(checks, "rubric_spec_valid", _artifact_ready(rubric_ref), {"artifact": rubric_ref}, {"schema": "rubric_spec", "passed": True})
    _add_check(
        checks,
        "review_calibration_present",
        calibration_path is not None and calibration_ref["exists"],
        {"artifact": calibration_ref},
        {"exists": True},
    )
    _add_check(
        checks,
        "review_calibration_passed",
        _artifact_ready(calibration_ref) and calibration.get("passed") is True,
        {"artifact": calibration_ref, "source_passed": calibration_ref.get("source_passed")},
        {"schema": "review_calibration", "passed": True, "source_passed": True},
    )
    agreement_rate = _calibration_agreement_rate(calibration)
    _add_check(
        checks,
        "min_calibration_agreement_rate",
        agreement_rate >= min_calibration_agreement_rate,
        {"agreement_rate": agreement_rate},
        {"min": min_calibration_agreement_rate},
    )
    if max_disagreements is not None:
        disagreement_count = _calibration_disagreement_count(calibration)
        _add_check(
            checks,
            "max_calibration_disagreements",
            disagreement_count <= max_disagreements,
            {"disagreement_count": disagreement_count},
            {"max": max_disagreements},
        )
    _add_check(
        checks,
        "dry_run_labels_not_previously_admitted",
        _labels_admitted_count(dry_run) == 0,
        {"labels_admitted_count": _labels_admitted_count(dry_run)},
        {"labels_admitted_count": 0},
    )
    _add_check(
        checks,
        "dry_run_human_review_queue_resolved",
        human_review_queue_resolved,
        {
            "labels_requiring_human_review_count": labels_requiring_human_review,
            "disagreement_queue_count": dry_run_disagreement_queue_count,
            "override_receipt_required": overrides_required,
            "override_receipt_passed": override_receipt.get("passed") if isinstance(override_receipt, dict) else None,
            "override_resolved_count": override_resolved_count,
            "override_unresolved_count": override_unresolved_count,
        },
        {"override_receipt_required": overrides_required, "unresolved_count": 0},
    )
    _add_check(
        checks,
        "paid_model_grader_not_called",
        _paid_calls_started(dry_run) is False,
        {"paid_model_grader_calls_started": _paid_calls_started(dry_run)},
        {"paid_model_grader_calls_started": False},
    )
    failed = [check for check in checks if not check["passed"]]
    passed = not failed
    return {
        "schema_version": MODEL_GRADER_GATE_SCHEMA_VERSION,
        "created_at": created_at or _now(),
        "passed": passed,
        "readiness": "labels_calibrated_for_curated_handoff" if passed else "blocked",
        "recommendation": "allow_curated_grader_labels" if passed else "route_to_human_review_or_calibration",
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "checks": checks,
        "blocked_reasons": [check["summary"] for check in failed],
        "source_artifacts": {
            "dry_run_receipt": dry_ref,
            "rubric_spec": rubric_ref,
            "review_calibration": calibration_ref,
            "model_grader_override_receipt": override_ref,
        },
        "admission": {
            "labels_allowed_for_training": passed,
            "labels_admitted_count": dry_run.get("graded_item_count", 0) if passed else 0,
            "uncalibrated_labels_admitted": 0,
            "human_override_required_for_disagreements": True,
        },
        "metrics": {
            "graded_item_count": dry_run.get("graded_item_count", 0) if isinstance(dry_run, dict) else 0,
            "agreement_rate": agreement_rate,
            "calibration_disagreement_count": _calibration_disagreement_count(calibration),
            "dry_run_disagreement_queue_count": dry_run_disagreement_queue_count,
            "dry_run_labels_requiring_human_review_count": labels_requiring_human_review,
            "human_override_receipt_present": override_receipt_path is not None,
            "human_override_resolved_count": override_resolved_count,
            "human_override_unresolved_count": override_unresolved_count,
        },
        "execution_boundary": _boundary(),
        "notes": [
            "The gate is the only artifact that can make model-grader labels eligible for downstream curation.",
            "Missing or failing calibration blocks by default.",
            "Dry-run disagreement queues or labels requiring human review block until resolved by a future override contract.",
            "The gate records no provider calls and admits zero uncalibrated labels.",
        ],
    }


def build_model_grader_override_receipt(
    *,
    dry_run_path: str | Path,
    overrides_path: str | Path,
    created_at: str | None = None,
    preserve_paths: bool = False,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a human override receipt resolving model-grader dry-run queue items."""
    dry_file = Path(dry_run_path)
    override_file = Path(overrides_path)
    output_path = Path(out_path) if out_path is not None else None
    _require_evidence_file(dry_file, "model-grader dry-run receipt")
    _require_evidence_file(override_file, "model-grader override rows")
    dry_ref = _json_artifact_ref("model_grader_dry_run", dry_file, "model_grader_dry_run", preserve_paths, output_path)
    dry_run = _read_json(dry_file)
    queue = [item for item in dry_run.get("disagreement_queue", []) if isinstance(item, dict)] if isinstance(dry_run, dict) else []
    queue_ids = {str(item.get("review_item_id") or "") for item in queue if item.get("review_item_id")}
    rows, row_errors = _read_override_rows(override_file)
    overrides = [_override_record(index, row, queue_ids) for index, row in enumerate(rows)]
    resolved_ids = {row["review_item_id"] for row in overrides if row.get("accepted") and row.get("resolves_queue_item")}
    unresolved_ids = sorted(queue_ids - resolved_ids)
    unmatched_count = sum(1 for row in overrides if not row.get("resolves_queue_item"))
    invalid_count = sum(1 for row in overrides if not row.get("accepted"))
    checks: list[dict[str, Any]] = []
    _add_check(checks, "dry_run_receipt_valid", _artifact_ready(dry_ref), {"artifact": dry_ref}, {"schema": "model_grader_dry_run", "passed": True})
    _add_check(checks, "override_rows_readable", not row_errors, {"errors": row_errors}, {"errors": []})
    _add_check(
        checks,
        "override_rows_present_when_queue_non_empty",
        bool(rows) or not queue_ids,
        {"override_row_count": len(rows), "queue_count": len(queue_ids)},
        {"override_row_count": ">0 when queue_count > 0"},
    )
    _add_check(
        checks,
        "all_overrides_match_queue",
        unmatched_count == 0,
        {"unmatched_override_count": unmatched_count},
        {"unmatched_override_count": 0},
    )
    _add_check(
        checks,
        "override_labels_finalized",
        invalid_count == 0,
        {"invalid_override_count": invalid_count},
        {"invalid_override_count": 0},
    )
    _add_check(
        checks,
        "all_queue_items_resolved",
        not unresolved_ids,
        {"unresolved_review_item_ids": unresolved_ids},
        {"unresolved_review_item_ids": []},
    )
    _add_check(
        checks,
        "flight_recorder_did_not_admit_labels",
        True,
        {"labels_admitted_to_training": False, "weights_updated_by_flight_recorder": False},
        {"labels_admitted_to_training": False, "weights_updated_by_flight_recorder": False},
    )
    failed = [check for check in checks if not check["passed"]]
    passed = not failed
    return {
        "schema_version": MODEL_GRADER_OVERRIDE_RECEIPT_SCHEMA_VERSION,
        "created_at": created_at or _now(),
        "passed": passed,
        "readiness": "ready_for_model_grader_gate" if passed else "blocked",
        "recommendation": "use_overrides_for_grader_gate" if passed else "complete_human_override_review",
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "checks": checks,
        "blocked_reasons": [check["summary"] for check in failed],
        "source_artifacts": {
            "dry_run_receipt": dry_ref,
            "override_rows": _file_ref("model_grader_override_rows", override_file, preserve_paths, output_path),
        },
        "queue": {
            "dry_run_disagreement_queue_count": len(queue_ids),
            "required_review_item_ids": sorted(queue_ids),
            "resolved_review_item_ids": sorted(resolved_ids),
            "unresolved_review_item_ids": unresolved_ids,
        },
        "overrides": overrides,
        "metrics": {
            "override_row_count": len(rows),
            "resolved_queue_count": len(resolved_ids),
            "unresolved_queue_count": len(unresolved_ids),
            "unmatched_override_count": unmatched_count,
            "invalid_override_count": invalid_count,
        },
        "training_admission": {
            "labels_allowed_for_training": False,
            "labels_admitted_count": 0,
            "requires_model_grader_gate": True,
        },
        "execution_boundary": _boundary(),
        "notes": [
            "Override receipts record human resolution of model-grader dry-run queue items only.",
            "Flight Recorder does not admit labels to training from this receipt; model-grader gate must consume it.",
        ],
    }


def write_model_grader_artifact(path: str | Path, payload: dict[str, Any]) -> None:
    """Write stable JSON for any model-grader contract."""
    out_path = Path(path)
    _require_output_file(out_path, "model-grader artifact output")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_review_items(review_export_dir: Path) -> list[dict[str, Any]]:
    path = review_export_dir / "review_items.jsonl"
    _require_evidence_file(path, "review_items.jsonl")
    rows = _read_jsonl(path, "review_items.jsonl")
    if not rows:
        raise ModelGraderError(f"review_items.jsonl contains no review items: {path}")
    return rows


def _require_evidence_dir(path: Path, label: str) -> None:
    if _path_has_symlink_component(path, include_leaf=True):
        raise ModelGraderError(f"{label} must resolve to a regular non-symlink directory: {path}")
    if not path.exists():
        raise ModelGraderError(f"{label} directory not found: {path}")
    if not path.is_dir():
        raise ModelGraderError(f"{label} path is not a directory: {path}")


def _require_evidence_file(path: Path, label: str) -> None:
    if _path_has_symlink_component(path, include_leaf=True):
        raise ModelGraderError(f"{label} must resolve to a regular non-symlink file: {path}")
    if not path.exists():
        raise ModelGraderError(f"{label} not found: {path}")
    if not path.is_file():
        raise ModelGraderError(f"{label} must be a file: {path}")


def _require_output_file(path: Path, label: str) -> None:
    if _path_has_symlink_component(path, include_leaf=True):
        raise ModelGraderError(f"{label} must resolve to a regular non-symlink file: {path}")
    if path.exists() and not path.is_file():
        raise ModelGraderError(f"{label} must be a file: {path}")


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ModelGraderError(f"{label}:{line_number} contains invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ModelGraderError(f"{label}:{line_number} must contain a JSON object")
        rows.append(row)
    return rows


def _criterion_rows(criteria: list[str] | None) -> list[dict[str, Any]]:
    values = criteria or [
        "Ground the label in observable trace events and scorecard rules.",
        "Reject unsafe, incomplete, or instruction-violating outputs even when fluent.",
        "Route ambiguous or evidence-light items to human review instead of training.",
    ]
    rows: list[dict[str, Any]] = []
    for index, text in enumerate(values, start=1):
        if not text or not str(text).strip():
            raise ModelGraderError("Rubric criteria must be non-empty strings")
        rows.append(
            {
                "criterion_id": f"criterion_{index:02d}",
                "description": str(text).strip(),
                "required": True,
            }
        )
    return rows


def _mock_label(item: dict[str, Any], grader_id: str, provider: str) -> dict[str, Any]:
    scorecard = item.get("scorecard") if isinstance(item.get("scorecard"), dict) else {}
    suggested = str(item.get("suggested_human_label") or ("accept" if scorecard.get("passed") is True else "reject"))
    if suggested not in REVIEW_LABELS:
        suggested = "needs_review"
    confidence = "medium" if suggested in {"accept", "reject"} else "low"
    requires_human_review = suggested in {"needs_review", "unsafe", "incomplete"} or confidence == "low"
    row = {
        "review_item_id": str(item.get("review_item_id") or ""),
        "episode_id": str(item.get("episode_id") or ""),
        "scenario_id": str(item.get("scenario_id") or ""),
        "task_family": str(item.get("task_family") or "unknown"),
        "review_item_sha256": str(item.get("review_item_sha256") or review_item_sha256(item)),
        "mock_model_label": suggested,
        "mock_confidence": confidence,
        "requires_human_review": requires_human_review,
        "rationale": "Dry-run mock label mirrors the existing scorecard suggestion; no model grader was called.",
        "grader_id": grader_id,
        "provider": provider,
        "label_sha256": "",
    }
    row["label_sha256"] = _stable_sha(row)
    return row


def _disagreement_candidate(item: dict[str, Any], label: dict[str, Any]) -> dict[str, Any]:
    return {
        "review_item_id": label["review_item_id"],
        "episode_id": label["episode_id"],
        "scenario_id": label["scenario_id"],
        "task_family": label["task_family"],
        "review_item_sha256": label["review_item_sha256"],
        "mock_model_label": label["mock_model_label"],
        "reason": "human_review_required_by_rubric_or_low_confidence",
        "source_report": (item.get("source_artifacts") or {}).get("report") if isinstance(item.get("source_artifacts"), dict) else None,
    }


def _item_fingerprint(item: dict[str, Any]) -> dict[str, Any]:
    expected = str(item.get("review_item_sha256") or review_item_sha256(item))
    return {
        "review_item_id": str(item.get("review_item_id") or ""),
        "episode_id": str(item.get("episode_id") or ""),
        "scenario_id": str(item.get("scenario_id") or ""),
        "review_item_sha256": expected,
    }


def _label_counts(labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = {label: 0 for label in REVIEW_LABELS}
    for row in labels:
        label = str(row.get("mock_model_label") or "")
        if label in counts:
            counts[label] += 1
    return [{"label": label, "count": counts[label]} for label in REVIEW_LABELS]


def _review_export_ref(path: Path, preserve_paths: bool, output_path: Path | None = None) -> dict[str, Any]:
    manifest = path / "manifest.json"
    items = path / "review_items.jsonl"
    return {
        "path": _display_path(path, preserve_paths, output_path),
        "manifest": _file_ref("review_manifest", manifest, preserve_paths, output_path),
        "review_items": _file_ref("review_items", items, preserve_paths, output_path),
    }


def _json_artifact_ref(
    role: str,
    path: Path,
    schema_name: str,
    preserve_paths: bool,
    output_path: Path | None = None,
) -> dict[str, Any]:
    ref = _file_ref(role, path, preserve_paths, output_path)
    schema = _schema_check(path, schema_name) if ref["exists"] else {"passed": False, "error_count": 1, "errors": ["artifact not found"]}
    payload = _read_json(path)
    ref.update(
        {
            "schema_name": schema_name,
            "schema_passed": schema["passed"],
            "schema_error_count": schema["error_count"],
            "schema_errors": schema["errors"],
            "source_passed": payload.get("passed") if isinstance(payload.get("passed"), bool) else None,
            "source_recommendation": str(payload.get("recommendation") or ""),
        }
    )
    return ref


def _file_ref(role: str, path: Path, preserve_paths: bool, output_path: Path | None = None) -> dict[str, Any]:
    exists = path.exists() and path.is_file() and not _path_has_symlink_component(path, include_leaf=True)
    return {
        "role": role,
        "path": _display_path(path, preserve_paths, output_path),
        "exists": exists,
        "sha256": _sha256(path) if exists else None,
        "size_bytes": path.stat().st_size if exists else None,
    }


def _missing_ref(role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": None,
        "exists": False,
        "sha256": None,
        "size_bytes": None,
        "schema_name": role,
        "schema_passed": False,
        "schema_error_count": 1,
        "schema_errors": ["artifact not provided"],
        "source_passed": None,
        "source_recommendation": "",
    }


def _schema_check(path: Path, schema_name: str) -> dict[str, Any]:
    try:
        return check_schema_file(path, schema_name)
    except (OSError, json.JSONDecodeError, SchemaRegistryError) as exc:
        return {"passed": False, "error_count": 1, "errors": [str(exc)]}


def _artifact_ready(ref: dict[str, Any]) -> bool:
    return bool(ref.get("exists") and ref.get("schema_passed") and ref.get("source_passed") is not False)


def _add_check(
    checks: list[dict[str, Any]],
    check_id: str,
    passed: bool,
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    checks.append(
        {
            "id": check_id,
            "passed": bool(passed),
            "actual": actual,
            "expected": expected,
            "summary": f"{check_id}: {'passed' if passed else 'failed'}",
        }
    )


def _labels_admitted_count(dry_run: dict[str, Any]) -> int:
    admission = dry_run.get("training_admission") if isinstance(dry_run.get("training_admission"), dict) else {}
    value = admission.get("labels_admitted_count")
    return int(value) if isinstance(value, int) and value >= 0 else 0


def _labels_requiring_human_review_count(dry_run: dict[str, Any]) -> int:
    labels = dry_run.get("grader_labels") if isinstance(dry_run.get("grader_labels"), list) else []
    return sum(1 for row in labels if isinstance(row, dict) and row.get("requires_human_review") is True)


def _dry_run_disagreement_queue_count(dry_run: dict[str, Any]) -> int:
    queue = dry_run.get("disagreement_queue") if isinstance(dry_run.get("disagreement_queue"), list) else []
    return len(queue)


def _paid_calls_started(dry_run: dict[str, Any]) -> bool:
    grader = dry_run.get("grader") if isinstance(dry_run.get("grader"), dict) else {}
    return bool(grader.get("paid_model_grader_calls_started"))


def _calibration_agreement_rate(calibration: dict[str, Any]) -> float:
    metrics = calibration.get("metrics") if isinstance(calibration.get("metrics"), dict) else {}
    value = metrics.get("agreement_rate")
    return float(value) if isinstance(value, (int, float)) else 0.0


def _calibration_disagreement_count(calibration: dict[str, Any]) -> int:
    metrics = calibration.get("metrics") if isinstance(calibration.get("metrics"), dict) else {}
    value = metrics.get("disagreement_count")
    return int(value) if isinstance(value, int) and value >= 0 else 0


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists() or not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_override_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return [], [f"file not found: {path}"]
    except OSError as exc:
        return [], [str(exc)]
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_number}: invalid JSON: {exc.msg}")
            continue
        if not isinstance(row, dict):
            errors.append(f"line {line_number}: row must be an object")
            continue
        rows.append(row)
    return rows, errors


def _override_record(index: int, row: dict[str, Any], queue_ids: set[str]) -> dict[str, Any]:
    review_item_id = str(row.get("review_item_id") or "")
    human_label = str(row.get("human_label") or "")
    reviewer_confidence = str(row.get("reviewer_confidence") or "")
    reviewer = str(row.get("reviewer") or "")
    reviewed_at = str(row.get("reviewed_at") or "")
    notes = str(row.get("notes") or "")
    errors: list[str] = []
    if not review_item_id:
        errors.append("review_item_id is required")
    if human_label not in REVIEW_LABELS or human_label == "needs_review":
        errors.append("human_label must be a finalized review label")
    if reviewer_confidence not in {"high", "medium"}:
        errors.append("reviewer_confidence must be high or medium")
    if not reviewer:
        errors.append("reviewer is required")
    if not reviewed_at:
        errors.append("reviewed_at is required")
    resolves_queue_item = review_item_id in queue_ids
    if not resolves_queue_item:
        errors.append("review_item_id does not match a dry-run queue item")
    record = {
        "index": index,
        "review_item_id": review_item_id,
        "review_item_sha256": str(row.get("review_item_sha256") or ""),
        "human_label": human_label,
        "reviewer_confidence": reviewer_confidence,
        "reviewer": reviewer,
        "reviewed_at": reviewed_at,
        "notes": notes,
        "resolves_queue_item": resolves_queue_item,
        "accepted": not errors,
        "errors": errors,
        "override_sha256": "",
    }
    record["override_sha256"] = _stable_sha(record)
    return record


def _override_resolved_queue_count(receipt: dict[str, Any]) -> int:
    metrics = receipt.get("metrics") if isinstance(receipt.get("metrics"), dict) else {}
    value = metrics.get("resolved_queue_count")
    return int(value) if isinstance(value, int) and value >= 0 else 0


def _override_unresolved_queue_count(receipt: dict[str, Any]) -> int:
    metrics = receipt.get("metrics") if isinstance(receipt.get("metrics"), dict) else {}
    value = metrics.get("unresolved_queue_count")
    return int(value) if isinstance(value, int) and value >= 0 else 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_sha(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "label_sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _display_path(path: Path, preserve_paths: bool, output_path: Path | None = None) -> str:
    if preserve_paths or output_path is None:
        return str(path) if preserve_paths else path.name
    return os.path.relpath(path.resolve(), output_path.parent.resolve())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _boundary() -> dict[str, Any]:
    return {
        "dry_run_only": True,
        "provider_api_called": False,
        "paid_model_grader_calls_started": False,
        "cloud_cost_incurred_usd": 0,
        "credential_values_recorded": False,
        "labels_admitted_to_training": False,
        "weights_updated_by_flight_recorder": False,
    }
