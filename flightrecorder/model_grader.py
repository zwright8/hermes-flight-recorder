"""Fail-closed rubric and model-grader contracts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .review import REVIEW_LABELS, review_item_sha256
from .schema_registry import SchemaRegistryError, check_schema_file

RUBRIC_SPEC_SCHEMA_VERSION = "hfr.rubric_spec.v1"
MODEL_GRADER_DRY_RUN_SCHEMA_VERSION = "hfr.model_grader_dry_run.v1"
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
) -> dict[str, Any]:
    """Build a schema-checkable rubric bound to a human-review queue."""
    review_dir = Path(review_export_dir)
    items = _read_review_items(review_dir)
    criterion_rows = _criterion_rows(criteria)
    item_fingerprints = [_item_fingerprint(row) for row in items]
    return {
        "schema_version": RUBRIC_SPEC_SCHEMA_VERSION,
        "rubric_id": rubric_id,
        "created_at": created_at or _now(),
        "review_export": _review_export_ref(review_dir, preserve_paths),
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
) -> dict[str, Any]:
    """Build a deterministic dry-run model-grader receipt without model calls."""
    review_dir = Path(review_export_dir)
    rubric_file = Path(rubric_path)
    items = _read_review_items(review_dir)
    labels = [_mock_label(item, grader_id, provider) for item in items]
    label_counts = _label_counts(labels)
    disagreement_queue = [_disagreement_candidate(item, label) for item, label in zip(items, labels) if label["requires_human_review"]]
    checks: list[dict[str, Any]] = []
    rubric_ref = _json_artifact_ref("rubric_spec", rubric_file, "rubric_spec", preserve_paths)
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
            "review_export": _review_export_ref(review_dir, preserve_paths),
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


def build_model_grader_gate(
    *,
    dry_run_path: str | Path,
    rubric_path: str | Path,
    calibration_path: str | Path | None = None,
    min_calibration_agreement_rate: float = 0.8,
    max_disagreements: int | None = None,
    created_at: str | None = None,
    preserve_paths: bool = False,
) -> dict[str, Any]:
    """Gate model-grader labels before trainer-facing data admission."""
    dry_ref = _json_artifact_ref("model_grader_dry_run", Path(dry_run_path), "model_grader_dry_run", preserve_paths)
    rubric_ref = _json_artifact_ref("rubric_spec", Path(rubric_path), "rubric_spec", preserve_paths)
    calibration_ref = (
        _json_artifact_ref("review_calibration", Path(calibration_path), "review_calibration", preserve_paths)
        if calibration_path
        else _missing_ref("review_calibration")
    )
    dry_run = _read_json(Path(dry_run_path))
    calibration = _read_json(Path(calibration_path)) if calibration_path else {}
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
            "dry_run_disagreement_queue_count": len(dry_run.get("disagreement_queue", [])) if isinstance(dry_run, dict) else 0,
        },
        "execution_boundary": _boundary(),
        "notes": [
            "The gate is the only artifact that can make model-grader labels eligible for downstream curation.",
            "Missing or failing calibration blocks by default.",
            "The gate records no provider calls and admits zero uncalibrated labels.",
        ],
    }


def write_model_grader_artifact(path: str | Path, payload: dict[str, Any]) -> None:
    """Write stable JSON for any model-grader contract."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_review_items(review_export_dir: Path) -> list[dict[str, Any]]:
    path = review_export_dir / "review_items.jsonl"
    if not path.exists():
        raise ModelGraderError(f"review_items.jsonl not found: {path}")
    rows = _read_jsonl(path, "review_items.jsonl")
    if not rows:
        raise ModelGraderError(f"review_items.jsonl contains no review items: {path}")
    return rows


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


def _review_export_ref(path: Path, preserve_paths: bool) -> dict[str, Any]:
    manifest = path / "manifest.json"
    items = path / "review_items.jsonl"
    return {
        "path": _display_path(path, preserve_paths),
        "manifest": _file_ref("review_manifest", manifest, preserve_paths),
        "review_items": _file_ref("review_items", items, preserve_paths),
    }


def _json_artifact_ref(role: str, path: Path, schema_name: str, preserve_paths: bool) -> dict[str, Any]:
    ref = _file_ref(role, path, preserve_paths)
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


def _file_ref(role: str, path: Path, preserve_paths: bool) -> dict[str, Any]:
    exists = path.exists() and path.is_file()
    return {
        "role": role,
        "path": _display_path(path, preserve_paths),
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


def _display_path(path: Path, preserve_paths: bool) -> str:
    return str(path) if preserve_paths else path.name


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
