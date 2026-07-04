"""Calibration reports for deterministic scorecards versus human labels."""

from __future__ import annotations

import json
import os
from pathlib import Path, PureWindowsPath
from typing import Any

from .path_safety import path_has_symlink_component as _path_has_symlink_component
from .review import REVIEW_LABELS, TRAINING_NEGATIVE_LABELS

REVIEW_CALIBRATION_SCHEMA_VERSION = "hfr.review_calibration.v1"
HUMAN_POSITIVE_LABELS = {"accept"}
HUMAN_NEGATIVE_LABELS = set(TRAINING_NEGATIVE_LABELS)
COMPARABLE_LABELS = HUMAN_POSITIVE_LABELS | HUMAN_NEGATIVE_LABELS


class ReviewCalibrationError(ValueError):
    """Raised when a review-calibration report cannot be built."""


def build_review_calibration(
    reviewed_export_dir: str | Path,
    *,
    min_agreement_rate: float | None = None,
    max_disagreements: int | None = None,
    max_false_positives: int | None = None,
    max_false_negatives: int | None = None,
    min_comparable_labels: int | None = None,
    validation_summary: dict[str, Any] | None = None,
    require_valid_export: bool = True,
    preserve_paths: bool = False,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compare reviewed human labels with deterministic scorecard outcomes."""
    export_dir = Path(reviewed_export_dir)
    _require_regular_dir(export_dir, "reviewed export")
    labels_path = export_dir / "reviewed_labels.jsonl"
    _require_regular_file(labels_path, "reviewed_labels.jsonl")
    rows = _read_jsonl(labels_path, "reviewed_labels.jsonl")
    metrics, disagreements = _calibration_metrics(rows)
    metrics["validation"] = _validation_metrics(validation_summary)
    checks: list[dict[str, Any]] = []
    output_dir = Path(out_path).parent if out_path else None
    reviewed_export_ref = _display_path(export_dir, preserve_paths, output_dir)
    reviewed_labels_ref = _display_path(labels_path, preserve_paths, output_dir)
    _add_source_paths_check(checks, reviewed_export_ref, reviewed_labels_ref)
    if require_valid_export:
        _add_validation_check(checks, "valid_reviewed_export", validation_summary)
    if min_comparable_labels is not None:
        _add_min_check(checks, "min_comparable_labels", metrics["comparable_label_count"], min_comparable_labels)
    if min_agreement_rate is not None:
        _add_min_check(checks, "min_agreement_rate", metrics["agreement_rate"], min_agreement_rate)
    if max_disagreements is not None:
        _add_max_check(checks, "max_disagreements", metrics["disagreement_count"], max_disagreements)
    if max_false_positives is not None:
        _add_max_check(checks, "max_false_positives", metrics["false_positive_count"], max_false_positives)
    if max_false_negatives is not None:
        _add_max_check(checks, "max_false_negatives", metrics["false_negative_count"], max_false_negatives)

    failed_check_count = sum(1 for check in checks if not check["passed"])
    return {
        "schema_version": REVIEW_CALIBRATION_SCHEMA_VERSION,
        "reviewed_export": reviewed_export_ref,
        "source": {
            "reviewed_labels": reviewed_labels_ref,
        },
        "passed": failed_check_count == 0,
        "check_count": len(checks),
        "failed_check_count": failed_check_count,
        "checks": checks,
        "metrics": metrics,
        "disagreements": disagreements,
        "notes": [
            "Calibration compares deterministic scorecard pass/fail outcomes with human review labels.",
            "By default, calibration fails when the reviewed export no longer passes artifact validation.",
            "Disagreements are evidence for scenario-policy review; they are not automatic proof that either side is correct.",
            "Rows labeled needs_review are tracked but excluded from agreement-rate denominators.",
        ],
    }


def write_review_calibration(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a review-calibration report without following symlinked outputs."""
    out_path = Path(path)
    _require_output_file(out_path, "review calibration output")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = json.loads(json.dumps(payload))
    safe_payload["reviewed_export"] = _output_relative_path(safe_payload.get("reviewed_export"), out_path.parent)
    if isinstance(safe_payload.get("source"), dict):
        safe_payload["source"]["reviewed_labels"] = _output_relative_path(safe_payload["source"].get("reviewed_labels"), out_path.parent)
    out_path.write_text(json.dumps(safe_payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _calibration_metrics(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    label_counts = {label: 0 for label in REVIEW_LABELS}
    score_totals: dict[str, list[int]] = {}
    task_families: set[str] = set()
    comparable_count = 0
    agreement_count = 0
    scorecard_positive_count = 0
    scorecard_negative_count = 0
    human_positive_count = 0
    human_negative_count = 0
    false_positive_count = 0
    false_negative_count = 0
    disagreements: list[dict[str, Any]] = []

    for row in rows:
        label = str(row.get("human_label") or "unknown")
        if label in label_counts:
            label_counts[label] += 1
        task_family = str(row.get("task_family") or "unknown")
        task_families.add(task_family)
        score = _score(row)
        score_totals.setdefault(label, []).append(score)
        if label not in COMPARABLE_LABELS:
            continue

        comparable_count += 1
        human_positive = label in HUMAN_POSITIVE_LABELS
        scorecard = row.get("scorecard") if isinstance(row.get("scorecard"), dict) else {}
        scorecard_positive = bool(scorecard.get("passed"))
        if scorecard_positive:
            scorecard_positive_count += 1
        else:
            scorecard_negative_count += 1
        if human_positive:
            human_positive_count += 1
        else:
            human_negative_count += 1

        if scorecard_positive == human_positive:
            agreement_count += 1
            continue

        if scorecard_positive and not human_positive:
            false_positive_count += 1
            disagreement_type = "scorecard_passed_human_rejected"
        else:
            false_negative_count += 1
            disagreement_type = "scorecard_failed_human_accepted"
        disagreements.append(_disagreement_row(row, disagreement_type))

    disagreement_count = len(disagreements)
    metrics = {
        "reviewed_label_count": len(rows),
        "comparable_label_count": comparable_count,
        "needs_review_count": label_counts.get("needs_review", 0),
        "agreement_count": agreement_count,
        "disagreement_count": disagreement_count,
        "agreement_rate": _rate(agreement_count, comparable_count),
        "scorecard_positive_count": scorecard_positive_count,
        "scorecard_negative_count": scorecard_negative_count,
        "human_positive_count": human_positive_count,
        "human_negative_count": human_negative_count,
        "false_positive_count": false_positive_count,
        "false_negative_count": false_negative_count,
        "label_counts": [{"label": label, "count": label_counts[label]} for label in REVIEW_LABELS],
        "mean_score_by_human_label": _mean_score_rows(score_totals),
        "task_families": sorted(task_families),
    }
    return metrics, disagreements


def _disagreement_row(row: dict[str, Any], disagreement_type: str) -> dict[str, Any]:
    scorecard = row.get("scorecard") if isinstance(row.get("scorecard"), dict) else {}
    source_artifacts = row.get("source_artifacts") if isinstance(row.get("source_artifacts"), dict) else {}
    return {
        "review_item_id": row.get("review_item_id"),
        "episode_id": row.get("episode_id"),
        "scenario_id": row.get("scenario_id"),
        "task_family": row.get("task_family"),
        "human_label": row.get("human_label"),
        "scorecard_passed": bool(scorecard.get("passed")),
        "scorecard_score": _scorecard_score(scorecard),
        "disagreement_type": disagreement_type,
        "failed_rules": scorecard.get("failed_rules", []) if isinstance(scorecard.get("failed_rules"), list) else [],
        "critical_failures": scorecard.get("critical_failures", []) if isinstance(scorecard.get("critical_failures"), list) else [],
        "source_report": source_artifacts.get("report"),
        "source_lineage": source_artifacts.get("lineage"),
    }


def _mean_score_rows(score_totals: dict[str, list[int]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in sorted(score_totals):
        scores = score_totals[label]
        rows.append(
            {
                "label": label,
                "count": len(scores),
                "average_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
            }
        )
    return rows


def _add_min_check(checks: list[dict[str, Any]], check_id: str, actual: int | float, minimum: int | float) -> None:
    checks.append(
        {
            "id": check_id,
            "passed": actual >= minimum,
            "actual": actual,
            "expected": {"min": minimum},
            "summary": f"{check_id}: actual={actual}, min={minimum}",
        }
    )


def _add_max_check(checks: list[dict[str, Any]], check_id: str, actual: int | float, maximum: int | float) -> None:
    checks.append(
        {
            "id": check_id,
            "passed": actual <= maximum,
            "actual": actual,
            "expected": {"max": maximum},
            "summary": f"{check_id}: actual={actual}, max={maximum}",
        }
    )


def _add_validation_check(checks: list[dict[str, Any]], check_id: str, validation_summary: dict[str, Any] | None) -> None:
    metrics = _validation_metrics(validation_summary)
    checks.append(
        {
            "id": check_id,
            "passed": bool(metrics.get("available") and metrics.get("passed")),
            "actual": metrics,
            "expected": {"passed": True, "error_count": 0},
            "summary": (
                f"{check_id}: passed={metrics['passed']}, "
                f"errors={metrics['error_count']}, warnings={metrics['warning_count']}"
            ),
        }
    )


def _add_source_paths_check(checks: list[dict[str, Any]], reviewed_export: str, reviewed_labels: str) -> None:
    passed = _is_public_review_calibration_ref_path(reviewed_export) and _is_public_review_calibration_ref_path(reviewed_labels)
    checks.append(
        {
            "id": "source_paths_replayable",
            "passed": passed,
            "actual": {"reviewed_export": reviewed_export, "reviewed_labels": reviewed_labels},
            "expected": {"safe_relative_paths": True},
            "summary": f"source_paths_replayable: passed={passed}",
        }
    )


def _validation_metrics(validation_summary: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(validation_summary, dict):
        return {
            "available": False,
            "passed": False,
            "strict": False,
            "target_count": 0,
            "error_count": 0,
            "warning_count": 0,
        }
    return {
        "available": True,
        "passed": bool(validation_summary.get("passed")),
        "strict": bool(validation_summary.get("strict")),
        "target_count": _int_value(validation_summary.get("target_count")),
        "error_count": _int_value(validation_summary.get("error_count")),
        "warning_count": _int_value(validation_summary.get("warning_count")),
    }


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise ReviewCalibrationError(f"{label} not found: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReviewCalibrationError(f"{label}:{line_number} contains invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ReviewCalibrationError(f"{label}:{line_number} must contain a JSON object")
        rows.append(value)
    return rows


def _require_regular_file(path: Path, label: str) -> None:
    if _path_has_symlink_component(path, include_leaf=True):
        raise ReviewCalibrationError(f"{label} must resolve to a regular non-symlink file: {path}")
    if not path.exists():
        raise ReviewCalibrationError(f"{label} not found: {path}")
    if not path.is_file():
        raise ReviewCalibrationError(f"{label} must be a file: {path}")


def _require_regular_dir(path: Path, label: str) -> None:
    if _path_has_symlink_component(path, include_leaf=True):
        raise ReviewCalibrationError(f"{label} must resolve to a regular non-symlink directory: {path}")
    if not path.exists():
        raise ReviewCalibrationError(f"{label} directory not found: {path}")
    if not path.is_dir():
        raise ReviewCalibrationError(f"{label} path is not a directory: {path}")


def _require_output_file(path: Path, label: str) -> None:
    if _path_has_symlink_component(path, include_leaf=True):
        raise ReviewCalibrationError(f"{label} must resolve to a regular non-symlink file: {path}")
    if path.exists() and not path.is_file():
        raise ReviewCalibrationError(f"{label} must be a file: {path}")


def _score(row: dict[str, Any]) -> int:
    try:
        return max(0, min(100, int(row.get("score", 0))))
    except (TypeError, ValueError):
        return 0


def _scorecard_score(scorecard: dict[str, Any]) -> int:
    try:
        return max(0, min(100, int(scorecard.get("score", 0))))
    except (TypeError, ValueError):
        return 0


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 1.0
    return round(numerator / denominator, 4)


def _display_path(path: Path, preserve_paths: bool = False, output_dir: Path | None = None) -> str:
    raw = str(path)
    if output_dir is not None:
        try:
            relative = os.path.relpath(path.resolve(), output_dir.resolve())
        except OSError:
            return f"<redacted:{_basename(raw)}>"
        return relative if _is_public_review_calibration_ref_path(relative) else f"<redacted:{_basename(raw)}>"
    if preserve_paths:
        return raw if _is_public_review_calibration_ref_path(raw) else f"<redacted:{_basename(raw)}>"
    if not path.is_absolute():
        return raw if _is_public_review_calibration_ref_path(raw) else f"<redacted:{_basename(raw)}>"
    try:
        relative = str(path.resolve().relative_to(Path.cwd().resolve()))
    except (OSError, ValueError):
        return f"<redacted:{_basename(raw)}>"
    return relative if _is_public_review_calibration_ref_path(relative) else f"<redacted:{_basename(raw)}>"


def _output_relative_path(value: Any, output_dir: Path) -> Any:
    if not isinstance(value, str) or not value:
        return value
    if value.startswith("<redacted:"):
        return value
    path = Path(value)
    if not path.is_absolute():
        if not _is_public_review_calibration_ref_path(value):
            return f"<redacted:{_basename(value)}>"
        if not path.exists():
            return value
        path = path.resolve()
    try:
        relative = os.path.relpath(path.resolve(), output_dir.resolve())
    except OSError:
        return f"<redacted:{_basename(value)}>"
    return relative if _is_public_review_calibration_ref_path(relative) else f"<redacted:{_basename(value)}>"


def _is_public_review_calibration_ref_path(value: str) -> bool:
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
