"""Calibration reports for deterministic scorecards versus human labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
    preserve_paths: bool = False,
) -> dict[str, Any]:
    """Compare reviewed human labels with deterministic scorecard outcomes."""
    export_dir = Path(reviewed_export_dir)
    if not export_dir.exists():
        raise ReviewCalibrationError(f"Reviewed export directory not found: {export_dir}")
    if not export_dir.is_dir():
        raise ReviewCalibrationError(f"Reviewed export path is not a directory: {export_dir}")

    labels_path = export_dir / "reviewed_labels.jsonl"
    rows = _read_jsonl(labels_path, "reviewed_labels.jsonl")
    metrics, disagreements = _calibration_metrics(rows)
    checks: list[dict[str, Any]] = []
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
        "reviewed_export": _display_path(export_dir, preserve_paths),
        "source": {
            "reviewed_labels": _display_path(labels_path, preserve_paths),
        },
        "passed": failed_check_count == 0,
        "check_count": len(checks),
        "failed_check_count": failed_check_count,
        "checks": checks,
        "metrics": metrics,
        "disagreements": disagreements,
        "notes": [
            "Calibration compares deterministic scorecard pass/fail outcomes with human review labels.",
            "Disagreements are evidence for scenario-policy review; they are not automatic proof that either side is correct.",
            "Rows labeled needs_review are tracked but excluded from agreement-rate denominators.",
        ],
    }


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


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 1.0
    return round(numerator / denominator, 4)


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
