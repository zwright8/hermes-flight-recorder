"""Policy gates over human-reviewed trainer-ready exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .review import REVIEW_LABELS, TRAINING_NEGATIVE_LABELS

REVIEWED_GATE_SCHEMA_VERSION = "hfr.reviewed_gate.v1"
REVIEWED_GATE_POLICY_SCHEMA_VERSION = "hfr.reviewed_gate.policy.v1"

_COUNT_POLICY_FIELDS = {
    "min_reviewed_labels",
    "min_accepted",
    "min_rejected",
    "min_sft",
    "min_reward_model",
    "min_preferences",
    "min_dpo",
    "max_needs_review",
}
_LIST_POLICY_FIELDS = {"forbid_labels", "require_task_families"}
_BOOLEAN_POLICY_FIELDS = {"require_valid_export", "strict_validation"}
_POLICY_FIELDS = {"schema_version", "description", *_COUNT_POLICY_FIELDS, *_LIST_POLICY_FIELDS, *_BOOLEAN_POLICY_FIELDS}
_REVIEW_LABEL_SET = set(REVIEW_LABELS)


class ReviewedGatePolicyError(ValueError):
    """Raised when a reviewed-export gate policy file is malformed."""


def load_reviewed_gate_policy(path: str | Path) -> dict[str, Any]:
    """Load and validate a versioned reviewed-export gate policy JSON file."""
    policy_path = Path(path)
    try:
        raw = json.loads(policy_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReviewedGatePolicyError(f"Invalid JSON in reviewed gate policy {policy_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ReviewedGatePolicyError(f"Reviewed gate policy must be a JSON object: {policy_path}")

    version = raw.get("schema_version")
    if version != REVIEWED_GATE_POLICY_SCHEMA_VERSION:
        raise ReviewedGatePolicyError(
            f"reviewed gate policy schema_version must be {REVIEWED_GATE_POLICY_SCHEMA_VERSION!r}; got {version!r}"
        )

    unknown = sorted(set(raw) - _POLICY_FIELDS)
    if unknown:
        raise ReviewedGatePolicyError(f"Unknown reviewed gate policy field(s): {', '.join(unknown)}")

    policy: dict[str, Any] = {"schema_version": REVIEWED_GATE_POLICY_SCHEMA_VERSION}
    if "description" in raw:
        if not isinstance(raw["description"], str):
            raise ReviewedGatePolicyError("reviewed gate policy field description must be a string")
        policy["description"] = raw["description"]

    for field in _COUNT_POLICY_FIELDS:
        if field not in raw or raw[field] is None:
            continue
        policy[field] = _policy_non_negative_int(field, raw[field])

    for field in _LIST_POLICY_FIELDS:
        if field not in raw or raw[field] is None:
            continue
        values = _policy_string_list(field, raw[field])
        if field == "forbid_labels":
            invalid = sorted(set(values) - _REVIEW_LABEL_SET)
            if invalid:
                raise ReviewedGatePolicyError(
                    f"reviewed gate policy field {field} has invalid label value(s): {', '.join(invalid)}"
                )
        policy[field] = values

    for field in _BOOLEAN_POLICY_FIELDS:
        if field not in raw or raw[field] is None:
            continue
        if not isinstance(raw[field], bool):
            raise ReviewedGatePolicyError(f"reviewed gate policy field {field} must be a boolean")
        policy[field] = raw[field]

    return policy


def evaluate_reviewed_gate(
    manifest: dict[str, Any],
    *,
    reviewed_export_path: str | Path,
    min_reviewed_labels: int | None = None,
    min_accepted: int | None = None,
    min_rejected: int | None = None,
    min_sft: int | None = None,
    min_reward_model: int | None = None,
    min_preferences: int | None = None,
    min_dpo: int | None = None,
    max_needs_review: int | None = None,
    forbid_labels: list[str] | None = None,
    require_task_families: list[str] | None = None,
    validation_summary: dict[str, Any] | None = None,
    require_valid_export: bool = True,
) -> dict[str, Any]:
    """Evaluate readiness checks against an apply-review manifest."""
    label_counts = _label_counts(manifest.get("label_counts"))
    task_families = _task_families(manifest.get("task_families"))
    accepted_count = label_counts.get("accept", 0)
    rejected_count = sum(label_counts.get(label, 0) for label in TRAINING_NEGATIVE_LABELS)
    checks: list[dict[str, Any]] = []

    if require_valid_export:
        _add_validation_check(checks, "valid_reviewed_export", validation_summary)

    if min_reviewed_labels is not None:
        _add_min_check(checks, "min_reviewed_labels", _int_value(manifest.get("reviewed_label_count")), min_reviewed_labels)
    if min_accepted is not None:
        _add_min_check(checks, "min_accepted", accepted_count, min_accepted)
    if min_rejected is not None:
        _add_min_check(checks, "min_rejected", rejected_count, min_rejected)
    if min_sft is not None:
        _add_min_check(checks, "min_sft", _int_value(manifest.get("sft_count")), min_sft)
    if min_reward_model is not None:
        _add_min_check(checks, "min_reward_model", _int_value(manifest.get("reward_model_count")), min_reward_model)
    if min_preferences is not None:
        _add_min_check(checks, "min_preferences", _int_value(manifest.get("preference_count")), min_preferences)
    if min_dpo is not None:
        _add_min_check(checks, "min_dpo", _int_value(manifest.get("dpo_count")), min_dpo)
    if max_needs_review is not None:
        _add_max_check(checks, "max_needs_review", label_counts.get("needs_review", 0), max_needs_review)

    for label in forbid_labels or []:
        _add_absence_check(checks, "forbid_label", label, label_counts.get(label, 0))
    for family in require_task_families or []:
        _add_presence_check(checks, "require_task_family", family in task_families, {"task_family": family})

    passed = all(check["passed"] for check in checks)
    return {
        "schema_version": REVIEWED_GATE_SCHEMA_VERSION,
        "reviewed_export": str(reviewed_export_path),
        "passed": passed,
        "check_count": len(checks),
        "failed_check_count": sum(1 for check in checks if not check["passed"]),
        "checks": checks,
        "metrics": {
            "reviewed_label_count": _int_value(manifest.get("reviewed_label_count")),
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "sft_count": _int_value(manifest.get("sft_count")),
            "reward_model_count": _int_value(manifest.get("reward_model_count")),
            "preference_count": _int_value(manifest.get("preference_count")),
            "dpo_count": _int_value(manifest.get("dpo_count")),
            "label_counts": label_counts,
            "task_families": sorted(task_families),
            "validation": _validation_metrics(validation_summary),
        },
    }


def _add_min_check(
    checks: list[dict[str, Any]],
    check_id: str,
    actual: int,
    minimum: int,
    scope: dict[str, str] | None = None,
) -> None:
    check = {
        "id": check_id,
        "passed": actual >= minimum,
        "actual": actual,
        "expected": {"min": minimum},
        "summary": f"{check_id}: actual={actual}, min={minimum}",
    }
    _append_check(checks, check, scope)


def _add_max_check(
    checks: list[dict[str, Any]],
    check_id: str,
    actual: int,
    maximum: int,
    scope: dict[str, str] | None = None,
) -> None:
    check = {
        "id": check_id,
        "passed": actual <= maximum,
        "actual": actual,
        "expected": {"max": maximum},
        "summary": f"{check_id}: actual={actual}, max={maximum}",
    }
    _append_check(checks, check, scope)


def _add_absence_check(
    checks: list[dict[str, Any]],
    check_id: str,
    item_id: str,
    count: int,
    scope: dict[str, str] | None = None,
) -> None:
    check = {
        "id": check_id,
        "passed": count == 0,
        "actual": {"id": item_id, "count": count},
        "expected": {"count": 0},
        "summary": f"{check_id}: id={item_id}, count={count}",
    }
    _append_check(checks, check, scope)


def _add_presence_check(checks: list[dict[str, Any]], check_id: str, present: bool, scope: dict[str, str]) -> None:
    check = {
        "id": check_id,
        "passed": present,
        "actual": {"present": present},
        "expected": {"present": True},
        "summary": f"{check_id}: present={present}",
    }
    _append_check(checks, check, scope)


def _append_check(checks: list[dict[str, Any]], check: dict[str, Any], scope: dict[str, str] | None = None) -> None:
    if scope:
        check["scope"] = scope
    checks.append(check)


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


def _label_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    counts: dict[str, int] = {}
    for key, count in value.items():
        if not isinstance(key, str) or key not in _REVIEW_LABEL_SET:
            continue
        counts[key] = _int_value(count)
    return counts


def _task_families(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {item for item in value if isinstance(item, str) and item}


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0


def _policy_non_negative_int(field: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReviewedGatePolicyError(f"reviewed gate policy field {field} must be a non-negative integer")
    return value


def _policy_string_list(field: str, value: Any) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ReviewedGatePolicyError(f"reviewed gate policy field {field} must be a list of non-empty strings")
    return list(dict.fromkeys(value))
