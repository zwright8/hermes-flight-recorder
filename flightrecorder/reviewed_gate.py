"""Policy gates over human-reviewed trainer-ready exports."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .gate_contract import build_gate_decision
from .path_safety import (
    _fingerprint_open_directory,
    _open_directory_path_bound,
)
from .review import REVIEW_CONFIDENCE_LEVELS, REVIEW_LABELS, TRAINING_NEGATIVE_LABELS

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
    "min_high_confidence_labels",
    "min_medium_or_high_confidence_labels",
    "max_needs_review",
    "max_low_confidence_labels",
    "max_unknown_confidence_labels",
}
_LIST_POLICY_FIELDS = {"forbid_labels", "require_task_families"}
_BOOLEAN_POLICY_FIELDS = {"require_valid_export", "strict_validation"}
_POLICY_FIELDS = {"schema_version", "description", *_COUNT_POLICY_FIELDS, *_LIST_POLICY_FIELDS, *_BOOLEAN_POLICY_FIELDS}
_REVIEW_LABEL_SET = set(REVIEW_LABELS)
_REVIEW_CONFIDENCE_SET = set(REVIEW_CONFIDENCE_LEVELS)
REVIEWED_EXPORT_SOURCE_ARTIFACT_FIELDS = frozenset(
    {
        "path",
        "kind",
        "exists",
        "tree_hash_algorithm",
        "sha256",
        "file_count",
        "size_bytes",
        "contains_symlinks",
        "manifest_path",
        "manifest_sha256",
        "manifest_size_bytes",
        "dataset_version",
    }
)
REVIEWED_EXPORT_CONTENT_FILES = (
    "dataset_registry.json",
    "manifest.json",
    "provenance/completed_labels.jsonl",
    "provenance/label_template.jsonl",
    "provenance/review_items.jsonl",
    "provenance/review_manifest.json",
    "reviewed_dpo.jsonl",
    "reviewed_action_sft.jsonl",
    "reviewed_labels.jsonl",
    "reviewed_preferences.jsonl",
    "reviewed_reward_model.jsonl",
    "reviewed_sft.jsonl",
)


class ReviewedGatePolicyError(ValueError):
    """Raised when a reviewed-export gate policy file is malformed."""


class ReviewedGateError(ValueError):
    """Raised when a reviewed-export gate cannot be created safely."""


@dataclass(frozen=True)
class ReviewedExportSnapshot:
    """One descriptor-attested reviewed-export source and its exact manifest."""

    source_artifact: dict[str, Any]
    manifest: dict[str, Any]


def snapshot_reviewed_export(
    reviewed_export_path: str | Path,
    *,
    display_path: str,
) -> ReviewedExportSnapshot:
    """Capture a reviewed tree and parse only the manifest bytes bound to it."""
    export_path = Path(reviewed_export_path)
    try:
        with _open_directory_path_bound(export_path) as root_descriptor:
            fingerprint = _fingerprint_open_directory(
                root_descriptor,
                display_path=export_path,
                declared_files=frozenset(REVIEWED_EXPORT_CONTENT_FILES),
                reject_undeclared=True,
                selected_files=frozenset({"manifest.json"}),
                expose_selected_files=True,
            )
            selected_file_sha256 = fingerprint.pop("selected_file_sha256", None)
            descriptor_manifest_sha256 = (
                selected_file_sha256.get("manifest.json")
                if isinstance(selected_file_sha256, dict)
                else None
            )
            if not isinstance(descriptor_manifest_sha256, str):
                raise ValueError("tree fingerprint did not capture manifest.json")
            manifest_bytes = _read_reviewed_manifest_bytes(root_descriptor)
            fingerprint_after = _fingerprint_open_directory(
                root_descriptor,
                display_path=export_path,
                declared_files=frozenset(REVIEWED_EXPORT_CONTENT_FILES),
                reject_undeclared=True,
            )
    except (NotImplementedError, OSError, TypeError, ValueError) as exc:
        raise ReviewedGateError(f"could not snapshot reviewed export {export_path}: {exc}") from exc
    manifest_path = export_path / "manifest.json"
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReviewedGateError(f"could not parse reviewed export manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ReviewedGateError(f"reviewed export manifest must be a JSON object: {manifest_path}")
    dataset_version = manifest.get("dataset_version")
    if not isinstance(dataset_version, str) or not dataset_version:
        raise ReviewedGateError(f"reviewed export manifest dataset_version must be a non-empty string: {manifest_path}")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha256 != descriptor_manifest_sha256:
        raise ReviewedGateError(
            "reviewed export manifest bytes did not match the descriptor-bound "
            f"tree fingerprint: {manifest_path}"
        )
    if fingerprint_after != fingerprint:
        raise ReviewedGateError(
            f"reviewed export changed while its manifest was being captured: {export_path}"
        )
    source_artifact = {
        "path": display_path,
        "kind": "directory",
        "exists": True,
        **fingerprint,
        "manifest_path": f"{display_path.rstrip('/')}/manifest.json",
        "manifest_sha256": manifest_sha256,
        "manifest_size_bytes": len(manifest_bytes),
        "dataset_version": dataset_version,
    }
    return ReviewedExportSnapshot(
        source_artifact=source_artifact,
        manifest=manifest,
    )


def build_reviewed_export_source_artifact(
    reviewed_export_path: str | Path,
    *,
    display_path: str,
) -> dict[str, Any]:
    """Bind a reviewed gate to one exact, relocatable reviewed-export tree."""
    return snapshot_reviewed_export(
        reviewed_export_path,
        display_path=display_path,
    ).source_artifact


def _read_reviewed_manifest_bytes(root_descriptor: int) -> bytes:
    """Read manifest.json through the descriptor used for its tree fingerprint."""
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    before = os.stat(
        "manifest.json",
        dir_fd=root_descriptor,
        follow_symlinks=False,
    )
    descriptor = os.open(
        "manifest.json",
        flags,
        dir_fd=root_descriptor,
    )
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise ValueError("manifest.json changed while being opened")
        chunks: list[bytes] = []
        captured = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            captured += len(chunk)
            if captured > before.st_size:
                raise ValueError("manifest.json grew while being read")
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    namespace_after = os.stat(
        "manifest.json",
        dir_fd=root_descriptor,
        follow_symlinks=False,
    )
    signatures = {
        (
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )
        for item in (before, opened, after, namespace_after)
    }
    if len(signatures) != 1 or captured != before.st_size:
        raise ValueError("manifest.json changed while being read")
    return b"".join(chunks)


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
    reviewed_export_source: dict[str, Any],
    min_reviewed_labels: int | None = None,
    min_accepted: int | None = None,
    min_rejected: int | None = None,
    min_sft: int | None = None,
    min_reward_model: int | None = None,
    min_preferences: int | None = None,
    min_dpo: int | None = None,
    min_high_confidence_labels: int | None = None,
    min_medium_or_high_confidence_labels: int | None = None,
    max_needs_review: int | None = None,
    max_low_confidence_labels: int | None = None,
    max_unknown_confidence_labels: int | None = None,
    forbid_labels: list[str] | None = None,
    require_task_families: list[str] | None = None,
    validation_summary: dict[str, Any] | None = None,
    require_valid_export: bool = True,
    strict_validation: bool = False,
) -> dict[str, Any]:
    """Evaluate readiness checks against an apply-review manifest."""
    label_counts = _label_counts(manifest.get("label_counts"))
    task_families = _task_families(manifest.get("task_families"))
    accepted_count = label_counts.get("accept", 0)
    rejected_count = sum(label_counts.get(label, 0) for label in TRAINING_NEGATIVE_LABELS)
    reviewed_label_count = _int_value(manifest.get("reviewed_label_count"))
    confidence_counts = _confidence_counts(manifest.get("confidence_counts"), reviewed_label_count)
    high_confidence_count = confidence_counts.get("high", 0)
    medium_or_high_confidence_count = high_confidence_count + confidence_counts.get("medium", 0)
    low_confidence_count = confidence_counts.get("low", 0)
    unknown_confidence_count = confidence_counts.get("unknown", 0)
    checks: list[dict[str, Any]] = []

    _add_validation_check(
        checks,
        "valid_reviewed_export",
        validation_summary if require_valid_export else None,
    )

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
    if min_high_confidence_labels is not None:
        _add_min_check(
            checks,
            "min_high_confidence_labels",
            high_confidence_count,
            min_high_confidence_labels,
        )
    if min_medium_or_high_confidence_labels is not None:
        _add_min_check(
            checks,
            "min_medium_or_high_confidence_labels",
            medium_or_high_confidence_count,
            min_medium_or_high_confidence_labels,
        )
    if max_needs_review is not None:
        _add_max_check(checks, "max_needs_review", label_counts.get("needs_review", 0), max_needs_review)
    if max_low_confidence_labels is not None:
        _add_max_check(checks, "max_low_confidence_labels", low_confidence_count, max_low_confidence_labels)
    if max_unknown_confidence_labels is not None:
        _add_max_check(
            checks,
            "max_unknown_confidence_labels",
            unknown_confidence_count,
            max_unknown_confidence_labels,
        )

    for label in forbid_labels or []:
        _add_absence_check(checks, "forbid_label", label, label_counts.get(label, 0))
    for family in require_task_families or []:
        _add_presence_check(checks, "require_task_family", family in task_families, {"task_family": family})

    passed = all(check["passed"] for check in checks)
    gate_metrics = {
        "reviewed_label_count": reviewed_label_count,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "sft_count": _int_value(manifest.get("sft_count")),
        "reward_model_count": _int_value(manifest.get("reward_model_count")),
        "preference_count": _int_value(manifest.get("preference_count")),
        "dpo_count": _int_value(manifest.get("dpo_count")),
        "label_counts": label_counts,
        "confidence_counts": confidence_counts,
        "high_confidence_label_count": high_confidence_count,
        "medium_or_high_confidence_label_count": medium_or_high_confidence_count,
        "low_confidence_label_count": low_confidence_count,
        "unknown_confidence_label_count": unknown_confidence_count,
        "task_families": sorted(task_families),
        "validation": _validation_metrics(validation_summary),
    }
    return {
        "schema_version": REVIEWED_GATE_SCHEMA_VERSION,
        "reviewed_export": str(reviewed_export_path),
        "source_artifacts": {"reviewed_export": dict(reviewed_export_source)},
        "effective_policy": {
            "min_reviewed_labels": min_reviewed_labels,
            "min_accepted": min_accepted,
            "min_rejected": min_rejected,
            "min_sft": min_sft,
            "min_reward_model": min_reward_model,
            "min_preferences": min_preferences,
            "min_dpo": min_dpo,
            "min_high_confidence_labels": min_high_confidence_labels,
            "min_medium_or_high_confidence_labels": min_medium_or_high_confidence_labels,
            "max_needs_review": max_needs_review,
            "max_low_confidence_labels": max_low_confidence_labels,
            "max_unknown_confidence_labels": max_unknown_confidence_labels,
            "forbid_labels": list(forbid_labels or []),
            "require_task_families": list(require_task_families or []),
            "require_valid_export": require_valid_export,
            "strict_validation": strict_validation,
        },
        "passed": passed,
        "check_count": len(checks),
        "failed_check_count": sum(1 for check in checks if not check["passed"]),
        "checks": checks,
        "metrics": gate_metrics,
        "decision": build_gate_decision(passed=passed, checks=checks, metrics=gate_metrics),
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


def _confidence_counts(value: Any, total_labels: int) -> dict[str, int]:
    counts = {level: 0 for level in REVIEW_CONFIDENCE_LEVELS}
    if not isinstance(value, dict):
        counts["unknown"] = total_labels
        return counts
    for key, count in value.items():
        if not isinstance(key, str) or key not in _REVIEW_CONFIDENCE_SET:
            continue
        counts[key] = _int_value(count)
    counted = sum(counts.values())
    if counted < total_labels:
        counts["unknown"] += total_labels - counted
    return counts


def _task_families(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {item for item in value if isinstance(item, str) and item}


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int) and value >= 0:
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
