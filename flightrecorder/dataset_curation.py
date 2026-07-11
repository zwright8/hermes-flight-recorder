"""Side-effect-free dataset curation receipts for agentic training loops."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

from .atomic_json import atomic_write_json_cas, json_file_sha256
from .path_safety import path_has_symlink_component
from .source_contract import inspect_artifact_source
from .training import episode_events_sha256

DATASET_CURATION_RECEIPT_SCHEMA_VERSION = "hfr.dataset_curation_receipt.v1"
_EXPECTED_SHA256_UNSET = object()


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
    lineage_passed, lineage_actual = training_export_lineage_status(
        [Path(path) for path in rejection_sampling_gate_paths],
        [Path(path) for path in training_export_paths],
    )
    _add_check(
        checks,
        "training_exports_cover_admitted_lineage",
        lineage_passed,
        lineage_actual,
        {
            "training_exports_all_complete": True,
            "reviewed_item_missing_count": 0,
            "reviewed_item_mismatch_count": 0,
            "reviewed_label_mismatch_count": 0,
            "rollout_scenario_missing_count": 0,
            "rollout_scenario_mismatch_count": 0,
        },
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
            "Every selected training export must independently cover the exact reviewed episode-event digests and rollout scenario fingerprints admitted by rejection sampling.",
            "Trainer-ready SFT, reward-model, and DPO roles must preserve each admitted human label, including exclusion of needs_review rows.",
        ],
    }


def training_export_lineage_status(
    rejection_sampling_gate_paths: list[Path],
    training_export_paths: list[Path],
) -> tuple[bool, dict[str, Any]]:
    """Check each trainer export against all reviewed and rollout evidence."""
    reviewed_items: dict[str, dict[str, Any]] = {}
    reviewed_labels: dict[str, dict[str, Any]] = {}
    rollout_scenarios: dict[str, set[str]] = {}
    errors: list[str] = []

    for gate_path in rejection_sampling_gate_paths:
        try:
            if path_has_symlink_component(gate_path, include_leaf=True):
                raise DatasetCurationReceiptError(
                    f"rejection gate path contains a symlink: {gate_path}"
                )
            gate = _read_json_object(gate_path)
            inputs = gate.get("input_artifacts")
            if not isinstance(inputs, dict):
                raise DatasetCurationReceiptError("rejection gate input_artifacts is missing")
            for row in inputs.get("reviewed_gate", []):
                if not isinstance(row, dict):
                    raise DatasetCurationReceiptError("reviewed gate reference is invalid")
                reviewed_gate_path = _resolve_ref(gate_path, row.get("path"))
                reviewed_gate = _read_json_object(reviewed_gate_path)
                sources = reviewed_gate.get("source_artifacts")
                reviewed_export_record = (
                    sources.get("reviewed_export")
                    if isinstance(sources, dict)
                    else None
                )
                if not isinstance(reviewed_export_record, dict):
                    raise DatasetCurationReceiptError(
                        "reviewed gate has no reviewed-export source record"
                    )
                reviewed_export = _resolve_ref(
                    reviewed_gate_path,
                    reviewed_export_record.get("path"),
                )
                for item in _read_jsonl_objects(
                    reviewed_export / "provenance" / "review_items.jsonl"
                ):
                    item_id = item.get("review_item_id")
                    if isinstance(item_id, str) and item_id:
                        existing = reviewed_items.get(item_id)
                        if existing is not None and existing != item:
                            errors.append(
                                f"reviewed item {item_id!r} has conflicting provenance"
                            )
                        reviewed_items[item_id] = item
                for label in _read_jsonl_objects(
                    reviewed_export / "reviewed_labels.jsonl"
                ):
                    item_id = label.get("review_item_id")
                    if isinstance(item_id, str) and item_id:
                        existing = reviewed_labels.get(item_id)
                        identity = {
                            "episode_id": label.get("episode_id"),
                            "human_label": label.get("human_label"),
                            "review_item_sha256": label.get("review_item_sha256"),
                        }
                        if existing is not None and existing != identity:
                            errors.append(
                                f"reviewed item {item_id!r} has conflicting human-label provenance"
                            )
                        reviewed_labels[item_id] = identity
            for row in inputs.get("agentic_rollout_receipt", []):
                if not isinstance(row, dict):
                    raise DatasetCurationReceiptError("rollout receipt reference is invalid")
                receipt_path = _resolve_ref(gate_path, row.get("path"))
                receipt = _read_json_object(receipt_path)
                for rollout in receipt.get("mock_rollouts", []):
                    if not isinstance(rollout, dict):
                        continue
                    scenario_id = rollout.get("scenario_id")
                    scenario_sha256 = rollout.get("scenario_sha256")
                    if (
                        isinstance(scenario_id, str)
                        and scenario_id
                        and isinstance(scenario_sha256, str)
                        and scenario_sha256
                    ):
                        rollout_scenarios.setdefault(scenario_id, set()).add(
                            scenario_sha256
                        )
        except (OSError, UnicodeError, json.JSONDecodeError, DatasetCurationReceiptError) as exc:
            errors.append(f"{gate_path}: {exc}")

    for item_id, label in sorted(reviewed_labels.items()):
        item = reviewed_items.get(item_id)
        if item is None:
            errors.append(f"reviewed label {item_id!r} has no provenance review item")
            continue
        if label.get("episode_id") != item.get("episode_id"):
            errors.append(
                f"reviewed label {item_id!r} episode_id does not match its provenance item"
            )
        if label.get("review_item_sha256") != item.get("review_item_sha256"):
            errors.append(
                f"reviewed label {item_id!r} hash does not match its provenance item"
            )
        if label.get("human_label") not in {
            "accept",
            "reject",
            "unsafe",
            "incomplete",
            "needs_review",
        }:
            errors.append(f"reviewed label {item_id!r} has an unsupported human_label")

    export_results: list[dict[str, Any]] = []
    for export_index, export_path in enumerate(training_export_paths):
        export_errors: list[str] = []
        try:
            if path_has_symlink_component(export_path, include_leaf=True):
                raise DatasetCurationReceiptError(
                    f"training export path contains a symlink: {export_path}"
                )
            episodes = _read_jsonl_objects(export_path / "episodes.jsonl")
            sft = _read_jsonl_objects(export_path / "sft.jsonl")
            reward_model = _read_jsonl_objects(export_path / "reward_model.jsonl")
            dpo = _read_jsonl_objects(export_path / "dpo.jsonl")
        except (OSError, UnicodeError, json.JSONDecodeError, DatasetCurationReceiptError) as exc:
            export_errors.append(str(exc))
            episodes = []
            sft = []
            reward_model = []
            dpo = []
        export_result = _training_export_coverage(
            export_index=export_index,
            episodes=episodes,
            sft=sft,
            reward_model=reward_model,
            dpo=dpo,
            reviewed_items=reviewed_items,
            reviewed_labels=reviewed_labels,
            rollout_scenarios=rollout_scenarios,
            errors=export_errors,
        )
        export_results.append(export_result)
        errors.extend(
            f"training_export[{export_index}]: {error}"
            for error in export_errors
        )

    missing_reviewed = sorted(
        {
            item_id
            for result in export_results
            for item_id in result["missing_reviewed_item_ids"]
        }
    )
    mismatched_reviewed = sorted(
        {
            item_id
            for result in export_results
            for item_id in result["mismatched_reviewed_item_ids"]
        }
    )
    mismatched_labels = sorted(
        {
            item_id
            for result in export_results
            for item_id in result["mismatched_reviewed_label_ids"]
        }
    )
    missing_rollouts = sorted(
        {
            scenario_id
            for result in export_results
            for scenario_id in result["missing_rollout_scenario_ids"]
        }
    )
    mismatched_rollouts = sorted(
        {
            scenario_id
            for result in export_results
            for scenario_id in result["mismatched_rollout_scenario_ids"]
        }
    )

    actual = {
        "training_export_count": len(training_export_paths),
        "training_exports_all_complete": bool(export_results)
        and all(result["passed"] for result in export_results),
        "reviewed_item_count": len(reviewed_labels),
        "reviewed_item_missing_count": sum(
            result["reviewed_item_missing_count"] for result in export_results
        ),
        "reviewed_item_mismatch_count": sum(
            result["reviewed_item_mismatch_count"] for result in export_results
        ),
        "reviewed_label_mismatch_count": sum(
            result["reviewed_label_mismatch_count"] for result in export_results
        ),
        "rollout_scenario_count": len(rollout_scenarios),
        "rollout_scenario_missing_count": sum(
            result["rollout_scenario_missing_count"] for result in export_results
        ),
        "rollout_scenario_mismatch_count": sum(
            result["rollout_scenario_mismatch_count"] for result in export_results
        ),
        "missing_reviewed_item_ids": missing_reviewed,
        "mismatched_reviewed_item_ids": mismatched_reviewed,
        "mismatched_reviewed_label_ids": mismatched_labels,
        "missing_rollout_scenario_ids": missing_rollouts,
        "mismatched_rollout_scenario_ids": mismatched_rollouts,
        "training_exports": export_results,
        "error_count": len(errors),
        "errors": errors,
    }
    passed = (
        bool(reviewed_labels)
        and bool(rollout_scenarios)
        and bool(export_results)
        and all(result["passed"] for result in export_results)
        and not errors
    )
    return passed, actual


def _training_export_coverage(
    *,
    export_index: int,
    episodes: list[dict[str, Any]],
    sft: list[dict[str, Any]],
    reward_model: list[dict[str, Any]],
    dpo: list[dict[str, Any]],
    reviewed_items: dict[str, dict[str, Any]],
    reviewed_labels: dict[str, dict[str, Any]],
    rollout_scenarios: dict[str, set[str]],
    errors: list[str],
) -> dict[str, Any]:
    episodes_by_id = _rows_by_string_field(episodes, "episode_id")
    sft_by_id = _rows_by_string_field(sft, "episode_id")
    reward_model_by_id = _rows_by_string_field(reward_model, "episode_id")
    dpo_chosen_ids = {
        row["chosen_episode_id"]
        for row in dpo
        if isinstance(row.get("chosen_episode_id"), str)
        and row["chosen_episode_id"]
    }
    dpo_rejected_ids = {
        row["rejected_episode_id"]
        for row in dpo
        if isinstance(row.get("rejected_episode_id"), str)
        and row["rejected_episode_id"]
    }

    missing_reviewed: list[str] = []
    behavior_mismatches: list[str] = []
    label_mismatches: list[dict[str, Any]] = []
    for item_id, label in sorted(reviewed_labels.items()):
        item = reviewed_items.get(item_id)
        episode_id = item.get("episode_id") if isinstance(item, dict) else None
        candidates = (
            episodes_by_id.get(episode_id, [])
            if isinstance(episode_id, str) and episode_id
            else []
        )
        if item is None or not candidates:
            missing_reviewed.append(item_id)
            continue
        matching = [
            episode
            for episode in candidates
            if _episode_matches_review_item(episode, item)
        ]
        if len(candidates) != 1 or len(matching) != 1:
            behavior_mismatches.append(item_id)
            continue
        reasons = _review_label_semantic_mismatches(
            human_label=label.get("human_label"),
            episode=matching[0],
            sft_rows=sft_by_id.get(episode_id, []),
            reward_model_rows=reward_model_by_id.get(episode_id, []),
            dpo_chosen=episode_id in dpo_chosen_ids,
            dpo_rejected=episode_id in dpo_rejected_ids,
        )
        if reasons:
            label_mismatches.append(
                {
                    "review_item_id": item_id,
                    "human_label": label.get("human_label"),
                    "reasons": reasons,
                }
            )

    episodes_by_scenario = _rows_by_string_field(episodes, "scenario_id")
    missing_rollouts: list[str] = []
    mismatched_rollouts: list[str] = []
    for scenario_id, expected_hashes in sorted(rollout_scenarios.items()):
        candidates = episodes_by_scenario.get(scenario_id, [])
        if not candidates:
            missing_rollouts.append(scenario_id)
            continue
        actual_hashes = {
            str(
                (
                    episode.get("source_fingerprints", {}).get("scenario", {})
                    if isinstance(episode.get("source_fingerprints"), dict)
                    else {}
                ).get("sha256")
                or ""
            )
            for episode in candidates
        }
        if not expected_hashes.issubset(actual_hashes):
            mismatched_rollouts.append(scenario_id)

    mismatched_label_ids = [
        row["review_item_id"] for row in label_mismatches
    ]
    mismatched_reviewed = sorted(
        set(behavior_mismatches) | set(mismatched_label_ids)
    )
    passed = not any(
        (
            missing_reviewed,
            mismatched_reviewed,
            missing_rollouts,
            mismatched_rollouts,
            errors,
        )
    )
    return {
        "export_index": export_index,
        "passed": passed,
        "reviewed_item_missing_count": len(missing_reviewed),
        "reviewed_item_mismatch_count": len(mismatched_reviewed),
        "reviewed_label_mismatch_count": len(label_mismatches),
        "rollout_scenario_missing_count": len(missing_rollouts),
        "rollout_scenario_mismatch_count": len(mismatched_rollouts),
        "missing_reviewed_item_ids": missing_reviewed,
        "mismatched_reviewed_item_ids": mismatched_reviewed,
        "mismatched_reviewed_label_ids": mismatched_label_ids,
        "reviewed_label_mismatches": label_mismatches,
        "missing_rollout_scenario_ids": missing_rollouts,
        "mismatched_rollout_scenario_ids": mismatched_rollouts,
        "error_count": len(errors),
        "errors": list(errors),
    }


def _rows_by_string_field(
    rows: list[dict[str, Any]],
    field_name: str,
) -> dict[str, list[dict[str, Any]]]:
    indexed: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        value = row.get(field_name)
        if isinstance(value, str) and value:
            indexed.setdefault(value, []).append(row)
    return indexed


def _review_label_semantic_mismatches(
    *,
    human_label: Any,
    episode: dict[str, Any],
    sft_rows: list[dict[str, Any]],
    reward_model_rows: list[dict[str, Any]],
    dpo_chosen: bool,
    dpo_rejected: bool,
) -> list[str]:
    reasons: list[str] = []
    positive_sft = (
        len(sft_rows) == 1
        and _positive_sft_matches_episode(sft_rows[0], episode)
    )
    positive_reward = (
        len(reward_model_rows) == 1
        and _reward_model_matches_episode(
            reward_model_rows[0], episode, positive=True
        )
    )
    negative_reward = (
        len(reward_model_rows) == 1
        and _reward_model_matches_episode(
            reward_model_rows[0], episode, positive=False
        )
    )
    if human_label == "accept":
        if not positive_sft:
            reasons.append("accept_requires_one_positive_sft_row")
        if not positive_reward:
            reasons.append("accept_requires_one_positive_reward_model_row")
        if dpo_rejected:
            reasons.append("accept_forbids_rejected_dpo_role")
    elif human_label in {"reject", "unsafe", "incomplete"}:
        if sft_rows:
            reasons.append("negative_label_forbids_sft_rows")
        if not negative_reward:
            reasons.append("negative_label_requires_one_negative_reward_model_row")
        if dpo_chosen:
            reasons.append("negative_label_forbids_chosen_dpo_role")
    elif human_label == "needs_review":
        if sft_rows:
            reasons.append("needs_review_forbids_sft_rows")
        if reward_model_rows:
            reasons.append("needs_review_forbids_reward_model_rows")
        if dpo_chosen or dpo_rejected:
            reasons.append("needs_review_forbids_dpo_roles")
    else:
        reasons.append("unsupported_human_label")
    return reasons


def _positive_sft_matches_episode(
    row: dict[str, Any],
    episode: dict[str, Any],
) -> bool:
    provenance = row.get("label_provenance")
    return (
        _trainer_row_matches_episode(row, episode)
        and row.get("quality_gate") == "passed_scorecard"
        and isinstance(provenance, dict)
        and provenance.get("label_type") == "sft_positive"
    )


def _reward_model_matches_episode(
    row: dict[str, Any],
    episode: dict[str, Any],
    *,
    positive: bool,
) -> bool:
    provenance = row.get("label_provenance")
    return (
        _trainer_row_matches_episode(row, episode)
        and row.get("passed") is positive
        and isinstance(provenance, dict)
        and provenance.get("label_type")
        == ("reward_model_positive" if positive else "reward_model_negative")
    )


def _trainer_row_matches_episode(
    row: dict[str, Any],
    episode: dict[str, Any],
) -> bool:
    return (
        row.get("episode_id") == episode.get("episode_id")
        and row.get("scenario_id") == episode.get("scenario_id")
        and row.get("task_family") == episode.get("task_family")
        and row.get("prompt") == episode.get("prompt")
        and row.get("response") == episode.get("final_answer")
    )


def _episode_matches_review_item(
    episode: dict[str, Any],
    item: dict[str, Any],
) -> bool:
    outcome = episode.get("outcome")
    scorecard = item.get("scorecard")
    return (
        episode.get("episode_id") == item.get("episode_id")
        and episode.get("scenario_id") == item.get("scenario_id")
        and episode.get("scenario_title") == item.get("scenario_title")
        and episode.get("task_family") == item.get("task_family")
        and episode.get("prompt") == item.get("prompt")
        and episode.get("final_answer") == item.get("final_answer")
        and isinstance(episode.get("events"), list)
        and len(episode["events"]) == item.get("event_count")
        and episode_events_sha256(episode["events"])
        == item.get("episode_events_sha256")
        and isinstance(outcome, dict)
        and isinstance(scorecard, dict)
        and outcome.get("passed") == scorecard.get("passed")
        and outcome.get("score") == scorecard.get("score")
        and outcome.get("failed_rules") == scorecard.get("failed_rules")
        and outcome.get("critical_failures") == scorecard.get("critical_failures")
    )


def _resolve_ref(source_path: Path, value: Any) -> Path:
    if not isinstance(value, str) or not _is_public_dataset_curation_ref_path(value):
        raise DatasetCurationReceiptError("artifact reference must be a safe relative path")
    resolved = source_path.parent / value
    if path_has_symlink_component(resolved, include_leaf=True):
        raise DatasetCurationReceiptError(
            f"artifact reference must not contain symlink components: {value}"
        )
    return resolved


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DatasetCurationReceiptError(f"artifact must contain one JSON object: {path}")
    return value


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise DatasetCurationReceiptError(
                f"JSONL row {line_number} must be an object: {path}"
            )
        rows.append(value)
    return rows


def write_dataset_curation_receipt(
    path: str | Path,
    receipt: dict[str, Any],
    *,
    expected_sha256: str | None | object = _EXPECTED_SHA256_UNSET,
) -> None:
    """Publish a curation receipt with compare-and-swap semantics."""
    out_path = Path(path)
    effective_expected = (
        json_file_sha256(out_path)
        if expected_sha256 is _EXPECTED_SHA256_UNSET
        else expected_sha256
    )
    if effective_expected is not None and not isinstance(effective_expected, str):
        raise DatasetCurationReceiptError("expected_sha256 must be a SHA-256 string or null")
    atomic_write_json_cas(
        out_path,
        receipt,
        expected_sha256=effective_expected,
        new_file_mode=0o666,
    )


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
