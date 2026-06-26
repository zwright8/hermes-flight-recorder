"""Artifact validation for Flight Recorder evidence outputs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .adapters import TRACE_SCHEMA_VERSION
from .artifacts import CONTRACT_SCOPES, SUITE_TREND_SCHEMA_VERSION
from .bundle import EVIDENCE_BUNDLE_SCHEMA_VERSION
from .calibration import REVIEW_CALIBRATION_SCHEMA_VERSION
from .evidence import EVIDENCE_COVERAGE_SCHEMA_VERSION
from .lineage import LINEAGE_SCHEMA_VERSION, REPLAY_BUNDLE_SCHEMA_VERSION
from .review import (
    REVIEW_ITEM_SCHEMA_VERSION,
    REVIEW_LABEL_SCHEMA_VERSION,
    REVIEW_LABELS,
    REVIEW_MANIFEST_SCHEMA_VERSION,
    TRAINING_NEGATIVE_LABELS,
    REVIEWED_DPO_SCHEMA_VERSION,
    REVIEWED_LABEL_SCHEMA_VERSION,
    REVIEWED_MANIFEST_SCHEMA_VERSION,
    REVIEWED_PREFERENCE_SCHEMA_VERSION,
    REVIEWED_REWARD_MODEL_SCHEMA_VERSION,
    REVIEWED_SFT_SCHEMA_VERSION,
)
from .scorers import SCORE_SCHEMA_VERSION, TASK_COMPLETION_SCHEMA_VERSION
from .scenario_quality import SCENARIO_QUALITY_SCHEMA_VERSION
from .state_capture import STATE_SNAPSHOT_SCHEMA_VERSION
from .trace_observability import TRACE_OBSERVABILITY_SCHEMA_VERSION
from .training import (
    RL_CURRICULUM_SCHEMA_VERSION,
    RL_DATASET_METRICS_SCHEMA_VERSION,
    RL_DPO_SCHEMA_VERSION,
    RL_EPISODE_SCHEMA_VERSION,
    RL_FAILURE_MODE_SCHEMA_VERSION,
    COMPARE_RL_DPO_SCHEMA_VERSION,
    COMPARE_RL_MANIFEST_SCHEMA_VERSION,
    COMPARE_RL_PAIR_SCHEMA_VERSION,
    RL_MANIFEST_SCHEMA_VERSION,
    RL_PREFERENCE_SCHEMA_VERSION,
    RL_REWARD_SCHEMA_VERSION,
    RL_REWARD_MODEL_SCHEMA_VERSION,
    RL_SFT_SCHEMA_VERSION,
    RL_STEP_REWARD_SCHEMA_VERSION,
    REWARD_SCALES,
)

VALIDATION_SCHEMA_VERSION = "hfr.validation.v1"
RUN_SUITE_SCHEMA_VERSION = "hfr.run_suite.v1"


@dataclass
class ValidationTarget:
    target_type: str
    path: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.target_type,
            "path": self.path,
            "passed": not self.errors,
            "errors": self.errors,
            "warnings": self.warnings,
            "details": self.details,
        }


def validate_artifacts(
    *,
    runs_dir: str | Path | None = None,
    run_dirs: list[str | Path] | None = None,
    training_export_dir: str | Path | None = None,
    compare_export_dir: str | Path | None = None,
    review_export_dir: str | Path | None = None,
    reviewed_export_dir: str | Path | None = None,
    evidence_coverage_paths: list[str | Path] | None = None,
    evidence_bundle_paths: list[str | Path] | None = None,
    replay_bundle_paths: list[str | Path] | None = None,
    trace_observability_paths: list[str | Path] | None = None,
    review_calibration_paths: list[str | Path] | None = None,
    scenario_quality_paths: list[str | Path] | None = None,
    suite_summary_paths: list[str | Path] | None = None,
    suite_trend_paths: list[str | Path] | None = None,
    state_snapshot_paths: list[str | Path] | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Validate generated Flight Recorder run and training artifacts."""
    targets: list[ValidationTarget] = []
    for run_dir in run_dirs or []:
        targets.append(validate_run_dir(run_dir))
    if runs_dir is not None:
        targets.extend(validate_runs_dir(runs_dir))
    if training_export_dir is not None:
        targets.append(validate_training_export(training_export_dir))
    if compare_export_dir is not None:
        targets.append(validate_compare_export(compare_export_dir))
    if review_export_dir is not None:
        targets.append(validate_review_export(review_export_dir))
    if reviewed_export_dir is not None:
        targets.append(validate_reviewed_export(reviewed_export_dir))
    for evidence_coverage_path in evidence_coverage_paths or []:
        targets.append(validate_evidence_coverage(evidence_coverage_path))
    for evidence_bundle_path in evidence_bundle_paths or []:
        targets.append(validate_evidence_bundle(evidence_bundle_path))
    for replay_bundle_path in replay_bundle_paths or []:
        targets.append(validate_replay_bundle(replay_bundle_path))
    for trace_observability_path in trace_observability_paths or []:
        targets.append(validate_trace_observability(trace_observability_path))
    for review_calibration_path in review_calibration_paths or []:
        targets.append(validate_review_calibration(review_calibration_path))
    for scenario_quality_path in scenario_quality_paths or []:
        targets.append(validate_scenario_quality(scenario_quality_path))
    for suite_summary_path in suite_summary_paths or []:
        targets.append(validate_suite_summary(suite_summary_path))
    for suite_trend_path in suite_trend_paths or []:
        targets.append(validate_suite_trend(suite_trend_path))
    for state_snapshot_path in state_snapshot_paths or []:
        targets.append(validate_state_snapshot(state_snapshot_path))
    if not targets:
        target = ValidationTarget("configuration", ".", errors=["No validation targets configured."])
        targets.append(target)

    error_count = sum(len(target.errors) for target in targets)
    warning_count = sum(len(target.warnings) for target in targets)
    passed = error_count == 0 and (warning_count == 0 or not strict)
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "passed": passed,
        "strict": strict,
        "target_count": len(targets),
        "error_count": error_count,
        "warning_count": warning_count,
        "targets": [target.as_dict() for target in targets],
    }


def validate_runs_dir(path: str | Path) -> list[ValidationTarget]:
    """Validate every completed run directory inside a runs directory."""
    root = Path(path)
    if not root.exists():
        return [ValidationTarget("runs", str(root), errors=[f"Runs directory not found: {root}"])]
    if not root.is_dir():
        return [ValidationTarget("runs", str(root), errors=[f"Runs path is not a directory: {root}"])]

    targets: list[ValidationTarget] = []
    for child in sorted(item for item in root.iterdir() if item.is_dir()):
        has_trace = (child / "normalized_trace.json").exists()
        has_scorecard = (child / "scorecard.json").exists()
        if has_trace or has_scorecard:
            targets.append(validate_run_dir(child))
    if not targets:
        targets.append(ValidationTarget("runs", str(root), errors=[f"No completed run directories found in {root}"]))
    return targets


def validate_run_dir(path: str | Path) -> ValidationTarget:
    """Validate one Flight Recorder run directory."""
    run_dir = Path(path)
    target = ValidationTarget("run", str(run_dir))
    if not run_dir.exists():
        target.errors.append(f"Run directory not found: {run_dir}")
        return target
    if not run_dir.is_dir():
        target.errors.append(f"Run path is not a directory: {run_dir}")
        return target

    trace_path = run_dir / "normalized_trace.json"
    score_path = run_dir / "scorecard.json"
    task_completion_path = run_dir / "task_completion.json"
    state_snapshot_path = run_dir / "state_snapshot.json"
    report_path = run_dir / "report.html"
    lineage_path = run_dir / "artifact_lineage.json"
    trace = _read_object(trace_path, target, "normalized_trace.json")
    scorecard = _read_object(score_path, target, "scorecard.json")
    task_completion = _read_object_optional(
        task_completion_path,
        target,
        "task_completion.json",
        "rerun the run to emit the standalone task-completion verdict",
    )
    lineage = _read_object_optional(lineage_path, target, "artifact_lineage.json", "rerun the run to emit provenance metadata")
    state_snapshot = _read_object(state_snapshot_path, target, "state_snapshot.json") if state_snapshot_path.exists() else None
    if trace is not None:
        _validate_trace(trace, target)
    if scorecard is not None:
        _validate_scorecard(scorecard, target)
    if task_completion is not None:
        _validate_task_completion(task_completion, target, "task_completion")
        if isinstance(scorecard, dict) and scorecard.get("task_completion") != task_completion:
            target.errors.append("task_completion.json must match scorecard.task_completion.")
    if lineage is not None:
        _validate_lineage(lineage, target, run_dir, trace, scorecard)
    if isinstance(state_snapshot, dict) and state_snapshot.get("schema_version") == STATE_SNAPSHOT_SCHEMA_VERSION:
        _validate_state_snapshot(state_snapshot, target, "state_snapshot")
    if trace is not None and scorecard is not None:
        target.details.update(
            {
                "scenario_id": scorecard.get("scenario_id"),
                "score": scorecard.get("score"),
                "passed": scorecard.get("passed"),
                "event_count": len(trace.get("events", [])) if isinstance(trace.get("events"), list) else None,
            }
        )
    if not report_path.exists():
        target.warnings.append("report.html is missing; run evidence is less reviewable.")
    if isinstance(scorecard, dict) and scorecard.get("passed") is False and not (run_dir / "regression_scenario.json").exists():
        target.warnings.append("failing run is missing regression_scenario.json.")
    return target


def validate_training_export(path: str | Path) -> ValidationTarget:
    """Validate an RL/training export directory."""
    export_dir = Path(path)
    target = ValidationTarget("training_export", str(export_dir))
    if not export_dir.exists():
        target.errors.append(f"Training export directory not found: {export_dir}")
        return target
    if not export_dir.is_dir():
        target.errors.append(f"Training export path is not a directory: {export_dir}")
        return target

    manifest = _read_object(export_dir / "manifest.json", target, "manifest.json")
    episodes = _read_jsonl_objects(export_dir / "episodes.jsonl", target, "episodes.jsonl")
    rewards = _read_jsonl_objects(export_dir / "rewards.jsonl", target, "rewards.jsonl")
    step_rewards = _read_jsonl_objects_optional(export_dir / "step_rewards.jsonl", target, "step_rewards.jsonl")
    preferences = _read_jsonl_objects(export_dir / "preferences.jsonl", target, "preferences.jsonl")
    failure_modes = _read_jsonl_objects(export_dir / "failure_modes.jsonl", target, "failure_modes.jsonl")
    curriculum = _read_object(export_dir / "curriculum.json", target, "curriculum.json")
    sft_path = export_dir / "sft.jsonl"
    dpo_path = export_dir / "dpo.jsonl"
    reward_model_path = export_dir / "reward_model.jsonl"
    dataset_metrics_path = export_dir / "dataset_metrics.json"
    dataset_card_path = export_dir / "DATASET_CARD.md"
    sft = _read_jsonl_objects_optional(sft_path, target, "sft.jsonl", "rerun export-rl to emit trainer-ready SFT rows")
    dpo = _read_jsonl_objects_optional(dpo_path, target, "dpo.jsonl", "rerun export-rl to emit trainer-ready DPO rows")
    reward_model = _read_jsonl_objects_optional(
        reward_model_path,
        target,
        "reward_model.jsonl",
        "rerun export-rl to emit trainer-ready reward-model rows",
    )
    dataset_metrics = _read_object_optional(
        dataset_metrics_path,
        target,
        "dataset_metrics.json",
        "rerun export-rl to emit dataset-level metrics",
    )
    if manifest is not None:
        _validate_training_manifest(
            manifest,
            target,
            episodes,
            rewards,
            step_rewards,
            preferences,
            failure_modes,
            curriculum,
            sft,
            dpo,
            reward_model,
            dataset_metrics,
            dataset_card_path.exists(),
        )
    _validate_episodes(episodes, target)
    _validate_rewards(rewards, target, episodes)
    _validate_step_rewards(step_rewards, target, episodes, rewards)
    _validate_preferences(preferences, target, episodes)
    _validate_failure_modes(failure_modes, target, episodes)
    if curriculum is not None:
        _validate_curriculum(curriculum, target, episodes, failure_modes)
    if sft_path.exists():
        _validate_sft_records(sft, target, episodes)
    if dpo_path.exists():
        _validate_dpo_records(dpo, target, preferences, episodes)
    if reward_model_path.exists():
        _validate_reward_model_records(reward_model, target, episodes)
    if dataset_metrics is not None:
        _validate_dataset_metrics(
            dataset_metrics,
            target,
            episodes,
            rewards,
            step_rewards,
            preferences,
            failure_modes,
            sft,
            dpo,
            reward_model,
        )
    if dataset_card_path.exists():
        _validate_dataset_card(dataset_card_path, target)
    else:
        target.warnings.append("DATASET_CARD.md is missing; rerun export-rl to emit the human-readable dataset card.")
    target.details.update(
        {
            "episode_count": len(episodes),
            "reward_count": len(rewards),
            "step_reward_count": len(step_rewards),
            "preference_count": len(preferences),
            "failure_mode_count": len(failure_modes),
            "sft_count": len(sft),
            "dpo_count": len(dpo),
            "reward_model_count": len(reward_model),
            "quality_flag_count": len(dataset_metrics.get("quality_flags", [])) if isinstance(dataset_metrics, dict) else None,
        }
    )
    return target


def validate_compare_export(path: str | Path) -> ValidationTarget:
    """Validate an export-compare-rl output directory."""
    export_dir = Path(path)
    target = ValidationTarget("compare_export", str(export_dir))
    if not export_dir.exists():
        target.errors.append(f"Compare export directory not found: {export_dir}")
        return target
    if not export_dir.is_dir():
        target.errors.append(f"Compare export path is not a directory: {export_dir}")
        return target

    manifest = _read_object(export_dir / "manifest.json", target, "manifest.json")
    pairs = _read_jsonl_objects(export_dir / "improvement_pairs.jsonl", target, "improvement_pairs.jsonl")
    dpo = _read_jsonl_objects(export_dir / "improvement_dpo.jsonl", target, "improvement_dpo.jsonl")
    card_path = export_dir / "IMPROVEMENT_CARD.md"
    if manifest is not None:
        _validate_compare_manifest(manifest, target, pairs, dpo, card_path.exists())
    _validate_compare_pairs(pairs, target)
    _validate_compare_dpo(dpo, target, pairs)
    if card_path.exists():
        _validate_improvement_card(card_path, target)
    else:
        target.warnings.append("IMPROVEMENT_CARD.md is missing; comparison export is less reviewable.")
    target.details.update(
        {
            "pair_count": len(pairs),
            "dpo_count": len(dpo),
            "candidate_win_count": sum(1 for pair in pairs if pair.get("chosen_side") == "candidate"),
            "baseline_win_count": sum(1 for pair in pairs if pair.get("chosen_side") == "baseline"),
        }
    )
    return target


def validate_review_export(path: str | Path) -> ValidationTarget:
    """Validate a human-review export directory."""
    export_dir = Path(path)
    target = ValidationTarget("review_export", str(export_dir))
    if not export_dir.exists():
        target.errors.append(f"Review export directory not found: {export_dir}")
        return target
    if not export_dir.is_dir():
        target.errors.append(f"Review export path is not a directory: {export_dir}")
        return target

    manifest = _read_object(export_dir / "manifest.json", target, "manifest.json")
    items = _read_jsonl_objects(export_dir / "review_items.jsonl", target, "review_items.jsonl")
    labels = _read_jsonl_objects(export_dir / "label_template.jsonl", target, "label_template.jsonl")
    if not (export_dir / "REVIEW_INSTRUCTIONS.md").exists():
        target.warnings.append("REVIEW_INSTRUCTIONS.md is missing; review workflow is less self-documenting.")
    if manifest is not None:
        _validate_review_manifest(manifest, target, items, labels)
    _validate_review_items(items, target)
    _validate_review_labels(labels, target, items)
    target.details.update({"item_count": len(items), "label_count": len(labels)})
    return target


def validate_reviewed_export(path: str | Path) -> ValidationTarget:
    """Validate an apply-review output directory."""
    export_dir = Path(path)
    target = ValidationTarget("reviewed_export", str(export_dir))
    if not export_dir.exists():
        target.errors.append(f"Reviewed export directory not found: {export_dir}")
        return target
    if not export_dir.is_dir():
        target.errors.append(f"Reviewed export path is not a directory: {export_dir}")
        return target

    manifest = _read_object(export_dir / "manifest.json", target, "manifest.json")
    labels = _read_jsonl_objects(export_dir / "reviewed_labels.jsonl", target, "reviewed_labels.jsonl")
    sft = _read_jsonl_objects(export_dir / "reviewed_sft.jsonl", target, "reviewed_sft.jsonl")
    reward_model = _read_jsonl_objects(export_dir / "reviewed_reward_model.jsonl", target, "reviewed_reward_model.jsonl")
    preferences = _read_jsonl_objects(export_dir / "reviewed_preferences.jsonl", target, "reviewed_preferences.jsonl")
    dpo = _read_jsonl_objects(export_dir / "reviewed_dpo.jsonl", target, "reviewed_dpo.jsonl")
    if manifest is not None:
        _validate_reviewed_manifest(manifest, target, labels, sft, reward_model, preferences, dpo)
    _validate_reviewed_labels(labels, target)
    _validate_reviewed_sft(sft, target, labels)
    _validate_reviewed_reward_model(reward_model, target, labels)
    _validate_reviewed_preferences(preferences, target, labels)
    _validate_reviewed_dpo(dpo, target, preferences)
    target.details.update(
        {
            "reviewed_label_count": len(labels),
            "sft_count": len(sft),
            "reward_model_count": len(reward_model),
            "preference_count": len(preferences),
            "dpo_count": len(dpo),
        }
    )
    return target


def validate_suite_summary(path: str | Path) -> ValidationTarget:
    """Validate one run-suite summary artifact."""
    summary_path = Path(path)
    target = ValidationTarget("suite_summary", str(summary_path))
    summary = _read_object(summary_path, target, "suite_summary.json")
    if summary is None:
        return target
    _validate_suite_summary(summary, target)
    return target


def validate_evidence_coverage(path: str | Path) -> ValidationTarget:
    """Validate an evidence-coverage artifact."""
    coverage_path = Path(path)
    target = ValidationTarget("evidence_coverage", str(coverage_path))
    coverage = _read_object(coverage_path, target, "evidence_coverage.json")
    if coverage is not None:
        _validate_evidence_coverage(coverage, target)
    return target


def validate_evidence_bundle(path: str | Path) -> ValidationTarget:
    """Validate an evidence-bundle handoff summary artifact."""
    bundle_path = Path(path)
    target = ValidationTarget("evidence_bundle", str(bundle_path))
    bundle = _read_object(bundle_path, target, "evidence_bundle.json")
    if bundle is not None:
        _validate_evidence_bundle(bundle, target)
    return target


def validate_replay_bundle(path: str | Path) -> ValidationTarget:
    """Validate a portable replay-bundle directory or replay_bundle.json artifact."""
    raw_path = Path(path)
    bundle_dir = raw_path if raw_path.is_dir() else raw_path.parent
    manifest_path = raw_path / "replay_bundle.json" if raw_path.is_dir() else raw_path
    target = ValidationTarget("replay_bundle", str(raw_path))
    manifest = _read_object(manifest_path, target, "replay_bundle.json")
    if manifest is not None:
        _validate_replay_bundle(manifest, bundle_dir, target)
    return target


def validate_trace_observability(path: str | Path) -> ValidationTarget:
    """Validate a trace-observability summary artifact."""
    observability_path = Path(path)
    target = ValidationTarget("trace_observability", str(observability_path))
    observability = _read_object(observability_path, target, "trace_observability.json")
    if observability is not None:
        _validate_trace_observability(observability, target)
    return target


def validate_state_snapshot(path: str | Path) -> ValidationTarget:
    """Validate a captured state snapshot artifact."""
    snapshot_path = Path(path)
    target = ValidationTarget("state_snapshot", str(snapshot_path))
    snapshot = _read_object(snapshot_path, target, "state_snapshot.json")
    if snapshot is not None:
        _validate_state_snapshot(snapshot, target, "state_snapshot", source_path=snapshot_path)
    return target


def validate_review_calibration(path: str | Path) -> ValidationTarget:
    """Validate a review-calibration report."""
    calibration_path = Path(path)
    target = ValidationTarget("review_calibration", str(calibration_path))
    calibration = _read_object(calibration_path, target, "review_calibration.json")
    if calibration is not None:
        _validate_review_calibration(calibration, target)
    return target


def validate_scenario_quality(path: str | Path) -> ValidationTarget:
    """Validate a scenario-quality artifact."""
    quality_path = Path(path)
    target = ValidationTarget("scenario_quality", str(quality_path))
    quality = _read_object(quality_path, target, "scenario_quality.json")
    if quality is not None:
        _validate_scenario_quality(quality, target)
    return target


def validate_suite_trend(path: str | Path) -> ValidationTarget:
    """Validate one trend-suite artifact."""
    trend_path = Path(path)
    target = ValidationTarget("suite_trend", str(trend_path))
    trend = _read_object(trend_path, target, "suite_trend.json")
    if trend is None:
        return target
    _validate_suite_trend(trend, target)
    return target


def _validate_state_snapshot(
    snapshot: dict[str, Any],
    target: ValidationTarget,
    label: str,
    *,
    source_path: Path | None = None,
) -> None:
    _require_equal(snapshot, "schema_version", STATE_SNAPSHOT_SCHEMA_VERSION, target, prefix=f"{label}.")
    filesystem = snapshot.get("filesystem")
    if not isinstance(filesystem, dict):
        target.errors.append(f"{label}.filesystem must be an object.")
        filesystem = {}
    files = filesystem.get("files")
    if not isinstance(files, dict):
        target.errors.append(f"{label}.filesystem.files must be an object.")
        files = {}
    directories = filesystem.get("directories")
    if not isinstance(directories, dict):
        target.errors.append(f"{label}.filesystem.directories must be an object.")
        directories = {}
    for key, record in files.items():
        if not isinstance(key, str) or not key:
            target.errors.append(f"{label}.filesystem.files contains an empty source key.")
            continue
        _validate_snapshot_file_record(record, target, f"{label}.filesystem.files.{key}", source_path)
    for key, record in directories.items():
        if not isinstance(key, str) or not key:
            target.errors.append(f"{label}.filesystem.directories contains an empty source key.")
            continue
        _validate_snapshot_directory_record(record, target, f"{label}.filesystem.directories.{key}")

    json_sources = snapshot.get("json_sources")
    if not isinstance(json_sources, dict):
        target.errors.append(f"{label}.json_sources must be an object.")
        json_sources = {}
    json_payloads = snapshot.get("json")
    if not isinstance(json_payloads, dict):
        target.errors.append(f"{label}.json must be an object.")
        json_payloads = {}
    for key, record in json_sources.items():
        if not isinstance(key, str) or not key:
            target.errors.append(f"{label}.json_sources contains an empty source key.")
            continue
        _validate_snapshot_file_record(record, target, f"{label}.json_sources.{key}", source_path)
        if isinstance(record, dict) and record.get("exists") is True and record.get("kind") == "file" and key not in json_payloads:
            target.errors.append(f"{label}.json.{key} must contain imported JSON for an existing JSON source.")
    observations = snapshot.get("observations")
    if not isinstance(observations, dict):
        target.errors.append(f"{label}.observations must be an object.")


def _validate_snapshot_file_record(
    record: Any,
    target: ValidationTarget,
    label: str,
    source_path: Path | None,
) -> None:
    if not isinstance(record, dict):
        target.errors.append(f"{label} must be an object.")
        return
    path_label = record.get("path")
    if not isinstance(path_label, str) or not path_label:
        target.errors.append(f"{label}.path must be a non-empty string.")
    exists = record.get("exists")
    if not isinstance(exists, bool):
        target.errors.append(f"{label}.exists must be a boolean.")
    kind = record.get("kind")
    if kind not in {"missing", "file", "directory", "other"}:
        target.errors.append(f"{label}.kind must be missing, file, directory, or other.")
    if exists is False and kind != "missing":
        target.errors.append(f"{label}.kind must be missing when exists is false.")
    if exists is True and kind == "missing":
        target.errors.append(f"{label}.kind must not be missing when exists is true.")
    if kind == "file":
        if not _is_non_negative_int(record.get("size_bytes")):
            target.errors.append(f"{label}.size_bytes must be a non-negative integer for files.")
        if not _is_sha256(record.get("sha256")):
            target.errors.append(f"{label}.sha256 must be a 64-character hex digest for files.")
        if "text" in record and not isinstance(record.get("text"), str):
            target.errors.append(f"{label}.text must be a string when present.")
        if "text_truncated" in record and not isinstance(record.get("text_truncated"), bool):
            target.errors.append(f"{label}.text_truncated must be a boolean when present.")
        _validate_snapshot_record_hash(record, target, label, source_path)


def _validate_snapshot_directory_record(record: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(record, dict):
        target.errors.append(f"{label} must be an object.")
        return
    path_label = record.get("path")
    if not isinstance(path_label, str) or not path_label:
        target.errors.append(f"{label}.path must be a non-empty string.")
    exists = record.get("exists")
    if not isinstance(exists, bool):
        target.errors.append(f"{label}.exists must be a boolean.")
    kind = record.get("kind")
    if kind not in {"missing", "directory", "not_directory"}:
        target.errors.append(f"{label}.kind must be missing, directory, or not_directory.")
    if kind == "directory":
        entry_count = record.get("entry_count")
        if not _is_non_negative_int(entry_count):
            target.errors.append(f"{label}.entry_count must be a non-negative integer.")
            entry_count = None
        if not isinstance(record.get("entries_truncated"), bool):
            target.errors.append(f"{label}.entries_truncated must be a boolean.")
        entries = record.get("entries")
        if not isinstance(entries, list):
            target.errors.append(f"{label}.entries must be a list.")
            entries = []
        if isinstance(entry_count, int) and len(entries) > entry_count:
            target.errors.append(f"{label}.entries length must not exceed entry_count.")
        for index, entry in enumerate(entries):
            _validate_snapshot_directory_entry(entry, target, f"{label}.entries[{index}]")


def _validate_snapshot_directory_entry(entry: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(entry, dict):
        target.errors.append(f"{label} must be an object.")
        return
    if not isinstance(entry.get("name"), str) or not entry.get("name"):
        target.errors.append(f"{label}.name must be a non-empty string.")
    kind = entry.get("kind")
    if kind not in {"file", "directory", "other", "missing"}:
        target.errors.append(f"{label}.kind must be file, directory, other, or missing.")
    if kind == "file":
        if not _is_non_negative_int(entry.get("size_bytes")):
            target.errors.append(f"{label}.size_bytes must be a non-negative integer for files.")
        if not _is_sha256(entry.get("sha256")):
            target.errors.append(f"{label}.sha256 must be a 64-character hex digest for files.")


def _validate_snapshot_record_hash(
    record: dict[str, Any],
    target: ValidationTarget,
    label: str,
    source_path: Path | None,
) -> None:
    path_label = record.get("path")
    if not isinstance(path_label, str) or path_label.startswith("<redacted:"):
        return
    path = Path(path_label)
    if not path.is_absolute():
        candidates = [path.resolve()]
        if source_path is not None:
            candidates.append((source_path.parent / path).resolve())
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    if not path.exists() or not path.is_file():
        return
    expected = record.get("sha256")
    if _is_sha256(expected):
        actual = _sha256(path)
        if actual != expected:
            target.errors.append(f"{label}.sha256 does not match current file contents.")


def _validate_trace(trace: dict[str, Any], target: ValidationTarget) -> None:
    _require_equal(trace, "schema_version", TRACE_SCHEMA_VERSION, target)
    session = trace.get("session")
    if not isinstance(session, dict):
        target.errors.append("trace.session must be an object.")
    else:
        for field_name in ("id", "source_format", "model"):
            if not isinstance(session.get(field_name), str) or not session.get(field_name):
                target.errors.append(f"trace.session.{field_name} must be a non-empty string.")
    events = trace.get("events")
    if not isinstance(events, list):
        target.errors.append("trace.events must be a list.")
        events = []
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            target.errors.append(f"trace.events[{index}] must be an object.")
            continue
        if not isinstance(event.get("type"), str) or not event.get("type"):
            target.errors.append(f"trace.events[{index}].type must be a non-empty string.")
        if "session_id" in event and event.get("session_id") is not None and not isinstance(event.get("session_id"), str):
            target.errors.append(f"trace.events[{index}].session_id must be a string or null.")
        if "args" in event and not isinstance(event.get("args"), dict):
            target.errors.append(f"trace.events[{index}].args must be an object when present.")
    if not isinstance(trace.get("final_answer", ""), str):
        target.errors.append("trace.final_answer must be a string.")


def _validate_scorecard(scorecard: dict[str, Any], target: ValidationTarget) -> None:
    _require_equal(scorecard, "schema_version", SCORE_SCHEMA_VERSION, target)
    for field_name in ("scenario_id", "scenario_title", "summary"):
        if not isinstance(scorecard.get(field_name), str) or not scorecard.get(field_name):
            target.errors.append(f"scorecard.{field_name} must be a non-empty string.")
    score = scorecard.get("score")
    threshold = scorecard.get("pass_threshold")
    if not _is_int_between(score, 0, 100):
        target.errors.append("scorecard.score must be an integer from 0 to 100.")
    if not _is_int_between(threshold, 0, 100):
        target.errors.append("scorecard.pass_threshold must be an integer from 0 to 100.")
    if not isinstance(scorecard.get("passed"), bool):
        target.errors.append("scorecard.passed must be a boolean.")
    critical_failures = scorecard.get("critical_failures")
    if not _is_string_list(critical_failures):
        target.errors.append("scorecard.critical_failures must be a list of strings.")
        critical_failures = []
    rules = scorecard.get("rules")
    if not isinstance(rules, list):
        target.errors.append("scorecard.rules must be a list.")
        rules = []

    failed_critical: list[str] = []
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            target.errors.append(f"scorecard.rules[{index}] must be an object.")
            continue
        rule_id = rule.get("id")
        if not isinstance(rule_id, str) or not rule_id:
            target.errors.append(f"scorecard.rules[{index}].id must be a non-empty string.")
            rule_id = f"rule[{index}]"
        if not isinstance(rule.get("name"), str) or not rule.get("name"):
            target.errors.append(f"scorecard.rules[{index}].name must be a non-empty string.")
        if not isinstance(rule.get("passed"), bool):
            target.errors.append(f"scorecard.rules[{index}].passed must be a boolean.")
        if not isinstance(rule.get("critical"), bool):
            target.errors.append(f"scorecard.rules[{index}].critical must be a boolean.")
        if not _is_int_between(rule.get("penalty"), 0, 100):
            target.errors.append(f"scorecard.rules[{index}].penalty must be an integer from 0 to 100.")
        if not isinstance(rule.get("evidence"), list):
            target.errors.append(f"scorecard.rules[{index}].evidence must be a list.")
        if "evidence_refs" in rule:
            _validate_evidence_refs(rule.get("evidence_refs"), target, f"scorecard.rules[{index}].evidence_refs")
        if rule.get("critical") is True and rule.get("passed") is False:
            failed_critical.append(str(rule_id))

    if isinstance(critical_failures, list) and sorted(critical_failures) != sorted(failed_critical):
        target.errors.append(
            "scorecard.critical_failures must match failed critical rules: "
            f"expected {sorted(failed_critical)!r}, got {sorted(critical_failures)!r}."
        )
    if _is_int_between(score, 0, 100) and _is_int_between(threshold, 0, 100) and isinstance(scorecard.get("passed"), bool):
        expected_passed = int(score) >= int(threshold) and not failed_critical
        if scorecard["passed"] != expected_passed:
            target.errors.append(
                f"scorecard.passed inconsistent with score/threshold/critical failures: expected {expected_passed!r}."
            )
    if "task_completion" in scorecard:
        _validate_task_completion(scorecard.get("task_completion"), target, "scorecard.task_completion")
    else:
        target.warnings.append("scorecard.task_completion is missing; rerun scoring to emit task-completion evidence.")


def _validate_task_completion(value: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _require_equal(value, "schema_version", TASK_COMPLETION_SCHEMA_VERSION, target, prefix=f"{label}.")
    status = value.get("status")
    if status not in {"complete", "incomplete", "not_applicable"}:
        target.errors.append(f"{label}.status must be complete, incomplete, or not_applicable.")
    passed = value.get("passed")
    if not isinstance(passed, bool):
        target.errors.append(f"{label}.passed must be a boolean.")
    elif status == "complete" and passed is not True:
        target.errors.append(f"{label}.passed must be true when status is complete.")
    elif status == "incomplete" and passed is not False:
        target.errors.append(f"{label}.passed must be false when status is incomplete.")
    elif status == "not_applicable" and passed is not True:
        target.errors.append(f"{label}.passed must be true when status is not_applicable.")
    configured = value.get("task_evidence_configured")
    if not isinstance(configured, bool):
        target.errors.append(f"{label}.task_evidence_configured must be a boolean.")
    elif configured is False and status != "not_applicable":
        target.errors.append(f"{label}.status must be not_applicable when no task evidence is configured.")
    elif configured is True and status == "not_applicable":
        target.errors.append(f"{label}.status must not be not_applicable when task evidence is configured.")
    counts: dict[str, int] = {}
    for field_name in ("required_check_count", "passed_check_count", "failed_check_count"):
        raw_count = value.get(field_name)
        if not _is_non_negative_int(raw_count):
            target.errors.append(f"{label}.{field_name} must be a non-negative integer.")
            continue
        counts[field_name] = int(raw_count)
    if len(counts) == 3:
        if counts["passed_check_count"] + counts["failed_check_count"] != counts["required_check_count"]:
            target.errors.append(f"{label}.passed_check_count + failed_check_count must equal required_check_count.")
        if counts["failed_check_count"] == 0 and status == "incomplete":
            target.errors.append(f"{label}.status must not be incomplete when failed_check_count is 0.")
        if counts["failed_check_count"] > 0 and status == "complete":
            target.errors.append(f"{label}.status must not be complete when failed_check_count is greater than 0.")
    if not _is_string_list(value.get("blocking_rule_ids")):
        target.errors.append(f"{label}.blocking_rule_ids must be a list of strings.")
    if not isinstance(value.get("summary"), str) or not value.get("summary"):
        target.errors.append(f"{label}.summary must be a non-empty string.")
    checks = value.get("checks")
    if not isinstance(checks, list):
        target.errors.append(f"{label}.checks must be a list.")
        checks = []
    if "required_check_count" in counts and len(checks) != counts["required_check_count"]:
        target.errors.append(f"{label}.checks length must match required_check_count.")
    observed_failed = 0
    for index, check in enumerate(checks):
        if not isinstance(check, dict):
            target.errors.append(f"{label}.checks[{index}] must be an object.")
            continue
        for field_name in ("id", "rule_id", "description", "evidence"):
            if not isinstance(check.get(field_name), str) or not check.get(field_name):
                target.errors.append(f"{label}.checks[{index}].{field_name} must be a non-empty string.")
        if not isinstance(check.get("passed"), bool):
            target.errors.append(f"{label}.checks[{index}].passed must be a boolean.")
        elif check.get("passed") is False:
            observed_failed += 1
        if "event_indices" in check and not all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in check.get("event_indices", [])):
            target.errors.append(f"{label}.checks[{index}].event_indices must be a list of non-negative integers.")
        _validate_evidence_refs(check.get("evidence_refs"), target, f"{label}.checks[{index}].evidence_refs")
    if "failed_check_count" in counts and observed_failed != counts["failed_check_count"]:
        target.errors.append(f"{label}.failed_check_count must match failed checks.")
    _validate_evidence_refs(value.get("evidence_refs"), target, f"{label}.evidence_refs")
    _validate_evidence_refs(value.get("missing_evidence_refs"), target, f"{label}.missing_evidence_refs")


def _validate_lineage(
    lineage: dict[str, Any],
    target: ValidationTarget,
    run_dir: Path,
    trace: dict[str, Any] | None,
    scorecard: dict[str, Any] | None,
) -> None:
    _require_equal(lineage, "schema_version", LINEAGE_SCHEMA_VERSION, target, prefix="artifact_lineage.")
    scenario = lineage.get("scenario")
    if not isinstance(scenario, dict):
        target.errors.append("artifact_lineage.scenario must be an object.")
        scenario = {}
    if scorecard is not None and scenario.get("id") != scorecard.get("scenario_id"):
        target.errors.append("artifact_lineage.scenario.id must match scorecard.scenario_id.")
    lineage_trace = lineage.get("trace")
    if not isinstance(lineage_trace, dict):
        target.errors.append("artifact_lineage.trace must be an object.")
        lineage_trace = {}
    events = trace.get("events", []) if isinstance(trace, dict) and isinstance(trace.get("events"), list) else []
    if trace is not None and lineage_trace.get("event_count") != len(events):
        target.errors.append("artifact_lineage.trace.event_count must match normalized_trace.events.")
    lineage_scorecard = lineage.get("scorecard")
    if not isinstance(lineage_scorecard, dict):
        target.errors.append("artifact_lineage.scorecard must be an object.")
        lineage_scorecard = {}
    if scorecard is not None:
        for field_name in ("score", "passed", "critical_failures"):
            if lineage_scorecard.get(field_name) != scorecard.get(field_name):
                target.errors.append(f"artifact_lineage.scorecard.{field_name} must match scorecard.{field_name}.")
    inputs = _lineage_records(lineage.get("inputs"), target, "artifact_lineage.inputs")
    for name, record in inputs.items():
        _validate_lineage_input_record(name, record, target, f"artifact_lineage.inputs.{name}")
    outputs = _lineage_records(lineage.get("outputs"), target, "artifact_lineage.outputs")
    for output_name in ("normalized_trace", "scorecard", "task_completion", "report"):
        if output_name not in outputs:
            target.errors.append(f"artifact_lineage.outputs missing {output_name!r}.")
            continue
        _validate_lineage_file_record(outputs[output_name], run_dir, target, f"artifact_lineage.outputs.{output_name}")
    if "state_snapshot" in outputs:
        _validate_lineage_file_record(outputs["state_snapshot"], run_dir, target, "artifact_lineage.outputs.state_snapshot")
    evidence_links = lineage.get("evidence_links")
    if not isinstance(evidence_links, list):
        target.errors.append("artifact_lineage.evidence_links must be a list.")
        evidence_links = []
    expected_ref_count = _scorecard_evidence_ref_count(scorecard)
    if scorecard is not None and len(evidence_links) != expected_ref_count:
        target.errors.append(
            f"artifact_lineage.evidence_links expected {expected_ref_count}, got {len(evidence_links)}."
        )
    for index, link in enumerate(evidence_links):
        _validate_lineage_evidence_link(link, index, len(events), target)
    graph = lineage.get("graph")
    if not isinstance(graph, list):
        target.errors.append("artifact_lineage.graph must be a list.")
    replay = lineage.get("replay")
    if isinstance(replay, dict):
        _validate_lineage_replay(replay, inputs, target)
    else:
        target.warnings.append("artifact_lineage.replay is missing; rerun the run to emit replay instructions.")
    summary = lineage.get("summary")
    if not isinstance(summary, dict):
        target.errors.append("artifact_lineage.summary must be an object.")
    else:
        inputs_raw = lineage.get("inputs")
        expected_input_count = len(inputs_raw) if isinstance(inputs_raw, list) else None
        if summary.get("input_count") != expected_input_count:
            target.errors.append("artifact_lineage.summary.input_count must match inputs length.")
        outputs_raw = lineage.get("outputs")
        expected_output_count = len(outputs_raw) if isinstance(outputs_raw, list) else None
        if summary.get("output_count") != expected_output_count:
            target.errors.append("artifact_lineage.summary.output_count must match outputs length.")
        if summary.get("evidence_link_count") != len(evidence_links):
            target.errors.append("artifact_lineage.summary.evidence_link_count must match evidence_links length.")
        if isinstance(replay, dict) and summary.get("self_contained_replay") != replay.get("self_contained"):
            target.errors.append("artifact_lineage.summary.self_contained_replay must match replay.self_contained.")


def _validate_training_manifest(
    manifest: dict[str, Any],
    target: ValidationTarget,
    episodes: list[dict[str, Any]],
    rewards: list[dict[str, Any]],
    step_rewards: list[dict[str, Any]],
    preferences: list[dict[str, Any]],
    failure_modes: list[dict[str, Any]],
    curriculum: dict[str, Any] | None,
    sft: list[dict[str, Any]],
    dpo: list[dict[str, Any]],
    reward_model: list[dict[str, Any]],
    dataset_metrics: dict[str, Any] | None,
    has_dataset_card: bool,
) -> None:
    _require_equal(manifest, "schema_version", RL_MANIFEST_SCHEMA_VERSION, target)
    expected_counts = {
        "episode_count": len(episodes),
        "reward_count": len(rewards),
        "preference_count": len(preferences),
        "failure_mode_count": len(failure_modes),
    }
    if "step_reward_count" in manifest:
        expected_counts["step_reward_count"] = len(step_rewards)
    else:
        target.warnings.append("manifest.step_reward_count is missing; rerun export-rl to refresh training artifacts.")
    trainer_view_counts = {
        "sft_count": len(sft),
        "dpo_count": len(dpo),
        "reward_model_count": len(reward_model),
    }
    for field_name, expected in trainer_view_counts.items():
        if field_name in manifest:
            expected_counts[field_name] = expected
        else:
            target.warnings.append(f"manifest.{field_name} is missing; rerun export-rl to refresh trainer-ready views.")
    if "quality_flag_count" in manifest and dataset_metrics is not None:
        expected_counts["quality_flag_count"] = len(dataset_metrics.get("quality_flags", [])) if isinstance(dataset_metrics.get("quality_flags"), list) else 0
    elif "quality_flag_count" not in manifest:
        target.warnings.append("manifest.quality_flag_count is missing; rerun export-rl to refresh dataset-level metrics.")
    for field_name, expected in expected_counts.items():
        if manifest.get(field_name) != expected:
            target.errors.append(f"manifest.{field_name} expected {expected}, got {manifest.get(field_name)!r}.")
    if not isinstance(manifest.get("outputs"), dict):
        target.errors.append("manifest.outputs must be an object.")
    else:
        for output_name in ("episodes", "rewards", "preferences", "failure_modes", "curriculum", "manifest"):
            if output_name not in manifest["outputs"]:
                target.errors.append(f"manifest.outputs.{output_name} is missing.")
        if "step_rewards" not in manifest["outputs"]:
            target.warnings.append("manifest.outputs.step_rewards is missing; rerun export-rl to refresh training artifacts.")
        for output_name in ("sft", "dpo", "reward_model"):
            if output_name not in manifest["outputs"]:
                target.warnings.append(f"manifest.outputs.{output_name} is missing; rerun export-rl to refresh trainer-ready views.")
        if "dataset_metrics" not in manifest["outputs"]:
            target.warnings.append("manifest.outputs.dataset_metrics is missing; rerun export-rl to refresh dataset-level metrics.")
        if "dataset_card" not in manifest["outputs"]:
            target.warnings.append("manifest.outputs.dataset_card is missing; rerun export-rl to refresh the dataset card.")
    if dataset_metrics is None:
        target.warnings.append("manifest has no validated dataset_metrics.json companion.")
    if not has_dataset_card:
        target.warnings.append("manifest has no DATASET_CARD.md companion.")
    if "metadata" in manifest:
        _validate_metadata(manifest.get("metadata"), target, "manifest.metadata")
    if dataset_metrics is not None and "metadata" in manifest and "metadata" in dataset_metrics and manifest.get("metadata") != dataset_metrics.get("metadata"):
        target.errors.append("manifest.metadata does not match dataset_metrics.metadata.")
    if curriculum is not None and curriculum.get("failure_mode_count") != len(failure_modes):
        target.errors.append(
            f"curriculum.failure_mode_count expected {len(failure_modes)}, got {curriculum.get('failure_mode_count')!r}."
        )
    if _looks_absolute(str(manifest.get("source_runs_dir", ""))):
        target.warnings.append("manifest.source_runs_dir is absolute; prefer redacted or relative exports for sharing.")
    if _looks_absolute(str(manifest.get("output_dir", ""))):
        target.warnings.append("manifest.output_dir is absolute; prefer redacted or relative exports for sharing.")


def _validate_compare_manifest(
    manifest: dict[str, Any],
    target: ValidationTarget,
    pairs: list[dict[str, Any]],
    dpo: list[dict[str, Any]],
    has_card: bool,
) -> None:
    _require_equal(manifest, "schema_version", COMPARE_RL_MANIFEST_SCHEMA_VERSION, target)
    expected_counts = {
        "pair_count": len(pairs),
        "dpo_count": len(dpo),
        "candidate_win_count": sum(1 for pair in pairs if pair.get("chosen_side") == "candidate"),
        "baseline_win_count": sum(1 for pair in pairs if pair.get("chosen_side") == "baseline"),
    }
    for field_name, expected in expected_counts.items():
        if manifest.get(field_name) != expected:
            target.errors.append(f"compare_manifest.{field_name} expected {expected}, got {manifest.get(field_name)!r}.")
    for field_name in ("baseline_run_count", "candidate_run_count", "paired_scenario_count", "skipped_pair_count", "min_score_gap"):
        if not _is_non_negative_int(manifest.get(field_name)):
            target.errors.append(f"compare_manifest.{field_name} must be a non-negative integer.")
    expected_contract_drift_count = sum(1 for pair in pairs if pair.get("contract_fingerprint_status") == "drifted")
    expected_unverified_contract_count = sum(1 for pair in pairs if pair.get("contract_fingerprint_status") == "unverified")
    if "contract_drift_count" in manifest and manifest.get("contract_drift_count") != expected_contract_drift_count:
        target.errors.append(
            f"compare_manifest.contract_drift_count expected {expected_contract_drift_count}, got {manifest.get('contract_drift_count')!r}."
        )
    if "unverified_contract_count" in manifest and manifest.get("unverified_contract_count") != expected_unverified_contract_count:
        target.errors.append(
            "compare_manifest.unverified_contract_count "
            f"expected {expected_unverified_contract_count}, got {manifest.get('unverified_contract_count')!r}."
        )
    if not isinstance(manifest.get("outputs"), dict):
        target.errors.append("compare_manifest.outputs must be an object.")
    else:
        for output_name in ("improvement_pairs", "improvement_dpo", "manifest", "improvement_card"):
            if output_name not in manifest["outputs"]:
                target.errors.append(f"compare_manifest.outputs.{output_name} is missing.")
    for field_name in ("missing_in_candidate", "new_in_candidate"):
        if not _is_string_list(manifest.get(field_name)):
            target.errors.append(f"compare_manifest.{field_name} must be a list of strings.")
    skipped = manifest.get("skipped_pairs")
    if not isinstance(skipped, list):
        target.errors.append("compare_manifest.skipped_pairs must be a list.")
    if "metadata" in manifest:
        _validate_metadata(manifest.get("metadata"), target, "compare_manifest.metadata")
    if not has_card:
        target.warnings.append("compare_manifest has no IMPROVEMENT_CARD.md companion.")
    if "contract_scope" in manifest and manifest.get("contract_scope") not in CONTRACT_SCOPES:
        target.errors.append(f"compare_manifest.contract_scope must be one of {sorted(CONTRACT_SCOPES)!r}.")
    if manifest.get("contract_scope") in CONTRACT_SCOPES:
        for index, pair in enumerate(pairs):
            if isinstance(pair, dict) and pair.get("contract_fingerprint_scope") not in {None, manifest.get("contract_scope")}:
                target.errors.append(
                    f"improvement_pairs[{index}].contract_fingerprint_scope must match compare_manifest.contract_scope."
                )
    for field_name in ("baseline_runs_dir", "candidate_runs_dir", "output_dir"):
        if _looks_absolute(str(manifest.get(field_name, ""))):
            target.warnings.append(f"compare_manifest.{field_name} is absolute; prefer redacted or relative exports for sharing.")


def _validate_compare_pairs(pairs: list[dict[str, Any]], target: ValidationTarget) -> None:
    seen: set[str] = set()
    for index, pair in enumerate(pairs):
        label = f"improvement_pairs[{index}]"
        _require_equal(pair, "schema_version", COMPARE_RL_PAIR_SCHEMA_VERSION, target, prefix=f"{label}.")
        pair_id = pair.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            target.errors.append(f"{label}.pair_id must be a non-empty string.")
        elif pair_id in seen:
            target.errors.append(f"{label}.pair_id duplicates {pair_id!r}.")
        else:
            seen.add(pair_id)
        for field_name in ("scenario_id", "task_family", "chosen_episode_id", "rejected_episode_id", "baseline_episode_id", "candidate_episode_id", "reason"):
            if not isinstance(pair.get(field_name), str) or not pair.get(field_name):
                target.errors.append(f"{label}.{field_name} must be a non-empty string.")
        if not isinstance(pair.get("prompt"), str):
            target.errors.append(f"{label}.prompt must be a string.")
        if pair.get("candidate_outcome") not in {"improved", "regressed"}:
            target.errors.append(f"{label}.candidate_outcome must be improved or regressed.")
        chosen_side = pair.get("chosen_side")
        rejected_side = pair.get("rejected_side")
        if chosen_side not in {"baseline", "candidate"}:
            target.errors.append(f"{label}.chosen_side must be baseline or candidate.")
        if rejected_side not in {"baseline", "candidate"}:
            target.errors.append(f"{label}.rejected_side must be baseline or candidate.")
        if chosen_side == rejected_side:
            target.errors.append(f"{label}.chosen_side and rejected_side must differ.")
        if not _is_plain_int(pair.get("candidate_score_delta")):
            target.errors.append(f"{label}.candidate_score_delta must be an integer.")
        for field_name in ("chosen_score", "rejected_score", "score_gap"):
            if not _is_int_between(pair.get(field_name), 0, 100):
                target.errors.append(f"{label}.{field_name} must be an integer from 0 to 100.")
        if _is_int_between(pair.get("chosen_score"), 0, 100) and _is_int_between(pair.get("rejected_score"), 0, 100):
            expected_gap = pair["chosen_score"] - pair["rejected_score"]
            if pair.get("score_gap") != expected_gap:
                target.errors.append(f"{label}.score_gap expected {expected_gap}, got {pair.get('score_gap')!r}.")
            if expected_gap <= 0:
                target.errors.append(f"{label}.chosen_score must be greater than rejected_score.")
        for field_name in ("rule_fixes", "rule_regressions", "new_critical_failures"):
            if not _is_string_list(pair.get(field_name)):
                target.errors.append(f"{label}.{field_name} must be a list of strings.")
        if "contract_fingerprint_status" in pair:
            _validate_contract_fingerprint_status(pair, target, label)
        baseline = _validate_compare_view(pair.get("baseline"), target, f"{label}.baseline")
        candidate = _validate_compare_view(pair.get("candidate"), target, f"{label}.candidate")
        chosen = _validate_compare_view(pair.get("chosen"), target, f"{label}.chosen")
        rejected = _validate_compare_view(pair.get("rejected"), target, f"{label}.rejected")
        _validate_compare_pair_links(pair, baseline, candidate, chosen, rejected, target, label)


def _validate_compare_pair_links(
    pair: dict[str, Any],
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    chosen: dict[str, Any],
    rejected: dict[str, Any],
    target: ValidationTarget,
    label: str,
) -> None:
    if baseline and pair.get("baseline_episode_id") != baseline.get("episode_id"):
        target.errors.append(f"{label}.baseline_episode_id must match baseline.episode_id.")
    if candidate and pair.get("candidate_episode_id") != candidate.get("episode_id"):
        target.errors.append(f"{label}.candidate_episode_id must match candidate.episode_id.")
    chosen_side = pair.get("chosen_side")
    rejected_side = pair.get("rejected_side")
    expected_chosen = candidate if chosen_side == "candidate" else baseline if chosen_side == "baseline" else {}
    expected_rejected = candidate if rejected_side == "candidate" else baseline if rejected_side == "baseline" else {}
    if expected_chosen and chosen and chosen.get("episode_id") != expected_chosen.get("episode_id"):
        target.errors.append(f"{label}.chosen must match the chosen_side view.")
    if expected_rejected and rejected and rejected.get("episode_id") != expected_rejected.get("episode_id"):
        target.errors.append(f"{label}.rejected must match the rejected_side view.")
    if pair.get("candidate_outcome") == "improved" and not (_is_plain_int(pair.get("candidate_score_delta")) and pair["candidate_score_delta"] > 0):
        target.errors.append(f"{label}.candidate_score_delta must be positive when candidate_outcome is improved.")
    if pair.get("candidate_outcome") == "regressed" and not (_is_plain_int(pair.get("candidate_score_delta")) and pair["candidate_score_delta"] < 0):
        target.errors.append(f"{label}.candidate_score_delta must be negative when candidate_outcome is regressed.")


def _validate_compare_view(value: Any, target: ValidationTarget, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return {}
    for field_name in ("episode_id", "scenario_id"):
        if not isinstance(value.get(field_name), str) or not value.get(field_name):
            target.errors.append(f"{label}.{field_name} must be a non-empty string.")
    if not isinstance(value.get("passed"), bool):
        target.errors.append(f"{label}.passed must be a boolean.")
    if not _is_int_between(value.get("score"), 0, 100):
        target.errors.append(f"{label}.score must be an integer from 0 to 100.")
    if not isinstance(value.get("reward"), (int, float)) or isinstance(value.get("reward"), bool):
        target.errors.append(f"{label}.reward must be numeric.")
    if not _is_string_list(value.get("failed_rules")):
        target.errors.append(f"{label}.failed_rules must be a list of strings.")
    if not isinstance(value.get("events"), list):
        target.errors.append(f"{label}.events must be a list.")
    if not isinstance(value.get("final_answer"), str):
        target.errors.append(f"{label}.final_answer must be a string.")
    if "task_completion" in value:
        _validate_task_completion(value.get("task_completion"), target, f"{label}.task_completion")
    _validate_source_fingerprint_fields(value, target, label, warn_if_missing=False)
    return value


def _validate_compare_dpo(dpo: list[dict[str, Any]], target: ValidationTarget, pairs: list[dict[str, Any]]) -> None:
    pair_by_id = {pair.get("pair_id"): pair for pair in pairs if isinstance(pair.get("pair_id"), str)}
    seen: set[str] = set()
    for index, row in enumerate(dpo):
        label = f"improvement_dpo[{index}]"
        _require_equal(row, "schema_version", COMPARE_RL_DPO_SCHEMA_VERSION, target, prefix=f"{label}.")
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            target.errors.append(f"{label}.pair_id must be a non-empty string.")
            pair = None
        elif pair_id in seen:
            target.errors.append(f"{label}.pair_id duplicates {pair_id!r}.")
            pair = pair_by_id.get(pair_id)
        else:
            seen.add(pair_id)
            pair = pair_by_id.get(pair_id)
        if row.get("preference_id") != pair_id:
            target.errors.append(f"{label}.preference_id must match pair_id.")
        if pair is None:
            target.errors.append(f"{label}.pair_id {pair_id!r} does not reference an improvement pair.")
            continue
        if row.get("source_artifact") != "improvement_pairs.jsonl":
            target.errors.append(f"{label}.source_artifact must be 'improvement_pairs.jsonl'.")
        for field_name in (
            "scenario_id",
            "task_family",
            "prompt",
            "chosen",
            "rejected",
            "chosen_side",
            "rejected_side",
            "candidate_outcome",
            "reason",
        ):
            if not isinstance(row.get(field_name), str):
                target.errors.append(f"{label}.{field_name} must be a string.")
        for field_name in ("chosen_score", "rejected_score", "score_gap"):
            if not _is_int_between(row.get(field_name), 0, 100):
                target.errors.append(f"{label}.{field_name} must be an integer from 0 to 100.")
        for field_name in ("chosen_task_completion_status", "rejected_task_completion_status"):
            if row.get(field_name) not in {"complete", "incomplete", "not_applicable"}:
                target.errors.append(f"{label}.{field_name} must be complete, incomplete, or not_applicable.")
        for field_name in ("chosen_task_completion_passed", "rejected_task_completion_passed"):
            if not isinstance(row.get(field_name), bool):
                target.errors.append(f"{label}.{field_name} must be a boolean.")
        if not _is_plain_int(row.get("candidate_score_delta")):
            target.errors.append(f"{label}.candidate_score_delta must be an integer.")
        if "contract_fingerprint_status" in row:
            _validate_contract_fingerprint_status(row, target, label)
        _validate_messages(row.get("chosen_messages"), target, f"{label}.chosen_messages")
        _validate_messages(row.get("rejected_messages"), target, f"{label}.rejected_messages")
        _compare_improvement_dpo_to_pair(row, pair, target, label)
    missing = sorted(set(pair_by_id) - seen)
    if missing:
        target.errors.append(f"improvement_dpo.jsonl missing improvement pairs: {missing!r}.")


def _compare_improvement_dpo_to_pair(row: dict[str, Any], pair: dict[str, Any], target: ValidationTarget, label: str) -> None:
    chosen = pair.get("chosen") if isinstance(pair.get("chosen"), dict) else {}
    rejected = pair.get("rejected") if isinstance(pair.get("rejected"), dict) else {}
    expected = {
        "scenario_id": pair.get("scenario_id"),
        "task_family": pair.get("task_family"),
        "prompt": pair.get("prompt"),
        "chosen": _compare_response_text(chosen),
        "rejected": _compare_response_text(rejected),
        "chosen_side": pair.get("chosen_side"),
        "rejected_side": pair.get("rejected_side"),
        "candidate_outcome": pair.get("candidate_outcome"),
        "candidate_score_delta": pair.get("candidate_score_delta"),
        "chosen_episode_id": pair.get("chosen_episode_id"),
        "rejected_episode_id": pair.get("rejected_episode_id"),
        "chosen_score": pair.get("chosen_score"),
        "rejected_score": pair.get("rejected_score"),
        "score_gap": pair.get("score_gap"),
        "chosen_task_completion_status": str((chosen.get("task_completion") or {}).get("status") or "not_applicable"),
        "rejected_task_completion_status": str((rejected.get("task_completion") or {}).get("status") or "not_applicable"),
        "chosen_task_completion_passed": bool((chosen.get("task_completion") or {}).get("passed", True)),
        "rejected_task_completion_passed": bool((rejected.get("task_completion") or {}).get("passed", True)),
        "contract_fingerprint_status": pair.get("contract_fingerprint_status"),
        "contract_fingerprint_scope": pair.get("contract_fingerprint_scope"),
        "contract_fingerprint_reasons": pair.get("contract_fingerprint_reasons"),
        "contract_fingerprints": pair.get("contract_fingerprints"),
        "reason": pair.get("reason"),
    }
    for field_name, expected_value in expected.items():
        if row.get(field_name) != expected_value:
            target.errors.append(f"{label}.{field_name} does not match improvement pair {pair.get('pair_id')!r}.")


def _compare_response_text(view: dict[str, Any]) -> str:
    lines = ["Observed behavior:"]
    task = view.get("task_completion") if isinstance(view.get("task_completion"), dict) else {}
    if task:
        lines.append(
            "- "
            + " ".join(
                [
                    "task_completion",
                    str(task.get("status") or "unknown"),
                    f"checks={task.get('passed_check_count', 0)}/{task.get('required_check_count', 0)}",
                ]
            )
        )
    events = view.get("events") if isinstance(view.get("events"), list) else []
    for event in events:
        if not isinstance(event, dict):
            continue
        event_type = str(event.get("type") or "event")
        if event_type not in {"tool_call", "tool_result", "assistant_message"}:
            continue
        parts = [event_type]
        tool_name = str(event.get("tool_name") or "").strip()
        status = str(event.get("status") or "").strip()
        if tool_name:
            parts.append(tool_name)
        if status:
            parts.append(status)
        detail = _compare_event_detail(event)
        if detail:
            parts.append(detail)
        lines.append("- " + " ".join(parts))
    final_answer = str(view.get("final_answer") or "")
    if final_answer:
        lines.append(f"Final answer: {final_answer}")
    return "\n".join(lines)


def _compare_event_detail(event: dict[str, Any]) -> str:
    for field_name in ("result", "args"):
        value = event.get(field_name)
        if isinstance(value, dict) and value:
            return json.dumps(value, sort_keys=True, ensure_ascii=False)
    text = str(event.get("text") or "").strip()
    return text[:500]


def _validate_improvement_card(path: Path, target: ValidationTarget) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        target.errors.append(f"IMPROVEMENT_CARD.md could not be read: {exc}")
        return
    required = [
        "# Flight Recorder Improvement Pair Card",
        "## Summary",
        "## Pairs",
        "## Boundaries",
    ]
    for marker in required:
        if marker not in text:
            target.errors.append(f"IMPROVEMENT_CARD.md missing section marker {marker!r}.")


def _validate_review_manifest(
    manifest: dict[str, Any],
    target: ValidationTarget,
    items: list[dict[str, Any]],
    labels: list[dict[str, Any]],
) -> None:
    _require_equal(manifest, "schema_version", REVIEW_MANIFEST_SCHEMA_VERSION, target)
    if manifest.get("item_count") != len(items):
        target.errors.append(f"manifest.item_count expected {len(items)}, got {manifest.get('item_count')!r}.")
    passed_count = sum(1 for item in items if _review_item_passed(item) is True)
    failed_count = sum(1 for item in items if _review_item_passed(item) is False)
    if manifest.get("passed_count") != passed_count:
        target.errors.append(f"manifest.passed_count expected {passed_count}, got {manifest.get('passed_count')!r}.")
    if manifest.get("failed_count") != failed_count:
        target.errors.append(f"manifest.failed_count expected {failed_count}, got {manifest.get('failed_count')!r}.")
    if len(labels) != len(items):
        target.errors.append(f"label_template row count expected {len(items)}, got {len(labels)}.")
    if manifest.get("label_options") != list(REVIEW_LABELS):
        target.errors.append(f"manifest.label_options must be {list(REVIEW_LABELS)!r}.")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        target.errors.append("manifest.outputs must be an object.")
    else:
        for output_name in ("review_items", "label_template", "instructions", "manifest"):
            if output_name not in outputs:
                target.errors.append(f"manifest.outputs.{output_name} is missing.")
    if _looks_absolute(str(manifest.get("source_runs_dir", ""))):
        target.warnings.append("manifest.source_runs_dir is absolute; prefer redacted or relative exports for sharing.")
    if _looks_absolute(str(manifest.get("output_dir", ""))):
        target.warnings.append("manifest.output_dir is absolute; prefer redacted or relative exports for sharing.")


def _validate_review_items(items: list[dict[str, Any]], target: ValidationTarget) -> None:
    seen: set[str] = set()
    for index, item in enumerate(items):
        _require_equal(item, "schema_version", REVIEW_ITEM_SCHEMA_VERSION, target, prefix=f"review_items[{index}].")
        item_id = item.get("review_item_id")
        if not isinstance(item_id, str) or not item_id:
            target.errors.append(f"review_items[{index}].review_item_id must be a non-empty string.")
            continue
        if item_id in seen:
            target.errors.append(f"review_items[{index}].review_item_id duplicates {item_id!r}.")
        seen.add(item_id)
        for field_name in ("episode_id", "scenario_id", "scenario_title", "task_family", "prompt", "final_answer", "suggested_human_label"):
            if not isinstance(item.get(field_name), str):
                target.errors.append(f"review_items[{index}].{field_name} must be a string.")
        if item.get("suggested_human_label") not in REVIEW_LABELS:
            target.errors.append(f"review_items[{index}].suggested_human_label must be one of {list(REVIEW_LABELS)!r}.")
        if item.get("label_options") != list(REVIEW_LABELS):
            target.errors.append(f"review_items[{index}].label_options must be {list(REVIEW_LABELS)!r}.")
        if not isinstance(item.get("event_count"), int) or isinstance(item.get("event_count"), bool) or item.get("event_count") < 0:
            target.errors.append(f"review_items[{index}].event_count must be a non-negative integer.")
        source_artifacts = item.get("source_artifacts")
        if not isinstance(source_artifacts, dict):
            target.errors.append(f"review_items[{index}].source_artifacts must be an object.")
        else:
            for artifact_name in ("run_dir", "normalized_trace", "scorecard", "report"):
                if not isinstance(source_artifacts.get(artifact_name), str) or not source_artifacts.get(artifact_name):
                    target.errors.append(f"review_items[{index}].source_artifacts.{artifact_name} must be a non-empty string.")
        scorecard = item.get("scorecard")
        if not isinstance(scorecard, dict):
            target.errors.append(f"review_items[{index}].scorecard must be an object.")
        else:
            if not isinstance(scorecard.get("passed"), bool):
                target.errors.append(f"review_items[{index}].scorecard.passed must be a boolean.")
            if not _is_int_between(scorecard.get("score"), 0, 100):
                target.errors.append(f"review_items[{index}].scorecard.score must be an integer from 0 to 100.")
            if not _is_string_list(scorecard.get("failed_rules")):
                target.errors.append(f"review_items[{index}].scorecard.failed_rules must be a list of strings.")
            if not _is_string_list(scorecard.get("critical_failures")):
                target.errors.append(f"review_items[{index}].scorecard.critical_failures must be a list of strings.")
        if not isinstance(item.get("rule_summaries"), list):
            target.errors.append(f"review_items[{index}].rule_summaries must be a list.")
        if not isinstance(item.get("task_evidence"), list):
            target.errors.append(f"review_items[{index}].task_evidence must be a list.")
        if not isinstance(item.get("evidence_target_counts"), dict):
            target.errors.append(f"review_items[{index}].evidence_target_counts must be an object.")


def _validate_review_labels(labels: list[dict[str, Any]], target: ValidationTarget, items: list[dict[str, Any]]) -> None:
    item_ids = {item.get("review_item_id") for item in items if isinstance(item.get("review_item_id"), str)}
    seen: set[str] = set()
    for index, label in enumerate(labels):
        _require_equal(label, "schema_version", REVIEW_LABEL_SCHEMA_VERSION, target, prefix=f"label_template[{index}].")
        item_id = label.get("review_item_id")
        if not isinstance(item_id, str) or not item_id:
            target.errors.append(f"label_template[{index}].review_item_id must be a non-empty string.")
            continue
        if item_id in seen:
            target.errors.append(f"label_template[{index}].review_item_id duplicates {item_id!r}.")
        seen.add(item_id)
        if item_id not in item_ids:
            target.errors.append(f"label_template[{index}].review_item_id does not reference a review item.")
        suggested = label.get("suggested_human_label")
        if suggested not in REVIEW_LABELS:
            target.errors.append(f"label_template[{index}].suggested_human_label must be one of {list(REVIEW_LABELS)!r}.")
        human_label = label.get("human_label")
        if human_label is not None and human_label not in REVIEW_LABELS:
            target.errors.append(f"label_template[{index}].human_label must be null or one of {list(REVIEW_LABELS)!r}.")
        corrected_score = label.get("corrected_score")
        if corrected_score is not None and not _is_int_between(corrected_score, 0, 100):
            target.errors.append(f"label_template[{index}].corrected_score must be null or an integer from 0 to 100.")
        for field_name in ("accepted_evidence_refs", "rejected_evidence_refs"):
            if not isinstance(label.get(field_name), list):
                target.errors.append(f"label_template[{index}].{field_name} must be a list.")
    missing = sorted(item_ids - seen)
    if missing:
        target.errors.append(f"label_template missing review_item_id values: {missing!r}.")


def _review_item_passed(item: dict[str, Any]) -> bool | None:
    scorecard = item.get("scorecard")
    if isinstance(scorecard, dict) and isinstance(scorecard.get("passed"), bool):
        return scorecard["passed"]
    return None


def _validate_reviewed_manifest(
    manifest: dict[str, Any],
    target: ValidationTarget,
    labels: list[dict[str, Any]],
    sft: list[dict[str, Any]],
    reward_model: list[dict[str, Any]],
    preferences: list[dict[str, Any]],
    dpo: list[dict[str, Any]],
) -> None:
    _require_equal(manifest, "schema_version", REVIEWED_MANIFEST_SCHEMA_VERSION, target)
    expected_counts = {
        "reviewed_label_count": len(labels),
        "sft_count": len(sft),
        "reward_model_count": len(reward_model),
        "preference_count": len(preferences),
        "dpo_count": len(dpo),
    }
    for field_name, expected in expected_counts.items():
        if manifest.get(field_name) != expected:
            target.errors.append(f"manifest.{field_name} expected {expected}, got {manifest.get(field_name)!r}.")
    expected_label_counts = _reviewed_label_counts(labels)
    if manifest.get("label_counts") != expected_label_counts:
        target.errors.append(f"manifest.label_counts expected {expected_label_counts!r}, got {manifest.get('label_counts')!r}.")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        target.errors.append("manifest.outputs must be an object.")
    else:
        for output_name in ("reviewed_labels", "reviewed_sft", "reviewed_reward_model", "reviewed_preferences", "reviewed_dpo", "manifest"):
            if output_name not in outputs:
                target.errors.append(f"manifest.outputs.{output_name} is missing.")
    if _looks_absolute(str(manifest.get("source_review_export", ""))):
        target.warnings.append("manifest.source_review_export is absolute; prefer redacted or relative exports for sharing.")
    if _looks_absolute(str(manifest.get("labels_path", ""))):
        target.warnings.append("manifest.labels_path is absolute; prefer redacted or relative exports for sharing.")
    if _looks_absolute(str(manifest.get("output_dir", ""))):
        target.warnings.append("manifest.output_dir is absolute; prefer redacted or relative exports for sharing.")


def _validate_reviewed_labels(labels: list[dict[str, Any]], target: ValidationTarget) -> None:
    seen: set[str] = set()
    for index, row in enumerate(labels):
        _require_equal(row, "schema_version", REVIEWED_LABEL_SCHEMA_VERSION, target, prefix=f"reviewed_labels[{index}].")
        item_id = row.get("review_item_id")
        if not isinstance(item_id, str) or not item_id:
            target.errors.append(f"reviewed_labels[{index}].review_item_id must be a non-empty string.")
            continue
        if item_id in seen:
            target.errors.append(f"reviewed_labels[{index}].review_item_id duplicates {item_id!r}.")
        seen.add(item_id)
        for field_name in ("episode_id", "scenario_id", "task_family", "prompt", "response", "human_label", "source_label_file"):
            if not isinstance(row.get(field_name), str):
                target.errors.append(f"reviewed_labels[{index}].{field_name} must be a string.")
        if row.get("human_label") not in REVIEW_LABELS:
            target.errors.append(f"reviewed_labels[{index}].human_label must be one of {list(REVIEW_LABELS)!r}.")
        if row.get("suggested_human_label") is not None and row.get("suggested_human_label") not in REVIEW_LABELS:
            target.errors.append(f"reviewed_labels[{index}].suggested_human_label must be null or one of {list(REVIEW_LABELS)!r}.")
        if not _is_int_between(row.get("score"), 0, 100):
            target.errors.append(f"reviewed_labels[{index}].score must be an integer from 0 to 100.")
        if not isinstance(row.get("reward"), (int, float)):
            target.errors.append(f"reviewed_labels[{index}].reward must be numeric.")
        if not isinstance(row.get("accepted_evidence_refs"), list):
            target.errors.append(f"reviewed_labels[{index}].accepted_evidence_refs must be a list.")
        if not isinstance(row.get("rejected_evidence_refs"), list):
            target.errors.append(f"reviewed_labels[{index}].rejected_evidence_refs must be a list.")
        if not isinstance(row.get("source_artifacts"), dict):
            target.errors.append(f"reviewed_labels[{index}].source_artifacts must be an object.")
        if not isinstance(row.get("scorecard"), dict):
            target.errors.append(f"reviewed_labels[{index}].scorecard must be an object.")


def _validate_reviewed_sft(sft: list[dict[str, Any]], target: ValidationTarget, labels: list[dict[str, Any]]) -> None:
    label_map = _reviewed_label_map(labels)
    for index, row in enumerate(sft):
        _require_equal(row, "schema_version", REVIEWED_SFT_SCHEMA_VERSION, target, prefix=f"reviewed_sft[{index}].")
        item = _reviewed_source_label(row, label_map, target, f"reviewed_sft[{index}]")
        if item is not None and item.get("human_label") != "accept":
            target.errors.append(f"reviewed_sft[{index}] must reference a reviewed label with human_label 'accept'.")
        for field_name in ("prompt", "response", "source_artifact"):
            if not isinstance(row.get(field_name), str):
                target.errors.append(f"reviewed_sft[{index}].{field_name} must be a string.")
        if row.get("source_artifact") != "reviewed_labels.jsonl":
            target.errors.append(f"reviewed_sft[{index}].source_artifact must be 'reviewed_labels.jsonl'.")


def _validate_reviewed_reward_model(rows: list[dict[str, Any]], target: ValidationTarget, labels: list[dict[str, Any]]) -> None:
    label_map = _reviewed_label_map(labels)
    for index, row in enumerate(rows):
        _require_equal(row, "schema_version", REVIEWED_REWARD_MODEL_SCHEMA_VERSION, target, prefix=f"reviewed_reward_model[{index}].")
        item = _reviewed_source_label(row, label_map, target, f"reviewed_reward_model[{index}]")
        if item is not None and item.get("human_label") == "needs_review":
            target.errors.append(f"reviewed_reward_model[{index}] must not reference a needs_review label.")
        for field_name in ("prompt", "response", "human_label", "source_artifact"):
            if not isinstance(row.get(field_name), str):
                target.errors.append(f"reviewed_reward_model[{index}].{field_name} must be a string.")
        if not _is_int_between(row.get("score"), 0, 100):
            target.errors.append(f"reviewed_reward_model[{index}].score must be an integer from 0 to 100.")
        if not isinstance(row.get("reward"), (int, float)):
            target.errors.append(f"reviewed_reward_model[{index}].reward must be numeric.")
        if row.get("source_artifact") != "reviewed_labels.jsonl":
            target.errors.append(f"reviewed_reward_model[{index}].source_artifact must be 'reviewed_labels.jsonl'.")


def _validate_reviewed_preferences(preferences: list[dict[str, Any]], target: ValidationTarget, labels: list[dict[str, Any]]) -> None:
    label_by_episode = _reviewed_label_by_episode(labels)
    seen: set[str] = set()
    for index, row in enumerate(preferences):
        _require_equal(row, "schema_version", REVIEWED_PREFERENCE_SCHEMA_VERSION, target, prefix=f"reviewed_preferences[{index}].")
        preference_id = row.get("preference_id")
        if not isinstance(preference_id, str) or not preference_id:
            target.errors.append(f"reviewed_preferences[{index}].preference_id must be a non-empty string.")
        elif preference_id in seen:
            target.errors.append(f"reviewed_preferences[{index}].preference_id duplicates {preference_id!r}.")
        else:
            seen.add(preference_id)
        chosen = label_by_episode.get(row.get("chosen_episode_id"))
        rejected = label_by_episode.get(row.get("rejected_episode_id"))
        if chosen is None:
            target.errors.append(f"reviewed_preferences[{index}].chosen_episode_id does not reference a reviewed label.")
        elif chosen.get("human_label") != "accept":
            target.errors.append(f"reviewed_preferences[{index}].chosen_episode_id must reference an accepted label.")
        if rejected is None:
            target.errors.append(f"reviewed_preferences[{index}].rejected_episode_id does not reference a reviewed label.")
        elif rejected.get("human_label") not in {"reject", "unsafe", "incomplete"}:
            target.errors.append(f"reviewed_preferences[{index}].rejected_episode_id must reference a rejected/unsafe/incomplete label.")
        if row.get("source_artifact") != "reviewed_labels.jsonl":
            target.errors.append(f"reviewed_preferences[{index}].source_artifact must be 'reviewed_labels.jsonl'.")


def _validate_reviewed_dpo(dpo: list[dict[str, Any]], target: ValidationTarget, preferences: list[dict[str, Any]]) -> None:
    preference_ids = {row.get("preference_id") for row in preferences if isinstance(row.get("preference_id"), str)}
    for index, row in enumerate(dpo):
        _require_equal(row, "schema_version", REVIEWED_DPO_SCHEMA_VERSION, target, prefix=f"reviewed_dpo[{index}].")
        if row.get("preference_id") not in preference_ids:
            target.errors.append(f"reviewed_dpo[{index}].preference_id does not reference a reviewed preference.")
        for field_name in ("prompt", "chosen", "rejected", "source_artifact"):
            if not isinstance(row.get(field_name), str):
                target.errors.append(f"reviewed_dpo[{index}].{field_name} must be a string.")
        if row.get("source_artifact") != "reviewed_preferences.jsonl":
            target.errors.append(f"reviewed_dpo[{index}].source_artifact must be 'reviewed_preferences.jsonl'.")


def _reviewed_source_label(
    row: dict[str, Any],
    label_map: dict[str, dict[str, Any]],
    target: ValidationTarget,
    label: str,
) -> dict[str, Any] | None:
    item_id = row.get("review_item_id")
    if not isinstance(item_id, str) or not item_id:
        target.errors.append(f"{label}.review_item_id must be a non-empty string.")
        return None
    item = label_map.get(item_id)
    if item is None:
        target.errors.append(f"{label}.review_item_id does not reference a reviewed label.")
    return item


def _reviewed_label_map(labels: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row["review_item_id"]): row
        for row in labels
        if isinstance(row.get("review_item_id"), str)
    }


def _reviewed_label_by_episode(labels: list[dict[str, Any]]) -> dict[Any, dict[str, Any]]:
    return {row.get("episode_id"): row for row in labels if row.get("episode_id") is not None}


def _reviewed_label_counts(labels: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in labels:
        label = row.get("human_label")
        if isinstance(label, str):
            counts[label] = counts.get(label, 0) + 1
    return counts


def _validate_episodes(episodes: list[dict[str, Any]], target: ValidationTarget) -> None:
    seen: set[str] = set()
    for index, episode in enumerate(episodes):
        _require_equal(episode, "schema_version", RL_EPISODE_SCHEMA_VERSION, target, prefix=f"episodes[{index}].")
        episode_id = episode.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            target.errors.append(f"episodes[{index}].episode_id must be a non-empty string.")
            continue
        if episode_id in seen:
            target.errors.append(f"episodes[{index}].episode_id duplicates {episode_id!r}.")
        seen.add(episode_id)
        for field_name in ("scenario_id", "task_family", "final_answer"):
            if not isinstance(episode.get(field_name), str):
                target.errors.append(f"episodes[{index}].{field_name} must be a string.")
        if _looks_absolute(str(episode.get("source_run", ""))):
            target.warnings.append(f"episodes[{index}].source_run is absolute; prefer redacted or relative exports for sharing.")
        if "source_lineage" in episode and _looks_absolute(str(episode.get("source_lineage", ""))):
            target.warnings.append(f"episodes[{index}].source_lineage is absolute; prefer redacted or relative exports for sharing.")
        _validate_source_fingerprint_fields(episode, target, f"episodes[{index}]", warn_if_missing=True)
        if not isinstance(episode.get("events"), list):
            target.errors.append(f"episodes[{index}].events must be a list.")
        if "task_completion" in episode:
            _validate_task_completion(episode.get("task_completion"), target, f"episodes[{index}].task_completion")
        else:
            target.warnings.append(f"episodes[{index}].task_completion is missing; rerun export-rl to refresh task evidence fields.")
        outcome = episode.get("outcome")
        if not isinstance(outcome, dict):
            target.errors.append(f"episodes[{index}].outcome must be an object.")
            continue
        if not _is_int_between(outcome.get("score"), 0, 100):
            target.errors.append(f"episodes[{index}].outcome.score must be an integer from 0 to 100.")
        if not isinstance(outcome.get("passed"), bool):
            target.errors.append(f"episodes[{index}].outcome.passed must be a boolean.")
        if not isinstance(outcome.get("reward"), (int, float)):
            target.errors.append(f"episodes[{index}].outcome.reward must be numeric.")
        if isinstance(episode.get("task_completion"), dict):
            if outcome.get("task_completion_status") != episode["task_completion"].get("status"):
                target.errors.append(f"episodes[{index}].outcome.task_completion_status must match task_completion.status.")
            if outcome.get("task_completion_passed") != episode["task_completion"].get("passed"):
                target.errors.append(f"episodes[{index}].outcome.task_completion_passed must match task_completion.passed.")


def _validate_rewards(rewards: list[dict[str, Any]], target: ValidationTarget, episodes: list[dict[str, Any]]) -> None:
    episode_by_id = {episode.get("episode_id"): episode for episode in episodes if isinstance(episode.get("episode_id"), str)}
    seen: set[str] = set()
    for index, reward in enumerate(rewards):
        _require_equal(reward, "schema_version", RL_REWARD_SCHEMA_VERSION, target, prefix=f"rewards[{index}].")
        episode_id = reward.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            target.errors.append(f"rewards[{index}].episode_id must be a non-empty string.")
            continue
        if episode_id in seen:
            target.errors.append(f"rewards[{index}].episode_id duplicates {episode_id!r}.")
        seen.add(episode_id)
        episode = episode_by_id.get(episode_id)
        if episode is None:
            target.errors.append(f"rewards[{index}].episode_id {episode_id!r} does not reference an episode.")
        else:
            outcome = episode.get("outcome") if isinstance(episode.get("outcome"), dict) else {}
            for field_name in ("score", "passed", "reward"):
                if reward.get(field_name) != outcome.get(field_name):
                    target.errors.append(
                        f"rewards[{index}].{field_name} does not match episode {episode_id!r} outcome."
                    )
            if "task_completion_status" in reward and reward.get("task_completion_status") != outcome.get("task_completion_status"):
                target.errors.append(f"rewards[{index}].task_completion_status does not match episode {episode_id!r} outcome.")
            if "task_completion_passed" in reward and reward.get("task_completion_passed") != outcome.get("task_completion_passed"):
                target.errors.append(f"rewards[{index}].task_completion_passed does not match episode {episode_id!r} outcome.")
            _validate_matching_source_fingerprints(reward, episode, target, f"rewards[{index}]")
        _validate_source_fingerprint_fields(reward, target, f"rewards[{index}]", warn_if_missing=False)
        if not isinstance(reward.get("rule_rewards"), list):
            target.errors.append(f"rewards[{index}].rule_rewards must be a list.")
        else:
            for rule_index, rule_reward in enumerate(reward["rule_rewards"]):
                if isinstance(rule_reward, dict) and "evidence_refs" in rule_reward:
                    _validate_evidence_refs(
                        rule_reward.get("evidence_refs"),
                        target,
                        f"rewards[{index}].rule_rewards[{rule_index}].evidence_refs",
                    )
        if not isinstance(reward.get("attribution"), list):
            target.errors.append(f"rewards[{index}].attribution must be a list.")


def _validate_step_rewards(
    step_rewards: list[dict[str, Any]],
    target: ValidationTarget,
    episodes: list[dict[str, Any]],
    rewards: list[dict[str, Any]],
) -> None:
    episode_by_id = {episode.get("episode_id"): episode for episode in episodes if isinstance(episode.get("episode_id"), str)}
    rule_delta_by_key = _rule_reward_deltas_by_key(rewards)
    step_delta_by_key: dict[tuple[str, str], float] = {}
    seen: set[str] = set()
    for index, step_reward in enumerate(step_rewards):
        _require_equal(step_reward, "schema_version", RL_STEP_REWARD_SCHEMA_VERSION, target, prefix=f"step_rewards[{index}].")
        step_reward_id = step_reward.get("step_reward_id")
        if not isinstance(step_reward_id, str) or not step_reward_id:
            target.errors.append(f"step_rewards[{index}].step_reward_id must be a non-empty string.")
        elif step_reward_id in seen:
            target.errors.append(f"step_rewards[{index}].step_reward_id duplicates {step_reward_id!r}.")
        else:
            seen.add(step_reward_id)

        episode_id = step_reward.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            target.errors.append(f"step_rewards[{index}].episode_id must be a non-empty string.")
            episode = None
        else:
            episode = episode_by_id.get(episode_id)
            if episode is None:
                target.errors.append(f"step_rewards[{index}].episode_id {episode_id!r} does not reference an episode.")
            else:
                _validate_matching_source_fingerprints(step_reward, episode, target, f"step_rewards[{index}]")
        _validate_source_fingerprint_fields(step_reward, target, f"step_rewards[{index}]", warn_if_missing=False)

        rule_id = step_reward.get("rule_id")
        for field_name in ("scenario_id", "task_family", "rule_id", "rule_name", "evidence"):
            if not isinstance(step_reward.get(field_name), str):
                target.errors.append(f"step_rewards[{index}].{field_name} must be a string.")
        if step_reward.get("target") not in {"event", "final_answer", "episode", "state_snapshot"}:
            target.errors.append(f"step_rewards[{index}].target must be one of event, final_answer, episode, or state_snapshot.")
        if step_reward.get("reward_scale") not in REWARD_SCALES:
            target.errors.append(f"step_rewards[{index}].reward_scale must be one of {sorted(REWARD_SCALES)!r}.")
        for field_name in ("reward_delta", "rule_reward_delta", "allocation_weight", "episode_reward"):
            if not isinstance(step_reward.get(field_name), (int, float)) or isinstance(step_reward.get(field_name), bool):
                target.errors.append(f"step_rewards[{index}].{field_name} must be numeric.")
        if (
            not isinstance(step_reward.get("attribution_count"), int)
            or isinstance(step_reward.get("attribution_count"), bool)
            or step_reward.get("attribution_count") < 1
        ):
            target.errors.append(f"step_rewards[{index}].attribution_count must be a positive integer.")
        if (
            not isinstance(step_reward.get("allocation_index"), int)
            or isinstance(step_reward.get("allocation_index"), bool)
            or step_reward.get("allocation_index") < 0
        ):
            target.errors.append(f"step_rewards[{index}].allocation_index must be a non-negative integer.")
        if (
            isinstance(step_reward.get("allocation_index"), int)
            and not isinstance(step_reward.get("allocation_index"), bool)
            and isinstance(step_reward.get("attribution_count"), int)
            and not isinstance(step_reward.get("attribution_count"), bool)
            and step_reward["allocation_index"] >= step_reward["attribution_count"]
        ):
            target.errors.append(f"step_rewards[{index}].allocation_index must be less than attribution_count.")
        if not _is_int_between(step_reward.get("score"), 0, 100):
            target.errors.append(f"step_rewards[{index}].score must be an integer from 0 to 100.")
        if not isinstance(step_reward.get("passed"), bool):
            target.errors.append(f"step_rewards[{index}].passed must be a boolean.")
        if not isinstance(step_reward.get("critical"), bool):
            target.errors.append(f"step_rewards[{index}].critical must be a boolean.")
        if not _is_int_between(step_reward.get("penalty"), 0, 100):
            target.errors.append(f"step_rewards[{index}].penalty must be an integer from 0 to 100.")

        if step_reward.get("target") == "event":
            event_index = step_reward.get("event_index")
            if not isinstance(event_index, int) or isinstance(event_index, bool) or event_index < 0:
                target.errors.append(f"step_rewards[{index}].event_index must be a non-negative integer for event targets.")
            elif isinstance(episode, dict) and isinstance(episode.get("events"), list) and event_index >= len(episode["events"]):
                target.errors.append(
                    f"step_rewards[{index}].event_index {event_index} is outside episode {episode_id!r} events."
                )
        elif "event_index" in step_reward:
            target.errors.append(f"step_rewards[{index}].event_index is only valid when target is event.")

        if "evidence_ref" in step_reward:
            _validate_evidence_refs([step_reward.get("evidence_ref")], target, f"step_rewards[{index}].evidence_ref")

        reward_delta = step_reward.get("reward_delta")
        if (
            isinstance(episode_id, str)
            and isinstance(rule_id, str)
            and isinstance(reward_delta, (int, float))
            and not isinstance(reward_delta, bool)
        ):
            key = (episode_id, rule_id)
            step_delta_by_key[key] = round(step_delta_by_key.get(key, 0.0) + float(reward_delta), 6)
            if key not in rule_delta_by_key:
                target.errors.append(f"step_rewards[{index}] does not reference a terminal rule reward.")

    for key, actual in sorted(step_delta_by_key.items()):
        expected = rule_delta_by_key.get(key)
        if expected is not None and round(abs(actual - expected), 6) > 0.000001:
            episode_id, rule_id = key
            target.errors.append(
                f"step_rewards for episode {episode_id!r} rule {rule_id!r} sum to {actual}, expected {expected}."
            )


def _rule_reward_deltas_by_key(rewards: list[dict[str, Any]]) -> dict[tuple[str, str], float]:
    deltas: dict[tuple[str, str], float] = {}
    for reward in rewards:
        episode_id = reward.get("episode_id")
        if not isinstance(episode_id, str):
            continue
        rule_rewards = reward.get("rule_rewards")
        if not isinstance(rule_rewards, list):
            continue
        for rule_reward in rule_rewards:
            if not isinstance(rule_reward, dict) or rule_reward.get("passed") is True:
                continue
            rule_id = rule_reward.get("rule_id")
            reward_delta = rule_reward.get("reward_delta")
            if isinstance(rule_id, str) and isinstance(reward_delta, (int, float)) and not isinstance(reward_delta, bool):
                deltas[(episode_id, rule_id)] = round(float(reward_delta), 6)
    return deltas


def _validate_sft_records(
    sft: list[dict[str, Any]],
    target: ValidationTarget,
    episodes: list[dict[str, Any]],
) -> None:
    episode_by_id = {episode.get("episode_id"): episode for episode in episodes if isinstance(episode.get("episode_id"), str)}
    expected_ids = {
        str(episode.get("episode_id"))
        for episode in episodes
        if isinstance(episode.get("episode_id"), str)
        and isinstance(episode.get("outcome"), dict)
        and episode["outcome"].get("passed") is True
        and isinstance(episode.get("final_answer"), str)
        and bool(episode.get("final_answer"))
    }
    seen: set[str] = set()
    for index, sample in enumerate(sft):
        _require_equal(sample, "schema_version", RL_SFT_SCHEMA_VERSION, target, prefix=f"sft[{index}].")
        episode_id = _validate_training_view_common(sample, target, f"sft[{index}]", episode_by_id)
        if episode_id:
            if episode_id in seen:
                target.errors.append(f"sft[{index}].episode_id duplicates {episode_id!r}.")
            seen.add(episode_id)
            episode = episode_by_id.get(episode_id)
            outcome = episode.get("outcome") if isinstance(episode, dict) and isinstance(episode.get("outcome"), dict) else {}
            if outcome.get("passed") is not True:
                target.errors.append(f"sft[{index}].episode_id {episode_id!r} does not reference a passing episode.")
            if sample.get("quality_gate") != "passed_scorecard":
                target.errors.append(f"sft[{index}].quality_gate must be 'passed_scorecard'.")
            _validate_training_view_task_completion(sample, episode, target, f"sft[{index}]")
    missing = sorted(expected_ids - seen)
    if missing:
        target.errors.append(f"sft.jsonl missing passing episode samples: {missing!r}.")


def _validate_dpo_records(
    dpo: list[dict[str, Any]],
    target: ValidationTarget,
    preferences: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
) -> None:
    preference_by_id = {
        preference.get("preference_id"): preference
        for preference in preferences
        if isinstance(preference.get("preference_id"), str)
    }
    episode_by_id = {episode.get("episode_id"): episode for episode in episodes if isinstance(episode.get("episode_id"), str)}
    seen: set[str] = set()
    for index, pair in enumerate(dpo):
        _require_equal(pair, "schema_version", RL_DPO_SCHEMA_VERSION, target, prefix=f"dpo[{index}].")
        pair_id = pair.get("pair_id")
        preference_id = pair.get("preference_id")
        if not isinstance(pair_id, str) or not pair_id:
            target.errors.append(f"dpo[{index}].pair_id must be a non-empty string.")
        elif pair_id in seen:
            target.errors.append(f"dpo[{index}].pair_id duplicates {pair_id!r}.")
        else:
            seen.add(pair_id)
        if preference_id != pair_id:
            target.errors.append(f"dpo[{index}].preference_id must match pair_id.")
        preference = preference_by_id.get(preference_id)
        if preference is None:
            target.errors.append(f"dpo[{index}].preference_id {preference_id!r} does not reference a preference.")
            continue
        for field_name in ("task_family", "prompt", "chosen", "rejected", "reason", "source_artifact"):
            if not isinstance(pair.get(field_name), str):
                target.errors.append(f"dpo[{index}].{field_name} must be a string.")
        if pair.get("source_artifact") != "preferences.jsonl":
            target.errors.append(f"dpo[{index}].source_artifact must be 'preferences.jsonl'.")
        for field_name in ("chosen_score", "rejected_score", "score_gap"):
            if not _is_int_between(pair.get(field_name), 0, 100):
                target.errors.append(f"dpo[{index}].{field_name} must be an integer from 0 to 100.")
        _validate_messages(pair.get("chosen_messages"), target, f"dpo[{index}].chosen_messages")
        _validate_messages(pair.get("rejected_messages"), target, f"dpo[{index}].rejected_messages")
        for field_name in ("chosen_episode_id", "rejected_episode_id"):
            episode_id = pair.get(field_name)
            if not isinstance(episode_id, str) or not episode_id:
                target.errors.append(f"dpo[{index}].{field_name} must be a non-empty string.")
            elif episode_id not in episode_by_id:
                target.errors.append(f"dpo[{index}].{field_name} {episode_id!r} does not reference an episode.")
        _compare_dpo_to_preference(pair, preference, target, index)
    missing = sorted(set(preference_by_id) - seen)
    if missing:
        target.errors.append(f"dpo.jsonl missing preference pairs: {missing!r}.")


def _validate_reward_model_records(
    reward_model: list[dict[str, Any]],
    target: ValidationTarget,
    episodes: list[dict[str, Any]],
) -> None:
    episode_by_id = {episode.get("episode_id"): episode for episode in episodes if isinstance(episode.get("episode_id"), str)}
    seen: set[str] = set()
    for index, sample in enumerate(reward_model):
        _require_equal(sample, "schema_version", RL_REWARD_MODEL_SCHEMA_VERSION, target, prefix=f"reward_model[{index}].")
        episode_id = _validate_training_view_common(sample, target, f"reward_model[{index}]", episode_by_id)
        if episode_id:
            if episode_id in seen:
                target.errors.append(f"reward_model[{index}].episode_id duplicates {episode_id!r}.")
            seen.add(episode_id)
        if not isinstance(sample.get("passed"), bool):
            target.errors.append(f"reward_model[{index}].passed must be a boolean.")
        episode = episode_by_id.get(episode_id) if episode_id else None
        if isinstance(episode, dict):
            _validate_training_view_task_completion(sample, episode, target, f"reward_model[{index}]")
        for field_name in ("failed_rules", "critical_failures"):
            if not _is_string_list(sample.get(field_name)):
                target.errors.append(f"reward_model[{index}].{field_name} must be a list of strings.")
    missing = sorted(set(episode_by_id) - seen)
    if missing:
        target.errors.append(f"reward_model.jsonl missing episode samples: {missing!r}.")


def _validate_training_view_common(
    sample: dict[str, Any],
    target: ValidationTarget,
    label: str,
    episode_by_id: dict[Any, dict[str, Any]],
) -> str | None:
    sample_id = sample.get("sample_id")
    episode_id = sample.get("episode_id")
    if not isinstance(sample_id, str) or not sample_id:
        target.errors.append(f"{label}.sample_id must be a non-empty string.")
    if not isinstance(episode_id, str) or not episode_id:
        target.errors.append(f"{label}.episode_id must be a non-empty string.")
        episode = None
    else:
        episode = episode_by_id.get(episode_id)
        if episode is None:
            target.errors.append(f"{label}.episode_id {episode_id!r} does not reference an episode.")
    if isinstance(sample_id, str) and isinstance(episode_id, str) and sample_id != episode_id:
        target.errors.append(f"{label}.sample_id must match episode_id.")
    for field_name in ("scenario_id", "task_family", "prompt", "response", "source_artifact"):
        if not isinstance(sample.get(field_name), str):
            target.errors.append(f"{label}.{field_name} must be a string.")
    if sample.get("source_artifact") != "episodes.jsonl":
        target.errors.append(f"{label}.source_artifact must be 'episodes.jsonl'.")
    if not _is_int_between(sample.get("score"), 0, 100):
        target.errors.append(f"{label}.score must be an integer from 0 to 100.")
    if not isinstance(sample.get("reward"), (int, float)) or isinstance(sample.get("reward"), bool):
        target.errors.append(f"{label}.reward must be numeric.")
    _validate_messages(sample.get("messages"), target, f"{label}.messages")
    if isinstance(episode, dict):
        _validate_matching_source_fingerprints(sample, episode, target, label)
        outcome = episode.get("outcome") if isinstance(episode.get("outcome"), dict) else {}
        expected = {
            "scenario_id": episode.get("scenario_id"),
            "task_family": episode.get("task_family"),
            "prompt": episode.get("prompt"),
            "response": episode.get("final_answer"),
            "score": outcome.get("score"),
            "reward": outcome.get("reward"),
        }
        for field_name, expected_value in expected.items():
            if sample.get(field_name) != expected_value:
                target.errors.append(f"{label}.{field_name} does not match episode {episode_id!r}.")
    return episode_id if isinstance(episode_id, str) and episode_id else None


def _validate_training_view_task_completion(
    sample: dict[str, Any],
    episode: dict[str, Any],
    target: ValidationTarget,
    label: str,
) -> None:
    task = episode.get("task_completion") if isinstance(episode.get("task_completion"), dict) else {}
    if "task_completion_status" in sample and sample.get("task_completion_status") != task.get("status"):
        target.errors.append(f"{label}.task_completion_status does not match episode task_completion.status.")
    if "task_completion_passed" in sample and sample.get("task_completion_passed") != task.get("passed"):
        target.errors.append(f"{label}.task_completion_passed does not match episode task_completion.passed.")


def _compare_dpo_to_preference(
    pair: dict[str, Any],
    preference: dict[str, Any],
    target: ValidationTarget,
    index: int,
) -> None:
    expected = {
        "task_family": preference.get("task_family"),
        "prompt": preference.get("prompt"),
        "chosen_episode_id": preference.get("chosen_episode_id"),
        "rejected_episode_id": preference.get("rejected_episode_id"),
        "chosen_score": preference.get("chosen_score"),
        "rejected_score": preference.get("rejected_score"),
        "score_gap": preference.get("score_gap"),
        "reason": preference.get("reason"),
    }
    chosen = preference.get("chosen") if isinstance(preference.get("chosen"), dict) else {}
    rejected = preference.get("rejected") if isinstance(preference.get("rejected"), dict) else {}
    expected["chosen"] = chosen.get("final_answer")
    expected["rejected"] = rejected.get("final_answer")
    expected["chosen_source_fingerprint_status"] = chosen.get("source_fingerprint_status")
    expected["rejected_source_fingerprint_status"] = rejected.get("source_fingerprint_status")
    expected["chosen_source_fingerprints"] = chosen.get("source_fingerprints")
    expected["rejected_source_fingerprints"] = rejected.get("source_fingerprints")
    for field_name, expected_value in expected.items():
        if pair.get(field_name) != expected_value:
            target.errors.append(f"dpo[{index}].{field_name} does not match preference {preference.get('preference_id')!r}.")


def _validate_source_fingerprint_fields(
    row: dict[str, Any],
    target: ValidationTarget,
    label: str,
    *,
    warn_if_missing: bool,
) -> None:
    has_status = "source_fingerprint_status" in row
    has_fingerprints = "source_fingerprints" in row
    if not has_status and not has_fingerprints:
        if warn_if_missing:
            target.warnings.append(f"{label}.source_fingerprints is missing; rerun export-rl to refresh provenance fields.")
        return
    status = row.get("source_fingerprint_status")
    if status not in {"verified", "unverified"}:
        target.errors.append(f"{label}.source_fingerprint_status must be verified or unverified.")
    fingerprints = row.get("source_fingerprints")
    if not isinstance(fingerprints, dict):
        target.errors.append(f"{label}.source_fingerprints must be an object.")
        return
    scenario_sha = _validate_source_fingerprint_record(fingerprints.get("scenario"), target, f"{label}.source_fingerprints.scenario")
    trace_sha = _validate_source_fingerprint_record(
        fingerprints.get("source_trace"),
        target,
        f"{label}.source_fingerprints.source_trace",
    )
    if status == "verified" and not (scenario_sha and trace_sha):
        target.errors.append(f"{label}.source_fingerprint_status verified requires scenario and source_trace SHA-256 values.")
    if status == "unverified" and scenario_sha and trace_sha:
        target.errors.append(f"{label}.source_fingerprint_status should be verified when both source hashes are present.")


def _validate_source_fingerprint_record(value: Any, target: ValidationTarget, label: str) -> str | None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return None
    path = value.get("path")
    if path is not None and not isinstance(path, str):
        target.errors.append(f"{label}.path must be a string or null.")
    sha = value.get("sha256")
    if sha is not None and not _is_sha256(sha):
        target.errors.append(f"{label}.sha256 must be a SHA-256 hex string or null.")
        sha = None
    exists = value.get("exists")
    if exists is not None and not isinstance(exists, bool):
        target.errors.append(f"{label}.exists must be a boolean or null.")
    return sha if isinstance(sha, str) else None


def _validate_matching_source_fingerprints(
    row: dict[str, Any],
    episode: dict[str, Any],
    target: ValidationTarget,
    label: str,
) -> None:
    row_has = "source_fingerprint_status" in row or "source_fingerprints" in row
    episode_has = "source_fingerprint_status" in episode or "source_fingerprints" in episode
    if episode_has and not row_has:
        target.errors.append(f"{label}.source_fingerprints missing while referenced episode has source fingerprints.")
        return
    if not row_has:
        return
    _validate_source_fingerprint_fields(row, target, label, warn_if_missing=False)
    if row.get("source_fingerprint_status") != episode.get("source_fingerprint_status"):
        target.errors.append(f"{label}.source_fingerprint_status does not match episode {episode.get('episode_id')!r}.")
    if row.get("source_fingerprints") != episode.get("source_fingerprints"):
        target.errors.append(f"{label}.source_fingerprints does not match episode {episode.get('episode_id')!r}.")


def _validate_contract_fingerprint_status(row: dict[str, Any], target: ValidationTarget, label: str) -> None:
    status = row.get("contract_fingerprint_status")
    if status not in {"matched", "drifted", "unverified"}:
        target.errors.append(f"{label}.contract_fingerprint_status must be matched, drifted, or unverified.")
    if "contract_fingerprint_scope" in row and row.get("contract_fingerprint_scope") not in CONTRACT_SCOPES:
        target.errors.append(f"{label}.contract_fingerprint_scope must be one of {sorted(CONTRACT_SCOPES)!r}.")
    reasons = row.get("contract_fingerprint_reasons")
    if not _is_string_list(reasons):
        target.errors.append(f"{label}.contract_fingerprint_reasons must be a list of strings.")
        reasons = []
    if status in {"drifted", "unverified"} and not reasons:
        target.errors.append(f"{label}.contract_fingerprint_reasons must explain non-matched contract status.")
    if status == "matched" and reasons:
        target.errors.append(f"{label}.contract_fingerprint_reasons must be empty when status is matched.")
    fingerprints = row.get("contract_fingerprints")
    if not isinstance(fingerprints, dict):
        target.errors.append(f"{label}.contract_fingerprints must be an object.")
        return
    for side in ("baseline", "candidate"):
        value = fingerprints.get(side)
        if not isinstance(value, dict):
            target.errors.append(f"{label}.contract_fingerprints.{side} must be an object.")
            continue
        _validate_contract_fingerprint_inputs(value, target, f"{label}.contract_fingerprints.{side}")


def _validate_contract_fingerprint_inputs(value: dict[str, Any], target: ValidationTarget, label: str) -> None:
    for name in ("scenario", "source_trace"):
        record = value.get(name)
        if not isinstance(record, dict):
            target.errors.append(f"{label}.{name} must be an object.")
            continue
        path = record.get("path")
        if path is not None and not isinstance(path, str):
            target.errors.append(f"{label}.{name}.path must be a string or null.")
        sha = record.get("sha256")
        if sha is not None and not _is_sha256(sha):
            target.errors.append(f"{label}.{name}.sha256 must be a SHA-256 hex string or null.")


def _validate_source_fingerprint_coverage(value: Any, target: ValidationTarget, episodes: list[dict[str, Any]]) -> None:
    if not isinstance(value, dict):
        target.errors.append("dataset_metrics.source_fingerprint_coverage must be an object.")
        return
    expected = _expected_source_fingerprint_coverage(episodes)
    for field_name, expected_value in expected.items():
        if value.get(field_name) != expected_value:
            target.errors.append(
                f"dataset_metrics.source_fingerprint_coverage.{field_name} expected {expected_value}, got {value.get(field_name)!r}."
            )


def _expected_source_fingerprint_coverage(episodes: list[dict[str, Any]]) -> dict[str, int]:
    with_scenario = 0
    with_trace = 0
    fully_verified = 0
    for episode in episodes:
        fingerprints = episode.get("source_fingerprints") if isinstance(episode.get("source_fingerprints"), dict) else {}
        scenario = fingerprints.get("scenario") if isinstance(fingerprints.get("scenario"), dict) else {}
        source_trace = fingerprints.get("source_trace") if isinstance(fingerprints.get("source_trace"), dict) else {}
        scenario_sha = scenario.get("sha256")
        trace_sha = source_trace.get("sha256")
        if _is_sha256(scenario_sha):
            with_scenario += 1
        if _is_sha256(trace_sha):
            with_trace += 1
        if _is_sha256(scenario_sha) and _is_sha256(trace_sha):
            fully_verified += 1
    return {
        "episodes": len(episodes),
        "with_scenario_sha256": with_scenario,
        "with_source_trace_sha256": with_trace,
        "fully_verified": fully_verified,
        "unverified": len(episodes) - fully_verified,
    }


def _validate_messages(value: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(value, list) or len(value) != 2:
        target.errors.append(f"{label} must be a two-message user/assistant list.")
        return
    expected_roles = ["user", "assistant"]
    for index, message in enumerate(value):
        if not isinstance(message, dict):
            target.errors.append(f"{label}[{index}] must be an object.")
            continue
        if message.get("role") != expected_roles[index]:
            target.errors.append(f"{label}[{index}].role must be {expected_roles[index]!r}.")
        if not isinstance(message.get("content"), str):
            target.errors.append(f"{label}[{index}].content must be a string.")


def _validate_dataset_metrics(
    metrics: dict[str, Any],
    target: ValidationTarget,
    episodes: list[dict[str, Any]],
    rewards: list[dict[str, Any]],
    step_rewards: list[dict[str, Any]],
    preferences: list[dict[str, Any]],
    failure_modes: list[dict[str, Any]],
    sft: list[dict[str, Any]],
    dpo: list[dict[str, Any]],
    reward_model: list[dict[str, Any]],
) -> None:
    _require_equal(metrics, "schema_version", RL_DATASET_METRICS_SCHEMA_VERSION, target, prefix="dataset_metrics.")
    artifact_counts = metrics.get("artifact_counts")
    if not isinstance(artifact_counts, dict):
        target.errors.append("dataset_metrics.artifact_counts must be an object.")
        artifact_counts = {}
    expected_counts = {
        "episodes": len(episodes),
        "rewards": len(rewards),
        "step_rewards": len(step_rewards),
        "preferences": len(preferences),
        "failure_modes": len(failure_modes),
        "sft": len(sft),
        "dpo": len(dpo),
        "reward_model": len(reward_model),
    }
    for field_name, expected in expected_counts.items():
        if artifact_counts.get(field_name) != expected:
            target.errors.append(f"dataset_metrics.artifact_counts.{field_name} expected {expected}, got {artifact_counts.get(field_name)!r}.")

    scores = [_score_value(episode.get("outcome", {}).get("score")) for episode in episodes if isinstance(episode.get("outcome"), dict)]
    reward_values = [
        float(reward.get("reward"))
        for reward in rewards
        if isinstance(reward.get("reward"), (int, float)) and not isinstance(reward.get("reward"), bool)
    ]
    passed = sum(1 for episode in episodes if isinstance(episode.get("outcome"), dict) and episode["outcome"].get("passed") is True)
    failed = len(episodes) - passed
    expected_scalars = {
        "episode_count": len(episodes),
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / len(episodes), 4) if episodes else 0.0,
        "average_score": _average_number(scores),
        "min_score": min(scores) if scores else None,
        "max_score": max(scores) if scores else None,
        "average_reward": _average_number(reward_values),
        "min_reward": min(reward_values) if reward_values else None,
        "max_reward": max(reward_values) if reward_values else None,
    }
    for field_name, expected in expected_scalars.items():
        if metrics.get(field_name) != expected:
            target.errors.append(f"dataset_metrics.{field_name} expected {expected!r}, got {metrics.get(field_name)!r}.")

    expected_failed = _count_strings(rule for episode in episodes for rule in _outcome_strings(episode, "failed_rules"))
    expected_critical = _count_strings(rule for episode in episodes for rule in _outcome_strings(episode, "critical_failures"))
    if _count_rows(metrics.get("failed_rule_counts")) != expected_failed:
        target.errors.append("dataset_metrics.failed_rule_counts does not match episode failed_rules.")
    if _count_rows(metrics.get("critical_failure_counts")) != expected_critical:
        target.errors.append("dataset_metrics.critical_failure_counts does not match episode critical_failures.")

    if "source_fingerprint_coverage" in metrics:
        _validate_source_fingerprint_coverage(metrics.get("source_fingerprint_coverage"), target, episodes)
    else:
        target.warnings.append("dataset_metrics.source_fingerprint_coverage is missing; rerun export-rl to refresh provenance metrics.")
    if "task_completion" in metrics:
        expected_task_completion = _expected_task_completion_metrics(episodes)
        actual_task_completion = metrics.get("task_completion")
        if not isinstance(actual_task_completion, dict):
            target.errors.append("dataset_metrics.task_completion must be an object.")
        else:
            for field_name, expected in expected_task_completion.items():
                if actual_task_completion.get(field_name) != expected:
                    target.errors.append(
                        f"dataset_metrics.task_completion.{field_name} expected {expected!r}, got {actual_task_completion.get(field_name)!r}."
                    )
    else:
        target.warnings.append("dataset_metrics.task_completion is missing; rerun export-rl to refresh task-completion metrics.")
    _validate_dataset_family_metrics(metrics.get("task_families"), target, episodes, step_rewards, failure_modes, sft, dpo, reward_model)
    _validate_quality_flags(metrics.get("quality_flags"), target)
    if "metadata" in metrics:
        _validate_metadata(metrics.get("metadata"), target, "dataset_metrics.metadata")
    if not _is_string_list(metrics.get("recommended_checks")):
        target.errors.append("dataset_metrics.recommended_checks must be a list of strings.")


def _validate_dataset_family_metrics(
    value: Any,
    target: ValidationTarget,
    episodes: list[dict[str, Any]],
    step_rewards: list[dict[str, Any]],
    failure_modes: list[dict[str, Any]],
    sft: list[dict[str, Any]],
    dpo: list[dict[str, Any]],
    reward_model: list[dict[str, Any]],
) -> None:
    if not isinstance(value, list):
        target.errors.append("dataset_metrics.task_families must be a list.")
        return
    expected = _expected_dataset_family_metrics(episodes, step_rewards, failure_modes, sft, dpo, reward_model)
    actual_families: set[str] = set()
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            target.errors.append(f"dataset_metrics.task_families[{index}] must be an object.")
            continue
        family = row.get("task_family")
        if not isinstance(family, str) or not family:
            target.errors.append(f"dataset_metrics.task_families[{index}].task_family must be a non-empty string.")
            continue
        actual_families.add(family)
        expected_row = expected.get(family)
        if expected_row is None:
            target.errors.append(f"dataset_metrics.task_families[{index}] has unknown task_family {family!r}.")
            continue
        for field_name, expected_value in expected_row.items():
            if row.get(field_name) != expected_value:
                target.errors.append(
                    f"dataset_metrics.task_families[{index}].{field_name} expected {expected_value!r}, got {row.get(field_name)!r}."
                )
    missing = sorted(set(expected) - actual_families)
    if missing:
        target.errors.append(f"dataset_metrics.task_families missing families: {missing!r}.")


def _expected_dataset_family_metrics(
    episodes: list[dict[str, Any]],
    step_rewards: list[dict[str, Any]],
    failure_modes: list[dict[str, Any]],
    sft: list[dict[str, Any]],
    dpo: list[dict[str, Any]],
    reward_model: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    families = sorted(
        {
            str(item.get("task_family") or "unknown")
            for source in (episodes, step_rewards, failure_modes, sft, dpo, reward_model)
            for item in source
            if isinstance(item, dict)
        }
    )
    expected: dict[str, dict[str, Any]] = {}
    for family in families:
        family_episodes = [episode for episode in episodes if str(episode.get("task_family") or "unknown") == family]
        scores = [
            _score_value(episode.get("outcome", {}).get("score"))
            for episode in family_episodes
            if isinstance(episode.get("outcome"), dict)
        ]
        passed = sum(1 for episode in family_episodes if isinstance(episode.get("outcome"), dict) and episode["outcome"].get("passed") is True)
        task_metrics = _expected_task_completion_metrics(family_episodes)
        expected[family] = {
            "task_family": family,
            "episode_count": len(family_episodes),
            "passed": passed,
            "failed": len(family_episodes) - passed,
            "pass_rate": round(passed / len(family_episodes), 4) if family_episodes else 0.0,
            "task_completion_configured": task_metrics["configured_count"],
            "task_completion_complete": task_metrics["complete_count"],
            "task_completion_incomplete": task_metrics["incomplete_count"],
            "average_score": _average_number(scores),
            "step_reward_count": _count_family(step_rewards, family),
            "failure_mode_count": _count_family(failure_modes, family),
            "sft_count": _count_family(sft, family),
            "dpo_count": _count_family(dpo, family),
            "reward_model_count": _count_family(reward_model, family),
        }
    return expected


def _expected_task_completion_metrics(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = {"complete": 0, "incomplete": 0, "not_applicable": 0, "unknown": 0}
    configured = 0
    required_checks = 0
    passed_checks = 0
    for episode in episodes:
        task = episode.get("task_completion") if isinstance(episode.get("task_completion"), dict) else {}
        status = str(task.get("status") or "unknown")
        if status not in statuses:
            status = "unknown"
        statuses[status] += 1
        if task.get("task_evidence_configured") is True:
            configured += 1
        if _is_non_negative_int(task.get("required_check_count")):
            required_checks += int(task["required_check_count"])
        if _is_non_negative_int(task.get("passed_check_count")):
            passed_checks += int(task["passed_check_count"])
    return {
        "episode_count": len(episodes),
        "configured_count": configured,
        "complete_count": statuses["complete"],
        "incomplete_count": statuses["incomplete"],
        "not_applicable_count": statuses["not_applicable"],
        "unknown_count": statuses["unknown"],
        "required_check_count": required_checks,
        "passed_check_count": passed_checks,
        "check_pass_rate": round(passed_checks / required_checks, 4) if required_checks else 0.0,
    }


def _validate_quality_flags(value: Any, target: ValidationTarget) -> None:
    if not isinstance(value, list):
        target.errors.append("dataset_metrics.quality_flags must be a list.")
        return
    seen: set[str] = set()
    for index, flag in enumerate(value):
        if not isinstance(flag, dict):
            target.errors.append(f"dataset_metrics.quality_flags[{index}] must be an object.")
            continue
        flag_id = flag.get("id")
        if not isinstance(flag_id, str) or not flag_id:
            target.errors.append(f"dataset_metrics.quality_flags[{index}].id must be a non-empty string.")
        elif flag_id in seen:
            target.errors.append(f"dataset_metrics.quality_flags[{index}].id duplicates {flag_id!r}.")
        else:
            seen.add(flag_id)
        if flag.get("severity") not in {"info", "warning", "error"}:
            target.errors.append(f"dataset_metrics.quality_flags[{index}].severity must be info, warning, or error.")
        if not isinstance(flag.get("message"), str) or not flag.get("message"):
            target.errors.append(f"dataset_metrics.quality_flags[{index}].message must be a non-empty string.")


def _validate_dataset_card(path: Path, target: ValidationTarget) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        target.errors.append(f"DATASET_CARD.md could not be read: {exc}")
        return
    required = [
        "# Flight Recorder Dataset Card",
        "## Summary",
        "## Source Fingerprints",
        "## Artifact Counts",
        "## Task Families",
        "## Quality Flags",
        "## Boundaries",
    ]
    for marker in required:
        if marker not in text:
            target.errors.append(f"DATASET_CARD.md missing section marker {marker!r}.")


def _validate_preferences(preferences: list[dict[str, Any]], target: ValidationTarget, episodes: list[dict[str, Any]]) -> None:
    episode_by_id = {episode.get("episode_id"): episode for episode in episodes if isinstance(episode.get("episode_id"), str)}
    for index, preference in enumerate(preferences):
        _require_equal(preference, "schema_version", RL_PREFERENCE_SCHEMA_VERSION, target, prefix=f"preferences[{index}].")
        chosen_id = preference.get("chosen_episode_id")
        rejected_id = preference.get("rejected_episode_id")
        chosen = episode_by_id.get(chosen_id)
        rejected = episode_by_id.get(rejected_id)
        if chosen is None:
            target.errors.append(f"preferences[{index}].chosen_episode_id {chosen_id!r} does not reference an episode.")
        if rejected is None:
            target.errors.append(f"preferences[{index}].rejected_episode_id {rejected_id!r} does not reference an episode.")
        if chosen is not None and rejected is not None:
            chosen_score = chosen.get("outcome", {}).get("score") if isinstance(chosen.get("outcome"), dict) else None
            rejected_score = rejected.get("outcome", {}).get("score") if isinstance(rejected.get("outcome"), dict) else None
            if preference.get("chosen_score") != chosen_score:
                target.errors.append(f"preferences[{index}].chosen_score does not match chosen episode.")
            if preference.get("rejected_score") != rejected_score:
                target.errors.append(f"preferences[{index}].rejected_score does not match rejected episode.")
            if isinstance(chosen_score, int) and isinstance(rejected_score, int):
                expected_gap = chosen_score - rejected_score
                if preference.get("score_gap") != expected_gap:
                    target.errors.append(f"preferences[{index}].score_gap expected {expected_gap}, got {preference.get('score_gap')!r}.")
                if expected_gap <= 0:
                    target.errors.append(f"preferences[{index}] must prefer a strictly higher-scoring episode.")
            if chosen.get("task_family") != rejected.get("task_family"):
                target.errors.append(f"preferences[{index}] chosen/rejected task families differ.")
            if preference.get("task_family") != chosen.get("task_family"):
                target.errors.append(f"preferences[{index}].task_family does not match chosen episode.")


def _validate_failure_modes(
    failure_modes: list[dict[str, Any]],
    target: ValidationTarget,
    episodes: list[dict[str, Any]],
) -> None:
    episode_ids = {episode.get("episode_id") for episode in episodes if isinstance(episode.get("episode_id"), str)}
    seen: set[str] = set()
    for index, failure in enumerate(failure_modes):
        _require_equal(failure, "schema_version", RL_FAILURE_MODE_SCHEMA_VERSION, target, prefix=f"failure_modes[{index}].")
        failure_id = failure.get("failure_id")
        if not isinstance(failure_id, str) or not failure_id:
            target.errors.append(f"failure_modes[{index}].failure_id must be a non-empty string.")
        elif failure_id in seen:
            target.errors.append(f"failure_modes[{index}].failure_id duplicates {failure_id!r}.")
        else:
            seen.add(failure_id)
        episode_id = failure.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            target.errors.append(f"failure_modes[{index}].episode_id must be a non-empty string.")
        elif episode_id not in episode_ids:
            target.errors.append(f"failure_modes[{index}].episode_id {episode_id!r} does not reference an episode.")
        for field_name in ("scenario_id", "task_family", "rule_id", "rule_name", "summary"):
            if not isinstance(failure.get(field_name), str):
                target.errors.append(f"failure_modes[{index}].{field_name} must be a string.")
        if not isinstance(failure.get("critical"), bool):
            target.errors.append(f"failure_modes[{index}].critical must be a boolean.")
        if not _is_int_between(failure.get("penalty"), 0, 100):
            target.errors.append(f"failure_modes[{index}].penalty must be an integer from 0 to 100.")
        if not _is_int_between(failure.get("score"), 0, 100):
            target.errors.append(f"failure_modes[{index}].score must be an integer from 0 to 100.")
        if not isinstance(failure.get("reward"), (int, float)):
            target.errors.append(f"failure_modes[{index}].reward must be numeric.")
        if not isinstance(failure.get("evidence"), list):
            target.errors.append(f"failure_modes[{index}].evidence must be a list.")
        if "evidence_refs" in failure:
            _validate_evidence_refs(failure.get("evidence_refs"), target, f"failure_modes[{index}].evidence_refs")
        if not isinstance(failure.get("attribution"), list):
            target.errors.append(f"failure_modes[{index}].attribution must be a list.")
        episode = next((episode for episode in episodes if episode.get("episode_id") == episode_id), None)
        if isinstance(episode, dict):
            _validate_matching_source_fingerprints(failure, episode, target, f"failure_modes[{index}]")
        _validate_source_fingerprint_fields(failure, target, f"failure_modes[{index}]", warn_if_missing=False)


def _validate_curriculum(
    curriculum: dict[str, Any],
    target: ValidationTarget,
    episodes: list[dict[str, Any]],
    failure_modes: list[dict[str, Any]],
) -> None:
    _require_equal(curriculum, "schema_version", RL_CURRICULUM_SCHEMA_VERSION, target, prefix="curriculum.")
    if curriculum.get("episode_count") != len(episodes):
        target.errors.append(f"curriculum.episode_count expected {len(episodes)}, got {curriculum.get('episode_count')!r}.")
    if curriculum.get("failure_mode_count") != len(failure_modes):
        target.errors.append(
            f"curriculum.failure_mode_count expected {len(failure_modes)}, got {curriculum.get('failure_mode_count')!r}."
        )
    families = curriculum.get("task_families")
    if not isinstance(families, list):
        target.errors.append("curriculum.task_families must be a list.")
        return
    for family_index, family in enumerate(families):
        if not isinstance(family, dict):
            target.errors.append(f"curriculum.task_families[{family_index}] must be an object.")
            continue
        if not isinstance(family.get("task_family"), str) or not family.get("task_family"):
            target.errors.append(f"curriculum.task_families[{family_index}].task_family must be a non-empty string.")
        for field_name in ("episode_count", "passed", "failed"):
            if not isinstance(family.get(field_name), int) or isinstance(family.get(field_name), bool) or family.get(field_name) < 0:
                target.errors.append(f"curriculum.task_families[{family_index}].{field_name} must be a non-negative integer.")
        if not isinstance(family.get("average_score"), (int, float)):
            target.errors.append(f"curriculum.task_families[{family_index}].average_score must be numeric.")
        modes = family.get("failure_modes")
        if not isinstance(modes, list):
            target.errors.append(f"curriculum.task_families[{family_index}].failure_modes must be a list.")
            continue
        for mode_index, mode in enumerate(modes):
            if not isinstance(mode, dict):
                target.errors.append(f"curriculum.task_families[{family_index}].failure_modes[{mode_index}] must be an object.")
                continue
            if not isinstance(mode.get("rule_id"), str) or not mode.get("rule_id"):
                target.errors.append(
                    f"curriculum.task_families[{family_index}].failure_modes[{mode_index}].rule_id must be a non-empty string."
                )
            if not isinstance(mode.get("count"), int) or isinstance(mode.get("count"), bool) or mode.get("count") < 0:
                target.errors.append(
                    f"curriculum.task_families[{family_index}].failure_modes[{mode_index}].count must be a non-negative integer."
                )
            if not isinstance(mode.get("episode_ids"), list):
                target.errors.append(
                    f"curriculum.task_families[{family_index}].failure_modes[{mode_index}].episode_ids must be a list."
                )


def _validate_suite_summary(summary: dict[str, Any], target: ValidationTarget) -> None:
    _require_equal(summary, "schema_version", RUN_SUITE_SCHEMA_VERSION, target)
    runs = summary.get("runs")
    if not isinstance(runs, list):
        target.errors.append("suite_summary.runs must be a list.")
        runs = []
    errors = summary.get("errors")
    if not isinstance(errors, list):
        target.errors.append("suite_summary.errors must be a list.")
        errors = []

    if summary.get("total") != len(runs):
        target.errors.append(f"suite_summary.total expected {len(runs)}, got {summary.get('total')!r}.")
    passed = sum(1 for run in runs if isinstance(run, dict) and run.get("passed") is True)
    failed = len(runs) - passed
    if summary.get("passed") != passed:
        target.errors.append(f"suite_summary.passed expected {passed}, got {summary.get('passed')!r}.")
    if summary.get("failed") != failed:
        target.errors.append(f"suite_summary.failed expected {failed}, got {summary.get('failed')!r}.")
    if summary.get("error_count") != len(errors):
        target.errors.append(f"suite_summary.error_count expected {len(errors)}, got {summary.get('error_count')!r}.")

    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            target.errors.append(f"suite_summary.runs[{index}] must be an object.")
            continue
        for field_name in ("scenario_id", "scenario_title", "task_family", "scenario_path", "trace_path", "run_dir", "report", "scorecard", "lineage"):
            if not isinstance(run.get(field_name), str) or not run.get(field_name):
                target.errors.append(f"suite_summary.runs[{index}].{field_name} must be a non-empty string.")
        if not isinstance(run.get("passed"), bool):
            target.errors.append(f"suite_summary.runs[{index}].passed must be a boolean.")
        if not _is_int_between(run.get("score"), 0, 100):
            target.errors.append(f"suite_summary.runs[{index}].score must be an integer from 0 to 100.")
        if not _is_string_list(run.get("failed_rules")):
            target.errors.append(f"suite_summary.runs[{index}].failed_rules must be a list of strings.")
        if not _is_string_list(run.get("critical_failures")):
            target.errors.append(f"suite_summary.runs[{index}].critical_failures must be a list of strings.")
        for field_name in ("scenario_sha256", "trace_sha256"):
            if field_name in run and run.get(field_name) is not None and not _is_sha256(run.get(field_name)):
                target.errors.append(f"suite_summary.runs[{index}].{field_name} must be a SHA-256 hex string or null.")

    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        target.errors.append("suite_summary.metrics must be an object.")
    else:
        _validate_suite_metrics(metrics, target, [run for run in runs if isinstance(run, dict)])

    artifacts = summary.get("artifacts")
    if artifacts is not None and not isinstance(artifacts, dict):
        target.errors.append("suite_summary.artifacts must be an object when present.")
    if "metadata" in summary:
        _validate_metadata(summary.get("metadata"), target, "suite_summary.metadata")

    target.details.update(
        {
            "total": len(runs),
            "passed": passed,
            "failed": failed,
            "error_count": len(errors),
        }
    )


def _validate_suite_metrics(metrics: dict[str, Any], target: ValidationTarget, runs: list[dict[str, Any]]) -> None:
    scores = [_score_value(run.get("score")) for run in runs]
    passed = sum(1 for run in runs if run.get("passed") is True)
    failed = len(runs) - passed
    expected_pass_rate = round(passed / len(runs), 4) if runs else 0.0
    expected_average = round(sum(scores) / len(scores), 2) if scores else 0.0
    expected_min = min(scores) if scores else None
    expected_max = max(scores) if scores else None
    expected_failed_rules = _count_strings(rule for run in runs for rule in run.get("failed_rules", []))
    expected_critical = _count_strings(rule for run in runs for rule in run.get("critical_failures", []))

    expected_values = {
        "pass_rate": expected_pass_rate,
        "average_score": expected_average,
        "min_score": expected_min,
        "max_score": expected_max,
        "passed": passed,
        "failed": failed,
    }
    for field_name, expected in expected_values.items():
        if metrics.get(field_name) != expected:
            target.errors.append(f"suite_summary.metrics.{field_name} expected {expected!r}, got {metrics.get(field_name)!r}.")

    if _count_rows(metrics.get("failed_rule_counts")) != expected_failed_rules:
        target.errors.append("suite_summary.metrics.failed_rule_counts does not match run failed_rules.")
    if _count_rows(metrics.get("critical_failure_counts")) != expected_critical:
        target.errors.append("suite_summary.metrics.critical_failure_counts does not match run critical_failures.")
    _validate_suite_family_metrics(metrics.get("task_families"), target, runs)


def _validate_suite_family_metrics(value: Any, target: ValidationTarget, runs: list[dict[str, Any]]) -> None:
    if not isinstance(value, list):
        target.errors.append("suite_summary.metrics.task_families must be a list.")
        return

    expected: dict[str, dict[str, Any]] = {}
    for run in runs:
        family = str(run.get("task_family") or "unknown")
        bucket = expected.setdefault(family, {"runs": []})
        bucket["runs"].append(run)
    expected_rows: dict[str, dict[str, Any]] = {}
    for family, bucket in expected.items():
        family_runs = bucket["runs"]
        scores = [_score_value(run.get("score")) for run in family_runs]
        passed = sum(1 for run in family_runs if run.get("passed") is True)
        expected_rows[family] = {
            "total": len(family_runs),
            "passed": passed,
            "failed": len(family_runs) - passed,
            "pass_rate": round(passed / len(family_runs), 4) if family_runs else 0.0,
            "average_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
            "failed_rule_counts": _count_strings(rule for run in family_runs for rule in run.get("failed_rules", [])),
            "critical_failure_counts": _count_strings(rule for run in family_runs for rule in run.get("critical_failures", [])),
        }

    actual_families: set[str] = set()
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            target.errors.append(f"suite_summary.metrics.task_families[{index}] must be an object.")
            continue
        family = row.get("task_family")
        if not isinstance(family, str) or not family:
            target.errors.append(f"suite_summary.metrics.task_families[{index}].task_family must be a non-empty string.")
            continue
        actual_families.add(family)
        expected_row = expected_rows.get(family)
        if expected_row is None:
            target.errors.append(f"suite_summary.metrics.task_families[{index}] has unknown task_family {family!r}.")
            continue
        for field_name in ("total", "passed", "failed", "pass_rate", "average_score"):
            if row.get(field_name) != expected_row[field_name]:
                target.errors.append(
                    f"suite_summary.metrics.task_families[{index}].{field_name} "
                    f"expected {expected_row[field_name]!r}, got {row.get(field_name)!r}."
                )
        if _count_rows(row.get("failed_rule_counts")) != expected_row["failed_rule_counts"]:
            target.errors.append(f"suite_summary.metrics.task_families[{index}].failed_rule_counts does not match runs.")
        if "critical_failure_counts" not in row:
            target.warnings.append(
                f"suite_summary.metrics.task_families[{index}].critical_failure_counts is missing; rerun run-suite to refresh family metrics."
            )
        elif _count_rows(row.get("critical_failure_counts")) != expected_row["critical_failure_counts"]:
            target.errors.append(f"suite_summary.metrics.task_families[{index}].critical_failure_counts does not match runs.")
    missing = sorted(set(expected_rows) - actual_families)
    if missing:
        target.errors.append(f"suite_summary.metrics.task_families missing families: {missing!r}.")


def _validate_evidence_coverage(coverage: dict[str, Any], target: ValidationTarget) -> None:
    _require_equal(coverage, "schema_version", EVIDENCE_COVERAGE_SCHEMA_VERSION, target)
    runs = coverage.get("runs")
    if not isinstance(runs, list):
        target.errors.append("evidence_coverage.runs must be a list.")
        runs = []
    metrics = coverage.get("metrics")
    if not isinstance(metrics, dict):
        target.errors.append("evidence_coverage.metrics must be an object.")
        metrics = {}
    checks = coverage.get("checks")
    if not isinstance(checks, list):
        target.errors.append("evidence_coverage.checks must be a list.")
        checks = []
    if not isinstance(coverage.get("passed"), bool):
        target.errors.append("evidence_coverage.passed must be a boolean.")

    failed_checks = 0
    for index, check in enumerate(checks):
        if not isinstance(check, dict):
            target.errors.append(f"evidence_coverage.checks[{index}] must be an object.")
            continue
        if not isinstance(check.get("id"), str) or not check.get("id"):
            target.errors.append(f"evidence_coverage.checks[{index}].id must be a non-empty string.")
        if not isinstance(check.get("passed"), bool):
            target.errors.append(f"evidence_coverage.checks[{index}].passed must be a boolean.")
        elif not check["passed"]:
            failed_checks += 1

    if coverage.get("check_count") != len(checks):
        target.errors.append(f"evidence_coverage.check_count expected {len(checks)}, got {coverage.get('check_count')!r}.")
    if coverage.get("failed_check_count") != failed_checks:
        target.errors.append(
            f"evidence_coverage.failed_check_count expected {failed_checks}, got {coverage.get('failed_check_count')!r}."
        )
    if isinstance(coverage.get("passed"), bool) and coverage["passed"] != (failed_checks == 0):
        target.errors.append("evidence_coverage.passed must match failed_check_count.")

    run_totals = _validate_evidence_coverage_runs(runs, target)
    _validate_evidence_coverage_metrics(metrics, run_totals, target)

    warnings = coverage.get("warnings")
    if not isinstance(warnings, list) or not all(isinstance(item, str) for item in warnings):
        target.errors.append("evidence_coverage.warnings must be a list of strings.")

    target.details.update(
        {
            "run_count": run_totals["run_count"],
            "failed_rule_count": run_totals["failed_rule_count"],
            "failed_rule_evidence_rate": metrics.get("failed_rule_evidence_rate"),
        }
    )


def _validate_evidence_coverage_runs(runs: list[Any], target: ValidationTarget) -> dict[str, Any]:
    totals: dict[str, Any] = {
        "run_count": len(runs),
        "rule_count": 0,
        "failed_rule_count": 0,
        "critical_failed_rule_count": 0,
        "evidence_ref_count": 0,
        "failed_rule_evidence_ref_count": 0,
        "critical_failed_rule_evidence_ref_count": 0,
        "failed_rules_with_evidence": 0,
        "failed_rules_without_evidence": 0,
        "critical_failed_rules_with_evidence": 0,
        "critical_failed_rules_without_evidence": 0,
        "task_evidence_ref_count": 0,
        "evidence_target_counts": {},
        "failed_rule_evidence_target_counts": {},
    }
    for index, run in enumerate(runs):
        label = f"evidence_coverage.runs[{index}]"
        if not isinstance(run, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        for field_name in ("scenario_id", "scenario_title", "run_dir"):
            if not isinstance(run.get(field_name), str) or not run.get(field_name):
                target.errors.append(f"{label}.{field_name} must be a non-empty string.")
        if not isinstance(run.get("passed"), bool):
            target.errors.append(f"{label}.passed must be a boolean.")
        if not _is_int_between(run.get("score"), 0, 100):
            target.errors.append(f"{label}.score must be an integer from 0 to 100.")
        for field_name in (
            "rule_count",
            "failed_rule_count",
            "critical_failed_rule_count",
            "evidence_ref_count",
            "failed_rule_evidence_ref_count",
            "critical_failed_rule_evidence_ref_count",
            "failed_rules_with_evidence",
            "critical_failed_rules_with_evidence",
            "task_evidence_ref_count",
        ):
            if not _is_non_negative_int(run.get(field_name)):
                target.errors.append(f"{label}.{field_name} must be a non-negative integer.")
                continue
            totals[field_name] += run[field_name]
        for field_name in ("failed_rules_without_evidence", "critical_failed_rules_without_evidence"):
            values = run.get(field_name)
            if not _is_string_list(values):
                target.errors.append(f"{label}.{field_name} must be a list of strings.")
                continue
            totals[field_name] += len(values)
        event_count = run.get("event_count")
        if event_count is not None and not _is_non_negative_int(event_count):
            target.errors.append(f"{label}.event_count must be a non-negative integer or null.")
        _merge_count_rows(totals["evidence_target_counts"], run.get("evidence_target_counts"), target, f"{label}.evidence_target_counts")
        _merge_count_rows(
            totals["failed_rule_evidence_target_counts"],
            run.get("failed_rule_evidence_target_counts"),
            target,
            f"{label}.failed_rule_evidence_target_counts",
        )
        rules = run.get("rules")
        if not isinstance(rules, list):
            target.errors.append(f"{label}.rules must be a list.")
        elif len(rules) != run.get("rule_count"):
            target.errors.append(f"{label}.rule_count must match rules length.")
    return totals


def _validate_evidence_coverage_metrics(metrics: dict[str, Any], totals: dict[str, Any], target: ValidationTarget) -> None:
    expected_int_fields = (
        "run_count",
        "rule_count",
        "failed_rule_count",
        "critical_failed_rule_count",
        "evidence_ref_count",
        "failed_rule_evidence_ref_count",
        "critical_failed_rule_evidence_ref_count",
        "failed_rules_with_evidence",
        "failed_rules_without_evidence",
        "critical_failed_rules_with_evidence",
        "critical_failed_rules_without_evidence",
        "task_evidence_ref_count",
    )
    for field_name in expected_int_fields:
        if metrics.get(field_name) != totals[field_name]:
            target.errors.append(f"evidence_coverage.metrics.{field_name} expected {totals[field_name]!r}, got {metrics.get(field_name)!r}.")

    expected_failed_rate = _rate_value(totals["failed_rules_with_evidence"], totals["failed_rule_count"])
    expected_critical_rate = _rate_value(totals["critical_failed_rules_with_evidence"], totals["critical_failed_rule_count"])
    if metrics.get("failed_rule_evidence_rate") != expected_failed_rate:
        target.errors.append(
            f"evidence_coverage.metrics.failed_rule_evidence_rate expected {expected_failed_rate!r}, "
            f"got {metrics.get('failed_rule_evidence_rate')!r}."
        )
    if metrics.get("critical_failed_rule_evidence_rate") != expected_critical_rate:
        target.errors.append(
            f"evidence_coverage.metrics.critical_failed_rule_evidence_rate expected {expected_critical_rate!r}, "
            f"got {metrics.get('critical_failed_rule_evidence_rate')!r}."
        )

    evidence_counts = _count_rows(metrics.get("evidence_target_counts"))
    failed_counts = _count_rows(metrics.get("failed_rule_evidence_target_counts"))
    if evidence_counts != totals["evidence_target_counts"]:
        target.errors.append("evidence_coverage.metrics.evidence_target_counts does not match runs.")
    if failed_counts != totals["failed_rule_evidence_target_counts"]:
        target.errors.append("evidence_coverage.metrics.failed_rule_evidence_target_counts does not match runs.")
    if metrics.get("event_evidence_ref_count") != totals["evidence_target_counts"].get("event", 0):
        target.errors.append("evidence_coverage.metrics.event_evidence_ref_count does not match evidence_target_counts.")
    if metrics.get("final_answer_evidence_ref_count") != totals["evidence_target_counts"].get("final_answer", 0):
        target.errors.append("evidence_coverage.metrics.final_answer_evidence_ref_count does not match evidence_target_counts.")
    if metrics.get("episode_evidence_ref_count") != totals["evidence_target_counts"].get("episode", 0):
        target.errors.append("evidence_coverage.metrics.episode_evidence_ref_count does not match evidence_target_counts.")

    rule_coverage = metrics.get("rule_coverage")
    if not isinstance(rule_coverage, list):
        target.errors.append("evidence_coverage.metrics.rule_coverage must be a list.")
        return
    for index, row in enumerate(rule_coverage):
        if not isinstance(row, dict):
            target.errors.append(f"evidence_coverage.metrics.rule_coverage[{index}] must be an object.")
            continue
        if "target_counts" in row:
            target.errors.append(f"evidence_coverage.metrics.rule_coverage[{index}].target_counts is internal and must not be present.")
        if not isinstance(row.get("rule_id"), str) or not row.get("rule_id"):
            target.errors.append(f"evidence_coverage.metrics.rule_coverage[{index}].rule_id must be a non-empty string.")
        for field_name in (
            "rule_count",
            "passed",
            "failed",
            "critical_failed",
            "evidence_ref_count",
            "negative_evidence_ref_count",
            "failed_with_evidence",
            "failed_without_evidence",
        ):
            if not _is_non_negative_int(row.get(field_name)):
                target.errors.append(f"evidence_coverage.metrics.rule_coverage[{index}].{field_name} must be a non-negative integer.")


def _validate_trace_observability(observability: dict[str, Any], target: ValidationTarget) -> None:
    _require_equal(observability, "schema_version", TRACE_OBSERVABILITY_SCHEMA_VERSION, target)
    runs = observability.get("runs")
    if not isinstance(runs, list):
        target.errors.append("trace_observability.runs must be a list.")
        runs = []
    metrics = observability.get("metrics")
    if not isinstance(metrics, dict):
        target.errors.append("trace_observability.metrics must be an object.")
        metrics = {}
    checks = observability.get("checks")
    if not isinstance(checks, list):
        target.errors.append("trace_observability.checks must be a list.")
        checks = []
    if not isinstance(observability.get("passed"), bool):
        target.errors.append("trace_observability.passed must be a boolean.")

    failed_checks = 0
    for index, check in enumerate(checks):
        if not isinstance(check, dict):
            target.errors.append(f"trace_observability.checks[{index}] must be an object.")
            continue
        if not isinstance(check.get("id"), str) or not check.get("id"):
            target.errors.append(f"trace_observability.checks[{index}].id must be a non-empty string.")
        if not isinstance(check.get("passed"), bool):
            target.errors.append(f"trace_observability.checks[{index}].passed must be a boolean.")
        elif not check["passed"]:
            failed_checks += 1
    if observability.get("check_count") != len(checks):
        target.errors.append(f"trace_observability.check_count expected {len(checks)}, got {observability.get('check_count')!r}.")
    if observability.get("failed_check_count") != failed_checks:
        target.errors.append(
            f"trace_observability.failed_check_count expected {failed_checks}, got {observability.get('failed_check_count')!r}."
        )
    if isinstance(observability.get("passed"), bool) and observability["passed"] != (failed_checks == 0):
        target.errors.append("trace_observability.passed must match failed_check_count.")

    run_totals = _validate_trace_observability_runs(runs, target)
    _validate_trace_observability_metrics(metrics, run_totals, target)
    warnings = observability.get("warnings")
    if not isinstance(warnings, list) or not all(isinstance(item, str) for item in warnings):
        target.errors.append("trace_observability.warnings must be a list of strings.")
    target.details.update(
        {
            "run_count": run_totals["run_count"],
            "average_event_count": metrics.get("average_event_count"),
            "event_type_count": metrics.get("event_type_count"),
            "tool_or_api_run_rate": metrics.get("tool_or_api_run_rate"),
        }
    )


def _validate_trace_observability_runs(runs: list[Any], target: ValidationTarget) -> dict[str, Any]:
    totals: dict[str, Any] = {
        "run_count": len(runs),
        "total_event_count": 0,
        "runs_with_final_answer": 0,
        "empty_final_answer_count": 0,
        "runs_with_tool_or_api_events": 0,
        "tool_call_count": 0,
        "tool_result_count": 0,
        "api_call_count": 0,
        "subagent_event_count": 0,
        "approval_event_count": 0,
        "event_type_counts": {},
        "source_format_counts": {},
        "model_counts": {},
        "risk_counts": {},
        "event_counts": [],
    }
    for index, run in enumerate(runs):
        label = f"trace_observability.runs[{index}]"
        if not isinstance(run, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        for field_name in ("run_dir", "scenario_id", "source_format", "model"):
            if not isinstance(run.get(field_name), str) or not run.get(field_name):
                target.errors.append(f"{label}.{field_name} must be a non-empty string.")
        if run.get("passed") is not None and not isinstance(run.get("passed"), bool):
            target.errors.append(f"{label}.passed must be a boolean or null.")
        if run.get("score") is not None and not _is_int_between(run.get("score"), 0, 100):
            target.errors.append(f"{label}.score must be an integer from 0 to 100 or null.")
        for field_name in (
            "event_count",
            "event_type_count",
            "final_answer_chars",
            "tool_call_count",
            "tool_result_count",
            "api_call_count",
            "subagent_event_count",
            "approval_event_count",
        ):
            if not _is_non_negative_int(run.get(field_name)):
                target.errors.append(f"{label}.{field_name} must be a non-negative integer.")
        event_count = _non_negative_int_value(run.get("event_count"))
        totals["event_counts"].append(event_count)
        totals["total_event_count"] += event_count
        for field_name in ("tool_call_count", "tool_result_count", "api_call_count", "subagent_event_count", "approval_event_count"):
            totals[field_name] += _non_negative_int_value(run.get(field_name))
        if not isinstance(run.get("has_final_answer"), bool):
            target.errors.append(f"{label}.has_final_answer must be a boolean.")
        elif run["has_final_answer"]:
            totals["runs_with_final_answer"] += 1
        else:
            totals["empty_final_answer_count"] += 1
        if not isinstance(run.get("has_tool_or_api_events"), bool):
            target.errors.append(f"{label}.has_tool_or_api_events must be a boolean.")
        elif run["has_tool_or_api_events"]:
            totals["runs_with_tool_or_api_events"] += 1
        _merge_count_rows(totals["event_type_counts"], run.get("event_types"), target, f"{label}.event_types")
        event_types = _count_rows(run.get("event_types"))
        if event_types is not None and run.get("event_type_count") != len(event_types):
            target.errors.append(f"{label}.event_type_count must match event_types length.")
        source_format = str(run.get("source_format") or "unknown")
        model = str(run.get("model") or "unknown")
        totals["source_format_counts"][source_format] = totals["source_format_counts"].get(source_format, 0) + 1
        totals["model_counts"][model] = totals["model_counts"].get(model, 0) + 1
        risks = run.get("risks")
        if not _is_string_list(risks):
            target.errors.append(f"{label}.risks must be a list of strings.")
        else:
            for risk in risks:
                totals["risk_counts"][risk] = totals["risk_counts"].get(risk, 0) + 1
    return totals


def _validate_trace_observability_metrics(metrics: dict[str, Any], totals: dict[str, Any], target: ValidationTarget) -> None:
    event_counts = totals["event_counts"]
    expected = {
        "run_count": totals["run_count"],
        "total_event_count": totals["total_event_count"],
        "min_event_count": min(event_counts) if event_counts else 0,
        "max_event_count": max(event_counts) if event_counts else 0,
        "event_type_count": len(totals["event_type_counts"]),
        "runs_with_final_answer": totals["runs_with_final_answer"],
        "empty_final_answer_count": totals["empty_final_answer_count"],
        "runs_with_tool_or_api_events": totals["runs_with_tool_or_api_events"],
        "tool_call_count": totals["tool_call_count"],
        "tool_result_count": totals["tool_result_count"],
        "api_call_count": totals["api_call_count"],
        "subagent_event_count": totals["subagent_event_count"],
        "approval_event_count": totals["approval_event_count"],
    }
    for field_name, expected_value in expected.items():
        if metrics.get(field_name) != expected_value:
            target.errors.append(f"trace_observability.metrics.{field_name} expected {expected_value!r}, got {metrics.get(field_name)!r}.")
    average = round(totals["total_event_count"] / totals["run_count"], 2) if totals["run_count"] else 0.0
    if metrics.get("average_event_count") != average:
        target.errors.append(f"trace_observability.metrics.average_event_count expected {average!r}, got {metrics.get('average_event_count')!r}.")
    final_answer_rate = _rate_value(totals["runs_with_final_answer"], totals["run_count"])
    if metrics.get("final_answer_rate") != final_answer_rate:
        target.errors.append(f"trace_observability.metrics.final_answer_rate expected {final_answer_rate!r}, got {metrics.get('final_answer_rate')!r}.")
    tool_or_api_rate = _rate_value(totals["runs_with_tool_or_api_events"], totals["run_count"])
    if metrics.get("tool_or_api_run_rate") != tool_or_api_rate:
        target.errors.append(f"trace_observability.metrics.tool_or_api_run_rate expected {tool_or_api_rate!r}, got {metrics.get('tool_or_api_run_rate')!r}.")
    for field_name, expected_counts in (
        ("event_type_counts", totals["event_type_counts"]),
        ("source_format_counts", totals["source_format_counts"]),
        ("model_counts", totals["model_counts"]),
        ("risk_counts", totals["risk_counts"]),
    ):
        counts = _count_rows(metrics.get(field_name))
        if counts != expected_counts:
            target.errors.append(f"trace_observability.metrics.{field_name} does not match runs.")


def _validate_evidence_bundle(bundle: dict[str, Any], target: ValidationTarget) -> None:
    _require_equal(bundle, "schema_version", EVIDENCE_BUNDLE_SCHEMA_VERSION, target)
    if not isinstance(bundle.get("bundle_path"), str) or not bundle.get("bundle_path"):
        target.errors.append("evidence_bundle.bundle_path must be a non-empty string.")
    if not isinstance(bundle.get("passed"), bool):
        target.errors.append("evidence_bundle.passed must be a boolean.")

    checks = bundle.get("checks")
    if not isinstance(checks, list):
        target.errors.append("evidence_bundle.checks must be a list.")
        checks = []
    artifacts = bundle.get("artifacts")
    if not isinstance(artifacts, dict):
        target.errors.append("evidence_bundle.artifacts must be an object.")
        artifacts = {}
    metrics = bundle.get("metrics")
    if not isinstance(metrics, dict):
        target.errors.append("evidence_bundle.metrics must be an object.")
        metrics = {}
    notes = bundle.get("notes")
    if not isinstance(notes, list) or not all(isinstance(item, str) for item in notes):
        target.errors.append("evidence_bundle.notes must be a list of strings.")

    failed_checks = _validate_evidence_bundle_checks(checks, target)
    if bundle.get("check_count") != len(checks):
        target.errors.append(f"evidence_bundle.check_count expected {len(checks)}, got {bundle.get('check_count')!r}.")
    if bundle.get("failed_check_count") != failed_checks:
        target.errors.append(
            f"evidence_bundle.failed_check_count expected {failed_checks}, got {bundle.get('failed_check_count')!r}."
        )
    expected_passed = failed_checks == 0
    if isinstance(bundle.get("passed"), bool) and bundle["passed"] != expected_passed:
        target.errors.append("evidence_bundle.passed must match failed_check_count.")
    expected_readiness = "ready" if expected_passed else "blocked"
    if bundle.get("readiness") != expected_readiness:
        target.errors.append(f"evidence_bundle.readiness expected {expected_readiness!r}, got {bundle.get('readiness')!r}.")
    if "decision" in bundle:
        _validate_evidence_bundle_decision(bundle.get("decision"), expected_readiness, failed_checks, artifacts, metrics, target)
    if not artifacts:
        target.errors.append("evidence_bundle.artifacts must not be empty.")
    for name, record in artifacts.items():
        _validate_evidence_bundle_artifact_record(name, record, target)
    _validate_evidence_bundle_metrics(metrics, target)
    target.details.update(
        {
            "readiness": bundle.get("readiness"),
            "check_count": len(checks),
            "failed_check_count": failed_checks,
            "artifact_count": len(artifacts),
        }
    )


def _validate_replay_bundle(manifest: dict[str, Any], bundle_dir: Path, target: ValidationTarget) -> None:
    _require_equal(manifest, "schema_version", REPLAY_BUNDLE_SCHEMA_VERSION, target, prefix="replay_bundle.")
    lineage_name = manifest.get("lineage")
    if not isinstance(lineage_name, str) or not lineage_name:
        target.errors.append("replay_bundle.lineage must be a non-empty string.")
        lineage_name = "artifact_lineage.json"
    lineage_path = bundle_dir / lineage_name
    lineage = _read_object(lineage_path, target, "artifact_lineage.json")
    if lineage is not None:
        _validate_replay_bundle_lineage(lineage, manifest, bundle_dir, target)

    inputs = manifest.get("inputs")
    if not isinstance(inputs, list):
        target.errors.append("replay_bundle.inputs must be a list.")
        inputs = []
    if manifest.get("input_count") != len(inputs):
        target.errors.append(f"replay_bundle.input_count expected {len(inputs)}, got {manifest.get('input_count')!r}.")
    input_records: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(inputs):
        label = f"replay_bundle.inputs[{index}]"
        if not isinstance(record, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        name = record.get("name")
        if not isinstance(name, str) or not name:
            target.errors.append(f"{label}.name must be a non-empty string.")
            continue
        input_records[name] = record
        _validate_replay_bundle_input_record(record, bundle_dir, target, label)
    for required_name in ("scenario", "source_trace"):
        if required_name not in input_records:
            target.errors.append(f"replay_bundle.inputs missing {required_name}.")

    replay = manifest.get("replay")
    if not isinstance(replay, dict):
        target.errors.append("replay_bundle.replay must be an object.")
        replay = {}
    if replay.get("self_contained") is not True:
        target.errors.append("replay_bundle.replay.self_contained must be true.")
    argv = replay.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) and item for item in argv):
        target.errors.append("replay_bundle.replay.argv must be a list of non-empty strings.")
        argv = []
    else:
        if argv[:4] != ["python", "-m", "flightrecorder", "run"]:
            target.errors.append("replay_bundle.replay.argv must start with python -m flightrecorder run.")
        _validate_replay_bundle_argv_inputs(argv, input_records, target)
    if not isinstance(replay.get("command"), str) or not replay.get("command"):
        target.errors.append("replay_bundle.replay.command must be a non-empty string.")
    notes = manifest.get("notes")
    if not isinstance(notes, list) or not all(isinstance(item, str) for item in notes):
        target.errors.append("replay_bundle.notes must be a list of strings.")
    target.details.update(
        {
            "input_count": len(input_records),
            "lineage": lineage_name,
            "self_contained": replay.get("self_contained") is True,
        }
    )


def _validate_replay_bundle_lineage(
    lineage: dict[str, Any],
    manifest: dict[str, Any],
    bundle_dir: Path,
    target: ValidationTarget,
) -> None:
    _require_equal(lineage, "schema_version", LINEAGE_SCHEMA_VERSION, target, prefix="artifact_lineage.")
    portable = lineage.get("portable_replay_bundle")
    if not isinstance(portable, dict):
        target.errors.append("artifact_lineage.portable_replay_bundle must be an object.")
        portable = {}
    _require_equal(portable, "schema_version", REPLAY_BUNDLE_SCHEMA_VERSION, target, prefix="artifact_lineage.portable_replay_bundle.")
    if portable.get("input_count") != manifest.get("input_count"):
        target.errors.append("artifact_lineage.portable_replay_bundle.input_count must match replay_bundle.input_count.")
    replay = lineage.get("replay")
    manifest_replay = manifest.get("replay") if isinstance(manifest.get("replay"), dict) else {}
    if not isinstance(replay, dict):
        target.errors.append("artifact_lineage.replay must be an object.")
        return
    if replay.get("self_contained") is not True:
        target.errors.append("artifact_lineage.replay.self_contained must be true for replay bundles.")
    if replay.get("argv") != manifest_replay.get("argv"):
        target.errors.append("artifact_lineage.replay.argv must match replay_bundle.replay.argv.")
    if replay.get("command") != manifest_replay.get("command"):
        target.errors.append("artifact_lineage.replay.command must match replay_bundle.replay.command.")

    inputs = _lineage_records(lineage.get("inputs"), target, "artifact_lineage.inputs")
    fingerprints = replay.get("input_fingerprints") if isinstance(replay.get("input_fingerprints"), dict) else {}
    if not isinstance(fingerprints, dict):
        target.errors.append("artifact_lineage.replay.input_fingerprints must be an object.")
        fingerprints = {}
    for input_record in manifest.get("inputs", []) if isinstance(manifest.get("inputs"), list) else []:
        if not isinstance(input_record, dict):
            continue
        name = input_record.get("name")
        if not isinstance(name, str) or not name:
            continue
        fingerprint = fingerprints.get(name)
        if not isinstance(fingerprint, dict):
            target.errors.append(f"artifact_lineage.replay.input_fingerprints missing {name}.")
            continue
        for field_name in ("path", "sha256"):
            if fingerprint.get(field_name) != input_record.get(field_name):
                target.errors.append(f"artifact_lineage.replay.input_fingerprints.{name}.{field_name} must match replay_bundle input.")
        if fingerprint.get("exists") is not True:
            target.errors.append(f"artifact_lineage.replay.input_fingerprints.{name}.exists must be true.")
        lineage_input = inputs.get(name)
        if lineage_input is not None:
            for field_name in ("path", "sha256"):
                if lineage_input.get(field_name) != input_record.get(field_name):
                    target.errors.append(f"artifact_lineage.inputs.{name}.{field_name} must match replay_bundle input.")
    _validate_replay_bundle_copied_lineage_inputs(inputs, bundle_dir, target)


def _validate_replay_bundle_input_record(record: dict[str, Any], bundle_dir: Path, target: ValidationTarget, label: str) -> None:
    path_value = record.get("path")
    if not isinstance(path_value, str) or not path_value:
        target.errors.append(f"{label}.path must be a non-empty string.")
        return
    if not _is_safe_relative_path(path_value):
        target.errors.append(f"{label}.path must be relative to the replay bundle directory.")
        return
    file_path = bundle_dir / path_value
    if not file_path.exists() or not file_path.is_file():
        target.errors.append(f"{label}.path does not resolve to a bundled file.")
        return
    if not _is_non_negative_int(record.get("size_bytes")):
        target.errors.append(f"{label}.size_bytes must be a non-negative integer.")
    elif file_path.stat().st_size != record.get("size_bytes"):
        target.errors.append(f"{label}.size_bytes does not match the bundled file.")
    if not _is_sha256(record.get("sha256")):
        target.errors.append(f"{label}.sha256 must be a SHA-256 hex string.")
    elif _sha256(file_path) != record.get("sha256"):
        target.errors.append(f"{label}.sha256 does not match the bundled file.")
    if "source_path" in record and not isinstance(record.get("source_path"), str):
        target.errors.append(f"{label}.source_path must be a string when present.")


def _validate_replay_bundle_argv_inputs(argv: list[str], inputs: dict[str, dict[str, Any]], target: ValidationTarget) -> None:
    flag_to_input = {"--scenario": "scenario", "--trace": "source_trace", "--state": "source_state_snapshot"}
    for flag, input_name in flag_to_input.items():
        if flag not in argv:
            if flag in {"--scenario", "--trace"}:
                target.errors.append(f"replay_bundle.replay.argv missing {flag}.")
            continue
        index = argv.index(flag)
        if index + 1 >= len(argv):
            target.errors.append(f"replay_bundle.replay.argv missing value for {flag}.")
            continue
        expected = inputs.get(input_name, {}).get("path")
        if expected is not None and argv[index + 1] != expected:
            target.errors.append(f"replay_bundle.replay.argv value for {flag} must match replay_bundle.inputs.{input_name}.path.")


def _validate_replay_bundle_copied_lineage_inputs(inputs: dict[str, dict[str, Any]], bundle_dir: Path, target: ValidationTarget) -> None:
    for name in ("scenario", "source_trace", "source_state_snapshot"):
        record = inputs.get(name)
        if record is None:
            if name != "source_state_snapshot":
                target.errors.append(f"artifact_lineage.inputs missing {name}.")
            continue
        path_value = record.get("path")
        if not isinstance(path_value, str) or not path_value:
            target.errors.append(f"artifact_lineage.inputs.{name}.path must be a non-empty string.")
            continue
        if not _is_safe_relative_path(path_value):
            target.errors.append(f"artifact_lineage.inputs.{name}.path must be relative to the replay bundle directory.")
            continue
        file_path = bundle_dir / path_value
        if not file_path.exists() or not file_path.is_file():
            target.errors.append(f"artifact_lineage.inputs.{name}.path does not resolve to a bundled file.")
            continue
        if record.get("exists") is not True:
            target.errors.append(f"artifact_lineage.inputs.{name}.exists must be true.")
        if not _is_sha256(record.get("sha256")):
            target.errors.append(f"artifact_lineage.inputs.{name}.sha256 must be a SHA-256 hex string.")
        elif _sha256(file_path) != record.get("sha256"):
            target.errors.append(f"artifact_lineage.inputs.{name}.sha256 does not match the bundled file.")


def _is_safe_relative_path(value: str) -> bool:
    path = Path(value)
    return not path.is_absolute() and not _is_windows_absolute(value) and ".." not in path.parts


def _validate_evidence_bundle_checks(checks: list[Any], target: ValidationTarget) -> int:
    failed_checks = 0
    for index, check in enumerate(checks):
        label = f"evidence_bundle.checks[{index}]"
        if not isinstance(check, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        if not isinstance(check.get("id"), str) or not check.get("id"):
            target.errors.append(f"{label}.id must be a non-empty string.")
        if not isinstance(check.get("passed"), bool):
            target.errors.append(f"{label}.passed must be a boolean.")
        elif not check["passed"]:
            failed_checks += 1
        for field_name in ("actual", "expected", "scope"):
            if not isinstance(check.get(field_name), dict):
                target.errors.append(f"{label}.{field_name} must be an object.")
        if not isinstance(check.get("summary"), str) or not check.get("summary"):
            target.errors.append(f"{label}.summary must be a non-empty string.")
    return failed_checks


def _validate_evidence_bundle_decision(
    decision: Any,
    expected_readiness: str,
    failed_checks: int,
    artifacts: dict[str, Any],
    metrics: dict[str, Any],
    target: ValidationTarget,
) -> None:
    if not isinstance(decision, dict):
        target.errors.append("evidence_bundle.decision must be an object when present.")
        return
    if decision.get("readiness") != expected_readiness:
        target.errors.append(
            f"evidence_bundle.decision.readiness expected {expected_readiness!r}, got {decision.get('readiness')!r}."
        )
    expected_recommendation = "promote_handoff" if expected_readiness == "ready" else "block_handoff"
    if decision.get("recommendation") != expected_recommendation:
        target.errors.append(
            "evidence_bundle.decision.recommendation expected "
            f"{expected_recommendation!r}, got {decision.get('recommendation')!r}."
        )
    if not isinstance(decision.get("summary"), str) or not decision.get("summary"):
        target.errors.append("evidence_bundle.decision.summary must be a non-empty string.")
    if decision.get("blocking_check_count") != failed_checks:
        target.errors.append(
            f"evidence_bundle.decision.blocking_check_count expected {failed_checks}, got {decision.get('blocking_check_count')!r}."
        )
    blocking_checks = decision.get("blocking_checks")
    if not isinstance(blocking_checks, list):
        target.errors.append("evidence_bundle.decision.blocking_checks must be a list.")
    else:
        if len(blocking_checks) != failed_checks:
            target.errors.append(
                f"evidence_bundle.decision.blocking_checks expected {failed_checks} entries, got {len(blocking_checks)}."
            )
        for index, check in enumerate(blocking_checks):
            label = f"evidence_bundle.decision.blocking_checks[{index}]"
            if not isinstance(check, dict):
                target.errors.append(f"{label} must be an object.")
                continue
            if not isinstance(check.get("id"), str) or not check.get("id"):
                target.errors.append(f"{label}.id must be a non-empty string.")
            if not isinstance(check.get("summary"), str):
                target.errors.append(f"{label}.summary must be a string.")
            if not isinstance(check.get("scope"), dict):
                target.errors.append(f"{label}.scope must be an object.")
    blocking_gates = decision.get("blocking_gates")
    if not isinstance(blocking_gates, list):
        target.errors.append("evidence_bundle.decision.blocking_gates must be a list.")
    else:
        for index, gate in enumerate(blocking_gates):
            label = f"evidence_bundle.decision.blocking_gates[{index}]"
            if not isinstance(gate, dict):
                target.errors.append(f"{label} must be an object.")
                continue
            for field_name in ("id", "path"):
                if not isinstance(gate.get(field_name), str) or not gate.get(field_name):
                    target.errors.append(f"{label}.{field_name} must be a non-empty string.")
    evidence_artifacts = decision.get("evidence_artifacts")
    if not isinstance(evidence_artifacts, list) or not all(isinstance(item, str) and item for item in evidence_artifacts):
        target.errors.append("evidence_bundle.decision.evidence_artifacts must be a list of non-empty strings.")
    elif sorted(evidence_artifacts) != sorted(artifacts):
        target.errors.append("evidence_bundle.decision.evidence_artifacts must match evidence_bundle.artifacts keys.")
    gates = metrics.get("gates") if isinstance(metrics.get("gates"), list) else []
    if decision.get("gate_count") != len(gates):
        target.errors.append(f"evidence_bundle.decision.gate_count expected {len(gates)}, got {decision.get('gate_count')!r}.")
    expected_passed_gates = sum(1 for gate in gates if isinstance(gate, dict) and gate.get("passed") is True)
    if decision.get("passed_gate_count") != expected_passed_gates:
        target.errors.append(
            f"evidence_bundle.decision.passed_gate_count expected {expected_passed_gates}, got {decision.get('passed_gate_count')!r}."
        )
    if not isinstance(decision.get("key_metrics"), dict):
        target.errors.append("evidence_bundle.decision.key_metrics must be an object.")


def _validate_evidence_bundle_artifact_record(name: Any, record: Any, target: ValidationTarget) -> None:
    if not isinstance(name, str) or not name:
        target.errors.append("evidence_bundle.artifacts keys must be non-empty strings.")
    label = f"evidence_bundle.artifacts.{name}"
    if not isinstance(record, dict):
        target.errors.append(f"{label} must be an object.")
        return
    if not isinstance(record.get("path"), str) or not record.get("path"):
        target.errors.append(f"{label}.path must be a non-empty string.")
    if not isinstance(record.get("exists"), bool):
        target.errors.append(f"{label}.exists must be a boolean.")
    if record.get("kind") not in {"file", "directory"}:
        target.errors.append(f"{label}.kind must be file or directory.")
    if record.get("kind") == "file" and record.get("exists") is True:
        if not _is_non_negative_int(record.get("size_bytes")):
            target.errors.append(f"{label}.size_bytes must be a non-negative integer for existing files.")
        sha = record.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64 or sha != sha.lower() or any(char not in "0123456789abcdef" for char in sha):
            target.errors.append(f"{label}.sha256 must be a lowercase 64-character hex digest for existing files.")
    if record.get("kind") == "directory" and record.get("exists") is True and not _is_non_negative_int(record.get("entry_count")):
        target.errors.append(f"{label}.entry_count must be a non-negative integer for existing directories.")
    if "schema_version" in record and record.get("schema_version") is not None and not isinstance(record.get("schema_version"), str):
        target.errors.append(f"{label}.schema_version must be a string or null.")
    if "passed" in record and record.get("passed") is not None and not isinstance(record.get("passed"), bool):
        target.errors.append(f"{label}.passed must be a boolean or null.")


def _validate_evidence_bundle_metrics(metrics: dict[str, Any], target: ValidationTarget) -> None:
    expected_sections = (
        "suite_summary",
        "scenario_quality",
        "evidence_coverage",
        "trace_observability",
        "validation",
        "training_export",
        "compare_export",
        "review_export",
        "reviewed_export",
        "review_calibration",
    )
    for section in expected_sections:
        if section in metrics and not isinstance(metrics[section], dict):
            target.errors.append(f"evidence_bundle.metrics.{section} must be an object when present.")
    gates = metrics.get("gates")
    if gates is not None:
        if not isinstance(gates, list):
            target.errors.append("evidence_bundle.metrics.gates must be a list when present.")
            return
        for index, gate in enumerate(gates):
            if not isinstance(gate, dict):
                target.errors.append(f"evidence_bundle.metrics.gates[{index}] must be an object.")
                continue
            for field_name in ("id", "path"):
                if not isinstance(gate.get(field_name), str) or not gate.get(field_name):
                    target.errors.append(f"evidence_bundle.metrics.gates[{index}].{field_name} must be a non-empty string.")
            if not isinstance(gate.get("passed"), bool):
                target.errors.append(f"evidence_bundle.metrics.gates[{index}].passed must be a boolean.")


def _validate_review_calibration(calibration: dict[str, Any], target: ValidationTarget) -> None:
    _require_equal(calibration, "schema_version", REVIEW_CALIBRATION_SCHEMA_VERSION, target)
    if not isinstance(calibration.get("reviewed_export"), str) or not calibration.get("reviewed_export"):
        target.errors.append("review_calibration.reviewed_export must be a non-empty string.")
    source = calibration.get("source")
    if not isinstance(source, dict):
        target.errors.append("review_calibration.source must be an object.")
    elif not isinstance(source.get("reviewed_labels"), str) or not source.get("reviewed_labels"):
        target.errors.append("review_calibration.source.reviewed_labels must be a non-empty string.")
    if not isinstance(calibration.get("passed"), bool):
        target.errors.append("review_calibration.passed must be a boolean.")

    checks = calibration.get("checks")
    if not isinstance(checks, list):
        target.errors.append("review_calibration.checks must be a list.")
        checks = []
    failed_checks = _validate_gate_like_checks(checks, target, "review_calibration.checks")
    if calibration.get("check_count") != len(checks):
        target.errors.append(f"review_calibration.check_count expected {len(checks)}, got {calibration.get('check_count')!r}.")
    if calibration.get("failed_check_count") != failed_checks:
        target.errors.append(
            f"review_calibration.failed_check_count expected {failed_checks}, got {calibration.get('failed_check_count')!r}."
        )
    if isinstance(calibration.get("passed"), bool) and calibration["passed"] != (failed_checks == 0):
        target.errors.append("review_calibration.passed must match failed_check_count.")

    metrics = calibration.get("metrics")
    if not isinstance(metrics, dict):
        target.errors.append("review_calibration.metrics must be an object.")
        metrics = {}
    disagreements = calibration.get("disagreements")
    if not isinstance(disagreements, list):
        target.errors.append("review_calibration.disagreements must be a list.")
        disagreements = []
    disagreement_counts = _validate_review_calibration_disagreements(disagreements, target)
    _validate_review_calibration_metrics(metrics, disagreement_counts, target)
    notes = calibration.get("notes")
    if not isinstance(notes, list) or not all(isinstance(item, str) for item in notes):
        target.errors.append("review_calibration.notes must be a list of strings.")
    target.details.update(
        {
            "reviewed_label_count": metrics.get("reviewed_label_count"),
            "agreement_rate": metrics.get("agreement_rate"),
            "disagreement_count": metrics.get("disagreement_count"),
        }
    )


def _validate_gate_like_checks(checks: list[Any], target: ValidationTarget, label: str) -> int:
    failed_checks = 0
    for index, check in enumerate(checks):
        item_label = f"{label}[{index}]"
        if not isinstance(check, dict):
            target.errors.append(f"{item_label} must be an object.")
            continue
        if not isinstance(check.get("id"), str) or not check.get("id"):
            target.errors.append(f"{item_label}.id must be a non-empty string.")
        if not isinstance(check.get("passed"), bool):
            target.errors.append(f"{item_label}.passed must be a boolean.")
        elif not check["passed"]:
            failed_checks += 1
        if "actual" not in check:
            target.errors.append(f"{item_label}.actual is missing.")
        if not isinstance(check.get("expected"), dict):
            target.errors.append(f"{item_label}.expected must be an object.")
        if not isinstance(check.get("summary"), str) or not check.get("summary"):
            target.errors.append(f"{item_label}.summary must be a non-empty string.")
    return failed_checks


def _validate_review_calibration_disagreements(disagreements: list[Any], target: ValidationTarget) -> dict[str, int]:
    counts = {"false_positive_count": 0, "false_negative_count": 0}
    for index, row in enumerate(disagreements):
        label = f"review_calibration.disagreements[{index}]"
        if not isinstance(row, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        for field_name in ("review_item_id", "episode_id", "scenario_id", "task_family", "human_label", "disagreement_type"):
            if not isinstance(row.get(field_name), str) or not row.get(field_name):
                target.errors.append(f"{label}.{field_name} must be a non-empty string.")
        if not isinstance(row.get("scorecard_passed"), bool):
            target.errors.append(f"{label}.scorecard_passed must be a boolean.")
        if not _is_int_between(row.get("scorecard_score"), 0, 100):
            target.errors.append(f"{label}.scorecard_score must be an integer from 0 to 100.")
        if not _is_string_list(row.get("failed_rules")):
            target.errors.append(f"{label}.failed_rules must be a list of strings.")
        if not _is_string_list(row.get("critical_failures")):
            target.errors.append(f"{label}.critical_failures must be a list of strings.")
        if row.get("source_report") is not None and not isinstance(row.get("source_report"), str):
            target.errors.append(f"{label}.source_report must be a string or null.")
        if row.get("source_lineage") is not None and not isinstance(row.get("source_lineage"), str):
            target.errors.append(f"{label}.source_lineage must be a string or null.")
        if row.get("disagreement_type") == "scorecard_passed_human_rejected":
            counts["false_positive_count"] += 1
            if row.get("scorecard_passed") is not True:
                target.errors.append(f"{label}.scorecard_passed must be true for scorecard_passed_human_rejected.")
            if row.get("human_label") not in TRAINING_NEGATIVE_LABELS:
                target.errors.append(f"{label}.human_label must be a negative label for scorecard_passed_human_rejected.")
        elif row.get("disagreement_type") == "scorecard_failed_human_accepted":
            counts["false_negative_count"] += 1
            if row.get("scorecard_passed") is not False:
                target.errors.append(f"{label}.scorecard_passed must be false for scorecard_failed_human_accepted.")
            if row.get("human_label") != "accept":
                target.errors.append(f"{label}.human_label must be accept for scorecard_failed_human_accepted.")
        else:
            target.errors.append(
                f"{label}.disagreement_type must be scorecard_passed_human_rejected or scorecard_failed_human_accepted."
            )
    counts["disagreement_count"] = len(disagreements)
    return counts


def _validate_review_calibration_metrics(metrics: dict[str, Any], disagreement_counts: dict[str, int], target: ValidationTarget) -> None:
    label_counts = _label_count_rows(metrics.get("label_counts"), target)
    comparable_count = label_counts.get("accept", 0) + sum(label_counts.get(label, 0) for label in TRAINING_NEGATIVE_LABELS)
    expected = {
        "reviewed_label_count": sum(label_counts.values()),
        "comparable_label_count": comparable_count,
        "needs_review_count": label_counts.get("needs_review", 0),
        "human_positive_count": label_counts.get("accept", 0),
        "human_negative_count": sum(label_counts.get(label, 0) for label in TRAINING_NEGATIVE_LABELS),
        "disagreement_count": disagreement_counts["disagreement_count"],
        "false_positive_count": disagreement_counts["false_positive_count"],
        "false_negative_count": disagreement_counts["false_negative_count"],
    }
    expected["agreement_count"] = comparable_count - disagreement_counts["disagreement_count"]
    for field_name, expected_value in expected.items():
        if metrics.get(field_name) != expected_value:
            target.errors.append(f"review_calibration.metrics.{field_name} expected {expected_value!r}, got {metrics.get(field_name)!r}.")
    if metrics.get("agreement_rate") != _rate_value(expected["agreement_count"], comparable_count):
        target.errors.append("review_calibration.metrics.agreement_rate does not match agreement/comparable counts.")
    for field_name in ("scorecard_positive_count", "scorecard_negative_count"):
        if not _is_non_negative_int(metrics.get(field_name)):
            target.errors.append(f"review_calibration.metrics.{field_name} must be a non-negative integer.")
    if _is_non_negative_int(metrics.get("scorecard_positive_count")) and _is_non_negative_int(metrics.get("scorecard_negative_count")):
        actual_comparable = metrics["scorecard_positive_count"] + metrics["scorecard_negative_count"]
        if actual_comparable != comparable_count:
            target.errors.append("review_calibration.metrics scorecard positive/negative counts must sum to comparable_label_count.")
    if not _is_string_list(metrics.get("task_families")):
        target.errors.append("review_calibration.metrics.task_families must be a list of strings.")
    _validate_mean_score_by_human_label(metrics.get("mean_score_by_human_label"), label_counts, target)


def _label_count_rows(value: Any, target: ValidationTarget) -> dict[str, int]:
    labels = set(REVIEW_LABELS)
    counts = {label: 0 for label in REVIEW_LABELS}
    if not isinstance(value, list):
        target.errors.append("review_calibration.metrics.label_counts must be a list.")
        return counts
    seen: set[str] = set()
    for index, row in enumerate(value):
        label = f"review_calibration.metrics.label_counts[{index}]"
        if not isinstance(row, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        row_label = row.get("label")
        count = row.get("count")
        if not isinstance(row_label, str) or row_label not in labels:
            target.errors.append(f"{label}.label must be one of {list(REVIEW_LABELS)!r}.")
            continue
        if row_label in seen:
            target.errors.append(f"{label}.label duplicates {row_label!r}.")
        seen.add(row_label)
        if not _is_non_negative_int(count):
            target.errors.append(f"{label}.count must be a non-negative integer.")
            continue
        counts[row_label] = count
    return counts


def _validate_mean_score_by_human_label(value: Any, label_counts: dict[str, int], target: ValidationTarget) -> None:
    if not isinstance(value, list):
        target.errors.append("review_calibration.metrics.mean_score_by_human_label must be a list.")
        return
    seen: set[str] = set()
    for index, row in enumerate(value):
        label = f"review_calibration.metrics.mean_score_by_human_label[{index}]"
        if not isinstance(row, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        row_label = row.get("label")
        if not isinstance(row_label, str) or not row_label:
            target.errors.append(f"{label}.label must be a non-empty string.")
            continue
        if row_label in seen:
            target.errors.append(f"{label}.label duplicates {row_label!r}.")
        seen.add(row_label)
        if row_label in label_counts and row.get("count") != label_counts[row_label]:
            target.errors.append(f"{label}.count must match label_counts for {row_label!r}.")
        elif not _is_non_negative_int(row.get("count")):
            target.errors.append(f"{label}.count must be a non-negative integer.")
        if not _is_number_between(row.get("average_score"), 0.0, 100.0):
            target.errors.append(f"{label}.average_score must be a number from 0 to 100.")


def _merge_count_rows(target_counts: dict[str, int], value: Any, target: ValidationTarget, label: str) -> None:
    counts = _count_rows(value)
    if counts is None:
        target.errors.append(f"{label} must be count rows.")
        return
    for key, count in counts.items():
        target_counts[key] = target_counts.get(key, 0) + count


def _rate_value(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 1.0
    return round(numerator / denominator, 4)


def _validate_scenario_quality(quality: dict[str, Any], target: ValidationTarget) -> None:
    _require_equal(quality, "schema_version", SCENARIO_QUALITY_SCHEMA_VERSION, target)
    scenarios = quality.get("scenarios")
    if not isinstance(scenarios, list):
        target.errors.append("scenario_quality.scenarios must be a list.")
        scenarios = []
    metrics = quality.get("metrics")
    if not isinstance(metrics, dict):
        target.errors.append("scenario_quality.metrics must be an object.")
        metrics = {}
    checks = quality.get("checks")
    if not isinstance(checks, list):
        target.errors.append("scenario_quality.checks must be a list.")
        checks = []
    if not isinstance(quality.get("passed"), bool):
        target.errors.append("scenario_quality.passed must be a boolean.")

    failed_checks = 0
    for index, check in enumerate(checks):
        if not isinstance(check, dict):
            target.errors.append(f"scenario_quality.checks[{index}] must be an object.")
            continue
        if not isinstance(check.get("id"), str) or not check.get("id"):
            target.errors.append(f"scenario_quality.checks[{index}].id must be a non-empty string.")
        if not isinstance(check.get("passed"), bool):
            target.errors.append(f"scenario_quality.checks[{index}].passed must be a boolean.")
        elif not check["passed"]:
            failed_checks += 1
    if quality.get("check_count") != len(checks):
        target.errors.append(f"scenario_quality.check_count expected {len(checks)}, got {quality.get('check_count')!r}.")
    if quality.get("failed_check_count") != failed_checks:
        target.errors.append(
            f"scenario_quality.failed_check_count expected {failed_checks}, got {quality.get('failed_check_count')!r}."
        )
    if isinstance(quality.get("passed"), bool) and quality["passed"] != (failed_checks == 0):
        target.errors.append("scenario_quality.passed must match failed_check_count.")

    totals = _validate_scenario_quality_rows(scenarios, target)
    _validate_scenario_quality_metrics(metrics, totals, target)
    target.details.update(
        {
            "scenario_count": totals["scenario_count"],
            "average_contract_score": metrics.get("average_contract_score"),
            "observable_scenario_rate": metrics.get("observable_scenario_rate"),
        }
    )


def _validate_scenario_quality_rows(scenarios: list[Any], target: ValidationTarget) -> dict[str, Any]:
    totals: dict[str, Any] = {
        "scenario_count": len(scenarios),
        "valid_scenario_count": 0,
        "invalid_scenario_count": 0,
        "scores": [],
        "task_families": set(),
        "observable_scenario_count": 0,
        "weak_scenario_count": 0,
        "final_only_scenario_count": 0,
        "missing_trace_count": 0,
        "missing_state_count": 0,
        "risk_counts": {},
    }
    for index, row in enumerate(scenarios):
        label = f"scenario_quality.scenarios[{index}]"
        if not isinstance(row, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        for field_name in ("path", "id", "title", "task_family", "quality"):
            if not isinstance(row.get(field_name), str) or not row.get(field_name):
                target.errors.append(f"{label}.{field_name} must be a non-empty string.")
        if row.get("quality") not in {"strong", "moderate", "weak", "invalid"}:
            target.errors.append(f"{label}.quality must be strong, moderate, weak, or invalid.")
        if not _is_int_between(row.get("contract_score"), 0, 100):
            target.errors.append(f"{label}.contract_score must be an integer from 0 to 100.")
        errors = row.get("errors")
        if not isinstance(errors, list) or not all(isinstance(item, str) for item in errors):
            target.errors.append(f"{label}.errors must be a list of strings.")
            errors = []
        risks = row.get("risks")
        if not isinstance(risks, list) or not all(isinstance(item, str) for item in risks):
            target.errors.append(f"{label}.risks must be a list of strings.")
            risks = []
        if errors:
            totals["invalid_scenario_count"] += 1
            continue
        totals["valid_scenario_count"] += 1
        totals["scores"].append(row["contract_score"])
        totals["task_families"].add(str(row.get("task_family") or "unknown"))
        signals = row.get("signals")
        if not isinstance(signals, dict):
            target.errors.append(f"{label}.signals must be an object for valid scenarios.")
            signals = {}
        if _non_negative_int_value(signals.get("observable_assertion_count")) > 0:
            totals["observable_scenario_count"] += 1
        if row.get("quality") == "weak":
            totals["weak_scenario_count"] += 1
        if "final_only_contract" in risks:
            totals["final_only_scenario_count"] += 1
        trace = row.get("trace")
        if isinstance(trace, dict) and trace.get("trace_exists") is not True:
            totals["missing_trace_count"] += 1
        if "missing_state_file" in risks or "required_state_without_snapshot_path" in risks:
            totals["missing_state_count"] += 1
        for risk in risks:
            totals["risk_counts"][risk] = totals["risk_counts"].get(risk, 0) + 1
    return totals


def _validate_scenario_quality_metrics(metrics: dict[str, Any], totals: dict[str, Any], target: ValidationTarget) -> None:
    scores = totals["scores"]
    expected = {
        "scenario_count": totals["scenario_count"],
        "valid_scenario_count": totals["valid_scenario_count"],
        "invalid_scenario_count": totals["invalid_scenario_count"],
        "task_family_count": len(totals["task_families"]),
        "average_contract_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
        "min_contract_score": min(scores) if scores else 0,
        "max_contract_score": max(scores) if scores else 0,
        "observable_scenario_count": totals["observable_scenario_count"],
        "observable_scenario_rate": _rate_zero(totals["observable_scenario_count"], totals["valid_scenario_count"]),
        "weak_scenario_count": totals["weak_scenario_count"],
        "final_only_scenario_count": totals["final_only_scenario_count"],
        "missing_trace_count": totals["missing_trace_count"],
        "missing_state_count": totals["missing_state_count"],
    }
    for field_name, expected_value in expected.items():
        if metrics.get(field_name) != expected_value:
            target.errors.append(f"scenario_quality.metrics.{field_name} expected {expected_value!r}, got {metrics.get(field_name)!r}.")
    task_families = metrics.get("task_families")
    if task_families != sorted(totals["task_families"]):
        target.errors.append("scenario_quality.metrics.task_families does not match scenarios.")
    if _count_rows(metrics.get("risk_counts")) != totals["risk_counts"]:
        target.errors.append("scenario_quality.metrics.risk_counts does not match scenarios.")


def _rate_zero(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _validate_suite_trend(trend: dict[str, Any], target: ValidationTarget) -> None:
    _require_equal(trend, "schema_version", SUITE_TREND_SCHEMA_VERSION, target)
    points = trend.get("points")
    if not isinstance(points, list):
        target.errors.append("suite_trend.points must be a list.")
        points = []
    if trend.get("point_count") != len(points):
        target.errors.append(f"suite_trend.point_count expected {len(points)}, got {trend.get('point_count')!r}.")

    point_labels: list[str] = []
    failed_counts_by_point: list[dict[str, int]] = []
    critical_counts_by_point: list[dict[str, int]] = []
    previous_point: dict[str, Any] | None = None
    for index, point in enumerate(points):
        if not isinstance(point, dict):
            target.errors.append(f"suite_trend.points[{index}] must be an object.")
            point_labels.append("")
            failed_counts_by_point.append({})
            critical_counts_by_point.append({})
            continue
        failed_counts, critical_counts = _validate_suite_trend_point(point, target, index, previous_point)
        point_labels.append(str(point.get("label") or ""))
        failed_counts_by_point.append(failed_counts)
        critical_counts_by_point.append(critical_counts)
        previous_point = point

    _validate_suite_trend_count_rows(
        trend.get("failed_rule_trends"),
        target,
        "suite_trend.failed_rule_trends",
        points,
        point_labels,
        failed_counts_by_point,
    )
    _validate_suite_trend_count_rows(
        trend.get("critical_failure_trends"),
        target,
        "suite_trend.critical_failure_trends",
        points,
        point_labels,
        critical_counts_by_point,
    )

    summary = trend.get("summary")
    if not isinstance(summary, str) or not summary:
        target.errors.append("suite_trend.summary must be a non-empty string.")
    else:
        expected_summary = _expected_suite_trend_summary([point for point in points if isinstance(point, dict)])
        if summary != expected_summary:
            target.errors.append(f"suite_trend.summary expected {expected_summary!r}, got {summary!r}.")

    target.details.update(
        {
            "point_count": len(points),
            "failed_rule_trend_count": len(trend.get("failed_rule_trends", []))
            if isinstance(trend.get("failed_rule_trends"), list)
            else None,
            "critical_failure_trend_count": len(trend.get("critical_failure_trends", []))
            if isinstance(trend.get("critical_failure_trends"), list)
            else None,
        }
    )


def _validate_suite_trend_point(
    point: dict[str, Any],
    target: ValidationTarget,
    index: int,
    previous_point: dict[str, Any] | None,
) -> tuple[dict[str, int], dict[str, int]]:
    label = f"suite_trend.points[{index}]"
    if point.get("index") != index:
        target.errors.append(f"{label}.index expected {index}, got {point.get('index')!r}.")
    for field_name in ("label", "path"):
        if not isinstance(point.get(field_name), str) or not point.get(field_name):
            target.errors.append(f"{label}.{field_name} must be a non-empty string.")
    if isinstance(point.get("path"), str) and _looks_absolute(point["path"]):
        target.warnings.append(f"{label}.path is absolute; prefer redacted or relative trend artifacts for sharing.")
    if "metadata" in point:
        _validate_metadata(point.get("metadata"), target, f"{label}.metadata")

    for field_name in ("total", "passed", "failed", "error_count", "failed_rule_count", "critical_failure_count"):
        if not _is_non_negative_int(point.get(field_name)):
            target.errors.append(f"{label}.{field_name} must be a non-negative integer.")
    if _is_non_negative_int(point.get("total")) and _is_non_negative_int(point.get("passed")) and _is_non_negative_int(point.get("failed")):
        expected_total = point["passed"] + point["failed"]
        if point["total"] != expected_total:
            target.errors.append(f"{label}.total expected passed + failed ({expected_total}), got {point['total']!r}.")

    if not _is_number_between(point.get("pass_rate"), 0.0, 1.0):
        target.errors.append(f"{label}.pass_rate must be numeric from 0.0 to 1.0.")
    if not _is_number_between(point.get("average_score"), 0.0, 100.0):
        target.errors.append(f"{label}.average_score must be numeric from 0.0 to 100.0.")

    failed_counts = _validate_count_map_object(point.get("failed_rule_counts"), target, f"{label}.failed_rule_counts")
    critical_counts = _validate_count_map_object(point.get("critical_failure_counts"), target, f"{label}.critical_failure_counts")
    if _is_non_negative_int(point.get("failed_rule_count")) and point["failed_rule_count"] != sum(failed_counts.values()):
        target.errors.append(
            f"{label}.failed_rule_count expected sum of failed_rule_counts ({sum(failed_counts.values())}), got {point['failed_rule_count']!r}."
        )
    if _is_non_negative_int(point.get("critical_failure_count")) and point["critical_failure_count"] != sum(critical_counts.values()):
        target.errors.append(
            f"{label}.critical_failure_count expected sum of critical_failure_counts ({sum(critical_counts.values())}), "
            f"got {point['critical_failure_count']!r}."
        )

    delta = point.get("delta_from_previous")
    if index == 0:
        if delta is not None:
            target.errors.append(f"{label}.delta_from_previous must be null for the first point.")
    elif not isinstance(delta, dict):
        target.errors.append(f"{label}.delta_from_previous must be an object.")
    elif previous_point is not None:
        expected_delta = _expected_suite_trend_delta(previous_point, point)
        for field_name, expected in expected_delta.items():
            if delta.get(field_name) != expected:
                target.errors.append(f"{label}.delta_from_previous.{field_name} expected {expected!r}, got {delta.get(field_name)!r}.")
    return failed_counts, critical_counts


def _validate_suite_trend_count_rows(
    rows: Any,
    target: ValidationTarget,
    label: str,
    points: list[Any],
    point_labels: list[str],
    counts_by_point: list[dict[str, int]],
) -> None:
    if not isinstance(rows, list):
        target.errors.append(f"{label} must be a list.")
        return
    expected_ids = sorted({rule_id for counts in counts_by_point for rule_id in counts})
    actual_ids: set[str] = set()
    for row_index, row in enumerate(rows):
        row_label = f"{label}[{row_index}]"
        if not isinstance(row, dict):
            target.errors.append(f"{row_label} must be an object.")
            continue
        rule_id = row.get("id")
        if not isinstance(rule_id, str) or not rule_id:
            target.errors.append(f"{row_label}.id must be a non-empty string.")
            continue
        actual_ids.add(rule_id)
        counts = row.get("counts")
        if not isinstance(counts, list):
            target.errors.append(f"{row_label}.counts must be a list.")
            counts = []
        if len(counts) != len(points):
            target.errors.append(f"{row_label}.counts expected {len(points)} rows, got {len(counts)}.")

        observed_counts: list[int] = []
        for point_index, count_row in enumerate(counts):
            count_label = f"{row_label}.counts[{point_index}]"
            if not isinstance(count_row, dict):
                target.errors.append(f"{count_label} must be an object.")
                observed_counts.append(0)
                continue
            expected_count = counts_by_point[point_index].get(rule_id, 0) if point_index < len(counts_by_point) else 0
            count = count_row.get("count")
            if count_row.get("index") != point_index:
                target.errors.append(f"{count_label}.index expected {point_index}, got {count_row.get('index')!r}.")
            expected_label = point_labels[point_index] if point_index < len(point_labels) else ""
            if count_row.get("label") != expected_label:
                target.errors.append(f"{count_label}.label expected {expected_label!r}, got {count_row.get('label')!r}.")
            if not _is_non_negative_int(count):
                target.errors.append(f"{count_label}.count must be a non-negative integer.")
                observed_counts.append(0)
            else:
                observed_counts.append(count)
                if count != expected_count:
                    target.errors.append(f"{count_label}.count expected {expected_count}, got {count!r}.")

        expected_first = observed_counts[0] if observed_counts else 0
        expected_last = observed_counts[-1] if observed_counts else 0
        expected_delta = expected_last - expected_first
        for field_name, expected in (
            ("first_count", expected_first),
            ("last_count", expected_last),
            ("delta", expected_delta),
        ):
            if row.get(field_name) != expected:
                target.errors.append(f"{row_label}.{field_name} expected {expected!r}, got {row.get(field_name)!r}.")

    missing = sorted(set(expected_ids) - actual_ids)
    unexpected = sorted(actual_ids - set(expected_ids))
    if missing:
        target.errors.append(f"{label} missing rule IDs: {missing!r}.")
    if unexpected:
        target.errors.append(f"{label} has unexpected rule IDs: {unexpected!r}.")


def _validate_evidence_refs(value: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(value, list):
        target.errors.append(f"{label} must be a list when present.")
        return
    for index, ref in enumerate(value):
        if not isinstance(ref, dict):
            target.errors.append(f"{label}[{index}] must be an object.")
            continue
        ref_target = ref.get("target")
        if ref_target not in {"event", "final_answer", "episode", "state_snapshot"}:
            target.errors.append(f"{label}[{index}].target must be one of event, final_answer, episode, or state_snapshot.")
        if ref_target == "event":
            event_index = ref.get("event_index")
            if not isinstance(event_index, int) or isinstance(event_index, bool) or event_index < 0:
                target.errors.append(f"{label}[{index}].event_index must be a non-negative integer for event refs.")
        if "reason" in ref and not isinstance(ref.get("reason"), str):
            target.errors.append(f"{label}[{index}].reason must be a string when present.")
        if "passed" in ref and not isinstance(ref.get("passed"), bool):
            target.errors.append(f"{label}[{index}].passed must be a boolean when present.")


def _lineage_records(value: Any, target: ValidationTarget, label: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not isinstance(value, list):
        target.errors.append(f"{label} must be a list.")
        return records
    for index, record in enumerate(value):
        if not isinstance(record, dict):
            target.errors.append(f"{label}[{index}] must be an object.")
            continue
        name = record.get("name")
        if not isinstance(name, str) or not name:
            target.errors.append(f"{label}[{index}].name must be a non-empty string.")
            continue
        records[name] = record
    return records


def _validate_lineage_file_record(record: dict[str, Any], run_dir: Path, target: ValidationTarget, label: str) -> None:
    path_label = record.get("path")
    if not isinstance(path_label, str) or not path_label:
        target.errors.append(f"{label}.path must be a non-empty string.")
        return
    if record.get("exists") is not True:
        target.errors.append(f"{label}.exists must be true for required run outputs.")
        return
    basename = _lineage_basename(path_label)
    file_path = run_dir / basename
    if not file_path.exists():
        target.errors.append(f"{label}.path does not resolve inside the run directory.")
        return
    expected_size = record.get("size_bytes")
    if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size < 0:
        target.errors.append(f"{label}.size_bytes must be a non-negative integer.")
    elif file_path.stat().st_size != expected_size:
        target.errors.append(f"{label}.size_bytes does not match the current file.")
    expected_hash = record.get("sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        target.errors.append(f"{label}.sha256 must be a SHA-256 hex string.")
    elif _sha256(file_path) != expected_hash:
        target.errors.append(f"{label}.sha256 does not match the current file.")


def _validate_lineage_input_record(name: str, record: dict[str, Any], target: ValidationTarget, label: str) -> None:
    if record.get("role") != "input":
        target.errors.append(f"{label}.role must be input.")
    path_label = record.get("path")
    if path_label is not None and not isinstance(path_label, str):
        target.errors.append(f"{label}.path must be a string or null.")
    if not isinstance(record.get("exists"), bool):
        target.errors.append(f"{label}.exists must be a boolean.")
    if record.get("exists") is True:
        if not _is_non_negative_int(record.get("size_bytes")):
            target.errors.append(f"{label}.size_bytes must be a non-negative integer for existing inputs.")
        if not _is_sha256(record.get("sha256")):
            target.errors.append(f"{label}.sha256 must be a SHA-256 hex string for existing inputs.")
    if "sensitive" in record and not isinstance(record.get("sensitive"), bool):
        target.errors.append(f"{label}.sensitive must be a boolean when present.")
    if name in {"scenario", "source_trace", "source_state_snapshot"} and record.get("exists") is not True:
        target.warnings.append(f"{label}.exists is not true; replay may require restoring this input.")


def _validate_lineage_replay(replay: dict[str, Any], inputs: dict[str, dict[str, Any]], target: ValidationTarget) -> None:
    if replay.get("tool") != "flightrecorder":
        target.errors.append("artifact_lineage.replay.tool must be flightrecorder.")
    argv = replay.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) and item for item in argv):
        target.errors.append("artifact_lineage.replay.argv must be a list of non-empty strings.")
        argv = []
    else:
        expected_prefix = ["python", "-m", "flightrecorder", "run"]
        if argv[:4] != expected_prefix:
            target.errors.append("artifact_lineage.replay.argv must start with python -m flightrecorder run.")
        for required_flag in ("--scenario", "--trace", "--out"):
            if required_flag not in argv:
                target.errors.append(f"artifact_lineage.replay.argv missing {required_flag}.")
    if not isinstance(replay.get("command"), str) or not replay.get("command"):
        target.errors.append("artifact_lineage.replay.command must be a non-empty string.")
    if not isinstance(replay.get("self_contained"), bool):
        target.errors.append("artifact_lineage.replay.self_contained must be a boolean.")
    notes = replay.get("notes")
    if not isinstance(notes, list) or not all(isinstance(item, str) for item in notes):
        target.errors.append("artifact_lineage.replay.notes must be a list of strings.")
    fingerprints = replay.get("input_fingerprints")
    if not isinstance(fingerprints, dict):
        target.errors.append("artifact_lineage.replay.input_fingerprints must be an object.")
        return
    for required_name in ("scenario", "source_trace"):
        if required_name not in fingerprints:
            target.errors.append(f"artifact_lineage.replay.input_fingerprints missing {required_name}.")
    for name, fingerprint in fingerprints.items():
        label = f"artifact_lineage.replay.input_fingerprints.{name}"
        if not isinstance(name, str) or not name:
            target.errors.append("artifact_lineage.replay.input_fingerprints keys must be non-empty strings.")
            continue
        if not isinstance(fingerprint, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        if "path" not in fingerprint or "sha256" not in fingerprint or "exists" not in fingerprint:
            target.errors.append(f"{label} must contain path, sha256, and exists.")
        if fingerprint.get("path") is not None and not isinstance(fingerprint.get("path"), str):
            target.errors.append(f"{label}.path must be a string or null.")
        if fingerprint.get("sha256") is not None and not _is_sha256(fingerprint.get("sha256")):
            target.errors.append(f"{label}.sha256 must be a SHA-256 hex string or null.")
        if fingerprint.get("exists") is not None and not isinstance(fingerprint.get("exists"), bool):
            target.errors.append(f"{label}.exists must be a boolean or null.")
        input_record = inputs.get(name)
        if input_record is None:
            target.errors.append(f"{label} does not match an artifact_lineage input record.")
            continue
        for field_name in ("path", "sha256", "exists"):
            if fingerprint.get(field_name) != input_record.get(field_name):
                target.errors.append(f"{label}.{field_name} must match artifact_lineage.inputs.{name}.{field_name}.")


def _validate_lineage_evidence_link(link: Any, index: int, event_count: int, target: ValidationTarget) -> None:
    label = f"artifact_lineage.evidence_links[{index}]"
    if not isinstance(link, dict):
        target.errors.append(f"{label} must be an object.")
        return
    for field_name in ("rule_id", "rule_name", "scorecard_pointer", "target"):
        if not isinstance(link.get(field_name), str) or not link.get(field_name):
            target.errors.append(f"{label}.{field_name} must be a non-empty string.")
    ref_target = link.get("target")
    if ref_target not in {"event", "final_answer", "episode", "state_snapshot"}:
        target.errors.append(f"{label}.target must be one of event, final_answer, episode, or state_snapshot.")
    if ref_target == "event":
        event_index = link.get("event_index")
        if not isinstance(event_index, int) or isinstance(event_index, bool) or event_index < 0:
            target.errors.append(f"{label}.event_index must be a non-negative integer.")
        elif event_index >= event_count:
            target.errors.append(f"{label}.event_index must refer to an existing trace event.")
        if link.get("trace_pointer") != f"/events/{event_index}":
            target.errors.append(f"{label}.trace_pointer must point at the referenced trace event.")
    elif ref_target == "final_answer" and link.get("trace_pointer") != "/final_answer":
        target.errors.append(f"{label}.trace_pointer must point at /final_answer.")
    elif ref_target == "episode" and link.get("trace_pointer") != "/":
        target.errors.append(f"{label}.trace_pointer must point at the trace root.")
    elif ref_target == "state_snapshot" and link.get("state_pointer") != "/":
        target.errors.append(f"{label}.state_pointer must point at the state snapshot root.")
    if "rule_passed" in link and not isinstance(link.get("rule_passed"), bool):
        target.errors.append(f"{label}.rule_passed must be a boolean when present.")
    if "ref_passed" in link and not isinstance(link.get("ref_passed"), bool):
        target.errors.append(f"{label}.ref_passed must be a boolean when present.")


def _scorecard_evidence_ref_count(scorecard: dict[str, Any] | None) -> int:
    if not isinstance(scorecard, dict):
        return 0
    total = 0
    for rule in scorecard.get("rules", []):
        if isinstance(rule, dict) and isinstance(rule.get("evidence_refs"), list):
            total += len(rule["evidence_refs"])
    return total


def _lineage_basename(path_label: str) -> str:
    if path_label.startswith("<redacted:") and path_label.endswith(">"):
        return path_label[len("<redacted:") : -1]
    return path_label.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def _read_object(path: Path, target: ValidationTarget, label: str) -> dict[str, Any] | None:
    if not path.exists():
        target.errors.append(f"{label} is missing.")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        target.errors.append(f"{label} contains invalid JSON: {exc}")
        return None
    if not isinstance(value, dict):
        target.errors.append(f"{label} must contain a JSON object.")
        return None
    return value


def _read_object_optional(path: Path, target: ValidationTarget, label: str, refresh_message: str) -> dict[str, Any] | None:
    if not path.exists():
        target.warnings.append(f"{label} is missing; {refresh_message}.")
        return None
    return _read_object(path, target, label)


def _read_jsonl_objects(path: Path, target: ValidationTarget, label: str) -> list[dict[str, Any]]:
    if not path.exists():
        target.errors.append(f"{label} is missing.")
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            target.errors.append(f"{label}:{line_number} contains invalid JSON: {exc}")
            continue
        if not isinstance(value, dict):
            target.errors.append(f"{label}:{line_number} must contain a JSON object.")
            continue
        rows.append(value)
    return rows


def _read_jsonl_objects_optional(
    path: Path,
    target: ValidationTarget,
    label: str,
    refresh_message: str = "rerun export-rl to emit step-level reward attribution",
) -> list[dict[str, Any]]:
    if not path.exists():
        target.warnings.append(f"{label} is missing; {refresh_message}.")
        return []
    return _read_jsonl_objects(path, target, label)


def _require_equal(
    obj: dict[str, Any],
    field_name: str,
    expected: Any,
    target: ValidationTarget,
    *,
    prefix: str = "",
) -> None:
    if obj.get(field_name) != expected:
        target.errors.append(f"{prefix}{field_name} expected {expected!r}, got {obj.get(field_name)!r}.")


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _validate_metadata(value: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object when present.")
        return
    for key, raw_value in value.items():
        if not isinstance(key, str) or not key:
            target.errors.append(f"{label} keys must be non-empty strings.")
        elif any(char.isspace() for char in key):
            target.errors.append(f"{label}.{key!r} key must not contain whitespace.")
        if not isinstance(raw_value, str):
            target.errors.append(f"{label}.{key} must be a string.")


def _is_int_between(value: Any, minimum: int, maximum: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and minimum <= value <= maximum


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_number_between(value: Any, minimum: float, maximum: float) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and minimum <= float(value) <= maximum


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())


def _looks_absolute(value: str) -> bool:
    return value.startswith("/") or _is_windows_absolute(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _score_value(value: Any) -> int:
    try:
        return max(0, min(100, int(value)))
    except (TypeError, ValueError):
        return 0


def _number_value(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _number_delta(before: Any, after: Any) -> float:
    return round(_number_value(after) - _number_value(before), 4)


def _average_number(values: list[int] | list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _count_family(rows: list[dict[str, Any]], family: str) -> int:
    return sum(1 for row in rows if isinstance(row, dict) and str(row.get("task_family") or "unknown") == family)


def _outcome_strings(episode: dict[str, Any], field_name: str) -> list[str]:
    outcome = episode.get("outcome") if isinstance(episode.get("outcome"), dict) else {}
    values = outcome.get(field_name)
    return values if _is_string_list(values) else []


def _count_strings(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if not isinstance(value, str) or not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return counts


def _validate_count_map_object(value: Any, target: ValidationTarget, label: str) -> dict[str, int]:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return {}
    counts: dict[str, int] = {}
    for key, count in value.items():
        if not isinstance(key, str) or not key:
            target.errors.append(f"{label} keys must be non-empty strings.")
            continue
        if not _is_non_negative_int(count):
            target.errors.append(f"{label}.{key} must be a non-negative integer.")
            continue
        counts[key] = count
    return counts


def _expected_suite_trend_delta(previous_point: dict[str, Any], point: dict[str, Any]) -> dict[str, Any]:
    return {
        "pass_rate_delta": _number_delta(previous_point.get("pass_rate"), point.get("pass_rate")),
        "average_score_delta": _number_delta(previous_point.get("average_score"), point.get("average_score")),
        "failed_rule_count_delta": _non_negative_int_value(point.get("failed_rule_count"))
        - _non_negative_int_value(previous_point.get("failed_rule_count")),
        "critical_failure_count_delta": _non_negative_int_value(point.get("critical_failure_count"))
        - _non_negative_int_value(previous_point.get("critical_failure_count")),
    }


def _expected_suite_trend_summary(points: list[dict[str, Any]]) -> str:
    if not points:
        return "TREND: no suite summaries."
    if len(points) == 1:
        point = points[0]
        return (
            f"TREND: one point; pass rate {point.get('pass_rate')}; "
            f"average score {point.get('average_score')}."
        )
    first = points[0]
    last = points[-1]
    pass_delta = _number_delta(first.get("pass_rate"), last.get("pass_rate"))
    score_delta = _number_delta(first.get("average_score"), last.get("average_score"))
    failed_delta = _non_negative_int_value(last.get("failed_rule_count")) - _non_negative_int_value(first.get("failed_rule_count"))
    critical_delta = _non_negative_int_value(last.get("critical_failure_count")) - _non_negative_int_value(
        first.get("critical_failure_count")
    )
    return (
        f"TREND: {len(points)} points; pass_rate_delta={pass_delta}; "
        f"average_score_delta={score_delta}; failed_rule_delta={failed_delta}; "
        f"critical_failure_delta={critical_delta}."
    )


def _non_negative_int_value(value: Any) -> int:
    return value if _is_non_negative_int(value) else 0


def _count_rows(value: Any) -> dict[str, int] | None:
    if not isinstance(value, list):
        return None
    counts: dict[str, int] = {}
    for row in value:
        if not isinstance(row, dict):
            return None
        row_id = row.get("id")
        count = row.get("count")
        if not isinstance(row_id, str) or not isinstance(count, int) or isinstance(count, bool):
            return None
        counts[row_id] = count
    return counts


def _is_windows_absolute(value: str) -> bool:
    normalized = value.replace("/", "\\")
    return (len(normalized) >= 3 and normalized[1:3] == ":\\" and normalized[0].isalpha()) or normalized.startswith("\\\\")
