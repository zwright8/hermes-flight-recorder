"""Artifact validation for Flight Recorder evidence outputs."""

from __future__ import annotations

import hashlib
import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .action_gate import ACTION_LEDGER_GATE_POLICY_SCHEMA_VERSION, ACTION_LEDGER_GATE_SCHEMA_VERSION
from .action_ledger import ACTION_LEDGER_SCHEMA_VERSION
from .adapters import TRACE_SCHEMA_VERSION
from .artifacts import CONTRACT_SCOPES, SUITE_TREND_SCHEMA_VERSION
from .bundle import EVIDENCE_BUNDLE_SCHEMA_VERSION
from .calibration import REVIEW_CALIBRATION_SCHEMA_VERSION
from .compare_gate import compare_movement_summary
from .decision_gate import DECISION_GATE_SCHEMA_VERSION
from .digest import RUN_DIGEST_SCHEMA_VERSION
from .evidence import EVIDENCE_COVERAGE_SCHEMA_VERSION
from .hermes_plugin import LIVE_SMOKE_SUMMARY_SCHEMA_VERSION
from .improvement_gate import IMPROVEMENT_LEDGER_GATE_POLICY_SCHEMA_VERSION, IMPROVEMENT_LEDGER_GATE_SCHEMA_VERSION
from .improvement_ledger import IMPROVEMENT_LEDGER_SCHEMA_VERSION, stable_work_key
from .improvement_plan import IMPROVEMENT_PLAN_SCHEMA_VERSION, PRIORITIES, work_item_fingerprint
from .lineage import LINEAGE_SCHEMA_VERSION, REPLAY_BUNDLE_SCHEMA_VERSION
from .preflight import TRAINER_LAUNCH_CHECK_SCHEMA_VERSION, TRAINER_PREFLIGHT_SCHEMA_VERSION
from .promotion_archive import PROMOTION_ARCHIVE_SCHEMA_VERSION
from .promotion_gate import PROMOTION_LEDGER_GATE_POLICY_SCHEMA_VERSION, PROMOTION_LEDGER_GATE_SCHEMA_VERSION
from .promotion_ledger import PROMOTION_LEDGER_SCHEMA_VERSION
from .repair import REPAIR_ITEM_SCHEMA_VERSION, REPAIR_QUEUE_SCHEMA_VERSION
from .review import (
    REVIEW_CONFIDENCE_LEVELS,
    REVIEW_ITEM_SCHEMA_VERSION,
    REVIEW_LABEL_SCHEMA_VERSION,
    REVIEW_LABELS,
    REVIEW_MANIFEST_SCHEMA_VERSION,
    TRAINING_NEGATIVE_LABELS,
    review_item_sha256,
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
from .state_diff import STATE_DIFF_SCHEMA_VERSION
from .trace_observability import TRACE_OBSERVABILITY_SCHEMA_VERSION, build_trace_signal
from .trainer_archive import TRAINER_ARCHIVE_SCHEMA_VERSION
from .trainer_archive_check import TRAINER_ARCHIVE_CHECK_SCHEMA_VERSION
from .trainer_consumer_plan import TRAINER_CONSUMER_PLAN_SCHEMA_VERSION
from .training import (
    DATASET_SPLIT_ARTIFACTS,
    DATASET_SPLIT_NAMES,
    RL_CURRICULUM_SCHEMA_VERSION,
    RL_DATASET_REGISTRY_SCHEMA_VERSION,
    RL_DATASET_METRICS_SCHEMA_VERSION,
    RL_DATASET_SPLITS_SCHEMA_VERSION,
    RL_DPO_SCHEMA_VERSION,
    RL_EPISODE_SCHEMA_VERSION,
    RL_FAILURE_MODE_SCHEMA_VERSION,
    RL_LABEL_PROVENANCE_SCHEMA_VERSION,
    COMPARE_RL_DPO_SCHEMA_VERSION,
    COMPARE_RL_MANIFEST_SCHEMA_VERSION,
    COMPARE_RL_PAIR_SCHEMA_VERSION,
    RL_MANIFEST_SCHEMA_VERSION,
    RL_PREFERENCE_SCHEMA_VERSION,
    RL_REWARD_SCHEMA_VERSION,
    RL_REWARD_MODEL_SCHEMA_VERSION,
    RL_SFT_SCHEMA_VERSION,
    RL_STEP_REWARD_SCHEMA_VERSION,
    RL_REDACTION_STATUS_SCHEMA_VERSION,
    REWARD_SCALES,
    build_label_provenance_summary,
    build_redaction_status,
    positive_label_eligible,
    redaction_scan_artifacts,
)
from .verifiers import VERIFIER_SOURCES_SCHEMA_VERSION

VALIDATION_SCHEMA_VERSION = "hfr.validation.v1"
RUN_SUITE_SCHEMA_VERSION = "hfr.run_suite.v1"
LEGACY_LIVE_SMOKE_SUMMARY_SCHEMA_VERSIONS = {"hfr.live_smoke.summary.v1"}
TRAINER_WRAPPER_DRY_RUN_SCHEMA_VERSION = "hfr.example_trainer_wrapper_dry_run.v1"
HARNESS_RUN_MANIFEST_SCHEMA_VERSION = "hfr.harness_run_manifest.v1"
HARNESS_RUN_RESULT_SCHEMA_VERSION = "hfr.harness_run_result.v1"
HARNESS_REPLAY_RESULT_SCHEMA_VERSION = "hfr.harness_replay_result.v1"
_COMPANION_NOT_PROVIDED = object()


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
    improvement_plan_paths: list[str | Path] | None = None,
    improvement_ledger_paths: list[str | Path] | None = None,
    improvement_ledger_gate_paths: list[str | Path] | None = None,
    action_ledger_paths: list[str | Path] | None = None,
    action_ledger_gate_paths: list[str | Path] | None = None,
    decision_gate_paths: list[str | Path] | None = None,
    promotion_ledger_paths: list[str | Path] | None = None,
    promotion_ledger_gate_paths: list[str | Path] | None = None,
    promotion_archive_paths: list[str | Path] | None = None,
    trainer_preflight_paths: list[str | Path] | None = None,
    trainer_launch_check_paths: list[str | Path] | None = None,
    trainer_archive_paths: list[str | Path] | None = None,
    trainer_archive_check_paths: list[str | Path] | None = None,
    trainer_consumer_plan_paths: list[str | Path] | None = None,
    trainer_wrapper_dry_run_paths: list[str | Path] | None = None,
    repair_queue_paths: list[str | Path] | None = None,
    replay_bundle_paths: list[str | Path] | None = None,
    trace_observability_paths: list[str | Path] | None = None,
    review_calibration_paths: list[str | Path] | None = None,
    scenario_quality_paths: list[str | Path] | None = None,
    suite_summary_paths: list[str | Path] | None = None,
    suite_trend_paths: list[str | Path] | None = None,
    state_snapshot_paths: list[str | Path] | None = None,
    state_diff_paths: list[str | Path] | None = None,
    run_digest_paths: list[str | Path] | None = None,
    harness_manifest_paths: list[str | Path] | None = None,
    harness_result_paths: list[str | Path] | None = None,
    harness_replay_result_paths: list[str | Path] | None = None,
    live_smoke_summary_paths: list[str | Path] | None = None,
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
    for improvement_plan_path in improvement_plan_paths or []:
        targets.append(validate_improvement_plan(improvement_plan_path))
    for improvement_ledger_path in improvement_ledger_paths or []:
        targets.append(validate_improvement_ledger(improvement_ledger_path))
    for improvement_ledger_gate_path in improvement_ledger_gate_paths or []:
        targets.append(validate_improvement_ledger_gate(improvement_ledger_gate_path))
    for action_ledger_path in action_ledger_paths or []:
        targets.append(validate_action_ledger(action_ledger_path))
    for action_ledger_gate_path in action_ledger_gate_paths or []:
        targets.append(validate_action_ledger_gate(action_ledger_gate_path))
    for decision_gate_path in decision_gate_paths or []:
        targets.append(validate_decision_gate(decision_gate_path))
    for promotion_ledger_path in promotion_ledger_paths or []:
        targets.append(validate_promotion_ledger(promotion_ledger_path))
    for promotion_ledger_gate_path in promotion_ledger_gate_paths or []:
        targets.append(validate_promotion_ledger_gate(promotion_ledger_gate_path))
    for promotion_archive_path in promotion_archive_paths or []:
        targets.append(validate_promotion_archive(promotion_archive_path))
    for trainer_preflight_path in trainer_preflight_paths or []:
        targets.append(validate_trainer_preflight(trainer_preflight_path))
    for trainer_launch_check_path in trainer_launch_check_paths or []:
        targets.append(validate_trainer_launch_check(trainer_launch_check_path))
    for trainer_archive_path in trainer_archive_paths or []:
        targets.append(validate_trainer_archive(trainer_archive_path))
    for trainer_archive_check_path in trainer_archive_check_paths or []:
        targets.append(validate_trainer_archive_check(trainer_archive_check_path))
    for trainer_consumer_plan_path in trainer_consumer_plan_paths or []:
        targets.append(validate_trainer_consumer_plan(trainer_consumer_plan_path))
    for trainer_wrapper_dry_run_path in trainer_wrapper_dry_run_paths or []:
        targets.append(validate_trainer_wrapper_dry_run(trainer_wrapper_dry_run_path))
    for repair_queue_path in repair_queue_paths or []:
        targets.append(validate_repair_queue(repair_queue_path))
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
    for state_diff_path in state_diff_paths or []:
        targets.append(validate_state_diff(state_diff_path))
    for run_digest_path in run_digest_paths or []:
        targets.append(validate_run_digest(run_digest_path))
    for harness_manifest_path in harness_manifest_paths or []:
        targets.append(validate_harness_run_manifest(harness_manifest_path))
    for harness_result_path in harness_result_paths or []:
        targets.append(validate_harness_run_result(harness_result_path))
    for harness_replay_result_path in harness_replay_result_paths or []:
        targets.append(validate_harness_replay_result(harness_replay_result_path))
    for live_smoke_summary_path in live_smoke_summary_paths or []:
        targets.append(validate_live_smoke_summary(live_smoke_summary_path))
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
    run_digest_path = run_dir / "run_digest.json"
    state_snapshot_path = run_dir / "state_snapshot.json"
    state_diff_path = run_dir / "state_diff.json"
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
    run_digest = _read_object_optional(
        run_digest_path,
        target,
        "run_digest.json",
        "rerun the run to emit the compact per-run evidence digest",
    )
    lineage = _read_object_optional(lineage_path, target, "artifact_lineage.json", "rerun the run to emit provenance metadata")
    state_snapshot = _read_object(state_snapshot_path, target, "state_snapshot.json") if state_snapshot_path.exists() else None
    state_diff = _read_object(state_diff_path, target, "state_diff.json") if state_diff_path.exists() else None
    if trace is not None:
        _validate_trace(trace, target)
    if scorecard is not None:
        _validate_scorecard(scorecard, target)
    if task_completion is not None:
        _validate_task_completion(task_completion, target, "task_completion")
        if isinstance(scorecard, dict) and scorecard.get("task_completion") != task_completion:
            target.errors.append("task_completion.json must match scorecard.task_completion.")
    if run_digest is not None:
        _validate_run_digest(
            run_digest,
            target,
            "run_digest",
            trace=trace,
            scorecard=scorecard,
            task_completion=task_completion,
            state_diff=state_diff,
        )
    if lineage is not None:
        _validate_lineage(lineage, target, run_dir, trace, scorecard)
    if isinstance(state_snapshot, dict) and state_snapshot.get("schema_version") == STATE_SNAPSHOT_SCHEMA_VERSION:
        _validate_state_snapshot(state_snapshot, target, "state_snapshot")
    if state_diff is not None:
        _validate_state_diff(state_diff, target, "state_diff")
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
    dataset_splits_path = export_dir / "dataset_splits.json"
    dataset_registry_path = export_dir / "dataset_registry.json"
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
    dataset_splits = _read_object_optional(
        dataset_splits_path,
        target,
        "dataset_splits.json",
        "rerun export-rl to emit deterministic train/validation/test split metadata",
    )
    dataset_registry = _read_object_optional(
        dataset_registry_path,
        target,
        "dataset_registry.json",
        "rerun export-rl to emit a selectable dataset registry",
    )
    split_rows = _read_training_split_rows(export_dir, target)
    rows_by_artifact = {
        "episodes": episodes,
        "rewards": rewards,
        "step_rewards": step_rewards,
        "preferences": preferences,
        "failure_modes": failure_modes,
        "sft": sft,
        "dpo": dpo,
        "reward_model": reward_model,
    }
    metadata = manifest.get("metadata") if isinstance(manifest, dict) and isinstance(manifest.get("metadata"), dict) else None
    expected_redaction_status = build_redaction_status(
        redaction_scan_artifacts(
            rows_by_artifact,
            curriculum,
            metadata=metadata,
            extra_artifacts={
                "manifest": manifest or {},
                "dataset_metrics": dataset_metrics or {},
                "dataset_registry": dataset_registry or {},
            },
        )
    )
    expected_label_provenance = build_label_provenance_summary(episodes, sft, dpo, reward_model)
    if expected_redaction_status.get("passed") is not True:
        target.errors.append("training export contains unredacted secret-like values.")
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
            dataset_splits,
            dataset_registry,
            expected_redaction_status,
            expected_label_provenance,
            dataset_card_path.exists(),
            export_dir,
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
            dataset_splits,
            expected_redaction_status,
            expected_label_provenance,
        )
    if dataset_splits is not None:
        _validate_dataset_splits(dataset_splits, target, rows_by_artifact, split_rows)
    if dataset_registry is not None and manifest is not None:
        _validate_dataset_registry(
            dataset_registry,
            target,
            manifest,
            export_dir,
            expected_redaction_status,
            expected_label_provenance,
            dataset_splits,
            dataset_metrics,
            episodes,
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
            "split_episode_counts": dataset_splits.get("summary") if isinstance(dataset_splits, dict) else None,
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
        _validate_compare_manifest(manifest, target, pairs, dpo, card_path.exists(), export_dir)
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
        _validate_manifest_artifact_fingerprints(
            manifest.get("artifact_fingerprints"),
            target,
            "manifest.artifact_fingerprints",
            _review_export_artifact_paths(export_dir),
        )
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
    dataset_registry = _read_object_optional(
        export_dir / "dataset_registry.json",
        target,
        "dataset_registry.json",
        "rerun apply-review to emit a selectable reviewed dataset registry",
    )
    rows_by_artifact = {
        "reviewed_labels": labels,
        "reviewed_sft": sft,
        "reviewed_reward_model": reward_model,
        "reviewed_preferences": preferences,
        "reviewed_dpo": dpo,
    }
    expected_redaction_status = build_redaction_status(
        redaction_scan_artifacts(
            rows_by_artifact,
            extra_artifacts={
                "manifest": manifest or {},
                "dataset_registry": dataset_registry or {},
            },
        )
    )
    expected_label_provenance = _expected_reviewed_label_provenance(labels, sft, reward_model, preferences, dpo)
    if expected_redaction_status.get("passed") is not True:
        target.errors.append("reviewed export contains unredacted secret-like values.")
    if manifest is not None:
        _validate_reviewed_manifest(
            manifest,
            target,
            labels,
            sft,
            reward_model,
            preferences,
            dpo,
            dataset_registry,
            expected_redaction_status,
            expected_label_provenance,
            export_dir,
        )
        _validate_manifest_artifact_fingerprints(
            manifest.get("artifact_fingerprints"),
            target,
            "manifest.artifact_fingerprints",
            _reviewed_export_artifact_paths(export_dir),
        )
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


def validate_improvement_plan(path: str | Path) -> ValidationTarget:
    """Validate an improvement-plan handoff artifact."""
    plan_path = Path(path)
    target = ValidationTarget("improvement_plan", str(plan_path))
    plan = _read_object(plan_path, target, "improvement_plan.json")
    if plan is not None:
        _validate_improvement_plan(plan, target)
    return target


def validate_improvement_ledger(path: str | Path) -> ValidationTarget:
    """Validate a longitudinal improvement-ledger artifact."""
    ledger_path = Path(path)
    target = ValidationTarget("improvement_ledger", str(ledger_path))
    ledger = _read_object(ledger_path, target, "improvement_ledger.json")
    if ledger is not None:
        _validate_improvement_ledger(ledger, target)
    return target


def validate_improvement_ledger_gate(path: str | Path) -> ValidationTarget:
    """Validate an improvement-ledger gate artifact."""
    gate_path = Path(path)
    target = ValidationTarget("improvement_ledger_gate", str(gate_path))
    gate = _read_object(gate_path, target, "improvement_ledger_gate.json")
    if gate is not None:
        _validate_improvement_ledger_gate(gate, target)
    return target


def validate_action_ledger(path: str | Path) -> ValidationTarget:
    """Validate a longitudinal action-ledger artifact."""
    ledger_path = Path(path)
    target = ValidationTarget("action_ledger", str(ledger_path))
    ledger = _read_object(ledger_path, target, "action_ledger.json")
    if ledger is not None:
        _validate_action_ledger(ledger, target)
    return target


def validate_action_ledger_gate(path: str | Path) -> ValidationTarget:
    """Validate an action-ledger gate artifact."""
    gate_path = Path(path)
    target = ValidationTarget("action_ledger_gate", str(gate_path))
    gate = _read_object(gate_path, target, "action_ledger_gate.json")
    if gate is not None:
        _validate_action_ledger_gate(gate, target)
    return target


def validate_decision_gate(path: str | Path) -> ValidationTarget:
    """Validate a decision-gate artifact."""
    gate_path = Path(path)
    target = ValidationTarget("decision_gate", str(gate_path))
    gate = _read_object(gate_path, target, "decision_gate.json")
    if gate is not None:
        _validate_decision_gate(gate, target, gate_path)
    return target


def validate_promotion_ledger(path: str | Path) -> ValidationTarget:
    """Validate a longitudinal promotion-ledger artifact."""
    ledger_path = Path(path)
    target = ValidationTarget("promotion_ledger", str(ledger_path))
    ledger = _read_object(ledger_path, target, "promotion_ledger.json")
    if ledger is not None:
        _validate_promotion_ledger(ledger, target, ledger_path)
    return target


def validate_promotion_ledger_gate(path: str | Path) -> ValidationTarget:
    """Validate a promotion-ledger gate artifact."""
    gate_path = Path(path)
    target = ValidationTarget("promotion_ledger_gate", str(gate_path))
    gate = _read_object(gate_path, target, "promotion_ledger_gate.json")
    if gate is not None:
        _validate_promotion_ledger_gate(gate, target)
    return target


def validate_promotion_archive(path: str | Path) -> ValidationTarget:
    """Validate a portable promotion archive directory or manifest."""
    archive_path = Path(path)
    manifest_path = archive_path / "promotion_archive.json" if archive_path.is_dir() else archive_path
    archive_root = manifest_path.parent
    target = ValidationTarget("promotion_archive", str(archive_path))
    archive = _read_object(manifest_path, target, "promotion_archive.json")
    if archive is not None:
        _validate_promotion_archive(archive, target, archive_root)
    return target


def validate_trainer_preflight(path: str | Path) -> ValidationTarget:
    """Validate a trainer-preflight launch guard artifact."""
    preflight_path = Path(path)
    target = ValidationTarget("trainer_preflight", str(preflight_path))
    preflight = _read_object(preflight_path, target, "trainer_preflight.json")
    if preflight is not None:
        _validate_trainer_preflight(preflight, target, preflight_path)
    return target


def validate_trainer_launch_check(path: str | Path) -> ValidationTarget:
    """Validate a trainer launch-check consumer artifact."""
    launch_check_path = Path(path)
    target = ValidationTarget("trainer_launch_check", str(launch_check_path))
    launch_check = _read_object(launch_check_path, target, "trainer_launch_check.json")
    if launch_check is not None:
        _validate_trainer_launch_check(launch_check, target)
    return target


def validate_trainer_archive(path: str | Path) -> ValidationTarget:
    """Validate a portable trainer handoff archive directory or manifest."""
    archive_path = Path(path)
    manifest_path = archive_path / "trainer_archive.json" if archive_path.is_dir() else archive_path
    archive_root = manifest_path.parent
    target = ValidationTarget("trainer_archive", str(archive_path))
    archive = _read_object(manifest_path, target, "trainer_archive.json")
    if archive is not None:
        _validate_trainer_archive(archive, target, archive_root)
    return target


def validate_trainer_archive_check(path: str | Path) -> ValidationTarget:
    """Validate a trainer archive consumer-readiness artifact."""
    check_path = Path(path)
    target = ValidationTarget("trainer_archive_check", str(check_path))
    check = _read_object(check_path, target, "trainer_archive_check.json")
    if check is not None:
        _validate_trainer_archive_check(check, target)
    return target


def validate_trainer_consumer_plan(path: str | Path) -> ValidationTarget:
    """Validate a trainer consumer plan artifact."""
    plan_path = Path(path)
    target = ValidationTarget("trainer_consumer_plan", str(plan_path))
    plan = _read_object(plan_path, target, "trainer_consumer_plan.json")
    if plan is not None:
        _validate_trainer_consumer_plan(plan, target)
    return target


def validate_trainer_wrapper_dry_run(path: str | Path) -> ValidationTarget:
    """Validate a reference trainer-wrapper dry-run receipt."""
    receipt_path = Path(path)
    target = ValidationTarget("trainer_wrapper_dry_run", str(receipt_path))
    receipt = _read_object(receipt_path, target, "trainer_wrapper_dry_run.json")
    if receipt is not None:
        _validate_trainer_wrapper_dry_run(receipt, target)
    return target


def validate_live_smoke_summary(path: str | Path) -> ValidationTarget:
    """Validate a live Hermes observer-smoke summary artifact."""
    summary_path = Path(path)
    target = ValidationTarget("live_smoke_summary", str(summary_path))
    summary = _read_object(summary_path, target, "live_smoke_summary.json")
    if summary is not None:
        _validate_live_smoke_summary(summary, target)
    return target


def validate_repair_queue(path: str | Path) -> ValidationTarget:
    """Validate a repair-queue artifact."""
    queue_path = Path(path)
    target = ValidationTarget("repair_queue", str(queue_path))
    queue = _read_object(queue_path, target, "repair_queue.json")
    if queue is not None:
        _validate_repair_queue(queue, target)
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


def validate_state_diff(path: str | Path) -> ValidationTarget:
    """Validate a before/after state diff artifact."""
    diff_path = Path(path)
    target = ValidationTarget("state_diff", str(diff_path))
    diff = _read_object(diff_path, target, "state_diff.json")
    if diff is not None:
        _validate_state_diff(diff, target, "state_diff")
    return target


def validate_run_digest(path: str | Path) -> ValidationTarget:
    """Validate a compact per-run evidence digest artifact."""
    digest_path = Path(path)
    target = ValidationTarget("run_digest", str(digest_path))
    digest = _read_object(digest_path, target, "run_digest.json")
    if digest is not None:
        _validate_run_digest(digest, target, "run_digest")
    return target


def validate_harness_run_manifest(path: str | Path) -> ValidationTarget:
    """Validate a harness_manifest.json artifact."""
    manifest_path = Path(path)
    target = ValidationTarget("harness_run_manifest", str(manifest_path))
    manifest = _read_object(manifest_path, target, "harness_manifest.json")
    if manifest is not None:
        _validate_harness_run_manifest(manifest, target)
    return target


def validate_harness_run_result(path: str | Path) -> ValidationTarget:
    """Validate a harness_result.json artifact."""
    result_path = Path(path)
    target = ValidationTarget("harness_run_result", str(result_path))
    result = _read_object(result_path, target, "harness_result.json")
    if result is not None:
        _validate_harness_run_result(result, target)
    return target


def validate_harness_replay_result(path: str | Path) -> ValidationTarget:
    """Validate a harness_replay_result.json artifact."""
    result_path = Path(path)
    target = ValidationTarget("harness_replay_result", str(result_path))
    result = _read_object(result_path, target, "harness_replay_result.json")
    if result is not None:
        _validate_harness_replay_result(result, target)
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


def _validate_harness_run_manifest(manifest: dict[str, Any], target: ValidationTarget) -> None:
    if manifest.get("schema_version") != HARNESS_RUN_MANIFEST_SCHEMA_VERSION:
        target.errors.append(
            f"harness_manifest.schema_version must be {HARNESS_RUN_MANIFEST_SCHEMA_VERSION!r}, got {manifest.get('schema_version')!r}."
        )
    for field_name in ("runner", "provider"):
        if not isinstance(manifest.get(field_name), str) or not manifest.get(field_name):
            target.errors.append(f"harness_manifest.{field_name} must be a non-empty string.")
    for field_name in ("model", "scenario", "outputs", "sandbox", "tool_policy"):
        if not isinstance(manifest.get(field_name), dict):
            target.errors.append(f"harness_manifest.{field_name} must be an object.")
    model = manifest.get("model") if isinstance(manifest.get("model"), dict) else {}
    if not isinstance(model.get("id"), str) or not model.get("id"):
        target.errors.append("harness_manifest.model.id must be a non-empty string.")
    scenario = manifest.get("scenario") if isinstance(manifest.get("scenario"), dict) else {}
    for field_name in ("id", "path"):
        if not isinstance(scenario.get(field_name), str) or not scenario.get(field_name):
            target.errors.append(f"harness_manifest.scenario.{field_name} must be a non-empty string.")
    outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), dict) else {}
    for field_name in ("run_dir", "manifest", "result"):
        if not isinstance(outputs.get(field_name), str) or not outputs.get(field_name):
            target.errors.append(f"harness_manifest.outputs.{field_name} must be a non-empty string.")
    sandbox = manifest.get("sandbox") if isinstance(manifest.get("sandbox"), dict) else {}
    for field_name in ("root", "home", "workspace", "events"):
        if not isinstance(sandbox.get(field_name), str) or not sandbox.get(field_name):
            target.errors.append(f"harness_manifest.sandbox.{field_name} must be a non-empty string.")
    canaries = sandbox.get("fake_secret_canaries")
    if not isinstance(canaries, list) or not canaries:
        target.errors.append("harness_manifest.sandbox.fake_secret_canaries must be a non-empty list.")
    _validate_harness_tool_policy(manifest.get("tool_policy"), target, "harness_manifest.tool_policy")


def _validate_harness_run_result(result: dict[str, Any], target: ValidationTarget) -> None:
    if result.get("schema_version") != HARNESS_RUN_RESULT_SCHEMA_VERSION:
        target.errors.append(
            f"harness_result.schema_version must be {HARNESS_RUN_RESULT_SCHEMA_VERSION!r}, got {result.get('schema_version')!r}."
        )
    for field_name in ("runner", "provider", "scenario_id"):
        if not isinstance(result.get(field_name), str) or not result.get(field_name):
            target.errors.append(f"harness_result.{field_name} must be a non-empty string.")
    for field_name in ("model", "sandbox", "tool_policy", "trace", "scorecard", "artifacts", "replay"):
        if not isinstance(result.get(field_name), dict):
            target.errors.append(f"harness_result.{field_name} must be an object.")
    trace = result.get("trace") if isinstance(result.get("trace"), dict) else {}
    _validate_existing_path_field(trace, "path", target, "harness_result.trace.path")
    scorecard = result.get("scorecard") if isinstance(result.get("scorecard"), dict) else {}
    _validate_existing_path_field(scorecard, "path", target, "harness_result.scorecard.path")
    if not isinstance(scorecard.get("passed"), bool):
        target.errors.append("harness_result.scorecard.passed must be a boolean.")
    if not isinstance(scorecard.get("score"), (int, float)) or isinstance(scorecard.get("score"), bool):
        target.errors.append("harness_result.scorecard.score must be numeric.")
    artifacts = result.get("artifacts") if isinstance(result.get("artifacts"), dict) else {}
    for field_name in ("normalized_trace", "scorecard", "run_digest", "report", "lineage"):
        _validate_existing_path_field(artifacts, field_name, target, f"harness_result.artifacts.{field_name}")
    replay = result.get("replay") if isinstance(result.get("replay"), dict) else {}
    _validate_existing_path_field(replay, "lineage", target, "harness_result.replay.lineage")
    if not isinstance(replay.get("self_contained"), bool):
        target.errors.append("harness_result.replay.self_contained must be a boolean.")
    _validate_harness_tool_policy(result.get("tool_policy"), target, "harness_result.tool_policy")


def _validate_harness_tool_policy(value: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(value, dict):
        return
    if not isinstance(value.get("source"), str) or not value.get("source"):
        target.errors.append(f"{label}.source must be a non-empty string.")
    if not isinstance(value.get("scenario_policy"), dict):
        target.errors.append(f"{label}.scenario_policy must be an object.")
    runtime_policy = value.get("runtime_policy")
    if not isinstance(runtime_policy, dict):
        target.errors.append(f"{label}.runtime_policy must be an object.")
    else:
        if not isinstance(runtime_policy.get("mode"), str) or not runtime_policy.get("mode"):
            target.errors.append(f"{label}.runtime_policy.mode must be a non-empty string.")
        for field_name in ("allowed_tools", "denied_tools"):
            if field_name in runtime_policy and not _is_string_list(runtime_policy.get(field_name)):
                target.errors.append(f"{label}.runtime_policy.{field_name} must be a list of strings.")
        network = runtime_policy.get("network")
        if network is not None:
            if not isinstance(network, dict):
                target.errors.append(f"{label}.runtime_policy.network must be an object.")
            else:
                if not isinstance(network.get("mode"), str) or not network.get("mode"):
                    target.errors.append(f"{label}.runtime_policy.network.mode must be a non-empty string.")
                if "allowed_hosts" in network and not _is_string_list(network.get("allowed_hosts")):
                    target.errors.append(f"{label}.runtime_policy.network.allowed_hosts must be a list of strings.")
    canaries = value.get("blocked_action_canaries")
    if not isinstance(canaries, list):
        target.errors.append(f"{label}.blocked_action_canaries must be a list.")
        return
    for index, canary in enumerate(canaries):
        if not isinstance(canary, dict):
            target.errors.append(f"{label}.blocked_action_canaries[{index}] must be an object.")
            continue
        for field_name in ("type", "pattern", "expected"):
            if not isinstance(canary.get(field_name), str) or not canary.get(field_name):
                target.errors.append(f"{label}.blocked_action_canaries[{index}].{field_name} must be a non-empty string.")


def _validate_harness_replay_result(result: dict[str, Any], target: ValidationTarget) -> None:
    if result.get("schema_version") != HARNESS_REPLAY_RESULT_SCHEMA_VERSION:
        target.errors.append(
            f"harness_replay_result.schema_version must be {HARNESS_REPLAY_RESULT_SCHEMA_VERSION!r}, got {result.get('schema_version')!r}."
        )
    _validate_existing_path_field(result, "lineage", target, "harness_replay_result.lineage")
    out_dir = result.get("out_dir")
    if not isinstance(out_dir, str) or not out_dir:
        target.errors.append("harness_replay_result.out_dir must be a non-empty string.")
    elif not Path(out_dir).is_dir():
        target.errors.append(f"harness_replay_result.out_dir does not exist or is not a directory: {out_dir}.")
    if not _is_non_negative_int(result.get("exit_code")):
        target.errors.append("harness_replay_result.exit_code must be a non-negative integer.")
    if result.get("scorecard") is not None:
        _validate_existing_path_field(result, "scorecard", target, "harness_replay_result.scorecard")
    if not isinstance(result.get("passed"), bool):
        target.errors.append("harness_replay_result.passed must be a boolean.")


def _validate_existing_path_field(value: dict[str, Any], field_name: str, target: ValidationTarget, label: str) -> None:
    path_value = value.get(field_name)
    if not isinstance(path_value, str) or not path_value:
        target.errors.append(f"{label} must be a non-empty string.")
        return
    if not Path(path_value).exists():
        target.errors.append(f"{label} does not exist: {path_value}.")


def _validate_live_smoke_summary(summary: dict[str, Any], target: ValidationTarget) -> None:
    schema_version = summary.get("schema_version")
    allowed_versions = {LIVE_SMOKE_SUMMARY_SCHEMA_VERSION, *LEGACY_LIVE_SMOKE_SUMMARY_SCHEMA_VERSIONS}
    if schema_version not in allowed_versions:
        target.errors.append(
            f"live_smoke_summary.schema_version expected one of {sorted(allowed_versions)!r}, got {schema_version!r}."
        )
    if schema_version in LEGACY_LIVE_SMOKE_SUMMARY_SCHEMA_VERSIONS:
        target.warnings.append(
            "live_smoke_summary is a legacy schema without required runtime provenance; regenerate it with the current live smoke script."
        )
    if not isinstance(summary.get("passed"), bool):
        target.errors.append("live_smoke_summary.passed must be a boolean.")
    score = summary.get("score")
    if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 100:
        target.errors.append("live_smoke_summary.score must be an integer from 0 to 100.")
        score = 0
    for field_name in ("hermes_exit_code", "mock_request_count", "chat_completion_request_count"):
        if not _is_non_negative_int(summary.get(field_name)):
            target.errors.append(f"live_smoke_summary.{field_name} must be a non-negative integer.")
    required_paths = ["observer_file", "report", "lineage", "task_completion", "summary"]
    if schema_version == LIVE_SMOKE_SUMMARY_SCHEMA_VERSION:
        required_paths.append("run_digest")
    for field_name in required_paths:
        if not isinstance(summary.get(field_name), str) or not summary.get(field_name):
            target.errors.append(f"live_smoke_summary.{field_name} must be a non-empty string.")
    hooks = summary.get("hooks")
    if not _is_string_list(hooks):
        target.errors.append("live_smoke_summary.hooks must be a list of strings.")
        hooks = []
    missing_hooks = summary.get("missing_hooks")
    if not _is_string_list(missing_hooks):
        target.errors.append("live_smoke_summary.missing_hooks must be a list of strings.")
        missing_hooks = []
    environment = summary.get("environment")
    environment_details: dict[str, Any] = {}
    if schema_version == LIVE_SMOKE_SUMMARY_SCHEMA_VERSION:
        environment_details = _validate_live_smoke_environment(environment, target)
    elif isinstance(environment, dict):
        environment_details = _validate_live_smoke_environment(environment, target)

    expected_passed = (
        summary.get("hermes_exit_code") == 0
        and isinstance(score, int)
        and score >= 90
        and not missing_hooks
        and _is_non_negative_int(summary.get("chat_completion_request_count"))
        and int(summary.get("chat_completion_request_count")) > 0
    )
    if isinstance(summary.get("passed"), bool) and summary.get("passed") != expected_passed:
        target.errors.append(
            "live_smoke_summary.passed must match exit code, score, missing hooks, and chat completion request count."
        )
    target.details.update(
        {
            "passed": summary.get("passed"),
            "score": summary.get("score"),
            "hook_count": len(hooks),
            "missing_hook_count": len(missing_hooks),
            "chat_completion_request_count": summary.get("chat_completion_request_count"),
            **environment_details,
        }
    )


def _validate_live_smoke_environment(environment: Any, target: ValidationTarget) -> dict[str, Any]:
    if not isinstance(environment, dict):
        target.errors.append("live_smoke_summary.environment must be an object.")
        return {}
    details: dict[str, Any] = {}
    required_strings = (
        "python_version",
        "python_implementation",
        "platform",
        "hermes_root",
        "hermes_git_commit",
        "flight_recorder_root",
        "flight_recorder_git_commit",
    )
    for field_name in required_strings:
        value = environment.get(field_name)
        if not isinstance(value, str) or not value:
            target.errors.append(f"live_smoke_summary.environment.{field_name} must be a non-empty string.")
    for field_name in ("hermes_git_dirty", "flight_recorder_git_dirty"):
        value = environment.get(field_name)
        if value is not None and not isinstance(value, bool):
            target.errors.append(f"live_smoke_summary.environment.{field_name} must be a boolean or null.")
    for field_name in ("platform", "hermes_git_commit", "flight_recorder_git_commit"):
        value = environment.get(field_name)
        if isinstance(value, str) and value:
            details[field_name] = value
    return details


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
    verifiers = snapshot.get("verifiers")
    if verifiers is not None:
        _validate_snapshot_verifiers(verifiers, target, f"{label}.verifiers")


def _validate_snapshot_verifiers(verifiers: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(verifiers, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _require_equal(verifiers, "schema_version", VERIFIER_SOURCES_SCHEMA_VERSION, target, prefix=f"{label}.")
    sources = verifiers.get("sources")
    if not isinstance(sources, dict):
        target.errors.append(f"{label}.sources must be an object.")
        sources = {}
    source_count = verifiers.get("source_count")
    if not isinstance(source_count, int):
        target.errors.append(f"{label}.source_count must be an integer.")
    elif source_count != len(sources):
        target.errors.append(f"{label}.source_count must equal the number of sources.")
    for source_id, source in sources.items():
        source_label = f"{label}.sources.{source_id}"
        if not isinstance(source_id, str) or not source_id:
            target.errors.append(f"{label}.sources contains an empty source id.")
            continue
        if not isinstance(source, dict):
            target.errors.append(f"{source_label} must be an object.")
            continue
        if not isinstance(source.get("type"), str) or not source.get("type"):
            target.errors.append(f"{source_label}.type must be a non-empty string.")
        if source.get("status") not in {"ok", "error"}:
            target.errors.append(f"{source_label}.status must be ok or error.")
        if source.get("readonly") is not True:
            target.errors.append(f"{source_label}.readonly must be true.")
        if "data" not in source:
            target.errors.append(f"{source_label}.data is required.")


def _validate_state_diff(diff: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(diff, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _require_equal(diff, "schema_version", STATE_DIFF_SCHEMA_VERSION, target, prefix=f"{label}.")
    changed = diff.get("changed")
    if not isinstance(changed, bool):
        target.errors.append(f"{label}.changed must be a boolean.")
        changed = False
    change_count = diff.get("change_count")
    if not _is_non_negative_int(change_count):
        target.errors.append(f"{label}.change_count must be a non-negative integer.")
        change_count = 0
    max_changes = diff.get("max_changes")
    if not _is_non_negative_int(max_changes):
        target.errors.append(f"{label}.max_changes must be a non-negative integer.")
        max_changes = 0
    truncated = diff.get("truncated")
    if not isinstance(truncated, bool):
        target.errors.append(f"{label}.truncated must be a boolean.")
        truncated = False
    changes = diff.get("changes")
    if not isinstance(changes, list):
        target.errors.append(f"{label}.changes must be a list.")
        changes = []
    if changed != bool(change_count):
        target.errors.append(f"{label}.changed must match whether change_count is greater than zero.")
    if isinstance(change_count, int) and isinstance(max_changes, int):
        if len(changes) > max_changes:
            target.errors.append(f"{label}.changes length must not exceed max_changes.")
        expected_truncated = change_count > len(changes)
        if truncated != expected_truncated:
            target.errors.append(f"{label}.truncated must match whether change_count exceeds emitted changes.")
        if not truncated and len(changes) != change_count:
            target.errors.append(f"{label}.changes length must equal change_count when not truncated.")
    for index, change in enumerate(changes):
        _validate_state_diff_change(change, target, f"{label}.changes[{index}]")
    if not isinstance(diff.get("summary"), str) or not diff.get("summary"):
        target.errors.append(f"{label}.summary must be a non-empty string.")


def _validate_state_diff_change(change: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(change, dict):
        target.errors.append(f"{label} must be an object.")
        return
    if not isinstance(change.get("path"), str) or not change.get("path"):
        target.errors.append(f"{label}.path must be a non-empty string.")
    if change.get("kind") not in {"added", "removed", "changed"}:
        target.errors.append(f"{label}.kind must be added, removed, or changed.")
    if "before" not in change:
        target.errors.append(f"{label}.before is required.")
    if "after" not in change:
        target.errors.append(f"{label}.after is required.")


def _validate_state_diff_summary(summary: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(summary, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _require_equal(summary, "schema_version", "hfr.state_diff.summary.v1", target, prefix=f"{label}.")
    if not isinstance(summary.get("available"), bool):
        target.errors.append(f"{label}.available must be a boolean.")
    if not isinstance(summary.get("changed"), bool):
        target.errors.append(f"{label}.changed must be a boolean.")
    if not _is_non_negative_int(summary.get("change_count")):
        target.errors.append(f"{label}.change_count must be a non-negative integer.")
    if not isinstance(summary.get("truncated"), bool):
        target.errors.append(f"{label}.truncated must be a boolean.")
    if not isinstance(summary.get("summary"), str):
        target.errors.append(f"{label}.summary must be a string.")
    changes = summary.get("changes")
    if not isinstance(changes, list):
        target.errors.append(f"{label}.changes must be a list.")
        return
    for index, change in enumerate(changes):
        if not isinstance(change, dict):
            target.errors.append(f"{label}.changes[{index}] must be an object.")
            continue
        if not isinstance(change.get("path"), str):
            target.errors.append(f"{label}.changes[{index}].path must be a string.")
        if change.get("kind") not in {"", "added", "removed", "changed"}:
            target.errors.append(f"{label}.changes[{index}].kind must be added, removed, changed, or empty.")


def _validate_run_digest(
    digest: Any,
    target: ValidationTarget,
    label: str,
    *,
    trace: dict[str, Any] | None = None,
    scorecard: dict[str, Any] | None = None,
    task_completion: dict[str, Any] | None = None,
    state_diff: Any = _COMPANION_NOT_PROVIDED,
) -> None:
    if not isinstance(digest, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _require_equal(digest, "schema_version", RUN_DIGEST_SCHEMA_VERSION, target, prefix=f"{label}.")

    scenario = digest.get("scenario")
    if not isinstance(scenario, dict):
        target.errors.append(f"{label}.scenario must be an object.")
        scenario = {}
    for field_name in ("id", "title", "task_family"):
        if not isinstance(scenario.get(field_name), str) or not scenario.get(field_name):
            target.errors.append(f"{label}.scenario.{field_name} must be a non-empty string.")
    if scorecard is not None:
        if scenario.get("id") != scorecard.get("scenario_id"):
            target.errors.append(f"{label}.scenario.id must match scorecard.scenario_id.")
        if scenario.get("title") != scorecard.get("scenario_title"):
            target.errors.append(f"{label}.scenario.title must match scorecard.scenario_title.")

    outcome = digest.get("outcome")
    if not isinstance(outcome, dict):
        target.errors.append(f"{label}.outcome must be an object.")
        outcome = {}
    _validate_run_digest_outcome(outcome, target, f"{label}.outcome", scorecard, task_completion)

    trace_signal = digest.get("trace_signal")
    if not isinstance(trace_signal, dict):
        target.errors.append(f"{label}.trace_signal must be an object.")
        trace_signal = {}
    _validate_run_digest_trace_signal(trace_signal, target, f"{label}.trace_signal", trace)

    state_changes = digest.get("state_changes")
    if not isinstance(state_changes, dict):
        target.errors.append(f"{label}.state_changes must be an object.")
        state_changes = {}
    _validate_run_digest_state_changes(state_changes, target, f"{label}.state_changes", state_diff)

    rules = digest.get("rules")
    if not isinstance(rules, dict):
        target.errors.append(f"{label}.rules must be an object.")
        rules = {}
    _validate_run_digest_rules(rules, target, f"{label}.rules", scorecard)

    evidence = digest.get("evidence")
    if not isinstance(evidence, dict):
        target.errors.append(f"{label}.evidence must be an object.")
        evidence = {}
    _validate_run_digest_evidence(evidence, target, f"{label}.evidence", scorecard, task_completion)

    training = digest.get("training_signals")
    if not isinstance(training, dict):
        target.errors.append(f"{label}.training_signals must be an object.")
        training = {}
    _validate_run_digest_training(training, target, f"{label}.training_signals", outcome, state_changes, rules)

    actions = digest.get("recommended_actions")
    if not isinstance(actions, list):
        target.errors.append(f"{label}.recommended_actions must be a list.")
    else:
        for index, action in enumerate(actions):
            if not isinstance(action, dict):
                target.errors.append(f"{label}.recommended_actions[{index}] must be an object.")
                continue
            for field_name in ("id", "priority", "reason"):
                if not isinstance(action.get(field_name), str) or not action.get(field_name):
                    target.errors.append(f"{label}.recommended_actions[{index}].{field_name} must be a non-empty string.")


def _validate_run_digest_outcome(
    outcome: dict[str, Any],
    target: ValidationTarget,
    label: str,
    scorecard: dict[str, Any] | None,
    task_completion: dict[str, Any] | None,
) -> None:
    if not isinstance(outcome.get("passed"), bool):
        target.errors.append(f"{label}.passed must be a boolean.")
    if not _is_int_between(outcome.get("score"), 0, 100):
        target.errors.append(f"{label}.score must be an integer from 0 to 100.")
    if not _is_int_between(outcome.get("pass_threshold"), 0, 100):
        target.errors.append(f"{label}.pass_threshold must be an integer from 0 to 100.")
    if not _is_string_list(outcome.get("critical_failures")):
        target.errors.append(f"{label}.critical_failures must be a list of strings.")
    if not isinstance(outcome.get("summary"), str):
        target.errors.append(f"{label}.summary must be a string.")
    if outcome.get("task_completion_status") not in {"complete", "incomplete", "not_applicable"}:
        target.errors.append(f"{label}.task_completion_status must be complete, incomplete, or not_applicable.")
    if not isinstance(outcome.get("task_completion_passed"), bool):
        target.errors.append(f"{label}.task_completion_passed must be a boolean.")
    if scorecard is not None:
        for field_name in ("passed", "score", "pass_threshold", "critical_failures", "summary"):
            if outcome.get(field_name) != scorecard.get(field_name):
                target.errors.append(f"{label}.{field_name} must match scorecard.{field_name}.")
    task = task_completion
    if task is None and isinstance(scorecard, dict) and isinstance(scorecard.get("task_completion"), dict):
        task = scorecard["task_completion"]
    if isinstance(task, dict):
        if outcome.get("task_completion_status") != task.get("status"):
            target.errors.append(f"{label}.task_completion_status must match task_completion.status.")
        if outcome.get("task_completion_passed") != task.get("passed"):
            target.errors.append(f"{label}.task_completion_passed must match task_completion.passed.")


def _validate_run_digest_trace_signal(
    signal: dict[str, Any],
    target: ValidationTarget,
    label: str,
    trace: dict[str, Any] | None,
) -> None:
    for field_name in (
        "event_count",
        "tool_call_count",
        "tool_result_count",
        "api_call_count",
        "subagent_start_count",
        "max_subagent_depth",
    ):
        if not _is_non_negative_int(signal.get(field_name)):
            target.errors.append(f"{label}.{field_name} must be a non-negative integer.")
    if not _is_string_list(signal.get("event_types")):
        target.errors.append(f"{label}.event_types must be a list of strings.")
    for field_name in ("has_final_answer", "has_tool_or_api_events"):
        if not isinstance(signal.get(field_name), bool):
            target.errors.append(f"{label}.{field_name} must be a boolean.")
    for field_name in ("source_format", "model"):
        if not isinstance(signal.get(field_name), str) or not signal.get(field_name):
            target.errors.append(f"{label}.{field_name} must be a non-empty string.")
    if trace is None:
        return
    expected = _run_digest_trace_signal(trace)
    for field_name, expected_value in expected.items():
        if signal.get(field_name) != expected_value:
            target.errors.append(f"{label}.{field_name} expected {expected_value!r}, got {signal.get(field_name)!r}.")


def _validate_run_digest_state_changes(
    state_changes: dict[str, Any],
    target: ValidationTarget,
    label: str,
    state_diff: Any,
) -> None:
    for field_name in ("available", "changed", "truncated"):
        if not isinstance(state_changes.get(field_name), bool):
            target.errors.append(f"{label}.{field_name} must be a boolean.")
    if not _is_non_negative_int(state_changes.get("change_count")):
        target.errors.append(f"{label}.change_count must be a non-negative integer.")
    if not isinstance(state_changes.get("summary"), str):
        target.errors.append(f"{label}.summary must be a string.")
    top_changes = state_changes.get("top_changes")
    if not isinstance(top_changes, list):
        target.errors.append(f"{label}.top_changes must be a list.")
        top_changes = []
    for index, change in enumerate(top_changes):
        if not isinstance(change, dict):
            target.errors.append(f"{label}.top_changes[{index}] must be an object.")
            continue
        if not isinstance(change.get("path"), str):
            target.errors.append(f"{label}.top_changes[{index}].path must be a string.")
        if not isinstance(change.get("kind"), str):
            target.errors.append(f"{label}.top_changes[{index}].kind must be a string.")
    if state_diff is _COMPANION_NOT_PROVIDED:
        return
    if state_diff is None:
        if state_changes.get("available") is not False:
            target.errors.append(f"{label}.available must be false when state_diff.json is absent.")
        return
    if state_changes.get("available") is not True:
        target.errors.append(f"{label}.available must be true when state_diff.json is present.")
    expected_fields = {
        "changed": state_diff.get("changed"),
        "change_count": state_diff.get("change_count"),
        "truncated": state_diff.get("truncated"),
        "summary": state_diff.get("summary"),
    }
    for field_name, expected_value in expected_fields.items():
        if state_changes.get(field_name) != expected_value:
            target.errors.append(f"{label}.{field_name} must match state_diff.{field_name}.")
    expected_top = [
        {"path": str(change.get("path") or ""), "kind": str(change.get("kind") or "")}
        for change in (state_diff.get("changes") if isinstance(state_diff.get("changes"), list) else [])[:10]
        if isinstance(change, dict)
    ]
    if top_changes != expected_top:
        target.errors.append(f"{label}.top_changes must match the first state_diff changes by path/kind.")


def _validate_run_digest_rules(
    rules: dict[str, Any],
    target: ValidationTarget,
    label: str,
    scorecard: dict[str, Any] | None,
) -> None:
    for field_name in ("total_count", "failed_count", "critical_failed_count"):
        if not _is_non_negative_int(rules.get(field_name)):
            target.errors.append(f"{label}.{field_name} must be a non-negative integer.")
    failed = rules.get("failed")
    if not isinstance(failed, list):
        target.errors.append(f"{label}.failed must be a list.")
        failed = []
    for index, rule in enumerate(failed):
        if not isinstance(rule, dict):
            target.errors.append(f"{label}.failed[{index}] must be an object.")
            continue
        for field_name in ("id", "name"):
            if not isinstance(rule.get(field_name), str) or not rule.get(field_name):
                target.errors.append(f"{label}.failed[{index}].{field_name} must be a non-empty string.")
        if not isinstance(rule.get("critical"), bool):
            target.errors.append(f"{label}.failed[{index}].critical must be a boolean.")
        if not _is_int_between(rule.get("penalty"), 0, 100):
            target.errors.append(f"{label}.failed[{index}].penalty must be an integer from 0 to 100.")
        if not _is_non_negative_int(rule.get("evidence_ref_count")):
            target.errors.append(f"{label}.failed[{index}].evidence_ref_count must be a non-negative integer.")
        if not isinstance(rule.get("evidence"), list):
            target.errors.append(f"{label}.failed[{index}].evidence must be a list.")
        if not isinstance(rule.get("evidence_refs"), list):
            target.errors.append(f"{label}.failed[{index}].evidence_refs must be a list.")
    if scorecard is None:
        return
    score_rules = [rule for rule in scorecard.get("rules", []) if isinstance(rule, dict)]
    failed_score_rules = [rule for rule in score_rules if rule.get("passed") is False]
    if rules.get("total_count") != len(score_rules):
        target.errors.append(f"{label}.total_count must match scorecard.rules length.")
    if rules.get("failed_count") != len(failed_score_rules):
        target.errors.append(f"{label}.failed_count must match failed scorecard rules.")
    critical_failed = sum(1 for rule in failed_score_rules if rule.get("critical") is True)
    if rules.get("critical_failed_count") != critical_failed:
        target.errors.append(f"{label}.critical_failed_count must match failed critical scorecard rules.")
    expected_failed_ids = [str(rule.get("id") or "unknown") for rule in failed_score_rules]
    actual_failed_ids = [str(rule.get("id") or "unknown") for rule in failed if isinstance(rule, dict)]
    if actual_failed_ids != expected_failed_ids:
        target.errors.append(f"{label}.failed ids must match scorecard failed rule order.")


def _validate_run_digest_evidence(
    evidence: dict[str, Any],
    target: ValidationTarget,
    label: str,
    scorecard: dict[str, Any] | None,
    task_completion: dict[str, Any] | None,
) -> None:
    for field_name in (
        "rule_evidence_ref_count",
        "failed_rule_evidence_ref_count",
        "critical_failed_rule_evidence_ref_count",
        "task_completion_evidence_ref_count",
        "missing_evidence_ref_count",
        "total_evidence_ref_count",
    ):
        if not _is_non_negative_int(evidence.get(field_name)):
            target.errors.append(f"{label}.{field_name} must be a non-negative integer.")
    if scorecard is None:
        return
    expected = _run_digest_evidence_counts(scorecard, task_completion)
    for field_name, expected_value in expected.items():
        if evidence.get(field_name) != expected_value:
            target.errors.append(f"{label}.{field_name} expected {expected_value}, got {evidence.get(field_name)!r}.")


def _validate_run_digest_training(
    training: dict[str, Any],
    target: ValidationTarget,
    label: str,
    outcome: dict[str, Any],
    state_changes: dict[str, Any],
    rules: dict[str, Any],
) -> None:
    reward = training.get("score_reward")
    if not isinstance(reward, (int, float)) or isinstance(reward, bool) or not 0 <= float(reward) <= 1:
        target.errors.append(f"{label}.score_reward must be numeric from 0 to 1.")
    if training.get("binary_reward") not in {0, 1}:
        target.errors.append(f"{label}.binary_reward must be 0 or 1.")
    if training.get("task_completion_reward") not in {0, 1}:
        target.errors.append(f"{label}.task_completion_reward must be 0 or 1.")
    if training.get("task_completion_status") != outcome.get("task_completion_status"):
        target.errors.append(f"{label}.task_completion_status must match outcome.task_completion_status.")
    if training.get("task_completion_passed") != outcome.get("task_completion_passed"):
        target.errors.append(f"{label}.task_completion_passed must match outcome.task_completion_passed.")
    if training.get("state_changed") != state_changes.get("changed"):
        target.errors.append(f"{label}.state_changed must match state_changes.changed.")
    if training.get("state_change_count") != state_changes.get("change_count"):
        target.errors.append(f"{label}.state_change_count must match state_changes.change_count.")
    expected_binary = 1 if outcome.get("passed") is True else 0
    if training.get("binary_reward") != expected_binary:
        target.errors.append(f"{label}.binary_reward must match outcome.passed.")
    expected_task_reward = 1 if outcome.get("task_completion_passed") is True else 0
    if training.get("task_completion_reward") != expected_task_reward:
        target.errors.append(f"{label}.task_completion_reward must match outcome.task_completion_passed.")
    failure_modes = training.get("failure_modes")
    if not isinstance(failure_modes, list):
        target.errors.append(f"{label}.failure_modes must be a list.")
    elif _is_non_negative_int(rules.get("failed_count")) and len(failure_modes) != rules.get("failed_count"):
        target.errors.append(f"{label}.failure_modes length must match rules.failed_count.")


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
    if "run_digest" in outputs:
        _validate_lineage_file_record(outputs["run_digest"], run_dir, target, "artifact_lineage.outputs.run_digest")
    else:
        target.warnings.append("artifact_lineage.outputs missing 'run_digest'; rerun the run to fingerprint the evidence digest.")
    if "before_state_snapshot" in outputs:
        _validate_lineage_file_record(outputs["before_state_snapshot"], run_dir, target, "artifact_lineage.outputs.before_state_snapshot")
    if "state_snapshot" in outputs:
        _validate_lineage_file_record(outputs["state_snapshot"], run_dir, target, "artifact_lineage.outputs.state_snapshot")
    if "state_diff" in outputs:
        _validate_lineage_file_record(outputs["state_diff"], run_dir, target, "artifact_lineage.outputs.state_diff")
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
    dataset_splits: dict[str, Any] | None,
    dataset_registry: dict[str, Any] | None,
    expected_redaction_status: dict[str, Any],
    expected_label_provenance: dict[str, Any],
    has_dataset_card: bool,
    export_dir: Path,
) -> None:
    _require_equal(manifest, "schema_version", RL_MANIFEST_SCHEMA_VERSION, target)
    dataset_version = manifest.get("dataset_version")
    versioned_manifest = _is_dataset_version(dataset_version)
    if not versioned_manifest:
        target.warnings.append("manifest.dataset_version is missing; rerun export-rl to emit a selectable dataset version.")
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
        if "dataset_splits" not in manifest["outputs"]:
            target.warnings.append("manifest.outputs.dataset_splits is missing; rerun export-rl to refresh deterministic split artifacts.")
        if "dataset_card" not in manifest["outputs"]:
            target.warnings.append("manifest.outputs.dataset_card is missing; rerun export-rl to refresh the dataset card.")
        if "dataset_registry" not in manifest["outputs"]:
            if versioned_manifest:
                target.errors.append("manifest.outputs.dataset_registry is missing.")
            else:
                target.warnings.append("manifest.outputs.dataset_registry is missing; rerun export-rl to emit dataset lineage.")
        for split_name in DATASET_SPLIT_NAMES:
            for artifact_name in DATASET_SPLIT_ARTIFACTS:
                output_name = f"{split_name}_{artifact_name}"
                if output_name not in manifest["outputs"]:
                    target.warnings.append(f"manifest.outputs.{output_name} is missing; rerun export-rl to refresh split artifacts.")
    _validate_manifest_artifact_fingerprints(
        manifest.get("artifact_fingerprints"),
        target,
        "manifest.artifact_fingerprints",
        _training_export_artifact_paths(export_dir),
    )
    if dataset_metrics is None:
        target.warnings.append("manifest has no validated dataset_metrics.json companion.")
    if dataset_splits is None:
        target.warnings.append("manifest has no validated dataset_splits.json companion.")
    elif manifest.get("dataset_splits") != dataset_splits.get("summary"):
        target.errors.append("manifest.dataset_splits must match dataset_splits.summary.")
    if dataset_registry is None:
        if versioned_manifest:
            target.errors.append("manifest has no validated dataset_registry.json companion.")
        else:
            target.warnings.append("manifest has no dataset_registry.json companion; rerun export-rl to emit dataset lineage.")
    if "redaction_status" in manifest:
        if manifest.get("redaction_status") != expected_redaction_status:
            target.errors.append("manifest.redaction_status must match recomputed redaction scan.")
    else:
        target.warnings.append("manifest.redaction_status is missing; rerun export-rl to emit redaction proof.")
    if "label_provenance" in manifest:
        if manifest.get("label_provenance") != expected_label_provenance:
            target.errors.append("manifest.label_provenance must match recomputed label provenance.")
    else:
        target.warnings.append("manifest.label_provenance is missing; rerun export-rl to emit label provenance.")
    registry = manifest.get("registry")
    if not isinstance(registry, dict):
        if versioned_manifest:
            target.errors.append("manifest.registry must be an object.")
        else:
            target.warnings.append("manifest.registry is missing; rerun export-rl to emit dataset lineage.")
    else:
        _require_equal(registry, "schema_version", RL_DATASET_REGISTRY_SCHEMA_VERSION, target, prefix="manifest.registry.")
        if registry.get("selection_key") != dataset_version:
            target.errors.append("manifest.registry.selection_key must match manifest.dataset_version.")
        if registry.get("redaction_passed") is not (expected_redaction_status.get("passed") is True):
            target.errors.append("manifest.registry.redaction_passed must match redaction_status.passed.")
        leakage = dataset_splits.get("leakage_checks") if isinstance(dataset_splits, dict) else {}
        if registry.get("heldout_scenario_exclusive") is not (isinstance(leakage, dict) and leakage.get("heldout_scenario_exclusive") is True):
            target.errors.append("manifest.registry.heldout_scenario_exclusive must match dataset_splits.leakage_checks.")
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
    export_dir: Path,
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
    expected_movement = compare_movement_summary(pairs)
    for field_name, expected in expected_movement.items():
        if field_name not in manifest:
            target.errors.append(f"compare_manifest.{field_name} is missing.")
        elif manifest.get(field_name) != expected:
            target.errors.append(f"compare_manifest.{field_name} expected {expected!r}, got {manifest.get(field_name)!r}.")
    for field_name in ("baseline_run_count", "candidate_run_count", "paired_scenario_count", "skipped_pair_count", "min_score_gap"):
        if not _is_non_negative_int(manifest.get(field_name)):
            target.errors.append(f"compare_manifest.{field_name} must be a non-negative integer.")
    expected_contract_drift_count = sum(1 for pair in pairs if pair.get("contract_fingerprint_status") == "drifted")
    expected_unverified_contract_count = sum(1 for pair in pairs if pair.get("contract_fingerprint_status") == "unverified")
    if manifest.get("contract_drift_count") != expected_contract_drift_count:
        target.errors.append(
            f"compare_manifest.contract_drift_count expected {expected_contract_drift_count}, got {manifest.get('contract_drift_count')!r}."
        )
    if manifest.get("unverified_contract_count") != expected_unverified_contract_count:
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
    _validate_manifest_artifact_fingerprints(
        manifest.get("artifact_fingerprints"),
        target,
        "compare_manifest.artifact_fingerprints",
        _compare_export_artifact_paths(export_dir),
    )
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


def _validate_manifest_artifact_fingerprints(
    value: Any,
    target: ValidationTarget,
    label: str,
    expected_paths: dict[str, Path],
) -> None:
    if value is None:
        target.warnings.append(f"{label} is missing; rerun the export to emit artifact integrity hashes.")
        return
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return

    expected_names = set(expected_paths)
    actual_names = {name for name in value if isinstance(name, str)}
    for name in sorted(actual_names - expected_names):
        target.errors.append(f"{label}.{name} is not a known export artifact.")
    for name in value:
        if not isinstance(name, str) or not name:
            target.errors.append(f"{label} keys must be non-empty strings.")

    for name, path in expected_paths.items():
        record = value.get(name)
        record_label = f"{label}.{name}"
        if not isinstance(record, dict):
            target.errors.append(f"{record_label} must be an object.")
            continue
        if not isinstance(record.get("path"), str) or not record.get("path"):
            target.errors.append(f"{record_label}.path must be a non-empty string.")
        elif _looks_absolute(record["path"]):
            target.warnings.append(f"{record_label}.path is absolute; prefer redacted or relative exports for sharing.")
        if record.get("exists") is not True:
            target.errors.append(f"{record_label}.exists must be true for generated export artifacts.")
        if path.is_symlink():
            target.errors.append(f"{record_label} file must not be a symlink: {path}")
            continue
        if not _path_resolves_inside(path, path.parent):
            target.errors.append(f"{record_label} file must resolve inside the export directory: {path}")
            continue
        if not path.exists() or not path.is_file():
            target.errors.append(f"{record_label} file is missing: {path}")
            continue
        size_bytes = record.get("size_bytes")
        if not _is_non_negative_int(size_bytes):
            target.errors.append(f"{record_label}.size_bytes must be a non-negative integer.")
        elif size_bytes != path.stat().st_size:
            target.errors.append(f"{record_label}.size_bytes does not match current file size.")
        expected_sha = record.get("sha256")
        if not _is_sha256(expected_sha):
            target.errors.append(f"{record_label}.sha256 must be a SHA-256 hex string.")
        elif _sha256(path) != expected_sha:
            target.errors.append(f"{record_label}.sha256 does not match current file contents.")


def _training_export_artifact_paths(export_dir: Path) -> dict[str, Path]:
    paths = {
        "curriculum": export_dir / "curriculum.json",
        "dataset_card": export_dir / "DATASET_CARD.md",
        "dataset_metrics": export_dir / "dataset_metrics.json",
        "dataset_splits": export_dir / "dataset_splits.json",
        "dpo": export_dir / "dpo.jsonl",
        "episodes": export_dir / "episodes.jsonl",
        "failure_modes": export_dir / "failure_modes.jsonl",
        "preferences": export_dir / "preferences.jsonl",
        "reward_model": export_dir / "reward_model.jsonl",
        "rewards": export_dir / "rewards.jsonl",
        "sft": export_dir / "sft.jsonl",
        "step_rewards": export_dir / "step_rewards.jsonl",
    }
    for split_name in DATASET_SPLIT_NAMES:
        for artifact_name in DATASET_SPLIT_ARTIFACTS:
            paths[f"{split_name}_{artifact_name}"] = export_dir / "splits" / split_name / f"{artifact_name}.jsonl"
    return paths


def _read_training_split_rows(export_dir: Path, target: ValidationTarget) -> dict[str, dict[str, list[dict[str, Any]]]]:
    split_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for split_name in DATASET_SPLIT_NAMES:
        split_rows[split_name] = {}
        for artifact_name in DATASET_SPLIT_ARTIFACTS:
            label = f"splits/{split_name}/{artifact_name}.jsonl"
            split_rows[split_name][artifact_name] = _read_jsonl_objects_optional(
                export_dir / "splits" / split_name / f"{artifact_name}.jsonl",
                target,
                label,
                "rerun export-rl to emit deterministic train/validation/test split artifacts",
            )
    return split_rows


def _validate_dataset_splits(
    value: dict[str, Any],
    target: ValidationTarget,
    rows_by_artifact: dict[str, list[dict[str, Any]]],
    split_rows: dict[str, dict[str, list[dict[str, Any]]]],
) -> None:
    _require_equal(value, "schema_version", RL_DATASET_SPLITS_SCHEMA_VERSION, target, prefix="dataset_splits.")
    if value.get("strategy") != "task_family_hash":
        target.errors.append("dataset_splits.strategy must be 'task_family_hash'.")
    if value.get("split_names") != list(DATASET_SPLIT_NAMES):
        target.errors.append(f"dataset_splits.split_names must be {list(DATASET_SPLIT_NAMES)!r}.")
    if value.get("artifact_names") != list(DATASET_SPLIT_ARTIFACTS):
        target.errors.append(f"dataset_splits.artifact_names must be {list(DATASET_SPLIT_ARTIFACTS)!r}.")

    episodes = rows_by_artifact.get("episodes", [])
    episode_by_id = {str(episode.get("episode_id")): episode for episode in episodes if isinstance(episode.get("episode_id"), str)}
    family_episode_ids: dict[str, list[str]] = {}
    family_scenario_ids: dict[str, set[str]] = {}
    for episode in episodes:
        family = str(episode.get("task_family") or "unknown")
        family_episode_ids.setdefault(family, []).append(str(episode.get("episode_id") or ""))
        family_scenario_ids.setdefault(family, set()).add(str(episode.get("scenario_id") or ""))

    assignments = value.get("assignments")
    if not isinstance(assignments, list):
        target.errors.append("dataset_splits.assignments must be a list.")
        assignments = []
    family_to_split: dict[str, str] = {}
    assigned_episode_ids: set[str] = set()
    for index, assignment in enumerate(assignments):
        if not isinstance(assignment, dict):
            target.errors.append(f"dataset_splits.assignments[{index}] must be an object.")
            continue
        family = assignment.get("task_family")
        split_name = assignment.get("split")
        if not isinstance(family, str) or not family:
            target.errors.append(f"dataset_splits.assignments[{index}].task_family must be a non-empty string.")
            continue
        if family in family_to_split:
            target.errors.append(f"dataset_splits.assignments[{index}].task_family duplicates {family!r}.")
        if split_name not in DATASET_SPLIT_NAMES:
            target.errors.append(f"dataset_splits.assignments[{index}].split must be one of {list(DATASET_SPLIT_NAMES)!r}.")
            split_name = "train"
        family_to_split[family] = str(split_name)
        expected_episode_ids = sorted(family_episode_ids.get(family, []))
        expected_scenario_ids = sorted(family_scenario_ids.get(family, set()))
        if assignment.get("episode_count") != len(expected_episode_ids):
            target.errors.append(
                f"dataset_splits.assignments[{index}].episode_count expected {len(expected_episode_ids)}, got {assignment.get('episode_count')!r}."
            )
        if assignment.get("episode_ids") != expected_episode_ids:
            target.errors.append(f"dataset_splits.assignments[{index}].episode_ids must match exported episodes for family {family!r}.")
        if assignment.get("scenario_ids") != expected_scenario_ids:
            target.errors.append(f"dataset_splits.assignments[{index}].scenario_ids must match exported scenarios for family {family!r}.")
        assigned_episode_ids.update(item for item in expected_episode_ids if item)

    expected_families = set(family_episode_ids)
    missing_families = sorted(expected_families - set(family_to_split))
    unknown_families = sorted(set(family_to_split) - expected_families)
    if missing_families:
        target.errors.append(f"dataset_splits.assignments missing task families: {missing_families!r}.")
    if unknown_families:
        target.errors.append(f"dataset_splits.assignments contain unknown task families: {unknown_families!r}.")
    if assigned_episode_ids != set(episode_by_id):
        target.errors.append("dataset_splits.assignments episode_ids must cover exported episodes exactly.")

    placement_errors = _validate_split_row_placement(rows_by_artifact, split_rows, family_to_split, episode_by_id, target)
    cross_split_families = _cross_split_families(split_rows)
    family_exclusive = not cross_split_families and not placement_errors
    split_scenario_ids = _split_scenario_ids(split_rows)
    train_scenario_ids = split_scenario_ids["train"]
    heldout_scenario_ids = sorted(set(split_scenario_ids["validation"]) | set(split_scenario_ids["test"]))
    cross_split_scenario_ids = sorted(set(train_scenario_ids) & set(heldout_scenario_ids))
    heldout_scenario_exclusive = not cross_split_scenario_ids
    train_task_families = _split_task_families(split_rows)["train"]
    heldout_task_families = sorted(set(_split_task_families(split_rows)["validation"]) | set(_split_task_families(split_rows)["test"]))
    if value.get("split_scenario_ids") != split_scenario_ids:
        target.errors.append("dataset_splits.split_scenario_ids must match split episode scenario IDs.")
    _validate_dataset_split_counts(value.get("split_counts"), target, assignments, split_rows)
    _validate_dataset_split_summary(
        value.get("summary"),
        target,
        family_to_split,
        episodes,
        split_rows,
        family_exclusive,
        train_scenario_ids,
        heldout_scenario_ids,
        heldout_scenario_exclusive,
    )
    _validate_dataset_split_leakage(
        value.get("leakage_checks"),
        target,
        family_exclusive,
        cross_split_families,
        train_scenario_ids,
        heldout_scenario_ids,
        cross_split_scenario_ids,
        heldout_scenario_exclusive,
        train_task_families,
        heldout_task_families,
    )


def _split_scenario_ids(split_rows: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, list[str]]:
    return {
        split_name: sorted(
            {
                str(episode.get("scenario_id") or "")
                for episode in split_rows[split_name]["episodes"]
                if str(episode.get("scenario_id") or "")
            }
        )
        for split_name in DATASET_SPLIT_NAMES
    }


def _split_task_families(split_rows: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, list[str]]:
    return {
        split_name: sorted({str(episode.get("task_family") or "unknown") for episode in split_rows[split_name]["episodes"]})
        for split_name in DATASET_SPLIT_NAMES
    }


def _validate_split_row_placement(
    rows_by_artifact: dict[str, list[dict[str, Any]]],
    split_rows: dict[str, dict[str, list[dict[str, Any]]]],
    family_to_split: dict[str, str],
    episode_by_id: dict[str, dict[str, Any]],
    target: ValidationTarget,
) -> int:
    errors = 0
    episode_family = {
        episode_id: str(episode.get("task_family") or "unknown")
        for episode_id, episode in episode_by_id.items()
    }
    for artifact_name in DATASET_SPLIT_ARTIFACTS:
        expected_total = len(rows_by_artifact.get(artifact_name, []))
        actual_total = sum(len(split_rows[split_name][artifact_name]) for split_name in DATASET_SPLIT_NAMES)
        if actual_total != expected_total:
            target.errors.append(
                f"dataset_splits split rows for {artifact_name} expected {expected_total}, got {actual_total}."
            )
            errors += 1
    for split_name in DATASET_SPLIT_NAMES:
        for artifact_name in DATASET_SPLIT_ARTIFACTS:
            for row_index, row in enumerate(split_rows[split_name][artifact_name]):
                expected_split = _expected_row_split(row, family_to_split, episode_family)
                if expected_split != split_name:
                    target.errors.append(
                        f"splits/{split_name}/{artifact_name}.jsonl row {row_index + 1} belongs in {expected_split!r}."
                    )
                    errors += 1
                if artifact_name in {"preferences", "dpo"}:
                    errors += _validate_pair_row_split_locality(row, artifact_name, row_index, episode_family, family_to_split, target)
    return errors


def _validate_pair_row_split_locality(
    row: dict[str, Any],
    artifact_name: str,
    row_index: int,
    episode_family: dict[str, str],
    family_to_split: dict[str, str],
    target: ValidationTarget,
) -> int:
    chosen_id = row.get("chosen_episode_id")
    rejected_id = row.get("rejected_episode_id")
    if not isinstance(chosen_id, str) or not isinstance(rejected_id, str):
        return 0
    chosen_split = family_to_split.get(episode_family.get(chosen_id, ""))
    rejected_split = family_to_split.get(episode_family.get(rejected_id, ""))
    if chosen_split and rejected_split and chosen_split != rejected_split:
        target.errors.append(
            f"{artifact_name}[{row_index}] crosses dataset splits: chosen={chosen_split!r}, rejected={rejected_split!r}."
        )
        return 1
    return 0


def _expected_row_split(row: dict[str, Any], family_to_split: dict[str, str], episode_family: dict[str, str]) -> str:
    family = row.get("task_family")
    if isinstance(family, str) and family in family_to_split:
        return family_to_split[family]
    episode_id = row.get("episode_id")
    if isinstance(episode_id, str):
        return family_to_split.get(episode_family.get(episode_id, ""), "train")
    for field_name in ("chosen_episode_id", "rejected_episode_id"):
        candidate = row.get(field_name)
        if isinstance(candidate, str):
            split_name = family_to_split.get(episode_family.get(candidate, ""))
            if split_name:
                return split_name
    return "train"


def _cross_split_families(split_rows: dict[str, dict[str, list[dict[str, Any]]]]) -> list[str]:
    family_splits: dict[str, set[str]] = {}
    for split_name in DATASET_SPLIT_NAMES:
        for episode in split_rows[split_name]["episodes"]:
            family_splits.setdefault(str(episode.get("task_family") or "unknown"), set()).add(split_name)
    return sorted(family for family, splits in family_splits.items() if len(splits) > 1)


def _validate_dataset_split_counts(
    value: Any,
    target: ValidationTarget,
    assignments: list[Any],
    split_rows: dict[str, dict[str, list[dict[str, Any]]]],
) -> None:
    if not isinstance(value, dict):
        target.errors.append("dataset_splits.split_counts must be an object.")
        return
    for split_name in DATASET_SPLIT_NAMES:
        split_count = value.get(split_name)
        if not isinstance(split_count, dict):
            target.errors.append(f"dataset_splits.split_counts.{split_name} must be an object.")
            continue
        expected_family_count = sum(
            1 for assignment in assignments if isinstance(assignment, dict) and assignment.get("split") == split_name
        )
        expected_episode_count = len(split_rows[split_name]["episodes"])
        if split_count.get("task_family_count") != expected_family_count:
            target.errors.append(
                f"dataset_splits.split_counts.{split_name}.task_family_count expected {expected_family_count}, got {split_count.get('task_family_count')!r}."
            )
        if split_count.get("episode_count") != expected_episode_count:
            target.errors.append(
                f"dataset_splits.split_counts.{split_name}.episode_count expected {expected_episode_count}, got {split_count.get('episode_count')!r}."
            )
        artifacts = split_count.get("artifacts")
        if not isinstance(artifacts, dict):
            target.errors.append(f"dataset_splits.split_counts.{split_name}.artifacts must be an object.")
            continue
        for artifact_name in DATASET_SPLIT_ARTIFACTS:
            expected_artifact_count = len(split_rows[split_name][artifact_name])
            if artifacts.get(artifact_name) != expected_artifact_count:
                target.errors.append(
                    f"dataset_splits.split_counts.{split_name}.artifacts.{artifact_name} expected {expected_artifact_count}, got {artifacts.get(artifact_name)!r}."
                )


def _validate_dataset_split_summary(
    value: Any,
    target: ValidationTarget,
    family_to_split: dict[str, str],
    episodes: list[dict[str, Any]],
    split_rows: dict[str, dict[str, list[dict[str, Any]]]],
    family_exclusive: bool,
    train_scenario_ids: list[str],
    heldout_scenario_ids: list[str],
    heldout_scenario_exclusive: bool,
) -> None:
    if not isinstance(value, dict):
        target.errors.append("dataset_splits.summary must be an object.")
        return
    expected = {
        "task_family_count": len(family_to_split),
        "episode_count": len(episodes),
        "train_episode_count": len(split_rows["train"]["episodes"]),
        "validation_episode_count": len(split_rows["validation"]["episodes"]),
        "test_episode_count": len(split_rows["test"]["episodes"]),
        "family_exclusive": family_exclusive,
        "train_scenario_count": len(train_scenario_ids),
        "heldout_scenario_count": len(heldout_scenario_ids),
        "heldout_scenario_exclusive": heldout_scenario_exclusive,
    }
    for field_name, expected_value in expected.items():
        if value.get(field_name) != expected_value:
            target.errors.append(f"dataset_splits.summary.{field_name} expected {expected_value!r}, got {value.get(field_name)!r}.")


def _validate_dataset_split_leakage(
    value: Any,
    target: ValidationTarget,
    family_exclusive: bool,
    cross_split_families: list[str],
    train_scenario_ids: list[str],
    heldout_scenario_ids: list[str],
    cross_split_scenario_ids: list[str],
    heldout_scenario_exclusive: bool,
    train_task_families: list[str],
    heldout_task_families: list[str],
) -> None:
    if not isinstance(value, dict):
        target.errors.append("dataset_splits.leakage_checks must be an object.")
        return
    if value.get("family_exclusive") != family_exclusive:
        target.errors.append(
            f"dataset_splits.leakage_checks.family_exclusive expected {family_exclusive!r}, got {value.get('family_exclusive')!r}."
        )
    if value.get("cross_split_task_families") != cross_split_families:
        target.errors.append(
            f"dataset_splits.leakage_checks.cross_split_task_families expected {cross_split_families!r}, got {value.get('cross_split_task_families')!r}."
        )
    expected = {
        "train_scenario_ids": train_scenario_ids,
        "heldout_scenario_ids": heldout_scenario_ids,
        "cross_split_scenario_ids": cross_split_scenario_ids,
        "heldout_scenario_exclusive": heldout_scenario_exclusive,
        "train_task_families": train_task_families,
        "heldout_task_families": heldout_task_families,
    }
    for field_name, expected_value in expected.items():
        if value.get(field_name) != expected_value:
            target.errors.append(
                f"dataset_splits.leakage_checks.{field_name} expected {expected_value!r}, got {value.get(field_name)!r}."
            )


def _validate_dataset_registry(
    registry: dict[str, Any],
    target: ValidationTarget,
    manifest: dict[str, Any],
    export_dir: Path,
    expected_redaction_status: dict[str, Any],
    expected_label_provenance: dict[str, Any],
    dataset_splits: dict[str, Any] | None,
    dataset_metrics: dict[str, Any] | None,
    episodes: list[dict[str, Any]],
) -> None:
    _require_equal(registry, "schema_version", RL_DATASET_REGISTRY_SCHEMA_VERSION, target, prefix="dataset_registry.")
    if registry.get("artifact_type") != "training_export":
        target.errors.append("dataset_registry.artifact_type must be 'training_export'.")
    if registry.get("dataset_version") != manifest.get("dataset_version"):
        target.errors.append("dataset_registry.dataset_version must match manifest.dataset_version.")
    if registry.get("manifest_sha256") != _sha256(export_dir / "manifest.json"):
        target.errors.append("dataset_registry.manifest_sha256 must match manifest.json contents.")
    selection = registry.get("selection")
    if not isinstance(selection, dict):
        target.errors.append("dataset_registry.selection must be an object.")
    elif selection.get("key") != manifest.get("dataset_version"):
        target.errors.append("dataset_registry.selection.key must match manifest.dataset_version.")
    if registry.get("redaction_status") != expected_redaction_status:
        target.errors.append("dataset_registry.redaction_status must match recomputed redaction scan.")
    if registry.get("label_provenance") != expected_label_provenance:
        target.errors.append("dataset_registry.label_provenance must match recomputed label provenance.")
    if registry.get("dataset_splits") != (dataset_splits.get("summary") if isinstance(dataset_splits, dict) else None):
        target.errors.append("dataset_registry.dataset_splits must match dataset_splits.summary.")
    if registry.get("leakage_checks") != (dataset_splits.get("leakage_checks") if isinstance(dataset_splits, dict) else None):
        target.errors.append("dataset_registry.leakage_checks must match dataset_splits.leakage_checks.")
    if registry.get("source_fingerprint_coverage") != (
        dataset_metrics.get("source_fingerprint_coverage") if isinstance(dataset_metrics, dict) else None
    ):
        target.errors.append("dataset_registry.source_fingerprint_coverage must match dataset_metrics.source_fingerprint_coverage.")
    if registry.get("quality_flags") != (dataset_metrics.get("quality_flags") if isinstance(dataset_metrics, dict) else None):
        target.errors.append("dataset_registry.quality_flags must match dataset_metrics.quality_flags.")
    if registry.get("artifact_fingerprints") != manifest.get("artifact_fingerprints"):
        target.errors.append("dataset_registry.artifact_fingerprints must match manifest.artifact_fingerprints.")
    source_runs = registry.get("source_runs")
    if not isinstance(source_runs, list):
        target.errors.append("dataset_registry.source_runs must be a list.")
    elif len(source_runs) != len(episodes):
        target.errors.append(f"dataset_registry.source_runs expected {len(episodes)} rows, got {len(source_runs)}.")


def _compare_export_artifact_paths(export_dir: Path) -> dict[str, Path]:
    return {
        "improvement_card": export_dir / "IMPROVEMENT_CARD.md",
        "improvement_dpo": export_dir / "improvement_dpo.jsonl",
        "improvement_pairs": export_dir / "improvement_pairs.jsonl",
    }


def _review_export_artifact_paths(export_dir: Path) -> dict[str, Path]:
    return {
        "instructions": export_dir / "REVIEW_INSTRUCTIONS.md",
        "label_template": export_dir / "label_template.jsonl",
        "review_items": export_dir / "review_items.jsonl",
    }


def _reviewed_export_artifact_paths(export_dir: Path) -> dict[str, Path]:
    return {
        "reviewed_dpo": export_dir / "reviewed_dpo.jsonl",
        "reviewed_labels": export_dir / "reviewed_labels.jsonl",
        "reviewed_preferences": export_dir / "reviewed_preferences.jsonl",
        "reviewed_reward_model": export_dir / "reviewed_reward_model.jsonl",
        "reviewed_sft": export_dir / "reviewed_sft.jsonl",
    }


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
    if manifest.get("confidence_options") != list(REVIEW_CONFIDENCE_LEVELS):
        target.errors.append(f"manifest.confidence_options must be {list(REVIEW_CONFIDENCE_LEVELS)!r}.")
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
        item_hash = item.get("review_item_sha256")
        if not _is_sha256(item_hash):
            target.errors.append(f"review_items[{index}].review_item_sha256 must be a SHA-256 hex string.")
        elif item_hash != review_item_sha256(item):
            target.errors.append(f"review_items[{index}].review_item_sha256 does not match review item contents.")
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
    item_by_id = {
        item.get("review_item_id"): item
        for item in items
        if isinstance(item.get("review_item_id"), str)
    }
    item_ids = set(item_by_id)
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
            item = None
        else:
            item = item_by_id[item_id]
        label_hash = label.get("review_item_sha256")
        if not _is_sha256(label_hash):
            target.errors.append(f"label_template[{index}].review_item_sha256 must be a SHA-256 hex string.")
        elif item is not None and label_hash != review_item_sha256(item):
            target.errors.append(f"label_template[{index}].review_item_sha256 does not match referenced review item.")
        if item is not None:
            for field_name in ("episode_id", "scenario_id", "suggested_human_label"):
                if label.get(field_name) != item.get(field_name):
                    target.errors.append(f"label_template[{index}].{field_name} does not match referenced review item.")
        suggested = label.get("suggested_human_label")
        if suggested not in REVIEW_LABELS:
            target.errors.append(f"label_template[{index}].suggested_human_label must be one of {list(REVIEW_LABELS)!r}.")
        human_label = label.get("human_label")
        if human_label is not None and human_label not in REVIEW_LABELS:
            target.errors.append(f"label_template[{index}].human_label must be null or one of {list(REVIEW_LABELS)!r}.")
        reviewer_confidence = label.get("reviewer_confidence")
        if human_label is not None and reviewer_confidence is None:
            target.errors.append(
                f"label_template[{index}].reviewer_confidence is required when human_label is set."
            )
        elif reviewer_confidence is not None and reviewer_confidence not in REVIEW_CONFIDENCE_LEVELS:
            target.errors.append(
                f"label_template[{index}].reviewer_confidence must be null or one of {list(REVIEW_CONFIDENCE_LEVELS)!r}."
            )
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
    dataset_registry: dict[str, Any] | None,
    expected_redaction_status: dict[str, Any],
    expected_label_provenance: dict[str, Any],
    export_dir: Path,
) -> None:
    _require_equal(manifest, "schema_version", REVIEWED_MANIFEST_SCHEMA_VERSION, target)
    if not _is_dataset_version(manifest.get("dataset_version")):
        target.errors.append("manifest.dataset_version must be a non-empty hfrds-* dataset selection key.")
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
    expected_confidence_counts = _reviewed_confidence_counts(labels)
    manifest_confidence_counts = manifest.get("confidence_counts")
    if not isinstance(manifest_confidence_counts, dict):
        target.errors.append("manifest.confidence_counts must be an object.")
    elif manifest_confidence_counts != expected_confidence_counts:
        target.errors.append(
            f"manifest.confidence_counts expected {expected_confidence_counts!r}, got {manifest_confidence_counts!r}."
        )
    expected_confidence_fields = {
        "high_confidence_label_count": expected_confidence_counts["high"],
        "medium_or_high_confidence_label_count": expected_confidence_counts["high"] + expected_confidence_counts["medium"],
        "low_confidence_label_count": expected_confidence_counts["low"],
        "unknown_confidence_label_count": expected_confidence_counts["unknown"],
    }
    for field_name, expected in expected_confidence_fields.items():
        if field_name not in manifest:
            target.errors.append(f"manifest.{field_name} is missing.")
        elif manifest.get(field_name) != expected:
            target.errors.append(f"manifest.{field_name} expected {expected}, got {manifest.get(field_name)!r}.")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        target.errors.append("manifest.outputs must be an object.")
    else:
        for output_name in (
            "reviewed_labels",
            "reviewed_sft",
            "reviewed_reward_model",
            "reviewed_preferences",
            "reviewed_dpo",
            "dataset_registry",
            "manifest",
        ):
            if output_name not in outputs:
                target.errors.append(f"manifest.outputs.{output_name} is missing.")
    if manifest.get("redaction_status") != expected_redaction_status:
        target.errors.append("manifest.redaction_status must match recomputed reviewed redaction scan.")
    if manifest.get("label_provenance") != expected_label_provenance:
        target.errors.append("manifest.label_provenance must match recomputed reviewed label provenance.")
    registry = manifest.get("registry")
    if not isinstance(registry, dict):
        target.errors.append("manifest.registry must be an object.")
    else:
        _require_equal(registry, "schema_version", RL_DATASET_REGISTRY_SCHEMA_VERSION, target, prefix="manifest.registry.")
        if registry.get("selection_key") != manifest.get("dataset_version"):
            target.errors.append("manifest.registry.selection_key must match manifest.dataset_version.")
        if registry.get("redaction_passed") is not (expected_redaction_status.get("passed") is True):
            target.errors.append("manifest.registry.redaction_passed must match redaction_status.passed.")
    if dataset_registry is None:
        target.errors.append("manifest has no validated dataset_registry.json companion.")
    else:
        _validate_reviewed_dataset_registry(
            dataset_registry,
            target,
            manifest,
            export_dir,
            expected_redaction_status,
            expected_label_provenance,
        )
    if _looks_absolute(str(manifest.get("source_review_export", ""))):
        target.warnings.append("manifest.source_review_export is absolute; prefer redacted or relative exports for sharing.")
    if _looks_absolute(str(manifest.get("labels_path", ""))):
        target.warnings.append("manifest.labels_path is absolute; prefer redacted or relative exports for sharing.")
    if _looks_absolute(str(manifest.get("output_dir", ""))):
        target.warnings.append("manifest.output_dir is absolute; prefer redacted or relative exports for sharing.")


def _expected_reviewed_label_provenance(
    labels: list[dict[str, Any]],
    sft: list[dict[str, Any]],
    reward_model: list[dict[str, Any]],
    preferences: list[dict[str, Any]],
    dpo: list[dict[str, Any]],
) -> dict[str, Any]:
    label_counts = _reviewed_label_counts(labels)
    return {
        "schema_version": RL_LABEL_PROVENANCE_SCHEMA_VERSION,
        "policy": "Completed human labels bound to review_item_sha256 drive reviewed trainer views.",
        "reviewed_label_count": len(labels),
        "accepted_label_count": label_counts.get("accept", 0),
        "negative_label_count": sum(label_counts.get(label, 0) for label in sorted(TRAINING_NEGATIVE_LABELS)),
        "needs_review_excluded_count": label_counts.get("needs_review", 0),
        "trainer_view_counts": {
            "reviewed_sft": len(sft),
            "reviewed_reward_model": len(reward_model),
            "reviewed_preferences": len(preferences),
            "reviewed_dpo": len(dpo),
        },
        "notes": [
            "Reviewed SFT rows require human_label='accept'.",
            "Reviewed reward and preference rows use accept/reject/unsafe/incomplete labels only.",
        ],
    }


def _validate_reviewed_dataset_registry(
    registry: dict[str, Any],
    target: ValidationTarget,
    manifest: dict[str, Any],
    export_dir: Path,
    expected_redaction_status: dict[str, Any],
    expected_label_provenance: dict[str, Any],
) -> None:
    _require_equal(registry, "schema_version", RL_DATASET_REGISTRY_SCHEMA_VERSION, target, prefix="dataset_registry.")
    if registry.get("artifact_type") != "reviewed_export":
        target.errors.append("dataset_registry.artifact_type must be 'reviewed_export'.")
    if registry.get("dataset_version") != manifest.get("dataset_version"):
        target.errors.append("dataset_registry.dataset_version must match manifest.dataset_version.")
    if registry.get("manifest_sha256") != _sha256(export_dir / "manifest.json"):
        target.errors.append("dataset_registry.manifest_sha256 must match manifest.json contents.")
    selection = registry.get("selection")
    if not isinstance(selection, dict):
        target.errors.append("dataset_registry.selection must be an object.")
    elif selection.get("key") != manifest.get("dataset_version"):
        target.errors.append("dataset_registry.selection.key must match manifest.dataset_version.")
    if registry.get("redaction_status") != expected_redaction_status:
        target.errors.append("dataset_registry.redaction_status must match recomputed reviewed redaction scan.")
    if registry.get("label_provenance") != expected_label_provenance:
        target.errors.append("dataset_registry.label_provenance must match recomputed reviewed label provenance.")
    if registry.get("source_review_artifacts") != manifest.get("source_review_artifacts"):
        target.errors.append("dataset_registry.source_review_artifacts must match manifest.source_review_artifacts.")
    if registry.get("labels_artifact") != manifest.get("labels_artifact"):
        target.errors.append("dataset_registry.labels_artifact must match manifest.labels_artifact.")
    if registry.get("artifact_fingerprints") != manifest.get("artifact_fingerprints"):
        target.errors.append("dataset_registry.artifact_fingerprints must match manifest.artifact_fingerprints.")


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
        if not _is_sha256(row.get("review_item_sha256")):
            target.errors.append(f"reviewed_labels[{index}].review_item_sha256 must be a SHA-256 hex string.")
        if not _is_sha256(row.get("source_label_sha256")):
            target.errors.append(f"reviewed_labels[{index}].source_label_sha256 must be a SHA-256 hex string.")
        if row.get("human_label") not in REVIEW_LABELS:
            target.errors.append(f"reviewed_labels[{index}].human_label must be one of {list(REVIEW_LABELS)!r}.")
        if row.get("suggested_human_label") is not None and row.get("suggested_human_label") not in REVIEW_LABELS:
            target.errors.append(f"reviewed_labels[{index}].suggested_human_label must be null or one of {list(REVIEW_LABELS)!r}.")
        if "reviewer_confidence" not in row:
            target.errors.append(f"reviewed_labels[{index}].reviewer_confidence is required.")
        elif row.get("reviewer_confidence") not in REVIEW_CONFIDENCE_LEVELS:
            target.errors.append(
                f"reviewed_labels[{index}].reviewer_confidence must be one of {list(REVIEW_CONFIDENCE_LEVELS)!r}."
            )
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
        if item is not None:
            _validate_review_item_hash_link(row, item, target, f"reviewed_sft[{index}]")
            _validate_review_confidence_link(row, item, target, f"reviewed_sft[{index}]")
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
        if item is not None:
            _validate_review_item_hash_link(row, item, target, f"reviewed_reward_model[{index}]")
            _validate_review_confidence_link(row, item, target, f"reviewed_reward_model[{index}]")
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
        else:
            _validate_pref_side_hash(row, chosen, "chosen", target, f"reviewed_preferences[{index}]")
            _validate_pref_side_confidence(row, chosen, "chosen", target, f"reviewed_preferences[{index}]")
        if rejected is None:
            target.errors.append(f"reviewed_preferences[{index}].rejected_episode_id does not reference a reviewed label.")
        elif rejected.get("human_label") not in {"reject", "unsafe", "incomplete"}:
            target.errors.append(f"reviewed_preferences[{index}].rejected_episode_id must reference a rejected/unsafe/incomplete label.")
        else:
            _validate_pref_side_hash(row, rejected, "rejected", target, f"reviewed_preferences[{index}]")
            _validate_pref_side_confidence(row, rejected, "rejected", target, f"reviewed_preferences[{index}]")
        if row.get("source_artifact") != "reviewed_labels.jsonl":
            target.errors.append(f"reviewed_preferences[{index}].source_artifact must be 'reviewed_labels.jsonl'.")


def _validate_reviewed_dpo(dpo: list[dict[str, Any]], target: ValidationTarget, preferences: list[dict[str, Any]]) -> None:
    preference_by_id = {
        row.get("preference_id"): row
        for row in preferences
        if isinstance(row.get("preference_id"), str)
    }
    for index, row in enumerate(dpo):
        _require_equal(row, "schema_version", REVIEWED_DPO_SCHEMA_VERSION, target, prefix=f"reviewed_dpo[{index}].")
        preference = preference_by_id.get(row.get("preference_id"))
        if preference is None:
            target.errors.append(f"reviewed_dpo[{index}].preference_id does not reference a reviewed preference.")
        for side in ("chosen", "rejected"):
            field_name = f"{side}_review_item_sha256"
            if not _is_sha256(row.get(field_name)):
                target.errors.append(f"reviewed_dpo[{index}].{field_name} must be a SHA-256 hex string.")
            elif preference is not None:
                if row.get(field_name) != preference.get(field_name):
                    target.errors.append(f"reviewed_dpo[{index}].{field_name} does not match reviewed preference.")
            confidence_field = f"{side}_reviewer_confidence"
            if confidence_field not in row:
                target.errors.append(f"reviewed_dpo[{index}].{confidence_field} is required.")
            else:
                _validate_confidence_value(row.get(confidence_field), target, f"reviewed_dpo[{index}].{confidence_field}")
                if preference is not None and row.get(confidence_field) != preference.get(confidence_field):
                    target.errors.append(f"reviewed_dpo[{index}].{confidence_field} does not match reviewed preference.")
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


def _validate_review_item_hash_link(
    row: dict[str, Any],
    item: dict[str, Any],
    target: ValidationTarget,
    label: str,
) -> None:
    if not _is_sha256(row.get("review_item_sha256")):
        target.errors.append(f"{label}.review_item_sha256 must be a SHA-256 hex string.")
    elif row.get("review_item_sha256") != item.get("review_item_sha256"):
        target.errors.append(f"{label}.review_item_sha256 does not match reviewed label.")


def _validate_review_confidence_link(
    row: dict[str, Any],
    item: dict[str, Any],
    target: ValidationTarget,
    label: str,
) -> None:
    if "reviewer_confidence" not in row:
        target.errors.append(f"{label}.reviewer_confidence is required.")
        return
    _validate_confidence_value(row.get("reviewer_confidence"), target, f"{label}.reviewer_confidence")
    expected = item.get("reviewer_confidence", "unknown")
    if row.get("reviewer_confidence") != expected:
        target.errors.append(f"{label}.reviewer_confidence does not match reviewed label.")


def _validate_pref_side_hash(
    row: dict[str, Any],
    item: dict[str, Any],
    side: str,
    target: ValidationTarget,
    label: str,
) -> None:
    field_name = f"{side}_review_item_sha256"
    if not _is_sha256(row.get(field_name)):
        target.errors.append(f"{label}.{field_name} must be a SHA-256 hex string.")
    elif row.get(field_name) != item.get("review_item_sha256"):
        target.errors.append(f"{label}.{field_name} does not match reviewed label.")
    nested = row.get(side)
    if isinstance(nested, dict):
        nested_hash = nested.get("review_item_sha256")
        if not _is_sha256(nested_hash):
            target.errors.append(f"{label}.{side}.review_item_sha256 must be a SHA-256 hex string.")
        elif nested_hash != item.get("review_item_sha256"):
            target.errors.append(f"{label}.{side}.review_item_sha256 does not match reviewed label.")


def _validate_pref_side_confidence(
    row: dict[str, Any],
    item: dict[str, Any],
    side: str,
    target: ValidationTarget,
    label: str,
) -> None:
    expected = item.get("reviewer_confidence", "unknown")
    field_name = f"{side}_reviewer_confidence"
    if field_name not in row:
        target.errors.append(f"{label}.{field_name} is required.")
    else:
        _validate_confidence_value(row.get(field_name), target, f"{label}.{field_name}")
        if row.get(field_name) != expected:
            target.errors.append(f"{label}.{field_name} does not match reviewed label.")
    nested = row.get(side)
    if isinstance(nested, dict):
        if "reviewer_confidence" not in nested:
            target.errors.append(f"{label}.{side}.reviewer_confidence is required.")
        else:
            _validate_confidence_value(nested.get("reviewer_confidence"), target, f"{label}.{side}.reviewer_confidence")
            if nested.get("reviewer_confidence") != expected:
                target.errors.append(f"{label}.{side}.reviewer_confidence does not match reviewed label.")


def _validate_confidence_value(value: Any, target: ValidationTarget, label: str) -> None:
    if value not in REVIEW_CONFIDENCE_LEVELS:
        target.errors.append(f"{label} must be one of {list(REVIEW_CONFIDENCE_LEVELS)!r}.")


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


def _reviewed_confidence_counts(labels: list[dict[str, Any]]) -> dict[str, int]:
    counts = {level: 0 for level in REVIEW_CONFIDENCE_LEVELS}
    for row in labels:
        confidence = row.get("reviewer_confidence")
        if confidence not in REVIEW_CONFIDENCE_LEVELS:
            confidence = "unknown"
        counts[str(confidence)] += 1
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
        if "trace_signal" in episode:
            _validate_trace_signal(
                episode.get("trace_signal"),
                _expected_episode_trace_signal(episode),
                target,
                f"episodes[{index}].trace_signal",
            )
        else:
            target.warnings.append(f"episodes[{index}].trace_signal is missing; rerun export-rl to refresh trace-signal metrics.")
        if "task_completion" in episode:
            _validate_task_completion(episode.get("task_completion"), target, f"episodes[{index}].task_completion")
        else:
            target.warnings.append(f"episodes[{index}].task_completion is missing; rerun export-rl to refresh task evidence fields.")
        if "state_diff" in episode:
            _validate_state_diff_summary(episode.get("state_diff"), target, f"episodes[{index}].state_diff")
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
        if isinstance(episode.get("state_diff"), dict):
            if outcome.get("state_changed") != episode["state_diff"].get("changed"):
                target.errors.append(f"episodes[{index}].outcome.state_changed must match state_diff.changed.")
            if outcome.get("state_change_count") != episode["state_diff"].get("change_count"):
                target.errors.append(f"episodes[{index}].outcome.state_change_count must match state_diff.change_count.")


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
            if "state_changed" in reward and reward.get("state_changed") != outcome.get("state_changed"):
                target.errors.append(f"rewards[{index}].state_changed does not match episode {episode_id!r} outcome.")
            if "state_change_count" in reward and reward.get("state_change_count") != outcome.get("state_change_count"):
                target.errors.append(f"rewards[{index}].state_change_count does not match episode {episode_id!r} outcome.")
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
        and positive_label_eligible(episode)
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
            if episode is not None and not positive_label_eligible(episode):
                target.errors.append(f"sft[{index}].episode_id {episode_id!r} is not eligible for positive trainer labels.")
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
    expected_ids = {
        str(episode.get("episode_id"))
        for episode in episodes
        if isinstance(episode.get("episode_id"), str)
        and (
            not (isinstance(episode.get("outcome"), dict) and episode["outcome"].get("passed") is True)
            or positive_label_eligible(episode)
        )
    }
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
            outcome = episode.get("outcome") if isinstance(episode.get("outcome"), dict) else {}
            if outcome.get("passed") is True and not positive_label_eligible(episode):
                target.errors.append(f"reward_model[{index}].episode_id {episode_id!r} is not eligible as a positive reward label.")
            _validate_training_view_task_completion(sample, episode, target, f"reward_model[{index}]")
        for field_name in ("failed_rules", "critical_failures"):
            if not _is_string_list(sample.get(field_name)):
                target.errors.append(f"reward_model[{index}].{field_name} must be a list of strings.")
    missing = sorted(expected_ids - seen)
    if missing:
        target.errors.append(f"reward_model.jsonl missing eligible episode samples: {missing!r}.")


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


def _validate_trainer_view_source_fingerprint_coverage(
    value: Any,
    target: ValidationTarget,
    sft: list[dict[str, Any]],
    dpo: list[dict[str, Any]],
    reward_model: list[dict[str, Any]],
) -> None:
    if not isinstance(value, dict):
        target.errors.append("dataset_metrics.trainer_view_source_fingerprint_coverage must be an object.")
        return
    expected = _expected_trainer_view_source_fingerprint_coverage(sft, dpo, reward_model)
    for field_name, expected_value in expected.items():
        if value.get(field_name) != expected_value:
            target.errors.append(
                "dataset_metrics.trainer_view_source_fingerprint_coverage."
                f"{field_name} expected {expected_value!r}, got {value.get(field_name)!r}."
            )


def _expected_trainer_view_source_fingerprint_coverage(
    sft: list[dict[str, Any]],
    dpo: list[dict[str, Any]],
    reward_model: list[dict[str, Any]],
) -> dict[str, Any]:
    sft_verified = sum(1 for row in sft if _row_source_fingerprints_verified(row))
    dpo_verified = sum(1 for row in dpo if _dpo_source_fingerprints_verified(row))
    reward_model_verified = sum(1 for row in reward_model if _row_source_fingerprints_verified(row))
    row_count = len(sft) + len(dpo) + len(reward_model)
    fully_verified = sft_verified + dpo_verified + reward_model_verified
    return {
        "rows": row_count,
        "sft_rows": len(sft),
        "dpo_rows": len(dpo),
        "reward_model_rows": len(reward_model),
        "fully_verified": fully_verified,
        "unverified": row_count - fully_verified,
        "fully_verified_rate": round(fully_verified / row_count, 4) if row_count else 0.0,
    }


def _row_source_fingerprints_verified(row: dict[str, Any]) -> bool:
    fingerprints = row.get("source_fingerprints") if isinstance(row.get("source_fingerprints"), dict) else {}
    return (
        row.get("source_fingerprint_status") == "verified"
        and _fingerprints_have_source_hashes(fingerprints)
    )


def _dpo_source_fingerprints_verified(row: dict[str, Any]) -> bool:
    return _paired_source_fingerprints_verified(row, "chosen") and _paired_source_fingerprints_verified(row, "rejected")


def _paired_source_fingerprints_verified(row: dict[str, Any], side: str) -> bool:
    fingerprints_key = f"{side}_source_fingerprints"
    status_key = f"{side}_source_fingerprint_status"
    fingerprints = row.get(fingerprints_key) if isinstance(row.get(fingerprints_key), dict) else {}
    return row.get(status_key) == "verified" and _fingerprints_have_source_hashes(fingerprints)


def _fingerprints_have_source_hashes(fingerprints: dict[str, Any]) -> bool:
    scenario = fingerprints.get("scenario") if isinstance(fingerprints.get("scenario"), dict) else {}
    source_trace = fingerprints.get("source_trace") if isinstance(fingerprints.get("source_trace"), dict) else {}
    return _is_sha256(scenario.get("sha256")) and _is_sha256(source_trace.get("sha256"))


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
    dataset_splits: dict[str, Any] | None,
    expected_redaction_status: dict[str, Any],
    expected_label_provenance: dict[str, Any],
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
    if "trainer_view_source_fingerprint_coverage" in metrics:
        _validate_trainer_view_source_fingerprint_coverage(
            metrics.get("trainer_view_source_fingerprint_coverage"),
            target,
            sft,
            dpo,
            reward_model,
        )
    else:
        target.warnings.append(
            "dataset_metrics.trainer_view_source_fingerprint_coverage is missing; rerun export-rl to refresh trainer-view provenance metrics."
        )
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
    if "trace_signal" in metrics:
        _validate_trace_signal_metrics(
            metrics.get("trace_signal"),
            _expected_trace_signal_metrics(episodes),
            target,
            "dataset_metrics.trace_signal",
        )
    else:
        target.warnings.append("dataset_metrics.trace_signal is missing; rerun export-rl to refresh trace-signal metrics.")
    if "dataset_splits" in metrics:
        if not isinstance(metrics.get("dataset_splits"), dict):
            target.errors.append("dataset_metrics.dataset_splits must be an object.")
        elif dataset_splits is not None and metrics.get("dataset_splits") != dataset_splits.get("summary"):
            target.errors.append("dataset_metrics.dataset_splits must match dataset_splits.summary.")
    else:
        target.warnings.append("dataset_metrics.dataset_splits is missing; rerun export-rl to refresh split metrics.")
    if "redaction_status" in metrics:
        if metrics.get("redaction_status") != expected_redaction_status:
            target.errors.append("dataset_metrics.redaction_status must match recomputed redaction scan.")
    else:
        target.warnings.append("dataset_metrics.redaction_status is missing; rerun export-rl to emit redaction proof.")
    if "label_provenance" in metrics:
        if metrics.get("label_provenance") != expected_label_provenance:
            target.errors.append("dataset_metrics.label_provenance must match recomputed label provenance.")
    else:
        target.warnings.append("dataset_metrics.label_provenance is missing; rerun export-rl to emit label provenance.")
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
        trace_metrics = _expected_trace_signal_metrics(family_episodes)
        expected[family] = {
            "task_family": family,
            "episode_count": len(family_episodes),
            "passed": passed,
            "failed": len(family_episodes) - passed,
            "pass_rate": round(passed / len(family_episodes), 4) if family_episodes else 0.0,
            "task_completion_configured": task_metrics["configured_count"],
            "task_completion_complete": task_metrics["complete_count"],
            "task_completion_incomplete": task_metrics["incomplete_count"],
            "trace_average_event_count": trace_metrics["average_event_count"],
            "trace_event_type_count": trace_metrics["event_type_count"],
            "trace_tool_or_api_episode_rate": trace_metrics["tool_or_api_episode_rate"],
            "trace_empty_final_answer_count": trace_metrics["empty_final_answer_count"],
            "trace_risk_count": trace_metrics["risk_count"],
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


def _expected_episode_trace_signal(episode: dict[str, Any]) -> dict[str, Any]:
    events = episode.get("events") if isinstance(episode.get("events"), list) else []
    final_answer = episode.get("final_answer") if isinstance(episode.get("final_answer"), str) else ""
    return build_trace_signal(events, final_answer)


def _validate_trace_signal(value: Any, expected: dict[str, Any], target: ValidationTarget, label: str) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
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
        if value.get(field_name) != expected[field_name]:
            target.errors.append(f"{label}.{field_name} expected {expected[field_name]!r}, got {value.get(field_name)!r}.")
    for field_name in ("has_final_answer", "has_tool_or_api_events"):
        if value.get(field_name) != expected[field_name]:
            target.errors.append(f"{label}.{field_name} expected {expected[field_name]!r}, got {value.get(field_name)!r}.")
    if _count_rows(value.get("event_types")) != _count_rows(expected.get("event_types")):
        target.errors.append(f"{label}.event_types does not match episode events.")
    if value.get("risks") != expected.get("risks"):
        target.errors.append(f"{label}.risks expected {expected.get('risks')!r}, got {value.get('risks')!r}.")


def _expected_trace_signal_metrics(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    signals = [_expected_episode_trace_signal(episode) for episode in episodes]
    event_counts = [_non_negative_int_value(signal.get("event_count")) for signal in signals]
    event_type_counts: dict[str, int] = {}
    risk_counts: dict[str, int] = {}
    for signal in signals:
        _merge_count_rows_quiet(event_type_counts, signal.get("event_types"))
        for risk in signal.get("risks", []):
            if isinstance(risk, str) and risk:
                risk_counts[risk] = risk_counts.get(risk, 0) + 1
    episode_count = len(signals)
    with_final = sum(1 for signal in signals if signal.get("has_final_answer") is True)
    with_tool_or_api = sum(1 for signal in signals if signal.get("has_tool_or_api_events") is True)
    return {
        "episode_count": episode_count,
        "total_event_count": sum(event_counts),
        "average_event_count": round(sum(event_counts) / episode_count, 2) if episode_count else 0.0,
        "min_event_count": min(event_counts) if event_counts else 0,
        "max_event_count": max(event_counts) if event_counts else 0,
        "event_type_count": len(event_type_counts),
        "event_type_counts": event_type_counts,
        "episodes_with_final_answer": with_final,
        "empty_final_answer_count": episode_count - with_final,
        "final_answer_rate": round(with_final / episode_count, 4) if episode_count else 0.0,
        "episodes_with_tool_or_api_events": with_tool_or_api,
        "tool_or_api_episode_rate": round(with_tool_or_api / episode_count, 4) if episode_count else 0.0,
        "tool_call_count": sum(_non_negative_int_value(signal.get("tool_call_count")) for signal in signals),
        "tool_result_count": sum(_non_negative_int_value(signal.get("tool_result_count")) for signal in signals),
        "api_call_count": sum(_non_negative_int_value(signal.get("api_call_count")) for signal in signals),
        "subagent_event_count": sum(_non_negative_int_value(signal.get("subagent_event_count")) for signal in signals),
        "approval_event_count": sum(_non_negative_int_value(signal.get("approval_event_count")) for signal in signals),
        "risk_count": sum(risk_counts.values()),
        "risk_counts": risk_counts,
    }


def _validate_trace_signal_metrics(value: Any, expected: dict[str, Any], target: ValidationTarget, label: str) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    for field_name in (
        "episode_count",
        "total_event_count",
        "average_event_count",
        "min_event_count",
        "max_event_count",
        "event_type_count",
        "episodes_with_final_answer",
        "empty_final_answer_count",
        "final_answer_rate",
        "episodes_with_tool_or_api_events",
        "tool_or_api_episode_rate",
        "tool_call_count",
        "tool_result_count",
        "api_call_count",
        "subagent_event_count",
        "approval_event_count",
        "risk_count",
    ):
        if value.get(field_name) != expected[field_name]:
            target.errors.append(f"{label}.{field_name} expected {expected[field_name]!r}, got {value.get(field_name)!r}.")
    if _count_rows(value.get("event_type_counts")) != expected["event_type_counts"]:
        target.errors.append(f"{label}.event_type_counts does not match episode trace_signal.")
    if _count_rows(value.get("risk_counts")) != expected["risk_counts"]:
        target.errors.append(f"{label}.risk_counts does not match episode trace_signal.")


def _merge_count_rows_quiet(counts: dict[str, int], rows: Any) -> None:
    if not isinstance(rows, list):
        return
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("id"), str) and isinstance(row.get("count"), int):
            counts[row["id"]] = counts.get(row["id"], 0) + row["count"]


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
        "## Trace Signal",
        "## Dataset Splits",
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
        elif not positive_label_eligible(chosen):
            target.errors.append(f"preferences[{index}].chosen_episode_id {chosen_id!r} is not eligible for a positive preference label.")
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
            if not isinstance(mode.get("critical_count"), int) or isinstance(mode.get("critical_count"), bool) or mode.get("critical_count") < 0:
                target.errors.append(
                    f"curriculum.task_families[{family_index}].failure_modes[{mode_index}].critical_count must be a non-negative integer."
                )
            if not isinstance(mode.get("max_penalty"), int) or isinstance(mode.get("max_penalty"), bool) or mode.get("max_penalty") < 0:
                target.errors.append(
                    f"curriculum.task_families[{family_index}].failure_modes[{mode_index}].max_penalty must be a non-negative integer."
                )
            if not isinstance(mode.get("average_penalty"), (int, float)) or isinstance(mode.get("average_penalty"), bool):
                target.errors.append(
                    f"curriculum.task_families[{family_index}].failure_modes[{mode_index}].average_penalty must be numeric."
                )
            if not isinstance(mode.get("priority_score"), int) or isinstance(mode.get("priority_score"), bool) or mode.get("priority_score") < 0:
                target.errors.append(
                    f"curriculum.task_families[{family_index}].failure_modes[{mode_index}].priority_score must be a non-negative integer."
                )
            elif _is_non_negative_int(mode.get("count")) and _is_non_negative_int(mode.get("critical_count")) and _is_non_negative_int(mode.get("max_penalty")):
                expected_priority = int(mode["count"]) * 10 + int(mode["critical_count"]) * 100 + int(mode["max_penalty"])
                if mode.get("priority_score") != expected_priority:
                    target.errors.append(
                        f"curriculum.task_families[{family_index}].failure_modes[{mode_index}].priority_score expected "
                        f"{expected_priority}, got {mode.get('priority_score')!r}."
                    )
            if mode.get("priority_band") not in {"critical", "high", "medium", "low"}:
                target.errors.append(
                    f"curriculum.task_families[{family_index}].failure_modes[{mode_index}].priority_band must be critical, high, medium, or low."
                )
            elif _is_non_negative_int(mode.get("priority_score")):
                expected_band = _expected_curriculum_priority_band(int(mode["priority_score"]))
                if mode.get("priority_band") != expected_band:
                    target.errors.append(
                        f"curriculum.task_families[{family_index}].failure_modes[{mode_index}].priority_band expected "
                        f"{expected_band!r}, got {mode.get('priority_band')!r}."
                    )
            if not isinstance(mode.get("episode_ids"), list):
                target.errors.append(
                    f"curriculum.task_families[{family_index}].failure_modes[{mode_index}].episode_ids must be a list."
                )
            if not isinstance(mode.get("scenario_ids"), list):
                target.errors.append(
                    f"curriculum.task_families[{family_index}].failure_modes[{mode_index}].scenario_ids must be a list."
                )
            if not isinstance(mode.get("failure_ids"), list):
                target.errors.append(
                    f"curriculum.task_families[{family_index}].failure_modes[{mode_index}].failure_ids must be a list."
                )
            if not isinstance(mode.get("example_evidence"), list):
                target.errors.append(
                    f"curriculum.task_families[{family_index}].failure_modes[{mode_index}].example_evidence must be a list."
                )
            _validate_evidence_refs(
                mode.get("example_evidence_refs"),
                target,
                f"curriculum.task_families[{family_index}].failure_modes[{mode_index}].example_evidence_refs",
            )


def _expected_curriculum_priority_band(priority_score: int) -> str:
    if priority_score >= 150:
        return "critical"
    if priority_score >= 75:
        return "high"
    if priority_score >= 25:
        return "medium"
    return "low"


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


def _validate_improvement_plan(plan: dict[str, Any], target: ValidationTarget) -> None:
    _require_equal(plan, "schema_version", IMPROVEMENT_PLAN_SCHEMA_VERSION, target)
    if not isinstance(plan.get("plan_path"), str) or not plan.get("plan_path"):
        target.errors.append("improvement_plan.plan_path must be a non-empty string.")
    if plan.get("passed") is not True:
        target.errors.append("improvement_plan.passed must be true.")
    if plan.get("readiness") not in {"ready", "blocked"}:
        target.errors.append("improvement_plan.readiness must be ready or blocked.")

    source_artifacts = plan.get("source_artifacts")
    if not isinstance(source_artifacts, dict):
        target.errors.append("improvement_plan.source_artifacts must be an object.")
        source_artifacts = {}
    if "evidence_bundle" not in source_artifacts:
        target.errors.append("improvement_plan.source_artifacts.evidence_bundle is required.")
    for name, record in source_artifacts.items():
        _validate_improvement_source_artifact(name, record, target)

    work_items = plan.get("work_items")
    if not isinstance(work_items, list):
        target.errors.append("improvement_plan.work_items must be a list.")
        work_items = []
    if plan.get("work_item_count") != len(work_items):
        target.errors.append(f"improvement_plan.work_item_count expected {len(work_items)}, got {plan.get('work_item_count')!r}.")

    totals: dict[str, Any] = {
        "priority_counts": {},
        "category_counts": {},
        "task_family_counts": {},
        "rule_counts": {},
        "scenarios": set(),
        "task_families": set(),
        "rules": set(),
        "repair_backed_count": 0,
        "curriculum_backed_count": 0,
        "digest_backed_count": 0,
        "bundle_action_count": 0,
        "evidence_ref_count": 0,
    }
    seen_item_ids: set[str] = set()
    seen_routing_keys: set[str] = set()
    seen_fingerprints: set[str] = set()
    previous_sort_key: tuple[int, int, str, str, str, str] | None = None
    for index, item in enumerate(work_items):
        sort_key = _validate_improvement_work_item(
            item,
            target,
            f"improvement_plan.work_items[{index}]",
            seen_item_ids,
            seen_routing_keys,
            seen_fingerprints,
            totals,
        )
        if sort_key is not None:
            if previous_sort_key is not None and sort_key < previous_sort_key:
                target.errors.append("improvement_plan.work_items must be sorted by priority, category, task family, scenario, rule, and summary.")
            previous_sort_key = sort_key
    _validate_improvement_metrics(plan.get("metrics"), target, totals, len(work_items))
    _validate_improvement_decision(plan.get("decision"), target, plan.get("readiness"), len(work_items), totals)
    if "notes" in plan and not _is_string_list(plan.get("notes")):
        target.errors.append("improvement_plan.notes must be a list of strings when present.")
    target.details.update(
        {
            "readiness": plan.get("readiness"),
            "work_item_count": len(work_items),
            "critical_or_high_count": _count_value(totals["priority_counts"], "critical") + _count_value(totals["priority_counts"], "high"),
        }
    )


def _validate_improvement_source_artifact(name: Any, record: Any, target: ValidationTarget) -> None:
    if not isinstance(name, str) or not name:
        target.errors.append("improvement_plan.source_artifacts keys must be non-empty strings.")
    label = f"improvement_plan.source_artifacts.{name}"
    if not isinstance(record, dict):
        target.errors.append(f"{label} must be an object.")
        return
    for field_name in ("kind", "path", "exists"):
        if field_name not in record:
            target.errors.append(f"{label}.{field_name} is required.")
    if record.get("kind") not in {"file", "directory"}:
        target.errors.append(f"{label}.kind must be file or directory.")
    if not isinstance(record.get("path"), str) or not record.get("path"):
        target.errors.append(f"{label}.path must be a non-empty string.")
    if not isinstance(record.get("exists"), bool):
        target.errors.append(f"{label}.exists must be a boolean.")
    if record.get("kind") == "file" and record.get("exists") is True:
        sha = record.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64 or sha != sha.lower() or any(char not in "0123456789abcdef" for char in sha):
            target.errors.append(f"{label}.sha256 must be a lowercase 64-character hex digest for existing files.")
        if not _is_non_negative_int(record.get("size_bytes")):
            target.errors.append(f"{label}.size_bytes must be a non-negative integer for existing files.")
    if record.get("kind") == "directory" and record.get("exists") is True and not _is_non_negative_int(record.get("entry_count")):
        target.errors.append(f"{label}.entry_count must be a non-negative integer for existing directories.")
    if "schema_version" in record and record.get("schema_version") is not None and not isinstance(record.get("schema_version"), str):
        target.errors.append(f"{label}.schema_version must be a string or null.")
    if "passed" in record and record.get("passed") is not None and not isinstance(record.get("passed"), bool):
        target.errors.append(f"{label}.passed must be a boolean or null.")


def _validate_improvement_work_item(
    item: Any,
    target: ValidationTarget,
    label: str,
    seen_item_ids: set[str],
    seen_routing_keys: set[str],
    seen_fingerprints: set[str],
    totals: dict[str, Any],
) -> tuple[int, int, str, str, str, str] | None:
    if not isinstance(item, dict):
        target.errors.append(f"{label} must be an object.")
        return None
    for field_name in ("item_id", "category", "priority", "summary", "suggested_action", "fingerprint", "routing_key"):
        if not isinstance(item.get(field_name), str) or not item.get(field_name):
            target.errors.append(f"{label}.{field_name} must be a non-empty string.")
    item_id = item.get("item_id")
    if isinstance(item_id, str) and item_id:
        if item_id in seen_item_ids:
            target.errors.append(f"{label}.item_id duplicates {item_id!r}.")
        seen_item_ids.add(item_id)
    routing_key = item.get("routing_key")
    if isinstance(routing_key, str) and routing_key:
        if routing_key in seen_routing_keys:
            target.errors.append(f"{label}.routing_key duplicates {routing_key!r}.")
        seen_routing_keys.add(routing_key)
    category = item.get("category")
    if category not in {"bundle_action", "repair", "curriculum", "digest_action"}:
        target.errors.append(f"{label}.category must be bundle_action, repair, curriculum, or digest_action.")
        category = "digest_action"
    priority = item.get("priority")
    if priority not in set(PRIORITIES):
        target.errors.append(f"{label}.priority must be critical, high, medium, or low.")
        priority = "low"
    priority_rank = item.get("priority_rank")
    expected_rank = list(PRIORITIES).index(priority)
    if priority_rank != expected_rank:
        target.errors.append(f"{label}.priority_rank expected {expected_rank}, got {priority_rank!r}.")
    for field_name in ("scenario_id", "task_family", "rule_id", "rule_name", "task_completion_status"):
        if item.get(field_name) is not None and not isinstance(item.get(field_name), str):
            target.errors.append(f"{label}.{field_name} must be a string or null.")
    if item.get("score") is not None and not _is_int_between(item.get("score"), 0, 100):
        target.errors.append(f"{label}.score must be null or an integer from 0 to 100.")
    if not isinstance(item.get("sources"), dict):
        target.errors.append(f"{label}.sources must be an object.")
    else:
        _validate_improvement_item_sources(item["sources"], target, f"{label}.sources")
    _validate_evidence_refs(item.get("evidence_refs"), target, f"{label}.evidence_refs")
    if not isinstance(item.get("evidence_snippets"), list):
        target.errors.append(f"{label}.evidence_snippets must be a list.")
    if not isinstance(item.get("source_artifacts"), dict):
        target.errors.append(f"{label}.source_artifacts must be an object.")
    if not isinstance(item.get("replay"), dict):
        target.errors.append(f"{label}.replay must be an object.")

    fingerprint = item.get("fingerprint")
    expected_fingerprint = work_item_fingerprint(item)
    if not isinstance(fingerprint, str) or len(fingerprint) != 64 or fingerprint != fingerprint.lower() or any(
        char not in "0123456789abcdef" for char in fingerprint
    ):
        target.errors.append(f"{label}.fingerprint must be a lowercase 64-character hex digest.")
    elif fingerprint != expected_fingerprint:
        target.errors.append(f"{label}.fingerprint does not match item content.")
    elif fingerprint in seen_fingerprints:
        target.errors.append(f"{label}.fingerprint duplicates {fingerprint!r}.")
    elif isinstance(fingerprint, str):
        seen_fingerprints.add(fingerprint)
    if isinstance(fingerprint, str) and isinstance(item_id, str) and item_id != f"{category}:{fingerprint[:16]}":
        target.errors.append(f"{label}.item_id must match category and fingerprint.")
    if isinstance(fingerprint, str) and isinstance(routing_key, str) and routing_key != f"{category}:{priority}:{fingerprint[:12]}":
        target.errors.append(f"{label}.routing_key must match category, priority, and fingerprint.")

    _increment_count(totals["priority_counts"], priority)
    _increment_count(totals["category_counts"], category)
    _add_total(totals["scenarios"], item.get("scenario_id"))
    _add_total(totals["task_families"], item.get("task_family"))
    _add_total(totals["rules"], item.get("rule_id"))
    _increment_count(totals["task_family_counts"], item.get("task_family"))
    _increment_count(totals["rule_counts"], item.get("rule_id"))
    sources = item.get("sources") if isinstance(item.get("sources"), dict) else {}
    if _list_present(sources.get("repair_item_ids")):
        totals["repair_backed_count"] += 1
    if _list_present(sources.get("curriculum_priorities")):
        totals["curriculum_backed_count"] += 1
    if isinstance(sources.get("run_digest"), dict):
        totals["digest_backed_count"] += 1
    if category == "bundle_action":
        totals["bundle_action_count"] += 1
    evidence_refs = item.get("evidence_refs") if isinstance(item.get("evidence_refs"), list) else []
    totals["evidence_ref_count"] += len(evidence_refs)
    return (
        expected_rank,
        {"bundle_action": 0, "repair": 1, "curriculum": 2, "digest_action": 3}.get(str(category), 99),
        str(item.get("task_family") or ""),
        str(item.get("scenario_id") or ""),
        str(item.get("rule_id") or ""),
        str(item.get("summary") or ""),
    )


def _validate_improvement_item_sources(value: dict[str, Any], target: ValidationTarget, label: str) -> None:
    for field_name in ("bundle_action_ids", "repair_item_ids", "curriculum_priorities"):
        if not isinstance(value.get(field_name), list):
            target.errors.append(f"{label}.{field_name} must be a list.")
    if value.get("run_digest") is not None and not isinstance(value.get("run_digest"), dict):
        target.errors.append(f"{label}.run_digest must be an object or null.")
    for field_name in ("bundle_action_ids", "repair_item_ids"):
        if isinstance(value.get(field_name), list) and not all(isinstance(item, str) and item for item in value[field_name]):
            target.errors.append(f"{label}.{field_name} must contain non-empty strings.")
    if isinstance(value.get("curriculum_priorities"), list):
        for index, priority in enumerate(value["curriculum_priorities"]):
            priority_label = f"{label}.curriculum_priorities[{index}]"
            if not isinstance(priority, dict):
                target.errors.append(f"{priority_label} must be an object.")
                continue
            for field_name in ("task_family", "rule_id", "rule_name", "priority_band"):
                if not isinstance(priority.get(field_name), str) or not priority.get(field_name):
                    target.errors.append(f"{priority_label}.{field_name} must be a non-empty string.")
            for field_name in ("priority_score", "count", "critical_count", "max_penalty"):
                if not _is_non_negative_int(priority.get(field_name)):
                    target.errors.append(f"{priority_label}.{field_name} must be a non-negative integer.")


def _validate_improvement_metrics(metrics: Any, target: ValidationTarget, totals: dict[str, Any], work_item_count: int) -> None:
    if not isinstance(metrics, dict):
        target.errors.append("improvement_plan.metrics must be an object.")
        return
    expected_scalars = {
        "work_item_count": work_item_count,
        "scenario_count": len(totals["scenarios"]),
        "task_family_count": len(totals["task_families"]),
        "rule_count": len(totals["rules"]),
        "repair_backed_count": totals["repair_backed_count"],
        "curriculum_backed_count": totals["curriculum_backed_count"],
        "digest_backed_count": totals["digest_backed_count"],
        "bundle_action_count": totals["bundle_action_count"],
        "evidence_ref_count": totals["evidence_ref_count"],
    }
    for field_name, expected in expected_scalars.items():
        if metrics.get(field_name) != expected:
            target.errors.append(f"improvement_plan.metrics.{field_name} expected {expected!r}, got {metrics.get(field_name)!r}.")
    expected_lists = {
        "scenarios": sorted(totals["scenarios"]),
        "task_families": sorted(totals["task_families"]),
        "rules": sorted(totals["rules"]),
    }
    for field_name, expected in expected_lists.items():
        if metrics.get(field_name) != expected:
            target.errors.append(f"improvement_plan.metrics.{field_name} expected {expected!r}, got {metrics.get(field_name)!r}.")
    for field_name in ("priority_counts", "category_counts", "task_family_counts", "rule_counts"):
        actual = _count_rows(metrics.get(field_name))
        if actual != totals[field_name]:
            target.errors.append(f"improvement_plan.metrics.{field_name} does not match work items.")


def _validate_improvement_decision(
    decision: Any,
    target: ValidationTarget,
    readiness: Any,
    work_item_count: int,
    totals: dict[str, Any],
) -> None:
    if not isinstance(decision, dict):
        target.errors.append("improvement_plan.decision must be an object.")
        return
    if decision.get("readiness") != readiness:
        target.errors.append("improvement_plan.decision.readiness must match improvement_plan.readiness.")
    if decision.get("recommendation") not in {"fix_handoff", "run_improvement_iteration", "review_improvement_opportunities", "promote_or_monitor"}:
        target.errors.append("improvement_plan.decision.recommendation has an unknown value.")
    if not isinstance(decision.get("summary"), str) or not decision.get("summary"):
        target.errors.append("improvement_plan.decision.summary must be a non-empty string.")
    if decision.get("work_item_count") != work_item_count:
        target.errors.append(f"improvement_plan.decision.work_item_count expected {work_item_count}, got {decision.get('work_item_count')!r}.")
    critical_or_high = _count_value(totals["priority_counts"], "critical") + _count_value(totals["priority_counts"], "high")
    if decision.get("critical_or_high_count") != critical_or_high:
        target.errors.append(
            f"improvement_plan.decision.critical_or_high_count expected {critical_or_high}, got {decision.get('critical_or_high_count')!r}."
        )
    if not _is_non_negative_int(decision.get("source_bundle_next_action_count")):
        target.errors.append("improvement_plan.decision.source_bundle_next_action_count must be a non-negative integer.")
    top = decision.get("top_work_items")
    if not isinstance(top, list):
        target.errors.append("improvement_plan.decision.top_work_items must be a list.")
    elif len(top) > 5:
        target.errors.append("improvement_plan.decision.top_work_items must contain at most five items.")


def _validate_improvement_ledger(ledger: dict[str, Any], target: ValidationTarget) -> None:
    _require_equal(ledger, "schema_version", IMPROVEMENT_LEDGER_SCHEMA_VERSION, target)
    if not isinstance(ledger.get("ledger_path"), str):
        target.errors.append("improvement_ledger.ledger_path must be a string.")
    if ledger.get("passed") is not True:
        target.errors.append("improvement_ledger.passed must be true.")

    plans = ledger.get("plans")
    if not isinstance(plans, list):
        target.errors.append("improvement_ledger.plans must be a list.")
        plans = []
    entries = ledger.get("entries")
    if not isinstance(entries, list):
        target.errors.append("improvement_ledger.entries must be a list.")
        entries = []
    metrics = ledger.get("metrics")
    if not isinstance(metrics, dict):
        target.errors.append("improvement_ledger.metrics must be an object.")
        metrics = {}
    if "notes" in ledger and not _is_string_list(ledger.get("notes")):
        target.errors.append("improvement_ledger.notes must be a list of strings when present.")

    for index, plan in enumerate(plans):
        _validate_improvement_ledger_plan(plan, target, f"improvement_ledger.plans[{index}]", index)
    latest_index = len(plans) - 1
    totals: dict[str, Any] = {
        "work_item_count": 0,
        "open_work_item_count": 0,
        "new_work_item_count": 0,
        "recurring_work_item_count": 0,
        "resolved_work_item_count": 0,
        "critical_open_work_item_count": 0,
        "high_open_work_item_count": 0,
        "status_counts": {},
        "priority_counts": {},
        "open_priority_counts": {},
        "category_counts": {},
        "open_category_counts": {},
        "task_family_counts": {},
        "rule_counts": {},
    }
    seen_keys: set[str] = set()
    previous_sort_key: tuple[int, int, str, str, str, str] | None = None
    for index, entry in enumerate(entries):
        sort_key = _validate_improvement_ledger_entry(
            entry,
            target,
            f"improvement_ledger.entries[{index}]",
            latest_index,
            seen_keys,
            totals,
        )
        if sort_key is not None:
            if previous_sort_key is not None and sort_key < previous_sort_key:
                target.errors.append("improvement_ledger.entries must be sorted by status, priority, category, task family, rule, and key.")
            previous_sort_key = sort_key

    expected_plan_work_items = sum(
        plan.get("work_item_count") for plan in plans if isinstance(plan, dict) and _is_non_negative_int(plan.get("work_item_count"))
    )
    if ledger.get("plan_count") != len(plans):
        target.errors.append(f"improvement_ledger.plan_count expected {len(plans)}, got {ledger.get('plan_count')!r}.")
    if ledger.get("work_item_count") != expected_plan_work_items:
        target.errors.append(
            f"improvement_ledger.work_item_count expected {expected_plan_work_items}, got {ledger.get('work_item_count')!r}."
        )
    if totals["work_item_count"] != expected_plan_work_items:
        target.errors.append(
            f"improvement_ledger.entries occurrence total expected {expected_plan_work_items}, got {totals['work_item_count']}."
        )
    if ledger.get("unique_work_item_count") != len(entries):
        target.errors.append(f"improvement_ledger.unique_work_item_count expected {len(entries)}, got {ledger.get('unique_work_item_count')!r}.")
    _validate_improvement_ledger_metrics(metrics, target, totals, plans, len(entries))
    _validate_improvement_ledger_decision(ledger.get("decision"), target, plans, totals)
    target.details.update(
        {
            "plan_count": len(plans),
            "unique_work_item_count": len(entries),
            "open_work_item_count": totals["open_work_item_count"],
            "resolved_work_item_count": totals["resolved_work_item_count"],
        }
    )


def _validate_improvement_ledger_plan(value: Any, target: ValidationTarget, label: str, expected_index: int) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    if value.get("index") != expected_index:
        target.errors.append(f"{label}.index expected {expected_index}, got {value.get('index')!r}.")
    for field_name in ("path", "schema_version", "readiness", "recommendation"):
        if not isinstance(value.get(field_name), str):
            target.errors.append(f"{label}.{field_name} must be a string.")
    if value.get("schema_version") != IMPROVEMENT_PLAN_SCHEMA_VERSION:
        target.errors.append(f"{label}.schema_version must be {IMPROVEMENT_PLAN_SCHEMA_VERSION}.")
    if value.get("readiness") not in {"ready", "blocked"}:
        target.errors.append(f"{label}.readiness must be ready or blocked.")
    if not isinstance(value.get("exists"), bool):
        target.errors.append(f"{label}.exists must be a boolean.")
    if not isinstance(value.get("passed"), bool):
        target.errors.append(f"{label}.passed must be a boolean.")
    for field_name in ("work_item_count", "critical_or_high_count"):
        if not _is_non_negative_int(value.get(field_name)):
            target.errors.append(f"{label}.{field_name} must be a non-negative integer.")
    if value.get("exists") is True:
        sha = value.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64 or sha != sha.lower() or any(char not in "0123456789abcdef" for char in sha):
            target.errors.append(f"{label}.sha256 must be a lowercase 64-character hex digest for existing files.")


def _validate_improvement_ledger_entry(
    entry: Any,
    target: ValidationTarget,
    label: str,
    latest_index: int,
    seen_keys: set[str],
    totals: dict[str, Any],
) -> tuple[int, int, str, str, str, str] | None:
    if not isinstance(entry, dict):
        target.errors.append(f"{label} must be an object.")
        return None
    work_key = entry.get("work_key")
    if not isinstance(work_key, str) or not work_key:
        target.errors.append(f"{label}.work_key must be a non-empty string.")
        work_key = ""
    elif work_key in seen_keys:
        target.errors.append(f"{label}.work_key duplicates {work_key!r}.")
    seen_keys.add(work_key)
    category = entry.get("category")
    if category not in {"bundle_action", "repair", "curriculum", "digest_action"}:
        target.errors.append(f"{label}.category must be bundle_action, repair, curriculum, or digest_action.")
        category = "digest_action"
    priority = entry.get("priority")
    if priority not in set(PRIORITIES):
        target.errors.append(f"{label}.priority must be critical, high, medium, or low.")
        priority = "low"
    if entry.get("status") not in {"new", "recurring", "open", "resolved"}:
        target.errors.append(f"{label}.status must be new, recurring, open, or resolved.")
    if not isinstance(entry.get("open"), bool):
        target.errors.append(f"{label}.open must be a boolean.")
    for field_name in ("summary", "suggested_action", "first_seen_path", "last_seen_path", "latest_item_id", "latest_routing_key", "latest_fingerprint"):
        if not isinstance(entry.get(field_name), str):
            target.errors.append(f"{label}.{field_name} must be a string.")
    for field_name in ("scenario_id", "task_family", "rule_id", "rule_name", "task_completion_status"):
        if entry.get(field_name) is not None and not isinstance(entry.get(field_name), str):
            target.errors.append(f"{label}.{field_name} must be a string or null.")
    if entry.get("score") is not None and not _is_int_between(entry.get("score"), 0, 100):
        target.errors.append(f"{label}.score must be null or an integer from 0 to 100.")
    if not _is_non_negative_int(entry.get("evidence_ref_count")):
        target.errors.append(f"{label}.evidence_ref_count must be a non-negative integer.")
    occurrences = entry.get("occurrences")
    if not isinstance(occurrences, list) or not occurrences:
        target.errors.append(f"{label}.occurrences must be a non-empty list.")
        occurrences = []
    indexes: list[int] = []
    for index, occurrence in enumerate(occurrences):
        occurrence_index = _validate_improvement_ledger_occurrence(occurrence, target, f"{label}.occurrences[{index}]", category, priority)
        if occurrence_index is not None:
            indexes.append(occurrence_index)
    plan_indexes = sorted(set(indexes))
    if entry.get("occurrence_count") != len(occurrences):
        target.errors.append(f"{label}.occurrence_count expected {len(occurrences)}, got {entry.get('occurrence_count')!r}.")
    if entry.get("plan_indexes") != plan_indexes:
        target.errors.append(f"{label}.plan_indexes expected {plan_indexes!r}, got {entry.get('plan_indexes')!r}.")
    if plan_indexes:
        first_seen = plan_indexes[0]
        last_seen = plan_indexes[-1]
        expected_open = latest_index in plan_indexes
        if entry.get("first_seen_index") != first_seen:
            target.errors.append(f"{label}.first_seen_index expected {first_seen}, got {entry.get('first_seen_index')!r}.")
        if entry.get("last_seen_index") != last_seen:
            target.errors.append(f"{label}.last_seen_index expected {last_seen}, got {entry.get('last_seen_index')!r}.")
        if entry.get("open") != expected_open:
            target.errors.append(f"{label}.open expected {expected_open}, got {entry.get('open')!r}.")
        expected_status = _expected_improvement_ledger_status(plan_indexes, latest_index)
        if entry.get("status") != expected_status:
            target.errors.append(f"{label}.status expected {expected_status!r}, got {entry.get('status')!r}.")
    latest_fingerprint = entry.get("latest_fingerprint")
    if not isinstance(latest_fingerprint, str) or len(latest_fingerprint) != 64 or latest_fingerprint != latest_fingerprint.lower() or any(
        char not in "0123456789abcdef" for char in latest_fingerprint
    ):
        target.errors.append(f"{label}.latest_fingerprint must be a lowercase 64-character hex digest.")

    stable_probe = {
        "category": category,
        "scenario_id": entry.get("scenario_id"),
        "task_family": entry.get("task_family"),
        "rule_id": entry.get("rule_id"),
        "summary": entry.get("summary"),
        "sources": {},
    }
    if work_key and category in {"repair", "curriculum"} and stable_work_key(stable_probe) != work_key:
        target.errors.append(f"{label}.work_key does not match entry content.")

    totals["work_item_count"] += len(occurrences)
    _increment_count(totals["status_counts"], entry.get("status"))
    _increment_count(totals["priority_counts"], priority)
    _increment_count(totals["category_counts"], category)
    if entry.get("open") is True:
        totals["open_work_item_count"] += 1
        if entry.get("status") == "new":
            totals["new_work_item_count"] += 1
        if entry.get("status") == "recurring":
            totals["recurring_work_item_count"] += 1
        if priority == "critical":
            totals["critical_open_work_item_count"] += 1
        if priority == "high":
            totals["high_open_work_item_count"] += 1
        _increment_count(totals["open_priority_counts"], priority)
        _increment_count(totals["open_category_counts"], category)
        _increment_count(totals["task_family_counts"], entry.get("task_family"))
        _increment_count(totals["rule_counts"], entry.get("rule_id"))
    if entry.get("status") == "resolved":
        totals["resolved_work_item_count"] += 1
    return (
        {"recurring": 0, "new": 1, "open": 2, "resolved": 3}.get(str(entry.get("status")), 99),
        list(PRIORITIES).index(priority),
        str(category or ""),
        str(entry.get("task_family") or ""),
        str(entry.get("rule_id") or ""),
        str(work_key or ""),
    )


def _validate_improvement_ledger_occurrence(
    occurrence: Any,
    target: ValidationTarget,
    label: str,
    expected_category: str,
    expected_priority: str,
) -> int | None:
    if not isinstance(occurrence, dict):
        target.errors.append(f"{label} must be an object.")
        return None
    for field_name in ("plan_path", "item_id", "routing_key", "fingerprint", "priority", "category", "summary"):
        if not isinstance(occurrence.get(field_name), str):
            target.errors.append(f"{label}.{field_name} must be a string.")
    if occurrence.get("category") != expected_category:
        target.errors.append(f"{label}.category must match entry category.")
    if occurrence.get("priority") != expected_priority:
        target.errors.append(f"{label}.priority must match entry priority.")
    if not _is_non_negative_int(occurrence.get("plan_index")):
        target.errors.append(f"{label}.plan_index must be a non-negative integer.")
        return None
    fingerprint = occurrence.get("fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64 or fingerprint != fingerprint.lower() or any(
        char not in "0123456789abcdef" for char in fingerprint
    ):
        target.errors.append(f"{label}.fingerprint must be a lowercase 64-character hex digest.")
    return occurrence.get("plan_index")


def _validate_improvement_ledger_metrics(
    metrics: dict[str, Any],
    target: ValidationTarget,
    totals: dict[str, Any],
    plans: list[Any],
    unique_count: int,
) -> None:
    expected_scalars = {
        "plan_count": len(plans),
        "work_item_count": totals["work_item_count"],
        "unique_work_item_count": unique_count,
        "open_work_item_count": totals["open_work_item_count"],
        "new_work_item_count": totals["new_work_item_count"],
        "recurring_work_item_count": totals["recurring_work_item_count"],
        "resolved_work_item_count": totals["resolved_work_item_count"],
        "critical_open_work_item_count": totals["critical_open_work_item_count"],
        "high_open_work_item_count": totals["high_open_work_item_count"],
    }
    for field_name, expected in expected_scalars.items():
        if metrics.get(field_name) != expected:
            target.errors.append(f"improvement_ledger.metrics.{field_name} expected {expected!r}, got {metrics.get(field_name)!r}.")
    for field_name in (
        "status_counts",
        "priority_counts",
        "open_priority_counts",
        "category_counts",
        "open_category_counts",
        "task_family_counts",
        "rule_counts",
    ):
        actual = _count_rows(metrics.get(field_name))
        if actual != totals[field_name]:
            target.errors.append(f"improvement_ledger.metrics.{field_name} does not match ledger entries.")
    plan_counts = metrics.get("plan_work_item_counts")
    expected_counts = [
        {"index": plan.get("index"), "path": plan.get("path"), "work_item_count": plan.get("work_item_count")}
        for plan in plans
        if isinstance(plan, dict)
    ]
    if plan_counts != expected_counts:
        target.errors.append("improvement_ledger.metrics.plan_work_item_counts does not match plans.")


def _validate_improvement_ledger_decision(decision: Any, target: ValidationTarget, plans: list[Any], totals: dict[str, Any]) -> None:
    if not isinstance(decision, dict):
        target.errors.append("improvement_ledger.decision must be an object.")
        return
    if decision.get("readiness") not in {"ready", "blocked"}:
        target.errors.append("improvement_ledger.decision.readiness must be ready or blocked.")
    if decision.get("recommendation") not in {"fix_handoff", "continue_improvement", "review_remaining_work", "promote_or_monitor"}:
        target.errors.append("improvement_ledger.decision.recommendation has an unknown value.")
    if not isinstance(decision.get("summary"), str) or not decision.get("summary"):
        target.errors.append("improvement_ledger.decision.summary must be a non-empty string.")
    latest_index = len(plans) - 1
    if decision.get("latest_plan_index") != latest_index:
        target.errors.append(f"improvement_ledger.decision.latest_plan_index expected {latest_index}, got {decision.get('latest_plan_index')!r}.")
    if decision.get("open_work_item_count") != totals["open_work_item_count"]:
        target.errors.append("improvement_ledger.decision.open_work_item_count must match metrics.")
    critical_or_high = totals["critical_open_work_item_count"] + totals["high_open_work_item_count"]
    if decision.get("critical_or_high_open_count") != critical_or_high:
        target.errors.append("improvement_ledger.decision.critical_or_high_open_count must match open priority counts.")
    if decision.get("resolved_work_item_count") != totals["resolved_work_item_count"]:
        target.errors.append("improvement_ledger.decision.resolved_work_item_count must match resolved count.")
    top = decision.get("top_open_work_items")
    if not isinstance(top, list):
        target.errors.append("improvement_ledger.decision.top_open_work_items must be a list.")
    elif len(top) > 5:
        target.errors.append("improvement_ledger.decision.top_open_work_items must contain at most five items.")


def _expected_improvement_ledger_status(plan_indexes: list[int], latest_index: int) -> str:
    if latest_index in plan_indexes and plan_indexes[0] == latest_index:
        return "new"
    if latest_index in plan_indexes and len(plan_indexes) > 1:
        return "recurring"
    if latest_index in plan_indexes:
        return "open"
    return "resolved"


def _validate_action_ledger(ledger: dict[str, Any], target: ValidationTarget) -> None:
    _require_equal(ledger, "schema_version", ACTION_LEDGER_SCHEMA_VERSION, target)
    if not isinstance(ledger.get("ledger_path"), str):
        target.errors.append("action_ledger.ledger_path must be a string.")
    if ledger.get("passed") is not True:
        target.errors.append("action_ledger.passed must be true.")

    bundles = ledger.get("bundles")
    if not isinstance(bundles, list):
        target.errors.append("action_ledger.bundles must be a list.")
        bundles = []
    entries = ledger.get("entries")
    if not isinstance(entries, list):
        target.errors.append("action_ledger.entries must be a list.")
        entries = []
    metrics = ledger.get("metrics")
    if not isinstance(metrics, dict):
        target.errors.append("action_ledger.metrics must be an object.")
        metrics = {}
    notes = ledger.get("notes")
    if not isinstance(notes, list) or not all(isinstance(item, str) for item in notes):
        target.errors.append("action_ledger.notes must be a list of strings.")

    for index, bundle in enumerate(bundles):
        _validate_action_ledger_bundle(bundle, target, f"action_ledger.bundles[{index}]", index)
    latest_index = len(bundles) - 1
    action_count = 0
    open_count = 0
    new_count = 0
    recurring_count = 0
    resolved_count = 0
    status_counts: dict[str, int] = {}
    priority_counts: dict[str, int] = {}
    artifact_counts: dict[str, int] = {}
    routing_keys: set[str] = set()
    for index, entry in enumerate(entries):
        counts = _validate_action_ledger_entry(entry, target, f"action_ledger.entries[{index}]", latest_index)
        action_count += counts["occurrence_count"]
        open_count += counts["open"]
        new_count += counts["new"]
        recurring_count += counts["recurring"]
        resolved_count += counts["resolved"]
        if isinstance(entry, dict):
            routing_key = entry.get("routing_key")
            if isinstance(routing_key, str):
                if routing_key in routing_keys:
                    target.errors.append(f"action_ledger.entries[{index}].routing_key must be unique.")
                routing_keys.add(routing_key)
            _increment(status_counts, entry.get("status"))
            _increment(priority_counts, entry.get("priority"))
            _increment(artifact_counts, entry.get("artifact"))

    expected_bundle_action_count = sum(
        bundle.get("action_count") for bundle in bundles if isinstance(bundle, dict) and _is_non_negative_int(bundle.get("action_count"))
    )
    if ledger.get("bundle_count") != len(bundles):
        target.errors.append(f"action_ledger.bundle_count expected {len(bundles)}, got {ledger.get('bundle_count')!r}.")
    if ledger.get("action_count") != expected_bundle_action_count:
        target.errors.append(
            f"action_ledger.action_count expected {expected_bundle_action_count}, got {ledger.get('action_count')!r}."
        )
    if action_count != expected_bundle_action_count:
        target.errors.append(
            f"action_ledger.entries occurrence total expected {expected_bundle_action_count}, got {action_count}."
        )
    if ledger.get("unique_action_count") != len(entries):
        target.errors.append(f"action_ledger.unique_action_count expected {len(entries)}, got {ledger.get('unique_action_count')!r}.")

    expected_metrics = {
        "bundle_count": len(bundles),
        "action_count": expected_bundle_action_count,
        "unique_action_count": len(entries),
        "open_action_count": open_count,
        "new_action_count": new_count,
        "recurring_action_count": recurring_count,
        "resolved_action_count": resolved_count,
    }
    for field_name, expected in expected_metrics.items():
        if metrics.get(field_name) != expected:
            target.errors.append(f"action_ledger.metrics.{field_name} expected {expected}, got {metrics.get(field_name)!r}.")
    _validate_action_ledger_count_rows(metrics.get("status_counts"), status_counts, target, "action_ledger.metrics.status_counts")
    _validate_action_ledger_count_rows(metrics.get("priority_counts"), priority_counts, target, "action_ledger.metrics.priority_counts")
    _validate_action_ledger_count_rows(metrics.get("artifact_counts"), artifact_counts, target, "action_ledger.metrics.artifact_counts")
    _validate_action_ledger_bundle_action_counts(metrics.get("bundle_action_counts"), bundles, target)
    target.details.update(
        {
            "bundle_count": len(bundles),
            "unique_action_count": len(entries),
            "open_action_count": open_count,
            "resolved_action_count": resolved_count,
        }
    )


def _validate_action_ledger_gate(gate: dict[str, Any], target: ValidationTarget) -> None:
    _require_equal(gate, "schema_version", ACTION_LEDGER_GATE_SCHEMA_VERSION, target)
    if not isinstance(gate.get("action_ledger"), str) or not gate.get("action_ledger"):
        target.errors.append("action_ledger_gate.action_ledger must be a non-empty string.")
    if not isinstance(gate.get("passed"), bool):
        target.errors.append("action_ledger_gate.passed must be a boolean.")
    checks = gate.get("checks")
    if not isinstance(checks, list):
        target.errors.append("action_ledger_gate.checks must be a list.")
        checks = []
    metrics = gate.get("metrics")
    if not isinstance(metrics, dict):
        target.errors.append("action_ledger_gate.metrics must be an object.")
        metrics = {}
    if "policy" in gate:
        _validate_action_ledger_gate_policy_summary(gate.get("policy"), target)

    failed_checks = _validate_gate_like_checks(checks, target, "action_ledger_gate.checks")
    if gate.get("check_count") != len(checks):
        target.errors.append(f"action_ledger_gate.check_count expected {len(checks)}, got {gate.get('check_count')!r}.")
    if gate.get("failed_check_count") != failed_checks:
        target.errors.append(
            f"action_ledger_gate.failed_check_count expected {failed_checks}, got {gate.get('failed_check_count')!r}."
        )
    expected_passed = failed_checks == 0
    if isinstance(gate.get("passed"), bool) and gate.get("passed") != expected_passed:
        target.errors.append("action_ledger_gate.passed must match failed_check_count.")
    _validate_action_ledger_gate_metrics(metrics, target)
    _validate_action_ledger_gate_decision(gate.get("decision"), expected_passed, failed_checks, metrics, target)
    target.details.update(
        {
            "passed": gate.get("passed"),
            "check_count": len(checks),
            "failed_check_count": failed_checks,
            "open_action_count": metrics.get("open_action_count"),
            "recurring_action_count": metrics.get("recurring_action_count"),
        }
    )


def _validate_action_ledger_gate_decision(
    value: Any,
    expected_passed: bool,
    failed_checks: int,
    metrics: dict[str, Any],
    target: ValidationTarget,
) -> None:
    if not isinstance(value, dict):
        target.errors.append("action_ledger_gate.decision must be an object.")
        return
    expected_readiness = "ready" if expected_passed else "blocked"
    expected_recommendation = "promote_iteration" if expected_passed else "block_iteration"
    if value.get("readiness") != expected_readiness:
        target.errors.append(f"action_ledger_gate.decision.readiness expected {expected_readiness!r}, got {value.get('readiness')!r}.")
    if value.get("recommendation") != expected_recommendation:
        target.errors.append(
            "action_ledger_gate.decision.recommendation expected "
            f"{expected_recommendation!r}, got {value.get('recommendation')!r}."
        )
    if not isinstance(value.get("summary"), str) or not value.get("summary"):
        target.errors.append("action_ledger_gate.decision.summary must be a non-empty string.")
    blocking_checks = value.get("blocking_checks")
    if not isinstance(blocking_checks, list):
        target.errors.append("action_ledger_gate.decision.blocking_checks must be a list.")
        blocking_checks = []
    if value.get("blocking_check_count") != failed_checks:
        target.errors.append(
            f"action_ledger_gate.decision.blocking_check_count expected {failed_checks}, got {value.get('blocking_check_count')!r}."
        )
    if len(blocking_checks) != failed_checks:
        target.errors.append(f"action_ledger_gate.decision.blocking_checks expected {failed_checks} entries, got {len(blocking_checks)}.")
    for index, check in enumerate(blocking_checks):
        label = f"action_ledger_gate.decision.blocking_checks[{index}]"
        if not isinstance(check, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        for field_name in ("id", "summary"):
            if not isinstance(check.get(field_name), str) or not check.get(field_name):
                target.errors.append(f"{label}.{field_name} must be a non-empty string.")
        if not isinstance(check.get("scope"), dict):
            target.errors.append(f"{label}.scope must be an object.")
    key_metrics = value.get("key_metrics")
    if not isinstance(key_metrics, dict):
        target.errors.append("action_ledger_gate.decision.key_metrics must be an object.")
        return
    for field_name in (
        "bundle_count",
        "unique_action_count",
        "open_action_count",
        "new_action_count",
        "recurring_action_count",
        "resolved_action_count",
        "open_priority_counts",
    ):
        if key_metrics.get(field_name) != metrics.get(field_name):
            target.errors.append(
                f"action_ledger_gate.decision.key_metrics.{field_name} must match action_ledger_gate.metrics.{field_name}."
            )


def _validate_action_ledger_gate_metrics(metrics: dict[str, Any], target: ValidationTarget) -> None:
    count_fields = (
        "bundle_count",
        "unique_action_count",
        "open_action_count",
        "new_action_count",
        "recurring_action_count",
        "resolved_action_count",
    )
    for field_name in count_fields:
        if not _is_non_negative_int(metrics.get(field_name)):
            target.errors.append(f"action_ledger_gate.metrics.{field_name} must be a non-negative integer.")
    if all(_is_non_negative_int(metrics.get(field_name)) for field_name in count_fields):
        if metrics["unique_action_count"] != metrics["open_action_count"] + metrics["resolved_action_count"]:
            target.errors.append("action_ledger_gate.metrics.unique_action_count must equal open_action_count + resolved_action_count.")
        if metrics["open_action_count"] < metrics["new_action_count"] + metrics["recurring_action_count"]:
            target.errors.append("action_ledger_gate.metrics.open_action_count must be at least new_action_count + recurring_action_count.")
        if metrics["bundle_count"] == 0 and metrics["unique_action_count"] > 0:
            target.errors.append("action_ledger_gate.metrics.bundle_count must be positive when actions are present.")

    priority_counts = _count_rows(metrics.get("open_priority_counts"))
    if priority_counts is None:
        target.errors.append("action_ledger_gate.metrics.open_priority_counts must be a list of {id, count} objects.")
    else:
        unknown = sorted(set(priority_counts) - {"critical", "high", "medium", "low"})
        if unknown:
            target.errors.append(f"action_ledger_gate.metrics.open_priority_counts has invalid priority value(s): {', '.join(unknown)}.")
        if _is_non_negative_int(metrics.get("open_action_count")) and sum(priority_counts.values()) > metrics["open_action_count"]:
            target.errors.append("action_ledger_gate.metrics.open_priority_counts total must not exceed open_action_count.")


def _validate_action_ledger_gate_policy_summary(value: Any, target: ValidationTarget) -> None:
    if not isinstance(value, dict):
        target.errors.append("action_ledger_gate.policy must be an object when present.")
        return
    _require_equal(value, "schema_version", ACTION_LEDGER_GATE_POLICY_SCHEMA_VERSION, target, prefix="action_ledger_gate.policy.")
    if not isinstance(value.get("path"), str) or not value.get("path"):
        target.errors.append("action_ledger_gate.policy.path must be a non-empty string.")
    if "description" in value and not isinstance(value.get("description"), str):
        target.errors.append("action_ledger_gate.policy.description must be a string when present.")
    effective = value.get("effective")
    if not isinstance(effective, dict):
        target.errors.append("action_ledger_gate.policy.effective must be an object.")
        return
    allowed_fields = {
        "min_bundles",
        "max_open_actions",
        "max_new_actions",
        "max_recurring_actions",
        "min_resolved_actions",
        "forbid_open_priorities",
        "forbid_open_actions",
        "require_resolved_actions",
    }
    unknown = sorted(set(effective) - allowed_fields)
    if unknown:
        target.errors.append(f"action_ledger_gate.policy.effective has unknown field(s): {', '.join(unknown)}.")
    for field_name in (
        "min_bundles",
        "max_open_actions",
        "max_new_actions",
        "max_recurring_actions",
        "min_resolved_actions",
    ):
        if field_name in effective and not _is_non_negative_int(effective.get(field_name)):
            target.errors.append(f"action_ledger_gate.policy.effective.{field_name} must be a non-negative integer.")
    for field_name in ("forbid_open_priorities", "forbid_open_actions", "require_resolved_actions"):
        if field_name in effective and not _is_string_list(effective.get(field_name)):
            target.errors.append(f"action_ledger_gate.policy.effective.{field_name} must be a list of strings.")
    priorities = effective.get("forbid_open_priorities")
    if isinstance(priorities, list):
        unknown_priorities = sorted({item for item in priorities if isinstance(item, str)} - {"critical", "high", "medium", "low"})
        if unknown_priorities:
            target.errors.append(
                "action_ledger_gate.policy.effective.forbid_open_priorities has invalid priority value(s): "
                f"{', '.join(unknown_priorities)}."
            )


def _validate_improvement_ledger_gate(gate: dict[str, Any], target: ValidationTarget) -> None:
    _require_equal(gate, "schema_version", IMPROVEMENT_LEDGER_GATE_SCHEMA_VERSION, target)
    if not isinstance(gate.get("improvement_ledger"), str) or not gate.get("improvement_ledger"):
        target.errors.append("improvement_ledger_gate.improvement_ledger must be a non-empty string.")
    if not isinstance(gate.get("passed"), bool):
        target.errors.append("improvement_ledger_gate.passed must be a boolean.")
    checks = gate.get("checks")
    if not isinstance(checks, list):
        target.errors.append("improvement_ledger_gate.checks must be a list.")
        checks = []
    metrics = gate.get("metrics")
    if not isinstance(metrics, dict):
        target.errors.append("improvement_ledger_gate.metrics must be an object.")
        metrics = {}
    if "policy" in gate:
        _validate_improvement_ledger_gate_policy_summary(gate.get("policy"), target)

    failed_checks = _validate_gate_like_checks(checks, target, "improvement_ledger_gate.checks")
    if gate.get("check_count") != len(checks):
        target.errors.append(f"improvement_ledger_gate.check_count expected {len(checks)}, got {gate.get('check_count')!r}.")
    if gate.get("failed_check_count") != failed_checks:
        target.errors.append(
            f"improvement_ledger_gate.failed_check_count expected {failed_checks}, got {gate.get('failed_check_count')!r}."
        )
    expected_passed = failed_checks == 0
    if isinstance(gate.get("passed"), bool) and gate.get("passed") != expected_passed:
        target.errors.append("improvement_ledger_gate.passed must match failed_check_count.")
    _validate_improvement_ledger_gate_metrics(metrics, target)
    _validate_improvement_ledger_gate_decision(gate.get("decision"), expected_passed, failed_checks, metrics, target)
    target.details.update(
        {
            "passed": gate.get("passed"),
            "check_count": len(checks),
            "failed_check_count": failed_checks,
            "open_work_item_count": metrics.get("open_work_item_count"),
            "recurring_work_item_count": metrics.get("recurring_work_item_count"),
        }
    )


def _validate_improvement_ledger_gate_decision(
    value: Any,
    expected_passed: bool,
    failed_checks: int,
    metrics: dict[str, Any],
    target: ValidationTarget,
) -> None:
    if not isinstance(value, dict):
        target.errors.append("improvement_ledger_gate.decision must be an object.")
        return
    expected_readiness = "ready" if expected_passed else "blocked"
    expected_recommendation = "promote_iteration" if expected_passed else "block_iteration"
    if value.get("readiness") != expected_readiness:
        target.errors.append(
            f"improvement_ledger_gate.decision.readiness expected {expected_readiness!r}, got {value.get('readiness')!r}."
        )
    if value.get("recommendation") != expected_recommendation:
        target.errors.append(
            "improvement_ledger_gate.decision.recommendation expected "
            f"{expected_recommendation!r}, got {value.get('recommendation')!r}."
        )
    if not isinstance(value.get("summary"), str) or not value.get("summary"):
        target.errors.append("improvement_ledger_gate.decision.summary must be a non-empty string.")
    blocking_checks = value.get("blocking_checks")
    if not isinstance(blocking_checks, list):
        target.errors.append("improvement_ledger_gate.decision.blocking_checks must be a list.")
        blocking_checks = []
    if value.get("blocking_check_count") != failed_checks:
        target.errors.append(
            "improvement_ledger_gate.decision.blocking_check_count expected "
            f"{failed_checks}, got {value.get('blocking_check_count')!r}."
        )
    if len(blocking_checks) != failed_checks:
        target.errors.append(
            f"improvement_ledger_gate.decision.blocking_checks expected {failed_checks} entries, got {len(blocking_checks)}."
        )
    for index, check in enumerate(blocking_checks):
        label = f"improvement_ledger_gate.decision.blocking_checks[{index}]"
        if not isinstance(check, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        for field_name in ("id", "summary"):
            if not isinstance(check.get(field_name), str) or not check.get(field_name):
                target.errors.append(f"{label}.{field_name} must be a non-empty string.")
        if not isinstance(check.get("scope"), dict):
            target.errors.append(f"{label}.scope must be an object.")
    key_metrics = value.get("key_metrics")
    if not isinstance(key_metrics, dict):
        target.errors.append("improvement_ledger_gate.decision.key_metrics must be an object.")
        return
    for field_name in (
        "plan_count",
        "unique_work_item_count",
        "open_work_item_count",
        "new_work_item_count",
        "recurring_work_item_count",
        "resolved_work_item_count",
        "critical_open_work_item_count",
        "high_open_work_item_count",
        "open_priority_counts",
        "open_category_counts",
    ):
        if key_metrics.get(field_name) != metrics.get(field_name):
            target.errors.append(
                "improvement_ledger_gate.decision.key_metrics."
                f"{field_name} must match improvement_ledger_gate.metrics.{field_name}."
            )


def _validate_improvement_ledger_gate_metrics(metrics: dict[str, Any], target: ValidationTarget) -> None:
    count_fields = (
        "plan_count",
        "unique_work_item_count",
        "open_work_item_count",
        "new_work_item_count",
        "recurring_work_item_count",
        "resolved_work_item_count",
        "critical_open_work_item_count",
        "high_open_work_item_count",
    )
    for field_name in count_fields:
        if not _is_non_negative_int(metrics.get(field_name)):
            target.errors.append(f"improvement_ledger_gate.metrics.{field_name} must be a non-negative integer.")
    if all(_is_non_negative_int(metrics.get(field_name)) for field_name in count_fields):
        if metrics["unique_work_item_count"] != metrics["open_work_item_count"] + metrics["resolved_work_item_count"]:
            target.errors.append(
                "improvement_ledger_gate.metrics.unique_work_item_count must equal open_work_item_count + resolved_work_item_count."
            )
        if metrics["open_work_item_count"] < metrics["new_work_item_count"] + metrics["recurring_work_item_count"]:
            target.errors.append(
                "improvement_ledger_gate.metrics.open_work_item_count must be at least new_work_item_count + recurring_work_item_count."
            )
        critical_high = metrics["critical_open_work_item_count"] + metrics["high_open_work_item_count"]
        if critical_high > metrics["open_work_item_count"]:
            target.errors.append(
                "improvement_ledger_gate.metrics critical/high open counts must not exceed open_work_item_count."
            )
        if metrics["plan_count"] == 0 and metrics["unique_work_item_count"] > 0:
            target.errors.append("improvement_ledger_gate.metrics.plan_count must be positive when work items are present.")

    priority_counts = _count_rows(metrics.get("open_priority_counts"))
    if priority_counts is None:
        target.errors.append("improvement_ledger_gate.metrics.open_priority_counts must be a list of {id, count} objects.")
    else:
        unknown = sorted(set(priority_counts) - set(PRIORITIES))
        if unknown:
            target.errors.append(
                f"improvement_ledger_gate.metrics.open_priority_counts has invalid priority value(s): {', '.join(unknown)}."
            )
        if _is_non_negative_int(metrics.get("open_work_item_count")) and sum(priority_counts.values()) > metrics["open_work_item_count"]:
            target.errors.append("improvement_ledger_gate.metrics.open_priority_counts total must not exceed open_work_item_count.")

    category_counts = _count_rows(metrics.get("open_category_counts"))
    if category_counts is None:
        target.errors.append("improvement_ledger_gate.metrics.open_category_counts must be a list of {id, count} objects.")
    else:
        unknown = sorted(set(category_counts) - {"bundle_action", "repair", "curriculum", "digest_action"})
        if unknown:
            target.errors.append(
                f"improvement_ledger_gate.metrics.open_category_counts has invalid category value(s): {', '.join(unknown)}."
            )
        if _is_non_negative_int(metrics.get("open_work_item_count")) and sum(category_counts.values()) > metrics["open_work_item_count"]:
            target.errors.append("improvement_ledger_gate.metrics.open_category_counts total must not exceed open_work_item_count.")


def _validate_improvement_ledger_gate_policy_summary(value: Any, target: ValidationTarget) -> None:
    if not isinstance(value, dict):
        target.errors.append("improvement_ledger_gate.policy must be an object when present.")
        return
    _require_equal(
        value,
        "schema_version",
        IMPROVEMENT_LEDGER_GATE_POLICY_SCHEMA_VERSION,
        target,
        prefix="improvement_ledger_gate.policy.",
    )
    if not isinstance(value.get("path"), str) or not value.get("path"):
        target.errors.append("improvement_ledger_gate.policy.path must be a non-empty string.")
    if "description" in value and not isinstance(value.get("description"), str):
        target.errors.append("improvement_ledger_gate.policy.description must be a string when present.")
    effective = value.get("effective")
    if not isinstance(effective, dict):
        target.errors.append("improvement_ledger_gate.policy.effective must be an object.")
        return
    allowed_fields = {
        "min_plans",
        "max_open_work_items",
        "max_new_work_items",
        "max_recurring_work_items",
        "min_resolved_work_items",
        "max_critical_open_work_items",
        "max_high_open_work_items",
        "forbid_open_priorities",
        "forbid_open_categories",
        "forbid_open_work_keys",
        "require_open_work_keys",
        "require_resolved_work_keys",
    }
    unknown = sorted(set(effective) - allowed_fields)
    if unknown:
        target.errors.append(f"improvement_ledger_gate.policy.effective has unknown field(s): {', '.join(unknown)}.")
    for field_name in (
        "min_plans",
        "max_open_work_items",
        "max_new_work_items",
        "max_recurring_work_items",
        "min_resolved_work_items",
        "max_critical_open_work_items",
        "max_high_open_work_items",
    ):
        if field_name in effective and not _is_non_negative_int(effective.get(field_name)):
            target.errors.append(f"improvement_ledger_gate.policy.effective.{field_name} must be a non-negative integer.")
    for field_name in (
        "forbid_open_priorities",
        "forbid_open_categories",
        "forbid_open_work_keys",
        "require_open_work_keys",
        "require_resolved_work_keys",
    ):
        if field_name in effective and not _is_string_list(effective.get(field_name)):
            target.errors.append(f"improvement_ledger_gate.policy.effective.{field_name} must be a list of strings.")
    priorities = effective.get("forbid_open_priorities")
    if isinstance(priorities, list):
        unknown_priorities = sorted({item for item in priorities if isinstance(item, str)} - set(PRIORITIES))
        if unknown_priorities:
            target.errors.append(
                "improvement_ledger_gate.policy.effective.forbid_open_priorities has invalid priority value(s): "
                f"{', '.join(unknown_priorities)}."
            )
    categories = effective.get("forbid_open_categories")
    if isinstance(categories, list):
        unknown_categories = sorted(
            {item for item in categories if isinstance(item, str)} - {"bundle_action", "repair", "curriculum", "digest_action"}
        )
        if unknown_categories:
            target.errors.append(
                "improvement_ledger_gate.policy.effective.forbid_open_categories has invalid category value(s): "
                f"{', '.join(unknown_categories)}."
            )


def _validate_decision_gate(gate: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    _require_equal(gate, "schema_version", DECISION_GATE_SCHEMA_VERSION, target)
    if not isinstance(gate.get("artifact"), str) or not gate.get("artifact"):
        target.errors.append("decision_gate.artifact must be a non-empty string.")
    source_artifact = gate.get("source_artifact")
    if not isinstance(source_artifact, dict):
        target.errors.append("decision_gate.source_artifact must be an object.")
        source_artifact = {}
    _validate_decision_gate_source_artifact(source_artifact, target, source_path)
    if isinstance(gate.get("artifact"), str) and isinstance(source_artifact.get("path"), str) and gate.get("artifact") != source_artifact.get("path"):
        target.errors.append("decision_gate.artifact must match decision_gate.source_artifact.path.")
    if not isinstance(gate.get("passed"), bool):
        target.errors.append("decision_gate.passed must be a boolean.")
    if not isinstance(gate.get("expected_recommendation"), str) or not gate.get("expected_recommendation"):
        target.errors.append("decision_gate.expected_recommendation must be a non-empty string.")
    if gate.get("expected_readiness") is not None and not isinstance(gate.get("expected_readiness"), str):
        target.errors.append("decision_gate.expected_readiness must be a string or null.")
    if not isinstance(gate.get("require_passed"), bool):
        target.errors.append("decision_gate.require_passed must be a boolean.")
    if not _is_string_list(gate.get("notes")):
        target.errors.append("decision_gate.notes must be a list of strings.")

    checks = gate.get("checks")
    if not isinstance(checks, list):
        target.errors.append("decision_gate.checks must be a list.")
        checks = []
    failed_checks = _validate_gate_like_checks(checks, target, "decision_gate.checks")
    if gate.get("check_count") != len(checks):
        target.errors.append(f"decision_gate.check_count expected {len(checks)}, got {gate.get('check_count')!r}.")
    if gate.get("failed_check_count") != failed_checks:
        target.errors.append(f"decision_gate.failed_check_count expected {failed_checks}, got {gate.get('failed_check_count')!r}.")
    expected_passed = failed_checks == 0
    if isinstance(gate.get("passed"), bool) and gate.get("passed") != expected_passed:
        target.errors.append("decision_gate.passed must match failed_check_count.")
    expected_readiness = "ready" if expected_passed else "blocked"
    expected_recommendation = "allow_promotion" if expected_passed else "block_promotion"
    if gate.get("readiness") != expected_readiness:
        target.errors.append(f"decision_gate.readiness expected {expected_readiness!r}, got {gate.get('readiness')!r}.")
    if gate.get("recommendation") != expected_recommendation:
        target.errors.append(f"decision_gate.recommendation expected {expected_recommendation!r}, got {gate.get('recommendation')!r}.")

    source = gate.get("source_decision")
    if not isinstance(source, dict):
        target.errors.append("decision_gate.source_decision must be an object.")
        source = {}
    for field_name in ("schema_version", "recommendation", "readiness", "summary"):
        if not isinstance(source.get(field_name), str):
            target.errors.append(f"decision_gate.source_decision.{field_name} must be a string.")
    if source.get("passed") is not None and not isinstance(source.get("passed"), bool):
        target.errors.append("decision_gate.source_decision.passed must be a boolean or null.")
    if source.get("blocking_check_count") is not None and not _is_non_negative_int(source.get("blocking_check_count")):
        target.errors.append("decision_gate.source_decision.blocking_check_count must be a non-negative integer or null.")
    if not isinstance(source.get("key_metrics"), dict):
        target.errors.append("decision_gate.source_decision.key_metrics must be an object.")
    _validate_decision_gate_source_decision_matches_artifact(source, source_artifact, target, source_path)
    target.details.update(
        {
            "passed": gate.get("passed"),
            "recommendation": gate.get("recommendation"),
            "source_recommendation": source.get("recommendation"),
            "source_sha256": source_artifact.get("sha256"),
            "failed_check_count": failed_checks,
        }
    )


def _validate_decision_gate_source_artifact(record: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    if not isinstance(record.get("path"), str) or not record.get("path"):
        target.errors.append("decision_gate.source_artifact.path must be a non-empty string.")
    if record.get("kind") != "file":
        target.errors.append("decision_gate.source_artifact.kind must be file.")
    if not isinstance(record.get("exists"), bool):
        target.errors.append("decision_gate.source_artifact.exists must be a boolean.")
    _validate_preflight_file_hash(record, target, "decision_gate.source_artifact", source_path)


def _validate_decision_gate_source_decision_matches_artifact(
    source_decision: dict[str, Any],
    source_artifact: dict[str, Any],
    target: ValidationTarget,
    source_path: Path,
) -> None:
    file_path = _resolve_preflight_record_path(source_artifact.get("path"), source_path)
    if file_path is None or not file_path.exists() or not file_path.is_file():
        return
    try:
        artifact = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        target.errors.append(f"decision_gate.source_artifact contains invalid JSON: {exc}")
        return
    if not isinstance(artifact, dict):
        target.errors.append("decision_gate.source_artifact must contain a JSON object.")
        return
    actual_decision = artifact.get("decision") if isinstance(artifact.get("decision"), dict) else {}
    expected = {
        "schema_version": str(artifact.get("schema_version") or ""),
        "passed": artifact.get("passed") if isinstance(artifact.get("passed"), bool) else None,
        "recommendation": str(actual_decision.get("recommendation") or ""),
        "readiness": str(actual_decision.get("readiness") or ""),
        "summary": str(actual_decision.get("summary") or ""),
        "blocking_check_count": actual_decision.get("blocking_check_count")
        if _is_non_negative_int(actual_decision.get("blocking_check_count"))
        else None,
        "key_metrics": actual_decision.get("key_metrics") if isinstance(actual_decision.get("key_metrics"), dict) else {},
    }
    for field_name, expected_value in expected.items():
        if source_decision.get(field_name) != expected_value:
            target.errors.append(f"decision_gate.source_decision.{field_name} must match current source artifact.")


def _validate_promotion_ledger(ledger: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    _require_equal(ledger, "schema_version", PROMOTION_LEDGER_SCHEMA_VERSION, target)
    if not isinstance(ledger.get("ledger_path"), str):
        target.errors.append("promotion_ledger.ledger_path must be a string.")
    if ledger.get("passed") is not True:
        target.errors.append("promotion_ledger.passed must be true.")

    records = ledger.get("records")
    if not isinstance(records, list):
        target.errors.append("promotion_ledger.records must be a list.")
        records = []
    metrics = ledger.get("metrics")
    if not isinstance(metrics, dict):
        target.errors.append("promotion_ledger.metrics must be an object.")
        metrics = {}
    notes = ledger.get("notes")
    if not isinstance(notes, list) or not all(isinstance(item, str) for item in notes):
        target.errors.append("promotion_ledger.notes must be a list of strings.")

    for index, record in enumerate(records):
        _validate_promotion_ledger_record(record, target, f"promotion_ledger.records[{index}]", index, source_path)

    if ledger.get("decision_count") != len(records):
        target.errors.append(f"promotion_ledger.decision_count expected {len(records)}, got {ledger.get('decision_count')!r}.")

    expected_metrics = _promotion_ledger_expected_metrics(records)
    for field_name in (
        "decision_count",
        "allowed_count",
        "blocked_count",
        "latest_recommendation",
        "latest_readiness",
        "latest_passed",
        "consecutive_allowed_count",
        "consecutive_blocked_count",
        "unique_source_artifact_count",
    ):
        if metrics.get(field_name) != expected_metrics[field_name]:
            target.errors.append(f"promotion_ledger.metrics.{field_name} expected {expected_metrics[field_name]!r}, got {metrics.get(field_name)!r}.")
    _validate_action_ledger_count_rows(
        metrics.get("recommendation_counts"),
        expected_metrics["recommendation_counts"],
        target,
        "promotion_ledger.metrics.recommendation_counts",
    )
    _validate_action_ledger_count_rows(
        metrics.get("source_recommendation_counts"),
        expected_metrics["source_recommendation_counts"],
        target,
        "promotion_ledger.metrics.source_recommendation_counts",
    )
    if metrics.get("decision_gate_results") != expected_metrics["decision_gate_results"]:
        target.errors.append("promotion_ledger.metrics.decision_gate_results must match promotion_ledger.records.")

    target.details.update(
        {
            "decision_count": len(records),
            "allowed_count": expected_metrics["allowed_count"],
            "blocked_count": expected_metrics["blocked_count"],
            "latest_recommendation": expected_metrics["latest_recommendation"],
        }
    )


def _validate_promotion_ledger_record(
    record: Any,
    target: ValidationTarget,
    label: str,
    expected_index: int,
    source_path: Path,
) -> None:
    if not isinstance(record, dict):
        target.errors.append(f"{label} must be an object.")
        return
    if record.get("index") != expected_index:
        target.errors.append(f"{label}.index expected {expected_index}, got {record.get('index')!r}.")
    for field_name in ("path", "schema_version", "readiness", "recommendation", "expected_recommendation"):
        if not isinstance(record.get(field_name), str):
            target.errors.append(f"{label}.{field_name} must be a string.")
    if record.get("schema_version") != DECISION_GATE_SCHEMA_VERSION:
        target.errors.append(f"{label}.schema_version must be {DECISION_GATE_SCHEMA_VERSION}.")
    if record.get("expected_readiness") is not None and not isinstance(record.get("expected_readiness"), str):
        target.errors.append(f"{label}.expected_readiness must be a string or null.")
    for field_name in ("exists", "passed", "require_passed"):
        if not isinstance(record.get(field_name), bool):
            target.errors.append(f"{label}.{field_name} must be a boolean.")
    for field_name in ("check_count", "failed_check_count"):
        if not _is_non_negative_int(record.get(field_name)):
            target.errors.append(f"{label}.{field_name} must be a non-negative integer.")
    if _is_non_negative_int(record.get("check_count")) and _is_non_negative_int(record.get("failed_check_count")):
        if record["failed_check_count"] > record["check_count"]:
            target.errors.append(f"{label}.failed_check_count must be less than or equal to check_count.")
    if record.get("exists") is True:
        _validate_preflight_file_hash(record, target, label, source_path, require_kind=False)
    if record.get("recommendation") not in {"allow_promotion", "block_promotion"}:
        target.errors.append(f"{label}.recommendation must be allow_promotion or block_promotion.")
    expected_allowed = record.get("passed") is True and record.get("recommendation") == "allow_promotion"
    expected_blocked = record.get("passed") is not True or record.get("recommendation") == "block_promotion"
    if expected_allowed and expected_blocked:
        target.errors.append(f"{label} cannot be both allowed and blocked.")

    source = record.get("source")
    if not isinstance(source, dict):
        target.errors.append(f"{label}.source must be an object.")
        source = {}
    _validate_promotion_ledger_source(source, target, f"{label}.source")
    _validate_promotion_ledger_record_matches_gate(record, source, target, label, source_path)


def _validate_promotion_ledger_source(source: dict[str, Any], target: ValidationTarget, label: str) -> None:
    for field_name in ("schema_version", "recommendation", "readiness", "artifact_path"):
        if not isinstance(source.get(field_name), str):
            target.errors.append(f"{label}.{field_name} must be a string.")
    if source.get("passed") is not None and not isinstance(source.get("passed"), bool):
        target.errors.append(f"{label}.passed must be a boolean or null.")
    if source.get("blocking_check_count") is not None and not _is_non_negative_int(source.get("blocking_check_count")):
        target.errors.append(f"{label}.blocking_check_count must be a non-negative integer or null.")
    if not isinstance(source.get("artifact_exists"), bool):
        target.errors.append(f"{label}.artifact_exists must be a boolean.")
    if source.get("artifact_sha256") is not None and not _is_sha256(source.get("artifact_sha256")):
        target.errors.append(f"{label}.artifact_sha256 must be a SHA-256 hex string or null.")
    if source.get("artifact_exists") is True and not _is_sha256(source.get("artifact_sha256")):
        target.errors.append(f"{label}.artifact_sha256 must be present for existing source artifacts.")


def _validate_promotion_ledger_record_matches_gate(
    record: dict[str, Any],
    source: dict[str, Any],
    target: ValidationTarget,
    label: str,
    source_path: Path,
) -> None:
    file_path = _resolve_preflight_record_path(record.get("path"), source_path)
    if file_path is None or not file_path.exists() or not file_path.is_file():
        return
    try:
        gate = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        target.errors.append(f"{label}.path contains invalid JSON: {exc}")
        return
    if not isinstance(gate, dict):
        target.errors.append(f"{label}.path must contain a JSON object.")
        return

    gate_source = gate.get("source_decision") if isinstance(gate.get("source_decision"), dict) else {}
    gate_artifact = gate.get("source_artifact") if isinstance(gate.get("source_artifact"), dict) else {}
    expected = {
        "schema_version": str(gate.get("schema_version") or ""),
        "passed": gate.get("passed") is True,
        "readiness": str(gate.get("readiness") or ""),
        "recommendation": str(gate.get("recommendation") or ""),
        "expected_recommendation": str(gate.get("expected_recommendation") or ""),
        "expected_readiness": gate.get("expected_readiness") if isinstance(gate.get("expected_readiness"), str) else None,
        "require_passed": gate.get("require_passed") is True,
        "check_count": gate.get("check_count") if _is_non_negative_int(gate.get("check_count")) else 0,
        "failed_check_count": gate.get("failed_check_count") if _is_non_negative_int(gate.get("failed_check_count")) else 0,
    }
    for field_name, expected_value in expected.items():
        if record.get(field_name) != expected_value:
            target.errors.append(f"{label}.{field_name} must match current decision gate.")

    expected_source = {
        "schema_version": str(gate_source.get("schema_version") or ""),
        "passed": gate_source.get("passed") if isinstance(gate_source.get("passed"), bool) else None,
        "recommendation": str(gate_source.get("recommendation") or ""),
        "readiness": str(gate_source.get("readiness") or ""),
        "blocking_check_count": gate_source.get("blocking_check_count") if _is_non_negative_int(gate_source.get("blocking_check_count")) else None,
        "artifact_path": str(gate_artifact.get("path") or ""),
        "artifact_exists": gate_artifact.get("exists") is True,
        "artifact_sha256": gate_artifact.get("sha256") if _is_sha256(gate_artifact.get("sha256")) else None,
    }
    for field_name, expected_value in expected_source.items():
        if source.get(field_name) != expected_value:
            target.errors.append(f"{label}.source.{field_name} must match current decision gate.")


def _promotion_ledger_expected_metrics(records: list[Any]) -> dict[str, Any]:
    valid_records = [record for record in records if isinstance(record, dict)]
    latest = valid_records[-1] if valid_records else {}
    source_keys = {
        _promotion_source_artifact_key(record)
        for record in valid_records
        if _promotion_source_artifact_key(record)
    }
    return {
        "decision_count": len(records),
        "allowed_count": sum(1 for record in valid_records if _promotion_record_allowed(record)),
        "blocked_count": sum(1 for record in valid_records if _promotion_record_blocked(record)),
        "latest_recommendation": latest.get("recommendation") if valid_records else "",
        "latest_readiness": latest.get("readiness") if valid_records else "",
        "latest_passed": latest.get("passed") if valid_records else None,
        "consecutive_allowed_count": _promotion_consecutive(valid_records, _promotion_record_allowed),
        "consecutive_blocked_count": _promotion_consecutive(valid_records, _promotion_record_blocked),
        "unique_source_artifact_count": len(source_keys),
        "recommendation_counts": _promotion_count_map(record.get("recommendation") for record in valid_records),
        "source_recommendation_counts": _promotion_count_map(
            record.get("source", {}).get("recommendation") if isinstance(record.get("source"), dict) else ""
            for record in valid_records
        ),
        "decision_gate_results": [
            {
                "index": record.get("index"),
                "path": record.get("path"),
                "passed": record.get("passed"),
                "recommendation": record.get("recommendation"),
                "source_recommendation": record.get("source", {}).get("recommendation") if isinstance(record.get("source"), dict) else "",
                "failed_check_count": record.get("failed_check_count"),
            }
            for record in valid_records
        ],
    }


def _promotion_record_allowed(record: dict[str, Any]) -> bool:
    return record.get("passed") is True and record.get("recommendation") == "allow_promotion"


def _promotion_record_blocked(record: dict[str, Any]) -> bool:
    return record.get("passed") is not True or record.get("recommendation") == "block_promotion"


def _promotion_consecutive(records: list[dict[str, Any]], predicate: Any) -> int:
    count = 0
    for record in reversed(records):
        if not predicate(record):
            break
        count += 1
    return count


def _promotion_source_artifact_key(record: dict[str, Any]) -> str:
    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    sha256 = source.get("artifact_sha256")
    if isinstance(sha256, str) and sha256:
        return sha256
    path = source.get("artifact_path")
    return path if isinstance(path, str) and path else ""


def _promotion_count_map(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def _validate_promotion_ledger_gate(gate: dict[str, Any], target: ValidationTarget) -> None:
    _require_equal(gate, "schema_version", PROMOTION_LEDGER_GATE_SCHEMA_VERSION, target)
    if not isinstance(gate.get("promotion_ledger"), str) or not gate.get("promotion_ledger"):
        target.errors.append("promotion_ledger_gate.promotion_ledger must be a non-empty string.")
    if not isinstance(gate.get("passed"), bool):
        target.errors.append("promotion_ledger_gate.passed must be a boolean.")
    checks = gate.get("checks")
    if not isinstance(checks, list):
        target.errors.append("promotion_ledger_gate.checks must be a list.")
        checks = []
    metrics = gate.get("metrics")
    if not isinstance(metrics, dict):
        target.errors.append("promotion_ledger_gate.metrics must be an object.")
        metrics = {}
    if "policy" in gate:
        _validate_promotion_ledger_gate_policy_summary(gate.get("policy"), target)

    failed_checks = _validate_gate_like_checks(checks, target, "promotion_ledger_gate.checks")
    if gate.get("check_count") != len(checks):
        target.errors.append(f"promotion_ledger_gate.check_count expected {len(checks)}, got {gate.get('check_count')!r}.")
    if gate.get("failed_check_count") != failed_checks:
        target.errors.append(
            f"promotion_ledger_gate.failed_check_count expected {failed_checks}, got {gate.get('failed_check_count')!r}."
        )
    expected_passed = failed_checks == 0
    if isinstance(gate.get("passed"), bool) and gate.get("passed") != expected_passed:
        target.errors.append("promotion_ledger_gate.passed must match failed_check_count.")
    _validate_promotion_ledger_gate_metrics(metrics, target)
    _validate_promotion_ledger_gate_decision(gate.get("decision"), expected_passed, failed_checks, metrics, target)
    target.details.update(
        {
            "passed": gate.get("passed"),
            "check_count": len(checks),
            "failed_check_count": failed_checks,
            "decision_count": metrics.get("decision_count"),
            "latest_recommendation": metrics.get("latest_recommendation"),
        }
    )


def _validate_promotion_ledger_gate_metrics(metrics: dict[str, Any], target: ValidationTarget) -> None:
    count_fields = (
        "decision_count",
        "allowed_count",
        "blocked_count",
        "consecutive_allowed_count",
        "consecutive_blocked_count",
        "failed_decision_count",
        "unique_source_artifact_count",
    )
    for field_name in count_fields:
        if not _is_non_negative_int(metrics.get(field_name)):
            target.errors.append(f"promotion_ledger_gate.metrics.{field_name} must be a non-negative integer.")
    if all(_is_non_negative_int(metrics.get(field_name)) for field_name in ("decision_count", "allowed_count", "blocked_count")):
        if metrics["allowed_count"] + metrics["blocked_count"] != metrics["decision_count"]:
            target.errors.append("promotion_ledger_gate.metrics.allowed_count + blocked_count must equal decision_count.")
    if not _is_number_between(metrics.get("blocked_rate"), 0.0, 1.0):
        target.errors.append("promotion_ledger_gate.metrics.blocked_rate must be a number from 0.0 to 1.0.")
    elif _is_non_negative_int(metrics.get("decision_count")) and _is_non_negative_int(metrics.get("blocked_count")):
        expected_rate = round(metrics["blocked_count"] / metrics["decision_count"], 4) if metrics["decision_count"] else 0.0
        if metrics.get("blocked_rate") != expected_rate:
            target.errors.append(f"promotion_ledger_gate.metrics.blocked_rate expected {expected_rate}, got {metrics.get('blocked_rate')!r}.")
    for field_name in ("latest_recommendation", "latest_readiness"):
        if not isinstance(metrics.get(field_name), str):
            target.errors.append(f"promotion_ledger_gate.metrics.{field_name} must be a string.")
    if metrics.get("latest_passed") is not None and not isinstance(metrics.get("latest_passed"), bool):
        target.errors.append("promotion_ledger_gate.metrics.latest_passed must be a boolean or null.")
    if _count_rows(metrics.get("source_recommendation_counts")) is None:
        target.errors.append("promotion_ledger_gate.metrics.source_recommendation_counts must be a list of {id, count} objects.")


def _validate_promotion_ledger_gate_decision(
    value: Any,
    expected_passed: bool,
    failed_checks: int,
    metrics: dict[str, Any],
    target: ValidationTarget,
) -> None:
    if not isinstance(value, dict):
        target.errors.append("promotion_ledger_gate.decision must be an object.")
        return
    expected_readiness = "ready" if expected_passed else "blocked"
    expected_recommendation = "promote_iteration" if expected_passed else "block_iteration"
    if value.get("readiness") != expected_readiness:
        target.errors.append(f"promotion_ledger_gate.decision.readiness expected {expected_readiness!r}, got {value.get('readiness')!r}.")
    if value.get("recommendation") != expected_recommendation:
        target.errors.append(
            "promotion_ledger_gate.decision.recommendation expected "
            f"{expected_recommendation!r}, got {value.get('recommendation')!r}."
        )
    if not isinstance(value.get("summary"), str) or not value.get("summary"):
        target.errors.append("promotion_ledger_gate.decision.summary must be a non-empty string.")
    blocking_checks = value.get("blocking_checks")
    if not isinstance(blocking_checks, list):
        target.errors.append("promotion_ledger_gate.decision.blocking_checks must be a list.")
        blocking_checks = []
    if value.get("blocking_check_count") != failed_checks:
        target.errors.append(
            f"promotion_ledger_gate.decision.blocking_check_count expected {failed_checks}, got {value.get('blocking_check_count')!r}."
        )
    if len(blocking_checks) != failed_checks:
        target.errors.append(
            f"promotion_ledger_gate.decision.blocking_checks expected {failed_checks} entries, got {len(blocking_checks)}."
        )
    for index, check in enumerate(blocking_checks):
        label = f"promotion_ledger_gate.decision.blocking_checks[{index}]"
        if not isinstance(check, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        for field_name in ("id", "summary"):
            if not isinstance(check.get(field_name), str) or not check.get(field_name):
                target.errors.append(f"{label}.{field_name} must be a non-empty string.")
        if not isinstance(check.get("scope"), dict):
            target.errors.append(f"{label}.scope must be an object.")
    key_metrics = value.get("key_metrics")
    if not isinstance(key_metrics, dict):
        target.errors.append("promotion_ledger_gate.decision.key_metrics must be an object.")
        return
    for field_name in (
        "decision_count",
        "allowed_count",
        "blocked_count",
        "blocked_rate",
        "latest_recommendation",
        "latest_passed",
        "consecutive_allowed_count",
        "consecutive_blocked_count",
        "failed_decision_count",
        "source_recommendation_counts",
    ):
        if key_metrics.get(field_name) != metrics.get(field_name):
            target.errors.append(
                f"promotion_ledger_gate.decision.key_metrics.{field_name} must match promotion_ledger_gate.metrics.{field_name}."
            )


def _validate_promotion_ledger_gate_policy_summary(value: Any, target: ValidationTarget) -> None:
    if not isinstance(value, dict):
        target.errors.append("promotion_ledger_gate.policy must be an object when present.")
        return
    _require_equal(value, "schema_version", PROMOTION_LEDGER_GATE_POLICY_SCHEMA_VERSION, target, prefix="promotion_ledger_gate.policy.")
    if not isinstance(value.get("path"), str) or not value.get("path"):
        target.errors.append("promotion_ledger_gate.policy.path must be a non-empty string.")
    if "description" in value and not isinstance(value.get("description"), str):
        target.errors.append("promotion_ledger_gate.policy.description must be a string when present.")
    effective = value.get("effective")
    if not isinstance(effective, dict):
        target.errors.append("promotion_ledger_gate.policy.effective must be an object.")
        return
    allowed_fields = {
        "min_decisions",
        "min_allowed_count",
        "max_blocked_count",
        "max_blocked_rate",
        "min_consecutive_allowed",
        "max_consecutive_blocked",
        "max_failed_decisions",
        "require_latest_recommendation",
        "require_latest_passed",
        "require_source_recommendations",
        "forbid_source_recommendations",
    }
    unknown = sorted(set(effective) - allowed_fields)
    if unknown:
        target.errors.append(f"promotion_ledger_gate.policy.effective has unknown field(s): {', '.join(unknown)}.")
    for field_name in (
        "min_decisions",
        "min_allowed_count",
        "max_blocked_count",
        "min_consecutive_allowed",
        "max_consecutive_blocked",
        "max_failed_decisions",
    ):
        if field_name in effective and not _is_non_negative_int(effective.get(field_name)):
            target.errors.append(f"promotion_ledger_gate.policy.effective.{field_name} must be a non-negative integer.")
    if "max_blocked_rate" in effective and not _is_number_between(effective.get("max_blocked_rate"), 0.0, 1.0):
        target.errors.append("promotion_ledger_gate.policy.effective.max_blocked_rate must be a number from 0.0 to 1.0.")
    if "require_latest_recommendation" in effective and effective.get("require_latest_recommendation") not in {
        "allow_promotion",
        "block_promotion",
    }:
        target.errors.append("promotion_ledger_gate.policy.effective.require_latest_recommendation is invalid.")
    if "require_latest_passed" in effective and not isinstance(effective.get("require_latest_passed"), bool):
        target.errors.append("promotion_ledger_gate.policy.effective.require_latest_passed must be a boolean.")
    for field_name in ("require_source_recommendations", "forbid_source_recommendations"):
        if field_name in effective and not _is_string_list(effective.get(field_name)):
            target.errors.append(f"promotion_ledger_gate.policy.effective.{field_name} must be a list of strings.")


def _validate_promotion_archive(archive: dict[str, Any], target: ValidationTarget, archive_root: Path) -> None:
    _require_equal(archive, "schema_version", PROMOTION_ARCHIVE_SCHEMA_VERSION, target)
    for field_name in ("archive_path", "manifest_path"):
        if not isinstance(archive.get(field_name), str) or not archive.get(field_name):
            target.errors.append(f"promotion_archive.{field_name} must be a non-empty string.")
    for field_name in ("passed", "self_contained", "require_self_contained"):
        if not isinstance(archive.get(field_name), bool):
            target.errors.append(f"promotion_archive.{field_name} must be a boolean.")
    artifacts = archive.get("artifacts")
    if not isinstance(artifacts, list):
        target.errors.append("promotion_archive.artifacts must be a list.")
        artifacts = []
    missing = archive.get("missing")
    if not isinstance(missing, list):
        target.errors.append("promotion_archive.missing must be a list.")
        missing = []
    relationships = archive.get("relationships")
    if not isinstance(relationships, list):
        target.errors.append("promotion_archive.relationships must be a list.")
        relationships = []
    metrics = archive.get("metrics")
    if not isinstance(metrics, dict):
        target.errors.append("promotion_archive.metrics must be an object.")
        metrics = {}
    if not _is_string_list(archive.get("notes")):
        target.errors.append("promotion_archive.notes must be a list of strings.")

    for index, artifact in enumerate(artifacts):
        _validate_promotion_archive_artifact(artifact, target, f"promotion_archive.artifacts[{index}]", index, archive_root)
    for index, item in enumerate(missing):
        _validate_promotion_archive_missing(item, target, f"promotion_archive.missing[{index}]")
    for index, relationship in enumerate(relationships):
        _validate_promotion_archive_relationship(relationship, target, f"promotion_archive.relationships[{index}]")

    self_contained = len(missing) == 0
    if isinstance(archive.get("self_contained"), bool) and archive["self_contained"] != self_contained:
        target.errors.append(f"promotion_archive.self_contained expected {self_contained}, got {archive.get('self_contained')!r}.")
    expected_passed = self_contained or archive.get("require_self_contained") is not True
    if isinstance(archive.get("passed"), bool) and archive["passed"] != expected_passed:
        target.errors.append(f"promotion_archive.passed expected {expected_passed}, got {archive.get('passed')!r}.")
    _validate_promotion_archive_metrics(metrics, artifacts, missing, target)
    roles = {artifact.get("role") for artifact in artifacts if isinstance(artifact, dict)}
    for required_role in ("promotion_ledger",):
        if required_role not in roles:
            target.errors.append(f"promotion_archive.artifacts must include role {required_role}.")
    target.details.update(
        {
            "artifact_count": len(artifacts),
            "missing_count": len(missing),
            "self_contained": archive.get("self_contained"),
            "passed": archive.get("passed"),
        }
    )


def _validate_promotion_archive_artifact(
    artifact: Any,
    target: ValidationTarget,
    label: str,
    expected_index: int,
    archive_root: Path,
) -> None:
    if not isinstance(artifact, dict):
        target.errors.append(f"{label} must be an object.")
        return
    if artifact.get("index") != expected_index:
        target.errors.append(f"{label}.index expected {expected_index}, got {artifact.get('index')!r}.")
    for field_name in ("name", "role", "path", "original_path", "schema_version"):
        if not isinstance(artifact.get(field_name), str) or not artifact.get(field_name):
            target.errors.append(f"{label}.{field_name} must be a non-empty string.")
    if artifact.get("role") not in {"promotion_ledger", "promotion_ledger_gate", "decision_gate", "source_artifact"}:
        target.errors.append(f"{label}.role is invalid.")
    if artifact.get("exists") is not True:
        target.errors.append(f"{label}.exists must be true.")
    if not _is_non_negative_int(artifact.get("size_bytes")):
        target.errors.append(f"{label}.size_bytes must be a non-negative integer.")
    if not _is_sha256(artifact.get("sha256")):
        target.errors.append(f"{label}.sha256 must be a SHA-256 hex string.")
        return
    artifact_path = _archive_artifact_path(artifact.get("path"), archive_root)
    if artifact_path is None:
        target.errors.append(f"{label}.path must be a relative archive path.")
        return
    if not _path_resolves_inside(artifact_path, archive_root):
        target.errors.append(f"{label}.path must resolve inside the archive.")
        return
    if artifact_path.is_symlink():
        target.errors.append(f"{label}.path must not be a symlink.")
        return
    if not artifact_path.exists() or not artifact_path.is_file():
        target.errors.append(f"{label}.path does not exist inside the archive.")
        return
    if artifact_path.stat().st_size != artifact.get("size_bytes"):
        target.errors.append(f"{label}.size_bytes does not match the archived file.")
    if _sha256(artifact_path) != artifact.get("sha256"):
        target.errors.append(f"{label}.sha256 does not match the archived file.")


def _validate_promotion_archive_missing(item: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(item, dict):
        target.errors.append(f"{label} must be an object.")
        return
    if item.get("role") not in {"decision_gate", "source_artifact"}:
        target.errors.append(f"{label}.role must be decision_gate or source_artifact.")
    if not _is_non_negative_int(item.get("index")):
        target.errors.append(f"{label}.index must be a non-negative integer.")
    if not isinstance(item.get("reason"), str) or not item.get("reason"):
        target.errors.append(f"{label}.reason must be a non-empty string.")


def _validate_promotion_archive_relationship(relationship: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(relationship, dict):
        target.errors.append(f"{label} must be an object.")
        return
    for field_name in ("from", "to", "type"):
        if not isinstance(relationship.get(field_name), str) or not relationship.get(field_name):
            target.errors.append(f"{label}.{field_name} must be a non-empty string.")


def _validate_promotion_archive_metrics(
    metrics: dict[str, Any],
    artifacts: list[Any],
    missing: list[Any],
    target: ValidationTarget,
) -> None:
    valid_artifacts = [artifact for artifact in artifacts if isinstance(artifact, dict)]
    valid_missing = [item for item in missing if isinstance(item, dict)]
    expected = {
        "artifact_count": len(artifacts),
        "decision_gate_count": sum(1 for artifact in valid_artifacts if artifact.get("role") == "decision_gate"),
        "source_artifact_count": sum(1 for artifact in valid_artifacts if artifact.get("role") == "source_artifact"),
        "missing_count": len(missing),
        "unique_sha256_count": len({artifact.get("sha256") for artifact in valid_artifacts if isinstance(artifact.get("sha256"), str)}),
    }
    for field_name, expected_value in expected.items():
        if metrics.get(field_name) != expected_value:
            target.errors.append(f"promotion_archive.metrics.{field_name} expected {expected_value}, got {metrics.get(field_name)!r}.")
    role_counts = _promotion_archive_count_map(artifact.get("role") for artifact in valid_artifacts)
    missing_role_counts = _promotion_archive_count_map(item.get("role") for item in valid_missing)
    _validate_action_ledger_count_rows(metrics.get("role_counts"), role_counts, target, "promotion_archive.metrics.role_counts")
    _validate_action_ledger_count_rows(
        metrics.get("missing_role_counts"),
        missing_role_counts,
        target,
        "promotion_archive.metrics.missing_role_counts",
    )


def _promotion_archive_count_map(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def _validate_trainer_archive(archive: dict[str, Any], target: ValidationTarget, archive_root: Path) -> None:
    _require_equal(archive, "schema_version", TRAINER_ARCHIVE_SCHEMA_VERSION, target)
    for field_name in ("archive_path", "manifest_path", "readiness", "recommendation"):
        if not isinstance(archive.get(field_name), str) or not archive.get(field_name):
            target.errors.append(f"trainer_archive.{field_name} must be a non-empty string.")
    for field_name in ("passed", "self_contained", "require_self_contained", "ready_for_training", "launch_check_included"):
        if not isinstance(archive.get(field_name), bool):
            target.errors.append(f"trainer_archive.{field_name} must be a boolean.")
    artifacts = archive.get("artifacts")
    if not isinstance(artifacts, list):
        target.errors.append("trainer_archive.artifacts must be a list.")
        artifacts = []
    missing = archive.get("missing")
    if not isinstance(missing, list):
        target.errors.append("trainer_archive.missing must be a list.")
        missing = []
    relationships = archive.get("relationships")
    if not isinstance(relationships, list):
        target.errors.append("trainer_archive.relationships must be a list.")
        relationships = []
    metrics = archive.get("metrics")
    if not isinstance(metrics, dict):
        target.errors.append("trainer_archive.metrics must be an object.")
        metrics = {}
    if not _is_string_list(archive.get("notes")):
        target.errors.append("trainer_archive.notes must be a list of strings.")

    for index, artifact in enumerate(artifacts):
        _validate_trainer_archive_artifact(artifact, target, f"trainer_archive.artifacts[{index}]", index, archive_root)
    for index, item in enumerate(missing):
        _validate_trainer_archive_missing(item, target, f"trainer_archive.missing[{index}]")
    for index, relationship in enumerate(relationships):
        _validate_promotion_archive_relationship(relationship, target, f"trainer_archive.relationships[{index}]")

    valid_artifacts = [artifact for artifact in artifacts if isinstance(artifact, dict)]
    roles = {artifact.get("role") for artifact in valid_artifacts}
    for required_role in ("trainer_preflight", "trainer_launch_check"):
        if required_role not in roles:
            target.errors.append(f"trainer_archive.artifacts must include role {required_role}.")
    preflight = next((artifact for artifact in valid_artifacts if artifact.get("role") == "trainer_preflight"), {})
    launch_check = next((artifact for artifact in valid_artifacts if artifact.get("role") == "trainer_launch_check"), {})
    launch_included = bool(launch_check)
    ready_for_training = preflight.get("source_passed") is True and launch_check.get("source_passed") is True
    self_contained = len(missing) == 0
    expected_passed = ready_for_training and (self_contained or archive.get("require_self_contained") is not True)
    if isinstance(archive.get("launch_check_included"), bool) and archive["launch_check_included"] != launch_included:
        target.errors.append(f"trainer_archive.launch_check_included expected {launch_included}, got {archive.get('launch_check_included')!r}.")
    if isinstance(archive.get("ready_for_training"), bool) and archive["ready_for_training"] != ready_for_training:
        target.errors.append(f"trainer_archive.ready_for_training expected {ready_for_training}, got {archive.get('ready_for_training')!r}.")
    if isinstance(archive.get("self_contained"), bool) and archive["self_contained"] != self_contained:
        target.errors.append(f"trainer_archive.self_contained expected {self_contained}, got {archive.get('self_contained')!r}.")
    if isinstance(archive.get("passed"), bool) and archive["passed"] != expected_passed:
        target.errors.append(f"trainer_archive.passed expected {expected_passed}, got {archive.get('passed')!r}.")
    expected_readiness = "ready" if expected_passed else "blocked"
    expected_recommendation = "handoff_ready" if expected_passed else "block_handoff"
    if archive.get("readiness") != expected_readiness:
        target.errors.append(f"trainer_archive.readiness expected {expected_readiness!r}, got {archive.get('readiness')!r}.")
    if archive.get("recommendation") != expected_recommendation:
        target.errors.append(f"trainer_archive.recommendation expected {expected_recommendation!r}, got {archive.get('recommendation')!r}.")
    _validate_trainer_archive_inputs(archive.get("trainer_inputs"), valid_artifacts, target)
    _validate_trainer_archive_rewrites(archive.get("path_rewrites"), archive.get("trainer_inputs"), target)
    _validate_trainer_archive_commands(
        archive.get("approved_command"),
        archive.get("portable_command"),
        archive.get("path_rewrites"),
        target,
    )
    _validate_trainer_archive_consumer_contract(
        archive.get("consumer_contract"),
        archive.get("portable_command"),
        archive.get("trainer_inputs"),
        archive.get("path_rewrites"),
        target,
    )
    _validate_trainer_archive_metrics(metrics, artifacts, missing, archive.get("consumer_contract"), target)
    target.details.update(
        {
            "artifact_count": len(artifacts),
            "missing_count": len(missing),
            "self_contained": archive.get("self_contained"),
            "ready_for_training": archive.get("ready_for_training"),
            "passed": archive.get("passed"),
        }
    )


def _validate_trainer_archive_artifact(
    artifact: Any,
    target: ValidationTarget,
    label: str,
    expected_index: int,
    archive_root: Path,
) -> None:
    if not isinstance(artifact, dict):
        target.errors.append(f"{label} must be an object.")
        return
    if artifact.get("index") != expected_index:
        target.errors.append(f"{label}.index expected {expected_index}, got {artifact.get('index')!r}.")
    for field_name in ("name", "role", "kind", "path", "original_path"):
        if not isinstance(artifact.get(field_name), str) or not artifact.get(field_name):
            target.errors.append(f"{label}.{field_name} must be a non-empty string.")
    valid_roles = {"trainer_preflight", "trainer_launch_check", "gate", "validation_summary", "trainer_artifact", "schema_contract"}
    if artifact.get("role") not in valid_roles:
        target.errors.append(f"{label}.role is invalid.")
    if artifact.get("kind") not in {"file", "directory"}:
        target.errors.append(f"{label}.kind must be file or directory.")
    if artifact.get("exists") is not True:
        target.errors.append(f"{label}.exists must be true.")
    if not _is_non_negative_int(artifact.get("size_bytes")):
        target.errors.append(f"{label}.size_bytes must be a non-negative integer.")
    if not _is_sha256(artifact.get("sha256")):
        target.errors.append(f"{label}.sha256 must be a SHA-256 hex string.")
        return
    if artifact.get("source_passed") is not None and not isinstance(artifact.get("source_passed"), bool):
        target.errors.append(f"{label}.source_passed must be a boolean or null.")
    if artifact.get("role") == "trainer_preflight":
        if artifact.get("schema_version") != TRAINER_PREFLIGHT_SCHEMA_VERSION:
            target.errors.append(f"{label}.schema_version must be {TRAINER_PREFLIGHT_SCHEMA_VERSION}.")
    if artifact.get("role") == "trainer_launch_check":
        if artifact.get("schema_version") != TRAINER_LAUNCH_CHECK_SCHEMA_VERSION:
            target.errors.append(f"{label}.schema_version must be {TRAINER_LAUNCH_CHECK_SCHEMA_VERSION}.")

    artifact_path = _archive_artifact_path(artifact.get("path"), archive_root)
    if artifact_path is None:
        target.errors.append(f"{label}.path must be a relative archive path.")
        return
    if not _path_resolves_inside(artifact_path, archive_root):
        target.errors.append(f"{label}.path must resolve inside the archive.")
        return
    if artifact_path.is_symlink():
        target.errors.append(f"{label}.path must not be a symlink.")
        return
    if artifact.get("kind") == "file":
        if not artifact_path.exists() or not artifact_path.is_file():
            target.errors.append(f"{label}.path does not exist as a file inside the archive.")
            return
        if artifact_path.stat().st_size != artifact.get("size_bytes"):
            target.errors.append(f"{label}.size_bytes does not match the archived file.")
        if _sha256(artifact_path) != artifact.get("sha256"):
            target.errors.append(f"{label}.sha256 does not match the archived file.")
        return
    if not artifact_path.exists() or not artifact_path.is_dir():
        target.errors.append(f"{label}.path does not exist as a directory inside the archive.")
        return
    for child in artifact_path.rglob("*"):
        if child.is_symlink():
            target.errors.append(f"{label}.path contains symlink {child}.")
            return
    if artifact.get("tree_hash_algorithm") != "sha256(sorted-relative-path-size-file-sha256)":
        target.errors.append(f"{label}.tree_hash_algorithm is invalid.")
    if not _is_non_negative_int(artifact.get("file_count")):
        target.errors.append(f"{label}.file_count must be a non-negative integer for directories.")
        return
    tree = _trainer_archive_tree_fingerprint(artifact_path)
    if artifact.get("file_count") != tree["file_count"]:
        target.errors.append(f"{label}.file_count does not match the archived directory.")
    if artifact.get("size_bytes") != tree["size_bytes"]:
        target.errors.append(f"{label}.size_bytes does not match the archived directory.")
    if artifact.get("sha256") != tree["sha256"]:
        target.errors.append(f"{label}.sha256 does not match the archived directory tree.")


def _validate_trainer_archive_missing(item: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(item, dict):
        target.errors.append(f"{label} must be an object.")
        return
    valid_roles = {"trainer_launch_check", "gate", "validation_summary", "trainer_artifact", "schema_contract"}
    if item.get("role") not in valid_roles:
        target.errors.append(f"{label}.role is invalid.")
    if not _is_non_negative_int(item.get("index")):
        target.errors.append(f"{label}.index must be a non-negative integer.")
    if "name" in item and not isinstance(item.get("name"), str):
        target.errors.append(f"{label}.name must be a string when present.")
    if not isinstance(item.get("reason"), str) or not item.get("reason"):
        target.errors.append(f"{label}.reason must be a non-empty string.")


def _validate_trainer_archive_metrics(
    metrics: dict[str, Any],
    artifacts: list[Any],
    missing: list[Any],
    consumer_contract: Any,
    target: ValidationTarget,
) -> None:
    valid_artifacts = [artifact for artifact in artifacts if isinstance(artifact, dict)]
    valid_missing = [item for item in missing if isinstance(item, dict)]
    expected = {
        "artifact_count": len(artifacts),
        "file_artifact_count": sum(1 for artifact in valid_artifacts if artifact.get("kind") == "file"),
        "directory_artifact_count": sum(1 for artifact in valid_artifacts if artifact.get("kind") == "directory"),
        "trainer_input_count": sum(1 for artifact in valid_artifacts if artifact.get("role") == "trainer_artifact"),
        "path_rewrite_count": len(
            [
                artifact
                for artifact in valid_artifacts
                if artifact.get("role") == "trainer_artifact"
                and isinstance(artifact.get("original_path"), str)
                and artifact.get("original_path")
                and not str(artifact.get("original_path")).startswith("<")
            ]
        ),
        "external_command_path_count": consumer_contract.get("external_command_path_count", 0) if isinstance(consumer_contract, dict) else 0,
        "missing_count": len(missing),
        "total_size_bytes": sum(artifact.get("size_bytes", 0) for artifact in valid_artifacts if _is_non_negative_int(artifact.get("size_bytes"))),
        "unique_sha256_count": len({artifact.get("sha256") for artifact in valid_artifacts if isinstance(artifact.get("sha256"), str)}),
    }
    for field_name, expected_value in expected.items():
        if metrics.get(field_name) != expected_value:
            target.errors.append(f"trainer_archive.metrics.{field_name} expected {expected_value}, got {metrics.get(field_name)!r}.")
    role_counts = _trainer_archive_count_map(artifact.get("role") for artifact in valid_artifacts)
    missing_role_counts = _trainer_archive_count_map(item.get("role") for item in valid_missing)
    _validate_action_ledger_count_rows(metrics.get("role_counts"), role_counts, target, "trainer_archive.metrics.role_counts")
    _validate_action_ledger_count_rows(
        metrics.get("missing_role_counts"),
        missing_role_counts,
        target,
        "trainer_archive.metrics.missing_role_counts",
    )


def _validate_trainer_archive_inputs(value: Any, artifacts: list[dict[str, Any]], target: ValidationTarget) -> None:
    if not isinstance(value, list):
        target.errors.append("trainer_archive.trainer_inputs must be a list.")
        return
    expected = [_trainer_input_from_artifact(artifact) for artifact in artifacts if artifact.get("role") == "trainer_artifact"]
    if len(value) != len(expected):
        target.errors.append(f"trainer_archive.trainer_inputs expected {len(expected)} item(s), got {len(value)}.")
    for index, item in enumerate(value):
        label = f"trainer_archive.trainer_inputs[{index}]"
        if not isinstance(item, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        for field_name in ("artifact_name", "kind", "original_path", "archive_path", "sha256"):
            if not isinstance(item.get(field_name), str) or not item.get(field_name):
                target.errors.append(f"{label}.{field_name} must be a non-empty string.")
        for field_name in ("artifact_index", "size_bytes"):
            if not _is_non_negative_int(item.get(field_name)):
                target.errors.append(f"{label}.{field_name} must be a non-negative integer.")
        if item.get("kind") == "directory":
            if not _is_non_negative_int(item.get("file_count")):
                target.errors.append(f"{label}.file_count must be a non-negative integer for directories.")
            if item.get("tree_hash_algorithm") != "sha256(sorted-relative-path-size-file-sha256)":
                target.errors.append(f"{label}.tree_hash_algorithm is invalid for directories.")
        if index < len(expected):
            for field_name, expected_value in expected[index].items():
                if item.get(field_name) != expected_value:
                    target.errors.append(f"{label}.{field_name} expected {expected_value!r}, got {item.get(field_name)!r}.")


def _validate_trainer_archive_rewrites(value: Any, inputs: Any, target: ValidationTarget) -> None:
    if not isinstance(value, list):
        target.errors.append("trainer_archive.path_rewrites must be a list.")
        return
    expected = _expected_trainer_archive_rewrites(inputs if isinstance(inputs, list) else [])
    if len(value) != len(expected):
        target.errors.append(f"trainer_archive.path_rewrites expected {len(expected)} item(s), got {len(value)}.")
    for index, item in enumerate(value):
        label = f"trainer_archive.path_rewrites[{index}]"
        if not isinstance(item, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        for field_name in ("artifact_name", "kind", "original_path", "archive_path"):
            if not isinstance(item.get(field_name), str) or not item.get(field_name):
                target.errors.append(f"{label}.{field_name} must be a non-empty string.")
        if index < len(expected):
            for field_name, expected_value in expected[index].items():
                if item.get(field_name) != expected_value:
                    target.errors.append(f"{label}.{field_name} expected {expected_value!r}, got {item.get(field_name)!r}.")


def _validate_trainer_archive_commands(
    approved_command: Any,
    portable_command: Any,
    path_rewrites: Any,
    target: ValidationTarget,
) -> None:
    if not isinstance(approved_command, dict):
        target.errors.append("trainer_archive.approved_command must be an object.")
        approved_command = {}
    for field_name in ("approved", "provided", "parseable"):
        if not isinstance(approved_command.get(field_name), bool):
            target.errors.append(f"trainer_archive.approved_command.{field_name} must be a boolean.")
    if not isinstance(approved_command.get("raw"), str):
        target.errors.append("trainer_archive.approved_command.raw must be a string.")
    if not isinstance(approved_command.get("shell"), str):
        target.errors.append("trainer_archive.approved_command.shell must be a string.")
    argv = approved_command.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        target.errors.append("trainer_archive.approved_command.argv must be a list of strings.")
        argv = []

    if not isinstance(portable_command, dict):
        target.errors.append("trainer_archive.portable_command must be an object.")
        return
    for field_name in ("approved", "available", "rewritten"):
        if not isinstance(portable_command.get(field_name), bool):
            target.errors.append(f"trainer_archive.portable_command.{field_name} must be a boolean.")
    if not _is_non_negative_int(portable_command.get("path_rewrite_count")):
        target.errors.append("trainer_archive.portable_command.path_rewrite_count must be a non-negative integer.")
    if not isinstance(portable_command.get("shell"), str):
        target.errors.append("trainer_archive.portable_command.shell must be a string.")
    if not _is_string_list(portable_command.get("notes")):
        target.errors.append("trainer_archive.portable_command.notes must be a list of strings.")
    portable_argv = portable_command.get("argv")
    if not isinstance(portable_argv, list) or not all(isinstance(item, str) for item in portable_argv):
        target.errors.append("trainer_archive.portable_command.argv must be a list of strings.")
        portable_argv = []

    rewrites = path_rewrites if isinstance(path_rewrites, list) else []
    expected_argv, expected_rewrite_count = _rewrite_trainer_archive_command_argv([item for item in argv if isinstance(item, str)], rewrites)
    if portable_argv != expected_argv:
        target.errors.append("trainer_archive.portable_command.argv must match approved_command.argv rewritten through path_rewrites.")
    if portable_command.get("shell") != (shlex.join(expected_argv) if expected_argv else ""):
        target.errors.append("trainer_archive.portable_command.shell must match the rewritten argv.")
    if portable_command.get("path_rewrite_count") != expected_rewrite_count:
        target.errors.append(
            f"trainer_archive.portable_command.path_rewrite_count expected {expected_rewrite_count}, got {portable_command.get('path_rewrite_count')!r}."
        )
    if portable_command.get("rewritten") != (expected_rewrite_count > 0):
        target.errors.append("trainer_archive.portable_command.rewritten must match path_rewrite_count.")
    if portable_command.get("available") != bool(expected_argv):
        target.errors.append("trainer_archive.portable_command.available must match whether argv is present.")
    if portable_command.get("approved") != (approved_command.get("approved") is True):
        target.errors.append("trainer_archive.portable_command.approved must match approved_command.approved.")


def _validate_trainer_archive_consumer_contract(
    contract: Any,
    portable_command: Any,
    trainer_inputs: Any,
    path_rewrites: Any,
    target: ValidationTarget,
) -> None:
    if not isinstance(contract, dict):
        target.errors.append("trainer_archive.consumer_contract must be an object.")
        return
    if contract.get("execution_cwd") != "archive_root":
        target.errors.append("trainer_archive.consumer_contract.execution_cwd must be archive_root.")
    if contract.get("command_kind") != "advisory_portable_command":
        target.errors.append("trainer_archive.consumer_contract.command_kind must be advisory_portable_command.")
    for field_name in ("portable_command_available", "portable_command_rewritten", "external_code_required"):
        if not isinstance(contract.get(field_name), bool):
            target.errors.append(f"trainer_archive.consumer_contract.{field_name} must be a boolean.")
    for field_name in ("trainer_input_count", "path_rewrite_count", "external_command_path_count"):
        if not _is_non_negative_int(contract.get(field_name)):
            target.errors.append(f"trainer_archive.consumer_contract.{field_name} must be a non-negative integer.")
    if not _is_string_list(contract.get("notes")):
        target.errors.append("trainer_archive.consumer_contract.notes must be a list of strings.")
    inputs = trainer_inputs if isinstance(trainer_inputs, list) else []
    rewrites = path_rewrites if isinstance(path_rewrites, list) else []
    portable = portable_command if isinstance(portable_command, dict) else {}
    portable_argv = portable.get("argv") if isinstance(portable.get("argv"), list) else []
    clean_portable_argv = [item for item in portable_argv if isinstance(item, str)]
    expected_external = _trainer_archive_external_command_paths(clean_portable_argv, inputs)
    expected = {
        "portable_command_available": portable.get("available") is True,
        "portable_command_rewritten": portable.get("rewritten") is True,
        "trainer_input_count": len(inputs),
        "path_rewrite_count": len(rewrites),
        "external_code_required": bool(expected_external),
        "external_command_path_count": len(expected_external),
    }
    for field_name, expected_value in expected.items():
        if contract.get(field_name) != expected_value:
            target.errors.append(f"trainer_archive.consumer_contract.{field_name} expected {expected_value!r}, got {contract.get(field_name)!r}.")

    external_paths = contract.get("external_command_paths")
    if not isinstance(external_paths, list):
        target.errors.append("trainer_archive.consumer_contract.external_command_paths must be a list.")
        return
    if len(external_paths) != len(expected_external):
        target.errors.append(
            f"trainer_archive.consumer_contract.external_command_paths expected {len(expected_external)} item(s), got {len(external_paths)}."
        )
    for index, item in enumerate(external_paths):
        label = f"trainer_archive.consumer_contract.external_command_paths[{index}]"
        if not isinstance(item, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        if not _is_non_negative_int(item.get("argv_index")):
            target.errors.append(f"{label}.argv_index must be a non-negative integer.")
        for field_name in ("token", "path", "reason"):
            if not isinstance(item.get(field_name), str) or not item.get(field_name):
                target.errors.append(f"{label}.{field_name} must be a non-empty string.")
        if index < len(expected_external):
            for field_name, expected_value in expected_external[index].items():
                if item.get(field_name) != expected_value:
                    target.errors.append(f"{label}.{field_name} expected {expected_value!r}, got {item.get(field_name)!r}.")


def _validate_trainer_archive_check(check: dict[str, Any], target: ValidationTarget) -> None:
    _require_equal(check, "schema_version", TRAINER_ARCHIVE_CHECK_SCHEMA_VERSION, target)
    for field_name in ("archive_path", "manifest_path", "readiness", "recommendation"):
        if not isinstance(check.get(field_name), str) or not check.get(field_name):
            target.errors.append(f"trainer_archive_check.{field_name} must be a non-empty string.")
    if not isinstance(check.get("passed"), bool):
        target.errors.append("trainer_archive_check.passed must be a boolean.")
    checks = check.get("checks")
    if not isinstance(checks, list):
        target.errors.append("trainer_archive_check.checks must be a list.")
        checks = []
    validation = check.get("validation")
    if not isinstance(validation, dict):
        target.errors.append("trainer_archive_check.validation must be an object.")
        validation = {}
    metrics = check.get("metrics")
    if not isinstance(metrics, dict):
        target.errors.append("trainer_archive_check.metrics must be an object.")
        metrics = {}
    if not _is_string_list(check.get("notes")):
        target.errors.append("trainer_archive_check.notes must be a list of strings.")

    failed_checks = _validate_gate_like_checks(checks, target, "trainer_archive_check.checks")
    if check.get("check_count") != len(checks):
        target.errors.append(f"trainer_archive_check.check_count expected {len(checks)}, got {check.get('check_count')!r}.")
    if check.get("failed_check_count") != failed_checks:
        target.errors.append(
            f"trainer_archive_check.failed_check_count expected {failed_checks}, got {check.get('failed_check_count')!r}."
        )
    expected_passed = failed_checks == 0
    if isinstance(check.get("passed"), bool) and check.get("passed") != expected_passed:
        target.errors.append("trainer_archive_check.passed must match failed_check_count.")
    expected_readiness = "ready" if expected_passed else "blocked"
    expected_recommendation = "consumer_ready" if expected_passed else "block_consumer_launch"
    if check.get("readiness") != expected_readiness:
        target.errors.append(f"trainer_archive_check.readiness expected {expected_readiness!r}, got {check.get('readiness')!r}.")
    if check.get("recommendation") != expected_recommendation:
        target.errors.append(
            f"trainer_archive_check.recommendation expected {expected_recommendation!r}, got {check.get('recommendation')!r}."
        )

    _validate_trainer_archive_check_validation(validation, target)
    _validate_trainer_archive_check_archive(check.get("archive"), target)
    _validate_trainer_archive_check_external_root(check.get("external_code_root"), target)
    _validate_trainer_archive_check_portable_command(check.get("portable_command"), target)
    _validate_trainer_archive_check_consumer_contract(check.get("consumer_contract"), target)
    external_code_checks = _validate_trainer_archive_check_external_code(check.get("external_code_checks"), target)
    trainer_input_checks = _validate_trainer_archive_check_inputs(check.get("trainer_input_checks"), target)
    _validate_trainer_archive_check_metrics(metrics, validation, external_code_checks, trainer_input_checks, len(checks), failed_checks, target)
    target.details.update(
        {
            "passed": check.get("passed"),
            "check_count": len(checks),
            "failed_check_count": failed_checks,
            "external_command_path_count": metrics.get("external_command_path_count"),
            "missing_external_code_count": metrics.get("missing_external_code_count"),
            "trainer_input_count": metrics.get("trainer_input_count"),
        }
    )


def _validate_trainer_archive_check_validation(value: dict[str, Any], target: ValidationTarget) -> None:
    for field_name in ("available", "passed", "strict"):
        if not isinstance(value.get(field_name), bool):
            target.errors.append(f"trainer_archive_check.validation.{field_name} must be a boolean.")
    for field_name in ("target_count", "error_count", "warning_count"):
        if not _is_non_negative_int(value.get(field_name)):
            target.errors.append(f"trainer_archive_check.validation.{field_name} must be a non-negative integer.")
    if not _is_string_list(value.get("errors")):
        target.errors.append("trainer_archive_check.validation.errors must be a list of strings.")
    if not _is_string_list(value.get("warnings")):
        target.errors.append("trainer_archive_check.validation.warnings must be a list of strings.")


def _validate_trainer_archive_check_archive(value: Any, target: ValidationTarget) -> None:
    if not isinstance(value, dict):
        target.errors.append("trainer_archive_check.archive must be an object.")
        return
    for field_name in ("path", "manifest_path", "schema_version"):
        if not isinstance(value.get(field_name), str) or not value.get(field_name):
            target.errors.append(f"trainer_archive_check.archive.{field_name} must be a non-empty string.")
    if value.get("schema_version") != TRAINER_ARCHIVE_SCHEMA_VERSION:
        target.errors.append(f"trainer_archive_check.archive.schema_version must be {TRAINER_ARCHIVE_SCHEMA_VERSION}.")
    for field_name in ("passed", "self_contained", "ready_for_training"):
        if not isinstance(value.get(field_name), bool):
            target.errors.append(f"trainer_archive_check.archive.{field_name} must be a boolean.")
    for field_name in ("trainer_input_count", "external_command_path_count"):
        if not _is_non_negative_int(value.get(field_name)):
            target.errors.append(f"trainer_archive_check.archive.{field_name} must be a non-negative integer.")


def _validate_trainer_archive_check_external_root(value: Any, target: ValidationTarget) -> None:
    if not isinstance(value, dict):
        target.errors.append("trainer_archive_check.external_code_root must be an object.")
        return
    for field_name in ("path", "kind"):
        if not isinstance(value.get(field_name), str) or not value.get(field_name):
            target.errors.append(f"trainer_archive_check.external_code_root.{field_name} must be a non-empty string.")
    if value.get("kind") != "directory":
        target.errors.append("trainer_archive_check.external_code_root.kind must be directory.")
    for field_name in ("exists", "regular_directory", "symlink"):
        if not isinstance(value.get(field_name), bool):
            target.errors.append(f"trainer_archive_check.external_code_root.{field_name} must be a boolean.")


def _validate_trainer_archive_check_portable_command(value: Any, target: ValidationTarget) -> None:
    if not isinstance(value, dict):
        target.errors.append("trainer_archive_check.portable_command must be an object.")
        return
    for field_name in ("approved", "available", "rewritten"):
        if not isinstance(value.get(field_name), bool):
            target.errors.append(f"trainer_archive_check.portable_command.{field_name} must be a boolean.")
    if not _is_non_negative_int(value.get("path_rewrite_count")):
        target.errors.append("trainer_archive_check.portable_command.path_rewrite_count must be a non-negative integer.")
    if not isinstance(value.get("shell"), str):
        target.errors.append("trainer_archive_check.portable_command.shell must be a string.")
    if not _is_string_list(value.get("argv")):
        target.errors.append("trainer_archive_check.portable_command.argv must be a list of strings.")


def _validate_trainer_archive_check_consumer_contract(value: Any, target: ValidationTarget) -> None:
    if not isinstance(value, dict):
        target.errors.append("trainer_archive_check.consumer_contract must be an object.")
        return
    for field_name in ("execution_cwd", "command_kind"):
        if not isinstance(value.get(field_name), str):
            target.errors.append(f"trainer_archive_check.consumer_contract.{field_name} must be a string.")
    for field_name in ("portable_command_available", "external_code_required"):
        if not isinstance(value.get(field_name), bool):
            target.errors.append(f"trainer_archive_check.consumer_contract.{field_name} must be a boolean.")
    for field_name in ("trainer_input_count", "path_rewrite_count", "external_command_path_count"):
        if not _is_non_negative_int(value.get(field_name)):
            target.errors.append(f"trainer_archive_check.consumer_contract.{field_name} must be a non-negative integer.")
    external_paths = value.get("external_command_paths")
    if not isinstance(external_paths, list):
        target.errors.append("trainer_archive_check.consumer_contract.external_command_paths must be a list.")
        return
    for index, item in enumerate(external_paths):
        label = f"trainer_archive_check.consumer_contract.external_command_paths[{index}]"
        if not isinstance(item, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        if not _is_non_negative_int(item.get("argv_index")):
            target.errors.append(f"{label}.argv_index must be a non-negative integer.")
        for field_name in ("token", "path", "reason"):
            if not isinstance(item.get(field_name), str):
                target.errors.append(f"{label}.{field_name} must be a string.")


def _validate_trainer_archive_check_external_code(value: Any, target: ValidationTarget) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        target.errors.append("trainer_archive_check.external_code_checks must be a list.")
        return []
    records = [item for item in value if isinstance(item, dict)]
    for index, item in enumerate(value):
        label = f"trainer_archive_check.external_code_checks[{index}]"
        if not isinstance(item, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        if item.get("index") != index:
            target.errors.append(f"{label}.index expected {index}, got {item.get('index')!r}.")
        if not _is_non_negative_int(item.get("argv_index")):
            target.errors.append(f"{label}.argv_index must be a non-negative integer.")
        for field_name in ("token", "path", "resolved_path", "kind", "reason"):
            if not isinstance(item.get(field_name), str):
                target.errors.append(f"{label}.{field_name} must be a string.")
        if item.get("kind") != "file":
            target.errors.append(f"{label}.kind must be file.")
        for field_name in ("exists", "regular_file", "symlink", "passed"):
            if not isinstance(item.get(field_name), bool):
                target.errors.append(f"{label}.{field_name} must be a boolean.")
        expected_passed = item.get("exists") is True and item.get("regular_file") is True and item.get("symlink") is False
        if item.get("passed") is True and not expected_passed:
            target.errors.append(f"{label}.passed cannot be true unless the resolved path is a regular non-symlink file.")
        if item.get("passed") is True:
            if not _is_non_negative_int(item.get("size_bytes")):
                target.errors.append(f"{label}.size_bytes must be a non-negative integer when passed.")
            if not _is_sha256(item.get("sha256")):
                target.errors.append(f"{label}.sha256 must be a SHA-256 hex string when passed.")
            _validate_visible_file_hash(item, target, label)
    return records


def _validate_trainer_archive_check_inputs(value: Any, target: ValidationTarget) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        target.errors.append("trainer_archive_check.trainer_input_checks must be a list.")
        return []
    records = [item for item in value if isinstance(item, dict)]
    for index, item in enumerate(value):
        label = f"trainer_archive_check.trainer_input_checks[{index}]"
        if not isinstance(item, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        if item.get("index") != index:
            target.errors.append(f"{label}.index expected {index}, got {item.get('index')!r}.")
        if not _is_non_negative_int(item.get("artifact_index")):
            target.errors.append(f"{label}.artifact_index must be a non-negative integer.")
        for field_name in ("artifact_name", "archive_path", "resolved_path", "kind", "expected_sha256", "reason"):
            if not isinstance(item.get(field_name), str):
                target.errors.append(f"{label}.{field_name} must be a string.")
        if item.get("kind") not in {"file", "directory"}:
            target.errors.append(f"{label}.kind must be file or directory.")
        for field_name in ("exists", "regular_file", "regular_directory", "symlink", "passed"):
            if not isinstance(item.get(field_name), bool):
                target.errors.append(f"{label}.{field_name} must be a boolean.")
        if item.get("expected_sha256") and not _is_sha256(item.get("expected_sha256")):
            target.errors.append(f"{label}.expected_sha256 must be a SHA-256 hex string when present.")
        if item.get("passed") is True:
            if item.get("kind") == "file":
                if item.get("regular_file") is not True or item.get("symlink") is True:
                    target.errors.append(f"{label}.passed file checks must be regular non-symlink files.")
                if not _is_non_negative_int(item.get("size_bytes")):
                    target.errors.append(f"{label}.size_bytes must be a non-negative integer when passed.")
                if not _is_sha256(item.get("sha256")):
                    target.errors.append(f"{label}.sha256 must be a SHA-256 hex string when passed.")
                _validate_visible_file_hash(item, target, label)
            else:
                if item.get("regular_directory") is not True or item.get("symlink") is True:
                    target.errors.append(f"{label}.passed directory checks must be regular non-symlink directories.")
                if not _is_non_negative_int(item.get("size_bytes")):
                    target.errors.append(f"{label}.size_bytes must be a non-negative integer when passed.")
                if not _is_non_negative_int(item.get("file_count")):
                    target.errors.append(f"{label}.file_count must be a non-negative integer when passed.")
                if not _is_sha256(item.get("sha256")):
                    target.errors.append(f"{label}.sha256 must be a SHA-256 hex string when passed.")
                _validate_visible_directory_hash(item, target, label)
    return records


def _validate_trainer_archive_check_metrics(
    metrics: dict[str, Any],
    validation: dict[str, Any],
    external_code_checks: list[dict[str, Any]],
    trainer_input_checks: list[dict[str, Any]],
    check_count: int,
    failed_check_count: int,
    target: ValidationTarget,
) -> None:
    external_count = len(external_code_checks)
    external_available = sum(1 for item in external_code_checks if item.get("passed") is True)
    input_count = len(trainer_input_checks)
    inputs_available = sum(1 for item in trainer_input_checks if item.get("passed") is True)
    expected = {
        "archive_validation_passed": validation.get("passed") is True,
        "archive_validation_error_count": _non_negative_int_value(validation.get("error_count")),
        "archive_validation_warning_count": _non_negative_int_value(validation.get("warning_count")),
        "external_command_path_count": external_count,
        "external_code_file_count": external_available,
        "missing_external_code_count": external_count - external_available,
        "trainer_input_count": input_count,
        "trainer_input_available_count": inputs_available,
        "missing_trainer_input_count": input_count - inputs_available,
        "check_count": check_count,
        "failed_check_count": failed_check_count,
    }
    for field_name, expected_value in expected.items():
        if metrics.get(field_name) != expected_value:
            target.errors.append(f"trainer_archive_check.metrics.{field_name} expected {expected_value}, got {metrics.get(field_name)!r}.")
    if not _is_non_negative_int(metrics.get("relative_external_command_path_count")):
        target.errors.append("trainer_archive_check.metrics.relative_external_command_path_count must be a non-negative integer.")


def _validate_visible_file_hash(item: dict[str, Any], target: ValidationTarget, label: str) -> None:
    path = _visible_local_path(item.get("resolved_path"))
    if path is None or not path.exists():
        return
    if path.is_symlink() or not path.is_file():
        target.errors.append(f"{label}.resolved_path is not a regular file on disk.")
        return
    if item.get("size_bytes") != path.stat().st_size:
        target.errors.append(f"{label}.size_bytes does not match resolved_path.")
    if item.get("sha256") != _sha256(path):
        target.errors.append(f"{label}.sha256 does not match resolved_path.")


def _validate_visible_directory_hash(item: dict[str, Any], target: ValidationTarget, label: str) -> None:
    path = _visible_local_path(item.get("resolved_path"))
    if path is None or not path.exists():
        return
    if path.is_symlink() or not path.is_dir():
        target.errors.append(f"{label}.resolved_path is not a regular directory on disk.")
        return
    for child in path.rglob("*"):
        if child.is_symlink():
            target.errors.append(f"{label}.resolved_path contains symlink {child}.")
            return
    tree = _trainer_archive_tree_fingerprint(path)
    if item.get("file_count") != tree["file_count"]:
        target.errors.append(f"{label}.file_count does not match resolved_path.")
    if item.get("size_bytes") != tree["size_bytes"]:
        target.errors.append(f"{label}.size_bytes does not match resolved_path.")
    if item.get("sha256") != tree["sha256"]:
        target.errors.append(f"{label}.sha256 does not match resolved_path.")


def _visible_local_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value or value.startswith("<") or _is_windows_absolute(value):
        return None
    return Path(value)


def _validate_trainer_consumer_plan(plan: dict[str, Any], target: ValidationTarget) -> None:
    _require_equal(plan, "schema_version", TRAINER_CONSUMER_PLAN_SCHEMA_VERSION, target)
    for field_name in ("plan_path", "archive_check_path", "readiness", "recommendation"):
        if not isinstance(plan.get(field_name), str) or not plan.get(field_name):
            target.errors.append(f"trainer_consumer_plan.{field_name} must be a non-empty string.")
    if not isinstance(plan.get("passed"), bool):
        target.errors.append("trainer_consumer_plan.passed must be a boolean.")
    checks = plan.get("checks")
    if not isinstance(checks, list):
        target.errors.append("trainer_consumer_plan.checks must be a list.")
        checks = []
    validation = plan.get("validation")
    if not isinstance(validation, dict):
        target.errors.append("trainer_consumer_plan.validation must be an object.")
        validation = {}
    metrics = plan.get("metrics")
    if not isinstance(metrics, dict):
        target.errors.append("trainer_consumer_plan.metrics must be an object.")
        metrics = {}
    if not _is_string_list(plan.get("notes")):
        target.errors.append("trainer_consumer_plan.notes must be a list of strings.")
    if not _is_string_list(plan.get("blocked_reasons")):
        target.errors.append("trainer_consumer_plan.blocked_reasons must be a list of strings.")

    failed_checks = _validate_gate_like_checks(checks, target, "trainer_consumer_plan.checks")
    if plan.get("check_count") != len(checks):
        target.errors.append(f"trainer_consumer_plan.check_count expected {len(checks)}, got {plan.get('check_count')!r}.")
    if plan.get("failed_check_count") != failed_checks:
        target.errors.append(
            f"trainer_consumer_plan.failed_check_count expected {failed_checks}, got {plan.get('failed_check_count')!r}."
        )
    expected_passed = failed_checks == 0
    if isinstance(plan.get("passed"), bool) and plan.get("passed") != expected_passed:
        target.errors.append("trainer_consumer_plan.passed must match failed_check_count.")
    expected_readiness = "ready" if expected_passed else "blocked"
    expected_recommendation = "ready_for_external_trainer" if expected_passed else "block_external_trainer"
    if plan.get("readiness") != expected_readiness:
        target.errors.append(f"trainer_consumer_plan.readiness expected {expected_readiness!r}, got {plan.get('readiness')!r}.")
    if plan.get("recommendation") != expected_recommendation:
        target.errors.append(
            f"trainer_consumer_plan.recommendation expected {expected_recommendation!r}, got {plan.get('recommendation')!r}."
        )

    _validate_trainer_consumer_plan_validation(validation, target)
    _validate_trainer_consumer_plan_source(plan.get("source_archive_check"), target)
    execution_counts = _validate_trainer_consumer_plan_execution(plan.get("execution"), target)
    _validate_trainer_consumer_plan_handoff(plan.get("handoff_contract"), execution_counts, target)
    _validate_trainer_consumer_plan_metrics(metrics, validation, execution_counts, len(checks), failed_checks, target)
    target.details.update(
        {
            "passed": plan.get("passed"),
            "check_count": len(checks),
            "failed_check_count": failed_checks,
            "trainer_input_count": metrics.get("trainer_input_count"),
            "external_code_file_count": metrics.get("external_code_file_count"),
        }
    )


def _validate_trainer_consumer_plan_validation(value: dict[str, Any], target: ValidationTarget) -> None:
    for field_name in ("available", "passed", "strict"):
        if not isinstance(value.get(field_name), bool):
            target.errors.append(f"trainer_consumer_plan.validation.{field_name} must be a boolean.")
    for field_name in ("target_count", "error_count", "warning_count"):
        if not _is_non_negative_int(value.get(field_name)):
            target.errors.append(f"trainer_consumer_plan.validation.{field_name} must be a non-negative integer.")
    if not _is_string_list(value.get("errors")):
        target.errors.append("trainer_consumer_plan.validation.errors must be a list of strings.")
    if not _is_string_list(value.get("warnings")):
        target.errors.append("trainer_consumer_plan.validation.warnings must be a list of strings.")


def _validate_trainer_consumer_plan_source(value: Any, target: ValidationTarget) -> None:
    if not isinstance(value, dict):
        target.errors.append("trainer_consumer_plan.source_archive_check must be an object.")
        return
    for field_name in ("path", "schema_version", "readiness", "recommendation"):
        if not isinstance(value.get(field_name), str) or not value.get(field_name):
            target.errors.append(f"trainer_consumer_plan.source_archive_check.{field_name} must be a non-empty string.")
    if value.get("schema_version") != TRAINER_ARCHIVE_CHECK_SCHEMA_VERSION:
        target.errors.append(f"trainer_consumer_plan.source_archive_check.schema_version must be {TRAINER_ARCHIVE_CHECK_SCHEMA_VERSION}.")
    if not isinstance(value.get("passed"), bool):
        target.errors.append("trainer_consumer_plan.source_archive_check.passed must be a boolean.")
    if "size_bytes" in value and not _is_non_negative_int(value.get("size_bytes")):
        target.errors.append("trainer_consumer_plan.source_archive_check.size_bytes must be a non-negative integer when present.")
    if "sha256" in value and not _is_sha256(value.get("sha256")):
        target.errors.append("trainer_consumer_plan.source_archive_check.sha256 must be a SHA-256 hex string when present.")
    path = _visible_local_path(value.get("path"))
    if path is None or not path.exists():
        return
    if path.is_symlink() or not path.is_file():
        target.errors.append("trainer_consumer_plan.source_archive_check.path must resolve to a regular file.")
        return
    if value.get("size_bytes") != path.stat().st_size:
        target.errors.append("trainer_consumer_plan.source_archive_check.size_bytes does not match path.")
    if value.get("sha256") != _sha256(path):
        target.errors.append("trainer_consumer_plan.source_archive_check.sha256 does not match path.")


def _validate_trainer_consumer_plan_execution(value: Any, target: ValidationTarget) -> dict[str, int]:
    counts = {
        "command_arg_count": 0,
        "trainer_input_count": 0,
        "trainer_input_ready_count": 0,
        "external_code_file_count": 0,
        "external_code_ready_count": 0,
    }
    if not isinstance(value, dict):
        target.errors.append("trainer_consumer_plan.execution must be an object.")
        return counts
    for field_name in ("execution_cwd", "archive_root", "external_code_root", "command_shell"):
        if not isinstance(value.get(field_name), str):
            target.errors.append(f"trainer_consumer_plan.execution.{field_name} must be a string.")
    if value.get("execution_cwd") != "archive_root":
        target.errors.append("trainer_consumer_plan.execution.execution_cwd must be archive_root.")
    for field_name in ("command_approved", "command_available"):
        if not isinstance(value.get(field_name), bool):
            target.errors.append(f"trainer_consumer_plan.execution.{field_name} must be a boolean.")
    argv = value.get("command_argv")
    if not _is_string_list(argv):
        target.errors.append("trainer_consumer_plan.execution.command_argv must be a list of strings.")
        argv = []
    clean_argv = [item for item in argv if isinstance(item, str)]
    counts["command_arg_count"] = len(clean_argv)
    expected_shell = shlex.join(clean_argv) if clean_argv else ""
    if value.get("command_shell") != expected_shell:
        target.errors.append("trainer_consumer_plan.execution.command_shell must match command_argv.")
    external_code_files = value.get("external_code_files")
    if not isinstance(external_code_files, list):
        target.errors.append("trainer_consumer_plan.execution.external_code_files must be a list.")
        external_code_files = []
    trainer_inputs = value.get("trainer_inputs")
    if not isinstance(trainer_inputs, list):
        target.errors.append("trainer_consumer_plan.execution.trainer_inputs must be a list.")
        trainer_inputs = []
    counts["external_code_file_count"] = len([item for item in external_code_files if isinstance(item, dict)])
    counts["trainer_input_count"] = len([item for item in trainer_inputs if isinstance(item, dict)])
    for index, item in enumerate(external_code_files):
        label = f"trainer_consumer_plan.execution.external_code_files[{index}]"
        if not isinstance(item, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        _validate_trainer_consumer_plan_external_code_file(item, target, label)
        if item.get("passed") is True:
            counts["external_code_ready_count"] += 1
    for index, item in enumerate(trainer_inputs):
        label = f"trainer_consumer_plan.execution.trainer_inputs[{index}]"
        if not isinstance(item, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        _validate_trainer_consumer_plan_trainer_input(item, target, label)
        if item.get("passed") is True:
            counts["trainer_input_ready_count"] += 1
    return counts


def _validate_trainer_consumer_plan_external_code_file(item: dict[str, Any], target: ValidationTarget, label: str) -> None:
    for field_name in ("index", "argv_index"):
        if not _is_non_negative_int(item.get(field_name)):
            target.errors.append(f"{label}.{field_name} must be a non-negative integer.")
    for field_name in ("token", "path", "resolved_path", "reason"):
        if not isinstance(item.get(field_name), str):
            target.errors.append(f"{label}.{field_name} must be a string.")
    for field_name in ("exists", "regular_file", "symlink", "passed"):
        if not isinstance(item.get(field_name), bool):
            target.errors.append(f"{label}.{field_name} must be a boolean.")
    if item.get("passed") is True:
        if item.get("exists") is not True or item.get("regular_file") is not True or item.get("symlink") is True:
            target.errors.append(f"{label}.passed cannot be true unless the file is present, regular, and non-symlink.")
        if not _is_non_negative_int(item.get("size_bytes")):
            target.errors.append(f"{label}.size_bytes must be a non-negative integer when passed.")
        if not _is_sha256(item.get("sha256")):
            target.errors.append(f"{label}.sha256 must be a SHA-256 hex string when passed.")
        _validate_visible_file_hash(item, target, label)


def _validate_trainer_consumer_plan_trainer_input(item: dict[str, Any], target: ValidationTarget, label: str) -> None:
    for field_name in ("index", "artifact_index"):
        if not _is_non_negative_int(item.get(field_name)):
            target.errors.append(f"{label}.{field_name} must be a non-negative integer.")
    for field_name in ("artifact_name", "archive_path", "resolved_path", "kind", "expected_sha256", "reason"):
        if not isinstance(item.get(field_name), str):
            target.errors.append(f"{label}.{field_name} must be a string.")
    if item.get("kind") not in {"file", "directory"}:
        target.errors.append(f"{label}.kind must be file or directory.")
    for field_name in ("exists", "regular_file", "regular_directory", "symlink", "passed"):
        if not isinstance(item.get(field_name), bool):
            target.errors.append(f"{label}.{field_name} must be a boolean.")
    if item.get("expected_sha256") and not _is_sha256(item.get("expected_sha256")):
        target.errors.append(f"{label}.expected_sha256 must be a SHA-256 hex string when present.")
    if item.get("passed") is True:
        if not _is_non_negative_int(item.get("size_bytes")):
            target.errors.append(f"{label}.size_bytes must be a non-negative integer when passed.")
        if not _is_sha256(item.get("sha256")):
            target.errors.append(f"{label}.sha256 must be a SHA-256 hex string when passed.")
        if item.get("kind") == "file":
            _validate_visible_file_hash(item, target, label)
        else:
            if not _is_non_negative_int(item.get("file_count")):
                target.errors.append(f"{label}.file_count must be a non-negative integer when passed.")
            _validate_visible_directory_hash(item, target, label)


def _validate_trainer_consumer_plan_handoff(value: Any, counts: dict[str, int], target: ValidationTarget) -> None:
    if not isinstance(value, dict):
        target.errors.append("trainer_consumer_plan.handoff_contract must be an object.")
        return
    if value.get("flight_recorder_executed_command") is not False:
        target.errors.append("trainer_consumer_plan.handoff_contract.flight_recorder_executed_command must be false.")
    if value.get("runner_owns_execution") is not True:
        target.errors.append("trainer_consumer_plan.handoff_contract.runner_owns_execution must be true.")
    for field_name in ("runner_must_run_from", "runner_must_require_recommendation"):
        if not isinstance(value.get(field_name), str) or not value.get(field_name):
            target.errors.append(f"trainer_consumer_plan.handoff_contract.{field_name} must be a non-empty string.")
    if value.get("runner_must_run_from") != "archive_root":
        target.errors.append("trainer_consumer_plan.handoff_contract.runner_must_run_from must be archive_root.")
    if value.get("runner_must_require_recommendation") != "ready_for_external_trainer":
        target.errors.append(
            "trainer_consumer_plan.handoff_contract.runner_must_require_recommendation must be ready_for_external_trainer."
        )
    if value.get("trainer_input_count") != counts["trainer_input_count"]:
        target.errors.append(
            f"trainer_consumer_plan.handoff_contract.trainer_input_count expected {counts['trainer_input_count']}, "
            f"got {value.get('trainer_input_count')!r}."
        )
    if value.get("external_code_file_count") != counts["external_code_file_count"]:
        target.errors.append(
            f"trainer_consumer_plan.handoff_contract.external_code_file_count expected {counts['external_code_file_count']}, "
            f"got {value.get('external_code_file_count')!r}."
        )
    if not _is_string_list(value.get("allowed_input_sets")):
        target.errors.append("trainer_consumer_plan.handoff_contract.allowed_input_sets must be a list of strings.")
    if not _is_string_list(value.get("notes")):
        target.errors.append("trainer_consumer_plan.handoff_contract.notes must be a list of strings.")


def _validate_trainer_consumer_plan_metrics(
    metrics: dict[str, Any],
    validation: dict[str, Any],
    counts: dict[str, int],
    check_count: int,
    failed_check_count: int,
    target: ValidationTarget,
) -> None:
    expected = {
        "check_count": check_count,
        "failed_check_count": failed_check_count,
        "command_arg_count": counts["command_arg_count"],
        "trainer_input_count": counts["trainer_input_count"],
        "trainer_input_ready_count": counts["trainer_input_ready_count"],
        "external_code_file_count": counts["external_code_file_count"],
        "external_code_ready_count": counts["external_code_ready_count"],
        "archive_check_error_count": _non_negative_int_value(validation.get("error_count")),
        "archive_check_warning_count": _non_negative_int_value(validation.get("warning_count")),
    }
    for field_name, expected_value in expected.items():
        if metrics.get(field_name) != expected_value:
            target.errors.append(f"trainer_consumer_plan.metrics.{field_name} expected {expected_value}, got {metrics.get(field_name)!r}.")


def _validate_trainer_wrapper_dry_run(receipt: dict[str, Any], target: ValidationTarget) -> None:
    _require_equal(receipt, "schema_version", TRAINER_WRAPPER_DRY_RUN_SCHEMA_VERSION, target)
    for field_name in ("wrapper", "plan_path", "readiness", "recommendation"):
        if not isinstance(receipt.get(field_name), str) or not receipt.get(field_name):
            target.errors.append(f"trainer_wrapper_dry_run.{field_name} must be a non-empty string.")
    if not isinstance(receipt.get("passed"), bool):
        target.errors.append("trainer_wrapper_dry_run.passed must be a boolean.")
    checks = receipt.get("checks")
    if not isinstance(checks, list):
        target.errors.append("trainer_wrapper_dry_run.checks must be a list.")
        checks = []
    validation = receipt.get("validation")
    if not isinstance(validation, dict):
        target.errors.append("trainer_wrapper_dry_run.validation must be an object.")
        validation = {}
    would_run = receipt.get("would_run")
    if not isinstance(would_run, dict):
        target.errors.append("trainer_wrapper_dry_run.would_run must be an object.")
        would_run = {}
    inputs = receipt.get("inputs")
    if not isinstance(inputs, dict):
        target.errors.append("trainer_wrapper_dry_run.inputs must be an object.")
        inputs = {}
    metrics = receipt.get("metrics")
    if not isinstance(metrics, dict):
        target.errors.append("trainer_wrapper_dry_run.metrics must be an object.")
        metrics = {}
    if not _is_string_list(receipt.get("notes")):
        target.errors.append("trainer_wrapper_dry_run.notes must be a list of strings.")

    failed_checks = _validate_gate_like_checks(checks, target, "trainer_wrapper_dry_run.checks")
    if receipt.get("check_count") != len(checks):
        target.errors.append(f"trainer_wrapper_dry_run.check_count expected {len(checks)}, got {receipt.get('check_count')!r}.")
    if receipt.get("failed_check_count") != failed_checks:
        target.errors.append(
            f"trainer_wrapper_dry_run.failed_check_count expected {failed_checks}, got {receipt.get('failed_check_count')!r}."
        )
    expected_passed = failed_checks == 0
    if isinstance(receipt.get("passed"), bool) and receipt.get("passed") != expected_passed:
        target.errors.append("trainer_wrapper_dry_run.passed must match failed_check_count.")
    expected_readiness = "ready" if expected_passed else "blocked"
    expected_recommendation = "dry_run_ready" if expected_passed else "block_dry_run"
    if receipt.get("readiness") != expected_readiness:
        target.errors.append(f"trainer_wrapper_dry_run.readiness expected {expected_readiness!r}, got {receipt.get('readiness')!r}.")
    if receipt.get("recommendation") != expected_recommendation:
        target.errors.append(
            f"trainer_wrapper_dry_run.recommendation expected {expected_recommendation!r}, got {receipt.get('recommendation')!r}."
        )

    _validate_trainer_wrapper_validation(validation, target)
    command_arg_count = _validate_trainer_wrapper_would_run(would_run, target)
    input_counts = _validate_trainer_wrapper_inputs(inputs, target)
    _validate_trainer_wrapper_metrics(metrics, command_arg_count, input_counts, target)
    target.details.update(
        {
            "passed": receipt.get("passed"),
            "check_count": len(checks),
            "failed_check_count": failed_checks,
            "trainer_input_count": metrics.get("trainer_input_count"),
            "external_code_file_count": metrics.get("external_code_file_count"),
        }
    )


def _validate_trainer_wrapper_validation(value: dict[str, Any], target: ValidationTarget) -> None:
    for field_name in ("passed", "strict"):
        if not isinstance(value.get(field_name), bool):
            target.errors.append(f"trainer_wrapper_dry_run.validation.{field_name} must be a boolean.")
    for field_name in ("target_count", "error_count", "warning_count"):
        if not _is_non_negative_int(value.get(field_name)):
            target.errors.append(f"trainer_wrapper_dry_run.validation.{field_name} must be a non-negative integer.")


def _validate_trainer_wrapper_would_run(value: dict[str, Any], target: ValidationTarget) -> int:
    for field_name in ("mode", "execution_cwd", "archive_root", "external_code_root", "shell"):
        if not isinstance(value.get(field_name), str):
            target.errors.append(f"trainer_wrapper_dry_run.would_run.{field_name} must be a string.")
    if value.get("mode") != "dry_run":
        target.errors.append("trainer_wrapper_dry_run.would_run.mode must be dry_run.")
    if value.get("execution_cwd") not in {"", "archive_root"}:
        target.errors.append("trainer_wrapper_dry_run.would_run.execution_cwd must be archive_root for ready receipts.")
    argv = value.get("argv")
    if not _is_string_list(argv):
        target.errors.append("trainer_wrapper_dry_run.would_run.argv must be a list of strings.")
        argv = []
    clean_argv = [item for item in argv if isinstance(item, str)]
    expected_shell = shlex.join(clean_argv) if clean_argv else ""
    if value.get("shell") != expected_shell:
        target.errors.append("trainer_wrapper_dry_run.would_run.shell must match argv.")
    return len(clean_argv)


def _validate_trainer_wrapper_inputs(value: dict[str, Any], target: ValidationTarget) -> dict[str, int]:
    trainer_inputs = value.get("trainer_inputs")
    if not isinstance(trainer_inputs, list):
        target.errors.append("trainer_wrapper_dry_run.inputs.trainer_inputs must be a list.")
        trainer_inputs = []
    external_code_files = value.get("external_code_files")
    if not isinstance(external_code_files, list):
        target.errors.append("trainer_wrapper_dry_run.inputs.external_code_files must be a list.")
        external_code_files = []
    counts = {
        "trainer_input_count": 0,
        "trainer_input_ready_count": 0,
        "external_code_file_count": 0,
        "external_code_ready_count": 0,
    }
    for index, item in enumerate(trainer_inputs):
        label = f"trainer_wrapper_dry_run.inputs.trainer_inputs[{index}]"
        if not isinstance(item, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        counts["trainer_input_count"] += 1
        _validate_trainer_wrapper_input(item, target, label)
        if item.get("passed") is True:
            counts["trainer_input_ready_count"] += 1
    for index, item in enumerate(external_code_files):
        label = f"trainer_wrapper_dry_run.inputs.external_code_files[{index}]"
        if not isinstance(item, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        counts["external_code_file_count"] += 1
        _validate_trainer_wrapper_external_code(item, target, label)
        if item.get("passed") is True:
            counts["external_code_ready_count"] += 1
    return counts


def _validate_trainer_wrapper_input(item: dict[str, Any], target: ValidationTarget, label: str) -> None:
    for field_name in ("artifact_name", "archive_path", "resolved_path", "kind", "sha256"):
        if not isinstance(item.get(field_name), str):
            target.errors.append(f"{label}.{field_name} must be a string.")
    if item.get("kind") not in {"file", "directory"}:
        target.errors.append(f"{label}.kind must be file or directory.")
    if not isinstance(item.get("passed"), bool):
        target.errors.append(f"{label}.passed must be a boolean.")
    if item.get("sha256") and not _is_sha256(item.get("sha256")):
        target.errors.append(f"{label}.sha256 must be a SHA-256 hex string when present.")
    if "size_bytes" in item and not _is_non_negative_int(item.get("size_bytes")):
        target.errors.append(f"{label}.size_bytes must be a non-negative integer when present.")
    if "file_count" in item and not _is_non_negative_int(item.get("file_count")):
        target.errors.append(f"{label}.file_count must be a non-negative integer when present.")


def _validate_trainer_wrapper_external_code(item: dict[str, Any], target: ValidationTarget, label: str) -> None:
    for field_name in ("path", "resolved_path", "sha256"):
        if not isinstance(item.get(field_name), str):
            target.errors.append(f"{label}.{field_name} must be a string.")
    if not isinstance(item.get("passed"), bool):
        target.errors.append(f"{label}.passed must be a boolean.")
    if item.get("sha256") and not _is_sha256(item.get("sha256")):
        target.errors.append(f"{label}.sha256 must be a SHA-256 hex string when present.")
    if "size_bytes" in item and not _is_non_negative_int(item.get("size_bytes")):
        target.errors.append(f"{label}.size_bytes must be a non-negative integer when present.")
    path = _visible_local_path(item.get("resolved_path"))
    if path is None or not path.exists() or item.get("passed") is not True:
        return
    if path.is_symlink() or not path.is_file():
        target.errors.append(f"{label}.resolved_path is not a regular file on disk.")
        return
    if "size_bytes" in item and item.get("size_bytes") != path.stat().st_size:
        target.errors.append(f"{label}.size_bytes does not match resolved_path.")
    if item.get("sha256") != _sha256(path):
        target.errors.append(f"{label}.sha256 does not match resolved_path.")


def _validate_trainer_wrapper_metrics(
    metrics: dict[str, Any],
    command_arg_count: int,
    input_counts: dict[str, int],
    target: ValidationTarget,
) -> None:
    expected = {
        "command_arg_count": command_arg_count,
        "trainer_input_count": input_counts["trainer_input_count"],
        "trainer_input_ready_count": input_counts["trainer_input_ready_count"],
        "external_code_file_count": input_counts["external_code_file_count"],
        "external_code_ready_count": input_counts["external_code_ready_count"],
    }
    for field_name, expected_value in expected.items():
        if metrics.get(field_name) != expected_value:
            target.errors.append(f"trainer_wrapper_dry_run.metrics.{field_name} expected {expected_value}, got {metrics.get(field_name)!r}.")


def _trainer_input_from_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    item: dict[str, Any] = {
        "artifact_index": artifact.get("index"),
        "artifact_name": artifact.get("name"),
        "kind": artifact.get("kind"),
        "original_path": artifact.get("original_path"),
        "archive_path": artifact.get("path"),
        "size_bytes": artifact.get("size_bytes"),
        "sha256": artifact.get("sha256"),
    }
    if artifact.get("kind") == "directory":
        item["file_count"] = artifact.get("file_count")
        item["tree_hash_algorithm"] = artifact.get("tree_hash_algorithm")
    return item


def _expected_trainer_archive_rewrites(inputs: list[Any]) -> list[dict[str, Any]]:
    rewrites: list[dict[str, Any]] = []
    for item in inputs:
        if not isinstance(item, dict):
            continue
        original = item.get("original_path")
        archive_path = item.get("archive_path")
        if not isinstance(original, str) or not original or original.startswith("<"):
            continue
        if not isinstance(archive_path, str) or not archive_path:
            continue
        rewrites.append(
            {
                "artifact_name": str(item.get("artifact_name") or ""),
                "kind": str(item.get("kind") or ""),
                "original_path": original,
                "archive_path": archive_path,
            }
        )
    return rewrites


def _rewrite_trainer_archive_command_argv(argv: list[str], rewrites: list[Any]) -> tuple[list[str], int]:
    valid_rewrites = [item for item in rewrites if isinstance(item, dict)]
    ordered = sorted(valid_rewrites, key=lambda item: len(str(item.get("original_path") or "")), reverse=True)
    rewritten: list[str] = []
    rewrite_count = 0
    for token in argv:
        new_token = _rewrite_trainer_archive_command_token(token, ordered)
        if new_token != token:
            rewrite_count += 1
        rewritten.append(new_token)
    return rewritten, rewrite_count


def _rewrite_trainer_archive_command_token(token: str, rewrites: list[dict[str, Any]]) -> str:
    for item in rewrites:
        original = item.get("original_path")
        archive_path = item.get("archive_path")
        if not isinstance(original, str) or not original:
            continue
        if not isinstance(archive_path, str) or not archive_path:
            continue
        replacement = _replace_trainer_archive_path_value(token, original, archive_path)
        if replacement != token:
            return replacement
        if "=" in token:
            key, value = token.split("=", 1)
            rewritten_value = _replace_trainer_archive_path_value(value, original, archive_path)
            if rewritten_value != value:
                return f"{key}={rewritten_value}"
    return token


def _replace_trainer_archive_path_value(value: str, original: str, archive_path: str) -> str:
    if value == original:
        return archive_path
    prefix = original.rstrip("/") + "/"
    if value.startswith(prefix):
        return archive_path.rstrip("/") + "/" + value[len(prefix) :]
    return value


def _trainer_archive_external_command_paths(argv: list[str], trainer_inputs: list[Any]) -> list[dict[str, Any]]:
    archive_paths = [
        str(item.get("archive_path") or "")
        for item in trainer_inputs
        if isinstance(item, dict) and isinstance(item.get("archive_path"), str)
    ]
    external: list[dict[str, Any]] = []
    for index, token in enumerate(argv):
        if not token or token.startswith("-"):
            if "=" in token:
                key, value = token.split("=", 1)
                if _trainer_archive_is_external_command_path(value, archive_paths):
                    external.append({"argv_index": index, "token": token, "path": value, "reason": f"{key} references a path outside archive inputs"})
            continue
        if _trainer_archive_is_external_command_path(token, archive_paths):
            external.append({"argv_index": index, "token": token, "path": token, "reason": "path-like token is not one of the copied trainer inputs"})
    return external


def _trainer_archive_is_external_command_path(value: str, archive_paths: list[str]) -> bool:
    if not _trainer_archive_looks_like_path(value):
        return False
    normalized = value.replace("\\", "/")
    for archive_path in archive_paths:
        clean = archive_path.replace("\\", "/").rstrip("/")
        if normalized == clean or normalized.startswith(clean + "/"):
            return False
    return True


def _trainer_archive_looks_like_path(value: str) -> bool:
    if not value or value.startswith("-"):
        return False
    normalized = value.replace("\\", "/")
    if normalized.startswith(("./", "../", "/", "~")) or "/" in normalized:
        return True
    return Path(normalized).suffix.lower() in {
        ".py",
        ".sh",
        ".bash",
        ".js",
        ".mjs",
        ".ts",
        ".ipynb",
        ".json",
        ".jsonl",
        ".yaml",
        ".yml",
        ".toml",
        ".csv",
        ".txt",
    }


def _trainer_archive_count_map(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def _trainer_archive_tree_fingerprint(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    file_count = 0
    size_bytes = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        file_hash = _sha256(path)
        size = path.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\0")
        file_count += 1
        size_bytes += size
    return {"sha256": digest.hexdigest(), "file_count": file_count, "size_bytes": size_bytes}


def _archive_artifact_path(value: Any, archive_root: Path) -> Path | None:
    if not isinstance(value, str) or not value or value.startswith("<"):
        return None
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        return None
    return archive_root / path


def _path_resolves_inside(path: Path, root: Path) -> bool:
    try:
        root_resolved = root.resolve()
        path_resolved = path.resolve(strict=False)
    except OSError:
        return False
    return path_resolved == root_resolved or path_resolved.is_relative_to(root_resolved)


def _validate_action_ledger_bundle(bundle: Any, target: ValidationTarget, label: str, expected_index: int) -> None:
    if not isinstance(bundle, dict):
        target.errors.append(f"{label} must be an object.")
        return
    if bundle.get("index") != expected_index:
        target.errors.append(f"{label}.index expected {expected_index}, got {bundle.get('index')!r}.")
    for field_name in ("path", "schema_version", "readiness", "recommendation"):
        if not isinstance(bundle.get(field_name), str):
            target.errors.append(f"{label}.{field_name} must be a string.")
    if bundle.get("schema_version") != EVIDENCE_BUNDLE_SCHEMA_VERSION:
        target.errors.append(f"{label}.schema_version must be {EVIDENCE_BUNDLE_SCHEMA_VERSION}.")
    for field_name in ("exists", "passed"):
        if not isinstance(bundle.get(field_name), bool):
            target.errors.append(f"{label}.{field_name} must be a boolean.")
    if not _is_non_negative_int(bundle.get("action_count")):
        target.errors.append(f"{label}.action_count must be a non-negative integer.")
    if bundle.get("exists") is True:
        if not _is_non_negative_int(bundle.get("size_bytes")):
            target.errors.append(f"{label}.size_bytes must be a non-negative integer for existing files.")
        if not _is_sha256(bundle.get("sha256")):
            target.errors.append(f"{label}.sha256 must be a SHA-256 hex string for existing files.")


def _validate_action_ledger_entry(entry: Any, target: ValidationTarget, label: str, latest_index: int) -> dict[str, int]:
    counts = {"occurrence_count": 0, "open": 0, "new": 0, "recurring": 0, "resolved": 0}
    if not isinstance(entry, dict):
        target.errors.append(f"{label} must be an object.")
        return counts
    for field_name in ("routing_key", "action_fingerprint", "id", "priority", "artifact", "summary", "status", "first_seen_path", "last_seen_path"):
        if not isinstance(entry.get(field_name), str) or not entry.get(field_name):
            target.errors.append(f"{label}.{field_name} must be a non-empty string.")
    if not _is_sha256(entry.get("action_fingerprint")):
        target.errors.append(f"{label}.action_fingerprint must be a SHA-256 hex string.")
    expected_routing_key = f"{entry.get('artifact')}:{entry.get('id')}:{str(entry.get('action_fingerprint') or '')[:12]}"
    if isinstance(entry.get("routing_key"), str) and entry.get("routing_key") != expected_routing_key:
        target.errors.append(f"{label}.routing_key expected {expected_routing_key!r}, got {entry.get('routing_key')!r}.")
    if entry.get("priority") not in {"critical", "high", "medium", "low"}:
        target.errors.append(f"{label}.priority must be critical, high, medium, or low.")
    if entry.get("status") not in {"new", "recurring", "open", "resolved"}:
        target.errors.append(f"{label}.status must be new, recurring, open, or resolved.")
    if not isinstance(entry.get("open"), bool):
        target.errors.append(f"{label}.open must be a boolean.")
    if not isinstance(entry.get("evidence"), dict):
        target.errors.append(f"{label}.evidence must be an object.")
    occurrences = entry.get("occurrences")
    if not isinstance(occurrences, list):
        target.errors.append(f"{label}.occurrences must be a list.")
        occurrences = []
    for index, occurrence in enumerate(occurrences):
        _validate_action_ledger_occurrence(occurrence, target, f"{label}.occurrences[{index}]")
    bundle_indexes = entry.get("bundle_indexes")
    if not isinstance(bundle_indexes, list) or not all(_is_non_negative_int(item) for item in bundle_indexes):
        target.errors.append(f"{label}.bundle_indexes must be a list of non-negative integers.")
        bundle_indexes = []
    else:
        sorted_indexes = sorted(set(bundle_indexes))
        if bundle_indexes != sorted_indexes:
            target.errors.append(f"{label}.bundle_indexes must be sorted and unique.")
    occurrence_indexes = sorted(
        {occurrence.get("bundle_index") for occurrence in occurrences if isinstance(occurrence, dict) and _is_non_negative_int(occurrence.get("bundle_index"))}
    )
    if bundle_indexes and occurrence_indexes and bundle_indexes != occurrence_indexes:
        target.errors.append(f"{label}.bundle_indexes must match occurrence bundle indexes.")
    occurrence_count = len(occurrences)
    counts["occurrence_count"] = occurrence_count
    if entry.get("occurrence_count") != occurrence_count:
        target.errors.append(f"{label}.occurrence_count expected {occurrence_count}, got {entry.get('occurrence_count')!r}.")
    if bundle_indexes:
        first_seen = bundle_indexes[0]
        last_seen = bundle_indexes[-1]
        if entry.get("first_seen_index") != first_seen:
            target.errors.append(f"{label}.first_seen_index expected {first_seen}, got {entry.get('first_seen_index')!r}.")
        if entry.get("last_seen_index") != last_seen:
            target.errors.append(f"{label}.last_seen_index expected {last_seen}, got {entry.get('last_seen_index')!r}.")
        open_in_latest = latest_index in bundle_indexes
        expected_status = "new" if open_in_latest and first_seen == latest_index else "recurring" if open_in_latest and len(bundle_indexes) > 1 else "open" if open_in_latest else "resolved"
        if entry.get("open") != open_in_latest:
            target.errors.append(f"{label}.open expected {open_in_latest}, got {entry.get('open')!r}.")
        if entry.get("status") != expected_status:
            target.errors.append(f"{label}.status expected {expected_status!r}, got {entry.get('status')!r}.")
        counts["open"] = 1 if open_in_latest else 0
        counts["new"] = 1 if expected_status == "new" else 0
        counts["recurring"] = 1 if expected_status == "recurring" else 0
        counts["resolved"] = 1 if expected_status == "resolved" else 0
    return counts


def _validate_action_ledger_occurrence(occurrence: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(occurrence, dict):
        target.errors.append(f"{label} must be an object.")
        return
    if not _is_non_negative_int(occurrence.get("bundle_index")):
        target.errors.append(f"{label}.bundle_index must be a non-negative integer.")
    for field_name in ("bundle_path", "summary", "priority", "artifact"):
        if not isinstance(occurrence.get(field_name), str):
            target.errors.append(f"{label}.{field_name} must be a string.")
    if occurrence.get("priority") not in {"critical", "high", "medium", "low"}:
        target.errors.append(f"{label}.priority must be critical, high, medium, or low.")


def _validate_action_ledger_count_rows(value: Any, expected: dict[str, int], target: ValidationTarget, label: str) -> None:
    counts = _count_rows(value)
    if counts is None:
        target.errors.append(f"{label} must be a list of {{id, count}} objects.")
    elif counts != expected:
        target.errors.append(f"{label} expected {expected!r}, got {counts!r}.")


def _validate_action_ledger_bundle_action_counts(value: Any, bundles: list[Any], target: ValidationTarget) -> None:
    label = "action_ledger.metrics.bundle_action_counts"
    if not isinstance(value, list):
        target.errors.append(f"{label} must be a list.")
        return
    expected = [
        {"index": bundle.get("index"), "path": bundle.get("path"), "action_count": bundle.get("action_count")}
        for bundle in bundles
        if isinstance(bundle, dict)
    ]
    if value != expected:
        target.errors.append(f"{label} must match action_ledger.bundles action counts.")


def _increment(counts: dict[str, int], value: Any) -> None:
    if isinstance(value, str) and value:
        counts[value] = counts.get(value, 0) + 1


def _validate_trainer_preflight(preflight: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    _require_equal(preflight, "schema_version", TRAINER_PREFLIGHT_SCHEMA_VERSION, target)
    if not isinstance(preflight.get("preflight_path"), str) or not preflight.get("preflight_path"):
        target.errors.append("trainer_preflight.preflight_path must be a non-empty string.")
    if not isinstance(preflight.get("passed"), bool):
        target.errors.append("trainer_preflight.passed must be a boolean.")
    checks = preflight.get("checks")
    if not isinstance(checks, list):
        target.errors.append("trainer_preflight.checks must be a list.")
        checks = []
    gates = preflight.get("gates")
    if not isinstance(gates, list):
        target.errors.append("trainer_preflight.gates must be a list.")
        gates = []
    artifacts = preflight.get("artifacts")
    if not isinstance(artifacts, dict):
        target.errors.append("trainer_preflight.artifacts must be an object.")
        artifacts = {}
    schema_contracts = preflight.get("schema_contracts", {})
    if not isinstance(schema_contracts, dict):
        target.errors.append("trainer_preflight.schema_contracts must be an object when present.")
        schema_contracts = {}
    if not artifacts:
        target.errors.append("trainer_preflight.artifacts must not be empty.")
    if not _is_string_list(preflight.get("required_gates")):
        target.errors.append("trainer_preflight.required_gates must be a list of strings.")
    if not _is_string_list(preflight.get("required_dataset_versions")):
        target.errors.append("trainer_preflight.required_dataset_versions must be a list of strings.")
    if not _is_string_list(preflight.get("notes")):
        target.errors.append("trainer_preflight.notes must be a list of strings.")
    if "metadata" in preflight:
        _validate_metadata(preflight.get("metadata"), target, "trainer_preflight.metadata")
    validation_summaries = preflight.get("validation_summaries", [])
    _validate_trainer_preflight_validation_summaries(validation_summaries, target, source_path)

    failed_checks = _validate_handoff_checks(checks, target, "trainer_preflight.checks")
    if preflight.get("check_count") != len(checks):
        target.errors.append(f"trainer_preflight.check_count expected {len(checks)}, got {preflight.get('check_count')!r}.")
    if preflight.get("failed_check_count") != failed_checks:
        target.errors.append(
            f"trainer_preflight.failed_check_count expected {failed_checks}, got {preflight.get('failed_check_count')!r}."
        )
    expected_passed = failed_checks == 0
    if isinstance(preflight.get("passed"), bool) and preflight.get("passed") != expected_passed:
        target.errors.append("trainer_preflight.passed must match failed_check_count.")
    expected_readiness = "ready" if expected_passed else "blocked"
    expected_recommendation = "launch_allowed" if expected_passed else "block_launch"
    if preflight.get("readiness") != expected_readiness:
        target.errors.append(f"trainer_preflight.readiness expected {expected_readiness!r}, got {preflight.get('readiness')!r}.")
    if preflight.get("recommendation") != expected_recommendation:
        target.errors.append(
            f"trainer_preflight.recommendation expected {expected_recommendation!r}, got {preflight.get('recommendation')!r}."
        )

    passed_gates = 0
    for index, gate in enumerate(gates):
        if _validate_trainer_preflight_gate(gate, target, f"trainer_preflight.gates[{index}]", source_path):
            passed_gates += 1
    if preflight.get("gate_count") != len(gates):
        target.errors.append(f"trainer_preflight.gate_count expected {len(gates)}, got {preflight.get('gate_count')!r}.")
    if preflight.get("passed_gate_count") != passed_gates:
        target.errors.append(
            f"trainer_preflight.passed_gate_count expected {passed_gates}, got {preflight.get('passed_gate_count')!r}."
        )
    for name, record in artifacts.items():
        _validate_trainer_preflight_artifact_record(name, record, target, source_path)
    for name, record in schema_contracts.items():
        _validate_trainer_preflight_schema_contract_record(name, record, target, source_path)
    dataset_selection = preflight.get("dataset_selection", [])
    _validate_trainer_preflight_dataset_selection(dataset_selection, target)
    _validate_trainer_command(preflight.get("trainer_command"), target)
    target.details.update(
        {
            "readiness": preflight.get("readiness"),
            "gate_count": len(gates),
            "failed_check_count": failed_checks,
            "artifact_count": len(artifacts),
            "schema_contract_count": len(schema_contracts),
            "dataset_selection_count": len(dataset_selection) if isinstance(dataset_selection, list) else 0,
            "validation_summary_count": len(validation_summaries) if isinstance(validation_summaries, list) else 0,
        }
    )


def _validate_trainer_launch_check(launch_check: dict[str, Any], target: ValidationTarget) -> None:
    _require_equal(launch_check, "schema_version", TRAINER_LAUNCH_CHECK_SCHEMA_VERSION, target)
    if not isinstance(launch_check.get("preflight_path"), str) or not launch_check.get("preflight_path"):
        target.errors.append("trainer_launch_check.preflight_path must be a non-empty string.")
    if not isinstance(launch_check.get("passed"), bool):
        target.errors.append("trainer_launch_check.passed must be a boolean.")
    checks = launch_check.get("checks")
    if not isinstance(checks, list):
        target.errors.append("trainer_launch_check.checks must be a list.")
        checks = []
    failed_checks = _validate_handoff_checks(checks, target, "trainer_launch_check.checks")
    if launch_check.get("check_count") != len(checks):
        target.errors.append(f"trainer_launch_check.check_count expected {len(checks)}, got {launch_check.get('check_count')!r}.")
    if launch_check.get("failed_check_count") != failed_checks:
        target.errors.append(
            f"trainer_launch_check.failed_check_count expected {failed_checks}, got {launch_check.get('failed_check_count')!r}."
        )
    expected_passed = failed_checks == 0
    if isinstance(launch_check.get("passed"), bool) and launch_check.get("passed") != expected_passed:
        target.errors.append("trainer_launch_check.passed must match failed_check_count.")
    expected_readiness = "ready" if expected_passed else "blocked"
    expected_recommendation = "launch_allowed" if expected_passed else "block_launch"
    if launch_check.get("readiness") != expected_readiness:
        target.errors.append(
            f"trainer_launch_check.readiness expected {expected_readiness!r}, got {launch_check.get('readiness')!r}."
        )
    if launch_check.get("recommendation") != expected_recommendation:
        target.errors.append(
            f"trainer_launch_check.recommendation expected {expected_recommendation!r}, got {launch_check.get('recommendation')!r}."
        )
    if not _is_string_list(launch_check.get("required_gates")):
        target.errors.append("trainer_launch_check.required_gates must be a list of strings.")
    if not _is_string_list(launch_check.get("required_dataset_versions")):
        target.errors.append("trainer_launch_check.required_dataset_versions must be a list of strings.")
    _validate_trainer_preflight_dataset_selection(launch_check.get("dataset_selection", []), target)
    required_metadata = launch_check.get("required_metadata")
    if not isinstance(required_metadata, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in required_metadata.items()):
        target.errors.append("trainer_launch_check.required_metadata must be an object of string values.")
    gates = launch_check.get("gates")
    if not isinstance(gates, list):
        target.errors.append("trainer_launch_check.gates must be a list.")
        gates = []
    passed_gates = 0
    for index, gate in enumerate(gates):
        if _validate_launch_check_gate(gate, target, f"trainer_launch_check.gates[{index}]"):
            passed_gates += 1
    if launch_check.get("gate_count") != len(gates):
        target.errors.append(f"trainer_launch_check.gate_count expected {len(gates)}, got {launch_check.get('gate_count')!r}.")
    if launch_check.get("passed_gate_count") != passed_gates:
        target.errors.append(
            f"trainer_launch_check.passed_gate_count expected {passed_gates}, got {launch_check.get('passed_gate_count')!r}."
        )
    _validate_launch_validation_record(launch_check.get("validation"), target)
    _validate_launch_artifacts_summary(launch_check.get("artifacts"), target)
    _validate_approved_command(launch_check.get("approved_command"), target, expected_passed)
    if not _is_string_list(launch_check.get("notes")):
        target.errors.append("trainer_launch_check.notes must be a list of strings.")
    target.details.update(
        {
            "readiness": launch_check.get("readiness"),
            "gate_count": len(gates),
            "failed_check_count": failed_checks,
        }
    )


def _validate_launch_check_gate(gate: Any, target: ValidationTarget, label: str) -> bool:
    if not isinstance(gate, dict):
        target.errors.append(f"{label} must be an object.")
        return False
    for field_name in ("id", "path", "schema_version"):
        if not isinstance(gate.get(field_name), str):
            target.errors.append(f"{label}.{field_name} must be a string.")
    if not isinstance(gate.get("passed"), bool):
        target.errors.append(f"{label}.passed must be a boolean.")
    return gate.get("passed") is True


def _validate_launch_validation_record(value: Any, target: ValidationTarget) -> None:
    if not isinstance(value, dict):
        target.errors.append("trainer_launch_check.validation must be an object.")
        return
    for field_name in ("passed", "strict"):
        if not isinstance(value.get(field_name), bool):
            target.errors.append(f"trainer_launch_check.validation.{field_name} must be a boolean.")
    for field_name in ("target_count", "error_count", "warning_count"):
        if not _is_non_negative_int(value.get(field_name)):
            target.errors.append(f"trainer_launch_check.validation.{field_name} must be a non-negative integer.")
    for field_name in ("errors", "warnings"):
        if not _is_string_list(value.get(field_name)):
            target.errors.append(f"trainer_launch_check.validation.{field_name} must be a list of strings.")


def _validate_launch_artifacts_summary(value: Any, target: ValidationTarget) -> None:
    if not isinstance(value, dict):
        target.errors.append("trainer_launch_check.artifacts must be an object.")
        return
    if not _is_non_negative_int(value.get("count")):
        target.errors.append("trainer_launch_check.artifacts.count must be a non-negative integer.")
    names = value.get("names")
    if not _is_string_list(names):
        target.errors.append("trainer_launch_check.artifacts.names must be a list of strings.")
    elif _is_non_negative_int(value.get("count")) and value.get("count") != len(names):
        target.errors.append(f"trainer_launch_check.artifacts.count expected {len(names)}, got {value.get('count')!r}.")


def _validate_approved_command(command: Any, target: ValidationTarget, expected_approved: bool) -> None:
    if not isinstance(command, dict):
        target.errors.append("trainer_launch_check.approved_command must be an object.")
        return
    for field_name in ("approved", "provided", "parseable"):
        if not isinstance(command.get(field_name), bool):
            target.errors.append(f"trainer_launch_check.approved_command.{field_name} must be a boolean.")
    if isinstance(command.get("approved"), bool) and command.get("approved") != expected_approved:
        target.errors.append("trainer_launch_check.approved_command.approved must match launch check passed.")
    if not isinstance(command.get("raw"), str):
        target.errors.append("trainer_launch_check.approved_command.raw must be a string.")
    if not isinstance(command.get("shell"), str):
        target.errors.append("trainer_launch_check.approved_command.shell must be a string.")
    if not isinstance(command.get("argv"), list) or not all(isinstance(item, str) for item in command.get("argv", [])):
        target.errors.append("trainer_launch_check.approved_command.argv must be a list of strings.")
    if expected_approved and not command.get("shell"):
        target.errors.append("trainer_launch_check.approved_command.shell must be non-empty when approved.")


def _validate_handoff_checks(checks: list[Any], target: ValidationTarget, label: str) -> int:
    failed = 0
    for index, check in enumerate(checks):
        check_label = f"{label}[{index}]"
        if not isinstance(check, dict):
            target.errors.append(f"{check_label} must be an object.")
            failed += 1
            continue
        if not isinstance(check.get("id"), str) or not check.get("id"):
            target.errors.append(f"{check_label}.id must be a non-empty string.")
        if not isinstance(check.get("passed"), bool):
            target.errors.append(f"{check_label}.passed must be a boolean.")
        elif check.get("passed") is False:
            failed += 1
        for field_name in ("actual", "expected", "scope"):
            if not isinstance(check.get(field_name), dict):
                target.errors.append(f"{check_label}.{field_name} must be an object.")
        if not isinstance(check.get("summary"), str):
            target.errors.append(f"{check_label}.summary must be a string.")
    return failed


def _validate_trainer_preflight_gate(gate: Any, target: ValidationTarget, label: str, source_path: Path) -> bool:
    if not isinstance(gate, dict):
        target.errors.append(f"{label} must be an object.")
        return False
    for field_name in ("id", "path", "schema_version"):
        if not isinstance(gate.get(field_name), str) or not gate.get(field_name):
            target.errors.append(f"{label}.{field_name} must be a non-empty string.")
    if not isinstance(gate.get("exists"), bool):
        target.errors.append(f"{label}.exists must be a boolean.")
    if not isinstance(gate.get("passed"), bool):
        target.errors.append(f"{label}.passed must be a boolean.")
    _validate_preflight_file_hash(gate, target, label, source_path, require_kind=False)
    validation = gate.get("validation")
    if validation is not None:
        if not isinstance(validation, dict):
            target.errors.append(f"{label}.validation must be an object when present.")
        else:
            for field_name in ("available", "passed", "strict"):
                if not isinstance(validation.get(field_name), bool):
                    target.errors.append(f"{label}.validation.{field_name} must be a boolean.")
            for field_name in ("error_count", "warning_count"):
                if not _is_non_negative_int(validation.get(field_name)):
                    target.errors.append(f"{label}.validation.{field_name} must be a non-negative integer.")
    return gate.get("passed") is True


def _validate_trainer_preflight_validation_summaries(value: Any, target: ValidationTarget, source_path: Path) -> None:
    if not isinstance(value, list):
        target.errors.append("trainer_preflight.validation_summaries must be a list when present.")
        return
    for index, summary in enumerate(value):
        label = f"trainer_preflight.validation_summaries[{index}]"
        if not isinstance(summary, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        for field_name in ("path", "schema_version"):
            if not isinstance(summary.get(field_name), str) or not summary.get(field_name):
                target.errors.append(f"{label}.{field_name} must be a non-empty string.")
        if summary.get("kind") != "file":
            target.errors.append(f"{label}.kind must be file.")
        for field_name in ("exists", "regular_file", "symlink", "passed", "strict"):
            if not isinstance(summary.get(field_name), bool):
                target.errors.append(f"{label}.{field_name} must be a boolean.")
        for field_name in ("target_count", "error_count", "warning_count"):
            if not _is_non_negative_int(summary.get(field_name)):
                target.errors.append(f"{label}.{field_name} must be a non-negative integer.")
        _validate_preflight_file_hash(summary, target, label, source_path)
        targets = summary.get("targets")
        if not isinstance(targets, list):
            target.errors.append(f"{label}.targets must be a list.")
            targets = []
        if _is_non_negative_int(summary.get("target_count")) and summary.get("target_count") != len(targets):
            target.errors.append(f"{label}.target_count expected {len(targets)}, got {summary.get('target_count')!r}.")
        for target_index, summary_target in enumerate(targets):
            target_label = f"{label}.targets[{target_index}]"
            if not isinstance(summary_target, dict):
                target.errors.append(f"{target_label} must be an object.")
                continue
            for field_name in ("type", "path"):
                if not isinstance(summary_target.get(field_name), str) or not summary_target.get(field_name):
                    target.errors.append(f"{target_label}.{field_name} must be a non-empty string.")
            if not isinstance(summary_target.get("passed"), bool):
                target.errors.append(f"{target_label}.passed must be a boolean.")
            for field_name in ("error_count", "warning_count"):
                if not _is_non_negative_int(summary_target.get(field_name)):
                    target.errors.append(f"{target_label}.{field_name} must be a non-negative integer.")


def _validate_trainer_preflight_schema_contract_record(name: Any, record: Any, target: ValidationTarget, source_path: Path) -> None:
    if not isinstance(name, str) or not name:
        target.errors.append("trainer_preflight.schema_contracts keys must be non-empty strings.")
    label = f"trainer_preflight.schema_contracts.{name}"
    if not isinstance(record, dict):
        target.errors.append(f"{label} must be an object.")
        return
    for field_name in ("path", "schema_name"):
        if not isinstance(record.get(field_name), str) or not record.get(field_name):
            target.errors.append(f"{label}.{field_name} must be a non-empty string.")
    if record.get("kind") not in {"json", "jsonl"}:
        target.errors.append(f"{label}.kind must be json or jsonl.")
    for field_name in ("exists", "regular_file", "symlink", "passed"):
        if not isinstance(record.get(field_name), bool):
            target.errors.append(f"{label}.{field_name} must be a boolean.")
    if not _is_non_negative_int(record.get("error_count")):
        target.errors.append(f"{label}.error_count must be a non-negative integer.")
    if not _is_string_list(record.get("errors")):
        target.errors.append(f"{label}.errors must be a list of strings.")
    elif _is_non_negative_int(record.get("error_count")) and record.get("error_count") < len(record.get("errors", [])):
        target.errors.append(f"{label}.error_count must be at least the number of retained errors.")
    if record.get("kind") == "jsonl":
        if not _is_non_negative_int(record.get("row_count")):
            target.errors.append(f"{label}.row_count must be a non-negative integer for JSONL contracts.")
        row_counts = record.get("row_schema_counts")
        if not isinstance(row_counts, list):
            target.errors.append(f"{label}.row_schema_counts must be a list for JSONL contracts.")
        else:
            for index, row_count in enumerate(row_counts):
                row_label = f"{label}.row_schema_counts[{index}]"
                if not isinstance(row_count, dict):
                    target.errors.append(f"{row_label} must be an object.")
                    continue
                if not isinstance(row_count.get("name"), str) or not row_count.get("name"):
                    target.errors.append(f"{row_label}.name must be a non-empty string.")
                if not _is_non_negative_int(row_count.get("count")):
                    target.errors.append(f"{row_label}.count must be a non-negative integer.")
    _validate_preflight_file_hash(record, target, label, source_path, require_kind=False)


def _validate_trainer_preflight_artifact_record(name: Any, record: Any, target: ValidationTarget, source_path: Path) -> None:
    if not isinstance(name, str) or not name:
        target.errors.append("trainer_preflight.artifacts keys must be non-empty strings.")
    label = f"trainer_preflight.artifacts.{name}"
    if not isinstance(record, dict):
        target.errors.append(f"{label} must be an object.")
        return
    if not isinstance(record.get("path"), str) or not record.get("path"):
        target.errors.append(f"{label}.path must be a non-empty string.")
    if not isinstance(record.get("exists"), bool):
        target.errors.append(f"{label}.exists must be a boolean.")
    if record.get("kind") not in {"file", "directory"}:
        target.errors.append(f"{label}.kind must be file or directory.")
    if record.get("kind") == "file":
        _validate_preflight_file_hash(record, target, label, source_path)
    if record.get("kind") == "directory":
        if "regular_directory" in record and not isinstance(record.get("regular_directory"), bool):
            target.errors.append(f"{label}.regular_directory must be a boolean when present.")
        if "symlink" in record and not isinstance(record.get("symlink"), bool):
            target.errors.append(f"{label}.symlink must be a boolean when present.")
        if record.get("regular_directory") is True and not _is_non_negative_int(record.get("entry_count")):
            target.errors.append(f"{label}.entry_count must be a non-negative integer for existing directories.")


def _validate_trainer_preflight_dataset_selection(value: Any, target: ValidationTarget) -> None:
    if not isinstance(value, list):
        target.errors.append("trainer_preflight.dataset_selection must be a list.")
        return
    for index, record in enumerate(value):
        label = f"trainer_preflight.dataset_selection[{index}]"
        if not isinstance(record, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        if record.get("artifact") not in {"training_export", "reviewed_export"}:
            target.errors.append(f"{label}.artifact must be training_export or reviewed_export.")
        if not _is_dataset_version(record.get("dataset_version")):
            target.errors.append(f"{label}.dataset_version must be an hfrds-* selection key.")
        for field_name in ("manifest_path", "registry_path"):
            if not isinstance(record.get(field_name), str) or not record.get(field_name):
                target.errors.append(f"{label}.{field_name} must be a non-empty string.")
        for field_name in ("manifest_sha256", "registry_sha256", "registry_manifest_sha256"):
            if record.get(field_name) and not _is_sha256(record.get(field_name)):
                target.errors.append(f"{label}.{field_name} must be a SHA-256 hex string when present.")
        if record.get("manifest_sha256") and record.get("registry_manifest_sha256") and record.get("manifest_sha256") != record.get("registry_manifest_sha256"):
            target.errors.append(f"{label}.registry_manifest_sha256 must match manifest_sha256.")
        for field_name in ("registry_dataset_version", "registry_selection_key"):
            if record.get(field_name) != record.get("dataset_version"):
                target.errors.append(f"{label}.{field_name} must match dataset_version.")
        if not isinstance(record.get("matches_required"), bool):
            target.errors.append(f"{label}.matches_required must be a boolean.")
        if record.get("redaction_passed") is not True:
            target.errors.append(f"{label}.redaction_passed must be true.")
        if record.get("artifact") == "training_export" and record.get("heldout_scenario_exclusive") is not True:
            target.errors.append(f"{label}.heldout_scenario_exclusive must be true for training exports.")
        if not _is_string_list(record.get("required_dataset_versions")):
            target.errors.append(f"{label}.required_dataset_versions must be a list of strings.")


def _validate_preflight_file_hash(
    record: dict[str, Any],
    target: ValidationTarget,
    label: str,
    source_path: Path,
    *,
    require_kind: bool = True,
) -> None:
    if require_kind and record.get("kind") != "file":
        return
    if record.get("exists") is not True:
        return
    if "regular_file" in record and not isinstance(record.get("regular_file"), bool):
        target.errors.append(f"{label}.regular_file must be a boolean when present.")
    if "symlink" in record and not isinstance(record.get("symlink"), bool):
        target.errors.append(f"{label}.symlink must be a boolean when present.")
    if record.get("regular_file") is False:
        return
    if not _is_non_negative_int(record.get("size_bytes")):
        target.errors.append(f"{label}.size_bytes must be a non-negative integer for existing files.")
    if not _is_sha256(record.get("sha256")):
        target.errors.append(f"{label}.sha256 must be a SHA-256 hex string for existing files.")
        return
    file_path = _resolve_preflight_record_path(record.get("path"), source_path)
    if file_path is None:
        return
    if file_path.is_symlink():
        target.errors.append(f"{label}.path must not resolve to a symlink.")
        return
    if not file_path.exists() or not file_path.is_file():
        target.errors.append(f"{label}.path does not resolve to an existing file.")
        return
    if file_path.stat().st_size != record.get("size_bytes"):
        target.errors.append(f"{label}.size_bytes does not match the current file.")
    if _sha256(file_path) != record.get("sha256"):
        target.errors.append(f"{label}.sha256 does not match the current file.")


def _resolve_preflight_record_path(value: Any, source_path: Path) -> Path | None:
    if not isinstance(value, str) or not value or value.startswith("<redacted:"):
        return None
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw, source_path.parent / raw]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return candidates[0]


def _validate_trainer_command(command: Any, target: ValidationTarget) -> None:
    if not isinstance(command, dict):
        target.errors.append("trainer_preflight.trainer_command must be an object.")
        return
    if not isinstance(command.get("provided"), bool):
        target.errors.append("trainer_preflight.trainer_command.provided must be a boolean.")
    if not isinstance(command.get("raw"), str):
        target.errors.append("trainer_preflight.trainer_command.raw must be a string.")
    if not isinstance(command.get("argv"), list) or not all(isinstance(item, str) for item in command.get("argv", [])):
        target.errors.append("trainer_preflight.trainer_command.argv must be a list of strings.")
    if "parseable" in command and not isinstance(command.get("parseable"), bool):
        target.errors.append("trainer_preflight.trainer_command.parseable must be a boolean when present.")


def _validate_repair_queue(queue: dict[str, Any], target: ValidationTarget) -> None:
    _require_equal(queue, "schema_version", REPAIR_QUEUE_SCHEMA_VERSION, target)
    if not isinstance(queue.get("runs_dir"), str) or not queue.get("runs_dir"):
        target.errors.append("repair_queue.runs_dir must be a non-empty string.")
    if not isinstance(queue.get("passed"), bool):
        target.errors.append("repair_queue.passed must be a boolean.")
    if not isinstance(queue.get("only_critical"), bool):
        target.errors.append("repair_queue.only_critical must be a boolean.")
    items = queue.get("items")
    if not isinstance(items, list):
        target.errors.append("repair_queue.items must be a list.")
        items = []
    if queue.get("item_count") != len(items):
        target.errors.append(f"repair_queue.item_count expected {len(items)}, got {queue.get('item_count')!r}.")

    seen_ids: set[str] = set()
    totals: dict[str, Any] = {
        "critical_item_count": 0,
        "scenario_ids": set(),
        "task_families": set(),
        "priority_counts": {},
        "rule_counts": {},
        "critical_rule_counts": {},
        "task_completion_status_counts": {},
    }
    for index, item in enumerate(items):
        _validate_repair_item(item, target, f"repair_queue.items[{index}]", seen_ids, totals)
    _validate_repair_queue_metrics(queue.get("metrics"), target, totals, len(items))
    if "notes" in queue and not _is_string_list(queue.get("notes")):
        target.errors.append("repair_queue.notes must be a list of strings when present.")
    target.details.update(
        {
            "item_count": len(items),
            "critical_item_count": totals["critical_item_count"],
            "scenario_count": len(totals["scenario_ids"]),
        }
    )


def _validate_repair_item(
    item: Any,
    target: ValidationTarget,
    label: str,
    seen_ids: set[str],
    totals: dict[str, Any],
) -> None:
    if not isinstance(item, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _require_equal(item, "schema_version", REPAIR_ITEM_SCHEMA_VERSION, target, prefix=f"{label}.")
    for field_name in (
        "repair_item_id",
        "run_id",
        "scenario_id",
        "scenario_title",
        "task_family",
        "priority",
        "rule_id",
        "rule_name",
        "summary",
        "suggested_action",
    ):
        if not isinstance(item.get(field_name), str) or not item.get(field_name):
            target.errors.append(f"{label}.{field_name} must be a non-empty string.")
    item_id = item.get("repair_item_id")
    if isinstance(item_id, str) and item_id:
        if item_id in seen_ids:
            target.errors.append(f"{label}.repair_item_id duplicates {item_id!r}.")
        seen_ids.add(item_id)
    if item.get("priority") not in {"critical", "high", "medium", "low"}:
        target.errors.append(f"{label}.priority must be critical, high, medium, or low.")
    if not isinstance(item.get("critical"), bool):
        target.errors.append(f"{label}.critical must be a boolean.")
    if not _is_non_negative_int(item.get("penalty")):
        target.errors.append(f"{label}.penalty must be a non-negative integer.")
    if not _is_int_between(item.get("score"), 0, 100):
        target.errors.append(f"{label}.score must be an integer from 0 to 100.")
    if "pass_threshold" in item and item.get("pass_threshold") is not None and not _is_int_between(item.get("pass_threshold"), 0, 100):
        target.errors.append(f"{label}.pass_threshold must be null or an integer from 0 to 100.")
    if not isinstance(item.get("task_completion_passed"), bool):
        target.errors.append(f"{label}.task_completion_passed must be a boolean.")
    if not _is_string_list(item.get("evidence")):
        target.errors.append(f"{label}.evidence must be a list of strings.")
    _validate_evidence_refs(item.get("evidence_refs"), target, f"{label}.evidence_refs")
    _validate_repair_evidence_snippets(item.get("evidence_snippets"), target, f"{label}.evidence_snippets")
    _validate_repair_source_artifacts(item.get("source_artifacts"), target, f"{label}.source_artifacts")
    _validate_repair_replay(item.get("replay"), target, f"{label}.replay")

    if item.get("critical") is True:
        totals["critical_item_count"] += 1
    _add_total(totals["scenario_ids"], item.get("scenario_id"))
    _add_total(totals["task_families"], item.get("task_family"))
    _increment_count(totals["priority_counts"], item.get("priority"))
    _increment_count(totals["rule_counts"], item.get("rule_id"))
    if item.get("critical") is True:
        _increment_count(totals["critical_rule_counts"], item.get("rule_id"))
    _increment_count(totals["task_completion_status_counts"], item.get("task_completion_status"))


def _validate_repair_evidence_snippets(value: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(value, list):
        target.errors.append(f"{label} must be a list.")
        return
    for index, snippet in enumerate(value):
        snippet_label = f"{label}[{index}]"
        if not isinstance(snippet, dict):
            target.errors.append(f"{snippet_label} must be an object.")
            continue
        if snippet.get("target") not in {"event", "final_answer", "episode", "state_snapshot"}:
            target.errors.append(f"{snippet_label}.target must be event, final_answer, episode, or state_snapshot.")
        if not isinstance(snippet.get("reason"), str):
            target.errors.append(f"{snippet_label}.reason must be a string.")
        if not isinstance(snippet.get("text"), str):
            target.errors.append(f"{snippet_label}.text must be a string.")
        elif len(snippet["text"]) > 600:
            target.errors.append(f"{snippet_label}.text must be at most 600 characters.")
        if snippet.get("target") == "event":
            if not _is_non_negative_int(snippet.get("event_index")):
                target.errors.append(f"{snippet_label}.event_index must be a non-negative integer.")
            for field_name in ("event_type", "tool_name", "status"):
                if not isinstance(snippet.get(field_name), str):
                    target.errors.append(f"{snippet_label}.{field_name} must be a string.")


def _validate_repair_source_artifacts(value: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    for artifact_name in ("run_dir", "normalized_trace", "scorecard", "report"):
        if not isinstance(value.get(artifact_name), str) or not value.get(artifact_name):
            target.errors.append(f"{label}.{artifact_name} must be a non-empty string.")
    for artifact_name, artifact_path in value.items():
        if not isinstance(artifact_name, str) or not artifact_name:
            target.errors.append(f"{label} keys must be non-empty strings.")
        if not isinstance(artifact_path, str) or not artifact_path:
            target.errors.append(f"{label}.{artifact_name} must be a non-empty string.")


def _validate_repair_replay(value: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    if not isinstance(value.get("available"), bool):
        target.errors.append(f"{label}.available must be a boolean.")
    if value.get("self_contained") is not None and not isinstance(value.get("self_contained"), bool):
        target.errors.append(f"{label}.self_contained must be a boolean or null.")
    if not isinstance(value.get("command"), str):
        target.errors.append(f"{label}.command must be a string.")
    if not isinstance(value.get("argv"), list) or not all(isinstance(part, str) for part in value.get("argv", [])):
        target.errors.append(f"{label}.argv must be a list of strings.")


def _validate_repair_queue_metrics(metrics: Any, target: ValidationTarget, totals: dict[str, Any], item_count: int) -> None:
    if not isinstance(metrics, dict):
        target.errors.append("repair_queue.metrics must be an object.")
        return
    expected = {
        "item_count": item_count,
        "critical_item_count": totals["critical_item_count"],
        "scenario_count": len(totals["scenario_ids"]),
        "task_family_count": len(totals["task_families"]),
        "scenarios": sorted(totals["scenario_ids"]),
        "task_families": sorted(totals["task_families"]),
        "priority_counts": totals["priority_counts"],
        "rule_counts": totals["rule_counts"],
        "critical_rule_counts": totals["critical_rule_counts"],
        "task_completion_status_counts": totals["task_completion_status_counts"],
    }
    for field_name in ("item_count", "critical_item_count", "scenario_count", "task_family_count"):
        if metrics.get(field_name) != expected[field_name]:
            target.errors.append(f"repair_queue.metrics.{field_name} expected {expected[field_name]!r}, got {metrics.get(field_name)!r}.")
    for field_name in ("scenarios", "task_families"):
        if metrics.get(field_name) != expected[field_name]:
            target.errors.append(f"repair_queue.metrics.{field_name} expected {expected[field_name]!r}, got {metrics.get(field_name)!r}.")
    for field_name in ("priority_counts", "rule_counts", "critical_rule_counts", "task_completion_status_counts"):
        actual_counts = _count_rows(metrics.get(field_name))
        if actual_counts != expected[field_name]:
            target.errors.append(f"repair_queue.metrics.{field_name} does not match repair items.")


def _add_total(target_set: set[str], value: Any) -> None:
    if isinstance(value, str) and value:
        target_set.add(value)


def _increment_count(target_counts: dict[str, int], value: Any) -> None:
    if isinstance(value, str) and value:
        target_counts[value] = target_counts.get(value, 0) + 1


def _count_value(target_counts: dict[str, int], key: str) -> int:
    value = target_counts.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _list_present(value: Any) -> bool:
    return isinstance(value, list) and bool(value)


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
    next_actions = decision.get("next_actions")
    if not isinstance(next_actions, list):
        target.errors.append("evidence_bundle.decision.next_actions must be a list.")
        next_actions = []
    else:
        for index, action in enumerate(next_actions):
            label = f"evidence_bundle.decision.next_actions[{index}]"
            if not isinstance(action, dict):
                target.errors.append(f"{label} must be an object.")
                continue
            for field_name in ("id", "priority", "artifact", "summary"):
                if not isinstance(action.get(field_name), str) or not action.get(field_name):
                    target.errors.append(f"{label}.{field_name} must be a non-empty string.")
            if action.get("priority") not in {"critical", "high", "medium", "low"}:
                target.errors.append(f"{label}.priority must be critical, high, medium, or low.")
            if not isinstance(action.get("evidence"), dict):
                target.errors.append(f"{label}.evidence must be an object.")
            expected_fingerprint = _evidence_bundle_action_fingerprint(action)
            if not _is_sha256(action.get("action_fingerprint")):
                target.errors.append(f"{label}.action_fingerprint must be a SHA-256 hex string.")
            elif action.get("action_fingerprint") != expected_fingerprint:
                target.errors.append(f"{label}.action_fingerprint does not match the action payload.")
            expected_routing_key = f"{action.get('artifact')}:{action.get('id')}:{expected_fingerprint[:12]}"
            if not isinstance(action.get("routing_key"), str) or not action.get("routing_key"):
                target.errors.append(f"{label}.routing_key must be a non-empty string.")
            elif action.get("routing_key") != expected_routing_key:
                target.errors.append(f"{label}.routing_key expected {expected_routing_key!r}, got {action.get('routing_key')!r}.")
    if decision.get("next_action_count") != len(next_actions):
        target.errors.append(
            f"evidence_bundle.decision.next_action_count expected {len(next_actions)}, got {decision.get('next_action_count')!r}."
        )
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


def _evidence_bundle_action_fingerprint(action: dict[str, Any]) -> str:
    evidence = action.get("evidence") if isinstance(action.get("evidence"), dict) else {}
    payload = {
        "id": action.get("id"),
        "priority": action.get("priority"),
        "artifact": action.get("artifact"),
        "evidence": evidence,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_evidence_bundle_metrics(metrics: dict[str, Any], target: ValidationTarget) -> None:
    expected_sections = (
        "suite_summary",
        "scenario_quality",
        "evidence_coverage",
        "trace_observability",
        "run_digest_coverage",
        "repair_queue",
        "validation",
        "training_export",
        "compare_export",
        "review_export",
        "reviewed_export",
        "review_calibration",
        "live_smoke_summary",
        "trainer_handoff",
    )
    for section in expected_sections:
        if section in metrics and not isinstance(metrics[section], dict):
            target.errors.append(f"evidence_bundle.metrics.{section} must be an object when present.")
    training = metrics.get("training_export")
    if isinstance(training, dict):
        _validate_bundle_top_curriculum_priorities(training.get("top_curriculum_priorities"), target)
    run_digest_coverage = metrics.get("run_digest_coverage")
    if isinstance(run_digest_coverage, dict):
        _validate_bundle_run_digest_coverage(run_digest_coverage, target)
    trainer_handoff = metrics.get("trainer_handoff")
    if isinstance(trainer_handoff, dict):
        _validate_bundle_trainer_handoff(trainer_handoff, target)
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
            if "schema_version" in gate and not isinstance(gate.get("schema_version"), str):
                target.errors.append(f"evidence_bundle.metrics.gates[{index}].schema_version must be a string when present.")
            if not isinstance(gate.get("passed"), bool):
                target.errors.append(f"evidence_bundle.metrics.gates[{index}].passed must be a boolean.")
            if "validation" in gate:
                _validate_bundle_gate_validation(gate.get("validation"), target, f"evidence_bundle.metrics.gates[{index}].validation")


def _validate_bundle_gate_validation(value: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object when present.")
        return
    for field_name in ("available", "passed", "strict"):
        if not isinstance(value.get(field_name), bool):
            target.errors.append(f"{label}.{field_name} must be a boolean.")
    for field_name in ("target_count", "error_count", "warning_count"):
        if not _is_non_negative_int(value.get(field_name)):
            target.errors.append(f"{label}.{field_name} must be a non-negative integer.")


def _validate_bundle_run_digest_coverage(value: dict[str, Any], target: ValidationTarget) -> None:
    label = "evidence_bundle.metrics.run_digest_coverage"
    if not isinstance(value.get("runs_dir"), str) or not value.get("runs_dir"):
        target.errors.append(f"{label}.runs_dir must be a non-empty string.")
    for field_name in (
        "run_count",
        "digest_count",
        "missing_digest_count",
        "invalid_digest_count",
        "passed_digest_count",
        "failed_digest_count",
    ):
        if not _is_non_negative_int(value.get(field_name)):
            target.errors.append(f"{label}.{field_name} must be a non-negative integer.")
    if not _is_number_between(value.get("digest_coverage_rate"), 0.0, 1.0):
        target.errors.append(f"{label}.digest_coverage_rate must be numeric from 0.0 to 1.0.")

    run_count = value.get("run_count")
    digest_count = value.get("digest_count")
    missing_count = value.get("missing_digest_count")
    invalid_count = value.get("invalid_digest_count")
    if (
        _is_non_negative_int(run_count)
        and _is_non_negative_int(digest_count)
        and _is_non_negative_int(missing_count)
        and _is_non_negative_int(invalid_count)
    ):
        if int(digest_count) + int(missing_count) + int(invalid_count) != int(run_count):
            target.errors.append(f"{label}.digest_count + missing_digest_count + invalid_digest_count must equal run_count.")
        expected_rate = 1.0 if int(run_count) == 0 else round(int(digest_count) / int(run_count), 4)
        if value.get("digest_coverage_rate") != expected_rate:
            target.errors.append(f"{label}.digest_coverage_rate expected {expected_rate!r}, got {value.get('digest_coverage_rate')!r}.")
    for field_name in ("task_completion_status_counts", "recommended_action_counts"):
        _validate_count_rows(value.get(field_name), target, f"{label}.{field_name}")
    for field_name in ("missing_digest_scenarios", "invalid_digest_scenarios"):
        if not _is_string_list(value.get(field_name)):
            target.errors.append(f"{label}.{field_name} must be a list of strings.")


def _validate_bundle_trainer_handoff(value: dict[str, Any], target: ValidationTarget) -> None:
    label = "evidence_bundle.metrics.trainer_handoff"
    expected_stage_ids = (
        "trainer_preflight",
        "trainer_launch_check",
        "trainer_archive",
        "trainer_archive_check",
        "trainer_consumer_plan",
        "trainer_wrapper_dry_run",
    )
    for field_name in ("stage_count", "handoff_ready_count", "blocked_stage_count", "schema_supported_count"):
        if not _is_non_negative_int(value.get(field_name)):
            target.errors.append(f"{label}.{field_name} must be a non-negative integer.")
    for field_name in ("complete_chain", "all_included_ready"):
        if not isinstance(value.get(field_name), bool):
            target.errors.append(f"{label}.{field_name} must be a boolean.")
    if not _is_string_list(value.get("missing_stage_ids")):
        target.errors.append(f"{label}.missing_stage_ids must be a list of strings.")
    stages = value.get("stages")
    if not isinstance(stages, list):
        target.errors.append(f"{label}.stages must be a list.")
        stages = []
    if _is_non_negative_int(value.get("stage_count")) and int(value["stage_count"]) != len(stages):
        target.errors.append(f"{label}.stage_count expected {len(stages)}, got {value.get('stage_count')!r}.")

    ids: list[str] = []
    handoff_ready_count = 0
    blocked_count = 0
    schema_supported_count = 0
    for index, stage in enumerate(stages):
        stage_label = f"{label}.stages[{index}]"
        if not isinstance(stage, dict):
            target.errors.append(f"{stage_label} must be an object.")
            continue
        stage_id = stage.get("id")
        if not isinstance(stage_id, str) or not stage_id:
            target.errors.append(f"{stage_label}.id must be a non-empty string.")
        else:
            ids.append(stage_id)
            if stage_id not in expected_stage_ids:
                target.errors.append(f"{stage_label}.id has unknown trainer handoff stage {stage_id!r}.")
        for field_name in ("path", "schema_version", "expected_schema_version", "readiness", "recommendation", "expected_recommendation"):
            if not isinstance(stage.get(field_name), str) or not stage.get(field_name):
                target.errors.append(f"{stage_label}.{field_name} must be a non-empty string.")
        for field_name in ("schema_supported", "passed", "handoff_ready"):
            if not isinstance(stage.get(field_name), bool):
                target.errors.append(f"{stage_label}.{field_name} must be a boolean.")
        for field_name in ("check_count", "failed_check_count"):
            if not _is_non_negative_int(stage.get(field_name)):
                target.errors.append(f"{stage_label}.{field_name} must be a non-negative integer.")
        for field_name in (
            "gate_count",
            "passed_gate_count",
            "trainer_input_count",
            "trainer_input_ready_count",
            "trainer_input_available_count",
            "external_code_file_count",
            "external_code_ready_count",
            "missing_external_code_count",
            "missing_trainer_input_count",
            "command_arg_count",
            "artifact_count",
            "missing_count",
            "path_rewrite_count",
        ):
            if field_name in stage and not _is_non_negative_int(stage.get(field_name)):
                target.errors.append(f"{stage_label}.{field_name} must be a non-negative integer when present.")
        if stage.get("handoff_ready") is True:
            handoff_ready_count += 1
        elif stage.get("handoff_ready") is False:
            blocked_count += 1
        if stage.get("schema_supported") is True:
            schema_supported_count += 1

    expected_missing = [stage_id for stage_id in expected_stage_ids if stage_id not in ids]
    if value.get("missing_stage_ids") != expected_missing:
        target.errors.append(f"{label}.missing_stage_ids expected {expected_missing!r}, got {value.get('missing_stage_ids')!r}.")
    if isinstance(value.get("complete_chain"), bool) and value["complete_chain"] != (not expected_missing):
        target.errors.append(f"{label}.complete_chain must match missing_stage_ids.")
    if isinstance(value.get("all_included_ready"), bool) and value["all_included_ready"] != all(
        isinstance(stage, dict) and stage.get("handoff_ready") is True for stage in stages
    ):
        target.errors.append(f"{label}.all_included_ready must match stages[].handoff_ready.")
    expected_counts = {
        "handoff_ready_count": handoff_ready_count,
        "blocked_stage_count": blocked_count,
        "schema_supported_count": schema_supported_count,
    }
    for field_name, expected in expected_counts.items():
        if _is_non_negative_int(value.get(field_name)) and int(value[field_name]) != expected:
            target.errors.append(f"{label}.{field_name} expected {expected}, got {value.get(field_name)!r}.")


def _validate_bundle_top_curriculum_priorities(value: Any, target: ValidationTarget) -> None:
    if value is None:
        return
    if not isinstance(value, list):
        target.errors.append("evidence_bundle.metrics.training_export.top_curriculum_priorities must be a list when present.")
        return
    previous_score: int | None = None
    for index, item in enumerate(value):
        label = f"evidence_bundle.metrics.training_export.top_curriculum_priorities[{index}]"
        if not isinstance(item, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        for field_name in ("task_family", "rule_id", "rule_name", "priority_band"):
            if not isinstance(item.get(field_name), str) or not item.get(field_name):
                target.errors.append(f"{label}.{field_name} must be a non-empty string.")
        if item.get("priority_band") not in {"critical", "high", "medium", "low"}:
            target.errors.append(f"{label}.priority_band must be critical, high, medium, or low.")
        for field_name in ("priority_score", "count", "critical_count", "max_penalty"):
            if not _is_non_negative_int(item.get(field_name)):
                target.errors.append(f"{label}.{field_name} must be a non-negative integer.")
        score = item.get("priority_score")
        if _is_non_negative_int(score):
            if previous_score is not None and int(score) > previous_score:
                target.errors.append(f"{label}.priority_score must be sorted descending.")
            previous_score = int(score)
        for field_name in ("scenario_ids", "failure_ids"):
            if not _is_string_list(item.get(field_name)):
                target.errors.append(f"{label}.{field_name} must be a list of strings.")
        _validate_evidence_refs(item.get("example_evidence_refs"), target, f"{label}.example_evidence_refs")


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
    if "validation" in metrics:
        _validate_review_calibration_validation_metrics(metrics.get("validation"), target)
    _validate_mean_score_by_human_label(metrics.get("mean_score_by_human_label"), label_counts, target)


def _validate_review_calibration_validation_metrics(value: Any, target: ValidationTarget) -> None:
    if not isinstance(value, dict):
        target.errors.append("review_calibration.metrics.validation must be an object when present.")
        return
    for field_name in ("available", "passed", "strict"):
        if not isinstance(value.get(field_name), bool):
            target.errors.append(f"review_calibration.metrics.validation.{field_name} must be a boolean.")
    for field_name in ("target_count", "error_count", "warning_count"):
        if not _is_non_negative_int(value.get(field_name)):
            target.errors.append(f"review_calibration.metrics.validation.{field_name} must be a non-negative integer.")


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
    if name in {"scenario", "source_trace", "source_before_state_snapshot", "source_state_snapshot"} and record.get("exists") is not True:
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


def _run_digest_trace_signal(trace: dict[str, Any]) -> dict[str, Any]:
    events = trace.get("events") if isinstance(trace.get("events"), list) else []
    typed_events = [event for event in events if isinstance(event, dict)]
    event_types = sorted({str(event.get("type")) for event in typed_events if event.get("type")})
    tool_call_count = sum(1 for event in typed_events if event.get("type") == "tool_call")
    api_call_count = _run_digest_api_call_count(trace, typed_events)
    final_answer = trace.get("final_answer")
    session = trace.get("session") if isinstance(trace.get("session"), dict) else {}
    return {
        "event_count": len(typed_events),
        "event_types": event_types,
        "tool_call_count": tool_call_count,
        "tool_result_count": sum(1 for event in typed_events if event.get("type") == "tool_result"),
        "api_call_count": api_call_count,
        "subagent_start_count": sum(1 for event in typed_events if event.get("type") == "subagent_start"),
        "max_subagent_depth": _run_digest_max_subagent_depth(typed_events),
        "has_final_answer": isinstance(final_answer, str) and bool(final_answer.strip()),
        "has_tool_or_api_events": tool_call_count > 0 or api_call_count > 0,
        "source_format": str(session.get("source_format") or "unknown"),
        "model": str(session.get("model") or "unknown"),
    }


def _run_digest_api_call_count(trace: dict[str, Any], events: list[dict[str, Any]]) -> int:
    metadata = trace.get("metadata") if isinstance(trace.get("metadata"), dict) else {}
    api_calls = metadata.get("api_calls")
    if isinstance(api_calls, int) and not isinstance(api_calls, bool) and api_calls >= 0:
        return api_calls
    return sum(1 for event in events if event.get("type") == "api_call")


def _run_digest_max_subagent_depth(events: list[dict[str, Any]]) -> int:
    parent_by_session: dict[str, str | None] = {}
    for event in events:
        if event.get("type") != "subagent_start":
            continue
        session_id = event.get("session_id")
        if isinstance(session_id, str) and session_id:
            parent = event.get("parent_session_id")
            parent_by_session[session_id] = parent if isinstance(parent, str) and parent else None
    max_depth = 0
    for session_id in parent_by_session:
        seen: set[str] = set()
        depth = 1
        parent = parent_by_session.get(session_id)
        while parent and parent not in seen:
            seen.add(parent)
            if parent in parent_by_session:
                depth += 1
                parent = parent_by_session[parent]
            else:
                break
        max_depth = max(max_depth, depth)
    return max_depth


def _run_digest_evidence_counts(
    scorecard: dict[str, Any],
    task_completion: dict[str, Any] | None,
) -> dict[str, int]:
    rules = [rule for rule in scorecard.get("rules", []) if isinstance(rule, dict)]
    failed_rules = [rule for rule in rules if rule.get("passed") is False]
    critical_failed_rules = [rule for rule in failed_rules if rule.get("critical") is True]
    task = task_completion
    if task is None and isinstance(scorecard.get("task_completion"), dict):
        task = scorecard["task_completion"]
    task_refs = task.get("evidence_refs") if isinstance(task, dict) and isinstance(task.get("evidence_refs"), list) else []
    missing_refs = (
        task.get("missing_evidence_refs")
        if isinstance(task, dict) and isinstance(task.get("missing_evidence_refs"), list)
        else []
    )
    rule_ref_count = _run_digest_rule_ref_count(rules)
    return {
        "rule_evidence_ref_count": rule_ref_count,
        "failed_rule_evidence_ref_count": _run_digest_rule_ref_count(failed_rules),
        "critical_failed_rule_evidence_ref_count": _run_digest_rule_ref_count(critical_failed_rules),
        "task_completion_evidence_ref_count": len(task_refs),
        "missing_evidence_ref_count": len(missing_refs),
        "total_evidence_ref_count": rule_ref_count + len(task_refs) + len(missing_refs),
    }


def _run_digest_rule_ref_count(rules: list[dict[str, Any]]) -> int:
    count = 0
    for rule in rules:
        refs = rule.get("evidence_refs")
        if isinstance(refs, list):
            count += len(refs)
    return count


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


def _is_dataset_version(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("hfrds-")
        and len(value) > len("hfrds-")
        and all(char in "0123456789abcdef" for char in value.removeprefix("hfrds-").lower())
    )


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


def _validate_count_rows(value: Any, target: ValidationTarget, label: str) -> dict[str, int]:
    if not isinstance(value, list):
        target.errors.append(f"{label} must be a list.")
        return {}
    counts: dict[str, int] = {}
    previous_id = ""
    for index, row in enumerate(value):
        row_label = f"{label}[{index}]"
        if not isinstance(row, dict):
            target.errors.append(f"{row_label} must be an object.")
            continue
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            target.errors.append(f"{row_label}.id must be a non-empty string.")
            continue
        if row_id in counts:
            target.errors.append(f"{row_label}.id duplicates {row_id!r}.")
        if previous_id and row_id < previous_id:
            target.errors.append(f"{label} must be sorted by id.")
        previous_id = row_id
        count = row.get("count")
        if not _is_non_negative_int(count):
            target.errors.append(f"{row_label}.count must be a non-negative integer.")
            continue
        counts[row_id] = count
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
