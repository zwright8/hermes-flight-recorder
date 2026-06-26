"""Artifact validation for Flight Recorder evidence outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .adapters import TRACE_SCHEMA_VERSION
from .scorers import SCORE_SCHEMA_VERSION
from .training import (
    RL_CURRICULUM_SCHEMA_VERSION,
    RL_DATASET_METRICS_SCHEMA_VERSION,
    RL_DPO_SCHEMA_VERSION,
    RL_EPISODE_SCHEMA_VERSION,
    RL_FAILURE_MODE_SCHEMA_VERSION,
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
    suite_summary_paths: list[str | Path] | None = None,
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
    for suite_summary_path in suite_summary_paths or []:
        targets.append(validate_suite_summary(suite_summary_path))
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
    report_path = run_dir / "report.html"
    trace = _read_object(trace_path, target, "normalized_trace.json")
    scorecard = _read_object(score_path, target, "scorecard.json")
    if trace is not None:
        _validate_trace(trace, target)
    if scorecard is not None:
        _validate_scorecard(scorecard, target)
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


def validate_suite_summary(path: str | Path) -> ValidationTarget:
    """Validate one run-suite summary artifact."""
    summary_path = Path(path)
    target = ValidationTarget("suite_summary", str(summary_path))
    summary = _read_object(summary_path, target, "suite_summary.json")
    if summary is None:
        return target
    _validate_suite_summary(summary, target)
    return target


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
    if curriculum is not None and curriculum.get("failure_mode_count") != len(failure_modes):
        target.errors.append(
            f"curriculum.failure_mode_count expected {len(failure_modes)}, got {curriculum.get('failure_mode_count')!r}."
        )
    if _looks_absolute(str(manifest.get("source_runs_dir", ""))):
        target.warnings.append("manifest.source_runs_dir is absolute; prefer redacted or relative exports for sharing.")
    if _looks_absolute(str(manifest.get("output_dir", ""))):
        target.warnings.append("manifest.output_dir is absolute; prefer redacted or relative exports for sharing.")


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
        if not isinstance(episode.get("events"), list):
            target.errors.append(f"episodes[{index}].events must be a list.")
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

        rule_id = step_reward.get("rule_id")
        for field_name in ("scenario_id", "task_family", "rule_id", "rule_name", "evidence"):
            if not isinstance(step_reward.get(field_name), str):
                target.errors.append(f"step_rewards[{index}].{field_name} must be a string.")
        if step_reward.get("target") not in {"event", "final_answer", "episode"}:
            target.errors.append(f"step_rewards[{index}].target must be one of event, final_answer, or episode.")
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
    for field_name, expected_value in expected.items():
        if pair.get(field_name) != expected_value:
            target.errors.append(f"dpo[{index}].{field_name} does not match preference {preference.get('preference_id')!r}.")


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

    _validate_dataset_family_metrics(metrics.get("task_families"), target, episodes, step_rewards, failure_modes, sft, dpo, reward_model)
    _validate_quality_flags(metrics.get("quality_flags"), target)
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
        expected[family] = {
            "task_family": family,
            "episode_count": len(family_episodes),
            "passed": passed,
            "failed": len(family_episodes) - passed,
            "pass_rate": round(passed / len(family_episodes), 4) if family_episodes else 0.0,
            "average_score": _average_number(scores),
            "step_reward_count": _count_family(step_rewards, family),
            "failure_mode_count": _count_family(failure_modes, family),
            "sft_count": _count_family(sft, family),
            "dpo_count": _count_family(dpo, family),
            "reward_model_count": _count_family(reward_model, family),
        }
    return expected


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
        for field_name in ("scenario_id", "scenario_title", "task_family", "scenario_path", "trace_path", "run_dir", "report", "scorecard"):
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

    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        target.errors.append("suite_summary.metrics must be an object.")
    else:
        _validate_suite_metrics(metrics, target, [run for run in runs if isinstance(run, dict)])

    artifacts = summary.get("artifacts")
    if artifacts is not None and not isinstance(artifacts, dict):
        target.errors.append("suite_summary.artifacts must be an object when present.")

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


def _validate_evidence_refs(value: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(value, list):
        target.errors.append(f"{label} must be a list when present.")
        return
    for index, ref in enumerate(value):
        if not isinstance(ref, dict):
            target.errors.append(f"{label}[{index}] must be an object.")
            continue
        ref_target = ref.get("target")
        if ref_target not in {"event", "final_answer", "episode"}:
            target.errors.append(f"{label}[{index}].target must be one of event, final_answer, or episode.")
        if ref_target == "event":
            event_index = ref.get("event_index")
            if not isinstance(event_index, int) or isinstance(event_index, bool) or event_index < 0:
                target.errors.append(f"{label}[{index}].event_index must be a non-negative integer for event refs.")
        if "reason" in ref and not isinstance(ref.get("reason"), str):
            target.errors.append(f"{label}[{index}].reason must be a string when present.")
        if "passed" in ref and not isinstance(ref.get("passed"), bool):
            target.errors.append(f"{label}[{index}].passed must be a boolean when present.")


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


def _is_int_between(value: Any, minimum: int, maximum: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and minimum <= value <= maximum


def _looks_absolute(value: str) -> bool:
    return value.startswith("/") or _is_windows_absolute(value)


def _score_value(value: Any) -> int:
    try:
        return max(0, min(100, int(value)))
    except (TypeError, ValueError):
        return 0


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
