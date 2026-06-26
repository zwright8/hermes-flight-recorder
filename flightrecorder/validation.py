"""Artifact validation for Flight Recorder evidence outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .adapters import TRACE_SCHEMA_VERSION
from .scorers import SCORE_SCHEMA_VERSION
from .training import (
    RL_EPISODE_SCHEMA_VERSION,
    RL_MANIFEST_SCHEMA_VERSION,
    RL_PREFERENCE_SCHEMA_VERSION,
    RL_REWARD_SCHEMA_VERSION,
)

VALIDATION_SCHEMA_VERSION = "hfr.validation.v1"


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
    preferences = _read_jsonl_objects(export_dir / "preferences.jsonl", target, "preferences.jsonl")
    if manifest is not None:
        _validate_training_manifest(manifest, target, episodes, rewards, preferences)
    _validate_episodes(episodes, target)
    _validate_rewards(rewards, target, episodes)
    _validate_preferences(preferences, target, episodes)
    target.details.update(
        {
            "episode_count": len(episodes),
            "reward_count": len(rewards),
            "preference_count": len(preferences),
        }
    )
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
    preferences: list[dict[str, Any]],
) -> None:
    _require_equal(manifest, "schema_version", RL_MANIFEST_SCHEMA_VERSION, target)
    expected_counts = {
        "episode_count": len(episodes),
        "reward_count": len(rewards),
        "preference_count": len(preferences),
    }
    for field_name, expected in expected_counts.items():
        if manifest.get(field_name) != expected:
            target.errors.append(f"manifest.{field_name} expected {expected}, got {manifest.get(field_name)!r}.")
    if not isinstance(manifest.get("outputs"), dict):
        target.errors.append("manifest.outputs must be an object.")
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
        if not isinstance(reward.get("attribution"), list):
            target.errors.append(f"rewards[{index}].attribution must be a list.")


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


def _is_windows_absolute(value: str) -> bool:
    normalized = value.replace("/", "\\")
    return (len(normalized) >= 3 and normalized[1:3] == ":\\" and normalized[0].isalpha()) or normalized.startswith("\\\\")
