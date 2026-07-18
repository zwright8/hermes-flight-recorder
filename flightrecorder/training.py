"""Training-data exports for future agent-improvement loops."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import CONTRACT_SCOPES, compare_scorecards
from .compare_gate import compare_movement_summary
from .redaction import redact_text
from .scorers import TASK_COMPLETION_SCHEMA_VERSION
from .trace_observability import build_trace_signal

RL_MANIFEST_SCHEMA_VERSION = "hfr.rl.manifest.v1"
RL_EPISODE_SCHEMA_VERSION = "hfr.rl.episode.v1"
RL_REWARD_SCHEMA_VERSION = "hfr.rl.reward.v1"
RL_STEP_REWARD_SCHEMA_VERSION = "hfr.rl.step_reward.v1"
RL_PREFERENCE_SCHEMA_VERSION = "hfr.rl.preference.v1"
RL_FAILURE_MODE_SCHEMA_VERSION = "hfr.rl.failure_mode.v1"
RL_CURRICULUM_SCHEMA_VERSION = "hfr.rl.curriculum.v1"
RL_SFT_SCHEMA_VERSION = "hfr.rl.sft.v1"
RL_DPO_SCHEMA_VERSION = "hfr.rl.dpo.v1"
RL_REWARD_MODEL_SCHEMA_VERSION = "hfr.rl.reward_model.v1"
RL_DATASET_METRICS_SCHEMA_VERSION = "hfr.rl.dataset_metrics.v1"
RL_DATASET_SPLITS_SCHEMA_VERSION = "hfr.rl.dataset_splits.v1"
RL_DATASET_REGISTRY_SCHEMA_VERSION = "hfr.rl.dataset_registry.v1"
RL_REDACTION_STATUS_SCHEMA_VERSION = "hfr.rl.redaction_status.v1"
RL_LABEL_PROVENANCE_SCHEMA_VERSION = "hfr.rl.label_provenance.v1"
COMPARE_RL_MANIFEST_SCHEMA_VERSION = "hfr.compare_rl.manifest.v1"
COMPARE_RL_PAIR_SCHEMA_VERSION = "hfr.compare_rl.pair.v1"
COMPARE_RL_DPO_SCHEMA_VERSION = "hfr.compare_rl.dpo.v1"

REWARD_SCALES = {"score", "binary", "signed"}
DATASET_SPLIT_NAMES = ("train", "validation", "test")
DATASET_SPLIT_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}
DATASET_SPLIT_SEED = "hfr.dataset_split.v1"
DATASET_SPLIT_ARTIFACTS = ("episodes", "rewards", "step_rewards", "preferences", "failure_modes", "sft", "dpo", "reward_model")
EVENT_INDEX_RE = re.compile(r"event #(\d+)")
FAMILY_SUFFIX_RE = re.compile(r"([_-](good|bad|pass|fail|passing|failing|chosen|rejected))+$", re.IGNORECASE)
UNREDACTED_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)(?<![:\w.-])[\w.-]*(?:api[_-]?key|secret|token|password|authorization|bearer)[\w.-]*"
    r"\s*[:=]\s*[\"']?(?!\[REDACTED\]\b)[^\"'\s,;}]+"
)


class TrainingExportError(ValueError):
    """Raised when RL training artifacts cannot be exported."""


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    run_dir: Path
    trace: dict[str, Any]
    scorecard: dict[str, Any]
    lineage_path: Path | None = None
    lineage: dict[str, Any] | None = None
    state_diff: dict[str, Any] | None = None


def export_rl_dataset(
    runs_dir: str | Path,
    out_dir: str | Path,
    *,
    reward_scale: str = "score",
    min_score_gap: int = 1,
    max_pairs_per_family: int = 0,
    preserve_paths: bool = False,
    metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Export completed run directories as RL-ready JSONL artifacts."""
    if reward_scale not in REWARD_SCALES:
        raise TrainingExportError(f"Unsupported reward scale {reward_scale!r}; choose one of {sorted(REWARD_SCALES)}")
    if min_score_gap < 0:
        raise TrainingExportError("min_score_gap must be non-negative")
    if max_pairs_per_family < 0:
        raise TrainingExportError("max_pairs_per_family must be non-negative")

    source = Path(runs_dir)
    target = Path(out_dir)
    export_metadata = _metadata(metadata)
    records = load_run_records(source)
    target.mkdir(parents=True, exist_ok=True)

    episodes = [_episode_record(record, reward_scale, preserve_paths) for record in records]
    rewards = [_reward_record(record, reward_scale) for record in records]
    step_rewards = _step_reward_records(records, reward_scale)
    preferences = _preference_records(episodes, min_score_gap=min_score_gap, max_pairs_per_family=max_pairs_per_family)
    failure_modes = [_failure_mode_record(record, rule, reward_scale) for record in records for rule in _failed_rules(record.scorecard)]
    curriculum = _curriculum_record(episodes, failure_modes)
    sft = _sft_records(episodes)
    dpo = _dpo_records(preferences)
    reward_model = _reward_model_records(episodes)
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
    dataset_splits, split_rows = _dataset_splits(rows_by_artifact)
    redaction_status = build_redaction_status(
        redaction_scan_artifacts(rows_by_artifact, curriculum, metadata=export_metadata)
    )
    if not redaction_status["passed"]:
        raise TrainingExportError(
            "Training export contains unredacted secret-like values; rerun collection with redaction enabled."
        )
    dataset_metrics = _dataset_metrics(
        episodes,
        rewards,
        step_rewards,
        preferences,
        failure_modes,
        sft,
        dpo,
        reward_model,
        dataset_splits,
        redaction_status,
        reward_scale,
        export_metadata,
    )

    paths = {
        "episodes": target / "episodes.jsonl",
        "rewards": target / "rewards.jsonl",
        "step_rewards": target / "step_rewards.jsonl",
        "preferences": target / "preferences.jsonl",
        "failure_modes": target / "failure_modes.jsonl",
        "curriculum": target / "curriculum.json",
        "sft": target / "sft.jsonl",
        "dpo": target / "dpo.jsonl",
        "reward_model": target / "reward_model.jsonl",
        "dataset_metrics": target / "dataset_metrics.json",
        "dataset_splits": target / "dataset_splits.json",
        "dataset_card": target / "DATASET_CARD.md",
        "dataset_registry": target / "dataset_registry.json",
        "manifest": target / "manifest.json",
    }
    for split_name in DATASET_SPLIT_NAMES:
        for artifact_name in DATASET_SPLIT_ARTIFACTS:
            paths[f"{split_name}_{artifact_name}"] = target / "splits" / split_name / f"{artifact_name}.jsonl"
    _write_jsonl(paths["episodes"], episodes)
    _write_jsonl(paths["rewards"], rewards)
    _write_jsonl(paths["step_rewards"], step_rewards)
    _write_jsonl(paths["preferences"], preferences)
    _write_jsonl(paths["failure_modes"], failure_modes)
    _write_json(paths["curriculum"], curriculum)
    _write_jsonl(paths["sft"], sft)
    _write_jsonl(paths["dpo"], dpo)
    _write_jsonl(paths["reward_model"], reward_model)
    _write_json(paths["dataset_metrics"], dataset_metrics)
    _write_json(paths["dataset_splits"], dataset_splits)
    for split_name in DATASET_SPLIT_NAMES:
        for artifact_name in DATASET_SPLIT_ARTIFACTS:
            _write_jsonl(paths[f"{split_name}_{artifact_name}"], split_rows[split_name][artifact_name])

    pre_card_fingerprints = _artifact_fingerprints(
        paths,
        preserve_paths,
        exclude={"manifest", "dataset_registry", "dataset_card"},
    )
    dataset_version = _dataset_version_id(pre_card_fingerprints)
    manifest = {
        "schema_version": RL_MANIFEST_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": dataset_version,
        "source_runs_dir": _display_path(source, preserve_paths),
        "output_dir": _display_path(target, preserve_paths),
        "reward_scale": reward_scale,
        "min_score_gap": min_score_gap,
        "max_pairs_per_family": max_pairs_per_family,
        "run_count": len(records),
        "episode_count": len(episodes),
        "reward_count": len(rewards),
        "step_reward_count": len(step_rewards),
        "preference_count": len(preferences),
        "failure_mode_count": len(failure_modes),
        "sft_count": len(sft),
        "dpo_count": len(dpo),
        "reward_model_count": len(reward_model),
        "quality_flag_count": len(dataset_metrics.get("quality_flags", [])),
        "source_fingerprint_coverage": dataset_metrics.get("source_fingerprint_coverage"),
        "dataset_splits": dataset_splits["summary"],
        "redaction_status": redaction_status,
        "label_provenance": dataset_metrics.get("label_provenance"),
        "task_families": sorted({str(episode["task_family"]) for episode in episodes}),
        "outputs": {name: _display_path(path, preserve_paths) for name, path in paths.items()},
        "registry": {
            "schema_version": RL_DATASET_REGISTRY_SCHEMA_VERSION,
            "path": _display_path(paths["dataset_registry"], preserve_paths),
            "selection_key": dataset_version,
            "notes": [
                "dataset_registry.json fingerprints this manifest after it is written.",
                "manifest.json does not fingerprint dataset_registry.json to avoid a circular hash.",
            ],
        },
        "notes": [
            "Exports are built from normalized_trace.json and scorecard.json.",
            "Use these artifacts as reward/eval data, not as a complete trainer.",
            "failure_modes.jsonl exposes one failed-rule record per episode for curriculum construction.",
            "New scorecards include evidence_refs for structured event/final-answer/episode attribution.",
            "step_rewards.jsonl flattens failed-rule attribution into one row per event/final-answer/episode target.",
            "sft.jsonl, dpo.jsonl, and reward_model.jsonl are trainer-ready views over the canonical evidence files.",
            "dataset_metrics.json and DATASET_CARD.md summarize export quality and coverage.",
            "dataset_splits.json and splits/<split>/*.jsonl partition rows by task family to reduce train/eval leakage.",
            "dataset_registry.json records the registered dataset version, manifest hash, source hashes, redaction status, split metadata, quality flags, and label provenance.",
            "SFT and positive preference labels require configured task-completion evidence; final-answer-only success claims are excluded from trainer-ready views.",
            "episodes.jsonl includes source_lineage and source_fingerprints when the originating run emitted artifact_lineage.json.",
            "Reward labels are deterministic scenario-policy judgments and can be reward-hacked if scenarios are weak.",
        ],
    }
    if export_metadata:
        manifest["metadata"] = export_metadata
    _write_text(paths["dataset_card"], _dataset_card(manifest, dataset_metrics))
    manifest["artifact_fingerprints"] = _artifact_fingerprints(paths, preserve_paths, exclude={"manifest", "dataset_registry"})
    _write_json(paths["manifest"], manifest)
    _write_json(paths["dataset_registry"], _dataset_registry_record(manifest, dataset_metrics, dataset_splits, episodes, paths, preserve_paths))
    return manifest


def export_compare_rl_dataset(
    baseline_dir: str | Path,
    candidate_dir: str | Path,
    out_dir: str | Path,
    *,
    reward_scale: str = "score",
    min_score_gap: int = 1,
    contract_scope: str = "scenario",
    preserve_paths: bool = False,
    metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Export baseline-vs-candidate comparisons as preference training artifacts."""
    if reward_scale not in REWARD_SCALES:
        raise TrainingExportError(f"Unsupported reward scale {reward_scale!r}; choose one of {sorted(REWARD_SCALES)}")
    if min_score_gap < 0:
        raise TrainingExportError("min_score_gap must be non-negative")
    if contract_scope not in CONTRACT_SCOPES:
        raise TrainingExportError(f"contract_scope must be one of {sorted(CONTRACT_SCOPES)!r}; got {contract_scope!r}")

    baseline_root = Path(baseline_dir)
    candidate_root = Path(candidate_dir)
    target = Path(out_dir)
    export_metadata = _metadata(metadata)
    baseline = _records_by_scenario(load_run_records(baseline_root), "baseline")
    candidate = _records_by_scenario(load_run_records(candidate_root), "candidate")
    target.mkdir(parents=True, exist_ok=True)

    pairs: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    missing_in_candidate = sorted(set(baseline) - set(candidate))
    new_in_candidate = sorted(set(candidate) - set(baseline))
    for scenario_id in sorted(set(baseline) & set(candidate)):
        baseline_record = baseline[scenario_id]
        candidate_record = candidate[scenario_id]
        baseline_episode = _comparison_episode_record(baseline_record, "baseline", reward_scale, preserve_paths)
        candidate_episode = _comparison_episode_record(candidate_record, "candidate", reward_scale, preserve_paths)
        contract = _episode_contract_comparison(baseline_episode, candidate_episode, contract_scope)
        baseline_score = _score(baseline_record.scorecard)
        candidate_score = _score(candidate_record.scorecard)
        score_delta = candidate_score - baseline_score
        if abs(score_delta) < min_score_gap:
            skipped.append(
                {
                    "scenario_id": scenario_id,
                    "reason": f"score gap {abs(score_delta)} is below min_score_gap {min_score_gap}",
                    "baseline_score": baseline_score,
                    "candidate_score": candidate_score,
                    "contract_fingerprint_status": contract["status"],
                    "contract_fingerprint_reasons": contract["reasons"],
                }
            )
            continue
        comparison = compare_scorecards(
            baseline_record.scorecard,
            candidate_record.scorecard,
            baseline_label=_display_path(baseline_record.run_dir, preserve_paths),
            candidate_label=_display_path(candidate_record.run_dir, preserve_paths),
        )
        if score_delta > 0:
            chosen_side = "candidate"
            rejected_side = "baseline"
        else:
            chosen_side = "baseline"
            rejected_side = "candidate"
        pairs.append(
            _comparison_pair_record(
                scenario_id,
                baseline_episode,
                candidate_episode,
                comparison,
                chosen_side,
                rejected_side,
                contract,
            )
        )

    dpo = _comparison_dpo_records(pairs)
    paths = {
        "improvement_pairs": target / "improvement_pairs.jsonl",
        "improvement_dpo": target / "improvement_dpo.jsonl",
        "manifest": target / "manifest.json",
        "improvement_card": target / "IMPROVEMENT_CARD.md",
    }
    candidate_win_count = sum(1 for pair in pairs if pair.get("chosen_side") == "candidate")
    baseline_win_count = sum(1 for pair in pairs if pair.get("chosen_side") == "baseline")
    contract_drift_count = sum(1 for pair in pairs if pair.get("contract_fingerprint_status") == "drifted")
    unverified_contract_count = sum(1 for pair in pairs if pair.get("contract_fingerprint_status") == "unverified")
    movement = compare_movement_summary(pairs)
    manifest: dict[str, Any] = {
        "schema_version": COMPARE_RL_MANIFEST_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_runs_dir": _display_path(baseline_root, preserve_paths),
        "candidate_runs_dir": _display_path(candidate_root, preserve_paths),
        "output_dir": _display_path(target, preserve_paths),
        "reward_scale": reward_scale,
        "min_score_gap": min_score_gap,
        "contract_scope": contract_scope,
        "baseline_run_count": len(baseline),
        "candidate_run_count": len(candidate),
        "paired_scenario_count": len(set(baseline) & set(candidate)),
        "pair_count": len(pairs),
        "dpo_count": len(dpo),
        "candidate_win_count": candidate_win_count,
        "baseline_win_count": baseline_win_count,
        "candidate_win_scenarios": movement["candidate_win_scenarios"],
        "baseline_win_scenarios": movement["baseline_win_scenarios"],
        "task_completion_improvement_count": movement["task_completion_improvement_count"],
        "task_completion_regression_count": movement["task_completion_regression_count"],
        "task_completion_improvement_scenarios": movement["task_completion_improvement_scenarios"],
        "task_completion_regression_scenarios": movement["task_completion_regression_scenarios"],
        "fixed_rule_counts": movement["fixed_rule_counts"],
        "regressed_rule_counts": movement["regressed_rule_counts"],
        "new_critical_failure_counts": movement["new_critical_failure_counts"],
        "contract_drift_count": contract_drift_count,
        "unverified_contract_count": unverified_contract_count,
        "skipped_pair_count": len(skipped),
        "missing_in_candidate": missing_in_candidate,
        "new_in_candidate": new_in_candidate,
        "skipped_pairs": skipped,
        "outputs": {name: _display_path(path, preserve_paths) for name, path in paths.items()},
        "notes": [
            "Comparison exports are built from paired baseline/candidate normalized_trace.json and scorecard.json files.",
            "The chosen side is whichever paired run has the higher deterministic scorecard score.",
            "Candidate wins describe measurable improvements; baseline wins describe regressions to avoid.",
            "Manifest movement fields summarize task-completion, scenario, and rule deltas so exports remain comparable without a separate gate artifact.",
            "Pairs include contract_fingerprint_status so trainer handoffs can reject drifted or unverified comparisons.",
            "Use these artifacts as preference/eval data, not as a complete trainer.",
        ],
    }
    if export_metadata:
        manifest["metadata"] = export_metadata
    _write_jsonl(paths["improvement_pairs"], pairs)
    _write_jsonl(paths["improvement_dpo"], dpo)
    _write_text(paths["improvement_card"], _improvement_card(manifest, pairs))
    manifest["artifact_fingerprints"] = _artifact_fingerprints(paths, preserve_paths, exclude={"manifest"})
    _write_json(paths["manifest"], manifest)
    return manifest


def load_run_records(runs_dir: str | Path) -> list[RunRecord]:
    """Load run directories that contain normalized traces and scorecards."""
    root = Path(runs_dir)
    if not root.exists():
        raise TrainingExportError(f"Runs directory not found: {root}")
    if not root.is_dir():
        raise TrainingExportError(f"Runs path is not a directory: {root}")

    records: list[RunRecord] = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        trace_path = run_dir / "normalized_trace.json"
        score_path = run_dir / "scorecard.json"
        lineage_path = run_dir / "artifact_lineage.json"
        state_diff_path = run_dir / "state_diff.json"
        if not trace_path.exists() or not score_path.exists():
            continue
        trace = _read_json(trace_path)
        scorecard = _read_json(score_path)
        if not isinstance(trace, dict) or not isinstance(scorecard, dict):
            raise TrainingExportError(f"Run {run_dir} must contain JSON objects")
        lineage: dict[str, Any] | None = None
        if lineage_path.exists():
            raw_lineage = _read_json(lineage_path)
            if isinstance(raw_lineage, dict):
                lineage = raw_lineage
        state_diff: dict[str, Any] | None = None
        if state_diff_path.exists():
            raw_state_diff = _read_json(state_diff_path)
            if not isinstance(raw_state_diff, dict):
                raise TrainingExportError(f"Run {run_dir} state_diff.json must contain a JSON object")
            state_diff = raw_state_diff
        records.append(
            RunRecord(
                run_id=run_dir.name,
                run_dir=run_dir,
                trace=trace,
                scorecard=scorecard,
                lineage_path=lineage_path if lineage_path.exists() else None,
                lineage=lineage,
                state_diff=state_diff,
            )
        )

    if not records:
        raise TrainingExportError(f"No completed Flight Recorder runs found in {root}")
    return records


def _episode_record(record: RunRecord, reward_scale: str, preserve_paths: bool) -> dict[str, Any]:
    trace = record.trace
    scorecard = record.scorecard
    score = _score(scorecard)
    passed = bool(scorecard.get("passed"))
    scenario_id = str(scorecard.get("scenario_id") or record.run_id)
    scenario_title = str(scorecard.get("scenario_title") or scenario_id)
    events = [_event_record(index, event) for index, event in enumerate(trace.get("events", []))]
    final_answer = str(trace.get("final_answer") or "")
    failed_rules = _failed_rule_ids(scorecard)
    source_fingerprints = _source_fingerprints(record)
    source_fingerprint_status = _source_fingerprint_status(source_fingerprints)
    task_completion = _task_completion(scorecard)
    state_diff = _state_diff_summary(record.state_diff)
    episode = {
        "schema_version": RL_EPISODE_SCHEMA_VERSION,
        "episode_id": record.run_id,
        "source_run": _display_path(record.run_dir, preserve_paths),
        "scenario_id": scenario_id,
        "scenario_title": scenario_title,
        "task_family": _scorecard_task_family(scorecard, scenario_id),
        "prompt": _prompt_from_trace(trace),
        "source_format": trace.get("session", {}).get("source_format", "unknown"),
        "model": trace.get("session", {}).get("model", "unknown"),
        "events": events,
        "final_answer": final_answer,
        "trace_signal": build_trace_signal(events, final_answer),
        "source_fingerprint_status": source_fingerprint_status,
        "source_fingerprints": source_fingerprints,
        "task_completion": task_completion,
        "state_diff": state_diff,
        "outcome": {
            "passed": passed,
            "score": score,
            "pass_threshold": scorecard.get("pass_threshold"),
            "reward": _reward_value(scorecard, reward_scale),
            "critical_failures": scorecard.get("critical_failures", []),
            "failed_rules": failed_rules,
            "task_completion_status": task_completion["status"],
            "task_completion_passed": task_completion["passed"],
            "state_changed": state_diff["changed"],
            "state_change_count": state_diff["change_count"],
            "summary": scorecard.get("summary", ""),
        },
    }
    if record.lineage_path is not None:
        episode["source_lineage"] = _display_path(record.lineage_path, preserve_paths)
    return episode


def _task_completion(scorecard: dict[str, Any]) -> dict[str, Any]:
    value = scorecard.get("task_completion")
    if isinstance(value, dict):
        return value
    return _default_task_completion()


def _default_task_completion() -> dict[str, Any]:
    return {
        "schema_version": TASK_COMPLETION_SCHEMA_VERSION,
        "status": "not_applicable",
        "passed": True,
        "task_evidence_configured": False,
        "required_check_count": 0,
        "passed_check_count": 0,
        "failed_check_count": 0,
        "blocking_rule_ids": [],
        "summary": "No task-completion evidence assertions were configured.",
        "checks": [],
        "evidence_refs": [],
        "missing_evidence_refs": [],
    }


def _state_diff_summary(diff: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(diff, dict):
        return {
            "schema_version": "hfr.state_diff.summary.v1",
            "available": False,
            "changed": False,
            "change_count": 0,
            "truncated": False,
            "summary": "No state diff artifact was available.",
            "changes": [],
        }
    changes = diff.get("changes") if isinstance(diff.get("changes"), list) else []
    return {
        "schema_version": "hfr.state_diff.summary.v1",
        "available": True,
        "changed": bool(diff.get("changed")),
        "change_count": _int_value(diff.get("change_count")),
        "truncated": bool(diff.get("truncated")),
        "summary": str(diff.get("summary") or ""),
        "changes": [
            {
                "path": str(change.get("path") or ""),
                "kind": str(change.get("kind") or ""),
            }
            for change in changes
            if isinstance(change, dict)
        ],
    }


def _reward_record(record: RunRecord, reward_scale: str) -> dict[str, Any]:
    scorecard = record.scorecard
    scenario_id = str(scorecard.get("scenario_id") or record.run_id)
    rule_rewards = [_rule_reward(rule) for rule in scorecard.get("rules", []) if isinstance(rule, dict)]
    source_fingerprints = _source_fingerprints(record)
    task_completion = _task_completion(scorecard)
    state_diff = _state_diff_summary(record.state_diff)
    return {
        "schema_version": RL_REWARD_SCHEMA_VERSION,
        "episode_id": record.run_id,
        "scenario_id": scenario_id,
        "task_family": _scorecard_task_family(scorecard, scenario_id),
        "source_fingerprint_status": _source_fingerprint_status(source_fingerprints),
        "source_fingerprints": source_fingerprints,
        "reward_scale": reward_scale,
        "reward": _reward_value(scorecard, reward_scale),
        "score": _score(scorecard),
        "passed": bool(scorecard.get("passed")),
        "task_completion_status": task_completion["status"],
        "task_completion_passed": task_completion["passed"],
        "state_changed": state_diff["changed"],
        "state_change_count": state_diff["change_count"],
        "terminal": True,
        "critical_failures": scorecard.get("critical_failures", []),
        "rule_rewards": rule_rewards,
        "attribution": _reward_attribution(scorecard),
    }


def _step_reward_records(records: list[RunRecord], reward_scale: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        scorecard = record.scorecard
        scenario_id = str(scorecard.get("scenario_id") or record.run_id)
        task_family = _scorecard_task_family(scorecard, scenario_id)
        source_fingerprints = _source_fingerprints(record)
        source_fingerprint_status = _source_fingerprint_status(source_fingerprints)
        rules_by_id = {
            str(rule.get("id")): rule
            for rule in _failed_rules(scorecard)
            if rule.get("id")
        }
        attributions = _reward_attribution(scorecard)
        attribution_counts: dict[str, int] = {}
        attribution_indexes: dict[str, int] = {}
        allocated_by_rule: dict[str, float] = {}
        for attribution in attributions:
            rule_id = str(attribution.get("rule_id") or "unknown_rule")
            attribution_counts[rule_id] = attribution_counts.get(rule_id, 0) + 1

        for index, attribution in enumerate(attributions):
            rule_id = str(attribution.get("rule_id") or "unknown_rule")
            rule = rules_by_id.get(rule_id, {})
            target = attribution.get("target") if attribution.get("target") in {"event", "final_answer", "episode"} else "episode"
            attribution_count = attribution_counts.get(rule_id, 1)
            allocation_index = attribution_indexes.get(rule_id, 0)
            attribution_indexes[rule_id] = allocation_index + 1
            rule_reward_delta = float(attribution.get("reward_delta", 0.0) or 0.0)
            if allocation_index == attribution_count - 1:
                reward_delta = round(rule_reward_delta - allocated_by_rule.get(rule_id, 0.0), 6)
            else:
                reward_delta = round(rule_reward_delta / attribution_count, 6)
                allocated_by_rule[rule_id] = round(allocated_by_rule.get(rule_id, 0.0) + reward_delta, 6)
            allocation_weight = round(reward_delta / rule_reward_delta, 6) if rule_reward_delta else 0.0
            row = {
                "schema_version": RL_STEP_REWARD_SCHEMA_VERSION,
                "step_reward_id": f"{record.run_id}:{rule_id}:{index}",
                "episode_id": record.run_id,
                "scenario_id": scenario_id,
                "scenario_title": str(scorecard.get("scenario_title") or scenario_id),
                "task_family": task_family,
                "source_fingerprint_status": source_fingerprint_status,
                "source_fingerprints": source_fingerprints,
                "target": target,
                "rule_id": rule_id,
                "rule_name": str(rule.get("name") or rule_id),
                "critical": bool(rule.get("critical")),
                "penalty": int(rule.get("penalty", 0) or 0),
                "score": _score(scorecard),
                "episode_reward": _reward_value(scorecard, reward_scale),
                "reward_scale": reward_scale,
                "reward_delta": reward_delta,
                "rule_reward_delta": rule_reward_delta,
                "allocation_weight": allocation_weight,
                "allocation_index": allocation_index,
                "attribution_count": attribution_count,
                "passed": bool(scorecard.get("passed")),
                "evidence": str(attribution.get("evidence") or ""),
            }
            if target == "event" and isinstance(attribution.get("event_index"), int) and not isinstance(attribution.get("event_index"), bool):
                row["event_index"] = attribution["event_index"]
            if isinstance(attribution.get("evidence_ref"), dict):
                row["evidence_ref"] = attribution["evidence_ref"]
            rows.append(row)
    return rows


def _preference_records(
    episodes: list[dict[str, Any]],
    *,
    min_score_gap: int,
    max_pairs_per_family: int,
) -> list[dict[str, Any]]:
    preferences: list[dict[str, Any]] = []
    by_family: dict[str, list[dict[str, Any]]] = {}
    for episode in episodes:
        by_family.setdefault(str(episode["task_family"]), []).append(episode)

    for family, family_episodes in sorted(by_family.items()):
        pairs_for_family = 0
        ordered = sorted(
            family_episodes,
            key=lambda item: (-int(item["outcome"]["score"]), str(item["episode_id"])),
        )
        for chosen in ordered:
            for rejected in reversed(ordered):
                chosen_score = int(chosen["outcome"]["score"])
                rejected_score = int(rejected["outcome"]["score"])
                if chosen_score - rejected_score < min_score_gap:
                    continue
                if not _positive_label_eligible(chosen):
                    continue
                preferences.append(_preference_record(family, chosen, rejected))
                pairs_for_family += 1
                if max_pairs_per_family and pairs_for_family >= max_pairs_per_family:
                    break
            if max_pairs_per_family and pairs_for_family >= max_pairs_per_family:
                break
    return preferences


def _preference_record(family: str, chosen: dict[str, Any], rejected: dict[str, Any]) -> dict[str, Any]:
    chosen_score = int(chosen["outcome"]["score"])
    rejected_score = int(rejected["outcome"]["score"])
    rejected_failures = rejected["outcome"].get("failed_rules", [])
    reason = f"Chosen score {chosen_score} exceeded rejected score {rejected_score}."
    if rejected_failures:
        reason += f" Rejected failed rules: {', '.join(str(rule) for rule in rejected_failures)}."
    return {
        "schema_version": RL_PREFERENCE_SCHEMA_VERSION,
        "preference_id": f"{family}:{chosen['episode_id']}>{rejected['episode_id']}",
        "task_family": family,
        "prompt": chosen.get("prompt") or rejected.get("prompt") or "",
        "chosen_episode_id": chosen["episode_id"],
        "rejected_episode_id": rejected["episode_id"],
        "chosen_score": chosen_score,
        "rejected_score": rejected_score,
        "score_gap": chosen_score - rejected_score,
        "reason": reason,
        "chosen": _preference_view(chosen),
        "rejected": _preference_view(rejected),
        "label_provenance": _paired_label_provenance(chosen, rejected, "dpo_preference"),
    }


def _preference_view(episode: dict[str, Any]) -> dict[str, Any]:
    return {
        "episode_id": episode["episode_id"],
        "scenario_id": episode["scenario_id"],
        "passed": episode["outcome"]["passed"],
        "score": episode["outcome"]["score"],
        "reward": episode["outcome"]["reward"],
        "failed_rules": episode["outcome"]["failed_rules"],
        "task_completion": episode.get("task_completion", _default_task_completion()),
        "state_diff": episode.get("state_diff", _state_diff_summary(None)),
        "source_fingerprint_status": episode.get("source_fingerprint_status", "unverified"),
        "source_fingerprints": episode.get("source_fingerprints", {}),
        "events": episode["events"],
        "final_answer": episode["final_answer"],
    }


def _failure_mode_record(record: RunRecord, rule: dict[str, Any], reward_scale: str) -> dict[str, Any]:
    scorecard = record.scorecard
    scenario_id = str(scorecard.get("scenario_id") or record.run_id)
    rule_id = str(rule.get("id") or "unknown_rule")
    attribution = [item for item in _reward_attribution(scorecard) if item.get("rule_id") == rule_id]
    source_fingerprints = _source_fingerprints(record)
    return {
        "schema_version": RL_FAILURE_MODE_SCHEMA_VERSION,
        "failure_id": f"{record.run_id}:{rule_id}",
        "episode_id": record.run_id,
        "scenario_id": scenario_id,
        "scenario_title": str(scorecard.get("scenario_title") or scenario_id),
        "task_family": _scorecard_task_family(scorecard, scenario_id),
        "source_fingerprint_status": _source_fingerprint_status(source_fingerprints),
        "source_fingerprints": source_fingerprints,
        "rule_id": rule_id,
        "rule_name": str(rule.get("name") or rule_id),
        "critical": bool(rule.get("critical")),
        "penalty": int(rule.get("penalty", 0) or 0),
        "score": _score(scorecard),
        "reward": _reward_value(scorecard, reward_scale),
        "summary": str(scorecard.get("summary") or ""),
        "evidence": [str(item) for item in rule.get("evidence", [])],
        "evidence_refs": _evidence_refs(rule),
        "attribution": attribution,
    }


def _curriculum_record(episodes: list[dict[str, Any]], failure_modes: list[dict[str, Any]]) -> dict[str, Any]:
    families: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        family = str(episode.get("task_family") or "unknown")
        outcome = episode.get("outcome") if isinstance(episode.get("outcome"), dict) else {}
        bucket = families.setdefault(
            family,
            {
                "task_family": family,
                "episode_count": 0,
                "passed": 0,
                "failed": 0,
                "scores": [],
                "failure_modes": {},
            },
        )
        bucket["episode_count"] += 1
        if outcome.get("passed"):
            bucket["passed"] += 1
        else:
            bucket["failed"] += 1
        if isinstance(outcome.get("score"), int):
            bucket["scores"].append(outcome["score"])

    for failure in failure_modes:
        family = str(failure.get("task_family") or "unknown")
        rule_id = str(failure.get("rule_id") or "unknown_rule")
        bucket = families.setdefault(
            family,
            {
                "task_family": family,
                "episode_count": 0,
                "passed": 0,
                "failed": 0,
                "scores": [],
                "failure_modes": {},
            },
        )
        mode = bucket["failure_modes"].setdefault(
            rule_id,
            {
                "rule_id": rule_id,
                "rule_name": failure.get("rule_name", rule_id),
                "count": 0,
                "critical_count": 0,
                "episode_ids": [],
                "scenario_ids": [],
                "failure_ids": [],
                "example_evidence": [],
                "example_evidence_refs": [],
                "_penalties": [],
            },
        )
        mode["count"] += 1
        if failure.get("critical"):
            mode["critical_count"] += 1
        penalty = _int_value(failure.get("penalty"))
        mode["_penalties"].append(penalty)
        episode_id = str(failure.get("episode_id") or "")
        if episode_id and episode_id not in mode["episode_ids"]:
            mode["episode_ids"].append(episode_id)
        scenario_id = str(failure.get("scenario_id") or "")
        if scenario_id and scenario_id not in mode["scenario_ids"]:
            mode["scenario_ids"].append(scenario_id)
        failure_id = str(failure.get("failure_id") or "")
        if failure_id and failure_id not in mode["failure_ids"]:
            mode["failure_ids"].append(failure_id)
        for evidence in failure.get("evidence", []):
            text = str(evidence)
            if text and text not in mode["example_evidence"]:
                mode["example_evidence"].append(text)
            if len(mode["example_evidence"]) >= 3:
                break
        for ref in failure.get("evidence_refs", []):
            if not isinstance(ref, dict):
                continue
            if ref not in mode["example_evidence_refs"]:
                mode["example_evidence_refs"].append(ref)
            if len(mode["example_evidence_refs"]) >= 3:
                break

    family_rows: list[dict[str, Any]] = []
    for family, bucket in sorted(families.items()):
        scores = bucket.pop("scores")
        failure_map = bucket.pop("failure_modes")
        bucket["average_score"] = round(sum(scores) / len(scores), 2) if scores else 0.0
        for mode in failure_map.values():
            penalties = mode.pop("_penalties", [])
            mode["max_penalty"] = max(penalties) if penalties else 0
            mode["average_penalty"] = round(sum(penalties) / len(penalties), 2) if penalties else 0.0
            mode["priority_score"] = _curriculum_priority_score(mode)
            mode["priority_band"] = _curriculum_priority_band(mode["priority_score"])
        bucket["failure_modes"] = sorted(
            failure_map.values(),
            key=lambda item: (-int(item["priority_score"]), -int(item["count"]), str(item["rule_id"])),
        )
        family_rows.append(bucket)

    return {
        "schema_version": RL_CURRICULUM_SCHEMA_VERSION,
        "episode_count": len(episodes),
        "failure_mode_count": len(failure_modes),
        "task_families": family_rows,
        "recommended_use": [
            "Use high-count critical failure modes as regression priorities.",
            "Sort failure_modes by priority_score to decide which scenario contracts or repair tasks to address first.",
            "Use passing episodes in the same task family as positive references.",
            "Treat this as curriculum metadata, not as an automatic trainer policy.",
        ],
    }


def _curriculum_priority_score(mode: dict[str, Any]) -> int:
    count = _int_value(mode.get("count"))
    critical_count = _int_value(mode.get("critical_count"))
    max_penalty = _int_value(mode.get("max_penalty"))
    return count * 10 + critical_count * 100 + max_penalty


def _curriculum_priority_band(priority_score: int) -> str:
    if priority_score >= 150:
        return "critical"
    if priority_score >= 75:
        return "high"
    if priority_score >= 25:
        return "medium"
    return "low"


def _sft_records(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        outcome = episode.get("outcome") if isinstance(episode.get("outcome"), dict) else {}
        response = str(episode.get("final_answer") or "")
        if outcome.get("passed") is not True or not response:
            continue
        if not _positive_label_eligible(episode):
            continue
        prompt = str(episode.get("prompt") or "")
        label_provenance = _label_provenance(episode, "sft_positive")
        rows.append(
            {
                "schema_version": RL_SFT_SCHEMA_VERSION,
                "sample_id": str(episode.get("episode_id") or ""),
                "episode_id": str(episode.get("episode_id") or ""),
                "scenario_id": str(episode.get("scenario_id") or ""),
                "task_family": str(episode.get("task_family") or "unknown"),
                "prompt": prompt,
                "response": response,
                "messages": _messages(prompt, response),
                "score": _score_value(outcome.get("score")),
                "reward": _numeric_value(outcome.get("reward")),
                "quality_gate": "passed_scorecard",
                "task_completion_status": str((episode.get("task_completion") or {}).get("status") or "not_applicable"),
                "task_completion_passed": bool((episode.get("task_completion") or {}).get("passed", True)),
                "state_changed": bool((episode.get("state_diff") or {}).get("changed", False)),
                "state_change_count": _int_value((episode.get("state_diff") or {}).get("change_count")),
                "source_fingerprint_status": str(episode.get("source_fingerprint_status") or "unverified"),
                "source_fingerprints": episode.get("source_fingerprints", {}),
                "label_provenance": label_provenance,
                "source_artifact": "episodes.jsonl",
            }
        )
    return rows


def _dpo_records(preferences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for preference in preferences:
        prompt = str(preference.get("prompt") or "")
        chosen_view = preference.get("chosen") if isinstance(preference.get("chosen"), dict) else {}
        rejected_view = preference.get("rejected") if isinstance(preference.get("rejected"), dict) else {}
        chosen = str(chosen_view.get("final_answer") or "")
        rejected = str(rejected_view.get("final_answer") or "")
        preference_id = str(preference.get("preference_id") or "")
        rows.append(
            {
                "schema_version": RL_DPO_SCHEMA_VERSION,
                "pair_id": preference_id,
                "preference_id": preference_id,
                "task_family": str(preference.get("task_family") or "unknown"),
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "chosen_messages": _messages(prompt, chosen),
                "rejected_messages": _messages(prompt, rejected),
                "chosen_episode_id": str(preference.get("chosen_episode_id") or ""),
                "rejected_episode_id": str(preference.get("rejected_episode_id") or ""),
                "chosen_score": _score_value(preference.get("chosen_score")),
                "rejected_score": _score_value(preference.get("rejected_score")),
                "score_gap": _score_value(preference.get("score_gap")),
                "reason": str(preference.get("reason") or ""),
                "chosen_source_fingerprint_status": str(chosen_view.get("source_fingerprint_status") or "unverified"),
                "rejected_source_fingerprint_status": str(rejected_view.get("source_fingerprint_status") or "unverified"),
                "chosen_source_fingerprints": chosen_view.get("source_fingerprints", {}),
                "rejected_source_fingerprints": rejected_view.get("source_fingerprints", {}),
                "label_provenance": preference.get("label_provenance", {}),
                "source_artifact": "preferences.jsonl",
            }
        )
    return rows


def _records_by_scenario(records: list[RunRecord], side: str) -> dict[str, RunRecord]:
    by_scenario: dict[str, RunRecord] = {}
    for record in records:
        scenario_id = str(record.scorecard.get("scenario_id") or record.run_id)
        if scenario_id in by_scenario:
            raise TrainingExportError(f"Duplicate scenario_id {scenario_id!r} in {side} runs")
        by_scenario[scenario_id] = record
    return by_scenario


def _comparison_episode_record(record: RunRecord, side: str, reward_scale: str, preserve_paths: bool) -> dict[str, Any]:
    episode = _episode_record(record, reward_scale, preserve_paths)
    episode["episode_id"] = f"{side}:{record.run_id}"
    episode["comparison_side"] = side
    return episode


def _comparison_pair_record(
    scenario_id: str,
    baseline_episode: dict[str, Any],
    candidate_episode: dict[str, Any],
    comparison: dict[str, Any],
    chosen_side: str,
    rejected_side: str,
    contract: dict[str, Any],
) -> dict[str, Any]:
    chosen = candidate_episode if chosen_side == "candidate" else baseline_episode
    rejected = baseline_episode if rejected_side == "baseline" else candidate_episode
    candidate_score_delta = int(comparison.get("score_delta", 0) or 0)
    candidate_outcome = "improved" if candidate_score_delta > 0 else "regressed"
    chosen_score = _score_value(chosen.get("outcome", {}).get("score") if isinstance(chosen.get("outcome"), dict) else None)
    rejected_score = _score_value(rejected.get("outcome", {}).get("score") if isinstance(rejected.get("outcome"), dict) else None)
    reason = (
        f"{chosen_side} score {chosen_score} beat {rejected_side} score {rejected_score}; "
        f"candidate_delta={candidate_score_delta}."
    )
    if comparison.get("fixes"):
        reason += f" Fixed rules: {', '.join(str(rule) for rule in comparison['fixes'])}."
    if comparison.get("regressions"):
        reason += f" Regressed rules: {', '.join(str(rule) for rule in comparison['regressions'])}."
    return {
        "schema_version": COMPARE_RL_PAIR_SCHEMA_VERSION,
        "pair_id": f"{scenario_id}:{chosen_side}>{rejected_side}",
        "scenario_id": scenario_id,
        "task_family": _task_family(scenario_id),
        "prompt": str(chosen.get("prompt") or rejected.get("prompt") or ""),
        "candidate_outcome": candidate_outcome,
        "candidate_score_delta": candidate_score_delta,
        "chosen_side": chosen_side,
        "rejected_side": rejected_side,
        "chosen_episode_id": chosen["episode_id"],
        "rejected_episode_id": rejected["episode_id"],
        "baseline_episode_id": baseline_episode["episode_id"],
        "candidate_episode_id": candidate_episode["episode_id"],
        "chosen_score": chosen_score,
        "rejected_score": rejected_score,
        "score_gap": chosen_score - rejected_score,
        "rule_fixes": comparison.get("fixes", []),
        "rule_regressions": comparison.get("regressions", []),
        "new_critical_failures": comparison.get("new_critical_failures", []),
        "contract_fingerprint_status": contract["status"],
        "contract_fingerprint_scope": contract["scope"],
        "contract_fingerprint_reasons": contract["reasons"],
        "contract_fingerprints": contract["fingerprints"],
        "reason": reason,
        "baseline": _preference_view(baseline_episode),
        "candidate": _preference_view(candidate_episode),
        "chosen": _preference_view(chosen),
        "rejected": _preference_view(rejected),
    }


def _comparison_dpo_records(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        prompt = str(pair.get("prompt") or "")
        chosen_view = pair.get("chosen") if isinstance(pair.get("chosen"), dict) else {}
        rejected_view = pair.get("rejected") if isinstance(pair.get("rejected"), dict) else {}
        chosen = _comparison_response_text(chosen_view)
        rejected = _comparison_response_text(rejected_view)
        pair_id = str(pair.get("pair_id") or "")
        rows.append(
            {
                "schema_version": COMPARE_RL_DPO_SCHEMA_VERSION,
                "pair_id": pair_id,
                "preference_id": pair_id,
                "scenario_id": str(pair.get("scenario_id") or ""),
                "task_family": str(pair.get("task_family") or "unknown"),
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "chosen_messages": _messages(prompt, chosen),
                "rejected_messages": _messages(prompt, rejected),
                "chosen_side": str(pair.get("chosen_side") or ""),
                "rejected_side": str(pair.get("rejected_side") or ""),
                "candidate_outcome": str(pair.get("candidate_outcome") or ""),
                "candidate_score_delta": int(pair.get("candidate_score_delta", 0) or 0),
                "chosen_episode_id": str(pair.get("chosen_episode_id") or ""),
                "rejected_episode_id": str(pair.get("rejected_episode_id") or ""),
                "chosen_score": _score_value(pair.get("chosen_score")),
                "rejected_score": _score_value(pair.get("rejected_score")),
                "score_gap": _score_value(pair.get("score_gap")),
                "chosen_task_completion_status": str((chosen_view.get("task_completion") or {}).get("status") or "not_applicable"),
                "rejected_task_completion_status": str((rejected_view.get("task_completion") or {}).get("status") or "not_applicable"),
                "chosen_task_completion_passed": bool((chosen_view.get("task_completion") or {}).get("passed", True)),
                "rejected_task_completion_passed": bool((rejected_view.get("task_completion") or {}).get("passed", True)),
                "contract_fingerprint_status": str(pair.get("contract_fingerprint_status") or "unverified"),
                "contract_fingerprint_scope": str(pair.get("contract_fingerprint_scope") or "scenario"),
                "contract_fingerprint_reasons": _string_list(pair.get("contract_fingerprint_reasons")),
                "contract_fingerprints": pair.get("contract_fingerprints") if isinstance(pair.get("contract_fingerprints"), dict) else {},
                "reason": str(pair.get("reason") or ""),
                "source_artifact": "improvement_pairs.jsonl",
            }
        )
    return rows


def _comparison_response_text(view: dict[str, Any]) -> str:
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
        detail = _comparison_event_detail(event)
        if detail:
            parts.append(detail)
        lines.append("- " + " ".join(parts))
    final_answer = str(view.get("final_answer") or "")
    if final_answer:
        lines.append(f"Final answer: {final_answer}")
    return "\n".join(lines)


def _comparison_event_detail(event: dict[str, Any]) -> str:
    for field_name in ("result", "args"):
        value = event.get(field_name)
        if isinstance(value, dict) and value:
            return json.dumps(value, sort_keys=True, ensure_ascii=False)
    text = str(event.get("text") or "").strip()
    return text[:500]


def _reward_model_records(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        outcome = episode.get("outcome") if isinstance(episode.get("outcome"), dict) else {}
        if outcome.get("passed") is True and not _positive_label_eligible(episode):
            continue
        prompt = str(episode.get("prompt") or "")
        response = str(episode.get("final_answer") or "")
        label_provenance = _label_provenance(episode, "reward_model")
        rows.append(
            {
                "schema_version": RL_REWARD_MODEL_SCHEMA_VERSION,
                "sample_id": str(episode.get("episode_id") or ""),
                "episode_id": str(episode.get("episode_id") or ""),
                "scenario_id": str(episode.get("scenario_id") or ""),
                "task_family": str(episode.get("task_family") or "unknown"),
                "prompt": prompt,
                "response": response,
                "messages": _messages(prompt, response),
                "score": _score_value(outcome.get("score")),
                "reward": _numeric_value(outcome.get("reward")),
                "passed": bool(outcome.get("passed")),
                "task_completion_status": str((episode.get("task_completion") or {}).get("status") or "not_applicable"),
                "task_completion_passed": bool((episode.get("task_completion") or {}).get("passed", True)),
                "state_changed": bool((episode.get("state_diff") or {}).get("changed", False)),
                "state_change_count": _int_value((episode.get("state_diff") or {}).get("change_count")),
                "failed_rules": _string_list(outcome.get("failed_rules")),
                "critical_failures": _string_list(outcome.get("critical_failures")),
                "source_fingerprint_status": str(episode.get("source_fingerprint_status") or "unverified"),
                "source_fingerprints": episode.get("source_fingerprints", {}),
                "label_provenance": label_provenance,
                "source_artifact": "episodes.jsonl",
            }
        )
    return rows


def build_redaction_status(artifacts: dict[str, Any]) -> dict[str, Any]:
    findings = scan_unredacted_secret_findings(artifacts)
    return {
        "schema_version": RL_REDACTION_STATUS_SCHEMA_VERSION,
        "passed": not findings,
        "scanner": "flightrecorder.generic_secret_assignment.v1",
        "unredacted_secret_count": len(findings),
        "finding_count": len(findings),
        "findings": findings,
        "notes": [
            "The scanner rejects generic secret assignments such as api_key=..., token=..., password=..., authorization=..., or bearer=....",
            "Export metadata is scanned as key=value pairs so secret-like metadata keys cannot carry raw values.",
            "Findings include redacted previews only; raw matched values are never written to dataset artifacts.",
        ],
    }


def scan_unredacted_secret_findings(artifacts: dict[str, Any], *, limit: int = 50) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for artifact_name, payload in sorted(artifacts.items()):
        for path, text in _walk_strings(payload):
            if _skip_redaction_scan_path(path):
                continue
            if not UNREDACTED_SECRET_ASSIGNMENT_RE.search(text):
                continue
            redacted = redact_text(text)
            findings.append(
                {
                    "artifact": artifact_name,
                    "path": path,
                    "issue": "unredacted_secret_assignment",
                    "redacted_preview": _preview(redacted),
                }
            )
            if len(findings) >= limit:
                return findings
    return findings


def _skip_redaction_scan_path(path: str) -> bool:
    field = path.rsplit(".", 1)[-1]
    if field.endswith("]"):
        field = field.rsplit("[", 1)[0]
    return field in {
        "check_id",
        "episode_id",
        "failure_mode_id",
        "preference_id",
        "reward_id",
        "rule_id",
        "scenario_id",
        "step_reward_id",
        "task_family",
    }


def build_label_provenance_summary(
    episodes: list[dict[str, Any]],
    sft: list[dict[str, Any]],
    dpo: list[dict[str, Any]],
    reward_model: list[dict[str, Any]],
) -> dict[str, Any]:
    positive_episode_ids = sorted(
        str(episode.get("episode_id") or "")
        for episode in episodes
        if isinstance(episode.get("outcome"), dict) and episode["outcome"].get("passed") is True
    )
    eligible_positive_episode_ids = sorted(
        str(episode.get("episode_id") or "")
        for episode in episodes
        if _positive_label_eligible(episode)
    )
    excluded_success_ids = sorted(set(positive_episode_ids) - set(eligible_positive_episode_ids))
    return {
        "schema_version": RL_LABEL_PROVENANCE_SCHEMA_VERSION,
        "positive_episode_count": len(positive_episode_ids),
        "eligible_positive_episode_count": len(eligible_positive_episode_ids),
        "final_answer_only_success_count": len(excluded_success_ids),
        "excluded_success_episode_ids": excluded_success_ids,
        "sft_label_count": len(sft),
        "dpo_pair_count": len(dpo),
        "reward_model_label_count": len(reward_model),
        "rules": [
            "Positive SFT rows require a passing scorecard, non-empty final answer, and task_completion.status=complete.",
            "DPO chosen rows require the chosen episode to satisfy the same positive-label eligibility rule.",
            "Reward-model exports keep failing episodes and eligible positive episodes; unsupported positive success claims are excluded.",
        ],
    }


def _positive_label_eligible(episode: dict[str, Any]) -> bool:
    outcome = episode.get("outcome") if isinstance(episode.get("outcome"), dict) else {}
    if outcome.get("passed") is not True:
        return False
    task = episode.get("task_completion") if isinstance(episode.get("task_completion"), dict) else {}
    return (
        task.get("task_evidence_configured") is True
        and task.get("status") == "complete"
        and task.get("passed") is True
    )


def _label_provenance(episode: dict[str, Any], label_type: str) -> dict[str, Any]:
    outcome = episode.get("outcome") if isinstance(episode.get("outcome"), dict) else {}
    task = episode.get("task_completion") if isinstance(episode.get("task_completion"), dict) else {}
    state_diff = episode.get("state_diff") if isinstance(episode.get("state_diff"), dict) else {}
    trace_signal = episode.get("trace_signal") if isinstance(episode.get("trace_signal"), dict) else {}
    scorecard_passed = outcome.get("passed") is True
    task_evidence_configured = task.get("task_evidence_configured") is True
    task_completion_passed = task.get("status") == "complete" and task.get("passed") is True
    evidence_sources: list[str] = []
    if task_evidence_configured:
        evidence_sources.append("task_completion")
    if state_diff.get("changed") is True:
        evidence_sources.append("state_diff")
    if trace_signal.get("has_tool_or_api_events") is True:
        evidence_sources.append("tool_or_api_trace")
    if not evidence_sources:
        evidence_sources.append("final_answer")
    final_answer_only_success = scorecard_passed and not task_completion_passed
    trainer_view_eligible = (not scorecard_passed) or _positive_label_eligible(episode)
    return {
        "schema_version": RL_LABEL_PROVENANCE_SCHEMA_VERSION,
        "label_type": label_type,
        "source": "scorecard_outcome",
        "scorecard_passed": scorecard_passed,
        "score": _score_value(outcome.get("score")),
        "reward": _numeric_value(outcome.get("reward")),
        "task_evidence_configured": task_evidence_configured,
        "task_completion_status": str(task.get("status") or "unknown"),
        "task_completion_passed": bool(task.get("passed", False)),
        "state_changed": bool(state_diff.get("changed", False)),
        "state_change_count": _int_value(state_diff.get("change_count")),
        "trace_has_tool_or_api_events": bool(trace_signal.get("has_tool_or_api_events", False)),
        "source_fingerprint_status": str(episode.get("source_fingerprint_status") or "unverified"),
        "evidence_sources": evidence_sources,
        "final_answer_only_success": final_answer_only_success,
        "trainer_view_eligible": trainer_view_eligible,
        "exclusion_reason": "missing_task_completion_success_evidence" if final_answer_only_success else "",
    }


def _paired_label_provenance(chosen: dict[str, Any], rejected: dict[str, Any], label_type: str) -> dict[str, Any]:
    chosen_provenance = _label_provenance(chosen, "chosen_positive")
    rejected_provenance = _label_provenance(rejected, "rejected_comparison")
    return {
        "schema_version": RL_LABEL_PROVENANCE_SCHEMA_VERSION,
        "label_type": label_type,
        "source": "scorecard_preference",
        "chosen": chosen_provenance,
        "rejected": rejected_provenance,
        "trainer_view_eligible": bool(chosen_provenance.get("trainer_view_eligible")),
        "exclusion_reason": str(chosen_provenance.get("exclusion_reason") or ""),
    }


def redaction_scan_artifacts(
    rows_by_artifact: dict[str, list[dict[str, Any]]],
    curriculum: dict[str, Any],
    *,
    metadata: dict[str, str] | None = None,
    extra_artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {name: rows for name, rows in rows_by_artifact.items()}
    artifacts["curriculum"] = curriculum
    if metadata:
        artifacts["metadata"] = metadata_redaction_scan_payload(metadata)
    if extra_artifacts:
        artifacts.update(extra_artifacts)
    return artifacts


def _redaction_scan_artifacts(rows_by_artifact: dict[str, list[dict[str, Any]]], curriculum: dict[str, Any]) -> dict[str, Any]:
    return redaction_scan_artifacts(rows_by_artifact, curriculum)


def metadata_redaction_scan_payload(metadata: dict[str, str]) -> list[str]:
    return [f"{key}={value}" for key, value in sorted(metadata.items())]


def _walk_strings(value: Any, path: str = "$"):
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_strings(item, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            yield from _walk_strings(item, f"{path}.{key}")


def _preview(value: str, limit: int = 160) -> str:
    rendered = value.replace("\n", "\\n")
    if len(rendered) <= limit:
        return rendered
    return rendered[: limit - 3] + "..."


def _dataset_version_id(artifact_fingerprints: dict[str, Any]) -> str:
    payload = {
        name: record.get("sha256")
        for name, record in sorted(artifact_fingerprints.items())
        if isinstance(record, dict) and isinstance(record.get("sha256"), str)
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return f"hfrds-{digest[:16]}"


def _dataset_registry_record(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    dataset_splits: dict[str, Any],
    episodes: list[dict[str, Any]],
    paths: dict[str, Path],
    preserve_paths: bool,
) -> dict[str, Any]:
    manifest_path = paths["manifest"]
    return {
        "schema_version": RL_DATASET_REGISTRY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": manifest["dataset_version"],
        "manifest": _file_fingerprint(manifest_path, preserve_paths),
        "registry_path": _display_path(paths["dataset_registry"], preserve_paths),
        "source_runs_dir": manifest.get("source_runs_dir"),
        "output_dir": manifest.get("output_dir"),
        "run_count": manifest.get("run_count"),
        "source_fingerprint_coverage": manifest.get("source_fingerprint_coverage"),
        "source_runs": _registry_source_runs(episodes),
        "split_metadata": dataset_splits.get("summary", {}),
        "split_assignments": dataset_splits.get("assignments", []),
        "redaction_status": manifest.get("redaction_status"),
        "quality_flags": metrics.get("quality_flags", []),
        "label_provenance": metrics.get("label_provenance", {}),
        "artifact_fingerprints": manifest.get("artifact_fingerprints", {}),
        "trainer_views": {
            "sft": {"path": manifest.get("outputs", {}).get("sft"), "count": manifest.get("sft_count")},
            "dpo": {"path": manifest.get("outputs", {}).get("dpo"), "count": manifest.get("dpo_count")},
            "reward_model": {
                "path": manifest.get("outputs", {}).get("reward_model"),
                "count": manifest.get("reward_model_count"),
            },
        },
        "selection": {
            "by_manifest": manifest.get("outputs", {}).get("manifest"),
            "by_dataset_version": manifest["dataset_version"],
        },
    }


def _file_fingerprint(path: Path, preserve_paths: bool) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": _display_path(path, preserve_paths),
        "exists": True,
        "size_bytes": stat.st_size,
        "sha256": _sha256_file(path),
    }


def _registry_source_runs(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "episode_id": str(episode.get("episode_id") or ""),
            "scenario_id": str(episode.get("scenario_id") or ""),
            "task_family": str(episode.get("task_family") or "unknown"),
            "source_fingerprint_status": str(episode.get("source_fingerprint_status") or "unverified"),
            "source_fingerprints": episode.get("source_fingerprints", {}),
        }
        for episode in episodes
    ]


def _dataset_splits(rows_by_artifact: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, Any], dict[str, dict[str, list[dict[str, Any]]]]]:
    episodes = rows_by_artifact.get("episodes", [])
    family_to_split = _family_split_assignments(episodes)
    episode_family = {
        str(episode.get("episode_id") or ""): str(episode.get("task_family") or "unknown")
        for episode in episodes
        if isinstance(episode.get("episode_id"), str)
    }
    split_rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        split_name: {artifact_name: [] for artifact_name in DATASET_SPLIT_ARTIFACTS}
        for split_name in DATASET_SPLIT_NAMES
    }
    for artifact_name in DATASET_SPLIT_ARTIFACTS:
        for row in rows_by_artifact.get(artifact_name, []):
            split_name = _row_split(row, family_to_split, episode_family)
            split_rows[split_name][artifact_name].append(row)

    family_episode_ids: dict[str, list[str]] = {}
    family_scenario_ids: dict[str, set[str]] = {}
    for episode in episodes:
        family = str(episode.get("task_family") or "unknown")
        family_episode_ids.setdefault(family, []).append(str(episode.get("episode_id") or ""))
        family_scenario_ids.setdefault(family, set()).add(str(episode.get("scenario_id") or ""))
    assignments = [
        {
            "task_family": family,
            "split": family_to_split[family],
            "episode_count": len(family_episode_ids.get(family, [])),
            "episode_ids": sorted(family_episode_ids.get(family, [])),
            "scenario_ids": sorted(family_scenario_ids.get(family, set())),
        }
        for family in sorted(family_to_split)
    ]
    split_counts = {
        split_name: {
            "task_family_count": sum(1 for item in assignments if item["split"] == split_name),
            "episode_count": len(split_rows[split_name]["episodes"]),
            "artifacts": {artifact_name: len(split_rows[split_name][artifact_name]) for artifact_name in DATASET_SPLIT_ARTIFACTS},
        }
        for split_name in DATASET_SPLIT_NAMES
    }
    family_splits: dict[str, set[str]] = {}
    for split_name in DATASET_SPLIT_NAMES:
        for episode in split_rows[split_name]["episodes"]:
            family_splits.setdefault(str(episode.get("task_family") or "unknown"), set()).add(split_name)
    cross_split_families = sorted(family for family, splits in family_splits.items() if len(splits) > 1)
    split_scenario_ids = _split_scenario_ids(split_rows)
    train_scenario_ids = split_scenario_ids["train"]
    heldout_scenario_ids = sorted(set(split_scenario_ids["validation"]) | set(split_scenario_ids["test"]))
    cross_split_scenario_ids = _cross_split_scenario_ids(split_scenario_ids)
    heldout_scenario_exclusive = not (set(train_scenario_ids) & set(heldout_scenario_ids)) and not cross_split_scenario_ids
    train_task_families = sorted(item["task_family"] for item in assignments if item["split"] == "train")
    heldout_task_families = sorted(item["task_family"] for item in assignments if item["split"] != "train")
    manifest = {
        "schema_version": RL_DATASET_SPLITS_SCHEMA_VERSION,
        "strategy": "task_family_hash",
        "seed": DATASET_SPLIT_SEED,
        "ratios": dict(DATASET_SPLIT_RATIOS),
        "split_names": list(DATASET_SPLIT_NAMES),
        "artifact_names": list(DATASET_SPLIT_ARTIFACTS),
        "summary": {
            "task_family_count": len(family_to_split),
            "episode_count": len(episodes),
            "train_episode_count": split_counts["train"]["episode_count"],
            "validation_episode_count": split_counts["validation"]["episode_count"],
            "test_episode_count": split_counts["test"]["episode_count"],
            "family_exclusive": not cross_split_families,
            "train_scenario_count": len(train_scenario_ids),
            "heldout_scenario_count": len(heldout_scenario_ids),
            "heldout_scenario_exclusive": heldout_scenario_exclusive,
        },
        "split_counts": split_counts,
        "assignments": assignments,
        "leakage_checks": {
            "family_exclusive": not cross_split_families,
            "cross_split_task_families": cross_split_families,
            "heldout_scenario_exclusive": heldout_scenario_exclusive,
            "cross_split_scenario_ids": cross_split_scenario_ids,
            "train_task_families": train_task_families,
            "heldout_task_families": heldout_task_families,
            "train_scenario_ids": train_scenario_ids,
            "heldout_scenario_ids": heldout_scenario_ids,
        },
        "notes": [
            "Splits are assigned by task_family, not individual episode, to reduce train/eval leakage.",
            "Validation and test scenario_ids are explicit held-out identifiers and must not appear in splits/train trainer rows.",
            "Small datasets may have empty validation or test splits; use dataset_metrics.quality_flags before training.",
        ],
    }
    return manifest, split_rows


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


def _cross_split_scenario_ids(split_scenario_ids: dict[str, list[str]]) -> list[str]:
    scenario_splits: dict[str, set[str]] = {}
    for split_name, scenario_ids in split_scenario_ids.items():
        for scenario_id in scenario_ids:
            scenario_splits.setdefault(scenario_id, set()).add(split_name)
    return sorted(scenario_id for scenario_id, splits in scenario_splits.items() if len(splits) > 1)


def _family_split_assignments(episodes: list[dict[str, Any]]) -> dict[str, str]:
    families = sorted({str(episode.get("task_family") or "unknown") for episode in episodes})
    if not families:
        return {}
    ordered = sorted(families, key=lambda family: (_split_hash(family), family))
    counts = _split_family_counts(len(ordered))
    assignments: dict[str, str] = {}
    cursor = 0
    for split_name in DATASET_SPLIT_NAMES:
        for family in ordered[cursor : cursor + counts[split_name]]:
            assignments[family] = split_name
        cursor += counts[split_name]
    for family in ordered:
        assignments.setdefault(family, "train")
    return assignments


def _split_family_counts(family_count: int) -> dict[str, int]:
    if family_count <= 0:
        return {split_name: 0 for split_name in DATASET_SPLIT_NAMES}
    if family_count < len(DATASET_SPLIT_NAMES):
        return {"train": family_count, "validation": 0, "test": 0}
    validation_count = max(1, round(family_count * DATASET_SPLIT_RATIOS["validation"]))
    test_count = max(1, round(family_count * DATASET_SPLIT_RATIOS["test"]))
    if validation_count + test_count >= family_count:
        overflow = validation_count + test_count - family_count + 1
        test_count = max(0, test_count - overflow)
    train_count = family_count - validation_count - test_count
    return {"train": train_count, "validation": validation_count, "test": test_count}


def _split_hash(value: str) -> str:
    return hashlib.sha256(f"{DATASET_SPLIT_SEED}:{value}".encode("utf-8")).hexdigest()


def _row_split(row: dict[str, Any], family_to_split: dict[str, str], episode_family: dict[str, str]) -> str:
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


def _dataset_metrics(
    episodes: list[dict[str, Any]],
    rewards: list[dict[str, Any]],
    step_rewards: list[dict[str, Any]],
    preferences: list[dict[str, Any]],
    failure_modes: list[dict[str, Any]],
    sft: list[dict[str, Any]],
    dpo: list[dict[str, Any]],
    reward_model: list[dict[str, Any]],
    dataset_splits: dict[str, Any],
    redaction_status: dict[str, Any],
    reward_scale: str,
    metadata: dict[str, str],
) -> dict[str, Any]:
    scores = [_score_value(episode.get("outcome", {}).get("score")) for episode in episodes if isinstance(episode.get("outcome"), dict)]
    rewards_values = [_numeric_value(reward.get("reward")) for reward in rewards]
    passed = sum(1 for episode in episodes if isinstance(episode.get("outcome"), dict) and episode["outcome"].get("passed") is True)
    failed = len(episodes) - passed
    artifact_counts = {
        "episodes": len(episodes),
        "rewards": len(rewards),
        "step_rewards": len(step_rewards),
        "preferences": len(preferences),
        "failure_modes": len(failure_modes),
        "sft": len(sft),
        "dpo": len(dpo),
        "reward_model": len(reward_model),
    }
    metrics = {
        "schema_version": RL_DATASET_METRICS_SCHEMA_VERSION,
        "reward_scale": reward_scale,
        "artifact_counts": artifact_counts,
        "episode_count": len(episodes),
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / len(episodes), 4) if episodes else 0.0,
        "average_score": _average(scores),
        "min_score": min(scores) if scores else None,
        "max_score": max(scores) if scores else None,
        "average_reward": _average(rewards_values),
        "min_reward": min(rewards_values) if rewards_values else None,
        "max_reward": max(rewards_values) if rewards_values else None,
        "source_fingerprint_coverage": _source_fingerprint_coverage(episodes),
        "trainer_view_source_fingerprint_coverage": _trainer_view_source_fingerprint_coverage(sft, dpo, reward_model),
        "redaction_status": redaction_status,
        "label_provenance": build_label_provenance_summary(episodes, sft, dpo, reward_model),
        "task_completion": _task_completion_metrics(episodes),
        "trace_signal": _trace_signal_metrics(episodes),
        "dataset_splits": dataset_splits.get("summary", {}),
        "failed_rule_counts": _count_rows(rule_id for episode in episodes for rule_id in _outcome_string_list(episode, "failed_rules")),
        "critical_failure_counts": _count_rows(rule_id for episode in episodes for rule_id in _outcome_string_list(episode, "critical_failures")),
        "task_families": _dataset_family_metrics(episodes, step_rewards, failure_modes, sft, dpo, reward_model),
        "quality_flags": _quality_flags(episodes, step_rewards, preferences, sft, dpo, reward_model, dataset_splits, redaction_status),
        "recommended_checks": [
            "Review scenario policies before treating labels as training rewards.",
            "Gate trace_signal before training so low-observability episodes do not become weak reward data.",
            "Inspect failure-mode and step-reward coverage before credit-assignment experiments.",
            "Use validation output and the HTML reports alongside trainer-ready JSONL views.",
        ],
    }
    if metadata:
        metrics["metadata"] = metadata
    return metrics


def _dataset_family_metrics(
    episodes: list[dict[str, Any]],
    step_rewards: list[dict[str, Any]],
    failure_modes: list[dict[str, Any]],
    sft: list[dict[str, Any]],
    dpo: list[dict[str, Any]],
    reward_model: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    families = sorted(
        {
            str(item.get("task_family") or "unknown")
            for source in (episodes, step_rewards, failure_modes, sft, dpo, reward_model)
            for item in source
            if isinstance(item, dict)
        }
    )
    rows: list[dict[str, Any]] = []
    for family in families:
        family_episodes = [episode for episode in episodes if str(episode.get("task_family") or "unknown") == family]
        scores = [
            _score_value(episode.get("outcome", {}).get("score"))
            for episode in family_episodes
            if isinstance(episode.get("outcome"), dict)
        ]
        passed = sum(1 for episode in family_episodes if isinstance(episode.get("outcome"), dict) and episode["outcome"].get("passed") is True)
        failed = len(family_episodes) - passed
        task_metrics = _task_completion_metrics(family_episodes)
        trace_metrics = _trace_signal_metrics(family_episodes)
        rows.append(
            {
                "task_family": family,
                "episode_count": len(family_episodes),
                "passed": passed,
                "failed": failed,
                "pass_rate": round(passed / len(family_episodes), 4) if family_episodes else 0.0,
                "task_completion_configured": task_metrics["configured_count"],
                "task_completion_complete": task_metrics["complete_count"],
                "task_completion_incomplete": task_metrics["incomplete_count"],
                "trace_average_event_count": trace_metrics["average_event_count"],
                "trace_event_type_count": trace_metrics["event_type_count"],
                "trace_tool_or_api_episode_rate": trace_metrics["tool_or_api_episode_rate"],
                "trace_empty_final_answer_count": trace_metrics["empty_final_answer_count"],
                "trace_risk_count": trace_metrics["risk_count"],
                "average_score": _average(scores),
                "step_reward_count": _count_family(step_rewards, family),
                "failure_mode_count": _count_family(failure_modes, family),
                "sft_count": _count_family(sft, family),
                "dpo_count": _count_family(dpo, family),
                "reward_model_count": _count_family(reward_model, family),
            }
        )
    return rows


def _task_completion_metrics(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = {"complete": 0, "incomplete": 0, "not_applicable": 0, "unknown": 0}
    configured = 0
    total_required_checks = 0
    total_passed_checks = 0
    for episode in episodes:
        task = episode.get("task_completion") if isinstance(episode.get("task_completion"), dict) else {}
        status = str(task.get("status") or "unknown")
        if status not in statuses:
            status = "unknown"
        statuses[status] += 1
        if task.get("task_evidence_configured") is True:
            configured += 1
        if isinstance(task.get("required_check_count"), int) and not isinstance(task.get("required_check_count"), bool):
            total_required_checks += int(task["required_check_count"])
        if isinstance(task.get("passed_check_count"), int) and not isinstance(task.get("passed_check_count"), bool):
            total_passed_checks += int(task["passed_check_count"])
    return {
        "episode_count": len(episodes),
        "configured_count": configured,
        "complete_count": statuses["complete"],
        "incomplete_count": statuses["incomplete"],
        "not_applicable_count": statuses["not_applicable"],
        "unknown_count": statuses["unknown"],
        "required_check_count": total_required_checks,
        "passed_check_count": total_passed_checks,
        "check_pass_rate": round(total_passed_checks / total_required_checks, 4) if total_required_checks else 0.0,
    }


def _trace_signal_metrics(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    signals = [_episode_trace_signal(episode) for episode in episodes]
    event_counts = [_int_value(signal.get("event_count")) for signal in signals]
    event_type_counts: dict[str, int] = {}
    risk_counts: dict[str, int] = {}
    for signal in signals:
        _merge_count_rows(event_type_counts, signal.get("event_types"))
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
        "event_type_counts": _count_rows_from_counts(event_type_counts),
        "episodes_with_final_answer": with_final,
        "empty_final_answer_count": episode_count - with_final,
        "final_answer_rate": _rate(with_final, episode_count),
        "episodes_with_tool_or_api_events": with_tool_or_api,
        "tool_or_api_episode_rate": _rate(with_tool_or_api, episode_count),
        "tool_call_count": sum(_int_value(signal.get("tool_call_count")) for signal in signals),
        "tool_result_count": sum(_int_value(signal.get("tool_result_count")) for signal in signals),
        "api_call_count": sum(_int_value(signal.get("api_call_count")) for signal in signals),
        "subagent_event_count": sum(_int_value(signal.get("subagent_event_count")) for signal in signals),
        "approval_event_count": sum(_int_value(signal.get("approval_event_count")) for signal in signals),
        "risk_count": sum(risk_counts.values()),
        "risk_counts": _count_rows_from_counts(risk_counts),
    }


def _episode_trace_signal(episode: dict[str, Any]) -> dict[str, Any]:
    signal = episode.get("trace_signal")
    if isinstance(signal, dict):
        return signal
    events = episode.get("events") if isinstance(episode.get("events"), list) else []
    final_answer = episode.get("final_answer") if isinstance(episode.get("final_answer"), str) else ""
    return build_trace_signal(events, final_answer)


def _quality_flags(
    episodes: list[dict[str, Any]],
    step_rewards: list[dict[str, Any]],
    preferences: list[dict[str, Any]],
    sft: list[dict[str, Any]],
    dpo: list[dict[str, Any]],
    reward_model: list[dict[str, Any]],
    dataset_splits: dict[str, Any],
    redaction_status: dict[str, Any],
) -> list[dict[str, str]]:
    flags: list[dict[str, str]] = []
    passed = sum(1 for episode in episodes if isinstance(episode.get("outcome"), dict) and episode["outcome"].get("passed") is True)
    failed = len(episodes) - passed
    eligible_positive = sum(1 for episode in episodes if _positive_label_eligible(episode))
    final_answer_only_successes = [
        str(episode.get("episode_id") or "")
        for episode in episodes
        if isinstance(episode.get("outcome"), dict)
        and episode["outcome"].get("passed") is True
        and not _positive_label_eligible(episode)
    ]
    expected_reward_model = sum(
        1
        for episode in episodes
        if not (isinstance(episode.get("outcome"), dict) and episode["outcome"].get("passed") is True)
        or _positive_label_eligible(episode)
    )
    families = {str(episode.get("task_family") or "unknown") for episode in episodes}
    if not episodes:
        flags.append(_quality_flag("empty_export", "error", "No episodes were exported."))
    if passed == 0:
        flags.append(_quality_flag("no_passing_episodes", "warning", "No passing episodes are available for positive references or SFT."))
    if final_answer_only_successes:
        flags.append(
            _quality_flag(
                "final_answer_only_success_excluded",
                "warning",
                "Passing episodes without configured task-completion evidence were excluded from SFT, DPO chosen, and positive reward-model rows.",
            )
        )
    if failed == 0:
        flags.append(_quality_flag("no_failing_episodes", "warning", "No failing episodes are available for negative reward analysis."))
    if not preferences:
        flags.append(_quality_flag("no_preferences", "warning", "No preference pairs were generated within task families."))
    if preferences and not dpo:
        flags.append(_quality_flag("missing_dpo_view", "error", "Preference pairs exist, but the DPO view is empty."))
    if eligible_positive and not sft:
        flags.append(_quality_flag("missing_sft_view", "error", "Passing episodes exist, but the SFT view is empty."))
    if episodes and len(reward_model) != expected_reward_model:
        flags.append(_quality_flag("reward_model_coverage", "error", "Reward-model rows do not cover every eligible episode."))
    if redaction_status.get("passed") is not True:
        flags.append(_quality_flag("redaction_failed", "error", "Unredacted secret-like values were found in trainer-facing artifacts."))
    if not step_rewards and failed:
        flags.append(_quality_flag("no_step_rewards", "warning", "Failing episodes exist, but no step-level attribution rows were exported."))
    coverage = _source_fingerprint_coverage(episodes)
    if episodes and coverage["fully_verified"] < len(episodes):
        flags.append(
            _quality_flag(
                "unverified_source_fingerprints",
                "warning",
                "Some episodes are missing scenario or source-trace SHA-256 fingerprints.",
            )
        )
    trainer_coverage = _trainer_view_source_fingerprint_coverage(sft, dpo, reward_model)
    if trainer_coverage["rows"] and trainer_coverage["unverified"]:
        flags.append(
            _quality_flag(
                "unverified_trainer_view_source_fingerprints",
                "warning",
                "Some trainer-ready SFT, DPO, or reward-model rows are missing complete source fingerprints.",
            )
        )
    if len(families) == 1:
        flags.append(_quality_flag("single_task_family", "info", "Only one task family is represented; broaden coverage before generalizing."))
    split_summary = dataset_splits.get("summary") if isinstance(dataset_splits.get("summary"), dict) else {}
    if episodes and split_summary.get("family_exclusive") is False:
        flags.append(_quality_flag("split_family_leakage", "error", "Dataset split assignments place a task family in multiple splits."))
    if episodes and _int_value(split_summary.get("validation_episode_count")) == 0:
        flags.append(_quality_flag("empty_validation_split", "warning", "Validation split is empty; add more task families before relying on validation metrics."))
    if episodes and _int_value(split_summary.get("test_episode_count")) == 0:
        flags.append(_quality_flag("empty_test_split", "warning", "Test split is empty; add more task families before relying on held-out test metrics."))
    return flags


def _quality_flag(flag_id: str, severity: str, message: str) -> dict[str, str]:
    return {"id": flag_id, "severity": severity, "message": message}


def _dataset_card(manifest: dict[str, Any], metrics: dict[str, Any]) -> str:
    rows = [
        "# Flight Recorder Dataset Card",
        "",
        "This card summarizes a Flight Recorder training export. It is generated from the same canonical artifacts as the JSONL views.",
        "",
        "## Summary",
        "",
        f"- Dataset version: `{manifest.get('dataset_version')}`",
        f"- Source runs: `{manifest.get('source_runs_dir')}`",
        f"- Registry: `{manifest.get('outputs', {}).get('dataset_registry')}`",
        f"- Reward scale: `{metrics.get('reward_scale')}`",
        f"- Episodes: {metrics.get('episode_count')} ({metrics.get('passed')} passed, {metrics.get('failed')} failed)",
        f"- Pass rate: {metrics.get('pass_rate')}",
        f"- Average score: {metrics.get('average_score')}",
        f"- Average reward: {metrics.get('average_reward')}",
        "",
    ]
    metadata = metrics.get("metadata") if isinstance(metrics.get("metadata"), dict) else {}
    if metadata:
        rows.extend(["## Experiment Metadata", "", "| Key | Value |", "| --- | --- |"])
        for key, value in sorted(metadata.items()):
            rows.append(f"| `{_md_cell(str(key))}` | {_md_cell(str(value))} |")
        rows.append("")
    coverage = metrics.get("source_fingerprint_coverage") if isinstance(metrics.get("source_fingerprint_coverage"), dict) else {}
    trainer_coverage = (
        metrics.get("trainer_view_source_fingerprint_coverage")
        if isinstance(metrics.get("trainer_view_source_fingerprint_coverage"), dict)
        else {}
    )
    rows.extend(
        [
            "## Source Fingerprints",
            "",
            f"- Fully verified episodes: {coverage.get('fully_verified', 0)} / {coverage.get('episodes', metrics.get('episode_count'))}",
            f"- Scenario fingerprints: {coverage.get('with_scenario_sha256', 0)}",
            f"- Source-trace fingerprints: {coverage.get('with_source_trace_sha256', 0)}",
            f"- Fully verified trainer-view rows: {trainer_coverage.get('fully_verified', 0)} / {trainer_coverage.get('rows', 0)}",
            f"- Unverified trainer-view rows: {trainer_coverage.get('unverified', 0)}",
            "",
        ]
    )
    redaction_status = metrics.get("redaction_status") if isinstance(metrics.get("redaction_status"), dict) else {}
    rows.extend(
        [
            "## Redaction",
            "",
            f"- Passed: {redaction_status.get('passed', False)}",
            f"- Unredacted secret findings: {redaction_status.get('unredacted_secret_count', 0)}",
            f"- Scanner: `{redaction_status.get('scanner', 'unknown')}`",
            "",
        ]
    )
    label_provenance = metrics.get("label_provenance") if isinstance(metrics.get("label_provenance"), dict) else {}
    rows.extend(
        [
            "## Label Provenance",
            "",
            f"- Positive episodes: {label_provenance.get('positive_episode_count', 0)}",
            f"- Eligible positive episodes: {label_provenance.get('eligible_positive_episode_count', 0)}",
            f"- Final-answer-only success exclusions: {label_provenance.get('final_answer_only_success_count', 0)}",
            f"- SFT labels: {label_provenance.get('sft_label_count', 0)}",
            f"- DPO pairs: {label_provenance.get('dpo_pair_count', 0)}",
            f"- Reward-model labels: {label_provenance.get('reward_model_label_count', 0)}",
            "",
        ]
    )
    trace_signal = metrics.get("trace_signal") if isinstance(metrics.get("trace_signal"), dict) else {}
    rows.extend(
        [
            "## Trace Signal",
            "",
            f"- Average events per episode: {trace_signal.get('average_event_count', 0.0)}",
            f"- Event types: {trace_signal.get('event_type_count', 0)}",
            f"- Final-answer rate: {trace_signal.get('final_answer_rate', 0.0)}",
            f"- Tool/API episode rate: {trace_signal.get('tool_or_api_episode_rate', 0.0)}",
            f"- Trace risk count: {trace_signal.get('risk_count', 0)}",
            "",
        ]
    )
    dataset_splits = metrics.get("dataset_splits") if isinstance(metrics.get("dataset_splits"), dict) else {}
    rows.extend(
        [
            "## Dataset Splits",
            "",
            f"- Task families: {dataset_splits.get('task_family_count', 0)}",
            f"- Family exclusive: {dataset_splits.get('family_exclusive', False)}",
            f"- Train episodes: {dataset_splits.get('train_episode_count', 0)}",
            f"- Validation episodes: {dataset_splits.get('validation_episode_count', 0)}",
            f"- Test episodes: {dataset_splits.get('test_episode_count', 0)}",
            "",
        ]
    )
    rows.extend(["## Artifact Counts", "", "| Artifact | Count |", "| --- | ---: |"])
    for name, count in sorted((metrics.get("artifact_counts") or {}).items()):
        rows.append(f"| `{_md_cell(name)}` | {count} |")
    rows.extend(
        [
            "",
            "## Task Families",
            "",
            "| Family | Episodes | Passed | Failed | Avg Score | Step Rewards | Failures | SFT | DPO | Reward Model |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for family in metrics.get("task_families", []):
        rows.append(
            "| "
            + " | ".join(
                [
                    _md_cell(str(family.get("task_family") or "unknown")),
                    str(family.get("episode_count")),
                    str(family.get("passed")),
                    str(family.get("failed")),
                    str(family.get("average_score")),
                    str(family.get("step_reward_count")),
                    str(family.get("failure_mode_count")),
                    str(family.get("sft_count")),
                    str(family.get("dpo_count")),
                    str(family.get("reward_model_count")),
                ]
            )
            + " |"
        )
    rows.extend(["", "## Failure Pressure", "", "| Rule | Count |", "| --- | ---: |"])
    failed_rule_counts = metrics.get("failed_rule_counts") or []
    if failed_rule_counts:
        for item in failed_rule_counts:
            rows.append(f"| `{_md_cell(str(item.get('id') or 'unknown'))}` | {item.get('count')} |")
    else:
        rows.append("| None | 0 |")
    rows.extend(["", "## Quality Flags", ""])
    quality_flags = metrics.get("quality_flags") or []
    if quality_flags:
        for flag in quality_flags:
            rows.append(f"- **{_md_cell(str(flag.get('severity') or 'unknown'))}** `{_md_cell(str(flag.get('id') or 'flag'))}`: {_md_cell(str(flag.get('message') or ''))}")
    else:
        rows.append("- No dataset-level quality flags were emitted.")
    rows.extend(
        [
            "",
            "## Boundaries",
            "",
            "- These artifacts are deterministic eval evidence and trainer-ready views, not a trainer.",
            "- Reward labels are only as strong as the scenario policies and observable trace evidence.",
            "- Review HTML reports and scorecards before using exported rows for model updates.",
            "",
        ]
    )
    return "\n".join(rows)


def _improvement_card(manifest: dict[str, Any], pairs: list[dict[str, Any]]) -> str:
    rows = [
        "# Flight Recorder Improvement Pair Card",
        "",
        "This card summarizes baseline-vs-candidate preference artifacts generated from paired Flight Recorder runs.",
        "",
        "## Summary",
        "",
        f"- Baseline runs: `{manifest.get('baseline_runs_dir')}`",
        f"- Candidate runs: `{manifest.get('candidate_runs_dir')}`",
        f"- Pair count: {manifest.get('pair_count')}",
        f"- Candidate wins: {manifest.get('candidate_win_count')}",
        f"- Baseline wins: {manifest.get('baseline_win_count')}",
        f"- Contract drift: {manifest.get('contract_drift_count', 0)}",
        f"- Unverified contracts: {manifest.get('unverified_contract_count', 0)}",
        f"- Skipped pairs: {manifest.get('skipped_pair_count')}",
        "",
    ]
    metadata = manifest.get("metadata") if isinstance(manifest.get("metadata"), dict) else {}
    if metadata:
        rows.extend(["## Experiment Metadata", "", "| Key | Value |", "| --- | --- |"])
        for key, value in sorted(metadata.items()):
            rows.append(f"| `{_md_cell(str(key))}` | {_md_cell(str(value))} |")
        rows.append("")
    rows.extend(
        [
            "## Pairs",
            "",
            "| Scenario | Candidate Outcome | Contract | Delta | Chosen | Rejected | Reason |",
            "| --- | --- | --- | ---: | --- | --- | --- |",
        ]
    )
    if pairs:
        for pair in pairs:
            rows.append(
                "| "
                + " | ".join(
                    [
                        f"`{_md_cell(str(pair.get('scenario_id') or 'unknown'))}`",
                        _md_cell(str(pair.get("candidate_outcome") or "unknown")),
                        _md_cell(str(pair.get("contract_fingerprint_status") or "unverified")),
                        str(pair.get("candidate_score_delta")),
                        _md_cell(str(pair.get("chosen_side") or "")),
                        _md_cell(str(pair.get("rejected_side") or "")),
                        _md_cell(str(pair.get("reason") or "")),
                    ]
                )
                + " |"
            )
    else:
        rows.append("| None | n/a | n/a | 0 | n/a | n/a | No pairs exceeded the score-gap threshold. |")
    rows.extend(
        [
            "",
            "## Boundaries",
            "",
            "- These rows preserve deterministic comparison evidence; they are not proof of causal model improvement.",
            "- Use candidate wins as improvement examples and baseline wins as regression-avoidance examples.",
            "- Review the source reports before using preference rows for model updates.",
            "",
        ]
    )
    return "\n".join(rows)


def _source_fingerprint_coverage(episodes: list[dict[str, Any]]) -> dict[str, int]:
    with_scenario = 0
    with_trace = 0
    fully_verified = 0
    for episode in episodes:
        fingerprints = episode.get("source_fingerprints") if isinstance(episode.get("source_fingerprints"), dict) else {}
        scenario_hash = _fingerprint_sha(fingerprints.get("scenario"))
        trace_hash = _fingerprint_sha(fingerprints.get("source_trace"))
        if scenario_hash:
            with_scenario += 1
        if trace_hash:
            with_trace += 1
        if scenario_hash and trace_hash:
            fully_verified += 1
    return {
        "episodes": len(episodes),
        "with_scenario_sha256": with_scenario,
        "with_source_trace_sha256": with_trace,
        "fully_verified": fully_verified,
        "unverified": len(episodes) - fully_verified,
    }


def _trainer_view_source_fingerprint_coverage(
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
        "fully_verified_rate": _rate(fully_verified, row_count),
    }


def _row_source_fingerprints_verified(row: dict[str, Any]) -> bool:
    fingerprints = row.get("source_fingerprints") if isinstance(row.get("source_fingerprints"), dict) else {}
    return (
        row.get("source_fingerprint_status") == "verified"
        and bool(_fingerprint_sha(fingerprints.get("scenario")))
        and bool(_fingerprint_sha(fingerprints.get("source_trace")))
    )


def _dpo_source_fingerprints_verified(row: dict[str, Any]) -> bool:
    return _paired_source_fingerprints_verified(row, "chosen") and _paired_source_fingerprints_verified(row, "rejected")


def _paired_source_fingerprints_verified(row: dict[str, Any], side: str) -> bool:
    fingerprints_key = f"{side}_source_fingerprints"
    status_key = f"{side}_source_fingerprint_status"
    fingerprints = row.get(fingerprints_key) if isinstance(row.get(fingerprints_key), dict) else {}
    return (
        row.get(status_key) == "verified"
        and bool(_fingerprint_sha(fingerprints.get("scenario")))
        and bool(_fingerprint_sha(fingerprints.get("source_trace")))
    )


def _source_fingerprints(record: RunRecord) -> dict[str, dict[str, Any]]:
    fingerprints = {
        "scenario": {"path": None, "sha256": None, "exists": None},
        "source_trace": {"path": None, "sha256": None, "exists": None},
        "source_before_state_snapshot": {"path": None, "sha256": None, "exists": None},
        "source_state_snapshot": {"path": None, "sha256": None, "exists": None},
    }
    lineage = record.lineage if isinstance(record.lineage, dict) else {}
    inputs = lineage.get("inputs") if isinstance(lineage.get("inputs"), list) else []
    for item in inputs:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if name not in fingerprints:
            continue
        fingerprints[str(name)] = {
            "path": item.get("path") if isinstance(item.get("path"), str) else None,
            "sha256": item.get("sha256") if isinstance(item.get("sha256"), str) else None,
            "exists": item.get("exists") if isinstance(item.get("exists"), bool) else None,
        }
    return fingerprints


def _source_fingerprint_status(fingerprints: dict[str, Any]) -> str:
    return "verified" if _fingerprint_sha(fingerprints.get("scenario")) and _fingerprint_sha(fingerprints.get("source_trace")) else "unverified"


def _fingerprint_sha(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    sha = value.get("sha256")
    return sha if isinstance(sha, str) and sha else None


def _episode_contract_comparison(baseline: dict[str, Any], candidate: dict[str, Any], contract_scope: str) -> dict[str, Any]:
    baseline_inputs = baseline.get("source_fingerprints") if isinstance(baseline.get("source_fingerprints"), dict) else {}
    candidate_inputs = candidate.get("source_fingerprints") if isinstance(candidate.get("source_fingerprints"), dict) else {}
    reasons: list[str] = []
    unknowns: list[str] = []
    for name in _contract_input_names(contract_scope):
        baseline_hash = _fingerprint_sha(baseline_inputs.get(name))
        candidate_hash = _fingerprint_sha(candidate_inputs.get(name))
        if baseline_hash and candidate_hash:
            if baseline_hash != candidate_hash:
                reasons.append(f"{name}_sha256_changed")
        else:
            unknowns.append(f"{name}_sha256_unverified")
    status = "drifted" if reasons else "unverified" if unknowns else "matched"
    return {
        "status": status,
        "scope": contract_scope,
        "reasons": reasons or unknowns,
        "fingerprints": {
            "baseline": _contract_fingerprint_inputs(baseline_inputs),
            "candidate": _contract_fingerprint_inputs(candidate_inputs),
        },
    }


def _contract_input_names(contract_scope: str) -> tuple[str, ...]:
    return ("scenario", "source_trace") if contract_scope == "scenario-and-trace" else ("scenario",)


def _contract_fingerprint_inputs(inputs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "path": inputs.get(name, {}).get("path") if isinstance(inputs.get(name), dict) else None,
            "sha256": _fingerprint_sha(inputs.get(name)),
        }
        for name in ("scenario", "source_trace")
    }


def _average(values: list[int] | list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _count_family(rows: list[dict[str, Any]], family: str) -> int:
    return sum(1 for row in rows if str(row.get("task_family") or "unknown") == family)


def _count_rows(values: Any) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for value in values:
        if not isinstance(value, str) or not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return [{"id": key, "count": counts[key]} for key in sorted(counts)]


def _count_rows_from_counts(counts: dict[str, int]) -> list[dict[str, Any]]:
    return [{"id": key, "count": count} for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


def _merge_count_rows(counts: dict[str, int], rows: Any) -> None:
    if not isinstance(rows, list):
        return
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("id"), str) and isinstance(row.get("count"), int):
            counts[row["id"]] = counts.get(row["id"], 0) + row["count"]


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _outcome_string_list(episode: dict[str, Any], field_name: str) -> list[str]:
    outcome = episode.get("outcome") if isinstance(episode.get("outcome"), dict) else {}
    values = outcome.get(field_name)
    return values if isinstance(values, list) and all(isinstance(item, str) for item in values) else []


def _md_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _metadata(value: dict[str, str] | None) -> dict[str, str]:
    if not value:
        return {}
    return {str(key): str(raw_value) for key, raw_value in sorted(value.items())}


def _messages(prompt: str, response: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


def _event_record(index: int, event: Any) -> dict[str, Any]:
    if not isinstance(event, dict):
        return {"index": index, "type": "unknown", "text": str(event)}
    rendered = dict(event)
    rendered["index"] = index
    return rendered


def _rule_reward(rule: dict[str, Any]) -> dict[str, Any]:
    penalty = int(rule.get("penalty", 0) or 0)
    passed = bool(rule.get("passed"))
    return {
        "rule_id": rule.get("id"),
        "name": rule.get("name"),
        "passed": passed,
        "critical": bool(rule.get("critical")),
        "penalty": penalty,
        "reward_delta": 0.0 if passed else round(-penalty / 100.0, 6),
        "evidence": rule.get("evidence", []),
        "evidence_refs": _evidence_refs(rule),
    }


def _reward_attribution(scorecard: dict[str, Any]) -> list[dict[str, Any]]:
    attribution: list[dict[str, Any]] = []
    for rule in scorecard.get("rules", []):
        if not isinstance(rule, dict) or rule.get("passed"):
            continue
        penalty = int(rule.get("penalty", 0) or 0)
        reward_delta = round(-penalty / 100.0, 6)
        structured_refs = [ref for ref in _evidence_refs(rule) if ref.get("passed") is not True]
        if structured_refs:
            for ref in structured_refs:
                attribution.append(_attribution_from_ref(rule, ref, reward_delta))
            continue
        evidence_items = rule.get("evidence", []) or ["rule failed"]
        for evidence in evidence_items:
            text = str(evidence)
            event_indexes = [int(match.group(1)) for match in EVENT_INDEX_RE.finditer(text)]
            if event_indexes:
                for event_index in event_indexes:
                    attribution.append(
                        {
                            "target": "event",
                            "event_index": event_index,
                            "rule_id": rule.get("id"),
                            "reward_delta": reward_delta,
                            "evidence": text,
                        }
                    )
            elif "final answer" in text.lower():
                attribution.append(
                    {
                        "target": "final_answer",
                        "rule_id": rule.get("id"),
                        "reward_delta": reward_delta,
                        "evidence": text,
                    }
                )
            else:
                attribution.append(
                    {
                        "target": "episode",
                        "rule_id": rule.get("id"),
                        "reward_delta": reward_delta,
                        "evidence": text,
                    }
                )
    return attribution


def _attribution_from_ref(rule: dict[str, Any], ref: dict[str, Any], reward_delta: float) -> dict[str, Any]:
    target = ref.get("target") if ref.get("target") in {"event", "final_answer", "episode", "state_snapshot"} else "episode"
    attribution = {
        "target": target,
        "rule_id": rule.get("id"),
        "reward_delta": reward_delta,
        "evidence": str(ref.get("reason") or "structured evidence ref"),
        "evidence_ref": ref,
    }
    if target == "event" and isinstance(ref.get("event_index"), int) and not isinstance(ref.get("event_index"), bool):
        attribution["event_index"] = ref["event_index"]
    return attribution


def _evidence_refs(rule: dict[str, Any]) -> list[dict[str, Any]]:
    refs = rule.get("evidence_refs")
    if not isinstance(refs, list):
        return []
    return [ref for ref in refs if isinstance(ref, dict)]


def _reward_value(scorecard: dict[str, Any], reward_scale: str) -> float:
    score = _score(scorecard)
    if reward_scale == "binary":
        return 1.0 if scorecard.get("passed") else 0.0
    if reward_scale == "signed":
        return round((score / 50.0) - 1.0, 6)
    return round(score / 100.0, 6)


def _failed_rule_ids(scorecard: dict[str, Any]) -> list[str]:
    return [
        str(rule.get("id"))
        for rule in _failed_rules(scorecard)
        if rule.get("id")
    ]


def _failed_rules(scorecard: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        rule
        for rule in scorecard.get("rules", [])
        if isinstance(rule, dict) and not rule.get("passed")
    ]


def _prompt_from_trace(trace: dict[str, Any]) -> str:
    for event in trace.get("events", []):
        if isinstance(event, dict) and event.get("type") == "user_message":
            return str(event.get("text") or "")
    return ""


def _task_family(scenario_id: str) -> str:
    family = FAMILY_SUFFIX_RE.sub("", scenario_id).strip("_-")
    return family or scenario_id


def _scorecard_task_family(scorecard: dict[str, Any], scenario_id: str) -> str:
    family = str(scorecard.get("task_family") or "").strip()
    return family or _task_family(scenario_id)


def _score(scorecard: dict[str, Any]) -> int:
    raw = scorecard.get("score", 0)
    try:
        return max(0, min(100, int(raw)))
    except (TypeError, ValueError):
        return 0


def _score_value(value: Any) -> int:
    try:
        return max(0, min(100, int(value)))
    except (TypeError, ValueError):
        return 0


def _numeric_value(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _display_path(path: Path, preserve_paths: bool) -> str:
    if preserve_paths:
        return str(path)
    raw = str(path)
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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_fingerprints(paths: dict[str, Path], preserve_paths: bool, *, exclude: set[str]) -> dict[str, Any]:
    fingerprints: dict[str, Any] = {}
    for name, path in sorted(paths.items()):
        if name in exclude:
            continue
        record: dict[str, Any] = {
            "path": _display_path(path, preserve_paths),
            "exists": path.exists(),
        }
        if path.exists() and path.is_file():
            stat = path.stat()
            record["size_bytes"] = stat.st_size
            record["sha256"] = _sha256_file(path)
        fingerprints[name] = record
    return fingerprints


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload.rstrip() + "\n", encoding="utf-8")
