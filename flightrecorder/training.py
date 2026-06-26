"""Training-data exports for future agent-improvement loops."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

REWARD_SCALES = {"score", "binary", "signed"}
EVENT_INDEX_RE = re.compile(r"event #(\d+)")
FAMILY_SUFFIX_RE = re.compile(r"([_-](good|bad|pass|fail|passing|failing|chosen|rejected))+$", re.IGNORECASE)


class TrainingExportError(ValueError):
    """Raised when RL training artifacts cannot be exported."""


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    run_dir: Path
    trace: dict[str, Any]
    scorecard: dict[str, Any]


def export_rl_dataset(
    runs_dir: str | Path,
    out_dir: str | Path,
    *,
    reward_scale: str = "score",
    min_score_gap: int = 1,
    max_pairs_per_family: int = 0,
    preserve_paths: bool = False,
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
    dataset_metrics = _dataset_metrics(
        episodes,
        rewards,
        step_rewards,
        preferences,
        failure_modes,
        sft,
        dpo,
        reward_model,
        reward_scale,
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
        "dataset_card": target / "DATASET_CARD.md",
        "manifest": target / "manifest.json",
    }
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

    manifest = {
        "schema_version": RL_MANIFEST_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
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
        "task_families": sorted({str(episode["task_family"]) for episode in episodes}),
        "outputs": {name: _display_path(path, preserve_paths) for name, path in paths.items()},
        "notes": [
            "Exports are built from normalized_trace.json and scorecard.json.",
            "Use these artifacts as reward/eval data, not as a complete trainer.",
            "failure_modes.jsonl exposes one failed-rule record per episode for curriculum construction.",
            "New scorecards include evidence_refs for structured event/final-answer/episode attribution.",
            "step_rewards.jsonl flattens failed-rule attribution into one row per event/final-answer/episode target.",
            "sft.jsonl, dpo.jsonl, and reward_model.jsonl are trainer-ready views over the canonical evidence files.",
            "dataset_metrics.json and DATASET_CARD.md summarize export quality and coverage.",
            "Reward labels are deterministic scenario-policy judgments and can be reward-hacked if scenarios are weak.",
        ],
    }
    _write_text(paths["dataset_card"], _dataset_card(manifest, dataset_metrics))
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
        if not trace_path.exists() or not score_path.exists():
            continue
        trace = _read_json(trace_path)
        scorecard = _read_json(score_path)
        if not isinstance(trace, dict) or not isinstance(scorecard, dict):
            raise TrainingExportError(f"Run {run_dir} must contain JSON objects")
        records.append(RunRecord(run_id=run_dir.name, run_dir=run_dir, trace=trace, scorecard=scorecard))

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
    failed_rules = _failed_rule_ids(scorecard)
    return {
        "schema_version": RL_EPISODE_SCHEMA_VERSION,
        "episode_id": record.run_id,
        "source_run": _display_path(record.run_dir, preserve_paths),
        "scenario_id": scenario_id,
        "scenario_title": scenario_title,
        "task_family": _task_family(scenario_id),
        "prompt": _prompt_from_trace(trace),
        "source_format": trace.get("session", {}).get("source_format", "unknown"),
        "model": trace.get("session", {}).get("model", "unknown"),
        "events": events,
        "final_answer": str(trace.get("final_answer") or ""),
        "outcome": {
            "passed": passed,
            "score": score,
            "pass_threshold": scorecard.get("pass_threshold"),
            "reward": _reward_value(scorecard, reward_scale),
            "critical_failures": scorecard.get("critical_failures", []),
            "failed_rules": failed_rules,
            "summary": scorecard.get("summary", ""),
        },
    }


def _reward_record(record: RunRecord, reward_scale: str) -> dict[str, Any]:
    scorecard = record.scorecard
    scenario_id = str(scorecard.get("scenario_id") or record.run_id)
    rule_rewards = [_rule_reward(rule) for rule in scorecard.get("rules", []) if isinstance(rule, dict)]
    return {
        "schema_version": RL_REWARD_SCHEMA_VERSION,
        "episode_id": record.run_id,
        "scenario_id": scenario_id,
        "task_family": _task_family(scenario_id),
        "reward_scale": reward_scale,
        "reward": _reward_value(scorecard, reward_scale),
        "score": _score(scorecard),
        "passed": bool(scorecard.get("passed")),
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
        task_family = _task_family(scenario_id)
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
    }


def _preference_view(episode: dict[str, Any]) -> dict[str, Any]:
    return {
        "episode_id": episode["episode_id"],
        "scenario_id": episode["scenario_id"],
        "passed": episode["outcome"]["passed"],
        "score": episode["outcome"]["score"],
        "reward": episode["outcome"]["reward"],
        "failed_rules": episode["outcome"]["failed_rules"],
        "events": episode["events"],
        "final_answer": episode["final_answer"],
    }


def _failure_mode_record(record: RunRecord, rule: dict[str, Any], reward_scale: str) -> dict[str, Any]:
    scorecard = record.scorecard
    scenario_id = str(scorecard.get("scenario_id") or record.run_id)
    rule_id = str(rule.get("id") or "unknown_rule")
    attribution = [item for item in _reward_attribution(scorecard) if item.get("rule_id") == rule_id]
    return {
        "schema_version": RL_FAILURE_MODE_SCHEMA_VERSION,
        "failure_id": f"{record.run_id}:{rule_id}",
        "episode_id": record.run_id,
        "scenario_id": scenario_id,
        "scenario_title": str(scorecard.get("scenario_title") or scenario_id),
        "task_family": _task_family(scenario_id),
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
                "example_evidence": [],
            },
        )
        mode["count"] += 1
        if failure.get("critical"):
            mode["critical_count"] += 1
        episode_id = str(failure.get("episode_id") or "")
        if episode_id and episode_id not in mode["episode_ids"]:
            mode["episode_ids"].append(episode_id)
        for evidence in failure.get("evidence", []):
            text = str(evidence)
            if text and text not in mode["example_evidence"]:
                mode["example_evidence"].append(text)
            if len(mode["example_evidence"]) >= 3:
                break

    family_rows: list[dict[str, Any]] = []
    for family, bucket in sorted(families.items()):
        scores = bucket.pop("scores")
        failure_map = bucket.pop("failure_modes")
        bucket["average_score"] = round(sum(scores) / len(scores), 2) if scores else 0.0
        bucket["failure_modes"] = sorted(
            failure_map.values(),
            key=lambda item: (-int(item["count"]), str(item["rule_id"])),
        )
        family_rows.append(bucket)

    return {
        "schema_version": RL_CURRICULUM_SCHEMA_VERSION,
        "episode_count": len(episodes),
        "failure_mode_count": len(failure_modes),
        "task_families": family_rows,
        "recommended_use": [
            "Use high-count critical failure modes as regression priorities.",
            "Use passing episodes in the same task family as positive references.",
            "Treat this as curriculum metadata, not as an automatic trainer policy.",
        ],
    }


def _sft_records(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        outcome = episode.get("outcome") if isinstance(episode.get("outcome"), dict) else {}
        response = str(episode.get("final_answer") or "")
        if outcome.get("passed") is not True or not response:
            continue
        prompt = str(episode.get("prompt") or "")
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
                "source_artifact": "preferences.jsonl",
            }
        )
    return rows


def _reward_model_records(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        outcome = episode.get("outcome") if isinstance(episode.get("outcome"), dict) else {}
        prompt = str(episode.get("prompt") or "")
        response = str(episode.get("final_answer") or "")
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
                "failed_rules": _string_list(outcome.get("failed_rules")),
                "critical_failures": _string_list(outcome.get("critical_failures")),
                "source_artifact": "episodes.jsonl",
            }
        )
    return rows


def _dataset_metrics(
    episodes: list[dict[str, Any]],
    rewards: list[dict[str, Any]],
    step_rewards: list[dict[str, Any]],
    preferences: list[dict[str, Any]],
    failure_modes: list[dict[str, Any]],
    sft: list[dict[str, Any]],
    dpo: list[dict[str, Any]],
    reward_model: list[dict[str, Any]],
    reward_scale: str,
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
        "failed_rule_counts": _count_rows(rule_id for episode in episodes for rule_id in _outcome_string_list(episode, "failed_rules")),
        "critical_failure_counts": _count_rows(rule_id for episode in episodes for rule_id in _outcome_string_list(episode, "critical_failures")),
        "task_families": _dataset_family_metrics(episodes, step_rewards, failure_modes, sft, dpo, reward_model),
        "quality_flags": _quality_flags(episodes, step_rewards, preferences, sft, dpo, reward_model),
        "recommended_checks": [
            "Review scenario policies before treating labels as training rewards.",
            "Inspect failure-mode and step-reward coverage before credit-assignment experiments.",
            "Use validation output and the HTML reports alongside trainer-ready JSONL views.",
        ],
    }
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
        rows.append(
            {
                "task_family": family,
                "episode_count": len(family_episodes),
                "passed": passed,
                "failed": failed,
                "pass_rate": round(passed / len(family_episodes), 4) if family_episodes else 0.0,
                "average_score": _average(scores),
                "step_reward_count": _count_family(step_rewards, family),
                "failure_mode_count": _count_family(failure_modes, family),
                "sft_count": _count_family(sft, family),
                "dpo_count": _count_family(dpo, family),
                "reward_model_count": _count_family(reward_model, family),
            }
        )
    return rows


def _quality_flags(
    episodes: list[dict[str, Any]],
    step_rewards: list[dict[str, Any]],
    preferences: list[dict[str, Any]],
    sft: list[dict[str, Any]],
    dpo: list[dict[str, Any]],
    reward_model: list[dict[str, Any]],
) -> list[dict[str, str]]:
    flags: list[dict[str, str]] = []
    passed = sum(1 for episode in episodes if isinstance(episode.get("outcome"), dict) and episode["outcome"].get("passed") is True)
    failed = len(episodes) - passed
    families = {str(episode.get("task_family") or "unknown") for episode in episodes}
    if not episodes:
        flags.append(_quality_flag("empty_export", "error", "No episodes were exported."))
    if passed == 0:
        flags.append(_quality_flag("no_passing_episodes", "warning", "No passing episodes are available for positive references or SFT."))
    if failed == 0:
        flags.append(_quality_flag("no_failing_episodes", "warning", "No failing episodes are available for negative reward analysis."))
    if not preferences:
        flags.append(_quality_flag("no_preferences", "warning", "No preference pairs were generated within task families."))
    if preferences and not dpo:
        flags.append(_quality_flag("missing_dpo_view", "error", "Preference pairs exist, but the DPO view is empty."))
    if passed and not sft:
        flags.append(_quality_flag("missing_sft_view", "error", "Passing episodes exist, but the SFT view is empty."))
    if episodes and len(reward_model) != len(episodes):
        flags.append(_quality_flag("reward_model_coverage", "error", "Reward-model rows do not cover every episode."))
    if not step_rewards and failed:
        flags.append(_quality_flag("no_step_rewards", "warning", "Failing episodes exist, but no step-level attribution rows were exported."))
    if len(families) == 1:
        flags.append(_quality_flag("single_task_family", "info", "Only one task family is represented; broaden coverage before generalizing."))
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
        f"- Source runs: `{manifest.get('source_runs_dir')}`",
        f"- Reward scale: `{metrics.get('reward_scale')}`",
        f"- Episodes: {metrics.get('episode_count')} ({metrics.get('passed')} passed, {metrics.get('failed')} failed)",
        f"- Pass rate: {metrics.get('pass_rate')}",
        f"- Average score: {metrics.get('average_score')}",
        f"- Average reward: {metrics.get('average_reward')}",
        "",
        "## Artifact Counts",
        "",
        "| Artifact | Count |",
        "| --- | ---: |",
    ]
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


def _outcome_string_list(episode: dict[str, Any], field_name: str) -> list[str]:
    outcome = episode.get("outcome") if isinstance(episode.get("outcome"), dict) else {}
    values = outcome.get(field_name)
    return values if isinstance(values, list) and all(isinstance(item, str) for item in values) else []


def _md_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


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
    target = ref.get("target") if ref.get("target") in {"event", "final_answer", "episode"} else "episode"
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
