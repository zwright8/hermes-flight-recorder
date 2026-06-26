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

    paths = {
        "episodes": target / "episodes.jsonl",
        "rewards": target / "rewards.jsonl",
        "step_rewards": target / "step_rewards.jsonl",
        "preferences": target / "preferences.jsonl",
        "failure_modes": target / "failure_modes.jsonl",
        "curriculum": target / "curriculum.json",
        "manifest": target / "manifest.json",
    }
    _write_jsonl(paths["episodes"], episodes)
    _write_jsonl(paths["rewards"], rewards)
    _write_jsonl(paths["step_rewards"], step_rewards)
    _write_jsonl(paths["preferences"], preferences)
    _write_jsonl(paths["failure_modes"], failure_modes)
    _write_json(paths["curriculum"], curriculum)

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
        "task_families": sorted({str(episode["task_family"]) for episode in episodes}),
        "outputs": {name: _display_path(path, preserve_paths) for name, path in paths.items()},
        "notes": [
            "Exports are built from normalized_trace.json and scorecard.json.",
            "Use these artifacts as reward/eval data, not as a complete trainer.",
            "failure_modes.jsonl exposes one failed-rule record per episode for curriculum construction.",
            "New scorecards include evidence_refs for structured event/final-answer/episode attribution.",
            "step_rewards.jsonl flattens failed-rule attribution into one row per event/final-answer/episode target.",
            "Reward labels are deterministic scenario-policy judgments and can be reward-hacked if scenarios are weak.",
        ],
    }
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
