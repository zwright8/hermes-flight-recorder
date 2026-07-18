#!/usr/bin/env python3
"""Build a reproducible agentic fine-tuning experiment bundle.

The bundle compares two data sources for a Hermes runtime model:

* trace-only SFT rows derived from every completed Hermes/Flight Recorder
  episode, with no scorecard gating; and
* Flight Recorder curated rows: reviewed SFT, reviewed/compare DPO, reward-model
  labels, split metadata, and gate summaries.

This script intentionally does not train a model. It prepares the deterministic
inputs and statistics needed before launching an expensive Qwen fine-tune.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from flightrecorder.training import _agentic_messages, _inferred_tool_definitions


DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def _load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _file_fingerprint(path: Path, base_dir: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(base_dir).as_posix(),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def _messages(prompt: str, response: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


def _task_family_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("task_family") or "unknown") for row in rows)
    return dict(sorted(counts.items()))


def _row_task_family(row: dict[str, Any]) -> str:
    family = row.get("task_family")
    if family:
        return str(family)
    scenario_id = str(row.get("scenario_id") or row.get("episode_id") or "")
    parts = scenario_id.split("_")
    if parts and parts[-1] in {"good", "bad", "pass", "fail", "passing", "failing", "chosen", "rejected"}:
        parts = parts[:-1]
    return "_".join(parts) or "unknown"


def _label_counts(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    counts = Counter(str(row.get(key) or "unknown") for row in rows)
    return dict(sorted(counts.items()))


def _gate_summary(path: Path) -> dict[str, Any]:
    data = _load_json(path, {})
    if not data:
        return {"available": False, "path": str(path)}
    return {
        "available": True,
        "path": str(path),
        "passed": bool(data.get("passed", data.get("failed_check_count", 1) == 0)),
        "check_count": data.get("check_count"),
        "failed_check_count": data.get("failed_check_count"),
        "metric_keys": sorted((data.get("metrics") or {}).keys()),
    }


def _build_trace_only_rows(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        prompt = _as_text(episode.get("prompt"))
        response = _as_text(episode.get("final_answer"))
        if not prompt or not response:
            continue
        rows.append(
            {
                "sample_id": episode.get("episode_id"),
                "episode_id": episode.get("episode_id"),
                "scenario_id": episode.get("scenario_id"),
                "task_family": episode.get("task_family"),
                "prompt": prompt,
                "response": response,
                "messages": _messages(prompt, response),
                "source_artifact": "runs/training_export/episodes.jsonl",
                "training_arm": "hermes_trace_only_sft",
            }
        )
    return rows


def _build_action_sft_rows(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        outcome = episode.get("outcome") or {}
        task_completion = episode.get("task_completion") or {}
        if not outcome.get("passed") or task_completion.get("status") != "complete":
            continue
        prompt = _as_text(episode.get("prompt"))
        if not prompt:
            continue
        events = sorted(
            [event for event in episode.get("events", []) if isinstance(event, dict)],
            key=lambda event: int(event.get("order", event.get("index", 0)) or 0),
        )
        response = _as_text(episode.get("final_answer"))
        tool_calls = [event for event in events if event.get("type") == "tool_call" and event.get("tool_name")]
        rows.append(
            {
                "sample_id": episode.get("episode_id"),
                "episode_id": episode.get("episode_id"),
                "scenario_id": episode.get("scenario_id"),
                "task_family": episode.get("task_family"),
                "prompt": prompt,
                "response": response,
                "messages": _agentic_messages(prompt, response, events),
                "tools": _inferred_tool_definitions(tool_calls),
                "tool_schema_provenance": "inferred_from_observed_argument_shapes" if tool_calls else "not_applicable",
                "source_artifact": "runs/training_export/episodes.jsonl",
                "training_arm": "flightrecorder_action_sft",
                "quality_gate": "passed_scorecard_and_task_completion",
            }
        )
    return rows


def _normalize_sft_rows(rows: list[dict[str, Any]], source: str, arm: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        prompt = _as_text(row.get("prompt"))
        response = _as_text(row.get("response"))
        if not prompt or not response:
            continue
        out = dict(row)
        out["messages"] = row.get("messages") or _messages(prompt, response)
        out["source_artifact"] = source
        out["training_arm"] = arm
        normalized.append(out)
    return normalized


def _normalize_dpo_rows(rows: list[dict[str, Any]], source: str, arm: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        prompt = _as_text(row.get("prompt"))
        chosen = _as_text(row.get("chosen"))
        rejected = _as_text(row.get("rejected"))
        if not prompt or not chosen or not rejected:
            continue
        out = dict(row)
        out["chosen_messages"] = row.get("chosen_messages") or _messages(prompt, chosen)
        out["rejected_messages"] = row.get("rejected_messages") or _messages(prompt, rejected)
        out["source_artifact"] = source
        out["training_arm"] = arm
        normalized.append(out)
    return normalized


def _split_scenarios(dataset_splits: dict[str, Any]) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for assignment in dataset_splits.get("assignments", []):
        split = str(assignment.get("split") or "unknown")
        by_split[split].append(
            {
                "task_family": assignment.get("task_family"),
                "episode_ids": assignment.get("episode_ids", []),
                "scenario_ids": assignment.get("scenario_ids", []),
            }
        )
    return {
        "strategy": dataset_splits.get("strategy"),
        "family_exclusive": (dataset_splits.get("leakage_checks") or {}).get("family_exclusive"),
        "train": by_split.get("train", []),
        "validation": by_split.get("validation", []),
        "test": by_split.get("test", []),
        "heldout": by_split.get("validation", []) + by_split.get("test", []),
    }


def _split_filter_sets(split_plan: dict[str, Any]) -> dict[str, set[str]]:
    train_families = {str(item.get("task_family")) for item in split_plan.get("train", []) if item.get("task_family")}
    heldout_families = {
        str(item.get("task_family"))
        for split in ("validation", "test", "heldout")
        for item in split_plan.get(split, [])
        if item.get("task_family")
    }
    heldout_scenario_ids = {
        str(scenario_id)
        for split in ("validation", "test", "heldout")
        for item in split_plan.get(split, [])
        for scenario_id in item.get("scenario_ids", [])
    }
    return {
        "train_families": train_families,
        "heldout_families": heldout_families,
        "heldout_scenario_ids": heldout_scenario_ids,
    }


def _filter_train_rows(
    rows: list[dict[str, Any]],
    *,
    train_families: set[str],
    heldout_scenario_ids: set[str],
) -> list[dict[str, Any]]:
    if not train_families:
        return []
    filtered: list[dict[str, Any]] = []
    for row in rows:
        scenario_ids = [
            str(row.get(key))
            for key in ("scenario_id", "episode_id", "chosen_episode_id", "rejected_episode_id")
            if row.get(key)
        ]
        if any(scenario_id in heldout_scenario_ids for scenario_id in scenario_ids):
            continue
        if _row_task_family(row) not in train_families:
            continue
        filtered.append(row)
    return filtered


def _known_trace_quality(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    passed = 0
    failed = 0
    critical = Counter()
    failed_rules = Counter()
    task_completion = Counter()
    for episode in episodes:
        outcome = episode.get("outcome") or {}
        if outcome.get("passed"):
            passed += 1
        else:
            failed += 1
        for item in outcome.get("critical_failures", []):
            critical[str(item)] += 1
        for item in outcome.get("failed_rules", []):
            failed_rules[str(item)] += 1
        completion = episode.get("task_completion") or {}
        task_completion[str(completion.get("status") or "unknown")] += 1
    total = passed + failed
    return {
        "rows": total,
        "known_passed": passed,
        "known_failed": failed,
        "known_failed_rate": round(failed / total, 4) if total else 0.0,
        "critical_failure_counts": dict(sorted(critical.items())),
        "failed_rule_counts": dict(sorted(failed_rules.items())),
        "task_completion_status_counts": dict(sorted(task_completion.items())),
    }


def _write_training_dataset_manifest(
    out_dir: Path,
    training_dir: Path,
    dataset_metrics: dict[str, Any],
    dataset_splits: dict[str, Any],
    training_gate: dict[str, Any],
) -> Path:
    data_files = {
        "trace_sft": "data/hermes_trace_only_sft.jsonl",
        "flightrecorder_sft": "data/flightrecorder_sft.jsonl",
        "train_sft": "data/flightrecorder_sft.jsonl",
        "flightrecorder_action_sft": "data/flightrecorder_action_sft.jsonl",
        "train_action_sft": "data/flightrecorder_action_sft.jsonl",
        "flightrecorder_combined_dpo": "data/flightrecorder_combined_dpo.jsonl",
        "train_dpo": "data/flightrecorder_combined_dpo.jsonl",
        "flightrecorder_reward_model": "data/flightrecorder_reward_model.jsonl",
        "train_reward_model": "data/flightrecorder_reward_model.jsonl",
        "flightrecorder_step_rewards": "data/flightrecorder_step_rewards.jsonl",
        "train_step_rewards": "data/flightrecorder_step_rewards.jsonl",
    }
    source_manifest = _load_json(training_dir / "manifest.json", {})
    manifest_path = out_dir / "dataset_training_manifest.json"
    artifact_fingerprints = {
        name: _file_fingerprint(out_dir / relative_path, out_dir)
        for name, relative_path in data_files.items()
        if name.startswith("flightrecorder_") or name == "trace_sft"
    }
    leakage = dataset_splits.get("leakage_checks") if isinstance(dataset_splits.get("leakage_checks"), dict) else {}
    manifest = {
        "schema_version": "hfr.dataset_registry_entry.v1",
        "dataset_id": str(source_manifest.get("dataset_version") or "flightrecorder-agentic-training"),
        "dataset_version": str(source_manifest.get("dataset_version") or ""),
        "source_manifest": str((training_dir / "manifest.json").resolve()),
        "redaction_status": dataset_metrics.get("redaction_status", {}),
        "gates": {"training_gate": {"passed": training_gate.get("passed") is True}},
        "dataset_splits": dataset_splits.get("summary", {}),
        "leakage_checks": leakage,
        "quality_flags": dataset_metrics.get("quality_flags", []),
        "source_fingerprint_coverage": dataset_metrics.get("source_fingerprint_coverage", {}),
        "data_files": data_files,
        "artifact_fingerprints": artifact_fingerprints,
        "notes": [
            "All trainer aliases point to heldout-filtered experiment artifacts.",
            "The trainer must revalidate these SHA-256 fingerprints before launch.",
        ],
    }
    _write_json(manifest_path, manifest)
    return manifest_path


def build_bundle(runs_dir: Path, out_dir: Path, model: str) -> dict[str, Any]:
    training_dir = runs_dir / "training_export"
    reviewed_dir = runs_dir / "reviewed_export"
    compare_dir = runs_dir / "compare_rl_export"

    dataset_metrics = _load_json(training_dir / "dataset_metrics.json", {})
    dataset_splits = _load_json(training_dir / "dataset_splits.json", {})
    leakage_checks = dataset_splits.get("leakage_checks") if isinstance(dataset_splits.get("leakage_checks"), dict) else {}
    if not dataset_splits or not isinstance(dataset_splits.get("assignments"), list) or not dataset_splits["assignments"]:
        raise ValueError("dataset_splits.json must contain deterministic split assignments before building training data")
    if leakage_checks.get("family_exclusive") is not True or leakage_checks.get("heldout_scenario_exclusive") is not True:
        raise ValueError("dataset split leakage checks must pass before building training data")
    redaction_status = dataset_metrics.get("redaction_status") if isinstance(dataset_metrics.get("redaction_status"), dict) else {}
    if redaction_status.get("passed") is not True:
        raise ValueError("dataset redaction status must pass before building training data")
    training_gate = _gate_summary(runs_dir / "training_gate.json")
    if training_gate.get("available") is not True or training_gate.get("passed") is not True:
        raise ValueError("a passing training_gate.json is required before building training data")
    split_plan = _split_scenarios(dataset_splits)
    split_filters = _split_filter_sets(split_plan)
    train_filter = {
        "train_task_families": sorted(split_filters["train_families"]),
        "heldout_task_families": sorted(split_filters["heldout_families"]),
        "heldout_scenario_ids": sorted(split_filters["heldout_scenario_ids"]),
        "policy": "exclude validation/test task families and scenario ids from all training views",
    }

    episodes_all = _load_jsonl(training_dir / "episodes.jsonl")
    train_episodes_path = training_dir / "splits" / "train" / "episodes.jsonl"
    episodes = _load_jsonl(train_episodes_path) if train_episodes_path.exists() else _filter_train_rows(
        episodes_all,
        train_families=split_filters["train_families"],
        heldout_scenario_ids=split_filters["heldout_scenario_ids"],
    )
    reviewed_sft = _normalize_sft_rows(
        _load_jsonl(reviewed_dir / "reviewed_sft.jsonl"),
        "runs/reviewed_export/reviewed_sft.jsonl",
        "flightrecorder_reviewed_sft",
    )
    if not reviewed_sft:
        train_sft_path = training_dir / "splits" / "train" / "sft.jsonl"
        reviewed_sft = _normalize_sft_rows(
            _load_jsonl(train_sft_path if train_sft_path.exists() else training_dir / "sft.jsonl"),
            "runs/training_export/splits/train/sft.jsonl" if train_sft_path.exists() else "runs/training_export/sft.jsonl",
            "flightrecorder_scorecard_sft",
        )
    reviewed_dpo = _normalize_dpo_rows(
        _load_jsonl(reviewed_dir / "reviewed_dpo.jsonl"),
        "runs/reviewed_export/reviewed_dpo.jsonl",
        "flightrecorder_reviewed_dpo",
    )
    train_dpo_path = training_dir / "splits" / "train" / "dpo.jsonl"
    scorecard_dpo = _normalize_dpo_rows(
        _load_jsonl(train_dpo_path if train_dpo_path.exists() else training_dir / "dpo.jsonl"),
        "runs/training_export/splits/train/dpo.jsonl" if train_dpo_path.exists() else "runs/training_export/dpo.jsonl",
        "flightrecorder_scorecard_dpo",
    )
    compare_dpo = _normalize_dpo_rows(
        _load_jsonl(compare_dir / "improvement_dpo.jsonl"),
        "runs/compare_rl_export/improvement_dpo.jsonl",
        "flightrecorder_compare_dpo",
    )
    reviewed_reward_model = _load_jsonl(reviewed_dir / "reviewed_reward_model.jsonl")
    train_reward_model_path = training_dir / "splits" / "train" / "reward_model.jsonl"
    reward_model = reviewed_reward_model or _load_jsonl(
        train_reward_model_path if train_reward_model_path.exists() else training_dir / "reward_model.jsonl"
    )
    train_step_rewards_path = training_dir / "splits" / "train" / "step_rewards.jsonl"
    step_rewards = _load_jsonl(
        train_step_rewards_path if train_step_rewards_path.exists() else training_dir / "step_rewards.jsonl"
    )
    root_action_sft = _load_jsonl(training_dir / "action_sft.jsonl")
    train_action_sft_path = training_dir / "splits" / "train" / "action_sft.jsonl"
    if train_action_sft_path.exists():
        action_sft_source = _load_jsonl(train_action_sft_path)
    elif root_action_sft:
        action_sft_source = root_action_sft
    else:
        action_sft_source = _build_action_sft_rows(episodes_all)

    raw_counts = {
        "episodes": len(episodes_all),
        "flightrecorder_sft": len(reviewed_sft),
        "flightrecorder_action_sft": len(root_action_sft or _build_action_sft_rows(episodes_all)),
        "flightrecorder_reviewed_dpo": len(reviewed_dpo),
        "flightrecorder_scorecard_dpo": len(scorecard_dpo),
        "flightrecorder_compare_dpo": len(compare_dpo),
        "flightrecorder_reward_model": len(reward_model),
        "flightrecorder_step_rewards": len(step_rewards),
    }

    reviewed_sft = _filter_train_rows(
        reviewed_sft,
        train_families=split_filters["train_families"],
        heldout_scenario_ids=split_filters["heldout_scenario_ids"],
    )
    reviewed_dpo = _filter_train_rows(
        reviewed_dpo,
        train_families=split_filters["train_families"],
        heldout_scenario_ids=split_filters["heldout_scenario_ids"],
    )
    scorecard_dpo = _filter_train_rows(
        scorecard_dpo,
        train_families=split_filters["train_families"],
        heldout_scenario_ids=split_filters["heldout_scenario_ids"],
    )
    compare_dpo = _filter_train_rows(
        compare_dpo,
        train_families=split_filters["train_families"],
        heldout_scenario_ids=split_filters["heldout_scenario_ids"],
    )
    reward_model = _filter_train_rows(
        reward_model,
        train_families=split_filters["train_families"],
        heldout_scenario_ids=split_filters["heldout_scenario_ids"],
    )
    step_rewards = _filter_train_rows(
        step_rewards,
        train_families=split_filters["train_families"],
        heldout_scenario_ids=split_filters["heldout_scenario_ids"],
    )

    trace_only_sft = _build_trace_only_rows(episodes)
    action_sft = _filter_train_rows(
        action_sft_source,
        train_families=split_filters["train_families"],
        heldout_scenario_ids=split_filters["heldout_scenario_ids"],
    )
    data_dir = out_dir / "data"
    counts = {
        "hermes_trace_only_sft": _write_jsonl(data_dir / "hermes_trace_only_sft.jsonl", trace_only_sft),
        "flightrecorder_sft": _write_jsonl(data_dir / "flightrecorder_sft.jsonl", reviewed_sft),
        "flightrecorder_action_sft": _write_jsonl(data_dir / "flightrecorder_action_sft.jsonl", action_sft),
        "flightrecorder_reviewed_dpo": _write_jsonl(data_dir / "flightrecorder_reviewed_dpo.jsonl", reviewed_dpo),
        "flightrecorder_scorecard_dpo": _write_jsonl(data_dir / "flightrecorder_scorecard_dpo.jsonl", scorecard_dpo),
        "flightrecorder_compare_dpo": _write_jsonl(data_dir / "flightrecorder_compare_dpo.jsonl", compare_dpo),
        "flightrecorder_reward_model": _write_jsonl(data_dir / "flightrecorder_reward_model.jsonl", reward_model),
        "flightrecorder_step_rewards": _write_jsonl(data_dir / "flightrecorder_step_rewards.jsonl", step_rewards),
    }
    combined_dpo = reviewed_dpo + compare_dpo
    if not combined_dpo:
        # A normal ``run-suite --export-rl`` handoff produces scorecard-gated
        # preference rows without requiring a separate human-review or
        # baseline/candidate comparison export. Keep reviewed and comparison
        # labels preferred when they exist, but do not strand the advertised
        # SFT-then-DPO path when the standard export is the only evidence source.
        combined_dpo = scorecard_dpo
    counts["flightrecorder_combined_dpo"] = _write_jsonl(data_dir / "flightrecorder_combined_dpo.jsonl", combined_dpo)
    _write_json(out_dir / "heldout_scenarios.json", split_plan)
    train_filter["raw_counts_before_filter"] = raw_counts
    train_filter["counts_after_filter"] = {
        "episodes": len(episodes),
        "flightrecorder_sft": len(reviewed_sft),
        "flightrecorder_action_sft": len(action_sft),
        "flightrecorder_reviewed_dpo": len(reviewed_dpo),
        "flightrecorder_scorecard_dpo": len(scorecard_dpo),
        "flightrecorder_compare_dpo": len(compare_dpo),
        "flightrecorder_reward_model": len(reward_model),
        "flightrecorder_step_rewards": len(step_rewards),
    }
    train_filter["excluded_counts"] = {
        key: raw_counts[key] - train_filter["counts_after_filter"][key]
        for key in raw_counts
        if key in train_filter["counts_after_filter"]
    }

    gate_summaries = {
        "training_gate": training_gate,
        "reviewed_gate": _gate_summary(runs_dir / "reviewed_gate.json"),
        "compare_gate": _gate_summary(runs_dir / "compare_gate.json"),
        "evidence_bundle": _gate_summary(runs_dir / "evidence_bundle.json"),
    }
    dataset_training_manifest_path = _write_training_dataset_manifest(
        out_dir,
        training_dir,
        dataset_metrics,
        dataset_splits,
        training_gate,
    )

    stats = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "source_runs_dir": str(runs_dir),
        "dataset_counts": counts,
        "training_filter": train_filter,
        "flightrecorder_dataset_metrics": {
            "episode_count": dataset_metrics.get("episode_count"),
            "passed": dataset_metrics.get("passed"),
            "failed": dataset_metrics.get("failed"),
            "pass_rate": dataset_metrics.get("pass_rate"),
            "average_score": dataset_metrics.get("average_score"),
            "average_reward": dataset_metrics.get("average_reward"),
            "quality_flags": dataset_metrics.get("quality_flags", []),
            "task_completion": dataset_metrics.get("task_completion"),
            "trace_signal": dataset_metrics.get("trace_signal"),
            "dataset_splits": dataset_metrics.get("dataset_splits"),
        },
        "hermes_trace_only_quality_audit_not_used_as_labels": _known_trace_quality(episodes),
        "all_source_trace_quality_audit": _known_trace_quality(episodes_all),
        "flightrecorder_sft": {
            "rows": len(reviewed_sft),
            "task_family_counts": _task_family_counts(reviewed_sft),
            "quality_gate_counts": _label_counts(reviewed_sft, "quality_gate"),
            "human_label_counts": _label_counts(reviewed_sft, "human_label"),
        },
        "flightrecorder_action_sft": {
            "rows": len(action_sft),
            "task_family_counts": _task_family_counts(action_sft),
            "quality_gate_counts": _label_counts(action_sft, "quality_gate"),
        },
        "flightrecorder_dpo": {
            "reviewed_rows": len(reviewed_dpo),
            "scorecard_rows": len(scorecard_dpo),
            "compare_rows": len(compare_dpo),
            "combined_rows": len(combined_dpo),
            "task_family_counts": _task_family_counts(combined_dpo),
        },
        "flightrecorder_reward_model": {
            "rows": len(reward_model),
            "label_counts": _label_counts(reward_model, "human_label"),
            "task_family_counts": _task_family_counts(reward_model),
        },
        "heldout": split_plan,
        "gates": gate_summaries,
        "promotion_requirements": {
            "baseline_eval": "Run Qwen3-4B-Instruct-2507 through the held-out Flight Recorder suite.",
            "trace_only_arm": "Fine-tune from data/hermes_trace_only_sft.jsonl, then rerun the exact held-out suite.",
            "flightrecorder_arm": "Fine-tune from Flight Recorder SFT plus DPO/reward views, then rerun the exact held-out suite.",
            "required_movement": [
                "higher pass rate than baseline and trace-only arm",
                "higher average score than baseline and trace-only arm",
                "fewer critical failures",
                "improved task-completion evidence",
                "no new forbidden-action regressions",
                "no new unsupported-claim regressions",
            ],
        },
    }
    _write_json(out_dir / "stats.json", stats)

    eval_status = _evaluation_status(out_dir)
    manifest = {
        "schema_version": "hfr.experiment.qwen_agentic_finetune.v1",
        "generated_at": stats["generated_at"],
        "model": model,
        "objective": "Demonstrate whether Flight Recorder curated outputs train a Hermes runtime model better than raw Hermes traces alone.",
        "artifacts": {
            "stats": "stats.json",
            "dataset_training_manifest": dataset_training_manifest_path.relative_to(out_dir).as_posix(),
            "heldout_scenarios": "heldout_scenarios.json",
            "trace_only_sft": "data/hermes_trace_only_sft.jsonl",
            "flightrecorder_sft": "data/flightrecorder_sft.jsonl",
            "flightrecorder_action_sft": "data/flightrecorder_action_sft.jsonl",
            "flightrecorder_combined_dpo": "data/flightrecorder_combined_dpo.jsonl",
            "flightrecorder_reward_model": "data/flightrecorder_reward_model.jsonl",
            "flightrecorder_step_rewards": "data/flightrecorder_step_rewards.jsonl",
            **eval_status["artifacts"],
        },
        "status": {
            "data_bundle_ready": bool(
                counts["flightrecorder_sft"]
                and counts["flightrecorder_action_sft"]
                and training_gate.get("passed") is True
                and redaction_status.get("passed") is True
            ),
            "baseline_model_eval_complete": eval_status["baseline_complete"],
            "trace_only_finetune_complete": eval_status["trace_only_complete"],
            "flightrecorder_finetune_complete": eval_status["flightrecorder_complete"],
            "promotion_comparison_complete": eval_status["promotion_complete"],
        },
        "next_commands": [
            "python3 -m flightrecorder validate --runs runs --training-export runs/training_export --strict",
            "python3 -m flightrecorder gate-export --training-export runs/training_export --policy examples/training_gate_policy.demo.json",
            "python3 scripts/build_agentic_finetune_experiment.py --runs-dir runs --out experiments/qwen3_4b_flightrecorder",
            "python3 scripts/train_agentic_lora.py --mode trace_sft --dry-run --experiment-dir experiments/qwen3_4b_flightrecorder",
            "python3 scripts/train_agentic_lora.py --mode fr_sft_dpo --dry-run --experiment-dir experiments/qwen3_4b_flightrecorder",
            "uv venv --python 3.11 .venv",
            "uv pip install --python .venv/bin/python torch transformers peft accelerate",
            ".venv/bin/python scripts/preflight_serving_runtime.py --model Qwen/Qwen3-4B-Instruct-2507 --runtime-python .venv/bin/python --adapter trace_only=experiments/qwen3_4b_flightrecorder/adapters/qwen3_4b_local_trace_sft/trace_sft_adapter --adapter flightrecorder=experiments/qwen3_4b_flightrecorder/adapters/qwen3_4b_local_fr_sft_dpo/fr_sft_dpo_adapter --out experiments/qwen3_4b_flightrecorder/serving/real_runtime_preflight/serving_runtime_preflight.json --report experiments/qwen3_4b_flightrecorder/serving/real_runtime_preflight/SERVING_RUNTIME_PREFLIGHT.md --allow-blocked",
            ".venv/bin/python scripts/serve_transformers_openai.py --model Qwen/Qwen3-4B-Instruct-2507 --port 8000",
            ".venv/bin/python scripts/serve_transformers_openai.py --model Qwen/Qwen3-4B-Instruct-2507 --adapter <adapter-dir> --port 8000",
            ".venv/bin/python scripts/check_openai_serving.py --engine transformers --arm <arm> --model <served-model> --adapter <adapter-dir> --base-url <openai-compatible-base-url> --out experiments/qwen3_4b_flightrecorder/serving/<arm>",
            ".venv/bin/python scripts/run_managed_serving_eval.py --server-command \".venv/bin/python scripts/serve_transformers_openai.py --model Qwen/Qwen3-4B-Instruct-2507 --adapter <adapter-dir> --host 127.0.0.1 --port 8000\" --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3-4B-Instruct-2507 --adapter <adapter-dir> --arm <arm> --out experiments/qwen3_4b_flightrecorder/serving/<arm> --eval-command \".venv/bin/python scripts/evaluate_hermes_heldout.py --arm <arm> --model Qwen/Qwen3-4B-Instruct-2507 --base-url {base_url} --serving-profile {serving_profile} --out experiments/qwen3_4b_flightrecorder/evaluations/<arm> --force\"",
            ".venv/bin/python scripts/evaluate_hermes_heldout.py --arm baseline --model Qwen/Qwen3-4B-Instruct-2507 --base-url <openai-compatible-base-url> --serving-profile experiments/qwen3_4b_flightrecorder/serving/baseline/serving_profile.json --out experiments/qwen3_4b_flightrecorder/evaluations/baseline --force",
            ".venv/bin/python scripts/evaluate_hermes_heldout.py --arm trace_only --model <served-trace-only-adapter-model> --base-url <openai-compatible-base-url> --serving-profile experiments/qwen3_4b_flightrecorder/serving/trace_only/serving_profile.json --out experiments/qwen3_4b_flightrecorder/evaluations/trace_only --force",
            ".venv/bin/python scripts/evaluate_hermes_heldout.py --arm flightrecorder --model <served-flightrecorder-adapter-model> --base-url <openai-compatible-base-url> --serving-profile experiments/qwen3_4b_flightrecorder/serving/flightrecorder/serving_profile.json --out experiments/qwen3_4b_flightrecorder/evaluations/flightrecorder --force",
            ".venv/bin/python scripts/verify_serving_profiles.py --profile baseline=experiments/qwen3_4b_flightrecorder/serving/baseline/serving_profile.json --profile trace_only=experiments/qwen3_4b_flightrecorder/serving/trace_only/serving_profile.json --profile flightrecorder=experiments/qwen3_4b_flightrecorder/serving/flightrecorder/serving_profile.json --required-arm baseline --required-arm trace_only --required-arm flightrecorder --require-structured-output --out experiments/qwen3_4b_flightrecorder/serving/serving_endpoint_suite.json --report experiments/qwen3_4b_flightrecorder/serving/SERVING_ENDPOINTS.md",
            "python3 scripts/compare_agentic_finetune_results.py --baseline <baseline-suite-summary> --trace-only <trace-suite-summary> --flightrecorder <flightrecorder-suite-summary>",
            ".venv/bin/python scripts/build_serving_demo_report.py --arm baseline=<baseline-evaluation-summary> --arm trace_only=<trace-only-evaluation-summary> --arm flightrecorder=<flightrecorder-evaluation-summary> --endpoint-suite experiments/qwen3_4b_flightrecorder/serving/serving_endpoint_suite.json --out experiments/qwen3_4b_flightrecorder/serving/demo_run.json --report experiments/qwen3_4b_flightrecorder/serving/DEMO_REPORT.md",
        ],
    }
    _write_json(out_dir / "manifest.json", manifest)
    _write_card(out_dir / "EXPERIMENT_CARD.md", stats, manifest)

    _copy_if_exists(training_dir / "DATASET_CARD.md", out_dir / "source_DATASET_CARD.md")
    return manifest


def _evaluation_status(out_dir: Path) -> dict[str, Any]:
    artifacts: dict[str, str] = {}

    def complete(name: str) -> bool:
        path = out_dir / "evaluations" / name / "evaluation_summary.json"
        if not path.exists():
            return False
        rel = path.relative_to(out_dir).as_posix()
        artifacts[f"{name}_evaluation_summary"] = rel
        data = _load_json(path, {})
        scenario_count = int(data.get("scenario_count") or 0)
        total = int(data.get("total") or 0)
        handoff = data.get("governance_handoff") if isinstance(data.get("governance_handoff"), dict) else {}
        return bool(data) and scenario_count > 0 and total == scenario_count and int(data.get("error_count") or 0) == 0 and handoff.get("ready") is True

    promotion_path = out_dir / "promotion_comparison.json"
    promotion_complete = False
    if promotion_path.exists():
        artifacts["promotion_comparison"] = promotion_path.relative_to(out_dir).as_posix()
        promotion_complete = bool((_load_json(promotion_path, {}) or {}).get("passed") is True)

    smoke_path = out_dir / "local_qwen3_4b_smoke.json"
    if smoke_path.exists():
        artifacts["local_qwen3_4b_smoke"] = smoke_path.relative_to(out_dir).as_posix()

    local_training_path = out_dir / "local_4b_training_results.json"
    if local_training_path.exists():
        artifacts["local_4b_training_results"] = local_training_path.relative_to(out_dir).as_posix()

    training_smoke_path = out_dir / "smoke_results.json"
    if training_smoke_path.exists():
        artifacts["training_smoke_results"] = training_smoke_path.relative_to(out_dir).as_posix()

    return {
        "artifacts": artifacts,
        "baseline_complete": complete("baseline"),
        "trace_only_complete": complete("trace_only"),
        "flightrecorder_complete": complete("flightrecorder"),
        "promotion_complete": promotion_complete,
    }


def _write_card(path: Path, stats: dict[str, Any], manifest: dict[str, Any]) -> None:
    trace_quality = stats["hermes_trace_only_quality_audit_not_used_as_labels"]
    metrics = stats["flightrecorder_dataset_metrics"]
    lines = [
        "# Qwen3-4B Flight Recorder Fine-Tune Experiment",
        "",
        f"- Model: `{stats['model']}`",
        f"- Generated: `{stats['generated_at']}`",
        f"- Source runs: `{stats['source_runs_dir']}`",
        "",
        "## Current Evidence",
        "",
        f"- Episodes: {metrics.get('episode_count')} ({metrics.get('passed')} passed, {metrics.get('failed')} failed)",
        f"- Pass rate: {metrics.get('pass_rate')}",
        f"- Average score: {metrics.get('average_score')}",
        f"- Raw trace-only SFT rows: {stats['dataset_counts']['hermes_trace_only_sft']}",
        f"- Raw trace-only known failed row rate: {trace_quality['known_failed_rate']}",
        f"- Flight Recorder SFT rows: {stats['dataset_counts']['flightrecorder_sft']}",
        f"- Flight Recorder action SFT rows: {stats['dataset_counts']['flightrecorder_action_sft']}",
        f"- Flight Recorder combined DPO rows: {stats['dataset_counts']['flightrecorder_combined_dpo']}",
        f"- Flight Recorder reward-model rows: {stats['dataset_counts']['flightrecorder_reward_model']}",
        f"- Flight Recorder step-reward rows: {stats['dataset_counts']['flightrecorder_step_rewards']}",
        "",
        "## Why This Bundle Matters",
        "",
        "The trace-only arm imitates every completed final answer, including known failed runs.",
        "The Flight Recorder arm separates accepted SFT examples, rejected behavior, DPO pairs, reward labels, step rewards, and held-out task families.",
        "",
        "## Held-Out Splits",
        "",
    ]
    for split in ("validation", "test"):
        entries = stats["heldout"].get(split, [])
        lines.append(f"### {split.title()}")
        if not entries:
            lines.append("")
            lines.append("- None")
        for entry in entries:
            lines.append(
                f"- `{entry.get('task_family')}`: scenarios {', '.join(entry.get('scenario_ids', []))}"
            )
        lines.append("")
    lines.extend(
        [
            "## Completion Status",
            "",
        ]
    )
    for key, value in manifest["status"].items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    lines.append(
        "The data bundle is ready, but the model training/evaluation proof is not complete until the baseline, trace-only, and Flight Recorder fine-tuned models are all evaluated on the same held-out scenarios."
    )
    lines.append("")
    lines.append("Use `scripts/serve_transformers_openai.py` to expose the base model or a PEFT adapter through a local OpenAI-compatible `/v1/chat/completions` endpoint.")
    lines.append("Use `scripts/preflight_serving_runtime.py` before starting real model servers so missing serving dependencies are recorded as artifacts instead of partial launches.")
    lines.append("Use `scripts/check_openai_serving.py` before evaluation to write a serving profile and compatibility report for the endpoint.")
    lines.append("Use `scripts/run_managed_serving_eval.py` when the evaluation loop should start, preflight, use, and stop a local serving process.")
    lines.append("Use `scripts/verify_serving_profiles.py` to prove all required endpoint profiles are ready before comparing arms or publishing a demo.")
    lines.append("")
    lines.append("Use `scripts/evaluate_hermes_heldout.py` to run those held-out scenarios through a live Hermes runtime and produce suite summaries for promotion comparison.")
    lines.append("")
    lines.append("Use `scripts/build_serving_demo_report.py` after baseline and candidate evals to create a replayable base-vs-candidate demo report with endpoint-suite readiness links.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"), help="Flight Recorder runs directory")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/qwen3_4b_flightrecorder"),
        help="Output experiment bundle directory",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base/instruct model id for the experiment")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_bundle(args.runs_dir, args.out, args.model)
    print(json.dumps({"wrote": str(args.out), "manifest": manifest["schema_version"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
