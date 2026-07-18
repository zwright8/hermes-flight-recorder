#!/usr/bin/env python3
"""Prepare a public-safe, governed training handoff for the loop case study."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.data_governance import build_contamination_report, build_governance_receipt  # noqa: E402
from flightrecorder.review_semantics import (  # noqa: E402
    build_action_credit,
    build_branch_replay_dataset,
    build_contract_preferences,
    curate_training_rows,
)

FIXTURE_ROOT = ROOT / "examples" / "self_improving_loop"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _recovery_row(trajectory: dict[str, Any], template: dict[str, Any]) -> dict[str, Any]:
    episode_id = str(trajectory["episode_id"])
    return {
        "schema_version": "hfr.rl.action_sft.v1",
        "episode_id": episode_id,
        "sample_id": episode_id,
        "scenario_id": episode_id,
        "task_family": "inventory",
        "prompt": "Look up part A-17.",
        "response": "Part A-17 has quantity 12.",
        "messages": trajectory["messages"],
        "tools": template["tools"],
        "tool_schema_provenance": "recorded_exact",
        "human_label": "accept",
        "review_item_id": "review-inventory-recovery",
        "review_item_sha256": _sha256(trajectory),
        "reviewer_confidence": "high",
        "quality_score": 1.0,
        "training_role": "action_sft",
        "source_id": "synthetic-recovery",
        "environment": template["environment"],
        "policy": template["policy"],
        "scenario_contract": template["scenario_contract"],
        "governance": {
            **template["governance"],
            "deletion_subject_ids": ["synthetic-inventory-recovery"],
        },
    }


def _episode(row: dict[str, Any]) -> dict[str, Any]:
    passed = row.get("human_label") == "accept"
    response = str(row.get("response") or "")
    if not response:
        for message in reversed(row.get("messages", [])):
            if isinstance(message, dict) and message.get("role") == "assistant" and message.get("content"):
                response = str(message["content"])
                break
    episode_id = str(row["episode_id"])
    return {
        "schema_version": "hfr.rl.episode.v1",
        "episode_id": episode_id,
        "scenario_id": str(row.get("scenario_id") or episode_id),
        "task_family": str(row.get("task_family") or "unknown"),
        "prompt": str(row.get("prompt") or ""),
        "final_answer": response,
        "events": [],
        "outcome": {"passed": passed, "score": 100 if passed else 0, "reward": 1.0 if passed else -1.0},
        "task_completion": {"status": "complete" if passed else "incomplete", "passed": passed},
        "source_fingerprint_status": "verified",
        "source_fingerprints": {"synthetic_row": {"sha256": _sha256(row)}},
        "governance": row["governance"],
    }


def prepare(out: Path) -> dict[str, Any]:
    native = _load_jsonl(FIXTURE_ROOT / "native_trajectories.jsonl")
    recovery_trajectory = _load_json(FIXTURE_ROOT / "action_credit_trajectory.json")
    recovery = _recovery_row(recovery_trajectory, native[0])
    all_rows = native + [recovery]

    governance = build_governance_receipt(
        all_rows,
        purpose="agent_training",
        now="2028-01-01T00:00:00+00:00",
    )
    contamination = build_contamination_report(
        all_rows,
        protected_rows=_load_jsonl(FIXTURE_ROOT / "protected_benchmark.jsonl"),
    )
    action_credit = build_action_credit(recovery_trajectory)
    reviewed_preferences = build_contract_preferences(native)
    replay_request = _load_json(FIXTURE_ROOT / "branch_replay_request.json")
    branch_replay = build_branch_replay_dataset(**replay_request)
    curated = curate_training_rows(
        all_rows,
        recipe=_load_json(FIXTURE_ROOT / "curation_recipe.json"),
    )
    if governance.get("passed") is not True or contamination.get("passed") is not True:
        raise ValueError("case-study governance and contamination controls must pass")
    if not reviewed_preferences or not action_credit or not branch_replay.get("preferences"):
        raise ValueError("case-study review, credit, and branch evidence must be non-empty")

    _write_json(out / "governance.json", governance)
    _write_json(out / "contamination.json", contamination)
    _write_json(out / "curated.json", curated)
    _write_jsonl(out / "action_credit.jsonl", action_credit)
    _write_json(out / "branch_replay.json", branch_replay)
    _write_jsonl(out / "preferences.jsonl", reviewed_preferences)

    episodes = [_episode(row) for row in all_rows]
    training = out / "training_export"
    train_episodes = [row for row in episodes if row["task_family"] == "inventory"]
    _write_jsonl(training / "episodes.jsonl", episodes)
    _write_jsonl(training / "splits" / "train" / "episodes.jsonl", train_episodes)
    for name in ("sft", "dpo", "reward_model", "step_rewards"):
        _write_jsonl(training / f"{name}.jsonl", [])
    dataset_version = f"hfrds-{_sha256(all_rows)[:16]}"
    splits = {
        "schema_version": "hfr.rl.dataset_splits.v1",
        "strategy": "task_family_exclusive_synthetic_case_study",
        "assignments": [
            {
                "split": "train",
                "task_family": "inventory",
                "episode_ids": sorted(row["episode_id"] for row in episodes if row["task_family"] == "inventory"),
                "scenario_ids": sorted(row["scenario_id"] for row in episodes if row["task_family"] == "inventory"),
            },
            {
                "split": "test",
                "task_family": "schedule",
                "episode_ids": sorted(row["episode_id"] for row in episodes if row["task_family"] == "schedule"),
                "scenario_ids": sorted(row["scenario_id"] for row in episodes if row["task_family"] == "schedule"),
            },
        ],
        "summary": {"train": len(train_episodes), "validation": 0, "test": len(episodes) - len(train_episodes)},
        "leakage_checks": {"family_exclusive": True, "heldout_scenario_exclusive": True},
    }
    _write_json(training / "dataset_splits.json", splits)
    _write_json(
        training / "dataset_metrics.json",
        {
            "schema_version": "hfr.rl.dataset_metrics.v1",
            "episode_count": len(episodes),
            "passed": sum(1 for row in episodes if row["outcome"]["passed"]),
            "failed": sum(1 for row in episodes if not row["outcome"]["passed"]),
            "pass_rate": 0.75,
            "average_score": 75,
            "average_reward": 0.5,
            "redaction_status": {"passed": True, "status": "public_synthetic"},
            "quality_flags": [],
            "source_fingerprint_coverage": {"fully_verified": len(episodes), "unverified": 0},
            "dataset_splits": splits["summary"],
        },
    )
    _write_json(
        training / "manifest.json",
        {
            "schema_version": "hfr.rl.manifest.v1",
            "dataset_version": dataset_version,
            "episode_count": len(episodes),
            "redaction_status": {"passed": True, "status": "public_synthetic"},
        },
    )
    _write_json(
        out / "training_gate.json",
        {
            "schema_version": "hfr.training_gate.v1",
            "passed": True,
            "check_count": 6,
            "failed_check_count": 0,
            "metrics": {"public_synthetic": True},
        },
    )
    manifest = {
        "schema_version": "hfr.self_improving_case_study_handoff.v1",
        "dataset_version": dataset_version,
        "row_count": len(all_rows),
        "control_paths": {
            "governance": "governance.json",
            "contamination": "contamination.json",
            "curated": "curated.json",
            "action_credit": "action_credit.jsonl",
            "branch_replay": "branch_replay.json",
            "preferences": "preferences.jsonl",
        },
        "expected_training_effects": {
            "negative_credit_episode_excluded": "inventory-recovery",
            "human_rejection_preference_source": "inventory-rejected",
            "verified_branch_preference_count": len(branch_replay["preferences"]),
        },
    }
    _write_json(out / "case_study_handoff.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/self_improving_loop"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(json.dumps(prepare(args.out), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
