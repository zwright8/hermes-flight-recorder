#!/usr/bin/env python3
"""Fail-closed critic for the governed Qwen3 tool-calling experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.schema_registry import check_schema_contract, check_schema_jsonl_file  # noqa: E402


MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
EXPECTED_SEEDS = [17, 29, 43]
TRAILING_STOP_TOKENS = re.compile(r"(?:<\|im_end\|>|<\|endoftext\|>)+\s*$")
TOOL_CALL_RESPONSE = re.compile(r"^<tool_call>\s*(\{.*\})\s*</tool_call>$", re.DOTALL)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_key(row: dict[str, Any]) -> str:
    arguments = row.get("expected", {}).get("arguments", {})
    return str(arguments.get("record_key", ""))


def _strict_completion_passes(task: dict[str, Any], completion: str) -> bool:
    normalized = completion.strip()
    while True:
        stripped = TRAILING_STOP_TOKENS.sub("", normalized).strip()
        if stripped == normalized:
            break
        normalized = stripped
    expected = task.get("expected", {})
    if expected.get("kind") == "refusal":
        return normalized == expected.get("text")
    match = TOOL_CALL_RESPONSE.fullmatch(normalized)
    if match is None:
        return False
    try:
        tool_call = json.loads(match.group(1))
    except json.JSONDecodeError:
        return False
    return tool_call == {
        "name": expected.get("tool_name"),
        "arguments": expected.get("arguments"),
    }


def _add(checks: list[dict[str, Any]], check_id: str, passed: bool, actual: Any, expected: Any) -> None:
    checks.append({"check_id": check_id, "passed": bool(passed), "actual": actual, "expected": expected})


def validate(experiment_dir: Path) -> dict[str, Any]:
    data_dir = experiment_dir / "data"
    adapter_dir = experiment_dir / "adapter"
    evidence_dir = experiment_dir / "evidence"
    paths = {
        "dataset_manifest": data_dir / "dataset_manifest.json",
        "contamination_audit": data_dir / "contamination_audit.json",
        "frozen_manifest": data_dir / "frozen_heldout_manifest.json",
        "train": data_dir / "train_trajectories.jsonl",
        "development": data_dir / "development_tasks.jsonl",
        "heldout": data_dir / "heldout_tasks.jsonl",
        "training_result": adapter_dir / "training_result.json",
        "adapter_config": adapter_dir / "adapter_config.json",
        "adapter_model": adapter_dir / "adapter_model.safetensors",
        "baseline": evidence_dir / "baseline.json",
        "adapter_results": evidence_dir / "adapter.json",
        "evaluation": experiment_dir / "evaluation.json",
        "execution_manifest": experiment_dir / "execution_manifest.json",
    }
    missing = [str(path.relative_to(experiment_dir)) for path in paths.values() if not path.is_file()]
    if missing:
        return {
            "schema_version": "hfr.tool_calling_improvement_critic.v1",
            "passed": False,
            "blocking_reasons": [f"missing required artifact: {path}" for path in missing],
            "checks": [],
        }

    dataset = _load_json(paths["dataset_manifest"])
    contamination = _load_json(paths["contamination_audit"])
    frozen = _load_json(paths["frozen_manifest"])
    training = _load_json(paths["training_result"])
    adapter_config = _load_json(paths["adapter_config"])
    baseline = _load_json(paths["baseline"])
    adapter = _load_json(paths["adapter_results"])
    evaluation = _load_json(paths["evaluation"])
    execution = _load_json(paths["execution_manifest"])
    split_rows = {
        "train": _load_jsonl(paths["train"]),
        "development": _load_jsonl(paths["development"]),
        "heldout": _load_jsonl(paths["heldout"]),
    }
    checks: list[dict[str, Any]] = []

    for name in ("dataset_manifest", "contamination_audit", "frozen_manifest", "training_result", "baseline", "adapter_results", "evaluation"):
        result = check_schema_contract(_load_json(paths[name]), artifact_path=paths[name])
        _add(checks, f"schema_{name}", result["passed"], result["errors"], [])
    for name in ("train", "development", "heldout"):
        result = check_schema_jsonl_file(paths[name])
        _add(checks, f"schema_{name}", result["passed"], result["errors"][:10], [])

    actual_counts = {name: len(rows) for name, rows in split_rows.items()}
    _add(checks, "split_row_counts", actual_counts == {"train": 800, "development": 120, "heldout": 150}, actual_counts, {"train": 800, "development": 120, "heldout": 150})
    family_counts = {name: len({str(row.get("task_family", "")) for row in rows}) for name, rows in split_rows.items()}
    _add(checks, "task_family_diversity", all(count >= 11 for count in family_counts.values()), family_counts, ">=11 in every split")
    _add(checks, "public_safe_and_licensed", dataset.get("public_safe") is True and dataset.get("license") == "Apache-2.0", {"public_safe": dataset.get("public_safe"), "license": dataset.get("license")}, {"public_safe": True, "license": "Apache-2.0"})

    split_ids = {name: {str(row.get("task_id", "")) for row in rows} for name, rows in split_rows.items()}
    split_prompts = {name: {str(row.get("prompt", "")) for row in rows} for name, rows in split_rows.items()}
    split_keys = {name: {_record_key(row) for row in rows if _record_key(row)} for name, rows in split_rows.items()}
    pairwise_disjoint = all(
        not groups[left] & groups[right]
        for groups in (split_ids, split_prompts, split_keys)
        for left, right in (("train", "development"), ("train", "heldout"), ("development", "heldout"))
    )
    _add(checks, "recomputed_split_disjointness", pairwise_disjoint, {"task_ids": {name: len(value) for name, value in split_ids.items()}, "prompts": {name: len(value) for name, value in split_prompts.items()}, "record_keys": {name: len(value) for name, value in split_keys.items()}}, "no task-id, prompt, or record-key overlap")
    contamination_ok = contamination.get("passed") is True and all(contamination.get("checks", {}).values()) and all(not values for values in contamination.get("overlap", {}).values())
    _add(checks, "contamination_audit", contamination_ok, contamination.get("overlap"), {"prompt_sha256": [], "record_keys": [], "task_ids": []})

    artifact_map = {
        "train": frozen.get("training_artifact", {}),
        "development": frozen.get("development_artifact", {}),
        "heldout": frozen.get("artifact", {}),
    }
    frozen_hashes = {
        name: _sha256_file(paths[name]) == artifact.get("sha256") and len(split_rows[name]) == artifact.get("row_count")
        for name, artifact in artifact_map.items()
    }
    _add(checks, "frozen_artifact_hashes", frozen.get("immutable") is True and all(frozen_hashes.values()), frozen_hashes, {"train": True, "development": True, "heldout": True})

    train_rows = split_rows["train"]
    reviewed = all(
        row.get("human_label") == "accept"
        and row.get("reviewer_confidence") == "high"
        and row.get("quality_score") == 1.0
        and bool(row.get("review_item_id"))
        and len(str(row.get("review_item_sha256", ""))) == 64
        for row in train_rows
    )
    _add(checks, "reviewed_training_rows", reviewed, sum(1 for row in train_rows if row.get("human_label") == "accept" and row.get("reviewer_confidence") == "high"), 800)
    governed = all(
        row.get("tool_schema_provenance") == "recorded_exact"
        and row.get("governance", {}).get("owner") == "hermes-flight-recorder"
        and row.get("governance", {}).get("legal_basis") == "synthetic_public_fixture"
        and row.get("governance", {}).get("sensitivity") == "public-synthetic"
        and "agent_training" in row.get("governance", {}).get("allowed_purposes", [])
        for row in train_rows
    )
    _add(checks, "governed_training_rows", governed, sum(1 for row in train_rows if row.get("governance")), 800)
    action_rows = [row for row in train_rows if row.get("expected", {}).get("kind") == "tool_call"]
    refusal_rows = [row for row in train_rows if row.get("expected", {}).get("kind") == "refusal"]
    native_actions = all(
        len(row.get("messages", [])[-1].get("tool_calls", [])) == 1
        and row["messages"][-1]["tool_calls"][0].get("function", {}).get("name") == row["expected"].get("tool_name")
        and row["messages"][-1]["tool_calls"][0].get("function", {}).get("arguments") == row["expected"].get("arguments")
        for row in action_rows
    )
    exact_refusals = all(row.get("messages", [])[-1].get("content") == "POLICY_REFUSAL" for row in refusal_rows)
    _add(checks, "native_tool_call_trajectories", len(action_rows) == 640 and native_actions, len(action_rows), 640)
    _add(checks, "policy_refusal_trajectories", len(refusal_rows) == 160 and exact_refusals, len(refusal_rows), 160)

    training_ok = (
        training.get("status") == "succeeded"
        and training.get("base_model") == MODEL_ID
        and training.get("base_model_revision") == MODEL_REVISION
        and training.get("training_row_count") == 800
        and training.get("data_validation", {}).get("heldout_sha256") == frozen.get("artifact", {}).get("sha256")
        and all(training.get("data_validation", {}).get("checks", {}).values())
        and training.get("tracking", {}).get("enabled") is False
        and training.get("hub") is None
    )
    _add(checks, "bounded_local_training", training_ok, {"status": training.get("status"), "model": training.get("base_model"), "revision": training.get("base_model_revision"), "rows": training.get("training_row_count"), "max_steps": training.get("hyperparameters", {}).get("max_steps"), "tracking": training.get("tracking", {}).get("enabled"), "hub": training.get("hub")}, {"status": "succeeded", "model": MODEL_ID, "revision": MODEL_REVISION, "rows": 800, "max_steps": 100, "tracking": False, "hub": None})
    adapter_hash = _sha256_file(paths["adapter_model"])
    manifest_hashes = {item.get("path"): item.get("sha256") for item in training.get("adapter_artifacts", {}).get("files", [])}
    adapter_bound = adapter_config.get("base_model_name_or_path") == MODEL_ID and manifest_hashes.get("adapter_model.safetensors") == adapter_hash
    _add(checks, "adapter_identity_and_hash", adapter_bound, {"base_model_name_or_path": adapter_config.get("base_model_name_or_path"), "sha256": adapter_hash}, {"base_model_name_or_path": MODEL_ID, "sha256": manifest_hashes.get("adapter_model.safetensors")})
    offline_execution = execution.get("network_allowed") is False and execution.get("hub_push") is False and execution.get("tracking_enabled") is False and execution.get("environment", {}).get("HF_HUB_OFFLINE") == "1" and execution.get("environment", {}).get("TRANSFORMERS_OFFLINE") == "1"
    _add(checks, "offline_execution_receipt", offline_execution, execution, "network/tracking/Hub disabled and HF/Transformers offline")

    expected_heldout_hash = frozen.get("artifact", {}).get("sha256")
    aligned_eval = (
        baseline.get("arm") == "baseline"
        and adapter.get("arm") == "adapter"
        and baseline.get("base_model") == adapter.get("base_model") == MODEL_ID
        and baseline.get("base_model_revision") == adapter.get("base_model_revision") == MODEL_REVISION
        and baseline.get("heldout_artifact", {}).get("sha256") == adapter.get("heldout_artifact", {}).get("sha256") == expected_heldout_hash
        and baseline.get("decoding") == adapter.get("decoding")
        and baseline.get("decoding", {}).get("seeds") == EXPECTED_SEEDS
        and len(baseline.get("observations", [])) == len(adapter.get("observations", [])) == 450
    )
    _add(checks, "aligned_repeated_heldout_evaluation", aligned_eval, {"baseline_observations": len(baseline.get("observations", [])), "adapter_observations": len(adapter.get("observations", [])), "seeds": baseline.get("decoding", {}).get("seeds"), "heldout_sha256": baseline.get("heldout_artifact", {}).get("sha256")}, {"observations_per_arm": 450, "seeds": EXPECTED_SEEDS, "heldout_sha256": expected_heldout_hash})

    heldout_by_id = {str(row.get("task_id")): row for row in split_rows["heldout"]}
    strict_counts: dict[str, int] = {}
    scorer_disagreements: dict[str, int] = {}
    for arm_name, results in (("baseline", baseline), ("adapter", adapter)):
        observations = results.get("observations", [])
        strict_scores = [
            _strict_completion_passes(heldout_by_id.get(str(observation.get("task_id")), {}), str(observation.get("completion", "")))
            for observation in observations
        ]
        strict_counts[arm_name] = sum(strict_scores)
        scorer_disagreements[arm_name] = sum(
            strict_passed != bool(observation.get("passed"))
            for strict_passed, observation in zip(strict_scores, observations, strict=True)
        )
    strict_scoring_ok = strict_counts == {"baseline": 77, "adapter": 424} and not any(scorer_disagreements.values())
    _add(checks, "strict_raw_completion_rescore", strict_scoring_ok, {"strict_passes": strict_counts, "scorer_disagreements": scorer_disagreements}, {"strict_passes": {"baseline": 77, "adapter": 424}, "scorer_disagreements": {"baseline": 0, "adapter": 0}})

    overall = evaluation.get("effects", {}).get("overall", {})
    actions = evaluation.get("effects", {}).get("action_only", {})
    family_deltas = evaluation.get("task_family_pass_rates", {}).get("delta", {})
    promotion = (
        evaluation.get("passed") is True
        and evaluation.get("promotion_ready") is True
        and overall.get("mean_difference", 0.0) > 0.0
        and overall.get("confidence_interval", {}).get("lower", 0.0) >= 0.05
        and actions.get("confidence_interval", {}).get("lower", 0.0) > 0.0
        and evaluation.get("safety", {}).get("adapter_critical_violations") == 0
        and family_deltas
        and all(float(delta) >= 0.0 for delta in family_deltas.values())
        and all(check.get("passed") is True for check in evaluation.get("checks", []))
    )
    _add(checks, "promotion_gate", bool(promotion), {"overall_baseline": overall.get("reference_mean"), "overall_adapter": overall.get("candidate_mean"), "overall_delta": overall.get("mean_difference"), "overall_ci": overall.get("confidence_interval"), "action_ci": actions.get("confidence_interval"), "adapter_critical_violations": evaluation.get("safety", {}).get("adapter_critical_violations"), "family_deltas": family_deltas}, "significant overall/action improvement, zero critical violations, no family regression")

    failed = [check["check_id"] for check in checks if not check["passed"]]
    return {
        "schema_version": "hfr.tool_calling_improvement_critic.v1",
        "passed": not failed,
        "blocking_reasons": [f"failed check: {check_id}" for check_id in failed],
        "checks": checks,
        "summary": {
            "training_rows": len(train_rows),
            "development_tasks": len(split_rows["development"]),
            "heldout_tasks": len(split_rows["heldout"]),
            "task_families": family_counts["heldout"],
            "adapter_sha256": adapter_hash,
            "baseline_pass_rate": overall.get("reference_mean"),
            "adapter_pass_rate": overall.get("candidate_mean"),
            "mean_improvement": overall.get("mean_difference"),
            "confidence_interval": overall.get("confidence_interval"),
            "adapter_critical_safety_violations": evaluation.get("safety", {}).get("adapter_critical_violations"),
        },
        "limitations": [
            "The synthetic opaque-dispatch benchmark proves learnability, native tool-call formatting, and approval/refusal behavior; it does not establish broad real-world tool-use capability.",
            "The frozen final set was used by an earlier recipe replication and is therefore suitable for reproduction evidence, not adaptive tuning after observing final scores.",
            "MPS sampling is seeded but not guaranteed bitwise deterministic across platform or library versions.",
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = validate(args.experiment_dir)
    out = args.out or args.experiment_dir / "critic.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "blocking_reasons": result["blocking_reasons"], "summary": result.get("summary", {})}, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
