#!/usr/bin/env python3
"""Run the deterministic offline governed LoRA recipe-search demonstration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from flightrecorder.lora_recipe_search import (
    build_promotion_handoff,
    build_search_plan,
    run_search,
    validate_promotion_handoff,
    validate_search_result,
    write_json,
)
from flightrecorder.repeated_eval import (
    build_observation,
    build_promotion_evidence,
    validate_promotion_evidence,
)

ARMS = ("baseline", "trace_only", "flightrecorder")
POOLS = ("frozen", "rolling", "adversarial")
REPEATS = 3
CREATED_AT = "2026-07-20T12:00:00+00:00"


class QueueProposer:
    """Deterministic stand-in for an agent proposing bounded hypotheses."""

    def __init__(self) -> None:
        self._cursor = 0
        self._proposals = [
            _proposal(
                "wider-low-rank-subspace",
                "A wider low-rank subspace may improve tool-routing representations.",
                {"lora_r": 32},
            ),
            _proposal(
                "raise-dropout",
                "More adapter dropout may regularize the small development workload.",
                {"lora_dropout": 0.2},
            ),
            _proposal(
                "raise-sft-learning-rate",
                "A moderately higher SFT learning rate may converge within the fixed step budget.",
                {"sft_learning_rate": 0.0002},
            ),
            _proposal(
                "shorter-training",
                "Fewer optimizer steps may retain the gain with less compute.",
                {"max_steps": 40},
            ),
            _proposal(
                "longer-training",
                "The current incumbent may benefit from a bounded increase in optimizer steps.",
                {"max_steps": 120},
            ),
        ]

    def propose(self, state: dict[str, Any]) -> dict[str, Any] | None:
        if state["remaining_budget"]["trials"] <= 0 or self._cursor >= len(self._proposals):
            return None
        proposal = self._proposals[self._cursor]
        self._cursor += 1
        return proposal


class DeterministicDevelopmentEvaluator:
    """Synthetic evaluator used only to exercise Flight Recorder contracts."""

    def evaluate(
        self,
        recipe: dict[str, Any],
        *,
        trial_id: str,
        development_suite_path: Path,
    ) -> dict[str, Any]:
        suite = json.loads(development_suite_path.read_text(encoding="utf-8"))
        if "development" not in {str(tag).lower() for tag in suite.get("tags", [])}:
            raise ValueError("evaluator received a non-development suite")
        score = 0.55
        score += {16: 0.0, 32: 0.08, 64: 0.06}.get(recipe["lora_r"], -0.1)
        if recipe["sft_learning_rate"] == 0.0002:
            score += 0.09
        if recipe["lora_dropout"] > 0.1:
            score -= 0.07
        if recipe["max_steps"] < 80:
            score -= 0.05
        elif recipe["max_steps"] > 80:
            score += 0.04
        score = round(score, 6)
        return {
            "status": "completed",
            "primary_metric": score,
            "critical_failures": 0,
            "cost_usd": 0.02,
            "duration_seconds": 2.0,
            "candidate_identity_sha256": _digest(
                json.dumps(
                    {"kind": "simulated_lora_adapter", "recipe": recipe},
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
            "diagnostics": [
                "Synthetic deterministic development score; no model was downloaded or trained.",
                f"Development suite: {suite['suite_id']}; trial: {trial_id}",
            ],
            "execution_mode": "simulation",
            "external_side_effects_observed": False,
            "model_weights_updated_externally": False,
        }


def run_demo(out_dir: str | Path) -> dict[str, Any]:
    root = Path(out_dir)
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"output directory must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)

    development_suite_path = root / "development_suite.json"
    write_json(
        development_suite_path,
        {
            "schema_version": "hfr.eval_suite_manifest.v1",
            "suite_id": "autoresearch_lora_development_v1",
            "description": "Synthetic development-only selector for bounded LoRA recipe search.",
            "tags": ["development", "synthetic", "lora_recipe_search"],
            "scenario_ids": [
                "prompt_injection_good",
                "tool_schema_good",
                "state_change_good",
            ],
            "notes": [
                "This selector is available to candidate search.",
                "Frozen, rolling, adversarial, and final promotion inputs are deliberately absent.",
            ],
        },
    )

    plan_path = root / "search_plan.json"
    plan = build_search_plan(
        campaign_id="autoresearch-lora-demo-v1",
        objective="Improve synthetic development task pass rate without critical failures.",
        development_suite_path=development_suite_path,
        base_recipe={
            "mode": "fr_sft",
            "sft_learning_rate": 0.0001,
            "dpo_learning_rate": 0.00001,
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "max_steps": 80,
            "max_length": 640,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "seed": 42,
            "data_seed": 42,
        },
        mutable_fields=[
            "sft_learning_rate",
            "batch_size",
            "gradient_accumulation_steps",
            "max_steps",
            "max_length",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
        ],
        budget={
            "max_trials": 6,
            "max_cost_usd": 0.18,
            "max_duration_seconds": 18.0,
            "per_trial_cost_ceiling_usd": 0.03,
            "per_trial_duration_ceiling_seconds": 3.0,
        },
        out_path=plan_path,
        primary_metric="development_pass_rate",
        direction="maximize",
        minimum_delta=0.01,
        max_critical_failures=0,
        created_at=CREATED_AT,
    )
    write_json(plan_path, plan)

    search_result_path = root / "search_result.json"
    search_result = run_search(
        plan_path=plan_path,
        out_path=search_result_path,
        proposer=QueueProposer(),
        evaluator=DeterministicDevelopmentEvaluator(),
        created_at=CREATED_AT,
    )
    search_validation = validate_search_result(search_result_path)
    if not search_validation["passed"]:
        raise ValueError("search result failed replay: " + "; ".join(search_validation["errors"]))

    champion_sha = search_result["champion"]["candidate_identity_sha256"]
    observation_paths = _write_promotion_observations(root / "promotion", champion_sha)
    promotion_evidence_path = root / "promotion_evidence.json"
    promotion_evidence = build_promotion_evidence(
        observation_paths=observation_paths,
        policy={"bootstrap_samples": 200, "bootstrap_seed": 19},
        out_path=promotion_evidence_path,
        created_at="2026-07-20T13:00:00+00:00",
    )
    write_json(promotion_evidence_path, promotion_evidence)
    evidence_validation = validate_promotion_evidence(promotion_evidence_path)
    if not evidence_validation["passed"]:
        raise ValueError("promotion evidence failed replay: " + "; ".join(evidence_validation["errors"]))

    handoff_path = root / "promotion_handoff.json"
    handoff = build_promotion_handoff(
        search_result_path=search_result_path,
        promotion_evidence_path=promotion_evidence_path,
        out_path=handoff_path,
        created_at="2026-07-20T13:05:00+00:00",
    )
    write_json(handoff_path, handoff)
    handoff_validation = validate_promotion_handoff(handoff_path)
    if not handoff_validation["passed"]:
        raise ValueError("promotion handoff failed replay: " + "; ".join(handoff_validation["errors"]))

    report_path = root / "REPORT.md"
    report_path.write_text(_report(search_result, promotion_evidence, handoff), encoding="utf-8")
    return {
        "passed": search_result["passed"] and promotion_evidence["passed"] and handoff["passed"],
        "search_plan": str(plan_path),
        "search_result": str(search_result_path),
        "promotion_evidence": str(promotion_evidence_path),
        "promotion_handoff": str(handoff_path),
        "report": str(report_path),
        "trial_count": search_result["trial_count"],
        "kept_trial_count": search_result["kept_trial_count"],
        "discarded_trial_count": search_result["discarded_trial_count"],
        "champion_trial_id": search_result["champion"]["trial_id"],
        "development_metric": search_result["champion"]["development_metric"],
        "promotion_ready": promotion_evidence["promotion_ready"],
        "governance_ready": handoff["readiness"] == "ready_for_governance_review",
    }


def _write_promotion_observations(root: Path, champion_sha: str) -> dict[str, list[str | Path]]:
    identities = {arm: _identity(arm, champion_sha) for arm in ARMS}
    cases: dict[str, dict[str, Any]] = {
        "baseline": {"passed": False, "score": 35},
        "trace_only": {"passed": False, "score": 50},
        "flightrecorder": {"passed": True, "score": 90},
    }
    paths: dict[str, list[str | Path]] = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for pool in POOLS:
            for repeat_index in range(REPEATS):
                observation_dir = root / arm / pool / str(repeat_index)
                identity_path = observation_dir / "arm_identity.json"
                evaluation_path = observation_dir / "evaluation_summary.json"
                suite_path = observation_dir / "suite_summary.json"
                request_path = observation_dir / "request_attestation.json"
                serving_path = observation_dir / "serving_profile.json"
                observation_path = observation_dir / "observation.json"
                identity = identities[arm]
                case = cases[arm]
                seed = 1000 + repeat_index
                decoding = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 256}
                scenario_id = f"autoresearch-{pool}-scenario"
                write_json(identity_path, identity)
                write_json(
                    evaluation_path,
                    {
                        "schema_version": "hfr.hermes_heldout_eval_summary.v1",
                        "arm": arm,
                        "model": identity["model"]["id"],
                        "base_url": "http://127.0.0.1:8000/v1",
                        "simulation": True,
                    },
                )
                write_json(
                    suite_path,
                    _suite_summary(
                        arm=arm,
                        pool=pool,
                        repeat_index=repeat_index,
                        scenario_id=scenario_id,
                        passed=case["passed"],
                        score=case["score"],
                    ),
                )
                write_json(request_path, _request_attestation(identity["model"]["id"], seed, decoding))
                if arm != "baseline":
                    write_json(serving_path, _serving_profile(identity))
                observation = build_observation(
                    arm_identity_path=identity_path,
                    evaluation_summary_path=evaluation_path,
                    suite_summary_path=suite_path,
                    request_attestation_path=request_path,
                    serving_profile_path=serving_path if arm != "baseline" else None,
                    repeat_index=repeat_index,
                    seed=seed,
                    decoding=decoding,
                    pool_type=pool,
                    pool_id=f"autoresearch-{pool}-v1",
                    risk_tier="standard",
                    out_path=observation_path,
                    created_at=f"2026-07-20T13:00:0{repeat_index}+00:00",
                )
                write_json(observation_path, observation)
                paths[arm].append(observation_path)
    return paths


def _suite_summary(
    *,
    arm: str,
    pool: str,
    repeat_index: int,
    scenario_id: str,
    passed: bool,
    score: int,
) -> dict[str, Any]:
    passed_count = 1 if passed else 0
    failed_count = 0 if passed else 1
    return {
        "schema_version": "hfr.run_suite.v1",
        "scenarios_dir": "synthetic_promotion_scenarios",
        "out_dir": ".",
        "total": 1,
        "passed": passed_count,
        "failed": failed_count,
        "error_count": 0,
        "errors": [],
        "metrics": {
            "pass_rate": float(passed_count),
            "average_score": float(score),
            "min_score": score,
            "max_score": score,
            "failed_rule_counts": [],
            "critical_failure_counts": [],
            "task_families": [
                {
                    "task_family": "agentic_tool_use",
                    "total": 1,
                    "passed": passed_count,
                    "failed": failed_count,
                    "pass_rate": float(passed_count),
                    "average_score": float(score),
                    "failed_rule_counts": [],
                    "critical_failure_counts": [],
                }
            ],
            "failed": failed_count,
            "passed": passed_count,
        },
        "runs": [
            {
                "scenario_id": scenario_id,
                "scenario_title": f"Synthetic {pool} promotion scenario",
                "scenario_path": f"scenarios/{scenario_id}.json",
                "scenario_sha256": _digest(scenario_id),
                "trace_path": "trace.json",
                "trace_sha256": _digest(f"trace:{arm}:{pool}:{repeat_index}"),
                "run_dir": ".",
                "report": "report.md",
                "report_sha256": _digest("report"),
                "report_size_bytes": 1,
                "scorecard": "scorecard.json",
                "scorecard_sha256": _digest("scorecard"),
                "scorecard_size_bytes": 1,
                "run_digest": "run_digest.json",
                "run_digest_sha256": _digest("run-digest"),
                "run_digest_size_bytes": 1,
                "lineage": "lineage.json",
                "lineage_sha256": _digest("lineage"),
                "lineage_size_bytes": 1,
                "passed": passed,
                "score": score,
                "failed_rules": [],
                "critical_failures": [],
                "tool_schema_valid": True,
                "cost_usd": 1.0,
                "latency_seconds": 1.0,
                "task_family": "agentic_tool_use",
                "risk_tier": "standard",
            }
        ],
        "artifacts": {},
    }


def _request_attestation(model: str, seed: int, decoding: dict[str, Any]) -> dict[str, Any]:
    configured = {"seed": seed, **decoding}
    config_sha256 = _digest(json.dumps(configured, sort_keys=True, separators=(",", ":")))
    return {
        "schema_version": "hfr.eval_request_attestation.v1",
        "endpoint_base_url": "http://127.0.0.1:8000/v1",
        "configured": {**configured, "config_sha256": config_sha256},
        "request_count": 1,
        "matching_request_count": 1,
        "observed_models": [model],
        "requests": [
            {
                "request_index": 0,
                "path": "/v1/chat/completions",
                "model": model,
                **configured,
                "config_sha256": config_sha256,
                "body_sha256": _digest(f"request:{model}:{seed}"),
                "matched": True,
            }
        ],
        "passed": True,
        "blocking_reasons": [],
    }


def _serving_profile(identity: dict[str, Any]) -> dict[str, Any]:
    adapter = identity["adapter"]
    model = identity["model"]["id"]
    served_model = f"{model}+{adapter['id']}"
    return {
        "schema_version": "hfr.serving_profile.v1",
        "generated_at": CREATED_AT,
        "profile_id": f"profile-{identity['arm']}",
        "arm": identity["arm"],
        "provider": "custom",
        "engine": "vllm",
        "endpoint": {"base_url": "http://127.0.0.1:8000/v1"},
        "model_identity": {
            "requested_model": model,
            "served_model_id": served_model,
            "observed_model_ids": [served_model],
            "metadata_model": served_model,
            "chat_response_model": served_model,
            "adapter": {
                "present": True,
                "local": False,
                "immutable": True,
                "observation_source": "endpoint_model_metadata",
                **adapter,
            },
        },
        "capabilities": {"chat_completions": True},
        "eval_preflight": {"ready": True, "readiness": "ready", "failed_checks": []},
    }


def _identity(arm: str, champion_sha: str) -> dict[str, Any]:
    adapter = None
    if arm == "trace_only":
        adapter = {
            "id": "demo/trace-only-lora",
            "revision": "trace-only-demo-r1",
            "sha256": _digest("trace-only-demo-adapter"),
        }
    elif arm == "flightrecorder":
        adapter = {
            "id": "demo/autoresearch-flightrecorder-lora",
            "revision": "autoresearch-demo-r1",
            "sha256": champion_sha,
        }
    return {
        "schema_version": "hfr.eval_arm_identity.v1",
        "arm": arm,
        "model": {
            "id": "Qwen/Qwen3-0.6B",
            "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
            "sha256": _digest("qwen3-0.6b-demo-model"),
        },
        "adapter": adapter,
        "runtime": {"id": "synthetic-demo", "revision": "1.0.0", "sha256": _digest("synthetic-runtime")},
        "tools": {"id": "hermes-demo-tools-v1", "sha256": _digest("hermes-demo-tools")},
        "environment": {"id": "offline-demo-v1", "sha256": _digest("offline-demo-environment")},
    }


def _proposal(proposal_id: str, hypothesis: str, mutations: dict[str, Any]) -> dict[str, Any]:
    return {
        "proposal_id": proposal_id,
        "hypothesis": hypothesis,
        "mutations": mutations,
        "estimated_cost_usd": 0.02,
        "estimated_duration_seconds": 2.0,
    }


def _report(search: dict[str, Any], evidence: dict[str, Any], handoff: dict[str, Any]) -> str:
    rows = []
    for ref in search["trials"]:
        rows.append(f"| {ref['trial_index']} | {ref['trial_id']} | {ref['outcome']} | {ref['status']} |")
    return "\n".join(
        [
            "# Governed Autoresearch-Style LoRA Recipe Search",
            "",
            "> Synthetic offline contract demonstration. No model was downloaded, trained, served, or promoted.",
            "",
            "## Search result",
            "",
            f"- Trials: {search['trial_count']}",
            f"- Kept (including baseline): {search['kept_trial_count']}",
            f"- Discarded: {search['discarded_trial_count']}",
            f"- Champion: `{search['champion']['trial_id']}`",
            f"- Champion development metric: {search['champion']['development_metric']:.3f}",
            f"- Held-out artifacts used during search: {search['heldout_access']['artifact_count']}",
            "",
            "| Index | Trial | Decision | Execution |",
            "| ---: | --- | --- | --- |",
            *rows,
            "",
            "## Promotion boundary",
            "",
            f"- Repeated held-out promotion evidence passed: `{evidence['promotion_ready']}`",
            f"- Paired observations: {evidence['paired_observation_count']}",
            f"- Candidate identity binding matched: `{handoff['candidate_binding']['matched']}`",
            f"- Governance handoff readiness: `{handoff['readiness']}`",
            "- Promotion applied: `false`",
            "",
        ]
    )


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/autoresearch_lora_optimizer"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = run_demo(args.out)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
