#!/usr/bin/env python3
"""Governed offline autoresearch controller for runtime-adapter LoRA recipes.

The controller wraps the existing Flight Recorder recipe-search contract around
the local trainer and runtime-adapter evaluator. It is safe by default:
candidate search records proposals and budget decisions, but it does not update
model weights unless ``--execute-local-training`` is supplied. Sealed-final
evaluation is a separate one-shot ``--finalize-sealed`` path.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.lora_recipe_search import (  # noqa: E402
    HELDOUT_TAGS,
    build_search_plan,
    run_search,
    validate_search_result,
)
from flightrecorder.path_safety import path_has_symlink_component  # noqa: E402

SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_runtime_adapter_candidates import (  # noqa: E402
    CHAT_TEMPLATE_SHA256,
    DEFAULT_THRESHOLDS,
    MODEL_ID,
    MODEL_REVISION,
    TOKENIZER_REVISION,
    adapter_directory_fingerprint,
    canonical_sha256,
    sha256_file,
    validate_candidate_identity,
    validate_evaluation_report,
)


TRAINER = SCRIPTS / "train_agentic_lora.py"
EVALUATOR = SCRIPTS / "evaluate_runtime_adapter_candidates.py"
CAMPAIGN_RECORD = "runtime_adapter_autoresearch_campaign.json"
SEALED_RECEIPT = "sealed_final_receipt.json"
SEALED_REPORT = "sealed_final_evaluation.json"
FORBIDDEN_SEARCH_PATH_TOKENS = HELDOUT_TAGS | {"sealed", "sealed_final"}
ALLOWED_EVALUATION_SCOPES = frozenset(
    {"browser", "database", "code_terminal", "generalist", "shared"}
)
RECIPE_MUTATION_QUEUE = (
    (
        "focus-first-action",
        "Increase exact tool-argument supervision while retaining every governed full trajectory.",
        {"action_turn_repeats": 4},
    ),
    (
        "rank-32",
        "Increase adapter rank to widen the tool-call representation subspace.",
        {"lora_r": 32, "lora_alpha": 64},
    ),
    (
        "lower-dropout",
        "Reduce dropout to preserve exact argument formatting on small corpora.",
        {"lora_dropout": 0.02},
    ),
    (
        "longer-development-budget",
        "Give the current incumbent more optimizer steps within the trial budget.",
        {"max_steps": 240},
    ),
    (
        "slower-sft-rate",
        "Lower SFT learning rate to reduce tool-call schema regressions.",
        {"sft_learning_rate": 0.00005},
    ),
    (
        "short-context-fast-pass",
        "Reduce sequence length to favor fast exact tool-call convergence.",
        {"max_length": 768, "max_steps": 160},
    ),
)


@dataclass(frozen=True)
class RunnerPaths:
    root: Path
    attempts: Path
    plan: Path
    result: Path
    campaign_record: Path


class DeterministicQueueProposer:
    """Deterministic bounded recipe queue for development-set search."""

    def __init__(self, proposals: Iterable[tuple[str, str, dict[str, Any]]] = RECIPE_MUTATION_QUEUE) -> None:
        self._proposals = list(proposals)
        self._cursor = 0

    def propose(self, state: dict[str, Any]) -> dict[str, Any] | None:
        if state["remaining_budget"]["trials"] <= 0 or self._cursor >= len(self._proposals):
            return None
        proposal_id, hypothesis, mutations = self._proposals[self._cursor]
        self._cursor += 1
        return {
            "proposal_id": proposal_id,
            "hypothesis": hypothesis,
            "mutations": dict(mutations),
            "estimated_cost_usd": state["plan_budget"]["per_trial_cost_ceiling_usd"],
            "estimated_duration_seconds": state["plan_budget"]["per_trial_duration_ceiling_seconds"],
        }


class RuntimeAdapterDevelopmentEvaluator:
    """Import local development outcomes into the RecipeEvaluator contract."""

    def __init__(self, args: argparse.Namespace, paths: RunnerPaths) -> None:
        self.args = args
        self.paths = paths
        self.candidate_records: list[dict[str, Any]] = []

    def evaluate(
        self,
        recipe: dict[str, Any],
        *,
        trial_id: str,
        development_suite_path: Path,
    ) -> dict[str, Any]:
        attempt_dir = self.paths.attempts / trial_id
        if attempt_dir.exists():
            raise RuntimeError(f"attempt directory already exists: {attempt_dir}")
        attempt_dir.mkdir(parents=True)
        proposal_path = attempt_dir / "proposal_launch_record.json"
        launch = self._launch_record(recipe, trial_id, development_suite_path, attempt_dir)
        atomic_write_json(proposal_path, launch)

        if not self.args.execute_local_training:
            record = {
                **launch,
                "status": "blocked",
                "reason": "local model-weight training requires --execute-local-training",
                "proposal_launch_record": str(proposal_path),
            }
            self.candidate_records.append(record)
            self._write_campaign_record("search_in_progress")
            return _recipe_evaluation(
                status="crashed",
                primary_metric=None,
                critical_failures=0,
                cost=0.0,
                duration=0.0,
                candidate_sha="",
                diagnostics=[
                    "local training was not executed",
                    f"proposal launch record persisted before any subprocess: {proposal_path}",
                ],
                side_effects=False,
                weights_updated=False,
            )

        training_result_path = attempt_dir / "training" / f"{recipe['mode']}_result.json"
        _run_subprocess(launch["trainer_command"], cwd=ROOT)
        if not training_result_path.is_file():
            raise RuntimeError(f"trainer did not produce expected result: {training_result_path}")

        candidate = build_candidate_json(
            recipe=recipe,
            trial_id=trial_id,
            attempt_dir=attempt_dir,
            training_result_path=training_result_path,
            args=self.args,
        )
        candidate_path = attempt_dir / "candidate.json"
        atomic_write_json(candidate_path, {"candidates": [candidate]})

        development_report_path = attempt_dir / "development_evaluation.json"
        observations_path = attempt_dir / "development_observations.jsonl"
        evaluator_command = build_evaluator_command(
            rows_jsonl=self.args.development_jsonl,
            candidates_path=candidate_path,
            out_path=development_report_path,
            observations_out=observations_path,
            device=self.args.device,
            model_identity=model_identity_from_args(self.args),
            evaluation_split="development",
        )
        _run_subprocess(evaluator_command, cwd=ROOT)
        report = load_json_object(development_report_path)
        outcome = development_outcome(report, candidate["candidate_id"])
        candidate_record = {
            **launch,
            "status": "evaluated",
            "candidate": candidate,
            "candidate_path": str(candidate_path),
            "development_report": str(development_report_path),
            "development_report_sha256": sha256_file(development_report_path),
            "development_observations": str(observations_path),
            "development_outcome": outcome,
            "trainer_result": str(training_result_path),
            "trainer_result_sha256": sha256_file(training_result_path),
        }
        self.candidate_records.append(candidate_record)
        self._write_campaign_record("search_in_progress")
        return _recipe_evaluation(
            status="completed",
            primary_metric=outcome["development_quality_score"],
            critical_failures=outcome["critical_unsafe_call_count"],
            cost=0.0,
            duration=float(self.args.per_trial_duration_ceiling_seconds),
            candidate_sha=candidate["candidate_identity_sha256"],
            diagnostics=[
                "development-only local inference completed",
                f"overall_pass_rate={outcome['overall_pass_rate']}",
                f"critical_unsafe_call_count={outcome['critical_unsafe_call_count']}",
            ],
            side_effects=True,
            weights_updated=True,
        )

    def _launch_record(
        self,
        recipe: dict[str, Any],
        trial_id: str,
        development_suite_path: Path,
        attempt_dir: Path,
    ) -> dict[str, Any]:
        trainer_command = build_trainer_command(
            recipe=recipe,
            trial_id=trial_id,
            attempt_dir=attempt_dir,
            args=self.args,
        )
        return {
            "schema_version": "hfr.runtime_adapter_autoresearch_launch.v1",
            "created_at": utc_now(),
            "campaign_id": self.args.campaign_id,
            "trial_id": trial_id,
            "recipe": recipe,
            "recipe_sha256": canonical_sha256(recipe),
            "attempt_dir": str(attempt_dir),
            "development_suite": str(development_suite_path),
            "development_jsonl": str(self.args.development_jsonl),
            "sealed_inputs_accessed": False,
            "execute_local_training": bool(self.args.execute_local_training),
            "trainer_command": trainer_command,
            "budget": {
                "trial_training_seconds": self.args.trial_training_seconds,
                "per_trial_duration_ceiling_seconds": self.args.per_trial_duration_ceiling_seconds,
                "per_trial_cost_ceiling_usd": self.args.per_trial_cost_ceiling_usd,
            },
            "offline_constraints": {
                "local_training": "--local-training" in trainer_command,
                "execute_flag_present": "--execute-local-training" in trainer_command,
                "push_to_hub": "--push-to-hub" in trainer_command,
                "trackio_disabled": "--disable-trackio" in trainer_command,
                "registered_inputs_required": "--require-registered-inputs" in trainer_command,
            },
        }

    def _write_campaign_record(self, status: str) -> None:
        atomic_write_json(
            self.paths.campaign_record,
            campaign_record(
                args=self.args,
                paths=self.paths,
                status=status,
                candidate_records=self.candidate_records,
            ),
        )


def build_trainer_command(
    *,
    recipe: dict[str, Any],
    trial_id: str,
    attempt_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        str(TRAINER),
        "--mode",
        str(recipe["mode"]),
        "--experiment-dir",
        str(args.experiment_dir),
        "--model",
        str(args.model),
        "--model-revision",
        str(args.model_revision),
        "--tokenizer-revision",
        str(args.tokenizer_revision),
        "--expected-chat-template-sha256",
        str(args.chat_template_sha256),
        "--model-manifest",
        str(args.model_manifest),
        "--dataset-manifest",
        str(args.dataset_manifest),
        "--require-registered-inputs",
        "--output-dir",
        str(attempt_dir / "training"),
        "--local-training",
        "--local-model-path",
        str(args.local_model_path),
        "--execute-local-training",
        "--disable-trackio",
        "--device",
        str(args.device),
        "--max-training-seconds",
        str(args.trial_training_seconds),
        "--sft-learning-rate",
        str(recipe["sft_learning_rate"]),
        "--dpo-learning-rate",
        str(recipe["dpo_learning_rate"]),
        "--batch-size",
        str(recipe["batch_size"]),
        "--gradient-accumulation-steps",
        str(recipe["gradient_accumulation_steps"]),
        "--max-steps",
        str(recipe["max_steps"]),
        "--max-length",
        str(recipe["max_length"]),
        "--action-turn-repeats",
        str(recipe["action_turn_repeats"]),
        "--lora-r",
        str(recipe["lora_r"]),
        "--lora-alpha",
        str(recipe["lora_alpha"]),
        "--lora-dropout",
        str(recipe["lora_dropout"]),
        "--seed",
        str(recipe["seed"]),
        "--data-seed",
        str(recipe["data_seed"]),
        "--run-name-prefix",
        f"{args.campaign_id}-{trial_id}",
    ]
    if args.limit:
        command.extend(["--limit", str(args.limit)])
    for task_family in args.task_family:
        command.extend(["--task-family", str(task_family)])
    if args.gradient_checkpointing:
        command.append("--gradient-checkpointing")
    return command


def build_evaluator_command(
    *,
    rows_jsonl: Path,
    candidates_path: Path,
    out_path: Path,
    observations_out: Path,
    device: str,
    model_identity: dict[str, str] | None = None,
    evaluation_split: str = "sealed_final",
) -> list[str]:
    if evaluation_split not in {"development", "sealed_final"}:
        raise ValueError("evaluation_split must be development or sealed_final")
    identity = model_identity or {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "chat_template_sha256": CHAT_TEMPLATE_SHA256,
    }
    return [
        sys.executable,
        str(EVALUATOR),
        "--heldout-jsonl",
        str(rows_jsonl),
        "--candidates",
        str(candidates_path),
        "--out",
        str(out_path),
        "--observations-out",
        str(observations_out),
        "--run-inference",
        "--device",
        device if device in {"cpu", "mps"} else "cpu",
        "--overall-min-pass-rate",
        str(DEFAULT_THRESHOLDS["overall_min_pass_rate"]),
        "--single-domain-min-pass-rate",
        str(DEFAULT_THRESHOLDS["single_domain_min_pass_rate"]),
        "--cross-domain-min-pass-rate",
        str(DEFAULT_THRESHOLDS["cross_domain_min_pass_rate"]),
        "--safety-min-pass-rate",
        str(DEFAULT_THRESHOLDS["safety_min_pass_rate"]),
        "--write-denial-min-pass-rate",
        str(DEFAULT_THRESHOLDS["write_denial_min_pass_rate"]),
        "--failure-recovery-min-pass-rate",
        str(DEFAULT_THRESHOLDS["failure_recovery_min_pass_rate"]),
        "--expected-model-id",
        identity["model_id"],
        "--expected-model-revision",
        identity["model_revision"],
        "--expected-tokenizer-revision",
        identity["tokenizer_revision"],
        "--expected-chat-template-sha256",
        identity["chat_template_sha256"],
        "--evaluation-split",
        evaluation_split,
    ]


def build_candidate_json(
    *,
    recipe: dict[str, Any],
    trial_id: str,
    attempt_dir: Path,
    training_result_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model_identity = model_identity_from_args(args)
    adapter_dir = adapter_dir_for_recipe(attempt_dir / "training", recipe["mode"])
    adapter_fingerprint = adapter_directory_fingerprint(adapter_dir)
    training_sha = sha256_file(training_result_path)
    identity = {
        "campaign_id": args.campaign_id,
        "trial_id": trial_id,
        "recipe_sha256": canonical_sha256(recipe),
        "adapter_sha256": adapter_fingerprint["sha256"],
        "training_result_sha256": training_sha,
        "base_model": model_identity["model_id"],
        "base_revision": model_identity["model_revision"],
        "tokenizer_revision": model_identity["tokenizer_revision"],
        "chat_template_sha256": model_identity["chat_template_sha256"],
        "training_task_families": list(args.task_family),
        "evaluation_scopes": list(args.evaluation_scope) or ["*"],
    }
    candidate_id = "runtime-adapter-" + canonical_sha256(identity)[:16]
    candidate = {
        "candidate_id": candidate_id,
        "id": candidate_id,
        "type": "lora_adapter",
        "status": "succeeded",
        "scope": "runtime_adapter_router",
        "base_model": model_identity["model_id"],
        "base_revision": model_identity["model_revision"],
        "tokenizer_revision": model_identity["tokenizer_revision"],
        "chat_template_sha256": model_identity["chat_template_sha256"],
        "local_model_path": str(args.local_model_path),
        "adapter_dir": str(adapter_dir),
        "adapter_sha256": adapter_fingerprint["sha256"],
        "adapter_directory_sha256": adapter_fingerprint["sha256"],
        "training_result_path": str(training_result_path),
        "training_result_sha256": training_sha,
        "recipe": recipe,
        "recipe_sha256": canonical_sha256(recipe),
        "training_task_families": list(args.task_family),
        "evaluation_scopes": list(args.evaluation_scope) or ["*"],
    }
    candidate["candidate_identity_sha256"] = canonical_sha256(candidate)
    return candidate


def adapter_dir_for_recipe(training_dir: Path, mode: str) -> Path:
    names = {
        "trace_sft": "trace_sft_adapter",
        "fr_sft": "fr_sft_adapter",
        "fr_action_sft": "fr_action_sft_adapter",
        "fr_dpo": "fr_dpo_adapter",
        "fr_sft_dpo": "fr_sft_dpo_adapter",
    }
    return training_dir / names.get(mode, f"{mode}_adapter")


def development_outcome(report: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    matching = [
        item
        for item in report.get("candidate_reports", [])
        if isinstance(item, dict) and item.get("candidate_id") == candidate_id
    ]
    if len(matching) != 1:
        raise RuntimeError(f"development report did not contain exactly one candidate {candidate_id!r}")
    candidate = matching[0]
    metrics = candidate.get("metrics", {})
    overall = metrics.get("overall", {}) if isinstance(metrics, dict) else {}
    safety = metrics.get("safety", {}) if isinstance(metrics, dict) else {}
    write_denial = metrics.get("write_denial", {}) if isinstance(metrics, dict) else {}
    failure_recovery = metrics.get("failure_recovery", {}) if isinstance(metrics, dict) else {}
    check_rates = metrics.get("check_pass_rates", {}) if isinstance(metrics, dict) else {}
    tool_calls_exact = check_rates.get("tool_calls_exact_order", {}) if isinstance(check_rates, dict) else {}
    tool_calls_functional = (
        check_rates.get("tool_calls_functional_order", tool_calls_exact)
        if isinstance(check_rates, dict)
        else {}
    )
    final_answer_exact = check_rates.get("final_answer_exact", {}) if isinstance(check_rates, dict) else {}
    overall_pass_rate = _metric_pass_rate(overall)
    safety_pass_rate = _metric_pass_rate(safety)
    write_denial_pass_rate = _metric_pass_rate(write_denial)
    failure_recovery_pass_rate = _metric_pass_rate(failure_recovery)
    tool_calls_exact_rate = _metric_pass_rate(tool_calls_exact)
    tool_calls_functional_rate = _metric_pass_rate(tool_calls_functional)
    final_answer_exact_rate = _metric_pass_rate(final_answer_exact)
    critical_unsafe = int(metrics.get("critical_safety_failures") or 0)
    # Development selection needs a smooth functionality signal while remaining
    # dominated by safety. Each critical unsafe call costs a full point, whereas
    # all positive quality components together are bounded by one point.
    quality = round(
        0.35 * overall_pass_rate
        + 0.15 * tool_calls_functional_rate
        + 0.15 * final_answer_exact_rate
        + 0.15 * safety_pass_rate
        + 0.10 * write_denial_pass_rate
        + 0.10 * failure_recovery_pass_rate
        - critical_unsafe,
        8,
    )
    return {
        "development_quality_score": quality,
        "overall_pass_rate": overall_pass_rate,
        "safety_pass_rate": safety_pass_rate,
        "write_denial_pass_rate": write_denial_pass_rate,
        "failure_recovery_pass_rate": failure_recovery_pass_rate,
        "tool_calls_exact_rate": tool_calls_exact_rate,
        "tool_calls_functional_rate": tool_calls_functional_rate,
        "final_answer_exact_rate": final_answer_exact_rate,
        "critical_unsafe_call_count": critical_unsafe,
        "promotion_eligible_on_development": bool(candidate.get("promotion_eligible")),
        "blocking_reasons": candidate.get("blocking_reasons", []),
    }


def _metric_pass_rate(metric: dict[str, Any]) -> float:
    value = metric.get("pass_rate")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    paths = runner_paths(args.campaign_dir)
    if not args.resume:
        existing_state = [
            path
            for path in (paths.plan, paths.result, paths.campaign_record)
            if path.exists()
        ]
        if paths.attempts.is_dir() and any(paths.attempts.iterdir()):
            existing_state.append(paths.attempts)
        if existing_state:
            joined = ", ".join(str(path) for path in existing_state)
            raise SystemExit(
                "campaign already contains autoresearch state; use --resume or a new "
                f"--campaign-dir: {joined}"
            )
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.attempts.mkdir(parents=True, exist_ok=True)
    validate_search_inputs(args)
    args.development_suite = materialize_development_suite(args.development_suite, paths.root)
    existing_plan = load_json_object(paths.plan) if paths.plan.is_file() else None
    plan = build_search_plan(
        campaign_id=args.campaign_id,
        objective=args.objective,
        development_suite_path=args.development_suite,
        base_recipe=base_recipe_from_args(args),
        mutable_fields=[
            "sft_learning_rate",
            "dpo_learning_rate",
            "batch_size",
            "gradient_accumulation_steps",
            "max_steps",
            "max_length",
            "action_turn_repeats",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
        ],
        budget={
            "max_trials": args.max_trials,
            "max_cost_usd": args.campaign_max_cost_usd,
            "max_duration_seconds": args.campaign_max_duration_seconds,
            "per_trial_cost_ceiling_usd": args.per_trial_cost_ceiling_usd,
            "per_trial_duration_ceiling_seconds": args.per_trial_duration_ceiling_seconds,
        },
        out_path=paths.plan,
        primary_metric="development_quality_score",
        direction="maximize",
        minimum_delta=args.minimum_delta,
        max_critical_failures=args.development_max_critical_failures,
        created_at=existing_plan.get("created_at") if existing_plan else None,
    )
    if existing_plan is not None:
        if existing_plan != plan:
            raise SystemExit(f"refusing to overwrite different immutable search plan: {paths.plan}")
    else:
        atomic_write_json(paths.plan, plan)

    evaluator = RuntimeAdapterDevelopmentEvaluator(args, paths)
    if args.resume and paths.campaign_record.is_file():
        existing_campaign = load_json_object(paths.campaign_record)
        if existing_campaign.get("campaign_id") != args.campaign_id:
            raise SystemExit("existing campaign record belongs to a different campaign_id")
        existing_records = existing_campaign.get("candidate_records")
        if not isinstance(existing_records, list) or not all(
            isinstance(record, dict) for record in existing_records
        ):
            raise SystemExit("existing campaign record has invalid candidate_records")
        evaluator.candidate_records.extend(existing_records)
    result = run_search_maybe_resume(
        plan_path=paths.plan,
        out_path=paths.result,
        proposer=DeterministicQueueProposer(),
        evaluator=evaluator,
        resume=args.resume,
    )
    validation = validate_search_result(paths.result)
    if not validation["passed"]:
        raise RuntimeError("search result failed validation: " + "; ".join(validation["errors"]))
    record = campaign_record(
        args=args,
        paths=paths,
        status="search_complete",
        candidate_records=evaluator.candidate_records,
        search_result=result,
    )
    atomic_write_json(paths.campaign_record, record)
    return record


def materialize_development_suite(source: Path, campaign_root: Path) -> Path:
    destination = campaign_root / "development_suite_manifest.json"
    source_payload = load_json_object(source)
    if destination.exists():
        existing = load_json_object(destination)
        if existing != source_payload:
            raise SystemExit(f"refusing to overwrite different immutable development suite: {destination}")
    else:
        atomic_write_json(destination, source_payload)
    return destination


def finalize_sealed(args: argparse.Namespace) -> dict[str, Any]:
    paths = runner_paths(args.campaign_dir)
    receipt_path = paths.root / SEALED_RECEIPT
    if receipt_path.exists():
        raise SystemExit(f"sealed final receipt already exists; refusing second sealed access: {receipt_path}")
    if args.sealed_jsonl is None:
        raise SystemExit("--finalize-sealed requires --sealed-jsonl")
    if not paths.result.is_file():
        raise SystemExit(f"search result is missing: {paths.result}")
    campaign = load_json_object(paths.campaign_record)
    search = load_json_object(paths.result)
    champion = search.get("champion") if isinstance(search.get("champion"), dict) else {}
    if not champion:
        raise SystemExit("search result has no champion to finalize")
    candidate_record = champion_candidate_record(campaign, champion)
    candidate = candidate_record.get("candidate")
    if not isinstance(candidate, dict):
        raise SystemExit("champion has no persisted candidate JSON")
    validate_development_champion_before_sealed(
        paths=paths,
        candidate_record=candidate_record,
        candidate=candidate,
    )
    champion_model_identity = model_identity_from_candidate(candidate)
    candidate_validation = validate_candidate_identity(
        candidate, expected_model_identity=champion_model_identity
    )
    if candidate_validation["status"] != "eligible":
        raise SystemExit(
            "champion candidate integrity failed before sealed access: "
            + ", ".join(candidate_validation["reasons"])
        )
    champion_candidate_path = paths.root / "sealed_champion_candidate.json"
    atomic_write_json(champion_candidate_path, {"candidates": [candidate]})
    report_path = paths.root / SEALED_REPORT
    observations_path = paths.root / "sealed_final_observations.jsonl"
    command = build_evaluator_command(
        rows_jsonl=args.sealed_jsonl,
        candidates_path=champion_candidate_path,
        out_path=report_path,
        observations_out=observations_path,
        device=args.device,
        model_identity=champion_model_identity,
        evaluation_split="sealed_final",
    )
    receipt = {
        "schema_version": "hfr.runtime_adapter_autoresearch_sealed_receipt.v1",
        "created_at": utc_now(),
        "status": "started",
        "campaign_id": str(campaign.get("campaign_id") or ""),
        "sealed_jsonl": str(args.sealed_jsonl),
        "champion_trial_id": champion.get("trial_id"),
        "champion_candidate_id": candidate.get("candidate_id"),
        "candidate_identity_sha256": candidate.get("candidate_identity_sha256"),
        "model_identity": champion_model_identity,
        "sealed_report": str(report_path),
        "sealed_report_sha256": "",
        "sealed_observations": str(observations_path),
        "evaluator_command": command,
        "sealed_access": {
            "during_search": False,
            "finalize_sealed": True,
            "one_time_receipt": True,
        },
    }
    # Persist the one-shot boundary before the evaluator can read sealed rows.
    # A crashed evaluation therefore remains closed to a second access.
    atomic_write_json(receipt_path, receipt)
    _run_subprocess(command, cwd=ROOT)
    receipt["status"] = "completed"
    receipt["sealed_report_sha256"] = sha256_file(report_path)
    atomic_write_json(receipt_path, receipt)
    return receipt


def run_search_maybe_resume(
    *,
    plan_path: Path,
    out_path: Path,
    proposer: DeterministicQueueProposer,
    evaluator: RuntimeAdapterDevelopmentEvaluator,
    resume: bool,
) -> dict[str, Any]:
    kwargs = {
        "plan_path": plan_path,
        "out_path": out_path,
        "proposer": proposer,
        "evaluator": evaluator,
    }
    if resume and "resume" in inspect.signature(run_search).parameters:
        kwargs["resume"] = True
    return run_search(**kwargs)


def validate_search_inputs(args: argparse.Namespace) -> None:
    invalid_task_families = [
        value
        for value in args.task_family
        if not isinstance(value, str)
        or not value.strip()
        or not value.strip().endswith("_train")
    ]
    if invalid_task_families:
        raise SystemExit(
            "--task-family values must be non-empty registered training families ending in _train"
        )
    invalid_scopes = sorted(set(args.evaluation_scope) - ALLOWED_EVALUATION_SCOPES)
    if invalid_scopes:
        raise SystemExit(
            "--evaluation-scope contains unsupported values: " + ", ".join(invalid_scopes)
        )
    for role, path in (
        ("development suite", args.development_suite),
        ("development jsonl", args.development_jsonl),
    ):
        if path is None:
            raise SystemExit(f"--{role.replace(' ', '-')} is required")
        if not path.is_file():
            raise SystemExit(f"{role} is missing: {path}")
        reject_search_path(role, path)
    if args.sealed_jsonl is not None and args.development_jsonl.resolve() == args.sealed_jsonl.resolve():
        raise SystemExit("development and sealed paths must be different")
    if args.execute_local_training:
        for role, path in (
            ("model manifest", args.model_manifest),
            ("dataset manifest", args.dataset_manifest),
            ("local model path", args.local_model_path),
        ):
            if path is None:
                raise SystemExit(f"--{role.replace(' ', '-')} is required with --execute-local-training")
            if not path.exists():
                raise SystemExit(f"{role} is missing: {path}")
        manifest = load_json_object(args.model_manifest)
        model_identity = model_identity_from_args(args)
        declared = {
            "model_id": str(manifest.get("model_id") or ""),
            "model_revision": str(
                (manifest.get("source") or {}).get("revision") or ""
            ),
            "tokenizer_revision": str(
                ((manifest.get("compatibility") or {}).get("tokenizer") or {}).get(
                    "revision"
                )
                or ""
            ),
            "chat_template_sha256": str(
                ((manifest.get("compatibility") or {}).get("chat_template") or {}).get(
                    "sha256"
                )
                or ""
            ),
        }
        if declared != model_identity:
            raise SystemExit(
                "model manifest identity does not match the governed campaign identity"
            )


def model_identity_from_args(args: argparse.Namespace) -> dict[str, str]:
    identity = {
        "model_id": str(getattr(args, "model", MODEL_ID)),
        "model_revision": str(getattr(args, "model_revision", MODEL_REVISION)),
        "tokenizer_revision": str(
            getattr(args, "tokenizer_revision", TOKENIZER_REVISION)
        ),
        "chat_template_sha256": str(
            getattr(args, "chat_template_sha256", CHAT_TEMPLATE_SHA256)
        ),
    }
    if not all(identity.values()):
        raise SystemExit("governed campaign model identity fields must be non-empty")
    digest = identity["chat_template_sha256"]
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise SystemExit("governed campaign chat-template identity must be lowercase SHA-256")
    return identity


def model_identity_from_candidate(candidate: dict[str, Any]) -> dict[str, str]:
    identity = {
        "model_id": str(candidate.get("base_model") or ""),
        "model_revision": str(candidate.get("base_revision") or ""),
        "tokenizer_revision": str(candidate.get("tokenizer_revision") or ""),
        "chat_template_sha256": str(candidate.get("chat_template_sha256") or ""),
    }
    if not all(identity.values()):
        raise SystemExit("champion candidate has an incomplete model identity")
    digest = identity["chat_template_sha256"]
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise SystemExit("champion candidate has an invalid chat-template identity")
    return identity


def reject_search_path(role: str, path: Path) -> None:
    tokens = {
        token
        for part in path.parts
        for token in part.lower().replace("-", "_").replace(".", "_").split("_")
        if token
    }
    forbidden = sorted(tokens & FORBIDDEN_SEARCH_PATH_TOKENS)
    if forbidden:
        raise SystemExit(f"{role} path appears to reference held-out/sealed data: {path}")


def base_recipe_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "mode": args.mode,
        "sft_learning_rate": args.sft_learning_rate,
        "dpo_learning_rate": args.dpo_learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_steps": args.max_steps,
        "max_length": args.max_length,
        "action_turn_repeats": args.action_turn_repeats,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "seed": args.seed,
        "data_seed": args.data_seed,
    }


def runner_paths(root: Path) -> RunnerPaths:
    return RunnerPaths(
        root=root,
        attempts=root / "attempts",
        plan=root / "search_plan.json",
        result=root / "search_result.json",
        campaign_record=root / CAMPAIGN_RECORD,
    )


def campaign_record(
    *,
    args: argparse.Namespace,
    paths: RunnerPaths,
    status: str,
    candidate_records: list[dict[str, Any]],
    search_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "hfr.runtime_adapter_autoresearch_campaign.v1",
        "created_at": utc_now(),
        "campaign_id": args.campaign_id,
        "status": status,
        "campaign_dir": str(paths.root),
        "search_plan": str(paths.plan),
        "search_result": str(paths.result),
        "execute_local_training": bool(args.execute_local_training),
        "development_suite": str(args.development_suite),
        "development_jsonl": str(args.development_jsonl),
        "training_task_families": list(args.task_family),
        "evaluation_scopes": list(args.evaluation_scope) or ["*"],
        "sealed_jsonl_known_to_search": False,
        "model_identity": model_identity_from_args(args),
        "resume_requested": bool(args.resume),
        "run_search_resume_supported": "resume" in inspect.signature(run_search).parameters,
        "candidate_records": candidate_records,
        "search_result_summary": {
            "passed": search_result.get("passed"),
            "readiness": search_result.get("readiness"),
            "champion": search_result.get("champion"),
            "trial_count": search_result.get("trial_count"),
        }
        if search_result
        else {},
    }


def champion_candidate_record(campaign: dict[str, Any], champion: dict[str, Any]) -> dict[str, Any]:
    champion_sha = str(champion.get("candidate_identity_sha256") or "")
    records = campaign.get("candidate_records", [])
    if not isinstance(records, list):
        raise SystemExit("campaign record has invalid candidate_records")
    matches = [
        record
        for record in records
        if isinstance(record, dict)
        and isinstance(record.get("candidate"), dict)
        and record["candidate"].get("candidate_identity_sha256") == champion_sha
    ]
    if len(matches) != 1:
        raise SystemExit("could not resolve exactly one persisted champion candidate")
    match = matches[0]
    candidate = match["candidate"]
    candidate_payload = dict(candidate)
    claimed_sha = str(candidate_payload.pop("candidate_identity_sha256", ""))
    if not claimed_sha or canonical_sha256(candidate_payload) != claimed_sha:
        raise SystemExit("persisted champion candidate content hash mismatch")
    recipe = candidate.get("recipe")
    recipe_sha = str(candidate.get("recipe_sha256") or "")
    if (
        not isinstance(recipe, dict)
        or canonical_sha256(recipe) != recipe_sha
        or recipe_sha != str(champion.get("recipe_sha256") or "")
    ):
        raise SystemExit("persisted champion candidate recipe does not match search champion")
    return match


def validate_development_champion_before_sealed(
    *,
    paths: RunnerPaths,
    candidate_record: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Replay development eligibility from immutable local evidence."""

    report_value = candidate_record.get("development_report")
    if not isinstance(report_value, str) or not report_value:
        raise SystemExit("champion has no persisted development report")
    resolved_report = resolve_campaign_attempt_artifact(
        report_value, paths=paths
    )
    if candidate_record.get("development_report_sha256") != sha256_file(
        resolved_report
    ):
        raise SystemExit("champion development report hash mismatch")
    report = load_json_object(resolved_report)
    semantic_errors = validate_evaluation_report(report)
    if semantic_errors:
        raise SystemExit(
            "champion development report failed semantic validation: "
            + "; ".join(semantic_errors)
        )
    if report.get("heldout", {}).get("split") != "development":
        raise SystemExit("champion development report does not declare development split")
    candidate_id = str(candidate.get("candidate_id") or "")
    candidate_reports = [
        item
        for item in report.get("candidate_reports", [])
        if isinstance(item, dict) and item.get("candidate_id") == candidate_id
    ]
    if len(candidate_reports) != 1:
        raise SystemExit("champion development report does not contain exactly one champion result")
    candidate_report = candidate_reports[0]
    identity = (
        candidate_report.get("identity", {})
        if isinstance(candidate_report.get("identity"), dict)
        else {}
    )
    adapter = identity.get("adapter", {}) if isinstance(identity.get("adapter"), dict) else {}
    if adapter.get("sha256") != candidate.get("adapter_sha256"):
        raise SystemExit("champion development report adapter hash mismatch")
    replayed_outcome = development_outcome(report, candidate_id)
    if candidate_record.get("development_outcome") != replayed_outcome:
        raise SystemExit("champion development outcome summary does not replay from report")
    if replayed_outcome["critical_unsafe_call_count"] != 0:
        raise SystemExit(
            "champion has critical unsafe calls on development; sealed final remains closed"
        )
    if replayed_outcome["promotion_eligible_on_development"] is not True:
        raise SystemExit(
            "champion is not promotion eligible on development; sealed final remains closed"
        )
    return replayed_outcome


def resolve_campaign_attempt_artifact(
    value: str, *, paths: RunnerPaths
) -> Path:
    """Resolve an absolute, repo-relative, or campaign-relative attempt artifact."""

    supplied = Path(value)
    if supplied.is_absolute():
        candidates = [supplied]
    else:
        candidates = [ROOT / supplied, paths.root / supplied]
        if not (paths.root / supplied).is_absolute():
            candidates[-1] = ROOT / candidates[-1]

    attempts = paths.attempts
    if not attempts.is_absolute():
        attempts = ROOT / attempts
    try:
        attempts = attempts.resolve(strict=True)
    except (OSError, RuntimeError):
        raise SystemExit(
            "campaign attempts directory is not a regular directory"
        ) from None

    matches: dict[str, Path] = {}
    for candidate in candidates:
        if path_has_symlink_component(candidate, include_leaf=True):
            continue
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(attempts)
        except (OSError, RuntimeError, ValueError):
            continue
        if resolved.is_file():
            matches[str(resolved)] = resolved
    if len(matches) != 1:
        raise SystemExit(
            "champion development report is not a regular campaign-attempt artifact"
        )
    return next(iter(matches.values()))


def _recipe_evaluation(
    *,
    status: str,
    primary_metric: float | None,
    critical_failures: int,
    cost: float,
    duration: float,
    candidate_sha: str,
    diagnostics: list[str],
    side_effects: bool,
    weights_updated: bool,
) -> dict[str, Any]:
    return {
        "status": status,
        "primary_metric": primary_metric,
        "critical_failures": critical_failures,
        "cost_usd": cost,
        "duration_seconds": duration,
        "candidate_identity_sha256": candidate_sha,
        "diagnostics": diagnostics,
        "execution_mode": "imported_external_result" if side_effects else "simulation",
        "external_side_effects_observed": side_effects,
        "model_weights_updated_externally": weights_updated,
    }


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        tmp_path = Path(tmp_name)
        if tmp_path.exists():
            tmp_path.unlink()


def _run_subprocess(command: list[str], *, cwd: Path) -> None:
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with exit {completed.returncode}: {command}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--campaign-id", default="runtime-adapter-autoresearch")
    parser.add_argument("--objective", default="Improve runtime-adapter tool-calling quality on development evidence.")
    parser.add_argument("--development-suite", type=Path)
    parser.add_argument("--development-jsonl", type=Path)
    parser.add_argument("--sealed-jsonl", type=Path)
    parser.add_argument("--finalize-sealed", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--experiment-dir", type=Path, default=Path("examples/case_studies/runtime_adapter_router/training"))
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--tokenizer-revision", default=TOKENIZER_REVISION)
    parser.add_argument("--chat-template-sha256", default=CHAT_TEMPLATE_SHA256)
    parser.add_argument("--model-manifest", type=Path)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--local-model-path", type=Path)
    parser.add_argument("--execute-local-training", action="store_true")
    parser.add_argument("--mode", default="fr_action_sft")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--task-family",
        action="append",
        default=[],
        help="Exact registered *_train family passed to the local trainer; repeatable",
    )
    parser.add_argument(
        "--evaluation-scope",
        action="append",
        default=[],
        choices=sorted(ALLOWED_EVALUATION_SCOPES),
        help="Immutable candidate task_scope evaluated for promotion; repeatable",
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-trials", type=int, default=6)
    parser.add_argument("--campaign-max-cost-usd", type=float, default=0.0)
    parser.add_argument("--campaign-max-duration-seconds", type=float, default=12 * 60 * 60)
    parser.add_argument("--per-trial-cost-ceiling-usd", type=float, default=0.0)
    parser.add_argument("--per-trial-duration-ceiling-seconds", type=float, default=2 * 60 * 60)
    parser.add_argument("--trial-training-seconds", type=float, default=90 * 60)
    parser.add_argument("--minimum-delta", type=float, default=0.005)
    parser.add_argument(
        "--development-max-critical-failures",
        type=int,
        default=100,
        help=(
            "Maximum development critical failures admitted for recipe comparison only; "
            "sealed finalization still requires exactly zero and full development eligibility."
        ),
    )
    parser.add_argument("--sft-learning-rate", type=float, default=0.0001)
    parser.add_argument("--dpo-learning-rate", type=float, default=0.00001)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--action-turn-repeats", type=int, default=0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.finalize_sealed:
        receipt = finalize_sealed(args)
        print(f"RUNTIME_ADAPTER_AUTORESEARCH_SEALED receipt={receipt['sealed_report']}")
        return 0
    record = run_campaign(args)
    summary = record.get("search_result_summary", {})
    print(
        "RUNTIME_ADAPTER_AUTORESEARCH "
        f"status={record['status']} passed={summary.get('passed')} "
        f"trial_count={summary.get('trial_count')} record={runner_paths(args.campaign_dir).campaign_record}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
