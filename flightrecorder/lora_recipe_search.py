"""Bounded, development-only LoRA recipe search with promotion handoff gates.

The search loop deliberately stops at candidate selection.  It never receives
held-out inputs and cannot promote a model.  A separate handoff binds the
selected candidate to Flight Recorder's replayable repeated-evaluation evidence.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from .path_safety import path_has_symlink_component
from .redaction import redact_text
from .repeated_eval import validate_promotion_evidence
from .schema_registry import SchemaRegistryError, check_schema_contract

SEARCH_PLAN_SCHEMA_VERSION = "hfr.lora_recipe_search_plan.v1"
TRIAL_SCHEMA_VERSION = "hfr.lora_recipe_trial.v1"
SEARCH_RESULT_SCHEMA_VERSION = "hfr.lora_recipe_search_result.v1"
PROMOTION_HANDOFF_SCHEMA_VERSION = "hfr.lora_recipe_promotion_handoff.v1"

RECIPE_FIELDS = (
    "mode",
    "sft_learning_rate",
    "dpo_learning_rate",
    "batch_size",
    "gradient_accumulation_steps",
    "max_steps",
    "max_length",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "seed",
    "data_seed",
)
MUTABLE_RECIPE_FIELDS = frozenset(
    {
        "sft_learning_rate",
        "dpo_learning_rate",
        "batch_size",
        "gradient_accumulation_steps",
        "max_steps",
        "max_length",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
    }
)
SUPPORTED_MODES = frozenset({"trace_sft", "fr_sft", "fr_action_sft", "fr_dpo", "fr_sft_dpo"})
HELDOUT_TAGS = frozenset({"heldout", "held_out", "frozen", "adversarial", "final", "test"})


class LoraRecipeSearchError(ValueError):
    """Raised when a search contract, proposal, or result is unsafe."""


class RecipeProposer(Protocol):
    """Supplies one candidate mutation from the current bounded search state."""

    def propose(self, state: dict[str, Any]) -> dict[str, Any] | None: ...


class RecipeEvaluator(Protocol):
    """Returns a simulated or imported result for the plan-bound development suite."""

    def evaluate(
        self,
        recipe: dict[str, Any],
        *,
        trial_id: str,
        development_suite_path: Path,
    ) -> dict[str, Any]: ...


def build_search_plan(
    *,
    campaign_id: str,
    objective: str,
    development_suite_path: str | Path,
    base_recipe: dict[str, Any],
    mutable_fields: list[str],
    budget: dict[str, Any],
    out_path: str | Path,
    primary_metric: str = "development_pass_rate",
    direction: str = "maximize",
    minimum_delta: float = 0.0,
    max_critical_failures: int = 0,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a replayable search plan that exposes no held-out input surface."""

    if not _safe_id(campaign_id):
        raise LoraRecipeSearchError("campaign_id must contain only letters, numbers, '.', '_', or '-'")
    if not isinstance(objective, str) or not objective.strip():
        raise LoraRecipeSearchError("objective must be a non-empty string")
    if not isinstance(primary_metric, str) or not primary_metric.strip():
        raise LoraRecipeSearchError("primary_metric must be a non-empty string")
    if direction not in {"maximize", "minimize"}:
        raise LoraRecipeSearchError("direction must be 'maximize' or 'minimize'")
    if not _is_non_negative_number(minimum_delta):
        raise LoraRecipeSearchError("minimum_delta must be a finite non-negative number")
    if not _is_non_negative_int(max_critical_failures):
        raise LoraRecipeSearchError("max_critical_failures must be a non-negative integer")

    recipe = _validated_recipe(base_recipe)
    normalized_mutable = _validated_mutable_fields(mutable_fields)
    normalized_budget = _validated_budget(budget)
    output_path = Path(out_path)
    suite_path = Path(development_suite_path)
    suite = _read_object(suite_path, "development suite")
    suite_schema = check_schema_contract(suite, name_or_id="eval_suite_manifest")
    if not suite_schema["passed"]:
        raise LoraRecipeSearchError("development suite failed eval_suite_manifest schema: " + "; ".join(suite_schema["errors"]))
    tags = {str(tag).strip().lower() for tag in suite.get("tags", []) if isinstance(tag, str)}
    if "development" not in tags:
        raise LoraRecipeSearchError("development suite must include the 'development' tag")
    forbidden_tags = sorted(tags & HELDOUT_TAGS)
    if forbidden_tags:
        raise LoraRecipeSearchError(f"development suite must not carry held-out tags: {forbidden_tags!r}")
    suite_ref = _artifact_ref(suite_path, output_path.parent, expected_schema="hfr.eval_suite_manifest.v1")

    plan = {
        "schema_version": SEARCH_PLAN_SCHEMA_VERSION,
        "created_at": created_at or _utc_now(),
        "campaign_id": campaign_id,
        "objective": redact_text(objective.strip()),
        "development_suite": {
            **suite_ref,
            "suite_id": str(suite["suite_id"]),
            "evaluation_role": "development",
            "heldout": False,
        },
        "base_recipe": recipe,
        "base_recipe_sha256": _canonical_sha256(recipe),
        "mutable_fields": normalized_mutable,
        "selection_policy": {
            "primary_metric": primary_metric.strip(),
            "direction": direction,
            "minimum_delta": float(minimum_delta),
            "max_critical_failures": max_critical_failures,
        },
        "budget": normalized_budget,
        "execution_boundary": {
            "development_only": True,
            "heldout_inputs_allowed": False,
            "promotion_allowed": False,
            "model_weight_training_executed_by_flight_recorder": False,
            "external_side_effects_launched_by_search": False,
        },
    }
    schema = check_schema_contract(plan, name_or_id="lora_recipe_search_plan")
    if not schema["passed"]:
        raise LoraRecipeSearchError("search plan schema violation: " + "; ".join(schema["errors"]))
    return plan


def run_search(
    *,
    plan_path: str | Path,
    out_path: str | Path,
    proposer: RecipeProposer,
    evaluator: RecipeEvaluator,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Run a baseline-first, keep-or-discard search under the immutable plan."""

    source_plan_path = Path(plan_path)
    plan_validation = validate_search_plan(source_plan_path)
    if not plan_validation["passed"]:
        raise LoraRecipeSearchError("search plan validation failed: " + "; ".join(plan_validation["errors"]))
    plan = _read_object(source_plan_path, "search plan")
    result_path = Path(out_path)
    result_root = result_path.parent
    plan_ref = _artifact_ref(source_plan_path, result_root, expected_schema=SEARCH_PLAN_SCHEMA_VERSION)
    development_suite_path = _resolve_ref(
        plan["development_suite"], source_plan_path.parent, [], "development_suite"
    )
    if development_suite_path is None:
        raise LoraRecipeSearchError("search plan development suite could not be resolved")

    trial_dir = result_root
    instant = created_at or _utc_now()
    trials: list[dict[str, Any]] = []
    trial_refs: list[dict[str, Any]] = []
    spent_cost = 0.0
    spent_duration = 0.0
    incumbent: dict[str, Any] | None = None
    terminal_error = ""
    stop_reason = "proposer_finished"

    baseline = {
        "proposal_id": "baseline",
        "hypothesis": "Establish the unmodified recipe baseline.",
        "mutations": {},
        "estimated_cost_usd": plan["budget"]["per_trial_cost_ceiling_usd"],
        "estimated_duration_seconds": plan["budget"]["per_trial_duration_ceiling_seconds"],
    }

    proposal: dict[str, Any] | None = baseline
    while proposal is not None:
        if len(trials) >= plan["budget"]["max_trials"]:
            stop_reason = "max_trials_reached"
            break
        trial_index = len(trials)
        is_baseline = trial_index == 0
        proposal_errors = _proposal_errors(proposal, plan, is_baseline=is_baseline)
        proposal_id = str(proposal.get("proposal_id") or f"proposal-{trial_index:03d}")
        trial_id = f"trial-{trial_index:03d}-{_slug(proposal_id)}"
        recipe = copy.deepcopy(plan["base_recipe"] if is_baseline else (incumbent or {}).get("recipe", plan["base_recipe"]))
        if not proposal_errors:
            recipe.update(copy.deepcopy(proposal.get("mutations", {})))
            try:
                recipe = _validated_recipe(recipe)
            except LoraRecipeSearchError as exc:
                proposal_errors.append(str(exc))

        estimate_cost = _number(proposal.get("estimated_cost_usd"))
        estimate_duration = _number(proposal.get("estimated_duration_seconds"))
        if not proposal_errors and (
            spent_cost + estimate_cost > plan["budget"]["max_cost_usd"]
            or spent_duration + estimate_duration > plan["budget"]["max_duration_seconds"]
        ):
            proposal_errors.append("proposal estimate exceeds remaining campaign budget")
            stop_reason = "estimated_budget_exhausted"

        evaluation = _empty_evaluation("not_run")
        status = "blocked" if proposal_errors else "completed"
        if not proposal_errors:
            try:
                evaluation = _validated_evaluation(
                    evaluator.evaluate(
                        copy.deepcopy(recipe),
                        trial_id=trial_id,
                        development_suite_path=development_suite_path,
                    ),
                    plan,
                )
                status = "completed" if evaluation["status"] == "completed" else "crashed"
            except Exception as exc:  # evaluator failures are recorded, not trusted
                evaluation = _empty_evaluation("crashed", diagnostics=[redact_text(str(exc))])
                status = "crashed"

        actual_cost = float(evaluation["cost_usd"])
        actual_duration = float(evaluation["duration_seconds"])
        budget_violation = (
            actual_cost > plan["budget"]["per_trial_cost_ceiling_usd"]
            or actual_duration > plan["budget"]["per_trial_duration_ceiling_seconds"]
            or spent_cost + actual_cost > plan["budget"]["max_cost_usd"]
            or spent_duration + actual_duration > plan["budget"]["max_duration_seconds"]
        )
        if budget_violation:
            status = "budget_violation"
            terminal_error = "evaluator reported usage outside the immutable budget"
            stop_reason = "actual_budget_violation"

        decision = _trial_decision(
            plan=plan,
            status=status,
            evaluation=evaluation,
            incumbent=incumbent,
            is_baseline=is_baseline,
            proposal_errors=proposal_errors,
        )
        if decision["outcome"] in {"baseline", "keep"}:
            decision["incumbent_trial_id_after"] = trial_id
        receipt = {
            "schema_version": TRIAL_SCHEMA_VERSION,
            "created_at": _offset_timestamp(instant, trial_index),
            "campaign_id": plan["campaign_id"],
            "trial_index": trial_index,
            "trial_id": trial_id,
            "parent_trial_id": (incumbent or {}).get("trial_id"),
            "source_plan": plan_ref,
            "development_suite": {
                "path": plan["development_suite"]["path"],
                "sha256": plan["development_suite"]["sha256"],
                "size_bytes": plan["development_suite"]["size_bytes"],
                "schema_version": plan["development_suite"]["schema_version"],
                "evaluation_role": "development",
                "heldout": False,
            },
            "proposal": {
                "proposal_id": proposal_id,
                "hypothesis": redact_text(str(proposal.get("hypothesis") or "")),
                "mutations": copy.deepcopy(proposal.get("mutations", {})),
                "estimated_cost_usd": estimate_cost,
                "estimated_duration_seconds": estimate_duration,
                "validation_errors": proposal_errors,
            },
            "recipe": recipe,
            "recipe_sha256": _canonical_sha256(recipe),
            "status": status,
            "evaluation": evaluation,
            "decision": decision,
            "execution_boundary": {
                "development_only": True,
                "heldout_accessed": False,
                "promotion_applied": False,
            },
        }
        schema = check_schema_contract(receipt, name_or_id="lora_recipe_trial")
        if not schema["passed"]:
            raise LoraRecipeSearchError("trial receipt schema violation: " + "; ".join(schema["errors"]))
        receipt_path = trial_dir / f"trial-{trial_index:03d}-{_slug(proposal_id)}.json"
        write_json(receipt_path, receipt)
        receipt_ref = _artifact_ref(receipt_path, result_root, expected_schema=TRIAL_SCHEMA_VERSION)
        receipt_ref.update(
            {
                "trial_index": trial_index,
                "trial_id": trial_id,
                "status": status,
                "outcome": decision["outcome"],
            }
        )
        trials.append(receipt)
        trial_refs.append(receipt_ref)
        spent_cost = round(spent_cost + actual_cost, 8)
        spent_duration = round(spent_duration + actual_duration, 6)
        if decision["outcome"] in {"baseline", "keep"}:
            incumbent = {
                "trial_id": trial_id,
                "recipe": copy.deepcopy(recipe),
                "recipe_sha256": receipt["recipe_sha256"],
                "candidate_identity_sha256": evaluation["candidate_identity_sha256"],
                "development_metric": evaluation["primary_metric"],
            }
        if terminal_error or (is_baseline and decision["outcome"] != "baseline"):
            if not terminal_error:
                terminal_error = "baseline evaluation did not produce an admissible incumbent"
                stop_reason = "baseline_failed"
            break
        if stop_reason == "estimated_budget_exhausted":
            break
        try:
            proposal = proposer.propose(_search_state(plan, trials, incumbent, spent_cost, spent_duration))
        except Exception as exc:
            terminal_error = redact_text(f"proposer failed: {exc}")
            stop_reason = "proposer_failed"
            break
        if proposal is not None and not isinstance(proposal, dict):
            terminal_error = "proposer returned a non-object proposal"
            stop_reason = "proposer_failed"
            break

    ready = incumbent is not None and not terminal_error
    result = {
        "schema_version": SEARCH_RESULT_SCHEMA_VERSION,
        "created_at": _offset_timestamp(instant, len(trials)),
        "campaign_id": plan["campaign_id"],
        "passed": ready,
        "status": "complete" if ready else "failed",
        "readiness": "ready_for_heldout_evaluation" if ready else "blocked",
        "recommendation": "run_governed_heldout_evaluation" if ready else "repair_search_campaign",
        "stop_reason": stop_reason,
        "terminal_error": terminal_error,
        "source_plan": plan_ref,
        "trials": trial_refs,
        "trial_count": len(trials),
        "kept_trial_count": sum(1 for trial in trials if trial["decision"]["outcome"] in {"baseline", "keep"}),
        "discarded_trial_count": sum(1 for trial in trials if trial["decision"]["outcome"] == "discard"),
        "blocked_trial_count": sum(1 for trial in trials if trial["decision"]["outcome"] == "blocked"),
        "budget_usage": {
            "cost_usd": spent_cost,
            "duration_seconds": spent_duration,
            "trial_count": len(trials),
        },
        "champion": incumbent,
        "heldout_access": {
            "used_during_search": False,
            "artifact_count": 0,
            "artifacts": [],
        },
        "execution_boundary": {
            "development_only": True,
            "model_weight_training_executed_by_flight_recorder": False,
            "promotion_applied": False,
            "external_side_effects_launched_by_search": False,
            "imported_external_side_effects_observed": any(
                trial["evaluation"]["external_side_effects_observed"] for trial in trials
            ),
            "model_weights_updated_externally": any(
                trial["evaluation"]["model_weights_updated_externally"] for trial in trials
            ),
        },
    }
    schema = check_schema_contract(result, name_or_id="lora_recipe_search_result")
    if not schema["passed"]:
        raise LoraRecipeSearchError("search result schema violation: " + "; ".join(schema["errors"]))
    write_json(result_path, result)
    return result


def validate_search_plan(path: str | Path) -> dict[str, Any]:
    """Validate plan shape, recipe bounds, and replayable development evidence."""

    plan_path = Path(path)
    errors: list[str] = []
    try:
        plan = _read_object(plan_path, "search plan")
        schema = check_schema_contract(plan, name_or_id="lora_recipe_search_plan")
        errors.extend(schema["errors"])
        if errors:
            return _validation(SEARCH_PLAN_SCHEMA_VERSION, errors)
        _validated_recipe(plan["base_recipe"])
        if _canonical_sha256(plan["base_recipe"]) != plan["base_recipe_sha256"]:
            errors.append("base_recipe_sha256 does not match base_recipe")
        _validated_mutable_fields(plan["mutable_fields"])
        _validated_budget(plan["budget"])
        suite_path = _resolve_ref(plan["development_suite"], plan_path.parent, errors, "development_suite")
        if suite_path is not None:
            suite = _read_object(suite_path, "development suite")
            suite_schema = check_schema_contract(suite, name_or_id="eval_suite_manifest")
            errors.extend(f"development_suite: {error}" for error in suite_schema["errors"])
            tags = {str(tag).strip().lower() for tag in suite.get("tags", []) if isinstance(tag, str)}
            if "development" not in tags:
                errors.append("development suite is missing the 'development' tag")
            if tags & HELDOUT_TAGS:
                errors.append("development suite contains a held-out tag")
        boundary = plan["execution_boundary"]
        if boundary != {
            "development_only": True,
            "heldout_inputs_allowed": False,
            "promotion_allowed": False,
            "model_weight_training_executed_by_flight_recorder": False,
            "external_side_effects_launched_by_search": False,
        }:
            errors.append("execution_boundary does not match the fail-closed search boundary")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError, SchemaRegistryError) as exc:
        errors.append(str(exc))
    return _validation(SEARCH_PLAN_SCHEMA_VERSION, errors)


def validate_search_result(path: str | Path) -> dict[str, Any]:
    """Replay a result's plan, trial lineage, decisions, counters, and budgets."""

    result_path = Path(path)
    errors: list[str] = []
    try:
        result = _read_object(result_path, "search result")
        schema = check_schema_contract(result, name_or_id="lora_recipe_search_result")
        errors.extend(schema["errors"])
        if errors:
            return _validation(SEARCH_RESULT_SCHEMA_VERSION, errors)
        plan_path = _resolve_ref(result["source_plan"], result_path.parent, errors, "source_plan")
        if plan_path is None:
            return _validation(SEARCH_RESULT_SCHEMA_VERSION, errors)
        plan_validation = validate_search_plan(plan_path)
        errors.extend(f"source_plan: {error}" for error in plan_validation["errors"])
        plan = _read_object(plan_path, "search plan")
        trial_payloads: list[dict[str, Any]] = []
        incumbent: dict[str, Any] | None = None
        cost = 0.0
        duration = 0.0
        seen_ids: set[str] = set()
        for expected_index, ref in enumerate(result["trials"]):
            trial_path = _resolve_ref(ref, result_path.parent, errors, f"trials[{expected_index}]")
            if trial_path is None:
                continue
            trial = _read_object(trial_path, "trial receipt")
            trial_payloads.append(trial)
            trial_schema = check_schema_contract(trial, name_or_id="lora_recipe_trial")
            errors.extend(f"trials[{expected_index}]: {error}" for error in trial_schema["errors"])
            if trial.get("trial_index") != expected_index:
                errors.append(f"trials[{expected_index}] has a non-contiguous trial_index")
            for field in ("trial_index", "trial_id", "status"):
                if ref.get(field) != trial.get(field):
                    errors.append(f"trials[{expected_index}] reference {field} does not match receipt")
            if ref.get("outcome") != trial.get("decision", {}).get("outcome"):
                errors.append(f"trials[{expected_index}] reference outcome does not match receipt")
            if trial.get("trial_id") in seen_ids:
                errors.append(f"trials[{expected_index}] repeats trial_id {trial.get('trial_id')!r}")
            seen_ids.add(str(trial.get("trial_id") or ""))
            if trial.get("campaign_id") != plan.get("campaign_id"):
                errors.append(f"trials[{expected_index}] campaign_id does not match plan")
            if trial.get("source_plan", {}).get("sha256") != result["source_plan"]["sha256"]:
                errors.append(f"trials[{expected_index}] source plan fingerprint does not match result")
            if _canonical_sha256(trial.get("recipe")) != trial.get("recipe_sha256"):
                errors.append(f"trials[{expected_index}] recipe_sha256 does not match recipe")
            if trial.get("execution_boundary") != {
                "development_only": True,
                "heldout_accessed": False,
                "promotion_applied": False,
            }:
                errors.append(f"trials[{expected_index}] violates the development-only boundary")
            if trial.get("development_suite", {}).get("sha256") != plan["development_suite"]["sha256"]:
                errors.append(f"trials[{expected_index}] development suite fingerprint does not match plan")
            proposal_value = trial.get("proposal")
            proposal_record: dict[str, Any] = proposal_value if isinstance(proposal_value, dict) else {}
            raw_proposal = {
                key: proposal_record.get(key)
                for key in (
                    "proposal_id",
                    "hypothesis",
                    "mutations",
                    "estimated_cost_usd",
                    "estimated_duration_seconds",
                )
            }
            expected_proposal_errors = _proposal_errors(raw_proposal, plan, is_baseline=expected_index == 0)
            mutations = proposal_record.get("mutations", {})
            parent_recipe = copy.deepcopy(plan["base_recipe"] if expected_index == 0 else (incumbent or {}).get("recipe", plan["base_recipe"]))
            candidate_recipe = copy.deepcopy(parent_recipe)
            if not expected_proposal_errors:
                candidate_recipe.update(mutations if isinstance(mutations, dict) else {})
                try:
                    candidate_recipe = _validated_recipe(candidate_recipe)
                except LoraRecipeSearchError as exc:
                    expected_proposal_errors.append(str(exc))
                    candidate_recipe = parent_recipe
            recorded_proposal_errors = [
                str(error)
                for error in proposal_record.get("validation_errors", [])
                if isinstance(error, str)
            ]
            if recorded_proposal_errors != expected_proposal_errors:
                errors.append(f"trials[{expected_index}] proposal validation errors do not replay")
            if expected_index == 0:
                expected_recipe = candidate_recipe
                if trial.get("parent_trial_id") is not None:
                    errors.append("baseline trial must not have a parent")
            else:
                expected_recipe = candidate_recipe
                if trial.get("parent_trial_id") != (incumbent or {}).get("trial_id"):
                    errors.append(f"trials[{expected_index}] parent does not match the active incumbent")
            if trial.get("recipe") != expected_recipe:
                errors.append(f"trials[{expected_index}] recipe does not replay from parent plus mutations")
            evaluation_value = trial.get("evaluation")
            evaluation_record: dict[str, Any] = evaluation_value if isinstance(evaluation_value, dict) else {}
            expected_decision = _trial_decision(
                plan=plan,
                status=str(trial.get("status") or ""),
                evaluation=evaluation_record,
                incumbent=incumbent,
                is_baseline=expected_index == 0,
                proposal_errors=expected_proposal_errors,
            )
            if expected_decision["outcome"] in {"baseline", "keep"}:
                expected_decision["incumbent_trial_id_after"] = trial.get("trial_id")
            if trial.get("decision") != expected_decision:
                errors.append(f"trials[{expected_index}] decision does not replay from policy and metrics")
            trial_cost = _number(evaluation_record.get("cost_usd"))
            trial_duration = _number(evaluation_record.get("duration_seconds"))
            evaluation_not_run = evaluation_record.get("status") == "not_run"
            if evaluation_not_run != (evaluation_record.get("execution_mode") == "not_run"):
                errors.append(f"trials[{expected_index}] execution_mode does not match evaluation status")
            if evaluation_not_run and (
                evaluation_record.get("external_side_effects_observed") is not False
                or evaluation_record.get("model_weights_updated_externally") is not False
            ):
                errors.append(f"trials[{expected_index}] not-run evaluation cannot report side effects")
            trial_usage_exceeded = (
                trial_cost > plan["budget"]["per_trial_cost_ceiling_usd"]
                or trial_duration > plan["budget"]["per_trial_duration_ceiling_seconds"]
                or cost + trial_cost > plan["budget"]["max_cost_usd"]
                or duration + trial_duration > plan["budget"]["max_duration_seconds"]
            )
            if trial_usage_exceeded != (trial.get("status") == "budget_violation"):
                errors.append(f"trials[{expected_index}] budget-violation status does not match reported usage")
            outcome = trial.get("decision", {}).get("outcome")
            if outcome in {"baseline", "keep"}:
                incumbent = {
                    "trial_id": trial["trial_id"],
                    "recipe": trial["recipe"],
                    "recipe_sha256": trial["recipe_sha256"],
                    "candidate_identity_sha256": trial["evaluation"]["candidate_identity_sha256"],
                    "development_metric": trial["evaluation"]["primary_metric"],
                }
            cost += trial_cost
            duration += trial_duration
        if result["trial_count"] != len(trial_payloads):
            errors.append("trial_count does not match replayed trial receipts")
        expected_kept = sum(1 for trial in trial_payloads if trial.get("decision", {}).get("outcome") in {"baseline", "keep"})
        expected_discarded = sum(1 for trial in trial_payloads if trial.get("decision", {}).get("outcome") == "discard")
        expected_blocked = sum(1 for trial in trial_payloads if trial.get("decision", {}).get("outcome") == "blocked")
        if result["kept_trial_count"] != expected_kept:
            errors.append("kept_trial_count does not match trial decisions")
        if result["discarded_trial_count"] != expected_discarded:
            errors.append("discarded_trial_count does not match trial decisions")
        if result["blocked_trial_count"] != expected_blocked:
            errors.append("blocked_trial_count does not match trial decisions")
        usage = result["budget_usage"]
        if not math.isclose(float(usage["cost_usd"]), cost, rel_tol=0.0, abs_tol=1e-8):
            errors.append("budget_usage.cost_usd does not match trial receipts")
        if not math.isclose(float(usage["duration_seconds"]), duration, rel_tol=0.0, abs_tol=1e-6):
            errors.append("budget_usage.duration_seconds does not match trial receipts")
        if usage["trial_count"] != len(trial_payloads):
            errors.append("budget_usage.trial_count does not match trial receipts")
        usage_exceeded = cost > plan["budget"]["max_cost_usd"] or duration > plan["budget"]["max_duration_seconds"]
        violation_reported = any(trial.get("status") == "budget_violation" for trial in trial_payloads)
        if usage_exceeded and not (
            violation_reported
            and result.get("passed") is False
            and result.get("status") == "failed"
            and result.get("readiness") == "blocked"
        ):
            errors.append("campaign usage exceeds the immutable plan budget without a fail-closed result")
        if result.get("champion") != incumbent:
            errors.append("champion does not match the final kept trial")
        if result["heldout_access"] != {"used_during_search": False, "artifact_count": 0, "artifacts": []}:
            errors.append("search result reports held-out access")
        expected_boundary = {
            "development_only": True,
            "model_weight_training_executed_by_flight_recorder": False,
            "promotion_applied": False,
            "external_side_effects_launched_by_search": False,
            "imported_external_side_effects_observed": any(
                trial.get("evaluation", {}).get("external_side_effects_observed") is True for trial in trial_payloads
            ),
            "model_weights_updated_externally": any(
                trial.get("evaluation", {}).get("model_weights_updated_externally") is True for trial in trial_payloads
            ),
        }
        if result.get("execution_boundary") != expected_boundary:
            errors.append("execution_boundary does not match replayed trial execution posture")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError, SchemaRegistryError) as exc:
        errors.append(str(exc))
    return _validation(SEARCH_RESULT_SCHEMA_VERSION, errors)


def build_promotion_handoff(
    *,
    search_result_path: str | Path,
    promotion_evidence_path: str | Path,
    out_path: str | Path,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Bind a development-selected champion to untouched promotion evidence."""

    output_path = Path(out_path)
    search_path = Path(search_result_path)
    evidence_path = Path(promotion_evidence_path)
    search_validation = validate_search_result(search_path)
    evidence_validation = validate_promotion_evidence(evidence_path)
    search = _read_object(search_path, "search result")
    evidence = _read_object(evidence_path, "promotion evidence")
    champion_value = search.get("champion")
    champion: dict[str, Any] = champion_value if isinstance(champion_value, dict) else {}
    arms_value = evidence.get("arms")
    arms: dict[str, Any] = arms_value if isinstance(arms_value, dict) else {}
    flightrecorder_value = arms.get("flightrecorder")
    flightrecorder_arm: dict[str, Any] = flightrecorder_value if isinstance(flightrecorder_value, dict) else {}
    identity_value = flightrecorder_arm.get("identity")
    arm_identity: dict[str, Any] = identity_value if isinstance(identity_value, dict) else {}
    adapter_value = arm_identity.get("adapter")
    adapter: dict[str, Any] = adapter_value if isinstance(adapter_value, dict) else {}
    selected_sha = str(champion.get("candidate_identity_sha256") or "")
    evaluated_sha = str(adapter.get("sha256") or "")
    checks = [
        _check("search_result_replays", search_validation["passed"], search_validation["errors"], []),
        _check("development_only_selection", search.get("heldout_access", {}).get("used_during_search") is False, search.get("heldout_access"), {"used_during_search": False}),
        _check("search_ready", search.get("readiness") == "ready_for_heldout_evaluation", search.get("readiness"), "ready_for_heldout_evaluation"),
        _check("promotion_evidence_replays", evidence_validation["passed"], evidence_validation["errors"], []),
        _check("promotion_evidence_passed", evidence.get("promotion_ready") is True, evidence.get("promotion_ready"), True),
        _check("selected_candidate_matches_evaluated_adapter", bool(selected_sha) and selected_sha == evaluated_sha, evaluated_sha, selected_sha),
    ]
    failed = [check for check in checks if not check["passed"]]
    handoff = {
        "schema_version": PROMOTION_HANDOFF_SCHEMA_VERSION,
        "created_at": created_at or _utc_now(),
        "passed": not failed,
        "readiness": "ready_for_governance_review" if not failed else "blocked",
        "recommendation": "send_to_governance" if not failed else "repair_search_or_evaluation",
        "source_artifacts": {
            "search_result": _artifact_ref(search_path, output_path.parent, expected_schema=SEARCH_RESULT_SCHEMA_VERSION),
            "promotion_evidence": _artifact_ref(evidence_path, output_path.parent, expected_schema="hfr.agentic_eval_promotion_evidence.v1"),
        },
        "candidate_binding": {
            "champion_trial_id": str(champion.get("trial_id") or ""),
            "recipe_sha256": str(champion.get("recipe_sha256") or ""),
            "selected_candidate_sha256": selected_sha,
            "evaluated_adapter_sha256": evaluated_sha,
            "matched": bool(selected_sha) and selected_sha == evaluated_sha,
        },
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "checks": checks,
        "blocking_reasons": [check["id"] for check in failed],
        "execution_boundary": {
            "search_executed_on_development_only": True,
            "heldout_evaluation_executed_after_selection": True,
            "promotion_applied": False,
            "registry_alias_updated": False,
            "model_weights_updated_by_flight_recorder": False,
        },
    }
    schema = check_schema_contract(handoff, name_or_id="lora_recipe_promotion_handoff")
    if not schema["passed"]:
        raise LoraRecipeSearchError("promotion handoff schema violation: " + "; ".join(schema["errors"]))
    return handoff


def validate_promotion_handoff(path: str | Path) -> dict[str, Any]:
    """Replay a promotion handoff from its bound search and evaluation inputs."""

    handoff_path = Path(path)
    errors: list[str] = []
    try:
        handoff = _read_object(handoff_path, "promotion handoff")
        schema = check_schema_contract(handoff, name_or_id="lora_recipe_promotion_handoff")
        errors.extend(schema["errors"])
        if errors:
            return _validation(PROMOTION_HANDOFF_SCHEMA_VERSION, errors)
        sources = handoff["source_artifacts"]
        search_path = _resolve_ref(sources["search_result"], handoff_path.parent, errors, "search_result")
        evidence_path = _resolve_ref(sources["promotion_evidence"], handoff_path.parent, errors, "promotion_evidence")
        if search_path is None or evidence_path is None or errors:
            return _validation(PROMOTION_HANDOFF_SCHEMA_VERSION, errors)
        rebuilt = build_promotion_handoff(
            search_result_path=search_path,
            promotion_evidence_path=evidence_path,
            out_path=handoff_path,
            created_at=handoff["created_at"],
        )
        if rebuilt != handoff:
            errors.append("promotion handoff does not match deterministic replay of its source artifacts")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError, SchemaRegistryError) as exc:
        errors.append(str(exc))
    return _validation(PROMOTION_HANDOFF_SCHEMA_VERSION, errors)


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _trial_decision(
    *,
    plan: dict[str, Any],
    status: str,
    evaluation: dict[str, Any],
    incumbent: dict[str, Any] | None,
    is_baseline: bool,
    proposal_errors: list[str],
) -> dict[str, Any]:
    before = (incumbent or {}).get("trial_id")
    metric = evaluation.get("primary_metric")
    incumbent_metric = (incumbent or {}).get("development_metric")
    delta = None
    if _is_number(metric) and _is_number(incumbent_metric):
        delta = _number(metric) - _number(incumbent_metric)
        if plan["selection_policy"]["direction"] == "minimize":
            delta = -delta
    if status == "blocked":
        outcome, reason = "blocked", "; ".join(proposal_errors) or "proposal blocked"
    elif status == "budget_violation":
        outcome, reason = "blocked", "reported usage exceeded an immutable budget"
    elif status != "completed" or evaluation.get("status") != "completed":
        outcome, reason = "discard", "evaluator did not complete successfully"
    elif evaluation["critical_failures"] > plan["selection_policy"]["max_critical_failures"]:
        outcome, reason = "discard", "development critical-failure ceiling exceeded"
    elif is_baseline:
        outcome, reason = "baseline", "baseline established"
    elif delta is not None and delta >= plan["selection_policy"]["minimum_delta"]:
        outcome, reason = "keep", "development metric improved by the required delta"
    else:
        outcome, reason = "discard", "development metric did not improve by the required delta"
    return {
        "outcome": outcome,
        "reason": reason,
        "metric_delta": delta,
        "incumbent_trial_id_before": before,
        "incumbent_trial_id_after": before,
    }


def _search_state(
    plan: dict[str, Any],
    trials: list[dict[str, Any]],
    incumbent: dict[str, Any] | None,
    spent_cost: float,
    spent_duration: float,
) -> dict[str, Any]:
    summaries = [
        {
            "trial_id": trial["trial_id"],
            "proposal_id": trial["proposal"]["proposal_id"],
            "hypothesis": trial["proposal"]["hypothesis"],
            "mutations": copy.deepcopy(trial["proposal"]["mutations"]),
            "status": trial["status"],
            "outcome": trial["decision"]["outcome"],
            "primary_metric": trial["evaluation"]["primary_metric"],
            "critical_failures": trial["evaluation"]["critical_failures"],
        }
        for trial in trials
    ]
    return {
        "campaign_id": plan["campaign_id"],
        "objective": plan["objective"],
        "trial_count": len(trials),
        "mutable_fields": list(plan["mutable_fields"]),
        "selection_policy": copy.deepcopy(plan["selection_policy"]),
        "incumbent": copy.deepcopy(incumbent),
        "trials": summaries,
        "remaining_budget": {
            "trials": max(0, plan["budget"]["max_trials"] - len(trials)),
            "cost_usd": max(0.0, plan["budget"]["max_cost_usd"] - spent_cost),
            "duration_seconds": max(0.0, plan["budget"]["max_duration_seconds"] - spent_duration),
        },
        "heldout_inputs": [],
    }


def _proposal_errors(proposal: Any, plan: dict[str, Any], *, is_baseline: bool) -> list[str]:
    if not isinstance(proposal, dict):
        return ["proposal must be an object"]
    errors: list[str] = []
    allowed = {"proposal_id", "hypothesis", "mutations", "estimated_cost_usd", "estimated_duration_seconds"}
    unknown = sorted(set(proposal) - allowed)
    if unknown:
        errors.append(f"proposal has unknown fields: {unknown!r}")
    if not _safe_id(proposal.get("proposal_id")):
        errors.append("proposal_id must be a safe non-empty identifier")
    if not isinstance(proposal.get("hypothesis"), str) or not proposal["hypothesis"].strip():
        errors.append("hypothesis must be a non-empty string")
    mutations = proposal.get("mutations")
    if not isinstance(mutations, dict):
        errors.append("mutations must be an object")
    else:
        if is_baseline and mutations:
            errors.append("baseline must not mutate the recipe")
        if not is_baseline and not mutations:
            errors.append("candidate proposal must mutate at least one field")
        unauthorized = sorted(set(mutations) - set(plan["mutable_fields"]))
        if unauthorized:
            errors.append(f"proposal mutates fields outside the plan allowlist: {unauthorized!r}")
    for field, ceiling in (
        ("estimated_cost_usd", plan["budget"]["per_trial_cost_ceiling_usd"]),
        ("estimated_duration_seconds", plan["budget"]["per_trial_duration_ceiling_seconds"]),
    ):
        value = proposal.get(field)
        if not _is_non_negative_number(value):
            errors.append(f"{field} must be a finite non-negative number")
        elif _number(value) > ceiling:
            errors.append(f"{field} exceeds the per-trial ceiling")
    return errors


def _validated_evaluation(value: Any, plan: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LoraRecipeSearchError("evaluator result must be an object")
    allowed = {
        "status",
        "primary_metric",
        "critical_failures",
        "cost_usd",
        "duration_seconds",
        "candidate_identity_sha256",
        "diagnostics",
        "execution_mode",
        "external_side_effects_observed",
        "model_weights_updated_externally",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise LoraRecipeSearchError(f"evaluator result has unknown fields: {unknown!r}")
    status = value.get("status")
    if status not in {"completed", "crashed"}:
        raise LoraRecipeSearchError("evaluator status must be 'completed' or 'crashed'")
    metric = value.get("primary_metric")
    if status == "completed" and not _is_number(metric):
        raise LoraRecipeSearchError("completed evaluator result requires a finite primary_metric")
    if metric is not None and not _is_number(metric):
        raise LoraRecipeSearchError("primary_metric must be null or a finite number")
    critical = value.get("critical_failures")
    if not _is_non_negative_int(critical):
        raise LoraRecipeSearchError("critical_failures must be a non-negative integer")
    for field in ("cost_usd", "duration_seconds"):
        if not _is_non_negative_number(value.get(field)):
            raise LoraRecipeSearchError(f"{field} must be a finite non-negative number")
    candidate_sha = value.get("candidate_identity_sha256")
    if status == "completed" and not _is_sha256(candidate_sha):
        raise LoraRecipeSearchError("completed evaluator result requires candidate_identity_sha256")
    diagnostics = value.get("diagnostics", [])
    if not isinstance(diagnostics, list) or not all(isinstance(item, str) for item in diagnostics):
        raise LoraRecipeSearchError("diagnostics must be a list of strings")
    execution_mode = value.get("execution_mode")
    if execution_mode not in {"simulation", "imported_external_result"}:
        raise LoraRecipeSearchError("execution_mode must be 'simulation' or 'imported_external_result'")
    for field in ("external_side_effects_observed", "model_weights_updated_externally"):
        if not isinstance(value.get(field), bool):
            raise LoraRecipeSearchError(f"{field} must be a boolean")
    return {
        "status": status,
        "primary_metric_name": plan["selection_policy"]["primary_metric"],
        "primary_metric": _number(metric) if metric is not None else None,
        "critical_failures": critical,
        "cost_usd": float(value["cost_usd"]),
        "duration_seconds": float(value["duration_seconds"]),
        "candidate_identity_sha256": str(candidate_sha or ""),
        "diagnostics": [redact_text(item) for item in diagnostics],
        "execution_mode": execution_mode,
        "external_side_effects_observed": value["external_side_effects_observed"],
        "model_weights_updated_externally": value["model_weights_updated_externally"],
    }


def _empty_evaluation(status: str, *, diagnostics: list[str] | None = None) -> dict[str, Any]:
    return {
        "status": status,
        "primary_metric_name": "",
        "primary_metric": None,
        "critical_failures": 0,
        "cost_usd": 0.0,
        "duration_seconds": 0.0,
        "candidate_identity_sha256": "",
        "diagnostics": [redact_text(item) for item in (diagnostics or [])],
        "execution_mode": "not_run",
        "external_side_effects_observed": False,
        "model_weights_updated_externally": False,
    }


def _validated_recipe(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LoraRecipeSearchError("recipe must be an object")
    missing = sorted(set(RECIPE_FIELDS) - set(value))
    unknown = sorted(set(value) - set(RECIPE_FIELDS))
    if missing or unknown:
        raise LoraRecipeSearchError(f"recipe fields mismatch (missing={missing!r}, unknown={unknown!r})")
    recipe = copy.deepcopy(value)
    if recipe["mode"] not in SUPPORTED_MODES:
        raise LoraRecipeSearchError(f"recipe mode must be one of {sorted(SUPPORTED_MODES)!r}")
    for field in ("sft_learning_rate", "dpo_learning_rate"):
        if not _is_number(recipe[field]) or not 1e-7 <= float(recipe[field]) <= 1e-2:
            raise LoraRecipeSearchError(f"recipe.{field} must be between 1e-7 and 1e-2")
        recipe[field] = float(recipe[field])
    integer_ranges = {
        "batch_size": (1, 128),
        "gradient_accumulation_steps": (1, 1024),
        "max_steps": (1, 100_000),
        "max_length": (128, 32_768),
        "lora_alpha": (1, 1024),
        "seed": (0, 2**31 - 1),
        "data_seed": (0, 2**31 - 1),
    }
    for field, (minimum, maximum) in integer_ranges.items():
        if not isinstance(recipe[field], int) or isinstance(recipe[field], bool) or not minimum <= recipe[field] <= maximum:
            raise LoraRecipeSearchError(f"recipe.{field} must be an integer from {minimum} to {maximum}")
    if recipe["lora_r"] not in {4, 8, 16, 32, 64, 128}:
        raise LoraRecipeSearchError("recipe.lora_r must be one of 4, 8, 16, 32, 64, or 128")
    if not _is_number(recipe["lora_dropout"]) or not 0.0 <= float(recipe["lora_dropout"]) < 1.0:
        raise LoraRecipeSearchError("recipe.lora_dropout must be in [0, 1)")
    recipe["lora_dropout"] = float(recipe["lora_dropout"])
    return recipe


def _validated_mutable_fields(value: Any) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise LoraRecipeSearchError("mutable_fields must be a non-empty list of strings")
    if len(value) != len(set(value)):
        raise LoraRecipeSearchError("mutable_fields must not contain duplicates")
    unsupported = sorted(set(value) - MUTABLE_RECIPE_FIELDS)
    if unsupported:
        raise LoraRecipeSearchError(f"mutable_fields contains unsupported fields: {unsupported!r}")
    return list(value)


def _validated_budget(value: Any) -> dict[str, Any]:
    required = {
        "max_trials",
        "max_cost_usd",
        "max_duration_seconds",
        "per_trial_cost_ceiling_usd",
        "per_trial_duration_ceiling_seconds",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise LoraRecipeSearchError(f"budget must contain exactly {sorted(required)!r}")
    if not isinstance(value["max_trials"], int) or isinstance(value["max_trials"], bool) or not 1 <= value["max_trials"] <= 10_000:
        raise LoraRecipeSearchError("budget.max_trials must be an integer from 1 to 10000")
    for field in required - {"max_trials"}:
        if not _is_non_negative_number(value[field]):
            raise LoraRecipeSearchError(f"budget.{field} must be a finite non-negative number")
    if value["max_duration_seconds"] <= 0 or value["per_trial_duration_ceiling_seconds"] <= 0:
        raise LoraRecipeSearchError("duration budgets must be positive")
    if value["per_trial_cost_ceiling_usd"] > value["max_cost_usd"]:
        raise LoraRecipeSearchError("per-trial cost ceiling cannot exceed campaign cost budget")
    if value["per_trial_duration_ceiling_seconds"] > value["max_duration_seconds"]:
        raise LoraRecipeSearchError("per-trial duration ceiling cannot exceed campaign duration budget")
    return {
        "max_trials": value["max_trials"],
        "max_cost_usd": float(value["max_cost_usd"]),
        "max_duration_seconds": float(value["max_duration_seconds"]),
        "per_trial_cost_ceiling_usd": float(value["per_trial_cost_ceiling_usd"]),
        "per_trial_duration_ceiling_seconds": float(value["per_trial_duration_ceiling_seconds"]),
    }


def _artifact_ref(path: Path, output_root: Path, *, expected_schema: str) -> dict[str, Any]:
    if path.is_symlink() or path_has_symlink_component(path, include_leaf=False):
        raise LoraRecipeSearchError(f"artifact path must not traverse symlinks: {path}")
    if not path.is_file():
        raise LoraRecipeSearchError(f"artifact path is not a file: {path}")
    try:
        relative = path.resolve().relative_to(output_root.resolve()).as_posix()
    except ValueError as exc:
        raise LoraRecipeSearchError(f"artifact must be inside the output artifact directory: {path}") from exc
    if not _is_safe_relative_path(relative):
        raise LoraRecipeSearchError(f"artifact reference is not public-safe: {relative}")
    payload = _read_object(path, "artifact")
    schema_version = str(payload.get("schema_version") or "")
    if schema_version != expected_schema:
        raise LoraRecipeSearchError(f"artifact schema_version must be {expected_schema!r}; got {schema_version!r}")
    return {
        "path": relative,
        "sha256": _file_sha256(path),
        "size_bytes": path.stat().st_size,
        "schema_version": schema_version,
    }


def _resolve_ref(ref: Any, base: Path, errors: list[str], label: str) -> Path | None:
    if not isinstance(ref, dict):
        errors.append(f"{label} reference must be an object")
        return None
    raw = ref.get("path")
    if not isinstance(raw, str) or not _is_safe_relative_path(raw):
        errors.append(f"{label}.path must be a safe relative path")
        return None
    path = base / PurePosixPath(raw)
    if path.is_symlink() or path_has_symlink_component(path, include_leaf=False):
        errors.append(f"{label}.path must not traverse symlinks")
        return None
    if not path.is_file():
        errors.append(f"{label}.path does not resolve to a file")
        return None
    if path.stat().st_size != ref.get("size_bytes"):
        errors.append(f"{label}.size_bytes does not match source")
    if _file_sha256(path) != ref.get("sha256"):
        errors.append(f"{label}.sha256 does not match source")
    payload = _read_object(path, label)
    if payload.get("schema_version") != ref.get("schema_version"):
        errors.append(f"{label}.schema_version does not match source")
    return path


def _check(check_id: str, passed: bool, actual: Any, expected: Any) -> dict[str, Any]:
    return {"id": check_id, "passed": bool(passed), "actual": actual, "expected": expected}


def _validation(schema_version: str, errors: list[str]) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "passed": not errors,
        "error_count": len(errors),
        "errors": errors,
    }


def _read_object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LoraRecipeSearchError(f"{label} must contain a JSON object: {path}")
    return payload


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _offset_timestamp(value: str, offset: int) -> str:
    try:
        instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LoraRecipeSearchError("created_at must be an ISO-8601 timestamp") from exc
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=timezone.utc)
    return (instant + timedelta(seconds=offset)).isoformat()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_id(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and all(character.isalnum() or character in "._-" for character in value)


def _slug(value: str) -> str:
    normalized = "".join(character.lower() if character.isalnum() else "-" for character in value)
    return "-".join(part for part in normalized.split("-") if part)[:80] or "proposal"


def _is_safe_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts and "\\" not in value


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _is_non_negative_number(value: Any) -> bool:
    return _is_number(value) and float(value) >= 0.0


def _is_non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _number(value: Any) -> float:
    return float(value) if _is_number(value) else 0.0
