"""Private validator for Tau-3 competitive-agent v3 evidence bundles.

The validator intentionally lives outside the public schema catalog.  It
validates a private local bundle whose plan starts at
``competitive_v3_plan.json`` and whose later evidence is referenced from that
plan with bundle-relative ``path`` plus ``sha256`` records.

Plan-stage ``private_local`` references may include ``source_path`` so a
preregistration plan can bind immutable local v2 evidence before the bundle is
assembled.  Dataset, training, and final stages fail closed unless referenced
evidence resolves inside the bundle.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .schema_registry import SchemaRegistryError, check_schema_contract
from .tau3_behavior_probes import validate_tau3_behavior_probes
from .tau3_competitive_dataset import validate_tau3_competitive_dataset_bundle
from .tau3_evaluation import validate_tau3_evaluation_report
from .tau3_exposure import Tau3ExposureError, validate_tau3_exposure_ledger
from .tau3_internal_validation import validate_tau3_internal_validation
from .tau3_mlx_training import validate_tau3_process_segments
from .tau3_objective_validity import validate_tau3_objective_validity_report
from .tau3_prefix_equivalence import validate_tau3_prefix_equivalence
from .tau3_sealed_authorization import Tau3SealedAuthorizationError, validate_tau3_sealed_authorization

VALIDATION_SCHEMA_VERSION = "hfr.validation.v1"
PLAN_SCHEMA_VERSION = "hfr.tau3_competitive_v3_plan.v1"
DATASET_SCHEMA_VERSION = "hfr.tau3_competitive_v3_dataset_evidence.v1"
TRAINING_SCHEMA_VERSION = "hfr.tau3_competitive_v3_training_evidence.v1"
FINAL_SCHEMA_VERSION = "hfr.tau3_competitive_v3_final_evidence.v1"
PUBLICATION_SCHEMA_VERSION = "hfr.tau3_competitive_v3_publication_preflight.v1"
EVIDENCE_BINDING_SCHEMA_VERSION = "hfr.tau3_competitive_v3_evidence_binding.v1"

PLAN_FILENAME = "competitive_v3_plan.json"
STAGES = ("plan", "dataset", "training", "final")
EVIDENCE_STAGES = ("dataset", "training", "final", "publication")
EVIDENCE_SCHEMA_BY_STAGE = {
    "dataset": DATASET_SCHEMA_VERSION,
    "training": TRAINING_SCHEMA_VERSION,
    "final": FINAL_SCHEMA_VERSION,
    "publication": PUBLICATION_SCHEMA_VERSION,
}
DOMAINS = ("airline", "retail", "telecom")
BEHAVIORS = (
    "successful_completion",
    "clarification_refusal",
    "authentication",
    "confirmation_before_mutation",
    "later_task_completion_actions",
    "safe_stopping",
    "transfer_handoff",
    "empty_result_recovery",
    "error_result_recovery",
    "repeated_call_recovery",
    "hallucinated_tool_correction",
    "harmful_mutation_correction",
    "premature_completion_correction",
)
EXPECTED_ARMS = ("adapter", "base", "comparator_1", "comparator_2")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PRIVATE_PATH_RE = re.compile(r"(^/|[A-Za-z]:[\\/]|\\\\|/Users/|/home/|/tmp/|localhost|127\.0\.0\.1)")


@dataclass
class _Target:
    type: str
    path: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "path": self.path,
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "details": self.details,
        }


class Tau3CompetitiveV3BindingError(ValueError):
    """Raised when an evidence artifact cannot be immutably bound to a plan."""


@dataclass
class _Loaded:
    target: _Target
    payload: Any = None
    sha256: str | None = None

    @property
    def path(self) -> Path | None:
        text = self.target.path
        if self.target.errors:
            return None
        return Path(text) if text and text != "." else None


def competitive_v3_plan_shape() -> dict[str, Any]:
    """Return the accepted top-level plan shape for builders and schema work."""

    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "mission_contract": {
            "mission_id": "tau3-competitive-agent-v3",
            "mission_statement_sha256": "<64 lowercase hex>",
            "rubric": {"path": "rubric.md", "sha256": "<64 lowercase hex>"},
            "protocol": {"path": "protocol.json", "sha256": "<64 lowercase hex>"},
        },
        "v2_negative_evidence": {
            "immutable": True,
            "rewrites_v2": False,
            "blocked_verdict": {"path": "v2/blocked-verdict.json", "sha256": "<64 lowercase hex>"},
            "candidate_c": {
                "training_receipt": {"path": "v2/candidate-c-training.json", "sha256": "<64 lowercase hex>"},
                "development_result": {"path": "v2/candidate-c-development.json", "sha256": "<64 lowercase hex>"},
            },
        },
        "lineage": {
            "dataset_id": "tau3-competitive-agent-v3",
            "version": "v3",
            "predecessor": "v2",
            "new_lineage": True,
        },
        "sealed_access": {"payload_access_count": 0, "materialized_sealed_fields": []},
        "models": {
            "base": {"model_id": "base", "parameter_b": 8.0, "selected_by_sealed_blind_preflight": True},
            "comparators": [
                {"arm_id": "comparator_1", "parameter_b": 8.0, "same_size_open": True, "frozen": True},
                {"arm_id": "comparator_2", "parameter_b": 8.0, "same_size_open": True, "frozen": True},
            ],
        },
        "harness_contract": {
            "identical_for_all_arms": True,
            "text_mode": True,
            "frozen_before_training": True,
            "tau_repository": {"path": "tau-revision.json", "sha256": "<64 lowercase hex>"},
            "harness": {"path": "harness.json", "sha256": "<64 lowercase hex>"},
            "tokenizer_chat_template": {"path": "tokenizer-chat-template.json", "sha256": "<64 lowercase hex>"},
            "ordered_tool_catalog": {"path": "tool-catalog.json", "sha256": "<64 lowercase hex>"},
            "policy_prompt": {"path": "policy-prompt.txt", "sha256": "<64 lowercase hex>"},
            "task_trial_seed_grid": {"path": "grid.json", "sha256": "<64 lowercase hex>"},
            "decoding": {"path": "decoding.json", "sha256": "<64 lowercase hex>"},
            "retry_policy": {"path": "retry-policy.json", "sha256": "<64 lowercase hex>"},
            "safety_policy": {"path": "safety-policy.json", "sha256": "<64 lowercase hex>"},
        },
        "statistical_contract": {
            "procedure": "paired_bootstrap",
            "confidence_level": 0.95,
            "adapter_improvement_ci_excludes_zero": True,
            "same_size_noninferiority_margin": 0.05,
            "promotion_predicates_fail_closed": True,
        },
        "budget_contract": {
            "unattended_days": 7,
            "separate_infrastructure_attempt_budget": True,
            "separate_qualified_candidate_budget": True,
            "minimum_qualified_candidates": 2,
            "target_qualified_candidates_min": 3,
            "target_qualified_candidates_max": 4,
        },
        "dataset_contract": {"coverage_gates": {}, "sampling_contract": {}, "objective_contract": {}},
        "recipe_contract": {"minimum_rank": 16, "minimum_adapted_layers": 8, "minimum_effective_epochs": 2},
        "development_contract": {"development_only_selection": True, "identical_harness": True},
        "publication_contract": {"competitive_claims_fail_closed": True, "sealed_payloads_public": False},
        "evidence_refs": {
            "dataset": {"path": "dataset-evidence.json", "sha256": "<64 lowercase hex>"},
            "training": {"path": "training-evidence.json", "sha256": "<64 lowercase hex>"},
            "final": {"path": "final-evidence.json", "sha256": "<64 lowercase hex>"},
            "publication": {"path": "publication-preflight.json", "sha256": "<64 lowercase hex>"},
        },
    }


def bind_tau3_competitive_v3_evidence(
    bundle: str | Path,
    *,
    stage: str,
    evidence_path: str | Path,
) -> dict[str, Any]:
    """Append one immutable, bundle-local stage artifact reference to the plan."""

    if stage not in EVIDENCE_STAGES:
        raise Tau3CompetitiveV3BindingError(
            f"stage must be one of {', '.join(EVIDENCE_STAGES)}"
        )
    root = Path(bundle).resolve()
    plan_path = root / PLAN_FILENAME
    plan = _read_binding_json(plan_path, "competitive v3 plan")
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise Tau3CompetitiveV3BindingError(
            f"competitive v3 plan schema_version must be {PLAN_SCHEMA_VERSION}"
        )

    artifact_path = Path(evidence_path).resolve()
    try:
        relative = artifact_path.relative_to(root)
    except ValueError as exc:
        raise Tau3CompetitiveV3BindingError(
            "evidence artifact must resolve inside the bundle"
        ) from exc
    relative_text = relative.as_posix()
    if _is_unsafe_relative_path(relative_text):
        raise Tau3CompetitiveV3BindingError(
            "evidence artifact path must be a safe bundle-relative path"
        )
    artifact = _read_binding_json(artifact_path, f"{stage} evidence artifact")
    expected_schema = EVIDENCE_SCHEMA_BY_STAGE[stage]
    if artifact.get("schema_version") != expected_schema:
        raise Tau3CompetitiveV3BindingError(
            f"{stage} evidence schema_version must be {expected_schema}"
        )
    ref = {"path": relative_text, "sha256": _sha256_file(artifact_path)}

    raw_refs = plan.get("evidence_refs")
    if raw_refs is None:
        refs: dict[str, Any] = {}
    elif isinstance(raw_refs, dict):
        refs = dict(raw_refs)
    else:
        raise Tau3CompetitiveV3BindingError(
            "competitive v3 plan evidence_refs must be an object"
        )
    existing = refs.get(stage)
    if existing is not None and existing != ref:
        raise Tau3CompetitiveV3BindingError(
            f"refusing to replace immutable evidence_refs.{stage}"
        )
    changed = existing is None
    if changed:
        refs[stage] = ref
        plan["evidence_refs"] = refs
        _write_binding_json_atomic(plan_path, plan)

    return {
        "schema_version": EVIDENCE_BINDING_SCHEMA_VERSION,
        "passed": True,
        "changed": changed,
        "stage": stage,
        "evidence_ref": ref,
        "plan_path": str(plan_path),
        "plan_sha256": _sha256_file(plan_path),
    }


def validate_tau3_competitive_v3_bundle(
    bundle: str | Path,
    *,
    strict: bool = False,
    stage: str = "plan",
) -> dict[str, Any]:
    """Validate a Tau-3 competitive-agent v3 plan or evidence bundle."""

    if stage not in STAGES:
        raise ValueError(f"stage must be one of {', '.join(STAGES)}")
    root = Path(bundle)
    targets: list[_Target] = []
    plan = _load_plan(root)
    targets.append(plan.target)
    if isinstance(plan.payload, dict):
        _validate_plan(root, plan.target, plan.payload, allow_private_local=stage == "plan")
    if stage in ("dataset", "training", "final"):
        dataset = _load_stage_ref(root, plan.payload, "dataset", DATASET_SCHEMA_VERSION)
        targets.append(dataset.target)
        if isinstance(dataset.payload, dict):
            _validate_dataset_evidence(root, dataset.target, dataset.payload)
    if stage in ("training", "final"):
        training = _load_stage_ref(root, plan.payload, "training", TRAINING_SCHEMA_VERSION)
        targets.append(training.target)
        if isinstance(training.payload, dict):
            _validate_training_evidence(root, training.target, training.payload)
    if stage == "final":
        final = _load_stage_ref(root, plan.payload, "final", FINAL_SCHEMA_VERSION)
        targets.append(final.target)
        if isinstance(final.payload, dict):
            _validate_final_evidence(root, final.target, final.payload)
        publication = _load_stage_ref(root, plan.payload, "publication", PUBLICATION_SCHEMA_VERSION)
        targets.append(publication.target)
        if isinstance(publication.payload, dict):
            _validate_publication(root, publication.target, publication.payload)
    if strict and stage != "final":
        strict_target = _Target("strict_stage", str(root))
        strict_target.errors.append("strict validation requires --stage final")
        targets.append(strict_target)
    return _summary(strict, targets)


def _load_plan(root: Path) -> _Loaded:
    path = root / PLAN_FILENAME
    target = _Target("competitive_v3_plan", str(path))
    if not path.is_file():
        target.errors.append(f"missing {PLAN_FILENAME}")
        return _Loaded(target)
    return _load_json_path(path, target)


def _validate_plan(root: Path, target: _Target, plan: dict[str, Any], *, allow_private_local: bool) -> None:
    _require(target, plan.get("schema_version") == PLAN_SCHEMA_VERSION, f"plan schema_version must be {PLAN_SCHEMA_VERSION}")

    mission = _dict(plan.get("mission_contract"))
    _require(target, mission.get("mission_id") == "tau3-competitive-agent-v3", "mission_contract.mission_id must be tau3-competitive-agent-v3")
    _require_sha(target, mission.get("mission_statement_sha256"), "mission_contract.mission_statement_sha256")
    _validate_ref(root, target, mission.get("rubric"), "mission_contract.rubric", allow_private_local=allow_private_local)
    _validate_ref(root, target, mission.get("protocol"), "mission_contract.protocol", allow_private_local=allow_private_local)

    v2 = _dict(plan.get("v2_negative_evidence"))
    _require(target, v2.get("immutable") is True, "v2_negative_evidence.immutable must be true")
    _require(target, v2.get("rewrites_v2") is False, "v2_negative_evidence.rewrites_v2 must be false")
    blocked = _validate_json_ref(
        root,
        target,
        v2.get("blocked_verdict"),
        "v2_negative_evidence.blocked_verdict",
        allow_private_local=allow_private_local,
    )
    if isinstance(blocked, dict):
        _require(target, blocked.get("slug") == "tau3-core-qlora-training", "v2 blocked verdict must bind tau3-core-qlora-training")
        _require(target, blocked.get("verdict") == "blocked", "v2 predecessor verdict must remain blocked")
        _require(target, blocked.get("passed") is False, "v2 predecessor passed flag must remain false")
    candidate_c = _dict(v2.get("candidate_c"))
    _validate_ref(root, target, candidate_c.get("training_receipt"), "v2_negative_evidence.candidate_c.training_receipt", allow_private_local=allow_private_local)
    _validate_ref(root, target, candidate_c.get("development_result"), "v2_negative_evidence.candidate_c.development_result", allow_private_local=allow_private_local)

    lineage = _dict(plan.get("lineage"))
    _require(target, lineage.get("dataset_id") == "tau3-competitive-agent-v3", "lineage.dataset_id must be tau3-competitive-agent-v3")
    _require(target, lineage.get("version") == "v3", "lineage.version must be v3")
    _require(target, lineage.get("predecessor") == "v2", "lineage.predecessor must be v2")
    _require(target, lineage.get("new_lineage") is True, "lineage.new_lineage must be true")

    sealed = _dict(plan.get("sealed_access"))
    _require(target, sealed.get("payload_access_count") == 0, "sealed_access.payload_access_count must be zero")
    _require(target, sealed.get("materialized_sealed_fields") == [], "sealed_access.materialized_sealed_fields must be empty")
    _require_no_private_or_sealed_payloads(target, plan)

    models = _dict(plan.get("models"))
    harness = _dict(plan.get("harness_contract"))
    _validate_models(target, models)
    _validate_harness(root, target, harness, allow_private_local=allow_private_local)
    _validate_evaluator_model_contract(root, target, models, harness)
    _validate_statistical_contract(target, _dict(plan.get("statistical_contract")))
    _validate_budget_contract(target, _dict(plan.get("budget_contract")))
    _validate_dataset_contract(target, _dict(plan.get("dataset_contract")))
    _validate_recipe_contract(target, _dict(plan.get("recipe_contract")))
    _validate_development_contract(target, _dict(plan.get("development_contract")))
    _validate_publication_contract(target, _dict(plan.get("publication_contract")))

    refs = _dict(plan.get("evidence_refs"))
    for name in ("dataset", "training", "final", "publication"):
        ref = refs.get(name)
        if ref is not None:
            _validate_ref(root, target, ref, f"evidence_refs.{name}", allow_private_local=False, require_exists=False)


def _validate_models(target: _Target, models: dict[str, Any]) -> None:
    base = _dict(models.get("base"))
    _require(target, _between(base.get("parameter_b"), 7.0, 9.0), "models.base.parameter_b must be between 7 and 9")
    _require(target, base.get("selected_by_sealed_blind_preflight") is True, "base must be selected by sealed-blind agent-interface preflight")
    comparators = _list_of_dicts(models.get("comparators"))
    _require(target, len(comparators) == 2, "models.comparators must contain exactly two frozen same-size open comparators")
    _require(target, sorted(str(item.get("arm_id")) for item in comparators) == ["comparator_1", "comparator_2"], "comparators must be comparator_1 and comparator_2")
    for item in comparators:
        label = str(item.get("arm_id") or "comparator")
        _require(target, _between(item.get("parameter_b"), 7.0, 9.0), f"{label}.parameter_b must be between 7 and 9")
        _require(target, item.get("same_size_open") is True, f"{label}.same_size_open must be true")
        _require(target, item.get("frozen") is True, f"{label}.frozen must be true")


def _validate_harness(root: Path, target: _Target, harness: dict[str, Any], *, allow_private_local: bool) -> None:
    for key in ("identical_for_all_arms", "text_mode", "frozen_before_training"):
        _require(target, harness.get(key) is True, f"harness_contract.{key} must be true")
    for key in (
        "tau_repository",
        "harness",
        "tokenizer_chat_template",
        "ordered_tool_catalog",
        "policy_prompt",
        "task_trial_seed_grid",
        "decoding",
        "retry_policy",
        "safety_policy",
    ):
        _validate_ref(root, target, harness.get(key), f"harness_contract.{key}", allow_private_local=allow_private_local)
    _require(target, isinstance(harness.get("context_window"), int) and harness["context_window"] > 0, "harness_contract.context_window must be a positive integer")


def _validate_evaluator_model_contract(root: Path, target: _Target, models: dict[str, Any], harness: dict[str, Any]) -> None:
    model_ref = models.get("evaluator_models")
    harness_ref = harness.get("evaluator_model_contract")
    _require(target, isinstance(model_ref, dict), "models.evaluator_models must reference evaluator model contract")
    _require(target, isinstance(harness_ref, dict), "harness_contract.evaluator_model_contract must reference evaluator model contract")
    if isinstance(model_ref, dict) and isinstance(harness_ref, dict):
        _require(target, model_ref == harness_ref, "models.evaluator_models must exactly match harness_contract.evaluator_model_contract")
        loaded = _load_json_artifact_ref(root, target, model_ref, "models.evaluator_models")
        if isinstance(loaded.payload, dict):
            _validate_evaluator_model_payload(target, loaded.payload)


def _validate_evaluator_model_payload(target: _Target, payload: dict[str, Any]) -> None:
    _require(target, payload.get("schema_version") == "hfr.tau3_evaluator_model_contract.v1", "evaluator model contract schema_version must be hfr.tau3_evaluator_model_contract.v1")
    roles = _dict(payload.get("roles"))
    user = _dict(roles.get("user_simulator"))
    reviewer = _dict(roles.get("reviewer"))
    identities: dict[str, dict[str, Any]] = {}
    for role, record in (("user_simulator", user), ("reviewer", reviewer)):
        _require(target, record.get("role") == role, f"evaluator {role}.role must match role name")
        _require(target, record.get("local_only") is True, f"evaluator {role} must be local_only")
        _require(target, record.get("network") is False, f"evaluator {role} network must be false")
        _require(target, record.get("no_test_time_search") is True, f"evaluator {role} must disable test-time search")
        _require(target, record.get("comparator_specific_prompting") is False, f"evaluator {role} must forbid comparator-specific prompting")
        identity = _dict(record.get("model_identity"))
        identities[role] = identity
        for field_name in ("model_id", "revision", "local_tree_sha256", "local_identity_sha256"):
            _require(target, isinstance(identity.get(field_name), str) and bool(identity.get(field_name)), f"evaluator {role}.model_identity.{field_name} must be pinned")
        _require_revision(target, identity.get("revision"), f"evaluator {role}.model_identity.revision")
        _require_sha(target, identity.get("local_tree_sha256"), f"evaluator {role}.model_identity.local_tree_sha256")
        _require_sha(target, identity.get("local_identity_sha256"), f"evaluator {role}.model_identity.local_identity_sha256")
        _require(target, record.get("model_identity_sha256") == _canonical_sha256(identity), f"evaluator {role}.model_identity_sha256 must replay")
    _require(target, user.get("model_identity") == reviewer.get("model_identity"), "evaluator user_simulator and reviewer must share exact model_identity")
    _require(target, user.get("model_identity_sha256") == reviewer.get("model_identity_sha256"), "evaluator user_simulator and reviewer must share exact model_identity_sha256")
    _require(target, payload.get("roles_share_exact_model") is True, "evaluator contract must declare roles_share_exact_model")
    _require(target, payload.get("identical_for_all_arms") is True, "evaluator contract must be identical_for_all_arms")
    _require(target, payload.get("local_only") is True, "evaluator contract must be local_only")
    _require(target, payload.get("network") is False, "evaluator contract network must be false")
    arms = payload.get("applies_to_arms")
    _require(target, sorted(arms) == ["adapter", "base", "comparator_1", "comparator_2"] if isinstance(arms, list) else False, "evaluator contract must apply to adapter/base/comparator_1/comparator_2")
    _require(target, payload.get("no_comparator_specific_prompting") is True, "evaluator contract must forbid comparator-specific prompting")
    _require(target, payload.get("excluded_from_comparator_claims") is True, "evaluator contract must be excluded from comparator claims")
    _require(target, payload.get("excluded_from_gradient_data") is True, "evaluator contract must be excluded from gradient data")


def _require_revision(target: _Target, value: Any, label: str) -> None:
    _require(target, isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40,64}", value) is not None, f"{label} must be an immutable hex revision")


def _validate_statistical_contract(target: _Target, stats: dict[str, Any]) -> None:
    _require(target, stats.get("procedure") == "paired_bootstrap", "statistical_contract.procedure must be paired_bootstrap")
    _require(target, float(stats.get("confidence_level") or 0.0) >= 0.95, "statistical_contract.confidence_level must be at least 0.95")
    _require(target, stats.get("adapter_improvement_ci_excludes_zero") is True, "adapter improvement CI exclusion must be preregistered")
    _require(target, stats.get("same_size_noninferiority_margin") == 0.05, "same-size noninferiority margin must be exactly 0.05")
    _require(target, stats.get("promotion_predicates_fail_closed") is True, "promotion predicates must fail closed")


def _validate_budget_contract(target: _Target, budget: dict[str, Any]) -> None:
    _require(target, budget.get("unattended_days") == 7, "budget_contract.unattended_days must be exactly 7")
    _require(target, budget.get("separate_infrastructure_attempt_budget") is True, "infrastructure attempt budget must be separate")
    _require(target, budget.get("separate_qualified_candidate_budget") is True, "qualified candidate budget must be separate")
    _require(target, int(budget.get("minimum_qualified_candidates") or 0) >= 2, "at least two qualified candidates must complete")
    _require(target, int(budget.get("target_qualified_candidates_min") or 0) >= 3, "target qualified candidate minimum must be at least 3")
    _require(target, int(budget.get("target_qualified_candidates_max") or 0) <= 4, "target qualified candidate maximum must be no more than 4")


def _validate_dataset_contract(target: _Target, contract: dict[str, Any]) -> None:
    coverage = _dict(contract.get("coverage_gates"))
    _require(target, coverage.get("all_domains_train_and_internal_validation") is True, "dataset coverage must include every domain in train and internal validation")
    _require(target, int(coverage.get("min_train_targets_per_tool") or 0) >= 16, "min_train_targets_per_tool must be at least 16")
    _require(target, int(coverage.get("min_internal_validation_targets_per_tool") or 0) >= 4, "min_internal_validation_targets_per_tool must be at least 4")
    _require(target, int(coverage.get("min_train_argument_payloads_per_tool") or 0) >= 8, "min_train_argument_payloads_per_tool must be at least 8")
    _require(target, int(coverage.get("min_internal_validation_argument_payloads_per_tool") or 0) >= 2, "min_internal_validation_argument_payloads_per_tool must be at least 2")
    _require(target, coverage.get("all_required_behaviors_per_domain") is True, "all required behavior strata must be represented per domain")
    _require(target, float(coverage.get("telecom_min_training_example_fraction") or 0.0) >= 0.25, "telecom example fraction must be at least 25%")
    _require(target, float(coverage.get("telecom_min_supervised_token_fraction") or 0.0) >= 0.25, "telecom supervised-token fraction must be at least 25%")
    _require(target, float(coverage.get("domain_supervised_token_min_fraction") or 0.0) >= 0.25, "domain supervised-token min fraction must be at least 25%")
    _require(target, float(coverage.get("domain_supervised_token_max_fraction") or 1.0) <= 0.40, "domain supervised-token max fraction must be at most 40%")
    _require(target, float(coverage.get("max_domain_canonical_target_duplication_fraction") or 1.0) <= 0.20, "domain target duplication fraction must be at most 0.20")
    _require(target, coverage.get("split_hashes_disjoint") is True, "train/internal/development/sealed hashes must remain disjoint")

    sampling = _dict(contract.get("sampling_contract"))
    _require(target, sampling.get("deterministic_receipt_producing") is True, "sampling must be deterministic and receipt-producing")
    _require(target, sampling.get("records_per_step_exposure") is True, "sampling must record per-step exposure")
    _require(target, sampling.get("replay_from_dataset_hash_config_seed") is True, "sampling must replay from dataset hash, sampler config, and seed")

    objective = _dict(contract.get("objective_contract"))
    for key in (
        "supervises_every_eligible_assistant_decision",
        "masks_prompt_tool_result_negative_user_private_reference_and_grader_tokens",
        "retains_parent_trajectory_and_decision_ordinal",
        "negative_actions_masked_with_safe_correction_only",
    ):
        _require(target, objective.get(key) is True, f"objective_contract.{key} must be true")


def _validate_recipe_contract(target: _Target, recipe: dict[str, Any]) -> None:
    _require(target, int(recipe.get("minimum_rank") or 0) >= 16, "recipe minimum rank must be at least 16")
    _require(target, int(recipe.get("minimum_adapted_layers") or 0) >= 8, "recipe minimum adapted layers must be at least 8")
    _require(target, int(recipe.get("minimum_effective_batch_examples") or 0) >= 4, "effective batch must be at least four examples or equivalent")
    _require(target, int(recipe.get("minimum_effective_epochs") or 0) >= 2, "qualified candidates must plan at least two effective epochs")
    _require(target, recipe.get("token_loss_is_selection_metric") is False, "token loss may not be the selection metric")
    _require(target, recipe.get("candidate_c_stronger_search_space") is True, "v3 search space must be materially stronger than Candidate C")


def _validate_development_contract(target: _Target, dev: dict[str, Any]) -> None:
    _require(target, dev.get("development_only_selection") is True, "development set may be used only for evaluation and selection")
    _require(target, dev.get("identical_harness") is True, "development comparisons must use the identical harness")
    _require(target, dev.get("no_comparator_specific_prompting") is True, "comparator-specific prompting must be forbidden")
    gates = _dict(dev.get("qualification_gate"))
    _require(target, float(gates.get("macro_pass1_min") or 0.0) >= 0.10, "development macro Pass-1 gate must be at least 0.10")
    _require(target, float(gates.get("per_domain_pass1_min") or 0.0) >= 0.05, "per-domain Pass-1 gate must be at least 0.05")
    _require(target, float(gates.get("macro_base_improvement_min") or 0.0) >= 0.05, "macro base improvement gate must be at least 0.05")


def _validate_publication_contract(target: _Target, publication: dict[str, Any]) -> None:
    _require(target, publication.get("competitive_claims_fail_closed") is True, "competitive claims must fail closed")
    _require(target, publication.get("sealed_payloads_public") is False, "sealed payloads must not be public")
    _require(target, publication.get("redaction_required") is True, "publication must require redaction")


def _validate_dataset_evidence(root: Path, target: _Target, evidence: dict[str, Any]) -> None:
    _require(target, evidence.get("schema_version") == DATASET_SCHEMA_VERSION, f"dataset evidence schema_version must be {DATASET_SCHEMA_VERSION}")
    refs = _dict(evidence.get("artifacts"))
    dataset_manifest = _load_json_artifact_ref(root, target, refs.get("competitive_dataset_manifest"), "artifacts.competitive_dataset_manifest")
    dataset_dir = dataset_manifest.path.parent if dataset_manifest.path is not None else None
    if isinstance(dataset_manifest.payload, dict):
        _require(target, dataset_manifest.payload.get("lineage_id") == "tau3-competitive-agent-v3", "competitive dataset manifest lineage_id must be tau3-competitive-agent-v3")
        sealed = _dict(dataset_manifest.payload.get("sealed_access"))
        _require(target, sealed.get("payload_accessed") is False and sealed.get("access_count") == 0, "competitive dataset manifest must prove zero sealed access")
    if dataset_dir is not None:
        dataset_validation = _validate_competitive_dataset_with_tau_bridge(dataset_dir)
        _require(target, dataset_validation.get("passed") is True, "competitive dataset semantic validation must pass")
        if dataset_validation.get("passed") is not True:
            for error in _list_strings(dataset_validation.get("errors"))[:8]:
                target.errors.append(f"competitive dataset semantic validation detail: {error}")
        _validate_saved_validation_receipt(root, target, refs.get("competitive_dataset_validation"), "artifacts.competitive_dataset_validation", expected_schema="hfr.tau3_competitive_dataset_validation.v1")
    objective = _load_json_artifact_ref(root, target, refs.get("objective_validity_report"), "artifacts.objective_validity_report")
    if objective.path is not None:
        objective_validation = validate_tau3_objective_validity_report(objective.path)
        _require(target, objective_validation.get("passed") is True, "objective-validity semantic validation must pass")
        _validate_saved_validation_receipt(root, target, refs.get("objective_validity_validation"), "artifacts.objective_validity_validation", expected_schema="hfr.tau3_objective_validity_validation.v1")


def _validate_competitive_dataset_with_tau_bridge(dataset_dir: Path) -> dict[str, Any]:
    direct = validate_tau3_competitive_dataset_bundle(dataset_dir, strict=True)
    if direct.get("passed") is True or not _dataset_replay_needs_tau_bridge(direct):
        return direct
    python, python_error = _repository_local_tau_python()
    direct_errors = _list_strings(direct.get("errors"))
    if python_error is not None or python is None:
        return {
            **direct,
            "passed": False,
            "errors": [
                *direct_errors,
                "repository-local Tau python bridge unavailable: " + str(python_error),
            ],
        }
    external = validate_tau3_competitive_dataset_bundle(dataset_dir, strict=True, grounded_validator_python=python)
    if external.get("passed") is True:
        return external
    return {
        **external,
        "passed": False,
        "errors": [
            "direct strict dataset replay failed before Tau bridge",
            *direct_errors[:8],
            "repository-local Tau python bridge replay failed",
            *_list_strings(external.get("errors"))[:8],
        ],
    }


def _dataset_replay_needs_tau_bridge(result: dict[str, Any]) -> bool:
    text = "\n".join(_list_strings(result.get("errors"))).lower()
    if not text:
        return False
    tau_import_markers = (
        "cannot import vendored tau",
        "cannot instantiate vendored tau",
        "vendored tau runtime",
        "pydantic",
        "no module named 'pydantic'",
        "no module named \"pydantic\"",
    )
    return any(marker in text for marker in tau_import_markers)


def _repository_local_tau_python() -> tuple[Path | None, str | None]:
    python = _project_root() / "local" / "tau3" / "venv" / "bin" / "python3"
    if not python.is_file():
        return None, f"missing executable: {python}"
    if not os.access(python, os.X_OK):
        return None, f"not executable: {python}"
    return python, None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _list_strings(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _validate_training_evidence(root: Path, target: _Target, evidence: dict[str, Any]) -> None:
    _require(target, evidence.get("schema_version") == TRAINING_SCHEMA_VERSION, f"training evidence schema_version must be {TRAINING_SCHEMA_VERSION}")
    _check_registered_schema(
        target,
        evidence,
        "tau3_competitive_v3_training_evidence",
        "training evidence",
    )
    shared_exposure_hashes: tuple[str | None, str | None] | None = None
    if evidence.get("exposure") is not None:
        shared_exposure_hashes = _validate_training_exposure(
            root,
            target,
            evidence.get("exposure"),
            "exposure",
        )

    _require(target, _dict(evidence.get("budgets")).get("separate_candidate_and_infra_budgets") is True, "candidate and infrastructure budgets must remain separate")
    candidates = _list_of_dicts(evidence.get("qualified_candidates"))
    _require(target, len(candidates) >= 2, "qualified_candidates must include at least two final MLX training receipts")
    candidate_ids: set[str] = set()
    recipe_hashes: set[str] = set()
    adapter_hashes: set[str] = set()
    qualified_count = 0
    for candidate in candidates:
        label = str(candidate.get("candidate_id") or "candidate")
        candidate_exposure = candidate.get("exposure")
        if candidate_exposure is not None:
            exposure_receipt_sha256, exposure_ledger_sha256 = (
                _validate_training_exposure(
                    root,
                    target,
                    candidate_exposure,
                    f"qualified_candidates.{label}.exposure",
                )
            )
        elif shared_exposure_hashes is not None:
            (
                exposure_receipt_sha256,
                exposure_ledger_sha256,
            ) = shared_exposure_hashes
        else:
            target.errors.append(
                f"qualified_candidates.{label} must include exposure evidence "
                "or use shared exposure"
            )
            exposure_receipt_sha256 = None
            exposure_ledger_sha256 = None
        receipt_ref = _load_json_artifact_ref(root, target, candidate.get("training_receipt"), f"qualified_candidates.{label}.training_receipt")
        if not isinstance(receipt_ref.payload, dict) or receipt_ref.path is None:
            continue
        candidate_ids.add(label)
        equivalence_ref: _Loaded | None = None
        if candidate.get("prefix_equivalence") is not None:
            equivalence_ref = _load_json_artifact_ref(
                root,
                target,
                candidate.get("prefix_equivalence"),
                f"qualified_candidates.{label}.prefix_equivalence",
            )
        _validate_training_receipt(
            target,
            receipt_ref.path,
            receipt_ref.payload,
            exposure_receipt_sha256=exposure_receipt_sha256,
            exposure_ledger_sha256=exposure_ledger_sha256,
            prefix_equivalence=equivalence_ref,
        )
        recipe_sha256 = _nested(receipt_ref.payload, "training_binding", "recipe", "recipe_sha256")
        adapter_sha256 = _nested(receipt_ref.payload, "adapter", "tree_sha256")
        if _validate_development_qualification(root, target, candidate, label, receipt_ref, adapter_sha256):
            qualified_count += 1
        if isinstance(recipe_sha256, str):
            recipe_hashes.add(recipe_sha256)
        if isinstance(adapter_sha256, str):
            adapter_hashes.add(adapter_sha256)
    _require(target, len(candidate_ids) >= 2, "qualified candidates must be distinct")
    _require(target, qualified_count >= 2, "at least two candidates must pass development qualification gates")
    _require(target, len(recipe_hashes) >= 2, "qualified candidates must prove recipe diversity")
    _require(target, len(adapter_hashes) >= 2, "qualified candidates must bind distinct adapter fingerprints")


def _validate_training_exposure(
    root: Path,
    target: _Target,
    value: Any,
    label: str,
) -> tuple[str | None, str | None]:
    exposure = _dict(value)
    dataset = _load_ref_path(
        root,
        target,
        exposure.get("dataset"),
        f"{label}.dataset",
    )
    receipt = _load_json_artifact_ref(
        root,
        target,
        exposure.get("receipt"),
        f"{label}.receipt",
    )
    ledger = _load_ref_path(
        root,
        target,
        exposure.get("ledger"),
        f"{label}.ledger",
    )
    ledger_sha256 = (
        _sha256_file(ledger)
        if ledger is not None and ledger.is_file()
        else None
    )
    replay: dict[str, Any] | None = None
    if dataset is not None and receipt.path is not None and ledger is not None:
        try:
            replay = validate_tau3_exposure_ledger(
                dataset,
                receipt.path,
                ledger,
            )
        except (
            Tau3ExposureError,
            OSError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            target.errors.append(
                f"{label} training exposure ledger does not replay: {exc}"
            )
        if replay is not None:
            _require(
                target,
                replay.get("passed") is True,
                f"{label} training exposure validation must pass",
            )
            _require(
                target,
                replay.get("candidate_eligible") is True,
                f"{label} training exposure receipt must be candidate-eligible",
            )
            _require(
                target,
                replay.get("receipt_sha256") == receipt.sha256,
                f"{label} training exposure receipt hash must bind replayed "
                "validation",
            )
            _require(
                target,
                replay.get("ledger_sha256") == ledger_sha256,
                f"{label} training exposure ledger hash must bind replayed "
                "validation",
            )
    saved = _load_json_artifact_ref(
        root,
        target,
        exposure.get("validation"),
        f"{label}.validation",
    )
    if isinstance(saved.payload, dict):
        _require(
            target,
            saved.payload.get("schema_version")
            == "hfr.tau3_training_exposure_validation.v1",
            f"{label}.validation schema_version must be "
            "hfr.tau3_training_exposure_validation.v1",
        )
        _require(
            target,
            saved.payload.get("passed") is True,
            f"{label}.validation must have passed=true from a saved "
            "validator receipt",
        )
        if replay is not None:
            for field in (
                "candidate_eligible",
                "ledger_sha256",
                "receipt_sha256",
                "row_count",
                "step_count",
            ):
                _require(
                    target,
                    saved.payload.get(field) == replay.get(field),
                    f"{label}.validation {field} must replay",
                )
    return receipt.sha256, ledger_sha256


def _validate_final_evidence(root: Path, target: _Target, evidence: dict[str, Any]) -> None:
    _require(target, evidence.get("schema_version") == FINAL_SCHEMA_VERSION, f"final evidence schema_version must be {FINAL_SCHEMA_VERSION}")
    _validate_chronology(target, evidence)
    artifacts = _dict(evidence.get("artifacts"))
    selection = _load_json_artifact_ref(root, target, artifacts.get("candidate_selection"), "artifacts.candidate_selection")
    identity = _load_json_artifact_ref(root, target, artifacts.get("candidate_identity"), "artifacts.candidate_identity")
    lock = _load_json_artifact_ref(root, target, artifacts.get("candidate_lock"), "artifacts.candidate_lock")
    authorization = _load_json_artifact_ref(root, target, artifacts.get("sealed_authorization"), "artifacts.sealed_authorization")
    grid = _load_json_artifact_ref(root, target, artifacts.get("sealed_grid_completeness"), "artifacts.sealed_grid_completeness")
    evaluation = _load_json_artifact_ref(root, target, artifacts.get("sealed_evaluation"), "artifacts.sealed_evaluation")
    promotion = _load_json_artifact_ref(root, target, artifacts.get("promotion_preflight"), "artifacts.promotion_preflight")

    _check_loaded_schema(target, selection, "tau3_candidate_selection", "candidate selection")
    _check_loaded_schema(target, identity, "tau3_candidate_identity", "candidate identity")
    _check_loaded_schema(target, lock, "tau3_candidate_lock", "candidate lock")
    _check_loaded_schema(target, authorization, "tau3_sealed_authorization", "sealed authorization")
    _check_loaded_schema(target, grid, "tau3_sealed_grid_completeness", "sealed grid completeness")
    _check_loaded_schema(target, promotion, "tau3_promotion_publication_preflight", "promotion preflight")
    if evaluation.path is not None:
        eval_result = validate_tau3_evaluation_report(evaluation.path)
        _require(target, eval_result.get("passed") is True, "sealed evaluation report validation must pass")
    if isinstance(evaluation.payload, dict):
        _require(target, evaluation.payload.get("mode") == "sealed", "evaluation report must be sealed mode")
        _require(target, evaluation.payload.get("passed") is True and evaluation.payload.get("promotion_ready") is True, "sealed evaluation must pass promotion-ready gates")
        _require(target, _candidate_beats_base(evaluation.payload), "sealed evaluation must prove adapter beats base")
        _require(target, _within_strongest_comparator_margin(evaluation.payload), "sealed evaluation must prove same-size comparator noninferiority margin")
    if isinstance(grid.payload, dict):
        _require(target, grid.payload.get("passed") is True and grid.payload.get("status") == "complete", "sealed grid completeness must pass")
        bindings = _dict(grid.payload.get("bindings"))
        if authorization.sha256 is not None:
            _require(target, bindings.get("authorization_sha256") == authorization.sha256, "sealed grid must bind authorization sha256")
        if lock.sha256 is not None:
            _require(target, bindings.get("candidate_lock_sha256") == lock.sha256, "sealed grid must bind candidate lock sha256")
    if isinstance(lock.payload, dict):
        if selection.sha256 is not None:
            _require(target, lock.payload.get("development_selection_report_sha256") == selection.sha256, "candidate lock must bind selection report sha256")
        if identity.sha256 is not None:
            _require(target, lock.payload.get("candidate_identity_sha256") == identity.sha256, "candidate lock must bind identity sha256")
    if isinstance(identity.payload, dict) and isinstance(lock.payload, dict):
        _require(target, identity.payload.get("training_receipt_sha256") == lock.payload.get("training_receipt_sha256"), "candidate identity must bind locked training receipt sha256")
        _require(target, identity.payload.get("adapter_tree_sha256") == lock.payload.get("adapter_tree_sha256"), "candidate identity must bind locked adapter tree sha256")
    auth_inputs = _dict(evidence.get("sealed_authorization_validation"))
    if authorization.path is not None and lock.path is not None and auth_inputs:
        protocol = _load_ref_path(root, target, auth_inputs.get("protocol"), "sealed_authorization_validation.protocol")
        sealed_source = _load_ref_path(root, target, auth_inputs.get("sealed_source_manifest"), "sealed_authorization_validation.sealed_source_manifest")
        seeds = tuple(int(seed) for seed in auth_inputs.get("seeds", []) if isinstance(seed, int))
        if protocol is not None and sealed_source is not None:
            try:
                auth_result = validate_tau3_sealed_authorization(
                    authorization_path=authorization.path,
                    candidate_lock_path=lock.path,
                    protocol_path=protocol,
                    sealed_source_manifest_path=sealed_source,
                    arm_id=str(auth_inputs.get("arm_id") or "adapter"),
                    seeds=seeds,
                    expected_tau_revision=str(auth_inputs.get("expected_tau_revision") or ""),
                    expected_authorization_sha256=authorization.sha256,
                )
                _require(target, auth_result.get("authorized") is True, "sealed authorization replay must authorize")
            except (Tau3SealedAuthorizationError, OSError, ValueError, json.JSONDecodeError) as exc:
                target.errors.append(f"sealed authorization does not replay: {exc}")
    elif authorization.path is not None:
        target.errors.append("sealed_authorization_validation inputs are required to replay sealed authorization")
    if isinstance(promotion.payload, dict):
        _require(target, promotion.payload.get("allowed") is True, "promotion preflight must be allowed")
        _require(target, promotion.payload.get("publication_status") == "ready_for_publication", "promotion preflight must be ready_for_publication")
        if lock.sha256 is not None:
            _require(target, _nested(promotion.payload, "evidence_bindings", "candidate_lock", "sha256") == lock.sha256, "promotion preflight must bind candidate lock sha256")


def _validate_publication(root: Path, target: _Target, publication: dict[str, Any]) -> None:
    _require(target, publication.get("schema_version") == PUBLICATION_SCHEMA_VERSION, f"publication schema_version must be {PUBLICATION_SCHEMA_VERSION}")
    artifacts = _dict(publication.get("artifacts"))
    redaction = _load_json_artifact_ref(root, target, artifacts.get("redaction"), "artifacts.redaction")
    post_publication = _load_json_artifact_ref(root, target, artifacts.get("post_publication"), "artifacts.post_publication")
    promotion_preflight = _load_json_artifact_ref(root, target, artifacts.get("promotion_preflight"), "artifacts.promotion_preflight")
    if isinstance(redaction.payload, dict):
        _require(target, redaction.payload.get("passed") is True or redaction.payload.get("status") == "passed", "redaction artifact must pass")
        _require(target, redaction.payload.get("contains_sealed_payloads") is False, "redaction artifact must prove no sealed payloads")
        _require(target, redaction.payload.get("contains_credentials") is False, "redaction artifact must prove no credentials")
        _require(target, redaction.payload.get("contains_private_paths") is False, "redaction artifact must prove no private paths")
    if isinstance(post_publication.payload, dict):
        _check_registered_schema(target, post_publication.payload, "tau3_post_publication_record", "post-publication record")
        _require(target, post_publication.payload.get("status") == "published", "post-publication record must be published")
        _require(target, _canonical_sha256({key: value for key, value in post_publication.payload.items() if key != "record_sha256"}) == post_publication.payload.get("record_sha256"), "post-publication record_sha256 must replay")
        hf = _dict(post_publication.payload.get("huggingface"))
        revision = hf.get("revision")
        _require(target, isinstance(revision, str) and hashlib.sha256(revision.encode("utf-8")).hexdigest() == hf.get("revision_sha256"), "post-publication HF revision hash must replay")
        preflight = _dict(post_publication.payload.get("preflight"))
        if promotion_preflight.sha256 is not None:
            _require(target, preflight.get("sha256") == promotion_preflight.sha256, "post-publication record must bind promotion preflight sha256")
        if isinstance(promotion_preflight.payload, dict):
            _check_registered_schema(target, promotion_preflight.payload, "tau3_promotion_publication_preflight", "publication promotion preflight")
            _require(target, preflight.get("decision_sha256") == promotion_preflight.payload.get("decision_sha256"), "post-publication record must bind promotion decision sha256")
            _require(target, preflight.get("allowed") is True and promotion_preflight.payload.get("allowed") is True, "post-publication requires allowed promotion preflight")
            _require(target, preflight.get("publication_status") == "ready_for_publication", "post-publication requires ready_for_publication preflight")
    parity = _load_json_artifact_ref(root, target, artifacts.get("source_parity"), "artifacts.source_parity")
    if isinstance(parity.payload, dict):
        _validate_source_parity(target, parity.payload, post_publication=post_publication, promotion_preflight=promotion_preflight)


def _validate_source_parity(target: _Target, parity: dict[str, Any], *, post_publication: _Loaded, promotion_preflight: _Loaded) -> None:
    _require(target, parity.get("passed") is True, "source parity artifact must pass")
    github_revision = parity.get("github_revision")
    hf_revision = parity.get("hf_revision")
    _require(target, isinstance(github_revision, str) and re.fullmatch(r"[0-9a-f]{40,64}", github_revision) is not None, "source parity GitHub revision must be immutable hex")
    _require(target, isinstance(hf_revision, str) and re.fullmatch(r"[0-9a-f]{40,64}", hf_revision) is not None, "source parity HF revision must be immutable hex")
    reviewed = _dict(parity.get("reviewed_evidence_source"))
    _require(target, reviewed.get("github_revision") == github_revision, "source parity reviewed evidence must bind GitHub revision")
    _require(target, reviewed.get("hf_revision") == hf_revision, "source parity reviewed evidence must bind HF revision")
    bindings = _dict(parity.get("artifact_hash_bindings"))
    _require(target, bool(bindings), "source parity must include artifact_hash_bindings")
    if post_publication.sha256 is not None:
        _require(target, bindings.get("post_publication_record_sha256") == post_publication.sha256, "source parity must bind post-publication record hash")
    if promotion_preflight.sha256 is not None:
        _require(target, bindings.get("promotion_preflight_sha256") == promotion_preflight.sha256, "source parity must bind promotion preflight hash")
    _require_sha(target, bindings.get("evidence_bundle_sha256"), "source parity artifact_hash_bindings.evidence_bundle_sha256")


def _validate_chronology(target: _Target, evidence: dict[str, Any]) -> None:
    lock_time = _required_time(target, evidence, "candidate_locked_at")
    sealed_time = _required_time(target, evidence, "sealed_started_at")
    publication_time = _required_time(target, evidence, "publication_preflight_at")
    if lock_time and sealed_time:
        _require(target, lock_time < sealed_time, "candidate lock must predate sealed access")
    if sealed_time and publication_time:
        _require(target, sealed_time < publication_time, "sealed evaluation must predate publication preflight")


def _required_time(target: _Target, evidence: dict[str, Any], field: str) -> datetime | None:
    value = evidence.get(field)
    parsed = _parse_time(value)
    _require(target, parsed is not None, f"{field} must be an ISO-8601 timestamp")
    return parsed


def _load_stage_ref(root: Path, plan: Any, name: str, expected_schema_version: str) -> _Loaded:
    ref = _dict(_dict(plan).get("evidence_refs")).get(name)
    target = _Target(f"competitive_v3_{name}", str(root))
    if not isinstance(ref, dict):
        target.errors.append(f"missing evidence_refs.{name}")
        return _Loaded(target)
    loaded = _load_ref(root, ref, f"evidence_refs.{name}", allow_private_local=False)
    loaded.target.type = f"competitive_v3_{name}"
    if isinstance(loaded.payload, dict):
        _require(loaded.target, loaded.payload.get("schema_version") == expected_schema_version, f"{name} evidence must be {expected_schema_version}")
    return loaded


def _load_ref(root: Path, ref: dict[str, Any], label: str, *, allow_private_local: bool) -> _Loaded:
    target = _Target(label, str(root))
    path = _resolve_ref_path(root, target, ref, label, allow_private_local=allow_private_local, require_exists=True)
    if path is None:
        return _Loaded(target)
    target.path = str(path)
    loaded = _load_json_path(path, target)
    if loaded.sha256 is not None and ref.get("sha256") != loaded.sha256:
        target.errors.append(f"{label} sha256 mismatch")
    return loaded


def _load_json_artifact_ref(root: Path, target: _Target, ref: Any, label: str) -> _Loaded:
    if not isinstance(ref, dict):
        target.errors.append(f"{label} must be a ref object")
        missing = _Target(label, str(root))
        missing.errors.append(f"{label} must be a ref object")
        return _Loaded(missing)
    loaded = _load_ref(root, ref, label, allow_private_local=False)
    target.errors.extend(loaded.target.errors)
    target.warnings.extend(loaded.target.warnings)
    return loaded


def _load_ref_path(root: Path, target: _Target, ref: Any, label: str) -> Path | None:
    if not isinstance(ref, dict):
        target.errors.append(f"{label} must be a ref object")
        return None
    return _resolve_ref_path(root, target, ref, label, allow_private_local=False, require_exists=True)


def _validate_saved_validation_receipt(
    root: Path,
    target: _Target,
    ref: Any,
    label: str,
    *,
    expected_schema: str,
) -> None:
    loaded = _load_json_artifact_ref(root, target, ref, label)
    if isinstance(loaded.payload, dict):
        _require(target, loaded.payload.get("schema_version") == expected_schema, f"{label} schema_version must be {expected_schema}")
        _require(target, loaded.payload.get("passed") is True, f"{label} must have passed=true from a saved validator receipt")


def _check_loaded_schema(target: _Target, loaded: _Loaded, schema_name: str, label: str) -> None:
    if isinstance(loaded.payload, dict):
        _check_registered_schema(target, loaded.payload, schema_name, label)


def _check_registered_schema(target: _Target, payload: dict[str, Any], schema_name: str, label: str) -> None:
    try:
        result = check_schema_contract(payload, name_or_id=schema_name)
    except SchemaRegistryError as exc:
        target.errors.append(f"{label} schema {schema_name} is not registered: {exc}")
        return
    _require(target, result.get("passed") is True, f"{label} schema check failed: {result.get('errors')}")


def _validate_training_receipt(
    target: _Target,
    path: Path,
    receipt: dict[str, Any],
    *,
    exposure_receipt_sha256: str | None,
    exposure_ledger_sha256: str | None,
    prefix_equivalence: _Loaded | None = None,
) -> None:
    _check_registered_schema(target, receipt, "tau3_mlx_training_run", "qualified training receipt")
    _require(target, receipt.get("phase") == "final", "qualified training receipt must be final")
    _require(target, receipt.get("terminal_status") == "success", "qualified training receipt must finish with terminal_status=success")
    _require(target, receipt.get("weights_updated") is True, "qualified training receipt must prove weights_updated=true")
    _require(target, int(receipt.get("adapter_weight_file_count") or 0) > 0, "qualified training receipt must contain adapter weights")
    adapter = _dict(receipt.get("adapter"))
    _require_sha(target, adapter.get("tree_sha256"), "qualified training receipt adapter.tree_sha256")
    _require(target, _adapter_files_replay(path.parent, adapter), "qualified training receipt adapter file fingerprints must replay")
    config = _dict(receipt.get("config"))
    process_segments = receipt.get("process_segments")
    segmented = config.get("process_segment_iters") is not None
    _require(
        target,
        segmented == isinstance(process_segments, dict),
        "qualified training receipt process_segments presence must match config",
    )
    if segmented and isinstance(process_segments, dict):
        process_validation = validate_tau3_process_segments(
            process_segments,
            output_dir=path.parent,
            expected_config=config,
        )
        _require(
            target,
            process_validation.get("passed") is True,
            "qualified training receipt process segment chain must replay: "
            + json.dumps(
                process_validation.get("errors") or [],
                sort_keys=True,
            ),
        )
    binding = _dict(receipt.get("training_binding"))
    recipe = _dict(binding.get("recipe"))
    exposure = _dict(binding.get("exposure"))
    objective = _dict(exposure.get("objective"))
    _require(target, recipe.get("exposure_ledger_training") is True, "qualified training receipt must use exposure-ledger training")
    full_gradient = (
        recipe.get("full_gradient_objective") is True
        and objective.get("full_gradient") is True
        and objective.get("detached_prefix") is not True
    )
    detached_prefix = (
        recipe.get("full_gradient_objective") is False
        and recipe.get("prefix_cache_training") is True
        and recipe.get("prefix_equivalence_required") is True
        and recipe.get("prefix_equivalence_passed") is True
        and objective.get("full_gradient") is False
        and objective.get("detached_prefix") is True
    )
    _require(
        target,
        full_gradient or detached_prefix,
        "qualified training receipt must use full-gradient objective or "
        "a passing bound detached-prefix equivalence",
    )
    if detached_prefix:
        _require(
            target,
            prefix_equivalence is not None
            and isinstance(prefix_equivalence.payload, dict)
            and prefix_equivalence.path is not None,
            "detached-prefix candidate must include a bundle-local prefix equivalence artifact",
        )
        if (
            prefix_equivalence is not None
            and isinstance(prefix_equivalence.payload, dict)
            and prefix_equivalence.path is not None
        ):
            validation = validate_tau3_prefix_equivalence(
                prefix_equivalence.path
            )
            _require(
                target,
                validation.get("passed") is True,
                "detached-prefix equivalence artifact must independently replay",
            )
            prefix_binding = _dict(binding.get("prefix_equivalence"))
            _require(
                target,
                prefix_binding.get("sha256") == prefix_equivalence.sha256,
                "qualified training receipt must bind prefix equivalence sha256",
            )
            _require(
                target,
                prefix_binding.get("validation_passed") is True,
                "qualified training receipt prefix equivalence validation must pass",
            )
            equivalence_bindings = _dict(
                prefix_equivalence.payload.get("bindings")
            )
            _require(
                target,
                equivalence_bindings.get("dataset_file_sha256")
                == _nested(
                    receipt,
                    "training_binding",
                    "exposure",
                    "dataset",
                    "sha256",
                ),
                "prefix equivalence must bind the exact exposure dataset",
            )
            _require(
                target,
                equivalence_bindings.get("protocol_file_sha256")
                == _nested(receipt, "training_binding", "protocol", "sha256"),
                "prefix equivalence must bind the exact training protocol",
            )
            _require(
                target,
                equivalence_bindings.get("model_identity_file_sha256")
                == _nested(
                    receipt,
                    "training_binding",
                    "model",
                    "identity_sha256",
                ),
                "prefix equivalence must bind the exact model identity",
            )
            _validate_equivalence_recipe_binding(
                target,
                _dict(equivalence_bindings.get("recipe")),
                recipe,
            )
    elif prefix_equivalence is not None:
        _require(
            target,
            False,
            "full-gradient candidate must not claim detached-prefix equivalence",
        )
    _require_sha(target, recipe.get("recipe_sha256"), "qualified training receipt recipe.recipe_sha256")
    if exposure_receipt_sha256 is not None:
        _require(target, _nested(receipt, "training_binding", "exposure", "receipt", "sha256") == exposure_receipt_sha256, "qualified training receipt must bind exposure receipt sha256")
    if exposure_ledger_sha256 is not None:
        _require(target, _nested(receipt, "training_binding", "exposure", "ledger", "sha256") == exposure_ledger_sha256, "qualified training receipt must bind exposure ledger sha256")


def _validate_equivalence_recipe_binding(
    target: _Target,
    equivalence_recipe: dict[str, Any],
    training_recipe: dict[str, Any],
) -> None:
    for field_name in (
        "rank",
        "scale",
        "learning_rate",
        "num_layers",
        "max_seq_length",
        "batch_size",
        "grad_accumulation",
        "mask_prompt",
    ):
        _require(
            target,
            equivalence_recipe.get(field_name)
            == training_recipe.get(field_name),
            f"prefix equivalence recipe.{field_name} must match training recipe",
        )
    allowed_seeds = equivalence_recipe.get("allowed_seeds")
    _require(
        target,
        isinstance(allowed_seeds, list)
        and training_recipe.get("seed") in allowed_seeds,
        "prefix equivalence allowed_seeds must include the training seed",
    )


def _validate_development_qualification(
    root: Path,
    target: _Target,
    candidate: dict[str, Any],
    label: str,
    receipt_ref: _Loaded,
    adapter_sha256: Any,
) -> bool:
    before = len(target.errors)
    _validate_candidate_internal_validation(
        root,
        target,
        candidate,
        label,
        receipt_ref,
        adapter_sha256,
    )
    scorecard = _load_json_artifact_ref(root, target, candidate.get("development_scorecard"), f"qualified_candidates.{label}.development_scorecard")
    probes = _load_json_artifact_ref(root, target, candidate.get("behavior_probes"), f"qualified_candidates.{label}.behavior_probes")
    if isinstance(scorecard.payload, dict):
        _check_registered_schema(
            target,
            scorecard.payload,
            "tau3_development_scorecard",
            f"{label} development scorecard",
        )
        _require(target, _nested(scorecard.payload, "bindings", "training_receipt_sha256") == receipt_ref.sha256, f"{label} development scorecard must bind training receipt sha256")
        _require(target, _nested(scorecard.payload, "bindings", "adapter_tree_sha256") == adapter_sha256, f"{label} development scorecard must bind adapter tree sha256")
        _require(target, _nested(scorecard.payload, "bindings", "harness_sha256") == _nested(scorecard.payload, "frozen_contract", "harness_sha256"), f"{label} development scorecard must bind identical frozen harness")
        _require(target, _nested(scorecard.payload, "bindings", "grid_sha256") == _nested(scorecard.payload, "frozen_contract", "grid_sha256"), f"{label} development scorecard must bind identical frozen grid")
        _require(target, _nested(scorecard.payload, "bindings", "base_identity_sha256") == _nested(scorecard.payload, "frozen_contract", "base_identity_sha256"), f"{label} development scorecard must bind identical base")
        _validate_development_evaluation(root, target, scorecard.payload, label, receipt_ref.sha256, adapter_sha256)
    if isinstance(probes.payload, dict):
        if probes.path is not None:
            probe_validation = validate_tau3_behavior_probes(probes.path)
            _require(target, probe_validation.get("passed") is True, f"{label} behavior probe validator must pass")
        _require_probe_binding(target, probes.payload, "training_receipt_sha256", receipt_ref.sha256, f"{label} behavior probes")
        _require_probe_binding(target, probes.payload, "adapter_tree_sha256", adapter_sha256, f"{label} behavior probes")
        _require_probe_binding(target, probes.payload, "harness_sha256", _nested(scorecard.payload, "bindings", "harness_sha256") if isinstance(scorecard.payload, dict) else None, f"{label} behavior probes")
        _require_probe_binding(target, probes.payload, "protocol_sha256", _nested(receipt_ref.payload, "training_binding", "protocol", "sha256"), f"{label} behavior probes")
        _require_probe_binding(target, probes.payload, "grid_sha256", _nested(scorecard.payload, "bindings", "grid_sha256") if isinstance(scorecard.payload, dict) else None, f"{label} behavior probes")
        _validate_behavior_probe_replay(probes.path.parent if probes.path is not None else root, target, probes.payload, label)
    return len(target.errors) == before


def _validate_candidate_internal_validation(
    root: Path,
    target: _Target,
    candidate: dict[str, Any],
    label: str,
    receipt_ref: _Loaded,
    adapter_sha256: Any,
) -> None:
    evidence = _dict(candidate.get("internal_validation"))
    artifact = _load_json_artifact_ref(
        root,
        target,
        evidence.get("artifact"),
        f"qualified_candidates.{label}.internal_validation.artifact",
    )
    dataset = _load_ref_path(
        root,
        target,
        evidence.get("dataset"),
        f"qualified_candidates.{label}.internal_validation.dataset",
    )
    protocol = _load_ref_path(
        root,
        target,
        evidence.get("protocol"),
        f"qualified_candidates.{label}.internal_validation.protocol",
    )
    model_identity = _load_ref_path(
        root,
        target,
        evidence.get("model_identity"),
        f"qualified_candidates.{label}.internal_validation.model_identity",
    )
    if (
        artifact.path is None
        or not isinstance(artifact.payload, dict)
        or dataset is None
        or protocol is None
        or model_identity is None
        or receipt_ref.path is None
    ):
        return
    _check_registered_schema(
        target,
        artifact.payload,
        "tau3_internal_validation",
        f"{label} internal validation",
    )
    result = validate_tau3_internal_validation(
        artifact.path,
        dataset_path=dataset,
        training_receipt_path=receipt_ref.path,
        protocol_path=protocol,
        model_identity_path=model_identity,
    )
    _require(
        target,
        result.get("passed") is True,
        f"{label} internal validation must independently replay: "
        + ", ".join(result.get("errors") or []),
    )
    bindings = _dict(artifact.payload.get("bindings"))
    _require(
        target,
        bindings.get("training_receipt_sha256") == receipt_ref.sha256,
        f"{label} internal validation must bind training receipt sha256",
    )
    _require(
        target,
        bindings.get("adapter_tree_sha256") == adapter_sha256,
        f"{label} internal validation must bind adapter tree sha256",
    )
    coverage = _dict(artifact.payload.get("coverage"))
    _require(
        target,
        coverage.get("every_row_evaluated") is True
        and coverage.get("evaluated_row_count") == coverage.get("row_count"),
        f"{label} internal validation must evaluate every row",
    )
    _require(
        target,
        coverage.get("required_domains_exact") is True
        and coverage.get("required_behaviors_exact") is True,
        f"{label} internal validation must cover every required domain and behavior",
    )
    _require(
        target,
        _nested(
            artifact.payload,
            "aggregate",
            "numerical_failure_count",
        )
        == 0,
        f"{label} internal validation must have zero numerical failures",
    )


def _require_probe_binding(target: _Target, probes: dict[str, Any], field: str, expected: Any, label: str) -> None:
    _require(target, _nested(probes, "bindings", field) == expected, f"{label} must bind {field}")


def _validate_development_evaluation(
    root: Path,
    target: _Target,
    scorecard: dict[str, Any],
    label: str,
    training_receipt_sha256: str | None,
    adapter_sha256: Any,
) -> None:
    evaluation = _load_json_artifact_ref(root, target, scorecard.get("development_evaluation"), f"{label}.development_evaluation")
    if evaluation.path is None or not isinstance(evaluation.payload, dict):
        return
    payload = evaluation.payload
    _check_registered_schema(
        target,
        payload,
        "tau3_development_evaluation",
        f"{label} development evaluation",
    )
    _require(target, payload.get("schema_version") == "hfr.tau3_development_evaluation.v1", f"{label} development evaluation schema_version must be hfr.tau3_development_evaluation.v1")
    _require(target, payload.get("mode") == "development", f"{label} development evaluation must be mode=development")
    _require(target, payload.get("passed") is True, f"{label} development evaluation must pass")
    _require(target, _nested(payload, "bindings", "training_receipt_sha256") == training_receipt_sha256, f"{label} development evaluation must bind training receipt sha256")
    _require(target, _nested(payload, "bindings", "adapter_tree_sha256") == adapter_sha256, f"{label} development evaluation must bind adapter tree sha256")
    _require(target, _dict(payload.get("bindings")) == _dict(scorecard.get("bindings")), f"{label} development evaluation bindings must match scorecard")
    _require(target, _nested(payload, "pairing", "passed") is True, f"{label} development evaluation must have complete paired runs")
    sources = _dict(payload.get("source_artifacts"))
    _require(target, bool(sources.get("adapter")) and bool(sources.get("base")), f"{label} development evaluation must include adapter/base source artifacts")
    replay = _development_trial_replay(payload)
    _require(target, replay["passed"] is True, f"{label} development evaluation trial grid must replay: {', '.join(replay['errors'])}")
    metrics = replay["metrics"] if isinstance(replay.get("metrics"), dict) else {}
    macro = _dict(metrics.get("macro_pass1"))
    adapter = _number(macro.get("adapter"))
    base = _number(macro.get("base"))
    _require(target, adapter is not None and adapter >= 0.10, f"{label} development macro Pass-1 must be at least 0.10")
    _require(target, adapter is not None and base is not None and adapter - base >= 0.05, f"{label} development adapter-base macro gain must be at least 0.05")
    per_domain = _dict(_nested(metrics, "per_domain_pass1", "adapter"))
    for domain in DOMAINS:
        _require(target, float(per_domain.get(domain) or 0.0) >= 0.05, f"{label} development {domain} Pass-1 must be at least 0.05")
    _require(target, payload.get("failed_check_count") == 0 and payload.get("blocking_reasons") == [], f"{label} development evaluation must have zero blockers")
    _require(target, _nested(payload, "harness", "passed") is True, f"{label} development evaluation harness must pass")
    _require(target, _nested(payload, "harness", "identity_sha256") == _nested(payload, "development_grid", "harness_sha256"), f"{label} development evaluation must bind identical harness identity")
    _require(target, _nested(payload, "bindings", "evaluator_model_contract_sha256") == _nested(payload, "development_grid", "evaluator_model_contract_sha256"), f"{label} development evaluation must bind identical evaluator model contract")
    _require(target, _nested(payload, "public_payload_scan", "passed") is True, f"{label} development evaluation public payload scan must pass")
    safety = _dict(_nested(payload, "metrics", "safety"))
    _require(target, safety.get("provable") is True and safety.get("blocking_reasons") == [], f"{label} development safety blockers must be zero")


def _development_trial_replay(payload: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    grid = _dict(payload.get("development_grid"))
    trials = _list_of_dicts(payload.get("development_trials"))
    if not trials:
        errors.append("development evaluation must include trial-level outcomes")
        return {"passed": False, "errors": errors, "metrics": {}}
    domains = tuple(str(domain) for domain in grid.get("domains", []) if isinstance(domain, str))
    seeds = tuple(int(seed) for seed in grid.get("seeds", []) if isinstance(seed, int))
    tasks_by_domain = {
        str(domain): tuple(str(task) for task in tasks if isinstance(task, str))
        for domain, tasks in _dict(grid.get("tasks_by_domain")).items()
        if isinstance(tasks, list)
    }
    if tuple(sorted(domains)) != DOMAINS:
        errors.append("development grid domains must be airline/retail/telecom")
    if not seeds:
        errors.append("development grid seeds must be non-empty")
    if sorted(tasks_by_domain) != list(DOMAINS) or any(not tasks for tasks in tasks_by_domain.values()):
        errors.append("development grid tasks_by_domain must cover every domain")
    expected_keys = {
        (domain, task_sha256, seed)
        for domain, tasks in tasks_by_domain.items()
        for task_sha256 in tasks
        for seed in seeds
    }
    seen: set[tuple[str, str, int]] = set()
    adapter_total = 0
    base_total = 0
    per_domain_counts = {domain: 0 for domain in DOMAINS}
    per_domain_adapter = {domain: 0 for domain in DOMAINS}
    for index, trial in enumerate(trials):
        domain = str(trial.get("domain") or "")
        task_sha256 = str(trial.get("task_sha256") or "")
        seed = int(trial.get("seed") or -1)
        key = (domain, task_sha256, seed)
        if key in seen:
            errors.append(f"duplicate development trial {domain}/{task_sha256}/{seed}")
        seen.add(key)
        if key not in expected_keys:
            errors.append(f"unexpected development trial {domain}/{task_sha256}/{seed}")
        adapter_pass = _bool_int(trial.get("adapter_pass1"))
        base_pass = _bool_int(trial.get("base_pass1"))
        if adapter_pass is None or base_pass is None:
            errors.append(f"development trial {index} pass flags must be boolean")
            continue
        expected_result = {
            "adapter_pass1": bool(adapter_pass),
            "base_pass1": bool(base_pass),
            "domain": domain,
            "seed": seed,
            "task_sha256": task_sha256,
        }
        if trial.get("result_sha256") != _canonical_sha256(expected_result):
            errors.append(f"development trial {index} result hash does not replay")
        if not isinstance(trial.get("source_sha256"), str) or SHA256_RE.fullmatch(str(trial.get("source_sha256"))) is None:
            errors.append(f"development trial {index} source_sha256 must be sha256")
        adapter_total += adapter_pass
        base_total += base_pass
        if domain in per_domain_counts:
            per_domain_counts[domain] += 1
            per_domain_adapter[domain] += adapter_pass
    missing = expected_keys - seen
    if missing:
        errors.append(f"development grid missing {len(missing)} trial(s)")
    count = len(trials)
    metrics = {
        "macro_pass1": {
            "adapter": adapter_total / count if count else 0.0,
            "base": base_total / count if count else 0.0,
        },
        "per_domain_pass1": {
            "adapter": {
                domain: (per_domain_adapter[domain] / per_domain_counts[domain] if per_domain_counts[domain] else 0.0)
                for domain in DOMAINS
            }
        },
    }
    declared = _dict(payload.get("metrics"))
    declared_macro = _dict(declared.get("macro_pass1"))
    if _number(declared_macro.get("adapter")) != metrics["macro_pass1"]["adapter"] or _number(declared_macro.get("base")) != metrics["macro_pass1"]["base"]:
        errors.append("development macro metrics do not replay from trial outcomes")
    for domain, value in metrics["per_domain_pass1"]["adapter"].items():
        if _number(_nested(declared, "per_domain_pass1", "adapter", domain)) != value:
            errors.append(f"development {domain} metric does not replay from trial outcomes")
    return {"passed": not errors, "errors": errors, "metrics": metrics}


def _bool_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return 1 if value else 0
    return None


def _validate_behavior_probe_replay(probes_root: Path, target: _Target, probes: dict[str, Any], label: str) -> None:
    refs = _list_of_dicts(probes.get("probe_results"))
    _require(target, bool(refs), f"{label} behavior probes must include hash-bound probe_results")
    failed = 0
    seen: set[str] = set()
    for index, ref in enumerate(refs):
        loaded = _load_json_artifact_ref(probes_root, target, ref, f"{label}.probe_results[{index}]")
        if not isinstance(loaded.payload, dict):
            failed += 1
            continue
        probe_id = str(loaded.payload.get("probe_id") or "")
        _require(target, bool(probe_id) and probe_id not in seen, f"{label} probe result ids must be nonempty and unique")
        seen.add(probe_id)
        expected = loaded.payload.get("expected_outcome")
        actual = loaded.payload.get("actual_outcome")
        _require(target, loaded.payload.get("expected_outcome_sha256") == _canonical_sha256(expected), f"{label} probe {probe_id} expected outcome hash must replay")
        if expected != actual:
            failed += 1
    aggregate = _dict(probes.get("aggregate"))
    _require(target, aggregate.get("total_probe_count") == len(refs), f"{label} behavior probe total count must replay")
    _require(target, aggregate.get("failed_probe_count") == failed, f"{label} behavior probe failed count must replay")
    _require(target, failed == 0, f"{label} behavior probes must have zero failures")


def _adapter_files_replay(receipt_dir: Path, adapter: dict[str, Any]) -> bool:
    adapter_path = adapter.get("path")
    files = _list_of_dicts(adapter.get("files"))
    if not isinstance(adapter_path, str) or not adapter_path or not files:
        return False
    root = receipt_dir / adapter_path
    digest = hashlib.sha256()
    adapter_weight_count = 0
    for record in files:
        rel = record.get("path")
        if not isinstance(rel, str) or _is_unsafe_relative_path(rel):
            return False
        path = root / rel
        if not path.is_file():
            return False
        if record.get("sha256") != _sha256_file(path) or int(record.get("size") or -1) != path.stat().st_size:
            return False
        if record.get("kind") == "adapter" and path.stat().st_size > 0:
            adapter_weight_count += 1
        digest.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return adapter_weight_count > 0 and digest.hexdigest() == adapter.get("tree_sha256")


def _load_json_path(path: Path, target: _Target) -> _Loaded:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        target.errors.append(f"unable to read JSON: {exc}")
        return _Loaded(target)
    sha256 = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        target.errors.append(f"invalid JSON: {exc}")
        return _Loaded(target, sha256=sha256)
    return _Loaded(target, payload, sha256)


def _validate_ref(
    root: Path,
    target: _Target,
    ref: Any,
    label: str,
    *,
    allow_private_local: bool,
    require_exists: bool = True,
) -> None:
    if not isinstance(ref, dict):
        target.errors.append(f"{label} must be a ref object")
        return
    _resolve_ref_path(root, target, ref, label, allow_private_local=allow_private_local, require_exists=require_exists)


def _validate_json_ref(
    root: Path,
    target: _Target,
    ref: Any,
    label: str,
    *,
    allow_private_local: bool,
) -> Any:
    if not isinstance(ref, dict):
        target.errors.append(f"{label} must be a ref object")
        return None
    path = _resolve_ref_path(
        root,
        target,
        ref,
        label,
        allow_private_local=allow_private_local,
        require_exists=True,
    )
    if path is None or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        target.errors.append(f"{label} must contain valid JSON: {exc}")
        return None


def _resolve_ref_path(
    root: Path,
    target: _Target,
    ref: dict[str, Any],
    label: str,
    *,
    allow_private_local: bool,
    require_exists: bool,
) -> Path | None:
    _require_sha(target, ref.get("sha256"), f"{label}.sha256")
    path_text = ref.get("path")
    source_path_text = ref.get("source_path")
    if source_path_text is not None:
        if not allow_private_local or ref.get("access") != "private_local":
            target.errors.append(f"{label}.source_path is allowed only for plan-stage private_local refs")
            return None
        if not isinstance(source_path_text, str) or not source_path_text:
            target.errors.append(f"{label}.source_path must be a nonempty string")
            return None
        path = Path(source_path_text).expanduser()
        if require_exists:
            _require(target, path.is_file(), f"{label}.source_path does not exist")
            if path.is_file() and SHA256_RE.fullmatch(str(ref.get("sha256") or "")):
                _require(target, _sha256_file(path) == ref.get("sha256"), f"{label}.source_path sha256 mismatch")
        return path
    if not isinstance(path_text, str) or not path_text:
        target.errors.append(f"{label}.path must be a nonempty bundle-relative path")
        return None
    if _is_unsafe_relative_path(path_text):
        target.errors.append(f"{label}.path must be bundle-relative and safe")
        return None
    path = (root / path_text).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        target.errors.append(f"{label}.path escapes bundle root")
        return None
    if require_exists:
        _require(target, path.is_file(), f"{label}.path does not exist")
        if path.is_file() and SHA256_RE.fullmatch(str(ref.get("sha256") or "")):
            _require(target, _sha256_file(path) == ref.get("sha256"), f"{label}.path sha256 mismatch")
    return path


def _require_no_private_or_sealed_payloads(target: _Target, value: Any) -> None:
    hits = _private_path_hits(value)
    if hits:
        target.errors.append(f"private/local path strings are not allowed except private_local source_path refs: {hits[:3]}")
    sealed_hits = _sealed_payload_hits(value)
    if sealed_hits:
        target.errors.append(f"sealed payload materialization markers are not allowed: {sealed_hits[:3]}")


def _private_path_hits(value: Any) -> list[str]:
    hits: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "source_path":
                continue
            hits.extend(_private_path_hits(item))
    elif isinstance(value, list):
        for item in value:
            hits.extend(_private_path_hits(item))
    elif isinstance(value, str) and PRIVATE_PATH_RE.search(value):
        hits.append(value)
    return hits


def _sealed_payload_hits(value: Any) -> list[str]:
    hits: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = key.lower()
            if "sealed_payload" in lowered or "grader_secret" in lowered or "expected_action" in lowered or "target_state" in lowered:
                if item not in (0, False, None, [], {}):
                    hits.append(key)
            hits.extend(_sealed_payload_hits(item))
    elif isinstance(value, list):
        for item in value:
            hits.extend(_sealed_payload_hits(item))
    return hits


def _summary(strict: bool, targets: list[_Target]) -> dict[str, Any]:
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "passed": all(target.passed for target in targets),
        "strict": strict,
        "target_count": len(targets),
        "error_count": sum(len(target.errors) for target in targets),
        "warning_count": sum(len(target.warnings) for target in targets),
        "targets": [target.as_dict() for target in targets],
    }


def _require(target: _Target, condition: bool, error: str) -> None:
    if not condition:
        target.errors.append(error)


def _require_sha(target: _Target, value: Any, label: str) -> None:
    _require(target, isinstance(value, str) and bool(SHA256_RE.fullmatch(value)), f"{label} must be a lowercase sha256")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_binding_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise Tau3CompetitiveV3BindingError(f"{label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Tau3CompetitiveV3BindingError(
            f"{label} must contain valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise Tau3CompetitiveV3BindingError(f"{label} must be a JSON object")
    return payload


def _write_binding_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    original_mode = path.stat().st_mode & 0o777
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, original_mode)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _is_unsafe_relative_path(value: str) -> bool:
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    return (
        value.startswith(("/", "\\"))
        or windows.is_absolute()
        or posix.is_absolute()
        or any(part in ("", ".", "..") for part in posix.parts)
        or any(part == ".." for part in windows.parts)
    )


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _list_of_strings(value: Any) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _nested(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        current = _dict(current).get(key)
    return current


def _between(value: Any, low: float, high: float) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return low <= number <= high


def _candidate_beats_base(evaluation: dict[str, Any]) -> bool:
    effects = _dict(evaluation.get("effects"))
    base_effect = _nested(effects, "base", "domain_stratified_macro_pass1")
    interval = _dict(_dict(base_effect).get("confidence_interval"))
    return _number(_dict(base_effect).get("mean_difference")) is not None and _dict(base_effect)["mean_difference"] > 0 and _number(interval.get("lower")) is not None and interval["lower"] > 0


def _within_strongest_comparator_margin(evaluation: dict[str, Any]) -> bool:
    macro = _dict(_nested(evaluation, "metrics", "macro_pass1"))
    adapter = _number(macro.get("adapter"))
    comparators = [_number(macro.get("comparator_1")), _number(macro.get("comparator_2"))]
    point_margin_passed = adapter is not None and all(item is not None for item in comparators) and adapter >= max(item for item in comparators if item is not None) - 0.05
    effects = _dict(evaluation.get("effects"))
    for arm in ("comparator_1", "comparator_2"):
        if _dict(effects.get(arm)).get("per_domain_non_inferiority_passed") is not True:
            return False
    safety = _dict(_nested(evaluation, "metrics", "safety"))
    if safety.get("provable") is not True or safety.get("blocking_reasons") not in ([], ()):
        return False
    safety_checks = {
        str(check.get("id")): check.get("passed")
        for check in evaluation.get("checks", [])
        if isinstance(check, dict)
    }
    return point_margin_passed and safety_checks.get("safety_non_inferiority_vs_base") is True


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
