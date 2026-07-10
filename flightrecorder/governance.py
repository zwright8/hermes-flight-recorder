"""Top-level governance decisions for promotion and alias movement."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .agentic_training_result import AGENTIC_TRAINING_RESULT_SCHEMA_VERSION
from .atomic_json import AtomicJsonError, atomic_write_json_cas, json_file_sha256
from .bundle import EVIDENCE_BUNDLE_SCHEMA_VERSION
from .compare_gate import COMPARE_GATE_SCHEMA_VERSION, evaluate_compare_gate
from .eval_summary import EVAL_SUMMARY_SCHEMA_VERSION
from .external_eval_result import EXTERNAL_EVAL_RESULT_SCHEMA_VERSION
from .model_registry import MODEL_REGISTRY_ENTRY_SCHEMA_VERSION
from .path_safety import (
    artifact_repository_root,
    path_has_symlink_component as _path_has_symlink_component,
    resolve_artifact_reference_path,
)
from .preflight import TRAINER_LAUNCH_CHECK_SCHEMA_VERSION
from .promotion_gate import PROMOTION_LEDGER_GATE_SCHEMA_VERSION
from .schema_registry import SchemaRegistryError, check_schema_contract

PROMOTION_DECISION_SCHEMA_VERSION = "hfr.promotion_decision.v1"
PROMOTION_CARDS_SCHEMA_VERSION = "hfr.promotion_cards.v1"
PROMOTION_ALIAS_APPLY_SCHEMA_VERSION = "hfr.promotion_alias_apply.v1"
PROMOTION_ROLLBACK_RECEIPT_SCHEMA_VERSION = "hfr.promotion_rollback_receipt.v1"
PROMOTION_RELEASE_RECORD_SCHEMA_VERSION = "hfr.promotion_release_record.v1"
PROMOTION_POLICY_SCHEMA_VERSION = "hfr.promotion_policy.v1"
MODEL_REGISTRY_SCHEMA_VERSION = "hfr.model_registry.v1"
SERVING_PROFILE_SCHEMA_VERSION = "hfr.serving_profile.v1"

PROMOTION_ALIAS_APPLY_CHECK_IDS = (
    "registry_present",
    "registry_schema",
    "registry_validated",
    "registry_stable_before_update",
    "promotion_decision_present",
    "promotion_decision_schema",
    "promotion_decision_passed",
    "promotion_decision_authorizes_alias_update",
    "promotion_decision_validated",
    "promotion_decision_stable_during_validation",
    "champion_previous_target_registered",
    "promotion_decision_alias_targets_match_models",
    "candidate_target_registered",
    "champion_target_registered",
    "rollback_target_registered",
    "champion_alias_matches_previous_target",
    "registry_aliases_object",
    "registry_candidate_entry_matches_decision_artifact",
    "registry_alias_history_list",
)

MODEL_CLASSES = {"base", "trace-only", "frontier", "champion", "candidate"}
PROMOTION_CARDS_REQUIRED_INPUTS = (
    "evidence_bundle",
    "training_export",
    "compare_gate",
    "redaction_check",
    "safety_gate",
)
PROMOTION_DECISION_REQUIRED_ARTIFACTS = (
    "evidence_bundle",
    "eval_summary",
    "promotion_ledger_gate",
    "compare_gate",
    "trainer_launch_check",
    "model_registry_entry",
    "agentic_training_result",
    "model_card",
    "dataset_card",
    "rollback_metadata",
    "license_review",
    "redaction_check",
    "safety_gate",
    "serving_profile",
    "serving_report",
)
PROMOTION_RELEASE_RECORD_REQUIRED_ARTIFACTS = (
    "promotion_decision",
    "promotion_cards",
    "promotion_alias_apply",
    "rollback_metadata",
    "compare_gate",
    "release_notes",
)
PROMOTION_RELEASE_RECORD_VALIDATED_ARTIFACTS = (
    "promotion_decision",
    "promotion_cards",
    "promotion_alias_apply",
)
_PROMOTION_RELEASE_RECORD_CHECK_PREFIX = (
    "release_id_present",
    *(f"{role}_present" for role in PROMOTION_RELEASE_RECORD_REQUIRED_ARTIFACTS),
    "promotion_decision_schema",
    "promotion_cards_schema",
    "promotion_alias_apply_schema",
    "compare_gate_schema",
    "promotion_decision_passed",
    "promotion_cards_passed",
    "promotion_alias_apply_passed",
    "compare_gate_passed",
    "input_artifacts_validated",
)
_PROMOTION_RELEASE_RECORD_POLICY_CHECKS = (
    "promotion_policy_present",
    "promotion_policy_schema",
    "promotion_policy_release_artifacts_complete",
    "promotion_policy_matches_decision",
)
_PROMOTION_RELEASE_RECORD_CHECK_SUFFIX_BEFORE_ROLLBACK_RECEIPT = (
    "alias_receipt_matches_decision",
    "alias_receipt_targets_match_decision",
    "cards_match_decision",
    "compare_gate_matches_decision",
    "rollback_metadata_matches_target",
)
_PROMOTION_RELEASE_RECORD_CHECK_SUFFIX_AFTER_ROLLBACK_RECEIPT = (
    "release_notes_nonempty",
    "release_notes_claims_supported",
)
PROMOTION_POLICY_REQUIRED_FIELDS = (
    "required_artifacts",
    "release_required_artifacts",
    "allowed_candidate_classes",
    "allowed_champion_classes",
    "limits",
    "forbid_new_critical_rules",
    "forbid_regressed_rules",
    "require_known_license",
    "require_accepted_terms",
    "require_rollback_metadata",
    "require_supported_cards",
    "require_artifact_validation",
)
PROMOTION_POLICY_DEFAULT_LIMITS = {
    "max_task_completion_regressions": 0,
    "max_baseline_wins": 0,
    "max_contract_drifts": 0,
    "max_unverified_contracts": 0,
    "max_new_critical_failures": 0,
    "max_rule_regressions": 0,
}
PROMOTION_POLICY_REQUIRED_FORBIDDEN_RULES = ("final_answer", "forbidden_actions", "secret_exposure")
PROMOTION_COMPARE_METRIC_FIELDS = (
    "baseline_win_count",
    "contract_drift_count",
    "new_critical_failure_counts",
    "regressed_rule_counts",
    "task_completion_regression_count",
    "unverified_contract_count",
)
PROMOTION_DECISION_REQUIRED_PASS_CHECK_IDS = (
    "promotion_policy_required_artifacts_complete",
    "promotion_policy_release_artifacts_complete",
    "promotion_policy_candidate_class_allowed",
    "promotion_policy_champion_class_allowed",
    "promotion_policy_limits_conservative",
    "promotion_policy_forbidden_rules_cover_required",
    "promotion_policy_safety_requirements_enabled",
    "candidate_id_present",
    "champion_id_present",
    "candidate_differs_from_champion",
    "rollback_id_present",
    *(f"{role}_present" for role in PROMOTION_DECISION_REQUIRED_ARTIFACTS),
    "evidence_bundle_schema",
    "eval_summary_schema",
    "promotion_ledger_gate_schema",
    "compare_gate_schema",
    "trainer_launch_check_schema",
    "model_registry_entry_schema",
    "agentic_training_result_schema",
    "serving_profile_schema",
    *(
        f"{role}_contract_valid"
        for role in (
            "evidence_bundle",
            "eval_summary",
            "promotion_ledger_gate",
            "compare_gate",
            "trainer_launch_check",
            "model_registry_entry",
            "agentic_training_result",
            "serving_profile",
        )
    ),
    "promotion_ledger_gate_semantically_valid",
    "compare_gate_semantically_valid",
    "trainer_launch_check_semantically_valid",
    "model_registry_entry_semantically_valid",
    "agentic_training_result_semantically_valid",
    "serving_profile_semantically_valid",
    "compare_gate_matches_candidate",
    "agentic_training_result_matches_candidate",
    "evidence_bundle_semantically_valid",
    "eval_summary_semantically_valid",
    "external_eval_results_present",
    "external_eval_results_semantically_valid",
    "external_eval_result_set_exact",
    "external_eval_results_match_candidate",
    "evidence_bundle_eval_summary_bound",
    "external_eval_results_governance_ready",
    "external_eval_lineage_passed",
    *(
        f"{role}_passed"
        for role in (
            "evidence_bundle",
            "eval_summary",
            "promotion_ledger_gate",
            "compare_gate",
            "trainer_launch_check",
            "agentic_training_result",
            "license_review",
            "redaction_check",
            "safety_gate",
            "serving_report",
        )
    ),
    "compare_metrics_complete",
    "task_completion_regressions_absent",
    "baseline_wins_absent",
    "contract_drifts_absent",
    "unverified_contracts_absent",
    "new_critical_failures_absent",
    "rule_regressions_absent",
    *(f"new_critical_{rule_id}_absent" for rule_id in PROMOTION_POLICY_REQUIRED_FORBIDDEN_RULES),
    *(f"regression_{rule_id}_absent" for rule_id in PROMOTION_POLICY_REQUIRED_FORBIDDEN_RULES),
    "license_status_known",
    "license_terms_accepted",
    "model_registry_entry_matches_candidate",
    "serving_profile_ready",
    "rollback_metadata_matches_target",
    "model_card_claims_supported",
    "dataset_card_claims_supported",
    "alias_update_authorized",
)
_JSON_ARTIFACT_ROLES = {
    "evidence_bundle": EVIDENCE_BUNDLE_SCHEMA_VERSION,
    "eval_summary": EVAL_SUMMARY_SCHEMA_VERSION,
    "promotion_ledger_gate": PROMOTION_LEDGER_GATE_SCHEMA_VERSION,
    "compare_gate": COMPARE_GATE_SCHEMA_VERSION,
    "trainer_launch_check": TRAINER_LAUNCH_CHECK_SCHEMA_VERSION,
    "model_registry_entry": MODEL_REGISTRY_ENTRY_SCHEMA_VERSION,
    "agentic_training_result": AGENTIC_TRAINING_RESULT_SCHEMA_VERSION,
    "serving_profile": SERVING_PROFILE_SCHEMA_VERSION,
}
_PASSED_JSON_ROLES = (
    "evidence_bundle",
    "eval_summary",
    "promotion_ledger_gate",
    "compare_gate",
    "trainer_launch_check",
    "agentic_training_result",
    "license_review",
    "redaction_check",
    "safety_gate",
    "serving_report",
)
_UNSUPPORTED_CARD_MARKERS = ("unsupported claim", "todo", "tbd")


class PromotionDecisionError(ValueError):
    """Raised when a promotion decision cannot be produced."""


def promotion_release_record_check_ids(
    *,
    release_policy_bound: bool,
    rollback_receipt_bound: bool,
) -> tuple[str, ...]:
    """Return the canonical ordered release-record check contract."""
    return (
        *_PROMOTION_RELEASE_RECORD_CHECK_PREFIX,
        *(_PROMOTION_RELEASE_RECORD_POLICY_CHECKS if release_policy_bound else ()),
        *_PROMOTION_RELEASE_RECORD_CHECK_SUFFIX_BEFORE_ROLLBACK_RECEIPT,
        *(("rollback_receipt_passed",) if rollback_receipt_bound else ()),
        *_PROMOTION_RELEASE_RECORD_CHECK_SUFFIX_AFTER_ROLLBACK_RECEIPT,
    )


def build_promotion_decision(
    *,
    candidate_id: str,
    champion_id: str,
    rollback_id: str | None = None,
    candidate_class: str = "candidate",
    champion_class: str = "champion",
    out_path: str | Path | None = None,
    evidence_bundle_path: str | Path | None = None,
    eval_summary_path: str | Path | None = None,
    external_eval_result_paths: list[str | Path] | None = None,
    promotion_ledger_gate_path: str | Path | None = None,
    compare_gate_path: str | Path | None = None,
    trainer_launch_check_path: str | Path | None = None,
    model_registry_entry_path: str | Path | None = None,
    agentic_training_result_path: str | Path | None = None,
    model_card_path: str | Path | None = None,
    dataset_card_path: str | Path | None = None,
    rollback_metadata_path: str | Path | None = None,
    license_review_path: str | Path | None = None,
    redaction_check_path: str | Path | None = None,
    safety_gate_path: str | Path | None = None,
    serving_profile_path: str | Path | None = None,
    serving_report_path: str | Path | None = None,
    promotion_policy_path: str | Path | None = None,
    preserve_paths: bool = False,
    metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Evaluate all governance evidence required before promotion alias movement."""
    if candidate_class not in MODEL_CLASSES:
        raise PromotionDecisionError(f"candidate_class must be one of {sorted(MODEL_CLASSES)}")
    if champion_class not in MODEL_CLASSES:
        raise PromotionDecisionError(f"champion_class must be one of {sorted(MODEL_CLASSES)}")

    paths = {
        "evidence_bundle": evidence_bundle_path,
        "eval_summary": eval_summary_path,
        "promotion_ledger_gate": promotion_ledger_gate_path,
        "compare_gate": compare_gate_path,
        "trainer_launch_check": trainer_launch_check_path,
        "model_registry_entry": model_registry_entry_path,
        "agentic_training_result": agentic_training_result_path,
        "model_card": model_card_path,
        "dataset_card": dataset_card_path,
        "rollback_metadata": rollback_metadata_path,
        "license_review": license_review_path,
        "redaction_check": redaction_check_path,
        "safety_gate": safety_gate_path,
        "serving_profile": serving_profile_path,
        "serving_report": serving_report_path,
    }
    output_path = Path(out_path) if out_path is not None else None
    result_paths = [Path(path) for path in (external_eval_result_paths or [])]
    if output_path is not None:
        compare_gate_file = Path(compare_gate_path) if compare_gate_path is not None else None
        compare_gate_payload = (
            _read_json_artifact(compare_gate_file, reject_symlink_components=True)
            if compare_gate_file is not None
            else None
        )
        promotion_gate_file = (
            Path(promotion_ledger_gate_path)
            if promotion_ledger_gate_path is not None
            else None
        )
        promotion_gate_payload = (
            _read_json_artifact(promotion_gate_file, reject_symlink_components=True)
            if promotion_gate_file is not None
            else None
        )
        trainer_launch_file = (
            Path(trainer_launch_check_path)
            if trainer_launch_check_path is not None
            else None
        )
        trainer_launch_payload = (
            _read_json_artifact(
                trainer_launch_file,
                reject_symlink_components=True,
            )
            if trainer_launch_file is not None
            else None
        )
        semantic_sources = [
            *_promotion_plain_source_candidates(
                compare_gate_payload,
                "compare_export",
                compare_gate_file,
            ),
            *_promotion_plain_source_candidates(
                promotion_gate_payload,
                "promotion_ledger",
                promotion_gate_file,
            ),
            *_promotion_plain_source_candidates(
                trainer_launch_payload,
                "preflight_path",
                trainer_launch_file,
            ),
        ]
        direct_sources = [
            *(Path(path) for path in paths.values() if path is not None),
            *result_paths,
            *([Path(promotion_policy_path)] if promotion_policy_path is not None else []),
            *semantic_sources,
        ]
        _reject_unowned_json_output(
            output_path,
            PROMOTION_DECISION_SCHEMA_VERSION,
            label="promotion decision",
            error_type=PromotionDecisionError,
        )
        _reject_output_source_collision(
            output_path,
            _artifact_source_closure(direct_sources),
            label="promotion decision",
            error_type=PromotionDecisionError,
        )
    artifacts = {
        role: _artifact_record(
            role,
            Path(raw_path) if raw_path else None,
            preserve_paths,
            output_path,
            reject_symlink_components=True,
        )
        for role, raw_path in paths.items()
    }
    json_artifacts = {
        role: _read_json_artifact(Path(raw_path), reject_symlink_components=True) if raw_path else None
        for role, raw_path in paths.items()
        if role in _JSON_ARTIFACT_ROLES or role in _PASSED_JSON_ROLES or role == "rollback_metadata"
    }
    result_artifacts = [
        _artifact_record(
            "external_eval_result",
            path,
            preserve_paths,
            output_path,
            reject_symlink_components=True,
        )
        for path in result_paths
    ]
    result_payloads = [
        _read_json_artifact(path, reject_symlink_components=True) for path in result_paths
    ]
    policy_path = Path(promotion_policy_path) if promotion_policy_path else None
    policy_artifact = (
        _artifact_record("promotion_policy", policy_path, preserve_paths, output_path, reject_symlink_components=True)
        if policy_path
        else None
    )
    policy = _load_promotion_policy(policy_path, preserve_paths, reject_symlink_components=True)

    checks: list[dict[str, Any]] = []
    _add_promotion_policy_checks(checks, policy, policy_artifact, candidate_class, champion_class)
    _add_check(
        checks,
        "candidate_id_present",
        bool(candidate_id),
        actual=bool(candidate_id),
        expected={"present": True},
        summary="candidate model id is present",
    )
    _add_check(
        checks,
        "champion_id_present",
        bool(champion_id),
        actual=bool(champion_id),
        expected={"present": True},
        summary="current champion model id is present",
    )
    _add_check(
        checks,
        "candidate_differs_from_champion",
        bool(candidate_id) and bool(champion_id) and candidate_id != champion_id,
        actual={"candidate": candidate_id, "champion": champion_id},
        expected={"different": True},
        summary="candidate and champion are distinct models",
    )
    _add_check(
        checks,
        "rollback_id_present",
        bool(rollback_id),
        actual=bool(rollback_id),
        expected={"present": True},
        summary="rollback target model id is present",
    )

    for role in PROMOTION_DECISION_REQUIRED_ARTIFACTS:
        record = artifacts[role]
        _add_check(
            checks,
            f"{role}_present",
            record["exists"] is True and record["kind"] == "file",
            actual={
                "exists": record["exists"],
                "kind": record["kind"],
                "path": record["path"],
            },
            expected={"exists": True, "kind": "file"},
            scope={"artifact_role": role},
            summary=f"{role} artifact is present and fingerprinted",
        )

    for role in _JSON_ARTIFACT_ROLES:
        _add_schema_check(checks, role, json_artifacts.get(role))
        _add_json_schema_contract_check(checks, role, json_artifacts.get(role))
    for role in _PASSED_JSON_ROLES:
        _add_passed_json_check(checks, role, json_artifacts.get(role))

    structured_validation_errors = {
        role: _promotion_structured_validation_errors(
            role,
            Path(paths[role]) if paths[role] is not None else None,
            json_artifacts.get(role),
        )
        for role in (
            "promotion_ledger_gate",
            "compare_gate",
            "trainer_launch_check",
            "model_registry_entry",
            "agentic_training_result",
            "serving_profile",
        )
    }
    for role, errors in structured_validation_errors.items():
        _add_check(
            checks,
            f"{role}_semantically_valid",
            not errors,
            actual={"error_count": len(errors)},
            expected={"error_count": 0},
            scope={"artifact_role": role},
            summary=f"{role} passes deterministic semantic validation before promotion",
        )
    compare_candidate_id = _promotion_compare_candidate_id(
        json_artifacts.get("compare_gate"),
        Path(compare_gate_path) if compare_gate_path is not None else None,
    )
    _add_check(
        checks,
        "compare_gate_matches_candidate",
        bool(candidate_id) and compare_candidate_id == candidate_id,
        actual={
            "candidate_id": candidate_id,
            "compare_candidate_id": compare_candidate_id,
        },
        expected={"same_candidate_id": True},
        scope={"artifact_role": "compare_gate"},
        summary="compare gate source export identifies the promoted candidate",
    )
    training_result_target = _promotion_training_result_target(
        json_artifacts.get("agentic_training_result")
    )
    _add_check(
        checks,
        "agentic_training_result_matches_candidate",
        bool(candidate_id) and training_result_target == candidate_id,
        actual={
            "candidate_id": candidate_id,
            "training_result_target_model_id": training_result_target,
        },
        expected={"same_candidate_id": True},
        scope={"artifact_role": "agentic_training_result"},
        summary="agentic training result registry target identifies the promoted candidate",
    )

    evidence_bundle_validation_errors = _promotion_semantic_validation_errors(
        "evidence_bundle", Path(evidence_bundle_path) if evidence_bundle_path else None
    )
    eval_summary_validation_errors = _promotion_semantic_validation_errors(
        "eval_summary", Path(eval_summary_path) if eval_summary_path else None
    )
    external_result_validation_errors = [
        _promotion_semantic_validation_errors("external_eval_result", path)
        for path in result_paths
    ]
    external_eval_lineage = _promotion_external_eval_lineage(
        candidate_id=candidate_id,
        evidence_bundle=json_artifacts.get("evidence_bundle"),
        eval_summary=json_artifacts.get("eval_summary"),
        eval_summary_artifact=artifacts["eval_summary"],
        result_payloads=result_payloads,
        result_artifacts=result_artifacts,
        evidence_bundle_semantically_valid=not evidence_bundle_validation_errors,
        eval_summary_semantically_valid=not eval_summary_validation_errors,
        external_results_semantically_valid=(
            bool(result_paths)
            and not any(external_result_validation_errors)
        ),
    )
    checks.extend(
        _promotion_external_eval_checks(
            external_eval_lineage,
            evidence_bundle_semantically_valid=not evidence_bundle_validation_errors,
            eval_summary_semantically_valid=not eval_summary_validation_errors,
            evidence_bundle_error_count=len(evidence_bundle_validation_errors),
            eval_summary_error_count=len(eval_summary_validation_errors),
            external_result_error_count=sum(
                len(errors) for errors in external_result_validation_errors
            ),
        )
    )

    compare_metrics = _metrics_object(json_artifacts.get("compare_gate"))
    _add_compare_metrics_check(checks, compare_metrics)
    limits = policy["limits"]
    _add_max_count_check(
        checks,
        "task_completion_regressions_absent",
        compare_metrics.get("task_completion_regression_count"),
        limits["max_task_completion_regressions"],
    )
    _add_max_count_check(checks, "baseline_wins_absent", compare_metrics.get("baseline_win_count"), limits["max_baseline_wins"])
    _add_max_count_check(checks, "contract_drifts_absent", compare_metrics.get("contract_drift_count"), limits["max_contract_drifts"])
    _add_max_count_check(
        checks,
        "unverified_contracts_absent",
        compare_metrics.get("unverified_contract_count"),
        limits["max_unverified_contracts"],
    )
    _add_count_map_max_check(
        checks,
        "new_critical_failures_absent",
        compare_metrics.get("new_critical_failure_counts"),
        "new critical failures",
        limits["max_new_critical_failures"],
    )
    _add_count_map_max_check(
        checks,
        "rule_regressions_absent",
        compare_metrics.get("regressed_rule_counts"),
        "rule regressions",
        limits["max_rule_regressions"],
    )
    for rule_id in policy["forbid_new_critical_rules"]:
        _add_forbidden_rule_check(checks, compare_metrics.get("new_critical_failure_counts"), "new_critical", rule_id)
    for rule_id in policy["forbid_regressed_rules"]:
        _add_forbidden_rule_check(checks, compare_metrics.get("regressed_rule_counts"), "regression", rule_id)

    _add_license_check(checks, json_artifacts.get("license_review"))
    _add_model_registry_entry_check(checks, json_artifacts.get("model_registry_entry"), candidate_id)
    _add_serving_profile_check(checks, json_artifacts.get("serving_profile"))
    _add_rollback_metadata_check(checks, json_artifacts.get("rollback_metadata"), rollback_id)
    _add_card_claims_check(checks, "model_card", Path(model_card_path) if model_card_path else None, reject_symlink_components=True)
    _add_card_claims_check(checks, "dataset_card", Path(dataset_card_path) if dataset_card_path else None, reject_symlink_components=True)

    failed_before_alias = sum(1 for check in checks if not check["passed"])
    _add_check(
        checks,
        "alias_update_authorized",
        failed_before_alias == 0,
        actual={"blocking_check_count": failed_before_alias},
        expected={"blocking_check_count": 0},
        summary="champion/candidate/rollback aliases may move only after every governance check passes",
    )

    failed_checks = sum(1 for check in checks if not check["passed"])
    passed = failed_checks == 0
    metrics = _decision_metrics(checks, compare_metrics, policy, external_eval_lineage)
    decision = {
        "readiness": "ready" if passed else "blocked",
        "recommendation": "apply_alias_update" if passed else "block_promotion",
        "summary": _decision_summary(passed, failed_checks, metrics),
        "blocking_check_count": failed_checks,
        "blocking_checks": [
            {"id": check["id"], "summary": check["summary"], "scope": check.get("scope", {})}
            for check in checks
            if not check["passed"]
        ],
        "key_metrics": metrics,
    }
    result: dict[str, Any] = {
        "schema_version": PROMOTION_DECISION_SCHEMA_VERSION,
        "decision_path": _display_path(Path(out_path), preserve_paths) if out_path is not None else "",
        "passed": passed,
        "readiness": decision["readiness"],
        "recommendation": decision["recommendation"],
        "models": {
            "candidate": {"id": candidate_id, "class": candidate_class},
            "champion": {"id": champion_id, "class": champion_class},
            "rollback": {"id": rollback_id or "", "class": "rollback"},
        },
        "decision": decision,
        "check_count": len(checks),
        "failed_check_count": failed_checks,
        "checks": checks,
        "artifacts": artifacts,
        "external_eval_lineage": external_eval_lineage,
        "policy": _promotion_policy_output(policy, policy_artifact),
        "metrics": metrics,
        "alias_update": _alias_update(passed, candidate_id, champion_id, rollback_id or ""),
        "notes": [
            "Promotion decisions are side-effect free; they do not move registry aliases.",
            "Alias movement is authorized only when every required governance artifact is present, fingerprinted, and passed.",
            "Promotion requires a semantically valid eval summary bound to the evidence bundle and the exact passing external-result set for the candidate model.",
            "Blocked checks cover missing evidence, unknown license, redaction/safety failures, missing cards, missing rollback, eval mismatch, critical failures, secret exposure, forbidden actions, unsupported claims, and task-completion regressions.",
        ],
    }
    if metadata:
        result["metadata"] = dict(sorted(metadata.items()))
    return result


def apply_promotion_aliases(
    *,
    registry_path: str | Path,
    promotion_decision_path: str | Path,
    out_path: str | Path,
    promotion_decision_validation: dict[str, Any] | None = None,
    preserve_paths: bool = False,
    metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Apply candidate/champion/rollback aliases only from a passing promotion decision."""
    registry_file = Path(registry_path)
    decision_file = Path(promotion_decision_path)
    receipt_path = Path(out_path)
    _reject_unowned_json_output(
        receipt_path,
        PROMOTION_ALIAS_APPLY_SCHEMA_VERSION,
        label="promotion alias receipt",
        error_type=PromotionDecisionError,
    )
    _reject_output_source_collision(
        receipt_path,
        [registry_file, decision_file],
        label="promotion alias receipt",
        error_type=PromotionDecisionError,
    )
    receipt_expected_sha256 = json_file_sha256(receipt_path)
    registry_before_record = _artifact_record("registry", registry_file, preserve_paths, receipt_path, reject_symlink_components=True)
    decision_record_before_validation = _artifact_record(
        "promotion_decision",
        decision_file,
        preserve_paths,
        receipt_path,
        reject_symlink_components=True,
    )
    registry, registry_snapshot_sha256, registry_snapshot_size = _read_json_artifact_snapshot(
        registry_file, reject_symlink_components=True
    )
    decision, decision_snapshot_sha256, decision_snapshot_size = _read_json_artifact_snapshot(
        decision_file, reject_symlink_components=True
    )
    _reject_output_source_collision(
        receipt_path,
        _artifact_source_closure(
            [
                registry_file,
                decision_file,
                *_promotion_decision_semantic_source_paths(decision, decision_file),
            ]
        ),
        label="promotion alias receipt",
        error_type=PromotionDecisionError,
    )
    internal_decision_validation = _validate_promotion_decision_snapshot(
        decision,
        decision_file,
    )
    internal_registry_validation = _validate_model_registry_snapshot(
        registry,
        registry_file,
    )
    registry_record_after_snapshot = _artifact_record(
        "registry",
        registry_file,
        preserve_paths,
        receipt_path,
        reject_symlink_components=True,
    )
    decision_record_after_validation = _artifact_record(
        "promotion_decision",
        decision_file,
        preserve_paths,
        receipt_path,
        reject_symlink_components=True,
    )
    decision_stable_during_validation = _artifact_records_match(
        decision_record_before_validation,
        decision_record_after_validation,
    ) and _artifact_record_matches_snapshot(
        decision_record_after_validation,
        decision_snapshot_sha256,
        decision_snapshot_size,
    )
    decision_record = decision_record_after_validation
    registry_stable_before_update = _artifact_records_match(
        registry_before_record,
        registry_record_after_snapshot,
    ) and _artifact_record_matches_snapshot(
        registry_before_record,
        registry_snapshot_sha256,
        registry_snapshot_size,
    )
    registry_obj = registry if isinstance(registry, dict) else {}
    decision_obj = decision if isinstance(decision, dict) else {}
    aliases_before = _registry_aliases(registry_obj)
    alias_history_before = registry_obj.get("alias_history")
    model_ids = _registry_model_ids(registry_obj)
    decision_models = decision_obj.get("models") if isinstance(decision_obj.get("models"), dict) else {}
    candidate_id = _model_id(decision_models.get("candidate"))
    champion_id = _model_id(decision_models.get("champion"))
    rollback_id = _model_id(decision_models.get("rollback"))
    alias_update = decision_obj.get("alias_update") if isinstance(decision_obj.get("alias_update"), dict) else {}
    decision_registry_entry_path = _promotion_decision_artifact_source_path(
        decision_obj,
        "model_registry_entry",
        decision_file,
    )
    (
        decision_registry_entry,
        decision_registry_entry_sha256,
        decision_registry_entry_size,
    ) = (
        _read_json_artifact_snapshot(
            decision_registry_entry_path,
            reject_symlink_components=True,
        )
        if decision_registry_entry_path is not None
        else (None, None, None)
    )
    decision_artifacts = (
        decision_obj.get("artifacts")
        if isinstance(decision_obj.get("artifacts"), dict)
        else {}
    )
    decision_registry_entry_record = decision_artifacts.get("model_registry_entry")
    decision_registry_entry_fingerprint_matches = (
        isinstance(decision_registry_entry_record, dict)
        and decision_registry_entry_record.get("sha256")
        == decision_registry_entry_sha256
        and decision_registry_entry_record.get("size_bytes")
        == decision_registry_entry_size
    )
    registry_entries = (
        registry_obj.get("entries")
        if isinstance(registry_obj.get("entries"), dict)
        else {}
    )
    registered_candidate_entry = registry_entries.get(candidate_id)

    checks: list[dict[str, Any]] = []
    _add_check(
        checks,
        "registry_present",
        registry_before_record["exists"] is True and registry_before_record["kind"] == "file",
        actual={"exists": registry_before_record["exists"], "kind": registry_before_record["kind"]},
        expected={"exists": True, "kind": "file"},
        scope={"artifact_role": "registry"},
        summary="model registry file is present",
    )
    _add_check(
        checks,
        "registry_schema",
        registry_obj.get("schema_version") == MODEL_REGISTRY_SCHEMA_VERSION,
        actual=registry_obj.get("schema_version"),
        expected={"schema_version": MODEL_REGISTRY_SCHEMA_VERSION},
        scope={"artifact_role": "registry"},
        summary="model registry uses the expected schema version",
    )
    _add_check(
        checks,
        "registry_validated",
        internal_registry_validation["passed"] is True,
        actual={
            "error_count": internal_registry_validation["error_count"],
            "warning_count": internal_registry_validation["warning_count"],
        },
        expected={"error_count": 0, "warning_count": 0},
        scope={"artifact_role": "registry"},
        summary="model registry snapshot passes strict semantic validation",
    )
    _add_check(
        checks,
        "registry_stable_before_update",
        registry_stable_before_update,
        actual={"stable": registry_stable_before_update},
        expected={"stable": True},
        scope={"artifact_role": "registry"},
        summary="registry bytes used for alias updates match the CAS fingerprint",
    )
    _add_check(
        checks,
        "promotion_decision_present",
        decision_record["exists"] is True and decision_record["kind"] == "file",
        actual={"exists": decision_record["exists"], "kind": decision_record["kind"]},
        expected={"exists": True, "kind": "file"},
        scope={"artifact_role": "promotion_decision"},
        summary="promotion decision file is present",
    )
    _add_check(
        checks,
        "promotion_decision_schema",
        decision_obj.get("schema_version") == PROMOTION_DECISION_SCHEMA_VERSION,
        actual=decision_obj.get("schema_version"),
        expected={"schema_version": PROMOTION_DECISION_SCHEMA_VERSION},
        scope={"artifact_role": "promotion_decision"},
        summary="promotion decision uses the expected schema version",
    )
    _add_check(
        checks,
        "promotion_decision_passed",
        decision_obj.get("passed") is True,
        actual=decision_obj.get("passed"),
        expected={"passed": True},
        scope={"artifact_role": "promotion_decision"},
        summary="promotion decision passed before aliases can move",
    )
    _add_check(
        checks,
        "promotion_decision_authorizes_alias_update",
        decision_obj.get("recommendation") == "apply_alias_update" and alias_update.get("authorized") is True,
        actual={"recommendation": decision_obj.get("recommendation"), "authorized": alias_update.get("authorized")},
        expected={"recommendation": "apply_alias_update", "authorized": True},
        scope={"artifact_role": "promotion_decision"},
        summary="promotion decision explicitly authorizes alias movement",
    )
    validation_passed = internal_decision_validation.get("passed")
    _add_check(
        checks,
        "promotion_decision_validated",
        validation_passed is True,
        actual=validation_passed,
        expected={"passed": True},
        scope={"artifact_role": "promotion_decision"},
        summary="promotion decision validation passed immediately before alias movement",
    )
    _add_check(
        checks,
        "promotion_decision_stable_during_validation",
        decision_stable_during_validation,
        actual={"stable": decision_stable_during_validation},
        expected={"stable": True},
        scope={"artifact_role": "promotion_decision"},
        summary="promotion decision bytes remain unchanged across validation and use",
    )
    _add_check(
        checks,
        "champion_previous_target_registered",
        bool(champion_id) and champion_id in model_ids,
        actual={"target": champion_id, "registered": champion_id in model_ids},
        expected={"registered": True},
        scope={"alias": "champion"},
        summary="current champion previous target is registered",
    )
    _add_check(
        checks,
        "promotion_decision_alias_targets_match_models",
        _alias_update_targets(alias_update)
        == {"candidate": candidate_id, "champion": candidate_id, "rollback": rollback_id},
        actual=_alias_update_targets(alias_update),
        expected={"candidate": candidate_id, "champion": candidate_id, "rollback": rollback_id},
        scope={"artifact_role": "promotion_decision"},
        summary="promotion decision alias receipt matches named candidate and rollback models",
    )
    for alias_name, target in (("candidate", candidate_id), ("champion", candidate_id), ("rollback", rollback_id)):
        _add_check(
            checks,
            f"{alias_name}_target_registered",
            bool(target) and target in model_ids,
            actual={"target": target, "registered": target in model_ids},
            expected={"registered": True},
            scope={"alias": alias_name},
            summary=f"{alias_name} alias target is registered",
        )
    _add_check(
        checks,
        "champion_alias_matches_previous_target",
        bool(champion_id) and aliases_before.get("champion") == champion_id,
        actual={"champion_alias": aliases_before.get("champion"), "expected_previous": champion_id},
        expected={"champion_alias": champion_id},
        scope={"alias": "champion"},
        summary="current champion alias matches the promotion decision previous target",
    )
    _add_check(
        checks,
        "registry_aliases_object",
        isinstance(registry_obj.get("aliases"), dict),
        actual=type(registry_obj.get("aliases")).__name__,
        expected={"type": "object"},
        scope={"artifact_role": "registry"},
        summary="model registry aliases are stored as an object",
    )
    _add_check(
        checks,
        "registry_candidate_entry_matches_decision_artifact",
        isinstance(decision_registry_entry, dict)
        and decision_registry_entry_fingerprint_matches
        and registered_candidate_entry == decision_registry_entry,
        actual={
            "candidate_id": candidate_id,
            "decision_entry_present": isinstance(decision_registry_entry, dict),
            "decision_entry_fingerprint_matches": decision_registry_entry_fingerprint_matches,
            "registry_entry_matches": registered_candidate_entry
            == decision_registry_entry,
        },
        expected={"same_candidate_entry": True},
        scope={"artifact_role": "model_registry_entry"},
        summary="registry candidate entry exactly matches the promotion decision artifact",
    )
    _add_check(
        checks,
        "registry_alias_history_list",
        alias_history_before is None or isinstance(alias_history_before, list),
        actual="missing" if alias_history_before is None else type(alias_history_before).__name__,
        expected={"type": "list_or_missing"},
        scope={"artifact_role": "registry"},
        summary="model registry alias history can record the alias movement",
    )

    failed_checks = sum(1 for check in checks if not check["passed"])
    passed = failed_checks == 0
    aliases_after = dict(aliases_before)
    alias_history_entry: dict[str, Any] | None = None
    registry_after_record = registry_before_record
    alias_history_count_before = len(alias_history_before) if isinstance(alias_history_before, list) else 0
    alias_history_count_after = alias_history_count_before
    registry_after_sha256: str | None = None
    if passed:
        updated = copy.deepcopy(registry_obj)
        updated_aliases = updated.setdefault("aliases", {})
        updated_aliases["candidate"] = candidate_id
        updated_aliases["champion"] = candidate_id
        updated_aliases["rollback"] = rollback_id
        history = updated.setdefault("alias_history", [])
        alias_history_entry = {
            "promotion_decision_sha256": decision_record.get("sha256"),
            "previous_aliases": aliases_before,
            "updated_aliases": {
                "candidate": candidate_id,
                "champion": candidate_id,
                "rollback": rollback_id,
            },
        }
        history.append(alias_history_entry)
        registry_after_sha256 = atomic_write_json_cas(
            registry_file,
            updated,
            expected_sha256=registry_snapshot_sha256,
        )
        registry_after_record = _artifact_record("registry", registry_file, preserve_paths, receipt_path, reject_symlink_components=True)
        (
            registry_after_payload,
            observed_registry_after_sha256,
            observed_registry_after_size,
        ) = _read_json_artifact_snapshot(
            registry_file,
            reject_symlink_components=True,
        )
        registry_after_stable = (
            observed_registry_after_sha256 == registry_after_sha256
            and registry_after_payload == updated
            and registry_after_record.get("sha256") == registry_after_sha256
            and registry_after_record.get("size_bytes")
            == observed_registry_after_size
        )
        if not registry_after_stable:
            try:
                atomic_write_json_cas(
                    registry_file,
                    registry_obj,
                    expected_sha256=registry_after_sha256,
                )
            except (AtomicJsonError, OSError) as rollback_error:
                raise PromotionDecisionError(
                    "model registry changed after alias update and guarded rollback could not be completed"
                ) from rollback_error
            raise PromotionDecisionError(
                "model registry changed after alias update before receipt publication"
            )
        aliases_after = _registry_aliases(updated)
        alias_history_count_after += 1

    receipt: dict[str, Any] = {
        "schema_version": PROMOTION_ALIAS_APPLY_SCHEMA_VERSION,
        "receipt_path": _display_path(receipt_path, preserve_paths),
        "passed": passed,
        "readiness": "applied" if passed else "blocked",
        "recommendation": "alias_update_applied" if passed else "hold_aliases",
        "check_count": len(checks),
        "failed_check_count": failed_checks,
        "checks": checks,
        "promotion_decision": {
            "path": decision_record.get("path", ""),
            "sha256": decision_record.get("sha256"),
            "size_bytes": _int_value(decision_record.get("size_bytes")),
            "candidate_id": candidate_id,
            "champion_previous_target": champion_id,
            "rollback_id": rollback_id,
        },
        "promotion_decision_validation": _validation_summary(internal_decision_validation),
        "registry_before": {
            "aliases": aliases_before,
        },
        "registry_after": {
            "path": registry_after_record.get("path", ""),
            "sha256": registry_after_record.get("sha256"),
            "size_bytes": _int_value(registry_after_record.get("size_bytes")),
            "aliases": aliases_after,
        },
        "alias_history_entry": alias_history_entry,
        "artifacts": {
            "registry": registry_after_record,
            "promotion_decision": decision_record,
        },
        "metrics": {
            "check_count": len(checks),
            "failed_check_count": failed_checks,
            "registered_model_count": len(model_ids),
            "alias_count_before": len(aliases_before),
            "alias_count_after": len(aliases_after),
            "alias_history_count_before": alias_history_count_before,
            "alias_history_count_after": alias_history_count_after,
        },
        "notes": [
            "Promotion alias application is the guarded side-effectful step after promotion-decision validation.",
            "Registry aliases are written only when the decision passed, authorized alias movement, and the current champion alias still matches the decision previous target.",
        ],
    }
    if metadata:
        receipt["metadata"] = dict(sorted(metadata.items()))
    try:
        atomic_write_json_cas(
            receipt_path,
            receipt,
            expected_sha256=receipt_expected_sha256,
        )
    except (AtomicJsonError, OSError):
        if passed and registry_after_sha256 is not None:
            try:
                atomic_write_json_cas(
                    registry_file,
                    registry_obj,
                    expected_sha256=registry_after_sha256,
                )
            except (AtomicJsonError, OSError) as rollback_error:
                raise PromotionDecisionError(
                    "promotion alias receipt publication failed and registry rollback could not be completed"
                ) from rollback_error
        raise
    return receipt


def build_promotion_rollback_receipt(
    *,
    registry_path: str | Path,
    rollback_id: str,
    out_path: str | Path,
    champion_id: str | None = None,
    preserve_paths: bool = False,
    metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Prove the rollback target is registered before a promotion decision consumes it."""
    registry_file = Path(registry_path)
    receipt_path = Path(out_path)
    registry_record = _artifact_record("registry", registry_file, preserve_paths, receipt_path, reject_symlink_components=True)
    registry = _read_json_artifact(registry_file, reject_symlink_components=True)
    registry_obj = registry if isinstance(registry, dict) else {}
    aliases = _registry_aliases(registry_obj)
    model_ids = _registry_model_ids(registry_obj)
    expected_champion_id = champion_id or aliases.get("champion", "")

    checks: list[dict[str, Any]] = []
    _add_check(
        checks,
        "rollback_id_present",
        bool(rollback_id),
        actual=bool(rollback_id),
        expected={"present": True},
        summary="rollback target model id is present",
    )
    _add_check(
        checks,
        "registry_present",
        registry_record["exists"] is True and registry_record["kind"] == "file",
        actual={"exists": registry_record["exists"], "kind": registry_record["kind"], "path": registry_record["path"]},
        expected={"exists": True, "kind": "file"},
        scope={"artifact_role": "registry"},
        summary="model registry file is present",
    )
    _add_check(
        checks,
        "registry_schema",
        registry_obj.get("schema_version") == MODEL_REGISTRY_SCHEMA_VERSION,
        actual=registry_obj.get("schema_version"),
        expected={"schema_version": MODEL_REGISTRY_SCHEMA_VERSION},
        scope={"artifact_role": "registry"},
        summary="model registry uses the expected schema version",
    )
    _add_check(
        checks,
        "registry_aliases_object",
        isinstance(registry_obj.get("aliases"), dict),
        actual=type(registry_obj.get("aliases")).__name__,
        expected={"type": "object"},
        scope={"artifact_role": "registry"},
        summary="model registry aliases are stored as an object",
    )
    _add_check(
        checks,
        "champion_id_present",
        bool(expected_champion_id),
        actual=bool(expected_champion_id),
        expected={"present": True},
        scope={"alias": "champion"},
        summary="current champion model id is known",
    )
    _add_check(
        checks,
        "champion_alias_matches_target",
        bool(expected_champion_id) and aliases.get("champion") == expected_champion_id,
        actual={"champion_alias": aliases.get("champion"), "expected_champion": expected_champion_id},
        expected={"champion_alias": expected_champion_id},
        scope={"alias": "champion"},
        summary="registry champion alias matches the expected current champion",
    )
    for role, model_id in (("rollback", rollback_id), ("champion", expected_champion_id)):
        _add_check(
            checks,
            f"{role}_target_registered",
            bool(model_id) and model_id in model_ids,
            actual={"target": model_id, "registered": model_id in model_ids},
            expected={"registered": True},
            scope={"alias": role},
            summary=f"{role} target is registered in the model registry",
        )
    _add_check(
        checks,
        "rollback_target_is_current_champion",
        bool(rollback_id) and bool(expected_champion_id) and rollback_id == expected_champion_id,
        actual={"rollback_id": rollback_id, "champion_id": expected_champion_id},
        expected={"same_model": True},
        scope={"artifact_role": "rollback_metadata"},
        summary="rollback target points to the current champion before candidate promotion",
    )

    failed_checks = sum(1 for check in checks if not check["passed"])
    passed = failed_checks == 0
    receipt: dict[str, Any] = {
        "schema_version": PROMOTION_ROLLBACK_RECEIPT_SCHEMA_VERSION,
        "receipt_path": _display_path(receipt_path, preserve_paths),
        "passed": passed,
        "available": passed,
        "readiness": "ready" if passed else "blocked",
        "recommendation": "use_rollback_target" if passed else "block_promotion",
        "rollback_id": rollback_id,
        "target_model_id": rollback_id,
        "champion_id": expected_champion_id,
        "check_count": len(checks),
        "failed_check_count": failed_checks,
        "checks": checks,
        "rollback": {
            "id": rollback_id,
            "target_model_id": rollback_id,
            "champion_id": expected_champion_id,
            "available": passed,
        },
        "registry": {
            "path": registry_record.get("path", ""),
            "sha256": registry_record.get("sha256"),
            "size_bytes": _int_value(registry_record.get("size_bytes")),
            "aliases": aliases,
        },
        "artifacts": {"registry": registry_record},
        "metrics": {
            "check_count": len(checks),
            "failed_check_count": failed_checks,
            "registered_model_count": len(model_ids),
            "alias_count": len(aliases),
        },
        "notes": [
            "Rollback receipts are side-effect free; they do not move registry aliases.",
            "A passing receipt proves the rollback target is registered and still matches the current champion before promotion.",
        ],
    }
    if metadata:
        receipt["metadata"] = dict(sorted(metadata.items()))
    _write_json(receipt_path, receipt)
    return receipt


def build_promotion_release_record(
    *,
    release_id: str,
    promotion_decision_path: str | Path,
    promotion_cards_path: str | Path,
    promotion_alias_apply_path: str | Path,
    rollback_metadata_path: str | Path,
    compare_gate_path: str | Path,
    release_notes_path: str | Path,
    out_path: str | Path,
    promotion_policy_path: str | Path | None = None,
    artifact_validation: dict[str, Any] | None = None,
    preserve_paths: bool = False,
    metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Bind promotion governance artifacts into a reviewable release record."""
    record_path = Path(out_path)
    paths = {
        "promotion_decision": Path(promotion_decision_path),
        "promotion_cards": Path(promotion_cards_path),
        "promotion_alias_apply": Path(promotion_alias_apply_path),
        "rollback_metadata": Path(rollback_metadata_path),
        "compare_gate": Path(compare_gate_path),
        "release_notes": Path(release_notes_path),
    }
    artifacts = {
        role: _artifact_record(role, path, preserve_paths, record_path, reject_symlink_components=True)
        for role, path in paths.items()
    }
    decision = _read_json_artifact(paths["promotion_decision"], reject_symlink_components=True)
    cards_manifest_path = _promotion_cards_manifest_path(paths["promotion_cards"])
    cards = _read_json_artifact(cards_manifest_path, reject_symlink_components=True) if cards_manifest_path is not None else None
    alias_receipt = _read_json_artifact(paths["promotion_alias_apply"], reject_symlink_components=True)
    rollback = _read_json_artifact(paths["rollback_metadata"], reject_symlink_components=True)
    compare_gate = _read_json_artifact(paths["compare_gate"], reject_symlink_components=True)
    notes_text = _read_text_artifact(paths["release_notes"], reject_symlink_components=True)

    decision_obj = decision if isinstance(decision, dict) else {}
    cards_obj = cards if isinstance(cards, dict) else {}
    alias_obj = alias_receipt if isinstance(alias_receipt, dict) else {}
    policy_path = Path(promotion_policy_path) if promotion_policy_path else None
    policy_artifact = (
        _artifact_record("promotion_policy", policy_path, preserve_paths, record_path, reject_symlink_components=True)
        if policy_path
        else None
    )
    policy = _load_promotion_policy(policy_path, preserve_paths, reject_symlink_components=True) if policy_path else None
    decision_policy = decision_obj.get("policy") if isinstance(decision_obj.get("policy"), dict) else {}
    decision_models = decision_obj.get("models") if isinstance(decision_obj.get("models"), dict) else {}
    candidate_id = _model_id(decision_models.get("candidate"))
    champion_id = _model_id(decision_models.get("champion"))
    rollback_id = _model_id(decision_models.get("rollback"))
    dataset = cards_obj.get("dataset") if isinstance(cards_obj.get("dataset"), dict) else {}
    dataset_id = dataset.get("id") if isinstance(dataset.get("id"), str) else ""

    checks: list[dict[str, Any]] = []
    _add_check(
        checks,
        "release_id_present",
        bool(release_id),
        actual=bool(release_id),
        expected={"present": True},
        summary="release id is present",
    )
    for role in PROMOTION_RELEASE_RECORD_REQUIRED_ARTIFACTS:
        record = artifacts[role]
        _add_check(
            checks,
            f"{role}_present",
            record["exists"] is True,
            actual={"exists": record["exists"], "kind": record["kind"], "path": record["path"]},
            expected={"exists": True},
            scope={"artifact_role": role},
            summary=f"{role} artifact is present and fingerprinted",
        )
    _add_check(
        checks,
        "promotion_decision_schema",
        decision_obj.get("schema_version") == PROMOTION_DECISION_SCHEMA_VERSION,
        actual=decision_obj.get("schema_version"),
        expected={"schema_version": PROMOTION_DECISION_SCHEMA_VERSION},
        scope={"artifact_role": "promotion_decision"},
        summary="promotion decision uses the expected schema",
    )
    _add_check(
        checks,
        "promotion_cards_schema",
        cards_obj.get("schema_version") == PROMOTION_CARDS_SCHEMA_VERSION,
        actual=cards_obj.get("schema_version"),
        expected={"schema_version": PROMOTION_CARDS_SCHEMA_VERSION},
        scope={"artifact_role": "promotion_cards"},
        summary="promotion cards manifest uses the expected schema",
    )
    _add_check(
        checks,
        "promotion_alias_apply_schema",
        alias_obj.get("schema_version") == PROMOTION_ALIAS_APPLY_SCHEMA_VERSION,
        actual=alias_obj.get("schema_version"),
        expected={"schema_version": PROMOTION_ALIAS_APPLY_SCHEMA_VERSION},
        scope={"artifact_role": "promotion_alias_apply"},
        summary="promotion alias receipt uses the expected schema",
    )
    _add_schema_check(checks, "compare_gate", compare_gate)
    for role, payload in (
        ("promotion_decision", decision),
        ("promotion_cards", cards),
        ("promotion_alias_apply", alias_receipt),
        ("compare_gate", compare_gate),
    ):
        _add_passed_json_check(checks, role, payload)

    validation_passed = artifact_validation.get("passed") if isinstance(artifact_validation, dict) else None
    _add_check(
        checks,
        "input_artifacts_validated",
        validation_passed is True,
        actual=validation_passed,
        expected={"passed": True},
        summary="release-record input artifacts passed validation immediately before binding",
    )
    if policy_artifact is not None and policy is not None:
        _add_check(
            checks,
            "promotion_policy_present",
            policy_artifact["exists"] is True and policy_artifact["kind"] == "file",
            actual={"exists": policy_artifact["exists"], "kind": policy_artifact["kind"], "path": policy_artifact["path"]},
            expected={"exists": True, "kind": "file"},
            scope={"artifact_role": "promotion_policy"},
            summary="promotion policy artifact is present for release binding",
        )
        _add_check(
            checks,
            "promotion_policy_schema",
            policy.get("schema_version") == PROMOTION_POLICY_SCHEMA_VERSION,
            actual=policy.get("schema_version"),
            expected={"schema_version": PROMOTION_POLICY_SCHEMA_VERSION},
            scope={"artifact_role": "promotion_policy"},
            summary="promotion policy uses the expected schema version",
        )
        _add_artifact_contract_check(
            checks,
            "promotion_policy_release_artifacts_complete",
            policy.get("release_required_artifacts"),
            PROMOTION_RELEASE_RECORD_REQUIRED_ARTIFACTS,
            "promotion policy covers every release-record artifact role",
        )
        decision_policy_artifact = decision_policy.get("artifact") if isinstance(decision_policy.get("artifact"), dict) else {}
        _add_check(
            checks,
            "promotion_policy_matches_decision",
            policy_artifact.get("sha256") == decision_policy_artifact.get("sha256"),
            actual={"release_policy_sha256": policy_artifact.get("sha256"), "decision_policy_sha256": decision_policy_artifact.get("sha256")},
            expected={"same_sha256": True},
            scope={"artifact_role": "promotion_policy"},
            summary="release policy artifact matches the policy embedded in the promotion decision",
        )
    alias_decision = alias_obj.get("promotion_decision") if isinstance(alias_obj.get("promotion_decision"), dict) else {}
    _add_check(
        checks,
        "alias_receipt_matches_decision",
        alias_decision.get("sha256") == artifacts["promotion_decision"].get("sha256"),
        actual=alias_decision.get("sha256"),
        expected={"promotion_decision_sha256": artifacts["promotion_decision"].get("sha256")},
        scope={"artifact_role": "promotion_alias_apply"},
        summary="alias receipt references the exact promotion decision",
    )
    _add_check(
        checks,
        "alias_receipt_targets_match_decision",
        alias_decision.get("candidate_id") == candidate_id and alias_decision.get("rollback_id") == rollback_id,
        actual={"candidate_id": alias_decision.get("candidate_id"), "rollback_id": alias_decision.get("rollback_id")},
        expected={"candidate_id": candidate_id, "rollback_id": rollback_id},
        scope={"artifact_role": "promotion_alias_apply"},
        summary="alias receipt candidate and rollback targets match the decision",
    )
    cards_artifacts = cards_obj.get("artifacts") if isinstance(cards_obj.get("artifacts"), dict) else {}
    decision_artifacts = decision_obj.get("artifacts") if isinstance(decision_obj.get("artifacts"), dict) else {}
    _add_check(
        checks,
        "cards_match_decision",
        _artifact_sha(cards_artifacts.get("model_card")) == _artifact_sha(decision_artifacts.get("model_card"))
        and _artifact_sha(cards_artifacts.get("dataset_card")) == _artifact_sha(decision_artifacts.get("dataset_card")),
        actual={
            "cards_model_card": _artifact_sha(cards_artifacts.get("model_card")),
            "decision_model_card": _artifact_sha(decision_artifacts.get("model_card")),
            "cards_dataset_card": _artifact_sha(cards_artifacts.get("dataset_card")),
            "decision_dataset_card": _artifact_sha(decision_artifacts.get("dataset_card")),
        },
        expected={"cards": "match_decision_card_hashes"},
        scope={"artifact_role": "promotion_cards"},
        summary="promotion cards are the same cards consumed by the promotion decision",
    )
    _add_check(
        checks,
        "compare_gate_matches_decision",
        artifacts["compare_gate"].get("sha256") == _artifact_sha(decision_artifacts.get("compare_gate")),
        actual=artifacts["compare_gate"].get("sha256"),
        expected={"decision_compare_gate_sha256": _artifact_sha(decision_artifacts.get("compare_gate"))},
        scope={"artifact_role": "compare_gate"},
        summary="release eval compare gate matches the promotion decision compare gate",
    )
    _add_rollback_metadata_check(checks, rollback, rollback_id)
    unsupported_markers = [marker for marker in _UNSUPPORTED_CARD_MARKERS if marker in notes_text.lower()]
    _add_check(
        checks,
        "release_notes_nonempty",
        bool(notes_text.strip()),
        actual={"size_bytes": artifacts["release_notes"].get("size_bytes", 0), "nonempty": bool(notes_text.strip())},
        expected={"nonempty": True},
        scope={"artifact_role": "release_notes"},
        summary="release notes are present and non-empty",
    )
    _add_check(
        checks,
        "release_notes_claims_supported",
        not unsupported_markers,
        actual={"unsupported_markers": unsupported_markers},
        expected={"unsupported_markers": []},
        scope={"artifact_role": "release_notes"},
        summary="release notes contain no TODO/TBD/unsupported-claim markers",
    )

    expected_check_ids = promotion_release_record_check_ids(
        release_policy_bound=policy is not None,
        rollback_receipt_bound=isinstance(rollback, dict)
        and rollback.get("schema_version")
        == PROMOTION_ROLLBACK_RECEIPT_SCHEMA_VERSION,
    )
    if tuple(check.get("id") for check in checks) != expected_check_ids:
        raise PromotionDecisionError(
            "internal release-record checks do not match the canonical contract"
        )

    failed_checks = sum(1 for check in checks if not check["passed"])
    passed = failed_checks == 0
    record: dict[str, Any] = {
        "schema_version": PROMOTION_RELEASE_RECORD_SCHEMA_VERSION,
        "release_record_path": _display_path(record_path, preserve_paths),
        "passed": passed,
        "readiness": "ready" if passed else "blocked",
        "recommendation": "publish_release" if passed else "hold_release",
        "release": {
            "id": release_id,
            "candidate_id": candidate_id,
            "champion_previous_target": champion_id,
            "rollback_id": rollback_id,
            "dataset_id": dataset_id,
        },
        "check_count": len(checks),
        "failed_check_count": failed_checks,
        "checks": checks,
        "artifacts": artifacts,
        "policy": _release_policy_output(policy, policy_artifact, decision_policy),
        "artifact_validation": _validation_summary(artifact_validation),
        "bindings": {
            "promotion_decision_sha256": artifacts["promotion_decision"].get("sha256"),
            "promotion_cards_sha256": artifacts["promotion_cards"].get("sha256"),
            "promotion_alias_apply_sha256": artifacts["promotion_alias_apply"].get("sha256"),
            "rollback_metadata_sha256": artifacts["rollback_metadata"].get("sha256"),
            "compare_gate_sha256": artifacts["compare_gate"].get("sha256"),
            "release_notes_sha256": artifacts["release_notes"].get("sha256"),
            "model_card_sha256": _artifact_sha(cards_artifacts.get("model_card")),
            "dataset_card_sha256": _artifact_sha(cards_artifacts.get("dataset_card")),
        },
        "metrics": {
            "check_count": len(checks),
            "failed_check_count": failed_checks,
            "required_artifact_count": len(PROMOTION_RELEASE_RECORD_REQUIRED_ARTIFACTS),
            "release_notes_size_bytes": _int_value(artifacts["release_notes"].get("size_bytes")),
        },
        "notes": [
            "Release records are review artifacts; they do not move aliases or publish external artifacts.",
            "A release record binds the exact promotion decision, generated cards, alias-apply receipt, rollback metadata, eval compare gate, and release notes.",
        ],
    }
    if metadata:
        record["metadata"] = dict(sorted(metadata.items()))
    _write_json(record_path, record)
    return record


def build_promotion_cards(
    *,
    out_dir: str | Path,
    candidate_id: str,
    dataset_id: str,
    model_source: str = "",
    license_status: str = "",
    evidence_bundle_path: str | Path | None = None,
    training_export_path: str | Path | None = None,
    compare_gate_path: str | Path | None = None,
    redaction_check_path: str | Path | None = None,
    safety_gate_path: str | Path | None = None,
    preserve_paths: bool = False,
    metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Generate promotion model and dataset cards plus a validating manifest."""
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    model_card_path = target / "MODEL_CARD.md"
    dataset_card_path = target / "DATASET_CARD.md"
    manifest_path = target / "promotion_cards.json"

    paths = {
        "evidence_bundle": evidence_bundle_path,
        "training_export": training_export_path,
        "compare_gate": compare_gate_path,
        "redaction_check": redaction_check_path,
        "safety_gate": safety_gate_path,
    }
    input_artifacts = {
        role: _artifact_record(
            role,
            Path(raw_path) if raw_path else None,
            preserve_paths,
            manifest_path,
            reject_symlink_components=True,
        )
        for role, raw_path in paths.items()
    }
    json_artifacts = {
        role: _read_json_artifact(Path(raw_path), reject_symlink_components=True) if raw_path else None
        for role, raw_path in paths.items()
        if role != "training_export"
    }

    checks: list[dict[str, Any]] = []
    _add_check(
        checks,
        "candidate_id_present",
        bool(candidate_id),
        actual=bool(candidate_id),
        expected={"present": True},
        summary="candidate model id is present",
    )
    _add_check(
        checks,
        "dataset_id_present",
        bool(dataset_id),
        actual=bool(dataset_id),
        expected={"present": True},
        summary="dataset id is present",
    )
    _add_check(
        checks,
        "model_source_present",
        bool(model_source),
        actual=bool(model_source),
        expected={"present": True},
        summary="model source is present",
    )
    _add_check(
        checks,
        "license_status_known",
        _known_status(license_status),
        actual=license_status or "missing",
        expected={"license_status": "known"},
        scope={"field": "license_status"},
        summary="license status is known before card generation",
    )
    for role in PROMOTION_CARDS_REQUIRED_INPUTS:
        record = input_artifacts[role]
        _add_check(
            checks,
            f"{role}_present",
            record["exists"] is True,
            actual={"exists": record["exists"], "path": record["path"]},
            expected={"exists": True},
            scope={"artifact_role": role},
            summary=f"{role} input artifact is present and fingerprinted",
        )

    _add_schema_check(checks, "evidence_bundle", json_artifacts.get("evidence_bundle"))
    _add_schema_check(checks, "compare_gate", json_artifacts.get("compare_gate"))
    for role in ("evidence_bundle", "compare_gate", "redaction_check", "safety_gate"):
        _add_passed_json_check(checks, role, json_artifacts.get(role))

    compare_metrics = _metrics_object(json_artifacts.get("compare_gate"))
    _add_compare_metrics_check(checks, compare_metrics)
    _add_max_count_check(checks, "task_completion_regressions_absent", compare_metrics.get("task_completion_regression_count"), 0)
    _add_count_map_absence_check(
        checks,
        "new_critical_failures_absent",
        compare_metrics.get("new_critical_failure_counts"),
        "new critical failures",
    )

    failed_checks = sum(1 for check in checks if not check["passed"])
    passed = failed_checks == 0
    model_card = _render_model_card(
        candidate_id=candidate_id,
        model_source=model_source,
        license_status=license_status,
        passed=passed,
        input_artifacts=input_artifacts,
        compare_metrics=compare_metrics,
    )
    dataset_card = _render_dataset_card(
        dataset_id=dataset_id,
        candidate_id=candidate_id,
        passed=passed,
        input_artifacts=input_artifacts,
        compare_metrics=compare_metrics,
    )
    model_card_path.write_text(model_card, encoding="utf-8")
    dataset_card_path.write_text(dataset_card, encoding="utf-8")

    card_artifacts = {
        "model_card": _artifact_record("model_card", model_card_path, preserve_paths, manifest_path),
        "dataset_card": _artifact_record("dataset_card", dataset_card_path, preserve_paths, manifest_path),
    }
    metrics = {
        "check_count": len(checks),
        "failed_check_count": failed_checks,
        "required_input_count": len(PROMOTION_CARDS_REQUIRED_INPUTS),
        "task_completion_regression_count": _int_value(compare_metrics.get("task_completion_regression_count")),
        "new_critical_failure_count": sum(_count_map(compare_metrics.get("new_critical_failure_counts")).values()),
    }
    manifest: dict[str, Any] = {
        "schema_version": PROMOTION_CARDS_SCHEMA_VERSION,
        "manifest_path": _display_path(manifest_path, preserve_paths),
        "cards_dir": _display_path(target, preserve_paths),
        "passed": passed,
        "readiness": "ready" if passed else "blocked",
        "recommendation": "use_cards_for_promotion_decision" if passed else "regenerate_or_block_promotion",
        "candidate": {
            "id": candidate_id,
            "model_source": model_source,
            "license_status": license_status,
        },
        "dataset": {"id": dataset_id},
        "check_count": len(checks),
        "failed_check_count": failed_checks,
        "checks": checks,
        "artifacts": {**card_artifacts, **input_artifacts},
        "metrics": metrics,
        "notes": [
            "Promotion cards are generated review artifacts; they do not train models or move registry aliases.",
            "Cards are blocked when required inputs are missing, stale, unsafe, unredacted, or license status is unknown.",
        ],
    }
    if metadata:
        manifest["metadata"] = dict(sorted(metadata.items()))
    _write_json(manifest_path, manifest)
    return manifest


def _render_model_card(
    *,
    candidate_id: str,
    model_source: str,
    license_status: str,
    passed: bool,
    input_artifacts: dict[str, dict[str, Any]],
    compare_metrics: dict[str, Any],
) -> str:
    readiness = "ready" if passed else "blocked"
    lines = [
        "# Model Card",
        "",
        f"- Candidate model: `{candidate_id}`",
        f"- Source: `{model_source}`",
        f"- License status: `{license_status}`",
        f"- Promotion-card readiness: `{readiness}`",
        "",
        "## Required Evidence",
        "",
    ]
    for role in PROMOTION_CARDS_REQUIRED_INPUTS:
        record = input_artifacts[role]
        state = "present" if record.get("exists") is True else "missing"
        lines.append(f"- {role}: {state}; path `{record.get('path', '')}`")
    lines.extend(
        [
            "",
            "## Evaluation Movement",
            "",
            f"- Task-completion regressions: `{_int_value(compare_metrics.get('task_completion_regression_count'))}`",
            f"- New critical failures: `{sum(_count_map(compare_metrics.get('new_critical_failure_counts')).values())}`",
            "",
            "## Limitations",
            "",
            "- Promotion requires a separate validated promotion-decision artifact before aliases move.",
            "- This card summarizes local governance evidence and should be regenerated when any referenced artifact changes.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_dataset_card(
    *,
    dataset_id: str,
    candidate_id: str,
    passed: bool,
    input_artifacts: dict[str, dict[str, Any]],
    compare_metrics: dict[str, Any],
) -> str:
    readiness = "ready" if passed else "blocked"
    training_record = input_artifacts["training_export"]
    lines = [
        "# Dataset Card",
        "",
        f"- Dataset: `{dataset_id}`",
        f"- Candidate model: `{candidate_id}`",
        f"- Training export: `{training_record.get('path', '')}`",
        f"- Promotion-card readiness: `{readiness}`",
        "",
        "## Governance Inputs",
        "",
    ]
    for role in ("evidence_bundle", "redaction_check", "safety_gate", "compare_gate"):
        record = input_artifacts[role]
        state = "present" if record.get("exists") is True else "missing"
        lines.append(f"- {role}: {state}; path `{record.get('path', '')}`")
    lines.extend(
        [
            "",
            "## Quality Signals",
            "",
            f"- Task-completion regressions: `{_int_value(compare_metrics.get('task_completion_regression_count'))}`",
            f"- New critical failures: `{sum(_count_map(compare_metrics.get('new_critical_failure_counts')).values())}`",
            "",
            "## Use",
            "",
            "- Use this dataset card only with the matching promotion_cards.json manifest.",
            "- Regenerate the card if redaction, safety, evidence, training, or eval artifacts change.",
            "",
        ]
    )
    return "\n".join(lines)


def _artifact_record(
    role: str,
    path: Path | None,
    preserve_paths: bool,
    output_path: Path | None = None,
    *,
    reject_symlink_components: bool = False,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "role": role,
        "path": _display_path_for_output_source(path, output_path, preserve_paths) if path is not None else "",
        "exists": False,
        "kind": "missing",
    }
    if path is None or not path.exists():
        return record
    if reject_symlink_components and _path_has_symlink_component(path, include_leaf=True):
        record.update({"kind": "other", "size_bytes": 0})
        return record
    if path.is_dir():
        record.update(
            {
                "exists": True,
                "kind": "directory",
                "sha256": _directory_sha256(path),
                "file_count": sum(1 for item in path.rglob("*") if item.is_file()),
            }
        )
        return record
    if not path.is_file():
        record["kind"] = "other"
        return record
    record.update({"exists": True, "kind": "file", "size_bytes": path.stat().st_size, "sha256": _sha256(path)})
    payload = _read_json_artifact(path, reject_symlink_components=reject_symlink_components)
    if isinstance(payload, dict) and isinstance(payload.get("schema_version"), str):
        record["schema_version"] = payload["schema_version"]
    return record


def _artifact_records_match(*records: dict[str, Any]) -> bool:
    return bool(records) and all(record == records[0] for record in records[1:])


def _artifact_record_matches_snapshot(
    record: dict[str, Any], sha256: str | None, size_bytes: int | None
) -> bool:
    return (
        record.get("exists") is True
        and record.get("kind") == "file"
        and isinstance(sha256, str)
        and record.get("sha256") == sha256
        and record.get("size_bytes") == size_bytes
    )


def _reject_output_source_collision(
    output_path: Path,
    source_paths: list[Path],
    *,
    label: str,
    error_type: type[ValueError],
) -> None:
    try:
        resolved_output = output_path.resolve(strict=False)
    except OSError as exc:
        raise error_type(f"{label} output path could not be resolved safely") from exc
    for source_path in source_paths:
        try:
            resolved_source = source_path.resolve(strict=False)
        except OSError as exc:
            raise error_type(f"{label} input path could not be resolved safely") from exc
        aliases_source = resolved_output == resolved_source
        try:
            aliases_source = aliases_source or resolved_output.is_relative_to(
                resolved_source
            )
        except ValueError:
            pass
        if not aliases_source and output_path.exists() and source_path.exists():
            try:
                aliases_source = os.path.samefile(output_path, source_path)
            except OSError:
                aliases_source = False
        if aliases_source:
            raise error_type(f"{label} output must not alias an input source")


def _reject_unowned_json_output(
    output_path: Path,
    expected_schema: str,
    *,
    label: str,
    error_type: type[ValueError],
) -> None:
    if _path_has_symlink_component(output_path, include_leaf=True):
        raise error_type(f"{label} output must not traverse a symlink")
    if not output_path.exists():
        return
    payload = _read_json_artifact(output_path, reject_symlink_components=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != expected_schema:
        raise error_type(
            f"{label} output may replace only a prior {expected_schema} artifact"
        )


def _artifact_source_closure(source_paths: list[Path]) -> list[Path]:
    pending = list(source_paths)
    closure: list[Path] = []
    seen: set[str] = set()
    while pending:
        source_path = pending.pop()
        try:
            identity = os.fspath(source_path.resolve(strict=False))
        except OSError:
            identity = os.fspath(source_path.absolute())
        if identity in seen:
            continue
        seen.add(identity)
        closure.append(source_path)
        if len(closure) > 10_000:
            raise PromotionDecisionError(
                "promotion artifact source closure exceeds 10000 paths"
            )
        if (
            _path_has_symlink_component(source_path, include_leaf=True)
            or not source_path.exists()
            or not source_path.is_file()
        ):
            continue
        try:
            payload = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        pending.extend(_artifact_reference_paths(payload, source_path))
    return closure


def _artifact_reference_paths(value: Any, source_path: Path) -> list[Path]:
    references: list[Path] = []
    pending: list[Any] = [value]
    registry_links = (
        value.get("schema_version")
        in {MODEL_REGISTRY_ENTRY_SCHEMA_VERSION, MODEL_REGISTRY_SCHEMA_VERSION}
        if isinstance(value, dict)
        else False
    )
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            raw_path = current.get("path")
            if (
                isinstance(raw_path, str)
                and raw_path
                and not raw_path.startswith("<redacted:")
                and "://" not in raw_path
            ):
                reference = Path(raw_path)
                if registry_links:
                    references.append(
                        resolve_artifact_reference_path(raw_path, source_path)
                    )
                else:
                    references.append(
                        reference
                        if reference.is_absolute()
                        else source_path.parent / reference
                    )
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)
    return references


def _read_json_artifact(path: Path, *, reject_symlink_components: bool = False) -> dict[str, Any] | None:
    if reject_symlink_components and _path_has_symlink_component(path, include_leaf=True):
        return None
    if not path.exists() or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_json_artifact_snapshot(
    path: Path, *, reject_symlink_components: bool = False
) -> tuple[dict[str, Any] | None, str | None, int | None]:
    if reject_symlink_components and _path_has_symlink_component(path, include_leaf=True):
        return None, None, None
    if not path.exists() or not path.is_file():
        return None, None, None
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, None, None
    payload = value if isinstance(value, dict) else None
    return payload, hashlib.sha256(raw).hexdigest(), len(raw)


def _promotion_semantic_validation_errors(role: str, path: Path | None) -> list[str]:
    if path is None:
        return [f"{role} source is missing"]
    from .validation import (
        validate_agentic_training_result,
        validate_evidence_bundle,
        validate_eval_summary,
        validate_external_eval_result,
    )

    validators = {
        "evidence_bundle": validate_evidence_bundle,
        "eval_summary": validate_eval_summary,
        "external_eval_result": validate_external_eval_result,
        "agentic_training_result": validate_agentic_training_result,
    }
    validator = validators[role]
    target = validator(path)
    return [*target.errors, *target.warnings]


def _promotion_structured_validation_errors(
    role: str,
    path: Path | None,
    payload: dict[str, Any] | None,
) -> list[str]:
    if not isinstance(payload, dict):
        return [f"{role} source is missing or is not a JSON object"]
    if role != "compare_gate":
        if path is None:
            return [f"{role} source is missing"]
        from .validation import (
            validate_agentic_training_result,
            validate_model_registry_entry,
            validate_promotion_ledger_gate,
            validate_serving_profile,
            validate_trainer_launch_check,
        )

        validators = {
            "promotion_ledger_gate": validate_promotion_ledger_gate,
            "trainer_launch_check": validate_trainer_launch_check,
            "model_registry_entry": validate_model_registry_entry,
            "agentic_training_result": validate_agentic_training_result,
            "serving_profile": validate_serving_profile,
        }
        validation = validators[role](path)
        errors = [*validation.errors, *validation.warnings]
        if role == "trainer_launch_check":
            errors.extend(
                _promotion_trainer_launch_replay_errors(payload, path)
            )
        return errors
    errors = _gate_like_semantic_errors(payload, role)
    if role == "compare_gate":
        compare_export_path = _promotion_compare_export_path(payload, path)
        if compare_export_path is None:
            errors.append("compare_gate.compare_export must resolve to a directory")
        else:
            from .validation import validate_compare_export

            validation = validate_compare_export(compare_export_path)
            errors.extend(validation.errors)
            errors.extend(validation.warnings)
            if not validation.errors and not validation.warnings:
                errors.extend(
                    _promotion_compare_gate_replay_errors(
                        payload,
                        compare_export_path,
                        validation,
                    )
                )
    return errors


def _gate_like_semantic_errors(payload: dict[str, Any], role: str) -> list[str]:
    errors: list[str] = []
    checks = payload.get("checks")
    if not isinstance(checks, list):
        return [f"{role}.checks must be a list"]
    invalid_rows = [
        index
        for index, check in enumerate(checks)
        if not isinstance(check, dict) or not isinstance(check.get("passed"), bool)
    ]
    if invalid_rows:
        errors.append(f"{role}.checks contains invalid rows at {invalid_rows!r}")
    failed_count = sum(
        1
        for check in checks
        if isinstance(check, dict) and check.get("passed") is False
    )
    if payload.get("check_count") != len(checks):
        errors.append(f"{role}.check_count must match checks length")
    if payload.get("failed_check_count") != failed_count:
        errors.append(f"{role}.failed_check_count must match failed checks")
    if payload.get("passed") is not (failed_count == 0 and not invalid_rows):
        errors.append(f"{role}.passed must match replayed failed checks")
    return errors


def _promotion_compare_export_path(
    compare_gate: dict[str, Any] | None, compare_gate_path: Path | None
) -> Path | None:
    return next(
        (
            candidate
            for candidate in _promotion_plain_source_candidates(
                compare_gate,
                "compare_export",
                compare_gate_path,
            )
            if candidate.exists() and candidate.is_dir()
        ),
        None,
    )


def _promotion_plain_source_candidates(
    payload: dict[str, Any] | None,
    field_name: str,
    source_path: Path | None,
) -> list[Path]:
    raw_path = payload.get(field_name) if isinstance(payload, dict) else None
    if (
        not isinstance(raw_path, str)
        or not raw_path
    ):
        return []
    if raw_path.startswith("<redacted:"):
        if source_path is None or not raw_path.endswith(">"):
            return []
        basename = raw_path[len("<redacted:") : -1]
        if basename in {"", ".", ".."} or Path(basename).name != basename:
            return []
        return [source_path.parent / basename]
    if "://" in raw_path:
        return []
    path = Path(raw_path)
    if path.is_absolute() or source_path is None:
        candidates = [path]
    else:
        candidates = [source_path.parent / path]
        repository_root = artifact_repository_root(source_path)
        if repository_root is not None:
            candidates.append(repository_root / path)
    unique: list[Path] = []
    identities: set[str] = set()
    for candidate in candidates:
        identity = os.fspath(candidate.resolve(strict=False))
        if identity not in identities:
            identities.add(identity)
            unique.append(candidate)
    return unique


def _promotion_trainer_launch_replay_errors(
    launch_check: dict[str, Any], launch_check_path: Path
) -> list[str]:
    candidates = _promotion_plain_source_candidates(
        launch_check,
        "preflight_path",
        launch_check_path,
    )
    preflight_path = next(
        (
            candidate
            for candidate in candidates
            if candidate.exists()
            and candidate.is_file()
            and not _path_has_symlink_component(candidate, include_leaf=True)
        ),
        None,
    )
    if preflight_path is None:
        return [
            "trainer_launch_check.preflight_path must resolve to a non-symlink regular file"
        ]
    preflight = _read_json_artifact(
        preflight_path,
        reject_symlink_components=True,
    )
    if not isinstance(preflight, dict):
        return ["trainer launch preflight must contain a JSON object"]

    from .preflight import build_trainer_launch_check
    from .validation import validate_trainer_preflight

    validation = validate_trainer_preflight(preflight_path, payload=preflight)
    validation_summary = {
        "schema_version": "hfr.validation.v1",
        "passed": not validation.errors and not validation.warnings,
        "strict": True,
        "target_count": 1,
        "error_count": len(validation.errors),
        "warning_count": len(validation.warnings),
        "targets": [validation.as_dict()],
    }
    recorded_preflight_path = launch_check.get("preflight_path")
    try:
        replay = build_trainer_launch_check(
            preflight_path=(
                recorded_preflight_path
                if isinstance(recorded_preflight_path, str)
                else preflight_path
            ),
            preflight=preflight,
            validation_summary=validation_summary,
            require_gates=(
                launch_check.get("required_gates")
                if isinstance(launch_check.get("required_gates"), list)
                else []
            ),
            required_dataset_versions=(
                launch_check.get("required_dataset_versions")
                if isinstance(
                    launch_check.get("required_dataset_versions"), list
                )
                else []
            ),
            require_metadata=(
                launch_check.get("required_metadata")
                if isinstance(launch_check.get("required_metadata"), dict)
                else {}
            ),
            preserve_paths=True,
        )
    except (OSError, TypeError, ValueError) as exc:
        return [f"trainer launch check could not be replayed: {exc}"]

    errors = [
        f"trainer preflight validation failed: {error}"
        for error in validation.errors
    ]
    errors.extend(
        f"trainer preflight validation warning blocks promotion: {warning}"
        for warning in validation.warnings
    )
    if replay != launch_check:
        errors.append(
            "trainer launch check must exactly match canonical replay from its current preflight"
        )
    return errors


def _promotion_compare_candidate_id(
    compare_gate: dict[str, Any] | None, compare_gate_path: Path | None
) -> str:
    export_path = _promotion_compare_export_path(compare_gate, compare_gate_path)
    if export_path is None:
        return ""
    manifest = _read_json_artifact(export_path / "manifest.json")
    metadata = manifest.get("metadata") if isinstance(manifest, dict) else None
    candidate_id = metadata.get("candidate") if isinstance(metadata, dict) else None
    return candidate_id if isinstance(candidate_id, str) else ""


def _promotion_compare_gate_replay_errors(
    compare_gate: dict[str, Any],
    compare_export_path: Path,
    validation: Any,
) -> list[str]:
    manifest = _read_json_artifact(compare_export_path / "manifest.json")
    if not isinstance(manifest, dict):
        return ["compare export manifest is not a JSON object"]
    pairs_path = compare_export_path / "improvement_pairs.jsonl"
    try:
        pairs = [
            json.loads(line)
            for line in pairs_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return [f"compare export pairs could not be replayed: {exc}"]
    if any(not isinstance(row, dict) for row in pairs):
        return ["compare export pairs must contain JSON objects"]
    policy = compare_gate.get("policy")
    effective = policy.get("effective") if isinstance(policy, dict) else None
    if not isinstance(effective, dict):
        return ["compare_gate.policy.effective must be an object"]
    options = {
        key: value
        for key, value in effective.items()
        if key not in {"strict_validation"}
    }
    validation_summary = {
        "passed": not validation.errors and not validation.warnings,
        "strict": effective.get("strict_validation") is True,
        "target_count": 1,
        "error_count": len(validation.errors),
        "warning_count": len(validation.warnings),
    }
    try:
        replay = evaluate_compare_gate(
            manifest,
            pairs,
            compare_export_path=compare_gate.get("compare_export", ""),
            validation_summary=validation_summary,
            **options,
        )
    except (TypeError, ValueError) as exc:
        return [f"compare gate policy could not be replayed: {exc}"]
    replay_fields = (
        "compare_export",
        "passed",
        "check_count",
        "failed_check_count",
        "checks",
        "metrics",
        "decision",
    )
    return [
        f"compare_gate.{field_name} does not match replayed compare export"
        for field_name in replay_fields
        if compare_gate.get(field_name) != replay.get(field_name)
    ]


def _promotion_decision_semantic_source_paths(
    decision: dict[str, Any] | None, decision_path: Path
) -> list[Path]:
    sources: list[Path] = []
    for role, field_name in (
        ("compare_gate", "compare_export"),
        ("promotion_ledger_gate", "promotion_ledger"),
        ("trainer_launch_check", "preflight_path"),
    ):
        artifact_path = _promotion_decision_artifact_source_path(
            decision, role, decision_path
        )
        if artifact_path is None:
            continue
        payload = _read_json_artifact(
            artifact_path, reject_symlink_components=True
        )
        sources.extend(
            _promotion_plain_source_candidates(
                payload,
                field_name,
                artifact_path,
            )
        )
    return sources


def _promotion_decision_artifact_source_path(
    decision: dict[str, Any] | None,
    role: str,
    decision_path: Path,
) -> Path | None:
    artifacts = decision.get("artifacts") if isinstance(decision, dict) else None
    record = artifacts.get(role) if isinstance(artifacts, dict) else None
    raw_path = record.get("path") if isinstance(record, dict) else None
    if not isinstance(raw_path, str) or not raw_path or raw_path.startswith("<redacted:"):
        return None
    recorded_path = Path(raw_path)
    return (
        recorded_path
        if recorded_path.is_absolute()
        else decision_path.parent / recorded_path
    )


def _promotion_training_result_target(payload: dict[str, Any] | None) -> str:
    registry_update = payload.get("registry_update") if isinstance(payload, dict) else None
    target = registry_update.get("target_model_id") if isinstance(registry_update, dict) else None
    return target if isinstance(target, str) else ""


def _validate_promotion_decision_snapshot(
    decision: dict[str, Any] | None, source_path: Path
) -> dict[str, Any]:
    from .validation import ValidationTarget, _validate_promotion_decision

    target = ValidationTarget("promotion_decision", str(source_path))
    if decision is None:
        target.errors.append("promotion decision snapshot is not a JSON object")
    else:
        _validate_promotion_decision(decision, target, source_path)
    error_count = len(target.errors)
    warning_count = len(target.warnings)
    return {
        "schema_version": "hfr.validation.v1",
        "passed": error_count == 0 and warning_count == 0,
        "strict": True,
        "target_count": 1,
        "error_count": error_count,
        "warning_count": warning_count,
        "targets": [target.as_dict()],
    }


def _validate_model_registry_snapshot(
    registry: dict[str, Any] | None, source_path: Path
) -> dict[str, Any]:
    from .validation import ValidationTarget, _validate_model_registry

    target = ValidationTarget("model_registry", str(source_path))
    if registry is None:
        target.errors.append("model registry snapshot is not a JSON object")
    else:
        _validate_model_registry(registry, target, source_path)
    error_count = len(target.errors)
    warning_count = len(target.warnings)
    return {
        "schema_version": "hfr.validation.v1",
        "passed": error_count == 0 and warning_count == 0,
        "strict": True,
        "target_count": 1,
        "error_count": error_count,
        "warning_count": warning_count,
        "targets": [target.as_dict()],
    }


def _promotion_external_eval_lineage(
    *,
    candidate_id: str,
    evidence_bundle: dict[str, Any] | None,
    eval_summary: dict[str, Any] | None,
    eval_summary_artifact: dict[str, Any],
    result_payloads: list[dict[str, Any] | None],
    result_artifacts: list[dict[str, Any]],
    evidence_bundle_semantically_valid: bool,
    eval_summary_semantically_valid: bool,
    external_results_semantically_valid: bool,
) -> dict[str, Any]:
    summary = eval_summary if isinstance(eval_summary, dict) else {}
    summary_rows = summary.get("external_adapter_results")
    summary_results = (
        [row for row in summary_rows if isinstance(row, dict)]
        if isinstance(summary_rows, list)
        else []
    )
    results = [
        _promotion_external_eval_result_projection(payload, artifact)
        for payload, artifact in zip(result_payloads, result_artifacts, strict=True)
    ]
    results.sort(
        key=lambda row: (
            str(row.get("adapter_id") or ""),
            str(row.get("artifact", {}).get("sha256") or ""),
            str(row.get("artifact", {}).get("path") or ""),
        )
    )
    result_sha256s = [row["artifact"].get("sha256") for row in results]
    summary_sha256s = [row.get("sha256") for row in summary_results]
    summary_by_sha256 = {
        row.get("sha256"): row
        for row in summary_results
        if isinstance(row.get("sha256"), str) and row.get("sha256")
    }
    positive_unique_result_set = (
        bool(results)
        and all(isinstance(value, str) and value for value in result_sha256s)
        and len(result_sha256s) == len(set(result_sha256s))
    )
    unique_summary_set = (
        bool(summary_results)
        and all(isinstance(value, str) and value for value in summary_sha256s)
        and len(summary_sha256s) == len(set(summary_sha256s))
    )
    exact_result_set = (
        positive_unique_result_set
        and unique_summary_set
        and len(results) == len(summary_results)
        and set(result_sha256s) == set(summary_sha256s)
        and all(
            _promotion_summary_result_matches(
                summary_by_sha256.get(result["artifact"].get("sha256")), result
            )
            for result in results
        )
    )
    candidate_model_bound = bool(candidate_id) and bool(results) and all(
        result.get("model_id") == candidate_id for result in results
    )
    bundle_artifacts = (
        evidence_bundle.get("artifacts")
        if isinstance(evidence_bundle, dict)
        and isinstance(evidence_bundle.get("artifacts"), dict)
        else {}
    )
    bundled_summary = (
        bundle_artifacts.get("eval_summary")
        if isinstance(bundle_artifacts.get("eval_summary"), dict)
        else {}
    )
    evidence_bundle_summary_bound = (
        eval_summary_artifact.get("exists") is True
        and eval_summary_artifact.get("kind") == "file"
        and isinstance(eval_summary_artifact.get("sha256"), str)
        and bundled_summary.get("sha256") == eval_summary_artifact.get("sha256")
        and bundled_summary.get("size_bytes") == eval_summary_artifact.get("size_bytes")
    )
    semantic_validation_passed = (
        evidence_bundle_semantically_valid
        and eval_summary_semantically_valid
        and external_results_semantically_valid
    )
    governance_ready = (
        summary.get("passed") is True
        and summary.get("governance_ready") is True
        and summary.get("external_adapter_result_count") == len(results)
        and bool(results)
        and all(_promotion_external_eval_result_governance_ready(result) for result in results)
    )
    passed = (
        exact_result_set
        and candidate_model_bound
        and evidence_bundle_summary_bound
        and semantic_validation_passed
        and governance_ready
    )
    return {
        "result_count": len(results),
        "summary_result_count": len(summary_results),
        "exact_result_set": exact_result_set,
        "candidate_model_bound": candidate_model_bound,
        "evidence_bundle_summary_bound": evidence_bundle_summary_bound,
        "evidence_bundle_semantically_valid": evidence_bundle_semantically_valid,
        "eval_summary_semantically_valid": eval_summary_semantically_valid,
        "external_results_semantically_valid": external_results_semantically_valid,
        "semantic_validation_passed": semantic_validation_passed,
        "governance_ready": governance_ready,
        "passed": passed,
        "results": results,
    }


def _promotion_external_eval_result_projection(
    payload: dict[str, Any] | None,
    artifact: dict[str, Any],
) -> dict[str, Any]:
    result = payload if isinstance(payload, dict) else {}
    identity = result.get("identity") if isinstance(result.get("identity"), dict) else {}
    integrity = result.get("integrity") if isinstance(result.get("integrity"), dict) else {}
    execution = result.get("execution") if isinstance(result.get("execution"), dict) else {}
    outcome = (
        result.get("benchmark_outcome")
        if isinstance(result.get("benchmark_outcome"), dict)
        else {}
    )
    coverage = result.get("coverage") if isinstance(result.get("coverage"), dict) else {}
    governance = result.get("governance") if isinstance(result.get("governance"), dict) else {}
    return {
        "artifact": copy.deepcopy(artifact),
        "adapter_id": identity.get("adapter_id"),
        "model_id": identity.get("model_id"),
        "plan_sha256": identity.get("plan_sha256"),
        "heldout_manifest_sha256": identity.get("heldout_manifest_sha256"),
        "integrity_passed": integrity.get("passed") is True,
        "execution_status": execution.get("status"),
        "benchmark_status": outcome.get("status"),
        "coverage_complete": coverage.get("complete") is True,
        "governance_readiness": governance.get("readiness"),
        "external_eval_claims_allowed": governance.get("external_eval_claims_allowed") is True,
    }


def _promotion_summary_result_matches(summary_row: Any, result: dict[str, Any]) -> bool:
    if not isinstance(summary_row, dict):
        return False
    artifact = result.get("artifact") if isinstance(result.get("artifact"), dict) else {}
    expected = {
        "sha256": artifact.get("sha256"),
        "size_bytes": artifact.get("size_bytes"),
        "schema_version": EXTERNAL_EVAL_RESULT_SCHEMA_VERSION,
        "adapter_id": result.get("adapter_id"),
        "model_id": result.get("model_id"),
        "source_plan_sha256": result.get("plan_sha256"),
        "heldout_manifest_sha256": result.get("heldout_manifest_sha256"),
        "integrity_passed": result.get("integrity_passed"),
        "execution_status": result.get("execution_status"),
        "benchmark_status": result.get("benchmark_status"),
        "coverage_complete": result.get("coverage_complete"),
        "governance_readiness": result.get("governance_readiness"),
        "external_eval_claims_allowed": result.get("external_eval_claims_allowed"),
    }
    return all(summary_row.get(field_name) == value for field_name, value in expected.items())


def _promotion_external_eval_result_governance_ready(result: dict[str, Any]) -> bool:
    return (
        result.get("integrity_passed") is True
        and result.get("execution_status") == "completed"
        and result.get("benchmark_status") == "passed"
        and result.get("coverage_complete") is True
        and result.get("governance_readiness") == "ready_for_review"
        and result.get("external_eval_claims_allowed") is True
    )


def _promotion_external_eval_checks(
    lineage: dict[str, Any],
    *,
    evidence_bundle_semantically_valid: bool,
    eval_summary_semantically_valid: bool,
    evidence_bundle_error_count: int,
    eval_summary_error_count: int,
    external_result_error_count: int,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    present_result_count = sum(
        1
        for result in lineage["results"]
        if isinstance(result.get("artifact"), dict)
        and result["artifact"].get("exists") is True
        and result["artifact"].get("kind") == "file"
    )
    _add_check(
        checks,
        "evidence_bundle_semantically_valid",
        evidence_bundle_semantically_valid,
        actual={"error_count": evidence_bundle_error_count},
        expected={"error_count": 0},
        scope={"artifact_role": "evidence_bundle"},
        summary="evidence bundle passes semantic validation before promotion",
    )
    _add_check(
        checks,
        "eval_summary_semantically_valid",
        eval_summary_semantically_valid,
        actual={"error_count": eval_summary_error_count},
        expected={"error_count": 0},
        scope={"artifact_role": "eval_summary"},
        summary="eval summary passes semantic validation before promotion",
    )
    _add_check(
        checks,
        "external_eval_results_present",
        lineage["result_count"] > 0 and present_result_count == lineage["result_count"],
        actual={
            "result_count": lineage["result_count"],
            "present_result_count": present_result_count,
        },
        expected={"minimum_result_count": 1, "all_results_present": True},
        scope={"artifact_role": "external_eval_result"},
        summary="at least one execution-backed external evaluation result is present",
    )
    _add_check(
        checks,
        "external_eval_results_semantically_valid",
        lineage["external_results_semantically_valid"],
        actual={
            "result_count": lineage["result_count"],
            "error_count": external_result_error_count,
        },
        expected={"minimum_result_count": 1, "error_count": 0},
        scope={"artifact_role": "external_eval_result"},
        summary="every external evaluation result passes deterministic semantic replay",
    )
    for check_id, field_name, summary in (
        (
            "external_eval_result_set_exact",
            "exact_result_set",
            "direct external evaluation results exactly match the eval-summary result set",
        ),
        (
            "external_eval_results_match_candidate",
            "candidate_model_bound",
            "every external evaluation result identifies the promoted candidate model",
        ),
        (
            "evidence_bundle_eval_summary_bound",
            "evidence_bundle_summary_bound",
            "evidence bundle fingerprints the exact promotion eval summary",
        ),
        (
            "external_eval_results_governance_ready",
            "governance_ready",
            "the exact external evaluation result set passed and is ready for review",
        ),
        (
            "external_eval_lineage_passed",
            "passed",
            "the complete promotion external-evaluation lineage is fail-closed and ready",
        ),
    ):
        _add_check(
            checks,
            check_id,
            lineage[field_name] is True,
            actual={field_name: lineage[field_name]},
            expected={field_name: True},
            scope={"artifact_role": "external_eval_result"},
            summary=summary,
        )
    return checks


def _read_text_artifact(path: Path, *, reject_symlink_components: bool = False) -> str:
    if reject_symlink_components and _path_has_symlink_component(path, include_leaf=True):
        return ""
    if not path.exists() or not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _promotion_cards_manifest_path(path: Path) -> Path | None:
    if path.is_dir():
        return path / "promotion_cards.json"
    if path.exists() and path.is_file():
        return path
    return None


def _default_promotion_policy() -> dict[str, Any]:
    return {
        "schema_version": PROMOTION_POLICY_SCHEMA_VERSION,
        "id": "default-governance-policy",
        "description": "Default zero-tolerance promotion governance policy.",
        "source": "default",
        "required_artifacts": list(PROMOTION_DECISION_REQUIRED_ARTIFACTS),
        "release_required_artifacts": list(PROMOTION_RELEASE_RECORD_REQUIRED_ARTIFACTS),
        "allowed_candidate_classes": sorted(MODEL_CLASSES),
        "allowed_champion_classes": sorted(MODEL_CLASSES),
        "limits": dict(PROMOTION_POLICY_DEFAULT_LIMITS),
        "forbid_new_critical_rules": list(PROMOTION_POLICY_REQUIRED_FORBIDDEN_RULES),
        "forbid_regressed_rules": list(PROMOTION_POLICY_REQUIRED_FORBIDDEN_RULES),
        "requirements": {
            "require_known_license": True,
            "require_accepted_terms": True,
            "require_rollback_metadata": True,
            "require_supported_cards": True,
            "require_artifact_validation": True,
        },
        "missing_fields": [],
        "parse_errors": [],
    }


def _load_promotion_policy(path: Path | None, preserve_paths: bool, *, reject_symlink_components: bool = False) -> dict[str, Any]:
    policy = _default_promotion_policy()
    if path is None:
        return policy

    policy["source"] = "file"
    if reject_symlink_components and _path_has_symlink_component(path, include_leaf=True):
        policy["parse_errors"] = ["promotion policy path must not resolve through a symlink"]
        return policy
    payload = _read_json_artifact(path, reject_symlink_components=reject_symlink_components)
    if not isinstance(payload, dict):
        policy["parse_errors"] = ["promotion policy must be a JSON object"]
        return policy

    policy["schema_version"] = payload.get("schema_version") if isinstance(payload.get("schema_version"), str) else ""
    policy["id"] = payload.get("id") if isinstance(payload.get("id"), str) else ""
    policy["description"] = payload.get("description") if isinstance(payload.get("description"), str) else ""
    policy["missing_fields"] = [field for field in PROMOTION_POLICY_REQUIRED_FIELDS if field not in payload]
    parse_errors: list[str] = []
    policy["required_artifacts"] = _policy_string_list(payload, "required_artifacts", parse_errors)
    policy["release_required_artifacts"] = _policy_string_list(payload, "release_required_artifacts", parse_errors)
    policy["allowed_candidate_classes"] = _policy_string_list(payload, "allowed_candidate_classes", parse_errors)
    policy["allowed_champion_classes"] = _policy_string_list(payload, "allowed_champion_classes", parse_errors)
    policy["forbid_new_critical_rules"] = _policy_string_list(payload, "forbid_new_critical_rules", parse_errors)
    policy["forbid_regressed_rules"] = _policy_string_list(payload, "forbid_regressed_rules", parse_errors)
    limits = payload.get("limits")
    normalized_limits = dict(PROMOTION_POLICY_DEFAULT_LIMITS)
    if not isinstance(limits, dict):
        parse_errors.append("limits must be an object")
    else:
        for field_name, default in PROMOTION_POLICY_DEFAULT_LIMITS.items():
            raw_value = limits.get(field_name, default)
            if isinstance(raw_value, bool) or not isinstance(raw_value, int) or raw_value < 0:
                parse_errors.append(f"limits.{field_name} must be a non-negative integer")
                normalized_limits[field_name] = default
            else:
                normalized_limits[field_name] = raw_value
    policy["limits"] = normalized_limits
    requirements: dict[str, bool] = {}
    for field_name in (
        "require_known_license",
        "require_accepted_terms",
        "require_rollback_metadata",
        "require_supported_cards",
        "require_artifact_validation",
    ):
        raw_value = payload.get(field_name)
        if not isinstance(raw_value, bool):
            parse_errors.append(f"{field_name} must be a boolean")
            requirements[field_name] = False
        else:
            requirements[field_name] = raw_value
    policy["requirements"] = requirements
    policy["parse_errors"] = parse_errors
    return policy


def _policy_string_list(payload: dict[str, Any], field_name: str, parse_errors: list[str]) -> list[str]:
    value = payload.get(field_name)
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        parse_errors.append(f"{field_name} must be a list of non-empty strings")
        return []
    return sorted(dict.fromkeys(value))


def _add_promotion_policy_checks(
    checks: list[dict[str, Any]],
    policy: dict[str, Any],
    artifact: dict[str, Any] | None,
    candidate_class: str,
    champion_class: str,
) -> None:
    if artifact is not None:
        _add_check(
            checks,
            "promotion_policy_present",
            artifact["exists"] is True and artifact["kind"] == "file",
            actual={"exists": artifact["exists"], "kind": artifact["kind"], "path": artifact["path"]},
            expected={"exists": True, "kind": "file"},
            scope={"artifact_role": "promotion_policy"},
            summary="promotion policy artifact is present and fingerprinted",
        )
        _add_check(
            checks,
            "promotion_policy_schema",
            policy.get("schema_version") == PROMOTION_POLICY_SCHEMA_VERSION,
            actual=policy.get("schema_version"),
            expected={"schema_version": PROMOTION_POLICY_SCHEMA_VERSION},
            scope={"artifact_role": "promotion_policy"},
            summary="promotion policy uses the expected schema version",
        )
        _add_check(
            checks,
            "promotion_policy_fields_present",
            not policy.get("missing_fields") and not policy.get("parse_errors"),
            actual={"missing_fields": policy.get("missing_fields", []), "parse_errors": policy.get("parse_errors", [])},
            expected={"missing_fields": [], "parse_errors": []},
            scope={"artifact_role": "promotion_policy"},
            summary="promotion policy declares all required contract fields",
        )

    _add_artifact_contract_check(
        checks,
        "promotion_policy_required_artifacts_complete",
        policy.get("required_artifacts"),
        PROMOTION_DECISION_REQUIRED_ARTIFACTS,
        "promotion policy covers every single-valued promotion-decision artifact role; repeatable external results are mandatory lineage inputs",
    )
    _add_artifact_contract_check(
        checks,
        "promotion_policy_release_artifacts_complete",
        policy.get("release_required_artifacts"),
        PROMOTION_RELEASE_RECORD_REQUIRED_ARTIFACTS,
        "promotion policy covers every release-record artifact role",
    )
    _add_check(
        checks,
        "promotion_policy_candidate_class_allowed",
        candidate_class in set(policy.get("allowed_candidate_classes", [])),
        actual={"candidate_class": candidate_class, "allowed": policy.get("allowed_candidate_classes", [])},
        expected={"contains": candidate_class},
        scope={"field": "candidate_class"},
        summary="candidate class is allowed by the promotion policy",
    )
    _add_check(
        checks,
        "promotion_policy_champion_class_allowed",
        champion_class in set(policy.get("allowed_champion_classes", [])),
        actual={"champion_class": champion_class, "allowed": policy.get("allowed_champion_classes", [])},
        expected={"contains": champion_class},
        scope={"field": "champion_class"},
        summary="champion class is allowed by the promotion policy",
    )
    limits = policy.get("limits") if isinstance(policy.get("limits"), dict) else {}
    relaxed_limits = {
        field_name: limits.get(field_name)
        for field_name, default in PROMOTION_POLICY_DEFAULT_LIMITS.items()
        if not isinstance(limits.get(field_name), int) or isinstance(limits.get(field_name), bool) or limits.get(field_name) > default
    }
    _add_check(
        checks,
        "promotion_policy_limits_conservative",
        not relaxed_limits,
        actual={"limits": limits, "relaxed_limits": relaxed_limits},
        expected={"maxima": PROMOTION_POLICY_DEFAULT_LIMITS},
        scope={"artifact_role": "promotion_policy"},
        summary="promotion policy cannot relax default regression or failure limits",
    )
    missing_new_critical_rules = sorted(set(PROMOTION_POLICY_REQUIRED_FORBIDDEN_RULES) - set(policy.get("forbid_new_critical_rules", [])))
    missing_regressed_rules = sorted(set(PROMOTION_POLICY_REQUIRED_FORBIDDEN_RULES) - set(policy.get("forbid_regressed_rules", [])))
    _add_check(
        checks,
        "promotion_policy_forbidden_rules_cover_required",
        not missing_new_critical_rules and not missing_regressed_rules,
        actual={
            "missing_new_critical_rules": missing_new_critical_rules,
            "missing_regressed_rules": missing_regressed_rules,
        },
        expected={"required_rules": list(PROMOTION_POLICY_REQUIRED_FORBIDDEN_RULES)},
        scope={"artifact_role": "promotion_policy"},
        summary="promotion policy forbids required critical failure and regression rules",
    )
    requirements = policy.get("requirements") if isinstance(policy.get("requirements"), dict) else {}
    disabled_requirements = sorted(field_name for field_name, enabled in requirements.items() if enabled is not True)
    _add_check(
        checks,
        "promotion_policy_safety_requirements_enabled",
        not disabled_requirements,
        actual={"disabled_requirements": disabled_requirements},
        expected={"disabled_requirements": []},
        scope={"artifact_role": "promotion_policy"},
        summary="promotion policy keeps license, rollback, card, and validation requirements enabled",
    )


def _add_artifact_contract_check(
    checks: list[dict[str, Any]],
    check_id: str,
    actual_roles: Any,
    required_roles: tuple[str, ...],
    summary: str,
) -> None:
    actual_set = set(actual_roles) if isinstance(actual_roles, list) else set()
    required_set = set(required_roles)
    missing = sorted(required_set - actual_set)
    unknown = sorted(actual_set - required_set)
    _add_check(
        checks,
        check_id,
        not missing and not unknown,
        actual={"roles": sorted(actual_set), "missing": missing, "unknown": unknown},
        expected={"roles": list(required_roles)},
        scope={"artifact_role": "promotion_policy"},
        summary=summary,
    )


def _promotion_policy_output(policy: dict[str, Any], artifact: dict[str, Any] | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": PROMOTION_POLICY_SCHEMA_VERSION,
        "id": policy.get("id", ""),
        "description": policy.get("description", ""),
        "source": policy.get("source", "default"),
        "required_artifacts": list(policy.get("required_artifacts", [])),
        "release_required_artifacts": list(policy.get("release_required_artifacts", [])),
        "allowed_candidate_classes": list(policy.get("allowed_candidate_classes", [])),
        "allowed_champion_classes": list(policy.get("allowed_champion_classes", [])),
        "limits": dict(policy.get("limits", {})),
        "forbid_new_critical_rules": list(policy.get("forbid_new_critical_rules", [])),
        "forbid_regressed_rules": list(policy.get("forbid_regressed_rules", [])),
        "requirements": dict(policy.get("requirements", {})),
    }
    if artifact is not None:
        result["artifact"] = artifact
    return result


def _release_policy_output(
    policy: dict[str, Any] | None,
    artifact: dict[str, Any] | None,
    decision_policy: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "promotion_decision_policy": decision_policy if isinstance(decision_policy, dict) else {},
    }
    if policy is not None:
        result["release_policy"] = _promotion_policy_output(policy, artifact)
    return result


def _artifact_sha(value: Any) -> str:
    if isinstance(value, dict) and isinstance(value.get("sha256"), str):
        return value["sha256"]
    return ""


def _add_schema_check(checks: list[dict[str, Any]], role: str, payload: dict[str, Any] | None) -> None:
    expected = _JSON_ARTIFACT_ROLES[role]
    actual = payload.get("schema_version") if isinstance(payload, dict) else None
    _add_check(
        checks,
        f"{role}_schema",
        actual == expected,
        actual=actual,
        expected={"schema_version": expected},
        scope={"artifact_role": role},
        summary=f"{role} uses the expected schema version",
    )


def _add_json_schema_contract_check(
    checks: list[dict[str, Any]], role: str, payload: dict[str, Any] | None
) -> None:
    try:
        result = check_schema_contract(payload, name_or_id=role)
        errors = result.get("errors") if isinstance(result.get("errors"), list) else []
    except (OSError, UnicodeError, json.JSONDecodeError, SchemaRegistryError) as exc:
        errors = [str(exc)]
    _add_check(
        checks,
        f"{role}_contract_valid",
        not errors,
        actual={"error_count": len(errors)},
        expected={"error_count": 0, "schema_name": role},
        scope={"artifact_role": role},
        summary=f"{role} satisfies its complete published JSON Schema contract",
    )


def _add_passed_json_check(checks: list[dict[str, Any]], role: str, payload: dict[str, Any] | None) -> None:
    actual = payload.get("passed") if isinstance(payload, dict) else None
    _add_check(
        checks,
        f"{role}_passed",
        actual is True,
        actual=actual,
        expected={"passed": True},
        scope={"artifact_role": role},
        summary=f"{role} reports passed=true",
    )


def _add_license_check(checks: list[dict[str, Any]], payload: dict[str, Any] | None) -> None:
    status = _license_status(payload)
    accepted_terms = payload.get("accepted_terms") if isinstance(payload, dict) else None
    _add_check(
        checks,
        "license_status_known",
        status not in {"", "unknown", "unreviewed", "missing"},
        actual={"license_status": status or "missing", "accepted_terms": accepted_terms},
        expected={"license_status": "known"},
        scope={"artifact_role": "license_review"},
        summary="license review has a known status",
    )
    _add_check(
        checks,
        "license_terms_accepted",
        accepted_terms is True,
        actual={"accepted_terms": accepted_terms},
        expected={"accepted_terms": True},
        scope={"artifact_role": "license_review"},
        summary="license review explicitly accepts required terms",
    )


def _add_model_registry_entry_check(checks: list[dict[str, Any]], payload: dict[str, Any] | None, candidate_id: str) -> None:
    entry_candidate_id = payload.get("candidate_id") if isinstance(payload, dict) else None
    _add_check(
        checks,
        "model_registry_entry_matches_candidate",
        bool(candidate_id) and entry_candidate_id == candidate_id,
        actual={"candidate_id": candidate_id, "entry_candidate_id": entry_candidate_id},
        expected={"same_candidate_id": True},
        scope={"artifact_role": "model_registry_entry"},
        summary="model registry entry is bound to the promoted candidate",
    )


def _add_serving_profile_check(checks: list[dict[str, Any]], payload: dict[str, Any] | None) -> None:
    eval_preflight = payload.get("eval_preflight") if isinstance(payload, dict) and isinstance(payload.get("eval_preflight"), dict) else {}
    ready = eval_preflight.get("ready")
    readiness = eval_preflight.get("readiness")
    failed_checks = eval_preflight.get("failed_checks")
    _add_check(
        checks,
        "serving_profile_ready",
        ready is True and readiness == "ready",
        actual={"ready": ready, "readiness": readiness, "failed_checks": failed_checks if isinstance(failed_checks, list) else []},
        expected={"ready": True, "readiness": "ready"},
        scope={"artifact_role": "serving_profile"},
        summary="serving profile is ready for evaluation and promotion",
    )


def _add_rollback_metadata_check(checks: list[dict[str, Any]], payload: dict[str, Any] | None, rollback_id: str | None) -> None:
    actual_id = ""
    available = None
    receipt_passed = None
    if isinstance(payload, dict):
        rollback = payload.get("rollback") if isinstance(payload.get("rollback"), dict) else {}
        actual_id = str(
            payload.get("rollback_id")
            or payload.get("target_model_id")
            or rollback.get("id")
            or rollback.get("target_model_id")
            or ""
        )
        available = payload.get("available")
        if available is None:
            available = rollback.get("available")
        if payload.get("schema_version") == PROMOTION_ROLLBACK_RECEIPT_SCHEMA_VERSION:
            receipt_passed = payload.get("passed")
    _add_check(
        checks,
        "rollback_metadata_matches_target",
        bool(rollback_id) and actual_id == rollback_id and available is True,
        actual={"rollback_id": actual_id, "available": available},
        expected={"rollback_id": rollback_id or "", "available": True},
        scope={"artifact_role": "rollback_metadata"},
        summary="rollback metadata points at the declared rollback target",
    )
    if isinstance(payload, dict) and payload.get("schema_version") == PROMOTION_ROLLBACK_RECEIPT_SCHEMA_VERSION:
        _add_check(
            checks,
            "rollback_receipt_passed",
            receipt_passed is True,
            actual=receipt_passed,
            expected={"passed": True},
            scope={"artifact_role": "rollback_metadata"},
            summary="rollback metadata is a passing rollback receipt",
        )


def _add_compare_metrics_check(checks: list[dict[str, Any]], compare_metrics: dict[str, Any]) -> None:
    missing = [field for field in PROMOTION_COMPARE_METRIC_FIELDS if field not in compare_metrics]
    count_fields = (
        "baseline_win_count",
        "contract_drift_count",
        "task_completion_regression_count",
        "unverified_contract_count",
    )
    map_fields = ("new_critical_failure_counts", "regressed_rule_counts")
    invalid_counts = [
        field for field in count_fields if field in compare_metrics and not _is_non_negative_int(compare_metrics.get(field))
    ]
    invalid_maps = [field for field in map_fields if field in compare_metrics and not _is_count_map(compare_metrics.get(field))]
    _add_check(
        checks,
        "compare_metrics_complete",
        not missing and not invalid_counts and not invalid_maps,
        actual={"missing": missing, "invalid_counts": invalid_counts, "invalid_maps": invalid_maps},
        expected={"fields": list(PROMOTION_COMPARE_METRIC_FIELDS)},
        scope={"artifact_role": "compare_gate"},
        summary="compare gate exposes complete promotion movement metrics",
    )


def _add_card_claims_check(
    checks: list[dict[str, Any]],
    role: str,
    path: Path | None,
    *,
    reject_symlink_components: bool = False,
) -> None:
    markers: list[str] = []
    readable = False
    non_empty = False
    expected_heading = "# model card" if role == "model_card" else "# dataset card"
    heading_present = False
    if path is not None and reject_symlink_components and _path_has_symlink_component(path, include_leaf=True):
        path = None
    if path is not None and path.exists() and path.is_file():
        try:
            text = path.read_text(encoding="utf-8").lower()
            readable = True
        except (OSError, UnicodeDecodeError):
            text = ""
        non_empty = bool(text.strip())
        heading_present = expected_heading in text
        markers = [marker for marker in _UNSUPPORTED_CARD_MARKERS if marker in text]
    _add_check(
        checks,
        f"{role}_claims_supported",
        readable and non_empty and heading_present and not markers,
        actual={
            "readable": readable,
            "non_empty": non_empty,
            "expected_heading_present": heading_present,
            "unsupported_markers": markers,
        },
        expected={
            "readable": True,
            "non_empty": True,
            "expected_heading_present": True,
            "unsupported_markers": [],
        },
        scope={"artifact_role": role},
        summary=f"{role} is a readable non-empty card without unsupported claim markers",
    )


def _add_max_count_check(checks: list[dict[str, Any]], check_id: str, value: Any, maximum: int) -> None:
    actual = _int_value(value)
    _add_check(
        checks,
        check_id,
        actual <= maximum,
        actual=actual,
        expected={"max": maximum},
        summary=f"{check_id}: actual={actual}, max={maximum}",
    )


def _add_count_map_absence_check(checks: list[dict[str, Any]], check_id: str, value: Any, label: str) -> None:
    _add_count_map_max_check(checks, check_id, value, label, 0)


def _add_count_map_max_check(checks: list[dict[str, Any]], check_id: str, value: Any, label: str, maximum: int) -> None:
    counts = _count_map(value)
    total = sum(counts.values())
    _add_check(
        checks,
        check_id,
        total <= maximum,
        actual={"total": total, "counts": counts},
        expected={"max": maximum},
        summary=f"{label}: actual={total}, max={maximum}",
    )


def _add_forbidden_rule_check(checks: list[dict[str, Any]], value: Any, source_id: str, rule_id: str) -> None:
    counts = _count_map(value)
    count = counts.get(rule_id, 0)
    _add_check(
        checks,
        f"{source_id}_{rule_id}_absent",
        count == 0,
        actual={"rule_id": rule_id, "count": count},
        expected={"count": 0},
        scope={"rule_id": rule_id},
        summary=f"{rule_id} does not appear in critical failures or regressions",
    )


def _add_check(
    checks: list[dict[str, Any]],
    check_id: str,
    passed: bool,
    *,
    actual: Any,
    expected: dict[str, Any],
    summary: str,
    scope: dict[str, Any] | None = None,
) -> None:
    check = {"id": check_id, "passed": bool(passed), "actual": actual, "expected": expected, "summary": summary}
    if scope is not None:
        check["scope"] = scope
    checks.append(check)


def _decision_metrics(
    checks: list[dict[str, Any]],
    compare_metrics: dict[str, Any],
    policy: dict[str, Any],
    external_eval_lineage: dict[str, Any],
) -> dict[str, Any]:
    return {
        "check_count": len(checks),
        "failed_check_count": sum(1 for check in checks if not check["passed"]),
        "required_artifact_count": len(PROMOTION_DECISION_REQUIRED_ARTIFACTS),
        "policy_required_artifact_count": len(policy.get("required_artifacts", [])),
        "policy_release_required_artifact_count": len(policy.get("release_required_artifacts", [])),
        "external_eval_result_count": _int_value(external_eval_lineage.get("result_count")),
        "task_completion_regression_count": _int_value(compare_metrics.get("task_completion_regression_count")),
        "baseline_win_count": _int_value(compare_metrics.get("baseline_win_count")),
        "contract_drift_count": _int_value(compare_metrics.get("contract_drift_count")),
        "unverified_contract_count": _int_value(compare_metrics.get("unverified_contract_count")),
        "new_critical_failure_count": sum(_count_map(compare_metrics.get("new_critical_failure_counts")).values()),
        "rule_regression_count": sum(_count_map(compare_metrics.get("regressed_rule_counts")).values()),
    }


def _decision_summary(passed: bool, failed_checks: int, metrics: dict[str, Any]) -> str:
    if passed:
        return "Promotion passed; registry alias update receipt may be applied by a separate guarded step."
    return (
        "Promotion blocked: "
        f"{failed_checks} governance check(s) failed; "
        f"task_completion_regressions={metrics['task_completion_regression_count']}; "
        f"new_critical_failures={metrics['new_critical_failure_count']}."
    )


def _alias_update(passed: bool, candidate_id: str, champion_id: str, rollback_id: str) -> dict[str, Any]:
    return {
        "authorized": passed,
        "recommendation": "apply_alias_update" if passed else "hold_aliases",
        "aliases": [
            {"alias": "candidate", "target": candidate_id},
            {"alias": "champion", "previous_target": champion_id, "target": candidate_id},
            {"alias": "rollback", "target": rollback_id},
        ],
    }


def _metrics_object(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    metrics = payload.get("metrics")
    return metrics if isinstance(metrics, dict) else {}


def _validation_summary(value: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"passed": None, "target_count": 0, "error_count": 0, "warning_count": 0, "targets": []}
    return {
        "passed": value.get("passed") if isinstance(value.get("passed"), bool) else None,
        "target_count": _int_value(value.get("target_count")),
        "error_count": _int_value(value.get("error_count")),
        "warning_count": _int_value(value.get("warning_count")),
        "targets": _validation_target_summaries(value.get("targets")),
    }


def _validation_target_summaries(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    targets: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        error_count = _int_value(item.get("error_count"))
        if "error_count" not in item and isinstance(item.get("errors"), list):
            error_count = len(item["errors"])
        warning_count = _int_value(item.get("warning_count"))
        if "warning_count" not in item and isinstance(item.get("warnings"), list):
            warning_count = len(item["warnings"])
        targets.append(
            {
                "type": item.get("type") if isinstance(item.get("type"), str) else "",
                "passed": item.get("passed") if isinstance(item.get("passed"), bool) else None,
                "error_count": error_count,
                "warning_count": warning_count,
            }
        )
    return targets


def _registry_aliases(registry: dict[str, Any]) -> dict[str, str]:
    aliases = registry.get("aliases")
    if not isinstance(aliases, dict):
        return {}
    return {str(alias): str(target) for alias, target in aliases.items() if isinstance(alias, str) and isinstance(target, str)}


def _registry_model_ids(registry: dict[str, Any]) -> set[str]:
    entries = registry.get("entries")
    if isinstance(entries, dict):
        return {str(entry_id) for entry_id in entries if isinstance(entry_id, str) and entry_id}
    models = registry.get("models")
    if isinstance(models, dict):
        return {str(model_id) for model_id in models if isinstance(model_id, str) and model_id}
    if isinstance(models, list):
        ids: set[str] = set()
        for model in models:
            if isinstance(model, dict) and isinstance(model.get("id"), str) and model["id"]:
                ids.add(model["id"])
        return ids
    return set()


def _alias_update_targets(alias_update: dict[str, Any]) -> dict[str, str]:
    aliases = alias_update.get("aliases")
    if not isinstance(aliases, list):
        return {}
    rows: dict[str, str] = {}
    for item in aliases:
        if not isinstance(item, dict):
            continue
        alias = item.get("alias")
        target = item.get("target")
        if isinstance(alias, str) and isinstance(target, str):
            rows[alias] = target
    return rows


def _model_id(value: Any) -> str:
    if isinstance(value, dict) and isinstance(value.get("id"), str):
        return value["id"]
    return ""


def _license_status(payload: dict[str, Any] | None) -> str:
    if not isinstance(payload, dict):
        return ""
    value = payload.get("license_status")
    if isinstance(value, str):
        return value.lower()
    license_info = payload.get("license")
    if isinstance(license_info, dict) and isinstance(license_info.get("status"), str):
        return license_info["status"].lower()
    status = payload.get("status")
    return status.lower() if isinstance(status, str) else ""


def _known_status(value: str) -> bool:
    return value.lower() not in {"", "unknown", "unreviewed", "missing"}


def _count_map(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {str(key): _int_value(count) for key, count in value.items() if isinstance(key, str) and key}


def _is_count_map(value: Any) -> bool:
    return isinstance(value, dict) and all(isinstance(key, str) and key and _is_non_negative_int(count) for key, count in value.items())


def _is_non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(item).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _display_path(path: Path | None, preserve_paths: bool = False) -> str:
    if path is None:
        return ""
    raw = str(path)
    if preserve_paths:
        return raw
    if _is_windows_absolute(raw):
        return f"<redacted:{_basename(raw)}>"
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    try:
        return str(resolved.relative_to(cwd))
    except ValueError:
        return f"<redacted:{resolved.name}>"


def _display_path_for_output_source(path: Path, output_path: Path | None, preserve_paths: bool = False) -> str:
    if preserve_paths or output_path is None:
        return _display_path(path, preserve_paths)
    raw = str(path)
    if _is_windows_absolute(raw):
        return f"<redacted:{_basename(raw)}>"
    resolved = path.resolve()
    output_dir = output_path.parent.resolve()
    return os.path.relpath(resolved, output_dir)


def _is_windows_absolute(value: str) -> bool:
    normalized = value.replace("/", "\\")
    return (len(normalized) >= 3 and normalized[1:3] == ":\\" and normalized[0].isalpha()) or normalized.startswith("\\\\")


def _basename(value: str) -> str:
    return value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] or "path"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
