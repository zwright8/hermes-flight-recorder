"""Top-level governance decisions for promotion and alias movement."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .agentic_training_result import AGENTIC_TRAINING_RESULT_SCHEMA_VERSION
from .bundle import EVIDENCE_BUNDLE_SCHEMA_VERSION
from .compare_gate import COMPARE_GATE_SCHEMA_VERSION
from .model_registry import MODEL_REGISTRY_ENTRY_SCHEMA_VERSION
from .preflight import TRAINER_LAUNCH_CHECK_SCHEMA_VERSION
from .promotion_gate import PROMOTION_LEDGER_GATE_SCHEMA_VERSION

PROMOTION_DECISION_SCHEMA_VERSION = "hfr.promotion_decision.v1"
PROMOTION_CARDS_SCHEMA_VERSION = "hfr.promotion_cards.v1"
PROMOTION_ALIAS_APPLY_SCHEMA_VERSION = "hfr.promotion_alias_apply.v1"
PROMOTION_ROLLBACK_RECEIPT_SCHEMA_VERSION = "hfr.promotion_rollback_receipt.v1"
PROMOTION_RELEASE_RECORD_SCHEMA_VERSION = "hfr.promotion_release_record.v1"
PROMOTION_POLICY_SCHEMA_VERSION = "hfr.promotion_policy.v1"
MODEL_REGISTRY_SCHEMA_VERSION = "hfr.model_registry.v1"
SERVING_PROFILE_SCHEMA_VERSION = "hfr.serving_profile.v1"

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
    "promotion_ledger_gate_schema",
    "compare_gate_schema",
    "trainer_launch_check_schema",
    "model_registry_entry_schema",
    "agentic_training_result_schema",
    "serving_profile_schema",
    *(
        f"{role}_passed"
        for role in (
            "evidence_bundle",
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
    "promotion_ledger_gate": PROMOTION_LEDGER_GATE_SCHEMA_VERSION,
    "compare_gate": COMPARE_GATE_SCHEMA_VERSION,
    "trainer_launch_check": TRAINER_LAUNCH_CHECK_SCHEMA_VERSION,
    "model_registry_entry": MODEL_REGISTRY_ENTRY_SCHEMA_VERSION,
    "agentic_training_result": AGENTIC_TRAINING_RESULT_SCHEMA_VERSION,
    "serving_profile": SERVING_PROFILE_SCHEMA_VERSION,
}
_PASSED_JSON_ROLES = (
    "evidence_bundle",
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


def build_promotion_decision(
    *,
    candidate_id: str,
    champion_id: str,
    rollback_id: str | None = None,
    candidate_class: str = "candidate",
    champion_class: str = "champion",
    out_path: str | Path | None = None,
    evidence_bundle_path: str | Path | None = None,
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
    artifacts = {
        role: _artifact_record(role, Path(raw_path) if raw_path else None, preserve_paths, output_path)
        for role, raw_path in paths.items()
    }
    json_artifacts = {
        role: _read_json_artifact(Path(raw_path)) if raw_path else None
        for role, raw_path in paths.items()
        if role in _JSON_ARTIFACT_ROLES or role in _PASSED_JSON_ROLES or role == "rollback_metadata"
    }
    policy_path = Path(promotion_policy_path) if promotion_policy_path else None
    policy_artifact = _artifact_record("promotion_policy", policy_path, preserve_paths, output_path) if policy_path else None
    policy = _load_promotion_policy(policy_path, preserve_paths)

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
            record["exists"] is True,
            actual={"exists": record["exists"], "path": record["path"]},
            expected={"exists": True},
            scope={"artifact_role": role},
            summary=f"{role} artifact is present and fingerprinted",
        )

    for role in _JSON_ARTIFACT_ROLES:
        _add_schema_check(checks, role, json_artifacts.get(role))
    for role in _PASSED_JSON_ROLES:
        _add_passed_json_check(checks, role, json_artifacts.get(role))

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
    _add_card_claims_check(checks, "model_card", Path(model_card_path) if model_card_path else None)
    _add_card_claims_check(checks, "dataset_card", Path(dataset_card_path) if dataset_card_path else None)

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
    metrics = _decision_metrics(checks, compare_metrics, policy)
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
        "policy": _promotion_policy_output(policy, policy_artifact),
        "metrics": metrics,
        "alias_update": _alias_update(passed, candidate_id, champion_id, rollback_id or ""),
        "notes": [
            "Promotion decisions are side-effect free; they do not move registry aliases.",
            "Alias movement is authorized only when every required governance artifact is present, fingerprinted, and passed.",
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
    registry_before_record = _artifact_record("registry", registry_file, preserve_paths, receipt_path)
    decision_record = _artifact_record("promotion_decision", decision_file, preserve_paths, receipt_path)
    registry = _read_json_artifact(registry_file)
    decision = _read_json_artifact(decision_file)
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
    validation_passed = promotion_decision_validation.get("passed") if isinstance(promotion_decision_validation, dict) else None
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
        _write_json(registry_file, updated)
        registry_after_record = _artifact_record("registry", registry_file, preserve_paths, receipt_path)
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
            "candidate_id": candidate_id,
            "champion_previous_target": champion_id,
            "rollback_id": rollback_id,
        },
        "promotion_decision_validation": _validation_summary(promotion_decision_validation),
        "registry_before": {
            "path": registry_before_record.get("path", ""),
            "sha256": registry_before_record.get("sha256"),
            "aliases": aliases_before,
        },
        "registry_after": {
            "path": registry_after_record.get("path", ""),
            "sha256": registry_after_record.get("sha256"),
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
    _write_json(receipt_path, receipt)
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
    registry_record = _artifact_record("registry", registry_file, preserve_paths, receipt_path)
    registry = _read_json_artifact(registry_file)
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
    artifacts = {role: _artifact_record(role, path, preserve_paths, record_path) for role, path in paths.items()}
    decision = _read_json_artifact(paths["promotion_decision"])
    cards_manifest_path = _promotion_cards_manifest_path(paths["promotion_cards"])
    cards = _read_json_artifact(cards_manifest_path) if cards_manifest_path is not None else None
    alias_receipt = _read_json_artifact(paths["promotion_alias_apply"])
    rollback = _read_json_artifact(paths["rollback_metadata"])
    compare_gate = _read_json_artifact(paths["compare_gate"])
    notes_text = _read_text_artifact(paths["release_notes"])

    decision_obj = decision if isinstance(decision, dict) else {}
    cards_obj = cards if isinstance(cards, dict) else {}
    alias_obj = alias_receipt if isinstance(alias_receipt, dict) else {}
    policy_path = Path(promotion_policy_path) if promotion_policy_path else None
    policy_artifact = _artifact_record("promotion_policy", policy_path, preserve_paths, record_path) if policy_path else None
    policy = _load_promotion_policy(policy_path, preserve_paths) if policy_path else None
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
        "release_notes_present",
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
        role: _artifact_record(role, Path(raw_path) if raw_path else None, preserve_paths, manifest_path)
        for role, raw_path in paths.items()
    }
    json_artifacts = {
        role: _read_json_artifact(Path(raw_path)) if raw_path else None
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
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "role": role,
        "path": _display_path_for_output_source(path, output_path, preserve_paths) if path is not None else "",
        "exists": False,
        "kind": "missing",
    }
    if path is None or not path.exists():
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
    payload = _read_json_artifact(path)
    if isinstance(payload, dict) and isinstance(payload.get("schema_version"), str):
        record["schema_version"] = payload["schema_version"]
    return record


def _read_json_artifact(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_text_artifact(path: Path) -> str:
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


def _load_promotion_policy(path: Path | None, preserve_paths: bool) -> dict[str, Any]:
    policy = _default_promotion_policy()
    if path is None:
        return policy

    policy["source"] = "file"
    payload = _read_json_artifact(path)
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
        "promotion policy covers every promotion-decision artifact role",
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
        bool(rollback_id) and actual_id == rollback_id and available is not False,
        actual={"rollback_id": actual_id, "available": available},
        expected={"rollback_id": rollback_id or "", "available": "not false"},
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


def _add_card_claims_check(checks: list[dict[str, Any]], role: str, path: Path | None) -> None:
    markers: list[str] = []
    if path is not None and path.exists() and path.is_file():
        try:
            text = path.read_text(encoding="utf-8").lower()
        except OSError:
            text = ""
        markers = [marker for marker in _UNSUPPORTED_CARD_MARKERS if marker in text]
    _add_check(
        checks,
        f"{role}_claims_supported",
        not markers,
        actual={"unsupported_markers": markers},
        expected={"unsupported_markers": []},
        scope={"artifact_role": role},
        summary=f"{role} contains no TODO/TBD/unsupported-claim markers",
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


def _decision_metrics(checks: list[dict[str, Any]], compare_metrics: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "check_count": len(checks),
        "failed_check_count": sum(1 for check in checks if not check["passed"]),
        "required_artifact_count": len(PROMOTION_DECISION_REQUIRED_ARTIFACTS),
        "policy_required_artifact_count": len(policy.get("required_artifacts", [])),
        "policy_release_required_artifact_count": len(policy.get("release_required_artifacts", [])),
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
