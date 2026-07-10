"""Fail-closed inspection of JSON artifacts used as readiness evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .path_safety import path_has_symlink_component
from .schema_registry import SchemaRegistryError, check_schema_contract

_SCHEMA_NAME_OVERRIDES = {
    "harness_manifest": "harness_run_manifest",
    "harness_result": "harness_run_result",
}

_DIRECTORY_MANIFESTS = {
    "promotion_archive": "promotion_archive.json",
    "promotion_cards": "promotion_cards.json",
}

_SEMANTIC_VALIDATOR_NAMES = {
    "action_ledger": "validate_action_ledger",
    "action_ledger_gate": "validate_action_ledger_gate",
    "agentic_loop_governance_receipt": "validate_agentic_loop_governance_receipt",
    "agentic_loop_ledger": "validate_agentic_loop_ledger",
    "agentic_rollout_plan": "validate_agentic_rollout_plan",
    "agentic_rollout_receipt": "validate_agentic_rollout_receipt",
    "agentic_training_flow": "validate_agentic_training_flow",
    "agentic_training_loop_plan": "validate_agentic_training_loop_plan",
    "agentic_training_plan": "validate_agentic_training_plan",
    "agentic_training_result": "validate_agentic_training_result",
    "agentic_training_runtime_preflight": "validate_agentic_training_runtime_preflight",
    "cloud_training_artifact_manifest": "validate_cloud_training_artifact_manifest",
    "cloud_training_launch_plan": "validate_cloud_training_launch_plan",
    "cloud_training_launch_receipt": "validate_cloud_training_launch_receipt",
    "cloud_training_preflight": "validate_cloud_training_preflight",
    "cloud_training_provider_registry": "validate_cloud_training_provider_registry",
    "cloud_training_status_receipt": "validate_cloud_training_status_receipt",
    "dataset_curation_receipt": "validate_dataset_curation_receipt",
    "decision_gate": "validate_decision_gate",
    "eval_summary": "validate_eval_summary",
    "evidence_bundle": "validate_evidence_bundle",
    "evidence_coverage": "validate_evidence_coverage",
    "external_eval_plan": "validate_external_eval_plan",
    "external_eval_receipt": "validate_external_eval_receipt",
    "harness_manifest": "validate_harness_run_manifest",
    "harness_result": "validate_harness_run_result",
    "heldout_manifest": "validate_heldout_manifest",
    "improvement_ledger": "validate_improvement_ledger",
    "improvement_ledger_gate": "validate_improvement_ledger_gate",
    "improvement_plan": "validate_improvement_plan",
    "live_smoke_summary": "validate_live_smoke_summary",
    "model_adapter_manifest": "validate_model_adapter_manifest",
    "model_candidate": "validate_model_candidate",
    "model_compatibility_report": "validate_model_compatibility_report",
    "model_grader_disagreement_queue": "validate_model_grader_disagreement_queue",
    "model_grader_dry_run": "validate_model_grader_dry_run",
    "model_grader_gate": "validate_model_grader_gate",
    "model_grader_override_receipt": "validate_model_grader_override_receipt",
    "model_registry": "validate_model_registry",
    "model_registry_entry": "validate_model_registry_entry",
    "model_scout_manifest": "validate_model_scout_manifest",
    "model_serving_probe_receipt": "validate_model_serving_probe_receipt",
    "next_iteration_schedule": "validate_next_iteration_schedule",
    "promotion_alias_apply": "validate_promotion_alias_apply",
    "promotion_decision": "validate_promotion_decision",
    "promotion_ledger": "validate_promotion_ledger",
    "promotion_ledger_gate": "validate_promotion_ledger_gate",
    "promotion_release_record": "validate_promotion_release_record",
    "promotion_rollback_receipt": "validate_promotion_rollback_receipt",
    "rejection_sampling_gate": "validate_rejection_sampling_gate",
    "repair_queue": "validate_repair_queue",
    "review_calibration": "validate_review_calibration",
    "rubric_spec": "validate_rubric_spec",
    "run_digest": "validate_run_digest",
    "scenario_check": "validate_scenario_check",
    "scenario_quality": "validate_scenario_quality",
    "serving_compatibility_report": "validate_serving_compatibility_report",
    "serving_demo_run": "validate_serving_demo_run",
    "serving_endpoint_check": "validate_serving_endpoint_check",
    "serving_lifecycle": "validate_serving_lifecycle",
    "serving_profile": "validate_serving_profile",
    "state_diff": "validate_state_diff",
    "state_snapshot": "validate_state_snapshot",
    "trace_observability": "validate_trace_observability",
    "trainer_archive": "validate_trainer_archive",
    "trainer_archive_check": "validate_trainer_archive_check",
    "trainer_consumer_plan": "validate_trainer_consumer_plan",
    "trainer_launch_check": "validate_trainer_launch_check",
    "trainer_preflight": "validate_trainer_preflight",
    "trainer_wrapper_dry_run": "validate_trainer_wrapper_dry_run",
}

_GATE_CONTRACT_ROLES = {"compare_gate", "reviewed_gate"}

_REQUIRED_VALUES: dict[str, dict[str, Any]] = {
    "action_ledger": {"passed": True},
    "agentic_loop_ledger": {"passed": True},
    "agentic_rollout_plan": {"passed": True, "readiness": "ready_for_harness_batch"},
    "agentic_rollout_receipt": {"passed": True, "readiness": "mock_rollouts_recorded"},
    "agentic_training_flow": {"passed": True},
    "agentic_training_plan": {"passed": True, "readiness": "ready"},
    "agentic_training_result": {"passed": True},
    "agentic_training_runtime_preflight": {"passed": True},
    "cloud_training_artifact_manifest": {"passed": True, "readiness": "ready"},
    "cloud_training_launch_plan": {"passed": True, "readiness": "ready_for_dry_run_launch"},
    "cloud_training_launch_receipt": {"passed": True, "readiness": "dry_run_recorded"},
    "cloud_training_preflight": {"passed": True, "readiness": "ready_for_dry_run_launch_plan"},
    "cloud_training_status_receipt": {"passed": True, "readiness": "status_recorded"},
    "compare_gate": {"passed": True},
    "dataset_curation_receipt": {"passed": True, "readiness": "ready_for_external_trainer_handoff"},
    "eval_summary": {"passed": True, "governance_ready": True},
    "evidence_bundle": {"passed": True, "readiness": "ready"},
    "external_eval_plan": {"ready": True},
    "external_eval_receipt": {"passed": True, "readiness": "dry_run_recorded"},
    "heldout_manifest": {"ready": True},
    "improvement_ledger": {"passed": True},
    "improvement_plan": {"passed": True},
    "model_grader_gate": {"passed": True, "readiness": "labels_calibrated_for_curated_handoff"},
    "promotion_archive": {"passed": True},
    "promotion_cards": {"passed": True, "readiness": "ready"},
    "promotion_decision": {"passed": True, "readiness": "ready"},
    "promotion_ledger": {"passed": True},
    "rejection_sampling_gate": {"passed": True, "readiness": "ready_for_dataset_curation"},
    "review_calibration": {"passed": True},
    "reviewed_gate": {"passed": True},
    "serving_lifecycle": {"passed": True, "ready": True, "readiness": "ready"},
    "trainer_launch_check": {"passed": True, "readiness": "ready"},
    "trainer_preflight": {"passed": True, "readiness": "ready"},
}


def inspect_artifact_source(
    path_value: str | Path,
    role: str,
    *,
    require_semantics: bool = True,
) -> dict[str, Any]:
    """Return a side-effect-free readiness assessment for one source artifact.

    ``ready`` means the source is a regular non-symlink artifact, conforms to
    the role's bundled schema, satisfies its success signal, and passes the
    role's full semantic validator when one exists. Physical existence alone
    is never treated as evidence readiness.
    """
    path = Path(path_value)
    if role == "training_export":
        return _inspect_training_export(path, require_semantics=require_semantics)
    if role in _DIRECTORY_MANIFESTS:
        return _inspect_directory(path, role, require_semantics=require_semantics)
    schema_name = _SCHEMA_NAME_OVERRIDES.get(role, role)
    return inspect_json_source(path, schema_name, role=role, require_semantics=require_semantics)


def inspect_json_source(
    path_value: str | Path,
    schema_name: str,
    *,
    role: str | None = None,
    require_semantics: bool = True,
) -> dict[str, Any]:
    """Inspect one JSON source without following symlinked path components."""
    path = Path(path_value)
    physical_exists = path.exists()
    has_symlink = path_has_symlink_component(path, include_leaf=True)
    regular_file = physical_exists and not has_symlink and path.is_file()
    payload: dict[str, Any] = {}
    parse_valid = False
    if regular_file:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raw = None
        if isinstance(raw, dict):
            payload = raw
            parse_valid = True

    schema_valid = False
    if parse_valid:
        try:
            schema_valid = bool(check_schema_contract(payload, name_or_id=schema_name).get("passed"))
        except (SchemaRegistryError, TypeError, ValueError):
            schema_valid = False

    semantic_valid = not require_semantics or (
        schema_valid
        and _semantic_ready(role or schema_name, payload)
        and _semantic_contract_valid(path, role or schema_name)
    )
    return {
        "path": path,
        "physical_exists": physical_exists,
        "regular_file": regular_file,
        "parse_valid": parse_valid,
        "schema_valid": schema_valid,
        "semantic_valid": semantic_valid,
        "ready": regular_file and parse_valid and schema_valid and semantic_valid,
        "payload": payload,
    }


def _inspect_training_export(path: Path, *, require_semantics: bool) -> dict[str, Any]:
    physical_exists = path.exists()
    has_symlink = path_has_symlink_component(path, include_leaf=True)
    regular_directory = (
        physical_exists
        and not has_symlink
        and path.is_dir()
        and not _directory_contains_symlink(path)
    )
    manifest_path = path / "manifest.json"
    manifest = (
        inspect_json_source(
            manifest_path,
            "training_manifest",
            role="training_export",
            require_semantics=require_semantics,
        )
        if regular_directory
        else _empty_inspection(manifest_path)
    )
    contract_valid = not require_semantics or _training_export_contract_valid(path)
    semantic_valid = manifest["semantic_valid"] and contract_valid
    return {
        "path": path,
        "physical_exists": physical_exists,
        "regular_directory": regular_directory,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "payload": manifest["payload"],
        "schema_valid": manifest["schema_valid"],
        "semantic_valid": semantic_valid,
        "ready": regular_directory and manifest["regular_file"] and manifest["schema_valid"] and semantic_valid,
    }


def _inspect_directory(path: Path, role: str, *, require_semantics: bool) -> dict[str, Any]:
    physical_exists = path.exists()
    has_symlink = path_has_symlink_component(path, include_leaf=True)
    regular_directory = (
        physical_exists
        and not has_symlink
        and path.is_dir()
        and not _directory_contains_symlink(path)
    )
    manifest_path = path / _DIRECTORY_MANIFESTS[role]
    manifest = (
        inspect_json_source(
            manifest_path,
            role,
            role=role,
            require_semantics=require_semantics,
        )
        if regular_directory
        else _empty_inspection(manifest_path)
    )
    contract_valid = manifest["schema_valid"] and _directory_contract_valid(path, role)
    semantic_valid = manifest["semantic_valid"] and contract_valid
    return {
        "path": path,
        "physical_exists": physical_exists,
        "regular_directory": regular_directory,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "payload": manifest["payload"],
        "schema_valid": manifest["schema_valid"],
        "semantic_valid": semantic_valid,
        "ready": regular_directory and manifest["regular_file"] and manifest["schema_valid"] and semantic_valid,
    }


def _directory_contract_valid(path: Path, role: str) -> bool:
    try:
        from .validation import validate_promotion_archive, validate_promotion_cards

        validator = validate_promotion_archive if role == "promotion_archive" else validate_promotion_cards
        return not validator(path).errors
    except (OSError, UnicodeError, json.JSONDecodeError, SchemaRegistryError, TypeError, ValueError):
        return False


def _training_export_contract_valid(path: Path) -> bool:
    try:
        from .validation import validate_training_export

        return not validate_training_export(path).errors
    except (OSError, UnicodeError, json.JSONDecodeError, SchemaRegistryError, TypeError, ValueError):
        return False


def _semantic_contract_valid(path: Path, role: str) -> bool:
    if role in _GATE_CONTRACT_ROLES:
        try:
            from .gate_contract import summarize_gate_contract

            payload = json.loads(path.read_text(encoding="utf-8"))
            summary = summarize_gate_contract(payload)
        except (OSError, TypeError, UnicodeError, ValueError, json.JSONDecodeError):
            return False
        checks = payload.get("checks") if isinstance(payload.get("checks"), list) else []
        decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
        return (
            summary.get("valid") is True
            and payload.get("check_count") == len(checks)
            and decision.get("key_metrics") == payload.get("metrics")
        )
    validator_name = _SEMANTIC_VALIDATOR_NAMES.get(role)
    if validator_name is None:
        return True
    try:
        from . import validation

        validator = getattr(validation, validator_name)
        result = validator(path)
    except (
        AttributeError,
        ImportError,
        IndexError,
        KeyError,
        OSError,
        SchemaRegistryError,
        TypeError,
        UnicodeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False
    errors = getattr(result, "errors", None)
    return isinstance(errors, list) and not errors


def _empty_inspection(path: Path) -> dict[str, Any]:
    return {
        "path": path,
        "physical_exists": False,
        "regular_file": False,
        "parse_valid": False,
        "schema_valid": False,
        "semantic_valid": False,
        "ready": False,
        "payload": {},
    }


def _semantic_ready(role: str, payload: dict[str, Any]) -> bool:
    for field_name, expected in _REQUIRED_VALUES.get(role, {}).items():
        actual = payload.get(field_name)
        if isinstance(expected, bool):
            if actual is not expected:
                return False
        elif actual != expected:
            return False

    if role == "rubric_spec":
        criteria = payload.get("criteria")
        count = payload.get("criterion_count")
        return isinstance(criteria, list) and isinstance(count, int) and not isinstance(count, bool) and count > 0 and count == len(criteria)
    if role == "harness_result":
        scorecard = payload.get("scorecard")
        return isinstance(scorecard, dict) and scorecard.get("passed") is True
    if role == "cloud_training_provider_registry":
        providers = payload.get("providers")
        count = payload.get("provider_count")
        return isinstance(providers, list) and isinstance(count, int) and not isinstance(count, bool) and count > 0 and count == len(providers)
    if role == "heldout_manifest":
        count = payload.get("scenario_count")
        return isinstance(count, int) and not isinstance(count, bool) and count > 0
    if role == "external_eval_plan":
        count = payload.get("adapter_count")
        ready_count = payload.get("ready_adapter_count")
        return (
            isinstance(count, int)
            and not isinstance(count, bool)
            and count > 0
            and ready_count == count
        )
    if role == "training_export":
        redaction = payload.get("redaction_status")
        episode_count = payload.get("episode_count")
        return (
            isinstance(redaction, dict)
            and redaction.get("passed") is True
            and isinstance(episode_count, int)
            and not isinstance(episode_count, bool)
            and episode_count > 0
        )
    return True


def _directory_contains_symlink(path: Path) -> bool:
    try:
        return any(item.is_symlink() for item in path.rglob("*"))
    except OSError:
        return True
