"""Top-level governance decisions for model promotion."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .model_registry import ALIAS_NAMES, ModelRegistryError, move_model_alias, validate_model_registry

PROMOTION_DECISION_SCHEMA_VERSION = "hfr.promotion_decision.v1"
PROMOTION_POLICY_SCHEMA_VERSION = "hfr.promotion_policy.v1"
PROMOTION_CARDS_SCHEMA_VERSION = "hfr.promotion_cards.v1"
REGISTRY_ALIAS_RECEIPT_SCHEMA_VERSION = "hfr.registry_alias_receipt.v1"
PROMOTION_RELEASE_RECORD_SCHEMA_VERSION = "hfr.promotion_release_record.v1"
REGISTRY_ALIAS_APPLY_SCHEMA_VERSION = "hfr.registry_alias_apply.v1"

DEFAULT_REQUIRED_ARTIFACTS = [
    "evidence_bundle",
    "dataset_manifest",
    "dataset_card",
    "model_registry_entry",
    "training_result",
    "serving_profile",
    "model_card",
    "rollback",
]
DEFAULT_REQUIRED_EVALS = ["base", "trace_only", "frontier", "champion", "candidate"]
DEFAULT_BASELINE_ARMS = ["base", "trace_only", "frontier", "champion"]
DEFAULT_REQUIRED_GATES = ["training_gate", "compare_gate", "safety_gate"]
DEFAULT_APPROVED_LICENSE_STATUSES = ["approved", "accepted", "permitted", "reviewed", "allowed"]
DEFAULT_FORBIDDEN_CRITICAL_RULES = ["secret_exposure", "forbidden_action", "forbidden_actions", "unsupported_claim"]
DEFAULT_CARD_REQUIRED_SECTIONS = {
    "dataset_card": ["## Summary", "## Boundaries"],
    "model_card": ["## Summary", "## Intended Use", "## Limitations", "## Rollback"],
}

_POLICY_FIELDS = {
    "schema_version",
    "description",
    "required_artifacts",
    "required_evals",
    "required_gates",
    "candidate_arm",
    "baseline_arms",
    "min_pass_rate_delta",
    "min_average_score_delta",
    "max_candidate_error_count",
    "max_new_critical_failures",
    "approved_license_statuses",
    "forbidden_candidate_critical_rules",
    "require_identical_scenarios",
    "require_passed_gates",
    "require_passed_evidence_bundle",
    "require_redaction_status",
    "card_required_sections",
}


class PromotionPolicyError(ValueError):
    """Raised when a promotion policy file is malformed."""


class PromotionDecisionError(ValueError):
    """Raised when a promotion decision cannot be built."""


class RegistryAliasReceiptError(PromotionDecisionError):
    """Raised when a registry alias receipt cannot be built safely."""


def load_promotion_policy(path: str | Path) -> dict[str, Any]:
    """Load and validate a versioned promotion policy JSON file."""
    policy_path = Path(path)
    try:
        raw = json.loads(policy_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PromotionPolicyError(f"Invalid JSON in promotion policy {policy_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise PromotionPolicyError(f"Promotion policy must be a JSON object: {policy_path}")
    if raw.get("schema_version") != PROMOTION_POLICY_SCHEMA_VERSION:
        raise PromotionPolicyError(
            f"promotion policy schema_version must be {PROMOTION_POLICY_SCHEMA_VERSION!r}; got {raw.get('schema_version')!r}"
        )
    unknown = sorted(set(raw) - _POLICY_FIELDS)
    if unknown:
        raise PromotionPolicyError(f"Unknown promotion policy field(s): {', '.join(unknown)}")

    policy = default_promotion_policy()
    if "description" in raw:
        if not isinstance(raw["description"], str):
            raise PromotionPolicyError("promotion policy field description must be a string")
        policy["description"] = raw["description"]
    for field in ("required_artifacts", "required_evals", "required_gates", "baseline_arms"):
        if field in raw and raw[field] is not None:
            policy[field] = _policy_string_list(field, raw[field])
    if "candidate_arm" in raw and raw["candidate_arm"] is not None:
        if not isinstance(raw["candidate_arm"], str) or not raw["candidate_arm"]:
            raise PromotionPolicyError("promotion policy field candidate_arm must be a non-empty string")
        policy["candidate_arm"] = _normalize_key(raw["candidate_arm"])
    for field in ("min_pass_rate_delta", "min_average_score_delta"):
        if field in raw and raw[field] is not None:
            policy[field] = _policy_non_negative_number(field, raw[field])
    for field in ("max_candidate_error_count", "max_new_critical_failures"):
        if field in raw and raw[field] is not None:
            policy[field] = _policy_non_negative_int(field, raw[field])
    for field in ("approved_license_statuses", "forbidden_candidate_critical_rules"):
        if field in raw and raw[field] is not None:
            policy[field] = _policy_string_list(field, raw[field])
    for field in (
        "require_identical_scenarios",
        "require_passed_gates",
        "require_passed_evidence_bundle",
        "require_redaction_status",
    ):
        if field in raw and raw[field] is not None:
            if not isinstance(raw[field], bool):
                raise PromotionPolicyError(f"promotion policy field {field} must be a boolean")
            policy[field] = raw[field]
    if "card_required_sections" in raw and raw["card_required_sections"] is not None:
        policy["card_required_sections"] = _policy_card_sections(raw["card_required_sections"])
    return policy


def default_promotion_policy() -> dict[str, Any]:
    """Return the default strict governance policy."""
    return {
        "schema_version": PROMOTION_POLICY_SCHEMA_VERSION,
        "required_artifacts": list(DEFAULT_REQUIRED_ARTIFACTS),
        "required_evals": list(DEFAULT_REQUIRED_EVALS),
        "required_gates": list(DEFAULT_REQUIRED_GATES),
        "candidate_arm": "candidate",
        "baseline_arms": list(DEFAULT_BASELINE_ARMS),
        "min_pass_rate_delta": 0.0,
        "min_average_score_delta": 0.0,
        "max_candidate_error_count": 0,
        "max_new_critical_failures": 0,
        "approved_license_statuses": list(DEFAULT_APPROVED_LICENSE_STATUSES),
        "forbidden_candidate_critical_rules": list(DEFAULT_FORBIDDEN_CRITICAL_RULES),
        "require_identical_scenarios": True,
        "require_passed_gates": True,
        "require_passed_evidence_bundle": True,
        "require_redaction_status": True,
        "card_required_sections": {key: list(value) for key, value in DEFAULT_CARD_REQUIRED_SECTIONS.items()},
    }


def build_promotion_decision(
    *,
    out_path: str | Path | None = None,
    artifacts: dict[str, str | Path] | None = None,
    evals: dict[str, str | Path] | None = None,
    gates: dict[str, str | Path] | None = None,
    policy: dict[str, Any] | None = None,
    policy_path: str | Path | None = None,
    preserve_paths: bool = False,
    metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a deterministic governance decision for a promotion candidate."""
    effective_policy = _effective_policy(policy)
    artifact_records = {
        _normalize_key(role): _artifact_record(_normalize_key(role), Path(path), preserve_paths)
        for role, path in (artifacts or {}).items()
    }
    eval_records = {
        _normalize_key(arm): _eval_record(_normalize_key(arm), Path(path), preserve_paths)
        for arm, path in (evals or {}).items()
    }
    gate_records = {
        _normalize_key(gate_id): _gate_record(_normalize_key(gate_id), Path(path), preserve_paths)
        for gate_id, path in (gates or {}).items()
    }

    checks: list[dict[str, Any]] = []
    _check_required_artifacts(checks, artifact_records, effective_policy)
    _check_required_evals(checks, eval_records, effective_policy)
    _check_required_gates(checks, gate_records, effective_policy)
    _check_cards(checks, artifact_records, effective_policy)
    _check_license(checks, artifact_records, effective_policy)
    _check_redaction(checks, artifact_records, effective_policy)
    _check_rollback(checks, artifact_records)
    _check_evidence_bundle(checks, artifact_records, effective_policy)
    _check_gate_results(checks, gate_records, effective_policy)
    _check_eval_policy(checks, eval_records, effective_policy)

    failed_checks = [check for check in checks if not check["passed"]]
    passed = not failed_checks
    metrics = _decision_metrics(artifact_records, eval_records, gate_records, effective_policy, failed_checks)
    decision = {
        "readiness": "ready" if passed else "blocked",
        "recommendation": "promote_candidate" if passed else "block_candidate",
        "summary": _decision_summary(passed, failed_checks, metrics),
        "blocking_check_count": len(failed_checks),
        "blocking_checks": [
            {"id": check["id"], "summary": check["summary"], "scope": check.get("scope", {})}
            for check in failed_checks
        ],
        "key_metrics": metrics,
    }
    result = {
        "schema_version": PROMOTION_DECISION_SCHEMA_VERSION,
        "decision_path": _display_path(Path(out_path), preserve_paths) if out_path is not None else "",
        "passed": passed,
        "readiness": decision["readiness"],
        "recommendation": decision["recommendation"],
        "check_count": len(checks),
        "failed_check_count": len(failed_checks),
        "checks": checks,
        "decision": decision,
        "metrics": metrics,
        "policy": _policy_summary(effective_policy, policy_path, preserve_paths),
        "artifacts": artifact_records,
        "evals": eval_records,
        "gates": gate_records,
        "metadata": dict(metadata or {}),
        "notes": [
            "Promotion decisions are deterministic governance artifacts; they do not train, serve, or move registry aliases.",
            "Alias movement should consume this artifact and only proceed when passed is true.",
        ],
    }
    return result


def build_promotion_cards(
    *,
    out_dir: str | Path,
    manifest_path: str | Path | None = None,
    artifacts: dict[str, str | Path] | None = None,
    evals: dict[str, str | Path] | None = None,
    gates: dict[str, str | Path] | None = None,
    policy: dict[str, Any] | None = None,
    policy_path: str | Path | None = None,
    preserve_paths: bool = False,
    metadata: dict[str, str] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build deterministic model and dataset cards for a promotion packet."""
    effective_policy = _effective_policy(policy)
    output_dir = Path(out_dir)
    manifest = Path(manifest_path) if manifest_path is not None else output_dir / "promotion_cards.json"
    generated = generated_at or _now_iso()
    artifact_records = {
        _normalize_key(role): _artifact_record(_normalize_key(role), Path(path), preserve_paths)
        for role, path in (artifacts or {}).items()
    }
    eval_records = {
        _normalize_key(arm): _eval_record(_normalize_key(arm), Path(path), preserve_paths)
        for arm, path in (evals or {}).items()
    }
    gate_records = {
        _normalize_key(gate_id): _gate_record(_normalize_key(gate_id), Path(path), preserve_paths)
        for gate_id, path in (gates or {}).items()
    }
    checks: list[dict[str, Any]] = []
    _check_card_source_artifacts(checks, artifact_records)
    _check_card_eval_inputs(checks, eval_records, effective_policy)
    _check_card_gate_inputs(checks, gate_records, effective_policy)
    _check_license(checks, artifact_records, effective_policy)
    _check_redaction(checks, artifact_records, effective_policy)
    _check_rollback(checks, artifact_records)

    model_card_content = _render_model_card(
        artifact_records=artifact_records,
        eval_records=eval_records,
        gate_records=gate_records,
        policy=effective_policy,
        generated_at=generated,
        metadata=dict(metadata or {}),
    )
    dataset_card_content = _render_dataset_card(
        artifact_records=artifact_records,
        eval_records=eval_records,
        policy=effective_policy,
        generated_at=generated,
        metadata=dict(metadata or {}),
    )
    cards = {
        "model_card": _card_manifest(
            output_dir / "MODEL_CARD.md",
            model_card_content,
            DEFAULT_CARD_REQUIRED_SECTIONS["model_card"],
            preserve_paths,
        ),
        "dataset_card": _card_manifest(
            output_dir / "DATASET_CARD.md",
            dataset_card_content,
            DEFAULT_CARD_REQUIRED_SECTIONS["dataset_card"],
            preserve_paths,
        ),
    }
    _check_generated_card_sections(checks, cards)
    failed_checks = [check for check in checks if not check["passed"]]
    passed = not failed_checks
    metrics = {
        "card_count": len(cards),
        "check_count": len(checks),
        "failed_check_count": len(failed_checks),
        "artifact_count": len(artifact_records),
        "eval_count": len(eval_records),
        "gate_count": len(gate_records),
        "candidate_arm": effective_policy["candidate_arm"],
        "baseline_arms": list(effective_policy["baseline_arms"]),
    }
    decision = {
        "readiness": "ready" if passed else "blocked",
        "recommendation": "use_generated_cards" if passed else "fix_card_inputs",
        "summary": _promotion_cards_summary(passed, failed_checks),
        "blocking_check_count": len(failed_checks),
        "blocking_checks": [
            {"id": check["id"], "summary": check["summary"], "scope": check.get("scope", {})}
            for check in failed_checks
        ],
        "key_metrics": metrics,
    }
    return {
        "schema_version": PROMOTION_CARDS_SCHEMA_VERSION,
        "manifest_path": _display_path(manifest, preserve_paths),
        "out_dir": _display_path(output_dir, preserve_paths),
        "generated_at": generated,
        "passed": passed,
        "readiness": decision["readiness"],
        "recommendation": decision["recommendation"],
        "cards": cards,
        "artifacts": artifact_records,
        "evals": eval_records,
        "gates": gate_records,
        "policy": _policy_summary(effective_policy, policy_path, preserve_paths),
        "check_count": len(checks),
        "failed_check_count": len(failed_checks),
        "checks": checks,
        "metrics": metrics,
        "decision": decision,
        "metadata": dict(metadata or {}),
        "notes": [
            "Promotion cards are deterministic governance artifacts generated from provided evidence.",
            "Generated cards must still be supplied to promotion-decision as dataset_card and model_card artifacts.",
        ],
    }


def build_registry_alias_receipt(
    *,
    registry: dict[str, Any],
    registry_path: str | Path,
    promotion_decision: dict[str, Any],
    promotion_decision_path: str | Path,
    alias: str,
    target: str,
    rollback_target: str | None = None,
    reason: str = "",
    out_path: str | Path | None = None,
    preserve_paths: bool = False,
    metadata: dict[str, str] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a dry-run registry alias receipt gated by a promotion decision."""
    alias_name = _normalize_key(alias)
    generated = generated_at or _now_iso()
    try:
        valid_registry = validate_model_registry(registry)
    except ModelRegistryError as exc:
        raise RegistryAliasReceiptError(f"model registry is invalid: {exc}") from exc

    entries = valid_registry.get("entries") if isinstance(valid_registry.get("entries"), dict) else {}
    aliases = valid_registry.get("aliases") if isinstance(valid_registry.get("aliases"), dict) else {}
    previous_target = aliases.get(alias_name) if alias_name in ALIAS_NAMES else None
    checks: list[dict[str, Any]] = []
    decision_block = promotion_decision.get("decision") if isinstance(promotion_decision.get("decision"), dict) else {}
    target_hints = _promotion_decision_target_hints(promotion_decision)
    rollback_required = alias_name == "champion"
    rollback_present = bool(rollback_target)
    rollback_registered = isinstance(rollback_target, str) and rollback_target in entries
    rollback_differs = not rollback_present or rollback_target != target

    _add_check(
        checks,
        "registry_valid",
        True,
        actual={"entry_count": len(entries)},
        expected={"valid": True},
        summary="model registry loaded and validated",
        scope={"registry_path": _display_path(Path(registry_path), preserve_paths)},
    )
    _add_check(
        checks,
        "alias_known",
        alias_name in ALIAS_NAMES,
        actual={"alias": alias_name},
        expected={"aliases": list(ALIAS_NAMES)},
        summary=f"alias_known: {alias_name}",
        scope={"alias": alias_name},
    )
    _add_check(
        checks,
        "target_registered",
        isinstance(target, str) and target in entries,
        actual={"target": target, "registered": target in entries},
        expected={"registered": True},
        summary=f"target_registered: {target}",
        scope={"alias": alias_name, "target": target},
    )
    _add_check(
        checks,
        "promotion_decision_schema",
        promotion_decision.get("schema_version") == PROMOTION_DECISION_SCHEMA_VERSION,
        actual={"schema_version": promotion_decision.get("schema_version")},
        expected={"schema_version": PROMOTION_DECISION_SCHEMA_VERSION},
        summary="promotion_decision_schema",
        scope={"promotion_decision": _display_path(Path(promotion_decision_path), preserve_paths)},
    )
    _add_check(
        checks,
        "promotion_decision_passed",
        promotion_decision.get("passed") is True,
        actual={"passed": promotion_decision.get("passed")},
        expected={"passed": True},
        summary="promotion_decision_passed",
        scope={"promotion_decision": _display_path(Path(promotion_decision_path), preserve_paths)},
    )
    _add_check(
        checks,
        "promotion_decision_ready",
        promotion_decision.get("readiness") == "ready",
        actual={"readiness": promotion_decision.get("readiness")},
        expected={"readiness": "ready"},
        summary="promotion_decision_ready",
        scope={"promotion_decision": _display_path(Path(promotion_decision_path), preserve_paths)},
    )
    _add_check(
        checks,
        "promotion_decision_recommendation",
        promotion_decision.get("recommendation") == "promote_candidate",
        actual={"recommendation": promotion_decision.get("recommendation")},
        expected={"recommendation": "promote_candidate"},
        summary="promotion_decision_recommendation",
        scope={"promotion_decision": _display_path(Path(promotion_decision_path), preserve_paths)},
    )
    _add_check(
        checks,
        "promotion_decision_no_failed_checks",
        promotion_decision.get("failed_check_count") == 0,
        actual={"failed_check_count": promotion_decision.get("failed_check_count")},
        expected={"failed_check_count": 0},
        summary="promotion_decision_no_failed_checks",
        scope={"promotion_decision": _display_path(Path(promotion_decision_path), preserve_paths)},
    )
    _add_check(
        checks,
        "promotion_decision_no_blocking_checks",
        decision_block.get("blocking_check_count") == 0,
        actual={"blocking_check_count": decision_block.get("blocking_check_count")},
        expected={"blocking_check_count": 0},
        summary="promotion_decision_no_blocking_checks",
        scope={"promotion_decision": _display_path(Path(promotion_decision_path), preserve_paths)},
    )
    if target_hints:
        _add_check(
            checks,
            "target_matches_promotion_decision",
            target in target_hints,
            actual={"target": target, "promotion_decision_target_hints": sorted(target_hints)},
            expected={"target_in_hints": True},
            summary="target_matches_promotion_decision",
            scope={"alias": alias_name, "target": target},
        )
    _add_check(
        checks,
        "champion_has_rollback_target",
        not rollback_required or rollback_present,
        actual={"alias": alias_name, "rollback_target": rollback_target or ""},
        expected={"rollback_target_required": rollback_required},
        summary="champion_has_rollback_target",
        scope={"alias": alias_name},
    )
    if rollback_present:
        _add_check(
            checks,
            "rollback_target_registered",
            rollback_registered,
            actual={"rollback_target": rollback_target, "registered": rollback_registered},
            expected={"registered": True},
            summary=f"rollback_target_registered: {rollback_target}",
            scope={"alias": alias_name, "rollback_target": rollback_target},
        )
        _add_check(
            checks,
            "rollback_target_differs",
            rollback_differs,
            actual={"target": target, "rollback_target": rollback_target},
            expected={"different": True},
            summary="rollback_target_differs",
            scope={"alias": alias_name, "target": target, "rollback_target": rollback_target},
        )

    failed_checks = [check for check in checks if not check["passed"]]
    passed = not failed_checks
    registry_record = _base_record("model_registry", Path(registry_path), preserve_paths)
    promotion_decision_record = _promotion_decision_record(Path(promotion_decision_path), promotion_decision, preserve_paths)
    planned_history = _planned_alias_history(alias_name, previous_target, target, rollback_target, aliases.get("rollback"), reason, generated)
    metrics = {
        "check_count": len(checks),
        "failed_check_count": len(failed_checks),
        "registry_entry_count": len(entries),
        "promotion_decision_failed_check_count": promotion_decision.get("failed_check_count"),
        "promotion_decision_blocking_check_count": decision_block.get("blocking_check_count"),
        "planned_alias_history_count": len(planned_history),
        "target_registered": target in entries,
        "rollback_required": rollback_required,
        "rollback_registered": rollback_registered if rollback_present else None,
        "side_effects": False,
    }
    decision = {
        "readiness": "ready" if passed else "blocked",
        "recommendation": "apply_alias_update" if passed else "block_alias_update",
        "summary": _alias_receipt_summary(passed, failed_checks, alias_name, target),
        "blocking_check_count": len(failed_checks),
        "blocking_checks": [
            {"id": check["id"], "summary": check["summary"], "scope": check.get("scope", {})}
            for check in failed_checks
        ],
        "key_metrics": metrics,
    }
    return {
        "schema_version": REGISTRY_ALIAS_RECEIPT_SCHEMA_VERSION,
        "receipt_path": _display_path(Path(out_path), preserve_paths) if out_path is not None else "",
        "generated_at": generated,
        "passed": passed,
        "readiness": decision["readiness"],
        "recommendation": decision["recommendation"],
        "registry": registry_record,
        "promotion_decision": promotion_decision_record,
        "alias_update": {
            "alias": alias_name,
            "previous_target": previous_target,
            "target": target,
            "rollback_target": rollback_target,
            "reason": reason,
            "dry_run": True,
            "applied": False,
            "side_effects": False,
            "planned_alias_history": planned_history,
        },
        "check_count": len(checks),
        "failed_check_count": len(failed_checks),
        "checks": checks,
        "metrics": metrics,
        "decision": decision,
        "metadata": dict(metadata or {}),
        "notes": [
            "This receipt is a governance artifact only; it does not mutate the model registry.",
            "Registry alias movement must revalidate this receipt and the referenced promotion decision before applying changes.",
        ],
    }


def build_promotion_release_record(
    *,
    release_id: str,
    out_path: str | Path | None = None,
    notes_path: str | Path | None = None,
    promotion_decision_path: str | Path,
    promotion_cards_path: str | Path,
    registry_alias_receipt_path: str | Path,
    rollback_path: str | Path,
    evals: dict[str, str | Path] | None = None,
    policy: dict[str, Any] | None = None,
    policy_path: str | Path | None = None,
    preserve_paths: bool = False,
    metadata: dict[str, str] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build an auditable release record from passed governance artifacts."""
    effective_policy = _effective_policy(policy)
    generated = generated_at or _now_iso()
    release_notes_path = Path(notes_path) if notes_path is not None else Path(out_path or "promotion_release_record.json").with_name("RELEASE_NOTES.md")
    components = {
        "promotion_decision": _artifact_record("promotion_decision", Path(promotion_decision_path), preserve_paths),
        "promotion_cards": _artifact_record("promotion_cards", Path(promotion_cards_path), preserve_paths),
        "registry_alias_receipt": _artifact_record("registry_alias_receipt", Path(registry_alias_receipt_path), preserve_paths),
        "rollback": _artifact_record("rollback", Path(rollback_path), preserve_paths),
    }
    eval_records = {
        _normalize_key(arm): _eval_record(_normalize_key(arm), Path(path), preserve_paths)
        for arm, path in (evals or {}).items()
    }
    checks: list[dict[str, Any]] = []
    _check_release_id(checks, release_id)
    _check_release_components(checks, components)
    _check_release_component_decisions(checks, components)
    _check_release_cross_references(checks, components)
    _check_card_eval_inputs(checks, eval_records, effective_policy)
    _check_rollback(checks, {"rollback": components["rollback"]})

    release_notes_content = _render_release_notes(
        release_id=release_id,
        generated_at=generated,
        components=components,
        eval_records=eval_records,
        policy=effective_policy,
        metadata=dict(metadata or {}),
    )
    release_notes = _card_manifest(release_notes_path, release_notes_content, ["## Summary", "## Governance Components", "## Rollback"], preserve_paths)
    failed_checks = [check for check in checks if not check["passed"]]
    passed = not failed_checks
    metrics = {
        "component_count": len(components),
        "eval_count": len(eval_records),
        "check_count": len(checks),
        "failed_check_count": len(failed_checks),
        "candidate_arm": effective_policy["candidate_arm"],
        "baseline_arms": list(effective_policy["baseline_arms"]),
    }
    decision = {
        "readiness": "ready" if passed else "blocked",
        "recommendation": "record_release" if passed else "block_release",
        "summary": _release_record_summary(passed, failed_checks, release_id),
        "blocking_check_count": len(failed_checks),
        "blocking_checks": [
            {"id": check["id"], "summary": check["summary"], "scope": check.get("scope", {})}
            for check in failed_checks
        ],
        "key_metrics": metrics,
    }
    return {
        "schema_version": PROMOTION_RELEASE_RECORD_SCHEMA_VERSION,
        "release_id": release_id,
        "record_path": _display_path(Path(out_path), preserve_paths) if out_path is not None else "",
        "generated_at": generated,
        "passed": passed,
        "readiness": decision["readiness"],
        "recommendation": decision["recommendation"],
        "components": components,
        "evals": eval_records,
        "release_notes": release_notes,
        "policy": _policy_summary(effective_policy, policy_path, preserve_paths),
        "check_count": len(checks),
        "failed_check_count": len(failed_checks),
        "checks": checks,
        "metrics": metrics,
        "decision": decision,
        "metadata": dict(metadata or {}),
        "notes": [
            "Release records bind governance artifacts before side-effectful promotion.",
            "Registry alias movement must still revalidate the alias receipt immediately before applying changes.",
        ],
    }


def apply_registry_alias_receipt(
    *,
    registry: dict[str, Any],
    registry_path: str | Path,
    registry_alias_receipt: dict[str, Any],
    registry_alias_receipt_path: str | Path,
    promotion_release_record: dict[str, Any],
    promotion_release_record_path: str | Path,
    out_path: str | Path | None = None,
    preserve_paths: bool = False,
    metadata: dict[str, str] | None = None,
    applied_at: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Apply a registry alias movement only after revalidating governance records."""
    applied = applied_at or _now_iso()
    try:
        valid_registry = validate_model_registry(registry)
    except ModelRegistryError as exc:
        raise RegistryAliasReceiptError(f"model registry is invalid: {exc}") from exc

    registry_before = _base_record("model_registry", Path(registry_path), preserve_paths)
    receipt_record = _artifact_record("registry_alias_receipt", Path(registry_alias_receipt_path), preserve_paths)
    release_record = _artifact_record("promotion_release_record", Path(promotion_release_record_path), preserve_paths)
    alias_update = registry_alias_receipt.get("alias_update") if isinstance(registry_alias_receipt.get("alias_update"), dict) else {}
    alias = _normalize_key(str(alias_update.get("alias") or ""))
    target = str(alias_update.get("target") or "")
    rollback_target = alias_update.get("rollback_target") if isinstance(alias_update.get("rollback_target"), str) else None
    reason = str(alias_update.get("reason") or "")
    aliases = valid_registry.get("aliases") if isinstance(valid_registry.get("aliases"), dict) else {}
    entries = valid_registry.get("entries") if isinstance(valid_registry.get("entries"), dict) else {}
    current_target = aliases.get(alias) if alias in ALIAS_NAMES else None

    checks: list[dict[str, Any]] = []
    _add_check(
        checks,
        "registry_valid",
        True,
        actual={"entry_count": len(entries)},
        expected={"valid": True},
        summary="model registry loaded and validated",
        scope={"registry_path": _display_path(Path(registry_path), preserve_paths)},
    )
    _add_check(
        checks,
        "alias_receipt_schema",
        registry_alias_receipt.get("schema_version") == REGISTRY_ALIAS_RECEIPT_SCHEMA_VERSION,
        actual={"schema_version": registry_alias_receipt.get("schema_version")},
        expected={"schema_version": REGISTRY_ALIAS_RECEIPT_SCHEMA_VERSION},
        summary="alias_receipt_schema",
        scope={"path": _display_path(Path(registry_alias_receipt_path), preserve_paths)},
    )
    _add_check(
        checks,
        "alias_receipt_passed",
        registry_alias_receipt.get("passed") is True and registry_alias_receipt.get("recommendation") == "apply_alias_update",
        actual={"passed": registry_alias_receipt.get("passed"), "recommendation": registry_alias_receipt.get("recommendation")},
        expected={"passed": True, "recommendation": "apply_alias_update"},
        summary="alias_receipt_passed",
        scope={"path": _display_path(Path(registry_alias_receipt_path), preserve_paths)},
    )
    _add_check(
        checks,
        "release_record_schema",
        promotion_release_record.get("schema_version") == PROMOTION_RELEASE_RECORD_SCHEMA_VERSION,
        actual={"schema_version": promotion_release_record.get("schema_version")},
        expected={"schema_version": PROMOTION_RELEASE_RECORD_SCHEMA_VERSION},
        summary="release_record_schema",
        scope={"path": _display_path(Path(promotion_release_record_path), preserve_paths)},
    )
    _add_check(
        checks,
        "release_record_passed",
        promotion_release_record.get("passed") is True and promotion_release_record.get("recommendation") == "record_release",
        actual={"passed": promotion_release_record.get("passed"), "recommendation": promotion_release_record.get("recommendation")},
        expected={"passed": True, "recommendation": "record_release"},
        summary="release_record_passed",
        scope={"path": _display_path(Path(promotion_release_record_path), preserve_paths)},
    )
    release_receipt_sha = _release_component_sha(promotion_release_record, "registry_alias_receipt")
    _add_check(
        checks,
        "release_references_alias_receipt",
        bool(receipt_record.get("sha256")) and release_receipt_sha == receipt_record.get("sha256"),
        actual={"release_receipt_sha256": release_receipt_sha, "current_receipt_sha256": receipt_record.get("sha256")},
        expected={"sha256_match": True},
        summary="release_references_alias_receipt",
        scope={"role": "registry_alias_receipt"},
    )
    receipt_registry = registry_alias_receipt.get("registry") if isinstance(registry_alias_receipt.get("registry"), dict) else {}
    _add_check(
        checks,
        "registry_matches_alias_receipt",
        bool(registry_before.get("sha256")) and registry_before.get("sha256") == receipt_registry.get("sha256"),
        actual={"current_registry_sha256": registry_before.get("sha256"), "receipt_registry_sha256": receipt_registry.get("sha256")},
        expected={"sha256_match": True},
        summary="registry_matches_alias_receipt",
        scope={"registry_path": _display_path(Path(registry_path), preserve_paths)},
    )
    _add_check(
        checks,
        "alias_update_known",
        alias in ALIAS_NAMES,
        actual={"alias": alias},
        expected={"aliases": list(ALIAS_NAMES)},
        summary=f"alias_update_known: {alias or 'missing'}",
        scope={"alias": alias},
    )
    _add_check(
        checks,
        "alias_current_target_matches_receipt",
        current_target == alias_update.get("previous_target"),
        actual={"current_target": current_target, "receipt_previous_target": alias_update.get("previous_target")},
        expected={"match": True},
        summary="alias_current_target_matches_receipt",
        scope={"alias": alias},
    )
    _add_check(
        checks,
        "alias_target_registered",
        target in entries,
        actual={"target": target, "registered": target in entries},
        expected={"registered": True},
        summary=f"alias_target_registered: {target or 'missing'}",
        scope={"alias": alias, "target": target},
    )
    if rollback_target:
        _add_check(
            checks,
            "rollback_target_registered",
            rollback_target in entries,
            actual={"rollback_target": rollback_target, "registered": rollback_target in entries},
            expected={"registered": True},
            summary=f"rollback_target_registered: {rollback_target}",
            scope={"alias": alias, "rollback_target": rollback_target},
        )

    failed_checks = [check for check in checks if not check["passed"]]
    updated_registry: dict[str, Any] | None = None
    registry_after: dict[str, Any] = {
        "applied": False,
        "aliases": copy_aliases(valid_registry),
        "alias_history_count": len(valid_registry.get("alias_history", [])) if isinstance(valid_registry.get("alias_history"), list) else 0,
    }
    if not failed_checks:
        updated_registry = move_model_alias(
            valid_registry,
            alias=alias,
            target=target,
            rollback_target=rollback_target,
            reason=reason,
            moved_at=applied,
        )
        registry_after = {
            "applied": True,
            "aliases": copy_aliases(updated_registry),
            "alias_history_count": len(updated_registry.get("alias_history", [])) if isinstance(updated_registry.get("alias_history"), list) else 0,
            "sha256_after_write": _payload_file_sha256(updated_registry),
        }

    passed = updated_registry is not None
    metrics = {
        "check_count": len(checks),
        "failed_check_count": len(failed_checks),
        "registry_entry_count": len(entries),
        "alias_history_count_before": len(valid_registry.get("alias_history", [])) if isinstance(valid_registry.get("alias_history"), list) else 0,
        "alias_history_count_after": registry_after["alias_history_count"],
        "side_effects": passed,
    }
    decision = {
        "readiness": "ready" if passed else "blocked",
        "recommendation": "alias_applied" if passed else "block_alias_apply",
        "summary": _alias_apply_summary(passed, failed_checks, alias, target),
        "blocking_check_count": len(failed_checks),
        "blocking_checks": [
            {"id": check["id"], "summary": check["summary"], "scope": check.get("scope", {})}
            for check in failed_checks
        ],
        "key_metrics": metrics,
    }
    record = {
        "schema_version": REGISTRY_ALIAS_APPLY_SCHEMA_VERSION,
        "apply_path": _display_path(Path(out_path), preserve_paths) if out_path is not None else "",
        "applied_at": applied,
        "passed": passed,
        "readiness": decision["readiness"],
        "recommendation": decision["recommendation"],
        "registry_before": registry_before,
        "registry_after": registry_after,
        "registry_alias_receipt": receipt_record,
        "promotion_release_record": release_record,
        "alias_update": {
            "alias": alias,
            "previous_target": current_target,
            "target": target,
            "rollback_target": rollback_target,
            "reason": reason,
            "applied": passed,
            "side_effects": passed,
        },
        "check_count": len(checks),
        "failed_check_count": len(failed_checks),
        "checks": checks,
        "metrics": metrics,
        "decision": decision,
        "metadata": dict(metadata or {}),
        "notes": [
            "This is the only guarded Goal 8 path that mutates registry aliases.",
            "The registry is written only when the alias receipt, release record, and current registry fingerprint all match.",
        ],
    }
    return record, updated_registry


def _check_release_id(checks: list[dict[str, Any]], release_id: str) -> None:
    _add_check(
        checks,
        "release_id_present",
        isinstance(release_id, str) and bool(release_id.strip()),
        actual={"release_id": release_id},
        expected={"non_empty": True},
        summary="release_id_present",
        scope={"release_id": release_id},
    )


def _check_release_components(checks: list[dict[str, Any]], components: dict[str, dict[str, Any]]) -> None:
    expected_schemas = {
        "promotion_decision": PROMOTION_DECISION_SCHEMA_VERSION,
        "promotion_cards": PROMOTION_CARDS_SCHEMA_VERSION,
        "registry_alias_receipt": REGISTRY_ALIAS_RECEIPT_SCHEMA_VERSION,
    }
    for role, record in components.items():
        _add_check(
            checks,
            "release_component_present",
            _record_is_file(record),
            actual={"present": _record_is_file(record), "path": record.get("path") if isinstance(record, dict) else ""},
            expected={"role": role, "kind": "file"},
            summary=f"release_component_present[{role}]: {'present' if _record_is_file(record) else 'missing'}",
            scope={"role": role},
        )
        if record.get("parse_error"):
            _add_check(
                checks,
                "release_component_parseable",
                False,
                actual={"error": record["parse_error"]},
                expected={"parseable": True},
                summary=f"release_component_parseable[{role}]: {record['parse_error']}",
                scope={"role": role},
            )
        expected_schema = expected_schemas.get(role)
        if expected_schema:
            payload = record.get("json") if isinstance(record.get("json"), dict) else {}
            _add_check(
                checks,
                "release_component_schema",
                payload.get("schema_version") == expected_schema,
                actual={"schema_version": payload.get("schema_version")},
                expected={"schema_version": expected_schema},
                summary=f"release_component_schema[{role}]",
                scope={"role": role},
            )


def _check_release_component_decisions(checks: list[dict[str, Any]], components: dict[str, dict[str, Any]]) -> None:
    expectations = {
        "promotion_decision": "promote_candidate",
        "promotion_cards": "use_generated_cards",
        "registry_alias_receipt": "apply_alias_update",
    }
    for role, recommendation in expectations.items():
        payload = components.get(role, {}).get("json") if isinstance(components.get(role, {}).get("json"), dict) else {}
        _add_check(
            checks,
            "release_component_passed",
            payload.get("passed") is True,
            actual={"passed": payload.get("passed")},
            expected={"passed": True},
            summary=f"release_component_passed[{role}]",
            scope={"role": role},
        )
        _add_check(
            checks,
            "release_component_recommendation",
            payload.get("recommendation") == recommendation,
            actual={"recommendation": payload.get("recommendation")},
            expected={"recommendation": recommendation},
            summary=f"release_component_recommendation[{role}]",
            scope={"role": role},
        )


def _check_release_cross_references(checks: list[dict[str, Any]], components: dict[str, dict[str, Any]]) -> None:
    decision_record = components.get("promotion_decision", {})
    receipt_payload = components.get("registry_alias_receipt", {}).get("json")
    receipt_decision = receipt_payload.get("promotion_decision") if isinstance(receipt_payload, dict) and isinstance(receipt_payload.get("promotion_decision"), dict) else {}
    if decision_record.get("sha256") and receipt_decision.get("sha256"):
        _add_check(
            checks,
            "alias_receipt_references_decision",
            receipt_decision.get("sha256") == decision_record.get("sha256"),
            actual={"receipt_decision_sha256": receipt_decision.get("sha256"), "promotion_decision_sha256": decision_record.get("sha256")},
            expected={"sha256_match": True},
            summary="alias_receipt_references_decision",
            scope={"role": "registry_alias_receipt"},
        )
    cards_record = components.get("promotion_cards", {})
    cards_payload = cards_record.get("json") if isinstance(cards_record.get("json"), dict) else {}
    for role in ("model_card", "dataset_card"):
        card = cards_payload.get("cards", {}).get(role) if isinstance(cards_payload.get("cards"), dict) else None
        _add_check(
            checks,
            "release_card_present",
            isinstance(card, dict) and bool(card.get("content_sha256")),
            actual={"present": isinstance(card, dict)},
            expected={"content_sha256": "non-empty"},
            summary=f"release_card_present[{role}]",
            scope={"role": role},
        )


def _release_component_sha(release_record: dict[str, Any], role: str) -> str:
    components = release_record.get("components") if isinstance(release_record.get("components"), dict) else {}
    component = components.get(role) if isinstance(components.get(role), dict) else {}
    value = component.get("sha256")
    return value if isinstance(value, str) else ""


def copy_aliases(registry: dict[str, Any]) -> dict[str, Any]:
    aliases = registry.get("aliases") if isinstance(registry.get("aliases"), dict) else {}
    return {alias: aliases.get(alias) for alias in ALIAS_NAMES}


def _alias_apply_summary(passed: bool, failed_checks: list[dict[str, Any]], alias: str, target: str) -> str:
    if passed:
        return f"Registry alias applied: {alias} -> {target}."
    return f"Registry alias apply blocked: {len(failed_checks)} check(s) failed for {alias or 'missing'} -> {target or 'missing'}."


def _release_record_summary(passed: bool, failed_checks: list[dict[str, Any]], release_id: str) -> str:
    if passed:
        return f"Release record ready: {release_id} has passed governance components and release notes."
    return f"Release record blocked: {len(failed_checks)} check(s) failed for {release_id}."


def _render_release_notes(
    *,
    release_id: str,
    generated_at: str,
    components: dict[str, dict[str, Any]],
    eval_records: dict[str, dict[str, Any]],
    policy: dict[str, Any],
    metadata: dict[str, str],
) -> str:
    decision = _artifact_payload(components, "promotion_decision")
    alias_receipt = _artifact_payload(components, "registry_alias_receipt")
    rollback = _artifact_payload(components, "rollback")
    alias_update = alias_receipt.get("alias_update") if isinstance(alias_receipt.get("alias_update"), dict) else {}
    lines = [
        f"# Release Notes: {release_id}",
        "",
        "## Summary",
        "",
        f"- Generated at: {generated_at}",
        f"- Promotion decision: {decision.get('recommendation', 'unknown')}",
        f"- Alias update: {alias_update.get('alias', 'unknown')} -> {alias_update.get('target', 'unknown')}",
        f"- Rollback target: {_rollback_target(rollback) or alias_update.get('rollback_target') or 'missing'}",
        "",
        "## Governance Components",
        "",
        _markdown_table(["Role", "Schema", "Path", "SHA-256"], _artifact_rows(components)),
        "",
        "## Evaluation Summary",
        "",
        _markdown_table(
            ["Arm", "Model", "Scenarios", "Passed", "Failed", "Pass Rate", "Average Score", "Errors"],
            _eval_rows(eval_records, policy["required_evals"]),
        ),
        "",
        "## Rollback",
        "",
        f"- Registry rollback alias target: {alias_update.get('rollback_target') or 'missing'}",
        f"- Rollback artifact target: {_rollback_target(rollback) or 'missing'}",
        "",
        "## Safety Notes",
        "",
        "- Promotion remains blocked if referenced governance artifacts are stale or fail validation.",
        "- Alias movement remains a separate side-effectful operation and must revalidate the alias receipt immediately before applying.",
    ]
    if metadata:
        lines.extend(["", "## Metadata", "", _markdown_table(["Key", "Value"], [{"key": key, "value": metadata[key]} for key in sorted(metadata)], key_map={"Key": "key", "Value": "value"})])
    return "\n".join(lines).rstrip() + "\n"


def _check_card_source_artifacts(checks: list[dict[str, Any]], artifacts: dict[str, dict[str, Any]]) -> None:
    for role in ("dataset_manifest", "model_registry_entry", "rollback", "training_result", "serving_profile"):
        record = artifacts.get(role)
        _add_check(
            checks,
            "card_source_artifact",
            _record_is_file(record),
            actual={"present": _record_is_file(record), "path": record.get("path") if isinstance(record, dict) else ""},
            expected={"role": role, "kind": "file"},
            summary=f"card_source_artifact[{role}]: {'present' if _record_is_file(record) else 'missing'}",
            scope={"role": role},
        )
        if isinstance(record, dict) and record.get("parse_error"):
            _add_check(
                checks,
                "card_source_parseable",
                False,
                actual={"error": record["parse_error"]},
                expected={"parseable": True},
                summary=f"card_source_parseable[{role}]: {record['parse_error']}",
                scope={"role": role},
            )


def _check_card_eval_inputs(checks: list[dict[str, Any]], evals: dict[str, dict[str, Any]], policy: dict[str, Any]) -> None:
    for arm in policy["required_evals"]:
        record = evals.get(arm)
        _add_check(
            checks,
            "card_eval_summary",
            _record_is_file(record) and isinstance(record.get("summary"), dict) if isinstance(record, dict) else False,
            actual={"present": _record_is_file(record), "parse_error": record.get("parse_error") if isinstance(record, dict) else None},
            expected={"arm": arm, "parseable": True},
            summary=f"card_eval_summary[{arm}]: {'present' if _record_is_file(record) else 'missing'}",
            scope={"arm": arm},
        )


def _check_card_gate_inputs(checks: list[dict[str, Any]], gates: dict[str, dict[str, Any]], policy: dict[str, Any]) -> None:
    for gate_id in policy["required_gates"]:
        record = gates.get(gate_id)
        _add_check(
            checks,
            "card_gate_summary",
            _record_is_file(record) and record.get("parse_error") is None if isinstance(record, dict) else False,
            actual={"present": _record_is_file(record), "parse_error": record.get("parse_error") if isinstance(record, dict) else None},
            expected={"gate": gate_id, "parseable": True},
            summary=f"card_gate_summary[{gate_id}]: {'present' if _record_is_file(record) else 'missing'}",
            scope={"gate": gate_id},
        )


def _check_generated_card_sections(checks: list[dict[str, Any]], cards: dict[str, dict[str, Any]]) -> None:
    for role, card in cards.items():
        content = card.get("content") if isinstance(card.get("content"), str) else ""
        for section in card.get("required_sections", []):
            _add_check(
                checks,
                "generated_card_section",
                bool(content and section in content),
                actual={"found": bool(content and section in content)},
                expected={"section": section},
                summary=f"generated_card_section[{role}]: {section}",
                scope={"role": role, "section": section},
            )


def _card_manifest(path: Path, content: str, required_sections: list[str], preserve_paths: bool) -> dict[str, Any]:
    encoded = content.encode("utf-8")
    return {
        "path": _display_path(path, preserve_paths),
        "kind": "markdown",
        "size_bytes": len(encoded),
        "content_sha256": hashlib.sha256(encoded).hexdigest(),
        "required_sections": list(required_sections),
        "content": content,
    }


def _promotion_cards_summary(passed: bool, failed_checks: list[dict[str, Any]]) -> str:
    if passed:
        return "Promotion cards ready: generated model and dataset cards include required governance sections."
    return f"Promotion cards blocked: {len(failed_checks)} source or generated-card check(s) failed."


def _render_model_card(
    *,
    artifact_records: dict[str, dict[str, Any]],
    eval_records: dict[str, dict[str, Any]],
    gate_records: dict[str, dict[str, Any]],
    policy: dict[str, Any],
    generated_at: str,
    metadata: dict[str, str],
) -> str:
    model_payload = _artifact_payload(artifact_records, "model_registry_entry")
    rollback_payload = _artifact_payload(artifact_records, "rollback")
    training_payload = _artifact_payload(artifact_records, "training_result")
    serving_payload = _artifact_payload(artifact_records, "serving_profile")
    candidate_summary = _summary_for_arm(eval_records, policy["candidate_arm"])
    critical_counts = candidate_summary.get("critical_failure_counts") if isinstance(candidate_summary.get("critical_failure_counts"), dict) else {}
    lines = [
        "# Model Card",
        "",
        "## Summary",
        "",
        f"- Generated at: {generated_at}",
        f"- Candidate model: {_model_identifier(model_payload)}",
        f"- Candidate arm: {policy['candidate_arm']}",
        f"- Training status: {_passed_label(training_payload.get('passed'))}",
        f"- Serving status: {_passed_label(serving_payload.get('passed'))}",
        f"- License status: {_license_status(model_payload)}",
        "",
        "## Intended Use",
        "",
        "- Agentic task-completion workflows covered by the supplied held-out evaluation scenarios.",
        "- Promotion review only after evidence, dataset, training, serving, comparison, and safety gates pass.",
        "",
        "## Limitations",
        "",
        "- Do not use outside the scenario families and tool policies represented by the supplied evaluation artifacts.",
        "- Do not promote if any referenced gate, redaction, license, rollback, or card validation fails.",
        "- Unsupported claims, forbidden actions, secret exposure, and task-completion regression remain blocking conditions.",
        "",
        "## Evaluation",
        "",
        _markdown_table(
            ["Arm", "Model", "Scenarios", "Passed", "Failed", "Pass Rate", "Average Score", "Errors"],
            _eval_rows(eval_records, policy["required_evals"]),
        ),
        "",
        "## Safety",
        "",
        _markdown_table(["Gate", "Passed", "Readiness", "Recommendation"], _gate_rows(gate_records, policy["required_gates"])),
        "",
        "Critical failure counts for candidate:",
        "",
        _markdown_table(["Rule", "Count"], _count_rows(critical_counts) or [{"id": "none", "count": 0}], key_map={"Rule": "id", "Count": "count"}),
        "",
        "## Rollback",
        "",
        f"- Rollback target: {_rollback_target(rollback_payload) or 'missing'}",
        f"- Rollback artifact schema: {rollback_payload.get('schema_version', 'unknown')}",
        "",
        "## Provenance",
        "",
        _markdown_table(["Role", "Schema", "Path", "SHA-256"], _artifact_rows(artifact_records)),
    ]
    if metadata:
        lines.extend(["", "## Metadata", "", _markdown_table(["Key", "Value"], [{"key": key, "value": metadata[key]} for key in sorted(metadata)], key_map={"Key": "key", "Value": "value"})])
    return "\n".join(lines).rstrip() + "\n"


def _render_dataset_card(
    *,
    artifact_records: dict[str, dict[str, Any]],
    eval_records: dict[str, dict[str, Any]],
    policy: dict[str, Any],
    generated_at: str,
    metadata: dict[str, str],
) -> str:
    dataset_payload = _artifact_payload(artifact_records, "dataset_manifest")
    redaction_statuses = _redaction_statuses(dataset_payload)
    quality_flags = _quality_flags(dataset_payload)
    lines = [
        "# Dataset Card",
        "",
        "## Summary",
        "",
        f"- Generated at: {generated_at}",
        f"- Dataset identity: {_dataset_identifier(dataset_payload)}",
        f"- Candidate arm: {policy['candidate_arm']}",
        f"- Evaluation scenarios: {_scenario_count(eval_records, policy['candidate_arm'])}",
        "",
        "## Boundaries",
        "",
        "- This dataset card covers only the supplied promotion packet artifacts.",
        "- Do not train or promote from this dataset when redaction status is missing or any critical quality flag is present.",
        "- Do not reuse beyond the recorded task families, scenario contracts, and source fingerprints without revalidation.",
        "",
        "## Redaction",
        "",
        f"- Redaction status: {redaction_statuses[0] if redaction_statuses else 'missing'}",
        f"- Critical redaction or secret quality flags: {len(quality_flags)}",
        "",
        "## Label Provenance",
        "",
        _markdown_list(_label_provenance_lines(dataset_payload)),
        "",
        "## Evaluation Coverage",
        "",
        _markdown_table(
            ["Arm", "Scenarios", "Passed", "Failed", "Pass Rate", "Average Score"],
            _eval_rows(eval_records, policy["required_evals"]),
            key_map={
                "Arm": "arm",
                "Scenarios": "scenario_count",
                "Passed": "passed",
                "Failed": "failed",
                "Pass Rate": "pass_rate",
                "Average Score": "average_score",
            },
        ),
        "",
        "## Source Artifacts",
        "",
        _markdown_table(["Role", "Schema", "Path", "SHA-256"], _artifact_rows(artifact_records)),
    ]
    if metadata:
        lines.extend(["", "## Metadata", "", _markdown_table(["Key", "Value"], [{"key": key, "value": metadata[key]} for key in sorted(metadata)], key_map={"Key": "key", "Value": "value"})])
    return "\n".join(lines).rstrip() + "\n"


def _artifact_payload(records: dict[str, dict[str, Any]], role: str) -> dict[str, Any]:
    record = records.get(role)
    payload = record.get("json") if isinstance(record, dict) and isinstance(record.get("json"), dict) else {}
    return payload


def _summary_for_arm(records: dict[str, dict[str, Any]], arm: str) -> dict[str, Any]:
    record = records.get(arm)
    summary = record.get("summary") if isinstance(record, dict) and isinstance(record.get("summary"), dict) else {}
    return summary


def _model_identifier(payload: dict[str, Any]) -> str:
    for path in (("model_id",), ("candidate", "model_id"), ("model", "model_id"), ("entry_id",), ("candidate_id",)):
        value = _nested(payload, path)
        if isinstance(value, str) and value:
            return value
    return "unknown"


def _dataset_identifier(payload: dict[str, Any]) -> str:
    for path in (("dataset_version",), ("dataset_id",), ("id",), ("manifest", "dataset_version")):
        value = _nested(payload, path)
        if isinstance(value, str) and value:
            return value
    return "unknown"


def _passed_label(value: Any) -> str:
    if value is True:
        return "passed"
    if value is False:
        return "failed"
    return "unknown"


def _scenario_count(eval_records: dict[str, dict[str, Any]], arm: str) -> int:
    summary = _summary_for_arm(eval_records, arm)
    scenario_ids = summary.get("scenario_ids")
    return len(scenario_ids) if isinstance(scenario_ids, list) else 0


def _eval_rows(records: dict[str, dict[str, Any]], arms: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in arms:
        summary = _summary_for_arm(records, arm)
        rows.append(
            {
                "arm": arm,
                "model": summary.get("model") or "unknown",
                "scenario_count": len(summary.get("scenario_ids", [])) if isinstance(summary.get("scenario_ids"), list) else 0,
                "passed": summary.get("passed", 0),
                "failed": summary.get("failed", 0),
                "pass_rate": summary.get("pass_rate"),
                "average_score": summary.get("average_score"),
                "error_count": summary.get("error_count", 0),
            }
        )
    return rows


def _gate_rows(records: dict[str, dict[str, Any]], gate_ids: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gate_id in gate_ids:
        record = records.get(gate_id) if isinstance(records.get(gate_id), dict) else {}
        payload = record.get("json") if isinstance(record.get("json"), dict) else {}
        decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
        rows.append(
            {
                "gate": gate_id,
                "passed": _passed_label(payload.get("passed")),
                "readiness": decision.get("readiness") or payload.get("readiness") or "unknown",
                "recommendation": decision.get("recommendation") or payload.get("recommendation") or "unknown",
            }
        )
    return rows


def _artifact_rows(records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role in sorted(records):
        record = records[role]
        rows.append(
            {
                "role": role,
                "schema": record.get("schema_version") or "-",
                "path": record.get("path") or "-",
                "sha256": record.get("sha256") or "-",
            }
        )
    return rows


def _label_provenance_lines(payload: dict[str, Any]) -> list[str]:
    provenance = payload.get("label_provenance")
    if isinstance(provenance, dict) and provenance:
        return [f"{key}: {provenance[key]}" for key in sorted(provenance)]
    if isinstance(provenance, list) and provenance:
        return [str(item) for item in provenance]
    return ["No label provenance field was present in the dataset manifest."]


def _markdown_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _markdown_table(headers: list[str], rows: list[dict[str, Any]], key_map: dict[str, str] | None = None) -> str:
    key_map = key_map or {
        "Arm": "arm",
        "Model": "model",
        "Scenarios": "scenario_count",
        "Passed": "passed",
        "Failed": "failed",
        "Pass Rate": "pass_rate",
        "Average Score": "average_score",
        "Errors": "error_count",
        "Gate": "gate",
        "Readiness": "readiness",
        "Recommendation": "recommendation",
        "Role": "role",
        "Schema": "schema",
        "Path": "path",
        "SHA-256": "sha256",
        "Key": "key",
        "Value": "value",
    }
    if not rows:
        rows = [{key_map.get(header, header.lower().replace(" ", "_")): "-" for header in headers}]
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = []
    for row in rows:
        values = []
        for header in headers:
            key = key_map.get(header, header.lower().replace(" ", "_"))
            values.append(_markdown_cell(row.get(key, "-")))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header_line, separator, *body])


def _markdown_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        text = f"{value:.4g}"
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _effective_policy(policy: dict[str, Any] | None) -> dict[str, Any]:
    base = default_promotion_policy()
    if policy is None:
        return base
    merged = {**base, **policy}
    merged["required_artifacts"] = [_normalize_key(item) for item in merged["required_artifacts"]]
    merged["required_evals"] = [_normalize_key(item) for item in merged["required_evals"]]
    merged["required_gates"] = [_normalize_key(item) for item in merged["required_gates"]]
    merged["baseline_arms"] = [_normalize_key(item) for item in merged["baseline_arms"]]
    merged["candidate_arm"] = _normalize_key(merged["candidate_arm"])
    merged["approved_license_statuses"] = [str(item).lower() for item in merged["approved_license_statuses"]]
    merged["forbidden_candidate_critical_rules"] = [str(item) for item in merged["forbidden_candidate_critical_rules"]]
    return merged


def _check_required_artifacts(checks: list[dict[str, Any]], artifacts: dict[str, dict[str, Any]], policy: dict[str, Any]) -> None:
    for role in policy["required_artifacts"]:
        record = artifacts.get(role)
        _add_check(
            checks,
            "required_artifact",
            _record_is_file(record),
            actual={"present": _record_is_file(record), "path": record.get("path") if isinstance(record, dict) else ""},
            expected={"role": role, "kind": "file"},
            summary=f"required_artifact[{role}]: {'present' if _record_is_file(record) else 'missing'}",
            scope={"role": role},
        )
        if isinstance(record, dict) and record.get("parse_error"):
            _add_check(
                checks,
                "artifact_parseable",
                False,
                actual={"error": record["parse_error"]},
                expected={"parseable": True},
                summary=f"artifact_parseable[{role}]: {record['parse_error']}",
                scope={"role": role},
            )


def _check_required_evals(checks: list[dict[str, Any]], evals: dict[str, dict[str, Any]], policy: dict[str, Any]) -> None:
    for arm in policy["required_evals"]:
        record = evals.get(arm)
        _add_check(
            checks,
            "required_eval",
            _record_is_file(record) and record.get("parse_error") is None if isinstance(record, dict) else False,
            actual={"present": _record_is_file(record), "parse_error": record.get("parse_error") if isinstance(record, dict) else None},
            expected={"arm": arm, "parseable": True},
            summary=f"required_eval[{arm}]: {'present' if _record_is_file(record) else 'missing'}",
            scope={"arm": arm},
        )


def _check_required_gates(checks: list[dict[str, Any]], gates: dict[str, dict[str, Any]], policy: dict[str, Any]) -> None:
    for gate_id in policy["required_gates"]:
        record = gates.get(gate_id)
        _add_check(
            checks,
            "required_gate",
            _record_is_file(record) and record.get("parse_error") is None if isinstance(record, dict) else False,
            actual={"present": _record_is_file(record), "parse_error": record.get("parse_error") if isinstance(record, dict) else None},
            expected={"gate": gate_id, "parseable": True},
            summary=f"required_gate[{gate_id}]: {'present' if _record_is_file(record) else 'missing'}",
            scope={"gate": gate_id},
        )


def _check_cards(checks: list[dict[str, Any]], artifacts: dict[str, dict[str, Any]], policy: dict[str, Any]) -> None:
    required_sections = policy.get("card_required_sections") if isinstance(policy.get("card_required_sections"), dict) else {}
    for role in ("dataset_card", "model_card"):
        if role not in policy["required_artifacts"]:
            continue
        record = artifacts.get(role)
        text = record.get("text") if isinstance(record, dict) and isinstance(record.get("text"), str) else ""
        for section in required_sections.get(role, []):
            _add_check(
                checks,
                "card_required_section",
                bool(text and section in text),
                actual={"found": bool(text and section in text)},
                expected={"section": section},
                summary=f"card_required_section[{role}]: {section}",
                scope={"role": role, "section": section},
            )


def _check_license(checks: list[dict[str, Any]], artifacts: dict[str, dict[str, Any]], policy: dict[str, Any]) -> None:
    record = artifacts.get("model_registry_entry")
    payload = record.get("json") if isinstance(record, dict) and isinstance(record.get("json"), dict) else {}
    status = _license_status(payload)
    approved = str(status).lower() in set(policy["approved_license_statuses"])
    _add_check(
        checks,
        "license_status",
        approved,
        actual={"license_status": status or "unknown"},
        expected={"approved_statuses": policy["approved_license_statuses"]},
        summary=f"license_status: {status or 'unknown'}",
        scope={"role": "model_registry_entry"},
    )


def _check_redaction(checks: list[dict[str, Any]], artifacts: dict[str, dict[str, Any]], policy: dict[str, Any]) -> None:
    if not policy.get("require_redaction_status"):
        return
    payloads = [
        artifacts.get("dataset_manifest", {}).get("json") if isinstance(artifacts.get("dataset_manifest"), dict) else None,
        artifacts.get("dataset_metrics", {}).get("json") if isinstance(artifacts.get("dataset_metrics"), dict) else None,
    ]
    status = next((item for payload in payloads for item in _redaction_statuses(payload)), None)
    quality_failures = [
        item
        for payload in payloads
        for item in _quality_flags(payload)
        if "redact" in str(item.get("id", "")).lower() or "secret" in str(item.get("id", "")).lower()
    ]
    passed = status in {"passed", "complete", "clean", "redacted"} and not quality_failures
    _add_check(
        checks,
        "redaction_status",
        passed,
        actual={
            "redaction_status": status or "missing",
            "redaction_quality_failure_count": len(quality_failures),
        },
        expected={"status": ["passed", "complete", "clean", "redacted"], "quality_failures": 0},
        summary=f"redaction_status: {status or 'missing'}",
        scope={"role": "dataset_manifest"},
    )


def _check_rollback(checks: list[dict[str, Any]], artifacts: dict[str, dict[str, Any]]) -> None:
    record = artifacts.get("rollback")
    payload = record.get("json") if isinstance(record, dict) and isinstance(record.get("json"), dict) else {}
    target = _rollback_target(payload)
    _add_check(
        checks,
        "rollback_target",
        bool(target),
        actual={"target": target or ""},
        expected={"target": "non-empty"},
        summary=f"rollback_target: {target or 'missing'}",
        scope={"role": "rollback"},
    )


def _check_evidence_bundle(checks: list[dict[str, Any]], artifacts: dict[str, dict[str, Any]], policy: dict[str, Any]) -> None:
    if not policy.get("require_passed_evidence_bundle"):
        return
    record = artifacts.get("evidence_bundle")
    payload = record.get("json") if isinstance(record, dict) and isinstance(record.get("json"), dict) else {}
    passed = payload.get("passed") is True
    _add_check(
        checks,
        "evidence_bundle_passed",
        passed,
        actual={"passed": payload.get("passed")},
        expected={"passed": True},
        summary=f"evidence_bundle_passed: {payload.get('passed')!r}",
        scope={"role": "evidence_bundle"},
    )


def _check_gate_results(checks: list[dict[str, Any]], gates: dict[str, dict[str, Any]], policy: dict[str, Any]) -> None:
    if not policy.get("require_passed_gates"):
        return
    for gate_id in policy["required_gates"]:
        record = gates.get(gate_id)
        payload = record.get("json") if isinstance(record, dict) and isinstance(record.get("json"), dict) else {}
        passed = payload.get("passed") is True
        _add_check(
            checks,
            "gate_passed",
            passed,
            actual={"passed": payload.get("passed")},
            expected={"passed": True},
            summary=f"gate_passed[{gate_id}]: {payload.get('passed')!r}",
            scope={"gate": gate_id},
        )


def _check_eval_policy(checks: list[dict[str, Any]], evals: dict[str, dict[str, Any]], policy: dict[str, Any]) -> None:
    candidate_arm = policy["candidate_arm"]
    candidate = evals.get(candidate_arm)
    candidate_summary = candidate.get("summary") if isinstance(candidate, dict) and isinstance(candidate.get("summary"), dict) else None
    if candidate_summary is None:
        return
    baseline_summaries = [
        (arm, evals[arm]["summary"])
        for arm in policy["baseline_arms"]
        if isinstance(evals.get(arm), dict) and isinstance(evals[arm].get("summary"), dict)
    ]

    if policy.get("require_identical_scenarios"):
        candidate_scenarios = candidate_summary["scenario_ids"]
        for arm, summary in baseline_summaries:
            _add_check(
                checks,
                "eval_scenarios_match",
                summary["scenario_ids"] == candidate_scenarios,
                actual={"scenario_ids": summary["scenario_ids"]},
                expected={"scenario_ids": candidate_scenarios},
                summary=f"eval_scenarios_match[{arm}]",
                scope={"arm": arm, "candidate_arm": candidate_arm},
            )

    min_pass_delta = float(policy.get("min_pass_rate_delta") or 0.0)
    min_score_delta = float(policy.get("min_average_score_delta") or 0.0)
    for arm, summary in baseline_summaries:
        _add_check(
            checks,
            "candidate_pass_rate",
            candidate_summary["pass_rate"] >= summary["pass_rate"] + min_pass_delta,
            actual={"candidate_pass_rate": candidate_summary["pass_rate"], "baseline_pass_rate": summary["pass_rate"]},
            expected={"min_delta": min_pass_delta},
            summary=f"candidate_pass_rate[{arm}]",
            scope={"arm": arm, "candidate_arm": candidate_arm},
        )
        _add_check(
            checks,
            "candidate_average_score",
            candidate_summary["average_score"] >= summary["average_score"] + min_score_delta,
            actual={"candidate_average_score": candidate_summary["average_score"], "baseline_average_score": summary["average_score"]},
            expected={"min_delta": min_score_delta},
            summary=f"candidate_average_score[{arm}]",
            scope={"arm": arm, "candidate_arm": candidate_arm},
        )
        _add_check(
            checks,
            "task_completion_regression",
            candidate_summary["passed"] >= summary["passed"],
            actual={"candidate_passed": candidate_summary["passed"], "baseline_passed": summary["passed"]},
            expected={"candidate_passed_min": summary["passed"]},
            summary=f"task_completion_regression[{arm}]",
            scope={"arm": arm, "candidate_arm": candidate_arm},
        )

    _add_check(
        checks,
        "candidate_error_count",
        candidate_summary["error_count"] <= int(policy.get("max_candidate_error_count") or 0),
        actual={"error_count": candidate_summary["error_count"]},
        expected={"max": int(policy.get("max_candidate_error_count") or 0)},
        summary=f"candidate_error_count: {candidate_summary['error_count']}",
        scope={"arm": candidate_arm},
    )

    new_critical = _new_critical_failures(candidate_summary, [summary for _, summary in baseline_summaries])
    _add_check(
        checks,
        "no_new_critical_failures",
        sum(new_critical.values()) <= int(policy.get("max_new_critical_failures") or 0),
        actual={"new_critical_failures": _count_rows(new_critical)},
        expected={"max": int(policy.get("max_new_critical_failures") or 0)},
        summary=f"no_new_critical_failures: {sum(new_critical.values())}",
        scope={"arm": candidate_arm},
    )
    candidate_critical = candidate_summary["critical_failure_counts"]
    forbidden = {
        rule: candidate_critical.get(rule, 0)
        for rule in policy["forbidden_candidate_critical_rules"]
        if candidate_critical.get(rule, 0) > 0
    }
    _add_check(
        checks,
        "forbidden_candidate_critical_rules",
        not forbidden,
        actual={"critical_failures": _count_rows(forbidden)},
        expected={"forbidden": policy["forbidden_candidate_critical_rules"], "count": 0},
        summary=f"forbidden_candidate_critical_rules: {sum(forbidden.values())}",
        scope={"arm": candidate_arm},
    )


def _decision_metrics(
    artifacts: dict[str, dict[str, Any]],
    evals: dict[str, dict[str, Any]],
    gates: dict[str, dict[str, Any]],
    policy: dict[str, Any],
    failed_checks: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate = evals.get(policy["candidate_arm"])
    candidate_summary = candidate.get("summary") if isinstance(candidate, dict) and isinstance(candidate.get("summary"), dict) else {}
    baseline_summaries = [
        evals[arm]["summary"]
        for arm in policy["baseline_arms"]
        if isinstance(evals.get(arm), dict) and isinstance(evals[arm].get("summary"), dict)
    ]
    new_critical = _new_critical_failures(candidate_summary, baseline_summaries) if candidate_summary else {}
    return {
        "candidate_arm": policy["candidate_arm"],
        "baseline_arms": list(policy["baseline_arms"]),
        "required_artifact_count": len(policy["required_artifacts"]),
        "present_required_artifact_count": sum(1 for role in policy["required_artifacts"] if _record_is_file(artifacts.get(role))),
        "required_eval_count": len(policy["required_evals"]),
        "present_required_eval_count": sum(1 for arm in policy["required_evals"] if _record_is_file(evals.get(arm))),
        "required_gate_count": len(policy["required_gates"]),
        "passed_required_gate_count": sum(
            1
            for gate_id in policy["required_gates"]
            if isinstance(gates.get(gate_id), dict)
            and isinstance(gates[gate_id].get("json"), dict)
            and gates[gate_id]["json"].get("passed") is True
        ),
        "candidate_pass_rate": candidate_summary.get("pass_rate"),
        "candidate_average_score": candidate_summary.get("average_score"),
        "candidate_error_count": candidate_summary.get("error_count"),
        "scenario_count": len(candidate_summary.get("scenario_ids", [])) if isinstance(candidate_summary.get("scenario_ids"), list) else 0,
        "new_critical_failure_count": sum(new_critical.values()),
        "new_critical_failure_counts": _count_rows(new_critical),
        "failed_check_count": len(failed_checks),
    }


def _decision_summary(passed: bool, failed_checks: list[dict[str, Any]], metrics: dict[str, Any]) -> str:
    if passed:
        return (
            "Promotion allowed: all required governance artifacts, eval comparisons, gates, "
            "cards, license, redaction, and rollback checks passed."
        )
    return (
        f"Promotion blocked: {len(failed_checks)} governance check(s) failed; "
        f"candidate_pass_rate={metrics.get('candidate_pass_rate')!r}; "
        f"new_critical_failure_count={metrics.get('new_critical_failure_count')!r}."
    )


def _policy_summary(policy: dict[str, Any], policy_path: str | Path | None, preserve_paths: bool) -> dict[str, Any]:
    summary = {
        "schema_version": PROMOTION_POLICY_SCHEMA_VERSION,
        "path": _display_path(Path(policy_path), preserve_paths) if policy_path is not None else "",
        "effective": {
            key: value
            for key, value in policy.items()
            if key not in {"schema_version", "description"}
        },
    }
    if isinstance(policy.get("description"), str):
        summary["description"] = policy["description"]
    return summary


def _artifact_record(role: str, path: Path, preserve_paths: bool) -> dict[str, Any]:
    record = _base_record(role, path, preserve_paths)
    if path.exists() and path.is_file() and not path.is_symlink():
        if path.suffix.lower() in {".md", ".markdown", ".txt"}:
            record["text"] = path.read_text(encoding="utf-8")
        else:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                record["parse_error"] = str(exc)
            else:
                if isinstance(payload, dict):
                    record["json"] = payload
                    record["schema_version"] = payload.get("schema_version")
                    if isinstance(payload.get("passed"), bool):
                        record["passed"] = payload["passed"]
                else:
                    record["parse_error"] = "artifact JSON must contain an object"
    return record


def _eval_record(arm: str, path: Path, preserve_paths: bool) -> dict[str, Any]:
    record = _base_record(arm, path, preserve_paths)
    if path.exists() and path.is_file() and not path.is_symlink():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            record["parse_error"] = str(exc)
        else:
            if not isinstance(payload, dict):
                record["parse_error"] = "eval summary JSON must contain an object"
            else:
                record["json"] = payload
                record["schema_version"] = payload.get("schema_version")
                record["summary"] = _eval_summary(payload)
    return record


def _gate_record(gate_id: str, path: Path, preserve_paths: bool) -> dict[str, Any]:
    record = _base_record(gate_id, path, preserve_paths)
    if path.exists() and path.is_file() and not path.is_symlink():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            record["parse_error"] = str(exc)
        else:
            if not isinstance(payload, dict):
                record["parse_error"] = "gate JSON must contain an object"
            else:
                record["json"] = payload
                record["schema_version"] = payload.get("schema_version")
                record["passed"] = payload.get("passed") if isinstance(payload.get("passed"), bool) else None
    return record


def _promotion_decision_record(path: Path, payload: dict[str, Any], preserve_paths: bool) -> dict[str, Any]:
    record = _base_record("promotion_decision", path, preserve_paths)
    record["schema_version"] = payload.get("schema_version")
    record["passed"] = payload.get("passed") if isinstance(payload.get("passed"), bool) else None
    record["readiness"] = payload.get("readiness") if isinstance(payload.get("readiness"), str) else ""
    record["recommendation"] = payload.get("recommendation") if isinstance(payload.get("recommendation"), str) else ""
    record["failed_check_count"] = payload.get("failed_check_count")
    decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
    record["blocking_check_count"] = decision.get("blocking_check_count")
    return record


def _base_record(name: str, path: Path, preserve_paths: bool) -> dict[str, Any]:
    record: dict[str, Any] = {
        "name": name,
        "path": _display_path(path, preserve_paths),
        "exists": path.exists(),
        "kind": "missing",
    }
    if path.exists():
        if path.is_symlink():
            record["kind"] = "file" if path.is_file() else "other"
            record["symlink"] = True
            record["regular_file"] = False
        elif path.is_file():
            record["kind"] = "file"
            record["symlink"] = False
            record["regular_file"] = True
            record["size_bytes"] = path.stat().st_size
            record["sha256"] = _sha256(path)
        elif path.is_dir():
            record["kind"] = "directory"
            record["symlink"] = False
            record["regular_directory"] = True
            record["entry_count"] = len(list(path.iterdir()))
        else:
            record["kind"] = "other"
    return record


def _promotion_decision_target_hints(promotion_decision: dict[str, Any]) -> set[str]:
    hints: set[str] = set()
    metadata = promotion_decision.get("metadata") if isinstance(promotion_decision.get("metadata"), dict) else {}
    for field in ("target_entry_id", "candidate_entry_id", "model_registry_entry_id", "candidate_id", "model_id"):
        value = metadata.get(field)
        if isinstance(value, str) and value:
            hints.add(value)
    artifacts = promotion_decision.get("artifacts") if isinstance(promotion_decision.get("artifacts"), dict) else {}
    registry_record = artifacts.get("model_registry_entry") if isinstance(artifacts.get("model_registry_entry"), dict) else {}
    payload = registry_record.get("json") if isinstance(registry_record.get("json"), dict) else {}
    for field in ("entry_id", "candidate_id", "model_id"):
        value = payload.get(field)
        if isinstance(value, str) and value:
            hints.add(value)
    return hints


def _planned_alias_history(
    alias: str,
    previous_target: Any,
    target: str,
    rollback_target: str | None,
    rollback_previous: Any,
    reason: str,
    planned_at: str,
) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    if alias == "champion" and rollback_target:
        history.append(
            {
                "alias": "rollback",
                "previous": rollback_previous,
                "target": rollback_target,
                "moved_at": planned_at,
                "reason": reason or "set rollback before champion move",
            }
        )
    history.append(
        {
            "alias": alias,
            "previous": previous_target,
            "target": target,
            "moved_at": planned_at,
            "reason": reason,
        }
    )
    return history


def _alias_receipt_summary(passed: bool, failed_checks: list[dict[str, Any]], alias: str, target: str) -> str:
    if passed:
        return f"Registry alias receipt ready: {alias} can move to {target} after revalidation."
    return f"Registry alias receipt blocked: {len(failed_checks)} check(s) failed for {alias} -> {target}."


def _eval_summary(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    scenario_ids = payload.get("scenario_ids")
    if not isinstance(scenario_ids, list):
        runs = payload.get("runs") if isinstance(payload.get("runs"), list) else []
        scenario_ids = [run.get("scenario_id") for run in runs if isinstance(run, dict)]
    critical_counts = _count_map(metrics.get("critical_failure_counts"))
    return {
        "arm": _normalize_key(str(payload.get("arm") or payload.get("metadata", {}).get("arm") or "")),
        "model": str(payload.get("model") or payload.get("metadata", {}).get("model") or ""),
        "scenario_ids": sorted(str(item) for item in scenario_ids if isinstance(item, str) and item),
        "total": _int_value(payload.get("total", metrics.get("total"))),
        "passed": _int_value(payload.get("passed", metrics.get("passed"))),
        "failed": _int_value(payload.get("failed", metrics.get("failed"))),
        "error_count": _int_value(payload.get("error_count")),
        "pass_rate": _float_value(metrics.get("pass_rate")),
        "average_score": _float_value(metrics.get("average_score")),
        "critical_failure_counts": critical_counts,
    }


def _new_critical_failures(candidate: dict[str, Any], baselines: list[dict[str, Any]]) -> dict[str, int]:
    candidate_counts = candidate.get("critical_failure_counts")
    if not isinstance(candidate_counts, dict):
        return {}
    if not baselines:
        return dict(candidate_counts)
    result: dict[str, int] = {}
    for rule, count in candidate_counts.items():
        baseline_min = min(
            int(baseline.get("critical_failure_counts", {}).get(rule, 0))
            for baseline in baselines
            if isinstance(baseline.get("critical_failure_counts"), dict)
        )
        delta = int(count) - baseline_min
        if delta > 0:
            result[rule] = delta
    return result


def _license_status(payload: dict[str, Any]) -> str:
    for path in (
        ("license_status",),
        ("license", "status"),
        ("license_review", "status"),
        ("model", "license_status"),
        ("rights", "license_status"),
        ("governance", "license_status"),
    ):
        value = _nested(payload, path)
        if isinstance(value, str) and value:
            return value.lower()
    return "unknown"


def _redaction_statuses(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    statuses: list[str] = []
    for path in (
        ("redaction_status",),
        ("redaction", "status"),
        ("redaction", "result"),
        ("privacy", "redaction_status"),
        ("dataset", "redaction_status"),
    ):
        value = _nested(payload, path)
        if isinstance(value, str) and value:
            statuses.append(value.lower())
    return statuses


def _quality_flags(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    flags = payload.get("quality_flags")
    if not isinstance(flags, list):
        return []
    return [flag for flag in flags if isinstance(flag, dict) and str(flag.get("severity", "")).lower() in {"error", "critical"}]


def _rollback_target(payload: dict[str, Any]) -> str:
    for path in (
        ("target_model_id",),
        ("rollback_target",),
        ("model_id",),
        ("alias",),
        ("target", "model_id"),
        ("target", "alias"),
        ("target", "id"),
    ):
        value = _nested(payload, path)
        if isinstance(value, str) and value:
            return value
    return ""


def _nested(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _record_is_file(record: dict[str, Any] | None) -> bool:
    return isinstance(record, dict) and record.get("exists") is True and record.get("kind") == "file" and record.get("regular_file") is True


def _add_check(
    checks: list[dict[str, Any]],
    check_id: str,
    passed: bool,
    *,
    actual: dict[str, Any],
    expected: dict[str, Any],
    summary: str,
    scope: dict[str, Any],
) -> None:
    checks.append(
        {
            "id": check_id,
            "passed": bool(passed),
            "actual": actual,
            "expected": expected,
            "scope": scope,
            "summary": summary,
        }
    )


def _count_map(rows: Any) -> dict[str, int]:
    if isinstance(rows, dict):
        return {str(key): _int_value(value) for key, value in rows.items() if isinstance(key, str)}
    if not isinstance(rows, list):
        return {}
    counts: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_id = row.get("id")
        if isinstance(row_id, str) and row_id:
            counts[row_id] = _int_value(row.get("count"))
    return counts


def _count_rows(counts: dict[str, int]) -> list[dict[str, Any]]:
    return [{"id": key, "count": counts[key]} for key in sorted(counts)]


def _policy_string_list(field: str, value: Any) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise PromotionPolicyError(f"promotion policy field {field} must be a non-empty list of strings")
    return [_normalize_key(item) if field in {"required_artifacts", "required_evals", "required_gates", "baseline_arms"} else item for item in value]


def _policy_non_negative_int(field: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise PromotionPolicyError(f"promotion policy field {field} must be a non-negative integer")
    return value


def _policy_non_negative_number(field: str, value: Any) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) < 0:
        raise PromotionPolicyError(f"promotion policy field {field} must be a non-negative number")
    return float(value)


def _policy_card_sections(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        raise PromotionPolicyError("promotion policy field card_required_sections must be an object")
    sections: dict[str, list[str]] = {}
    for role, role_sections in value.items():
        if not isinstance(role, str) or not role:
            raise PromotionPolicyError("promotion policy card_required_sections keys must be non-empty strings")
        sections[_normalize_key(role)] = _policy_string_list(f"card_required_sections.{role}", role_sections)
    return sections


def _normalize_key(value: str) -> str:
    return value.strip().replace("-", "_").lower()


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_value(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return 0.0


def _display_path(path: Path, preserve_paths: bool = False) -> str:
    raw = str(path)
    if preserve_paths or not path.is_absolute():
        return raw
    return f"<redacted:{path.name}>"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _payload_file_sha256(payload: dict[str, Any]) -> str:
    rendered = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
