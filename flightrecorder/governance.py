"""Top-level governance decisions for promotion and alias movement."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from .bundle import EVIDENCE_BUNDLE_SCHEMA_VERSION
from .compare_gate import COMPARE_GATE_SCHEMA_VERSION
from .preflight import TRAINER_LAUNCH_CHECK_SCHEMA_VERSION
from .promotion_gate import PROMOTION_LEDGER_GATE_SCHEMA_VERSION

PROMOTION_DECISION_SCHEMA_VERSION = "hfr.promotion_decision.v1"
PROMOTION_CARDS_SCHEMA_VERSION = "hfr.promotion_cards.v1"
PROMOTION_ALIAS_APPLY_SCHEMA_VERSION = "hfr.promotion_alias_apply.v1"
MODEL_REGISTRY_SCHEMA_VERSION = "hfr.model_registry.v1"

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
    "model_card",
    "dataset_card",
    "rollback_metadata",
    "license_review",
    "redaction_check",
    "safety_gate",
    "serving_report",
)
_JSON_ARTIFACT_ROLES = {
    "evidence_bundle": EVIDENCE_BUNDLE_SCHEMA_VERSION,
    "promotion_ledger_gate": PROMOTION_LEDGER_GATE_SCHEMA_VERSION,
    "compare_gate": COMPARE_GATE_SCHEMA_VERSION,
    "trainer_launch_check": TRAINER_LAUNCH_CHECK_SCHEMA_VERSION,
}
_PASSED_JSON_ROLES = (
    "evidence_bundle",
    "promotion_ledger_gate",
    "compare_gate",
    "trainer_launch_check",
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
    model_card_path: str | Path | None = None,
    dataset_card_path: str | Path | None = None,
    rollback_metadata_path: str | Path | None = None,
    license_review_path: str | Path | None = None,
    redaction_check_path: str | Path | None = None,
    safety_gate_path: str | Path | None = None,
    serving_report_path: str | Path | None = None,
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
        "model_card": model_card_path,
        "dataset_card": dataset_card_path,
        "rollback_metadata": rollback_metadata_path,
        "license_review": license_review_path,
        "redaction_check": redaction_check_path,
        "safety_gate": safety_gate_path,
        "serving_report": serving_report_path,
    }
    artifacts = {
        role: _artifact_record(role, Path(raw_path) if raw_path else None, preserve_paths)
        for role, raw_path in paths.items()
    }
    json_artifacts = {
        role: _read_json_artifact(Path(raw_path)) if raw_path else None
        for role, raw_path in paths.items()
        if role in _PASSED_JSON_ROLES or role == "rollback_metadata"
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

    _add_schema_check(checks, "evidence_bundle", json_artifacts.get("evidence_bundle"))
    _add_schema_check(checks, "promotion_ledger_gate", json_artifacts.get("promotion_ledger_gate"))
    _add_schema_check(checks, "compare_gate", json_artifacts.get("compare_gate"))
    _add_schema_check(checks, "trainer_launch_check", json_artifacts.get("trainer_launch_check"))
    for role in _PASSED_JSON_ROLES:
        _add_passed_json_check(checks, role, json_artifacts.get(role))

    compare_metrics = _metrics_object(json_artifacts.get("compare_gate"))
    _add_max_count_check(checks, "task_completion_regressions_absent", compare_metrics.get("task_completion_regression_count"), 0)
    _add_max_count_check(checks, "baseline_wins_absent", compare_metrics.get("baseline_win_count"), 0)
    _add_max_count_check(checks, "contract_drifts_absent", compare_metrics.get("contract_drift_count"), 0)
    _add_max_count_check(checks, "unverified_contracts_absent", compare_metrics.get("unverified_contract_count"), 0)
    _add_count_map_absence_check(
        checks,
        "new_critical_failures_absent",
        compare_metrics.get("new_critical_failure_counts"),
        "new critical failures",
    )
    _add_count_map_absence_check(
        checks,
        "rule_regressions_absent",
        compare_metrics.get("regressed_rule_counts"),
        "rule regressions",
    )
    _add_forbidden_rule_check(checks, compare_metrics.get("new_critical_failure_counts"), "new_critical", "forbidden_actions")
    _add_forbidden_rule_check(checks, compare_metrics.get("new_critical_failure_counts"), "new_critical", "secret_exposure")
    _add_forbidden_rule_check(checks, compare_metrics.get("regressed_rule_counts"), "regression", "forbidden_actions")
    _add_forbidden_rule_check(checks, compare_metrics.get("regressed_rule_counts"), "regression", "secret_exposure")

    _add_license_check(checks, json_artifacts.get("license_review"))
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
    metrics = _decision_metrics(checks, compare_metrics)
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
    registry_before_record = _artifact_record("registry", registry_file, preserve_paths)
    decision_record = _artifact_record("promotion_decision", decision_file, preserve_paths)
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
        registry_after_record = _artifact_record("registry", registry_file, preserve_paths)
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
        role: _artifact_record(role, Path(raw_path) if raw_path else None, preserve_paths)
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
        "model_card": _artifact_record("model_card", model_card_path, preserve_paths),
        "dataset_card": _artifact_record("dataset_card", dataset_card_path, preserve_paths),
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


def _artifact_record(role: str, path: Path | None, preserve_paths: bool) -> dict[str, Any]:
    record: dict[str, Any] = {
        "role": role,
        "path": _display_path(path, preserve_paths) if path is not None else "",
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
    passed = status not in {"", "unknown", "unreviewed", "missing"} and accepted_terms is not False
    _add_check(
        checks,
        "license_status_known",
        passed,
        actual={"license_status": status or "missing", "accepted_terms": accepted_terms},
        expected={"license_status": "known", "accepted_terms": "not false"},
        scope={"artifact_role": "license_review"},
        summary="license review has a known status and does not reject terms",
    )


def _add_rollback_metadata_check(checks: list[dict[str, Any]], payload: dict[str, Any] | None, rollback_id: str | None) -> None:
    actual_id = ""
    available = None
    if isinstance(payload, dict):
        actual_id = str(payload.get("rollback_id") or payload.get("target_model_id") or "")
        available = payload.get("available")
    _add_check(
        checks,
        "rollback_metadata_matches_target",
        bool(rollback_id) and actual_id == rollback_id and available is not False,
        actual={"rollback_id": actual_id, "available": available},
        expected={"rollback_id": rollback_id or "", "available": "not false"},
        scope={"artifact_role": "rollback_metadata"},
        summary="rollback metadata points at the declared rollback target",
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
    counts = _count_map(value)
    total = sum(counts.values())
    _add_check(
        checks,
        check_id,
        total == 0,
        actual={"total": total, "counts": counts},
        expected={"total": 0},
        summary=f"no {label} are present",
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


def _decision_metrics(checks: list[dict[str, Any]], compare_metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "check_count": len(checks),
        "failed_check_count": sum(1 for check in checks if not check["passed"]),
        "required_artifact_count": len(PROMOTION_DECISION_REQUIRED_ARTIFACTS),
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
        return {"passed": None, "target_count": 0, "error_count": 0, "warning_count": 0}
    return {
        "passed": value.get("passed") if isinstance(value.get("passed"), bool) else None,
        "target_count": _int_value(value.get("target_count")),
        "error_count": _int_value(value.get("error_count")),
        "warning_count": _int_value(value.get("warning_count")),
    }


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


def _is_windows_absolute(value: str) -> bool:
    normalized = value.replace("/", "\\")
    return (len(normalized) >= 3 and normalized[1:3] == ":\\" and normalized[0].isalpha()) or normalized.startswith("\\\\")


def _basename(value: str) -> str:
    return value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] or "path"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
