"""Trainer preflight manifests over passed Flight Recorder gates."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
from pathlib import Path
from typing import Any

from .schema_registry import SchemaRegistryError, check_schema_file, check_schema_jsonl_file
from .training import DATASET_SPLIT_ARTIFACTS, DATASET_SPLIT_NAMES

TRAINER_PREFLIGHT_SCHEMA_VERSION = "hfr.trainer_preflight.v1"
TRAINER_LAUNCH_CHECK_SCHEMA_VERSION = "hfr.trainer_launch_check.v1"

_TRAINING_EXPORT_BASE_FILES = (
    "manifest.json",
    "dataset_registry.json",
    "dataset_metrics.json",
    "dataset_splits.json",
    "DATASET_CARD.md",
    "episodes.jsonl",
    "rewards.jsonl",
    "step_rewards.jsonl",
    "preferences.jsonl",
    "failure_modes.jsonl",
    "curriculum.json",
    "sft.jsonl",
    "dpo.jsonl",
    "reward_model.jsonl",
)
_TRAINING_EXPORT_FILES = _TRAINING_EXPORT_BASE_FILES + tuple(
    f"splits/{split_name}/{artifact_name}.jsonl"
    for split_name in DATASET_SPLIT_NAMES
    for artifact_name in DATASET_SPLIT_ARTIFACTS
)
_TRAINING_ROW_SCHEMA_NAMES = {
    "episodes": "rl_episode",
    "rewards": "rl_reward",
    "step_rewards": "rl_step_reward",
    "preferences": "rl_preference",
    "failure_modes": "rl_failure_mode",
    "sft": "rl_sft",
    "dpo": "rl_dpo",
    "reward_model": "rl_reward_model",
}
_TRAINING_SCHEMA_CONTRACTS = (
    ("manifest.json", "training_manifest", False),
    ("dataset_registry.json", "dataset_registry", False),
    ("dataset_metrics.json", "rl_dataset_metrics", False),
    ("curriculum.json", "rl_curriculum", False),
    ("dataset_splits.json", "dataset_splits", False),
    ("episodes.jsonl", "rl_episode", True),
    ("rewards.jsonl", "rl_reward", True),
    ("step_rewards.jsonl", "rl_step_reward", True),
    ("preferences.jsonl", "rl_preference", True),
    ("failure_modes.jsonl", "rl_failure_mode", True),
    ("sft.jsonl", "rl_sft", True),
    ("dpo.jsonl", "rl_dpo", True),
    ("reward_model.jsonl", "rl_reward_model", True),
) + tuple(
    (f"splits/{split_name}/{artifact_name}.jsonl", _TRAINING_ROW_SCHEMA_NAMES[artifact_name], True)
    for split_name in DATASET_SPLIT_NAMES
    for artifact_name in DATASET_SPLIT_ARTIFACTS
)
_COMPARE_EXPORT_FILES = (
    "manifest.json",
    "IMPROVEMENT_CARD.md",
    "improvement_pairs.jsonl",
    "improvement_dpo.jsonl",
)
_COMPARE_SCHEMA_CONTRACTS = (
    ("manifest.json", "compare_rl_manifest", False),
    ("improvement_pairs.jsonl", "compare_rl_pair", True),
    ("improvement_dpo.jsonl", "compare_rl_dpo", True),
)
_REVIEWED_EXPORT_FILES = (
    "manifest.json",
    "dataset_registry.json",
    "reviewed_labels.jsonl",
    "reviewed_sft.jsonl",
    "reviewed_reward_model.jsonl",
    "reviewed_preferences.jsonl",
    "reviewed_dpo.jsonl",
)
_REVIEWED_SCHEMA_CONTRACTS = (
    ("manifest.json", "reviewed_manifest", False),
    ("dataset_registry.json", "dataset_registry", False),
)
_VALIDATION_REQUIRED_GATE_SCHEMAS = {
    "hfr.training_gate.v1",
    "hfr.compare_gate.v1",
    "hfr.improvement_ledger_gate.v1",
    "hfr.reviewed_gate.v1",
    "hfr.review_calibration.v1",
}


class TrainerPreflightError(ValueError):
    """Raised when a trainer preflight manifest cannot be built."""


def build_trainer_preflight(
    *,
    out_path: str | Path,
    gate_paths: list[str | Path],
    training_export_dir: str | Path | None = None,
    compare_export_dir: str | Path | None = None,
    reviewed_export_dir: str | Path | None = None,
    evidence_bundle_path: str | Path | None = None,
    agentic_training_plan_path: str | Path | None = None,
    validation_summary_paths: list[str | Path] | None = None,
    require_gates: list[str] | None = None,
    required_dataset_versions: list[str] | None = None,
    trainer_command: str | None = None,
    allow_unvalidated_gates: bool = False,
    preserve_paths: bool = False,
    metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a launch guard over trainer-facing exports and gate outputs."""
    if not gate_paths:
        raise TrainerPreflightError("At least one --gate path is required.")
    if not any((training_export_dir, compare_export_dir, reviewed_export_dir, evidence_bundle_path, agentic_training_plan_path)):
        raise TrainerPreflightError(
            "At least one trainer-facing artifact is required: --training-export, --compare-export, "
            "--reviewed-export, --evidence-bundle, or --agentic-training-plan."
        )

    checks: list[dict[str, Any]] = []
    required_dataset_versions = list(dict.fromkeys(required_dataset_versions or []))
    output_path = Path(out_path)
    validation_summaries, validation_targets = _validation_summary_records(validation_summary_paths or [], preserve_paths, output_path)
    gates = [_gate_record(Path(path), preserve_paths, validation_targets, output_path) for path in gate_paths]
    seen_gate_ids = {gate["id"] for gate in gates}
    for gate in gates:
        _add_bool_check(checks, "gate_passed", gate["passed"], {"gate": gate["id"], "path": gate["path"]})
        if _gate_requires_validation(gate) and not allow_unvalidated_gates:
            validation = gate.get("validation") if isinstance(gate.get("validation"), dict) else {}
            _add_bool_check(
                checks,
                "gate_validation_passed",
                bool(validation.get("available") and validation.get("passed")),
                {
                    "gate": gate["id"],
                    "validation_available": str(bool(validation.get("available"))).lower(),
                    "validation_error_count": str(_int_value(validation.get("error_count"))),
                },
            )

    for gate_id in require_gates or []:
        _add_bool_check(checks, "required_gate_present", gate_id in seen_gate_ids, {"gate": gate_id})

    artifacts: dict[str, Any] = {}
    schema_contracts: dict[str, Any] = {}
    dataset_selection: list[dict[str, Any]] = []
    if training_export_dir is not None:
        training_root = Path(training_export_dir)
        _add_export_artifacts(artifacts, checks, "training_export", training_root, _TRAINING_EXPORT_FILES, preserve_paths, output_path)
        _add_schema_contracts(schema_contracts, checks, "training_export", training_root, _TRAINING_SCHEMA_CONTRACTS, preserve_paths, output_path)
        _add_dataset_selection(
            dataset_selection,
            checks,
            "training_export",
            training_root,
            preserve_paths,
            output_path,
            required_dataset_versions,
        )
    if compare_export_dir is not None:
        compare_root = Path(compare_export_dir)
        _add_export_artifacts(artifacts, checks, "compare_export", compare_root, _COMPARE_EXPORT_FILES, preserve_paths, output_path)
        _add_schema_contracts(schema_contracts, checks, "compare_export", compare_root, _COMPARE_SCHEMA_CONTRACTS, preserve_paths, output_path)
    if reviewed_export_dir is not None:
        reviewed_root = Path(reviewed_export_dir)
        _add_export_artifacts(artifacts, checks, "reviewed_export", reviewed_root, _REVIEWED_EXPORT_FILES, preserve_paths, output_path)
        _add_schema_contracts(schema_contracts, checks, "reviewed_export", reviewed_root, _REVIEWED_SCHEMA_CONTRACTS, preserve_paths, output_path)
        _add_dataset_selection(
            dataset_selection,
            checks,
            "reviewed_export",
            reviewed_root,
            preserve_paths,
            output_path,
            required_dataset_versions,
        )
    if evidence_bundle_path is not None:
        bundle_path = Path(evidence_bundle_path)
        artifacts["evidence_bundle"] = _file_record(bundle_path, preserve_paths, output_path)
        _add_schema_contract(schema_contracts, checks, "evidence_bundle", bundle_path, "evidence_bundle", False, preserve_paths, output_path)
        bundle = _read_json_optional(bundle_path)
        _add_bool_check(checks, "evidence_bundle_exists", bundle_path.exists() and bundle_path.is_file(), {"artifact": "evidence_bundle"})
        _add_bool_check(
            checks,
            "evidence_bundle_passed",
            bool(isinstance(bundle, dict) and bundle.get("passed") is True),
            {"artifact": "evidence_bundle"},
        )
    if agentic_training_plan_path is not None:
        plan_path = Path(agentic_training_plan_path)
        artifacts["agentic_training_plan"] = _file_record(plan_path, preserve_paths, output_path)
        _add_schema_contract(schema_contracts, checks, "agentic_training_plan", plan_path, "agentic_training_plan", False, preserve_paths, output_path)
        plan = _read_json_optional(plan_path)
        _add_bool_check(checks, "agentic_training_plan_exists", plan_path.exists() and plan_path.is_file(), {"artifact": "agentic_training_plan"})
        _add_bool_check(
            checks,
            "agentic_training_plan_ready",
            bool(isinstance(plan, dict) and plan.get("passed") is True and plan.get("recommendation") == "ready_for_external_trainer_plan"),
            {"artifact": "agentic_training_plan"},
        )

    command = _trainer_command_record(trainer_command)
    failed_checks = sum(1 for check in checks if check.get("passed") is False)
    passed = failed_checks == 0
    preflight: dict[str, Any] = {
        "schema_version": TRAINER_PREFLIGHT_SCHEMA_VERSION,
        "preflight_path": _display_path(Path(out_path), preserve_paths),
        "passed": passed,
        "readiness": "ready" if passed else "blocked",
        "recommendation": "launch_allowed" if passed else "block_launch",
        "gate_count": len(gates),
        "passed_gate_count": sum(1 for gate in gates if gate.get("passed") is True),
        "required_gates": list(dict.fromkeys(require_gates or [])),
        "required_dataset_versions": required_dataset_versions,
        "check_count": len(checks),
        "failed_check_count": failed_checks,
        "checks": checks,
        "gates": gates,
        "validation_summaries": validation_summaries,
        "schema_contracts": schema_contracts,
        "artifacts": artifacts,
        "dataset_selection": dataset_selection,
        "trainer_command": command,
        "notes": [
            "Trainer preflight records evidence for a downstream launch guard; it does not train, sandbox, or execute commands.",
            "A ready preflight means the referenced gates and artifacts passed the configured handoff checks.",
            "Use --require-dataset-version to bind launch readiness to an exact manifest dataset_version.",
        ],
    }
    clean_metadata = _metadata(metadata)
    if clean_metadata:
        preflight["metadata"] = clean_metadata
    return preflight


def build_trainer_launch_check(
    *,
    preflight_path: str | Path,
    preflight: dict[str, Any],
    validation_summary: dict[str, Any],
    require_gates: list[str] | None = None,
    required_dataset_versions: list[str] | None = None,
    require_metadata: dict[str, str] | None = None,
    preserve_paths: bool = False,
) -> dict[str, Any]:
    """Build a side-effect-free trainer launch decision from a preflight artifact."""
    if not isinstance(preflight, dict):
        raise TrainerPreflightError(f"trainer preflight must contain a JSON object: {preflight_path}")

    checks: list[dict[str, Any]] = []
    display_preflight_path = _display_path(Path(preflight_path), preserve_paths)
    validation = _validation_record(validation_summary)
    gates = _launch_gate_records(preflight.get("gates"))
    metadata = _metadata(preflight.get("metadata") if isinstance(preflight.get("metadata"), dict) else None)
    required_gates = list(dict.fromkeys(require_gates or []))
    preflight_required_dataset_versions = (
        preflight.get("required_dataset_versions") if isinstance(preflight.get("required_dataset_versions"), list) else []
    )
    required_dataset_versions = list(
        dict.fromkeys(
            [
                str(version)
                for version in [*(required_dataset_versions or []), *preflight_required_dataset_versions]
                if isinstance(version, str) and version
            ]
        )
    )
    required_metadata = _metadata(require_metadata)
    command = _approved_command_record(preflight.get("trainer_command"))
    dataset_selection = preflight.get("dataset_selection") if isinstance(preflight.get("dataset_selection"), list) else []

    _add_bool_check(checks, "preflight_validation_passed", validation["passed"], {"preflight": display_preflight_path})
    _add_bool_check(
        checks,
        "preflight_schema_supported",
        preflight.get("schema_version") == TRAINER_PREFLIGHT_SCHEMA_VERSION,
        {"schema_version": str(preflight.get("schema_version") or "")},
    )
    _add_bool_check(checks, "preflight_passed", preflight.get("passed") is True, {"preflight": display_preflight_path})
    _add_bool_check(
        checks,
        "recommendation_launch_allowed",
        preflight.get("recommendation") == "launch_allowed",
        {"recommendation": str(preflight.get("recommendation") or "")},
    )
    _add_bool_check(
        checks,
        "trainer_command_ready",
        bool(command["provided"] and command["parseable"] and command["argv"]),
        {"provided": str(command["provided"]).lower(), "parseable": str(command["parseable"]).lower()},
    )
    for gate_id in required_gates:
        matching_gate = next((gate for gate in gates if gate["id"] == gate_id), None)
        _add_bool_check(checks, "required_gate_present", matching_gate is not None, {"gate": gate_id})
        _add_bool_check(
            checks,
            "required_gate_passed",
            bool(matching_gate and matching_gate.get("passed") is True),
            {"gate": gate_id},
        )
    for key, expected in required_metadata.items():
        actual = metadata.get(key)
        _add_bool_check(
            checks,
            "required_metadata_matches",
            actual == expected,
            {"key": key, "expected": expected, "actual": "" if actual is None else actual},
        )
    for version in required_dataset_versions:
        matching = [
            item
            for item in dataset_selection
            if isinstance(item, dict) and item.get("dataset_version") == version and item.get("matches_required") is True
        ]
        _add_bool_check(
            checks,
            "required_dataset_version_selected",
            bool(matching),
            {"dataset_version": version},
        )

    failed_checks = sum(1 for check in checks if check.get("passed") is False)
    passed = failed_checks == 0
    command["approved"] = passed
    artifacts = preflight.get("artifacts") if isinstance(preflight.get("artifacts"), dict) else {}
    return {
        "schema_version": TRAINER_LAUNCH_CHECK_SCHEMA_VERSION,
        "preflight_path": display_preflight_path,
        "passed": passed,
        "readiness": "ready" if passed else "blocked",
        "recommendation": "launch_allowed" if passed else "block_launch",
        "check_count": len(checks),
        "failed_check_count": failed_checks,
        "checks": checks,
        "validation": validation,
        "required_gates": required_gates,
        "required_dataset_versions": required_dataset_versions,
        "required_metadata": required_metadata,
        "gates": gates,
        "gate_count": len(gates),
        "passed_gate_count": sum(1 for gate in gates if gate.get("passed") is True),
        "artifacts": {"count": len(artifacts), "names": sorted(str(name) for name in artifacts.keys())},
        "dataset_selection": dataset_selection,
        "approved_command": command,
        "notes": [
            "Trainer launch check validates the preflight and emits the approved command; it does not execute it.",
            "A ready launch check means the preflight artifact, referenced hashes, required gates, and required metadata passed.",
            "Dataset selections are inherited from the trainer preflight and can be enforced again with --require-dataset-version.",
        ],
    }


def _gate_record(path: Path, preserve_paths: bool, validation_targets: dict[str, dict[str, Any]], output_path: Path) -> dict[str, Any]:
    gate = _read_json_required(path, "gate")
    schema_version = gate.get("schema_version") if isinstance(gate.get("schema_version"), str) else "unknown"
    metrics = gate.get("metrics") if isinstance(gate.get("metrics"), dict) else {}
    validation = metrics.get("validation") if isinstance(metrics.get("validation"), dict) else {}
    external_validation = _validation_target_for_path(path, validation_targets)
    record: dict[str, Any] = {
        "id": _gate_id(gate, path.stem),
        "path": _display_path_for_output_source(path, output_path, preserve_paths),
        "exists": path.exists(),
        "schema_version": schema_version,
        "passed": gate.get("passed") is True,
    }
    if path.exists() and path.is_file():
        record["size_bytes"] = path.stat().st_size
        record["sha256"] = _sha256(path)
    if validation:
        record["validation"] = {
            "available": bool(validation.get("available")),
            "passed": bool(validation.get("passed")),
            "strict": bool(validation.get("strict")),
            "error_count": _int_value(validation.get("error_count")),
            "warning_count": _int_value(validation.get("warning_count")),
        }
    elif external_validation:
        record["validation"] = external_validation
    return record


def _gate_id(gate: dict[str, Any], fallback: str) -> str:
    schema = str(gate.get("schema_version") or fallback)
    return schema.removeprefix("hfr.").removesuffix(".v1")


def _gate_requires_validation(gate: dict[str, Any]) -> bool:
    return gate.get("schema_version") in _VALIDATION_REQUIRED_GATE_SCHEMAS


def _add_schema_contracts(
    schema_contracts: dict[str, Any],
    checks: list[dict[str, Any]],
    name: str,
    root: Path,
    contracts: tuple[tuple[str, str, bool], ...],
    preserve_paths: bool,
    output_path: Path,
) -> None:
    for relative, schema_name, jsonl in contracts:
        _add_schema_contract(
            schema_contracts,
            checks,
            f"{name}_{_artifact_key(relative)}",
            root / relative,
            schema_name,
            jsonl,
            preserve_paths,
            output_path,
        )


def _add_schema_contract(
    schema_contracts: dict[str, Any],
    checks: list[dict[str, Any]],
    key: str,
    path: Path,
    schema_name: str,
    jsonl: bool,
    preserve_paths: bool,
    output_path: Path,
) -> None:
    record = _schema_contract_record(path, schema_name, jsonl, preserve_paths, output_path)
    schema_contracts[key] = record
    _add_bool_check(
        checks,
        "schema_contract_passed",
        record["passed"],
        {
            "artifact": key,
            "schema": schema_name,
            "path": record["path"],
        },
    )


def _schema_contract_record(path: Path, schema_name: str, jsonl: bool, preserve_paths: bool, output_path: Path) -> dict[str, Any]:
    regular_file = _is_regular_file(path)
    record: dict[str, Any] = {
        "path": _display_path_for_output_source(path, output_path, preserve_paths),
        "exists": path.exists(),
        "kind": "jsonl" if jsonl else "json",
        "schema_name": schema_name,
        "regular_file": regular_file,
        "symlink": path.is_symlink(),
        "passed": False,
        "error_count": 0,
        "errors": [],
    }
    if jsonl:
        record["row_count"] = 0
        record["row_schema_counts"] = []
    if regular_file:
        record["size_bytes"] = path.stat().st_size
        record["sha256"] = _sha256(path)
    else:
        record["errors"] = ["path is not a regular file"]
        record["error_count"] = 1
        return record
    try:
        result = check_schema_jsonl_file(path, schema_name) if jsonl else check_schema_file(path, schema_name)
    except (OSError, json.JSONDecodeError, SchemaRegistryError) as exc:
        record["errors"] = [str(exc)]
        record["error_count"] = 1
        return record
    record["passed"] = result.get("passed") is True
    record["error_count"] = _int_value(result.get("error_count"))
    record["errors"] = [str(error) for error in result.get("errors", []) if isinstance(error, str)][:20]
    if jsonl:
        record["row_count"] = _int_value(result.get("row_count"))
        record["row_schema_counts"] = result.get("row_schema_counts") if isinstance(result.get("row_schema_counts"), list) else []
    return record


def _validation_summary_records(
    paths: list[str | Path],
    preserve_paths: bool,
    output_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    targets_by_path: dict[str, dict[str, Any]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        summary = _read_json_required(path, "validation summary")
        record: dict[str, Any] = {
            "path": _display_path_for_output_source(path, output_path, preserve_paths),
            "exists": path.exists(),
            "kind": "file",
            "regular_file": _is_regular_file(path),
            "symlink": path.is_symlink(),
            "schema_version": str(summary.get("schema_version") or "unknown"),
            "passed": summary.get("passed") is True,
            "strict": summary.get("strict") is True,
            "target_count": _int_value(summary.get("target_count")),
            "error_count": _int_value(summary.get("error_count")),
            "warning_count": _int_value(summary.get("warning_count")),
            "targets": [],
        }
        if _is_regular_file(path):
            record["size_bytes"] = path.stat().st_size
            record["sha256"] = _sha256(path)
        targets = summary.get("targets") if isinstance(summary.get("targets"), list) else []
        for target in targets:
            if not isinstance(target, dict):
                continue
            target_path = target.get("path") if isinstance(target.get("path"), str) else ""
            errors = [str(error) for error in target.get("errors", []) if isinstance(error, str)]
            warnings = [str(warning) for warning in target.get("warnings", []) if isinstance(warning, str)]
            target_record = {
                "type": str(target.get("type") or ""),
                "path": _display_target_path(target_path, preserve_paths, output_path),
                "passed": target.get("passed") is True,
                "error_count": len(errors),
                "warning_count": len(warnings),
            }
            record["targets"].append(target_record)
            target_passed = target.get("passed") is True
            summary_passed = summary.get("passed") is True
            validation_record = {
                "available": True,
                "passed": target_passed and summary_passed,
                "strict": summary.get("strict") is True,
                "error_count": len(errors),
                "warning_count": len(warnings),
                "source": record["path"],
                "target_type": target_record["type"],
                "summary_passed": summary_passed,
            }
            for key in _path_match_keys(target_path):
                targets_by_path.setdefault(key, validation_record)
        records.append(record)
    return records, targets_by_path


def _validation_target_for_path(path: Path, targets_by_path: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    for key in _path_match_keys(path):
        if key in targets_by_path:
            return dict(targets_by_path[key])
    return None


def _path_match_keys(value: str | Path) -> tuple[str, ...]:
    raw = str(value)
    keys = [raw]
    try:
        path = Path(raw)
    except (OSError, ValueError):
        return tuple(dict.fromkeys(key for key in keys if key))
    keys.append(str(path))
    try:
        keys.append(str(path.resolve()))
    except (OSError, RuntimeError):
        pass
    return tuple(dict.fromkeys(key for key in keys if key))


def _display_target_path(value: str, preserve_paths: bool, output_path: Path) -> str:
    if not value:
        return ""
    return _display_path_for_output_source(Path(value), output_path, preserve_paths)


def _add_export_artifacts(
    artifacts: dict[str, Any],
    checks: list[dict[str, Any]],
    name: str,
    root: Path,
    files: tuple[str, ...],
    preserve_paths: bool,
    output_path: Path,
) -> None:
    artifacts[name] = _dir_record(root, preserve_paths, output_path)
    _add_bool_check(checks, "artifact_dir_exists", root.exists() and root.is_dir(), {"artifact": name})
    _add_bool_check(checks, "artifact_dir_regular", _is_regular_dir(root), {"artifact": name})
    for relative in files:
        key = f"{name}_{_artifact_key(relative)}"
        path = root / relative
        artifacts[key] = _file_record(path, preserve_paths, output_path)
        _add_bool_check(checks, "artifact_file_exists", path.exists() and path.is_file(), {"artifact": name, "file": relative})
        _add_bool_check(checks, "artifact_file_regular", _is_regular_file(path), {"artifact": name, "file": relative})


def _add_dataset_selection(
    dataset_selection: list[dict[str, Any]],
    checks: list[dict[str, Any]],
    artifact: str,
    root: Path,
    preserve_paths: bool,
    output_path: Path,
    required_dataset_versions: list[str],
) -> None:
    manifest_path = root / "manifest.json"
    registry_path = root / "dataset_registry.json"
    manifest = _read_json_safely(manifest_path)
    registry = _read_json_safely(registry_path)
    dataset_version = str(manifest.get("dataset_version") or "") if isinstance(manifest, dict) else ""
    manifest_sha256 = _sha256(manifest_path) if _is_regular_file(manifest_path) else ""
    registry_sha256 = _sha256(registry_path) if _is_regular_file(registry_path) else ""
    registry_selection = registry.get("selection") if isinstance(registry, dict) and isinstance(registry.get("selection"), dict) else {}
    registry_dataset_version = str(registry.get("dataset_version") or "") if isinstance(registry, dict) else ""
    registry_manifest_sha256 = str(registry.get("manifest_sha256") or "") if isinstance(registry, dict) else ""
    leakage = registry.get("leakage_checks") if isinstance(registry, dict) and isinstance(registry.get("leakage_checks"), dict) else {}
    manifest_registry = manifest.get("registry") if isinstance(manifest, dict) and isinstance(manifest.get("registry"), dict) else {}
    trainer_views = _dataset_trainer_views(manifest, registry)
    redaction_status = registry.get("redaction_status") if isinstance(registry, dict) and isinstance(registry.get("redaction_status"), dict) else {}
    matches_required = bool(dataset_version and (not required_dataset_versions or dataset_version in required_dataset_versions))
    record: dict[str, Any] = {
        "artifact": artifact,
        "root": _display_path_for_output_source(root, output_path, preserve_paths),
        "manifest_path": _display_path_for_output_source(manifest_path, output_path, preserve_paths),
        "manifest_sha256": manifest_sha256,
        "registry_path": _display_path_for_output_source(registry_path, output_path, preserve_paths),
        "registry_sha256": registry_sha256,
        "dataset_version": dataset_version,
        "required_dataset_versions": list(required_dataset_versions),
        "matches_required": matches_required,
        "registry_dataset_version": registry_dataset_version,
        "registry_selection_key": str(registry_selection.get("key") or ""),
        "registry_manifest_sha256": registry_manifest_sha256,
        "redaction_passed": redaction_status.get("passed") is True or manifest_registry.get("redaction_passed") is True,
        "trainer_views": trainer_views,
        "trainer_modes": sorted(trainer_views.get("mode_to_view", {})),
    }
    if artifact == "training_export":
        record["heldout_scenario_exclusive"] = (
            leakage.get("heldout_scenario_exclusive") is True
            or manifest_registry.get("heldout_scenario_exclusive") is True
        )
        record["heldout_scenario_ids"] = leakage.get("heldout_scenario_ids") if isinstance(leakage.get("heldout_scenario_ids"), list) else []
    dataset_selection.append(record)
    _add_bool_check(checks, "dataset_version_present", bool(dataset_version), {"artifact": artifact})
    if required_dataset_versions:
        _add_bool_check(
            checks,
            "dataset_version_matches_required",
            matches_required,
            {"artifact": artifact, "dataset_version": dataset_version, "required": ",".join(required_dataset_versions)},
        )
    _add_bool_check(checks, "dataset_registry_exists", _is_regular_file(registry_path), {"artifact": artifact})
    _add_bool_check(
        checks,
        "dataset_registry_version_matches_manifest",
        bool(dataset_version and registry_dataset_version == dataset_version and registry_selection.get("key") == dataset_version),
        {"artifact": artifact, "dataset_version": dataset_version},
    )
    _add_bool_check(
        checks,
        "dataset_registry_manifest_hash_matches",
        bool(manifest_sha256 and registry_manifest_sha256 == manifest_sha256),
        {"artifact": artifact, "dataset_version": dataset_version},
    )
    _add_bool_check(
        checks,
        "dataset_redaction_passed",
        record["redaction_passed"] is True,
        {"artifact": artifact, "dataset_version": dataset_version},
    )
    if artifact == "training_export":
        _add_bool_check(
            checks,
            "dataset_heldout_scenario_exclusive",
            record.get("heldout_scenario_exclusive") is True,
            {"artifact": artifact, "dataset_version": dataset_version},
        )


def _dataset_trainer_views(manifest: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    for source in (registry, manifest):
        if isinstance(source, dict) and isinstance(source.get("trainer_views"), dict):
            return source["trainer_views"]
    manifest_registry = manifest.get("registry") if isinstance(manifest, dict) and isinstance(manifest.get("registry"), dict) else {}
    mode_to_view = manifest_registry.get("mode_to_view") if isinstance(manifest_registry.get("mode_to_view"), dict) else {}
    root_views = manifest_registry.get("root_views") if isinstance(manifest_registry.get("root_views"), list) else []
    selection = registry.get("selection") if isinstance(registry, dict) and isinstance(registry.get("selection"), dict) else {}
    if not mode_to_view and isinstance(selection.get("mode_to_view"), dict):
        mode_to_view = selection["mode_to_view"]
    if not root_views and isinstance(selection.get("root_views"), list):
        root_views = selection["root_views"]
    if not mode_to_view and not root_views:
        return {}
    return {
        "mode_to_view": {str(key): str(value) for key, value in sorted(mode_to_view.items())},
        "root_views": [str(path) for path in root_views if isinstance(path, str)],
        "views": [],
    }


def _trainer_command_record(value: str | None) -> dict[str, Any]:
    if not value:
        return {"provided": False, "raw": "", "argv": []}
    try:
        argv = shlex.split(value)
    except ValueError:
        argv = []
    return {"provided": True, "raw": value, "argv": argv, "parseable": bool(argv)}


def _approved_command_record(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"approved": False, "provided": False, "raw": "", "argv": [], "parseable": False, "shell": ""}
    argv = value.get("argv") if isinstance(value.get("argv"), list) else []
    argv = [item for item in argv if isinstance(item, str)]
    raw = value.get("raw") if isinstance(value.get("raw"), str) else ""
    parseable = bool(value.get("parseable")) if "parseable" in value else bool(argv)
    return {
        "approved": False,
        "provided": value.get("provided") is True,
        "raw": raw,
        "argv": argv,
        "parseable": parseable,
        "shell": shlex.join(argv) if argv else raw,
    }


def _validation_record(summary: dict[str, Any]) -> dict[str, Any]:
    targets = summary.get("targets") if isinstance(summary.get("targets"), list) else []
    errors: list[str] = []
    warnings: list[str] = []
    for target in targets:
        if not isinstance(target, dict):
            continue
        errors.extend(str(error) for error in target.get("errors", []) if isinstance(error, str))
        warnings.extend(str(warning) for warning in target.get("warnings", []) if isinstance(warning, str))
    return {
        "passed": summary.get("passed") is True,
        "strict": summary.get("strict") is True,
        "target_count": _int_value(summary.get("target_count")),
        "error_count": _int_value(summary.get("error_count")),
        "warning_count": _int_value(summary.get("warning_count")),
        "errors": errors,
        "warnings": warnings,
    }


def _launch_gate_records(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    records: list[dict[str, Any]] = []
    for gate in value:
        if not isinstance(gate, dict):
            continue
        records.append(
            {
                "id": str(gate.get("id") or ""),
                "path": str(gate.get("path") or ""),
                "schema_version": str(gate.get("schema_version") or ""),
                "passed": gate.get("passed") is True,
            }
        )
    return records


def _add_bool_check(checks: list[dict[str, Any]], check_id: str, passed: bool, scope: dict[str, str]) -> None:
    checks.append(
        {
            "id": check_id,
            "passed": passed,
            "actual": {"passed": passed},
            "expected": {"passed": True},
            "scope": scope,
            "summary": f"{check_id}: passed={passed}",
        }
    )


def _read_json_required(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TrainerPreflightError(f"{label} must contain a JSON object: {path}")
    return payload


def _read_json_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _read_json_safely(path: Path) -> dict[str, Any]:
    try:
        payload = _read_json_optional(path)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload or {}


def _file_record(path: Path, preserve_paths: bool, output_path: Path) -> dict[str, Any]:
    regular_file = _is_regular_file(path)
    record: dict[str, Any] = {
        "path": _display_path_for_output_source(path, output_path, preserve_paths),
        "exists": path.exists(),
        "kind": "file",
        "regular_file": regular_file,
        "symlink": path.is_symlink(),
    }
    if regular_file:
        record["size_bytes"] = path.stat().st_size
        record["sha256"] = _sha256(path)
    return record


def _dir_record(path: Path, preserve_paths: bool, output_path: Path) -> dict[str, Any]:
    regular_directory = _is_regular_dir(path)
    record: dict[str, Any] = {
        "path": _display_path_for_output_source(path, output_path, preserve_paths),
        "exists": path.exists(),
        "kind": "directory",
        "regular_directory": regular_directory,
        "symlink": path.is_symlink(),
    }
    if regular_directory:
        record["entry_count"] = sum(1 for _ in path.iterdir())
    return record


def _is_regular_file(path: Path) -> bool:
    return path.exists() and path.is_file() and not path.is_symlink()


def _is_regular_dir(path: Path) -> bool:
    return path.exists() and path.is_dir() and not path.is_symlink()


def _artifact_key(value: str) -> str:
    return value.lower().replace("\\", "_").replace("/", "_").replace(".", "_").replace("-", "_")


def _metadata(value: dict[str, str] | None) -> dict[str, str]:
    if not value:
        return {}
    return {str(key): str(raw_value) for key, raw_value in sorted(value.items()) if str(key)}


def _int_value(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path, preserve_paths: bool = False) -> str:
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
