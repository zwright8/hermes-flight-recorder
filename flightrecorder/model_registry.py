"""Dependency-free model scouting, registry, and dry-run training contracts."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODEL_CANDIDATE_SCHEMA_VERSION = "hfr.model_candidate.v1"
MODEL_SCOUT_MANIFEST_SCHEMA_VERSION = "hfr.model_scout_manifest.v1"
MODEL_COMPATIBILITY_REPORT_SCHEMA_VERSION = "hfr.model_compatibility_report.v1"
MODEL_SERVING_PROBE_RECEIPT_SCHEMA_VERSION = "hfr.model_serving_probe_receipt.v1"
MODEL_REGISTRY_ENTRY_SCHEMA_VERSION = "hfr.model_registry_entry.v1"
MODEL_REGISTRY_SCHEMA_VERSION = "hfr.model_registry.v1"
TRAINING_PLAN_SCHEMA_VERSION = "hfr.training_plan.v1"

ALIAS_NAMES = ("candidate", "champion", "rollback")
MODEL_REGISTRY_LINK_COLLECTIONS = (
    "datasets",
    "training_runs",
    "adapters",
    "evals",
    "serving_probes",
    "promotion_decisions",
)
COMPATIBILITY_FIELDS = (
    "tokenizer",
    "chat_template",
    "serving",
    "tool_calls",
    "structured_outputs",
    "quantization",
    "memory",
)
COMPATIBILITY_REPORT_PROBE_IDS = ("context", *COMPATIBILITY_FIELDS)
SERVING_PROBE_RECEIPT_PROBE_IDS = (
    "health",
    "model_metadata",
    "chat",
    "tool_calls",
    "structured_outputs",
    "context",
    "memory",
)
SERVING_PROBE_RECEIPT_MODES = {"metadata_only", "external_receipt"}
TRAINING_APPROVED_LICENSE_STATUSES = {"approved"}
TRAINING_APPROVED_REVIEW_STATUSES = {"approved"}


class ModelRegistryError(ValueError):
    """Raised when a model-layer artifact cannot be safely used."""


def model_candidate_errors(candidate: Any, *, require_training_eligible: bool = False) -> list[str]:
    errors: list[str] = []
    if not isinstance(candidate, dict):
        return ["model_candidate must be a JSON object"]
    _require_equal(candidate, "schema_version", MODEL_CANDIDATE_SCHEMA_VERSION, errors, "model_candidate.")
    for field_name in ("candidate_id", "model_id"):
        _require_non_empty_string(candidate, field_name, errors, "model_candidate.")
    source = candidate.get("source")
    if not isinstance(source, dict):
        errors.append("model_candidate.source must be an object.")
    else:
        for field_name in ("type", "url"):
            _require_non_empty_string(source, field_name, errors, "model_candidate.source.")
    license_review = candidate.get("license")
    if not isinstance(license_review, dict):
        errors.append("model_candidate.license must be an object.")
    else:
        _license_errors(license_review, errors, "model_candidate.license.")
        if require_training_eligible:
            _training_license_errors(license_review, errors, "model_candidate.license.")
    compatibility = candidate.get("compatibility")
    if not isinstance(compatibility, dict):
        errors.append("model_candidate.compatibility must be an object.")
    else:
        if not _is_positive_int(compatibility.get("context_length")):
            errors.append("model_candidate.compatibility.context_length must be a positive integer.")
        for field_name in COMPATIBILITY_FIELDS:
            if not isinstance(compatibility.get(field_name), dict):
                errors.append(f"model_candidate.compatibility.{field_name} must be an object.")
        _require_bool_if_present(compatibility.get("tool_calls"), "supported", errors, "model_candidate.compatibility.tool_calls.")
        _require_bool_if_present(
            compatibility.get("structured_outputs"),
            "supported",
            errors,
            "model_candidate.compatibility.structured_outputs.",
        )
    notes = candidate.get("notes")
    if notes is not None and not _is_string_list(notes):
        errors.append("model_candidate.notes must be a list of strings when present.")
    return errors


def validate_model_candidate(candidate: Any, *, require_training_eligible: bool = False) -> dict[str, Any]:
    errors = model_candidate_errors(candidate, require_training_eligible=require_training_eligible)
    if errors:
        raise ModelRegistryError("; ".join(errors))
    return copy.deepcopy(candidate)


def model_scout_manifest_errors(manifest: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(manifest, dict):
        return ["model_scout_manifest must be a JSON object"]
    _require_equal(manifest, "schema_version", MODEL_SCOUT_MANIFEST_SCHEMA_VERSION, errors, "model_scout_manifest.")
    _require_non_empty_string(manifest, "updated_at", errors, "model_scout_manifest.")
    selection_policy = manifest.get("selection_policy")
    if not isinstance(selection_policy, dict):
        errors.append("model_scout_manifest.selection_policy must be an object.")
    else:
        for field_name in (
            "allow_unknown_license_for_scouting",
            "block_unknown_license_from_training",
            "require_candidate_source",
            "require_compatibility_metadata",
            "require_terms_review_for_training",
        ):
            if not isinstance(selection_policy.get(field_name), bool):
                errors.append(f"model_scout_manifest.selection_policy.{field_name} must be a boolean.")
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        errors.append("model_scout_manifest.candidates must be a non-empty list.")
        candidates = []
    seen: set[str] = set()
    for index, item in enumerate(candidates):
        label = f"model_scout_manifest.candidates[{index}]."
        if not isinstance(item, dict):
            errors.append(f"model_scout_manifest.candidates[{index}] must be an object.")
            continue
        for field_name in ("candidate_id", "manifest_path", "priority", "reason"):
            _require_non_empty_string(item, field_name, errors, label)
        candidate_id = item.get("candidate_id")
        if isinstance(candidate_id, str) and candidate_id:
            if candidate_id in seen:
                errors.append(f"{label}candidate_id duplicates {candidate_id!r}.")
            seen.add(candidate_id)
        for field_name in ("compatibility_report_path", "registry_entry_id", "license_status", "review_status"):
            if field_name in item:
                _require_non_empty_string(item, field_name, errors, label)
        _require_bool_if_present(item, "training_selection_eligible", errors, label)
    notes = manifest.get("notes")
    if notes is not None and not _is_string_list(notes):
        errors.append("model_scout_manifest.notes must be a list of strings when present.")
    return errors


def build_model_compatibility_report(
    candidate: dict[str, Any],
    *,
    out_path: str | Path,
    generated_at: str | None = None,
    preserve_paths: bool = False,
) -> dict[str, Any]:
    validate_model_candidate(candidate)
    compatibility = candidate["compatibility"]
    probes = [_context_probe(compatibility["context_length"])]
    probes.extend(_compatibility_probe(field_name, compatibility[field_name]) for field_name in COMPATIBILITY_FIELDS)
    missing_count = sum(1 for probe in probes if probe["status"] == "missing")
    metadata_only_count = sum(1 for probe in probes if probe["status"] == "metadata_only")
    verified_count = sum(1 for probe in probes if probe["verified"] is True)
    unsupported_count = sum(1 for probe in probes if probe.get("supported") is False)
    report = {
        "schema_version": MODEL_COMPATIBILITY_REPORT_SCHEMA_VERSION,
        "report_path": _display_path(Path(out_path), preserve_paths),
        "generated_at": generated_at or _now_iso(),
        "candidate_id": candidate["candidate_id"],
        "model_id": candidate["model_id"],
        "passed": missing_count == 0,
        "readiness": "metadata_recorded" if missing_count == 0 else "blocked",
        "recommendation": "ready_for_registry_dry_run" if missing_count == 0 else "block_training_selection",
        "download_policy": {
            "downloaded_weights": False,
            "downloaded_tokenizer": False,
            "imported_heavy_ml": False,
            "gpu_execution": False,
        },
        "probes": probes,
        "summary": {
            "probe_count": len(probes),
            "missing_count": missing_count,
            "verified_count": verified_count,
            "metadata_only_count": metadata_only_count,
            "unsupported_count": unsupported_count,
            "context_length": compatibility["context_length"],
        },
        "notes": [
            "This compatibility report records metadata without downloading model weights or tokenizers.",
            "Future serving checks should replace metadata_only probes with verified tokenizer and serving evidence.",
        ],
    }
    errors = model_compatibility_report_errors(report)
    if errors:
        raise ModelRegistryError("; ".join(errors))
    return report


def model_compatibility_report_errors(report: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(report, dict):
        return ["model_compatibility_report must be a JSON object"]
    _require_equal(report, "schema_version", MODEL_COMPATIBILITY_REPORT_SCHEMA_VERSION, errors, "model_compatibility_report.")
    for field_name in ("report_path", "generated_at", "candidate_id", "model_id", "readiness", "recommendation"):
        _require_non_empty_string(report, field_name, errors, "model_compatibility_report.")
    if not isinstance(report.get("passed"), bool):
        errors.append("model_compatibility_report.passed must be a boolean.")
    policy = report.get("download_policy")
    if not isinstance(policy, dict):
        errors.append("model_compatibility_report.download_policy must be an object.")
    else:
        for field_name in ("downloaded_weights", "downloaded_tokenizer", "imported_heavy_ml", "gpu_execution"):
            if policy.get(field_name) is not False:
                errors.append(f"model_compatibility_report.download_policy.{field_name} must be false.")
    probes = report.get("probes")
    if not isinstance(probes, list):
        errors.append("model_compatibility_report.probes must be a list.")
        probes = []
    seen: set[str] = set()
    for index, probe in enumerate(probes):
        if not isinstance(probe, dict):
            errors.append(f"model_compatibility_report.probes[{index}] must be an object.")
            continue
        probe_id = probe.get("id")
        if not isinstance(probe_id, str) or not probe_id:
            errors.append(f"model_compatibility_report.probes[{index}].id must be a non-empty string.")
        elif probe_id in seen:
            errors.append(f"model_compatibility_report.probes[{index}].id duplicates {probe_id!r}.")
        else:
            seen.add(probe_id)
        for field_name in ("status", "source", "notes"):
            _require_non_empty_string(probe, field_name, errors, f"model_compatibility_report.probes[{index}].")
        if not isinstance(probe.get("verified"), bool):
            errors.append(f"model_compatibility_report.probes[{index}].verified must be a boolean.")
        if "supported" in probe and probe.get("supported") is not None and not isinstance(probe.get("supported"), bool):
            errors.append(f"model_compatibility_report.probes[{index}].supported must be a boolean or null.")
    if seen != set(COMPATIBILITY_REPORT_PROBE_IDS):
        errors.append(f"model_compatibility_report.probes must contain exactly {sorted(COMPATIBILITY_REPORT_PROBE_IDS)!r}.")
    summary = report.get("summary")
    if not isinstance(summary, dict):
        errors.append("model_compatibility_report.summary must be an object.")
    else:
        for field_name in ("probe_count", "missing_count", "verified_count", "metadata_only_count", "unsupported_count", "context_length"):
            if not _is_non_negative_int(summary.get(field_name)):
                errors.append(f"model_compatibility_report.summary.{field_name} must be a non-negative integer.")
        if isinstance(summary.get("probe_count"), int) and summary.get("probe_count") != len(probes):
            errors.append("model_compatibility_report.summary.probe_count must match probes length.")
    return errors


def new_model_registry(*, registry_path: str | Path = "experiments/registry/model_registry.json") -> dict[str, Any]:
    return {
        "schema_version": MODEL_REGISTRY_SCHEMA_VERSION,
        "registry_path": str(registry_path),
        "updated_at": _now_iso(),
        "entries": {},
        "aliases": {alias: None for alias in ALIAS_NAMES},
        "alias_history": [],
        "notes": [],
    }


def load_model_registry(path: str | Path) -> dict[str, Any]:
    registry_path = Path(path)
    if not registry_path.exists():
        return new_model_registry(registry_path=registry_path)
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    registry = _registry_with_defaults(payload)
    validate_model_registry(registry)
    return registry


def model_registry_entry_errors(entry: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(entry, dict):
        return ["model_registry_entry must be a JSON object"]
    _require_equal(entry, "schema_version", MODEL_REGISTRY_ENTRY_SCHEMA_VERSION, errors, "model_registry_entry.")
    for field_name in ("entry_id", "candidate_id", "registered_at", "updated_at", "status"):
        _require_non_empty_string(entry, field_name, errors, "model_registry_entry.")
    candidate = entry.get("candidate")
    errors.extend(model_candidate_errors(candidate))
    if isinstance(candidate, dict) and entry.get("candidate_id") != candidate.get("candidate_id"):
        errors.append("model_registry_entry.candidate_id must match candidate.candidate_id.")
    if not isinstance(entry.get("training_eligible"), bool):
        errors.append("model_registry_entry.training_eligible must be a boolean.")
    if entry.get("training_eligible") is True and isinstance(candidate, dict):
        errors.extend(
            error.replace("model_candidate.", "model_registry_entry.candidate.")
            for error in model_candidate_errors(candidate, require_training_eligible=True)
        )
    if not _is_string_list(entry.get("notes", [])):
        errors.append("model_registry_entry.notes must be a list of strings when present.")
    _model_registry_links_errors(entry.get("links"), errors)
    return errors


def model_registry_errors(registry: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(registry, dict):
        return ["model_registry must be a JSON object"]
    _require_equal(registry, "schema_version", MODEL_REGISTRY_SCHEMA_VERSION, errors, "model_registry.")
    _require_non_empty_string(registry, "registry_path", errors, "model_registry.")
    _require_non_empty_string(registry, "updated_at", errors, "model_registry.")
    entries = registry.get("entries")
    if not isinstance(entries, dict):
        errors.append("model_registry.entries must be an object keyed by entry id.")
        entries = {}
    for entry_id, entry in entries.items():
        if not isinstance(entry_id, str) or not entry_id:
            errors.append("model_registry.entries keys must be non-empty strings.")
            continue
        errors.extend(error.replace("model_registry_entry.", f"model_registry.entries.{entry_id}.") for error in model_registry_entry_errors(entry))
        if isinstance(entry, dict) and entry.get("entry_id") != entry_id:
            errors.append(f"model_registry.entries.{entry_id}.entry_id must match its registry key.")
    aliases = registry.get("aliases")
    if not isinstance(aliases, dict):
        errors.append("model_registry.aliases must be an object.")
        aliases = {}
    for alias in ALIAS_NAMES:
        if alias not in aliases:
            errors.append(f"model_registry.aliases.{alias} must be present.")
            continue
        target = aliases.get(alias)
        if target is not None and (not isinstance(target, str) or not target):
            errors.append(f"model_registry.aliases.{alias} must be a non-empty string or null.")
        elif isinstance(target, str) and target not in entries:
            errors.append(f"model_registry.aliases.{alias} references unknown entry {target!r}.")
    champion = aliases.get("champion")
    rollback = aliases.get("rollback")
    if champion is not None and rollback is None:
        errors.append("model_registry.aliases.rollback must be explicit when champion is set.")
    if champion is not None and rollback == champion:
        errors.append("model_registry.aliases.rollback must differ from champion.")
    if not isinstance(registry.get("alias_history"), list):
        errors.append("model_registry.alias_history must be a list.")
    return errors


def validate_model_registry(registry: Any) -> dict[str, Any]:
    errors = model_registry_errors(registry)
    if errors:
        raise ModelRegistryError("; ".join(errors))
    return copy.deepcopy(registry)


def register_model_candidate(registry: dict[str, Any], candidate: dict[str, Any], *, status: str = "registered") -> dict[str, Any]:
    validate_model_candidate(candidate)
    next_registry = _registry_with_defaults(registry)
    candidate_id = candidate["candidate_id"]
    existing = next_registry["entries"].get(candidate_id) if isinstance(next_registry["entries"].get(candidate_id), dict) else {}
    license_review = candidate.get("license") if isinstance(candidate.get("license"), dict) else {}
    entry = {
        "schema_version": MODEL_REGISTRY_ENTRY_SCHEMA_VERSION,
        "entry_id": candidate_id,
        "candidate_id": candidate_id,
        "registered_at": existing.get("registered_at") or _now_iso(),
        "updated_at": _now_iso(),
        "status": status,
        "training_eligible": is_training_license_approved(candidate),
        "license_status": str(license_review.get("status") or ""),
        "candidate": copy.deepcopy(candidate),
        "links": _model_registry_links_with_defaults(existing.get("links")),
        "notes": list(existing.get("notes") if _is_string_list(existing.get("notes")) else []),
    }
    next_registry["entries"][candidate_id] = entry
    next_registry["updated_at"] = _now_iso()
    validate_model_registry(next_registry)
    return next_registry


def move_model_alias(
    registry: dict[str, Any],
    *,
    alias: str,
    target: str,
    rollback_target: str | None = None,
    reason: str = "",
) -> dict[str, Any]:
    if alias not in ALIAS_NAMES:
        raise ModelRegistryError(f"alias must be one of {', '.join(ALIAS_NAMES)}")
    next_registry = _registry_with_defaults(registry)
    _require_registered_target(next_registry, target)
    now = _now_iso()
    if alias == "champion":
        if not rollback_target:
            raise ModelRegistryError("moving champion requires explicit rollback_target")
        _require_registered_target(next_registry, rollback_target)
        if rollback_target == target:
            raise ModelRegistryError("rollback target must differ from champion target")
        _set_alias(next_registry, "rollback", rollback_target, now, reason or "set rollback before champion move")
    _set_alias(next_registry, alias, target, now, reason)
    next_registry["updated_at"] = _now_iso()
    validate_model_registry(next_registry)
    return next_registry


def list_model_registry_entries(registry: dict[str, Any]) -> list[dict[str, Any]]:
    validate_model_registry(registry)
    aliases = registry.get("aliases") if isinstance(registry.get("aliases"), dict) else {}
    rows: list[dict[str, Any]] = []
    for entry_id, entry in sorted(registry["entries"].items()):
        candidate = entry.get("candidate") if isinstance(entry.get("candidate"), dict) else {}
        rows.append(
            {
                "entry_id": entry_id,
                "candidate_id": entry.get("candidate_id"),
                "model_id": candidate.get("model_id"),
                "status": entry.get("status"),
                "training_eligible": entry.get("training_eligible") is True,
                "license_status": entry.get("license_status"),
                "aliases": sorted(alias for alias, target in aliases.items() if target == entry_id),
                "link_counts": _model_registry_link_counts(entry.get("links")),
            }
        )
    return rows


def resolve_model_registry_entry(registry: dict[str, Any], model_ref: str) -> dict[str, Any]:
    validate_model_registry(registry)
    aliases = registry.get("aliases") if isinstance(registry.get("aliases"), dict) else {}
    target = aliases.get(model_ref, model_ref)
    if target is None:
        raise ModelRegistryError(f"model alias {model_ref!r} is not set")
    entry = registry["entries"].get(target)
    if not isinstance(entry, dict):
        raise ModelRegistryError(f"model reference {model_ref!r} does not resolve to a registered entry")
    return copy.deepcopy(entry)


def select_model_for_training(registry: dict[str, Any], model_ref: str) -> dict[str, Any]:
    entry = resolve_model_registry_entry(registry, model_ref)
    validate_model_candidate(entry.get("candidate"), require_training_eligible=True)
    if entry.get("training_eligible") is not True:
        raise ModelRegistryError(f"model entry {entry.get('entry_id')!r} is not training eligible")
    return entry


def link_model_registry_artifact(
    registry: dict[str, Any],
    *,
    entry_id: str,
    collection: str,
    artifact_id: str,
    kind: str,
    status: str = "recorded",
    path: str | Path | None = None,
    sha256: str | None = None,
    metadata: dict[str, Any] | None = None,
    preserve_paths: bool = False,
) -> dict[str, Any]:
    next_registry = _registry_with_defaults(registry)
    if collection not in MODEL_REGISTRY_LINK_COLLECTIONS:
        raise ModelRegistryError(f"collection must be one of {', '.join(MODEL_REGISTRY_LINK_COLLECTIONS)}")
    entry = next_registry["entries"].get(entry_id)
    if not isinstance(entry, dict):
        raise ModelRegistryError(f"model registry entry {entry_id!r} is not registered")
    if not artifact_id:
        raise ModelRegistryError("artifact_id must be a non-empty string")
    if not kind:
        raise ModelRegistryError("kind must be a non-empty string")
    if not status:
        raise ModelRegistryError("status must be a non-empty string")
    record: dict[str, Any] = {
        "id": artifact_id,
        "kind": kind,
        "status": status,
        "recorded_at": _now_iso(),
        "metadata": copy.deepcopy(metadata or {}),
    }
    if path is not None and str(path):
        source_path = Path(path)
        if not source_path.exists() or not source_path.is_file() or source_path.is_symlink():
            raise ModelRegistryError(f"linked artifact path must be an existing regular file: {source_path}")
        computed_sha256 = _sha256(source_path)
        if sha256 is not None and sha256.lower() != computed_sha256:
            raise ModelRegistryError("linked artifact sha256 does not match path contents")
        record["path"] = _display_path(source_path, preserve_paths)
        record["sha256"] = computed_sha256
    elif sha256 is not None:
        normalized_sha256 = sha256.lower()
        if not _is_sha256(normalized_sha256):
            raise ModelRegistryError("sha256 must be a 64-character hex digest")
        record["sha256"] = normalized_sha256
    entry["links"] = _model_registry_links_with_defaults(entry.get("links"))
    _upsert_link_record(entry["links"][collection], record)
    entry["updated_at"] = _now_iso()
    next_registry["updated_at"] = _now_iso()
    validate_model_registry(next_registry)
    return next_registry


def build_dry_run_training_plan(
    registry: dict[str, Any],
    *,
    model_ref: str,
    dataset_id: str,
    dataset_manifest: str | Path,
    trainer: str,
    mode: str,
    output_dir: str | Path,
    out_path: str | Path,
    hyperparameters: dict[str, Any] | None = None,
    compute: dict[str, Any] | None = None,
    compatibility_report: dict[str, Any] | None = None,
    compatibility_report_path: str | Path | None = None,
    preserve_paths: bool = False,
) -> dict[str, Any]:
    entry = select_model_for_training(registry, model_ref)
    candidate = entry["candidate"]
    dataset_path = Path(dataset_manifest)
    if not dataset_id:
        raise ModelRegistryError("dataset_id must be a non-empty string")
    if not dataset_path.exists() or not dataset_path.is_file() or dataset_path.is_symlink():
        raise ModelRegistryError(f"dataset_manifest must be an existing regular file: {dataset_path}")
    if not trainer:
        raise ModelRegistryError("trainer must be a non-empty string")
    if not mode:
        raise ModelRegistryError("mode must be a non-empty string")
    checks = [
        {"id": "license_training_approved", "passed": True, "summary": "candidate license review is approved for training"},
        {"id": "dataset_manifest_exists", "passed": True, "summary": "dataset manifest exists and was fingerprinted"},
        {"id": "no_weight_download", "passed": True, "summary": "plan generation did not download model weights or tokenizers"},
        {"id": "no_gpu_execution", "passed": True, "summary": "plan generation did not launch GPU work"},
    ]
    plan = {
        "schema_version": TRAINING_PLAN_SCHEMA_VERSION,
        "plan_path": _display_path(Path(out_path), preserve_paths),
        "created_at": _now_iso(),
        "dry_run": True,
        "no_weight_download": True,
        "gpu_execution": False,
        "passed": True,
        "readiness": "ready",
        "recommendation": "ready_for_dry_run_review",
        "model": {
            "model_ref": model_ref,
            "entry_id": entry["entry_id"],
            "candidate_id": entry["candidate_id"],
            "model_id": candidate["model_id"],
            "candidate": copy.deepcopy(candidate),
            "license": copy.deepcopy(candidate["license"]),
            "compatibility": copy.deepcopy(candidate["compatibility"]),
        },
        "dataset": {
            "dataset_id": dataset_id,
            "manifest_path": _display_path(dataset_path, preserve_paths),
            "manifest_sha256": _sha256(dataset_path),
        },
        "trainer": {"name": trainer, "mode": mode, "hyperparameters": dict(hyperparameters or {})},
        "output": {"output_dir": _display_path(Path(output_dir), preserve_paths)},
        "compute": {"assumptions": dict(compute or {}), "dry_run_only": True, "accelerator_required": False},
        "execution": {
            "flight_recorder_downloaded_weights": False,
            "flight_recorder_downloaded_tokenizer": False,
            "flight_recorder_imported_heavy_ml": False,
            "flight_recorder_launched_gpu_job": False,
            "runner_must_revalidate_plan": True,
        },
        "checks": checks,
        "blocked_reasons": [],
        "notes": [
            "This is a dry-run plan. It records metadata for a future trainer but does not import heavy ML packages.",
            "Training runners must revalidate license, compatibility, dataset gates, and output paths before execution.",
        ],
    }
    if compatibility_report is not None:
        plan["compatibility_report"] = _training_compatibility_report_record(
            compatibility_report,
            candidate=candidate,
            report_path=compatibility_report_path,
            preserve_paths=preserve_paths,
        )
        plan["checks"].insert(
            2,
            {
                "id": "compatibility_report_passed",
                "passed": True,
                "summary": "compatibility report matches the selected model and passed no-download checks",
            },
        )
    errors = training_plan_errors(plan)
    if errors:
        raise ModelRegistryError("; ".join(errors))
    return plan


def build_model_serving_probe_receipt(
    registry: dict[str, Any],
    *,
    model_ref: str,
    out_path: str | Path,
    profile_id: str,
    provider: str,
    serving_engine: str,
    base_url: str,
    probe_mode: str = "metadata_only",
    compatibility_report: dict[str, Any] | None = None,
    compatibility_report_path: str | Path | None = None,
    preserve_paths: bool = False,
) -> dict[str, Any]:
    if probe_mode not in SERVING_PROBE_RECEIPT_MODES:
        raise ModelRegistryError(f"probe_mode must be one of {', '.join(sorted(SERVING_PROBE_RECEIPT_MODES))}")
    for field_name, value in (
        ("profile_id", profile_id),
        ("provider", provider),
        ("serving_engine", serving_engine),
        ("base_url", base_url),
    ):
        if not isinstance(value, str) or not value:
            raise ModelRegistryError(f"{field_name} must be a non-empty string")
    entry = resolve_model_registry_entry(registry, model_ref)
    candidate = validate_model_candidate(entry.get("candidate"))
    compatibility = candidate["compatibility"]
    probes = [
        _serving_probe(
            "health",
            status="not_run",
            verified=False,
            supported=None,
            source="serving_profile",
            notes="Flight Recorder did not launch a server or open a health-check connection.",
            metadata={"requires_external_endpoint": True},
        ),
        _serving_probe(
            "model_metadata",
            status="metadata_only",
            verified=False,
            supported=True,
            source="model_registry_entry.candidate",
            notes="Registry candidate metadata is available for an external serving probe.",
            metadata={"model_id": candidate["model_id"]},
        ),
        _serving_probe_from_compatibility("chat", "chat_template", compatibility["chat_template"]),
        _serving_probe_from_compatibility("tool_calls", "tool_calls", compatibility["tool_calls"]),
        _serving_probe_from_compatibility("structured_outputs", "structured_outputs", compatibility["structured_outputs"]),
        _serving_context_probe(compatibility["context_length"]),
        _serving_probe_from_compatibility("memory", "memory", compatibility["memory"]),
    ]
    summary = {
        "probe_count": len(probes),
        "verified_count": sum(1 for probe in probes if probe.get("verified") is True),
        "metadata_only_count": sum(1 for probe in probes if probe.get("status") == "metadata_only"),
        "not_run_count": sum(1 for probe in probes if probe.get("status") == "not_run"),
        "blocked_count": sum(1 for probe in probes if probe.get("status") == "blocked"),
        "context_length": compatibility["context_length"],
    }
    receipt = {
        "schema_version": MODEL_SERVING_PROBE_RECEIPT_SCHEMA_VERSION,
        "receipt_path": _display_path(Path(out_path), preserve_paths),
        "generated_at": _now_iso(),
        "model_ref": model_ref,
        "entry_id": entry["entry_id"],
        "candidate_id": entry["candidate_id"],
        "model_id": candidate["model_id"],
        "probe_mode": probe_mode,
        "passed": summary["blocked_count"] == 0,
        "readiness": "metadata_recorded" if probe_mode == "metadata_only" else "external_receipt_recorded",
        "recommendation": "ready_for_external_serving_probe",
        "serving_profile": {
            "profile_id": profile_id,
            "provider": provider,
            "serving_engine": serving_engine,
            "base_url": base_url,
            "launched_by_flight_recorder": False,
        },
        "download_policy": {
            "downloaded_weights": False,
            "downloaded_tokenizer": False,
            "imported_heavy_ml": False,
            "launched_server": False,
            "gpu_execution": False,
            "network_connection_attempted": False,
        },
        "execution": {
            "flight_recorder_downloaded_weights": False,
            "flight_recorder_downloaded_tokenizer": False,
            "flight_recorder_imported_heavy_ml": False,
            "flight_recorder_launched_server": False,
            "flight_recorder_launched_gpu_job": False,
            "flight_recorder_opened_network_connection": False,
            "external_endpoint_required_for_verification": True,
        },
        "probes": probes,
        "summary": summary,
        "notes": [
            "This receipt records serving-probe metadata without downloading weights or tokenizers.",
            "Flight Recorder did not launch a serving process, open a network connection, or run GPU work.",
            "A future serving worker must attach verified endpoint evidence before treating serving behavior as verified.",
        ],
    }
    if compatibility_report is not None:
        receipt["compatibility_report"] = _serving_compatibility_report_record(
            compatibility_report,
            candidate=candidate,
            report_path=compatibility_report_path,
            preserve_paths=preserve_paths,
        )
    errors = model_serving_probe_receipt_errors(receipt)
    if errors:
        raise ModelRegistryError("; ".join(errors))
    return receipt


def model_serving_probe_receipt_errors(receipt: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(receipt, dict):
        return ["model_serving_probe_receipt must be a JSON object"]
    _require_equal(receipt, "schema_version", MODEL_SERVING_PROBE_RECEIPT_SCHEMA_VERSION, errors, "model_serving_probe_receipt.")
    for field_name in (
        "receipt_path",
        "generated_at",
        "model_ref",
        "entry_id",
        "candidate_id",
        "model_id",
        "probe_mode",
        "readiness",
        "recommendation",
    ):
        _require_non_empty_string(receipt, field_name, errors, "model_serving_probe_receipt.")
    if receipt.get("probe_mode") not in SERVING_PROBE_RECEIPT_MODES:
        errors.append("model_serving_probe_receipt.probe_mode must be metadata_only or external_receipt.")
    if receipt.get("readiness") not in {"metadata_recorded", "external_receipt_recorded", "verified", "blocked"}:
        errors.append("model_serving_probe_receipt.readiness must be metadata_recorded, external_receipt_recorded, verified, or blocked.")
    if receipt.get("recommendation") not in {"ready_for_external_serving_probe", "serving_verified", "block_serving_selection"}:
        errors.append("model_serving_probe_receipt.recommendation must be a known serving recommendation.")
    if not isinstance(receipt.get("passed"), bool):
        errors.append("model_serving_probe_receipt.passed must be a boolean.")
    profile = receipt.get("serving_profile")
    if not isinstance(profile, dict):
        errors.append("model_serving_probe_receipt.serving_profile must be an object.")
    else:
        for field_name in ("profile_id", "provider", "serving_engine", "base_url"):
            _require_non_empty_string(profile, field_name, errors, "model_serving_probe_receipt.serving_profile.")
        if profile.get("launched_by_flight_recorder") is not False:
            errors.append("model_serving_probe_receipt.serving_profile.launched_by_flight_recorder must be false.")
    policy = receipt.get("download_policy")
    if not isinstance(policy, dict):
        errors.append("model_serving_probe_receipt.download_policy must be an object.")
    else:
        for field_name in (
            "downloaded_weights",
            "downloaded_tokenizer",
            "imported_heavy_ml",
            "launched_server",
            "gpu_execution",
            "network_connection_attempted",
        ):
            if policy.get(field_name) is not False:
                errors.append(f"model_serving_probe_receipt.download_policy.{field_name} must be false.")
    execution = receipt.get("execution")
    if not isinstance(execution, dict):
        errors.append("model_serving_probe_receipt.execution must be an object.")
    else:
        for field_name in (
            "flight_recorder_downloaded_weights",
            "flight_recorder_downloaded_tokenizer",
            "flight_recorder_imported_heavy_ml",
            "flight_recorder_launched_server",
            "flight_recorder_launched_gpu_job",
            "flight_recorder_opened_network_connection",
        ):
            if execution.get(field_name) is not False:
                errors.append(f"model_serving_probe_receipt.execution.{field_name} must be false.")
        if not isinstance(execution.get("external_endpoint_required_for_verification"), bool):
            errors.append("model_serving_probe_receipt.execution.external_endpoint_required_for_verification must be a boolean.")
    probes = receipt.get("probes")
    if not isinstance(probes, list):
        errors.append("model_serving_probe_receipt.probes must be a list.")
        probes = []
    seen: set[str] = set()
    for index, probe in enumerate(probes):
        prefix = f"model_serving_probe_receipt.probes[{index}]."
        if not isinstance(probe, dict):
            errors.append(f"model_serving_probe_receipt.probes[{index}] must be an object.")
            continue
        probe_id = probe.get("id")
        if not isinstance(probe_id, str) or not probe_id:
            errors.append(f"{prefix}id must be a non-empty string.")
        elif probe_id in seen:
            errors.append(f"{prefix}id duplicates {probe_id!r}.")
        else:
            seen.add(probe_id)
        for field_name in ("status", "source", "notes"):
            _require_non_empty_string(probe, field_name, errors, prefix)
        if not isinstance(probe.get("verified"), bool):
            errors.append(f"{prefix}verified must be a boolean.")
        if "supported" in probe and probe.get("supported") is not None and not isinstance(probe.get("supported"), bool):
            errors.append(f"{prefix}supported must be a boolean or null.")
        if not isinstance(probe.get("metadata", {}), dict):
            errors.append(f"{prefix}metadata must be an object when present.")
    if seen != set(SERVING_PROBE_RECEIPT_PROBE_IDS):
        errors.append(f"model_serving_probe_receipt.probes must contain exactly {sorted(SERVING_PROBE_RECEIPT_PROBE_IDS)!r}.")
    summary = receipt.get("summary")
    if not isinstance(summary, dict):
        errors.append("model_serving_probe_receipt.summary must be an object.")
    else:
        for field_name in ("probe_count", "verified_count", "metadata_only_count", "not_run_count", "blocked_count", "context_length"):
            if not _is_non_negative_int(summary.get(field_name)):
                errors.append(f"model_serving_probe_receipt.summary.{field_name} must be a non-negative integer.")
        if isinstance(summary.get("probe_count"), int) and summary.get("probe_count") != len(probes):
            errors.append("model_serving_probe_receipt.summary.probe_count must match probes length.")
    report = receipt.get("compatibility_report")
    if report is not None and not isinstance(report, dict):
        errors.append("model_serving_probe_receipt.compatibility_report must be an object when present.")
    elif isinstance(report, dict):
        _serving_probe_compatibility_report_errors(report, receipt, errors)
    return errors


def training_plan_errors(plan: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(plan, dict):
        return ["training_plan must be a JSON object"]
    _require_equal(plan, "schema_version", TRAINING_PLAN_SCHEMA_VERSION, errors, "training_plan.")
    for field_name in ("plan_path", "created_at", "readiness", "recommendation"):
        _require_non_empty_string(plan, field_name, errors, "training_plan.")
    for field_name, expected in (("dry_run", True), ("no_weight_download", True), ("gpu_execution", False), ("passed", True)):
        if plan.get(field_name) is not expected:
            errors.append(f"training_plan.{field_name} must be {expected!r}.")
    model = plan.get("model")
    if not isinstance(model, dict):
        errors.append("training_plan.model must be an object.")
    else:
        for field_name in ("model_ref", "entry_id", "candidate_id", "model_id"):
            _require_non_empty_string(model, field_name, errors, "training_plan.model.")
        errors.extend(error.replace("model_candidate.", "training_plan.model.candidate.") for error in model_candidate_errors(model.get("candidate"), require_training_eligible=True))
    report = plan.get("compatibility_report")
    if isinstance(report, dict) and isinstance(model, dict):
        _training_plan_compatibility_report_errors(report, model, errors)
    elif report is not None:
        errors.append("training_plan.compatibility_report must be an object when present.")
    dataset = plan.get("dataset")
    if not isinstance(dataset, dict):
        errors.append("training_plan.dataset must be an object.")
    else:
        for field_name in ("dataset_id", "manifest_path", "manifest_sha256"):
            _require_non_empty_string(dataset, field_name, errors, "training_plan.dataset.")
        if isinstance(dataset.get("manifest_sha256"), str) and not _is_sha256(dataset.get("manifest_sha256")):
            errors.append("training_plan.dataset.manifest_sha256 must be a 64-character hex digest.")
    for field_name in ("trainer", "output", "compute", "execution"):
        if not isinstance(plan.get(field_name), dict):
            errors.append(f"training_plan.{field_name} must be an object.")
    if not isinstance(plan.get("checks"), list):
        errors.append("training_plan.checks must be a list.")
    if not _is_string_list(plan.get("blocked_reasons", [])):
        errors.append("training_plan.blocked_reasons must be a list of strings when present.")
    return errors


def is_training_license_approved(candidate: dict[str, Any]) -> bool:
    license_review = candidate.get("license") if isinstance(candidate, dict) else None
    if not isinstance(license_review, dict):
        return False
    return (
        _normalized(license_review.get("status")) in TRAINING_APPROVED_LICENSE_STATUSES
        and _normalized(license_review.get("review_status")) in TRAINING_APPROVED_REVIEW_STATUSES
        and license_review.get("accepted_terms") is True
        and license_review.get("training_allowed") is True
    )


def _serving_compatibility_report_record(
    report: dict[str, Any],
    *,
    candidate: dict[str, Any],
    report_path: str | Path | None,
    preserve_paths: bool,
) -> dict[str, Any]:
    errors = model_compatibility_report_errors(report)
    if report.get("candidate_id") != candidate.get("candidate_id"):
        errors.append("compatibility_report.candidate_id must match selected candidate.")
    if report.get("model_id") != candidate.get("model_id"):
        errors.append("compatibility_report.model_id must match selected candidate.")
    if report.get("passed") is not True:
        errors.append("compatibility_report.passed must be true for serving-probe receipts.")
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    record = {
        "path": str(report.get("report_path") or ""),
        "candidate_id": report.get("candidate_id"),
        "model_id": report.get("model_id"),
        "passed": report.get("passed"),
        "readiness": report.get("readiness"),
        "recommendation": report.get("recommendation"),
        "probe_count": summary.get("probe_count"),
        "verified_count": summary.get("verified_count"),
        "metadata_only_count": summary.get("metadata_only_count"),
        "missing_count": summary.get("missing_count"),
        "unsupported_count": summary.get("unsupported_count"),
        "download_policy": copy.deepcopy(report.get("download_policy")),
    }
    if report_path is not None and str(report_path):
        source_path = Path(report_path)
        if not source_path.exists() or not source_path.is_file() or source_path.is_symlink():
            errors.append(f"compatibility_report_path must be an existing regular file: {source_path}")
        else:
            record["path"] = _display_path(source_path, preserve_paths)
            record["sha256"] = _sha256(source_path)
    if errors:
        raise ModelRegistryError("; ".join(errors))
    return record


def _serving_probe_compatibility_report_errors(
    report: dict[str, Any],
    receipt: dict[str, Any],
    errors: list[str],
) -> None:
    for field_name in ("path", "candidate_id", "model_id", "readiness", "recommendation"):
        _require_non_empty_string(report, field_name, errors, "model_serving_probe_receipt.compatibility_report.")
    for field_name in ("probe_count", "verified_count", "metadata_only_count", "missing_count", "unsupported_count"):
        if not _is_non_negative_int(report.get(field_name)):
            errors.append(f"model_serving_probe_receipt.compatibility_report.{field_name} must be a non-negative integer.")
    if report.get("passed") is not True:
        errors.append("model_serving_probe_receipt.compatibility_report.passed must be true.")
    if "sha256" in report and not _is_sha256(report.get("sha256")):
        errors.append("model_serving_probe_receipt.compatibility_report.sha256 must be a 64-character hex digest.")
    if report.get("candidate_id") != receipt.get("candidate_id"):
        errors.append("model_serving_probe_receipt.compatibility_report.candidate_id must match receipt.candidate_id.")
    if report.get("model_id") != receipt.get("model_id"):
        errors.append("model_serving_probe_receipt.compatibility_report.model_id must match receipt.model_id.")
    policy = report.get("download_policy")
    if not isinstance(policy, dict):
        errors.append("model_serving_probe_receipt.compatibility_report.download_policy must be an object.")
    else:
        for field_name in ("downloaded_weights", "downloaded_tokenizer", "imported_heavy_ml", "gpu_execution"):
            if policy.get(field_name) is not False:
                errors.append(f"model_serving_probe_receipt.compatibility_report.download_policy.{field_name} must be false.")


def _serving_probe_from_compatibility(
    probe_id: str,
    compatibility_field: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return _serving_probe(
        probe_id,
        status=str(metadata.get("status") or "metadata_only"),
        verified=metadata.get("verified") is True,
        supported=metadata.get("supported") if isinstance(metadata.get("supported"), bool) else None,
        source=f"model_candidate.compatibility.{compatibility_field}",
        notes=str(metadata.get("notes") or f"{compatibility_field} serving metadata recorded."),
        metadata=metadata,
    )


def _serving_context_probe(context_length: int) -> dict[str, Any]:
    return _serving_probe(
        "context",
        status="metadata_only",
        verified=False,
        supported=True,
        source="model_candidate.compatibility.context_length",
        notes=f"Recorded context length is {context_length} tokens; endpoint context handling is not verified.",
        metadata={"context_length": context_length},
    )


def _serving_probe(
    probe_id: str,
    *,
    status: str,
    verified: bool,
    supported: bool | None,
    source: str,
    notes: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": probe_id,
        "status": status,
        "verified": verified,
        "supported": supported,
        "source": source,
        "notes": notes,
        "metadata": copy.deepcopy(metadata),
    }


def _training_compatibility_report_record(
    report: dict[str, Any],
    *,
    candidate: dict[str, Any],
    report_path: str | Path | None,
    preserve_paths: bool,
) -> dict[str, Any]:
    errors = model_compatibility_report_errors(report)
    if report.get("candidate_id") != candidate.get("candidate_id"):
        errors.append("compatibility_report.candidate_id must match selected candidate.")
    if report.get("model_id") != candidate.get("model_id"):
        errors.append("compatibility_report.model_id must match selected candidate.")
    if report.get("passed") is not True:
        errors.append("compatibility_report.passed must be true for dry-run training plans.")
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    record = {
        "path": str(report.get("report_path") or ""),
        "candidate_id": report.get("candidate_id"),
        "model_id": report.get("model_id"),
        "passed": report.get("passed"),
        "readiness": report.get("readiness"),
        "recommendation": report.get("recommendation"),
        "probe_count": summary.get("probe_count"),
        "verified_count": summary.get("verified_count"),
        "metadata_only_count": summary.get("metadata_only_count"),
        "missing_count": summary.get("missing_count"),
        "unsupported_count": summary.get("unsupported_count"),
        "download_policy": copy.deepcopy(report.get("download_policy")),
    }
    if report_path is not None and str(report_path):
        source_path = Path(report_path)
        if not source_path.exists() or not source_path.is_file() or source_path.is_symlink():
            errors.append(f"compatibility_report_path must be an existing regular file: {source_path}")
        else:
            record["path"] = _display_path(source_path, preserve_paths)
            record["sha256"] = _sha256(source_path)
    if errors:
        raise ModelRegistryError("; ".join(errors))
    return record


def _training_plan_compatibility_report_errors(report: dict[str, Any], model: dict[str, Any], errors: list[str]) -> None:
    for field_name in ("path", "candidate_id", "model_id", "readiness", "recommendation"):
        _require_non_empty_string(report, field_name, errors, "training_plan.compatibility_report.")
    for field_name in ("probe_count", "verified_count", "metadata_only_count", "missing_count", "unsupported_count"):
        if not _is_non_negative_int(report.get(field_name)):
            errors.append(f"training_plan.compatibility_report.{field_name} must be a non-negative integer.")
    if report.get("passed") is not True:
        errors.append("training_plan.compatibility_report.passed must be true.")
    if "sha256" in report and not _is_sha256(report.get("sha256")):
        errors.append("training_plan.compatibility_report.sha256 must be a 64-character hex digest.")
    if report.get("candidate_id") != model.get("candidate_id"):
        errors.append("training_plan.compatibility_report.candidate_id must match training_plan.model.candidate_id.")
    if report.get("model_id") != model.get("model_id"):
        errors.append("training_plan.compatibility_report.model_id must match training_plan.model.model_id.")
    policy = report.get("download_policy")
    if not isinstance(policy, dict):
        errors.append("training_plan.compatibility_report.download_policy must be an object.")
    else:
        for field_name in ("downloaded_weights", "downloaded_tokenizer", "imported_heavy_ml", "gpu_execution"):
            if policy.get(field_name) is not False:
                errors.append(f"training_plan.compatibility_report.download_policy.{field_name} must be false.")


def _compatibility_probe(probe_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
    status = str(metadata.get("status") or "recorded")
    return {
        "id": probe_id,
        "status": status,
        "verified": metadata.get("verified") is True,
        "supported": metadata.get("supported") if isinstance(metadata.get("supported"), bool) else None,
        "source": "model_candidate.compatibility",
        "notes": str(metadata.get("notes") or f"{probe_id} compatibility metadata recorded."),
        "metadata": copy.deepcopy(metadata),
    }


def _context_probe(context_length: int) -> dict[str, Any]:
    return {
        "id": "context",
        "status": "recorded",
        "verified": False,
        "supported": True,
        "source": "model_candidate.compatibility.context_length",
        "notes": f"Recorded context length is {context_length} tokens.",
        "metadata": {"context_length": context_length},
    }


def _registry_with_defaults(registry: dict[str, Any]) -> dict[str, Any]:
    next_registry = copy.deepcopy(registry) if isinstance(registry, dict) else new_model_registry()
    next_registry.setdefault("schema_version", MODEL_REGISTRY_SCHEMA_VERSION)
    next_registry.setdefault("registry_path", "experiments/registry/model_registry.json")
    next_registry.setdefault("updated_at", _now_iso())
    next_registry.setdefault("entries", {})
    for entry in next_registry["entries"].values():
        if isinstance(entry, dict):
            entry["links"] = _model_registry_links_with_defaults(entry.get("links"))
    aliases = next_registry.setdefault("aliases", {})
    for alias in ALIAS_NAMES:
        aliases.setdefault(alias, None)
    next_registry.setdefault("alias_history", [])
    next_registry.setdefault("notes", [])
    return next_registry


def _set_alias(registry: dict[str, Any], alias: str, target: str, moved_at: str, reason: str) -> None:
    previous = registry["aliases"].get(alias)
    registry["aliases"][alias] = target
    registry["alias_history"].append(
        {"alias": alias, "previous": previous, "target": target, "moved_at": moved_at, "reason": reason}
    )


def _require_registered_target(registry: dict[str, Any], target: str) -> None:
    if not isinstance(target, str) or not target:
        raise ModelRegistryError("alias target must be a non-empty entry id")
    if target not in registry.get("entries", {}):
        raise ModelRegistryError(f"alias target {target!r} is not registered")


def _model_registry_links_errors(links: Any, errors: list[str]) -> None:
    if not isinstance(links, dict):
        errors.append("model_registry_entry.links must be an object.")
        return
    for collection in MODEL_REGISTRY_LINK_COLLECTIONS:
        records = links.get(collection)
        if not isinstance(records, list):
            errors.append(f"model_registry_entry.links.{collection} must be a list.")
            continue
        seen: set[tuple[str, str]] = set()
        for index, record in enumerate(records):
            prefix = f"model_registry_entry.links.{collection}[{index}]."
            if not isinstance(record, dict):
                errors.append(f"model_registry_entry.links.{collection}[{index}] must be an object.")
                continue
            for field_name in ("id", "kind", "status", "recorded_at"):
                _require_non_empty_string(record, field_name, errors, prefix)
            record_key = (str(record.get("id") or ""), str(record.get("kind") or ""))
            if record_key in seen:
                errors.append(f"{prefix}id and kind duplicate an earlier link record.")
            seen.add(record_key)
            if "path" in record:
                _require_non_empty_string(record, "path", errors, prefix)
            if "sha256" in record and not _is_sha256(str(record.get("sha256") or "").lower()):
                errors.append(f"{prefix}sha256 must be a 64-character hex digest.")
            if not isinstance(record.get("metadata", {}), dict):
                errors.append(f"{prefix}metadata must be an object when present.")


def _model_registry_links_with_defaults(links: Any) -> dict[str, list[dict[str, Any]]]:
    normalized: dict[str, list[dict[str, Any]]] = {collection: [] for collection in MODEL_REGISTRY_LINK_COLLECTIONS}
    if not isinstance(links, dict):
        return normalized
    for collection in MODEL_REGISTRY_LINK_COLLECTIONS:
        records = links.get(collection)
        if isinstance(records, list):
            normalized[collection] = [copy.deepcopy(record) for record in records if isinstance(record, dict)]
    return normalized


def _model_registry_link_counts(links: Any) -> dict[str, int]:
    normalized = _model_registry_links_with_defaults(links)
    return {collection: len(records) for collection, records in normalized.items()}


def _upsert_link_record(records: list[dict[str, Any]], record: dict[str, Any]) -> None:
    for index, existing in enumerate(records):
        if existing.get("id") == record["id"] and existing.get("kind") == record["kind"]:
            records[index] = record
            return
    records.append(record)


def _license_errors(license_review: dict[str, Any], errors: list[str], prefix: str) -> None:
    for field_name in ("status", "license_id", "source_url", "review_status", "terms_url"):
        _require_non_empty_string(license_review, field_name, errors, prefix)
    if not isinstance(license_review.get("accepted_terms"), bool):
        errors.append(f"{prefix}accepted_terms must be a boolean.")
    if not isinstance(license_review.get("training_allowed"), bool):
        errors.append(f"{prefix}training_allowed must be a boolean.")
    if _normalized(license_review.get("review_status")) == "approved":
        for field_name in ("reviewed_at", "reviewer"):
            _require_non_empty_string(license_review, field_name, errors, prefix)
    if license_review.get("notes") is not None and not _is_string_list(license_review.get("notes")):
        errors.append(f"{prefix}notes must be a list of strings when present.")


def _training_license_errors(license_review: dict[str, Any], errors: list[str], prefix: str) -> None:
    status = _normalized(license_review.get("status"))
    review_status = _normalized(license_review.get("review_status"))
    if status not in TRAINING_APPROVED_LICENSE_STATUSES:
        errors.append(f"{prefix}status {status or '<missing>'!r} is not approved for training.")
    if review_status not in TRAINING_APPROVED_REVIEW_STATUSES:
        errors.append(f"{prefix}review_status {review_status or '<missing>'!r} is not approved for training.")
    if license_review.get("accepted_terms") is not True:
        errors.append(f"{prefix}accepted_terms must be true for training selection.")
    if license_review.get("training_allowed") is not True:
        errors.append(f"{prefix}training_allowed must be true for training selection.")


def _require_equal(value: dict[str, Any], field_name: str, expected: Any, errors: list[str], prefix: str) -> None:
    if value.get(field_name) != expected:
        errors.append(f"{prefix}{field_name} must be {expected!r}.")


def _require_non_empty_string(value: dict[str, Any], field_name: str, errors: list[str], prefix: str) -> None:
    if not isinstance(value.get(field_name), str) or not value.get(field_name):
        errors.append(f"{prefix}{field_name} must be a non-empty string.")


def _require_bool_if_present(value: Any, field_name: str, errors: list[str], prefix: str) -> None:
    if isinstance(value, dict) and field_name in value and not isinstance(value.get(field_name), bool):
        errors.append(f"{prefix}{field_name} must be a boolean when present.")


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _normalized(value: Any) -> str:
    return str(value or "").strip().lower()


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path, preserve_paths: bool) -> str:
    return str(path.resolve()) if preserve_paths else str(path)
