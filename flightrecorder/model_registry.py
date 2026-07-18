"""Local model candidate, registry, and dry-run training-plan contracts."""

from __future__ import annotations

import copy
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODEL_CANDIDATE_SCHEMA_VERSION = "hfr.model_candidate.v1"
MODEL_SCOUT_MANIFEST_SCHEMA_VERSION = "hfr.model_scout_manifest.v1"
MODEL_REGISTRY_ENTRY_SCHEMA_VERSION = "hfr.model_registry_entry.v1"
MODEL_REGISTRY_SCHEMA_VERSION = "hfr.model_registry.v1"
MODEL_COMPATIBILITY_REPORT_SCHEMA_VERSION = "hfr.model_compatibility_report.v1"
TRAINING_PLAN_SCHEMA_VERSION = "hfr.training_plan.v1"

ALIAS_NAMES = ("candidate", "champion", "rollback")
TRAINING_APPROVED_LICENSE_STATUSES = {"approved"}
TRAINING_APPROVED_REVIEW_STATUSES = {"approved"}
LINK_FIELDS = ("datasets", "training_runs", "adapters", "evals", "promotion_decisions")
LINK_TYPE_ALIASES = {
    "dataset": "datasets",
    "datasets": "datasets",
    "training-run": "training_runs",
    "training_run": "training_runs",
    "training-runs": "training_runs",
    "training_runs": "training_runs",
    "adapter": "adapters",
    "adapters": "adapters",
    "eval": "evals",
    "evals": "evals",
    "promotion-decision": "promotion_decisions",
    "promotion_decision": "promotion_decisions",
    "promotion-decisions": "promotion_decisions",
    "promotion_decisions": "promotion_decisions",
}
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


class ModelRegistryError(ValueError):
    """Raised when model-layer artifacts cannot be safely used."""


def model_candidate_errors(candidate: Any, *, require_training_eligible: bool = False) -> list[str]:
    """Return semantic validation errors for a model-candidate artifact."""
    errors: list[str] = []
    if not isinstance(candidate, dict):
        return ["model candidate must be a JSON object"]
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
        context_length = compatibility.get("context_length")
        if not _is_positive_int(context_length):
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
    return errors


def validate_model_candidate(candidate: Any, *, require_training_eligible: bool = False) -> dict[str, Any]:
    """Validate and return a model-candidate artifact."""
    errors = model_candidate_errors(candidate, require_training_eligible=require_training_eligible)
    if errors:
        raise ModelRegistryError("; ".join(errors))
    return copy.deepcopy(candidate)


def model_scout_manifest_errors(manifest: Any) -> list[str]:
    """Return portable shape errors for a model-scout manifest."""
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
    seen_candidate_ids: set[str] = set()
    for index, item in enumerate(candidates):
        label = f"model_scout_manifest.candidates[{index}]."
        if not isinstance(item, dict):
            errors.append(f"model_scout_manifest.candidates[{index}] must be an object.")
            continue
        for field_name in ("candidate_id", "manifest_path", "priority", "reason"):
            _require_non_empty_string(item, field_name, errors, label)
        candidate_id = item.get("candidate_id")
        if isinstance(candidate_id, str) and candidate_id:
            if candidate_id in seen_candidate_ids:
                errors.append(f"{label}candidate_id duplicates {candidate_id!r}.")
            seen_candidate_ids.add(candidate_id)
        for field_name in ("compatibility_report_path", "registry_entry_id", "license_status", "review_status"):
            if field_name in item:
                _require_non_empty_string(item, field_name, errors, label)
        _require_bool_if_present(item, "training_selection_eligible", errors, label)
        notes = item.get("notes")
        if notes is not None and not _is_string_list(notes):
            errors.append(f"{label}notes must be a list of strings when present.")
    notes = manifest.get("notes")
    if notes is not None and not _is_string_list(notes):
        errors.append("model_scout_manifest.notes must be a list of strings when present.")
    return errors


def validate_model_scout_manifest(manifest: Any) -> dict[str, Any]:
    """Validate and return a model-scout manifest artifact."""
    errors = model_scout_manifest_errors(manifest)
    if errors:
        raise ModelRegistryError("; ".join(errors))
    return copy.deepcopy(manifest)


def model_registry_entry_errors(entry: Any) -> list[str]:
    """Return semantic validation errors for one registry entry."""
    errors: list[str] = []
    if not isinstance(entry, dict):
        return ["model_registry_entry must be a JSON object"]
    _require_equal(entry, "schema_version", MODEL_REGISTRY_ENTRY_SCHEMA_VERSION, errors, "model_registry_entry.")
    for field_name in ("entry_id", "candidate_id", "registered_at", "status"):
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
    for field_name in LINK_FIELDS:
        _link_list_errors(entry.get(field_name), errors, f"model_registry_entry.{field_name}")
    notes = entry.get("notes")
    if notes is not None and not _is_string_list(notes):
        errors.append("model_registry_entry.notes must be a list of strings when present.")
    return errors


def validate_model_registry_entry(entry: Any) -> dict[str, Any]:
    """Validate and return a model-registry-entry artifact."""
    errors = model_registry_entry_errors(entry)
    if errors:
        raise ModelRegistryError("; ".join(errors))
    return copy.deepcopy(entry)


def new_model_registry(*, registry_path: str | Path = "experiments/registry/model_registry.json") -> dict[str, Any]:
    """Build an empty local model registry artifact."""
    return {
        "schema_version": MODEL_REGISTRY_SCHEMA_VERSION,
        "registry_path": str(registry_path),
        "updated_at": _now_iso(),
        "entries": {},
        "aliases": {alias: None for alias in ALIAS_NAMES},
        "alias_history": [],
        "notes": [],
    }


def model_registry_errors(registry: Any) -> list[str]:
    """Return semantic validation errors for a registry container."""
    errors: list[str] = []
    if not isinstance(registry, dict):
        return ["model_registry must be a JSON object"]
    _require_equal(registry, "schema_version", MODEL_REGISTRY_SCHEMA_VERSION, errors, "model_registry.")
    _require_non_empty_string(registry, "registry_path", errors, "model_registry.")
    _require_non_empty_string(registry, "updated_at", errors, "model_registry.")
    entries = registry.get("entries")
    if not isinstance(entries, dict):
        errors.append("model_registry.entries must be an object keyed by entry_id.")
        entries = {}
    for entry_id, entry in entries.items():
        if not isinstance(entry_id, str) or not entry_id:
            errors.append("model_registry.entries keys must be non-empty strings.")
            continue
        entry_errors = model_registry_entry_errors(entry)
        errors.extend(error.replace("model_registry_entry.", f"model_registry.entries.{entry_id}.") for error in entry_errors)
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
    history = registry.get("alias_history")
    if not isinstance(history, list):
        errors.append("model_registry.alias_history must be a list.")
    else:
        for index, item in enumerate(history):
            if not isinstance(item, dict):
                errors.append(f"model_registry.alias_history[{index}] must be an object.")
                continue
            for field_name in ("alias", "target", "moved_at"):
                _require_non_empty_string(item, field_name, errors, f"model_registry.alias_history[{index}].")
            if item.get("alias") not in ALIAS_NAMES:
                errors.append(f"model_registry.alias_history[{index}].alias must be one of {ALIAS_NAMES!r}.")
    notes = registry.get("notes")
    if notes is not None and not _is_string_list(notes):
        errors.append("model_registry.notes must be a list of strings when present.")
    return errors


def validate_model_registry(registry: Any) -> dict[str, Any]:
    """Validate and return a registry artifact."""
    errors = model_registry_errors(registry)
    if errors:
        raise ModelRegistryError("; ".join(errors))
    return copy.deepcopy(registry)


def load_model_registry(path: str | Path) -> dict[str, Any]:
    """Load a registry or return an empty registry for a new path."""
    registry_path = Path(path)
    if not registry_path.exists():
        return new_model_registry(registry_path=registry_path)
    import json

    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    validate_model_registry(payload)
    return payload


def register_model_candidate(
    registry: dict[str, Any],
    candidate: dict[str, Any],
    *,
    status: str = "registered",
    registered_at: str | None = None,
) -> dict[str, Any]:
    """Insert or update a candidate entry while preserving lifecycle links."""
    validate_model_candidate(candidate)
    next_registry = _registry_with_defaults(registry)
    candidate_id = str(candidate["candidate_id"])
    existing = next_registry["entries"].get(candidate_id) if isinstance(next_registry["entries"].get(candidate_id), dict) else {}
    candidate_snapshot = copy.deepcopy(candidate)
    license_review = candidate_snapshot.get("license") if isinstance(candidate_snapshot.get("license"), dict) else {}
    entry = {
        "schema_version": MODEL_REGISTRY_ENTRY_SCHEMA_VERSION,
        "entry_id": candidate_id,
        "candidate_id": candidate_id,
        "registered_at": str(existing.get("registered_at") or registered_at or _now_iso()),
        "updated_at": _now_iso(),
        "status": status,
        "training_eligible": is_training_license_approved(candidate_snapshot),
        "license_status": str(license_review.get("status") or ""),
        "candidate": candidate_snapshot,
        "datasets": _copy_link_list(existing.get("datasets")),
        "training_runs": _copy_link_list(existing.get("training_runs")),
        "adapters": _copy_link_list(existing.get("adapters")),
        "evals": _copy_link_list(existing.get("evals")),
        "promotion_decisions": _copy_link_list(existing.get("promotion_decisions")),
        "notes": list(existing.get("notes") if _is_string_list(existing.get("notes")) else []),
    }
    validate_model_registry_entry(entry)
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
    moved_at: str | None = None,
) -> dict[str, Any]:
    """Move a registry alias, requiring explicit rollback for champion moves."""
    if alias not in ALIAS_NAMES:
        raise ModelRegistryError(f"alias must be one of {', '.join(ALIAS_NAMES)}")
    next_registry = _registry_with_defaults(registry)
    _require_registered_target(next_registry, target)
    now = moved_at or _now_iso()
    if alias == "champion":
        if not rollback_target:
            raise ModelRegistryError("moving champion requires explicit --rollback-target")
        _require_registered_target(next_registry, rollback_target)
        if rollback_target == target:
            raise ModelRegistryError("rollback target must differ from champion target")
        _set_alias(next_registry, "rollback", rollback_target, now, reason or "set rollback before champion move")
    _set_alias(next_registry, alias, target, now, reason)
    next_registry["updated_at"] = _now_iso()
    validate_model_registry(next_registry)
    return next_registry


def list_model_registry_entries(registry: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a stable summary of registry entries and aliases."""
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
            }
        )
    return rows


def add_model_registry_link(
    registry: dict[str, Any],
    *,
    entry_ref: str,
    link_type: str,
    artifact_id: str = "",
    path: str | Path = "",
    kind: str = "",
    metadata: dict[str, Any] | None = None,
    note: str = "",
    linked_at: str | None = None,
) -> dict[str, Any]:
    """Attach or update a lifecycle artifact link on one registry entry."""
    next_registry = _registry_with_defaults(registry)
    field_name = _normalize_link_type(link_type)
    entry_id = _resolve_entry_id(next_registry, entry_ref)
    artifact_path = str(path) if str(path) else ""
    if not artifact_id and not artifact_path:
        raise ModelRegistryError("registry links require --artifact-id or --path")

    entry = copy.deepcopy(next_registry["entries"][entry_id])
    link = {
        "kind": kind or field_name[:-1],
        "artifact_id": artifact_id,
        "path": artifact_path,
        "linked_at": linked_at or _now_iso(),
        "metadata": dict(metadata or {}),
    }
    if note:
        link["note"] = note
    link = {key: value for key, value in link.items() if value not in ("", None) and value != {}}

    links = _copy_link_list(entry.get(field_name))
    replaced = False
    for index, existing in enumerate(links):
        if _same_link(existing, link):
            links[index] = {**(existing if isinstance(existing, dict) else {}), **link}
            replaced = True
            break
    if not replaced:
        links.append(link)
    entry[field_name] = links
    entry["updated_at"] = _now_iso()
    validate_model_registry_entry(entry)

    next_registry["entries"][entry_id] = entry
    next_registry["updated_at"] = _now_iso()
    validate_model_registry(next_registry)
    return next_registry


def resolve_model_ref(registry: dict[str, Any], model_ref: str) -> dict[str, Any]:
    """Resolve a registry entry id or alias to an entry."""
    validate_model_registry(registry)
    if not isinstance(model_ref, str) or not model_ref:
        raise ModelRegistryError("model reference must be a non-empty string")
    aliases = registry.get("aliases") if isinstance(registry.get("aliases"), dict) else {}
    target = aliases.get(model_ref, model_ref)
    if target is None:
        raise ModelRegistryError(f"model alias {model_ref!r} is not set")
    entries = registry.get("entries") if isinstance(registry.get("entries"), dict) else {}
    entry = entries.get(target)
    if not isinstance(entry, dict):
        raise ModelRegistryError(f"model reference {model_ref!r} does not resolve to a registered entry")
    return copy.deepcopy(entry)


def select_model_for_training(registry: dict[str, Any], model_ref: str) -> dict[str, Any]:
    """Resolve a model reference and enforce training license eligibility."""
    entry = resolve_model_ref(registry, model_ref)
    candidate = entry.get("candidate")
    validate_model_candidate(candidate, require_training_eligible=True)
    if entry.get("training_eligible") is not True:
        raise ModelRegistryError(f"model entry {entry.get('entry_id')!r} is not training eligible")
    return entry


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
    created_at: str | None = None,
    preserve_paths: bool = False,
) -> dict[str, Any]:
    """Build a side-effect-free dry-run training plan without model downloads or GPU work."""
    entry = select_model_for_training(registry, model_ref)
    candidate = entry["candidate"]
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ModelRegistryError("dataset_id must be a non-empty string")
    dataset_path = Path(dataset_manifest)
    if not str(dataset_manifest):
        raise ModelRegistryError("dataset_manifest must be a non-empty path")
    if not dataset_path.exists() or not dataset_path.is_file() or dataset_path.is_symlink():
        raise ModelRegistryError(f"dataset_manifest must be an existing regular file: {dataset_path}")
    if not isinstance(trainer, str) or not trainer:
        raise ModelRegistryError("trainer must be a non-empty string")
    if not isinstance(mode, str) or not mode:
        raise ModelRegistryError("mode must be a non-empty string")
    if not str(output_dir):
        raise ModelRegistryError("output_dir must be a non-empty path")
    compatibility_report_record = None
    if compatibility_report is not None:
        compatibility_report_record = _training_compatibility_report_record(
            compatibility_report,
            candidate=candidate,
            report_path=compatibility_report_path,
            preserve_paths=preserve_paths,
        )

    checks = [
        {
            "id": "license_training_approved",
            "passed": True,
            "summary": "candidate license review is approved for training",
        },
        {
            "id": "dataset_manifest_exists",
            "passed": True,
            "summary": "dataset manifest exists and was fingerprinted",
        },
    ]
    if compatibility_report_record is not None:
        checks.append(
            {
                "id": "compatibility_report_passed",
                "passed": True,
                "summary": "compatibility report matches the selected model and passed no-download checks",
            }
        )
    checks.extend(
        [
            {
                "id": "no_weight_download",
                "passed": True,
                "summary": "plan generation did not download model weights or tokenizers",
            },
            {
                "id": "no_gpu_execution",
                "passed": True,
                "summary": "plan generation did not launch GPU work",
            },
        ]
    )

    plan = {
        "schema_version": TRAINING_PLAN_SCHEMA_VERSION,
        "plan_path": _display_path(Path(out_path), preserve_paths),
        "created_at": created_at or _now_iso(),
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
        "trainer": {
            "name": trainer,
            "mode": mode,
            "hyperparameters": dict(hyperparameters or {}),
        },
        "output": {
            "output_dir": _display_path(Path(output_dir), preserve_paths),
        },
        "compute": {
            "assumptions": dict(compute or {}),
            "dry_run_only": True,
            "accelerator_required": False,
        },
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
    if compatibility_report_record is not None:
        plan["compatibility_report"] = compatibility_report_record
    errors = training_plan_errors(plan)
    if errors:
        raise ModelRegistryError("; ".join(errors))
    return plan


def build_model_compatibility_report(
    candidate: dict[str, Any],
    *,
    out_path: str | Path,
    generated_at: str | None = None,
    preserve_paths: bool = False,
) -> dict[str, Any]:
    """Build a no-download compatibility report from recorded candidate metadata."""
    validate_model_candidate(candidate)
    compatibility = candidate["compatibility"]
    probes = [_context_probe(compatibility["context_length"])]
    probes.extend(_compatibility_probe(field_name, compatibility[field_name]) for field_name in COMPATIBILITY_FIELDS)
    missing_count = sum(1 for probe in probes if probe["status"] == "missing")
    verified_count = sum(1 for probe in probes if probe["verified"] is True)
    metadata_only_count = sum(1 for probe in probes if probe["status"] == "metadata_only")
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
            "This report records compatibility metadata without downloading weights or tokenizers.",
            "Future serving workers should replace metadata_only probes with verified serving/tokenizer checks.",
        ],
    }
    errors = model_compatibility_report_errors(report)
    if errors:
        raise ModelRegistryError("; ".join(errors))
    return report


def model_compatibility_report_errors(report: Any) -> list[str]:
    """Return semantic validation errors for a model compatibility report."""
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
    expected_probe_ids = set(COMPATIBILITY_REPORT_PROBE_IDS)
    if seen != expected_probe_ids:
        errors.append(f"model_compatibility_report.probes must contain exactly {sorted(expected_probe_ids)!r}.")
    summary = report.get("summary")
    if not isinstance(summary, dict):
        errors.append("model_compatibility_report.summary must be an object.")
    else:
        for field_name in ("probe_count", "missing_count", "verified_count", "metadata_only_count", "unsupported_count", "context_length"):
            if not isinstance(summary.get(field_name), int) or isinstance(summary.get(field_name), bool) or summary.get(field_name) < 0:
                errors.append(f"model_compatibility_report.summary.{field_name} must be a non-negative integer.")
        if isinstance(summary.get("probe_count"), int) and summary.get("probe_count") != len(probes):
            errors.append("model_compatibility_report.summary.probe_count must match probes length.")
    notes = report.get("notes")
    if notes is not None and not _is_string_list(notes):
        errors.append("model_compatibility_report.notes must be a list of strings when present.")
    return errors


def validate_model_compatibility_report(report: Any) -> dict[str, Any]:
    """Validate and return a model compatibility report artifact."""
    errors = model_compatibility_report_errors(report)
    if errors:
        raise ModelRegistryError("; ".join(errors))
    return copy.deepcopy(report)


def training_plan_errors(plan: Any) -> list[str]:
    """Return semantic validation errors for a dry-run training plan."""
    errors: list[str] = []
    if not isinstance(plan, dict):
        return ["training_plan must be a JSON object"]
    _require_equal(plan, "schema_version", TRAINING_PLAN_SCHEMA_VERSION, errors, "training_plan.")
    for field_name in ("plan_path", "created_at", "readiness", "recommendation"):
        _require_non_empty_string(plan, field_name, errors, "training_plan.")
    if plan.get("dry_run") is not True:
        errors.append("training_plan.dry_run must be true.")
    if plan.get("no_weight_download") is not True:
        errors.append("training_plan.no_weight_download must be true.")
    if plan.get("gpu_execution") is not False:
        errors.append("training_plan.gpu_execution must be false.")
    if plan.get("passed") is not True:
        errors.append("training_plan.passed must be true for a usable dry-run plan.")
    model = plan.get("model")
    if not isinstance(model, dict):
        errors.append("training_plan.model must be an object.")
    else:
        for field_name in ("model_ref", "entry_id", "candidate_id", "model_id"):
            _require_non_empty_string(model, field_name, errors, "training_plan.model.")
        candidate = model.get("candidate")
        candidate_errors = model_candidate_errors(candidate, require_training_eligible=True)
        errors.extend(error.replace("model_candidate.", "training_plan.model.candidate.") for error in candidate_errors)
    report = plan.get("compatibility_report")
    if report is not None:
        if not isinstance(report, dict):
            errors.append("training_plan.compatibility_report must be an object when present.")
        else:
            _training_plan_compatibility_report_errors(report, model if isinstance(model, dict) else {}, errors)
    dataset = plan.get("dataset")
    if not isinstance(dataset, dict):
        errors.append("training_plan.dataset must be an object.")
    else:
        for field_name in ("dataset_id", "manifest_path", "manifest_sha256"):
            _require_non_empty_string(dataset, field_name, errors, "training_plan.dataset.")
        if isinstance(dataset.get("manifest_sha256"), str) and not _is_sha256(dataset.get("manifest_sha256")):
            errors.append("training_plan.dataset.manifest_sha256 must be a 64-character hex digest.")
    trainer = plan.get("trainer")
    if not isinstance(trainer, dict):
        errors.append("training_plan.trainer must be an object.")
    else:
        for field_name in ("name", "mode"):
            _require_non_empty_string(trainer, field_name, errors, "training_plan.trainer.")
        if not isinstance(trainer.get("hyperparameters"), dict):
            errors.append("training_plan.trainer.hyperparameters must be an object.")
    output = plan.get("output")
    if not isinstance(output, dict):
        errors.append("training_plan.output must be an object.")
    else:
        _require_non_empty_string(output, "output_dir", errors, "training_plan.output.")
    compute = plan.get("compute")
    if not isinstance(compute, dict):
        errors.append("training_plan.compute must be an object.")
    elif compute.get("accelerator_required") is not False:
        errors.append("training_plan.compute.accelerator_required must be false for dry-run plans.")
    execution = plan.get("execution")
    if not isinstance(execution, dict):
        errors.append("training_plan.execution must be an object.")
    else:
        for field_name in (
            "flight_recorder_downloaded_weights",
            "flight_recorder_downloaded_tokenizer",
            "flight_recorder_imported_heavy_ml",
            "flight_recorder_launched_gpu_job",
        ):
            if execution.get(field_name) is not False:
                errors.append(f"training_plan.execution.{field_name} must be false.")
    checks = plan.get("checks")
    if not isinstance(checks, list):
        errors.append("training_plan.checks must be a list.")
    else:
        for index, check in enumerate(checks):
            if not isinstance(check, dict):
                errors.append(f"training_plan.checks[{index}] must be an object.")
                continue
            _require_non_empty_string(check, "id", errors, f"training_plan.checks[{index}].")
            if check.get("passed") is not True:
                errors.append(f"training_plan.checks[{index}].passed must be true.")
    blocked = plan.get("blocked_reasons")
    if blocked is not None and not _is_string_list(blocked):
        errors.append("training_plan.blocked_reasons must be a list of strings when present.")
    return errors


def validate_training_plan(plan: Any) -> dict[str, Any]:
    """Validate and return a dry-run training-plan artifact."""
    errors = training_plan_errors(plan)
    if errors:
        raise ModelRegistryError("; ".join(errors))
    return copy.deepcopy(plan)


def is_training_license_approved(candidate: dict[str, Any]) -> bool:
    """Return whether a candidate's license review can be selected for training."""
    license_review = candidate.get("license") if isinstance(candidate, dict) else None
    if not isinstance(license_review, dict):
        return False
    status = _normalized(license_review.get("status"))
    review_status = _normalized(license_review.get("review_status"))
    return (
        status in TRAINING_APPROVED_LICENSE_STATUSES
        and review_status in TRAINING_APPROVED_REVIEW_STATUSES
        and license_review.get("accepted_terms") is True
        and license_review.get("training_allowed") is True
    )


def _training_compatibility_report_record(
    report: dict[str, Any],
    *,
    candidate: dict[str, Any],
    report_path: str | Path | None,
    preserve_paths: bool,
) -> dict[str, Any]:
    validate_model_compatibility_report(report)
    errors: list[str] = []
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
    if not record["path"]:
        errors.append("compatibility_report.path must be recorded.")
    if errors:
        raise ModelRegistryError("; ".join(errors))
    return record


def _training_plan_compatibility_report_errors(
    report: dict[str, Any],
    model: dict[str, Any],
    errors: list[str],
) -> None:
    for field_name in ("path", "candidate_id", "model_id", "readiness", "recommendation"):
        _require_non_empty_string(report, field_name, errors, "training_plan.compatibility_report.")
    for field_name in ("probe_count", "verified_count", "metadata_only_count", "missing_count", "unsupported_count"):
        if not isinstance(report.get(field_name), int) or isinstance(report.get(field_name), bool) or report.get(field_name) < 0:
            errors.append(f"training_plan.compatibility_report.{field_name} must be a non-negative integer.")
    if report.get("passed") is not True:
        errors.append("training_plan.compatibility_report.passed must be true.")
    if "sha256" in report and not _is_sha256(report.get("sha256")):
        errors.append("training_plan.compatibility_report.sha256 must be a 64-character hex digest.")
    if model.get("candidate_id") and report.get("candidate_id") != model.get("candidate_id"):
        errors.append("training_plan.compatibility_report.candidate_id must match training_plan.model.candidate_id.")
    if model.get("model_id") and report.get("model_id") != model.get("model_id"):
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
    verified = metadata.get("verified") is True
    supported = metadata.get("supported") if isinstance(metadata.get("supported"), bool) else None
    notes = str(metadata.get("notes") or f"{probe_id} compatibility metadata recorded.")
    return {
        "id": probe_id,
        "status": status,
        "verified": verified,
        "supported": supported,
        "source": "model_candidate.compatibility",
        "notes": notes,
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
        {
            "alias": alias,
            "previous": previous,
            "target": target,
            "moved_at": moved_at,
            "reason": reason,
        }
    )


def _normalize_link_type(link_type: str) -> str:
    field_name = LINK_TYPE_ALIASES.get(str(link_type))
    if not field_name:
        raise ModelRegistryError(f"link type must be one of {', '.join(sorted(LINK_TYPE_ALIASES))}")
    return field_name


def _resolve_entry_id(registry: dict[str, Any], entry_ref: str) -> str:
    if not isinstance(entry_ref, str) or not entry_ref:
        raise ModelRegistryError("entry reference must be a non-empty entry id or alias")
    aliases = registry.get("aliases") if isinstance(registry.get("aliases"), dict) else {}
    entry_id = aliases.get(entry_ref, entry_ref)
    if not isinstance(entry_id, str) or not entry_id:
        raise ModelRegistryError(f"model alias {entry_ref!r} is not set")
    entries = registry.get("entries") if isinstance(registry.get("entries"), dict) else {}
    if entry_id not in entries:
        raise ModelRegistryError(f"model entry {entry_ref!r} does not resolve to a registered entry")
    return entry_id


def _same_link(existing: Any, new_link: dict[str, Any]) -> bool:
    if not isinstance(existing, dict):
        return False
    for key in ("artifact_id", "path"):
        if existing.get(key) and existing.get(key) == new_link.get(key):
            return True
    return False


def _require_registered_target(registry: dict[str, Any], target: str) -> None:
    if not isinstance(target, str) or not target:
        raise ModelRegistryError("alias target must be a non-empty entry id")
    entries = registry.get("entries") if isinstance(registry.get("entries"), dict) else {}
    if target not in entries:
        raise ModelRegistryError(f"alias target {target!r} is not registered")


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
    notes = license_review.get("notes")
    if notes is not None and not _is_string_list(notes):
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


def _link_list_errors(value: Any, errors: list[str], label: str) -> None:
    if not isinstance(value, list):
        errors.append(f"{label} must be a list.")
        return
    for index, item in enumerate(value):
        if isinstance(item, str):
            if not item:
                errors.append(f"{label}[{index}] must be a non-empty string.")
            continue
        if isinstance(item, dict):
            if not any(isinstance(item.get(field_name), str) and item.get(field_name) for field_name in ("id", "path", "artifact_id")):
                errors.append(f"{label}[{index}] object must include a non-empty id, path, or artifact_id.")
            continue
        errors.append(f"{label}[{index}] must be a string or object.")


def _copy_link_list(value: Any) -> list[Any]:
    if not isinstance(value, list):
        return []
    return copy.deepcopy(value)


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
