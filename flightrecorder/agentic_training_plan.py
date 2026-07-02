"""Registry-backed dry-run plans for agentic fine-tuning paths."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

AGENTIC_TRAINING_PLAN_SCHEMA_VERSION = "hfr.agentic_training_plan.v1"

SUPPORTED_MODES = (
    "sft",
    "action_sft",
    "dpo",
    "sft_then_dpo",
    "reward_model",
    "process_rewards",
    "grpo",
    "rl",
)

FUTURE_RL_MODES = {"grpo", "rl"}

MODE_VIEW_REQUIREMENTS: dict[str, tuple[tuple[str, ...], ...]] = {
    "sft": (("sft",),),
    "action_sft": (("action_sft", "sft"),),
    "dpo": (("dpo", "preferences"),),
    "sft_then_dpo": (("sft",), ("dpo", "preferences")),
    "reward_model": (("reward_model", "preferences"),),
    "process_rewards": (("process_rewards", "step_rewards"),),
    "grpo": (("episodes", "rollouts"),),
    "rl": (("episodes", "rollouts"),),
}

MODE_STAGE_SEQUENCES: dict[str, list[str]] = {
    "sft": ["sft"],
    "action_sft": ["action_sft"],
    "dpo": ["dpo"],
    "sft_then_dpo": ["sft", "dpo"],
    "reward_model": ["reward_model"],
    "process_rewards": ["process_rewards"],
    "grpo": ["future_grpo"],
    "rl": ["future_rl"],
}

TRAINING_LICENSE_STATUSES = {"approved", "allowed", "cleared", "permissive", "open"}
REDACTED_DATASET_STATUSES = {"redacted", "clean", "passed"}


class AgenticTrainingPlanError(ValueError):
    """Raised when an agentic training plan cannot be built."""


def build_agentic_training_plan(
    *,
    out_path: str | Path,
    mode: str,
    model_manifest_path: str | Path,
    dataset_manifest_path: str | Path,
    trainer_backend: str = "external",
    output_dir: str | Path | None = None,
    limit: int | None = None,
    allow_future_rl: bool = False,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a side-effect-free training plan from registered model and dataset manifests."""
    if mode not in SUPPORTED_MODES:
        raise AgenticTrainingPlanError(f"unsupported mode {mode!r}; expected one of {', '.join(SUPPORTED_MODES)}")
    if limit is not None and limit <= 0:
        raise AgenticTrainingPlanError("limit must be a positive integer when provided")

    model_path = Path(model_manifest_path)
    dataset_path = Path(dataset_manifest_path)
    model_manifest = _read_json_object(model_path, "model manifest")
    dataset_manifest = _read_json_object(dataset_path, "dataset manifest")

    model = _model_record(model_path, model_manifest)
    dataset = _dataset_record(dataset_path, dataset_manifest)
    selected_views = _selected_views(dataset["views"], mode)

    checks: list[dict[str, Any]] = []
    _add_check(checks, "mode_supported", True, {"mode": mode}, {"supported_modes": list(SUPPORTED_MODES)})
    _add_check(checks, "model_manifest_registered", bool(model["id"]), {"model_id": model["id"]}, {"required": True})
    _add_check(
        checks,
        "model_license_allows_training",
        model["license_allows_training"],
        {"status": model["license_status"], "allow_training": model["license_allow_training"]},
        {"status": sorted(TRAINING_LICENSE_STATUSES), "allow_training": True},
    )
    _add_check(
        checks,
        "model_compatibility_passed",
        model["compatibility_passed"],
        {"passed": model["compatibility_passed"]},
        {"passed": True},
    )
    _add_check(checks, "dataset_manifest_registered", bool(dataset["id"]), {"dataset_id": dataset["id"]}, {"required": True})
    _add_check(
        checks,
        "dataset_license_allows_training",
        dataset["license_allows_training"],
        {"status": dataset["license_status"], "allow_training": dataset["license_allow_training"]},
        {"status": sorted(TRAINING_LICENSE_STATUSES), "allow_training": True},
    )
    _add_check(
        checks,
        "dataset_redaction_passed",
        dataset["redaction_passed"],
        {"status": dataset["redaction_status"], "contains_unredacted_traces": dataset["contains_unredacted_traces"]},
        {"status": sorted(REDACTED_DATASET_STATUSES), "contains_unredacted_traces": False},
    )
    _add_check(
        checks,
        "mode_required_views_available",
        _requirements_satisfied(selected_views, mode),
        {"selected_views": [view["name"] for view in selected_views]},
        {"required_view_groups": [list(group) for group in MODE_VIEW_REQUIREMENTS[mode]]},
    )
    _add_check(
        checks,
        "future_rl_explicitly_enabled",
        mode not in FUTURE_RL_MODES or allow_future_rl,
        {"mode": mode, "allow_future_rl": allow_future_rl},
        {"future_rl_modes": sorted(FUTURE_RL_MODES), "allow_future_rl": True},
    )
    _add_check(checks, "flight_recorder_did_not_launch_training", True, {"training_started": False}, {"training_started": False})
    _add_check(checks, "model_downloads_not_started", True, {"model_downloads_started": False}, {"model_downloads_started": False})

    failed_checks = [check for check in checks if check["passed"] is False]
    passed = not failed_checks
    recommendation = "ready_for_external_trainer_plan" if passed else "block_external_training"
    output_path = Path(out_path)
    plan = {
        "schema_version": AGENTIC_TRAINING_PLAN_SCHEMA_VERSION,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "plan_path": str(output_path),
        "mode": mode,
        "passed": passed,
        "readiness": "ready" if passed else "blocked",
        "recommendation": recommendation,
        "check_count": len(checks),
        "failed_check_count": len(failed_checks),
        "checks": checks,
        "blocked_reasons": [check["summary"] for check in failed_checks],
        "input_manifests": {
            "model": model,
            "dataset": dataset,
        },
        "selected_views": selected_views,
        "trainer_plan": {
            "backend": trainer_backend,
            "output_dir": str(output_dir or ""),
            "limit": limit,
            "stage_sequence": MODE_STAGE_SEQUENCES[mode],
            "mode_required_view_groups": [list(group) for group in MODE_VIEW_REQUIREMENTS[mode]],
            "extension_points": _extension_points(mode),
        },
        "execution": {
            "dry_run_only": True,
            "training_started": False,
            "model_downloads_started": False,
            "external_runner_command": _external_runner_command(mode, trainer_backend, output_dir),
        },
        "handoff_contract": {
            "flight_recorder_executed_training": False,
            "runner_owns_execution": True,
            "runner_must_require_recommendation": "ready_for_external_trainer_plan",
            "requires_registered_model": True,
            "requires_registered_dataset": True,
            "requires_known_license_status": True,
            "requires_redacted_dataset": True,
            "disallow_unredacted_traces": True,
            "allowed_modes": list(SUPPORTED_MODES),
        },
        "notes": [
            "This is a dry-run plan only; Flight Recorder did not import trainer packages or update weights.",
            "External runners must revalidate manifests, redaction, license status, and file hashes immediately before launch.",
        ],
    }
    return plan


def write_agentic_training_plan(path: str | Path, plan: dict[str, Any]) -> None:
    """Write a deterministic JSON plan artifact."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AgenticTrainingPlanError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AgenticTrainingPlanError(f"{label} is not valid JSON: {path}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise AgenticTrainingPlanError(f"{label} must contain a JSON object: {path}")
    return payload


def _model_record(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    license_record = _dict_value(manifest, "license")
    compatibility = _dict_value(manifest, "compatibility") or _dict_value(manifest, "training_compatibility")
    model_id = _first_string(manifest, "model_id", "id", "name")
    candidate_id = _first_string(manifest, "candidate_id", "registry_entry", "alias")
    license_status = str(license_record.get("status") or "").lower()
    allow_training = license_record.get("allow_training") is True or license_record.get("training_allowed") is True
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "schema_version": str(manifest.get("schema_version") or ""),
        "id": model_id,
        "candidate_id": candidate_id,
        "license_status": license_status,
        "license_allow_training": allow_training,
        "license_allows_training": license_status in TRAINING_LICENSE_STATUSES and allow_training,
        "compatibility_passed": compatibility.get("passed") is True,
        "base_model": _first_string(manifest, "base_model", "base_model_id"),
    }


def _dataset_record(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    license_record = _dict_value(manifest, "license")
    redaction = _dict_value(manifest, "redaction") or _dict_value(manifest, "redaction_report")
    views = _views(manifest)
    license_status = str(license_record.get("status") or "").lower()
    allow_training = license_record.get("allow_training") is True or license_record.get("training_allowed") is True
    redaction_status = str(redaction.get("status") or "").lower()
    contains_unredacted = redaction.get("contains_unredacted_traces") is True
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "schema_version": str(manifest.get("schema_version") or ""),
        "id": _first_string(manifest, "dataset_id", "id", "name"),
        "version": _first_string(manifest, "dataset_version", "version"),
        "license_status": license_status,
        "license_allow_training": allow_training,
        "license_allows_training": license_status in TRAINING_LICENSE_STATUSES and allow_training,
        "redaction_status": redaction_status,
        "contains_unredacted_traces": contains_unredacted,
        "redaction_passed": redaction.get("passed") is True and redaction_status in REDACTED_DATASET_STATUSES and not contains_unredacted,
        "view_count": len(views),
        "views": views,
    }


def _views(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_views = manifest.get("views")
    if not isinstance(raw_views, dict):
        return {}
    views: dict[str, dict[str, Any]] = {}
    for name, value in raw_views.items():
        if not isinstance(name, str) or not isinstance(value, dict):
            continue
        row_count = value.get("row_count")
        if not isinstance(row_count, int) or isinstance(row_count, bool):
            row_count = 0
        views[name] = {
            "name": name,
            "path": str(value.get("path") or ""),
            "row_count": max(row_count, 0),
            "split": str(value.get("split") or ""),
            "schema_version": str(value.get("schema_version") or ""),
        }
    return views


def _selected_views(views: dict[str, dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in MODE_VIEW_REQUIREMENTS[mode]:
        match = next((views[name] for name in group if views.get(name, {}).get("row_count", 0) > 0), None)
        if match is not None and match["name"] not in seen:
            selected.append(dict(match))
            seen.add(match["name"])
    return selected


def _requirements_satisfied(selected_views: list[dict[str, Any]], mode: str) -> bool:
    selected = {str(view.get("name") or "") for view in selected_views if view.get("row_count", 0) > 0}
    return all(any(name in selected for name in group) for group in MODE_VIEW_REQUIREMENTS[mode])


def _extension_points(mode: str) -> list[dict[str, str]]:
    records = [
        {"id": "axolotl", "status": "planned", "scope": "sft,dpo,sft_then_dpo"},
        {"id": "llama_factory", "status": "planned", "scope": "sft,dpo,sft_then_dpo,reward_model"},
        {"id": "unsloth", "status": "planned", "scope": "sft,dpo,sft_then_dpo"},
        {"id": "process_reward_trainer", "status": "planned", "scope": "process_rewards,reward_model"},
        {"id": "grpo_rl_runner", "status": "future", "scope": "grpo,rl"},
    ]
    return [record for record in records if mode in record["scope"].split(",") or record["status"] == "future"]


def _external_runner_command(mode: str, trainer_backend: str, output_dir: str | Path | None) -> list[str]:
    command = [
        "<external-runner>",
        "--mode",
        mode,
        "--backend",
        trainer_backend,
        "--require-registered-inputs",
        "--require-redacted-dataset",
        "--require-license-approved",
    ]
    if output_dir:
        command.extend(["--output-dir", str(output_dir)])
    return command


def _add_check(
    checks: list[dict[str, Any]],
    check_id: str,
    passed: bool,
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    checks.append(
        {
            "id": check_id,
            "passed": bool(passed),
            "actual": actual,
            "expected": expected,
            "summary": f"{check_id}: passed={bool(passed)}",
        }
    )


def _dict_value(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _first_string(payload: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
