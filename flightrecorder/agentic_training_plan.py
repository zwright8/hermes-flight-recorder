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

DEFAULT_EXECUTABLE_MODES = {"sft", "action_sft", "dpo", "sft_then_dpo"}
ADVANCED_REWARD_MODES = {"reward_model", "process_rewards"}
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

MODE_DATA_REQUIREMENTS: dict[str, tuple[dict[str, Any], ...]] = {
    "sft": (
        {
            "id": "supervised_response_rows",
            "description": "Redacted supervised prompt/response rows for external SFT trainers.",
            "view_groups": (("sft",),),
            "required_schema_names": ("rl_sft",),
        },
    ),
    "action_sft": (
        {
            "id": "action_supervision_rows",
            "description": "Redacted action/tool-use supervised rows for action-SFT trainers.",
            "view_groups": (("action_sft", "sft"),),
            "required_schema_names": ("rl_sft",),
        },
    ),
    "dpo": (
        {
            "id": "preference_pair_rows",
            "description": "Reviewed chosen/rejected preference pairs for external DPO trainers.",
            "view_groups": (("dpo", "preferences"),),
            "required_schema_names": ("rl_dpo", "rl_preference"),
        },
    ),
    "sft_then_dpo": (
        {
            "id": "supervised_response_rows",
            "description": "Redacted supervised prompt/response rows for the SFT warmup stage.",
            "view_groups": (("sft",),),
            "required_schema_names": ("rl_sft",),
        },
        {
            "id": "preference_pair_rows",
            "description": "Reviewed chosen/rejected preference pairs for the DPO stage.",
            "view_groups": (("dpo", "preferences"),),
            "required_schema_names": ("rl_dpo", "rl_preference"),
        },
    ),
    "reward_model": (
        {
            "id": "reward_label_rows",
            "description": "Reviewed scalar reward labels or preference-derived reward-model rows.",
            "view_groups": (("reward_model", "preferences"),),
            "required_schema_names": ("rl_reward_model", "rl_preference"),
        },
    ),
    "process_rewards": (
        {
            "id": "step_reward_rows",
            "description": "Reviewed step-level process reward labels linked to rollout evidence.",
            "view_groups": (("process_rewards", "step_rewards"),),
            "required_schema_names": ("rl_step_reward",),
        },
    ),
    "grpo": (
        {
            "id": "rollout_episode_rows",
            "description": "Replayable rollout episodes for an external GRPO runner.",
            "view_groups": (("episodes", "rollouts"),),
            "required_schema_names": ("rl_episode",),
        },
        {
            "id": "external_reward_function_contract",
            "description": "A deterministic TRL/GRPO-style reward function supplied and validated by the external runner.",
            "view_groups": (("episodes", "rollouts"),),
            "required_schema_names": ("rl_episode",),
        },
    ),
    "rl": (
        {
            "id": "rollout_episode_rows",
            "description": "Replayable rollout episodes for an external RL runner.",
            "view_groups": (("episodes", "rollouts"),),
            "required_schema_names": ("rl_episode",),
        },
        {
            "id": "external_reward_function_contract",
            "description": "A deterministic reward function supplied and validated by the external RL runner.",
            "view_groups": (("episodes", "rollouts"),),
            "required_schema_names": ("rl_episode",),
        },
    ),
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
    allow_advanced_training: bool = False,
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
    mode_contract = _mode_contract(mode, selected_views, allow_advanced_training, allow_future_rl)

    checks: list[dict[str, Any]] = []
    _add_check(checks, "mode_supported", True, {"mode": mode}, {"supported_modes": list(SUPPORTED_MODES)})
    _add_check(
        checks,
        "default_trainer_flow_mode_ready",
        mode in DEFAULT_EXECUTABLE_MODES
        or (mode in ADVANCED_REWARD_MODES and allow_advanced_training)
        or (mode in FUTURE_RL_MODES and allow_future_rl),
        {"mode": mode, "allow_advanced_training": allow_advanced_training, "allow_future_rl": allow_future_rl},
        {
            "default_ready_modes": sorted(DEFAULT_EXECUTABLE_MODES),
            "advanced_reward_modes_require": "allow_advanced_training",
            "future_rl_modes_require": "allow_future_rl",
        },
    )
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
        "mode_contract_data_requirements_satisfied",
        all(requirement["satisfied"] for requirement in mode_contract["data_requirements"]),
        {
            "mode": mode,
            "unsatisfied_requirement_ids": [
                requirement["id"] for requirement in mode_contract["data_requirements"] if not requirement["satisfied"]
            ],
        },
        {"all_data_requirements_satisfied": True},
    )
    _add_check(
        checks,
        "advanced_reward_mode_explicitly_enabled",
        mode not in ADVANCED_REWARD_MODES or allow_advanced_training,
        {"mode": mode, "allow_advanced_training": allow_advanced_training},
        {"advanced_reward_modes": sorted(ADVANCED_REWARD_MODES), "allow_advanced_training": True},
    )
    _add_check(
        checks,
        "future_rl_explicitly_enabled",
        mode not in FUTURE_RL_MODES or allow_future_rl,
        {"mode": mode, "allow_future_rl": allow_future_rl},
        {"future_rl_modes": sorted(FUTURE_RL_MODES), "allow_future_rl": True},
    )
    _add_check(
        checks,
        "mode_contract_planning_gate_open",
        mode_contract["planning_gate"]["open"],
        mode_contract["planning_gate"],
        {"open": True},
    )
    _add_check(
        checks,
        "reward_contract_fail_closed",
        mode_contract["reward_contract"]["flight_recorder_supplies_callable"] is False
        and mode_contract["reward_contract"]["may_call_paid_services_by_default"] is False
        and mode_contract["reward_contract"]["may_require_secrets_by_default"] is False,
        {
            "flight_recorder_supplies_callable": mode_contract["reward_contract"]["flight_recorder_supplies_callable"],
            "may_call_paid_services_by_default": mode_contract["reward_contract"]["may_call_paid_services_by_default"],
            "may_require_secrets_by_default": mode_contract["reward_contract"]["may_require_secrets_by_default"],
        },
        {
            "flight_recorder_supplies_callable": False,
            "may_call_paid_services_by_default": False,
            "may_require_secrets_by_default": False,
        },
    )
    _add_check(checks, "flight_recorder_did_not_launch_training", True, {"training_started": False}, {"training_started": False})
    _add_check(checks, "model_downloads_not_started", True, {"model_downloads_started": False}, {"model_downloads_started": False})
    _add_check(checks, "cloud_jobs_not_started", True, {"cloud_jobs_started": False}, {"cloud_jobs_started": False})
    _add_check(
        checks,
        "paid_model_grader_calls_not_started",
        True,
        {"paid_model_grader_calls_started": False},
        {"paid_model_grader_calls_started": False},
    )
    _add_check(checks, "weights_not_updated", True, {"weights_updated": False}, {"weights_updated": False})

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
        "mode_contract": mode_contract,
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
            "cloud_jobs_started": False,
            "paid_model_grader_calls_started": False,
            "weights_updated": False,
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
            "advanced_modes_require_explicit_opt_in": True,
            "future_rl_requires_explicit_opt_in": True,
            "flight_recorder_started_cloud_provider": False,
            "paid_model_grader_calls_started": False,
            "weights_updated_by_flight_recorder": False,
            "reward_function_owned_by_external_runner": True,
            "allowed_modes": list(SUPPORTED_MODES),
            "default_executable_modes": sorted(DEFAULT_EXECUTABLE_MODES),
            "advanced_reward_modes_blocked_by_default": sorted(ADVANCED_REWARD_MODES),
            "future_rl_modes_blocked_by_default": sorted(FUTURE_RL_MODES),
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
        "size_bytes": path.stat().st_size,
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
        "size_bytes": path.stat().st_size,
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


def _mode_contract(
    mode: str,
    selected_views: list[dict[str, Any]],
    allow_advanced_training: bool,
    allow_future_rl: bool,
) -> dict[str, Any]:
    planning_gate = _planning_gate(mode, allow_advanced_training, allow_future_rl)
    return {
        "mode": mode,
        "category": _mode_category(mode),
        "planning_gate": planning_gate,
        "stage_sequence": MODE_STAGE_SEQUENCES[mode],
        "required_view_groups": [list(group) for group in MODE_VIEW_REQUIREMENTS[mode]],
        "selected_view_names": [str(view.get("name") or "") for view in selected_views],
        "data_requirements": _data_requirements(mode, selected_views),
        "reward_contract": _reward_contract(mode),
        "side_effect_boundary": {
            "dry_run_only": True,
            "training_started": False,
            "cloud_jobs_started": False,
            "model_downloads_started": False,
            "paid_model_grader_calls_started": False,
            "weights_updated": False,
            "provider_credentials_required_by_flight_recorder": False,
        },
        "external_runner_contract": {
            "runner_owns_execution": True,
            "runner_must_revalidate_inputs": True,
            "runner_must_require_recommendation": "ready_for_external_trainer_plan",
            "runner_must_validate_reward_contract": mode in {"dpo", "sft_then_dpo", "reward_model", "process_rewards", "grpo", "rl"},
            "runner_must_block_unredacted_traces": True,
        },
    }


def _mode_category(mode: str) -> str:
    if mode in DEFAULT_EXECUTABLE_MODES:
        return "default_executable"
    if mode in ADVANCED_REWARD_MODES:
        return "advanced_reward"
    return "future_rl"


def _planning_gate(mode: str, allow_advanced_training: bool, allow_future_rl: bool) -> dict[str, Any]:
    if mode in ADVANCED_REWARD_MODES:
        return {
            "open": allow_advanced_training,
            "required_flag": "--allow-advanced-training",
            "reason": "reward-model and process-reward planning is blocked by default",
        }
    if mode in FUTURE_RL_MODES:
        return {
            "open": allow_future_rl,
            "required_flag": "--allow-future-rl",
            "reason": "GRPO/RL planning is blocked by default",
        }
    return {
        "open": True,
        "required_flag": None,
        "reason": "default executable handoff mode",
    }


def _data_requirements(mode: str, selected_views: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected_by_name = {str(view.get("name") or ""): view for view in selected_views}
    requirements: list[dict[str, Any]] = []
    for spec in MODE_DATA_REQUIREMENTS[mode]:
        evidence = []
        group_satisfied = []
        for group in spec["view_groups"]:
            matched = next(
                (
                    selected_by_name[name]
                    for name in group
                    if name in selected_by_name and _positive_int(selected_by_name[name].get("row_count"))
                ),
                None,
            )
            group_satisfied.append(matched is not None)
            evidence.append(
                {
                    "candidate_views": list(group),
                    "selected_view": str(matched.get("name") or "") if matched else "",
                    "row_count": int(matched.get("row_count") or 0) if matched else 0,
                    "schema_version": str(matched.get("schema_version") or "") if matched else "",
                }
            )
        requirements.append(
            {
                "id": str(spec["id"]),
                "description": str(spec["description"]),
                "view_groups": [list(group) for group in spec["view_groups"]],
                "required_schema_names": list(spec["required_schema_names"]),
                "minimum_rows_per_group": 1,
                "satisfied": all(group_satisfied),
                "evidence": evidence,
            }
        )
    return requirements


def _reward_contract(mode: str) -> dict[str, Any]:
    base = {
        "required": mode not in {"sft", "action_sft"},
        "external_runner_must_supply": mode in {"grpo", "rl"},
        "external_runner_must_validate": mode in {"dpo", "sft_then_dpo", "reward_model", "process_rewards", "grpo", "rl"},
        "flight_recorder_supplies_callable": False,
        "may_call_paid_services_by_default": False,
        "may_require_secrets_by_default": False,
        "must_not_use_unredacted_traces": True,
        "requires_calibration_or_human_review_gate": mode in {"reward_model", "process_rewards", "grpo", "rl"},
    }
    if mode in {"sft", "action_sft"}:
        return {
            **base,
            "kind": "not_applicable",
            "source_views": [],
            "callable_signature": "",
            "notes": ["Supervised modes do not require a reward-function contract."],
        }
    if mode in {"dpo", "sft_then_dpo"}:
        return {
            **base,
            "kind": "preference_pairs",
            "source_views": ["dpo", "preferences"],
            "callable_signature": "",
            "notes": ["External DPO trainers must validate reviewed chosen/rejected preference pairs."],
        }
    if mode == "reward_model":
        return {
            **base,
            "kind": "scalar_or_preference_rewards",
            "source_views": ["reward_model", "preferences"],
            "callable_signature": "",
            "notes": ["External reward-model trainers must validate reviewed reward labels before use."],
        }
    if mode == "process_rewards":
        return {
            **base,
            "kind": "step_rewards",
            "source_views": ["process_rewards", "step_rewards"],
            "callable_signature": "",
            "notes": ["External process-reward trainers must validate step-level labels and episode lineage."],
        }
    if mode == "grpo":
        return {
            **base,
            "kind": "trl_grpo_reward_function",
            "source_views": ["episodes", "rollouts"],
            "callable_signature": "reward_fn(prompts, completions, **kwargs) -> list[float]",
            "notes": [
                "Flight Recorder records the required interface only.",
                "The external GRPO runner owns reward-function implementation, calibration, and execution.",
            ],
        }
    return {
        **base,
        "kind": "external_rl_reward_function",
        "source_views": ["episodes", "rollouts"],
        "callable_signature": "reward_fn(episodes, actions, **kwargs) -> list[float]",
        "notes": [
            "Flight Recorder records the required interface only.",
            "The external RL runner owns reward-function implementation, calibration, and execution.",
        ],
    }


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


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
