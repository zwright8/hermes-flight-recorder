"""Delegated trainer-flow receipts for executable agentic training modes."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agentic_training_plan import AGENTIC_TRAINING_PLAN_SCHEMA_VERSION, DEFAULT_EXECUTABLE_MODES
from .agentic_training_runtime import AGENTIC_TRAINING_RUNTIME_PREFLIGHT_SCHEMA_VERSION, RUNTIME_READY_RECOMMENDATION
from .path_safety import path_has_symlink_component as _path_has_symlink_component
from .schema_registry import SchemaRegistryError, check_schema_file
from .trainer_consumer_plan import TRAINER_CONSUMER_PLAN_SCHEMA_VERSION

AGENTIC_TRAINING_FLOW_SCHEMA_VERSION = "hfr.agentic_training_flow.v1"

FLOW_READY_RECOMMENDATION = "ready_for_delegated_trainer_execution"
FLOW_BLOCK_RECOMMENDATION = "block_delegated_trainer_execution"

EXECUTABLE_FLOW_MODES = tuple(sorted(DEFAULT_EXECUTABLE_MODES))
EXECUTABLE_STAGES = ("action_sft", "dpo", "sft")
BLOCKED_TRAINER_FLOW_MODES = ("grpo", "process_rewards", "reward_model", "rl")
BLOCKED_TRAINER_FLOW_STAGES = ("future_grpo", "future_rl", "process_rewards", "reward_model")
BLOCKED_FLOW_MODE_CATEGORIES = {
    "reward_model": "advanced_reward",
    "process_rewards": "advanced_reward",
    "grpo": "future_rl",
    "rl": "future_rl",
}

STAGE_VIEW_CANDIDATES: dict[str, tuple[str, ...]] = {
    "sft": ("sft",),
    "action_sft": ("action_sft", "sft"),
    "dpo": ("dpo", "preferences"),
    "reward_model": ("reward_model", "preferences"),
    "process_rewards": ("process_rewards", "step_rewards"),
    "future_grpo": ("episodes", "rollouts"),
    "future_rl": ("episodes", "rollouts"),
}


class AgenticTrainingFlowError(ValueError):
    """Raised when an agentic training flow receipt cannot be written."""


def build_agentic_training_flow(
    *,
    plan_path: str | Path,
    runtime_preflight_path: str | Path,
    trainer_consumer_plan_path: str | Path,
    out_path: str | Path | None = None,
    flow_id: str = "",
    preserve_paths: bool = False,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a fail-closed delegated trainer-flow receipt without executing training."""
    receipt_base = Path(out_path).parent if out_path is not None else None
    plan_file = Path(plan_path)
    runtime_file = Path(runtime_preflight_path)
    consumer_file = Path(trainer_consumer_plan_path)
    _reject_symlinked_source_input(plan_file, "agentic_training_flow.plan_path")
    _reject_symlinked_source_input(runtime_file, "agentic_training_flow.runtime_preflight_path")
    _reject_symlinked_source_input(consumer_file, "agentic_training_flow.trainer_consumer_plan_path")
    plan_payload, plan_read_errors = _read_json_object(plan_file)
    runtime_payload, runtime_read_errors = _read_json_object(runtime_file)
    consumer_payload, consumer_read_errors = _read_json_object(consumer_file)
    plan_schema = _schema_check(plan_file, "agentic_training_plan")
    runtime_schema = _schema_check(runtime_file, "agentic_training_runtime_preflight")
    consumer_schema = _schema_check(consumer_file, "trainer_consumer_plan")

    mode = str(plan_payload.get("mode") or "")
    trainer_plan = plan_payload.get("trainer_plan") if isinstance(plan_payload.get("trainer_plan"), dict) else {}
    backend = str(trainer_plan.get("backend") or runtime_payload.get("backend") or "")
    stage_sequence = _stage_sequence(trainer_plan)
    stages = _stage_records(stage_sequence, plan_payload)
    runtime_plan_sha = runtime_payload.get("plan_sha256") if isinstance(runtime_payload.get("plan_sha256"), str) else ""
    mode_contract_check = _mode_contract_check(runtime_payload, plan_payload, mode)
    flow_mode_gate = _flow_mode_gate(mode, mode_contract_check)
    plan_sha = _sha256_or_none(plan_file)
    consumer_execution = consumer_payload.get("execution") if isinstance(consumer_payload.get("execution"), dict) else {}
    command_argv = _string_list(consumer_execution.get("command_argv"))

    checks: list[dict[str, Any]] = []
    _add_check(checks, "plan_json_readable", not plan_read_errors, {"errors": plan_read_errors}, {"errors": []})
    _add_check(
        checks,
        "plan_schema_passed",
        plan_schema["passed"],
        {"error_count": plan_schema["error_count"], "errors": plan_schema["errors"]},
        {"schema_name": "agentic_training_plan", "error_count": 0},
    )
    _add_check(
        checks,
        "plan_ready_for_external_trainer",
        plan_payload.get("schema_version") == AGENTIC_TRAINING_PLAN_SCHEMA_VERSION
        and plan_payload.get("passed") is True
        and plan_payload.get("recommendation") == "ready_for_external_trainer_plan",
        {"passed": plan_payload.get("passed"), "recommendation": plan_payload.get("recommendation")},
        {"passed": True, "recommendation": "ready_for_external_trainer_plan"},
    )
    _add_check(checks, "runtime_preflight_json_readable", not runtime_read_errors, {"errors": runtime_read_errors}, {"errors": []})
    _add_check(
        checks,
        "runtime_preflight_schema_passed",
        runtime_schema["passed"],
        {"error_count": runtime_schema["error_count"], "errors": runtime_schema["errors"]},
        {"schema_name": "agentic_training_runtime_preflight", "error_count": 0},
    )
    _add_check(
        checks,
        "runtime_preflight_ready",
        runtime_payload.get("schema_version") == AGENTIC_TRAINING_RUNTIME_PREFLIGHT_SCHEMA_VERSION
        and runtime_payload.get("passed") is True
        and runtime_payload.get("recommendation") == RUNTIME_READY_RECOMMENDATION,
        {"passed": runtime_payload.get("passed"), "recommendation": runtime_payload.get("recommendation")},
        {"passed": True, "recommendation": RUNTIME_READY_RECOMMENDATION},
    )
    _add_check(
        checks,
        "runtime_preflight_matches_plan",
        bool(plan_sha) and runtime_plan_sha == plan_sha,
        {"runtime_plan_sha256": runtime_plan_sha, "plan_sha256": plan_sha or ""},
        {"runtime_plan_sha256": plan_sha or ""},
    )
    _add_check(
        checks,
        "mode_contract_ready_for_flow",
        mode_contract_check["present"] is True
        and mode_contract_check["mode_matches_plan"] is True
        and mode_contract_check["passed"] is True,
        {
            "mode": mode_contract_check["mode"],
            "category": mode_contract_check["category"],
            "present": mode_contract_check["present"],
            "mode_matches_plan": mode_contract_check["mode_matches_plan"],
            "passed": mode_contract_check["passed"],
            "error_count": mode_contract_check["error_count"],
            "errors": mode_contract_check["errors"],
        },
        {"present": True, "mode_matches_plan": True, "passed": True, "error_count": 0},
        summary=_mode_contract_summary(mode_contract_check),
    )
    _add_check(
        checks,
        "trainer_consumer_plan_json_readable",
        not consumer_read_errors,
        {"errors": consumer_read_errors},
        {"errors": []},
    )
    _add_check(
        checks,
        "trainer_consumer_plan_schema_passed",
        consumer_schema["passed"],
        {"error_count": consumer_schema["error_count"], "errors": consumer_schema["errors"]},
        {"schema_name": "trainer_consumer_plan", "error_count": 0},
    )
    _add_check(
        checks,
        "trainer_consumer_plan_ready",
        consumer_payload.get("schema_version") == TRAINER_CONSUMER_PLAN_SCHEMA_VERSION
        and consumer_payload.get("passed") is True
        and consumer_payload.get("recommendation") == "ready_for_external_trainer",
        {"passed": consumer_payload.get("passed"), "recommendation": consumer_payload.get("recommendation")},
        {"passed": True, "recommendation": "ready_for_external_trainer"},
    )
    _add_check(
        checks,
        "default_executable_flow_mode",
        mode in EXECUTABLE_FLOW_MODES,
        flow_mode_gate,
        {"executable_modes": list(EXECUTABLE_FLOW_MODES), "blocked_modes": list(BLOCKED_TRAINER_FLOW_MODES)},
        summary=_flow_mode_summary(flow_mode_gate),
    )
    _add_check(
        checks,
        "stage_sequence_executable",
        bool(stage_sequence) and all(stage in EXECUTABLE_STAGES for stage in stage_sequence),
        {"stage_sequence": stage_sequence},
        {"allowed_stages": list(EXECUTABLE_STAGES)},
        summary=_stage_sequence_summary(stage_sequence),
    )
    _add_check(
        checks,
        "selected_views_available_for_stages",
        bool(stages) and all(stage.get("view_ready") is True for stage in stages),
        {"stages_without_views": [stage["stage_id"] for stage in stages if stage.get("view_ready") is not True]},
        {"all_stages_have_selected_views": True},
    )
    _add_check(
        checks,
        "delegated_command_available",
        bool(command_argv) and consumer_execution.get("command_approved") is True and consumer_execution.get("command_available") is True,
        {
            "command_arg_count": len(command_argv),
            "command_approved": consumer_execution.get("command_approved"),
            "command_available": consumer_execution.get("command_available"),
        },
        {"command_arg_count": ">0", "command_approved": True, "command_available": True},
    )
    _add_check(
        checks,
        "flight_recorder_did_not_execute_trainer_flow",
        True,
        {
            "trainer_command_executed": False,
            "subprocess_started": False,
            "model_downloads_started": False,
            "weights_updated_by_flight_recorder": False,
        },
        {
            "trainer_command_executed": False,
            "subprocess_started": False,
            "model_downloads_started": False,
            "weights_updated_by_flight_recorder": False,
        },
    )

    failed_checks = [check for check in checks if check["passed"] is False]
    passed = not failed_checks
    external_code_files = consumer_execution.get("external_code_files") if isinstance(consumer_execution.get("external_code_files"), list) else []
    trainer_inputs = consumer_execution.get("trainer_inputs") if isinstance(consumer_execution.get("trainer_inputs"), list) else []
    metrics = {
        "check_count": len(checks),
        "failed_check_count": len(failed_checks),
        "stage_count": len(stages),
        "executable_stage_count": sum(1 for stage in stages if stage.get("stage_id") in EXECUTABLE_STAGES),
        "selected_view_count": len({stage.get("view_name") for stage in stages if stage.get("view_name")}),
        "command_arg_count": len(command_argv),
        "trainer_input_count": len([item for item in trainer_inputs if isinstance(item, dict)]),
        "external_code_file_count": len([item for item in external_code_files if isinstance(item, dict)]),
    }
    return {
        "schema_version": AGENTIC_TRAINING_FLOW_SCHEMA_VERSION,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "flow_path": _display_path(Path(out_path), receipt_base, preserve_paths) if out_path is not None else "",
        "flow_id": flow_id or _default_flow_id(mode, stage_sequence),
        "passed": passed,
        "readiness": "ready" if passed else "blocked",
        "recommendation": FLOW_READY_RECOMMENDATION if passed else FLOW_BLOCK_RECOMMENDATION,
        "check_count": len(checks),
        "failed_check_count": len(failed_checks),
        "checks": checks,
        "blocked_reasons": [check["summary"] for check in failed_checks],
        "mode_contract_check": mode_contract_check,
        "flow_mode_gate": flow_mode_gate,
        "source_artifacts": {
            "agentic_training_plan": _source_ref(plan_file, plan_payload, "agentic_training_plan", receipt_base, preserve_paths),
            "agentic_training_runtime_preflight": _source_ref(
                runtime_file,
                runtime_payload,
                "agentic_training_runtime_preflight",
                receipt_base,
                preserve_paths,
            ),
            "trainer_consumer_plan": _source_ref(consumer_file, consumer_payload, "trainer_consumer_plan", receipt_base, preserve_paths),
        },
        "delegated_flow": {
            "mode": mode,
            "backend": backend,
            "stage_sequence": stage_sequence,
            "stages": stages,
            "command": {
                "execution_cwd": str(consumer_execution.get("execution_cwd") or ""),
                "archive_root": str(consumer_execution.get("archive_root") or ""),
                "external_code_root": str(consumer_execution.get("external_code_root") or ""),
                "command_argv": command_argv,
                "command_shell": shlex.join(command_argv) if command_argv else "",
                "trainer_input_count": metrics["trainer_input_count"],
                "external_code_file_count": metrics["external_code_file_count"],
            },
        },
        "metrics": metrics,
        "execution_boundary": {
            "flow_plan_only": True,
            "training_started": False,
            "trainer_command_executed": False,
            "subprocess_started": False,
            "cloud_jobs_started": False,
            "model_downloads_started": False,
            "weights_updated_by_flight_recorder": False,
            "trainer_modules_imported": False,
            "credential_values_recorded": False,
        },
        "handoff_contract": {
            "runner_owns_execution": True,
            "runner_must_require_recommendation": FLOW_READY_RECOMMENDATION,
            "runner_must_emit_result_schema": "hfr.agentic_training_result.v1",
            "requires_runtime_preflight": True,
            "requires_trainer_consumer_plan": True,
            "requires_mode_contract": True,
            "requires_mode_contract_ready": True,
            "requires_registered_model_and_dataset": True,
            "requires_redacted_dataset": True,
            "executable_modes": list(EXECUTABLE_FLOW_MODES),
            "blocked_modes": list(BLOCKED_TRAINER_FLOW_MODES),
            "blocked_mode_categories": sorted(set(BLOCKED_FLOW_MODE_CATEGORIES.values())),
            "blocked_mode_stages": list(BLOCKED_TRAINER_FLOW_STAGES),
            "flight_recorder_executed_trainer": False,
        },
        "notes": [
            "This receipt delegates an executable trainer flow; Flight Recorder did not run the command.",
            "Only SFT, action-SFT, DPO, and SFT-then-DPO flows are executable by default.",
            "Reward-model, process-reward, GRPO, and RL modes remain blocked until their contracts are promoted.",
        ],
    }


def write_agentic_training_flow(path: str | Path, receipt: dict[str, Any]) -> None:
    """Write a deterministic delegated trainer-flow receipt."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _reject_symlinked_source_input(path: Path, label: str) -> None:
    if path.is_symlink():
        raise AgenticTrainingFlowError(f"{label} must not be a symlink: {path}")
    if _path_has_symlink_component(path, include_leaf=False):
        raise AgenticTrainingFlowError(f"{label} must not traverse symlinked components: {path}")


def _read_json_object(path: Path) -> tuple[dict[str, Any], list[str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}, [f"file not found: {path}"]
    except json.JSONDecodeError as exc:
        return {}, [f"invalid JSON: {exc.msg}"]
    except OSError as exc:
        return {}, [str(exc)]
    if not isinstance(payload, dict):
        return {}, ["artifact must contain a JSON object"]
    return payload, []


def _schema_check(path: Path, schema_name: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_name": schema_name,
        "passed": False,
        "error_count": 0,
        "errors": [],
    }
    try:
        result = check_schema_file(path, schema_name)
    except (OSError, json.JSONDecodeError, SchemaRegistryError) as exc:
        record["error_count"] = 1
        record["errors"] = [str(exc)]
        return record
    record["passed"] = result.get("passed") is True
    record["error_count"] = _non_negative_int(result.get("error_count"))
    record["errors"] = [str(error) for error in result.get("errors", []) if isinstance(error, str)][:20]
    return record


def _source_ref(
    path: Path,
    payload: dict[str, Any],
    role: str,
    receipt_base: Path | None,
    preserve_paths: bool,
) -> dict[str, Any]:
    symlinked_path = path.is_symlink() or _path_has_symlink_component(path, include_leaf=False)
    regular_file = path.is_file() and not symlinked_path
    ref: dict[str, Any] = {
        "role": role,
        "path": _display_path(path, receipt_base, preserve_paths),
        "exists": path.exists(),
        "regular_file": regular_file,
        "schema_version": str(payload.get("schema_version") or ""),
        "passed": payload.get("passed") if isinstance(payload.get("passed"), bool) else None,
        "recommendation": str(payload.get("recommendation") or ""),
        "sha256": _sha256_or_none(path) if regular_file else None,
        "size_bytes": path.stat().st_size if path.exists() and regular_file else None,
    }
    return ref


def _stage_sequence(trainer_plan: dict[str, Any]) -> list[str]:
    raw = trainer_plan.get("stage_sequence")
    return [stage for stage in raw if isinstance(stage, str) and stage] if isinstance(raw, list) else []


def _stage_records(stage_sequence: list[str], plan_payload: dict[str, Any]) -> list[dict[str, Any]]:
    selected_views = plan_payload.get("selected_views") if isinstance(plan_payload.get("selected_views"), list) else []
    views = [view for view in selected_views if isinstance(view, dict)]
    records: list[dict[str, Any]] = []
    for index, stage_id in enumerate(stage_sequence):
        view = _stage_view(stage_id, views)
        row_count = _non_negative_int(view.get("row_count")) if view else 0
        records.append(
            {
                "stage_index": index,
                "stage_id": stage_id,
                "view_name": str(view.get("name") or "") if view else "",
                "view_path": str(view.get("path") or "") if view else "",
                "view_schema_version": str(view.get("schema_version") or "") if view else "",
                "row_count": row_count,
                "view_ready": view is not None and row_count > 0,
            }
        )
    return records


def _mode_contract_check(
    runtime_payload: dict[str, Any],
    plan_payload: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    runtime_check = runtime_payload.get("mode_contract_check")
    if isinstance(runtime_check, dict):
        return _normalized_mode_contract_check(runtime_check, mode)
    contract = plan_payload.get("mode_contract") if isinstance(plan_payload.get("mode_contract"), dict) else {}
    planning_gate = contract.get("planning_gate") if isinstance(contract.get("planning_gate"), dict) else {}
    data_requirements = contract.get("data_requirements") if isinstance(contract.get("data_requirements"), list) else []
    reward_contract = contract.get("reward_contract") if isinstance(contract.get("reward_contract"), dict) else {}
    side_effect_boundary = contract.get("side_effect_boundary") if isinstance(contract.get("side_effect_boundary"), dict) else {}
    runner_contract = contract.get("external_runner_contract") if isinstance(contract.get("external_runner_contract"), dict) else {}
    errors = ["runtime preflight mode_contract_check is missing"]
    if not contract:
        errors.append("plan mode_contract is missing")
    return {
        "mode": str(contract.get("mode") or mode),
        "category": str(contract.get("category") or BLOCKED_FLOW_MODE_CATEGORIES.get(mode, "")),
        "present": bool(contract),
        "mode_matches_plan": contract.get("mode") == mode,
        "planning_gate_open": planning_gate.get("open") is True,
        "planning_required_flag": planning_gate.get("required_flag") if isinstance(planning_gate.get("required_flag"), str) else None,
        "data_requirement_count": len([item for item in data_requirements if isinstance(item, dict)]),
        "unsatisfied_data_requirement_ids": [
            str(item.get("id") or f"requirement_{index}")
            for index, item in enumerate(data_requirements)
            if isinstance(item, dict) and item.get("satisfied") is not True
        ],
        "reward_contract": _normalized_reward_contract(reward_contract),
        "side_effect_boundary": _normalized_side_effect_boundary(side_effect_boundary),
        "external_runner_contract": _normalized_external_runner_contract(runner_contract),
        "passed": False,
        "error_count": len(errors),
        "errors": errors,
    }


def _normalized_mode_contract_check(value: dict[str, Any], mode: str) -> dict[str, Any]:
    errors = [str(error) for error in value.get("errors", []) if isinstance(error, str)][:20]
    return {
        "mode": str(value.get("mode") or mode),
        "category": str(value.get("category") or BLOCKED_FLOW_MODE_CATEGORIES.get(mode, "")),
        "present": value.get("present") is True,
        "mode_matches_plan": value.get("mode_matches_plan") is True,
        "planning_gate_open": value.get("planning_gate_open") is True,
        "planning_required_flag": value.get("planning_required_flag") if isinstance(value.get("planning_required_flag"), str) else None,
        "data_requirement_count": _non_negative_int(value.get("data_requirement_count")),
        "unsatisfied_data_requirement_ids": _string_list(value.get("unsatisfied_data_requirement_ids")),
        "reward_contract": _normalized_reward_contract(value.get("reward_contract") if isinstance(value.get("reward_contract"), dict) else {}),
        "side_effect_boundary": _normalized_side_effect_boundary(
            value.get("side_effect_boundary") if isinstance(value.get("side_effect_boundary"), dict) else {}
        ),
        "external_runner_contract": _normalized_external_runner_contract(
            value.get("external_runner_contract") if isinstance(value.get("external_runner_contract"), dict) else {}
        ),
        "passed": value.get("passed") is True,
        "error_count": len(errors),
        "errors": errors,
    }


def _normalized_reward_contract(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": str(value.get("kind") or ""),
        "required": value.get("required") is True,
        "external_runner_must_supply": value.get("external_runner_must_supply") is True,
        "external_runner_must_validate": value.get("external_runner_must_validate") is True,
        "flight_recorder_supplies_callable": value.get("flight_recorder_supplies_callable") is True,
        "may_call_paid_services_by_default": value.get("may_call_paid_services_by_default") is True,
        "may_require_secrets_by_default": value.get("may_require_secrets_by_default") is True,
        "must_not_use_unredacted_traces": value.get("must_not_use_unredacted_traces") is True,
        "callable_signature": str(value.get("callable_signature") or ""),
    }


def _normalized_side_effect_boundary(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "dry_run_only": value.get("dry_run_only") is True,
        "training_started": value.get("training_started") is True,
        "cloud_jobs_started": value.get("cloud_jobs_started") is True,
        "model_downloads_started": value.get("model_downloads_started") is True,
        "paid_model_grader_calls_started": value.get("paid_model_grader_calls_started") is True,
        "weights_updated": value.get("weights_updated") is True,
        "provider_credentials_required_by_flight_recorder": value.get("provider_credentials_required_by_flight_recorder") is True,
    }


def _normalized_external_runner_contract(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "runner_owns_execution": value.get("runner_owns_execution") is True,
        "runner_must_revalidate_inputs": value.get("runner_must_revalidate_inputs") is True,
        "runner_must_require_recommendation": str(value.get("runner_must_require_recommendation") or ""),
        "runner_must_validate_reward_contract": value.get("runner_must_validate_reward_contract") is True,
        "runner_must_block_unredacted_traces": value.get("runner_must_block_unredacted_traces") is True,
    }


def _flow_mode_gate(mode: str, mode_contract_check: dict[str, Any]) -> dict[str, Any]:
    blocked_by_default = mode in BLOCKED_TRAINER_FLOW_MODES
    executable_by_default = mode in EXECUTABLE_FLOW_MODES
    category = str(mode_contract_check.get("category") or BLOCKED_FLOW_MODE_CATEGORIES.get(mode, ""))
    promotion_required = blocked_by_default or not executable_by_default
    reason = ""
    if blocked_by_default:
        reason = (
            f"{mode} is a {category or 'blocked'} mode; Flight Recorder records the contract but does not delegate "
            "trainer execution until this flow boundary is explicitly promoted"
        )
    elif not executable_by_default:
        reason = f"{mode or 'unknown'} is not a supported delegated trainer flow mode"
    return {
        "mode": mode,
        "category": category,
        "executable_by_default": executable_by_default,
        "blocked_by_default": blocked_by_default,
        "promotion_required": promotion_required,
        "promotion_status": "default_executable" if executable_by_default else "blocked_until_flow_promotion",
        "required_plan_opt_in_flag": mode_contract_check.get("planning_required_flag"),
        "mode_contract_ready": mode_contract_check.get("passed") is True,
        "reward_contract_kind": mode_contract_check.get("reward_contract", {}).get("kind", ""),
        "external_runner_must_supply_reward": mode_contract_check.get("reward_contract", {}).get("external_runner_must_supply") is True,
        "external_runner_must_validate_reward": mode_contract_check.get("reward_contract", {}).get("external_runner_must_validate") is True,
        "reason": reason,
    }


def _stage_view(stage_id: str, views: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = STAGE_VIEW_CANDIDATES.get(stage_id, ())
    for name in candidates:
        for view in views:
            if view.get("name") == name and _non_negative_int(view.get("row_count")) > 0:
                return view
    return None


def _display_path(path: Path, receipt_base: Path | None, preserve_paths: bool) -> str:
    raw = str(path)
    if preserve_paths:
        return raw
    resolved = path.resolve()
    if receipt_base is not None:
        base = receipt_base.resolve()
        try:
            relative = os.path.relpath(resolved, base)
        except OSError:
            relative = ""
        if relative and not relative.startswith("..") and not Path(relative).is_absolute():
            return relative
        cwd = Path.cwd().resolve()
        if relative and _is_relative_to(resolved, cwd) and _is_relative_to(base, cwd) and not Path(relative).is_absolute():
            return relative
        return f"<redacted:{resolved.name}>"
    if not path.is_absolute():
        return raw
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except (OSError, ValueError):
        return f"<redacted:{resolved.name}>"


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _sha256_or_none(path: Path) -> str | None:
    if not path.exists() or not path.is_file() or path.is_symlink():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_flow_id(mode: str, stage_sequence: list[str]) -> str:
    parts = [mode or "unknown", *stage_sequence]
    return "delegated-" + "-".join(part for part in parts if part)


def _string_list(value: Any) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _non_negative_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _mode_contract_summary(mode_contract_check: dict[str, Any]) -> str:
    if mode_contract_check.get("passed") is True:
        return (
            "mode_contract_ready_for_flow: "
            f"mode={mode_contract_check.get('mode')} category={mode_contract_check.get('category')} passed=True"
        )
    errors = "; ".join(str(error) for error in mode_contract_check.get("errors", []) if isinstance(error, str))
    return f"mode_contract_ready_for_flow: passed=False errors={errors or 'mode contract not ready'}"


def _flow_mode_summary(flow_mode_gate: dict[str, Any]) -> str:
    if flow_mode_gate.get("executable_by_default") is True:
        return f"default_executable_flow_mode: mode={flow_mode_gate.get('mode')} passed=True"
    return (
        "default_executable_flow_mode: passed=False "
        f"mode={flow_mode_gate.get('mode')} category={flow_mode_gate.get('category')} "
        f"promotion_status={flow_mode_gate.get('promotion_status')} reason={flow_mode_gate.get('reason')}"
    )


def _stage_sequence_summary(stage_sequence: list[str]) -> str:
    if stage_sequence and all(stage in EXECUTABLE_STAGES for stage in stage_sequence):
        return f"stage_sequence_executable: stage_sequence={stage_sequence} passed=True"
    blocked = [stage for stage in stage_sequence if stage not in EXECUTABLE_STAGES]
    return (
        "stage_sequence_executable: passed=False "
        f"blocked_stages={blocked} allowed_stages={list(EXECUTABLE_STAGES)}"
    )


def _add_check(
    checks: list[dict[str, Any]],
    check_id: str,
    passed: bool,
    actual: dict[str, Any],
    expected: dict[str, Any],
    summary: str | None = None,
) -> None:
    checks.append(
        {
            "id": check_id,
            "passed": bool(passed),
            "actual": actual,
            "expected": expected,
            "summary": summary or f"{check_id}: passed={bool(passed)}",
        }
    )
