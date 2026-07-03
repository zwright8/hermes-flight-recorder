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
from .schema_registry import SchemaRegistryError, check_schema_file
from .trainer_consumer_plan import TRAINER_CONSUMER_PLAN_SCHEMA_VERSION

AGENTIC_TRAINING_FLOW_SCHEMA_VERSION = "hfr.agentic_training_flow.v1"

FLOW_READY_RECOMMENDATION = "ready_for_delegated_trainer_execution"
FLOW_BLOCK_RECOMMENDATION = "block_delegated_trainer_execution"

EXECUTABLE_FLOW_MODES = tuple(sorted(DEFAULT_EXECUTABLE_MODES))
EXECUTABLE_STAGES = ("action_sft", "dpo", "sft")
BLOCKED_TRAINER_FLOW_MODES = ("grpo", "process_rewards", "reward_model", "rl")

STAGE_VIEW_CANDIDATES: dict[str, tuple[str, ...]] = {
    "sft": ("sft",),
    "action_sft": ("action_sft", "sft"),
    "dpo": ("dpo", "preferences"),
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
        {"mode": mode},
        {"executable_modes": list(EXECUTABLE_FLOW_MODES), "blocked_modes": list(BLOCKED_TRAINER_FLOW_MODES)},
    )
    _add_check(
        checks,
        "stage_sequence_executable",
        bool(stage_sequence) and all(stage in EXECUTABLE_STAGES for stage in stage_sequence),
        {"stage_sequence": stage_sequence},
        {"allowed_stages": list(EXECUTABLE_STAGES)},
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
            "requires_registered_model_and_dataset": True,
            "requires_redacted_dataset": True,
            "executable_modes": list(EXECUTABLE_FLOW_MODES),
            "blocked_modes": list(BLOCKED_TRAINER_FLOW_MODES),
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
    ref: dict[str, Any] = {
        "role": role,
        "path": _display_path(path, receipt_base, preserve_paths),
        "exists": path.exists(),
        "regular_file": path.is_file() and not path.is_symlink(),
        "schema_version": str(payload.get("schema_version") or ""),
        "passed": payload.get("passed") if isinstance(payload.get("passed"), bool) else None,
        "recommendation": str(payload.get("recommendation") or ""),
        "sha256": _sha256_or_none(path),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() and not path.is_symlink() else None,
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
    if not path.is_absolute():
        return raw
    resolved = path.resolve()
    if receipt_base is not None:
        try:
            relative = os.path.relpath(resolved, receipt_base.resolve())
        except OSError:
            relative = ""
        if relative and not relative.startswith("..") and not Path(relative).is_absolute():
            return relative
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except (OSError, ValueError):
        return f"<redacted:{resolved.name}>"


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
