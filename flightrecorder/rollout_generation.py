"""Fail-closed rollout generation plans for agentic training loops."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

from .source_contract import inspect_artifact_source

AGENTIC_ROLLOUT_PLAN_SCHEMA_VERSION = "hfr.agentic_rollout_plan.v1"
AGENTIC_ROLLOUT_RECEIPT_SCHEMA_VERSION = "hfr.agentic_rollout_receipt.v1"

POLICY_ROLES = ("baseline", "candidate", "teacher")


class RolloutGenerationError(ValueError):
    """Raised when a rollout generation plan cannot be built."""


def build_agentic_rollout_plan(
    *,
    out_path: str | Path,
    iteration_id: str,
    scenario_paths: list[str | Path],
    policies: dict[str, str],
    max_rollouts: int,
    verifier_paths: list[str | Path] | None = None,
    environment_id: str = "offline_mock",
    preserve_paths: bool = False,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic rollout generation contract without running policies."""
    if not iteration_id:
        raise RolloutGenerationError("iteration_id is required")
    if max_rollouts <= 0:
        raise RolloutGenerationError("max_rollouts must be positive")
    output_dir = Path(out_path).parent
    scenarios = [_scenario_ref(Path(path), preserve_paths, output_dir) for path in scenario_paths]
    if not scenarios:
        raise RolloutGenerationError("at least one scenario path is required")
    policy_rows = [_policy_ref(role, policies.get(role, "")) for role in POLICY_ROLES if policies.get(role)]
    if not policy_rows:
        raise RolloutGenerationError("at least one policy id is required")
    verifier_refs = [_file_ref("verifier_config", Path(path), preserve_paths, output_dir) for path in verifier_paths or []]
    verifier_gate = _external_state_verifier_gate(verifier_refs)
    batches = _batches([scenario for scenario in scenarios if scenario.get("exists") is True], policy_rows, max_rollouts)
    checks: list[dict[str, Any]] = []
    _add_check(checks, "scenario_inputs_exist", all(row["exists"] for row in scenarios), {"scenarios": scenarios}, {"all_scenarios_exist": True})
    _add_check(checks, "rollout_budget_positive", max_rollouts > 0, {"max_rollouts": max_rollouts}, {"max_rollouts": ">0"})
    _add_check(checks, "rollout_budget_not_exceeded", len(batches) <= max_rollouts, {"planned_rollouts": len(batches)}, {"max_rollouts": max_rollouts})
    _add_check(checks, "at_least_one_policy_configured", bool(policy_rows), {"policy_count": len(policy_rows)}, {"policy_count": ">=1"})
    _add_check(
        checks,
        "external_state_verifiers_resolved",
        verifier_gate["all_declared_verifiers_resolved"],
        {"external_state_verifier_gate": verifier_gate},
        {"all_declared_verifiers_resolved": True},
    )
    _add_check(checks, "flight_recorder_did_not_run_rollouts", True, {"rollouts_started": False}, {"rollouts_started": False})
    failed = [check for check in checks if not check["passed"]]
    return {
        "schema_version": AGENTIC_ROLLOUT_PLAN_SCHEMA_VERSION,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "iteration_id": iteration_id,
        "plan_path": _display_path(Path(out_path), preserve_paths, output_dir),
        "passed": not failed,
        "readiness": "ready_for_harness_batch" if not failed else "blocked",
        "recommendation": "run_mock_or_opted_in_harness_batch" if not failed else "fix_rollout_plan_inputs",
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "checks": checks,
        "blocked_reasons": [check["summary"] for check in failed],
        "environment": {
            "id": environment_id,
            "replayable": True,
            "network_default": "disabled",
            "external_state_verifiers": verifier_refs,
            "external_state_verifier_gate": verifier_gate,
        },
        "budget": {
            "max_rollouts": max_rollouts,
            "planned_rollouts": len(batches),
            "live_provider_calls_allowed": False,
        },
        "policies": policy_rows,
        "scenarios": scenarios,
        "harness_batches": batches,
        "rejection_sampling": {
            "enabled": True,
            "requires_scorecard": True,
            "requires_task_completion": True,
            "requires_review_calibration_before_training": True,
            "accepted_dataset_roles": ["sft", "action_sft", "dpo", "reward_model"],
        },
        "lineage": {
            "dataset_rows_created": False,
            "expected_trace_artifacts": ["normalized_trace", "scorecard", "task_completion", "run_digest", "artifact_lineage"],
            "preserve_source_hashes": True,
        },
        "execution_boundary": {
            "plan_only": True,
            "rollouts_started": False,
            "model_provider_calls_started": False,
            "paid_model_grader_calls_started": False,
            "dataset_rows_written": False,
        },
        "notes": [
            "This plan schedules rollout batches only; it does not call model providers or run harnesses.",
            "External-state verifier configs are referenced by hash so rollout receipts can prove which checks were expected.",
        ],
    }


def write_agentic_rollout_plan(path: str | Path, plan: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _plan_for_write(plan, out_path)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_agentic_rollout_receipt(
    *,
    plan_path: str | Path,
    out_path: str | Path | None = None,
    preserve_paths: bool = False,
    created_at: str | None = None,
    output_base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic mock rollout receipt from a rollout plan."""
    path = Path(plan_path)
    display_base_dir = Path(output_base_dir) if output_base_dir is not None else (Path(out_path).parent if out_path else None)
    source_path, source_replayable = _source_display_path(path, preserve_paths, display_base_dir)
    source = inspect_artifact_source(path, "agentic_rollout_plan") if source_replayable else {"payload": {}, "schema_valid": False}
    plan = source["payload"] if isinstance(source.get("payload"), dict) else {}
    source_contract_valid = source_replayable and source.get("schema_valid") is True
    batches = plan.get("harness_batches") if isinstance(plan.get("harness_batches"), list) else []
    checks: list[dict[str, Any]] = []
    _add_check(
        checks,
        "plan_schema_supported",
        source_contract_valid and plan.get("schema_version") == AGENTIC_ROLLOUT_PLAN_SCHEMA_VERSION,
        {"schema_version": plan.get("schema_version"), "source_contract_valid": source_contract_valid},
        {"schema_version": AGENTIC_ROLLOUT_PLAN_SCHEMA_VERSION, "source_contract_valid": True},
    )
    _add_check(
        checks,
        "plan_ready_for_mock_rollouts",
        plan.get("passed") is True and plan.get("readiness") == "ready_for_harness_batch",
        {"passed": plan.get("passed"), "readiness": plan.get("readiness")},
        {"passed": True, "readiness": "ready_for_harness_batch"},
    )
    _add_check(checks, "harness_batches_present", bool(batches), {"batch_count": len(batches)}, {"batch_count": ">=1"})
    _add_check(
        checks,
        "mock_rollout_only",
        all(isinstance(batch, dict) and batch.get("harness_mode") == "offline_mock" for batch in batches),
        {"harness_modes": sorted({str(batch.get("harness_mode")) for batch in batches if isinstance(batch, dict)})},
        {"harness_mode": "offline_mock"},
    )
    _add_check(
        checks,
        "flight_recorder_did_not_call_model_provider",
        True,
        {"model_provider_calls_started": False, "live_rollouts_started": False},
        {"model_provider_calls_started": False, "live_rollouts_started": False},
    )
    failed = [check for check in checks if not check["passed"]]
    records = [] if failed else [_mock_rollout_record(batch, index) for index, batch in enumerate(batches) if isinstance(batch, dict)]
    return {
        "schema_version": AGENTIC_ROLLOUT_RECEIPT_SCHEMA_VERSION,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "receipt_path": _display_path(Path(out_path), preserve_paths, Path(out_path).parent) if out_path else "",
        "passed": not failed,
        "readiness": "mock_rollouts_recorded" if not failed else "blocked",
        "recommendation": "score_and_review_mock_rollouts" if not failed else "fix_rollout_plan_before_mock_execution",
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "checks": checks,
        "blocked_reasons": [check["summary"] for check in failed],
        "source_plan": {
            "path": source_path,
            "exists": source_contract_valid,
            "sha256": _sha256(path) if source_contract_valid else None,
            "size_bytes": path.stat().st_size if source_contract_valid else None,
            "schema_version": plan.get("schema_version") if plan else AGENTIC_ROLLOUT_PLAN_SCHEMA_VERSION,
            "passed": plan.get("passed") if isinstance(plan.get("passed"), bool) else None,
            "readiness": plan.get("readiness") if isinstance(plan.get("readiness"), str) else "",
        },
        "iteration_id": str(plan.get("iteration_id") or ""),
        "environment": plan.get("environment") if isinstance(plan.get("environment"), dict) else _default_environment(),
        "mock_rollout_count": len(records),
        "mock_rollouts": records,
        "lineage": {
            "dataset_rows_created": False,
            "trace_files_written": False,
            "scorecards_written": False,
            "ready_for_rejection_sampling": not failed,
        },
        "execution_boundary": {
            "mock_receipt_only": True,
            "mock_rollouts_recorded": bool(records),
            "live_rollouts_started": False,
            "model_provider_calls_started": False,
            "paid_model_grader_calls_started": False,
            "dataset_rows_written": False,
        },
        "notes": [
            "This receipt records deterministic mock rollout rows only.",
            "It does not call model providers, run paid graders, write traces, or add rows to training datasets.",
        ],
    }


def write_agentic_rollout_receipt(path: str | Path, receipt: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _receipt_for_write(receipt, out_path)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _batches(scenarios: list[dict[str, Any]], policies: list[dict[str, Any]], max_rollouts: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        for policy in policies:
            if len(rows) >= max_rollouts:
                return rows
            rows.append(
                {
                    "batch_id": f"{scenario['id']}::{policy['role']}",
                    "scenario_id": scenario["id"],
                    "scenario_sha256": scenario["sha256"],
                    "policy_role": policy["role"],
                    "policy_id": policy["id"],
                    "harness_mode": "offline_mock",
                    "status": "planned",
                }
            )
    return rows


def _scenario_ref(path: Path, preserve_paths: bool, output_dir: Path) -> dict[str, Any]:
    displayed, replayable = _source_display_path(path, preserve_paths, output_dir)
    if not replayable:
        return _missing_scenario_ref(path, displayed)
    source = inspect_artifact_source(path, "scenario")
    exists = source["ready"] is True
    payload = source["payload"] if isinstance(source.get("payload"), dict) else {}
    return {
        "id": str(payload.get("id") or path.stem),
        "path": displayed,
        "exists": exists,
        "sha256": _sha256(path) if exists else None,
        "schema_version": str(payload.get("schema_version") or ""),
    }


def _policy_ref(role: str, policy_id: str) -> dict[str, Any]:
    return {"role": role, "id": policy_id, "live_calls_allowed": False}


def _file_ref(role: str, path: Path, preserve_paths: bool, output_dir: Path) -> dict[str, Any]:
    displayed, replayable = _source_display_path(path, preserve_paths, output_dir)
    if not replayable:
        return _missing_file_ref(role, displayed)
    source = inspect_artifact_source(path, role)
    exists = source["ready"] is True
    return {
        "role": role,
        "path": displayed,
        "exists": exists,
        "sha256": _sha256(path) if exists else None,
        "size_bytes": path.stat().st_size if exists else None,
    }


def _external_state_verifier_gate(verifier_refs: list[dict[str, Any]]) -> dict[str, Any]:
    resolved_count = sum(1 for ref in verifier_refs if ref.get("exists") is True)
    return {
        "declared_count": len(verifier_refs),
        "resolved_count": resolved_count,
        "all_declared_verifiers_resolved": resolved_count == len(verifier_refs),
        "required_for_external_state_checks": bool(verifier_refs),
        "verification_side_effects_started": False,
        "credential_values_recorded": False,
    }


def _default_environment() -> dict[str, Any]:
    verifier_refs: list[dict[str, Any]] = []
    return {
        "id": "offline_mock",
        "replayable": True,
        "network_default": "disabled",
        "external_state_verifiers": verifier_refs,
        "external_state_verifier_gate": _external_state_verifier_gate(verifier_refs),
    }


def _mock_rollout_record(batch: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "rollout_id": f"mock-rollout-{index + 1:04d}",
        "batch_id": str(batch.get("batch_id") or ""),
        "scenario_id": str(batch.get("scenario_id") or ""),
        "scenario_sha256": batch.get("scenario_sha256") if isinstance(batch.get("scenario_sha256"), str) else None,
        "policy_role": str(batch.get("policy_role") or ""),
        "policy_id": str(batch.get("policy_id") or ""),
        "harness_mode": "offline_mock",
        "status": "mock_recorded",
        "model_provider_called": False,
        "trace_written": False,
        "scorecard_written": False,
        "dataset_row_written": False,
    }


def _plan_for_write(plan: dict[str, Any], out_path: Path) -> dict[str, Any]:
    payload = copy.deepcopy(plan)
    payload["plan_path"] = _display_path(out_path, False, out_path.parent)
    changed = False
    scenarios = payload.get("scenarios") if isinstance(payload.get("scenarios"), list) else []
    for scenario in scenarios:
        if isinstance(scenario, dict) and _rewrite_source_ref_for_output(scenario, out_path.parent, scenario_role="scenario"):
            changed = True
    environment = payload.get("environment") if isinstance(payload.get("environment"), dict) else {}
    verifiers = environment.get("external_state_verifiers") if isinstance(environment.get("external_state_verifiers"), list) else []
    for verifier in verifiers:
        if isinstance(verifier, dict) and _rewrite_source_ref_for_output(verifier, out_path.parent, scenario_role=None):
            changed = True
    if isinstance(environment, dict):
        environment["external_state_verifier_gate"] = _external_state_verifier_gate(
            [verifier for verifier in verifiers if isinstance(verifier, dict)]
        )
    if changed:
        _refresh_plan_readiness(payload)
    return payload


def _receipt_for_write(receipt: dict[str, Any], out_path: Path) -> dict[str, Any]:
    payload = copy.deepcopy(receipt)
    payload["receipt_path"] = _display_path(out_path, False, out_path.parent)
    source_plan = payload.get("source_plan") if isinstance(payload.get("source_plan"), dict) else None
    if isinstance(source_plan, dict) and _rewrite_source_plan_for_output(source_plan, out_path.parent):
        _block_receipt_for_unreplayable_source_plan(payload, source_plan.get("path"))
    return payload


def _rewrite_source_ref_for_output(row: dict[str, Any], output_dir: Path, *, scenario_role: str | None) -> bool:
    value = row.get("path")
    if not isinstance(value, str) or not value:
        return False
    if value.startswith("<redacted:"):
        return _mark_missing_ref(row, value, scenario_role=scenario_role)
    if _is_safe_public_path(value):
        candidate = output_dir / value
        if row.get("exists") is not True or candidate.is_file():
            return False
        path = Path(value)
        if path.exists():
            relative = _safe_output_relative_path(path, output_dir)
            if relative is not None:
                row["path"] = relative
                return False
        return _mark_missing_ref(row, f"<redacted:{_basename(value)}>", scenario_role=scenario_role)
    path = Path(value)
    relative = _safe_output_relative_path(path, output_dir) if path.is_absolute() or path.exists() else None
    if relative is not None:
        row["path"] = relative
        return False
    return _mark_missing_ref(row, f"<redacted:{_basename(value)}>", scenario_role=scenario_role)


def _rewrite_source_plan_for_output(source_plan: dict[str, Any], output_dir: Path) -> bool:
    value = source_plan.get("path")
    if not isinstance(value, str) or not value:
        return False
    if value.startswith("<redacted:"):
        return _mark_source_plan_missing(source_plan, value)
    if _is_safe_public_path(value):
        if source_plan.get("exists") is not True or (output_dir / value).is_file():
            return False
        path = Path(value)
        if path.exists():
            relative = _safe_output_relative_path(path, output_dir)
            if relative is not None:
                source_plan["path"] = relative
                return False
        return _mark_source_plan_missing(source_plan, f"<redacted:{_basename(value)}>")
    path = Path(value)
    relative = _safe_output_relative_path(path, output_dir) if path.is_absolute() or path.exists() else None
    if relative is not None:
        source_plan["path"] = relative
        return False
    return _mark_source_plan_missing(source_plan, f"<redacted:{_basename(value)}>")


def _mark_missing_ref(row: dict[str, Any], displayed: str, *, scenario_role: str | None) -> bool:
    row["path"] = displayed
    row["exists"] = False
    row["sha256"] = None
    if scenario_role == "scenario":
        row["schema_version"] = ""
    else:
        row["size_bytes"] = None
    return True


def _mark_source_plan_missing(source_plan: dict[str, Any], displayed: str) -> bool:
    source_plan.update(
        {
            "path": displayed,
            "exists": False,
            "sha256": None,
            "size_bytes": None,
            "schema_version": AGENTIC_ROLLOUT_PLAN_SCHEMA_VERSION,
            "passed": None,
            "readiness": "",
        }
    )
    return True


def _refresh_plan_readiness(plan: dict[str, Any]) -> None:
    scenarios = plan.get("scenarios") if isinstance(plan.get("scenarios"), list) else []
    policies = plan.get("policies") if isinstance(plan.get("policies"), list) else []
    budget = plan.get("budget") if isinstance(plan.get("budget"), dict) else {}
    max_rollouts = budget.get("max_rollouts") if isinstance(budget.get("max_rollouts"), int) else 0
    batches = _batches(
        [scenario for scenario in scenarios if isinstance(scenario, dict) and scenario.get("exists") is True],
        [policy for policy in policies if isinstance(policy, dict)],
        max_rollouts,
    )
    plan["harness_batches"] = batches
    budget["planned_rollouts"] = len(batches)
    budget["live_provider_calls_allowed"] = False
    environment = plan.get("environment") if isinstance(plan.get("environment"), dict) else {}
    verifier_refs = environment.get("external_state_verifiers") if isinstance(environment.get("external_state_verifiers"), list) else []
    verifier_gate = _external_state_verifier_gate([ref for ref in verifier_refs if isinstance(ref, dict)])
    environment["external_state_verifier_gate"] = verifier_gate
    checks: list[dict[str, Any]] = []
    _add_check(checks, "scenario_inputs_exist", all(isinstance(row, dict) and row.get("exists") is True for row in scenarios), {"scenarios": scenarios}, {"all_scenarios_exist": True})
    _add_check(checks, "rollout_budget_positive", max_rollouts > 0, {"max_rollouts": max_rollouts}, {"max_rollouts": ">0"})
    _add_check(checks, "rollout_budget_not_exceeded", len(batches) <= max_rollouts, {"planned_rollouts": len(batches)}, {"max_rollouts": max_rollouts})
    _add_check(checks, "at_least_one_policy_configured", bool(policies), {"policy_count": len(policies)}, {"policy_count": ">=1"})
    _add_check(
        checks,
        "external_state_verifiers_resolved",
        verifier_gate["all_declared_verifiers_resolved"],
        {"external_state_verifier_gate": verifier_gate},
        {"all_declared_verifiers_resolved": True},
    )
    _add_check(checks, "flight_recorder_did_not_run_rollouts", True, {"rollouts_started": False}, {"rollouts_started": False})
    failed = [check for check in checks if not check["passed"]]
    plan["checks"] = checks
    plan["check_count"] = len(checks)
    plan["failed_check_count"] = len(failed)
    plan["passed"] = not failed
    plan["readiness"] = "ready_for_harness_batch" if not failed else "blocked"
    plan["recommendation"] = "run_mock_or_opted_in_harness_batch" if not failed else "fix_rollout_plan_inputs"
    plan["blocked_reasons"] = [check["summary"] for check in failed]


def _block_receipt_for_unreplayable_source_plan(receipt: dict[str, Any], path_value: Any) -> None:
    checks = receipt.get("checks") if isinstance(receipt.get("checks"), list) else []
    checks = [check for check in checks if not (isinstance(check, dict) and check.get("id") == "source_plan_replayable_from_receipt")]
    _add_check(
        checks,
        "source_plan_replayable_from_receipt",
        False,
        {"path": path_value, "exists": False},
        {"exists": True, "path": "safe relative path from receipt"},
    )
    failed = [check for check in checks if isinstance(check, dict) and check.get("passed") is False]
    receipt["checks"] = checks
    receipt["check_count"] = len(checks)
    receipt["failed_check_count"] = len(failed)
    receipt["passed"] = False
    receipt["readiness"] = "blocked"
    receipt["recommendation"] = "fix_rollout_plan_before_mock_execution"
    receipt["blocked_reasons"] = [str(check.get("summary") or "") for check in failed]
    receipt["mock_rollouts"] = []
    receipt["mock_rollout_count"] = 0
    lineage = receipt.get("lineage") if isinstance(receipt.get("lineage"), dict) else {}
    lineage["ready_for_rejection_sampling"] = False
    receipt["lineage"] = lineage
    boundary = receipt.get("execution_boundary") if isinstance(receipt.get("execution_boundary"), dict) else {}
    boundary["mock_rollouts_recorded"] = False
    receipt["execution_boundary"] = boundary


def _missing_scenario_ref(path: Path, displayed: str) -> dict[str, Any]:
    return {
        "id": path.stem or "scenario",
        "path": displayed,
        "exists": False,
        "sha256": None,
        "schema_version": "",
    }


def _missing_file_ref(role: str, displayed: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": displayed,
        "exists": False,
        "sha256": None,
        "size_bytes": None,
    }


def _source_display_path(path: Path, preserve_paths: bool, output_dir: Path | None = None) -> tuple[str, bool]:
    displayed = _source_ref_display_path(path, preserve_paths, output_dir)
    return displayed, _is_safe_public_path(displayed)


def _source_ref_display_path(path: Path, preserve_paths: bool, output_dir: Path | None = None) -> str:
    raw = str(path)
    if output_dir is not None:
        relative = _safe_output_relative_path(path, output_dir)
        return relative if relative is not None else f"<redacted:{_basename(raw)}>"
    if preserve_paths:
        return raw if _is_safe_public_path(raw) else f"<redacted:{_basename(raw)}>"
    if _is_windows_absolute(raw):
        return f"<redacted:{_basename(raw)}>"
    if not path.is_absolute():
        return raw if _is_safe_public_path(raw) else f"<redacted:{_basename(raw)}>"
    try:
        relative = str(path.resolve().relative_to(Path.cwd().resolve()))
    except (OSError, ValueError):
        return f"<redacted:{path.name}>"
    return relative if _is_safe_public_path(relative) else f"<redacted:{path.name}>"


def _safe_output_relative_path(path: Path, output_dir: Path) -> str | None:
    if _is_windows_absolute(str(path)):
        return None
    try:
        relative = os.path.relpath(path.resolve(), output_dir.resolve())
    except OSError:
        return None
    return relative if _is_safe_public_path(relative) else None


def _display_path(path: Path, preserve_paths: bool, output_dir: Path | None = None) -> str:
    return _source_ref_display_path(path, preserve_paths, output_dir)


def _is_safe_public_path(value: str) -> bool:
    if not value or value.startswith("<redacted:"):
        return False
    path = Path(value)
    windows_path = PureWindowsPath(value)
    return (
        not path.is_absolute()
        and not windows_path.is_absolute()
        and not windows_path.drive
        and "\\" not in value
        and "~" not in path.parts
        and ".." not in path.parts
    )


def _is_windows_absolute(value: str) -> bool:
    normalized = value.replace("/", "\\")
    return (len(normalized) >= 3 and normalized[1:3] == ":\\" and normalized[0].isalpha()) or normalized.startswith("\\\\")


def _basename(value: str) -> str:
    return value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] or "path"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _add_check(checks: list[dict[str, Any]], check_id: str, passed: bool, actual: dict[str, Any], expected: dict[str, Any]) -> None:
    checks.append(
        {
            "id": check_id,
            "passed": bool(passed),
            "actual": actual,
            "expected": expected,
            "summary": f"{check_id}: passed={bool(passed)}",
        }
    )
