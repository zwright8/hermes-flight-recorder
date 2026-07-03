"""Fail-closed rollout generation plans for agentic training loops."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    scenarios = [_scenario_ref(Path(path), preserve_paths) for path in scenario_paths]
    if not scenarios:
        raise RolloutGenerationError("at least one scenario path is required")
    policy_rows = [_policy_ref(role, policies.get(role, "")) for role in POLICY_ROLES if policies.get(role)]
    if not policy_rows:
        raise RolloutGenerationError("at least one policy id is required")
    verifier_refs = [_file_ref("verifier_config", Path(path), preserve_paths) for path in verifier_paths or []]
    batches = _batches(scenarios, policy_rows, max_rollouts)
    checks: list[dict[str, Any]] = []
    _add_check(checks, "scenario_inputs_exist", all(row["exists"] for row in scenarios), {"scenarios": scenarios}, {"all_scenarios_exist": True})
    _add_check(checks, "rollout_budget_positive", max_rollouts > 0, {"max_rollouts": max_rollouts}, {"max_rollouts": ">0"})
    _add_check(checks, "rollout_budget_not_exceeded", len(batches) <= max_rollouts, {"planned_rollouts": len(batches)}, {"max_rollouts": max_rollouts})
    _add_check(checks, "at_least_one_policy_configured", bool(policy_rows), {"policy_count": len(policy_rows)}, {"policy_count": ">=1"})
    _add_check(checks, "flight_recorder_did_not_run_rollouts", True, {"rollouts_started": False}, {"rollouts_started": False})
    failed = [check for check in checks if not check["passed"]]
    return {
        "schema_version": AGENTIC_ROLLOUT_PLAN_SCHEMA_VERSION,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "iteration_id": iteration_id,
        "plan_path": _display_path(Path(out_path), preserve_paths),
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
    out_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_agentic_rollout_receipt(
    *,
    plan_path: str | Path,
    out_path: str | Path | None = None,
    preserve_paths: bool = False,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic mock rollout receipt from a rollout plan."""
    path = Path(plan_path)
    plan = _read_required_json(path, "agentic rollout plan")
    batches = plan.get("harness_batches") if isinstance(plan.get("harness_batches"), list) else []
    checks: list[dict[str, Any]] = []
    _add_check(
        checks,
        "plan_schema_supported",
        plan.get("schema_version") == AGENTIC_ROLLOUT_PLAN_SCHEMA_VERSION,
        {"schema_version": plan.get("schema_version")},
        {"schema_version": AGENTIC_ROLLOUT_PLAN_SCHEMA_VERSION},
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
        "receipt_path": _display_path(Path(out_path), preserve_paths) if out_path else "",
        "passed": not failed,
        "readiness": "mock_rollouts_recorded" if not failed else "blocked",
        "recommendation": "score_and_review_mock_rollouts" if not failed else "fix_rollout_plan_before_mock_execution",
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "checks": checks,
        "blocked_reasons": [check["summary"] for check in failed],
        "source_plan": {
            "path": _display_path(path, preserve_paths),
            "exists": path.exists() and path.is_file(),
            "sha256": _sha256(path) if path.exists() and path.is_file() else None,
            "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
            "schema_version": plan.get("schema_version"),
            "passed": plan.get("passed") if isinstance(plan.get("passed"), bool) else None,
            "readiness": plan.get("readiness") if isinstance(plan.get("readiness"), str) else "",
        },
        "iteration_id": str(plan.get("iteration_id") or ""),
        "environment": plan.get("environment") if isinstance(plan.get("environment"), dict) else {},
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
    payload = dict(receipt)
    source_plan = payload.get("source_plan") if isinstance(payload.get("source_plan"), dict) else None
    if source_plan:
        source_plan = dict(source_plan)
        source_plan["path"] = _output_relative_path(source_plan.get("path"), out_path.parent)
        payload["source_plan"] = source_plan
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


def _scenario_ref(path: Path, preserve_paths: bool) -> dict[str, Any]:
    payload = _read_json(path)
    return {
        "id": str(payload.get("id") or path.stem),
        "path": _display_path(path, preserve_paths),
        "exists": path.exists() and path.is_file(),
        "sha256": _sha256(path) if path.exists() and path.is_file() else None,
        "schema_version": str(payload.get("schema_version") or ""),
    }


def _policy_ref(role: str, policy_id: str) -> dict[str, Any]:
    return {"role": role, "id": policy_id, "live_calls_allowed": False}


def _file_ref(role: str, path: Path, preserve_paths: bool) -> dict[str, Any]:
    exists = path.exists() and path.is_file()
    return {
        "role": role,
        "path": _display_path(path, preserve_paths),
        "exists": exists,
        "sha256": _sha256(path) if exists else None,
        "size_bytes": path.stat().st_size if exists else None,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_required_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RolloutGenerationError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RolloutGenerationError(f"{label} is not valid JSON: {path}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise RolloutGenerationError(f"{label} must contain a JSON object: {path}")
    return payload


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


def _output_relative_path(value: Any, output_dir: Path) -> Any:
    if not isinstance(value, str) or not value:
        return value
    path = Path(value)
    if not path.is_absolute():
        if not path.exists():
            return value
        path = path.resolve()
    return os.path.relpath(path.resolve(), output_dir.resolve())


def _display_path(path: Path, preserve_paths: bool) -> str:
    if preserve_paths:
        return str(path)
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except (OSError, ValueError):
        return path.name


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
