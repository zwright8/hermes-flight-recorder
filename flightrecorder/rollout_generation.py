"""Fail-closed rollout generation plans for agentic training loops."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

AGENTIC_ROLLOUT_PLAN_SCHEMA_VERSION = "hfr.agentic_rollout_plan.v1"

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
