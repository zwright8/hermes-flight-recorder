"""Opt-in, durable controller for the agent self-improvement lifecycle.

The recorder core remains side-effect free.  This module is an explicit
execution boundary: adapters receive stable idempotency keys, while the
controller persists plan-bound receipts before moving to the next phase.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised only on non-POSIX runtimes
    fcntl = None

from .atomic_json import atomic_write_json_cas, json_file_sha256
from .intervention_router import route_failure_cluster
from .redaction import contains_unredacted_secret_assignment, redact_text

CONTROLLER_PLAN_SCHEMA_VERSION = "hfr.agentic_loop_controller_plan.v1"
CONTROLLER_STATE_SCHEMA_VERSION = "hfr.agentic_loop_controller_state.v1"
CONTROLLER_PHASE_RECEIPT_SCHEMA_VERSION = "hfr.agentic_loop_phase_receipt.v1"

_TERMINAL_STATUSES = {"complete", "rolled_back"}
_DEFAULT_GUARDRAILS = {
    "min_task_success_rate": 0.0,
    "max_critical_failures": 0,
    "max_cost_delta": 1.0,
    "max_latency_delta": 1.0,
}
_OVERRIDABLE_PHASE_FIELDS = {
    "command",
    "reconcile_command",
    "result_path",
    "working_directory",
    "timeout_seconds",
    "estimated_cost_usd",
    "max_retries",
}
_AUXILIARY_PHASE_SPECS = {
    "failure_analysis": (False, False),
    "intervention_route": (False, False),
    "rollback": (True, True),
    "post_rollback_smoke": (True, True),
}


class ControllerError(ValueError):
    """Raised when controller inputs or durable state are unsafe."""


class RetryableControllerError(RuntimeError):
    """Raised by an adapter when retrying with the same key is safe."""


class ControllerAdapter(Protocol):
    def reconcile(
        self,
        phase: dict[str, Any],
        *,
        idempotency_key: str,
        context: dict[str, Any],
    ) -> dict[str, Any] | None: ...

    def execute(
        self,
        phase: dict[str, Any],
        *,
        idempotency_key: str,
        context: dict[str, Any],
    ) -> dict[str, Any]: ...


class InMemoryControllerAdapter:
    """Deterministic test/demo adapter with provider-style idempotency."""

    def __init__(
        self,
        *,
        outcomes: dict[str, dict[str, Any]] | None = None,
        fail_once: set[str] | None = None,
    ) -> None:
        self.outcomes = outcomes or {}
        self.fail_once = set(fail_once or set())
        self.failed: set[str] = set()
        self.results: dict[str, dict[str, Any]] = {}
        self.call_counts: Counter[str] = Counter()
        self.side_effect_counts: Counter[str] = Counter()

    def execute(
        self,
        phase: dict[str, Any],
        *,
        idempotency_key: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        phase_id = str(phase["id"])
        self.call_counts[phase_id] += 1
        if idempotency_key in self.results:
            return dict(self.results[idempotency_key])
        if phase_id in self.fail_once and phase_id not in self.failed:
            self.failed.add(phase_id)
            raise RetryableControllerError(f"simulated retryable failure in {phase_id}")
        default = {
            "passed": True,
            "cost_usd": 0.0,
            "duration_seconds": 0.01,
            "metrics": {
                "task_success_rate": 1.0,
                "critical_failures": 0,
                "cost_delta": 0.0,
                "latency_delta": 0.0,
            },
            "output_artifacts": [],
        }
        result = {**default, **self.outcomes.get(phase_id, {})}
        self.side_effect_counts[idempotency_key] += 1
        self.results[idempotency_key] = dict(result)
        return result

    def reconcile(
        self,
        phase: dict[str, Any],
        *,
        idempotency_key: str,
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        result = self.results.get(idempotency_key)
        return dict(result) if isinstance(result, dict) else None


class CommandControllerAdapter:
    """Explicit subprocess adapter for reviewed external/local job commands."""

    def __init__(self, *, allow_external: bool, environment: dict[str, str] | None = None) -> None:
        self.allow_external = allow_external
        self.environment = dict(environment or {})
        self._results: dict[str, dict[str, Any]] = {}

    def execute(
        self,
        phase: dict[str, Any],
        *,
        idempotency_key: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        if idempotency_key in self._results:
            return dict(self._results[idempotency_key])
        durable = self._load_durable_result(phase, idempotency_key, context)
        if durable is not None:
            self._results[idempotency_key] = durable
            return dict(durable)
        command = phase.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(value, str) and value for value in command):
            raise ControllerError(f"phase {phase.get('id')!r} requires a non-empty argv command")
        if phase.get("external_side_effect") is True and not self.allow_external:
            raise ControllerError(f"phase {phase.get('id')!r} requires allow_external=True")
        if phase.get("external_side_effect") is True:
            if not phase.get("result_path") or not phase.get("reconcile_command"):
                raise ControllerError(
                    f"external phase {phase.get('id')!r} requires result_path and reconcile_command for crash recovery"
                )
        rendered = " ".join(command)
        if contains_unredacted_secret_assignment(rendered):
            raise ControllerError("controller commands must not contain credential assignments")
        phase_timeout = _positive_number(phase.get("timeout_seconds"), default=3600.0)
        context_timeout = _number(context.get("timeout_seconds"))
        timeout = min(phase_timeout, context_timeout) if context_timeout > 0 else phase_timeout
        if timeout <= 0:
            raise ControllerError(f"phase {phase.get('id')!r} has no remaining duration budget")
        environment = os.environ.copy()
        environment.update(self.environment)
        environment["HFR_IDEMPOTENCY_KEY"] = idempotency_key
        environment["HFR_FENCING_TOKEN"] = str(context.get("fencing_token") or "")
        environment["HFR_MAX_COST_USD"] = str(context.get("cost_ceiling_usd") or 0.0)
        environment["HFR_TIMEOUT_SECONDS"] = str(context.get("timeout_seconds") or timeout)
        started_at = time.monotonic()
        completed = subprocess.run(
            command,
            cwd=str(phase.get("working_directory")) if phase.get("working_directory") else None,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        measured_duration = time.monotonic() - started_at
        result_path = Path(str(phase.get("result_path"))) if phase.get("result_path") else None
        result_payload: dict[str, Any] = {}
        if result_path is not None and result_path.is_file():
            loaded = json.loads(result_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ControllerError(f"phase result must be an object: {result_path}")
            result_payload = loaded
        if phase.get("external_side_effect") is True:
            if result_payload.get("idempotency_key") != idempotency_key:
                raise ControllerError("external phase result must echo the HFR idempotency key")
            if int(result_payload.get("fencing_token", -1)) != int(context.get("fencing_token", -2)):
                raise ControllerError("external phase result must echo the current HFR fencing token")
        result = {
            "passed": completed.returncode == 0 and result_payload.get("passed", True) is True,
            "returncode": completed.returncode,
            "cost_usd": _number(result_payload.get("cost_usd")),
            "duration_seconds": max(
                measured_duration,
                _number(result_payload.get("duration_seconds")),
            ),
            "metrics": result_payload.get("metrics") if isinstance(result_payload.get("metrics"), dict) else {},
            "output_artifacts": result_payload.get("output_artifacts")
            if isinstance(result_payload.get("output_artifacts"), list)
            else [],
            "stdout_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
            "stderr_sha256": hashlib.sha256(completed.stderr.encode("utf-8")).hexdigest(),
            "result": result_payload,
        }
        self._results[idempotency_key] = result
        self._write_durable_result(phase, idempotency_key, context, result)
        return dict(result)

    def reconcile(
        self,
        phase: dict[str, Any],
        *,
        idempotency_key: str,
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        durable = self._load_durable_result(phase, idempotency_key, context)
        if durable is not None:
            return durable
        if phase.get("external_side_effect") is not True:
            return None
        command = phase.get("reconcile_command")
        result_path = Path(str(phase.get("result_path"))) if phase.get("result_path") else None
        if not isinstance(command, list) or not command or result_path is None:
            raise ControllerError(f"external phase {phase.get('id')!r} has no durable reconciliation contract")
        environment = os.environ.copy()
        environment.update(self.environment)
        environment["HFR_IDEMPOTENCY_KEY"] = idempotency_key
        environment["HFR_FENCING_TOKEN"] = str(context.get("fencing_token") or "")
        environment["HFR_MAX_COST_USD"] = str(context.get("cost_ceiling_usd") or 0.0)
        environment["HFR_TIMEOUT_SECONDS"] = str(context.get("timeout_seconds") or phase.get("timeout_seconds") or 0)
        phase_timeout = _positive_number(phase.get("timeout_seconds"), default=3600.0)
        context_timeout = _number(context.get("timeout_seconds"))
        timeout = min(phase_timeout, context_timeout) if context_timeout > 0 else phase_timeout
        if timeout <= 0:
            raise ControllerError(f"phase {phase.get('id')!r} has no remaining reconciliation duration budget")
        completed = subprocess.run(
            command,
            cwd=str(phase.get("working_directory")) if phase.get("working_directory") else None,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if completed.returncode != 0 or not result_path.is_file():
            return None
        payload = _read_json(result_path)
        if payload.get("idempotency_key") != idempotency_key:
            raise ControllerError("reconciled result idempotency key does not match the prepared operation")
        if int(payload.get("fencing_token", -1)) != int(context.get("fencing_token", -2)):
            raise ControllerError("reconciled result fencing token is stale")
        result = {
            "passed": payload.get("passed") is True,
            "returncode": 0,
            "cost_usd": _number(payload.get("cost_usd")),
            "duration_seconds": _number(payload.get("duration_seconds")),
            "metrics": payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {},
            "output_artifacts": payload.get("output_artifacts") if isinstance(payload.get("output_artifacts"), list) else [],
            "stdout_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
            "stderr_sha256": hashlib.sha256(completed.stderr.encode("utf-8")).hexdigest(),
            "result": payload,
            "reconciled": True,
        }
        self._write_durable_result(phase, idempotency_key, context, result)
        return result

    @staticmethod
    def _durable_result_path(phase: dict[str, Any]) -> Path | None:
        if not phase.get("result_path"):
            return None
        result_path = Path(str(phase["result_path"]))
        return result_path.with_name(f".{result_path.name}.hfr-adapter-result.json")

    def _load_durable_result(
        self,
        phase: dict[str, Any],
        idempotency_key: str,
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        path = self._durable_result_path(phase)
        if path is None or not path.exists():
            return None
        envelope = _read_json(path)
        if envelope.get("idempotency_key") != idempotency_key:
            raise ControllerError(f"durable adapter result is bound to another operation: {path}")
        if int(envelope.get("fencing_token", -1)) > int(context.get("fencing_token", -1)):
            raise ControllerError(f"durable adapter result has a newer fencing token: {path}")
        result = envelope.get("result")
        if not isinstance(result, dict):
            raise ControllerError(f"durable adapter result payload is invalid: {path}")
        return result

    def _write_durable_result(
        self,
        phase: dict[str, Any],
        idempotency_key: str,
        context: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        path = self._durable_result_path(phase)
        if path is None:
            return
        atomic_write_json_cas(
            path,
            {
                "idempotency_key": idempotency_key,
                "fencing_token": int(context.get("fencing_token") or 0),
                "result": result,
            },
            expected_sha256=json_file_sha256(path),
        )


def build_controller_plan(
    *,
    controller_id: str,
    artifact_dir: str | Path,
    candidate_model: str,
    champion_model: str,
    canary_percentages: list[int],
    budget: dict[str, Any],
    deadline_at: str | datetime,
    canary_guardrails: dict[str, Any] | None = None,
    max_retries: int = 2,
    phase_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a deterministic complete-loop execution plan."""

    for label, value in (("controller_id", controller_id), ("candidate_model", candidate_model), ("champion_model", champion_model)):
        if not isinstance(value, str) or not value.strip():
            raise ControllerError(f"{label} must be a non-empty string")
    if candidate_model == champion_model:
        raise ControllerError("candidate_model and champion_model must differ")
    if canary_percentages != sorted(set(canary_percentages)):
        raise ControllerError("canary percentages must be unique and strictly increasing")
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 or value > 100 for value in canary_percentages):
        raise ControllerError("canary percentages must be integers from 1 to 100")
    if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0:
        raise ControllerError("max_retries must be a non-negative integer")
    normalized_budget = _budget(budget)
    built_at = _datetime(None)
    deadline = _datetime(deadline_at)
    if deadline <= built_at:
        raise ControllerError("deadline_at must be in the future when the controller plan is built")
    guardrails = {**_DEFAULT_GUARDRAILS, **(canary_guardrails or {})}
    overrides = phase_overrides or {}
    phase_specs = [
        ("collect", False, False),
        ("govern_curate", False, False),
        ("train", True, True),
        ("serve", True, True),
        ("evaluate", True, False),
        ("promotion_gate", False, False),
        ("shadow", True, True),
        *[(f"canary_{percentage:03d}", True, True) for percentage in canary_percentages],
        ("promote", True, True),
        ("monitor", False, False),
        ("next_iteration", False, False),
    ]
    phases: list[dict[str, Any]] = []
    previous = ""
    for phase_id, approval, side_effect in phase_specs:
        override = overrides.get(phase_id, {})
        _validate_phase_override(phase_id, override)
        phase = {
            "id": phase_id,
            "depends_on": [previous] if previous else [],
            "requires_approval": approval,
            "external_side_effect": side_effect,
            "max_retries": max_retries,
            "estimated_cost_usd": 0.0,
            "timeout_seconds": normalized_budget["max_duration_seconds"],
        }
        phase.update(override)
        _require_external_reconciliation_contract(phase)
        if phase_id.startswith("canary_"):
            phase["traffic_percentage"] = int(phase_id.rsplit("_", 1)[1])
            phase["guardrails"] = guardrails
        phases.append(phase)
        previous = phase_id
    known_override_ids = {phase_id for phase_id, _, _ in phase_specs} | set(_AUXILIARY_PHASE_SPECS)
    unknown_override_ids = sorted(set(overrides) - known_override_ids)
    if unknown_override_ids:
        raise ControllerError(f"phase overrides reference unknown phases: {unknown_override_ids!r}")
    failure_phases: dict[str, dict[str, Any]] = {}
    for phase_id, (approval, side_effect) in _AUXILIARY_PHASE_SPECS.items():
        override = overrides.get(phase_id, {})
        _validate_phase_override(phase_id, override)
        auxiliary = {
            "id": phase_id,
            "requires_approval": approval,
            "external_side_effect": side_effect,
            "max_retries": max_retries,
            "estimated_cost_usd": 0.0,
            "timeout_seconds": normalized_budget["max_duration_seconds"],
        }
        auxiliary.update(override)
        _require_external_reconciliation_contract(auxiliary)
        failure_phases[phase_id] = auxiliary
    plan_without_fingerprint = {
        "schema_version": CONTROLLER_PLAN_SCHEMA_VERSION,
        "controller_id": controller_id,
        "artifact_dir": str(Path(artifact_dir).resolve()),
        "candidate_model": candidate_model,
        "champion_model": champion_model,
        "budget": normalized_budget,
        "deadline_at": deadline.isoformat(),
        "canary_guardrails": guardrails,
        "phases": phases,
        "failure_phases": failure_phases,
        "execution_boundary": {
            "opt_in": True,
            "core_side_effect_free": True,
            "external_actions_require_plan_bound_approval": True,
            "adapter_idempotency_required": True,
            "durable_reconciliation_required_after_inflight_crash": True,
            "fencing_tokens_required": True,
            "hard_deadline_required": True,
        },
    }
    return {
        **plan_without_fingerprint,
        "plan_fingerprint": _canonical_sha256(plan_without_fingerprint),
    }


def _validate_phase_override(phase_id: str, override: Any) -> None:
    if not isinstance(override, dict):
        raise ControllerError(f"phase override {phase_id!r} must be an object")
    forbidden = sorted(set(override) - _OVERRIDABLE_PHASE_FIELDS)
    if forbidden:
        raise ControllerError(
            f"phase override {phase_id!r} cannot change identity, dependencies, approval, or side-effect safety: {forbidden!r}"
        )
    if "command" in override and (
        not isinstance(override["command"], list)
        or not override["command"]
        or not all(isinstance(value, str) and value for value in override["command"])
    ):
        raise ControllerError(f"phase override {phase_id!r} command must be a non-empty argv list")
    if "reconcile_command" in override and (
        not isinstance(override["reconcile_command"], list)
        or not override["reconcile_command"]
        or not all(isinstance(value, str) and value for value in override["reconcile_command"])
    ):
        raise ControllerError(f"phase override {phase_id!r} reconcile_command must be a non-empty argv list")


def _require_external_reconciliation_contract(phase: dict[str, Any]) -> None:
    if phase.get("external_side_effect") is not True or "command" not in phase:
        return
    if not phase.get("result_path") or not phase.get("reconcile_command"):
        raise ControllerError(
            f"external command phase {phase.get('id')!r} requires result_path and reconcile_command"
        )


def run_controller(
    plan: dict[str, Any],
    *,
    state_path: str | Path,
    adapter: ControllerAdapter,
    approvals: dict[str, str],
    owner_id: str,
    now: str | datetime | None = None,
    lease_seconds: int = 300,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Run or resume under an inter-process lock for exactly one state file."""

    target = Path(state_path)
    with _controller_process_lock(target, owner_id):
        return _run_controller_locked(
            plan,
            state_path=target,
            adapter=adapter,
            approvals=approvals,
            owner_id=owner_id,
            now=now,
            lease_seconds=lease_seconds,
            max_steps=max_steps,
        )


def _run_controller_locked(
    plan: dict[str, Any],
    *,
    state_path: str | Path,
    adapter: ControllerAdapter,
    approvals: dict[str, str],
    owner_id: str,
    now: str | datetime | None = None,
    lease_seconds: int = 300,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Run or resume a controller plan, stopping safely at any failed gate."""

    _validate_plan(plan)
    if not owner_id:
        raise ControllerError("owner_id must be non-empty")
    target = Path(state_path)
    current = _read_json_optional(target)
    if current and current.get("status") in _TERMINAL_STATUSES:
        _require_plan_binding(current, plan)
        _validate_state_replay(current, plan, target)
        return current
    instant = _datetime(now)
    state = _initialize_or_load_state(current, plan, target, instant)
    _acquire_lease(state, owner_id, instant, lease_seconds)
    _write_state(target, state)

    steps = 0
    try:
        for phase in plan["phases"]:
            phase_id = str(phase["id"])
            existing = _phase_record(state, phase_id)
            if existing and existing.get("status") == "completed":
                continue
            if max_steps is not None and steps >= max_steps:
                state["status"] = "paused"
                state["current_phase"] = phase_id
                return _persist_and_release(target, state, owner_id)
            state["current_phase"] = phase_id
            dependencies = phase.get("depends_on") if isinstance(phase.get("depends_on"), list) else []
            if any(not _phase_completed(state, dependency) for dependency in dependencies):
                state["status"] = "dependency_blocked"
                return _persist_and_release(target, state, owner_id)
            if phase.get("requires_approval") is True and approvals.get(phase_id) != plan["plan_fingerprint"]:
                state["status"] = "approval_required"
                state["pending_approvals"] = sorted(
                    set(state.get("pending_approvals", [])) | {phase_id}
                )
                return _persist_and_release(target, state, owner_id)
            state["pending_approvals"] = [value for value in state.get("pending_approvals", []) if value != phase_id]
            if _budget_would_exceed(state, plan, phase):
                state["status"] = "budget_exhausted"
                return _persist_and_release(target, state, owner_id)
            record = existing or _new_phase_record(plan, phase, state)
            if existing is None:
                state["phase_records"].append(record)
            if phase.get("requires_approval") is True:
                record["approval_fingerprint"] = plan["plan_fingerprint"]
            max_attempts = int(phase.get("max_retries", 0)) + 1
            if int(record.get("attempt_count", 0)) >= max_attempts:
                state["status"] = "failed"
                return _persist_and_release(target, state, owner_id)
            was_inflight = record.get("status") == "running" and record.get("operation_status") == "prepared"
            record["attempt_count"] = int(record.get("attempt_count", 0)) + 1
            record["status"] = "running"
            record["operation_status"] = "prepared"
            if not was_inflight:
                record["fencing_token"] = int(state.get("lease", {}).get("fencing_token") or 0)
            record["last_started_at"] = instant.isoformat()
            state["total_attempts"] = int(state.get("total_attempts", 0)) + 1
            effective_timeout = _effective_phase_timeout(state, plan, phase)
            _renew_lease(
                state,
                owner_id,
                _datetime(None),
                max(lease_seconds, int(effective_timeout) + 30),
            )
            _write_state(target, state)
            try:
                context = _adapter_context(plan, state, phase, record=record)
                adapter_result = None
                if was_inflight:
                    reconcile = getattr(adapter, "reconcile", None)
                    if callable(reconcile):
                        adapter_result = reconcile(
                            phase,
                            idempotency_key=record["idempotency_key"],
                            context=context,
                        )
                    if adapter_result is None:
                        record["status"] = "reconciliation_required"
                        state["status"] = "reconciliation_required"
                        return _persist_and_release(target, state, owner_id)
                if adapter_result is None:
                    adapter_result = adapter.execute(
                        phase,
                        idempotency_key=record["idempotency_key"],
                        context=context,
                    )
                _validate_result_budget(adapter_result, state, plan, phase, context)
                record["operation_status"] = "committed"
                _write_state(target, state)
            except RetryableControllerError as exc:
                if record["attempt_count"] < max_attempts:
                    record["status"] = "retryable_failure"
                    record["last_error"] = redact_text(str(exc))
                    state["status"] = "retryable_failure"
                    return _persist_and_release(target, state, owner_id)
                return _handle_execution_failure(
                    plan,
                    state,
                    target,
                    adapter,
                    owner_id,
                    instant,
                    phase,
                    record,
                    exc,
                    outcome_uncertain=False,
                )
            except Exception as exc:
                return _handle_execution_failure(
                    plan,
                    state,
                    target,
                    adapter,
                    owner_id,
                    instant,
                    phase,
                    record,
                    exc,
                    outcome_uncertain=phase.get("external_side_effect") is True,
                )

            receipt = _phase_receipt(plan, state, phase, record, adapter_result, instant)
            receipt_path = _write_phase_receipt(plan, phase_id, receipt)
            record.update(
                {
                    "status": "completed" if receipt["passed"] else "failed",
                    "receipt_path": str(receipt_path.resolve()),
                    "receipt_sha256": _sha256_file(receipt_path),
                    "completed_at": instant.isoformat(),
                    "cost_usd": receipt["cost_usd"],
                    "duration_seconds": receipt["duration_seconds"],
                    "metrics": receipt["metrics"],
                    "last_error": "",
                    "operation_status": "receipted",
                }
            )
            state["spent_cost_usd"] = round(_number(state.get("spent_cost_usd")) + receipt["cost_usd"], 8)
            state["elapsed_duration_seconds"] = round(
                _number(state.get("elapsed_duration_seconds")) + receipt["duration_seconds"],
                6,
            )
            if _budget_exceeded(state, plan):
                state["status"] = "budget_exhausted"
                return _persist_and_release(target, state, owner_id)
            if not receipt["passed"]:
                _route_failure(plan, state, target, adapter, owner_id, instant, phase_id, receipt)
                if phase_id.startswith("canary_"):
                    return _rollback(plan, state, target, adapter, approvals, owner_id, instant, phase_id)
                state["status"] = "repair_required"
                state["next_iteration"] = _next_iteration(plan, state, reason=f"{phase_id}_failed")
                return _persist_and_release(target, state, owner_id)
            if phase_id.startswith("canary_") and not _guardrails_pass(receipt["metrics"], phase.get("guardrails", {})):
                record["status"] = "guardrail_failed"
                _route_failure(plan, state, target, adapter, owner_id, instant, phase_id, receipt)
                return _rollback(plan, state, target, adapter, approvals, owner_id, instant, phase_id)
            if phase_id == "promote":
                state["active_model"] = plan["candidate_model"]
                state["rollback_model"] = plan["champion_model"]
            if phase_id == "next_iteration":
                state["next_iteration"] = _next_iteration(plan, state, reason="champion_monitoring")
            state["status"] = "running"
            steps += 1
            _renew_lease(state, owner_id, _datetime(None), lease_seconds)
            _write_state(target, state)

        state["status"] = "complete"
        state["current_phase"] = ""
        state["completed_at"] = instant.isoformat()
        return _persist_and_release(target, state, owner_id)
    finally:
        latest = _read_json_optional(target)
        if latest and latest.get("lease", {}).get("owner_id") == owner_id:
            latest["lease"] = {
                "owner_id": "",
                "expires_at": "",
                "heartbeat_at": latest.get("lease", {}).get("heartbeat_at", ""),
                "fencing_token": int(latest.get("lease", {}).get("fencing_token") or 0),
            }
            _write_state(target, latest)


def _handle_execution_failure(
    plan: dict[str, Any],
    state: dict[str, Any],
    state_path: Path,
    adapter: ControllerAdapter,
    owner_id: str,
    instant: datetime,
    phase: dict[str, Any],
    record: dict[str, Any],
    error: Exception,
    *,
    outcome_uncertain: bool,
) -> dict[str, Any]:
    phase_id = str(phase["id"])
    safe_error = redact_text(str(error))
    record["last_error"] = safe_error
    failure_result = {
        "passed": False,
        "cost_usd": 0.0,
        "duration_seconds": 0.0,
        "metrics": {"failure_modes": ["runtime_exception"]},
        "output_artifacts": [],
        "failure_modes": ["runtime_exception"],
        "error_type": type(error).__name__,
        "error": safe_error,
        "outcome_uncertain": outcome_uncertain,
    }
    if outcome_uncertain:
        record["status"] = "reconciliation_required"
        record["operation_status"] = "prepared"
        failed_receipt = _phase_receipt(plan, state, phase, record, failure_result, instant)
    else:
        record["operation_status"] = "committed"
        failed_receipt = _phase_receipt(plan, state, phase, record, failure_result, instant)
        receipt_path = _write_phase_receipt(plan, phase_id, failed_receipt)
        record.update(
            {
                "status": "failed",
                "receipt_path": str(receipt_path.resolve()),
                "receipt_sha256": _sha256_file(receipt_path),
                "completed_at": instant.isoformat(),
                "operation_status": "receipted",
                "cost_usd": 0.0,
                "duration_seconds": 0.0,
                "metrics": failed_receipt["metrics"],
            }
        )
    _write_state(state_path, state)
    _route_failure(plan, state, state_path, adapter, owner_id, instant, phase_id, failed_receipt)
    state["next_iteration"] = _next_iteration(plan, state, reason=f"{phase_id}_execution_failure")
    state["status"] = "reconciliation_required" if outcome_uncertain else "repair_required"
    return _persist_and_release(state_path, state, owner_id)


@contextmanager
def _controller_process_lock(state_path: Path, owner_id: str):
    """Serialize controllers sharing a filesystem; kernel locks release on crash."""

    lock_path = state_path.with_name(f".{state_path.name}.controller.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    locked = False
    fallback_created = False
    fallback_path = lock_path.with_suffix(lock_path.suffix + ".exclusive")
    try:
        if fcntl is not None:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ControllerError(f"controller state is locked by another owner: {state_path}") from exc
            locked = True
        else:  # pragma: no cover - POSIX is the supported test/runtime surface
            try:
                fallback_descriptor = os.open(fallback_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError as exc:
                raise ControllerError(f"controller state is locked by another owner: {state_path}") from exc
            else:
                os.close(fallback_descriptor)
                fallback_created = True
        metadata = json.dumps({"owner_id": owner_id, "state_path": str(state_path.resolve())}, sort_keys=True).encode("utf-8")
        os.ftruncate(descriptor, 0)
        os.write(descriptor, metadata)
        os.fsync(descriptor)
        yield
    finally:
        if locked and fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        if fallback_created:
            try:
                fallback_path.unlink()
            except FileNotFoundError:
                pass


def _rollback(
    plan: dict[str, Any],
    state: dict[str, Any],
    state_path: Path,
    adapter: ControllerAdapter,
    approvals: dict[str, str],
    owner_id: str,
    instant: datetime,
    failed_phase: str,
) -> dict[str, Any]:
    if approvals.get("rollback") != plan["plan_fingerprint"]:
        state["status"] = "rollback_approval_required"
        state["pending_approvals"] = sorted(set(state.get("pending_approvals", [])) | {"rollback"})
        return _persist_and_release(state_path, state, owner_id)
    if approvals.get("post_rollback_smoke") != plan["plan_fingerprint"]:
        state["status"] = "rollback_approval_required"
        state["pending_approvals"] = sorted(
            set(state.get("pending_approvals", [])) | {"post_rollback_smoke"}
        )
        return _persist_and_release(state_path, state, owner_id)
    rollback_receipt = _execute_auxiliary_phase(
        plan,
        state,
        state_path,
        adapter,
        phase_id="rollback",
        instant=instant,
        context={"failed_phase": failed_phase, "restore_model": plan["champion_model"]},
        owner_id=owner_id,
        approval_fingerprint=plan["plan_fingerprint"],
    )
    if not rollback_receipt["passed"]:
        state["status"] = "rollback_failed"
        return _persist_and_release(state_path, state, owner_id)
    state["active_model"] = plan["champion_model"]
    smoke = _execute_auxiliary_phase(
        plan,
        state,
        state_path,
        adapter,
        phase_id="post_rollback_smoke",
        instant=instant,
        context={"expected_model": plan["champion_model"]},
        owner_id=owner_id,
        approval_fingerprint=plan["plan_fingerprint"],
    )
    state["rollback"] = {
        "failed_phase": failed_phase,
        "restored_model": plan["champion_model"],
        "rollback_receipt_sha256": rollback_receipt["receipt_sha256"],
        "post_rollback_smoke_passed": smoke["passed"],
        "post_rollback_smoke_receipt_sha256": smoke["receipt_sha256"],
    }
    state["next_iteration"] = _next_iteration(plan, state, reason=f"rollback_after_{failed_phase}")
    state["status"] = "rolled_back" if smoke["passed"] else "rollback_smoke_failed"
    state["current_phase"] = ""
    return _persist_and_release(state_path, state, owner_id)


def _route_failure(
    plan: dict[str, Any],
    state: dict[str, Any],
    state_path: Path,
    adapter: ControllerAdapter,
    owner_id: str,
    instant: datetime,
    failed_phase: str,
    failed_receipt: dict[str, Any],
) -> None:
    cluster = _controller_failure_cluster(failed_phase, failed_receipt)
    analysis = _execute_auxiliary_phase(
        plan,
        state,
        state_path,
        adapter,
        phase_id="failure_analysis",
        instant=instant,
        context={
            "failed_phase": failed_phase,
            "failed_receipt_sha256": _canonical_sha256(failed_receipt),
            "failed_metrics": failed_receipt.get("metrics", {}),
            "failure_cluster": cluster,
        },
        owner_id=owner_id,
    )
    analyzed_cluster = _analyzed_failure_cluster(analysis, cluster)
    intervention = route_failure_cluster(analyzed_cluster)
    route = _execute_auxiliary_phase(
        plan,
        state,
        state_path,
        adapter,
        phase_id="intervention_route",
        instant=instant,
        context={
            "failed_phase": failed_phase,
            "failure_analysis_receipt_sha256": analysis["receipt_sha256"],
            "intervention_route": intervention,
        },
        owner_id=owner_id,
    )
    state["failure_analysis"] = {
        "failed_phase": failed_phase,
        "analysis_passed": analysis["passed"],
        "analysis_receipt_sha256": analysis["receipt_sha256"],
        "intervention_route_passed": route["passed"],
        "intervention_route_receipt_sha256": route["receipt_sha256"],
        "failure_cluster": analyzed_cluster,
        "intervention_route": intervention,
        "selected_intervention": intervention["selected_intervention"],
        "work_item": intervention["work_item"],
    }
    _write_state(state_path, state)


def _analyzed_failure_cluster(
    analysis_receipt: dict[str, Any],
    fallback: dict[str, Any],
) -> dict[str, Any]:
    adapter_result = (
        analysis_receipt.get("adapter_result")
        if isinstance(analysis_receipt.get("adapter_result"), dict)
        else {}
    )
    candidate = adapter_result.get("failure_cluster")
    if not isinstance(candidate, dict):
        return fallback
    # The router is the canonical validator for analyzed clusters. Calling it
    # here prevents an adapter from injecting an unrouteable result.
    route_failure_cluster(candidate)
    return candidate


def _controller_failure_cluster(failed_phase: str, failed_receipt: dict[str, Any]) -> dict[str, Any]:
    metrics = failed_receipt.get("metrics") if isinstance(failed_receipt.get("metrics"), dict) else {}
    adapter_result = (
        failed_receipt.get("adapter_result")
        if isinstance(failed_receipt.get("adapter_result"), dict)
        else {}
    )
    declared_modes = adapter_result.get("failure_modes", metrics.get("failure_modes"))
    failure_modes = sorted(
        {
            str(value).strip()
            for value in declared_modes if isinstance(value, str) and str(value).strip()
        }
    ) if isinstance(declared_modes, list) else []
    if not failure_modes:
        if _non_negative_int(metrics.get("critical_failures")) > 0:
            failure_modes = ["forbidden_action"]
        elif failed_phase in {"collect", "govern_curate"}:
            failure_modes = ["training_data_contamination"]
        elif failed_phase in {"evaluate", "promotion_gate"}:
            failure_modes = ["insufficient_eval_repeats"]
        elif failed_phase in {"shadow", "monitor"} or failed_phase.startswith("canary_"):
            failure_modes = ["low_final_answer_quality"]
        else:
            failure_modes = ["runtime_exception"]
    critical = _non_negative_int(metrics.get("critical_failures")) > 0
    return {
        "cluster_id": f"{failed_phase}-{_canonical_sha256(failed_receipt)[:16]}",
        "failure_modes": failure_modes,
        "severity": "critical" if critical else "high",
        "confidence": 0.95 if declared_modes else 0.75,
        "frequency": 1,
        "affected_task_families": [
            str(value) for value in metrics.get("affected_task_families", []) if isinstance(value, str)
        ],
        "affected_tools": [
            str(value) for value in metrics.get("affected_tools", []) if isinstance(value, str)
        ],
        "affected_policies": [
            str(value) for value in metrics.get("affected_policies", []) if isinstance(value, str)
        ],
        "evidence_refs": [
            {
                "artifact": "agentic_loop_phase_receipt",
                "phase_id": failed_phase,
                "sha256": _canonical_sha256(failed_receipt),
            }
        ],
    }


def _execute_auxiliary_phase(
    plan: dict[str, Any],
    state: dict[str, Any],
    state_path: Path,
    adapter: ControllerAdapter,
    *,
    phase_id: str,
    instant: datetime,
    context: dict[str, Any],
    owner_id: str,
    approval_fingerprint: str = "",
) -> dict[str, Any]:
    phase = plan.get("failure_phases", {}).get(phase_id)
    if not isinstance(phase, dict):
        raise ControllerError(f"auxiliary phase {phase_id!r} is not bound into the controller plan")
    existing = _phase_record(state, phase_id)
    record = existing or _new_phase_record(plan, phase, state)
    if existing is None:
        state["phase_records"].append(record)
    if phase.get("requires_approval") is True:
        if approval_fingerprint != plan["plan_fingerprint"]:
            raise ControllerError(f"auxiliary phase {phase_id!r} requires plan-bound approval")
        record["approval_fingerprint"] = approval_fingerprint
    if record.get("status") == "completed" and record.get("receipt_path"):
        receipt = _read_json(Path(record["receipt_path"]))
        return {**receipt, "receipt_sha256": record["receipt_sha256"]}
    if _budget_would_exceed(state, plan, phase):
        raise ControllerError(f"controller budget cannot admit auxiliary phase {phase_id!r}")
    max_attempts = int(phase.get("max_retries", 0)) + 1
    if int(record.get("attempt_count", 0)) >= max_attempts:
        raise ControllerError(f"auxiliary phase {phase_id!r} exhausted its retry budget")
    was_inflight = record.get("status") == "running" and record.get("operation_status") == "prepared"
    record["attempt_count"] = int(record.get("attempt_count", 0)) + 1
    record["status"] = "running"
    record["operation_status"] = "prepared"
    if not was_inflight:
        record["fencing_token"] = int(state.get("lease", {}).get("fencing_token") or 0)
    state["total_attempts"] = int(state.get("total_attempts", 0)) + 1
    _write_state(state_path, state)
    adapter_context = {
        **_adapter_context(plan, state, phase, record=record),
        **context,
    }
    result = None
    if was_inflight:
        reconcile = getattr(adapter, "reconcile", None)
        if callable(reconcile):
            result = reconcile(phase, idempotency_key=record["idempotency_key"], context=adapter_context)
        if result is None:
            raise ControllerError(f"auxiliary phase {phase_id!r} requires reconciliation before retry")
    if result is None:
        result = adapter.execute(
            phase,
            idempotency_key=record["idempotency_key"],
            context=adapter_context,
        )
    _validate_result_budget(result, state, plan, phase, adapter_context)
    record["operation_status"] = "committed"
    _write_state(state_path, state)
    receipt = _phase_receipt(plan, state, phase, record, result, instant)
    receipt_path = _write_phase_receipt(plan, phase_id, receipt)
    record.update(
        {
            "status": "completed" if receipt["passed"] else "failed",
            "receipt_path": str(receipt_path.resolve()),
            "receipt_sha256": _sha256_file(receipt_path),
            "completed_at": instant.isoformat(),
            "operation_status": "receipted",
            "cost_usd": receipt["cost_usd"],
            "duration_seconds": receipt["duration_seconds"],
            "metrics": receipt["metrics"],
        }
    )
    state["spent_cost_usd"] = round(_number(state.get("spent_cost_usd")) + receipt["cost_usd"], 8)
    state["elapsed_duration_seconds"] = round(
        _number(state.get("elapsed_duration_seconds")) + receipt["duration_seconds"], 6
    )
    _renew_lease(state, owner_id, _datetime(None), 300)
    _write_state(state_path, state)
    return {**receipt, "receipt_sha256": record["receipt_sha256"]}


def _phase_receipt(
    plan: dict[str, Any],
    state: dict[str, Any],
    phase: dict[str, Any],
    record: dict[str, Any],
    result: dict[str, Any],
    instant: datetime,
) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ControllerError(f"adapter result for {phase.get('id')!r} must be an object")
    return {
        "schema_version": CONTROLLER_PHASE_RECEIPT_SCHEMA_VERSION,
        "controller_id": plan["controller_id"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "phase_id": phase["id"],
        "idempotency_key": record["idempotency_key"],
        "input_fingerprint": record["input_fingerprint"],
        "attempt": record["attempt_count"],
        "approval_fingerprint": str(record.get("approval_fingerprint") or ""),
        "passed": result.get("passed") is True,
        "cost_usd": max(0.0, _number(result.get("cost_usd"))),
        "duration_seconds": max(0.0, _number(result.get("duration_seconds"))),
        "metrics": result.get("metrics") if isinstance(result.get("metrics"), dict) else {},
        "output_artifacts": result.get("output_artifacts") if isinstance(result.get("output_artifacts"), list) else [],
        "adapter_result": result,
        "created_at": instant.isoformat(),
    }


def _new_phase_record(plan: dict[str, Any], phase: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    phase_id = str(phase["id"])
    dependencies = [
        {"phase_id": row["phase_id"], "receipt_sha256": row.get("receipt_sha256", "")}
        for row in state.get("phase_records", [])
        if row.get("status") == "completed"
    ]
    input_fingerprint = _canonical_sha256(
        {"plan": plan["plan_fingerprint"], "phase": phase, "dependencies": dependencies}
    )
    return {
        "phase_id": phase_id,
        "status": "pending",
        "attempt_count": 0,
        "input_fingerprint": input_fingerprint,
        "idempotency_key": f"hfr-{plan['controller_id']}-{phase_id}-{input_fingerprint[:16]}",
        "receipt_path": "",
        "receipt_sha256": "",
        "operation_status": "not_started",
        "fencing_token": 0,
        "approval_fingerprint": "",
    }


def _initialize_or_load_state(
    current: dict[str, Any] | None,
    plan: dict[str, Any],
    state_path: Path,
    instant: datetime,
) -> dict[str, Any]:
    if current is not None:
        _require_plan_binding(current, plan)
        _validate_state_replay(current, plan, state_path)
        return current
    return {
        "schema_version": CONTROLLER_STATE_SCHEMA_VERSION,
        "controller_id": plan["controller_id"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "state_path": str(state_path.resolve()),
        "status": "running",
        "current_phase": "",
        "active_model": plan["champion_model"],
        "rollback_model": plan["champion_model"],
        "started_at": instant.isoformat(),
        "deadline_at": plan["deadline_at"],
        "completed_at": "",
        "spent_cost_usd": 0.0,
        "elapsed_duration_seconds": 0.0,
        "total_attempts": 0,
        "pending_approvals": [],
        "phase_records": [],
        "rollback": {},
        "failure_analysis": {},
        "next_iteration": {"scheduled": False},
        "lease": {"owner_id": "", "expires_at": "", "heartbeat_at": "", "fencing_token": 0},
    }


def _acquire_lease(state: dict[str, Any], owner_id: str, instant: datetime, lease_seconds: int) -> None:
    lease = state.get("lease") if isinstance(state.get("lease"), dict) else {}
    existing_owner = str(lease.get("owner_id") or "")
    expiry = _optional_datetime(lease.get("expires_at"))
    if existing_owner and existing_owner != owner_id and expiry is not None and expiry > instant:
        raise ControllerError(f"controller lease is held by {existing_owner!r} until {expiry.isoformat()}")
    fencing_token = int(lease.get("fencing_token") or 0) + 1
    state["lease"] = {
        "owner_id": owner_id,
        "expires_at": (instant + timedelta(seconds=max(1, lease_seconds))).isoformat(),
        "heartbeat_at": instant.isoformat(),
        "fencing_token": fencing_token,
    }


def _renew_lease(state: dict[str, Any], owner_id: str, instant: datetime, lease_seconds: int) -> None:
    lease = state.get("lease") if isinstance(state.get("lease"), dict) else {}
    if lease.get("owner_id") != owner_id:
        raise ControllerError("cannot renew a controller lease owned by another worker")
    lease["heartbeat_at"] = instant.isoformat()
    lease["expires_at"] = (instant + timedelta(seconds=max(1, lease_seconds))).isoformat()


def _persist_and_release(
    path: Path,
    state: dict[str, Any],
    owner_id: str,
) -> dict[str, Any]:
    if state.get("lease", {}).get("owner_id") == owner_id:
        lease = state["lease"]
        state["lease"] = {
            "owner_id": "",
            "expires_at": "",
            "heartbeat_at": lease.get("heartbeat_at", ""),
            "fencing_token": int(lease.get("fencing_token") or 0),
        }
    _write_state(path, state)
    return state


def _write_state(path: Path, state: dict[str, Any]) -> None:
    atomic_write_json_cas(path, state, expected_sha256=json_file_sha256(path))


def _write_phase_receipt(plan: dict[str, Any], phase_id: str, receipt: dict[str, Any]) -> Path:
    root = Path(plan["artifact_dir"]) / str(plan["controller_id"]) / "receipts"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{phase_id}.json"
    atomic_write_json_cas(path, receipt, expected_sha256=json_file_sha256(path))
    return path


def _adapter_context(
    plan: dict[str, Any],
    state: dict[str, Any],
    phase: dict[str, Any],
    *,
    record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    effective_timeout = _effective_phase_timeout(state, plan, phase)
    return {
        "controller_id": plan["controller_id"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "candidate_model": plan["candidate_model"],
        "champion_model": plan["champion_model"],
        "active_model": state.get("active_model"),
        "phase_id": phase["id"],
        "fencing_token": int((record or {}).get("fencing_token") or state.get("lease", {}).get("fencing_token") or 0),
        "cost_ceiling_usd": max(0.0, _number(phase.get("estimated_cost_usd"))),
        "remaining_cost_budget_usd": max(
            0.0,
            _number(plan.get("budget", {}).get("max_cost_usd")) - _number(state.get("spent_cost_usd")),
        ),
        "deadline_at": plan["deadline_at"],
        "remaining_duration_budget_seconds": _remaining_duration_seconds(state, plan),
        "timeout_seconds": effective_timeout,
        "prior_receipts": [
            {
                "phase_id": row["phase_id"],
                "receipt_path": row.get("receipt_path", ""),
                "receipt_sha256": row.get("receipt_sha256", ""),
            }
            for row in state.get("phase_records", [])
            if row.get("status") == "completed"
        ],
    }


def _next_iteration(plan: dict[str, Any], state: dict[str, Any], *, reason: str) -> dict[str, Any]:
    return {
        "scheduled": True,
        "iteration_id": f"{plan['controller_id']}-next-{len(state.get('phase_records', [])):03d}",
        "reason": reason,
        "source_plan_fingerprint": plan["plan_fingerprint"],
        "active_model": state.get("active_model"),
    }


def _guardrails_pass(metrics: dict[str, Any], guardrails: dict[str, Any]) -> bool:
    effective = {**_DEFAULT_GUARDRAILS, **guardrails}
    return (
        _number(metrics.get("task_success_rate")) >= _number(effective["min_task_success_rate"])
        and _non_negative_int(metrics.get("critical_failures")) <= _non_negative_int(effective["max_critical_failures"])
        and _number(metrics.get("cost_delta")) <= _number(effective["max_cost_delta"])
        and _number(metrics.get("latency_delta")) <= _number(effective["max_latency_delta"])
    )


def _budget(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ControllerError("budget must be an object")
    max_cost = _number(value.get("max_cost_usd"))
    max_duration = _number(value.get("max_duration_seconds"))
    max_attempts = _non_negative_int(value.get("max_attempts"))
    if max_cost < 0 or max_duration <= 0 or max_attempts <= 0:
        raise ControllerError("budget requires non-negative cost and positive duration/attempt limits")
    return {
        "max_cost_usd": max_cost,
        "max_duration_seconds": max_duration,
        "max_attempts": max_attempts,
    }


def _budget_would_exceed(state: dict[str, Any], plan: dict[str, Any], phase: dict[str, Any]) -> bool:
    budget = plan["budget"]
    return (
        _number(state.get("spent_cost_usd")) + max(0.0, _number(phase.get("estimated_cost_usd"))) > budget["max_cost_usd"]
        or _remaining_duration_seconds(state, plan) <= 0
        or _non_negative_int(state.get("total_attempts")) >= budget["max_attempts"]
    )


def _budget_exceeded(state: dict[str, Any], plan: dict[str, Any]) -> bool:
    budget = plan["budget"]
    return (
        _number(state.get("spent_cost_usd")) > budget["max_cost_usd"]
        or _remaining_duration_seconds(state, plan) <= 0
        or _non_negative_int(state.get("total_attempts")) > budget["max_attempts"]
    )


def _validate_result_budget(
    result: Any,
    state: dict[str, Any],
    plan: dict[str, Any],
    phase: dict[str, Any],
    context: dict[str, Any],
) -> None:
    if not isinstance(result, dict):
        raise ControllerError(f"adapter result for {phase.get('id')!r} must be an object")
    cost = _number(result.get("cost_usd"))
    ceiling = max(0.0, _number(phase.get("estimated_cost_usd")))
    remaining = max(0.0, _number(plan["budget"]["max_cost_usd"]) - _number(state.get("spent_cost_usd")))
    if cost < 0 or cost > ceiling + 1e-9 or cost > remaining + 1e-9:
        raise ControllerError(
            f"phase {phase.get('id')!r} violated its provider-enforced cost ceiling "
            f"(actual={cost}, phase_ceiling={ceiling}, remaining_budget={remaining})"
        )
    duration = _number(result.get("duration_seconds"))
    duration_ceiling = max(0.0, _number(context.get("timeout_seconds")))
    if duration < 0 or duration > duration_ceiling + 1e-6:
        raise ControllerError(
            f"phase {phase.get('id')!r} violated its provider-enforced duration ceiling "
            f"(actual={duration}, duration_ceiling={duration_ceiling})"
        )


def _remaining_duration_seconds(state: dict[str, Any], plan: dict[str, Any]) -> float:
    accounted = max(
        0.0,
        _number(plan["budget"]["max_duration_seconds"])
        - _number(state.get("elapsed_duration_seconds")),
    )
    deadline = _optional_datetime(plan.get("deadline_at"))
    if deadline is None:
        raise ControllerError("controller plan deadline_at is invalid")
    wall_clock = max(0.0, (deadline - _datetime(None)).total_seconds())
    return min(accounted, wall_clock)


def _effective_phase_timeout(
    state: dict[str, Any],
    plan: dict[str, Any],
    phase: dict[str, Any],
) -> float:
    return min(
        _positive_number(phase.get("timeout_seconds"), default=3600.0),
        _remaining_duration_seconds(state, plan),
    )


def _validate_plan(plan: dict[str, Any]) -> None:
    if not isinstance(plan, dict) or plan.get("schema_version") != CONTROLLER_PLAN_SCHEMA_VERSION:
        raise ControllerError(f"plan schema_version must be {CONTROLLER_PLAN_SCHEMA_VERSION!r}")
    supplied = plan.get("plan_fingerprint")
    replay = {key: value for key, value in plan.items() if key != "plan_fingerprint"}
    expected = _canonical_sha256(replay)
    if supplied != expected:
        raise ControllerError("controller plan fingerprint does not match plan contents")
    phases = plan.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ControllerError("controller plan phases must be a non-empty list")
    ids = [phase.get("id") for phase in phases if isinstance(phase, dict)]
    if len(ids) != len(phases) or len(set(ids)) != len(ids) or not all(isinstance(value, str) and value for value in ids):
        raise ControllerError("controller phase ids must be unique non-empty strings")
    _budget(plan.get("budget"))
    if _optional_datetime(plan.get("deadline_at")) is None:
        raise ControllerError("controller plan deadline_at must be an absolute date-time")
    fixed_prefix = [
        ("collect", False, False),
        ("govern_curate", False, False),
        ("train", True, True),
        ("serve", True, True),
        ("evaluate", True, False),
        ("promotion_gate", False, False),
        ("shadow", True, True),
    ]
    fixed_suffix = [
        ("promote", True, True),
        ("monitor", False, False),
        ("next_iteration", False, False),
    ]
    if len(phases) < len(fixed_prefix) + len(fixed_suffix):
        raise ControllerError("controller plan is missing mandatory lifecycle phases")
    canary_phases = phases[len(fixed_prefix) : len(phases) - len(fixed_suffix)]
    canary_specs: list[tuple[str, bool, bool]] = []
    percentages: list[int] = []
    for phase in canary_phases:
        phase_id = str(phase.get("id") or "")
        if not phase_id.startswith("canary_"):
            raise ControllerError("controller phases must preserve the canonical lifecycle order")
        try:
            percentage = int(phase_id.rsplit("_", 1)[1])
        except (TypeError, ValueError) as exc:
            raise ControllerError(f"invalid canary phase id: {phase_id!r}") from exc
        if phase_id != f"canary_{percentage:03d}" or not 1 <= percentage <= 100:
            raise ControllerError(f"invalid canary phase id: {phase_id!r}")
        percentages.append(percentage)
        canary_specs.append((phase_id, True, True))
    if percentages != sorted(set(percentages)):
        raise ControllerError("canary phases must be unique and strictly increasing")
    expected_specs = fixed_prefix + canary_specs + fixed_suffix
    previous = ""
    for phase, (phase_id, requires_approval, external_side_effect) in zip(phases, expected_specs):
        if phase.get("id") != phase_id:
            raise ControllerError("controller phases must preserve the canonical lifecycle order")
        if phase.get("depends_on") != ([previous] if previous else []):
            raise ControllerError(f"phase {phase_id!r} has unsafe or non-canonical dependencies")
        if phase.get("requires_approval") is not requires_approval:
            raise ControllerError(f"phase {phase_id!r} has an unsafe approval policy")
        if phase.get("external_side_effect") is not external_side_effect:
            raise ControllerError(f"phase {phase_id!r} has an unsafe side-effect policy")
        _validate_executable_phase(phase)
        if phase_id.startswith("canary_"):
            percentage = int(phase_id.rsplit("_", 1)[1])
            if phase.get("traffic_percentage") != percentage:
                raise ControllerError(f"phase {phase_id!r} has an invalid traffic percentage")
            if phase.get("guardrails") != plan.get("canary_guardrails"):
                raise ControllerError(f"phase {phase_id!r} is not bound to the plan guardrails")
        previous = phase_id

    failure_phases = plan.get("failure_phases")
    if not isinstance(failure_phases, dict) or set(failure_phases) != set(_AUXILIARY_PHASE_SPECS):
        raise ControllerError("controller plan must bind the complete canonical failure lifecycle")
    for phase_id, (requires_approval, external_side_effect) in _AUXILIARY_PHASE_SPECS.items():
        phase = failure_phases[phase_id]
        if not isinstance(phase, dict) or phase.get("id") != phase_id:
            raise ControllerError(f"invalid auxiliary phase {phase_id!r}")
        if phase.get("requires_approval") is not requires_approval:
            raise ControllerError(f"auxiliary phase {phase_id!r} has an unsafe approval policy")
        if phase.get("external_side_effect") is not external_side_effect:
            raise ControllerError(f"auxiliary phase {phase_id!r} has an unsafe side-effect policy")
        _validate_executable_phase(phase)

    boundary = plan.get("execution_boundary")
    required_boundary = {
        "opt_in",
        "core_side_effect_free",
        "external_actions_require_plan_bound_approval",
        "adapter_idempotency_required",
        "durable_reconciliation_required_after_inflight_crash",
        "fencing_tokens_required",
        "hard_deadline_required",
    }
    if not isinstance(boundary, dict) or any(boundary.get(key) is not True for key in required_boundary):
        raise ControllerError("controller plan has an unsafe execution boundary")


def _validate_executable_phase(phase: dict[str, Any]) -> None:
    phase_id = str(phase.get("id") or "")
    max_retries = phase.get("max_retries")
    if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0:
        raise ControllerError(f"phase {phase_id!r} max_retries must be a non-negative integer")
    if _number(phase.get("estimated_cost_usd")) < 0:
        raise ControllerError(f"phase {phase_id!r} estimated_cost_usd cannot be negative")
    if _number(phase.get("timeout_seconds")) <= 0:
        raise ControllerError(f"phase {phase_id!r} timeout_seconds must be positive")
    _require_external_reconciliation_contract(phase)


def _require_plan_binding(state: dict[str, Any], plan: dict[str, Any]) -> None:
    if state.get("schema_version") != CONTROLLER_STATE_SCHEMA_VERSION:
        raise ControllerError("controller state has an unsupported schema_version")
    if state.get("controller_id") != plan.get("controller_id") or state.get("plan_fingerprint") != plan.get("plan_fingerprint"):
        raise ControllerError("controller state is bound to a different immutable plan")


def _validate_state_replay(state: dict[str, Any], plan: dict[str, Any], state_path: Path) -> None:
    if state.get("state_path") != str(state_path.resolve()):
        raise ControllerError("controller state path binding does not match the loaded state file")
    records = state.get("phase_records")
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ControllerError("controller phase_records must be a list of objects")
    phase_by_id = {
        str(phase["id"]): phase
        for phase in [*plan["phases"], *plan["failure_phases"].values()]
    }
    seen: set[str] = set()
    completed_dependencies: list[dict[str, str]] = []
    total_attempts = 0
    spent_cost = 0.0
    elapsed = 0.0
    receipt_root = (Path(plan["artifact_dir"]) / str(plan["controller_id"]) / "receipts").resolve()
    for index, record in enumerate(records):
        phase_id = str(record.get("phase_id") or "")
        if phase_id not in phase_by_id or phase_id in seen:
            raise ControllerError(f"controller state has an unknown or duplicate phase record: {phase_id!r}")
        seen.add(phase_id)
        phase = phase_by_id[phase_id]
        attempts = record.get("attempt_count")
        if not isinstance(attempts, int) or isinstance(attempts, bool) or attempts < 0:
            raise ControllerError(f"phase record {phase_id!r} has an invalid attempt_count")
        total_attempts += attempts
        expected_input = _canonical_sha256(
            {
                "plan": plan["plan_fingerprint"],
                "phase": phase,
                "dependencies": completed_dependencies,
            }
        )
        if record.get("input_fingerprint") != expected_input:
            raise ControllerError(f"phase record {phase_id!r} input fingerprint does not replay")
        expected_key = f"hfr-{plan['controller_id']}-{phase_id}-{expected_input[:16]}"
        if record.get("idempotency_key") != expected_key:
            raise ControllerError(f"phase record {phase_id!r} idempotency key does not replay")
        if phase.get("requires_approval") is True:
            if record.get("approval_fingerprint") != plan["plan_fingerprint"]:
                raise ControllerError(f"phase record {phase_id!r} lacks plan-bound approval evidence")
        elif record.get("approval_fingerprint") not in {None, ""}:
            raise ControllerError(f"phase record {phase_id!r} has unexpected approval evidence")
        if phase in plan["phases"]:
            dependencies = phase.get("depends_on") if isinstance(phase.get("depends_on"), list) else []
            completed_ids = {row["phase_id"] for row in completed_dependencies}
            if any(dependency not in completed_ids for dependency in dependencies):
                raise ControllerError(f"phase record {phase_id!r} was created before its dependencies completed")

        receipt_path_value = record.get("receipt_path")
        receipt_sha = record.get("receipt_sha256")
        has_receipt = isinstance(receipt_path_value, str) and bool(receipt_path_value)
        if has_receipt:
            receipt_path = Path(receipt_path_value)
            expected_path = receipt_root / f"{phase_id}.json"
            if receipt_path.is_symlink() or receipt_path.resolve() != expected_path or not receipt_path.is_file():
                raise ControllerError(f"phase record {phase_id!r} receipt path is not the canonical regular file")
            if receipt_sha != _sha256_file(receipt_path):
                raise ControllerError(f"phase record {phase_id!r} receipt hash does not match")
            receipt = _read_json(receipt_path)
            expected_receipt_fields = {
                "schema_version": CONTROLLER_PHASE_RECEIPT_SCHEMA_VERSION,
                "controller_id": plan["controller_id"],
                "plan_fingerprint": plan["plan_fingerprint"],
                "phase_id": phase_id,
                "idempotency_key": expected_key,
                "input_fingerprint": expected_input,
                "attempt": attempts,
                "approval_fingerprint": str(record.get("approval_fingerprint") or ""),
            }
            if any(receipt.get(key) != value for key, value in expected_receipt_fields.items()):
                raise ControllerError(f"phase record {phase_id!r} receipt identity does not replay")
            if record.get("operation_status") != "receipted":
                raise ControllerError(f"phase record {phase_id!r} has a receipt without receipted operation status")
            receipt_passed = receipt.get("passed") is True
            allowed_statuses = {"completed", "guardrail_failed"} if receipt_passed else {"failed"}
            if record.get("status") not in allowed_statuses:
                raise ControllerError(f"phase record {phase_id!r} status conflicts with its receipt")
            for field in ("cost_usd", "duration_seconds", "metrics"):
                if record.get(field) != receipt.get(field):
                    raise ControllerError(f"phase record {phase_id!r} {field} conflicts with its receipt")
            spent_cost += _number(receipt.get("cost_usd"))
            elapsed += _number(receipt.get("duration_seconds"))
            if record.get("status") == "completed":
                completed_dependencies.append({"phase_id": phase_id, "receipt_sha256": str(receipt_sha)})
        else:
            if receipt_sha not in {None, ""}:
                raise ControllerError(f"phase record {phase_id!r} has a receipt hash without a receipt")
            if record.get("operation_status") == "receipted" or record.get("status") == "completed":
                raise ControllerError(f"phase record {phase_id!r} claims completion without a receipt")
            if record.get("status") in {"running", "reconciliation_required"} and record.get("operation_status") != "prepared":
                raise ControllerError(f"phase record {phase_id!r} has an invalid in-flight operation status")
    if total_attempts != state.get("total_attempts"):
        raise ControllerError("controller total_attempts does not replay from phase records")
    if total_attempts > plan["budget"]["max_attempts"]:
        raise ControllerError("controller state exceeds the immutable attempt budget")
    if round(spent_cost, 8) != round(_number(state.get("spent_cost_usd")), 8):
        raise ControllerError("controller spent_cost_usd does not replay from phase receipts")
    if round(elapsed, 6) != round(_number(state.get("elapsed_duration_seconds")), 6):
        raise ControllerError("controller elapsed_duration_seconds does not replay from phase receipts")
    if state.get("deadline_at") != plan.get("deadline_at"):
        raise ControllerError("controller state deadline does not match the immutable plan")
    if _optional_datetime(state.get("started_at")) is None:
        raise ControllerError("controller state started_at is invalid")
    if state.get("rollback_model") != plan["champion_model"]:
        raise ControllerError("controller rollback_model does not match the immutable champion")
    status = str(state.get("status") or "")
    expected_active_model = (
        plan["candidate_model"]
        if _phase_completed(state, "promote") and status != "rolled_back"
        else plan["champion_model"]
    )
    if state.get("active_model") != expected_active_model:
        raise ControllerError("controller active_model does not replay from completed phase receipts")
    if status == "complete":
        incomplete = [phase["id"] for phase in plan["phases"] if phase["id"] not in seen or not _phase_completed(state, phase["id"])]
        if incomplete or state.get("active_model") != plan["candidate_model"]:
            raise ControllerError(f"terminal complete state does not replay: incomplete={incomplete!r}")
    if status == "rolled_back":
        for required in ("rollback", "post_rollback_smoke"):
            if required not in seen or not _phase_completed(state, required):
                raise ControllerError(f"terminal rolled_back state is missing completed {required!r}")
        if state.get("active_model") != plan["champion_model"]:
            raise ControllerError("terminal rolled_back state does not restore the champion model")


def _phase_record(state: dict[str, Any], phase_id: str) -> dict[str, Any] | None:
    for row in state.get("phase_records", []):
        if isinstance(row, dict) and row.get("phase_id") == phase_id:
            return row
    return None


def _phase_completed(state: dict[str, Any], phase_id: str) -> bool:
    row = _phase_record(state, phase_id)
    return row is not None and row.get("status") == "completed"


def _read_json_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _read_json(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ControllerError(f"JSON artifact must be an object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _datetime(value: str | datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _optional_datetime(value: Any) -> datetime | None:
    try:
        return _datetime(str(value)) if value else None
    except (TypeError, ValueError):
        return None


def _number(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _positive_number(value: Any, *, default: float) -> float:
    parsed = _number(value)
    return parsed if parsed > 0 else default


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0
