"""Closed-loop agentic training iteration contracts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

AGENTIC_TRAINING_LOOP_PLAN_SCHEMA_VERSION = "hfr.agentic_training_loop_plan.v1"

PHASES: tuple[dict[str, Any], ...] = (
    {
        "id": "scenario_task_generation",
        "name": "Scenario and task generation",
        "required": (),
        "produces": ("scenario_quality", "harness_manifest"),
        "gate": "generated tasks must be versioned, replayable, and privacy reviewed before rollout.",
    },
    {
        "id": "rollout_collection",
        "name": "Rollout collection",
        "required": ("harness_result",),
        "produces": ("trace", "scorecard", "run_digest"),
        "gate": "rollouts must stay inside declared budget and environment descriptors.",
    },
    {
        "id": "evidence_scoring",
        "name": "Evidence scoring",
        "required": ("evidence_bundle",),
        "produces": ("evidence_coverage", "trace_observability", "repair_queue"),
        "gate": "evidence must be schema-checkable and traceable before training data selection.",
    },
    {
        "id": "rubric_model_grader_review",
        "name": "Rubric and model-grader review",
        "required": ("review_calibration",),
        "produces": ("review_manifest", "reviewed_manifest"),
        "gate": "model-grader labels are blocked until calibration and human override paths exist.",
    },
    {
        "id": "rejection_sampling",
        "name": "Rejection sampling",
        "required": ("reviewed_gate",),
        "produces": ("dataset_registry", "dataset_splits"),
        "gate": "uncalibrated or low-confidence labels must not enter trainer-ready datasets.",
    },
    {
        "id": "dataset_curation",
        "name": "Dataset curation",
        "required": ("training_export",),
        "produces": ("training_manifest", "dataset_registry"),
        "gate": "datasets must be redacted, licensed, hashed, and split before trainer handoff.",
    },
    {
        "id": "external_trainer_execution",
        "name": "External trainer execution",
        "required": ("agentic_training_plan", "trainer_preflight", "trainer_launch_check"),
        "produces": ("agentic_training_runtime_preflight", "agentic_training_result"),
        "gate": "live trainer launch requires explicit opt-in, credentials, and a passing preflight.",
    },
    {
        "id": "serving_checks",
        "name": "Serving checks",
        "required": ("serving_lifecycle",),
        "produces": ("serving_endpoint_check", "model_serving_probe_receipt"),
        "gate": "trained outputs must pass serving compatibility before held-out evals or promotion.",
    },
    {
        "id": "heldout_eval",
        "name": "Held-out eval and external benchmarks",
        "required": ("heldout_manifest", "external_eval_plan", "eval_summary"),
        "produces": ("eval_summary", "compare_gate", "decision_gate"),
        "gate": "live external benchmarks remain disabled until dependencies and held-out scenario parity pass.",
    },
    {
        "id": "improvement_planning",
        "name": "Improvement planning",
        "required": ("improvement_plan",),
        "produces": ("improvement_ledger", "action_ledger"),
        "gate": "next work must be evidence-backed, deduplicated, and linked to repair/curriculum signals.",
    },
    {
        "id": "governance_decision",
        "name": "Governance decision",
        "required": ("promotion_decision",),
        "produces": ("promotion_cards", "promotion_release_record"),
        "gate": "governance must approve, reject, rollback, or request another iteration before aliases move.",
    },
    {
        "id": "promotion_or_rollback",
        "name": "Promotion or rollback",
        "required": ("promotion_ledger",),
        "produces": ("promotion_alias_apply", "promotion_rollback_receipt"),
        "gate": "registry writes are guarded by promotion decisions and rollback receipts.",
    },
    {
        "id": "next_iteration",
        "name": "Next-iteration scheduling",
        "required": ("action_ledger",),
        "produces": ("agentic_training_loop_plan",),
        "gate": "new iterations are scheduled only from ledgered decisions and open repair actions.",
    },
)

ARTIFACT_ROLES: dict[str, str] = {
    "action_ledger": "action_ledger",
    "agentic_training_plan": "agentic_training_plan",
    "agentic_training_result": "agentic_training_result",
    "agentic_training_runtime_preflight": "agentic_training_runtime_preflight",
    "compare_gate": "compare_gate",
    "dataset_registry": "dataset_registry",
    "dataset_splits": "dataset_splits",
    "decision_gate": "decision_gate",
    "evidence_bundle": "evidence_bundle",
    "evidence_coverage": "evidence_coverage",
    "eval_summary": "eval_summary",
    "external_eval_plan": "external_eval_plan",
    "harness_manifest": "harness_manifest",
    "harness_result": "harness_result",
    "heldout_manifest": "heldout_manifest",
    "improvement_ledger": "improvement_ledger",
    "improvement_plan": "improvement_plan",
    "promotion_decision": "promotion_decision",
    "promotion_ledger": "promotion_ledger",
    "review_calibration": "review_calibration",
    "reviewed_gate": "reviewed_gate",
    "serving_lifecycle": "serving_lifecycle",
    "trace_observability": "trace_observability",
    "trainer_launch_check": "trainer_launch_check",
    "trainer_preflight": "trainer_preflight",
    "training_export": "training_export",
}


class AgenticTrainingLoopPlanError(ValueError):
    """Raised when a closed-loop iteration plan cannot be built."""


def build_agentic_training_loop_plan(
    *,
    out_path: str | Path,
    iteration_id: str,
    artifact_paths: dict[str, list[str | Path]] | None = None,
    objective: str | None = None,
    candidate: str | None = None,
    baseline: str | None = None,
    teacher: str | None = None,
    budget: dict[str, Any] | None = None,
    provider_constraints: dict[str, Any] | None = None,
    schedule: dict[str, Any] | None = None,
    preserve_paths: bool = False,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a side-effect-free contract for one closed-loop training iteration."""
    if not iteration_id:
        raise AgenticTrainingLoopPlanError("iteration_id is required")
    refs = _artifact_refs(artifact_paths or {}, preserve_paths)
    phases = [_phase_row(spec, refs) for spec in PHASES]
    checks: list[dict[str, Any]] = []
    _add_check(checks, "phase_contracts_present", len(phases) == len(PHASES), {"phase_count": len(phases)}, {"phase_count": len(PHASES)})
    _add_check(checks, "artifact_references_are_public_safe", _refs_are_safe(refs), {"unsafe_refs": _unsafe_refs(refs)}, {"unsafe_refs": []})
    _add_check(
        checks,
        "flight_recorder_did_not_launch_external_work",
        True,
        {
            "cloud_jobs_started": False,
            "paid_model_grader_calls_started": False,
            "live_benchmarks_started": False,
            "model_downloads_started": False,
            "weights_updated": False,
        },
        {
            "cloud_jobs_started": False,
            "paid_model_grader_calls_started": False,
            "live_benchmarks_started": False,
            "model_downloads_started": False,
            "weights_updated": False,
        },
    )
    _add_check(
        checks,
        "live_launches_require_explicit_opt_in",
        True,
        {"live_launch_opt_in": False, "environment_credentials_checked": False},
        {"live_launch_opt_in": True, "environment_credentials_checked": True},
    )
    _add_check(
        checks,
        "uncalibrated_labels_block_training_data",
        "review_calibration" in refs and "reviewed_gate" in refs,
        {"review_calibration_present": "review_calibration" in refs, "reviewed_gate_present": "reviewed_gate" in refs},
        {"review_calibration_present": True, "reviewed_gate_present": True},
    )
    _add_check(
        checks,
        "external_trainer_handoff_is_preflighted",
        "agentic_training_plan" in refs and "trainer_preflight" in refs and "trainer_launch_check" in refs,
        {
            "agentic_training_plan_present": "agentic_training_plan" in refs,
            "trainer_preflight_present": "trainer_preflight" in refs,
            "trainer_launch_check_present": "trainer_launch_check" in refs,
        },
        {
            "agentic_training_plan_present": True,
            "trainer_preflight_present": True,
            "trainer_launch_check_present": True,
        },
    )
    _add_check(
        checks,
        "heldout_eval_is_fail_closed",
        "heldout_manifest" in refs and "external_eval_plan" in refs,
        {"heldout_manifest_present": "heldout_manifest" in refs, "external_eval_plan_present": "external_eval_plan" in refs},
        {"heldout_manifest_present": True, "external_eval_plan_present": True},
    )
    _add_check(
        checks,
        "governance_required_for_promotion",
        "promotion_decision" in refs and "promotion_ledger" in refs,
        {"promotion_decision_present": "promotion_decision" in refs, "promotion_ledger_present": "promotion_ledger" in refs},
        {"promotion_decision_present": True, "promotion_ledger_present": True},
    )

    failed_checks = [check for check in checks if not check["passed"]]
    missing_phase_inputs = sorted({missing for phase in phases for missing in phase["missing_required_artifacts"]})
    readiness = "ready_for_governance_review" if not failed_checks and not missing_phase_inputs else "planned_fail_closed"
    recommendation = "approve_iteration_execution" if readiness == "ready_for_governance_review" else "collect_missing_receipts_before_live_execution"

    return {
        "schema_version": AGENTIC_TRAINING_LOOP_PLAN_SCHEMA_VERSION,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "iteration_id": iteration_id,
        "plan_path": _display_path(Path(out_path), preserve_paths),
        "objective": objective or "",
        "participants": {
            "baseline_policy": baseline or "",
            "candidate_policy": candidate or "",
            "teacher_policy": teacher or "",
        },
        "passed": not failed_checks,
        "readiness": readiness,
        "recommendation": recommendation,
        "check_count": len(checks),
        "failed_check_count": len(failed_checks),
        "checks": checks,
        "blocked_reasons": [check["summary"] for check in failed_checks],
        "missing_phase_inputs": missing_phase_inputs,
        "artifact_count": sum(len(rows) for rows in refs.values()),
        "artifact_role_counts": _role_counts(refs),
        "source_artifacts": refs,
        "phases": phases,
        "budget": _budget(budget or {}),
        "provider_constraints": _provider_constraints(provider_constraints or {}),
        "execution_boundary": {
            "dry_run_plan_only": True,
            "cloud_jobs_started": False,
            "paid_model_grader_calls_started": False,
            "live_benchmarks_started": False,
            "model_downloads_started": False,
            "weights_updated_by_flight_recorder": False,
            "credential_values_recorded": False,
            "public_artifact_paths_redacted": not preserve_paths,
        },
        "handoff_contract": {
            "flight_recorder_controls_preflight_and_receipts": True,
            "external_trainers_own_weight_updates": True,
            "live_launch_requires_explicit_opt_in": True,
            "requires_environment_credentials_for_live": True,
            "requires_trainer_preflight": True,
            "requires_trainer_launch_check": True,
            "requires_calibrated_review_before_training_data": True,
            "requires_heldout_eval_before_promotion": True,
            "requires_governance_decision_before_alias_update": True,
            "default_live_execution_allowed": False,
        },
        "next_iteration": {
            "scheduled": False,
            "requires_governance_decision": True,
            "schedule": schedule or {},
            "recommendation": (
                "Use promotion/action/improvement ledgers to schedule the next iteration after governance decides."
            ),
        },
        "notes": [
            "This artifact is a closed-loop iteration contract; it does not call graders, trainers, cloud APIs, or benchmarks.",
            "Missing phase inputs keep the loop fail-closed while preserving a schema-checkable plan for orchestration.",
            "Use dry-run/mock provider receipts first; live provider launches must be explicit, credentialed, and separately archived.",
        ],
    }


def write_agentic_training_loop_plan(path: str | Path, plan: dict[str, Any]) -> None:
    """Write a deterministic closed-loop plan artifact."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_refs(artifact_paths: dict[str, list[str | Path]], preserve_paths: bool) -> dict[str, list[dict[str, Any]]]:
    refs: dict[str, list[dict[str, Any]]] = {}
    for role in sorted(artifact_paths):
        normalized_role = ARTIFACT_ROLES.get(role, role)
        rows = [_artifact_ref(normalized_role, Path(path), preserve_paths) for path in artifact_paths[role] if str(path)]
        if rows:
            refs[normalized_role] = rows
    return refs


def _artifact_ref(role: str, path: Path, preserve_paths: bool) -> dict[str, Any]:
    exists = path.exists()
    is_file = path.is_file()
    is_dir = path.is_dir()
    payload = _read_json(path) if is_file and path.suffix == ".json" else {}
    return {
        "role": role,
        "path": _display_path(path, preserve_paths),
        "kind": "directory" if is_dir else "file",
        "exists": exists,
        "sha256": _sha256(path) if is_file else None,
        "size_bytes": path.stat().st_size if is_file else None,
        "schema_version": str(payload.get("schema_version") or "") if payload else "",
        "passed": payload.get("passed") if isinstance(payload.get("passed"), bool) else None,
        "readiness": str(payload.get("readiness") or payload.get("status") or ""),
    }


def _phase_row(spec: dict[str, Any], refs: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    required = list(spec["required"])
    present = [role for role in required if role in refs and refs[role]]
    missing = [role for role in required if role not in present]
    status = "ready" if required and not missing else "planned"
    if missing and required:
        status = "blocked"
    return {
        "id": spec["id"],
        "name": spec["name"],
        "status": status,
        "required_artifacts": required,
        "present_required_artifacts": present,
        "missing_required_artifacts": missing,
        "produces": list(spec["produces"]),
        "gate": spec["gate"],
    }


def _budget(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "max_rollouts": _non_negative_or_none(value.get("max_rollouts")),
        "max_training_examples": _non_negative_or_none(value.get("max_training_examples")),
        "max_cloud_cost_usd": _non_negative_number_or_none(value.get("max_cloud_cost_usd")),
        "max_gpu_hours": _non_negative_number_or_none(value.get("max_gpu_hours")),
        "live_spend_allowed": False,
    }


def _provider_constraints(value: dict[str, Any]) -> dict[str, Any]:
    regions = value.get("regions") if isinstance(value.get("regions"), list) else []
    gpus = value.get("gpu_classes") if isinstance(value.get("gpu_classes"), list) else []
    providers = value.get("providers") if isinstance(value.get("providers"), list) else []
    return {
        "providers": [str(item) for item in providers if str(item)],
        "regions": [str(item) for item in regions if str(item)],
        "gpu_classes": [str(item) for item in gpus if str(item)],
        "requires_cost_estimate": True,
        "requires_region_allowlist": bool(regions),
        "requires_secret_redaction": True,
    }


def _role_counts(refs: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    counter = Counter({role: len(rows) for role, rows in refs.items()})
    return [{"role": role, "count": counter[role]} for role in sorted(counter)]


def _refs_are_safe(refs: dict[str, list[dict[str, Any]]]) -> bool:
    return not _unsafe_refs(refs)


def _unsafe_refs(refs: dict[str, list[dict[str, Any]]]) -> list[str]:
    unsafe: list[str] = []
    for rows in refs.values():
        for row in rows:
            path = str(row.get("path") or "")
            if not path or Path(path).is_absolute() or ".." in Path(path).parts:
                unsafe.append(path)
    return unsafe


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


def _non_negative_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _non_negative_number_or_none(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and value >= 0:
        return value
    return None


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
