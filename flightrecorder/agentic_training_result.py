"""Result receipts for external agentic training runs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agentic_training_plan import AGENTIC_TRAINING_PLAN_SCHEMA_VERSION
from .agentic_training_runtime import (
    AGENTIC_TRAINING_RUNTIME_PREFLIGHT_SCHEMA_VERSION,
    PLAN_READY_RECOMMENDATION,
    RUNTIME_READY_RECOMMENDATION,
)
from .schema_registry import SchemaRegistryError, check_schema_file

AGENTIC_TRAINING_RESULT_SCHEMA_VERSION = "hfr.agentic_training_result.v1"

RESULT_STATUSES = ("completed", "failed", "blocked", "aborted")
FAILURE_CLASSES = (
    "none",
    "runtime_preflight_blocked",
    "dependency_missing",
    "view_validation_failed",
    "trainer_crash",
    "out_of_memory",
    "timeout",
    "interrupted",
    "license_or_redaction_block",
    "artifact_missing",
    "unknown",
)
REGISTER_RESULT_RECOMMENDATION = "register_training_result"
REGISTER_FAILURE_RECOMMENDATION = "register_training_failure"
BLOCK_REGISTRATION_RECOMMENDATION = "block_training_result_registration"

OUTPUT_ARTIFACT_ROLES = {"adapter", "checkpoint"}
RECOVERABLE_FAILURE_CLASSES = {
    "runtime_preflight_blocked",
    "dependency_missing",
    "view_validation_failed",
    "trainer_crash",
    "out_of_memory",
    "timeout",
    "interrupted",
    "artifact_missing",
}


class AgenticTrainingResultError(ValueError):
    """Raised when an agentic training result receipt cannot be built."""


def build_agentic_training_result(
    *,
    plan_path: str | Path,
    runtime_preflight_path: str | Path,
    out_path: str | Path | None = None,
    status: str,
    failure_class: str = "none",
    failure_message: str = "",
    runner_id: str = "external",
    run_id: str = "",
    output_dir: str | Path | None = None,
    artifacts: dict[str, Iterable[str | Path]] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a schema-checkable receipt for an externally executed training run.

    The receipt is intentionally side-effect free: it reads and fingerprints
    supplied artifacts but does not launch trainers, import trainer stacks, or
    mutate model weights.
    """
    normalized_status = status.strip().lower()
    normalized_failure_class = failure_class.strip().lower() or "none"
    if normalized_status not in RESULT_STATUSES:
        raise AgenticTrainingResultError(f"unsupported training status {status!r}; expected one of {', '.join(RESULT_STATUSES)}")
    if normalized_failure_class not in FAILURE_CLASSES:
        raise AgenticTrainingResultError(
            f"unsupported failure class {failure_class!r}; expected one of {', '.join(FAILURE_CLASSES)}"
        )

    plan_file = Path(plan_path)
    runtime_file = Path(runtime_preflight_path)
    plan_payload, plan_read_errors = _read_json_object(plan_file)
    runtime_payload, runtime_read_errors = _read_json_object(runtime_file)
    plan_schema_check = _json_schema_record(plan_file, "agentic_training_plan")
    runtime_schema_check = _json_schema_record(runtime_file, "agentic_training_runtime_preflight")
    artifact_refs = _artifact_refs(artifacts or {})

    plan_sha = _sha256_or_none(plan_file)
    runtime_plan_sha = runtime_payload.get("plan_sha256") if isinstance(runtime_payload.get("plan_sha256"), str) else ""
    plan_ready = (
        not plan_read_errors
        and plan_schema_check["passed"]
        and plan_payload.get("schema_version") == AGENTIC_TRAINING_PLAN_SCHEMA_VERSION
        and plan_payload.get("passed") is True
        and plan_payload.get("recommendation") == PLAN_READY_RECOMMENDATION
    )
    runtime_schema_valid = (
        not runtime_read_errors
        and runtime_schema_check["passed"]
        and runtime_payload.get("schema_version") == AGENTIC_TRAINING_RUNTIME_PREFLIGHT_SCHEMA_VERSION
    )
    runtime_ready = (
        runtime_schema_valid
        and runtime_payload.get("passed") is True
        and runtime_payload.get("recommendation") == RUNTIME_READY_RECOMMENDATION
    )
    runtime_matches_plan = bool(plan_sha) and runtime_plan_sha == plan_sha
    output_artifacts = [artifact for artifact in artifact_refs if artifact["role"] in OUTPUT_ARTIFACT_ROLES]
    artifacts_regular = all(artifact["regular_file"] for artifact in artifact_refs)
    classified_failure = (
        normalized_status == "completed"
        or (normalized_failure_class not in {"none", "unknown"} and bool(failure_message.strip()))
    )
    completed_has_no_failure = normalized_status != "completed" or normalized_failure_class == "none"
    completed_has_output = normalized_status != "completed" or bool(output_artifacts)
    runtime_ready_for_completed = normalized_status != "completed" or runtime_ready

    checks: list[dict[str, Any]] = []
    _add_check(checks, "status_supported", normalized_status in RESULT_STATUSES, {"status": normalized_status}, {"statuses": list(RESULT_STATUSES)})
    _add_check(checks, "plan_json_readable", not plan_read_errors, {"errors": plan_read_errors}, {"errors": []})
    _add_check(
        checks,
        "plan_schema_passed",
        plan_schema_check["passed"],
        {"error_count": plan_schema_check["error_count"], "errors": plan_schema_check["errors"]},
        {"schema_name": "agentic_training_plan", "error_count": 0},
    )
    _add_check(
        checks,
        "plan_recommendation_ready",
        plan_ready,
        {"passed": plan_payload.get("passed"), "recommendation": plan_payload.get("recommendation")},
        {"passed": True, "recommendation": PLAN_READY_RECOMMENDATION},
    )
    _add_check(checks, "runtime_preflight_json_readable", not runtime_read_errors, {"errors": runtime_read_errors}, {"errors": []})
    _add_check(
        checks,
        "runtime_preflight_schema_passed",
        runtime_schema_check["passed"],
        {"error_count": runtime_schema_check["error_count"], "errors": runtime_schema_check["errors"]},
        {"schema_name": "agentic_training_runtime_preflight", "error_count": 0},
    )
    _add_check(
        checks,
        "runtime_preflight_matches_plan",
        runtime_matches_plan,
        {"runtime_plan_sha256": runtime_plan_sha, "plan_sha256": plan_sha or ""},
        {"runtime_plan_sha256": plan_sha or ""},
    )
    _add_check(
        checks,
        "runtime_ready_for_completed_result",
        runtime_ready_for_completed,
        {"status": normalized_status, "runtime_recommendation": runtime_payload.get("recommendation")},
        {"completed_requires": RUNTIME_READY_RECOMMENDATION},
    )
    _add_check(
        checks,
        "non_completed_failure_classified",
        classified_failure,
        {"status": normalized_status, "failure_class": normalized_failure_class, "failure_message_present": bool(failure_message.strip())},
        {"non_completed_requires_classified_failure": True},
    )
    _add_check(
        checks,
        "completed_result_has_no_failure_class",
        completed_has_no_failure,
        {"status": normalized_status, "failure_class": normalized_failure_class},
        {"completed_failure_class": "none"},
    )
    _add_check(
        checks,
        "completed_result_has_output_artifact",
        completed_has_output,
        {"status": normalized_status, "output_artifact_count": len(output_artifacts)},
        {"output_artifact_roles": sorted(OUTPUT_ARTIFACT_ROLES)},
    )
    _add_check(
        checks,
        "artifact_refs_are_regular_files",
        artifacts_regular,
        {"artifact_count": len(artifact_refs), "missing_or_irregular": [artifact["path"] for artifact in artifact_refs if not artifact["regular_file"]]},
        {"all_artifact_refs_regular": True},
    )
    _add_check(
        checks,
        "flight_recorder_did_not_launch_training",
        True,
        {"training_started_by_flight_recorder": False},
        {"training_started_by_flight_recorder": False},
    )

    failed_checks = [check for check in checks if check["passed"] is False]
    passed = not failed_checks
    if passed and normalized_status == "completed":
        recommendation = REGISTER_RESULT_RECOMMENDATION
    elif passed:
        recommendation = REGISTER_FAILURE_RECOMMENDATION
    else:
        recommendation = BLOCK_REGISTRATION_RECOMMENDATION

    trainer_plan = plan_payload.get("trainer_plan") if isinstance(plan_payload.get("trainer_plan"), dict) else {}
    result = {
        "schema_version": AGENTIC_TRAINING_RESULT_SCHEMA_VERSION,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "artifact_path": str(out_path or ""),
        "passed": passed,
        "readiness": "ready" if passed else "blocked",
        "recommendation": recommendation,
        "check_count": len(checks),
        "failed_check_count": len(failed_checks),
        "checks": checks,
        "blocked_reasons": [check["summary"] for check in failed_checks],
        "training_result": {
            "status": normalized_status,
            "runner_id": runner_id,
            "run_id": run_id,
            "mode": str(plan_payload.get("mode") or ""),
            "backend": str(trainer_plan.get("backend") or runtime_payload.get("backend") or ""),
            "output_dir": str(output_dir if output_dir is not None else trainer_plan.get("output_dir") or ""),
            "external_runner_reported_status": normalized_status,
            "flight_recorder_executed_training": False,
            "model_downloads_started_by_flight_recorder": False,
        },
        "lineage": {
            "plan": _lineage_ref(plan_file, plan_sha, "agentic_training_plan"),
            "runtime_preflight": _lineage_ref(runtime_file, _sha256_or_none(runtime_file), "agentic_training_runtime_preflight"),
            "model": _manifest_summary(plan_payload, "model"),
            "dataset": _manifest_summary(plan_payload, "dataset"),
        },
        "failure": {
            "class": normalized_failure_class,
            "message": failure_message,
            "recoverable": normalized_failure_class in RECOVERABLE_FAILURE_CLASSES,
            "source": _failure_source(normalized_status, normalized_failure_class, runtime_payload),
        },
        "artifacts": artifact_refs,
        "metrics": {
            "artifact_count": len(artifact_refs),
            "regular_artifact_count": sum(1 for artifact in artifact_refs if artifact["regular_file"]),
            "output_artifact_count": len(output_artifacts),
            "config_count": _role_count(artifact_refs, "config"),
            "metrics_file_count": _role_count(artifact_refs, "metrics"),
            "adapter_count": _role_count(artifact_refs, "adapter"),
            "checkpoint_count": _role_count(artifact_refs, "checkpoint"),
            "log_count": _role_count(artifact_refs, "log"),
            "failure_report_count": _role_count(artifact_refs, "failure_report"),
        },
        "execution_boundary": {
            "archive_only": True,
            "runner_owns_execution": True,
            "flight_recorder_launched_training": False,
            "training_started_by_flight_recorder": False,
            "model_downloads_started_by_flight_recorder": False,
            "trainer_modules_imported_by_flight_recorder": False,
        },
        "handoff_contract": {
            "runner_owns_execution": True,
            "requires_agentic_training_plan": True,
            "requires_runtime_preflight": True,
            "requires_runtime_ready_for_completed": True,
            "requires_classified_failure_for_non_completed": True,
            "requires_output_artifact_for_completed": True,
            "requires_registered_model": True,
            "requires_registered_dataset": True,
            "requires_redacted_dataset": True,
            "flight_recorder_launched_training": False,
            "model_downloads_started_by_flight_recorder": False,
        },
        "notes": [
            "This receipt archives externally reported training status and supplied artifact hashes.",
            "Flight Recorder did not execute a trainer, import trainer stacks, download models, or mutate weights.",
            "Completed results must reference at least one adapter or checkpoint artifact; non-completed results must include a classified failure.",
        ],
    }
    return result


def write_agentic_training_result(path: str | Path, result: dict[str, Any]) -> None:
    """Write a deterministic JSON training result receipt."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _json_schema_record(path: Path, schema_name: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "schema_name": schema_name,
        "passed": False,
        "error_count": 0,
        "errors": [],
    }
    try:
        result = check_schema_file(path, schema_name)
    except (OSError, json.JSONDecodeError, SchemaRegistryError) as exc:
        record["errors"] = [str(exc)]
        record["error_count"] = 1
        return record
    record["passed"] = result.get("passed") is True
    record["error_count"] = _int_value(result.get("error_count"))
    record["errors"] = [str(error) for error in result.get("errors", []) if isinstance(error, str)][:20]
    return record


def _artifact_refs(artifacts: dict[str, Iterable[str | Path]]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for role in sorted(artifacts):
        for raw_path in artifacts[role]:
            path = Path(raw_path)
            refs.append(
                {
                    "role": role,
                    "path": str(path),
                    "exists": path.exists(),
                    "regular_file": path.is_file(),
                    "sha256": _sha256_or_none(path),
                    "size_bytes": path.stat().st_size if path.is_file() else 0,
                }
            )
    return refs


def _lineage_ref(path: Path, sha256: str | None, schema_name: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "regular_file": path.is_file(),
        "schema_name": schema_name,
        "sha256": sha256,
    }


def _manifest_summary(plan: dict[str, Any], key: str) -> dict[str, Any]:
    manifests = plan.get("input_manifests") if isinstance(plan.get("input_manifests"), dict) else {}
    record = manifests.get(key) if isinstance(manifests.get(key), dict) else {}
    return {
        "id": str(record.get("id") or ""),
        "path": str(record.get("path") or ""),
        "sha256": str(record.get("sha256") or ""),
        "license_allows_training": record.get("license_allows_training") is True,
    }


def _failure_source(status: str, failure_class: str, runtime_preflight: dict[str, Any]) -> str:
    if status == "completed":
        return "none"
    if failure_class == "runtime_preflight_blocked":
        return "runtime_preflight"
    if runtime_preflight.get("passed") is False:
        return "runtime_preflight"
    return "external_runner"


def _role_count(artifacts: list[dict[str, Any]], role: str) -> int:
    return sum(1 for artifact in artifacts if artifact.get("role") == role)


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


def _int_value(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _sha256_or_none(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
