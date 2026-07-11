"""Import-only receipts for externally executed cloud training jobs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

from .atomic_json import atomic_write_json_cas
from .path_safety import path_has_symlink_component
from .schema_registry import SchemaRegistryError, check_schema_contract
from .source_contract import (
    MAX_OPAQUE_TRAINING_OUTPUT_BYTES,
    MAX_OPAQUE_TRAINING_OUTPUT_FILES,
    MAX_OPAQUE_TRAINING_OUTPUT_TOTAL_BYTES,
    get_active_opaque_output_attestation,
)

CLOUD_TRAINING_COMPLETION_RECEIPT_SCHEMA_VERSION = (
    "hfr.cloud_training_completion_receipt.v1"
)
EXTERNAL_CLOUD_TRAINING_RUNNER_SCHEMA_VERSION = (
    "hfr.external_cloud_training_runner.v1"
)

EXECUTION_STATUSES = ("completed", "failed", "incomplete", "unknown")
FAILURE_CLASSES = (
    "none",
    "provider",
    "runner",
    "capacity",
    "timeout",
    "cancelled",
    "artifact",
    "unknown",
)
OUTPUT_ARTIFACT_ROLES = {"adapter", "checkpoint"}
MAX_CONTROL_SOURCE_BYTES = 4 * 1024 * 1024
MAX_RAW_PROVIDER_RESULT_BYTES = 64 * 1024 * 1024
MAX_JSON_DEPTH = 100
MAX_LINKED_CONTROL_SOURCES = 256

_METADATA_KEYS = {
    "schema_version",
    "provider_id",
    "provider_job_id",
    "execution_id",
    "candidate_model_id",
    "status",
    "terminal",
    "failure",
    "runner",
    "started_at",
    "finished_at",
    "exit_code",
    "provider_constraints",
    "source_sha256",
    "side_effects",
}
_FAILURE_KEYS = {"class", "message"}
_RUNNER_KEYS = {"id", "version"}
_CONSTRAINT_KEYS = {"region", "gpu_class", "reported_cost_usd"}
_SOURCE_HASH_KEYS = {
    "launch_plan",
    "launch_receipt",
    "status_receipt",
    "raw_provider_result",
    "output_artifact_manifest",
}
_SIDE_EFFECT_KEYS = {
    "external_provider_api_called",
    "external_cloud_job_started",
    "external_artifacts_uploaded",
    "external_artifacts_downloaded",
    "credential_values_recorded",
    "provider_api_called_by_flight_recorder",
    "cloud_job_started_by_flight_recorder",
    "provider_status_polled_by_flight_recorder",
    "artifacts_uploaded_by_flight_recorder",
    "artifacts_downloaded_by_flight_recorder",
    "model_downloads_started_by_flight_recorder",
    "weights_updated_by_flight_recorder",
    "provider_modules_imported_by_flight_recorder",
}
_HFR_SIDE_EFFECT_KEYS = {
    "provider_api_called_by_flight_recorder",
    "cloud_job_started_by_flight_recorder",
    "provider_status_polled_by_flight_recorder",
    "artifacts_uploaded_by_flight_recorder",
    "artifacts_downloaded_by_flight_recorder",
    "model_downloads_started_by_flight_recorder",
    "weights_updated_by_flight_recorder",
    "provider_modules_imported_by_flight_recorder",
}
_SECRET_LIKE = re.compile(
    r"(?i)(bearer\s+[a-z0-9._-]+|sk-[a-z0-9]{8,}|password\s*[:=]|api[_-]?key\s*[:=]|token\s*[:=])"
)


class CloudTrainingCompletionError(ValueError):
    """Raised when cloud completion evidence cannot be imported safely."""


class _BuiltCloudTrainingCompletionReceipt(dict[str, Any]):
    def __init__(
        self,
        payload: dict[str, Any],
        intended_output_path: Path | None,
    ) -> None:
        super().__init__(payload)
        self._intended_output_path = intended_output_path


class _DuplicateJsonKeyError(ValueError):
    pass


@dataclass(frozen=True)
class _SourceSnapshot:
    path: Path
    display_path: str
    exists: bool
    regular_file: bool
    replayable: bool
    sha256: str | None
    size_bytes: int | None
    data: bytes | None
    identity: tuple[int, int] | None
    errors: tuple[str, ...]


@dataclass(frozen=True)
class _StableFileFingerprint:
    identity: tuple[int, int]
    sha256: str
    size_bytes: int


def build_cloud_training_completion_receipt(
    *,
    launch_plan_path: str | Path,
    launch_receipt_path: str | Path,
    status_receipt_path: str | Path,
    runner_metadata_path: str | Path,
    raw_provider_result_path: str | Path,
    output_artifact_manifest_path: str | Path,
    out_path: str | Path | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Bind external cloud-run evidence without importing or calling a provider."""
    source_paths = {
        "launch_plan": Path(launch_plan_path),
        "launch_receipt": Path(launch_receipt_path),
        "status_receipt": Path(status_receipt_path),
        "runner_metadata": Path(runner_metadata_path),
        "raw_provider_result": Path(raw_provider_result_path),
        "output_artifact_manifest": Path(output_artifact_manifest_path),
    }
    output_path = Path(out_path) if out_path is not None else None
    if output_path is not None:
        _reject_output_source_collision(output_path, list(source_paths.values()))
        if path_has_symlink_component(output_path, include_leaf=True):
            raise CloudTrainingCompletionError(
                "cloud completion output path must not contain symlink components"
            )
        if _public_basename(output_path) != output_path.name:
            raise CloudTrainingCompletionError(
                "cloud completion output basename must be public-safe"
            )
    base_dir = output_path.parent if output_path is not None else Path.cwd()

    snapshots = {
        name: _capture_source(
            path,
            base_dir,
            name.replace("_", " "),
            MAX_RAW_PROVIDER_RESULT_BYTES
            if name == "raw_provider_result"
            else MAX_CONTROL_SOURCE_BYTES,
        )
        for name, path in source_paths.items()
    }
    payloads: dict[str, dict[str, Any]] = {}
    parse_errors: dict[str, list[str]] = {}
    for name in (
        "launch_plan",
        "launch_receipt",
        "status_receipt",
        "runner_metadata",
        "output_artifact_manifest",
    ):
        payloads[name], parse_errors[name] = _parse_json_object(
            snapshots[name], name.replace("_", " ")
        )

    launch_plan = payloads["launch_plan"]
    launch_receipt = payloads["launch_receipt"]
    status_receipt = payloads["status_receipt"]
    metadata = payloads["runner_metadata"]
    output_manifest = payloads["output_artifact_manifest"]

    metadata_errors = _runner_metadata_errors(metadata)
    status = _string(metadata.get("status")).lower()
    failure = metadata.get("failure") if isinstance(metadata.get("failure"), dict) else {}
    runner = metadata.get("runner") if isinstance(metadata.get("runner"), dict) else {}
    runner_constraints = (
        metadata.get("provider_constraints")
        if isinstance(metadata.get("provider_constraints"), dict)
        else {}
    )
    side_effects = (
        metadata.get("side_effects")
        if isinstance(metadata.get("side_effects"), dict)
        else {}
    )
    source_hashes = (
        metadata.get("source_sha256")
        if isinstance(metadata.get("source_sha256"), dict)
        else {}
    )

    upstream_schema_errors = {
        "launch_plan": _schema_errors(launch_plan, "cloud_training_launch_plan"),
        "launch_receipt": _schema_errors(
            launch_receipt, "cloud_training_launch_receipt"
        ),
        "status_receipt": _schema_errors(
            status_receipt, "cloud_training_status_receipt"
        ),
        "output_artifact_manifest": _schema_errors(
            output_manifest, "agentic_training_result"
        ),
    }
    invalid_direct_sources = [
        name for name, snapshot in snapshots.items() if not _source_ready(snapshot)
    ]
    invalid_json_sources = [
        name for name, errors in parse_errors.items() if errors
    ]
    invalid_schema_sources = [
        name for name, errors in upstream_schema_errors.items() if errors
    ]
    if invalid_direct_sources:
        raise CloudTrainingCompletionError(
            "cloud completion direct sources must be stable, bounded, and replayable: "
            + ", ".join(sorted(invalid_direct_sources))
        )
    if invalid_json_sources:
        raise CloudTrainingCompletionError(
            "cloud completion control sources must be duplicate-free finite JSON objects: "
            + ", ".join(sorted(invalid_json_sources))
        )
    if metadata_errors:
        raise CloudTrainingCompletionError(
            "cloud completion runner metadata is invalid: "
            + "; ".join(metadata_errors)
        )
    if invalid_schema_sources:
        raise CloudTrainingCompletionError(
            "cloud completion upstream sources must satisfy their public schemas: "
            + ", ".join(sorted(invalid_schema_sources))
        )
    created_at_value = created_at or datetime.now(timezone.utc).isoformat()
    parsed_created_at = _parsed_timestamp(created_at_value)
    parsed_finished_at = _parsed_timestamp(metadata.get("finished_at"))
    if parsed_created_at is None:
        raise CloudTrainingCompletionError(
            "cloud completion created_at must be a timezone-aware ISO-8601 timestamp"
        )
    if parsed_finished_at is not None and parsed_created_at < parsed_finished_at:
        raise CloudTrainingCompletionError(
            "cloud completion created_at must not precede runner finished_at"
        )
    upstream_semantic_errors = _upstream_semantic_errors(source_paths)

    launch_plan_sha = snapshots["launch_plan"].sha256
    launch_receipt_sha = snapshots["launch_receipt"].sha256
    status_receipt_sha = snapshots["status_receipt"].sha256
    raw_result_sha = snapshots["raw_provider_result"].sha256
    output_manifest_sha = snapshots["output_artifact_manifest"].sha256
    runner_metadata_sha = snapshots["runner_metadata"].sha256

    launch_plan_link = _source_link_sha(launch_receipt, "launch_plan")
    launch_receipt_link = _source_link_sha(status_receipt, "launch_receipt")
    provider_id = _string(metadata.get("provider_id"))
    launch_provider_id = _provider_id(launch_plan)

    preflight_path, preflight, preflight_errors = _linked_json_source(
        launch_plan,
        source_paths["launch_plan"],
        "preflight",
        "cloud launch-plan preflight",
    )
    preflight_constraints = (
        preflight.get("constraints")
        if isinstance(preflight.get("constraints"), dict)
        else {}
    )
    preflight_plan_sha = _source_link_sha(preflight, "agentic_training_plan")
    result_plan_sha = _lineage_sha(output_manifest, "plan")
    candidate_model_id = _string(metadata.get("candidate_model_id"))
    result_candidate_id = _training_result_candidate(output_manifest)

    control_paths, control_source_closure_bounded = _linked_control_source_paths(
        (
            (launch_plan, source_paths["launch_plan"]),
            (launch_receipt, source_paths["launch_receipt"]),
            (status_receipt, source_paths["status_receipt"]),
            (output_manifest, source_paths["output_artifact_manifest"]),
        )
    )
    control_paths.update(source_paths.values())
    if preflight_path is not None:
        control_paths.add(preflight_path)
    artifact_rows, artifact_errors, artifact_paths = _output_artifacts(
        output_manifest,
        source_paths["output_artifact_manifest"],
    )
    output_control_aliases = [
        artifact_path
        for artifact_path in artifact_paths
        if any(
            _paths_alias(artifact_path, control_path)
            for control_path in control_paths
        )
    ]
    if output_path is not None:
        _reject_output_source_collision(
            output_path,
            [*source_paths.values(), *artifact_paths, *([preflight_path] if preflight_path else [])],
        )
    output_artifact_count = sum(
        row.get("role") in OUTPUT_ARTIFACT_ROLES for row in artifact_rows
    )
    regular_artifact_count = sum(row.get("regular_file") is True for row in artifact_rows)
    artifact_set_sha = _canonical_sha256(artifact_rows)

    source_refs = {
        name: _source_ref(
            snapshots[name],
            payloads.get(name),
        )
        for name in source_paths
    }
    runner_source_bindings = {
        name: source_hashes.get(name)
        == snapshots[name].sha256
        and snapshots[name].sha256 is not None
        for name in _SOURCE_HASH_KEYS
    }
    source_identity_distinct = _source_identities_distinct(snapshots)
    source_stable_after_validation = True
    for snapshot in snapshots.values():
        if snapshot.sha256 is None:
            continue
        current = _stable_file_fingerprint(
            snapshot.path,
            max_bytes=MAX_RAW_PROVIDER_RESULT_BYTES,
        )
        if (
            current is None
            or current.sha256 != snapshot.sha256
            or current.size_bytes != snapshot.size_bytes
            or current.identity != snapshot.identity
        ):
            source_stable_after_validation = False
            break

    max_cost = _finite_nonnegative(preflight_constraints.get("max_cost_usd"))
    reported_cost = _finite_nonnegative(runner_constraints.get("reported_cost_usd"))
    cost_within_limit = (
        max_cost is not None
        and reported_cost is not None
        and reported_cost <= max_cost
    )
    region_matches = (
        bool(_string(preflight_constraints.get("region")))
        and runner_constraints.get("region") == preflight_constraints.get("region")
    )
    gpu_matches = (
        bool(_string(preflight_constraints.get("gpu_class")))
        and runner_constraints.get("gpu_class")
        == preflight_constraints.get("gpu_class")
    )

    hfr_side_effect_free = all(side_effects.get(key) is False for key in _HFR_SIDE_EFFECT_KEYS)
    failure_coherent = _failure_coherent(
        status,
        _string(failure.get("class")),
        _string(failure.get("message")),
    )
    terminal_coherent = status not in {"completed", "failed"} or metadata.get("terminal") is True
    runner_started_at = _parsed_timestamp(metadata.get("started_at"))
    launch_handoff_times = [
        parsed
        for parsed in (
            _parsed_timestamp(launch_plan.get("created_at")),
            _parsed_timestamp(launch_receipt.get("created_at")),
        )
        if parsed is not None
    ]
    runner_started_after_launch_handoff = (
        runner_started_at is not None
        and len(launch_handoff_times) == 2
        and runner_started_at >= max(launch_handoff_times)
    )
    status_created_at = _parsed_timestamp(status_receipt.get("created_at"))
    completion_after_status_receipt = (
        status_created_at is not None and parsed_created_at >= status_created_at
    )
    completed_result = (
        output_manifest.get("passed") is True
        and _training_result_status(output_manifest) == "completed"
    )

    checks: list[dict[str, Any]] = []
    _add_check(checks, "direct_sources_replayable", all(_source_ready(row) for row in snapshots.values()), "all direct evidence sources must be stable replayable regular files")
    _add_check(checks, "direct_sources_distinct", source_identity_distinct, "direct evidence sources must not alias one another")
    _add_check(checks, "source_json_duplicate_free", not any(parse_errors.values()), "control-plane JSON sources must be duplicate-free finite JSON objects")
    _add_check(checks, "upstream_schemas_valid", not any(upstream_schema_errors.values()), "cloud launch/status and output-manifest schemas must validate")
    _add_check(checks, "upstream_semantics_valid", not any(upstream_semantic_errors.values()), "cloud launch/status and output-manifest semantic validators must pass")
    _add_check(checks, "linked_control_source_closure_bounded", control_source_closure_bounded, "linked control evidence must remain a bounded replayable JSON closure")
    _add_check(checks, "launch_receipt_binds_launch_plan", bool(launch_plan_sha) and launch_plan_link == launch_plan_sha, "launch receipt must bind the supplied launch plan by SHA-256")
    _add_check(checks, "status_receipt_binds_launch_receipt", bool(launch_receipt_sha) and launch_receipt_link == launch_receipt_sha, "status receipt must bind the supplied launch receipt by SHA-256")
    _add_check(checks, "runner_metadata_valid", not metadata_errors, "runner metadata must use the public import envelope")
    _add_check(checks, "runner_binds_all_sources", all(runner_source_bindings.values()), "runner metadata must bind every supplied evidence source by SHA-256")
    _add_check(checks, "provider_identity_matches", bool(provider_id) and provider_id == launch_provider_id, "runner provider identity must match the launch-plan provider")
    _add_check(checks, "training_plan_lineage_converges", bool(preflight_plan_sha) and preflight_plan_sha == result_plan_sha, "cloud preflight and agentic training result must bind the same training plan")
    _add_check(checks, "provider_constraints_match", region_matches and gpu_matches and cost_within_limit, "reported region, GPU class, and cost must satisfy the preflight limits")
    _add_check(checks, "execution_failure_coherent", failure_coherent and terminal_coherent, "terminal outcomes and classified failures must be coherent")
    _add_check(checks, "runner_started_after_launch_handoff", runner_started_after_launch_handoff, "runner execution must not predate the launch plan and launch receipt")
    _add_check(checks, "completion_after_status_receipt", completion_after_status_receipt, "completion import must not predate its status receipt")
    _add_check(checks, "flight_recorder_import_only", hfr_side_effect_free, "runner metadata must attest that Flight Recorder performed no provider or training side effects")
    _add_check(checks, "credential_values_not_recorded", side_effects.get("credential_values_recorded") == "not_observed", "runner must explicitly attest that credential values were not recorded")
    _add_check(checks, "output_artifacts_current", not artifact_errors, "every output artifact must remain current, unique, nonempty when executable, and inside the declared output root")
    _add_check(checks, "output_artifacts_disjoint_from_evidence", not output_control_aliases, "output artifacts must not alias direct or linked control evidence")
    _add_check(checks, "completed_external_provider_api_called", status != "completed" or side_effects.get("external_provider_api_called") is True, "completed execution requires externally owned provider API activity")
    _add_check(checks, "completed_external_cloud_job_started", status != "completed" or side_effects.get("external_cloud_job_started") is True, "completed execution requires an externally owned cloud job")
    _add_check(checks, "completed_external_artifacts_downloaded", status != "completed" or side_effects.get("external_artifacts_downloaded") is True, "completed execution requires the external runner to materialize the declared local outputs")
    _add_check(checks, "completed_runner_exit_zero", status != "completed" or metadata.get("exit_code") == 0, "completed execution requires runner exit code zero")
    _add_check(checks, "completed_result_manifest_passed", status != "completed" or completed_result, "completed execution requires a passing completed agentic training result")
    _add_check(checks, "completed_candidate_matches", status != "completed" or (bool(candidate_model_id) and candidate_model_id == result_candidate_id), "completed execution candidate must match the agentic result registry target")
    _add_check(checks, "completed_outputs_nonempty", status != "completed" or output_artifact_count > 0, "completed execution requires at least one adapter or checkpoint")
    _add_check(checks, "sources_stable_after_validation", source_stable_after_validation, "direct evidence sources must remain stable throughout semantic validation")

    failed_checks = [row for row in checks if row["passed"] is not True]
    integrity_passed = not failed_checks
    completion_claims_allowed = (
        integrity_passed
        and status == "completed"
        and metadata.get("terminal") is True
        and completed_result
        and candidate_model_id == result_candidate_id
        and output_artifact_count > 0
    )
    governance_blockers = [row["summary"] for row in failed_checks]
    if status != "completed":
        governance_blockers.append("external cloud training execution is not completed")
    if not completed_result:
        governance_blockers.append("agentic training result is not a passing completed result")
    if candidate_model_id != result_candidate_id:
        governance_blockers.append("completion candidate does not match the training result")
    if output_artifact_count == 0:
        governance_blockers.append("training result has no adapter or checkpoint output")
    governance_blockers = _unique(governance_blockers)

    execution = {
        "status": status,
        "terminal": metadata.get("terminal") is True,
        "failure": {
            "class": _string(failure.get("class")),
            "message": _string(failure.get("message")),
            "classified": status == "completed"
            or (
                _string(failure.get("class")) in FAILURE_CLASSES[1:]
                and bool(_string(failure.get("message")))
            ),
        },
    }
    outputs = {
        "manifest_schema_version": output_manifest.get("schema_version"),
        "manifest_sha256": output_manifest_sha,
        "artifact_count": len(artifact_rows),
        "regular_artifact_count": regular_artifact_count,
        "output_artifact_count": output_artifact_count,
        "artifact_set_sha256": artifact_set_sha,
        "candidate_model_id": result_candidate_id,
    }
    provider_constraints_record = {
        "region": _string(runner_constraints.get("region")),
        "gpu_class": _string(runner_constraints.get("gpu_class")),
        "max_cost_usd": max_cost,
        "reported_cost_usd": reported_cost,
        "cost_within_limit": cost_within_limit,
    }
    identity_base = {
        "provider_id": provider_id,
        "provider_job_id": _string(metadata.get("provider_job_id")),
        "execution_id": _string(metadata.get("execution_id")),
        "candidate_model_id": candidate_model_id,
        "runner_id": _string(runner.get("id")),
        "runner_version": _string(runner.get("version")),
        "launch_plan_sha256": launch_plan_sha,
        "launch_receipt_sha256": launch_receipt_sha,
        "status_receipt_sha256": status_receipt_sha,
        "runner_metadata_sha256": runner_metadata_sha,
        "raw_provider_result_sha256": raw_result_sha,
        "output_artifact_manifest_sha256": output_manifest_sha,
        "output_artifact_set_sha256": artifact_set_sha,
        "agentic_training_plan_sha256": result_plan_sha,
    }
    identity = {
        **identity_base,
        "digest_sha256": _evidence_digest(
            identity_base,
            execution,
            outputs,
            provider_constraints_record,
            side_effects,
        ),
    }
    payload = {
        "schema_version": CLOUD_TRAINING_COMPLETION_RECEIPT_SCHEMA_VERSION,
        "created_at": created_at_value,
        "artifact_path": output_path.name if output_path is not None else "",
        "passed": integrity_passed,
        "integrity": {
            "passed": integrity_passed,
            "check_count": len(checks),
            "failed_check_count": len(failed_checks),
            "checks": checks,
            "blocking_reasons": [row["summary"] for row in failed_checks],
        },
        "execution": execution,
        "sources": source_refs,
        "identity": identity,
        "provider_constraints": provider_constraints_record,
        "runner_observation": {
            "runner_id": _string(runner.get("id")),
            "runner_version": _string(runner.get("version")),
            "started_at": _string(metadata.get("started_at")),
            "finished_at": _string(metadata.get("finished_at")),
            "exit_code": metadata.get("exit_code"),
            "side_effects": side_effects,
        },
        "outputs": outputs,
        "governance": {
            "readiness": "ready_for_review"
            if completion_claims_allowed
            else "blocked",
            "cloud_training_completion_claims_allowed": completion_claims_allowed,
            "recommendation": (
                "review_cloud_training_completion"
                if completion_claims_allowed
                else (
                    "archive_cloud_training_failure"
                    if integrity_passed and status == "failed"
                    else "block_cloud_training_completion_claims"
                )
            ),
            "blocking_reasons": governance_blockers,
        },
        "execution_boundary": _execution_boundary(),
        "notes": [
            "This receipt imports externally reported evidence; it does not verify provider authenticity.",
            "Flight Recorder does not call or poll providers, upload or download artifacts, import provider SDKs, start cloud jobs, or update weights while building this receipt.",
            "Receipt integrity is independent from execution outcome; valid failed, incomplete, and unknown outcomes remain auditable but cannot enable completion claims.",
        ],
    }
    intended_output_path = (
        output_path.resolve(strict=False) if output_path is not None else None
    )
    return _BuiltCloudTrainingCompletionReceipt(payload, intended_output_path)


def write_cloud_training_completion_receipt(
    receipt: dict[str, Any],
    out_path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> None:
    """Replay imported evidence and atomically publish the exact built receipt."""
    path = Path(out_path)
    intended = getattr(receipt, "_intended_output_path", None)
    try:
        resolved = path.resolve(strict=False)
    except OSError as exc:
        raise CloudTrainingCompletionError(
            "cloud completion output path could not be resolved safely"
        ) from exc
    if intended is None or intended != resolved:
        raise CloudTrainingCompletionError(
            "cloud completion receipt must be written to the exact build target"
        )
    if receipt.get("artifact_path") != path.name:
        raise CloudTrainingCompletionError(
            "cloud completion receipt artifact_path must match the output basename"
        )
    try:
        schema_check = check_schema_contract(
            dict(receipt),
            name_or_id="cloud_training_completion_receipt",
        )
    except (MemoryError, RecursionError, SchemaRegistryError, TypeError, ValueError) as exc:
        raise CloudTrainingCompletionError(
            "cloud completion receipt schema could not be checked safely"
        ) from exc
    if schema_check.get("passed") is not True:
        raise CloudTrainingCompletionError(
            "cloud completion receipt must satisfy its public schema before publication"
        )
    sources = receipt.get("sources") if isinstance(receipt.get("sources"), dict) else {}
    resolved_sources: dict[str, Path] = {}
    for name in (
        "launch_plan",
        "launch_receipt",
        "status_receipt",
        "runner_metadata",
        "raw_provider_result",
        "output_artifact_manifest",
    ):
        ref = sources.get(name) if isinstance(sources.get(name), dict) else {}
        raw_path = ref.get("path")
        if not isinstance(raw_path, str) or not _safe_relative_path(raw_path):
            raise CloudTrainingCompletionError(
                f"cloud completion source {name} is not replayable"
            )
        resolved_sources[name] = path.parent / raw_path
    replay = build_cloud_training_completion_receipt(
        launch_plan_path=resolved_sources["launch_plan"],
        launch_receipt_path=resolved_sources["launch_receipt"],
        status_receipt_path=resolved_sources["status_receipt"],
        runner_metadata_path=resolved_sources["runner_metadata"],
        raw_provider_result_path=resolved_sources["raw_provider_result"],
        output_artifact_manifest_path=resolved_sources["output_artifact_manifest"],
        out_path=path,
        created_at=_string(receipt.get("created_at")),
    )
    if dict(receipt) != dict(replay):
        raise CloudTrainingCompletionError(
            "cloud completion evidence changed after the receipt was built"
        )
    atomic_write_json_cas(
        path,
        dict(receipt),
        expected_sha256=expected_sha256,
    )


def cloud_training_completion_digest(receipt: dict[str, Any]) -> str:
    """Recompute the canonical imported-evidence digest."""
    identity = receipt.get("identity") if isinstance(receipt.get("identity"), dict) else {}
    identity_base = {key: value for key, value in identity.items() if key != "digest_sha256"}
    observation = (
        receipt.get("runner_observation")
        if isinstance(receipt.get("runner_observation"), dict)
        else {}
    )
    return _evidence_digest(
        identity_base,
        receipt.get("execution"),
        receipt.get("outputs"),
        receipt.get("provider_constraints"),
        observation.get("side_effects"),
    )


def _capture_source(
    path: Path,
    base_dir: Path,
    label: str,
    byte_limit: int,
) -> _SourceSnapshot:
    active_opaque = get_active_opaque_output_attestation(path)
    if active_opaque is not None:
        display, replayable = _display_path(path, base_dir)
        return _SourceSnapshot(
            path=path,
            display_path=display,
            exists=True,
            regular_file=True,
            replayable=replayable,
            sha256=active_opaque.sha256,
            size_bytes=active_opaque.size_bytes,
            data=None,
            identity=active_opaque.identity,
            errors=(),
        )
    if path_has_symlink_component(path, include_leaf=True):
        raise CloudTrainingCompletionError(
            f"{label} source path must not contain symlink components"
        )
    display, replayable = _display_path(path, base_dir)
    try:
        before = path.stat()
    except OSError:
        return _SourceSnapshot(path, display, False, False, replayable, None, None, None, None, (f"{label} not found",))
    if not path.is_file():
        return _SourceSnapshot(path, display, True, False, replayable, None, before.st_size, None, None, (f"{label} is not a regular file",))
    if before.st_size > byte_limit:
        return _SourceSnapshot(path, display, True, True, replayable, None, before.st_size, None, (before.st_dev, before.st_ino), (f"{label} exceeds {byte_limit} bytes",))
    try:
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            data = handle.read(byte_limit + 1)
            closed = os.fstat(handle.fileno())
        after = path.stat()
    except OSError:
        return _SourceSnapshot(path, display, True, True, replayable, None, before.st_size, None, (before.st_dev, before.st_ino), (f"{label} could not be read",))
    stable = (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
        == (closed.st_dev, closed.st_ino, closed.st_size, closed.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    )
    errors: list[str] = []
    if len(data) > byte_limit:
        errors.append(f"{label} exceeds {byte_limit} bytes")
        data = b""
    if not stable:
        errors.append(f"{label} changed while being imported")
        data = b""
    return _SourceSnapshot(
        path,
        display,
        True,
        True,
        replayable,
        hashlib.sha256(data).hexdigest() if not errors else None,
        len(data) if not errors else before.st_size,
        data if not errors else None,
        (before.st_dev, before.st_ino),
        tuple(errors),
    )


def _display_path(path: Path, base_dir: Path) -> tuple[str, bool]:
    try:
        relative = path.resolve().relative_to(base_dir.resolve())
        rendered = relative.as_posix()
        if not _safe_relative_path(rendered):
            raise ValueError
        return rendered, True
    except (OSError, ValueError):
        return f"<redacted:{_public_basename(path)}>", False


def _parse_json_object(
    source: _SourceSnapshot,
    label: str,
) -> tuple[dict[str, Any], list[str]]:
    errors = list(source.errors)
    if source.data is None:
        errors.append(f"{label} bytes are unavailable")
        return {}, errors
    try:
        value = json.loads(
            source.data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        _check_json_depth(value)
    except (_DuplicateJsonKeyError, UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError):
        errors.append(f"{label} must be duplicate-free finite JSON")
        return {}, errors
    if not isinstance(value, dict):
        errors.append(f"{label} must be a JSON object")
        return {}, errors
    return value, errors


def _runner_metadata_errors(metadata: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _unknown_keys(metadata, _METADATA_KEYS, "runner metadata", errors)
    if metadata.get("schema_version") != EXTERNAL_CLOUD_TRAINING_RUNNER_SCHEMA_VERSION:
        errors.append("runner metadata schema_version is unsupported")
    for field in ("provider_id", "provider_job_id", "execution_id", "candidate_model_id"):
        if not _public_identifier(metadata.get(field)):
            errors.append(f"runner metadata {field} must be a public identifier")
    status = _string(metadata.get("status")).lower()
    if status not in EXECUTION_STATUSES:
        errors.append("runner metadata status is unsupported")
    if not isinstance(metadata.get("terminal"), bool):
        errors.append("runner metadata terminal must be boolean")
    if not isinstance(metadata.get("exit_code"), int) or isinstance(metadata.get("exit_code"), bool):
        errors.append("runner metadata exit_code must be an integer")
    parsed_timestamps: dict[str, datetime | None] = {}
    for field in ("started_at", "finished_at"):
        parsed_timestamps[field] = _parsed_timestamp(metadata.get(field))
        if parsed_timestamps[field] is None:
            errors.append(f"runner metadata {field} must be an ISO-8601 timestamp")
    started_at = parsed_timestamps["started_at"]
    finished_at = parsed_timestamps["finished_at"]
    if (
        started_at is not None
        and finished_at is not None
        and finished_at < started_at
    ):
        errors.append("runner metadata finished_at must not precede started_at")
    failure = metadata.get("failure")
    if not isinstance(failure, dict):
        errors.append("runner metadata failure must be an object")
    else:
        _unknown_keys(failure, _FAILURE_KEYS, "runner failure", errors)
        if failure.get("class") not in FAILURE_CLASSES:
            errors.append("runner failure class is unsupported")
        if not isinstance(failure.get("message"), str) or not _public_message(failure.get("message")):
            errors.append("runner failure message must be public-safe")
    runner = metadata.get("runner")
    if not isinstance(runner, dict):
        errors.append("runner metadata runner must be an object")
    else:
        _unknown_keys(runner, _RUNNER_KEYS, "runner", errors)
        for field in ("id", "version"):
            if not _public_identifier(runner.get(field)):
                errors.append(f"runner {field} must be a public identifier")
    constraints = metadata.get("provider_constraints")
    if not isinstance(constraints, dict):
        errors.append("runner metadata provider_constraints must be an object")
    else:
        _unknown_keys(constraints, _CONSTRAINT_KEYS, "runner constraints", errors)
        for field in ("region", "gpu_class"):
            if not _public_identifier(constraints.get(field)):
                errors.append(f"runner constraint {field} must be public")
        if _finite_nonnegative(constraints.get("reported_cost_usd")) is None:
            errors.append("runner reported cost must be finite and non-negative")
    hashes = metadata.get("source_sha256")
    if not isinstance(hashes, dict):
        errors.append("runner metadata source_sha256 must be an object")
    else:
        _unknown_keys(hashes, _SOURCE_HASH_KEYS, "runner source hashes", errors)
        if set(hashes) != _SOURCE_HASH_KEYS or not all(_is_sha256(value) for value in hashes.values()):
            errors.append("runner metadata must bind every required source SHA-256")
    effects = metadata.get("side_effects")
    if not isinstance(effects, dict):
        errors.append("runner metadata side_effects must be an object")
    else:
        _unknown_keys(effects, _SIDE_EFFECT_KEYS, "runner side effects", errors)
        if set(effects) != _SIDE_EFFECT_KEYS:
            errors.append("runner side effects must include every required field")
        for field in _SIDE_EFFECT_KEYS - {"credential_values_recorded"}:
            if not isinstance(effects.get(field), bool):
                errors.append(f"runner side effect {field} must be boolean")
        if effects.get("credential_values_recorded") not in {"observed", "not_observed", "unknown"}:
            errors.append("credential observation status is unsupported")
    return errors


def _upstream_semantic_errors(source_paths: dict[str, Path]) -> dict[str, list[str]]:
    from .validation import (
        validate_agentic_training_result,
        validate_cloud_training_launch_plan,
        validate_cloud_training_launch_receipt,
        validate_cloud_training_status_receipt,
    )

    validators = {
        "launch_plan": validate_cloud_training_launch_plan,
        "launch_receipt": validate_cloud_training_launch_receipt,
        "status_receipt": validate_cloud_training_status_receipt,
        "output_artifact_manifest": validate_agentic_training_result,
    }
    result: dict[str, list[str]] = {}
    for name, validator in validators.items():
        try:
            validation = validator(source_paths[name])
            result[name] = [*validation.errors, *validation.warnings]
        except (
            MemoryError,
            OSError,
            RecursionError,
            TypeError,
            UnicodeError,
            ValueError,
        ) as exc:
            result[name] = [str(exc)]
    return result


def _linked_json_source(
    payload: dict[str, Any],
    payload_path: Path,
    source_name: str,
    label: str,
) -> tuple[Path | None, dict[str, Any], list[str]]:
    sources = payload.get("source_artifacts") if isinstance(payload.get("source_artifacts"), dict) else {}
    ref = sources.get(source_name) if isinstance(sources.get(source_name), dict) else {}
    raw_path = ref.get("path")
    if not isinstance(raw_path, str) or not _safe_relative_path(raw_path):
        return None, {}, [f"{label} path is not replayable"]
    path = payload_path.parent / raw_path
    if path_has_symlink_component(path, include_leaf=True) or not path.is_file():
        return path, {}, [f"{label} is not a regular non-symlink file"]
    snapshot = _capture_source(
        path,
        payload_path.parent,
        label,
        MAX_CONTROL_SOURCE_BYTES,
    )
    errors: list[str] = []
    if ref.get("sha256") != snapshot.sha256 or ref.get("size_bytes") != snapshot.size_bytes:
        errors.append(f"{label} fingerprint does not match")
    value, parse_errors = _parse_json_object(snapshot, label)
    errors.extend(parse_errors)
    return path, value, errors


def _output_artifacts(
    manifest: dict[str, Any],
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], list[str], list[Path]]:
    raw_rows = manifest.get("artifacts")
    if not isinstance(raw_rows, list):
        return [], ["agentic training result artifacts must be a list"], []
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    paths: list[Path] = []
    seen: set[tuple[Any, ...]] = set()
    seen_identities: set[tuple[int, int]] = set()
    training_result = (
        manifest.get("training_result")
        if isinstance(manifest.get("training_result"), dict)
        else {}
    )
    output_dir = training_result.get("output_dir")
    output_root = (
        manifest_path.parent / output_dir
        if isinstance(output_dir, str) and _safe_relative_path(output_dir)
        else None
    )
    output_rows = [
        row
        for row in raw_rows
        if isinstance(row, dict) and row.get("role") in OUTPUT_ARTIFACT_ROLES
    ]
    output_declared_sizes = [row.get("size_bytes") for row in output_rows]
    output_bounds_valid = (
        len(output_rows) <= MAX_OPAQUE_TRAINING_OUTPUT_FILES
        and all(
            isinstance(size, int)
            and not isinstance(size, bool)
            and 0 < size <= MAX_OPAQUE_TRAINING_OUTPUT_BYTES
            for size in output_declared_sizes
        )
        and sum(
            size
            for size in output_declared_sizes
            if isinstance(size, int) and not isinstance(size, bool)
        )
        <= MAX_OPAQUE_TRAINING_OUTPUT_TOTAL_BYTES
    )
    if not output_bounds_valid:
        errors.append("output artifacts exceed bounded count or byte limits")
    for index, raw_row in enumerate(raw_rows):
        if not isinstance(raw_row, dict):
            errors.append(f"output artifact {index} is not an object")
            continue
        path_value = raw_row.get("path")
        role = raw_row.get("role")
        if not isinstance(path_value, str) or not _safe_relative_path(path_value) or not isinstance(role, str) or not role:
            errors.append(f"output artifact {index} has an unsafe identity")
            continue
        path = manifest_path.parent / path_value
        paths.append(path)
        fingerprint = _stable_file_fingerprint(
            path,
            max_bytes=(
                MAX_OPAQUE_TRAINING_OUTPUT_BYTES
                if role in OUTPUT_ARTIFACT_ROLES and output_bounds_valid
                else MAX_CONTROL_SOURCE_BYTES
            ),
        )
        regular = fingerprint is not None
        sha = fingerprint.sha256 if fingerprint is not None else None
        size = fingerprint.size_bytes if fingerprint is not None else None
        identity = fingerprint.identity if fingerprint is not None else None
        normalized = {
            "role": role,
            "path": path_value,
            "sha256": raw_row.get("sha256"),
            "size_bytes": raw_row.get("size_bytes"),
            "regular_file": regular,
        }
        key = (role, path_value, raw_row.get("sha256"), raw_row.get("size_bytes"))
        if key in seen:
            errors.append(f"output artifact {index} duplicates another artifact")
        seen.add(key)
        if identity is not None:
            if identity in seen_identities:
                errors.append(
                    f"output artifact {index} aliases another artifact"
                )
            seen_identities.add(identity)
        if role in OUTPUT_ARTIFACT_ROLES:
            if not isinstance(size, int) or size <= 0:
                errors.append(
                    f"output artifact {index} must contain nonempty executable output"
                )
            if output_root is None or not _path_is_within(path, output_root):
                errors.append(
                    f"output artifact {index} is outside training_result.output_dir"
                )
        if (
            raw_row.get("exists") is not True
            or raw_row.get("regular_file") is not True
            or not regular
            or sha != raw_row.get("sha256")
            or size != raw_row.get("size_bytes")
        ):
            errors.append(f"output artifact {index} fingerprint is stale")
        rows.append(normalized)
    rows.sort(key=lambda row: (str(row["role"]), str(row["path"]), str(row["sha256"])))
    return rows, errors, paths


def _linked_control_source_paths(
    seeds: tuple[tuple[dict[str, Any], Path], ...],
) -> tuple[set[Path], bool]:
    """Collect only typed control-source links, never output artifact rows."""
    pending = list(seeds)
    paths: set[Path] = set()
    seen: set[tuple[int, int]] = set()
    complete = True
    while pending:
        payload, payload_path = pending.pop()
        for container_name in ("source_artifacts", "lineage"):
            container = payload.get(container_name)
            if not isinstance(container, dict):
                continue
            for raw_ref in container.values():
                refs = raw_ref if isinstance(raw_ref, list) else [raw_ref]
                for ref in refs:
                    if not isinstance(ref, dict) or ref.get("exists") is False:
                        continue
                    raw_path = ref.get("path")
                    if not isinstance(raw_path, str) or not _safe_relative_path(raw_path):
                        if ref.get("exists") is True:
                            complete = False
                        continue
                    path = payload_path.parent / raw_path
                    if path_has_symlink_component(path, include_leaf=True) or not path.is_file():
                        if ref.get("exists") is True:
                            complete = False
                        continue
                    try:
                        file_stat = path.stat()
                    except OSError:
                        complete = False
                        continue
                    identity = (file_stat.st_dev, file_stat.st_ino)
                    paths.add(path)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    if len(seen) > MAX_LINKED_CONTROL_SOURCES:
                        return paths, False
                    snapshot = _capture_source(
                        path,
                        payload_path.parent,
                        "linked control source",
                        MAX_CONTROL_SOURCE_BYTES,
                    )
                    linked_payload, errors = _parse_json_object(
                        snapshot,
                        "linked control source",
                    )
                    if errors:
                        complete = False
                        continue
                    pending.append((linked_payload, path))
    return paths, complete


def _paths_alias(left: Path, right: Path) -> bool:
    try:
        if left.resolve(strict=False) == right.resolve(strict=False):
            return True
        return left.exists() and right.exists() and os.path.samefile(left, right)
    except OSError:
        return False


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except (OSError, ValueError):
        return False
    return True


def _execution_boundary() -> dict[str, Any]:
    return {
        "import_only": True,
        "external_runner_owns_execution": True,
        "provider_api_called_by_flight_recorder": False,
        "cloud_job_started_by_flight_recorder": False,
        "provider_status_polled_by_flight_recorder": False,
        "artifacts_uploaded_by_flight_recorder": False,
        "artifacts_downloaded_by_flight_recorder": False,
        "model_downloads_started_by_flight_recorder": False,
        "weights_updated_by_flight_recorder": False,
        "provider_modules_imported_by_flight_recorder": False,
        "credential_values_recorded_by_flight_recorder": False,
        "raw_provider_result_embedded": False,
    }


def _source_ref(source: _SourceSnapshot, payload: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "path": source.display_path,
        "exists": source.exists,
        "regular_file": source.regular_file,
        "replayable": source.replayable,
        "sha256": source.sha256,
        "size_bytes": source.size_bytes,
        "schema_version": payload.get("schema_version")
        if isinstance(payload, dict) and isinstance(payload.get("schema_version"), str)
        else None,
    }


def _source_ready(source: _SourceSnapshot) -> bool:
    return source.exists and source.regular_file and source.replayable and source.sha256 is not None and not source.errors


def _source_link_sha(payload: dict[str, Any], source_name: str) -> str:
    sources = payload.get("source_artifacts") if isinstance(payload.get("source_artifacts"), dict) else {}
    ref = sources.get(source_name) if isinstance(sources.get(source_name), dict) else {}
    return _string(ref.get("sha256"))


def _provider_id(payload: dict[str, Any]) -> str:
    provider = payload.get("provider") if isinstance(payload.get("provider"), dict) else {}
    return _string(provider.get("id"))


def _lineage_sha(payload: dict[str, Any], name: str) -> str:
    lineage = payload.get("lineage") if isinstance(payload.get("lineage"), dict) else {}
    ref = lineage.get(name) if isinstance(lineage.get(name), dict) else {}
    return _string(ref.get("sha256"))


def _training_result_candidate(payload: dict[str, Any]) -> str:
    registry = payload.get("registry_update") if isinstance(payload.get("registry_update"), dict) else {}
    return _string(registry.get("target_model_id"))


def _training_result_status(payload: dict[str, Any]) -> str:
    result = payload.get("training_result") if isinstance(payload.get("training_result"), dict) else {}
    return _string(result.get("status"))


def _schema_errors(payload: dict[str, Any], schema_name: str) -> list[str]:
    try:
        result = check_schema_contract(payload, name_or_id=schema_name)
    except (SchemaRegistryError, TypeError, ValueError) as exc:
        return [str(exc)]
    return [str(error) for error in result.get("errors", [])]


def _source_identities_distinct(sources: dict[str, _SourceSnapshot]) -> bool:
    identities = [row.identity for row in sources.values() if row.identity is not None]
    return len(identities) == len(set(identities))


def _failure_coherent(status: str, failure_class: str, message: str) -> bool:
    if status not in EXECUTION_STATUSES or failure_class not in FAILURE_CLASSES:
        return False
    if status == "completed":
        return failure_class == "none" and not message
    return failure_class in FAILURE_CLASSES[1:] and bool(message)


def _evidence_digest(
    identity: dict[str, Any],
    execution: Any,
    outputs: Any,
    constraints: Any,
    side_effects: Any,
) -> str:
    return _canonical_sha256(
        {
            "identity": identity,
            "execution": execution,
            "outputs": outputs,
            "provider_constraints": constraints,
            "runner_side_effects": side_effects,
        }
    )


def _canonical_sha256(value: Any) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _reject_output_source_collision(output_path: Path, source_paths: list[Path]) -> None:
    try:
        resolved_output = output_path.resolve(strict=False)
    except OSError as exc:
        raise CloudTrainingCompletionError("cloud completion output path could not be resolved") from exc
    for source_path in source_paths:
        try:
            aliases = resolved_output == source_path.resolve(strict=False)
        except OSError as exc:
            raise CloudTrainingCompletionError("cloud completion source path could not be resolved") from exc
        if not aliases and output_path.exists() and source_path.exists():
            try:
                aliases = os.path.samefile(output_path, source_path)
            except OSError:
                aliases = False
        if aliases:
            raise CloudTrainingCompletionError("cloud completion output must not alias an input or output artifact")


def _safe_relative_path(value: str) -> bool:
    if not value or "\\" in value or "\x00" in value or value.startswith("~"):
        return False
    path = Path(value)
    windows = PureWindowsPath(value)
    return (
        not path.is_absolute()
        and not windows.is_absolute()
        and not windows.drive
        and ".." not in path.parts
        and "." not in path.parts
        and ":" not in value
        and not value.startswith("<redacted:")
        and all(_public_path_component(part) for part in path.parts)
    )


def _public_identifier(value: Any) -> bool:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 256
        or any(ord(char) < 0x20 for char in value)
        or "\\" in value
    ):
        return False
    lowered = value.lower()
    if any(token in lowered for token in ("/users/", "/home/", "c:\\")):
        return False
    if "/" in value:
        parts = value.split("/")
        if value.startswith(("/", "~/")) or any(
            part in {"", ".", "..", "~"} or part.startswith("~")
            for part in parts
        ):
            return False
    return (
        "\n" not in value
        and "\r" not in value
        and "://" not in value
        and not value.startswith(("/", "~", "\\"))
        and not _SECRET_LIKE.search(value)
    )


def _public_message(value: Any) -> bool:
    if (
        not isinstance(value, str)
        or len(value) > 512
        or value != value.strip()
    ):
        return False
    lowered = value.lower()
    if any(
        token in lowered
        for token in (
            "://",
            "api_key",
            "apikey",
            "token=",
            "password",
            "/users/",
            "/home/",
            "c:\\",
        )
    ):
        return False
    return not any(ord(char) < 0x20 and char not in {"\t"} for char in value) and not _SECRET_LIKE.search(value)


def _parsed_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def _timestamp(value: Any) -> bool:
    return _parsed_timestamp(value) is not None


def _finite_nonnegative(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        return None
    return value


def _current_sha(path: Path) -> str | None:
    fingerprint = _stable_file_fingerprint(
        path,
        max_bytes=MAX_RAW_PROVIDER_RESULT_BYTES,
    )
    return fingerprint.sha256 if fingerprint is not None else None


def _stable_file_fingerprint(
    path: Path,
    *,
    max_bytes: int,
) -> _StableFileFingerprint | None:
    active_opaque = get_active_opaque_output_attestation(path)
    if active_opaque is not None:
        if active_opaque.size_bytes > max_bytes:
            return None
        return _StableFileFingerprint(
            identity=active_opaque.identity,
            sha256=active_opaque.sha256,
            size_bytes=active_opaque.size_bytes,
        )
    if max_bytes < 0 or path_has_symlink_component(path, include_leaf=True):
        return None
    try:
        pathname_before = path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(pathname_before.st_mode)
            or pathname_before.st_size > max_bytes
        ):
            return None
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        descriptor = os.open(path, flags)
    except (MemoryError, OSError):
        return None
    digest = hashlib.sha256()
    captured = 0
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_size > max_bytes:
            return None
        while True:
            remaining = max_bytes - captured
            chunk = os.read(descriptor, min(1024 * 1024, remaining + 1))
            if not chunk:
                break
            captured += len(chunk)
            if captured > max_bytes:
                return None
            digest.update(chunk)
        after = os.fstat(descriptor)
    except (MemoryError, OSError):
        return None
    finally:
        os.close(descriptor)
    try:
        if path_has_symlink_component(path, include_leaf=True):
            return None
        pathname_after = path.stat(follow_symlinks=False)
    except (MemoryError, OSError):
        return None
    if not (
        _file_stat_signature(pathname_before)
        == _file_stat_signature(opened)
        == _file_stat_signature(after)
        == _file_stat_signature(pathname_after)
        and captured == after.st_size
    ):
        return None
    return _StableFileFingerprint(
        identity=(after.st_dev, after.st_ino),
        sha256=digest.hexdigest(),
        size_bytes=captured,
    )


def _file_stat_signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_mode,
        value.st_dev,
        value.st_ino,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _string(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _public_basename(path: Path) -> str:
    name = path.name
    return name if _public_path_component(name) else "artifact.json"


def _public_path_component(value: Any) -> bool:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or value != value.strip()
        or len(value) > 255
        or any(ord(char) < 0x20 for char in value)
        or any(char in value for char in ("/", "\\", ":"))
    ):
        return False
    lowered = value.lower()
    if any(
        token in lowered
        for token in ("api_key", "apikey", "token=", "password", "bearer ")
    ):
        return False
    return not _SECRET_LIKE.search(value)


def _unknown_keys(value: dict[str, Any], allowed: set[str], label: str, errors: list[str]) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        errors.append(f"{label} contains unknown fields: {unknown}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(key)
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ValueError(f"unsupported JSON constant {value}")


def _check_json_depth(value: Any, depth: int = 0) -> None:
    if depth > MAX_JSON_DEPTH:
        raise ValueError("JSON nesting exceeds limit")
    if isinstance(value, dict):
        for child in value.values():
            _check_json_depth(child, depth + 1)
    elif isinstance(value, list):
        for child in value:
            _check_json_depth(child, depth + 1)


def _add_check(checks: list[dict[str, Any]], check_id: str, passed: bool, summary: str) -> None:
    checks.append({"id": check_id, "passed": bool(passed), "summary": summary})


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
