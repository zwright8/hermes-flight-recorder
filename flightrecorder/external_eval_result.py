"""Portable, import-only result receipts for externally executed evaluations."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .path_safety import path_has_symlink_component
from .schema_registry import SchemaRegistryError, check_schema_contract

EXTERNAL_EVAL_RESULT_SCHEMA_VERSION = "hfr.external_eval_result.v1"

EXTERNAL_EVAL_PLAN_SCHEMA_VERSION = "hfr.external_eval_adapters.v1"
HELDOUT_MANIFEST_SCHEMA_VERSION = "hfr.heldout_scenario_manifest.v1"
RUN_SUITE_SCHEMA_VERSION = "hfr.run_suite.v1"

EXECUTION_STATUSES = ("completed", "incomplete", "failed")
FAILURE_CLASSES = (
    "none",
    "runner_error",
    "timeout",
    "interrupted",
    "dependency_missing",
    "invalid_output",
    "aggregate_only_output",
)
CASE_STATUSES = ("passed", "failed", "error", "skipped")
OUTCOME_STATUSES = ("passed", "failed", "inconclusive", "not_available")
SIDE_EFFECT_STATUSES = ("observed", "not_observed", "unknown")
SUPPORTED_RAW_FORMATS = ("hfr.run_suite.v1", "json", "jsonl", "aggregate_json")
EXTERNAL_EVAL_NORMALIZER_CONTRACTS: frozenset[tuple[str, str, str, str]] = frozenset(
    {
        ("local_mock", "hfr.local_mock.run_suite", "1", "hfr.run_suite.v1"),
        ("local_mock", "hfr.local_mock.per_case_json", "1", "json"),
        ("local_mock", "hfr.local_mock.aggregate_json", "1", "aggregate_json"),
        ("bfcl", "hfr.bfcl.per_case_json", "1", "json"),
        ("bfcl", "hfr.bfcl.per_case_jsonl", "1", "jsonl"),
        ("bfcl", "hfr.bfcl.aggregate_json", "1", "aggregate_json"),
        ("inspect_ai", "hfr.inspect_ai.samples_json", "1", "json"),
        ("inspect_ai", "hfr.inspect_ai.samples_jsonl", "1", "jsonl"),
        ("inspect_ai", "hfr.inspect_ai.aggregate_json", "1", "aggregate_json"),
        ("lm_eval_harness", "hfr.lm_eval_harness.samples_jsonl", "1", "jsonl"),
        (
            "lm_eval_harness",
            "hfr.lm_eval_harness.aggregate_json",
            "1",
            "aggregate_json",
        ),
        ("swe_bench", "hfr.swe_bench.per_instance_json", "1", "json"),
        ("swe_bench", "hfr.swe_bench.per_instance_jsonl", "1", "jsonl"),
        ("swe_bench", "hfr.swe_bench.aggregate_json", "1", "aggregate_json"),
    }
)

MAX_SOURCE_BYTES = 64 * 1024 * 1024
MAX_JSONL_LINE_BYTES = 8 * 1024 * 1024
MAX_JSON_DEPTH = 100


class ExternalEvalResultError(ValueError):
    """Raised when an external-evaluation result cannot be imported safely."""


class _BuiltExternalEvalResult(dict[str, Any]):
    """Result payload carrying a non-serialized, immutable write target."""

    def __init__(self, payload: dict[str, Any], intended_output_path: Path | None):
        super().__init__(payload)
        self._intended_output_path = intended_output_path


class _DuplicateJsonKeyError(ValueError):
    pass


@dataclass(frozen=True)
class _SourceSnapshot:
    path: Path | None
    display_path: str | None
    exists: bool
    regular_file: bool
    replayable: bool
    sha256: str | None
    size_bytes: int | None
    data: bytes | None
    errors: tuple[str, ...]


def build_external_eval_result(
    *,
    plan_path: str | Path,
    heldout_manifest_path: str | Path,
    raw_result_path: str | Path,
    runner_metadata_path: str | Path | None,
    adapter_id: str,
    execution_id: str,
    model_id: str,
    normalizer_id: str,
    normalizer_version: str,
    raw_format: str,
    execution_status: str,
    failure_class: str = "none",
    failure_message: str = "",
    runner_observation: dict[str, Any] | None = None,
    out_path: str | Path | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Import external runner evidence without launching or importing a benchmark.

    The local mock ``hfr.run_suite.v1`` normalizer is the primary supported
    path. Generic JSON/JSONL records are accepted only when every record has a
    public case identifier and an explicit per-case outcome. Aggregate-only
    JSON is always non-completing evidence.
    """
    status = execution_status.strip().lower()
    failure = failure_class.strip().lower() or "none"
    if status not in EXECUTION_STATUSES:
        raise ExternalEvalResultError(
            f"unsupported execution status {execution_status!r}; expected one of {', '.join(EXECUTION_STATUSES)}"
        )
    if failure not in FAILURE_CLASSES:
        raise ExternalEvalResultError(
            f"unsupported failure class {failure_class!r}; expected one of {', '.join(FAILURE_CLASSES)}"
        )
    if raw_format not in SUPPORTED_RAW_FORMATS:
        raise ExternalEvalResultError(
            f"unsupported raw format {raw_format!r}; expected one of {', '.join(SUPPORTED_RAW_FORMATS)}"
        )
    if not _identity_is_public(
        adapter_id, execution_id, model_id, normalizer_id, normalizer_version
    ):
        raise ExternalEvalResultError(
            "external eval result identity must contain public identifiers, not paths, URLs, or secret-like values"
        )
    if failure_message and not _public_message(failure_message):
        raise ExternalEvalResultError(
            "external eval failure message must be public-safe and must not contain paths, URLs, or secret-like values"
        )
    source_paths = [
        Path(plan_path),
        Path(heldout_manifest_path),
        Path(raw_result_path),
        *([Path(runner_metadata_path)] if runner_metadata_path is not None else []),
    ]
    if out_path is not None:
        _reject_output_source_collision(Path(out_path), source_paths)
    base_dir = Path(out_path).parent if out_path is not None else Path.cwd()

    plan_source = _capture_source(Path(plan_path), base_dir, "plan")
    heldout_source = _capture_source(
        Path(heldout_manifest_path), base_dir, "heldout manifest"
    )
    raw_source = _capture_source(Path(raw_result_path), base_dir, "raw result")
    metadata_source = (
        _capture_source(Path(runner_metadata_path), base_dir, "runner metadata")
        if runner_metadata_path is not None
        else _missing_optional_source()
    )

    plan, plan_errors = _parse_json_object(plan_source, "plan")
    heldout, heldout_errors = _parse_json_object(heldout_source, "heldout manifest")
    plan_schema_errors = (
        _schema_errors(plan, "external_eval_plan") if not plan_errors else []
    )
    heldout_schema_errors = (
        _schema_errors(heldout, "heldout_manifest") if not heldout_errors else []
    )
    plan_semantic_errors, heldout_semantic_errors = _upstream_semantic_errors(
        Path(plan_path),
        Path(heldout_manifest_path),
        plan_source.sha256,
        heldout_source.sha256,
    )
    metadata, metadata_errors = (
        _parse_json_object(metadata_source, "runner metadata")
        if runner_metadata_path is not None
        else ({}, [])
    )
    records, raw_payload, raw_errors, aggregate_only = _parse_raw_records(
        raw_source, raw_format
    )

    metadata_observation = (
        metadata.get("runner_observation")
        if isinstance(metadata.get("runner_observation"), dict)
        else None
    )
    selected_observation = (
        runner_observation if runner_observation is not None else metadata_observation
    )
    normalized_observation, observation_errors = _normalize_runner_observation(
        selected_observation
    )
    if runner_observation is not None and metadata_observation is not None:
        if _canonical_sha256(runner_observation) != _canonical_sha256(
            metadata_observation
        ):
            observation_errors.append(
                "inline runner observation does not match runner metadata"
            )

    expected_case_ids, heldout_case_errors = _heldout_case_ids(heldout)
    cases, observed_ids, normalization_errors = _normalize_records(records, raw_format)
    all_normalizer_errors = [*raw_errors, *normalization_errors]
    coverage = _coverage(expected_case_ids, observed_ids, cases, len(records))
    outcome = _benchmark_outcome(cases, aggregate_only or bool(raw_errors))

    source_refs = {
        "plan": _source_ref(plan_source, plan),
        "heldout_manifest": _source_ref(heldout_source, heldout),
        "raw_result": _source_ref(raw_source, raw_payload),
        "runner_metadata": _source_ref(
            metadata_source, metadata if runner_metadata_path is not None else None
        ),
    }
    observation_sha = _canonical_sha256(normalized_observation)
    execution_record = {
        "status": status,
        "failure": {
            "class": failure,
            "message": failure_message,
            "classified": status == "completed"
            or (failure in FAILURE_CLASSES[1:] and bool(failure_message.strip())),
        },
    }
    identity_base = {
        "adapter_id": adapter_id,
        "execution_id": execution_id,
        "model_id": model_id,
        "plan_sha256": plan_source.sha256,
        "heldout_manifest_sha256": heldout_source.sha256,
        "raw_result_sha256": raw_source.sha256,
        "runner_metadata_sha256": metadata_source.sha256,
        "runner_observation_sha256": observation_sha,
        "normalizer_id": normalizer_id,
        "normalizer_version": normalizer_version,
        "raw_format": raw_format,
    }
    identity = {
        **identity_base,
        "digest_sha256": _evidence_digest(
            identity=identity_base,
            execution=execution_record,
            benchmark_outcome=outcome,
            normalizer={
                "id": normalizer_id,
                "version": normalizer_version,
                "input_format": raw_format,
            },
            coverage=coverage,
            cases=cases,
            runner_observation=normalized_observation,
        ),
    }

    checks: list[dict[str, Any]] = []
    _add_check(
        checks,
        "plan_source_replayable",
        _source_ready(plan_source),
        "plan source must be a portable regular file",
    )
    _add_check(
        checks,
        "plan_json_readable",
        not plan_errors,
        "plan must be a duplicate-free finite JSON object",
    )
    _add_check(
        checks,
        "plan_schema_valid",
        not plan_schema_errors,
        "plan must conform to the external-evaluation plan schema",
    )
    _add_check(
        checks,
        "plan_semantically_valid",
        not plan_semantic_errors,
        "plan must pass external-evaluation semantic validation",
    )
    _add_check(
        checks,
        "plan_ready_for_adapter",
        _plan_ready(plan, adapter_id),
        "plan must be ready and include the selected ready adapter",
    )
    _add_check(
        checks,
        "heldout_source_replayable",
        _source_ready(heldout_source),
        "held-out manifest source must be a portable regular file",
    )
    _add_check(
        checks,
        "heldout_manifest_ready",
        not heldout_errors
        and not heldout_schema_errors
        and not heldout_semantic_errors
        and not heldout_case_errors
        and _heldout_ready(heldout),
        "held-out manifest must be ready with unique scenario IDs",
    )
    _add_check(
        checks,
        "plan_binds_heldout_manifest",
        _plan_manifest_sha(plan) == heldout_source.sha256
        and heldout_source.sha256 is not None,
        "plan scenario-manifest hash must match the imported held-out manifest",
    )
    _add_check(
        checks,
        "raw_result_source_replayable",
        _source_ready(raw_source),
        "raw result must be a portable regular file",
    )
    _add_check(
        checks,
        "runner_metadata_or_inline_observation",
        (
            runner_metadata_path is None
            or (_source_ready(metadata_source) and not metadata_errors)
        )
        and not observation_errors,
        "runner observation must be public-safe and metadata, when supplied, must be replayable",
    )
    _add_check(
        checks,
        "runner_metadata_identity_matches",
        runner_metadata_path is None
        or _metadata_identity_matches(metadata, adapter_id, execution_id, model_id),
        "runner metadata identity must match adapter, execution, and model identity",
    )
    _add_check(
        checks,
        "identity_is_public_and_plan_bound",
        _identity_is_public(
            adapter_id, execution_id, model_id, normalizer_id, normalizer_version
        )
        and _model_matches_plan(plan, model_id),
        "public result identity must be non-secret and match the plan model",
    )
    _add_check(
        checks,
        "normalizer_contract_supported",
        (adapter_id, normalizer_id, normalizer_version, raw_format)
        in EXTERNAL_EVAL_NORMALIZER_CONTRACTS,
        "adapter, normalizer ID, version, and raw format must match an allowlisted import contract",
    )
    _add_check(
        checks,
        "execution_failure_coherent",
        _execution_failure_coherent(status, failure, failure_message),
        "completed execution requires no failure; non-completed execution requires a classified failure",
    )
    _add_check(
        checks,
        "completed_runner_exit_zero",
        status != "completed" or normalized_observation["exit_code"] == 0,
        "completed execution requires a runner exit code of zero",
    )
    _add_check(
        checks,
        "completed_raw_result_readable",
        status != "completed" or not raw_errors,
        "completed execution requires duplicate-free finite per-case raw JSON",
    )
    _add_check(
        checks,
        "completed_has_per_case_records",
        status != "completed" or (not aggregate_only and bool(records)),
        "completed execution requires per-case records; aggregate-only output is insufficient",
    )
    _add_check(
        checks,
        "completed_normalization_error_free",
        status != "completed" or not all_normalizer_errors,
        "completed execution requires every raw record to normalize without error",
    )
    _add_check(
        checks,
        "completed_coverage_exact",
        status != "completed" or coverage["complete"],
        "completed execution requires exact expected, observed, and mapped case coverage",
    )
    _add_check(
        checks,
        "completed_benchmark_outcome_available",
        status != "completed" or outcome["status"] in {"passed", "failed"},
        "completed execution requires a conclusive benchmark outcome",
    )
    _add_check(
        checks,
        "flight_recorder_import_only",
        True,
        "Flight Recorder must not launch benchmark code while importing results",
    )

    failed_checks = [check for check in checks if not check["passed"]]
    integrity = {
        "passed": not failed_checks,
        "check_count": len(checks),
        "failed_check_count": len(failed_checks),
        "checks": checks,
        "blocking_reasons": [check["summary"] for check in failed_checks],
    }
    claims_allowed = (
        integrity["passed"]
        and status == "completed"
        and coverage["complete"]
        and outcome["status"] == "passed"
        and normalized_observation["side_effects"]["credential_values_recorded"]
        == "not_observed"
    )
    governance_blockers = list(integrity["blocking_reasons"])
    if status != "completed":
        governance_blockers.append("external evaluation execution is not completed")
    if not coverage["complete"]:
        governance_blockers.append("external evaluation case coverage is incomplete")
    if outcome["status"] == "failed":
        governance_blockers.append("external evaluation benchmark outcome failed")
    elif outcome["status"] != "passed":
        governance_blockers.append(
            "external evaluation benchmark outcome is unavailable"
        )
    credential_observation = normalized_observation["side_effects"][
        "credential_values_recorded"
    ]
    if credential_observation == "observed":
        governance_blockers.append("runner reported credential values in recorded evidence")
    elif credential_observation != "not_observed":
        governance_blockers.append(
            "runner did not explicitly attest that credential values were not recorded"
        )
    governance_blockers = _unique(governance_blockers)

    payload = {
        "schema_version": EXTERNAL_EVAL_RESULT_SCHEMA_VERSION,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "artifact_path": _public_basename(Path(out_path))
        if out_path is not None
        else None,
        "integrity": integrity,
        "execution": execution_record,
        "benchmark_outcome": outcome,
        "normalizer": {
            "id": normalizer_id,
            "version": normalizer_version,
            "input_format": raw_format,
            "implementation": "flightrecorder.external_eval_result",
            "implementation_version": "1",
            "aggregate_only": aggregate_only,
            "raw_record_count": len(records),
            "normalized_case_count": len(cases),
            "error_count": len(all_normalizer_errors),
            "errors": all_normalizer_errors,
        },
        "sources": source_refs,
        "identity": identity,
        "coverage": coverage,
        "cases": cases,
        "runner_observation": normalized_observation,
        "governance": {
            "readiness": "ready_for_review" if claims_allowed else "blocked",
            "external_eval_claims_allowed": claims_allowed,
            "recommendation": (
                "review_external_eval_result"
                if claims_allowed
                else "archive_external_eval_failure"
                if integrity["passed"] and status == "failed"
                else "block_external_eval_claims"
            ),
            "blocking_reasons": governance_blockers,
        },
        "execution_boundary": {
            "import_only": True,
            "runner_owns_execution": True,
            "benchmark_started_by_flight_recorder": False,
            "provider_api_called_by_flight_recorder": False,
            "model_downloads_started_by_flight_recorder": False,
            "external_modules_imported_by_flight_recorder": False,
            "raw_prompts_or_outputs_embedded": False,
            "credential_values_recorded_by_flight_recorder": False,
        },
        "notes": [
            "This artifact imports and fingerprints externally produced evaluation evidence; it does not execute a benchmark.",
            "Benchmark outcome is independent from receipt integrity: a completed, valid benchmark may fail, but failed outcomes cannot enable claims.",
            "Aggregate-only results and incomplete case mappings cannot enable external-evaluation claims.",
            "Normalized cases intentionally omit prompts, model outputs, tool arguments, patches, logs, provider URLs, and credentials.",
        ],
    }
    intended_output_path = (
        Path(out_path).resolve(strict=False) if out_path is not None else None
    )
    return _BuiltExternalEvalResult(payload, intended_output_path)


def write_external_eval_result(result: dict[str, Any], out_path: str | Path) -> None:
    """Write a result receipt as deterministic public JSON."""
    path = Path(out_path)
    intended_output_path = getattr(result, "_intended_output_path", None)
    try:
        resolved_output_path = path.resolve(strict=False)
    except OSError as exc:
        raise ExternalEvalResultError(
            "external eval result output path could not be resolved safely"
        ) from exc
    if intended_output_path is None or resolved_output_path != intended_output_path:
        raise ExternalEvalResultError(
            "external eval result must be written to the exact output path used while building it"
        )
    expected_name = result.get("artifact_path")
    if not isinstance(expected_name, str) or expected_name != _public_basename(path):
        raise ExternalEvalResultError(
            "external eval result must be written to the output basename used while building it"
        )
    sources = result.get("sources") if isinstance(result.get("sources"), dict) else {}
    replayable_source_paths: list[Path] = []
    for source in sources.values():
        ref = source.get("path") if isinstance(source, dict) else None
        if (
            isinstance(ref, str)
            and ref
            and not ref.startswith("<redacted:")
            and not Path(ref).is_absolute()
            and ".." not in Path(ref).parts
        ):
            replayable_source_paths.append(path.parent / ref)
    _reject_output_source_collision(path, replayable_source_paths)
    if path_has_symlink_component(path, include_leaf=True):
        raise ExternalEvalResultError(
            f"external eval result output path must not contain symlink components: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def external_eval_result_digest(result: dict[str, Any]) -> str:
    """Recompute the canonical evidence digest for semantic replay."""
    identity = (
        result.get("identity") if isinstance(result.get("identity"), dict) else {}
    )
    identity_base = {
        key: value for key, value in identity.items() if key != "digest_sha256"
    }
    normalizer = (
        result.get("normalizer") if isinstance(result.get("normalizer"), dict) else {}
    )
    return _evidence_digest(
        identity=identity_base,
        execution=result.get("execution"),
        benchmark_outcome=result.get("benchmark_outcome"),
        normalizer={
            "id": normalizer.get("id"),
            "version": normalizer.get("version"),
            "input_format": normalizer.get("input_format"),
        },
        coverage=result.get("coverage"),
        cases=result.get("cases"),
        runner_observation=result.get("runner_observation"),
    )


def _reject_output_source_collision(
    output_path: Path, source_paths: list[Path]
) -> None:
    try:
        resolved_output = output_path.resolve(strict=False)
    except OSError as exc:
        raise ExternalEvalResultError(
            "external eval result output path could not be resolved safely"
        ) from exc
    for source_path in source_paths:
        try:
            resolved_source = source_path.resolve(strict=False)
        except OSError as exc:
            raise ExternalEvalResultError(
                "external eval result source path could not be resolved safely"
            ) from exc
        aliases_source = resolved_output == resolved_source
        if not aliases_source and output_path.exists() and source_path.exists():
            try:
                aliases_source = os.path.samefile(output_path, source_path)
            except OSError:
                aliases_source = False
        if aliases_source:
            raise ExternalEvalResultError(
                "external eval result output must not alias an input source"
            )


def _upstream_semantic_errors(
    plan_path: Path,
    heldout_path: Path,
    captured_plan_sha256: str | None,
    captured_heldout_sha256: str | None,
) -> tuple[list[str], list[str]]:
    # Imported lazily to avoid the validation module's result-builder import cycle.
    from .validation import validate_external_eval_plan, validate_heldout_manifest

    plan_errors: list[str] = []
    heldout_errors: list[str] = []
    try:
        plan_validation = validate_external_eval_plan(plan_path)
        if plan_validation.errors or plan_validation.warnings:
            plan_errors.append("external eval plan failed semantic validation")
    except (OSError, TypeError, ValueError):
        plan_errors.append("external eval plan semantic validation could not complete")
    try:
        heldout_validation = validate_heldout_manifest(heldout_path)
        if heldout_validation.errors or heldout_validation.warnings:
            heldout_errors.append("held-out manifest failed semantic validation")
    except (OSError, TypeError, ValueError):
        heldout_errors.append("held-out manifest semantic validation could not complete")
    if _current_file_sha256(plan_path) != captured_plan_sha256:
        plan_errors.append("external eval plan changed during semantic validation")
    if _current_file_sha256(heldout_path) != captured_heldout_sha256:
        heldout_errors.append("held-out manifest changed during semantic validation")
    return plan_errors, heldout_errors


def _current_file_sha256(path: Path) -> str | None:
    try:
        if path_has_symlink_component(path, include_leaf=True) or not path.is_file():
            return None
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _capture_source(path: Path, base_dir: Path, label: str) -> _SourceSnapshot:
    if path_has_symlink_component(path, include_leaf=True):
        raise ExternalEvalResultError(
            f"{label} source path must not contain symlink components: {path}"
        )
    try:
        before = path.stat()
    except FileNotFoundError:
        return _SourceSnapshot(
            path,
            _display_path(path, base_dir)[0],
            False,
            False,
            False,
            None,
            None,
            None,
            (f"{label} not found",),
        )
    if not path.is_file():
        return _SourceSnapshot(
            path,
            _display_path(path, base_dir)[0],
            True,
            False,
            False,
            None,
            None,
            None,
            (f"{label} is not a regular file",),
        )
    if before.st_size > MAX_SOURCE_BYTES:
        display, replayable = _display_path(path, base_dir)
        return _SourceSnapshot(
            path,
            display,
            True,
            True,
            replayable,
            None,
            before.st_size,
            None,
            (f"{label} exceeds {MAX_SOURCE_BYTES} bytes",),
        )
    try:
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            data = handle.read(MAX_SOURCE_BYTES + 1)
            closed = os.fstat(handle.fileno())
        after = path.stat()
    except OSError:
        display, replayable = _display_path(path, base_dir)
        return _SourceSnapshot(
            path,
            display,
            True,
            True,
            replayable,
            None,
            before.st_size,
            None,
            (f"{label} could not be read",),
        )
    stable = (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
        == (closed.st_dev, closed.st_ino, closed.st_size, closed.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    )
    display, replayable = _display_path(path, base_dir)
    errors: list[str] = []
    if len(data) > MAX_SOURCE_BYTES:
        errors.append(f"{label} exceeds {MAX_SOURCE_BYTES} bytes")
        data = None
    if not stable:
        errors.append(f"{label} changed while it was being imported")
        data = None
    return _SourceSnapshot(
        path,
        display,
        True,
        True,
        replayable,
        hashlib.sha256(data).hexdigest() if data is not None else None,
        len(data) if data is not None else before.st_size,
        data,
        tuple(errors),
    )


def _missing_optional_source() -> _SourceSnapshot:
    return _SourceSnapshot(None, None, False, False, True, None, None, None, ())


def _display_path(path: Path, base_dir: Path) -> tuple[str, bool]:
    try:
        resolved = path.resolve()
        base = base_dir.resolve()
        relative = resolved.relative_to(base)
        if not relative.parts or ".." in relative.parts:
            raise ValueError
        return relative.as_posix(), True
    except (OSError, ValueError):
        return f"<redacted:{_public_basename(path)}>", False


def _source_ready(source: _SourceSnapshot) -> bool:
    return (
        source.exists
        and source.regular_file
        and source.replayable
        and source.sha256 is not None
        and not source.errors
    )


def _source_ref(source: _SourceSnapshot, payload: Any) -> dict[str, Any]:
    schema_version = (
        payload.get("schema_version")
        if isinstance(payload, dict) and isinstance(payload.get("schema_version"), str)
        else None
    )
    return {
        "path": source.display_path,
        "exists": source.exists,
        "regular_file": source.regular_file,
        "replayable": source.replayable,
        "sha256": source.sha256,
        "size_bytes": source.size_bytes,
        "schema_version": schema_version,
    }


def _parse_json_object(
    source: _SourceSnapshot, label: str
) -> tuple[dict[str, Any], list[str]]:
    payload, errors = _parse_json_bytes(source.data, label)
    errors = [*source.errors, *errors]
    if not isinstance(payload, dict):
        if payload is not None:
            errors.append(f"{label} must be a JSON object")
        return {}, errors
    return payload, errors


def _schema_errors(payload: dict[str, Any], schema_name: str) -> list[str]:
    try:
        result = check_schema_contract(payload, name_or_id=schema_name)
    except SchemaRegistryError as exc:
        return [f"schema check unavailable: {exc}"]
    return [str(error) for error in result.get("errors", [])]


def _parse_raw_records(
    source: _SourceSnapshot,
    raw_format: str,
) -> tuple[list[Any], Any, list[str], bool]:
    if raw_format == "jsonl":
        records, errors = _parse_jsonl_bytes(source.data)
        return records, records, [*source.errors, *errors], False
    payload, errors = _parse_json_bytes(source.data, "raw result")
    errors = [*source.errors, *errors]
    if raw_format == "aggregate_json":
        return [], payload, errors, True
    if raw_format == RUN_SUITE_SCHEMA_VERSION:
        if not isinstance(payload, dict):
            errors.append("hfr.run_suite.v1 raw result must be a JSON object")
            return [], payload, errors, False
        if payload.get("schema_version") != RUN_SUITE_SCHEMA_VERSION:
            errors.append(
                "hfr.run_suite.v1 raw result has an unsupported schema_version"
            )
        runs = payload.get("runs")
        if not isinstance(runs, list):
            errors.append("hfr.run_suite.v1 raw result must contain a runs list")
            return [], payload, errors, False
        errors.extend(_run_suite_semantic_errors(payload, runs))
        return runs, payload, errors, False
    if raw_format == "json":
        if isinstance(payload, list):
            return payload, payload, errors, False
        if isinstance(payload, dict):
            candidate_keys = [
                key
                for key in ("records", "samples", "instances")
                if isinstance(payload.get(key), list)
            ]
            if len(candidate_keys) == 1:
                return payload[candidate_keys[0]], payload, errors, False
            if not candidate_keys:
                return [], payload, errors, True
            errors.append(
                "generic JSON raw result contains multiple ambiguous record lists"
            )
            return [], payload, errors, False
        errors.append(
            "generic JSON raw result must be an array or object with one record list"
        )
        return [], payload, errors, False
    errors.append(f"unsupported raw format {raw_format!r}")
    return [], payload, errors, False


def _run_suite_semantic_errors(payload: dict[str, Any], runs: list[Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("total") != len(runs):
        errors.append("hfr.run_suite.v1 total does not match runs length")
    typed_runs = [row for row in runs if isinstance(row, dict)]
    passed_count = sum(row.get("passed") is True for row in typed_runs)
    failed_count = sum(row.get("passed") is False for row in typed_runs)
    if payload.get("passed") != passed_count:
        errors.append("hfr.run_suite.v1 passed count does not match per-case outcomes")
    if payload.get("failed") != failed_count:
        errors.append("hfr.run_suite.v1 failed count does not match per-case outcomes")
    raw_errors = payload.get("errors")
    if not isinstance(raw_errors, list):
        errors.append("hfr.run_suite.v1 errors must be a list")
    elif payload.get("error_count") != len(raw_errors):
        errors.append("hfr.run_suite.v1 error_count does not match errors length")
    elif raw_errors:
        errors.append("hfr.run_suite.v1 contains suite execution errors")
    return errors


def _parse_json_bytes(data: bytes | None, label: str) -> tuple[Any, list[str]]:
    if data is None:
        return None, [f"{label} bytes are unavailable"]
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        return None, [f"{label} is not valid UTF-8: {exc}"]
    try:
        payload = json.loads(
            text, object_pairs_hook=_unique_object, parse_constant=_reject_json_constant
        )
        _check_json_depth(payload)
    except _DuplicateJsonKeyError:
        return None, [f"{label} contains a duplicate JSON key"]
    except json.JSONDecodeError as exc:
        return None, [
            f"{label} is invalid JSON at line {exc.lineno}, column {exc.colno}"
        ]
    except (ValueError, RecursionError):
        return None, [f"{label} contains an unsupported JSON value or nesting"]
    return payload, []


def _parse_jsonl_bytes(data: bytes | None) -> tuple[list[Any], list[str]]:
    if data is None:
        return [], ["raw JSONL bytes are unavailable"]
    records: list[Any] = []
    errors: list[str] = []
    for line_number, raw_line in enumerate(data.splitlines(), start=1):
        if not raw_line.strip():
            continue
        if len(raw_line) > MAX_JSONL_LINE_BYTES:
            errors.append(
                f"raw JSONL line {line_number} exceeds {MAX_JSONL_LINE_BYTES} bytes"
            )
            continue
        payload, row_errors = _parse_json_bytes(
            raw_line, f"raw JSONL line {line_number}"
        )
        errors.extend(row_errors)
        if not row_errors:
            records.append(payload)
    return records, errors


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError("duplicate JSON key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _check_json_depth(value: Any) -> None:
    stack = [(value, 1)]
    while stack:
        current, depth = stack.pop()
        if depth > MAX_JSON_DEPTH:
            raise ValueError(f"JSON nesting exceeds {MAX_JSON_DEPTH}")
        if isinstance(current, dict):
            stack.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            stack.extend((item, depth + 1) for item in current)


def _normalize_records(
    records: list[Any], raw_format: str
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    cases: list[dict[str, Any]] = []
    observed_ids: list[str] = []
    errors: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"raw record {index} must be a JSON object")
            continue
        case_id = _case_id(record)
        if case_id is None:
            errors.append(f"raw record {index} has no public case identifier")
            continue
        observed_ids.append(case_id)
        status = _case_status(record)
        if status is None:
            errors.append(
                f"raw record {index} ({case_id}) has no explicit per-case outcome"
            )
            continue
        score = record.get("score")
        if score is not None and (not _is_finite_number(score)):
            errors.append(f"raw record {index} ({case_id}) has an invalid score")
            continue
        metrics = []
        raw_metrics = record.get("metrics")
        if isinstance(raw_metrics, dict):
            for name, value in sorted(
                raw_metrics.items(), key=lambda item: str(item[0])
            ):
                if not _public_text(name, allow_slash=True, allow_colon=True):
                    errors.append(
                        f"raw record {index} ({case_id}) contains a non-public metric name"
                    )
                    continue
                if not _is_finite_number(value):
                    errors.append(
                        f"raw record {index} ({case_id}) contains a non-finite metric value"
                    )
                    continue
                metrics.append({"name": name, "value": value})
        cases.append(
            {
                "case_id": case_id,
                "status": status,
                "score": score,
                "metrics": metrics,
                "raw_record_index": index,
                "raw_record_sha256": _canonical_sha256(record),
            }
        )
    return cases, observed_ids, errors


def _case_id(record: dict[str, Any]) -> str | None:
    for key in ("scenario_id", "case_id", "instance_id", "sample_id", "task_id", "id"):
        value = record.get(key)
        if (
            isinstance(value, str)
            and value.strip()
            and _public_text(value, allow_slash=True)
        ):
            return value.strip()
    return None


def _case_status(record: dict[str, Any]) -> str | None:
    passed = record.get("passed")
    if isinstance(passed, bool):
        return "passed" if passed else "failed"
    status = record.get("status")
    if not isinstance(status, str):
        return None
    normalized = status.strip().lower()
    mapping = {
        "pass": "passed",
        "passed": "passed",
        "success": "passed",
        "correct": "passed",
        "resolved": "passed",
        "fail": "failed",
        "failed": "failed",
        "incorrect": "failed",
        "unresolved": "failed",
        "error": "error",
        "errored": "error",
        "skipped": "skipped",
    }
    return mapping.get(normalized)


def _heldout_case_ids(heldout: dict[str, Any]) -> tuple[list[str], list[str]]:
    values = heldout.get("scenario_ids")
    if not isinstance(values, list):
        return [], ["held-out manifest scenario_ids must be a list"]
    ids = [
        value.strip()
        for value in values
        if isinstance(value, str)
        and value.strip()
        and _public_text(value, allow_slash=True)
    ]
    errors: list[str] = []
    if len(ids) != len(values):
        errors.append("held-out manifest scenario_ids must contain non-empty strings")
    if len(ids) != len(set(ids)):
        errors.append("held-out manifest scenario_ids must be unique")
    return sorted(set(ids)), errors


def _coverage(
    expected: list[str],
    observed: list[str],
    cases: list[dict[str, Any]],
    raw_record_count: int,
) -> dict[str, Any]:
    observed_set = set(observed)
    mapped = [case["case_id"] for case in cases]
    mapped_set = set(mapped)
    expected_set = set(expected)
    duplicate_ids = sorted(
        {case_id for case_id, count in Counter(observed).items() if count > 1}
        | {case_id for case_id, count in Counter(mapped).items() if count > 1}
    )
    mapped_indexes = {case["raw_record_index"] for case in cases}
    missing = sorted(expected_set - mapped_set)
    unexpected = sorted(observed_set - expected_set)
    unmapped = sorted(observed_set - mapped_set)
    unmapped_indexes = sorted(set(range(raw_record_count)) - mapped_indexes)
    complete = (
        bool(expected)
        and raw_record_count == len(expected)
        and len(observed) == len(expected)
        and len(cases) == len(expected)
        and expected_set == observed_set == mapped_set
        and not duplicate_ids
        and not unmapped_indexes
    )
    return {
        "complete": complete,
        "expected_count": len(expected),
        "observed_count": len(observed),
        "mapped_count": len(cases),
        "raw_record_count": raw_record_count,
        "expected_case_ids": expected,
        "observed_case_ids": sorted(observed_set),
        "mapped_case_ids": sorted(mapped_set),
        "missing_case_ids": missing,
        "unexpected_case_ids": unexpected,
        "unmapped_case_ids": unmapped,
        "duplicate_case_ids": duplicate_ids,
        "unmapped_record_indexes": unmapped_indexes,
    }


def _benchmark_outcome(
    cases: list[dict[str, Any]], unavailable: bool
) -> dict[str, Any]:
    counts = Counter(case["status"] for case in cases)
    scores = [case["score"] for case in cases if _is_finite_number(case.get("score"))]
    if unavailable or not cases:
        status = "not_available"
    elif counts["error"] or counts["skipped"]:
        status = "inconclusive"
    elif counts["failed"]:
        status = "failed"
    else:
        status = "passed"
    mean_score = sum(scores) / len(scores) if scores else None
    return {
        "status": status,
        "summary": _outcome_summary(status, len(cases), counts),
        "case_count": len(cases),
        "passed_case_count": counts["passed"],
        "failed_case_count": counts["failed"],
        "error_case_count": counts["error"],
        "skipped_case_count": counts["skipped"],
        "score_count": len(scores),
        "mean_score": mean_score,
    }


def _outcome_summary(status: str, case_count: int, counts: Counter[str]) -> str:
    if status == "not_available":
        return "No per-case benchmark outcome is available."
    return (
        f"Normalized {case_count} case(s): {counts['passed']} passed, {counts['failed']} failed, "
        f"{counts['error']} errored, and {counts['skipped']} skipped."
    )


def _normalize_runner_observation(value: Any) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    source = value if isinstance(value, dict) else {}
    if not isinstance(value, dict):
        errors.append("runner observation must be an object")
    runner_id = source.get("runner_id")
    if not isinstance(runner_id, str) or not _public_text(runner_id):
        errors.append(
            "runner_observation.runner_id must be a public non-empty identifier"
        )
        runner_id = "unknown"
    runner_version = source.get("runner_version")
    if runner_version is not None and (
        not isinstance(runner_version, str) or not _public_text(runner_version)
    ):
        errors.append(
            "runner_observation.runner_version must be null or a public identifier"
        )
        runner_version = None
    exit_code = source.get("exit_code")
    if exit_code is not None and (
        not isinstance(exit_code, int) or isinstance(exit_code, bool)
    ):
        errors.append("runner_observation.exit_code must be an integer or null")
        exit_code = None
    cost_source = source.get("cost") if isinstance(source.get("cost"), dict) else {}
    reported = cost_source.get("reported") is True
    amount = cost_source.get("amount")
    currency = cost_source.get("currency")
    if reported and (not _is_finite_number(amount) or amount < 0):
        errors.append("reported runner cost must have a non-negative finite amount")
        amount = None
    if not reported:
        amount = None
        currency = None
    elif (
        not isinstance(currency, str)
        or not currency.strip()
        or not _public_text(currency)
    ):
        errors.append("reported runner cost must have a public currency identifier")
        currency = None
    side_source = (
        source.get("side_effects")
        if isinstance(source.get("side_effects"), dict)
        else {}
    )
    side_effects: dict[str, str] = {}
    for field in (
        "network_access",
        "provider_api_calls",
        "model_downloads",
        "filesystem_writes",
        "credential_values_recorded",
    ):
        observed = side_source.get(field, "unknown")
        if observed not in SIDE_EFFECT_STATUSES:
            errors.append(
                f"runner_observation.side_effects.{field} has an unsupported status"
            )
            observed = "unknown"
        side_effects[field] = observed
    started_at = _optional_public_text(
        source.get("started_at"), "runner_observation.started_at", errors
    )
    finished_at = _optional_public_text(
        source.get("finished_at"), "runner_observation.finished_at", errors
    )
    return {
        "runner_id": runner_id,
        "runner_version": runner_version,
        "started_at": started_at,
        "finished_at": finished_at,
        "exit_code": exit_code,
        "cost": {"reported": reported, "amount": amount, "currency": currency},
        "side_effects": side_effects,
    }, errors


def _optional_public_text(value: Any, label: str, errors: list[str]) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _public_text(value, allow_colon=True):
        errors.append(f"{label} must be null or public text")
        return None
    return value


def _plan_ready(plan: dict[str, Any], adapter_id: str) -> bool:
    adapters = plan.get("adapters") if isinstance(plan.get("adapters"), list) else []
    return (
        plan.get("schema_version") == EXTERNAL_EVAL_PLAN_SCHEMA_VERSION
        and plan.get("ready") is True
        and adapter_id in (plan.get("selected_adapters") or [])
        and any(
            isinstance(row, dict)
            and row.get("id") == adapter_id
            and row.get("ready") is True
            for row in adapters
        )
    )


def _heldout_ready(heldout: dict[str, Any]) -> bool:
    scenario_ids = (
        heldout.get("scenario_ids")
        if isinstance(heldout.get("scenario_ids"), list)
        else []
    )
    return (
        heldout.get("schema_version") == HELDOUT_MANIFEST_SCHEMA_VERSION
        and heldout.get("ready") is True
        and bool(scenario_ids)
        and heldout.get("scenario_count") == len(scenario_ids)
        and not heldout.get("blocking_reasons")
    )


def _plan_manifest_sha(plan: dict[str, Any]) -> str | None:
    inputs = plan.get("inputs") if isinstance(plan.get("inputs"), dict) else {}
    manifest = (
        inputs.get("scenario_manifest")
        if isinstance(inputs.get("scenario_manifest"), dict)
        else {}
    )
    sha = manifest.get("sha256")
    return sha if isinstance(sha, str) else None


def _model_matches_plan(plan: dict[str, Any], model_id: str) -> bool:
    inputs = plan.get("inputs") if isinstance(plan.get("inputs"), dict) else {}
    declared_model = inputs.get("model")
    if isinstance(declared_model, str) and declared_model:
        return model_id == declared_model
    legacy_endpoint = inputs.get("model_endpoint")
    return (
        isinstance(legacy_endpoint, str)
        and _public_text(legacy_endpoint, allow_slash=True)
        and model_id == legacy_endpoint
    )


def _metadata_identity_matches(
    metadata: dict[str, Any], adapter_id: str, execution_id: str, model_id: str
) -> bool:
    return (
        metadata.get("adapter_id") == adapter_id
        and metadata.get("execution_id") == execution_id
        and metadata.get("model_id") == model_id
    )


def _identity_is_public(
    adapter_id: str, execution_id: str, model_id: str, normalizer_id: str, version: str
) -> bool:
    return all(
        _public_text(value, allow_slash=label == "model")
        for label, value in (
            ("adapter", adapter_id),
            ("execution", execution_id),
            ("model", model_id),
            ("normalizer", normalizer_id),
            ("version", version),
        )
    )


def _public_text(
    value: Any, *, allow_slash: bool = False, allow_colon: bool = False
) -> bool:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or len(value) > 256
    ):
        return False
    lowered = value.lower()
    if any(
        token in lowered
        for token in ("://", "api_key", "apikey", "secret", "token=", "password")
    ):
        return False
    if any(char in value for char in ("\n", "\r", "\x00", "\\")):
        return False
    if ":/" in value:
        return False
    if not allow_slash and "/" in value:
        return False
    if allow_slash and "/" in value:
        parts = value.split("/")
        if value.startswith(("/", "~/")) or any(
            part in {"", ".", "..", "~"} or part.startswith("~")
            for part in parts
        ):
            return False
    if not allow_colon and ":" in value:
        return False
    return True


def _public_message(value: str) -> bool:
    if not value.strip() or len(value) > 512:
        return False
    lowered = value.lower()
    if any(
        token in lowered
        for token in (
            "://",
            "api_key",
            "apikey",
            "secret",
            "token=",
            "password",
            "/users/",
            "/home/",
            "c:\\",
        )
    ):
        return False
    return not any(char in value for char in ("\n", "\r", "\x00"))


def _public_basename(path: Path) -> str:
    name = path.name
    return name if _public_text(name) else "source"


def _execution_failure_coherent(status: str, failure: str, message: str) -> bool:
    if status not in EXECUTION_STATUSES or failure not in FAILURE_CLASSES:
        return False
    if status == "completed":
        return failure == "none" and not message.strip()
    return failure != "none" and bool(message.strip())


def _add_check(
    checks: list[dict[str, Any]], check_id: str, passed: bool, summary: str
) -> None:
    checks.append({"id": check_id, "passed": bool(passed), "summary": summary})


def _evidence_digest(
    *,
    identity: Any,
    execution: Any,
    benchmark_outcome: Any,
    normalizer: Any,
    coverage: Any,
    cases: Any,
    runner_observation: Any,
) -> str:
    return _canonical_sha256(
        {
            "identity": identity,
            "execution": execution,
            "benchmark_outcome": benchmark_outcome,
            "normalizer": normalizer,
            "coverage": coverage,
            "cases": cases,
            "runner_observation": runner_observation,
        }
    )


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
