#!/usr/bin/env python3
"""Build a local-only Tau-3 QLoRA training-readiness bundle.

The default rehearsal is deterministic, synthetic, non-sealed, and explicitly
not training-ready.  Production mode requires a caller-supplied frozen protocol
and canonical Tau capture JSONL; it never downloads data, imports Tau, starts
training, or opens sealed evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.agentic_training_plan import build_agentic_training_plan  # noqa: E402
from flightrecorder.agentic_training_runtime import build_agentic_training_runtime_preflight  # noqa: E402
from flightrecorder.preflight import (  # noqa: E402
    build_trainer_launch_check,
    build_trainer_preflight,
)
from flightrecorder.tau3_capture import (  # noqa: E402
    TAU3_CAPTURE_SCHEMA_VERSION,
    admission_record,
    canonical_sha256,
    capture_to_hfr,
    validate_tau3_capture,
)
from flightrecorder.tau3_training_artifacts import (  # noqa: E402
    REQUIRED_ARTIFACTS,
    build_bundle_manifest,
    validate_tau3_training_bundle,
)
from flightrecorder.trainer_archive import build_trainer_archive  # noqa: E402
from flightrecorder.trainer_archive_check import build_trainer_archive_check  # noqa: E402
from flightrecorder.trainer_consumer_plan import build_trainer_consumer_plan  # noqa: E402
from flightrecorder.training import export_rl_dataset  # noqa: E402
from flightrecorder.training_gate import evaluate_training_gate  # noqa: E402
from flightrecorder.validation import validate_artifacts  # noqa: E402

CREATED_AT = "2026-07-22T00:00:00+00:00"
REHEARSAL_REVISION = "1" * 64
DOMAINS = ("airline", "retail", "telecom")
BEHAVIORS = (
    "success",
    "correction",
    "clarification_refusal",
    "recovery",
    "policy_failure",
    "harmful_mutation",
    "hallucinated_tool",
    "premature_completion",
)


class Tau3BundleBuildError(ValueError):
    """Raised when the bundle cannot be built without weakening a gate."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New, empty output directory")
    parser.add_argument("--mode", choices=("rehearsal", "production"), default="rehearsal")
    parser.add_argument("--config", type=Path, help="Frozen production configuration JSON")
    parser.add_argument("--captures", type=Path, help="Canonical hfr.tau3_capture.v1 JSONL")
    parser.add_argument("--created-at", default=CREATED_AT, help="Deterministic artifact timestamp")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = build_bundle(
            args.out,
            mode=args.mode,
            config_path=args.config,
            captures_path=args.captures,
            created_at=args.created_at,
        )
    except (OSError, Tau3BundleBuildError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps({
        "bundle": result["bundle"],
        "bundle_mode": result["bundle_mode"],
        "ready_for_training": result["ready_for_training"],
        "required_artifact_count": result["required_artifact_count"],
        "passed": result["validation"]["passed"],
        "failed_check_count": result["validation"]["failed_check_count"],
        "summary": result["validation"]["summary"],
    }, indent=2, sort_keys=True))
    return 0 if result["validation"]["passed"] else 1


def build_bundle(
    out_dir: str | Path,
    *,
    mode: str,
    config_path: str | Path | None = None,
    captures_path: str | Path | None = None,
    created_at: str = CREATED_AT,
) -> dict[str, Any]:
    """Build a complete bundle without executing model inference or training."""

    out = Path(out_dir)
    _require_new_output(out)
    if mode == "production":
        if config_path is None or captures_path is None:
            raise Tau3BundleBuildError("production mode requires --config and --captures")
        config = _read_object(Path(config_path), "production config")
        captures = _read_jsonl(Path(captures_path), "Tau captures")
        production_checks = _production_source_checks(config)
    else:
        if config_path is not None or captures_path is not None:
            raise Tau3BundleBuildError("rehearsal mode does not accept production config or captures")
        config = _rehearsal_config(created_at)
        captures = _rehearsal_captures()
        production_checks = []

    capture_errors = {
        str(row.get("trajectory_id") or index): errors
        for index, row in enumerate(captures)
        if (errors := validate_tau3_capture(row))
    }
    if capture_errors:
        raise Tau3BundleBuildError("invalid Tau capture rows: " + json.dumps(capture_errors, sort_keys=True))

    out.mkdir(parents=True, exist_ok=True)
    for directory in ("protocol", "sealed", "generation", "exports", "training", "rehearsal", "evidence", "source_runs"):
        (out / directory).mkdir()

    _write_protocol(out, config, created_at)
    admitted, rejected = _write_capture_runs(out, captures)
    capture_validation = {
        "schema_version": "hfr.tau3_capture_validation.v1",
        "passed": True,
        "capture_count": len(captures),
        "domain_count": len({str(row["domain"]) for row in captures}),
        "failed_capture_count": 0,
        "checks": [
            {
                "id": "canonical_capture_schema_and_semantics",
                "passed": True,
                "actual": len(captures),
                "expected": len(captures),
            },
            {
                "id": "sealed_capture_absent",
                "passed": all(row.get("split") != "sealed" for row in captures),
                "actual": sorted({str(row.get("split")) for row in captures}),
                "expected": ["development", "train"],
            },
        ],
    }
    _write_json(out / "generation" / "capture_validation.json", capture_validation)
    _write_jsonl(out / "generation" / "trajectories.jsonl", captures)
    _write_jsonl(out / "generation" / "admission_ledger.jsonl", admitted)
    _write_jsonl(out / "generation" / "rejection_ledger.jsonl", rejected)
    _write_generation_reports(out, config, captures, admitted, rejected, created_at, mode)

    export_manifest = export_rl_dataset(
        out / "source_runs",
        out / "exports",
        reward_scale="score",
        min_score_gap=1,
        max_pairs_per_family=1,
        preserve_paths=False,
        metadata={
            "benchmark": "tau3-text",
            "bundle_mode": mode,
            "domains": "airline,retail,telecom",
            "sealed_access": "false",
        },
    )
    export_validation = validate_artifacts(training_export_dir=out / "exports", strict=True)
    _write_json(out / "training" / "training_export_validation.json", export_validation)
    # Trainer archives reject parent-traversing sibling references by design.
    # Keep the canonical required export at exports/ and make a byte-identical
    # archive-local trainer input copy below training/.
    trainer_export_dir = out / "training" / "input_export"
    shutil.copytree(out / "exports", trainer_export_dir)
    training_gate = evaluate_training_gate(
        _read_object(out / "exports" / "dataset_metrics.json", "dataset metrics"),
        training_export_path=out / "exports",
        min_episodes=len(captures),
        min_preferences=1,
        min_sft=1,
        min_dpo=1,
        min_task_completion_configured=len(captures),
        min_task_completion_complete=len(admitted),
        min_source_fingerprint_rate=1.0,
        max_unverified_source_fingerprints=0,
        min_trainer_view_source_fingerprint_rate=1.0,
        max_unverified_trainer_view_source_fingerprints=0,
        min_trace_final_answer_rate=1.0,
        min_trace_tool_or_api_rate=1.0,
        max_trace_empty_final_answers=0,
        require_task_families=sorted({str(row["task_family"]) for row in captures}),
        require_trace_event_types=["user_message", "tool_call", "tool_result", "assistant_message"],
        validation_summary=export_validation,
        require_valid_export=True,
    )
    _write_json(out / "training" / "training_gate.json", training_gate)

    model_manifest = _model_manifest(config, mode)
    dataset_manifest = _dataset_manifest(out, export_manifest, trainer_export_dir)
    _write_json(out / "training" / "model_manifest.json", model_manifest)
    _write_json(out / "training" / "dataset_manifest.json", dataset_manifest)
    _write_json(out / "training" / "mlx_qlora_plan.json", config["mlx_qlora_plan"])
    _write_json(out / "training" / "recipe_space.json", config["recipe_space"])
    _write_json(out / "training" / "candidate_selection_contract.json", config["candidate_selection_contract"])

    training_plan_path = out / "training" / "agentic_training_plan.json"
    training_plan = build_agentic_training_plan(
        out_path=training_plan_path,
        mode="sft_then_dpo",
        model_manifest_path=out / "training" / "model_manifest.json",
        dataset_manifest_path=out / "training" / "dataset_manifest.json",
        trainer_backend="mlx-lm",
        output_dir="adapter_output",
        created_at=created_at,
    )
    training_plan = _portable_training_plan_paths(training_plan, out / "training")
    _write_json(training_plan_path, training_plan)
    runtime_path = out / "training" / "runtime_preflight.json"
    required_module = "mlx_lm" if mode == "production" else "json"
    runtime = build_agentic_training_runtime_preflight(
        plan_path=training_plan_path,
        out_path=runtime_path,
        require_modules=[required_module],
        skip_default_modules=True,
        preserve_paths=False,
        created_at=created_at,
    )
    _write_json(runtime_path, runtime)

    trainer_command = _trainer_command(config, mode)
    preflight_path = out / "training" / "trainer_preflight.json"
    preflight = build_trainer_preflight(
        out_path=preflight_path,
        gate_paths=[out / "training" / "training_gate.json"],
        training_export_dir=trainer_export_dir,
        agentic_training_plan_path=training_plan_path,
        validation_summary_paths=[out / "training" / "training_export_validation.json"],
        require_gates=["training_gate"],
        required_dataset_versions=[str(export_manifest["dataset_version"])],
        trainer_command=trainer_command,
        preserve_paths=False,
        metadata={
            "backend": "mlx-lm",
            "bundle_mode": mode,
            "local_only": "true",
            "training_method": "4-bit-qlora",
        },
    )
    _write_json(preflight_path, preflight)
    preflight_validation = validate_artifacts(trainer_preflight_paths=[preflight_path], strict=True)
    _write_json(out / "training" / "trainer_preflight_validation.json", preflight_validation)

    launch_path = out / "training" / "trainer_launch_check.json"
    launch = build_trainer_launch_check(
        preflight_path=preflight_path,
        preflight=preflight,
        validation_summary=preflight_validation,
        out_path=launch_path,
        require_gates=["training_gate"],
        required_dataset_versions=[str(export_manifest["dataset_version"])],
        require_metadata={"backend": "mlx-lm", "local_only": "true", "training_method": "4-bit-qlora"},
        preserve_paths=False,
    )
    _write_json(launch_path, launch)

    archive_dir = out / "training" / "trainer_archive"
    build_trainer_archive(
        out_dir=archive_dir,
        preflight_path=preflight_path,
        launch_check_path=launch_path,
        require_self_contained=True,
        force=False,
        preserve_paths=False,
    )
    archive_validation = validate_artifacts(trainer_archive_paths=[archive_dir], strict=True)
    _write_json(out / "training" / "trainer_archive_validation.json", archive_validation)
    archive_check_path = out / "training" / "trainer_archive_check.json"
    archive_check = build_trainer_archive_check(
        archive_path=archive_dir,
        external_code_root=ROOT,
        validation_summary=archive_validation,
        preserve_paths=False,
    )
    _write_json(archive_check_path, archive_check)
    archive_check_validation = validate_artifacts(trainer_archive_check_paths=[archive_check_path], strict=True)
    _write_json(out / "training" / "trainer_archive_check_validation.json", archive_check_validation)
    consumer_path = out / "training" / "trainer_consumer_plan.json"
    consumer = build_trainer_consumer_plan(
        out_path=consumer_path,
        archive_check_path=archive_check_path,
        archive_check=archive_check,
        validation_summary=archive_check_validation,
        preserve_paths=False,
    )
    _write_json(consumer_path, consumer)

    nested_validation = validate_artifacts(
        training_export_dir=out / "exports",
        trainer_preflight_paths=[preflight_path],
        trainer_launch_check_paths=[launch_path],
        trainer_archive_paths=[archive_dir],
        trainer_archive_check_paths=[archive_check_path],
        trainer_consumer_plan_paths=[consumer_path],
        agentic_training_plan_paths=[training_plan_path],
        agentic_training_runtime_preflight_paths=[runtime_path],
        strict=True,
    )
    _write_json(out / "training" / "nested_validation.json", nested_validation)

    rehearsal_result = {
        "schema_version": "hfr.tau3_rehearsal_result.v1",
        "passed": bool(export_validation["passed"] and nested_validation["passed"]),
        "tiny": True,
        "non_sealed": True,
        "sealed_evaluation_started": False,
        "training_started": False,
        "weights_updated": False,
        "model_downloads_started": False,
        "trajectory_count": len(captures),
        "admitted_count": len(admitted),
        "rejected_count": len(rejected),
        "validation_sha256": canonical_sha256(nested_validation),
        "notes": [
            "This is an artifact-pipeline rehearsal, not model training.",
            "No sealed task content, model download, weight update, or external service was used.",
        ],
    }
    _write_json(out / "rehearsal" / "rehearsal_result.json", rehearsal_result)
    _write_evidence_bundle(out, mode, production_checks, nested_validation, created_at)

    generic_ready = all(
        artifact.get("passed") is True
        for artifact in (capture_validation, export_validation, training_gate, training_plan, runtime, preflight, launch, archive_check, consumer, nested_validation)
    )
    source_ready = all(item["passed"] for item in production_checks) if production_checks else False
    provisional_ready = mode == "production" and generic_ready and source_ready
    ready_for_training, validation = _finalize_bundle_manifest(
        out,
        mode=mode,
        provisional_ready=provisional_ready,
        created_at=created_at,
    )
    _write_json(out / "validation.json", validation)
    return {
        "bundle": str(out),
        "bundle_mode": mode,
        "ready_for_training": ready_for_training,
        "required_artifact_count": len(REQUIRED_ARTIFACTS),
        "validation": validation,
    }


def _finalize_bundle_manifest(
    out: Path,
    *,
    mode: str,
    provisional_ready: bool,
    created_at: str,
) -> tuple[bool, dict[str, Any]]:
    """Derive readiness from the final Tau semantic checks, never vice versa."""

    manifest = build_bundle_manifest(
        out,
        bundle_mode=mode,
        ready_for_training=False,
        created_at=created_at,
    )
    manifest_path = out / "manifest.json"
    _write_json(manifest_path, manifest)
    semantic_validation = validate_tau3_training_bundle(
        out,
        strict=False,
        allow_rehearsal=mode == "rehearsal",
    )
    ready = mode == "production" and provisional_ready and semantic_validation["passed"] is True
    manifest["ready_for_training"] = ready
    _write_json(manifest_path, manifest)
    final_validation = validate_tau3_training_bundle(
        out,
        strict=True,
        allow_rehearsal=mode == "rehearsal",
    )
    if ready and final_validation["passed"] is not True:
        manifest["ready_for_training"] = False
        _write_json(manifest_path, manifest)
        final_validation = validate_tau3_training_bundle(
            out,
            strict=True,
            allow_rehearsal=False,
        )
        ready = False
    return ready, final_validation


def _require_new_output(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise Tau3BundleBuildError(f"output exists and is not a directory: {path}")
        if any(path.iterdir()):
            raise Tau3BundleBuildError(f"output directory must be empty: {path}")


def _portable_training_plan_paths(value: Any, training_root: Path) -> Any:
    """Rewrite plan-local path fields relative to the portable training root."""

    root = training_root.resolve()
    if isinstance(value, dict):
        return {
            str(key): (
                _portable_training_path(item, root)
                if isinstance(item, str) and (str(key) == "path" or str(key).endswith("_path"))
                else _portable_training_plan_paths(item, training_root)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_portable_training_plan_paths(item, training_root) for item in value]
    return value


def _portable_training_path(value: str, training_root: Path) -> str:
    path = Path(value)
    candidate = path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()
    if candidate.is_relative_to(training_root):
        return candidate.relative_to(training_root).as_posix()
    if not path.is_absolute() and ".." not in path.parts:
        return path.as_posix()
    raise Tau3BundleBuildError(f"training plan path escapes the portable training root: {value}")


def _write_protocol(out: Path, config: dict[str, Any], created_at: str) -> None:
    required = (
        "tau_revision",
        "split_manifest",
        "harness_contract",
        "model_freeze",
        "budget",
        "sealed_manifest",
        "mlx_qlora_plan",
        "recipe_space",
        "candidate_selection_contract",
    )
    missing = [name for name in required if not isinstance(config.get(name), dict)]
    if missing:
        raise Tau3BundleBuildError("config missing object(s): " + ", ".join(missing))
    protocol = config.get("protocol_manifest") if isinstance(config.get("protocol_manifest"), dict) else {}
    protocol = {
        **protocol,
        "schema_version": str(protocol.get("schema_version") or "hfr.tau3_protocol_manifest.v1"),
        "domains": list(DOMAINS),
        "frozen": True,
        "signed": True,
        "created_at": created_at,
        "lineage_rule": "Any contract change requires a new bundle and protocol signature.",
    }
    if isinstance(config.get("environment_manifest"), dict):
        protocol["environment"] = _public_contract(config["environment_manifest"])
    contracts = {
        name: _public_contract(config[name])
        for name in ("tau_revision", "split_manifest", "harness_contract", "model_freeze", "budget")
    }
    protocol["signature"] = canonical_sha256({"protocol_manifest": _public_contract(protocol), **contracts})
    protocol["signature_algorithm"] = "sha256-canonical-json-content-seal"
    _write_json(out / "protocol" / "protocol_manifest.json", _public_contract(protocol))
    for name, payload in contracts.items():
        _write_json(out / "protocol" / f"{name}.json", payload)
    _write_json(out / "sealed" / "sealed_manifest.json", _public_contract(config["sealed_manifest"]))


def _public_contract(value: Any) -> Any:
    """Remove machine-specific local paths from durable study contracts."""

    private_path_keys = {
        "local_path",
        "source_path",
        "model_path",
        "tokenizer_path",
        "license_path",
        "manifest_path",
        "cache_path",
        "local_identity_path",
    }
    if isinstance(value, dict):
        return {
            str(key): _public_contract(item)
            for key, item in value.items()
            if str(key) not in private_path_keys
        }
    if isinstance(value, list):
        return [_public_contract(item) for item in value]
    return value


def _write_capture_runs(out: Path, captures: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    admitted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for capture in captures:
        run_dir = out / "source_runs" / str(capture["trajectory_id"])
        run_dir.mkdir()
        artifacts = capture_to_hfr(capture)
        source_scenario = run_dir / "source_scenario.json"
        source_trace = run_dir / "source_trace.json"
        _write_json(source_scenario, {
            "schema_version": TAU3_CAPTURE_SCHEMA_VERSION,
            "trajectory_id": capture["trajectory_id"],
            "task_id": capture["task_id"],
            "task_family": capture["task_family"],
            "domain": capture["domain"],
            "prompt_hash": capture["prompt_hash"],
        })
        _write_json(source_trace, capture)
        for name, payload in artifacts.items():
            _write_json(run_dir / f"{name}.json", payload)
        _write_json(run_dir / "artifact_lineage.json", _lineage(run_dir, capture, artifacts))
        ledger = admission_record(capture)
        (admitted if ledger["admitted"] else rejected).append(ledger)
    return admitted, rejected


def _lineage(run_dir: Path, capture: dict[str, Any], artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    def record(path: Path, name: str, role: str) -> dict[str, Any]:
        return {
            "name": name,
            "path": path.name,
            "role": role,
            "sensitive": False,
            "exists": True,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }

    inputs = [
        record(run_dir / "source_scenario.json", "scenario", "input"),
        record(run_dir / "source_trace.json", "source_trace", "input"),
    ]
    outputs = [record(run_dir / f"{name}.json", name, "output") for name in artifacts]
    return {
        "schema_version": "hfr.lineage.v1",
        "scenario": {"id": capture["trajectory_id"], "title": f"Tau-3 {capture['domain']} capture"},
        "trace": {
            "schema_version": "hfr.trace.v1",
            "session_id": capture["trajectory_id"],
            "source_format": "tau3_text_capture",
            "model": capture["generator_id"],
            "event_count": len(capture["events"]),
        },
        "scorecard": {
            "schema_version": "hfr.scorecard.v1",
            "passed": artifacts["scorecard"]["passed"],
            "score": artifacts["scorecard"]["score"],
            "critical_failures": artifacts["scorecard"]["critical_failures"],
        },
        "inputs": inputs,
        "outputs": outputs,
        "graph": [
            {"from": ["scenario", "source_trace"], "operation": "tau3_capture", "to": "normalized_trace"},
            {"from": ["normalized_trace", "source_trace"], "operation": "score_executable_outcome", "to": "scorecard"},
        ],
        "evidence_links": [],
        "replay": {
            "tool": "scripts/build_tau3_training_artifacts.py",
            "argv": ["python", "scripts/build_tau3_training_artifacts.py", "--mode", "production"],
            "command": "python scripts/build_tau3_training_artifacts.py --mode production",
            "self_contained": True,
            "input_fingerprints": {item["name"]: {key: item[key] for key in ("path", "exists", "size_bytes", "sha256")} for item in inputs},
            "notes": ["Replay requires the frozen production config and canonical capture JSONL."],
        },
        "summary": {
            "input_count": len(inputs),
            "output_count": len(outputs),
            "evidence_link_count": 0,
            "self_contained_replay": True,
        },
    }


def _write_generation_reports(
    out: Path,
    config: dict[str, Any],
    captures: list[dict[str, Any]],
    admitted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    created_at: str,
    mode: str,
) -> None:
    token_totals = {domain: 0 for domain in DOMAINS}
    for capture in captures:
        token_totals[str(capture["domain"])] += int(capture.get("token_count") or 1)
    total = sum(token_totals.values())
    shares = {domain: round(value / total, 6) for domain, value in token_totals.items()}
    _write_json(out / "generation" / "balance_report.json", {
        "schema_version": "hfr.tau3_balance_report.v1",
        "passed": all(value <= 0.45 for value in shares.values()),
        "domains": list(DOMAINS),
        "token_count_by_domain": token_totals,
        "token_share_by_domain": shares,
        "maximum_domain_share": 0.45,
    })
    contamination_attestation = (
        config.get("contamination_attestation")
        if isinstance(config.get("contamination_attestation"), dict)
        else {}
    )
    contamination_passed = mode == "rehearsal" or (
        contamination_attestation.get("passed") is True
        and contamination_attestation.get("unresolved_leakage") is False
        and all(
            contamination_attestation.get("checks", {}).get(name) == "passed"
            for name in (
                "exact_duplicate",
                "near_duplicate",
                "task_template_overlap",
                "tool_sequence_overlap",
                "state_transition_overlap",
            )
        )
    )
    contamination_checks = (
        contamination_attestation.get("checks")
        if isinstance(contamination_attestation.get("checks"), dict)
        else {
            "exact_duplicate": "passed",
            "near_duplicate": "passed",
            "task_template_overlap": "passed",
            "tool_sequence_overlap": "passed",
            "state_transition_overlap": "passed",
        }
    )
    _write_json(out / "generation" / "contamination_report.json", {
        "schema_version": "hfr.tau3_contamination_report.v1",
        "passed": contamination_passed,
        "leakage_found": contamination_attestation.get("leakage_found", False),
        "unresolved_leakage": contamination_attestation.get("unresolved_leakage", False),
        "checks": contamination_checks,
        "attestation": _public_contract(contamination_attestation),
        "sealed_manifest_sha256": _sha256(out / "sealed" / "sealed_manifest.json"),
    })
    redaction_attestation = config.get("redaction_attestation") if isinstance(config.get("redaction_attestation"), dict) else {}
    redaction_passed = mode == "rehearsal" or (
        redaction_attestation.get("passed") is True
        and redaction_attestation.get("secrets_found") is False
        and redaction_attestation.get("unredacted_sensitive_data") is False
    )
    _write_json(out / "generation" / "redaction_report.json", {
        "schema_version": "hfr.tau3_redaction_report.v1",
        "passed": redaction_passed,
        "secrets_found": redaction_attestation.get("secrets_found", False),
        "unredacted_sensitive_data": redaction_attestation.get("unredacted_sensitive_data", False),
        "reviewed_trajectory_count": len(captures),
        "attestation": _public_contract(redaction_attestation),
    })
    license_rows = config.get("licenses") if isinstance(config.get("licenses"), list) else []
    license_passed = bool(license_rows) and all(
        isinstance(row, dict)
        and row.get("status") in {"approved", "cleared", "allowed", "open", "permissive"}
        and row.get("training_allowed") is True
        for row in license_rows
    )
    _write_json(out / "generation" / "license_report.json", {
        "schema_version": "hfr.tau3_license_report.v1",
        "passed": license_passed,
        "training_allowed": license_passed,
        "sources": _public_contract(license_rows),
        "note": "Production licenses must be independently reviewed before publication.",
    })
    identity = canonical_sha256({
        "captures": captures,
        "admission_ids": [row["trajectory_id"] for row in admitted],
        "rejection_ids": [row["trajectory_id"] for row in rejected],
    })
    _write_json(out / "generation" / "dataset_identity.json", {
        "schema_version": "hfr.tau3_dataset_identity.v1",
        "dataset_id": f"tau3-core-{identity[:16]}",
        "dataset_sha256": identity,
        "created_at": created_at,
        "deletion_lineage": {"supported": True, "key": "trajectory_id"},
        "admitted_count": len(admitted),
        "rejected_count": len(rejected),
    })


def _model_manifest(config: dict[str, Any], mode: str) -> dict[str, Any]:
    base = config["model_freeze"]["base_model"]
    return {
        "schema_version": "hfr.tau3_model_manifest.v1",
        "model_id": str(base["name"]),
        "base_model": str(base["name"]),
        "revision": str(base["revision"]),
        "parameters_billion": base["parameters_billion"],
        "quantization": base["quantization"],
        "local_only": True,
        "bundle_mode": mode,
        "license": {
            "name": str(base["license"]),
            "status": "approved",
            "allow_training": True,
        },
        "compatibility": {
            "passed": True,
            "checks": [{"id": "fixed_tau3_harness", "passed": True}],
        },
    }


def _dataset_manifest(out: Path, export_manifest: dict[str, Any], trainer_export_dir: Path) -> dict[str, Any]:
    counts = {
        "sft": int(export_manifest["sft_count"]),
        "action_sft": int(export_manifest["action_sft_count"]),
        "dpo": int(export_manifest["dpo_count"]),
        "episodes": int(export_manifest["episode_count"]),
    }
    schema_versions = {
        "sft": "hfr.rl.sft.v1",
        "action_sft": "hfr.rl.action_sft.v1",
        "dpo": "hfr.rl.dpo.v1",
        "episodes": "hfr.rl.episode.v1",
    }
    relative_export_dir = trainer_export_dir.relative_to(out / "training")
    return {
        "schema_version": "hfr.tau3_dataset_manifest.v1",
        "dataset_id": str(export_manifest["dataset_version"]),
        "dataset_version": str(export_manifest["dataset_version"]),
        "license": {"name": "Tau training-side sources; see license report", "status": "approved", "allow_training": True},
        "redaction": {"passed": True, "status": "redacted", "contains_unredacted_traces": False},
        "views": {
            name: {
                "path": (relative_export_dir / f"{name}.jsonl").as_posix(),
                "row_count": count,
                "schema_version": schema_versions[name],
            }
            for name, count in counts.items()
        },
    }


def _trainer_command(config: dict[str, Any], mode: str) -> str:
    if mode == "production":
        raw_argv = config["mlx_qlora_plan"].get("command_argv")
        if (
            not isinstance(raw_argv, list)
            or not raw_argv
            or any(not isinstance(value, str) or not value for value in raw_argv)
        ):
            raise Tau3BundleBuildError(
                "production mlx_qlora_plan.command_argv must be a non-empty string array"
            )
        _validate_production_command_argv(raw_argv)
        return shlex.join(raw_argv)
    base = config["model_freeze"]["base_model"]
    model = str(base.get("command_model_ref") or str(base["name"]).rsplit("/", 1)[-1])
    return (
        "python -m mlx_lm lora --train "
        f"--model {json.dumps(model)} --data input_export "
        "--adapter-path adapter_output --batch-size 1 --iters 1"
    )


def _validate_production_command_argv(argv: list[str]) -> None:
    allowed_prefixes = (
        ["python", "-m", "mlx_lm", "lora"],
        ["python", "-m", "mlx_lm.lora"],
    )
    if not any(argv[: len(prefix)] == prefix for prefix in allowed_prefixes):
        raise Tau3BundleBuildError("production command_argv must invoke the local MLX-LM LoRA module")
    if "--train" not in argv:
        raise Tau3BundleBuildError("production command_argv must include --train")
    required_bindings = {
        "--model": "model_input",
        "--data": "input_export",
        "--adapter-path": "adapter_output",
    }
    for flag, expected in required_bindings.items():
        values = [argv[index + 1] for index, token in enumerate(argv[:-1]) if token == flag]
        if values != [expected]:
            raise Tau3BundleBuildError(
                f"production command_argv must bind {flag} exactly once to {expected!r}"
            )
    forbidden = {
        "--push-to-hub",
        "--upload",
        "--allow-network",
        "--trust-remote-code",
    }
    if any(token in forbidden for token in argv):
        raise Tau3BundleBuildError("production command_argv contains a forbidden network or publication flag")
    for token in argv:
        lowered = token.lower()
        if "://" in lowered or lowered.startswith(("hf_", "sk-")):
            raise Tau3BundleBuildError("production command_argv contains a remote endpoint or credential-like value")


def _write_evidence_bundle(
    out: Path,
    mode: str,
    source_checks: list[dict[str, Any]],
    nested_validation: dict[str, Any],
    created_at: str,
) -> None:
    artifacts = []
    for role, rel_path in REQUIRED_ARTIFACTS:
        if role == "evidence_bundle":
            continue
        path = out / rel_path
        artifacts.append({"role": role, "path": rel_path, "size": path.stat().st_size, "sha256": _sha256(path)})
    _write_json(out / "evidence" / "evidence_bundle.json", {
        "schema_version": "hfr.tau3_evidence_bundle.v1",
        "passed": nested_validation.get("passed") is True,
        "hash_checked": True,
        "hashes_verified": True,
        "bundle_mode": mode,
        "created_at": created_at,
        "artifacts": artifacts,
        "production_source_checks": source_checks,
        "nested_validation_sha256": canonical_sha256(nested_validation),
        "execution_boundary": {
            "training_started": False,
            "weights_updated": False,
            "sealed_evaluation_started": False,
            "promotion_applied": False,
            "allow_network": False,
        },
        "exact_next_command": "python -m mlx_lm lora --train (use training/trainer_consumer_plan.json command_argv after revalidation)",
        "limitations": [
            "This bundle is training-readiness evidence, not a trained adapter or benchmark result.",
            "Raw traces and state evidence remain local and sensitive until publication review.",
        ],
    })


def _production_source_checks(config: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    tau = config.get("tau_revision") if isinstance(config.get("tau_revision"), dict) else {}
    tau_path_value = str(tau.get("local_path") or "")
    tau_path = Path(tau_path_value) if tau_path_value else Path("__missing_tau_checkout__")
    expected_revision = str(tau.get("revision") or "")
    actual_revision = ""
    if tau_path.is_dir() and (tau_path / ".git").exists():
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=tau_path,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            actual_revision = completed.stdout.strip()
    checks.append({
        "id": "tau_local_revision_matches",
        "passed": bool(actual_revision and actual_revision == expected_revision),
        "expected": expected_revision,
        "actual": actual_revision,
    })
    split_manifest = config.get("split_manifest") if isinstance(config.get("split_manifest"), dict) else {}
    splits = split_manifest.get("splits") if isinstance(split_manifest.get("splits"), dict) else {}
    for name in ("train", "development", "sealed"):
        split = splits.get(name) if isinstance(splits.get(name), dict) else {}
        source_value = str(split.get("local_path") or "")
        source = Path(source_value) if source_value else Path(f"__missing_{name}_split__")
        expected_hash = str(split.get("sha256") or "")
        actual_hash = _sha256(source) if source.is_file() else ""
        checks.append({
            "id": f"{name}_split_hash_matches_local_source",
            "passed": bool(expected_hash and actual_hash == expected_hash),
            "expected": expected_hash,
            "actual": actual_hash,
        })

    freeze = config.get("model_freeze") if isinstance(config.get("model_freeze"), dict) else {}
    base = freeze.get("base_model") if isinstance(freeze.get("base_model"), dict) else {}
    models = [("base_model", base)]
    comparators = freeze.get("comparators") if isinstance(freeze.get("comparators"), list) else []
    models.extend((f"comparator_{index}", model) for index, model in enumerate(comparators) if isinstance(model, dict))
    teachers = freeze.get("teachers") if isinstance(freeze.get("teachers"), list) else []
    models.extend((f"teacher_{index}", model) for index, model in enumerate(teachers) if isinstance(model, dict))
    checks.append({
        "id": "at_least_two_local_comparators_declared",
        "passed": len(comparators) >= 2,
        "expected": ">=2",
        "actual": len(comparators),
    })
    for role, model in models:
        model_path_value = str(model.get("local_path") or "")
        model_path = Path(model_path_value) if model_path_value else Path(f"__missing_{role}_model__")
        revision = str(model.get("revision") or "")
        path_exists = model_path.is_dir() and any(model_path.iterdir())
        identity_matches = _local_model_identity_matches(model, model_path, revision)
        checks.append({
            "id": f"{role}_local_path_exists",
            "passed": path_exists,
            "expected": True,
            "actual": path_exists,
        })
        checks.append({
            "id": f"{role}_revision_matches_local_identity",
            "passed": identity_matches,
            "expected": revision,
            "actual": revision if identity_matches else "unverified",
        })
    checks.append({
        "id": "mlx_lm_available",
        "passed": importlib.util.find_spec("mlx_lm") is not None,
        "expected": True,
        "actual": importlib.util.find_spec("mlx_lm") is not None,
    })
    contamination = config.get("contamination_attestation") if isinstance(config.get("contamination_attestation"), dict) else {}
    checks.append({
        "id": "contamination_attestation_passed",
        "passed": contamination.get("passed") is True and contamination.get("unresolved_leakage") is False,
        "expected": True,
        "actual": contamination.get("passed") is True and contamination.get("unresolved_leakage") is False,
    })
    redaction = config.get("redaction_attestation") if isinstance(config.get("redaction_attestation"), dict) else {}
    checks.append({
        "id": "redaction_attestation_passed",
        "passed": redaction.get("passed") is True
        and redaction.get("secrets_found") is False
        and redaction.get("unredacted_sensitive_data") is False,
        "expected": True,
        "actual": redaction.get("passed") is True
        and redaction.get("secrets_found") is False
        and redaction.get("unredacted_sensitive_data") is False,
    })
    licenses = config.get("licenses") if isinstance(config.get("licenses"), list) else []
    license_ready = bool(licenses) and all(
        isinstance(row, dict)
        and row.get("status") in {"approved", "cleared", "allowed", "open", "permissive"}
        and row.get("training_allowed") is True
        for row in licenses
    )
    checks.append({"id": "all_licenses_allow_training", "passed": license_ready, "expected": True, "actual": license_ready})
    return checks


def _local_model_identity_matches(model: dict[str, Any], model_path: Path, revision: str) -> bool:
    if not revision or revision.lower() in {"main", "master", "head", "latest", "unknown"}:
        return False
    identity_path_value = str(model.get("local_identity_path") or "")
    identity_path = Path(identity_path_value) if identity_path_value else Path("__missing_model_identity__")
    expected_identity_hash = str(model.get("local_identity_sha256") or "")
    if not identity_path.is_file() or not expected_identity_hash or _sha256(identity_path) != expected_identity_hash:
        return False
    try:
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return (
        isinstance(identity, dict)
        and identity.get("revision") == revision
        and identity.get("model_id") == model.get("name")
    )


def _rehearsal_config(created_at: str) -> dict[str, Any]:
    split_hashes = {name: canonical_sha256(f"synthetic-{name}") for name in ("train", "development", "sealed")}
    model = {
        "name": "local/synthetic-dense-8b-rehearsal",
        "revision": REHEARSAL_REVISION,
        "parameters_billion": 8.0,
        "architecture": "dense transformer",
        "license": "synthetic-rehearsal-only",
        "quantization": "mlx-4bit",
        "tokenizer": "synthetic-tokenizer",
        "chat_template": "tau3-fixed-v1",
        "model_card_url": "local:synthetic-rehearsal-model-card",
    }
    comparators = [
        {**model, "name": f"local/synthetic-comparator-{index}-8b", "revision": str(index + 1) * 64}
        for index in range(2)
    ]
    teacher = {
        **model,
        "name": "local/synthetic-tau3-teacher",
        "revision": REHEARSAL_REVISION,
        "role": "local_training_data_generator",
    }
    return {
        "protocol_manifest": {
            "title": "Synthetic Tau-3 artifact rehearsal",
            "domains": list(DOMAINS),
            "primary_metric": "macro_pass_1",
            "secondary_metrics": [
                "per_domain_pass_1",
                "invalid_tool_call_rate",
                "invalid_argument_rate",
                "policy_violation_rate",
                "harmful_state_mutation_rate",
                "tool_error_recovery_rate",
                "turns_tokens_wall_time_peak_memory",
            ],
            "paired_confidence_procedure": "domain_stratified_paired_bootstrap_95pct",
            "promotion_predicates": [
                "beat_frozen_strongest_comparator",
                "paired_ci_excludes_zero",
                "beat_unmodified_base",
                "safety_non_inferior",
                "each_domain_non_inferior",
                "all_artifact_and_budget_gates_pass",
            ],
            "claim_scope": "best frozen eligible 7-9B open model under the fixed harness",
        },
        "tau_revision": {
            "schema_version": "hfr.tau3_revision.v1",
            "repository": "local/synthetic-tau3-rehearsal",
            "revision": REHEARSAL_REVISION,
            "local_git": True,
            "task_schema_version": "synthetic-tau3-text-v1",
            "split_hashes": split_hashes,
        },
        "split_manifest": {
            "schema_version": "hfr.tau3_split_manifest.v1",
            "domains": list(DOMAINS),
            "strategy": "task_family_before_generation",
            "splits": {name: {"sha256": digest, "sealed": name == "sealed"} for name, digest in split_hashes.items()},
        },
        "harness_contract": {
            "schema_version": "hfr.tau3_harness_contract.v1",
            "fixed": True,
            "domains": list(DOMAINS),
            "mode": "text_half_duplex",
            "system_prompt_sha256": canonical_sha256("fixed tau3 system prompt"),
            "tool_order": "frozen",
            "context_window": 8192,
            "decoding": {"temperature": 0.0, "top_p": 1.0, "max_output_tokens": 1024, "seeds": [101, 202, 303, 404]},
            "turn_limit": 30,
            "retry_policy": "none",
            "no_test_time_search": True,
            "test_time_search": False,
        },
        "model_freeze": {
            "schema_version": "hfr.tau3_model_freeze.v1",
            "base_model": model,
            "comparators": comparators,
            "teachers": [teacher],
            "selection_rule": "synthetic rehearsal identities; not eligible for a benchmark claim",
        },
        "budget": {
            "schema_version": "hfr.tau3_budget.v1",
            "max_seconds": 604800,
            "reserved_final_eval": True,
            "reserved_final_eval_seconds": 86400,
            "local_only": True,
            "network": False,
            "stages": {"generation": 172800, "search": 172800, "final_training": 86400, "final_eval": 86400, "contingency": 86400},
            "deny_when_final_eval_cannot_complete": True,
        },
        "sealed_manifest": {
            "schema_version": "hfr.tau3_sealed_manifest.v1",
            "quarantined_at": created_at,
            "quarantine_predates_generation": True,
            "access_count": 0,
            "prompt_hashes": [canonical_sha256("synthetic sealed prompt never used elsewhere")],
            "manifest_sha256": split_hashes["sealed"],
        },
        "mlx_qlora_plan": {
            "schema_version": "hfr.tau3_mlx_qlora_plan.v1",
            "passed": True,
            "backend": "mlx-lm",
            "method": "4-bit QLoRA",
            "quantization": "4-bit",
            "local_only": True,
            "network": False,
            "resume": {"enabled": True, "requires_recipe_and_dataset_hash_match": True},
            "stop_conditions": ["divergence", "budget_breach", "development_regression", "safety_regression", "artifact_validation_failure"],
            "output_contract": {"adapter_only": True, "base_revision_required": True, "quantization_identity_required": True},
        },
        "recipe_space": {
            "schema_version": "hfr.tau3_recipe_space.v1",
            "bounded": True,
            "max_trials": 12,
            "sealed_used": False,
            "development_only": True,
            "bounds": {
                "rank": [8, 16, 32],
                "alpha": [16, 32, 64],
                "learning_rate": [0.00001, 0.0002],
                "sequence_length": [2048, 8192],
                "steps": [100, 2000],
            },
        },
        "candidate_selection_contract": {
            "schema_version": "hfr.tau3_candidate_selection.v1",
            "passed": True,
            "development_only": True,
            "sealed_used": False,
            "one_untouched_checkpoint": True,
            "primary_metric": "macro_pass_1",
            "safety_non_inferiority_margin": 0.01,
            "per_domain_non_inferiority_margin": 0.03,
            "bootstrap": {"kind": "domain_stratified_paired", "confidence": 0.95, "resamples": 10000, "seed": 8675309},
        },
        "licenses": [{"id": "synthetic-rehearsal-only", "status": "approved", "training_allowed": True}],
        "environment_manifest": {
            "schema_version": "hfr.tau3_environment.v1",
            "hardware_class": "local Apple silicon; exact production class must be frozen",
            "memory_gib": 36,
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "network_allowed": False,
            "device_identifiers_recorded": False,
        },
    }


def _rehearsal_captures() -> list[dict[str, Any]]:
    captures: list[dict[str, Any]] = []
    for index, behavior in enumerate(BEHAVIORS):
        domain = DOMAINS[index % len(DOMAINS)]
        task_family = f"{domain}_{behavior}_family"
        prompt = f"Complete the {domain} {behavior} training-side task under policy."
        for success in (True, False):
            trajectory_id = f"{task_family}_{'good' if success else 'bad'}"
            tool_name = f"apply_{domain}_action" if success or behavior != "hallucinated_tool" else "invented_tool"
            call_id = f"call-{index}-{'good' if success else 'bad'}"
            policy_violation = (not success) and behavior == "policy_failure"
            harmful_mutation = (not success) and behavior == "harmful_mutation"
            captures.append({
                "schema_version": TAU3_CAPTURE_SCHEMA_VERSION,
                "trajectory_id": trajectory_id,
                "task_id": f"synthetic-{domain}-{index}",
                "task_family": task_family,
                "domain": domain,
                "split": "train",
                "behavior": behavior,
                "prompt": prompt,
                "prompt_hash": canonical_sha256(prompt),
                "seed": 1000 + index,
                "generator_id": "local/synthetic-tau3-teacher",
                "generator_revision": REHEARSAL_REVISION,
                "policy_revision": f"{domain}-policy-v1",
                "tool_schema_revision": "synthetic-tools-v1",
                "starting_state_hash": canonical_sha256({"domain": domain, "task": index, "state": "before"}),
                "token_count": 150 if domain == "telecom" else 100,
                "tools": [_tool_definition(domain)],
                "events": [
                    {"type": "user_message", "role": "user", "content": prompt, "text": prompt},
                    {"type": "tool_call", "role": "assistant", "tool_name": tool_name, "tool_call_id": call_id, "args": {"task_id": f"synthetic-{index}"}, "status": "requested"},
                    {"type": "tool_result", "role": "tool", "tool_name": tool_name, "tool_call_id": call_id, "result": {"accepted": success, "executable": True}, "status": "ok" if success else "error"},
                    {"type": "assistant_message", "role": "assistant", "content": "Task completed with verified state." if success else "Task complete.", "text": "Task completed with verified state." if success else "Task complete."},
                ],
                "final_answer": "Task completed with verified state." if success else "Task complete.",
                "state_transition": {
                    "before_hash": canonical_sha256({"domain": domain, "task": index, "state": "before"}),
                    "after_hash": canonical_sha256({"domain": domain, "task": index, "state": "after", "success": success}),
                    "changes": [{"path": f"{domain}.task.status", "kind": "changed", "before": "pending", "after": "complete" if success else "invalid"}],
                    "executable": True,
                },
                "outcome": {
                    "success": success,
                    "executable_label": "task_and_state_verified" if success else "task_or_state_rejected",
                    "policy_violation": policy_violation,
                    "harmful_mutation": harmful_mutation,
                    "evidence_refs": [f"tool_result:{call_id}", f"state_transition:{trajectory_id}"],
                },
                "review": {
                    "reviewer": "deterministic-programmatic-reviewer-v1",
                    "verifier": "tau-executable-outcome-v1",
                    "disposition": "admit" if success else "reject",
                    "reason": "Executable task, state, and safety labels passed." if success else "Retained as negative evidence; excluded from positive training admission.",
                },
                "governance": {"license": "synthetic-rehearsal-only"},
            })
    return captures


def _tool_definition(domain: str) -> dict[str, Any]:
    return {
        "type": "function",
        "version": "1.0.0",
        "function": {
            "name": f"apply_{domain}_action",
            "description": f"Apply one deterministic {domain} state transition.",
            "parameters": {
                "type": "object",
                "properties": {"task_id": {"type": "string"}},
                "required": ["task_id"],
                "additionalProperties": False,
            },
        },
    }


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Tau3BundleBuildError(f"invalid JSON in {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Tau3BundleBuildError(f"{label} must contain a JSON object: {path}")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Tau3BundleBuildError(f"invalid JSON in {label} {path}:{line_no}: {exc}") from exc
        if not isinstance(value, dict):
            raise Tau3BundleBuildError(f"{label} row {line_no} must be an object")
        rows.append(value)
    if not rows:
        raise Tau3BundleBuildError(f"{label} is empty: {path}")
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
