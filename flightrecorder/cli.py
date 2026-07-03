"""Command line interface for Hermes Flight Recorder."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shlex
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .action_ledger import ActionLedgerError, build_action_ledger
from .action_gate import (
    ACTION_LEDGER_GATE_POLICY_SCHEMA_VERSION,
    ActionLedgerGateError,
    ActionLedgerGatePolicyError,
    evaluate_action_ledger_gate,
    load_action_ledger_gate_policy,
)
from .adapters import AdapterError, normalize_trace
from .agentic_training_loop_plan import (
    AgenticTrainingLoopPlanError,
    build_agentic_training_loop_plan,
    write_agentic_training_loop_plan,
)
from .artifacts import (
    ArtifactError,
    build_suite_trend,
    compare_scorecards,
    compare_suites,
    write_compare_report,
    write_junit,
    write_markdown_summary,
    write_suite_compare_report,
    write_suite_trend_report,
)
from .bundle import (
    HARNESS_RUN_MANIFEST_SCHEMA_VERSION,
    HARNESS_RUN_RESULT_SCHEMA_VERSION,
    EvidenceBundleError,
    build_evidence_bundle,
)
from .calibration import ReviewCalibrationError, build_review_calibration
from .compare_gate import (
    COMPARE_GATE_POLICY_SCHEMA_VERSION,
    CompareGatePolicyError,
    evaluate_compare_gate,
    load_compare_gate_policy,
)
from .decision_gate import DecisionGateError, evaluate_decision_gate
from .digest import RunDigestError, build_run_digest, render_run_digest_markdown
from .evidence import EvidenceCoverageError, build_evidence_coverage
from .governance import (
    PromotionDecisionError,
    apply_promotion_aliases,
    build_promotion_cards,
    build_promotion_decision,
    build_promotion_release_record,
    build_promotion_rollback_receipt,
)
from .improvement_gate import (
    IMPROVEMENT_LEDGER_GATE_POLICY_SCHEMA_VERSION,
    ImprovementLedgerGateError,
    ImprovementLedgerGatePolicyError,
    evaluate_improvement_ledger_gate,
    load_improvement_ledger_gate_policy,
)
from .improvement_ledger import ImprovementLedgerError, build_improvement_ledger
from .improvement_plan import ImprovementPlanError, build_improvement_plan
from .eval_summary import EvalSummaryError, build_eval_summary, render_eval_summary_markdown
from .external_eval import ExternalEvalPlanError, adapter_choices, build_external_eval_plan, write_external_eval_plan
from .heldout_manifest import HeldoutManifestError, build_heldout_manifest, write_heldout_manifest
from .lineage import REPLAY_BUNDLE_SCHEMA_VERSION, write_run_lineage
from .model_registry import (
    ALIAS_NAMES,
    MODEL_ADAPTER_MANIFEST_STATUSES,
    MODEL_REGISTRY_LINK_COLLECTIONS,
    ModelRegistryError,
    build_model_adapter_manifest,
    build_dry_run_training_plan,
    build_model_compatibility_report,
    build_model_serving_probe_receipt,
    link_model_registry_artifact,
    list_model_registry_entries,
    load_model_registry,
    model_candidate_errors,
    move_model_alias,
    register_model_candidate,
)
from .redaction import sanitize_trace
from .preflight import TrainerPreflightError, build_trainer_launch_check, build_trainer_preflight
from .promotion_archive import PromotionArchiveError, build_promotion_archive
from .promotion_gate import (
    PROMOTION_LEDGER_GATE_POLICY_SCHEMA_VERSION,
    PromotionLedgerGateError,
    PromotionLedgerGatePolicyError,
    evaluate_promotion_ledger_gate,
    load_promotion_ledger_gate_policy,
)
from .promotion_ledger import PromotionLedgerError, build_promotion_ledger
from .repair import RepairQueueError, build_repair_queue
from .report import write_index, write_report
from .review import REVIEW_LABELS, ReviewExportError, apply_review_labels, export_review_queue
from .reviewed_gate import (
    REVIEWED_GATE_POLICY_SCHEMA_VERSION,
    ReviewedGatePolicyError,
    evaluate_reviewed_gate,
    load_reviewed_gate_policy,
)
from .schema import ScenarioError, load_scenario, resolve_trace_path
from .schema_registry import (
    SchemaRegistryError,
    check_schema_file,
    check_schema_jsonl_file,
    list_schema_records,
    load_schema,
    write_schema_bundle,
)
from .scenario_check import check_scenarios, discover_scenarios
from .scenario_draft import draft_scenario, safe_scenario_id, score_draft, title_from_id
from .scenario_quality import build_scenario_quality
from .scorers import score_trace
from .state import (
    StateSnapshotError,
    load_state_snapshot,
    resolve_before_state_snapshot_path,
    resolve_state_snapshot_path,
    sanitize_state_snapshot,
)
from .state_capture import StateCaptureError, capture_state_snapshot
from .state_diff import StateDiffError, build_state_diff
from .state_validators import (
    StateValidatorError,
    build_monitor_catalog,
    build_state_validator_assertions,
    render_monitor_catalog_markdown,
)
from .suite_gate import SUITE_GATE_POLICY_SCHEMA_VERSION, SuiteGatePolicyError, evaluate_suite_gate, load_gate_policy
from .trace_observability import TraceObservabilityError, build_trace_observability
from .trainer_archive import TrainerArchiveError, build_trainer_archive
from .trainer_archive_check import TrainerArchiveCheckError, build_trainer_archive_check
from .trainer_consumer_plan import TrainerConsumerPlanError, build_trainer_consumer_plan
from .training import TrainingExportError, export_compare_rl_dataset, export_rl_dataset
from .training_gate import (
    TRAINING_GATE_POLICY_SCHEMA_VERSION,
    TrainingGatePolicyError,
    evaluate_training_gate,
    load_training_gate_policy,
)
from .validation import EVAL_SUITE_MANIFEST_SCHEMA_VERSION, validate_artifacts
from .verifiers import VerifierError, capture_verified_state

RUN_SUITE_SCHEMA_VERSION = "hfr.run_suite.v1"
GOAL3_HANDOFF_SCHEMA_VERSION = "hfr.goal3_handoff.v1"
FAMILY_SUFFIX_RE = re.compile(r"([_-](good|bad|pass|fail|passing|failing|chosen|rejected))+$", re.IGNORECASE)
TRACE_FORMAT_CHOICES = [
    "auto",
    "trajectory_jsonl",
    "observer_jsonl",
    "openclaw_jsonl",
    "coven_jsonl",
    "atof_jsonl",
    "atif_json",
    "normalized_json",
]


class ReplayError(ValueError):
    """Raised when a lineage replay contract cannot be safely rerun."""


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (
        AdapterError,
        ArtifactError,
        ScenarioError,
        StateCaptureError,
        StateValidatorError,
        VerifierError,
        StateDiffError,
        StateSnapshotError,
        SuiteGatePolicyError,
        ReviewExportError,
        ReviewedGatePolicyError,
        RepairQueueError,
        TrainerPreflightError,
        TrainerArchiveError,
        TrainerArchiveCheckError,
        TrainerConsumerPlanError,
        TrainingExportError,
        TrainingGatePolicyError,
        CompareGatePolicyError,
        DecisionGateError,
        RunDigestError,
        EvidenceCoverageError,
        EvidenceBundleError,

        PromotionDecisionError,
        EvalSummaryError,
        ExternalEvalPlanError,
        HeldoutManifestError,

        ReviewCalibrationError,
        TraceObservabilityError,
        ActionLedgerError,
        ActionLedgerGateError,
        ActionLedgerGatePolicyError,
        ImprovementLedgerGateError,
        ImprovementLedgerGatePolicyError,
        ImprovementLedgerError,
        ImprovementPlanError,
        PromotionLedgerGateError,
        PromotionLedgerGatePolicyError,
        PromotionLedgerError,
        PromotionArchiveError,
        AgenticTrainingLoopPlanError,
        ReplayError,
        SchemaRegistryError,
        ModelRegistryError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        parser.exit(2, f"flightrecorder: error: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "flightrecorder: interrupted\n")
    return 0


def cmd_normalize(args: argparse.Namespace) -> int:
    trace = normalize_trace(args.trace, args.format)
    if not args.no_redact:
        trace = sanitize_trace(trace, args.secret_pattern)
    _write_json(Path(args.out), trace)
    print(f"wrote {args.out}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario)
    trace = _read_json(Path(args.trace))
    state_path = resolve_state_snapshot_path(scenario, args.state)
    before_state_path = resolve_before_state_snapshot_path(scenario, args.before_state)
    state_snapshot = load_state_snapshot(state_path) if state_path is not None else None
    before_state_snapshot = load_state_snapshot(before_state_path) if before_state_path is not None else None
    scorecard = score_trace(scenario, trace, state_snapshot, before_state_snapshot)
    _write_json(Path(args.out), scorecard)
    _write_score_outputs(scorecard, args)
    print(f"wrote {args.out}")
    return 0 if scorecard["passed"] else 1


def cmd_report(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario)
    trace = _read_json(Path(args.trace))
    scorecard = _read_json(Path(args.score))
    state_diff = _read_json(Path(args.state_diff)) if args.state_diff else None
    write_report(scenario, trace, scorecard, args.out, state_diff=state_diff)
    print(f"wrote {args.out}")
    return 0


def cmd_digest(args: argparse.Namespace) -> int:
    scenario, trace, scorecard, state_diff = _load_digest_inputs(args)
    digest = build_run_digest(scenario, trace, scorecard, state_diff=state_diff)
    out_path = Path(args.out) if args.out else _default_digest_out(args)
    if out_path is None:
        raise RunDigestError("--out is required when --run is not supplied")
    _write_json(out_path, digest)
    if args.markdown_out:
        markdown_path = Path(args.markdown_out)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_run_digest_markdown(digest), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


def cmd_capture_state(args: argparse.Namespace) -> int:
    snapshot = capture_state_snapshot(
        files=args.file,
        directories=args.directory,
        json_sources=args.json_source,
        observations=args.observation,
        include_file_text=args.include_file_text,
        max_text_chars=args.max_text_chars,
        max_dir_entries=args.max_dir_entries,
        preserve_paths=args.preserve_paths,
        secret_patterns=args.secret_pattern,
    )
    _write_json(Path(args.out), snapshot)
    print(f"wrote {args.out}")
    return 0


def cmd_verify_state(args: argparse.Namespace) -> int:
    snapshot = capture_verified_state(
        args.config,
        preserve_paths=args.preserve_paths,
        secret_patterns=args.secret_pattern,
    )
    _write_json(Path(args.out), snapshot)
    print(f"wrote {args.out}")
    return 0


def cmd_state_validators(args: argparse.Namespace) -> int:
    if args.list:
        catalog = build_monitor_catalog()
        if args.out:
            _write_json(Path(args.out), catalog)
            print(f"wrote {args.out}")
        else:
            print(json.dumps(catalog, indent=2, sort_keys=True))
        if args.markdown_out:
            markdown_path = Path(args.markdown_out)
            markdown_path.parent.mkdir(parents=True, exist_ok=True)
            markdown_path.write_text(render_monitor_catalog_markdown(catalog), encoding="utf-8")
            print(f"wrote {args.markdown_out}")
        return 0

    if not args.config:
        raise StateValidatorError("state-validators requires --list or --config")
    compiled = build_state_validator_assertions(args.config)
    if args.out:
        _write_json(Path(args.out), compiled)
        print(f"wrote {args.out}")
    else:
        print(json.dumps(compiled, indent=2, sort_keys=True))
    return 0


def cmd_diff_state(args: argparse.Namespace) -> int:
    before = sanitize_state_snapshot(load_state_snapshot(args.before), args.secret_pattern)
    after = sanitize_state_snapshot(load_state_snapshot(args.after), args.secret_pattern)
    diff = build_state_diff(before, after, max_changes=args.max_changes)
    _write_json(Path(args.out), diff)
    print(f"wrote {args.out}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    result = _run_scenario_artifacts(
        args.scenario,
        args.out,
        trace_override=args.trace,
        state_override=args.state,
        before_state_override=args.before_state,
        trace_format=args.format,
        write_sensitive_trace=args.write_sensitive_trace,
        preserve_paths=args.preserve_paths,
        junit_out=args.junit_out,
        markdown_out=args.markdown_out,
    )
    scorecard = result["scorecard"]
    scenario = result["scenario"]
    report_path = result["paths"]["report"]
    print(f"{'PASS' if scorecard['passed'] else 'FAIL'} {scenario['id']} score={scorecard['score']} report={report_path}")
    return 1 if args.fail_on_score and not scorecard["passed"] else 0


def cmd_replay(args: argparse.Namespace) -> int:
    lineage_path = Path(args.lineage)
    lineage = _read_json(lineage_path)
    replay = lineage.get("replay")
    if not isinstance(replay, dict):
        raise ReplayError("artifact_lineage.replay is missing; rerun the original run to emit replay metadata")
    if replay.get("self_contained") is not True and not args.allow_non_self_contained:
        raise ReplayError("replay contract is not self-contained; restore paths or pass --allow-non-self-contained")
    argv = replay.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise ReplayError("artifact_lineage.replay.argv must be a list of strings")

    base_dir = Path(args.base_dir) if args.base_dir else _default_replay_base_dir(lineage_path, lineage)
    scenario_path = _replay_flag_path(argv, "--scenario", base_dir)
    trace_path = _replay_flag_path(argv, "--trace", base_dir)
    state_path = _replay_flag_path(argv, "--state", base_dir, required=False)
    before_state_path = _replay_flag_path(argv, "--before-state", base_dir, required=False)
    fingerprints = replay.get("input_fingerprints") if isinstance(replay.get("input_fingerprints"), dict) else {}
    _verify_replay_input("scenario", scenario_path, fingerprints)
    _verify_replay_input("source_trace", trace_path, fingerprints)
    if before_state_path is not None:
        _verify_replay_input("source_before_state_snapshot", before_state_path, fingerprints)
    if state_path is not None:
        _verify_replay_input("source_state_snapshot", state_path, fingerprints)

    result = _run_scenario_artifacts(
        scenario_path,
        args.out,
        trace_override=trace_path,
        state_override=state_path,
        before_state_override=before_state_path,
        trace_format=args.format,
        write_sensitive_trace=args.write_sensitive_trace,
        preserve_paths=args.preserve_paths,
    )
    scorecard = result["scorecard"]
    scenario = result["scenario"]
    print(f"{'PASS' if scorecard['passed'] else 'FAIL'} replay {scenario['id']} score={scorecard['score']} out={args.out}")
    return 1 if args.fail_on_score and not scorecard["passed"] else 0


def cmd_replay_bundle(args: argparse.Namespace) -> int:
    lineage_path = Path(args.lineage)
    lineage = _read_json(lineage_path)
    replay = lineage.get("replay")
    if not isinstance(replay, dict):
        raise ReplayError("artifact_lineage.replay is missing; rerun the original run to emit replay metadata")
    argv = replay.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise ReplayError("artifact_lineage.replay.argv must be a list of strings")

    base_dir = Path(args.base_dir) if args.base_dir else _default_replay_base_dir(lineage_path, lineage)
    scenario_path = _replay_flag_path(argv, "--scenario", base_dir)
    trace_path = _replay_flag_path(argv, "--trace", base_dir)
    state_path = _replay_flag_path(argv, "--state", base_dir, required=False)
    before_state_path = _replay_flag_path(argv, "--before-state", base_dir, required=False)
    fingerprints = replay.get("input_fingerprints") if isinstance(replay.get("input_fingerprints"), dict) else {}
    _verify_replay_input("scenario", scenario_path, fingerprints)
    _verify_replay_input("source_trace", trace_path, fingerprints)
    if before_state_path is not None:
        _verify_replay_input("source_before_state_snapshot", before_state_path, fingerprints)
    if state_path is not None:
        _verify_replay_input("source_state_snapshot", state_path, fingerprints)

    out_dir = Path(args.out)
    if out_dir.exists() and not out_dir.is_dir():
        raise ReplayError(f"replay bundle output is not a directory: {out_dir}")
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise ReplayError(f"replay bundle output is not empty: {out_dir}; pass --force to replace it")
    if out_dir.exists() and args.force:
        shutil.rmtree(out_dir)
    inputs_dir = out_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    copied_inputs = {
        "scenario": _copy_replay_input(scenario_path, inputs_dir / "scenario.json"),
        "source_trace": _copy_replay_input(trace_path, inputs_dir / _trace_bundle_name(trace_path)),
    }
    if state_path is not None:
        copied_inputs["source_state_snapshot"] = _copy_replay_input(state_path, inputs_dir / "source_state_snapshot.json")
    if before_state_path is not None:
        copied_inputs["source_before_state_snapshot"] = _copy_replay_input(
            before_state_path,
            inputs_dir / "source_before_state_snapshot.json",
        )

    bundle_lineage = _portable_replay_lineage(
        lineage=lineage,
        source_lineage_path=lineage_path,
        copied_inputs=copied_inputs,
        preserve_paths=args.preserve_paths,
    )
    bundle_lineage_path = out_dir / "artifact_lineage.json"
    _write_json(bundle_lineage_path, bundle_lineage)

    manifest = _replay_bundle_manifest(
        bundle_lineage=bundle_lineage,
        bundle_lineage_path=bundle_lineage_path,
        source_lineage_path=lineage_path,
        copied_inputs=copied_inputs,
        preserve_paths=args.preserve_paths,
    )
    manifest_path = out_dir / "replay_bundle.json"
    _write_json(manifest_path, manifest)
    print(f"wrote replay bundle {out_dir}")
    return 0


def _scenario_paths_from_suite_manifest(scenario_paths: list[Path], manifest_path: Path) -> list[Path]:
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != EVAL_SUITE_MANIFEST_SCHEMA_VERSION:
        raise ScenarioError(
            f"suite manifest schema_version must be {EVAL_SUITE_MANIFEST_SCHEMA_VERSION!r}; got {manifest.get('schema_version')!r}"
        )
    scenario_ids = manifest.get("scenario_ids")
    if not isinstance(scenario_ids, list) or not scenario_ids or not all(isinstance(item, str) and item for item in scenario_ids):
        raise ScenarioError("suite manifest scenario_ids must be a non-empty list of strings")
    duplicates = sorted({scenario_id for scenario_id in scenario_ids if scenario_ids.count(scenario_id) > 1})
    if duplicates:
        raise ScenarioError(f"suite manifest has duplicate scenario_ids: {', '.join(duplicates)}")

    by_id: dict[str, Path] = {}
    for scenario_path in scenario_paths:
        scenario = load_scenario(scenario_path)
        scenario_id = str(scenario["id"])
        if scenario_id in by_id:
            raise ScenarioError(f"Duplicate discovered scenario id {scenario_id!r}: {scenario_path} conflicts with {by_id[scenario_id]}")
        by_id[scenario_id] = scenario_path
    missing = [scenario_id for scenario_id in scenario_ids if scenario_id not in by_id]
    if missing:
        raise ScenarioError(f"suite manifest references missing scenario_ids: {', '.join(missing)}")
    return [by_id[scenario_id] for scenario_id in scenario_ids]


def cmd_run_suite(args: argparse.Namespace) -> int:
    scenario_paths = discover_scenarios(Path(args.scenarios), args.pattern, args.recursive)
    if args.suite_manifest:
        scenario_paths = _scenario_paths_from_suite_manifest(scenario_paths, Path(args.suite_manifest))
    out_dir = Path(args.out)
    metadata = _metadata_options(args.metadata)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    seen_run_ids: dict[str, Path] = {}
    for scenario_path in scenario_paths:
        try:
            scenario = load_scenario(scenario_path)
            run_id = _safe_run_id(str(scenario["id"]))
            if run_id in seen_run_ids:
                raise ScenarioError(
                    f"Duplicate scenario id/run directory {scenario['id']!r}: "
                    f"{scenario_path} conflicts with {seen_run_ids[run_id]}"
                )
            seen_run_ids[run_id] = scenario_path
            run_dir = out_dir / run_id
            result = _run_scenario_artifacts(
                scenario_path,
                run_dir,
                trace_format=args.format,
                write_sensitive_trace=args.write_sensitive_trace,
                preserve_paths=args.preserve_paths,
                junit_out=run_dir / "scorecard.junit.xml" if args.junit else None,
                markdown_out=run_dir / "scorecard.md" if args.markdown else None,
            )
            scorecard = result["scorecard"]
            runs.append(
                {
                    "scenario_id": result["scenario"]["id"],
                    "scenario_title": result["scenario"].get("title", result["scenario"]["id"]),
                    "task_family": _task_family(str(result["scenario"]["id"])),
                    "scenario_path": _display_path(scenario_path, args.preserve_paths),
                    "scenario_sha256": _lineage_input_hash(result["lineage"], "scenario"),
                    "trace_path": _display_path(result["trace_path"], args.preserve_paths),
                    "trace_sha256": _lineage_input_hash(result["lineage"], "source_trace"),
                    "before_state_path": (
                        _display_path(result["before_state_path"], args.preserve_paths)
                        if result.get("before_state_path")
                        else None
                    ),
                    "before_state_sha256": _lineage_input_hash(result["lineage"], "source_before_state_snapshot"),
                    "state_path": _display_path(result["state_path"], args.preserve_paths) if result.get("state_path") else None,
                    "state_sha256": _lineage_input_hash(result["lineage"], "source_state_snapshot"),
                    "run_dir": _display_path(run_dir, args.preserve_paths),
                    "report": _display_path(result["paths"]["report"], args.preserve_paths),
                    "scorecard": _display_path(result["paths"]["scorecard"], args.preserve_paths),
                    "run_digest": _display_path(result["paths"]["run_digest"], args.preserve_paths),
                    "lineage": _display_path(result["paths"]["lineage"], args.preserve_paths),
                    "passed": bool(scorecard["passed"]),
                    "score": scorecard["score"],
                    "failed_rules": _failed_rule_ids(scorecard),
                    "critical_failures": scorecard.get("critical_failures", []),
                }
            )
            print(
                f"{'PASS' if scorecard['passed'] else 'FAIL'} "
                f"{result['scenario']['id']} score={scorecard['score']} report={result['paths']['report']}"
            )
        except (AdapterError, ScenarioError, TrainingExportError, OSError, json.JSONDecodeError) as exc:
            errors.append({"scenario_path": _display_path(scenario_path, args.preserve_paths), "error": str(exc)})
            print(f"ERROR {scenario_path}: {exc}")

    artifacts: dict[str, str] = {}
    index_path = Path(args.index_out) if args.index_out else out_dir / "index.html"
    if not args.no_index:
        completed_run_dirs = [out_dir / _safe_run_id(str(run["scenario_id"])) for run in runs]
        write_index(completed_run_dirs, index_path, artifacts_dir=out_dir)
        artifacts["index"] = _display_path(index_path, args.preserve_paths)

    training_manifest: dict[str, Any] | None = None
    training_out = Path(args.training_export_out) if args.training_export_out else out_dir / "training_export"
    if args.export_rl:
        if runs:
            training_manifest = export_rl_dataset(
                out_dir,
                training_out,
                reward_scale=args.reward_scale,
                min_score_gap=args.min_score_gap,
                max_pairs_per_family=args.max_pairs_per_family,
                preserve_paths=args.preserve_paths,
                metadata=metadata,
            )
            artifacts["training_export"] = _display_path(training_out, args.preserve_paths)
        else:
            errors.append(
                {
                    "scenario_path": _display_path(Path(args.scenarios), args.preserve_paths),
                    "error": "Cannot export RL artifacts because no scenario runs completed.",
                }
            )

    validation_path = Path(args.validation_out) if args.validation_out else out_dir / "validation.json"
    summary_path = Path(args.summary_out) if args.summary_out else out_dir / "suite_summary.json"
    handoff_paths: dict[str, Path] = {}
    handoff_bundle: dict[str, Any] | None = None
    if args.evidence_handoff:
        if runs:
            handoff_paths = {
                "scenario_quality": out_dir / "scenario_quality.json",
                "evidence_coverage": out_dir / "evidence_coverage.json",
                "trace_observability": out_dir / "trace_observability.json",
                "repair_queue": out_dir / "repair_queue.json",
                "evidence_bundle": out_dir / "evidence_bundle.json",
            }
            scenario_quality = build_scenario_quality(
                Path(args.scenarios),
                pattern=args.pattern,
                recursive=args.recursive,
                require_traces=True,
                preserve_paths=args.preserve_paths,
            )
            _write_json(handoff_paths["scenario_quality"], scenario_quality)
            artifacts["scenario_quality"] = _display_path(handoff_paths["scenario_quality"], args.preserve_paths)

            evidence_coverage = build_evidence_coverage(out_dir, preserve_paths=args.preserve_paths)
            _write_json(handoff_paths["evidence_coverage"], evidence_coverage)
            artifacts["evidence_coverage"] = _display_path(handoff_paths["evidence_coverage"], args.preserve_paths)

            trace_observability = build_trace_observability(out_dir, preserve_paths=args.preserve_paths)
            _write_json(handoff_paths["trace_observability"], trace_observability)
            artifacts["trace_observability"] = _display_path(handoff_paths["trace_observability"], args.preserve_paths)

            repair_queue = build_repair_queue(out_dir, preserve_paths=args.preserve_paths)
            _write_json(handoff_paths["repair_queue"], repair_queue)
            artifacts["repair_queue"] = _display_path(handoff_paths["repair_queue"], args.preserve_paths)

            harness_paths = _write_run_suite_harness_handoff(out_dir, runs, summary_path=summary_path)
            if harness_paths:
                handoff_paths.update(harness_paths)
                artifacts["harness_manifest"] = _display_path(harness_paths["harness_manifest"], args.preserve_paths)
                artifacts["harness_result"] = _display_path(harness_paths["harness_result"], args.preserve_paths)
            artifacts["evidence_bundle"] = _display_path(handoff_paths["evidence_bundle"], args.preserve_paths)
        else:
            errors.append(
                {
                    "scenario_path": _display_path(Path(args.scenarios), args.preserve_paths),
                    "error": "Cannot build evidence handoff because no scenario runs completed.",
                }
            )

    validation_summary: dict[str, Any] | None = None
    if args.validate:
        artifacts["validation"] = _display_path(validation_path, args.preserve_paths)

    summary = _run_suite_summary(
        scenarios_dir=Path(args.scenarios),
        out_dir=out_dir,
        runs=runs,
        errors=errors,
        artifacts=artifacts,
        preserve_paths=args.preserve_paths,
        training_manifest=training_manifest,
        validation_summary=None,
        metadata=metadata,
    )
    _write_json(summary_path, summary)

    if args.validate:
        validation_summary = validate_artifacts(
            runs_dir=out_dir,
            training_export_dir=training_out if args.export_rl else None,
            evidence_coverage_paths=[handoff_paths["evidence_coverage"]] if args.evidence_handoff and handoff_paths else None,
            trace_observability_paths=[handoff_paths["trace_observability"]] if args.evidence_handoff and handoff_paths else None,
            scenario_quality_paths=[handoff_paths["scenario_quality"]] if args.evidence_handoff and handoff_paths else None,
            repair_queue_paths=[handoff_paths["repair_queue"]] if args.evidence_handoff and handoff_paths else None,
            harness_manifest_paths=(
                [handoff_paths["harness_manifest"]]
                if args.evidence_handoff and "harness_manifest" in handoff_paths
                else None
            ),
            harness_result_paths=(
                [handoff_paths["harness_result"]]
                if args.evidence_handoff and "harness_result" in handoff_paths
                else None
            ),
            suite_summary_paths=[summary_path],
            strict=args.strict,
        )
        _write_json(validation_path, validation_summary)

    summary = _run_suite_summary(
        scenarios_dir=Path(args.scenarios),
        out_dir=out_dir,
        runs=runs,
        errors=errors,
        artifacts=artifacts,
        preserve_paths=args.preserve_paths,
        training_manifest=training_manifest,
        validation_summary=validation_summary,
        metadata=metadata,
    )
    _write_json(summary_path, summary)

    if args.evidence_handoff and handoff_paths:
        handoff_bundle = build_evidence_bundle(
            out_path=handoff_paths["evidence_bundle"],
            runs_dir=out_dir,
            suite_summary_path=summary_path,
            scenario_quality_path=handoff_paths["scenario_quality"],
            evidence_coverage_path=handoff_paths["evidence_coverage"],
            trace_observability_path=handoff_paths["trace_observability"],
            repair_queue_path=handoff_paths["repair_queue"],
            validation_path=validation_path if args.validate else None,
            training_export_dir=training_out if args.export_rl else None,
            harness_manifest_paths=(
                [handoff_paths["harness_manifest"]] if "harness_manifest" in handoff_paths else None
            ),
            harness_result_paths=(
                [handoff_paths["harness_result"]] if "harness_result" in handoff_paths else None
            ),
            require_harness=True,
            preserve_paths=args.preserve_paths,
        )
        _write_json(handoff_paths["evidence_bundle"], handoff_bundle)

    if not args.no_index:
        completed_run_dirs = [out_dir / _safe_run_id(str(run["scenario_id"])) for run in runs]
        write_index(completed_run_dirs, index_path, artifacts_dir=out_dir)

    print(
        f"SUITE total={summary['total']} passed={summary['passed']} failed={summary['failed']} "
        f"errors={summary['error_count']} summary={summary_path}"
    )

    if errors:
        return 1
    if args.validate and validation_summary and not validation_summary["passed"]:
        return 1
    if args.evidence_handoff and handoff_bundle and not handoff_bundle["passed"]:
        return 1
    if args.fail_on_failed and summary["failed"] > 0:
        return 1
    return 0


def cmd_goal3_handoff(args: argparse.Namespace) -> int:
    target = Path(args.out)
    if target.exists() and not target.is_dir():
        raise ArtifactError(f"goal3 handoff output is not a directory: {target}")
    if target.exists() and any(target.iterdir()):
        if not args.force:
            raise ArtifactError(f"goal3 handoff output is not empty: {target}; pass --force to replace it")
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    metadata = _metadata_options(args.metadata)
    runs_dir = target / "runs"
    training_export_dir = target / "training_export"
    suite_summary_path = target / "suite_summary.json"
    validation_path = target / "validation.json"
    index_path = target / "index.html"
    gate_path = target / "training_gate.json"
    preflight_path = target / "trainer_preflight.json"
    handoff_path = target / "goal3_handoff.json"
    evidence_bundle_path = runs_dir / "evidence_bundle.json"

    suite_code = cmd_run_suite(
        argparse.Namespace(
            scenarios=args.scenarios,
            pattern=args.pattern,
            recursive=args.recursive,
            suite_manifest=args.suite_manifest,
            out=str(runs_dir),
            format=args.format,
            summary_out=str(suite_summary_path),
            index_out=str(index_path),
            no_index=False,
            junit=False,
            markdown=False,
            export_rl=True,
            training_export_out=str(training_export_dir),
            reward_scale=args.reward_scale,
            min_score_gap=args.min_score_gap,
            max_pairs_per_family=args.max_pairs_per_family,
            validate=True,
            validation_out=str(validation_path),
            strict=args.strict,
            evidence_handoff=True,
            write_sensitive_trace=False,
            preserve_paths=args.preserve_paths,
            metadata=args.metadata,
            fail_on_failed=False,
        )
    )

    training_manifest = _read_json(training_export_dir / "manifest.json")
    validation_summary = _read_json(validation_path)
    evidence_bundle = _read_json(evidence_bundle_path)
    dataset_version = str(training_manifest.get("dataset_version") or "")

    gate_code = cmd_gate_export(_goal3_training_gate_args(args, training_export_dir, gate_path))
    gate = _read_json(gate_path)

    preflight_code = cmd_trainer_preflight(
        argparse.Namespace(
            out=str(preflight_path),
            gate=[str(gate_path)],
            training_export=str(training_export_dir),
            compare_export=None,
            reviewed_export=None,
            evidence_bundle=str(evidence_bundle_path),
            agentic_training_plan=None,
            validation=[str(validation_path)],
            require_gate=["training_gate"],
            require_dataset_version=[dataset_version] if dataset_version else [],
            trainer_command=args.trainer_command,
            allow_unvalidated_gates=False,
            preserve_paths=args.preserve_paths,
            metadata=args.metadata,
        )
    )
    preflight = _read_json(preflight_path)

    artifacts = {
        "runs": _display_path(runs_dir, args.preserve_paths),
        "suite_summary": _display_path(suite_summary_path, args.preserve_paths),
        "training_export": _display_path(training_export_dir, args.preserve_paths),
        "validation": _display_path(validation_path, args.preserve_paths),
        "evidence_bundle": _display_path(evidence_bundle_path, args.preserve_paths),
        "training_gate": _display_path(gate_path, args.preserve_paths),
        "trainer_preflight": _display_path(preflight_path, args.preserve_paths),
    }
    if args.policy:
        artifacts["training_gate_policy"] = _display_path(Path(args.policy), args.preserve_paths)

    stages = [
        {
            "id": "run_suite",
            "passed": suite_code == 0,
            "artifact": artifacts["suite_summary"],
            "summary": "Scenario suite, optional RL export, validation, and evidence handoff generation.",
        },
        {
            "id": "training_export",
            "passed": bool(dataset_version),
            "artifact": artifacts["training_export"],
            "summary": "Trainer-ready RL dataset export with dataset_version selection key.",
        },
        {
            "id": "validation",
            "passed": validation_summary.get("passed") is True,
            "artifact": artifacts["validation"],
            "summary": "Structural validation over runs, training export, and evidence handoff artifacts.",
        },
        {
            "id": "evidence_bundle",
            "passed": evidence_bundle.get("passed") is True,
            "artifact": artifacts["evidence_bundle"],
            "summary": "Evidence handoff bundle over scenario quality, evidence coverage, observability, repair queue, and harness artifacts.",
        },
        {
            "id": "training_gate",
            "passed": gate.get("passed") is True and gate_code == 0,
            "artifact": artifacts["training_gate"],
            "summary": "Training dataset readiness gate.",
        },
        {
            "id": "trainer_preflight",
            "passed": preflight.get("passed") is True and preflight_code == 0,
            "artifact": artifacts["trainer_preflight"],
            "summary": "Trainer launch guard manifest that records but does not execute the trainer command.",
        },
    ]
    passed = all(stage["passed"] for stage in stages)
    handoff = {
        "schema_version": GOAL3_HANDOFF_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "recommendation": "handoff_ready" if passed else "fix_handoff",
        "output_dir": _display_path(target, args.preserve_paths),
        "dataset_version": dataset_version,
        "metadata": metadata,
        "artifacts": artifacts,
        "stages": stages,
        "notes": [
            "Goal 3 handoff builds export, validation, gate, evidence bundle, and trainer preflight artifacts in one reproducible sequence.",
            "The trainer command is recorded for downstream launch checks; it is not executed by this command.",
        ],
    }
    _write_json(handoff_path, handoff)
    print(
        f"{'READY' if passed else 'BLOCKED'} goal3-handoff "
        f"dataset_version={dataset_version or 'missing'} out={handoff_path}"
    )
    return 0 if passed else 1


def cmd_index(args: argparse.Namespace) -> int:
    runs_dir = Path(args.runs)
    run_dirs = sorted(path for path in runs_dir.iterdir() if path.is_dir())
    write_index(run_dirs, args.out, artifacts_dir=runs_dir)
    print(f"wrote {args.out}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    summary = _audit_runs(Path(args.runs), args.forbid_text)
    rendered = json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(rendered, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(rendered, end="")
    if args.fail_on_leak and summary["leaks"]:
        return 1
    if args.fail_on_failed and summary["failed"] > 0:
        return 1
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    baseline, baseline_label = _read_scorecard_ref(Path(args.baseline))
    candidate, candidate_label = _read_scorecard_ref(Path(args.candidate))
    comparison = compare_scorecards(
        baseline,
        candidate,
        baseline_label=baseline_label,
        candidate_label=candidate_label,
    )
    _write_json(Path(args.out), comparison)
    if args.html_out:
        write_compare_report(comparison, args.html_out)
    print(f"{'REGRESSION' if comparison['regressed'] else 'NO REGRESSION'} score_delta={comparison['score_delta']} wrote {args.out}")
    return 1 if args.fail_on_regression and comparison["regressed"] else 0


def cmd_compare_suite(args: argparse.Namespace) -> int:
    comparison = compare_suites(
        args.baseline,
        args.candidate,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        contract_scope=args.contract_scope,
    )
    _write_json(Path(args.out), comparison)
    if args.html_out:
        write_suite_compare_report(comparison, args.html_out)
    aggregate = comparison["aggregate"]
    print(
        f"{'REGRESSION' if comparison['regressed'] else 'NO REGRESSION'} "
        f"paired={aggregate['paired_count']} avg_score_delta={aggregate['avg_score_delta']} wrote {args.out}"
    )
    if args.fail_on_contract_drift and aggregate.get("contract_drift_count", 0) > 0:
        return 1
    if args.fail_on_unverified_contracts and aggregate.get("unverified_contract_count", 0) > 0:
        return 1
    return 1 if args.fail_on_regression and comparison["regressed"] else 0


def cmd_trend_suite(args: argparse.Namespace) -> int:
    trend = build_suite_trend(args.suite_summary)
    _write_json(Path(args.out), trend)
    if args.html_out:
        write_suite_trend_report(trend, args.html_out)
    print(f"TREND points={trend['point_count']} summary={trend['summary']} wrote {args.out}")
    return 0


def cmd_observer_template(args: argparse.Namespace) -> int:
    rendered = OBSERVER_TEMPLATE
    if args.out:
        path = Path(args.out)
        if path.exists() and not args.force:
            raise FileExistsError(f"Refusing to overwrite existing file without --force: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
        print(f"wrote {path}")
    else:
        print(rendered, end="")
    return 0


def cmd_check_scenarios(args: argparse.Namespace) -> int:
    summary = check_scenarios(
        args.scenarios,
        pattern=args.pattern,
        recursive=args.recursive,
        require_traces=args.require_traces,
        strict=args.strict,
        preserve_paths=args.preserve_paths,
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(rendered, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(rendered, end="")
    return 0 if summary["passed"] else 1


def cmd_scenario_quality(args: argparse.Namespace) -> int:
    summary = build_scenario_quality(
        args.scenarios,
        pattern=args.pattern,
        recursive=args.recursive,
        require_traces=args.require_traces,
        preserve_paths=args.preserve_paths,
        min_average_score=args.min_average_score,
        min_scenario_score=args.min_scenario_score,
        min_observable_rate=args.min_observable_rate,
        max_weak_scenarios=args.max_weak_scenarios,
        max_final_only_scenarios=args.max_final_only_scenarios,
        max_missing_traces=args.max_missing_traces,
        require_task_families=args.require_task_family,
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(rendered, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(rendered, end="")
    return 0 if summary["passed"] else 1


def cmd_validate(args: argparse.Namespace) -> int:
    summary = validate_artifacts(
        runs_dir=args.runs,
        run_dirs=args.run,
        training_export_dir=args.training_export,
        compare_export_dir=args.compare_export,
        review_export_dir=args.review_export,
        reviewed_export_dir=args.reviewed_export,
        evidence_coverage_paths=args.evidence_coverage,
        evidence_bundle_paths=args.evidence_bundle,
        improvement_plan_paths=args.improvement_plan,
        improvement_ledger_paths=args.improvement_ledger,
        improvement_ledger_gate_paths=args.improvement_ledger_gate,
        action_ledger_paths=args.action_ledger,
        action_ledger_gate_paths=args.action_ledger_gate,
        decision_gate_paths=args.decision_gate,
        promotion_cards_paths=args.promotion_cards,
        promotion_decision_paths=args.promotion_decision,
        promotion_alias_apply_paths=args.promotion_alias_apply,
        promotion_rollback_receipt_paths=args.promotion_rollback_receipt,
        promotion_release_record_paths=args.promotion_release_record,
        promotion_policy_paths=args.promotion_policy,
        promotion_ledger_paths=args.promotion_ledger,
        promotion_ledger_gate_paths=args.promotion_ledger_gate,
        promotion_archive_paths=args.promotion_archive,
        trainer_preflight_paths=args.trainer_preflight,
        trainer_launch_check_paths=args.trainer_launch_check,
        trainer_archive_paths=args.trainer_archive,
        trainer_archive_check_paths=args.trainer_archive_check,
        trainer_consumer_plan_paths=args.trainer_consumer_plan,
        trainer_wrapper_dry_run_paths=args.trainer_wrapper_dry_run,
        model_scout_manifest_paths=args.model_scout_manifest,
        model_candidate_paths=args.model_candidate,
        model_compatibility_report_paths=args.model_compatibility_report,
        model_serving_probe_receipt_paths=args.model_serving_probe_receipt,
        model_adapter_manifest_paths=args.model_adapter_manifest,
        model_registry_entry_paths=args.model_registry_entry,
        model_registry_paths=args.model_registry,
        training_plan_paths=args.training_plan,
        agentic_training_result_paths=args.agentic_training_result,
        agentic_training_loop_plan_paths=args.agentic_loop_plan,
        repair_queue_paths=args.repair_queue,
        replay_bundle_paths=args.replay_bundle,
        trace_observability_paths=args.trace_observability,
        review_calibration_paths=args.review_calibration,
        scenario_quality_paths=args.scenario_quality,
        suite_summary_paths=args.suite_summary,
        suite_trend_paths=args.suite_trend,
        eval_suite_manifest_paths=args.eval_suite_manifest,
        state_snapshot_paths=args.state_snapshot,
        state_diff_paths=args.state_diff,
        run_digest_paths=args.run_digest,
        harness_manifest_paths=args.harness_manifest,
        harness_result_paths=args.harness_result,
        harness_replay_result_paths=args.harness_replay_result,
        harness_suite_result_paths=args.harness_suite_result,
        live_smoke_summary_paths=args.live_smoke_summary,
        eval_summary_paths=args.eval_summary,
        external_eval_plan_paths=args.external_eval_plan,
        heldout_manifest_paths=args.heldout_manifest,
        serving_profile_paths=args.serving_profile,
        serving_compatibility_report_paths=args.serving_compatibility_report,
        serving_endpoint_check_paths=args.serving_endpoint_check,
        serving_lifecycle_paths=args.serving_lifecycle,
        serving_demo_run_paths=args.serving_demo_run,
        strict=args.strict,
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(rendered, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(rendered, end="")
    return 0 if summary["passed"] else 1



def cmd_model_scout_validate(args: argparse.Namespace) -> int:
    summary = validate_artifacts(model_scout_manifest_paths=[args.manifest], strict=args.strict)
    _emit_json_payload(summary, args.out)
    return 0 if summary["passed"] else 1


def cmd_model_candidate_validate(args: argparse.Namespace) -> int:
    candidate_path = Path(args.candidate)
    candidate = _read_json(candidate_path)
    errors = model_candidate_errors(candidate, require_training_eligible=args.require_training_eligible)
    target = {
        "type": "model_candidate",
        "path": str(candidate_path),
        "passed": not errors,
        "errors": errors,
        "warnings": [],
        "details": {
            "candidate_id": candidate.get("candidate_id") if isinstance(candidate, dict) else None,
            "model_id": candidate.get("model_id") if isinstance(candidate, dict) else None,
            "require_training_eligible": args.require_training_eligible,
        },
    }
    summary = {
        "schema_version": "hfr.validation.v1",
        "passed": not errors,
        "strict": False,
        "target_count": 1,
        "error_count": len(errors),
        "warning_count": 0,
        "targets": [target],
    }
    _emit_json_payload(summary, args.out)
    return 0 if summary["passed"] else 1


def cmd_model_candidate_compatibility_report(args: argparse.Namespace) -> int:
    candidate = _read_json(Path(args.candidate))
    report = build_model_compatibility_report(candidate, out_path=args.out, preserve_paths=args.preserve_paths)
    _write_json(Path(args.out), report)
    print(f"wrote {args.out}")
    return 0 if report["passed"] else 1


def cmd_model_registry_validate(args: argparse.Namespace) -> int:
    summary = validate_artifacts(model_registry_paths=[args.registry], strict=args.strict)
    _emit_json_payload(summary, args.out)
    return 0 if summary["passed"] else 1


def cmd_model_registry_register(args: argparse.Namespace) -> int:
    registry_path = Path(args.registry)
    registry = load_model_registry(registry_path)
    candidate = _read_json(Path(args.candidate))
    registry = register_model_candidate(registry, candidate, status=args.status)
    _write_json(registry_path, registry)
    entry = registry["entries"][candidate["candidate_id"]]
    if args.entry_out:
        _write_json(Path(args.entry_out), entry)
    print(f"registered {candidate['candidate_id']} in {registry_path}")
    return 0


def cmd_model_registry_list(args: argparse.Namespace) -> int:
    registry = load_model_registry(args.registry)
    rows = list_model_registry_entries(registry)
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True, ensure_ascii=False))
    elif rows:
        print("entry_id\tmodel_id\tstatus\ttraining_eligible\tlicense_status\taliases")
        for row in rows:
            print(
                "\t".join(
                    [
                        str(row["entry_id"]),
                        str(row["model_id"]),
                        str(row["status"]),
                        str(row["training_eligible"]).lower(),
                        str(row["license_status"]),
                        ",".join(row["aliases"]),
                    ]
                )
            )
    else:
        print("no model registry entries")
    return 0


def cmd_model_registry_alias(args: argparse.Namespace) -> int:
    registry_path = Path(args.registry)
    registry = load_model_registry(registry_path)
    registry = move_model_alias(
        registry,
        alias=args.alias,
        target=args.target,
        rollback_target=args.rollback_target,
        reason=args.reason or "",
    )
    _write_json(registry_path, registry)
    print(f"moved {args.alias} -> {args.target} in {registry_path}")
    return 0


def cmd_model_registry_link(args: argparse.Namespace) -> int:
    registry_path = Path(args.registry)
    registry = load_model_registry(registry_path)
    registry = link_model_registry_artifact(
        registry,
        entry_id=args.entry,
        collection=args.collection,
        artifact_id=args.artifact_id,
        kind=args.kind,
        status=args.status,
        path=args.path,
        sha256=args.sha256,
        metadata=_metadata_options(args.metadata),
        preserve_paths=args.preserve_paths,
    )
    _write_json(registry_path, registry)
    if args.entry_out:
        _write_json(Path(args.entry_out), registry["entries"][args.entry])
    print(f"linked {args.collection}:{args.artifact_id} to {args.entry} in {registry_path}")
    return 0


def cmd_model_registry_serving_probe_receipt(args: argparse.Namespace) -> int:
    registry_path = Path(args.registry)
    registry = load_model_registry(registry_path)
    compatibility_report = _read_json(Path(args.compatibility_report)) if args.compatibility_report else None
    receipt = build_model_serving_probe_receipt(
        registry,
        model_ref=args.model_ref,
        out_path=args.out,
        profile_id=args.profile_id,
        provider=args.provider,
        serving_engine=args.serving_engine,
        base_url=args.base_url,
        probe_mode=args.probe_mode,
        compatibility_report=compatibility_report,
        compatibility_report_path=args.compatibility_report,
        preserve_paths=args.preserve_paths,
    )
    receipt_path = Path(args.out)
    _write_json(receipt_path, receipt)
    if args.link:
        artifact_id = args.artifact_id or receipt_path.stem
        registry = link_model_registry_artifact(
            registry,
            entry_id=receipt["entry_id"],
            collection="serving_probes",
            artifact_id=artifact_id,
            kind="model_serving_probe_receipt",
            status=args.link_status,
            path=receipt_path,
            metadata={
                "probe_mode": receipt["probe_mode"],
                "readiness": receipt["readiness"],
                "provider": receipt["serving_profile"]["provider"],
                "serving_engine": receipt["serving_profile"]["serving_engine"],
            },
            preserve_paths=args.preserve_paths,
        )
        _write_json(registry_path, registry)
        if args.entry_out:
            _write_json(Path(args.entry_out), registry["entries"][receipt["entry_id"]])
    print(f"wrote {args.out}")
    return 0 if receipt["passed"] else 1


def cmd_model_registry_adapter_manifest(args: argparse.Namespace) -> int:
    registry_path = Path(args.registry)
    registry = load_model_registry(registry_path)
    training_plan = _read_json(Path(args.training_plan))
    manifest = build_model_adapter_manifest(
        registry,
        model_ref=args.model_ref,
        adapter_id=args.adapter_id,
        training_plan=training_plan,
        training_plan_path=args.training_plan,
        out_path=args.out,
        adapter_kind=args.kind,
        status=args.status,
        output_dir=args.output_dir,
        preserve_paths=args.preserve_paths,
    )
    manifest_path = Path(args.out)
    _write_json(manifest_path, manifest)
    if args.link:
        registry = link_model_registry_artifact(
            registry,
            entry_id=manifest["base_model"]["entry_id"],
            collection="adapters",
            artifact_id=manifest["registry_link"]["artifact_id"],
            kind=manifest["registry_link"]["kind"],
            status=args.link_status,
            path=manifest_path,
            metadata={
                "adapter_kind": manifest["adapter_kind"],
                "readiness": manifest["readiness"],
                "training_plan": manifest["training_plan"]["path"],
                "training_plan_sha256": manifest["training_plan"]["sha256"],
            },
            preserve_paths=args.preserve_paths,
        )
        _write_json(registry_path, registry)
        if args.entry_out:
            _write_json(Path(args.entry_out), registry["entries"][manifest["base_model"]["entry_id"]])
    print(f"wrote {args.out}")
    return 0 if manifest["passed"] else 1


def cmd_training_plan_dry_run(args: argparse.Namespace) -> int:
    registry = load_model_registry(args.registry)
    compatibility_report = _read_json(Path(args.compatibility_report)) if args.compatibility_report else None
    plan = build_dry_run_training_plan(
        registry,
        model_ref=args.model_ref,
        dataset_id=args.dataset_id,
        dataset_manifest=args.dataset_manifest,
        trainer=args.trainer,
        mode=args.mode,
        output_dir=args.output_dir,
        out_path=args.out,
        hyperparameters=dict(args.hyperparameter),
        compute=dict(args.compute),
        compatibility_report=compatibility_report,
        compatibility_report_path=args.compatibility_report,
        preserve_paths=args.preserve_paths,
    )
    _write_json(Path(args.out), plan)
    print(f"wrote {args.out}")
    return 0


def _emit_json_payload(payload: dict[str, Any], out: str | None) -> None:
    if out:
        _write_json(Path(out), payload)
        print(f"wrote {out}")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))

def cmd_eval_summary(args: argparse.Namespace) -> int:
    summary = build_eval_summary(
        suite_summary_specs=args.suite_summary,
        compare_export_specs=args.compare_export,
        compare_gate_specs=args.compare_gate,
        external_adapter_plan_specs=args.external_adapter_plan,
        serving_check_specs=args.serving_check,
        require_serving_preflight=args.require_serving_preflight,
        preserve_paths=args.preserve_paths,
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(rendered, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(rendered, end="")
    if args.markdown_out:
        Path(args.markdown_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown_out).write_text(render_eval_summary_markdown(summary), encoding="utf-8")
        print(f"wrote {args.markdown_out}")
    return 0 if summary["passed"] else 1


def cmd_external_eval_plan(args: argparse.Namespace) -> int:
    plan = build_external_eval_plan(
        adapters=args.adapter,
        scenario_manifest=args.scenario_manifest,
        model_endpoint=args.model_endpoint,
        model=args.model,
        tool_schema_set=args.tool_schema_set,
        inspect_task_set=args.inspect_task_set,
        lm_eval_task_list=args.lm_eval_task,
        swe_bench_task_set=args.swe_bench_task_set,
        sandbox_policy=args.sandbox_policy,
        allow_installed=args.allow_installed,
        preserve_paths=args.preserve_paths,
    )
    if args.out:
        write_external_eval_plan(plan, args.out, preserve_paths=args.preserve_paths)
        print(f"wrote {args.out}")
    else:
        print(json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if plan["ready"] else 1


def cmd_agentic_loop_plan(args: argparse.Namespace) -> int:
    artifact_paths = {
        "action_ledger": args.action_ledger,
        "agentic_training_plan": args.agentic_training_plan,
        "agentic_training_result": args.agentic_training_result,
        "agentic_training_runtime_preflight": args.agentic_training_runtime_preflight,
        "evidence_bundle": args.evidence_bundle,
        "eval_summary": args.eval_summary,
        "external_eval_plan": args.external_eval_plan,
        "harness_manifest": args.harness_manifest,
        "harness_result": args.harness_result,
        "heldout_manifest": args.heldout_manifest,
        "improvement_ledger": args.improvement_ledger,
        "improvement_plan": args.improvement_plan,
        "promotion_decision": args.promotion_decision,
        "promotion_ledger": args.promotion_ledger,
        "review_calibration": args.review_calibration,
        "reviewed_gate": args.reviewed_gate,
        "serving_lifecycle": args.serving_lifecycle,
        "trainer_launch_check": args.trainer_launch_check,
        "trainer_preflight": args.trainer_preflight,
        "training_export": args.training_export,
    }
    provider_constraints = {
        "providers": args.provider,
        "regions": args.region,
        "gpu_classes": args.gpu_class,
    }
    plan = build_agentic_training_loop_plan(
        out_path=args.out,
        iteration_id=args.iteration_id,
        objective=args.objective,
        candidate=args.candidate,
        baseline=args.baseline,
        teacher=args.teacher,
        artifact_paths=artifact_paths,
        budget=dict(args.budget or []),
        provider_constraints=provider_constraints,
        schedule=dict(args.schedule or []),
        preserve_paths=args.preserve_paths,
        created_at=args.created_at,
    )
    write_agentic_training_loop_plan(args.out, plan)
    print(
        f"wrote {args.out} readiness={plan['readiness']} "
        f"checks={plan['check_count'] - plan['failed_check_count']}/{plan['check_count']}"
    )
    return 0


def cmd_heldout_manifest(args: argparse.Namespace) -> int:
    manifest = build_heldout_manifest(
        suite_summary_specs=args.suite_summary,
        preserve_paths=args.preserve_paths,
    )
    if args.out:
        write_heldout_manifest(manifest, args.out)
        print(f"wrote {args.out}")
    else:
        print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if manifest["ready"] else 1



def cmd_schemas(args: argparse.Namespace) -> int:
    if args.check and args.check_jsonl:
        raise SchemaRegistryError("--check cannot be combined with --check-jsonl")
    if args.check:
        if args.write_dir:
            raise SchemaRegistryError("--check cannot be combined with --write-dir")
        if len(args.name) > 1:
            raise SchemaRegistryError("--check accepts at most one --name")
        result = check_schema_file(args.check, args.name[0] if args.name else None)
        rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(rendered, encoding="utf-8")
            print(f"wrote {args.out}")
        else:
            print(rendered, end="")
        return 0 if result["passed"] else 1
    if args.check_jsonl:
        if args.write_dir:
            raise SchemaRegistryError("--check-jsonl cannot be combined with --write-dir")
        if len(args.name) > 1:
            raise SchemaRegistryError("--check-jsonl accepts at most one --name")
        result = check_schema_jsonl_file(args.check_jsonl, args.name[0] if args.name else None)
        rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(rendered, encoding="utf-8")
            print(f"wrote {args.out}")
        else:
            print(rendered, end="")
        return 0 if result["passed"] else 1
    if args.write_dir:
        written = write_schema_bundle(args.write_dir, args.name or None, force=args.force)
        print(f"wrote {len(written)} schema file(s) to {args.write_dir}")
        return 0
    if args.out:
        if len(args.name) != 1:
            raise SchemaRegistryError("--out requires exactly one --name")
        _write_json(Path(args.out), load_schema(args.name[0]))
        print(f"wrote {args.out}")
        return 0
    if args.name:
        if len(args.name) != 1:
            raise SchemaRegistryError("printing to stdout requires exactly one --name")
        print(json.dumps(load_schema(args.name[0]), indent=2, sort_keys=True))
        return 0

    for record in list_schema_records():
        print(f"{record['name']}\t{record['artifact_schema_version']}\t{record['filename']}")
    return 0


def cmd_evidence_coverage(args: argparse.Namespace) -> int:
    coverage = build_evidence_coverage(
        args.runs,
        preserve_paths=args.preserve_paths,
        min_failed_rule_evidence_rate=args.min_failed_rule_evidence_rate,
        min_critical_failed_rule_evidence_rate=args.min_critical_failed_rule_evidence_rate,
        min_event_evidence_refs=args.min_event_evidence_refs,
        max_failed_rules_without_evidence=args.max_failed_rules_without_evidence,
        max_critical_failed_rules_without_evidence=args.max_critical_failed_rules_without_evidence,
        require_rule_evidence=args.require_rule_evidence,
    )
    rendered = json.dumps(coverage, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(rendered, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(rendered, end="")
    return 0 if coverage["passed"] else 1


def cmd_trace_observability(args: argparse.Namespace) -> int:
    observability = build_trace_observability(
        args.runs,
        preserve_paths=args.preserve_paths,
        min_average_events=args.min_average_events,
        min_event_type_count=args.min_event_type_count,
        min_tool_or_api_run_rate=args.min_tool_or_api_run_rate,
        max_empty_final_answers=args.max_empty_final_answers,
        require_event_types=args.require_event_type,
    )
    rendered = json.dumps(observability, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(rendered, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(rendered, end="")
    return 0 if observability["passed"] else 1


def cmd_repair_queue(args: argparse.Namespace) -> int:
    queue = build_repair_queue(
        args.runs,
        preserve_paths=args.preserve_paths,
        only_critical=args.only_critical,
    )
    rendered = json.dumps(queue, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(rendered, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(rendered, end="")
    return 0


def cmd_evidence_bundle(args: argparse.Namespace) -> int:
    bundle = build_evidence_bundle(
        out_path=args.out,
        runs_dir=args.runs,
        suite_summary_path=args.suite_summary,
        scenario_quality_path=args.scenario_quality,
        evidence_coverage_path=args.evidence_coverage,
        trace_observability_path=args.trace_observability,
        repair_queue_path=args.repair_queue,
        validation_path=args.validation,
        eval_summary_path=args.eval_summary,
        training_export_dir=args.training_export,
        compare_export_dir=args.compare_export,
        review_export_dir=args.review_export,
        reviewed_export_dir=args.reviewed_export,
        review_calibration_path=args.review_calibration,
        live_smoke_summary_path=args.live_smoke_summary,
        serving_lifecycle_path=args.serving_lifecycle,
        trainer_preflight_path=args.trainer_preflight,
        trainer_launch_check_path=args.trainer_launch_check,
        trainer_archive_path=args.trainer_archive,
        trainer_archive_check_path=args.trainer_archive_check,
        trainer_consumer_plan_path=args.trainer_consumer_plan,
        trainer_wrapper_dry_run_path=args.trainer_wrapper_dry_run,
        agentic_training_result_path=args.agentic_training_result,
        harness_manifest_paths=args.harness_manifest,
        harness_result_paths=args.harness_result,
        gate_paths=args.gate,
        require_harness=args.require_harness,
        require_gate=args.require_gate,
        preserve_paths=args.preserve_paths,
    )
    _write_json(Path(args.out), bundle)
    print(f"wrote {args.out}")
    return 0 if bundle["passed"] else 1


def cmd_improvement_plan(args: argparse.Namespace) -> int:
    plan = build_improvement_plan(
        out_path=args.out,
        evidence_bundle_path=args.evidence_bundle,
        repair_queue_path=args.repair_queue,
        training_export_dir=args.training_export,
        runs_dir=args.runs,
        eval_summary_path=args.eval_summary,
        preserve_paths=args.preserve_paths,
    )
    _write_json(Path(args.out), plan)
    print(f"wrote {args.out}")
    return 0


def cmd_improvement_ledger(args: argparse.Namespace) -> int:
    ledger = build_improvement_ledger(
        args.plan,
        out_path=args.out,
        preserve_paths=args.preserve_paths,
    )
    rendered = json.dumps(ledger, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(rendered, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(rendered, end="")
    return 0


def cmd_action_ledger(args: argparse.Namespace) -> int:
    ledger = build_action_ledger(
        args.bundle,
        out_path=args.out,
        preserve_paths=args.preserve_paths,
    )
    rendered = json.dumps(ledger, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(rendered, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(rendered, end="")
    return 0


def cmd_promotion_cards(args: argparse.Namespace) -> int:
    cards = build_promotion_cards(
        out_dir=args.out,
        candidate_id=args.candidate_id,
        dataset_id=args.dataset_id,
        model_source=args.model_source,
        license_status=args.license_status,
        evidence_bundle_path=args.evidence_bundle,
        training_export_path=args.training_export,
        compare_gate_path=args.compare_gate,
        redaction_check_path=args.redaction_check,
        safety_gate_path=args.safety_gate,
        preserve_paths=args.preserve_paths,
        metadata=_metadata_options(args.metadata),
    )
    print(
        f"{'READY' if cards['passed'] else 'BLOCKED'} promotion-cards "
        f"checks={cards['check_count'] - cards['failed_check_count']}/{cards['check_count']} "
        f"out={args.out}"
    )
    return 0 if cards["passed"] else 1


def cmd_promotion_alias_apply(args: argparse.Namespace) -> int:
    validation_summary = validate_artifacts(promotion_decision_paths=[args.promotion_decision], strict=True)
    receipt = apply_promotion_aliases(
        registry_path=args.registry,
        promotion_decision_path=args.promotion_decision,
        out_path=args.out,
        promotion_decision_validation=validation_summary,
        preserve_paths=args.preserve_paths,
        metadata=_metadata_options(args.metadata),
    )
    print(
        f"{'APPLIED' if receipt['passed'] else 'BLOCKED'} promotion-alias-apply "
        f"checks={receipt['check_count'] - receipt['failed_check_count']}/{receipt['check_count']} "
        f"out={args.out}"
    )
    return 0 if receipt["passed"] else 1


def cmd_promotion_rollback_receipt(args: argparse.Namespace) -> int:
    receipt = build_promotion_rollback_receipt(
        registry_path=args.registry,
        rollback_id=args.rollback_id,
        champion_id=args.champion_id,
        out_path=args.out,
        preserve_paths=args.preserve_paths,
        metadata=_metadata_options(args.metadata),
    )
    print(
        f"{'READY' if receipt['passed'] else 'BLOCKED'} promotion-rollback-receipt "
        f"checks={receipt['check_count'] - receipt['failed_check_count']}/{receipt['check_count']} "
        f"out={args.out}"
    )
    return 0 if receipt["passed"] else 1


def cmd_promotion_release_record(args: argparse.Namespace) -> int:
    validation_summary = validate_artifacts(
        promotion_decision_paths=[args.promotion_decision],
        promotion_cards_paths=[args.promotion_cards],
        promotion_alias_apply_paths=[args.promotion_alias_apply],
        strict=True,
    )
    record = build_promotion_release_record(
        release_id=args.release_id,
        promotion_decision_path=args.promotion_decision,
        promotion_cards_path=args.promotion_cards,
        promotion_alias_apply_path=args.promotion_alias_apply,
        rollback_metadata_path=args.rollback_metadata,
        compare_gate_path=args.compare_gate,
        release_notes_path=args.release_notes,
        out_path=args.out,
        promotion_policy_path=args.promotion_policy,
        artifact_validation=validation_summary,
        preserve_paths=args.preserve_paths,
        metadata=_metadata_options(args.metadata),
    )
    print(
        f"{'READY' if record['passed'] else 'BLOCKED'} promotion-release-record "
        f"checks={record['check_count'] - record['failed_check_count']}/{record['check_count']} "
        f"out={args.out}"
    )
    return 0 if record["passed"] else 1


def cmd_promotion_decision(args: argparse.Namespace) -> int:
    decision = build_promotion_decision(
        candidate_id=args.candidate_id,
        champion_id=args.champion_id,
        rollback_id=args.rollback_id,
        candidate_class=args.candidate_class,
        champion_class=args.champion_class,
        out_path=args.out,
        evidence_bundle_path=args.evidence_bundle,
        promotion_ledger_gate_path=args.promotion_ledger_gate,
        compare_gate_path=args.compare_gate,
        trainer_launch_check_path=args.trainer_launch_check,
        model_registry_entry_path=args.model_registry_entry,
        agentic_training_result_path=args.agentic_training_result,
        model_card_path=args.model_card,
        dataset_card_path=args.dataset_card,
        rollback_metadata_path=args.rollback_metadata,
        license_review_path=args.license_review,
        redaction_check_path=args.redaction_check,
        safety_gate_path=args.safety_gate,
        serving_profile_path=args.serving_profile,
        serving_report_path=args.serving_report,
        promotion_policy_path=args.promotion_policy,
        preserve_paths=args.preserve_paths,
        metadata=_metadata_options(args.metadata),
    )
    rendered = json.dumps(decision, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(rendered, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(rendered, end="")
    return 0 if decision["passed"] else 1


def cmd_promotion_ledger(args: argparse.Namespace) -> int:
    ledger = build_promotion_ledger(
        args.decision_gate,
        out_path=args.out,
        preserve_paths=args.preserve_paths,
    )
    rendered = json.dumps(ledger, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(rendered, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(rendered, end="")
    return 0


def cmd_promotion_archive(args: argparse.Namespace) -> int:
    archive = build_promotion_archive(
        out_dir=args.out,
        promotion_ledger_path=args.promotion_ledger,
        promotion_ledger_gate_path=args.promotion_ledger_gate,
        decision_gate_paths=args.decision_gate,
        promotion_release_record_paths=args.promotion_release_record,
        require_self_contained=args.require_self_contained,
        force=args.force,
        preserve_paths=args.preserve_paths,
    )
    print(
        f"{'READY' if archive['passed'] else 'INCOMPLETE'} promotion-archive "
        f"artifacts={archive['metrics']['artifact_count']} "
        f"missing={archive['metrics']['missing_count']} out={args.out}"
    )
    return 0 if archive["passed"] else 1


def cmd_trainer_archive(args: argparse.Namespace) -> int:
    archive = build_trainer_archive(
        out_dir=args.out,
        preflight_path=args.preflight,
        launch_check_path=args.launch_check,
        require_self_contained=args.require_self_contained,
        force=args.force,
        preserve_paths=args.preserve_paths,
    )
    print(
        f"{'READY' if archive['passed'] else 'BLOCKED'} trainer-archive "
        f"artifacts={archive['metrics']['artifact_count']} "
        f"missing={archive['metrics']['missing_count']} out={args.out}"
    )
    return 0 if archive["passed"] else 1


def cmd_trainer_archive_check(args: argparse.Namespace) -> int:
    validation_summary = validate_artifacts(trainer_archive_paths=[args.archive], strict=args.strict)
    check = build_trainer_archive_check(
        archive_path=args.archive,
        external_code_root=args.external_code_root,
        validation_summary=validation_summary,
        preserve_paths=args.preserve_paths,
    )
    _write_json(Path(args.out), check)
    print(
        f"{'READY' if check['passed'] else 'BLOCKED'} trainer-archive-check "
        f"external_paths={check['metrics']['external_command_path_count']} "
        f"missing_external={check['metrics']['missing_external_code_count']} out={args.out}"
    )
    return 0 if check["passed"] else 1


def cmd_trainer_consumer_plan(args: argparse.Namespace) -> int:
    archive_check_path = Path(args.archive_check)
    archive_check = _read_json(archive_check_path)
    validation_summary = validate_artifacts(trainer_archive_check_paths=[archive_check_path], strict=args.strict)
    plan = build_trainer_consumer_plan(
        out_path=args.out,
        archive_check_path=archive_check_path,
        archive_check=archive_check,
        validation_summary=validation_summary,
        preserve_paths=args.preserve_paths,
    )
    _write_json(Path(args.out), plan)
    print(
        f"{'READY' if plan['passed'] else 'BLOCKED'} trainer-consumer-plan "
        f"inputs={plan['metrics']['trainer_input_count']} "
        f"external_code={plan['metrics']['external_code_file_count']} out={args.out}"
    )
    return 0 if plan["passed"] else 1


def cmd_export_review(args: argparse.Namespace) -> int:
    manifest = export_review_queue(
        args.runs,
        args.out,
        only_failed=args.only_failed,
        preserve_paths=args.preserve_paths,
    )
    print(f"wrote review queue {args.out} items={manifest['item_count']}")
    return 0


def cmd_apply_review(args: argparse.Namespace) -> int:
    manifest = apply_review_labels(
        args.review_export,
        args.out,
        labels_path=args.labels,
        max_pairs_per_family=args.max_pairs_per_family,
        preserve_paths=args.preserve_paths,
    )
    print(f"wrote reviewed export {args.out} labels={manifest['reviewed_label_count']}")
    return 0


def cmd_review_calibration(args: argparse.Namespace) -> int:
    reviewed_dir = Path(args.reviewed_export)
    validation_summary = (
        validate_artifacts(reviewed_export_dir=reviewed_dir, strict=args.strict_validation)
        if not args.skip_validation
        else None
    )
    calibration = build_review_calibration(
        reviewed_dir,
        min_agreement_rate=args.min_agreement_rate,
        max_disagreements=args.max_disagreements,
        max_false_positives=args.max_false_positives,
        max_false_negatives=args.max_false_negatives,
        min_comparable_labels=args.min_comparable_labels,
        validation_summary=validation_summary,
        require_valid_export=not args.skip_validation,
        preserve_paths=args.preserve_paths,
    )
    _write_json(Path(args.out), calibration)
    metrics = calibration["metrics"]
    print(
        "wrote review calibration "
        f"agreement_rate={metrics['agreement_rate']} "
        f"disagreements={metrics['disagreement_count']} out={args.out}"
    )
    return 0 if calibration["passed"] else 1


def cmd_draft_scenario(args: argparse.Namespace) -> int:
    if args.run:
        source_path = Path(args.run) / "normalized_trace.json"
        trace = _read_json(source_path)
        trace_format = "normalized_json"
        source_label = str(source_path)
    else:
        source_path = Path(args.trace)
        trace = normalize_trace(source_path, args.format)
        trace_format = args.format
        source_label = str(source_path)

    scenario_id = safe_scenario_id(args.id or Path(args.run or args.trace).stem)
    scenario = draft_scenario(
        trace,
        scenario_id=scenario_id,
        title=args.title or f"Draft: {title_from_id(scenario_id)}",
        prompt=args.prompt or "TODO: describe the intended task for this run.",
        trace_path=source_path,
        trace_format=trace_format,
        out_path=args.out,
        max_actions=args.max_actions,
        preserve_paths=args.preserve_paths,
    )
    _write_json(Path(args.out), scenario)
    source_score = score_draft(scenario, trace)
    print(
        f"wrote {args.out} from {source_label} "
        f"source_score={source_score['score']} source_passed={str(source_score['passed']).lower()}"
    )
    return 0


def cmd_gate_suite(args: argparse.Namespace) -> int:
    suite_summary = _read_json(Path(args.suite_summary))
    options = _gate_suite_options(args)
    result = evaluate_suite_gate(
        suite_summary,
        suite_summary_path=_display_path(Path(args.suite_summary), args.preserve_paths),
        min_pass_rate=options["min_pass_rate"],
        min_average_score=options["min_average_score"],
        max_failed=options["max_failed"],
        max_errors=options["max_errors"],
        max_critical_failures=options["max_critical_failures"],
        forbid_failed_rules=options["forbid_failed_rules"],
        forbid_critical_rules=options["forbid_critical_rules"],
        task_family_gates=options["task_family_gates"],
    )
    if options["policy_path"]:
        result["policy"] = _gate_policy_summary(options)
    rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(rendered, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(rendered, end="")
    return 0 if result["passed"] else 1


def cmd_gate_export(args: argparse.Namespace) -> int:
    export_dir = Path(args.training_export)
    metrics_path = export_dir / "dataset_metrics.json"
    dataset_metrics = _read_json(metrics_path)
    options = _training_gate_options(args)
    validation_summary = (
        validate_artifacts(training_export_dir=export_dir, strict=options["strict_validation"])
        if options["require_valid_export"]
        else None
    )
    result = evaluate_training_gate(
        dataset_metrics,
        training_export_path=_display_path(export_dir, args.preserve_paths),
        min_episodes=options["min_episodes"],
        min_pass_rate=options["min_pass_rate"],
        min_average_score=options["min_average_score"],
        min_preferences=options["min_preferences"],
        min_sft=options["min_sft"],
        min_dpo=options["min_dpo"],
        min_reward_model=options["min_reward_model"],
        min_step_rewards=options["min_step_rewards"],
        min_task_completion_configured=options["min_task_completion_configured"],
        min_task_completion_complete=options["min_task_completion_complete"],
        max_task_completion_incomplete=options["max_task_completion_incomplete"],
        min_task_completion_check_pass_rate=options["min_task_completion_check_pass_rate"],
        min_source_fingerprint_rate=options["min_source_fingerprint_rate"],
        max_unverified_source_fingerprints=options["max_unverified_source_fingerprints"],
        min_trainer_view_source_fingerprint_rate=options["min_trainer_view_source_fingerprint_rate"],
        max_unverified_trainer_view_source_fingerprints=options["max_unverified_trainer_view_source_fingerprints"],
        min_trace_average_events=options["min_trace_average_events"],
        min_trace_event_type_count=options["min_trace_event_type_count"],
        min_trace_final_answer_rate=options["min_trace_final_answer_rate"],
        min_trace_tool_or_api_rate=options["min_trace_tool_or_api_rate"],
        max_trace_empty_final_answers=options["max_trace_empty_final_answers"],
        max_trace_risk_count=options["max_trace_risk_count"],
        min_split_task_families=options["min_split_task_families"],
        min_train_episodes=options["min_train_episodes"],
        min_validation_episodes=options["min_validation_episodes"],
        min_test_episodes=options["min_test_episodes"],
        require_family_exclusive_splits=options["require_family_exclusive_splits"],
        max_quality_flags=options["max_quality_flags"],
        forbid_quality_flags=options["forbid_quality_flags"],
        forbid_quality_severities=options["forbid_quality_severities"],
        require_task_families=options["require_task_families"],
        require_trace_event_types=options["require_trace_event_types"],
        task_family_gates=options["task_family_gates"],
        validation_summary=validation_summary,
        require_valid_export=options["require_valid_export"],
    )
    if options["policy_path"]:
        result["policy"] = _training_gate_policy_summary(options)
    rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(rendered, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(rendered, end="")
    return 0 if result["passed"] else 1


def cmd_gate_reviewed(args: argparse.Namespace) -> int:
    reviewed_dir = Path(args.reviewed_export)
    manifest = _read_json(reviewed_dir / "manifest.json")
    options = _reviewed_gate_options(args)
    validation_summary = (
        validate_artifacts(reviewed_export_dir=reviewed_dir, strict=options["strict_validation"])
        if options["require_valid_export"]
        else None
    )
    result = evaluate_reviewed_gate(
        manifest,
        reviewed_export_path=_display_path(reviewed_dir, args.preserve_paths),
        min_reviewed_labels=options["min_reviewed_labels"],
        min_accepted=options["min_accepted"],
        min_rejected=options["min_rejected"],
        min_sft=options["min_sft"],
        min_reward_model=options["min_reward_model"],
        min_preferences=options["min_preferences"],
        min_dpo=options["min_dpo"],
        min_high_confidence_labels=options["min_high_confidence_labels"],
        min_medium_or_high_confidence_labels=options["min_medium_or_high_confidence_labels"],
        max_needs_review=options["max_needs_review"],
        max_low_confidence_labels=options["max_low_confidence_labels"],
        max_unknown_confidence_labels=options["max_unknown_confidence_labels"],
        forbid_labels=options["forbid_labels"],
        require_task_families=options["require_task_families"],
        validation_summary=validation_summary,
        require_valid_export=options["require_valid_export"],
    )
    if options["policy_path"]:
        result["policy"] = _reviewed_gate_policy_summary(options)
    rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(rendered, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(rendered, end="")
    return 0 if result["passed"] else 1


def cmd_gate_compare_export(args: argparse.Namespace) -> int:
    compare_dir = Path(args.compare_export)
    manifest = _read_json(compare_dir / "manifest.json")
    pairs = _read_jsonl(compare_dir / "improvement_pairs.jsonl")
    options = _compare_gate_options(args)
    validation_summary = (
        validate_artifacts(compare_export_dir=compare_dir, strict=options["strict_validation"])
        if options["require_valid_export"]
        else None
    )
    result = evaluate_compare_gate(
        manifest,
        pairs,
        compare_export_path=_display_path(compare_dir, args.preserve_paths),
        min_pairs=options["min_pairs"],
        min_dpo=options["min_dpo"],
        min_candidate_wins=options["min_candidate_wins"],
        min_task_completion_improvements=options["min_task_completion_improvements"],
        max_baseline_wins=options["max_baseline_wins"],
        max_task_completion_regressions=options["max_task_completion_regressions"],
        max_skipped_pairs=options["max_skipped_pairs"],
        max_contract_drifts=options["max_contract_drifts"],
        max_unverified_contracts=options["max_unverified_contracts"],
        require_scenarios=options["require_scenarios"],
        require_candidate_win_scenarios=options["require_candidate_win_scenarios"],
        require_task_completion_improvement_scenarios=options["require_task_completion_improvement_scenarios"],
        forbid_regression_scenarios=options["forbid_regression_scenarios"],
        forbid_task_completion_regression_scenarios=options["forbid_task_completion_regression_scenarios"],
        require_rule_fixes=options["require_rule_fixes"],
        forbid_rule_regressions=options["forbid_rule_regressions"],
        forbid_new_critical_failures=options["forbid_new_critical_failures"],
        task_family_gates=options["task_family_gates"],
        validation_summary=validation_summary,
        require_valid_export=options["require_valid_export"],
    )
    if options["policy_path"]:
        result["policy"] = _compare_gate_policy_summary(options)
    rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(rendered, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(rendered, end="")
    return 0 if result["passed"] else 1


def cmd_gate_action_ledger(args: argparse.Namespace) -> int:
    ledger = _read_json(Path(args.action_ledger))
    options = _action_ledger_gate_options(args)
    output_path = Path(args.out) if args.out else None
    result = evaluate_action_ledger_gate(
        ledger,
        action_ledger_path=_display_path_for_output_source(
            Path(args.action_ledger),
            output_path,
            args.preserve_paths,
        ),
        min_bundles=options["min_bundles"],
        max_open_actions=options["max_open_actions"],
        max_new_actions=options["max_new_actions"],
        max_recurring_actions=options["max_recurring_actions"],
        min_resolved_actions=options["min_resolved_actions"],
        forbid_open_priorities=options["forbid_open_priorities"],
        forbid_open_actions=options["forbid_open_actions"],
        require_resolved_actions=options["require_resolved_actions"],
    )
    if options["policy_path"]:
        result["policy"] = _action_ledger_gate_policy_summary(options)
    rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(rendered, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(rendered, end="")
    return 0 if result["passed"] else 1


def cmd_gate_improvement_ledger(args: argparse.Namespace) -> int:
    ledger = _read_json(Path(args.improvement_ledger))
    options = _improvement_ledger_gate_options(args)
    result = evaluate_improvement_ledger_gate(
        ledger,
        improvement_ledger_path=_display_path(Path(args.improvement_ledger), args.preserve_paths),
        min_plans=options["min_plans"],
        max_open_work_items=options["max_open_work_items"],
        max_new_work_items=options["max_new_work_items"],
        max_recurring_work_items=options["max_recurring_work_items"],
        min_resolved_work_items=options["min_resolved_work_items"],
        max_critical_open_work_items=options["max_critical_open_work_items"],
        max_high_open_work_items=options["max_high_open_work_items"],
        forbid_open_priorities=options["forbid_open_priorities"],
        forbid_open_categories=options["forbid_open_categories"],
        forbid_open_work_keys=options["forbid_open_work_keys"],
        require_open_work_keys=options["require_open_work_keys"],
        require_resolved_work_keys=options["require_resolved_work_keys"],
    )
    if options["policy_path"]:
        result["policy"] = _improvement_ledger_gate_policy_summary(options)
    rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(rendered, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(rendered, end="")
    return 0 if result["passed"] else 1


def cmd_gate_promotion_ledger(args: argparse.Namespace) -> int:
    ledger = _read_json(Path(args.promotion_ledger))
    options = _promotion_ledger_gate_options(args)
    output_path = Path(args.out) if args.out else None
    result = evaluate_promotion_ledger_gate(
        ledger,
        promotion_ledger_path=_display_path_for_output_source(
            Path(args.promotion_ledger),
            output_path,
            args.preserve_paths,
        ),
        min_decisions=options["min_decisions"],
        min_allowed_count=options["min_allowed_count"],
        max_blocked_count=options["max_blocked_count"],
        max_blocked_rate=options["max_blocked_rate"],
        min_consecutive_allowed=options["min_consecutive_allowed"],
        max_consecutive_blocked=options["max_consecutive_blocked"],
        max_failed_decisions=options["max_failed_decisions"],
        require_latest_recommendation=options["require_latest_recommendation"],
        require_latest_passed=options["require_latest_passed"],
        require_source_recommendations=options["require_source_recommendations"],
        forbid_source_recommendations=options["forbid_source_recommendations"],
    )
    if options["policy_path"]:
        result["policy"] = _promotion_ledger_gate_policy_summary(options)
    rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(rendered, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(rendered, end="")
    return 0 if result["passed"] else 1


def cmd_gate_decision(args: argparse.Namespace) -> int:
    artifact = _read_json(Path(args.artifact))
    output_path = Path(args.out) if args.out else None
    result = evaluate_decision_gate(
        artifact,
        artifact_path=Path(args.artifact),
        artifact_display_path=_display_path_for_output_source(
            Path(args.artifact),
            output_path,
            args.preserve_paths,
        ),
        expect_recommendation=args.expect_recommendation,
        expect_readiness=args.expect_readiness,
        require_passed=args.require_passed,
        preserve_paths=args.preserve_paths,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(rendered, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(rendered, end="")
    return 0 if result["passed"] else 1


def cmd_trainer_preflight(args: argparse.Namespace) -> int:
    metadata = _metadata_options(args.metadata)
    preflight = build_trainer_preflight(
        out_path=args.out,
        gate_paths=args.gate,
        training_export_dir=args.training_export,
        compare_export_dir=args.compare_export,
        reviewed_export_dir=args.reviewed_export,
        evidence_bundle_path=args.evidence_bundle,
        agentic_training_plan_path=args.agentic_training_plan,
        validation_summary_paths=args.validation,
        require_gates=args.require_gate,
        required_dataset_versions=args.require_dataset_version,
        trainer_command=args.trainer_command,
        allow_unvalidated_gates=args.allow_unvalidated_gates,
        preserve_paths=args.preserve_paths,
        metadata=metadata,
    )
    _write_json(Path(args.out), preflight)
    print(
        f"{'READY' if preflight['passed'] else 'BLOCKED'} trainer-preflight "
        f"gates={preflight['passed_gate_count']}/{preflight['gate_count']} out={args.out}"
    )
    return 0 if preflight["passed"] else 1


def cmd_trainer_launch_check(args: argparse.Namespace) -> int:
    preflight_path = Path(args.preflight)
    preflight = _read_json(preflight_path)
    if not isinstance(preflight, dict):
        raise TrainerPreflightError(f"trainer preflight must contain a JSON object: {preflight_path}")
    validation_summary = validate_artifacts(trainer_preflight_paths=[preflight_path], strict=args.strict)
    launch_check = build_trainer_launch_check(
        preflight_path=preflight_path,
        preflight=preflight,
        validation_summary=validation_summary,
        require_gates=args.require_gate,
        required_dataset_versions=args.require_dataset_version,
        require_metadata=_metadata_options(args.require_metadata),
        preserve_paths=args.preserve_paths,
    )
    if args.out:
        _write_json(Path(args.out), launch_check)
    if args.print_command and launch_check["passed"]:
        print(launch_check["approved_command"]["shell"])
    elif args.out:
        print(
            f"{'READY' if launch_check['passed'] else 'BLOCKED'} trainer-launch-check "
            f"checks={launch_check['check_count'] - launch_check['failed_check_count']}/{launch_check['check_count']} "
            f"out={args.out}"
        )
    else:
        print(json.dumps(launch_check, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if launch_check["passed"] else 1


def cmd_export_rl(args: argparse.Namespace) -> int:
    metadata = _metadata_options(args.metadata)
    manifest = export_rl_dataset(
        args.runs,
        args.out,
        reward_scale=args.reward_scale,
        min_score_gap=args.min_score_gap,
        max_pairs_per_family=args.max_pairs_per_family,
        preserve_paths=args.preserve_paths,
        metadata=metadata,
    )
    print(
        "wrote RL export "
        f"episodes={manifest['episode_count']} rewards={manifest['reward_count']} "
        f"step_rewards={manifest['step_reward_count']} "
        f"preferences={manifest['preference_count']} failure_modes={manifest['failure_mode_count']} "
        f"sft={manifest['sft_count']} dpo={manifest['dpo_count']} "
        f"reward_model={manifest['reward_model_count']} "
        f"quality_flags={manifest['quality_flag_count']} out={args.out}"
    )
    return 0


def cmd_export_compare_rl(args: argparse.Namespace) -> int:
    metadata = _metadata_options(args.metadata)
    manifest = export_compare_rl_dataset(
        args.baseline,
        args.candidate,
        args.out,
        reward_scale=args.reward_scale,
        min_score_gap=args.min_score_gap,
        contract_scope=args.contract_scope,
        preserve_paths=args.preserve_paths,
        metadata=metadata,
    )
    print(
        "wrote compare RL export "
        f"pairs={manifest['pair_count']} dpo={manifest['dpo_count']} "
        f"candidate_wins={manifest['candidate_win_count']} "
        f"baseline_wins={manifest['baseline_win_count']} out={args.out}"
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flightrecorder", description="Hermes Autonomy Flight Recorder")
    subparsers = parser.add_subparsers(dest="command", required=True)

    normalize = subparsers.add_parser("normalize", help="Normalize a Hermes trace artifact")
    normalize.add_argument("--trace", required=True)
    normalize.add_argument("--format", default="auto", choices=TRACE_FORMAT_CHOICES)
    normalize.add_argument("--out", required=True)
    normalize.add_argument("--secret-pattern", action="append", default=[], help="Additional regex to redact from normalized output")
    normalize.add_argument("--no-redact", action="store_true", help="Write raw normalized trace without redaction")
    normalize.set_defaults(func=cmd_normalize)

    score = subparsers.add_parser("score", help="Score a normalized trace against a scenario")
    score.add_argument("--scenario", required=True)
    score.add_argument("--trace", required=True)
    score.add_argument("--state", help="Optional JSON state snapshot for required_state assertions")
    score.add_argument("--before-state", help="Optional JSON pre-run state snapshot for required_state_transitions assertions")
    score.add_argument("--out", required=True)
    score.add_argument("--junit-out", help="Also write a JUnit XML score report")
    score.add_argument("--markdown-out", help="Also write a Markdown score summary")
    score.set_defaults(func=cmd_score)

    report = subparsers.add_parser("report", help="Render a static HTML report")
    report.add_argument("--scenario", required=True)
    report.add_argument("--trace", required=True)
    report.add_argument("--score", required=True)
    report.add_argument("--state-diff", help="Optional hfr.state_diff.v1 JSON to render as a state-change table")
    report.add_argument("--out", required=True)
    report.set_defaults(func=cmd_report)

    digest = subparsers.add_parser("digest", help="Write a compact per-run evidence digest")
    digest.add_argument("--run", help="Existing run directory containing normalized_trace.json and scorecard.json")
    digest.add_argument("--scenario", help="Scenario JSON; optional with --run when lineage/scorecard metadata is enough")
    digest.add_argument("--trace", help="Normalized trace JSON; defaults to <run>/normalized_trace.json with --run")
    digest.add_argument("--score", help="Scorecard JSON; defaults to <run>/scorecard.json with --run")
    digest.add_argument("--state-diff", help="Optional hfr.state_diff.v1 JSON; defaults to <run>/state_diff.json if present")
    digest.add_argument("--out", help="Digest JSON output path; defaults to <run>/run_digest.json with --run")
    digest.add_argument("--markdown-out", help="Optional Markdown digest output path")
    digest.set_defaults(func=cmd_digest)

    capture_state = subparsers.add_parser(
        "capture-state",
        help="Capture a JSON state snapshot from local evidence sources",
    )
    capture_state.add_argument("--out", required=True, help="State snapshot JSON output path")
    capture_state.add_argument(
        "--file",
        action="append",
        default=[],
        type=_key_path_arg,
        metavar="KEY=PATH",
        help="Capture file existence, size, and sha256; may be repeated",
    )
    capture_state.add_argument(
        "--dir",
        dest="directory",
        action="append",
        default=[],
        type=_key_path_arg,
        metavar="KEY=PATH",
        help="Capture a directory listing; may be repeated",
    )
    capture_state.add_argument(
        "--json",
        dest="json_source",
        action="append",
        default=[],
        type=_key_path_arg,
        metavar="KEY=PATH",
        help="Import a JSON file under json.KEY; may be repeated",
    )
    capture_state.add_argument(
        "--set",
        dest="observation",
        action="append",
        default=[],
        type=_state_set_arg,
        metavar="PATH=VALUE",
        help="Set an observed value under observations using a dot path; VALUE may be JSON",
    )
    capture_state.add_argument("--include-file-text", action="store_true", help="Include UTF-8 file text for --file sources")
    capture_state.add_argument(
        "--max-text-chars",
        type=_non_negative_int_arg,
        default=4096,
        help="Maximum text characters captured per --file when --include-file-text is set",
    )
    capture_state.add_argument(
        "--max-dir-entries",
        type=_non_negative_int_arg,
        default=200,
        help="Maximum direct entries captured per --dir",
    )
    capture_state.add_argument("--secret-pattern", action="append", default=[], help="Regex pattern to redact from captured state")
    capture_state.add_argument("--preserve-paths", action="store_true", help="Allow absolute source paths in the snapshot")
    capture_state.set_defaults(func=cmd_capture_state)

    verify_state = subparsers.add_parser(
        "verify-state",
        help="Capture a JSON state snapshot from read-only external verifier adapters",
    )
    verify_state.add_argument("--config", required=True, help="Verifier config JSON path")
    verify_state.add_argument("--out", required=True, help="State snapshot JSON output path")
    verify_state.add_argument("--secret-pattern", action="append", default=[], help="Regex pattern to redact from captured state")
    verify_state.add_argument("--preserve-paths", action="store_true", help="Allow absolute source paths in the snapshot")
    verify_state.set_defaults(func=cmd_verify_state)

    state_validators = subparsers.add_parser(
        "state-validators",
        help="List external monitor targets or compile state-validator configs into scenario assertions",
    )
    state_validators.add_argument("--list", action="store_true", help="List monitorable external tool/state areas")
    state_validators.add_argument("--config", help="State-validator config JSON path")
    state_validators.add_argument("--out", help="JSON output path; prints to stdout when omitted")
    state_validators.add_argument("--markdown-out", help="Optional Markdown monitor catalog output path with --list")
    state_validators.set_defaults(func=cmd_state_validators)

    diff_state = subparsers.add_parser(
        "diff-state",
        help="Write a deterministic hfr.state_diff.v1 artifact from before/after state snapshots",
    )
    diff_state.add_argument("--before", required=True, help="Pre-run state snapshot JSON")
    diff_state.add_argument("--after", required=True, help="Post-run state snapshot JSON")
    diff_state.add_argument("--out", required=True)
    diff_state.add_argument(
        "--max-changes",
        type=_non_negative_int_arg,
        default=200,
        help="Maximum number of changed paths to include while still counting all changes",
    )
    diff_state.add_argument("--secret-pattern", action="append", default=[], help="Additional regex to redact from the diff")
    diff_state.set_defaults(func=cmd_diff_state)

    run = subparsers.add_parser("run", help="Normalize, score, and report in one command")
    run.add_argument("--scenario", required=True)
    run.add_argument("--trace")
    run.add_argument("--state", help="Optional JSON state snapshot for required_state assertions")
    run.add_argument("--before-state", help="Optional JSON pre-run state snapshot for required_state_transitions assertions")
    run.add_argument("--format", default="auto", choices=TRACE_FORMAT_CHOICES)
    run.add_argument("--out", required=True)
    run.add_argument("--write-sensitive-trace", action="store_true", help="Also write raw_trace.sensitive.json with unredacted evidence")
    run.add_argument("--preserve-paths", action="store_true", help="Allow absolute source paths in generated reports and regression files")
    run.add_argument("--junit-out", help="Also write a JUnit XML score report")
    run.add_argument("--markdown-out", help="Also write a Markdown score summary")
    run.add_argument("--fail-on-score", action="store_true", help="Exit nonzero when the scenario score fails")
    run.set_defaults(func=cmd_run)

    replay = subparsers.add_parser("replay", help="Rerun a scenario from artifact lineage replay metadata")
    replay.add_argument("--lineage", required=True, help="Path to artifact_lineage.json with replay metadata")
    replay.add_argument("--out", required=True, help="Output directory for replayed run artifacts")
    replay.add_argument("--base-dir", help="Base directory for relative replay paths; defaults to current directory")
    replay.add_argument("--format", default="auto", choices=TRACE_FORMAT_CHOICES)
    replay.add_argument("--write-sensitive-trace", action="store_true", help="Also write raw_trace.sensitive.json with unredacted evidence")
    replay.add_argument("--preserve-paths", action="store_true", help="Allow absolute source paths in generated replay artifacts")
    replay.add_argument("--allow-non-self-contained", action="store_true", help="Attempt replay even when replay.self_contained is false")
    replay.add_argument("--fail-on-score", action="store_true", help="Exit nonzero when the replayed score fails")
    replay.set_defaults(func=cmd_replay)

    replay_bundle = subparsers.add_parser("replay-bundle", help="Create a portable replay bundle from artifact lineage")
    replay_bundle.add_argument("--lineage", required=True, help="Path to source artifact_lineage.json")
    replay_bundle.add_argument("--out", required=True, help="Output directory for the portable replay bundle")
    replay_bundle.add_argument("--base-dir", help="Base directory for relative source replay paths; defaults to current directory")
    replay_bundle.add_argument("--force", action="store_true", help="Replace an existing non-empty bundle directory")
    replay_bundle.add_argument("--preserve-paths", action="store_true", help="Allow absolute source paths in generated bundle metadata")
    replay_bundle.set_defaults(func=cmd_replay_bundle)

    run_suite = subparsers.add_parser("run-suite", help="Run a directory of scenarios into a complete evidence bundle")
    run_suite.add_argument("--scenarios", required=True, help="Directory containing scenario JSON files")
    run_suite.add_argument("--out", required=True, help="Output directory for per-scenario run directories and suite artifacts")
    run_suite.add_argument("--pattern", default="*.json", help="Scenario filename glob relative to --scenarios")
    run_suite.add_argument("--recursive", action="store_true", help="Discover scenarios recursively with --pattern")
    run_suite.add_argument("--suite-manifest", help="Eval suite manifest with an explicit scenario_ids list to run")
    run_suite.add_argument("--format", default="auto", choices=TRACE_FORMAT_CHOICES)
    run_suite.add_argument("--summary-out", help="Suite summary JSON output path; defaults to <out>/suite_summary.json")
    run_suite.add_argument("--index-out", help="Report index output path; defaults to <out>/index.html")
    run_suite.add_argument("--no-index", action="store_true", help="Skip writing the report index")
    run_suite.add_argument("--junit", action="store_true", help="Write scorecard.junit.xml inside each run directory")
    run_suite.add_argument("--markdown", action="store_true", help="Write scorecard.md inside each run directory")
    run_suite.add_argument("--export-rl", action="store_true", help="Also export evidence and trainer-ready artifacts for the completed suite")
    run_suite.add_argument("--training-export-out", help="RL export directory; defaults to <out>/training_export")
    run_suite.add_argument("--reward-scale", default="score", choices=["score", "binary", "signed"])
    run_suite.add_argument("--min-score-gap", type=int, default=1)
    run_suite.add_argument("--max-pairs-per-family", type=int, default=0)
    run_suite.add_argument("--validate", action="store_true", help="Also validate generated run and optional training artifacts")
    run_suite.add_argument("--validation-out", help="Validation JSON output path; defaults to <out>/validation.json")
    run_suite.add_argument("--strict", action="store_true", help="Treat validation warnings as validation failure")
    run_suite.add_argument(
        "--evidence-handoff",
        action="store_true",
        help="Also write scenario quality, evidence coverage, trace observability, harness handoff, and evidence bundle artifacts",
    )
    run_suite.add_argument("--write-sensitive-trace", action="store_true", help="Also write raw_trace.sensitive.json for each scenario")
    run_suite.add_argument("--preserve-paths", action="store_true", help="Allow absolute source paths in generated artifacts")
    run_suite.add_argument(
        "--metadata",
        action="append",
        default=[],
        type=_metadata_arg,
        metavar="KEY=VALUE",
        help="Attach experiment metadata to suite and optional training-export artifacts; may be repeated",
    )
    run_suite.add_argument("--fail-on-failed", action="store_true", help="Exit nonzero when any scenario score fails")
    run_suite.set_defaults(func=cmd_run_suite)

    goal3_handoff = subparsers.add_parser(
        "goal3-handoff",
        help="Build a Goal 3 training-data handoff with export, validation, gate, evidence bundle, and preflight artifacts",
    )
    goal3_handoff.add_argument("--scenarios", required=True, help="Directory containing scenario JSON files")
    goal3_handoff.add_argument("--out", required=True, help="Output directory for the reproducible Goal 3 handoff")
    goal3_handoff.add_argument("--pattern", default="*.json", help="Scenario filename glob relative to --scenarios")
    goal3_handoff.add_argument("--recursive", action="store_true", help="Discover scenarios recursively with --pattern")
    goal3_handoff.add_argument("--suite-manifest", help="Eval suite manifest with an explicit scenario_ids list to run")
    goal3_handoff.add_argument("--format", default="auto", choices=TRACE_FORMAT_CHOICES)
    goal3_handoff.add_argument("--policy", help="Versioned training gate policy JSON file")
    goal3_handoff.add_argument("--trainer-command", required=True, help="Trainer command to record in preflight; not executed")
    goal3_handoff.add_argument("--reward-scale", default="score", choices=["score", "binary", "signed"])
    goal3_handoff.add_argument("--min-score-gap", type=int, default=1)
    goal3_handoff.add_argument("--max-pairs-per-family", type=int, default=0)
    goal3_handoff.add_argument("--strict", action="store_true", help="Treat validation warnings as handoff blockers")
    goal3_handoff.add_argument("--force", action="store_true", help="Replace an existing non-empty handoff directory")
    goal3_handoff.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in generated handoff artifacts")
    goal3_handoff.add_argument(
        "--metadata",
        action="append",
        default=[],
        type=_metadata_arg,
        metavar="KEY=VALUE",
        help="Attach metadata to suite, training export, and trainer preflight artifacts; may be repeated",
    )
    goal3_handoff.set_defaults(func=cmd_goal3_handoff)

    index = subparsers.add_parser("index", help="Build an index for generated run reports")
    index.add_argument("--runs", required=True)
    index.add_argument("--out", required=True)
    index.set_defaults(func=cmd_index)

    audit = subparsers.add_parser("audit", help="Summarize run outputs and scan generated artifacts")
    audit.add_argument("--runs", required=True)
    audit.add_argument("--out")
    audit.add_argument("--forbid-text", action="append", default=[], help="Literal text that must not appear in generated artifacts")
    audit.add_argument("--fail-on-leak", action="store_true", help="Exit nonzero if forbidden text is found")
    audit.add_argument("--fail-on-failed", action="store_true", help="Exit nonzero if any scorecard failed")
    audit.set_defaults(func=cmd_audit)

    compare = subparsers.add_parser("compare", help="Compare two scorecards or run directories")
    compare.add_argument("--baseline", required=True, help="Baseline scorecard.json or run directory")
    compare.add_argument("--candidate", required=True, help="Candidate scorecard.json or run directory")
    compare.add_argument("--out", required=True, help="Comparison JSON output path")
    compare.add_argument("--html-out", help="Optional static HTML comparison report")
    compare.add_argument("--fail-on-regression", action="store_true", help="Exit nonzero when the candidate regresses")
    compare.set_defaults(func=cmd_compare)

    compare_suite = subparsers.add_parser("compare-suite", help="Compare two directories of run scorecards")
    compare_suite.add_argument("--baseline", required=True, help="Baseline runs directory")
    compare_suite.add_argument("--candidate", required=True, help="Candidate runs directory")
    compare_suite.add_argument("--out", required=True, help="Suite comparison JSON output path")
    compare_suite.add_argument("--html-out", help="Optional static HTML suite comparison report")
    compare_suite.add_argument("--baseline-label", help="Human-readable baseline label")
    compare_suite.add_argument("--candidate-label", help="Human-readable candidate label")
    compare_suite.add_argument(
        "--contract-scope",
        default="scenario",
        choices=["scenario", "scenario-and-trace"],
        help="Fingerprint contract to compare: scenario for live improvement loops, scenario-and-trace for strict fixture replay",
    )
    compare_suite.add_argument("--fail-on-regression", action="store_true", help="Exit nonzero when the candidate suite regresses")
    compare_suite.add_argument("--fail-on-contract-drift", action="store_true", help="Exit nonzero when paired scenarios drift under --contract-scope")
    compare_suite.add_argument("--fail-on-unverified-contracts", action="store_true", help="Exit nonzero when paired scenarios are missing lineage fingerprints")
    compare_suite.set_defaults(func=cmd_compare_suite)

    trend_suite = subparsers.add_parser("trend-suite", help="Build a longitudinal trend over run-suite summaries")
    trend_suite.add_argument(
        "--suite-summary",
        action="append",
        required=True,
        help="Path to a suite_summary.json in chronological/order-of-comparison order; may be repeated",
    )
    trend_suite.add_argument("--out", required=True, help="Suite trend JSON output path")
    trend_suite.add_argument("--html-out", help="Optional static HTML suite trend report")
    trend_suite.set_defaults(func=cmd_trend_suite)

    check = subparsers.add_parser("check-scenarios", help="Validate scenario definitions before running them")
    check.add_argument("--scenarios", required=True, help="Directory containing scenario files")
    check.add_argument("--pattern", default="*.json", help="Scenario filename glob relative to --scenarios")
    check.add_argument("--recursive", action="store_true", help="Discover scenarios recursively with --pattern")
    check.add_argument("--out", help="Write scenario-check summary JSON to this path")
    check.add_argument("--require-traces", action="store_true", help="Fail when scenarios do not resolve to existing trace files")
    check.add_argument("--strict", action="store_true", help="Treat warnings as failure")
    check.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in generated check output")
    check.set_defaults(func=cmd_check_scenarios)

    quality = subparsers.add_parser("scenario-quality", help="Summarize and gate scenario contract quality")
    quality.add_argument("--scenarios", required=True, help="Directory containing scenario files")
    quality.add_argument("--pattern", default="*.json", help="Scenario filename glob relative to --scenarios")
    quality.add_argument("--recursive", action="store_true", help="Discover scenarios recursively with --pattern")
    quality.add_argument("--require-traces", action="store_true", help="Treat missing trace paths/files as scenario errors")
    quality.add_argument("--out", help="Write scenario-quality summary JSON to this path")
    quality.add_argument("--min-average-score", type=_score_arg, help="Minimum average scenario contract score")
    quality.add_argument("--min-scenario-score", type=_score_arg, help="Minimum allowed score for the weakest valid scenario")
    quality.add_argument("--min-observable-rate", type=_rate_arg, help="Minimum fraction of scenarios with observable assertions")
    quality.add_argument("--max-weak-scenarios", type=_non_negative_int_arg, help="Maximum allowed weak scenario contracts")
    quality.add_argument("--max-final-only-scenarios", type=_non_negative_int_arg, help="Maximum allowed final-answer-only contracts")
    quality.add_argument("--max-missing-traces", type=_non_negative_int_arg, help="Maximum valid scenarios with missing trace files")
    quality.add_argument("--require-task-family", action="append", default=[], help="Fail unless this derived task family is present")
    quality.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in generated quality output")
    quality.set_defaults(func=cmd_scenario_quality)

    validate = subparsers.add_parser("validate", help="Validate generated run and training artifacts")
    validate.add_argument("--run", action="append", default=[], help="Validate one run directory; may be repeated")
    validate.add_argument("--runs", help="Validate every completed run directory inside this runs directory")
    validate.add_argument("--training-export", help="Validate an export-rl output directory")
    validate.add_argument("--compare-export", help="Validate an export-compare-rl output directory")
    validate.add_argument("--review-export", help="Validate an export-review output directory")
    validate.add_argument("--reviewed-export", help="Validate an apply-review output directory")
    validate.add_argument("--evidence-coverage", action="append", default=[], help="Validate one evidence_coverage.json; may be repeated")
    validate.add_argument("--evidence-bundle", action="append", default=[], help="Validate one evidence_bundle.json; may be repeated")
    validate.add_argument("--improvement-plan", action="append", default=[], help="Validate one improvement_plan.json; may be repeated")
    validate.add_argument("--improvement-ledger", action="append", default=[], help="Validate one improvement_ledger.json; may be repeated")
    validate.add_argument(
        "--improvement-ledger-gate",
        action="append",
        default=[],
        help="Validate one improvement_ledger_gate.json; may be repeated",
    )
    validate.add_argument("--action-ledger", action="append", default=[], help="Validate one action_ledger.json; may be repeated")
    validate.add_argument("--action-ledger-gate", action="append", default=[], help="Validate one action_ledger_gate.json; may be repeated")
    validate.add_argument("--decision-gate", action="append", default=[], help="Validate one decision_gate.json; may be repeated")
    validate.add_argument("--promotion-cards", action="append", default=[], help="Validate one promotion-cards directory or manifest; may be repeated")
    validate.add_argument("--promotion-decision", action="append", default=[], help="Validate one promotion_decision.json; may be repeated")
    validate.add_argument("--promotion-alias-apply", action="append", default=[], help="Validate one promotion_alias_apply.json receipt; may be repeated")
    validate.add_argument("--promotion-rollback-receipt", action="append", default=[], help="Validate one promotion rollback receipt; may be repeated")
    validate.add_argument("--promotion-release-record", action="append", default=[], help="Validate one promotion_release_record.json; may be repeated")
    validate.add_argument("--promotion-policy", action="append", default=[], help="Validate one promotion_policy.json; may be repeated")
    validate.add_argument("--promotion-ledger", action="append", default=[], help="Validate one promotion_ledger.json; may be repeated")
    validate.add_argument("--promotion-ledger-gate", action="append", default=[], help="Validate one promotion_ledger_gate.json; may be repeated")
    validate.add_argument("--promotion-archive", action="append", default=[], help="Validate one promotion archive directory or manifest; may be repeated")
    validate.add_argument("--trainer-preflight", action="append", default=[], help="Validate one trainer_preflight.json; may be repeated")
    validate.add_argument(
        "--trainer-launch-check",
        action="append",
        default=[],
        help="Validate one trainer_launch_check.json; may be repeated",
    )
    validate.add_argument("--trainer-archive", action="append", default=[], help="Validate one trainer archive directory or manifest; may be repeated")
    validate.add_argument(
        "--trainer-archive-check",
        action="append",
        default=[],
        help="Validate one trainer_archive_check.json; may be repeated",
    )
    validate.add_argument(
        "--trainer-consumer-plan",
        action="append",
        default=[],
        help="Validate one trainer_consumer_plan.json; may be repeated",
    )
    validate.add_argument(
        "--trainer-wrapper-dry-run",
        action="append",
        default=[],
        help="Validate one trainer_wrapper_dry_run.json; may be repeated",
    )
    validate.add_argument(
        "--model-scout-manifest",
        action="append",
        default=[],
        help="Validate one model_scout_manifest.json; may be repeated",
    )
    validate.add_argument("--model-candidate", action="append", default=[], help="Validate one model candidate JSON; may be repeated")
    validate.add_argument(
        "--model-compatibility-report",
        action="append",
        default=[],
        help="Validate one model compatibility report JSON; may be repeated",
    )
    validate.add_argument(
        "--model-serving-probe-receipt",
        action="append",
        default=[],
        help="Validate one model serving-probe receipt JSON; may be repeated",
    )
    validate.add_argument(
        "--model-adapter-manifest",
        action="append",
        default=[],
        help="Validate one model adapter manifest JSON; may be repeated",
    )
    validate.add_argument(
        "--model-registry-entry",
        action="append",
        default=[],
        help="Validate one model registry entry JSON; may be repeated",
    )
    validate.add_argument("--model-registry", action="append", default=[], help="Validate one model registry JSON; may be repeated")
    validate.add_argument("--training-plan", action="append", default=[], help="Validate one dry-run training plan JSON; may be repeated")
    validate.add_argument(
        "--agentic-training-result",
        action="append",
        default=[],
        help="Validate one agentic_training_result.json receipt; may be repeated",
    )
    validate.add_argument(
        "--agentic-loop-plan",
        action="append",
        default=[],
        help="Validate one agentic_training_loop_plan.json contract; may be repeated",
    )
    validate.add_argument("--repair-queue", action="append", default=[], help="Validate one repair_queue.json; may be repeated")
    validate.add_argument("--replay-bundle", action="append", default=[], help="Validate one replay-bundle directory or replay_bundle.json; may be repeated")
    validate.add_argument("--trace-observability", action="append", default=[], help="Validate one trace_observability.json; may be repeated")
    validate.add_argument("--review-calibration", action="append", default=[], help="Validate one review_calibration.json; may be repeated")
    validate.add_argument("--scenario-quality", action="append", default=[], help="Validate one scenario_quality.json; may be repeated")
    validate.add_argument("--suite-summary", action="append", default=[], help="Validate one run-suite suite_summary.json; may be repeated")
    validate.add_argument("--suite-trend", action="append", default=[], help="Validate one trend-suite suite_trend.json; may be repeated")
    validate.add_argument("--eval-suite-manifest", action="append", default=[], help="Validate one hfr.eval_suite_manifest.v1 JSON file; may be repeated")
    validate.add_argument("--state-snapshot", action="append", default=[], help="Validate one hfr.state_snapshot.v1 JSON file; may be repeated")
    validate.add_argument("--state-diff", action="append", default=[], help="Validate one hfr.state_diff.v1 JSON file; may be repeated")
    validate.add_argument("--run-digest", action="append", default=[], help="Validate one hfr.run_digest.v1 JSON file; may be repeated")
    validate.add_argument("--harness-manifest", action="append", default=[], help="Validate one harness_manifest.json; may be repeated")
    validate.add_argument("--harness-result", action="append", default=[], help="Validate one harness_result.json; may be repeated")
    validate.add_argument("--harness-replay-result", action="append", default=[], help="Validate one harness_replay_result.json; may be repeated")
    validate.add_argument("--harness-suite-result", action="append", default=[], help="Validate one harness_suite_result.json; may be repeated")
    validate.add_argument("--live-smoke-summary", action="append", default=[], help="Validate one live_smoke_summary.json; may be repeated")
    validate.add_argument("--eval-summary", action="append", default=[], help="Validate one hfr.eval_summary.v1 JSON file; may be repeated")
    validate.add_argument(
        "--external-eval-plan",
        action="append",
        default=[],
        help="Validate one hfr.external_eval_adapters.v1 JSON file; may be repeated",
    )
    validate.add_argument(
        "--heldout-manifest",
        action="append",
        default=[],
        help="Validate one hfr.heldout_scenario_manifest.v1 JSON file; may be repeated",
    )
    validate.add_argument("--serving-profile", action="append", default=[], help="Validate one serving_profile.json; may be repeated")
    validate.add_argument(
        "--serving-compatibility-report",
        action="append",
        default=[],
        help="Validate one compatibility_report.json; may be repeated",
    )
    validate.add_argument("--serving-endpoint-check", action="append", default=[], help="Validate one serving_check.json; may be repeated")
    validate.add_argument("--serving-lifecycle", action="append", default=[], help="Validate one serving_lifecycle.json; may be repeated")
    validate.add_argument("--serving-demo-run", action="append", default=[], help="Validate one serving demo_run.json; may be repeated")
    validate.add_argument("--out", help="Write validation summary JSON to this path")
    validate.add_argument("--strict", action="store_true", help="Treat warnings as validation failure")
    validate.set_defaults(func=cmd_validate)


    model_scout = subparsers.add_parser("model-scout", help="Validate model-scout manifests")
    model_scout_subparsers = model_scout.add_subparsers(dest="model_scout_command", required=True)
    model_scout_validate = model_scout_subparsers.add_parser("validate", help="Validate a model-scout manifest")
    model_scout_validate.add_argument("manifest", help="Path to model_scout_manifest.json")
    model_scout_validate.add_argument("--out", help="Write validation summary JSON to this path")
    model_scout_validate.add_argument("--strict", action="store_true", help="Treat warnings as validation failure")
    model_scout_validate.set_defaults(func=cmd_model_scout_validate)

    model_candidate = subparsers.add_parser("model-candidate", help="Validate and report model-candidate artifacts")
    model_candidate_subparsers = model_candidate.add_subparsers(dest="model_candidate_command", required=True)
    model_candidate_validate = model_candidate_subparsers.add_parser("validate", help="Validate one model candidate")
    model_candidate_validate.add_argument("candidate", help="Path to model candidate JSON")
    model_candidate_validate.add_argument(
        "--require-training-eligible",
        action="store_true",
        help="Require license and terms posture that allows training selection",
    )
    model_candidate_validate.add_argument("--out", help="Write validation summary JSON to this path")
    model_candidate_validate.set_defaults(func=cmd_model_candidate_validate)
    model_candidate_report = model_candidate_subparsers.add_parser(
        "compatibility-report",
        help="Write metadata-only compatibility report for a model candidate",
    )
    model_candidate_report.add_argument("--candidate", required=True, help="Path to model candidate JSON")
    model_candidate_report.add_argument("--out", required=True, help="Write model compatibility report JSON to this path")
    model_candidate_report.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in generated report")
    model_candidate_report.set_defaults(func=cmd_model_candidate_compatibility_report)

    model_registry = subparsers.add_parser("model-registry", help="Manage the local model registry")
    model_registry_subparsers = model_registry.add_subparsers(dest="model_registry_command", required=True)
    model_registry_validate = model_registry_subparsers.add_parser("validate", help="Validate a model registry")
    model_registry_validate.add_argument("--registry", default="experiments/registry/model_registry.json", help="Path to model_registry.json")
    model_registry_validate.add_argument("--out", help="Write validation summary JSON to this path")
    model_registry_validate.add_argument("--strict", action="store_true", help="Treat warnings as validation failure")
    model_registry_validate.set_defaults(func=cmd_model_registry_validate)
    model_registry_register = model_registry_subparsers.add_parser("register", help="Register or update a model candidate")
    model_registry_register.add_argument("--registry", default="experiments/registry/model_registry.json", help="Path to model_registry.json")
    model_registry_register.add_argument("--candidate", required=True, help="Path to model candidate JSON")
    model_registry_register.add_argument("--status", default="registered", help="Registry entry status")
    model_registry_register.add_argument("--entry-out", help="Optionally write the resulting registry entry JSON")
    model_registry_register.set_defaults(func=cmd_model_registry_register)
    model_registry_list = model_registry_subparsers.add_parser("list", help="List registered model candidates")
    model_registry_list.add_argument("--registry", default="experiments/registry/model_registry.json", help="Path to model_registry.json")
    model_registry_list.add_argument("--json", action="store_true", help="Print machine-readable JSON rows")
    model_registry_list.set_defaults(func=cmd_model_registry_list)
    model_registry_alias = model_registry_subparsers.add_parser("alias", help="Move candidate, champion, or rollback aliases")
    model_registry_alias.add_argument("--registry", default="experiments/registry/model_registry.json", help="Path to model_registry.json")
    model_registry_alias.add_argument("--alias", choices=list(ALIAS_NAMES), required=True, help="Alias to move")
    model_registry_alias.add_argument("--target", required=True, help="Registry entry id to target")
    model_registry_alias.add_argument("--rollback-target", help="Required when moving champion")
    model_registry_alias.add_argument("--reason", default="", help="Reason recorded in alias history")
    model_registry_alias.set_defaults(func=cmd_model_registry_alias)
    model_registry_link = model_registry_subparsers.add_parser("link", help="Link dataset, training, adapter, eval, serving, or promotion artifacts")
    model_registry_link.add_argument("--registry", default="experiments/registry/model_registry.json", help="Path to model_registry.json")
    model_registry_link.add_argument("--entry", required=True, help="Model registry entry id to update")
    model_registry_link.add_argument("--collection", choices=list(MODEL_REGISTRY_LINK_COLLECTIONS), required=True, help="Link collection to update")
    model_registry_link.add_argument("--artifact-id", required=True, help="Stable linked artifact id")
    model_registry_link.add_argument("--kind", required=True, help="Linked artifact kind")
    model_registry_link.add_argument("--status", default="recorded", help="Linked artifact lifecycle status")
    model_registry_link.add_argument("--path", help="Optional local artifact path to hash and record")
    model_registry_link.add_argument("--sha256", help="Optional SHA-256 digest for pathless artifact refs or path verification")
    model_registry_link.add_argument("--entry-out", help="Optionally write the updated registry entry JSON")
    model_registry_link.add_argument("--metadata", action="append", default=[], type=_metadata_arg, help="Attach link metadata KEY=VALUE; may be repeated")
    model_registry_link.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in link records")
    model_registry_link.set_defaults(func=cmd_model_registry_link)
    model_registry_serving_probe = model_registry_subparsers.add_parser(
        "serving-probe-receipt",
        help="Write a no-download model serving-probe receipt and optionally link it to the registry",
    )
    model_registry_serving_probe.add_argument("--registry", default="experiments/registry/model_registry.json", help="Path to model_registry.json")
    model_registry_serving_probe.add_argument("--model-ref", required=True, help="Registry entry id or alias to resolve")
    model_registry_serving_probe.add_argument("--out", required=True, help="Write model serving-probe receipt JSON to this path")
    model_registry_serving_probe.add_argument("--profile-id", required=True, help="Stable serving profile id")
    model_registry_serving_probe.add_argument("--provider", required=True, help="Serving provider label, such as metadata_only or local")
    model_registry_serving_probe.add_argument("--serving-engine", required=True, help="Serving engine label, such as vllm-compatible")
    model_registry_serving_probe.add_argument("--base-url", required=True, help="Endpoint URL or metadata-only placeholder")
    model_registry_serving_probe.add_argument(
        "--probe-mode",
        choices=["metadata_only", "external_receipt"],
        default="metadata_only",
        help="Receipt mode; metadata_only records no endpoint execution",
    )
    model_registry_serving_probe.add_argument("--compatibility-report", help="Optional model compatibility report JSON to bind by hash")
    model_registry_serving_probe.add_argument("--link", action="store_true", help="Link the written receipt under registry links.serving_probes")
    model_registry_serving_probe.add_argument("--artifact-id", help="Stable linked artifact id; defaults to output filename stem")
    model_registry_serving_probe.add_argument("--link-status", default="metadata_receipt", help="Registry link lifecycle status")
    model_registry_serving_probe.add_argument("--entry-out", help="Optionally write the updated registry entry JSON when --link is used")
    model_registry_serving_probe.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in generated receipt and link records")
    model_registry_serving_probe.set_defaults(func=cmd_model_registry_serving_probe_receipt)
    model_registry_adapter = model_registry_subparsers.add_parser(
        "adapter-manifest",
        help="Write a no-download planned-adapter manifest and optionally link it to the registry",
    )
    model_registry_adapter.add_argument("--registry", default="experiments/registry/model_registry.json", help="Path to model_registry.json")
    model_registry_adapter.add_argument("--model-ref", required=True, help="Registry entry id or alias to resolve")
    model_registry_adapter.add_argument("--adapter-id", required=True, help="Stable planned adapter id")
    model_registry_adapter.add_argument("--kind", default="lora", help="Adapter kind, such as lora or qlora")
    model_registry_adapter.add_argument("--status", choices=sorted(MODEL_ADAPTER_MANIFEST_STATUSES), default="planned", help="Adapter manifest lifecycle status")
    model_registry_adapter.add_argument("--training-plan", required=True, help="Dry-run training plan JSON to fingerprint")
    model_registry_adapter.add_argument("--output-dir", help="Planned adapter output directory; defaults to training_plan.output.output_dir")
    model_registry_adapter.add_argument("--out", required=True, help="Write model adapter manifest JSON to this path")
    model_registry_adapter.add_argument("--link", action="store_true", help="Link the written manifest under registry links.adapters")
    model_registry_adapter.add_argument("--link-status", default="planned_adapter", help="Registry link lifecycle status")
    model_registry_adapter.add_argument("--entry-out", help="Optionally write the updated registry entry JSON when --link is used")
    model_registry_adapter.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in generated manifest and link records")
    model_registry_adapter.set_defaults(func=cmd_model_registry_adapter_manifest)

    training_plan = subparsers.add_parser("training-plan", help="Generate dry-run training plans")
    training_plan_subparsers = training_plan.add_subparsers(dest="training_plan_command", required=True)
    training_plan_dry_run = training_plan_subparsers.add_parser(
        "dry-run",
        help="Write a registry-backed dry-run training plan without downloads or GPU work",
    )
    training_plan_dry_run.add_argument("--registry", default="experiments/registry/model_registry.json", help="Path to model_registry.json")
    training_plan_dry_run.add_argument("--model-ref", required=True, help="Registry entry id or alias, such as candidate")
    training_plan_dry_run.add_argument("--dataset-id", required=True, help="Dataset version id to record in the plan")
    training_plan_dry_run.add_argument("--dataset-manifest", required=True, help="Dataset manifest file to fingerprint")
    training_plan_dry_run.add_argument("--trainer", required=True, help="Trainer or recipe name")
    training_plan_dry_run.add_argument("--mode", required=True, help="Training mode, such as sft, action_sft, dpo, or sft_then_dpo")
    training_plan_dry_run.add_argument("--output-dir", required=True, help="Planned trainer output directory")
    training_plan_dry_run.add_argument("--out", required=True, help="Write dry-run training plan JSON to this path")
    training_plan_dry_run.add_argument("--compatibility-report", help="Optional model compatibility report JSON to bind by hash")
    training_plan_dry_run.add_argument(
        "--hyperparameter",
        action="append",
        default=[],
        type=_state_set_arg,
        help="Attach trainer hyperparameter KEY=JSON_VALUE; may be repeated",
    )
    training_plan_dry_run.add_argument(
        "--compute",
        action="append",
        default=[],
        type=_state_set_arg,
        help="Attach compute assumption KEY=JSON_VALUE; may be repeated",
    )
    training_plan_dry_run.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in generated plan")
    training_plan_dry_run.set_defaults(func=cmd_training_plan_dry_run)

    heldout_manifest = subparsers.add_parser("heldout-manifest", help="Build a held-out scenario manifest from suite summaries")
    heldout_manifest.add_argument(
        "--suite-summary",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="run-suite suite_summary.json to include; may be repeated",
    )
    heldout_manifest.add_argument("--out", help="Write held-out manifest JSON to this path")
    heldout_manifest.add_argument("--preserve-paths", action="store_true", help="Allow absolute source paths in manifest output")
    heldout_manifest.set_defaults(func=cmd_heldout_manifest)

    external_eval_plan = subparsers.add_parser(
        "external-eval-plan",
        help="Plan fail-closed external BFCL/Inspect/lm-eval/SWE-bench adapter readiness",
    )
    external_eval_plan.add_argument(
        "--adapter",
        action="append",
        default=[],
        choices=adapter_choices(),
        help="External eval adapter to include; defaults to all supported adapters",
    )
    external_eval_plan.add_argument("--scenario-manifest", help="Held-out scenario manifest file shared by all external adapters")
    external_eval_plan.add_argument("--model-endpoint", help="Model endpoint or serving target used by external adapters")
    external_eval_plan.add_argument("--model", help="Model identifier included in adapter metadata")
    external_eval_plan.add_argument("--tool-schema-set", help="BFCL tool/function schema set identifier or file")
    external_eval_plan.add_argument("--inspect-task-set", help="Inspect AI task set identifier or file")
    external_eval_plan.add_argument("--lm-eval-task", action="append", default=[], help="lm-evaluation-harness task name; may be repeated")
    external_eval_plan.add_argument("--swe-bench-task-set", help="SWE-bench held-out task set identifier or file")
    external_eval_plan.add_argument("--sandbox-policy", help="Sandbox policy identifier or file for stateful external tasks")
    external_eval_plan.add_argument(
        "--allow-installed",
        action="store_true",
        help="Allow installed optional adapter dependencies to become ready when required inputs are present",
    )
    external_eval_plan.add_argument("--out", help="Write external eval adapter plan JSON to this path")
    external_eval_plan.add_argument("--preserve-paths", action="store_true", help="Allow absolute source paths in plan output")
    external_eval_plan.set_defaults(func=cmd_external_eval_plan)

    agentic_loop = subparsers.add_parser("agentic-loop", help="Plan closed-loop agentic training iterations")
    agentic_loop_subparsers = agentic_loop.add_subparsers(dest="agentic_loop_command", required=True)
    agentic_loop_plan = agentic_loop_subparsers.add_parser("plan", help="Write a fail-closed agentic training loop plan")
    agentic_loop_plan.add_argument("--iteration-id", required=True, help="Stable iteration id for this loop contract")
    agentic_loop_plan.add_argument("--out", required=True, help="Write hfr.agentic_training_loop_plan.v1 JSON to this path")
    agentic_loop_plan.add_argument("--objective", help="Human-readable iteration objective")
    agentic_loop_plan.add_argument("--created-at", help="Override generated timestamp for deterministic examples")
    agentic_loop_plan.add_argument("--baseline", help="Baseline policy/model id")
    agentic_loop_plan.add_argument("--candidate", help="Candidate policy/model id")
    agentic_loop_plan.add_argument("--teacher", help="Teacher policy/model id")
    agentic_loop_plan.add_argument("--provider", action="append", default=[], help="Allowed external trainer/provider id; may be repeated")
    agentic_loop_plan.add_argument("--region", action="append", default=[], help="Allowed cloud region; may be repeated")
    agentic_loop_plan.add_argument("--gpu-class", action="append", default=[], help="Allowed GPU class; may be repeated")
    agentic_loop_plan.add_argument("--budget", action="append", default=[], type=_state_set_arg, help="Attach budget KEY=JSON_VALUE; may be repeated")
    agentic_loop_plan.add_argument("--schedule", action="append", default=[], type=_state_set_arg, help="Attach next-iteration schedule KEY=JSON_VALUE; may be repeated")
    agentic_loop_plan.add_argument("--harness-manifest", action="append", default=[], help="harness_run_manifest artifact; may be repeated")
    agentic_loop_plan.add_argument("--harness-result", action="append", default=[], help="harness_run_result artifact; may be repeated")
    agentic_loop_plan.add_argument("--evidence-bundle", action="append", default=[], help="evidence_bundle artifact; may be repeated")
    agentic_loop_plan.add_argument("--review-calibration", action="append", default=[], help="review_calibration artifact; may be repeated")
    agentic_loop_plan.add_argument("--reviewed-gate", action="append", default=[], help="reviewed_gate artifact; may be repeated")
    agentic_loop_plan.add_argument("--training-export", action="append", default=[], help="export-rl directory; may be repeated")
    agentic_loop_plan.add_argument("--agentic-training-plan", action="append", default=[], help="agentic_training_plan artifact; may be repeated")
    agentic_loop_plan.add_argument(
        "--agentic-training-runtime-preflight",
        action="append",
        default=[],
        help="agentic_training_runtime_preflight artifact; may be repeated",
    )
    agentic_loop_plan.add_argument("--agentic-training-result", action="append", default=[], help="agentic_training_result artifact; may be repeated")
    agentic_loop_plan.add_argument("--trainer-preflight", action="append", default=[], help="trainer_preflight artifact; may be repeated")
    agentic_loop_plan.add_argument("--trainer-launch-check", action="append", default=[], help="trainer_launch_check artifact; may be repeated")
    agentic_loop_plan.add_argument("--serving-lifecycle", action="append", default=[], help="serving_lifecycle artifact; may be repeated")
    agentic_loop_plan.add_argument("--heldout-manifest", action="append", default=[], help="heldout manifest artifact; may be repeated")
    agentic_loop_plan.add_argument("--external-eval-plan", action="append", default=[], help="external_eval_plan artifact; may be repeated")
    agentic_loop_plan.add_argument("--eval-summary", action="append", default=[], help="eval_summary artifact; may be repeated")
    agentic_loop_plan.add_argument("--improvement-plan", action="append", default=[], help="improvement_plan artifact; may be repeated")
    agentic_loop_plan.add_argument("--improvement-ledger", action="append", default=[], help="improvement_ledger artifact; may be repeated")
    agentic_loop_plan.add_argument("--action-ledger", action="append", default=[], help="action_ledger artifact; may be repeated")
    agentic_loop_plan.add_argument("--promotion-decision", action="append", default=[], help="promotion_decision artifact; may be repeated")
    agentic_loop_plan.add_argument("--promotion-ledger", action="append", default=[], help="promotion_ledger artifact; may be repeated")
    agentic_loop_plan.add_argument("--preserve-paths", action="store_true", help="Allow absolute source paths in plan output")
    agentic_loop_plan.set_defaults(func=cmd_agentic_loop_plan)

    eval_summary = subparsers.add_parser("eval-summary", help="Build a governance-ready held-out eval summary")
    eval_summary.add_argument(
        "--suite-summary",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="run-suite suite_summary.json to include; may be repeated",
    )
    eval_summary.add_argument(
        "--compare-export",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="export-compare-rl directory or manifest.json to summarize; may be repeated",
    )
    eval_summary.add_argument(
        "--compare-gate",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="gate-compare-export JSON output to include; may be repeated",
    )
    eval_summary.add_argument(
        "--external-adapter-plan",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="External eval adapter readiness plan JSON to include; may be repeated",
    )
    eval_summary.add_argument(
        "--serving-check",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="serving_check.json endpoint preflight to attach to a matching suite-summary label; may be repeated",
    )
    eval_summary.add_argument(
        "--require-serving-preflight",
        action="store_true",
        help="Block suite arms that do not have a ready serving_check.json preflight attached",
    )
    eval_summary.add_argument("--out", help="Write eval summary JSON to this path")
    eval_summary.add_argument("--markdown-out", help="Write a compact Markdown eval handoff report")
    eval_summary.add_argument("--preserve-paths", action="store_true", help="Allow absolute source paths in summary output")
    eval_summary.set_defaults(func=cmd_eval_summary)


    schemas = subparsers.add_parser("schemas", help="List or export bundled JSON Schema contracts")
    schemas.add_argument("--name", action="append", default=[], help="Schema name, filename, schema version, or $id; may be repeated with --write-dir")
    schemas.add_argument("--check", help="Check one JSON artifact against a bundled schema; infers by schema_version unless --name is supplied")
    schemas.add_argument("--check-jsonl", help="Check every non-empty JSONL row against a bundled schema; infers by schema_version unless --name is supplied")
    schemas.add_argument("--out", help="Write exactly one selected schema, or a --check result, to this JSON file")
    schemas.add_argument("--write-dir", help="Write the selected schemas, or all schemas, plus catalog manifest to this directory")
    schemas.add_argument("--force", action="store_true", help="Overwrite existing files when using --write-dir")
    schemas.set_defaults(func=cmd_schemas)

    evidence_coverage = subparsers.add_parser(
        "evidence-coverage",
        help="Summarize structured evidence-ref coverage across completed runs",
    )
    evidence_coverage.add_argument("--runs", required=True, help="Directory containing Flight Recorder run subdirectories")
    evidence_coverage.add_argument("--out", help="Write evidence coverage JSON to this path")
    evidence_coverage.add_argument(
        "--min-failed-rule-evidence-rate",
        type=_rate_arg,
        help="Minimum fraction of failed rules that must have structured evidence refs",
    )
    evidence_coverage.add_argument(
        "--min-critical-failed-rule-evidence-rate",
        type=_rate_arg,
        help="Minimum fraction of failed critical rules that must have structured evidence refs",
    )
    evidence_coverage.add_argument(
        "--min-event-evidence-refs",
        type=_non_negative_int_arg,
        help="Minimum evidence refs pointing at trace events",
    )
    evidence_coverage.add_argument(
        "--max-failed-rules-without-evidence",
        type=_non_negative_int_arg,
        help="Maximum failed rules allowed to have no structured evidence refs",
    )
    evidence_coverage.add_argument(
        "--max-critical-failed-rules-without-evidence",
        type=_non_negative_int_arg,
        help="Maximum failed critical rules allowed to have no structured evidence refs",
    )
    evidence_coverage.add_argument(
        "--require-rule-evidence",
        action="append",
        default=[],
        help="Fail unless this rule id has at least one structured evidence ref across the suite",
    )
    evidence_coverage.add_argument("--preserve-paths", action="store_true", help="Allow absolute run paths in coverage output")
    evidence_coverage.set_defaults(func=cmd_evidence_coverage)

    trace_observability = subparsers.add_parser(
        "trace-observability",
        help="Summarize whether completed runs contain enough trace signal",
    )
    trace_observability.add_argument("--runs", required=True, help="Directory containing Flight Recorder run subdirectories")
    trace_observability.add_argument("--out", help="Write trace observability JSON to this path")
    trace_observability.add_argument("--min-average-events", type=_non_negative_float_arg, help="Minimum average normalized trace events per run")
    trace_observability.add_argument("--min-event-type-count", type=_non_negative_int_arg, help="Minimum distinct event types across the suite")
    trace_observability.add_argument("--min-tool-or-api-run-rate", type=_rate_arg, help="Minimum fraction of runs with tool or API events")
    trace_observability.add_argument("--max-empty-final-answers", type=_non_negative_int_arg, help="Maximum runs allowed to have empty final answers")
    trace_observability.add_argument("--require-event-type", action="append", default=[], help="Fail unless this normalized event type appears")
    trace_observability.add_argument("--preserve-paths", action="store_true", help="Allow absolute run paths in observability output")
    trace_observability.set_defaults(func=cmd_trace_observability)

    repair_queue = subparsers.add_parser(
        "repair-queue",
        help="Export failed scorecard rules as deterministic repair tasks",
    )
    repair_queue.add_argument("--runs", required=True, help="Directory containing Flight Recorder run subdirectories")
    repair_queue.add_argument("--out", help="Write repair queue JSON to this path")
    repair_queue.add_argument("--only-critical", action="store_true", help="Include only failed rules marked critical")
    repair_queue.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in repair queue output")
    repair_queue.set_defaults(func=cmd_repair_queue)

    evidence_bundle = subparsers.add_parser(
        "evidence-bundle",
        help="Summarize a complete evidence handoff bundle and readiness checks",
    )
    evidence_bundle.add_argument("--out", required=True, help="Write evidence bundle summary JSON to this path")
    evidence_bundle.add_argument("--runs", help="Runs directory included in the handoff")
    evidence_bundle.add_argument("--suite-summary", help="run-suite suite_summary.json included in the handoff")
    evidence_bundle.add_argument("--scenario-quality", help="scenario_quality.json included in the handoff")
    evidence_bundle.add_argument("--evidence-coverage", help="evidence_coverage.json included in the handoff")
    evidence_bundle.add_argument("--trace-observability", help="trace_observability.json included in the handoff")
    evidence_bundle.add_argument("--repair-queue", help="repair_queue.json included in the handoff")
    evidence_bundle.add_argument("--validation", help="validation.json included in the handoff")
    evidence_bundle.add_argument("--eval-summary", help="eval_summary.json included in the handoff")
    evidence_bundle.add_argument("--training-export", help="export-rl directory included in the handoff")
    evidence_bundle.add_argument("--compare-export", help="export-compare-rl directory included in the handoff")
    evidence_bundle.add_argument("--review-export", help="export-review directory included in the handoff")
    evidence_bundle.add_argument("--reviewed-export", help="apply-review directory included in the handoff")
    evidence_bundle.add_argument("--review-calibration", help="review_calibration.json included in the handoff")
    evidence_bundle.add_argument("--live-smoke-summary", help="live_smoke_summary.json included in the handoff")
    evidence_bundle.add_argument("--serving-lifecycle", help="serving_lifecycle.json included in the handoff")
    evidence_bundle.add_argument("--trainer-preflight", help="trainer_preflight.json included in the handoff")
    evidence_bundle.add_argument("--trainer-launch-check", help="trainer_launch_check.json included in the handoff")
    evidence_bundle.add_argument("--trainer-archive", help="Trainer archive directory or trainer_archive.json included in the handoff")
    evidence_bundle.add_argument("--trainer-archive-check", help="trainer_archive_check.json included in the handoff")
    evidence_bundle.add_argument("--trainer-consumer-plan", help="trainer_consumer_plan.json included in the handoff")
    evidence_bundle.add_argument("--trainer-wrapper-dry-run", help="trainer_wrapper_dry_run.json included in the handoff")
    evidence_bundle.add_argument("--agentic-training-result", help="agentic_training_result.json included in the handoff")
    evidence_bundle.add_argument("--harness-manifest", action="append", default=[], help="harness_manifest.json included in the handoff; may be repeated")
    evidence_bundle.add_argument("--harness-result", action="append", default=[], help="harness_result.json included in the handoff; may be repeated")
    evidence_bundle.add_argument("--gate", action="append", default=[], help="Gate result JSON to require; may be repeated")
    evidence_bundle.add_argument("--require-harness", action="store_true", help="Block unless at least one matched harness manifest/result pair is included")
    evidence_bundle.add_argument("--require-gate", action="store_true", help="Block unless at least one gate summary is included")
    evidence_bundle.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in the bundle summary")
    evidence_bundle.set_defaults(func=cmd_evidence_bundle)

    improvement_plan = subparsers.add_parser(
        "improvement-plan",
        help="Join bundle actions, repair items, curriculum priorities, and run digests into a next-iteration plan",
    )
    improvement_plan.add_argument("--evidence-bundle", required=True, help="evidence_bundle.json to summarize")
    improvement_plan.add_argument("--repair-queue", help="repair_queue.json with concrete failed-rule repair items")
    improvement_plan.add_argument("--training-export", help="export-rl directory containing curriculum.json")
    improvement_plan.add_argument("--runs", help="Runs directory containing per-run run_digest.json files")
    improvement_plan.add_argument("--eval-summary", help="eval_summary.json with eval repair/curriculum work items")
    improvement_plan.add_argument("--out", required=True, help="Write improvement plan JSON to this path")
    improvement_plan.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in the plan output")
    improvement_plan.set_defaults(func=cmd_improvement_plan)

    improvement_ledger = subparsers.add_parser(
        "improvement-ledger",
        help="Summarize improvement-plan work items across improvement iterations",
    )
    improvement_ledger.add_argument("--plan", action="append", required=True, help="Improvement plan JSON in chronological order; may be repeated")
    improvement_ledger.add_argument("--out", help="Write improvement ledger JSON to this path")
    improvement_ledger.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in the ledger output")
    improvement_ledger.set_defaults(func=cmd_improvement_ledger)

    gate_improvement_ledger = subparsers.add_parser(
        "gate-improvement-ledger",
        help="Evaluate concrete improvement-work thresholds against an improvement ledger",
    )
    gate_improvement_ledger.add_argument("--improvement-ledger", required=True, help="Path to improvement_ledger.json")
    gate_improvement_ledger.add_argument("--policy", help="Versioned improvement-ledger gate policy JSON file")
    gate_improvement_ledger.add_argument("--out", help="Write improvement-ledger gate JSON to this path")
    gate_improvement_ledger.add_argument("--min-plans", type=_non_negative_int_arg, help="Minimum improvement plans required")
    gate_improvement_ledger.add_argument(
        "--max-open-work-items",
        type=_non_negative_int_arg,
        help="Maximum concrete work items still open in the latest plan",
    )
    gate_improvement_ledger.add_argument(
        "--max-new-work-items",
        type=_non_negative_int_arg,
        help="Maximum work items first seen in the latest plan",
    )
    gate_improvement_ledger.add_argument(
        "--max-recurring-work-items",
        type=_non_negative_int_arg,
        help="Maximum work items recurring from earlier plans",
    )
    gate_improvement_ledger.add_argument(
        "--min-resolved-work-items",
        type=_non_negative_int_arg,
        help="Minimum work items resolved before the latest plan",
    )
    gate_improvement_ledger.add_argument(
        "--max-critical-open-work-items",
        type=_non_negative_int_arg,
        help="Maximum critical-priority work items still open",
    )
    gate_improvement_ledger.add_argument(
        "--max-high-open-work-items",
        type=_non_negative_int_arg,
        help="Maximum high-priority work items still open",
    )
    gate_improvement_ledger.add_argument(
        "--forbid-open-priority",
        action="append",
        default=[],
        choices=["critical", "high", "medium", "low"],
        help="Fail if any open work item has this priority; may be repeated",
    )
    gate_improvement_ledger.add_argument(
        "--forbid-open-category",
        action="append",
        default=[],
        choices=["bundle_action", "repair", "curriculum", "digest_action"],
        help="Fail if any open work item has this category; may be repeated",
    )
    gate_improvement_ledger.add_argument(
        "--forbid-open-work-key",
        action="append",
        default=[],
        help="Fail if this work key, routing key, item id, or fingerprint is open; may be repeated",
    )
    gate_improvement_ledger.add_argument(
        "--require-open-work-key",
        action="append",
        default=[],
        help="Fail unless this work key, routing key, item id, or fingerprint is open; may be repeated",
    )
    gate_improvement_ledger.add_argument(
        "--require-resolved-work-key",
        action="append",
        default=[],
        help="Fail unless this work key, routing key, item id, or fingerprint is resolved; may be repeated",
    )
    gate_improvement_ledger.add_argument("--preserve-paths", action="store_true", help="Allow absolute ledger paths in gate output")
    gate_improvement_ledger.set_defaults(func=cmd_gate_improvement_ledger)

    action_ledger = subparsers.add_parser(
        "action-ledger",
        help="Summarize evidence-bundle next actions across improvement iterations",
    )
    action_ledger.add_argument("--bundle", action="append", required=True, help="Evidence bundle JSON in chronological order; may be repeated")
    action_ledger.add_argument("--out", help="Write action ledger JSON to this path")
    action_ledger.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in the ledger output")
    action_ledger.set_defaults(func=cmd_action_ledger)

    promotion_cards = subparsers.add_parser(
        "promotion-cards",
        help="Generate model and dataset cards for promotion governance",
    )
    promotion_cards.add_argument("--candidate-id", required=True, help="Model id proposed for promotion")
    promotion_cards.add_argument("--dataset-id", required=True, help="Dataset id or version represented by the dataset card")
    promotion_cards.add_argument("--model-source", required=True, help="Source model or training output summarized by the model card")
    promotion_cards.add_argument(
        "--license-status",
        default="unknown",
        help="Reviewed license status; unknown/unreviewed/missing blocks promotion-card readiness",
    )
    promotion_cards.add_argument("--evidence-bundle", help="evidence_bundle.json used as governance evidence")
    promotion_cards.add_argument("--training-export", help="Training export directory used to produce the dataset card")
    promotion_cards.add_argument("--compare-gate", help="compare_gate.json used as eval movement evidence")
    promotion_cards.add_argument("--redaction-check", help="Redaction gate JSON")
    promotion_cards.add_argument("--safety-gate", help="Safety gate JSON")
    promotion_cards.add_argument("--out", required=True, help="Output directory for MODEL_CARD.md, DATASET_CARD.md, and promotion_cards.json")
    promotion_cards.add_argument(
        "--metadata",
        action="append",
        nargs=2,
        metavar=("KEY", "VALUE"),
        default=[],
        help="Attach metadata to the promotion cards manifest; may be repeated",
    )
    promotion_cards.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in card manifests")
    promotion_cards.set_defaults(func=cmd_promotion_cards)

    promotion_alias_apply = subparsers.add_parser(
        "promotion-alias-apply",
        help="Apply registry aliases from a validated passing promotion decision",
    )
    promotion_alias_apply.add_argument("--registry", required=True, help="Model registry JSON to mutate only if all checks pass")
    promotion_alias_apply.add_argument("--promotion-decision", required=True, help="Passing promotion_decision.json authorizing aliases")
    promotion_alias_apply.add_argument("--out", required=True, help="Write promotion alias application receipt JSON")
    promotion_alias_apply.add_argument(
        "--metadata",
        action="append",
        nargs=2,
        metavar=("KEY", "VALUE"),
        default=[],
        help="Attach metadata to the alias-apply receipt; may be repeated",
    )
    promotion_alias_apply.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in the receipt")
    promotion_alias_apply.set_defaults(func=cmd_promotion_alias_apply)

    promotion_rollback_receipt = subparsers.add_parser(
        "promotion-rollback-receipt",
        help="Prove the rollback target is registered before promotion",
    )
    promotion_rollback_receipt.add_argument("--registry", required=True, help="Model registry JSON used to verify the rollback target")
    promotion_rollback_receipt.add_argument("--rollback-id", required=True, help="Model id to use as rollback if the candidate is promoted")
    promotion_rollback_receipt.add_argument(
        "--champion-id",
        help="Expected current champion model id; defaults to the registry champion alias",
    )
    promotion_rollback_receipt.add_argument("--out", required=True, help="Write rollback receipt JSON")
    promotion_rollback_receipt.add_argument(
        "--metadata",
        action="append",
        nargs=2,
        metavar=("KEY", "VALUE"),
        default=[],
        help="Attach metadata to the rollback receipt; may be repeated",
    )
    promotion_rollback_receipt.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in the receipt")
    promotion_rollback_receipt.set_defaults(func=cmd_promotion_rollback_receipt)

    promotion_release_record = subparsers.add_parser(
        "promotion-release-record",
        help="Bind promotion decisions, cards, alias receipts, rollback, evals, and release notes",
    )
    promotion_release_record.add_argument("--release-id", required=True, help="Release identifier to bind to the governance artifacts")
    promotion_release_record.add_argument("--promotion-decision", required=True, help="Passing promotion_decision.json")
    promotion_release_record.add_argument("--promotion-cards", required=True, help="Promotion cards directory or promotion_cards.json")
    promotion_release_record.add_argument("--promotion-alias-apply", required=True, help="Passing promotion_alias_apply.json receipt")
    promotion_release_record.add_argument("--rollback-metadata", required=True, help="Rollback metadata JSON used by the promotion decision")
    promotion_release_record.add_argument("--compare-gate", required=True, help="compare_gate.json used as eval movement evidence")
    promotion_release_record.add_argument("--release-notes", required=True, help="Release notes markdown or text file")
    promotion_release_record.add_argument("--promotion-policy", help="Promotion policy JSON expected to match the decision policy fingerprint")
    promotion_release_record.add_argument("--out", required=True, help="Write promotion_release_record.json")
    promotion_release_record.add_argument(
        "--metadata",
        action="append",
        nargs=2,
        metavar=("KEY", "VALUE"),
        default=[],
        help="Attach metadata to the release record; may be repeated",
    )
    promotion_release_record.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in the release record")
    promotion_release_record.set_defaults(func=cmd_promotion_release_record)

    promotion_decision = subparsers.add_parser(
        "promotion-decision",
        help="Evaluate top-level governance evidence before registry alias movement",
    )
    promotion_decision.add_argument("--candidate-id", required=True, help="Model id proposed for promotion")
    promotion_decision.add_argument("--champion-id", required=True, help="Current champion model id")
    promotion_decision.add_argument("--rollback-id", help="Model id to assign to rollback if promotion passes")
    promotion_decision.add_argument(
        "--candidate-class",
        default="candidate",
        choices=["base", "trace-only", "frontier", "champion", "candidate"],
        help="Source class of the promoted candidate",
    )
    promotion_decision.add_argument(
        "--champion-class",
        default="champion",
        choices=["base", "trace-only", "frontier", "champion", "candidate"],
        help="Source class of the incumbent champion",
    )
    promotion_decision.add_argument("--evidence-bundle", help="evidence_bundle.json for the candidate run")
    promotion_decision.add_argument("--promotion-ledger-gate", help="promotion_ledger_gate.json proving clean promotion history")
    promotion_decision.add_argument("--compare-gate", help="compare_gate.json proving candidate/champion eval movement")
    promotion_decision.add_argument("--trainer-launch-check", help="trainer_launch_check.json proving trainer handoff readiness")
    promotion_decision.add_argument("--model-registry-entry", help="model_registry_entry.json for the promoted candidate")
    promotion_decision.add_argument("--agentic-training-result", help="agentic_training_result.json for the promoted candidate")
    promotion_decision.add_argument("--model-card", help="Candidate model card")
    promotion_decision.add_argument("--dataset-card", help="Dataset card for the training/eval data")
    promotion_decision.add_argument("--rollback-metadata", help="Rollback metadata JSON naming the rollback target")
    promotion_decision.add_argument("--license-review", help="License review JSON with known license status")
    promotion_decision.add_argument("--redaction-check", help="Redaction gate JSON")
    promotion_decision.add_argument("--safety-gate", help="Safety gate JSON")
    promotion_decision.add_argument("--serving-profile", help="serving_profile.json proving candidate endpoint readiness")
    promotion_decision.add_argument("--serving-report", help="Serving smoke or readiness JSON")
    promotion_decision.add_argument("--promotion-policy", help="Promotion policy JSON declaring required artifacts and zero-tolerance limits")
    promotion_decision.add_argument("--out", help="Write promotion decision JSON to this path")
    promotion_decision.add_argument(
        "--metadata",
        action="append",
        nargs=2,
        metavar=("KEY", "VALUE"),
        default=[],
        help="Attach metadata to the promotion decision; may be repeated",
    )
    promotion_decision.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in the decision output")
    promotion_decision.set_defaults(func=cmd_promotion_decision)

    promotion_ledger = subparsers.add_parser(
        "promotion-ledger",
        help="Summarize decision gates across promotion attempts",
    )
    promotion_ledger.add_argument(
        "--decision-gate",
        action="append",
        required=True,
        help="decision_gate.json in chronological order; may be repeated",
    )
    promotion_ledger.add_argument("--out", help="Write promotion ledger JSON to this path")
    promotion_ledger.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in the ledger output")
    promotion_ledger.set_defaults(func=cmd_promotion_ledger)

    promotion_archive = subparsers.add_parser(
        "promotion-archive",
        help="Copy promotion-history evidence into a portable hash-checked archive",
    )
    promotion_archive.add_argument("--promotion-ledger", required=True, help="Path to promotion_ledger.json")
    promotion_archive.add_argument("--promotion-ledger-gate", help="Optional path to promotion_ledger_gate.json")
    promotion_archive.add_argument("--decision-gate", action="append", default=[], help="Decision gate JSON to include; may be repeated")
    promotion_archive.add_argument(
        "--promotion-release-record",
        action="append",
        default=[],
        help="Promotion release record JSON to include as final publication evidence; may be repeated",
    )
    promotion_archive.add_argument("--out", required=True, help="Output directory for the portable promotion archive")
    promotion_archive.add_argument(
        "--require-self-contained",
        action="store_true",
        help="Return nonzero unless all referenced decision gates and source artifacts were copied",
    )
    promotion_archive.add_argument("--force", action="store_true", help="Replace an existing non-empty archive directory")
    promotion_archive.add_argument(
        "--preserve-paths",
        action="store_true",
        help="Allow absolute original paths in the archive manifest; use only for private local debugging",
    )
    promotion_archive.set_defaults(func=cmd_promotion_archive)

    gate_promotion_ledger = subparsers.add_parser(
        "gate-promotion-ledger",
        help="Evaluate promotion-history thresholds against a promotion ledger",
    )
    gate_promotion_ledger.add_argument("--promotion-ledger", required=True, help="Path to promotion_ledger.json")
    gate_promotion_ledger.add_argument("--policy", help="Versioned promotion-ledger gate policy JSON file")
    gate_promotion_ledger.add_argument("--out", help="Write promotion-ledger gate JSON to this path")
    gate_promotion_ledger.add_argument("--min-decisions", type=_non_negative_int_arg, help="Minimum promotion decisions required")
    gate_promotion_ledger.add_argument("--min-allowed-count", type=_non_negative_int_arg, help="Minimum allowed promotion decisions required")
    gate_promotion_ledger.add_argument("--max-blocked-count", type=_non_negative_int_arg, help="Maximum blocked promotion decisions allowed")
    gate_promotion_ledger.add_argument("--max-blocked-rate", type=_rate_arg, help="Maximum blocked promotion decision rate, from 0.0 to 1.0")
    gate_promotion_ledger.add_argument("--min-consecutive-allowed", type=_non_negative_int_arg, help="Minimum consecutive allow decisions required at the ledger tail")
    gate_promotion_ledger.add_argument("--max-consecutive-blocked", type=_non_negative_int_arg, help="Maximum consecutive block decisions allowed at the ledger tail")
    gate_promotion_ledger.add_argument("--max-failed-decisions", type=_non_negative_int_arg, help="Maximum decision gates with failed checks")
    gate_promotion_ledger.add_argument(
        "--require-latest-recommendation",
        choices=["allow_promotion", "block_promotion"],
        help="Required latest promotion-ledger recommendation",
    )
    gate_promotion_ledger.add_argument("--require-latest-passed", action="store_true", help="Require the latest decision gate to have passed")
    gate_promotion_ledger.add_argument(
        "--require-source-recommendation",
        action="append",
        default=[],
        help="Fail unless this source recommendation appears in the ledger history; may be repeated",
    )
    gate_promotion_ledger.add_argument(
        "--forbid-source-recommendation",
        action="append",
        default=[],
        help="Fail if this source recommendation appears in the ledger history; may be repeated",
    )
    gate_promotion_ledger.add_argument("--preserve-paths", action="store_true", help="Allow absolute ledger paths in gate output")
    gate_promotion_ledger.set_defaults(func=cmd_gate_promotion_ledger)

    gate_action_ledger = subparsers.add_parser(
        "gate-action-ledger",
        help="Evaluate improvement-loop thresholds against an action ledger",
    )
    gate_action_ledger.add_argument("--action-ledger", required=True, help="Path to action_ledger.json")
    gate_action_ledger.add_argument("--policy", help="Versioned action-ledger gate policy JSON file")
    gate_action_ledger.add_argument("--out", help="Write action-ledger gate JSON to this path")
    gate_action_ledger.add_argument("--min-bundles", type=_non_negative_int_arg, help="Minimum evidence bundles required in the ledger")
    gate_action_ledger.add_argument("--max-open-actions", type=_non_negative_int_arg, help="Maximum actions still open in the latest bundle")
    gate_action_ledger.add_argument("--max-new-actions", type=_non_negative_int_arg, help="Maximum actions first seen in the latest bundle")
    gate_action_ledger.add_argument("--max-recurring-actions", type=_non_negative_int_arg, help="Maximum actions recurring from earlier bundles")
    gate_action_ledger.add_argument("--min-resolved-actions", type=_non_negative_int_arg, help="Minimum actions resolved before the latest bundle")
    gate_action_ledger.add_argument(
        "--forbid-open-priority",
        action="append",
        default=[],
        choices=["critical", "high", "medium", "low"],
        help="Fail if any open action has this priority; may be repeated",
    )
    gate_action_ledger.add_argument(
        "--forbid-open-action",
        action="append",
        default=[],
        help="Fail if this action id, routing key, or fingerprint is open; may be repeated",
    )
    gate_action_ledger.add_argument(
        "--require-resolved-action",
        action="append",
        default=[],
        help="Fail unless this action id, routing key, or fingerprint is resolved; may be repeated",
    )
    gate_action_ledger.add_argument("--preserve-paths", action="store_true", help="Allow absolute ledger paths in gate output")
    gate_action_ledger.set_defaults(func=cmd_gate_action_ledger)

    gate_decision = subparsers.add_parser(
        "gate-decision",
        help="Gate a Flight Recorder decision recommendation for CI promotion",
    )
    gate_decision.add_argument("--artifact", required=True, help="Flight Recorder JSON artifact with a decision block")
    gate_decision.add_argument("--expect-recommendation", required=True, help="Required decision.recommendation value")
    gate_decision.add_argument("--expect-readiness", help="Optional required decision.readiness value")
    gate_decision.add_argument("--require-passed", action="store_true", help="Also require the source artifact root passed field to be true")
    gate_decision.add_argument("--out", help="Write decision gate JSON to this path")
    gate_decision.add_argument("--preserve-paths", action="store_true", help="Allow absolute artifact paths in gate output")
    gate_decision.set_defaults(func=cmd_gate_decision)

    draft = subparsers.add_parser("draft-scenario", help="Draft a scenario JSON file from an existing run or trace")
    draft_source = draft.add_mutually_exclusive_group(required=True)
    draft_source.add_argument("--trace", help="Trace artifact to normalize and use as source evidence")
    draft_source.add_argument("--run", help="Run directory containing normalized_trace.json")
    draft.add_argument("--format", default="auto", choices=TRACE_FORMAT_CHOICES)
    draft.add_argument("--out", required=True, help="Scenario JSON output path")
    draft.add_argument("--id", help="Scenario id; defaults to the source stem")
    draft.add_argument("--title", help="Scenario title; defaults from --id")
    draft.add_argument("--prompt", help="Scenario prompt; defaults to a TODO placeholder")
    draft.add_argument("--max-actions", type=_non_negative_int_arg, default=8, help="Maximum tool-result actions to draft")
    draft.add_argument("--preserve-paths", action="store_true", help="Write absolute trace paths instead of paths relative to --out")
    draft.set_defaults(func=cmd_draft_scenario)

    gate_suite = subparsers.add_parser("gate-suite", help="Evaluate CI thresholds against a run-suite summary")
    gate_suite.add_argument("--suite-summary", required=True, help="Path to suite_summary.json")
    gate_suite.add_argument("--policy", help="Versioned suite gate policy JSON file with committed threshold defaults")
    gate_suite.add_argument("--out", help="Write gate result JSON to this path")
    gate_suite.add_argument("--min-pass-rate", type=_rate_arg, help="Minimum allowed suite pass rate, from 0.0 to 1.0")
    gate_suite.add_argument("--min-average-score", type=_score_arg, help="Minimum allowed average score, from 0 to 100")
    gate_suite.add_argument("--max-failed", type=_non_negative_int_arg, help="Maximum allowed failed scenarios")
    gate_suite.add_argument("--max-errors", type=_non_negative_int_arg, help="Maximum allowed suite execution errors; defaults to policy value or 0")
    gate_suite.add_argument("--max-critical-failures", type=_non_negative_int_arg, help="Maximum allowed total critical failures")
    gate_suite.add_argument("--forbid-failed-rule", action="append", default=[], help="Fail if this rule id appears in failed_rule_counts")
    gate_suite.add_argument("--forbid-critical-rule", action="append", default=[], help="Fail if this rule id appears in critical_failure_counts")
    gate_suite.add_argument("--preserve-paths", action="store_true", help="Allow absolute suite summary paths in gate output")
    gate_suite.set_defaults(func=cmd_gate_suite)

    gate_export = subparsers.add_parser("gate-export", help="Evaluate readiness thresholds against an export-rl dataset")
    gate_export.add_argument("--training-export", required=True, help="Directory containing export-rl artifacts")
    gate_export.add_argument("--policy", help="Versioned training gate policy JSON file with committed threshold defaults")
    gate_export.add_argument("--out", help="Write gate result JSON to this path")
    gate_export.add_argument(
        "--strict-validation",
        action="store_true",
        help="Fail export-integrity validation on warnings as well as errors",
    )
    gate_export.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip export structure and artifact-fingerprint validation before gating",
    )
    gate_export.add_argument("--min-episodes", type=_non_negative_int_arg, help="Minimum episode count")
    gate_export.add_argument("--min-pass-rate", type=_rate_arg, help="Minimum dataset pass rate, from 0.0 to 1.0")
    gate_export.add_argument("--min-average-score", type=_score_arg, help="Minimum average score, from 0 to 100")
    gate_export.add_argument("--min-preferences", type=_non_negative_int_arg, help="Minimum preference pair count")
    gate_export.add_argument("--min-sft", type=_non_negative_int_arg, help="Minimum SFT row count")
    gate_export.add_argument("--min-dpo", type=_non_negative_int_arg, help="Minimum DPO row count")
    gate_export.add_argument("--min-reward-model", type=_non_negative_int_arg, help="Minimum reward-model row count")
    gate_export.add_argument("--min-step-rewards", type=_non_negative_int_arg, help="Minimum step-reward row count")
    gate_export.add_argument(
        "--min-task-completion-configured",
        type=_non_negative_int_arg,
        help="Minimum episodes with task-completion evidence configured",
    )
    gate_export.add_argument(
        "--min-task-completion-complete",
        type=_non_negative_int_arg,
        help="Minimum episodes with complete task-completion evidence",
    )
    gate_export.add_argument(
        "--max-task-completion-incomplete",
        type=_non_negative_int_arg,
        help="Maximum episodes with incomplete task-completion evidence",
    )
    gate_export.add_argument(
        "--min-task-completion-check-pass-rate",
        type=_rate_arg,
        help="Minimum required task-evidence check pass rate, from 0.0 to 1.0",
    )
    gate_export.add_argument(
        "--min-source-fingerprint-rate",
        type=_rate_arg,
        help="Minimum fraction of episodes with scenario and source-trace SHA-256 fingerprints",
    )
    gate_export.add_argument(
        "--max-unverified-source-fingerprints",
        type=_non_negative_int_arg,
        help="Maximum episodes allowed to lack scenario or source-trace SHA-256 fingerprints",
    )
    gate_export.add_argument(
        "--min-trainer-view-source-fingerprint-rate",
        type=_rate_arg,
        help="Minimum fraction of SFT/DPO/reward-model rows with complete source fingerprints",
    )
    gate_export.add_argument(
        "--max-unverified-trainer-view-source-fingerprints",
        type=_non_negative_int_arg,
        help="Maximum SFT/DPO/reward-model rows allowed to lack complete source fingerprints",
    )
    gate_export.add_argument("--min-trace-average-events", type=_non_negative_float_arg, help="Minimum average normalized events per exported episode")
    gate_export.add_argument("--min-trace-event-type-count", type=_non_negative_int_arg, help="Minimum distinct normalized event types in the export")
    gate_export.add_argument("--min-trace-final-answer-rate", type=_rate_arg, help="Minimum fraction of episodes with final answers")
    gate_export.add_argument("--min-trace-tool-or-api-rate", type=_rate_arg, help="Minimum fraction of episodes with tool or API events")
    gate_export.add_argument("--max-trace-empty-final-answers", type=_non_negative_int_arg, help="Maximum exported episodes allowed to have empty final answers")
    gate_export.add_argument("--max-trace-risk-count", type=_non_negative_int_arg, help="Maximum total trace observability risks allowed")
    gate_export.add_argument("--require-trace-event-type", action="append", default=[], help="Fail unless this normalized event type appears in exported traces")
    gate_export.add_argument("--min-split-task-families", type=_non_negative_int_arg, help="Minimum task families represented in dataset split metadata")
    gate_export.add_argument("--min-train-episodes", type=_non_negative_int_arg, help="Minimum episodes assigned to the train split")
    gate_export.add_argument("--min-validation-episodes", type=_non_negative_int_arg, help="Minimum episodes assigned to the validation split")
    gate_export.add_argument("--min-test-episodes", type=_non_negative_int_arg, help="Minimum episodes assigned to the test split")
    gate_export.add_argument(
        "--require-family-exclusive-splits",
        action="store_true",
        help="Fail unless dataset split metadata reports task-family-exclusive train/validation/test splits",
    )
    gate_export.add_argument("--max-quality-flags", type=_non_negative_int_arg, help="Maximum allowed dataset quality flags")
    gate_export.add_argument("--forbid-quality-flag", action="append", default=[], help="Fail if this quality flag id appears")
    gate_export.add_argument(
        "--forbid-quality-severity",
        action="append",
        default=[],
        choices=["info", "warning", "error"],
        help="Fail if any quality flag has this severity",
    )
    gate_export.add_argument("--require-task-family", action="append", default=[], help="Fail unless this task family is present")
    gate_export.add_argument("--preserve-paths", action="store_true", help="Allow absolute export paths in gate output")
    gate_export.set_defaults(func=cmd_gate_export)

    gate_reviewed = subparsers.add_parser("gate-reviewed", help="Evaluate readiness thresholds against an apply-review export")
    gate_reviewed.add_argument("--reviewed-export", required=True, help="Directory containing apply-review artifacts")
    gate_reviewed.add_argument("--policy", help="Versioned reviewed gate policy JSON file with committed threshold defaults")
    gate_reviewed.add_argument("--out", help="Write gate result JSON to this path")
    gate_reviewed.add_argument(
        "--strict-validation",
        action="store_true",
        help="Fail reviewed-export validation on warnings as well as errors",
    )
    gate_reviewed.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip reviewed export structure and artifact-fingerprint validation before gating",
    )
    gate_reviewed.add_argument("--min-reviewed-labels", type=_non_negative_int_arg, help="Minimum completed human label count")
    gate_reviewed.add_argument("--min-accepted", type=_non_negative_int_arg, help="Minimum accepted human label count")
    gate_reviewed.add_argument(
        "--min-rejected",
        type=_non_negative_int_arg,
        help="Minimum negative human label count across reject, unsafe, and incomplete labels",
    )
    gate_reviewed.add_argument("--min-sft", type=_non_negative_int_arg, help="Minimum reviewed SFT row count")
    gate_reviewed.add_argument("--min-reward-model", type=_non_negative_int_arg, help="Minimum reviewed reward-model row count")
    gate_reviewed.add_argument("--min-preferences", type=_non_negative_int_arg, help="Minimum reviewed preference pair count")
    gate_reviewed.add_argument("--min-dpo", type=_non_negative_int_arg, help="Minimum reviewed DPO row count")
    gate_reviewed.add_argument(
        "--min-high-confidence-labels",
        type=_non_negative_int_arg,
        help="Minimum reviewed labels marked reviewer_confidence='high'",
    )
    gate_reviewed.add_argument(
        "--min-medium-or-high-confidence-labels",
        type=_non_negative_int_arg,
        help="Minimum reviewed labels marked reviewer_confidence='medium' or 'high'",
    )
    gate_reviewed.add_argument("--max-needs-review", type=_non_negative_int_arg, help="Maximum allowed needs_review labels")
    gate_reviewed.add_argument(
        "--max-low-confidence-labels",
        type=_non_negative_int_arg,
        help="Maximum reviewed labels marked reviewer_confidence='low'",
    )
    gate_reviewed.add_argument(
        "--max-unknown-confidence-labels",
        type=_non_negative_int_arg,
        help="Maximum reviewed labels with missing or unknown reviewer_confidence",
    )
    gate_reviewed.add_argument(
        "--forbid-label",
        action="append",
        default=[],
        choices=list(REVIEW_LABELS),
        help="Fail if this human label appears in the reviewed export",
    )
    gate_reviewed.add_argument("--require-task-family", action="append", default=[], help="Fail unless this task family is present")
    gate_reviewed.add_argument("--preserve-paths", action="store_true", help="Allow absolute export paths in gate output")
    gate_reviewed.set_defaults(func=cmd_gate_reviewed)

    gate_compare = subparsers.add_parser("gate-compare-export", help="Evaluate readiness thresholds against an export-compare-rl dataset")
    gate_compare.add_argument("--compare-export", required=True, help="Directory containing export-compare-rl artifacts")
    gate_compare.add_argument("--policy", help="Versioned compare gate policy JSON file with committed threshold defaults")
    gate_compare.add_argument("--out", help="Write gate result JSON to this path")
    gate_compare.add_argument(
        "--strict-validation",
        action="store_true",
        help="Fail comparison-export validation on warnings as well as errors",
    )
    gate_compare.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip comparison export structure and artifact-fingerprint validation before gating",
    )
    gate_compare.add_argument("--min-pairs", type=_non_negative_int_arg, help="Minimum comparison pair count")
    gate_compare.add_argument("--min-dpo", type=_non_negative_int_arg, help="Minimum comparison DPO row count")
    gate_compare.add_argument("--min-candidate-wins", type=_non_negative_int_arg, help="Minimum candidate-win pair count")
    gate_compare.add_argument(
        "--min-task-completion-improvements",
        type=_non_negative_int_arg,
        help="Minimum pairs where the candidate completed the task and the baseline did not",
    )
    gate_compare.add_argument("--max-baseline-wins", type=_non_negative_int_arg, help="Maximum allowed baseline-win regression pairs")
    gate_compare.add_argument(
        "--max-task-completion-regressions",
        type=_non_negative_int_arg,
        help="Maximum pairs where the baseline completed the task and the candidate did not",
    )
    gate_compare.add_argument("--max-skipped-pairs", type=_non_negative_int_arg, help="Maximum allowed skipped paired scenarios")
    gate_compare.add_argument("--max-contract-drifts", type=_non_negative_int_arg, help="Maximum allowed drifted comparison contract fingerprints")
    gate_compare.add_argument(
        "--max-unverified-contracts",
        type=_non_negative_int_arg,
        help="Maximum allowed comparison pairs missing scenario or source-trace fingerprints",
    )
    gate_compare.add_argument("--require-scenario", action="append", default=[], help="Fail unless this scenario appears in the comparison pairs")
    gate_compare.add_argument(
        "--require-candidate-win-scenario",
        action="append",
        default=[],
        help="Fail unless this scenario is a candidate win",
    )
    gate_compare.add_argument(
        "--require-task-completion-improvement-scenario",
        action="append",
        default=[],
        help="Fail unless this scenario has candidate task completion and baseline non-completion",
    )
    gate_compare.add_argument(
        "--forbid-regression-scenario",
        action="append",
        default=[],
        help="Fail if this scenario is a baseline win",
    )
    gate_compare.add_argument(
        "--forbid-task-completion-regression-scenario",
        action="append",
        default=[],
        help="Fail if this scenario has baseline task completion and candidate non-completion",
    )
    gate_compare.add_argument("--require-rule-fix", action="append", default=[], help="Fail unless this rule id appears in rule_fixes")
    gate_compare.add_argument(
        "--forbid-rule-regression",
        action="append",
        default=[],
        help="Fail if this rule id appears in rule_regressions",
    )
    gate_compare.add_argument(
        "--forbid-new-critical-failure",
        action="append",
        default=[],
        help="Fail if this rule id appears in new_critical_failures",
    )
    gate_compare.add_argument("--preserve-paths", action="store_true", help="Allow absolute export paths in gate output")
    gate_compare.set_defaults(func=cmd_gate_compare_export)

    trainer_preflight = subparsers.add_parser(
        "trainer-preflight",
        help="Build a launch guard manifest over passed gates and trainer-facing artifacts",
    )
    trainer_preflight.add_argument("--gate", action="append", required=True, help="Gate JSON that must pass; may be repeated")
    trainer_preflight.add_argument("--out", required=True, help="Write trainer preflight JSON to this path")
    trainer_preflight.add_argument("--training-export", help="export-rl directory to fingerprint for the trainer handoff")
    trainer_preflight.add_argument("--compare-export", help="export-compare-rl directory to fingerprint for the trainer handoff")
    trainer_preflight.add_argument("--reviewed-export", help="apply-review directory to fingerprint for the trainer handoff")
    trainer_preflight.add_argument("--evidence-bundle", help="evidence_bundle.json that must pass before launch")
    trainer_preflight.add_argument("--agentic-training-plan", help="hfr.agentic_training_plan.v1 dry-run plan to fingerprint for the trainer handoff")
    trainer_preflight.add_argument(
        "--validation",
        action="append",
        default=[],
        help="flightrecorder validate --out summary that proves non-embedded gate artifacts; may be repeated",
    )
    trainer_preflight.add_argument("--require-gate", action="append", default=[], help="Require this gate id to be present")
    trainer_preflight.add_argument(
        "--require-dataset-version",
        action="append",
        default=[],
        help="Require an exact manifest dataset_version before trainer launch; may be repeated",
    )
    trainer_preflight.add_argument("--trainer-command", help="Trainer command to record but not execute")
    trainer_preflight.add_argument(
        "--allow-unvalidated-gates",
        action="store_true",
        help="Allow gates that skipped embedded or external validation proof",
    )
    trainer_preflight.add_argument("--preserve-paths", action="store_true", help="Allow absolute artifact paths in preflight output")
    trainer_preflight.add_argument(
        "--metadata",
        action="append",
        default=[],
        type=_metadata_arg,
        metavar="KEY=VALUE",
        help="Attach launch metadata to the preflight manifest; may be repeated",
    )
    trainer_preflight.set_defaults(func=cmd_trainer_preflight)

    trainer_launch_check = subparsers.add_parser(
        "trainer-launch-check",
        help="Validate a trainer preflight and emit the approved command without executing it",
    )
    trainer_launch_check.add_argument("--preflight", required=True, help="trainer_preflight.json to verify before trainer launch")
    trainer_launch_check.add_argument("--out", help="Write trainer launch-check JSON to this path")
    trainer_launch_check.add_argument("--require-gate", action="append", default=[], help="Require this preflight gate id to be present and passed")
    trainer_launch_check.add_argument(
        "--require-dataset-version",
        action="append",
        default=[],
        help="Require an exact dataset_version already selected by the trainer preflight; may be repeated",
    )
    trainer_launch_check.add_argument(
        "--require-metadata",
        action="append",
        default=[],
        type=_metadata_arg,
        metavar="KEY=VALUE",
        help="Require this metadata key/value from the preflight manifest; may be repeated",
    )
    trainer_launch_check.add_argument("--strict", action="store_true", help="Treat preflight validation warnings as launch blockers")
    trainer_launch_check.add_argument("--print-command", action="store_true", help="Print only the shell-escaped approved command when launch is allowed")
    trainer_launch_check.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in launch-check output")
    trainer_launch_check.set_defaults(func=cmd_trainer_launch_check)

    trainer_archive = subparsers.add_parser(
        "trainer-archive",
        help="Copy trainer handoff evidence into a portable hash-checked archive",
    )
    trainer_archive.add_argument("--preflight", required=True, help="trainer_preflight.json to include")
    trainer_archive.add_argument("--launch-check", required=True, help="trainer_launch_check.json to include")
    trainer_archive.add_argument("--out", required=True, help="Output directory for the portable trainer archive")
    trainer_archive.add_argument(
        "--require-self-contained",
        action="store_true",
        help="Return nonzero unless all preflight-referenced gates, validations, exports, and schema contracts were copied",
    )
    trainer_archive.add_argument("--force", action="store_true", help="Replace an existing non-empty trainer archive directory")
    trainer_archive.add_argument(
        "--preserve-paths",
        action="store_true",
        help="Allow absolute original paths in the archive manifest; use only for private local debugging",
    )
    trainer_archive.set_defaults(func=cmd_trainer_archive)

    trainer_archive_check = subparsers.add_parser(
        "trainer-archive-check",
        help="Validate a trainer archive plus local external trainer code without executing it",
    )
    trainer_archive_check.add_argument("--archive", required=True, help="trainer archive directory or trainer_archive.json to verify")
    trainer_archive_check.add_argument("--external-code-root", default=".", help="Directory containing external trainer code paths such as train.py")
    trainer_archive_check.add_argument("--out", required=True, help="Write trainer archive check JSON to this path")
    trainer_archive_check.add_argument("--strict", action="store_true", help="Treat archive validation warnings as consumer-readiness blockers")
    trainer_archive_check.add_argument(
        "--preserve-paths",
        action="store_true",
        help="Allow absolute local paths in trainer archive check output; use only for private local debugging",
    )
    trainer_archive_check.set_defaults(func=cmd_trainer_archive_check)

    trainer_consumer_plan = subparsers.add_parser(
        "trainer-consumer-plan",
        help="Build a side-effect-free execution plan for an external trainer wrapper",
    )
    trainer_consumer_plan.add_argument("--archive-check", required=True, help="trainer_archive_check.json to convert into a consumer plan")
    trainer_consumer_plan.add_argument("--out", required=True, help="Write trainer consumer plan JSON to this path")
    trainer_consumer_plan.add_argument("--strict", action="store_true", help="Treat archive-check validation warnings as plan blockers")
    trainer_consumer_plan.add_argument(
        "--preserve-paths",
        action="store_true",
        help="Allow absolute local paths in trainer consumer plan output; use only for private local debugging",
    )
    trainer_consumer_plan.set_defaults(func=cmd_trainer_consumer_plan)

    export_rl = subparsers.add_parser("export-rl", help="Export completed runs as future RL training artifacts")
    export_rl.add_argument("--runs", required=True, help="Directory containing Flight Recorder run subdirectories")
    export_rl.add_argument("--out", required=True, help="Output directory for evidence and trainer-ready artifacts")
    export_rl.add_argument(
        "--reward-scale",
        default="score",
        choices=["score", "binary", "signed"],
        help="Reward transform: score=0..1, binary=pass/fail, signed=-1..1",
    )
    export_rl.add_argument("--min-score-gap", type=int, default=1, help="Minimum score gap for a preference pair")
    export_rl.add_argument(
        "--max-pairs-per-family",
        type=int,
        default=0,
        help="Maximum preference pairs per task family; 0 means unlimited",
    )
    export_rl.add_argument("--preserve-paths", action="store_true", help="Allow absolute source/output paths in exported metadata")
    export_rl.add_argument(
        "--metadata",
        action="append",
        default=[],
        type=_metadata_arg,
        metavar="KEY=VALUE",
        help="Attach experiment metadata to manifest, dataset metrics, and dataset card; may be repeated",
    )
    export_rl.set_defaults(func=cmd_export_rl)

    export_compare_rl = subparsers.add_parser(
        "export-compare-rl",
        help="Export paired baseline/candidate runs as preference artifacts",
    )
    export_compare_rl.add_argument("--baseline", required=True, help="Baseline run-suite directory")
    export_compare_rl.add_argument("--candidate", required=True, help="Candidate run-suite directory")
    export_compare_rl.add_argument("--out", required=True, help="Output directory for comparison preference artifacts")
    export_compare_rl.add_argument(
        "--reward-scale",
        default="score",
        choices=["score", "binary", "signed"],
        help="Reward transform used inside episode views: score=0..1, binary=pass/fail, signed=-1..1",
    )
    export_compare_rl.add_argument("--min-score-gap", type=int, default=1, help="Minimum absolute score gap for an improvement pair")
    export_compare_rl.add_argument(
        "--contract-scope",
        default="scenario",
        choices=["scenario", "scenario-and-trace"],
        help="Fingerprint contract to compare: scenario for live improvement loops, scenario-and-trace for strict fixture replay",
    )
    export_compare_rl.add_argument("--preserve-paths", action="store_true", help="Allow absolute source/output paths in exported metadata")
    export_compare_rl.add_argument(
        "--metadata",
        action="append",
        default=[],
        type=_metadata_arg,
        metavar="KEY=VALUE",
        help="Attach experiment metadata to the comparison export; may be repeated",
    )
    export_compare_rl.set_defaults(func=cmd_export_compare_rl)

    export_review = subparsers.add_parser("export-review", help="Export completed runs as a human review queue")
    export_review.add_argument("--runs", required=True, help="Directory containing Flight Recorder run subdirectories")
    export_review.add_argument("--out", required=True, help="Output directory for review queue artifacts")
    export_review.add_argument("--only-failed", action="store_true", help="Include only failed runs in the review queue")
    export_review.add_argument("--preserve-paths", action="store_true", help="Allow absolute source/output paths in exported metadata")
    export_review.set_defaults(func=cmd_export_review)

    apply_review = subparsers.add_parser("apply-review", help="Apply completed human labels to a review queue")
    apply_review.add_argument("--review-export", required=True, help="Directory containing export-review artifacts")
    apply_review.add_argument("--out", required=True, help="Output directory for reviewed label/trainer artifacts")
    apply_review.add_argument("--labels", help="Completed labels JSONL; defaults to <review-export>/label_template.jsonl")
    apply_review.add_argument(
        "--max-pairs-per-family",
        type=_non_negative_int_arg,
        default=0,
        help="Maximum reviewed preference pairs per task family; 0 means unlimited",
    )
    apply_review.add_argument("--preserve-paths", action="store_true", help="Allow absolute source/output paths in exported metadata")
    apply_review.set_defaults(func=cmd_apply_review)

    review_calibration = subparsers.add_parser(
        "review-calibration",
        help="Compare deterministic scorecards with human-reviewed labels",
    )
    review_calibration.add_argument("--reviewed-export", required=True, help="Directory containing apply-review artifacts")
    review_calibration.add_argument("--out", required=True, help="Write calibration report JSON to this path")
    review_calibration.add_argument("--min-agreement-rate", type=_rate_arg, help="Minimum human/scorecard agreement rate")
    review_calibration.add_argument("--max-disagreements", type=_non_negative_int_arg, help="Maximum allowed comparable disagreements")
    review_calibration.add_argument("--max-false-positives", type=_non_negative_int_arg, help="Maximum scorecard-pass/human-negative rows")
    review_calibration.add_argument("--max-false-negatives", type=_non_negative_int_arg, help="Maximum scorecard-fail/human-accept rows")
    review_calibration.add_argument("--min-comparable-labels", type=_non_negative_int_arg, help="Minimum labels excluding needs_review")
    review_calibration.add_argument(
        "--strict-validation",
        action="store_true",
        help="Fail reviewed-export validation on warnings as well as errors",
    )
    review_calibration.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip reviewed export structure and artifact-fingerprint validation before calibration",
    )
    review_calibration.add_argument("--preserve-paths", action="store_true", help="Allow absolute source paths in calibration output")
    review_calibration.set_defaults(func=cmd_review_calibration)

    observer = subparsers.add_parser("observer-template", help="Print or write a read-only Hermes observer plugin template")
    observer.add_argument("--out", help="Write the template to this path instead of stdout")
    observer.add_argument("--force", action="store_true", help="Overwrite --out when it already exists")
    observer.set_defaults(func=cmd_observer_template)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ArtifactError(f"{path}:{line_number} must contain a JSON object")
        rows.append(value)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _rate_arg(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number from 0.0 to 1.0") from exc
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be from 0.0 to 1.0")
    return parsed


def _score_arg(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number from 0 to 100") from exc
    if not 0.0 <= parsed <= 100.0:
        raise argparse.ArgumentTypeError("must be from 0 to 100")
    return parsed


def _non_negative_int_arg(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _non_negative_float_arg(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative number") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative number")
    return parsed


def _metadata_arg(value: str) -> tuple[str, str]:
    key, separator, raw_value = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("must be KEY=VALUE")
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("metadata key must be non-empty")
    if any(char.isspace() for char in key):
        raise argparse.ArgumentTypeError("metadata key must not contain whitespace")
    return key, raw_value


def _key_path_arg(value: str) -> tuple[str, str]:
    key, separator, raw_path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("must be KEY=PATH")
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("key must be non-empty")
    if any(char.isspace() for char in key):
        raise argparse.ArgumentTypeError("key must not contain whitespace")
    if not raw_path:
        raise argparse.ArgumentTypeError("path must be non-empty")
    return key, raw_path


def _state_set_arg(value: str) -> tuple[str, Any]:
    key, separator, raw_value = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("must be PATH=VALUE")
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("path must be non-empty")
    try:
        parsed: Any = json.loads(raw_value)
    except json.JSONDecodeError:
        parsed = raw_value
    return key, parsed


def _metadata_options(items: list[tuple[str, str]]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for key, value in items:
        metadata[key] = value
    return metadata


def _write_score_outputs(scorecard: dict[str, Any], args: argparse.Namespace) -> None:
    if getattr(args, "junit_out", None):
        write_junit(scorecard, args.junit_out)
    if getattr(args, "markdown_out", None):
        write_markdown_summary(scorecard, args.markdown_out)


def _gate_suite_options(args: argparse.Namespace) -> dict[str, Any]:
    policy = load_gate_policy(args.policy) if args.policy else {}
    return {
        "policy_path": _display_path(Path(args.policy), args.preserve_paths) if args.policy else None,
        "policy_description": policy.get("description"),
        "min_pass_rate": args.min_pass_rate if args.min_pass_rate is not None else policy.get("min_pass_rate"),
        "min_average_score": args.min_average_score if args.min_average_score is not None else policy.get("min_average_score"),
        "max_failed": args.max_failed if args.max_failed is not None else policy.get("max_failed"),
        "max_errors": args.max_errors if args.max_errors is not None else policy.get("max_errors", 0),
        "max_critical_failures": (
            args.max_critical_failures if args.max_critical_failures is not None else policy.get("max_critical_failures")
        ),
        "forbid_failed_rules": _merge_gate_rule_ids(policy.get("forbid_failed_rules", []), args.forbid_failed_rule),
        "forbid_critical_rules": _merge_gate_rule_ids(policy.get("forbid_critical_rules", []), args.forbid_critical_rule),
        "task_family_gates": policy.get("task_family_gates", []),
    }


def _gate_policy_summary(options: dict[str, Any]) -> dict[str, Any]:
    effective_fields = (
        "min_pass_rate",
        "min_average_score",
        "max_failed",
        "max_errors",
        "max_critical_failures",
        "forbid_failed_rules",
        "forbid_critical_rules",
        "task_family_gates",
    )
    effective = {
        field: options[field]
        for field in effective_fields
        if options.get(field) is not None and options.get(field) != []
    }
    summary: dict[str, Any] = {
        "schema_version": SUITE_GATE_POLICY_SCHEMA_VERSION,
        "path": options["policy_path"],
        "effective": effective,
    }
    if options.get("policy_description"):
        summary["description"] = options["policy_description"]
    return summary


def _training_gate_options(args: argparse.Namespace) -> dict[str, Any]:
    policy = load_training_gate_policy(args.policy) if args.policy else {}
    return {
        "policy_path": _display_path(Path(args.policy), args.preserve_paths) if args.policy else None,
        "policy_description": policy.get("description"),
        "require_valid_export": False if args.skip_validation else policy.get("require_valid_export", True),
        "strict_validation": bool(args.strict_validation or policy.get("strict_validation", False)),
        "min_episodes": args.min_episodes if args.min_episodes is not None else policy.get("min_episodes"),
        "min_pass_rate": args.min_pass_rate if args.min_pass_rate is not None else policy.get("min_pass_rate"),
        "min_average_score": args.min_average_score if args.min_average_score is not None else policy.get("min_average_score"),
        "min_preferences": args.min_preferences if args.min_preferences is not None else policy.get("min_preferences"),
        "min_sft": args.min_sft if args.min_sft is not None else policy.get("min_sft"),
        "min_dpo": args.min_dpo if args.min_dpo is not None else policy.get("min_dpo"),
        "min_reward_model": args.min_reward_model if args.min_reward_model is not None else policy.get("min_reward_model"),
        "min_step_rewards": args.min_step_rewards if args.min_step_rewards is not None else policy.get("min_step_rewards"),
        "min_task_completion_configured": (
            args.min_task_completion_configured
            if args.min_task_completion_configured is not None
            else policy.get("min_task_completion_configured")
        ),
        "min_task_completion_complete": (
            args.min_task_completion_complete
            if args.min_task_completion_complete is not None
            else policy.get("min_task_completion_complete")
        ),
        "max_task_completion_incomplete": (
            args.max_task_completion_incomplete
            if args.max_task_completion_incomplete is not None
            else policy.get("max_task_completion_incomplete")
        ),
        "min_task_completion_check_pass_rate": (
            args.min_task_completion_check_pass_rate
            if args.min_task_completion_check_pass_rate is not None
            else policy.get("min_task_completion_check_pass_rate")
        ),
        "min_source_fingerprint_rate": (
            args.min_source_fingerprint_rate
            if args.min_source_fingerprint_rate is not None
            else policy.get("min_source_fingerprint_rate")
        ),
        "max_unverified_source_fingerprints": (
            args.max_unverified_source_fingerprints
            if args.max_unverified_source_fingerprints is not None
            else policy.get("max_unverified_source_fingerprints")
        ),
        "min_trainer_view_source_fingerprint_rate": (
            args.min_trainer_view_source_fingerprint_rate
            if args.min_trainer_view_source_fingerprint_rate is not None
            else policy.get("min_trainer_view_source_fingerprint_rate")
        ),
        "max_unverified_trainer_view_source_fingerprints": (
            args.max_unverified_trainer_view_source_fingerprints
            if args.max_unverified_trainer_view_source_fingerprints is not None
            else policy.get("max_unverified_trainer_view_source_fingerprints")
        ),
        "min_trace_average_events": (
            args.min_trace_average_events
            if args.min_trace_average_events is not None
            else policy.get("min_trace_average_events")
        ),
        "min_trace_event_type_count": (
            args.min_trace_event_type_count
            if args.min_trace_event_type_count is not None
            else policy.get("min_trace_event_type_count")
        ),
        "min_trace_final_answer_rate": (
            args.min_trace_final_answer_rate
            if args.min_trace_final_answer_rate is not None
            else policy.get("min_trace_final_answer_rate")
        ),
        "min_trace_tool_or_api_rate": (
            args.min_trace_tool_or_api_rate
            if args.min_trace_tool_or_api_rate is not None
            else policy.get("min_trace_tool_or_api_rate")
        ),
        "max_trace_empty_final_answers": (
            args.max_trace_empty_final_answers
            if args.max_trace_empty_final_answers is not None
            else policy.get("max_trace_empty_final_answers")
        ),
        "max_trace_risk_count": (
            args.max_trace_risk_count
            if args.max_trace_risk_count is not None
            else policy.get("max_trace_risk_count")
        ),
        "min_split_task_families": (
            args.min_split_task_families
            if args.min_split_task_families is not None
            else policy.get("min_split_task_families")
        ),
        "min_train_episodes": args.min_train_episodes if args.min_train_episodes is not None else policy.get("min_train_episodes"),
        "min_validation_episodes": (
            args.min_validation_episodes
            if args.min_validation_episodes is not None
            else policy.get("min_validation_episodes")
        ),
        "min_test_episodes": args.min_test_episodes if args.min_test_episodes is not None else policy.get("min_test_episodes"),
        "require_family_exclusive_splits": bool(
            args.require_family_exclusive_splits or policy.get("require_family_exclusive_splits", False)
        ),
        "max_quality_flags": args.max_quality_flags if args.max_quality_flags is not None else policy.get("max_quality_flags"),
        "forbid_quality_flags": _merge_unique_strings(policy.get("forbid_quality_flags", []), args.forbid_quality_flag),
        "forbid_quality_severities": _merge_unique_strings(
            policy.get("forbid_quality_severities", []),
            args.forbid_quality_severity,
        ),
        "require_task_families": _merge_unique_strings(policy.get("require_task_families", []), args.require_task_family),
        "require_trace_event_types": _merge_unique_strings(policy.get("require_trace_event_types", []), args.require_trace_event_type),
        "task_family_gates": policy.get("task_family_gates", []),
    }


def _goal3_training_gate_args(args: argparse.Namespace, training_export_dir: Path, gate_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        training_export=str(training_export_dir),
        policy=args.policy,
        out=str(gate_path),
        strict_validation=args.strict,
        skip_validation=False,
        min_episodes=None,
        min_pass_rate=None,
        min_average_score=None,
        min_preferences=None,
        min_sft=None,
        min_dpo=None,
        min_reward_model=None,
        min_step_rewards=None,
        min_task_completion_configured=None,
        min_task_completion_complete=None,
        max_task_completion_incomplete=None,
        min_task_completion_check_pass_rate=None,
        min_source_fingerprint_rate=None,
        max_unverified_source_fingerprints=None,
        min_trainer_view_source_fingerprint_rate=None,
        max_unverified_trainer_view_source_fingerprints=None,
        min_trace_average_events=None,
        min_trace_event_type_count=None,
        min_trace_final_answer_rate=None,
        min_trace_tool_or_api_rate=None,
        max_trace_empty_final_answers=None,
        max_trace_risk_count=None,
        require_trace_event_type=[],
        min_split_task_families=None,
        min_train_episodes=None,
        min_validation_episodes=None,
        min_test_episodes=None,
        require_family_exclusive_splits=False,
        max_quality_flags=None,
        forbid_quality_flag=[],
        forbid_quality_severity=[],
        require_task_family=[],
        preserve_paths=args.preserve_paths,
    )


def _training_gate_policy_summary(options: dict[str, Any]) -> dict[str, Any]:
    effective_fields = (
        "min_episodes",
        "min_pass_rate",
        "min_average_score",
        "min_preferences",
        "min_sft",
        "min_dpo",
        "min_reward_model",
        "min_step_rewards",
        "min_task_completion_configured",
        "min_task_completion_complete",
        "max_task_completion_incomplete",
        "min_task_completion_check_pass_rate",
        "min_source_fingerprint_rate",
        "max_unverified_source_fingerprints",
        "min_trainer_view_source_fingerprint_rate",
        "max_unverified_trainer_view_source_fingerprints",
        "min_trace_average_events",
        "min_trace_event_type_count",
        "min_trace_final_answer_rate",
        "min_trace_tool_or_api_rate",
        "max_trace_empty_final_answers",
        "max_trace_risk_count",
        "min_split_task_families",
        "min_train_episodes",
        "min_validation_episodes",
        "min_test_episodes",
        "require_family_exclusive_splits",
        "max_quality_flags",
        "forbid_quality_flags",
        "forbid_quality_severities",
        "require_task_families",
        "require_trace_event_types",
        "task_family_gates",
        "require_valid_export",
        "strict_validation",
    )
    effective = {
        field: options[field]
        for field in effective_fields
        if options.get(field) is not None and options.get(field) != []
    }
    summary: dict[str, Any] = {
        "schema_version": TRAINING_GATE_POLICY_SCHEMA_VERSION,
        "path": options["policy_path"],
        "effective": effective,
    }
    if options.get("policy_description"):
        summary["description"] = options["policy_description"]
    return summary


def _reviewed_gate_options(args: argparse.Namespace) -> dict[str, Any]:
    policy = load_reviewed_gate_policy(args.policy) if args.policy else {}
    return {
        "policy_path": _display_path(Path(args.policy), args.preserve_paths) if args.policy else None,
        "policy_description": policy.get("description"),
        "min_reviewed_labels": (
            args.min_reviewed_labels if args.min_reviewed_labels is not None else policy.get("min_reviewed_labels")
        ),
        "min_accepted": args.min_accepted if args.min_accepted is not None else policy.get("min_accepted"),
        "min_rejected": args.min_rejected if args.min_rejected is not None else policy.get("min_rejected"),
        "min_sft": args.min_sft if args.min_sft is not None else policy.get("min_sft"),
        "min_reward_model": args.min_reward_model if args.min_reward_model is not None else policy.get("min_reward_model"),
        "min_preferences": args.min_preferences if args.min_preferences is not None else policy.get("min_preferences"),
        "min_dpo": args.min_dpo if args.min_dpo is not None else policy.get("min_dpo"),
        "min_high_confidence_labels": (
            args.min_high_confidence_labels
            if args.min_high_confidence_labels is not None
            else policy.get("min_high_confidence_labels")
        ),
        "min_medium_or_high_confidence_labels": (
            args.min_medium_or_high_confidence_labels
            if args.min_medium_or_high_confidence_labels is not None
            else policy.get("min_medium_or_high_confidence_labels")
        ),
        "max_needs_review": args.max_needs_review if args.max_needs_review is not None else policy.get("max_needs_review"),
        "max_low_confidence_labels": (
            args.max_low_confidence_labels
            if args.max_low_confidence_labels is not None
            else policy.get("max_low_confidence_labels")
        ),
        "max_unknown_confidence_labels": (
            args.max_unknown_confidence_labels
            if args.max_unknown_confidence_labels is not None
            else policy.get("max_unknown_confidence_labels")
        ),
        "forbid_labels": _merge_unique_strings(policy.get("forbid_labels", []), args.forbid_label),
        "require_task_families": _merge_unique_strings(policy.get("require_task_families", []), args.require_task_family),
        "require_valid_export": False if args.skip_validation else policy.get("require_valid_export", True),
        "strict_validation": bool(args.strict_validation or policy.get("strict_validation", False)),
    }


def _reviewed_gate_policy_summary(options: dict[str, Any]) -> dict[str, Any]:
    effective_fields = (
        "min_reviewed_labels",
        "min_accepted",
        "min_rejected",
        "min_sft",
        "min_reward_model",
        "min_preferences",
        "min_dpo",
        "min_high_confidence_labels",
        "min_medium_or_high_confidence_labels",
        "max_needs_review",
        "max_low_confidence_labels",
        "max_unknown_confidence_labels",
        "forbid_labels",
        "require_task_families",
        "require_valid_export",
        "strict_validation",
    )
    effective = {
        field: options[field]
        for field in effective_fields
        if options.get(field) is not None and options.get(field) != []
    }
    summary: dict[str, Any] = {
        "schema_version": REVIEWED_GATE_POLICY_SCHEMA_VERSION,
        "path": options["policy_path"],
        "effective": effective,
    }
    if options.get("policy_description"):
        summary["description"] = options["policy_description"]
    return summary


def _compare_gate_options(args: argparse.Namespace) -> dict[str, Any]:
    policy = load_compare_gate_policy(args.policy) if args.policy else {}
    return {
        "policy_path": _display_path(Path(args.policy), args.preserve_paths) if args.policy else None,
        "policy_description": policy.get("description"),
        "require_valid_export": False if args.skip_validation else policy.get("require_valid_export", True),
        "strict_validation": bool(args.strict_validation or policy.get("strict_validation", False)),
        "min_pairs": args.min_pairs if args.min_pairs is not None else policy.get("min_pairs"),
        "min_dpo": args.min_dpo if args.min_dpo is not None else policy.get("min_dpo"),
        "min_candidate_wins": (
            args.min_candidate_wins if args.min_candidate_wins is not None else policy.get("min_candidate_wins")
        ),
        "min_task_completion_improvements": (
            args.min_task_completion_improvements
            if args.min_task_completion_improvements is not None
            else policy.get("min_task_completion_improvements")
        ),
        "max_baseline_wins": (
            args.max_baseline_wins if args.max_baseline_wins is not None else policy.get("max_baseline_wins")
        ),
        "max_task_completion_regressions": (
            args.max_task_completion_regressions
            if args.max_task_completion_regressions is not None
            else policy.get("max_task_completion_regressions")
        ),
        "max_skipped_pairs": args.max_skipped_pairs if args.max_skipped_pairs is not None else policy.get("max_skipped_pairs"),
        "max_contract_drifts": (
            args.max_contract_drifts if args.max_contract_drifts is not None else policy.get("max_contract_drifts")
        ),
        "max_unverified_contracts": (
            args.max_unverified_contracts
            if args.max_unverified_contracts is not None
            else policy.get("max_unverified_contracts")
        ),
        "require_scenarios": _merge_unique_strings(policy.get("require_scenarios", []), args.require_scenario),
        "require_candidate_win_scenarios": _merge_unique_strings(
            policy.get("require_candidate_win_scenarios", []),
            args.require_candidate_win_scenario,
        ),
        "require_task_completion_improvement_scenarios": _merge_unique_strings(
            policy.get("require_task_completion_improvement_scenarios", []),
            args.require_task_completion_improvement_scenario,
        ),
        "forbid_regression_scenarios": _merge_unique_strings(
            policy.get("forbid_regression_scenarios", []),
            args.forbid_regression_scenario,
        ),
        "forbid_task_completion_regression_scenarios": _merge_unique_strings(
            policy.get("forbid_task_completion_regression_scenarios", []),
            args.forbid_task_completion_regression_scenario,
        ),
        "require_rule_fixes": _merge_unique_strings(policy.get("require_rule_fixes", []), args.require_rule_fix),
        "forbid_rule_regressions": _merge_unique_strings(
            policy.get("forbid_rule_regressions", []),
            args.forbid_rule_regression,
        ),
        "forbid_new_critical_failures": _merge_unique_strings(
            policy.get("forbid_new_critical_failures", []),
            args.forbid_new_critical_failure,
        ),
        "task_family_gates": policy.get("task_family_gates", []),
    }


def _compare_gate_policy_summary(options: dict[str, Any]) -> dict[str, Any]:
    effective_fields = (
        "min_pairs",
        "min_dpo",
        "min_candidate_wins",
        "min_task_completion_improvements",
        "max_baseline_wins",
        "max_task_completion_regressions",
        "max_skipped_pairs",
        "max_contract_drifts",
        "max_unverified_contracts",
        "require_scenarios",
        "require_candidate_win_scenarios",
        "require_task_completion_improvement_scenarios",
        "forbid_regression_scenarios",
        "forbid_task_completion_regression_scenarios",
        "require_rule_fixes",
        "forbid_rule_regressions",
        "forbid_new_critical_failures",
        "task_family_gates",
        "require_valid_export",
        "strict_validation",
    )
    effective = {
        field: options[field]
        for field in effective_fields
        if options.get(field) is not None and options.get(field) != []
    }
    summary: dict[str, Any] = {
        "schema_version": COMPARE_GATE_POLICY_SCHEMA_VERSION,
        "path": options["policy_path"],
        "effective": effective,
    }
    if options.get("policy_description"):
        summary["description"] = options["policy_description"]
    return summary


def _action_ledger_gate_options(args: argparse.Namespace) -> dict[str, Any]:
    policy = load_action_ledger_gate_policy(args.policy) if args.policy else {}
    return {
        "policy_path": _display_path(Path(args.policy), args.preserve_paths) if args.policy else None,
        "policy_description": policy.get("description"),
        "min_bundles": args.min_bundles if args.min_bundles is not None else policy.get("min_bundles"),
        "max_open_actions": args.max_open_actions if args.max_open_actions is not None else policy.get("max_open_actions"),
        "max_new_actions": args.max_new_actions if args.max_new_actions is not None else policy.get("max_new_actions"),
        "max_recurring_actions": (
            args.max_recurring_actions if args.max_recurring_actions is not None else policy.get("max_recurring_actions")
        ),
        "min_resolved_actions": (
            args.min_resolved_actions if args.min_resolved_actions is not None else policy.get("min_resolved_actions")
        ),
        "forbid_open_priorities": _merge_unique_strings(policy.get("forbid_open_priorities", []), args.forbid_open_priority),
        "forbid_open_actions": _merge_unique_strings(policy.get("forbid_open_actions", []), args.forbid_open_action),
        "require_resolved_actions": _merge_unique_strings(policy.get("require_resolved_actions", []), args.require_resolved_action),
    }


def _action_ledger_gate_policy_summary(options: dict[str, Any]) -> dict[str, Any]:
    effective_fields = (
        "min_bundles",
        "max_open_actions",
        "max_new_actions",
        "max_recurring_actions",
        "min_resolved_actions",
        "forbid_open_priorities",
        "forbid_open_actions",
        "require_resolved_actions",
    )
    effective = {
        field: options[field]
        for field in effective_fields
        if options.get(field) is not None and options.get(field) != []
    }
    summary: dict[str, Any] = {
        "schema_version": ACTION_LEDGER_GATE_POLICY_SCHEMA_VERSION,
        "path": options["policy_path"],
        "effective": effective,
    }
    if options.get("policy_description"):
        summary["description"] = options["policy_description"]
    return summary


def _improvement_ledger_gate_options(args: argparse.Namespace) -> dict[str, Any]:
    policy = load_improvement_ledger_gate_policy(args.policy) if args.policy else {}
    return {
        "policy_path": _display_path(Path(args.policy), args.preserve_paths) if args.policy else None,
        "policy_description": policy.get("description"),
        "min_plans": args.min_plans if args.min_plans is not None else policy.get("min_plans"),
        "max_open_work_items": (
            args.max_open_work_items if args.max_open_work_items is not None else policy.get("max_open_work_items")
        ),
        "max_new_work_items": (
            args.max_new_work_items if args.max_new_work_items is not None else policy.get("max_new_work_items")
        ),
        "max_recurring_work_items": (
            args.max_recurring_work_items
            if args.max_recurring_work_items is not None
            else policy.get("max_recurring_work_items")
        ),
        "min_resolved_work_items": (
            args.min_resolved_work_items
            if args.min_resolved_work_items is not None
            else policy.get("min_resolved_work_items")
        ),
        "max_critical_open_work_items": (
            args.max_critical_open_work_items
            if args.max_critical_open_work_items is not None
            else policy.get("max_critical_open_work_items")
        ),
        "max_high_open_work_items": (
            args.max_high_open_work_items
            if args.max_high_open_work_items is not None
            else policy.get("max_high_open_work_items")
        ),
        "forbid_open_priorities": _merge_unique_strings(policy.get("forbid_open_priorities", []), args.forbid_open_priority),
        "forbid_open_categories": _merge_unique_strings(policy.get("forbid_open_categories", []), args.forbid_open_category),
        "forbid_open_work_keys": _merge_unique_strings(policy.get("forbid_open_work_keys", []), args.forbid_open_work_key),
        "require_open_work_keys": _merge_unique_strings(policy.get("require_open_work_keys", []), args.require_open_work_key),
        "require_resolved_work_keys": _merge_unique_strings(
            policy.get("require_resolved_work_keys", []),
            args.require_resolved_work_key,
        ),
    }


def _improvement_ledger_gate_policy_summary(options: dict[str, Any]) -> dict[str, Any]:
    effective_fields = (
        "min_plans",
        "max_open_work_items",
        "max_new_work_items",
        "max_recurring_work_items",
        "min_resolved_work_items",
        "max_critical_open_work_items",
        "max_high_open_work_items",
        "forbid_open_priorities",
        "forbid_open_categories",
        "forbid_open_work_keys",
        "require_open_work_keys",
        "require_resolved_work_keys",
    )
    effective = {
        field: options[field]
        for field in effective_fields
        if options.get(field) is not None and options.get(field) != []
    }
    summary: dict[str, Any] = {
        "schema_version": IMPROVEMENT_LEDGER_GATE_POLICY_SCHEMA_VERSION,
        "path": options["policy_path"],
        "effective": effective,
    }
    if options.get("policy_description"):
        summary["description"] = options["policy_description"]
    return summary


def _promotion_ledger_gate_options(args: argparse.Namespace) -> dict[str, Any]:
    policy = load_promotion_ledger_gate_policy(args.policy) if args.policy else {}
    return {
        "policy_path": _display_path(Path(args.policy), args.preserve_paths) if args.policy else None,
        "policy_description": policy.get("description"),
        "min_decisions": args.min_decisions if args.min_decisions is not None else policy.get("min_decisions"),
        "min_allowed_count": args.min_allowed_count if args.min_allowed_count is not None else policy.get("min_allowed_count"),
        "max_blocked_count": args.max_blocked_count if args.max_blocked_count is not None else policy.get("max_blocked_count"),
        "max_blocked_rate": args.max_blocked_rate if args.max_blocked_rate is not None else policy.get("max_blocked_rate"),
        "min_consecutive_allowed": (
            args.min_consecutive_allowed
            if args.min_consecutive_allowed is not None
            else policy.get("min_consecutive_allowed")
        ),
        "max_consecutive_blocked": (
            args.max_consecutive_blocked
            if args.max_consecutive_blocked is not None
            else policy.get("max_consecutive_blocked")
        ),
        "max_failed_decisions": (
            args.max_failed_decisions if args.max_failed_decisions is not None else policy.get("max_failed_decisions")
        ),
        "require_latest_recommendation": (
            args.require_latest_recommendation
            if args.require_latest_recommendation is not None
            else policy.get("require_latest_recommendation")
        ),
        "require_latest_passed": args.require_latest_passed or bool(policy.get("require_latest_passed")),
        "require_source_recommendations": _merge_unique_strings(
            policy.get("require_source_recommendations", []),
            args.require_source_recommendation,
        ),
        "forbid_source_recommendations": _merge_unique_strings(
            policy.get("forbid_source_recommendations", []),
            args.forbid_source_recommendation,
        ),
    }


def _promotion_ledger_gate_policy_summary(options: dict[str, Any]) -> dict[str, Any]:
    effective_fields = (
        "min_decisions",
        "min_allowed_count",
        "max_blocked_count",
        "max_blocked_rate",
        "min_consecutive_allowed",
        "max_consecutive_blocked",
        "max_failed_decisions",
        "require_latest_recommendation",
        "require_latest_passed",
        "require_source_recommendations",
        "forbid_source_recommendations",
    )
    effective = {
        field: options[field]
        for field in effective_fields
        if options.get(field) is not None and options.get(field) != []
    }
    summary: dict[str, Any] = {
        "schema_version": PROMOTION_LEDGER_GATE_POLICY_SCHEMA_VERSION,
        "path": options["policy_path"],
        "effective": effective,
    }
    if options.get("policy_description"):
        summary["description"] = options["policy_description"]
    return summary


def _merge_gate_rule_ids(policy_values: Any, cli_values: list[str]) -> list[str]:
    return _merge_unique_strings(policy_values, cli_values)


def _merge_unique_strings(policy_values: Any, cli_values: list[str]) -> list[str]:
    merged: list[str] = []
    for value in [*(policy_values or []), *cli_values]:
        if value not in merged:
            merged.append(value)
    return merged


def _load_digest_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    run_dir = Path(args.run) if args.run else None
    trace_path = Path(args.trace) if args.trace else (run_dir / "normalized_trace.json" if run_dir is not None else None)
    score_path = Path(args.score) if args.score else (run_dir / "scorecard.json" if run_dir is not None else None)
    state_diff_path = Path(args.state_diff) if args.state_diff else None
    if state_diff_path is None and run_dir is not None and (run_dir / "state_diff.json").exists():
        state_diff_path = run_dir / "state_diff.json"
    if trace_path is None or score_path is None:
        raise RunDigestError("--trace and --score are required when --run is not supplied")
    trace = _read_json(trace_path)
    scorecard = _read_json(score_path)
    state_diff = _read_json(state_diff_path) if state_diff_path is not None else None
    if args.scenario:
        scenario = load_scenario(args.scenario)
    elif run_dir is not None:
        scenario = _scenario_stub_from_run(run_dir, scorecard)
    else:
        raise RunDigestError("--scenario is required when --run is not supplied")
    return scenario, trace, scorecard, state_diff


def _scenario_stub_from_run(run_dir: Path, scorecard: dict[str, Any]) -> dict[str, Any]:
    lineage_path = run_dir / "artifact_lineage.json"
    if lineage_path.exists():
        lineage = _read_json(lineage_path)
        scenario_path = _lineage_record_path(lineage, "inputs", "scenario")
        if scenario_path is not None and not _looks_redacted_path(scenario_path):
            candidate = Path(scenario_path)
            if not candidate.is_absolute():
                candidate = (lineage_path.parent / candidate).resolve()
            if candidate.exists():
                return load_scenario(candidate)
        scenario = lineage.get("scenario") if isinstance(lineage.get("scenario"), dict) else {}
        if scenario.get("id") or scenario.get("title"):
            return {
                "id": str(scenario.get("id") or scorecard.get("scenario_id") or "unknown"),
                "title": str(scenario.get("title") or scorecard.get("scenario_title") or scenario.get("id") or "unknown"),
                "policy": {"secret_patterns": []},
                "assertions": {},
            }
    return {
        "id": str(scorecard.get("scenario_id") or "unknown"),
        "title": str(scorecard.get("scenario_title") or scorecard.get("scenario_id") or "unknown"),
        "policy": {"secret_patterns": []},
        "assertions": {},
    }


def _lineage_record_path(lineage: dict[str, Any], collection_name: str, record_name: str) -> str | None:
    records = lineage.get(collection_name)
    if not isinstance(records, list):
        return None
    for record in records:
        if isinstance(record, dict) and record.get("name") == record_name and isinstance(record.get("path"), str):
            return record["path"]
    return None


def _looks_redacted_path(value: str) -> bool:
    return value.startswith("<redacted:") or value.startswith("<missing-")


def _default_digest_out(args: argparse.Namespace) -> Path | None:
    return Path(args.run) / "run_digest.json" if args.run else None


def _run_scenario_artifacts(
    scenario_path: str | Path,
    out_dir: str | Path,
    *,
    trace_override: str | Path | None = None,
    state_override: str | Path | None = None,
    before_state_override: str | Path | None = None,
    trace_format: str = "auto",
    write_sensitive_trace: bool = False,
    preserve_paths: bool = False,
    junit_out: str | Path | None = None,
    markdown_out: str | Path | None = None,
) -> dict[str, Any]:
    scenario = load_scenario(scenario_path)
    trace_path = resolve_trace_path(scenario, trace_override)
    before_state_path = resolve_before_state_snapshot_path(scenario, before_state_override)
    state_path = resolve_state_snapshot_path(scenario, state_override)
    run_dir = Path(out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_trace = normalize_trace(trace_path, trace_format)
    raw_before_state_snapshot = load_state_snapshot(before_state_path) if before_state_path is not None else None
    raw_state_snapshot = load_state_snapshot(state_path) if state_path is not None else None
    trace_label = _display_path(trace_path, preserve_paths)
    scenario.setdefault("trace", {})["path"] = trace_label
    if before_state_path is not None:
        scenario.setdefault("state", {})["before_path"] = _display_path(before_state_path, preserve_paths)
        scenario["state"]["format"] = "json"
    if state_path is not None:
        scenario.setdefault("state", {})["path"] = _display_path(state_path, preserve_paths)
        scenario["state"]["format"] = "json"
    scorecard = score_trace(scenario, raw_trace, raw_state_snapshot, raw_before_state_snapshot)
    secret_patterns = scenario.get("policy", {}).get("secret_patterns") or []
    trace = sanitize_trace(raw_trace, secret_patterns)
    before_state_snapshot = (
        sanitize_state_snapshot(raw_before_state_snapshot, secret_patterns)
        if raw_before_state_snapshot is not None
        else None
    )
    state_snapshot = sanitize_state_snapshot(raw_state_snapshot, secret_patterns) if raw_state_snapshot is not None else None

    normalized_path = run_dir / "normalized_trace.json"
    score_path = run_dir / "scorecard.json"
    task_completion_path = run_dir / "task_completion.json"
    before_state_snapshot_path = run_dir / "before_state_snapshot.json" if before_state_snapshot is not None else None
    state_snapshot_path = run_dir / "state_snapshot.json" if state_snapshot is not None else None
    state_diff = (
        build_state_diff(before_state_snapshot, state_snapshot)
        if before_state_snapshot is not None and state_snapshot is not None
        else None
    )
    state_diff_path = run_dir / "state_diff.json" if state_diff is not None else None
    run_digest_path = run_dir / "run_digest.json"
    report_path = run_dir / "report.html"
    lineage_path = run_dir / "artifact_lineage.json"
    _write_json(normalized_path, trace)
    if before_state_snapshot_path is not None:
        _write_json(before_state_snapshot_path, before_state_snapshot)
    if state_snapshot_path is not None:
        _write_json(state_snapshot_path, state_snapshot)
    if state_diff_path is not None and state_diff is not None:
        _write_json(state_diff_path, state_diff)
    _write_json(score_path, scorecard)
    _write_json(task_completion_path, scorecard["task_completion"])
    run_digest = build_run_digest(scenario, trace, scorecard, state_diff=state_diff)
    _write_json(run_digest_path, run_digest)
    raw_trace_path = None
    if write_sensitive_trace:
        raw_trace_path = run_dir / "raw_trace.sensitive.json"
        _write_json(raw_trace_path, raw_trace)

    regression_path = None
    regression_display = None
    if not scorecard["passed"]:
        regression_path = run_dir / "regression_scenario.json"
        regression_display = _display_path(regression_path, preserve_paths)
        regression = _regression_scenario(scenario, trace_path, regression_path, preserve_paths)
        _write_json(regression_path, regression)

    if junit_out:
        write_junit(scorecard, junit_out)
    if markdown_out:
        write_markdown_summary(scorecard, markdown_out)
    write_report(scenario, trace, scorecard, report_path, regression_display, state_diff)
    lineage = write_run_lineage(
        scenario=scenario,
        trace=trace,
        scorecard=scorecard,
        source_trace_path=trace_path,
        source_before_state_snapshot_path=before_state_path,
        source_state_snapshot_path=state_path,
        artifacts={
            "normalized_trace": normalized_path,
            "before_state_snapshot": before_state_snapshot_path,
            "state_snapshot": state_snapshot_path,
            "state_diff": state_diff_path,
            "scorecard": score_path,
            "task_completion": task_completion_path,
            "run_digest": run_digest_path,
            "report": report_path,
            "regression_scenario": regression_path,
            "junit": junit_out,
            "markdown": markdown_out,
            "raw_trace_sensitive": raw_trace_path,
        },
        out_path=lineage_path,
        preserve_paths=preserve_paths,
    )

    return {
        "scenario": scenario,
        "trace_path": trace_path,
        "before_state_path": before_state_path,
        "state_path": state_path,
        "scorecard": scorecard,
        "paths": {
            "run_dir": run_dir,
            "normalized_trace": normalized_path,
            "before_state_snapshot": before_state_snapshot_path,
            "state_snapshot": state_snapshot_path,
            "state_diff": state_diff_path,
            "scorecard": score_path,
            "task_completion": task_completion_path,
            "run_digest": run_digest_path,
            "report": report_path,
            "lineage": lineage_path,
        },
        "lineage": lineage,
    }


def _write_run_suite_harness_handoff(
    out_dir: Path,
    runs: list[dict[str, Any]],
    *,
    summary_path: Path,
) -> dict[str, Path] | None:
    selected = _select_run_suite_harness_run(out_dir, runs)
    if selected is None:
        return None

    run, run_dir = selected
    harness_dir = out_dir / "harness_handoff"
    manifest_path = harness_dir / "harness_manifest.json"
    result_path = harness_dir / "harness_result.json"
    trace_path = run_dir / "normalized_trace.json"
    scorecard_path = run_dir / "scorecard.json"
    run_digest_path = run_dir / "run_digest.json"
    report_path = run_dir / "report.html"
    lineage_path = run_dir / "artifact_lineage.json"

    trace = _read_json(trace_path)
    scorecard = _read_json(scorecard_path)
    lineage = _read_json(lineage_path)
    scenario_id = str(run["scenario_id"])
    model_id = _run_suite_harness_model_id(trace)
    provider = _run_suite_harness_provider(trace)
    suite = _run_suite_harness_suite_context(
        harness_dir=harness_dir,
        out_dir=out_dir,
        summary_path=summary_path,
        runs=runs,
        selected_scenario_id=scenario_id,
        selected_run_dir=run_dir,
    )
    sandbox = {
        "root": _harness_relative_path(harness_dir, out_dir),
        "home": _harness_relative_path(harness_dir, out_dir),
        "workspace": _harness_relative_path(harness_dir, run_dir),
        "events": _harness_relative_path(harness_dir, trace_path),
        "fake_secret_canaries": [
            {
                "name": "HFR_RUN_SUITE_FAKE_SECRET_CANARY",
                "sha256": hashlib.sha256(b"HFR_RUN_SUITE_FAKE_SECRET_CANARY").hexdigest(),
            }
        ],
    }
    tool_policy = {
        "source": "flightrecorder run-suite --evidence-handoff",
        "scenario_policy": {},
        "runtime_policy": {
            "mode": "local_run_artifacts",
            "allowed_tools": [],
            "denied_tools": [],
            "network": {"mode": "not_required", "allowed_hosts": []},
        },
        "blocked_action_canaries": [],
    }

    manifest = {
        "schema_version": HARNESS_RUN_MANIFEST_SCHEMA_VERSION,
        "runner": "flightrecorder_run_suite",
        "provider": provider,
        "model": {"id": model_id},
        "scenario": {
            "id": scenario_id,
            "path": _run_suite_harness_scenario_path(lineage, run),
        },
        "outputs": {
            "run_dir": _harness_relative_path(harness_dir, run_dir),
            "manifest": "harness_manifest.json",
            "result": "harness_result.json",
        },
        "sandbox": sandbox,
        "tool_policy": tool_policy,
        "suite": suite,
    }
    result = {
        "schema_version": HARNESS_RUN_RESULT_SCHEMA_VERSION,
        "runner": "flightrecorder_run_suite",
        "provider": provider,
        "model": {"id": model_id},
        "scenario_id": scenario_id,
        "sandbox": sandbox,
        "tool_policy": tool_policy,
        "trace": {
            "path": _harness_relative_path(harness_dir, trace_path),
            "format": "normalized_json",
            "source_format": _run_suite_harness_source_format(trace),
        },
        "scorecard": {
            "path": _harness_relative_path(harness_dir, scorecard_path),
            "passed": scorecard.get("passed") is True,
            "score": scorecard.get("score", 0),
        },
        "artifacts": {
            "normalized_trace": _harness_relative_path(harness_dir, trace_path),
            "scorecard": _harness_relative_path(harness_dir, scorecard_path),
            "run_digest": _harness_relative_path(harness_dir, run_digest_path),
            "report": _harness_relative_path(harness_dir, report_path),
            "lineage": _harness_relative_path(harness_dir, lineage_path),
        },
        "replay": {
            "lineage": _harness_relative_path(harness_dir, lineage_path),
            "self_contained": _run_suite_harness_replay_self_contained(lineage),
        },
        "suite": suite,
    }
    _write_json(manifest_path, manifest)
    _write_json(result_path, result)
    return {"harness_manifest": manifest_path, "harness_result": result_path}


def _run_suite_harness_suite_context(
    *,
    harness_dir: Path,
    out_dir: Path,
    summary_path: Path,
    runs: list[dict[str, Any]],
    selected_scenario_id: str,
    selected_run_dir: Path,
) -> dict[str, Any]:
    passed = sum(1 for run in runs if run.get("passed") is True)
    total = len(runs)
    return {
        "source": "flightrecorder run-suite --evidence-handoff",
        "schema_version": RUN_SUITE_SCHEMA_VERSION,
        "summary": _harness_relative_path(harness_dir, summary_path),
        "runs_dir": _harness_relative_path(harness_dir, out_dir),
        "selected_scenario_id": selected_scenario_id,
        "selected_run_id": selected_run_dir.name,
        "selected_run_dir": _harness_relative_path(harness_dir, selected_run_dir),
        "total": total,
        "passed": passed,
        "failed": total - passed,
    }


def _select_run_suite_harness_run(out_dir: Path, runs: list[dict[str, Any]]) -> tuple[dict[str, Any], Path] | None:
    for run in sorted(runs, key=lambda item: str(item.get("scenario_id") or "")):
        scenario_id = run.get("scenario_id")
        if run.get("passed") is not True or not isinstance(scenario_id, str) or not scenario_id:
            continue
        run_dir = out_dir / _safe_run_id(scenario_id)
        required = (
            run_dir / "normalized_trace.json",
            run_dir / "scorecard.json",
            run_dir / "run_digest.json",
            run_dir / "report.html",
            run_dir / "artifact_lineage.json",
        )
        if all(path.exists() for path in required):
            return run, run_dir
    return None


def _run_suite_harness_model_id(trace: dict[str, Any]) -> str:
    session = trace.get("session") if isinstance(trace.get("session"), dict) else {}
    metadata = trace.get("metadata") if isinstance(trace.get("metadata"), dict) else {}
    for value in (session.get("model"), metadata.get("model"), metadata.get("model_id")):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def _run_suite_harness_provider(trace: dict[str, Any]) -> str:
    metadata = trace.get("metadata") if isinstance(trace.get("metadata"), dict) else {}
    for value in (metadata.get("provider"), metadata.get("source_provider")):
        if isinstance(value, str) and value.strip():
            return value.strip()
    source_format = _run_suite_harness_source_format(trace)
    normalized_source_format = source_format.lower()
    if "fixture" in normalized_source_format or "observer" in normalized_source_format:
        return "fixture"
    return "flightrecorder"


def _run_suite_harness_source_format(trace: dict[str, Any]) -> str:
    session = trace.get("session") if isinstance(trace.get("session"), dict) else {}
    value = session.get("source_format")
    return value.strip() if isinstance(value, str) and value.strip() else "unknown"


def _run_suite_harness_scenario_path(lineage: dict[str, Any], run: dict[str, Any]) -> str:
    path = _lineage_record_path(lineage, "inputs", "scenario")
    if isinstance(path, str) and path:
        return path
    scenario_path = run.get("scenario_path")
    return scenario_path if isinstance(scenario_path, str) and scenario_path else str(run.get("scenario_id") or "unknown")


def _run_suite_harness_replay_self_contained(lineage: dict[str, Any]) -> bool:
    replay = lineage.get("replay") if isinstance(lineage.get("replay"), dict) else {}
    return replay.get("self_contained") is True


def _harness_relative_path(base_dir: Path, path: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base_dir.resolve())).as_posix()


def _safe_run_id(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in value).strip("._-")
    return cleaned or "scenario"


def _run_suite_summary(
    *,
    scenarios_dir: Path,
    out_dir: Path,
    runs: list[dict[str, Any]],
    errors: list[dict[str, str]],
    artifacts: dict[str, str],
    preserve_paths: bool,
    training_manifest: dict[str, Any] | None,
    validation_summary: dict[str, Any] | None,
    metadata: dict[str, str] | None,
) -> dict[str, Any]:
    passed = sum(1 for run in runs if run["passed"])
    failed = len(runs) - passed
    summary: dict[str, Any] = {
        "schema_version": RUN_SUITE_SCHEMA_VERSION,
        "scenarios_dir": _display_path(scenarios_dir, preserve_paths),
        "out_dir": _display_path(out_dir, preserve_paths),
        "total": len(runs),
        "passed": passed,
        "failed": failed,
        "error_count": len(errors),
        "errors": errors,
        "metrics": _suite_metrics(runs),
        "runs": runs,
        "artifacts": artifacts,
    }
    if metadata:
        summary["metadata"] = dict(sorted(metadata.items()))
    if training_manifest is not None:
        summary["training_export"] = {
            "episode_count": training_manifest.get("episode_count"),
            "reward_count": training_manifest.get("reward_count"),
            "step_reward_count": training_manifest.get("step_reward_count"),
            "preference_count": training_manifest.get("preference_count"),
            "failure_mode_count": training_manifest.get("failure_mode_count"),
            "sft_count": training_manifest.get("sft_count"),
            "dpo_count": training_manifest.get("dpo_count"),
            "reward_model_count": training_manifest.get("reward_model_count"),
            "quality_flag_count": training_manifest.get("quality_flag_count"),
        }
    if validation_summary is not None:
        summary["validation"] = {
            "passed": validation_summary.get("passed"),
            "target_count": validation_summary.get("target_count"),
            "error_count": validation_summary.get("error_count"),
            "warning_count": validation_summary.get("warning_count"),
        }
    return summary


def _suite_metrics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [_int_score(run.get("score")) for run in runs]
    passed = sum(1 for run in runs if run.get("passed") is True)
    failed = len(runs) - passed
    return {
        "pass_rate": round(passed / len(runs), 4) if runs else 0.0,
        "average_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
        "min_score": min(scores) if scores else None,
        "max_score": max(scores) if scores else None,
        "failed_rule_counts": _count_values(
            str(rule_id)
            for run in runs
            for rule_id in run.get("failed_rules", [])
        ),
        "critical_failure_counts": _count_values(
            str(rule_id)
            for run in runs
            for rule_id in run.get("critical_failures", [])
        ),
        "task_families": _task_family_metrics(runs),
        "failed": failed,
        "passed": passed,
    }


def _task_family_metrics(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        buckets.setdefault(str(run.get("task_family") or _task_family(str(run.get("scenario_id") or ""))), []).append(run)

    metrics: list[dict[str, Any]] = []
    for family, family_runs in sorted(buckets.items()):
        scores = [_int_score(run.get("score")) for run in family_runs]
        passed = sum(1 for run in family_runs if run.get("passed") is True)
        metrics.append(
            {
                "task_family": family,
                "total": len(family_runs),
                "passed": passed,
                "failed": len(family_runs) - passed,
                "pass_rate": round(passed / len(family_runs), 4) if family_runs else 0.0,
                "average_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
                "failed_rule_counts": _count_values(
                    str(rule_id)
                    for run in family_runs
                    for rule_id in run.get("failed_rules", [])
                ),
                "critical_failure_counts": _count_values(
                    str(rule_id)
                    for run in family_runs
                    for rule_id in run.get("critical_failures", [])
                ),
            }
        )
    return metrics


def _count_values(values: Any) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for value in values:
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return [
        {"id": key, "count": count}
        for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _failed_rule_ids(scorecard: dict[str, Any]) -> list[str]:
    return [
        str(rule.get("id"))
        for rule in scorecard.get("rules", [])
        if isinstance(rule, dict) and rule.get("id") and not rule.get("passed")
    ]


def _lineage_input_hash(lineage: dict[str, Any], name: str) -> str | None:
    for record in lineage.get("inputs", []):
        if isinstance(record, dict) and record.get("name") == name and isinstance(record.get("sha256"), str):
            return record["sha256"]
    return None


def _default_replay_base_dir(lineage_path: Path, lineage: dict[str, Any]) -> Path:
    if isinstance(lineage.get("portable_replay_bundle"), dict):
        return lineage_path.parent
    return Path.cwd()


def _replay_flag_path(argv: list[str], flag: str, base_dir: Path, *, required: bool = True) -> Path | None:
    if flag not in argv:
        if required:
            raise ReplayError(f"artifact_lineage.replay.argv missing {flag}")
        return None
    index = argv.index(flag)
    if index + 1 >= len(argv) or not argv[index + 1]:
        raise ReplayError(f"artifact_lineage.replay.argv missing value for {flag}")
    raw = argv[index + 1]
    if raw.startswith("<redacted:") or raw.startswith("<missing-"):
        raise ReplayError(f"artifact_lineage.replay.argv contains non-replayable path for {flag}: {raw}")
    path = Path(raw)
    if not path.is_absolute():
        path = base_dir / path
    return path


def _verify_replay_input(name: str, path: Path, fingerprints: dict[str, Any]) -> None:
    record = fingerprints.get(name)
    if not isinstance(record, dict):
        raise ReplayError(f"artifact_lineage.replay.input_fingerprints missing {name}")
    expected = record.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ReplayError(f"artifact_lineage.replay.input_fingerprints.{name}.sha256 is missing")
    if not path.exists() or not path.is_file():
        raise ReplayError(f"replay input {name} not found: {path}")
    expected_size = record.get("size_bytes")
    if expected_size is not None:
        if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size < 0:
            raise ReplayError(f"artifact_lineage.replay.input_fingerprints.{name}.size_bytes is invalid")
        if path.stat().st_size != expected_size:
            raise ReplayError(f"replay input {name} size mismatch: {path}")
    actual = _sha256_file(path)
    if actual != expected:
        raise ReplayError(f"replay input {name} sha256 mismatch: {path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_replay_input(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "path": destination,
        "relative_path": f"inputs/{destination.name}",
        "source_path": source,
        "sha256": _sha256_file(destination),
        "size_bytes": destination.stat().st_size,
    }


def _trace_bundle_name(trace_path: Path) -> str:
    suffix = "".join(trace_path.suffixes)
    return "source_trace" + (suffix if suffix else ".jsonl")


def _portable_replay_lineage(
    *,
    lineage: dict[str, Any],
    source_lineage_path: Path,
    copied_inputs: dict[str, dict[str, Any]],
    preserve_paths: bool,
) -> dict[str, Any]:
    bundle_lineage = copy.deepcopy(lineage)
    replay = bundle_lineage.get("replay")
    if not isinstance(replay, dict):
        replay = {}
        bundle_lineage["replay"] = replay
    argv = replay.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise ReplayError("artifact_lineage.replay.argv must be a list of strings")
    argv = list(argv)
    _replace_replay_flag(argv, "--scenario", _bundle_relative_path(copied_inputs["scenario"]))
    _replace_replay_flag(argv, "--trace", _bundle_relative_path(copied_inputs["source_trace"]))
    _replace_replay_flag(argv, "--out", "replay")
    if "source_before_state_snapshot" in copied_inputs:
        _replace_replay_flag(
            argv,
            "--before-state",
            _bundle_relative_path(copied_inputs["source_before_state_snapshot"]),
            required=False,
        )
    if "source_state_snapshot" in copied_inputs:
        _replace_replay_flag(argv, "--state", _bundle_relative_path(copied_inputs["source_state_snapshot"]), required=False)
    replay["argv"] = argv
    replay["command"] = " ".join(shlex.quote(arg) for arg in argv)
    replay["self_contained"] = True
    input_fingerprints = replay.get("input_fingerprints")
    if not isinstance(input_fingerprints, dict):
        input_fingerprints = {}
        replay["input_fingerprints"] = input_fingerprints
    for name, copied in copied_inputs.items():
        record = input_fingerprints.get(name)
        if not isinstance(record, dict):
            record = {}
            input_fingerprints[name] = record
        record["path"] = _bundle_relative_path(copied)
        record["sha256"] = copied["sha256"]
        record["size_bytes"] = copied["size_bytes"]
        record["exists"] = True
    _rewrite_lineage_inputs_for_bundle(bundle_lineage, copied_inputs)
    summary = bundle_lineage.get("summary")
    if isinstance(summary, dict):
        summary["self_contained_replay"] = True
    bundle_lineage["portable_replay_bundle"] = {
        "schema_version": REPLAY_BUNDLE_SCHEMA_VERSION,
        "source_lineage": _display_path(source_lineage_path, preserve_paths),
        "input_count": len(copied_inputs),
        "notes": [
            "This lineage was rewritten for a portable replay bundle.",
            "Replay input paths are relative to the directory containing artifact_lineage.json.",
        ],
    }
    return bundle_lineage


def _rewrite_lineage_inputs_for_bundle(lineage: dict[str, Any], copied_inputs: dict[str, dict[str, Any]]) -> None:
    inputs = lineage.get("inputs")
    if not isinstance(inputs, list):
        return
    for record in inputs:
        if not isinstance(record, dict):
            continue
        name = record.get("name")
        copied = copied_inputs.get(name) if isinstance(name, str) else None
        if copied is None:
            continue
        record["path"] = _bundle_relative_path(copied)
        record["exists"] = True
        record["sha256"] = copied["sha256"]
        record["size_bytes"] = copied["size_bytes"]


def _replace_replay_flag(argv: list[str], flag: str, value: str, *, required: bool = True) -> None:
    if flag not in argv:
        if required:
            raise ReplayError(f"artifact_lineage.replay.argv missing {flag}")
        argv.extend([flag, value])
        return
    index = argv.index(flag)
    if index + 1 >= len(argv):
        raise ReplayError(f"artifact_lineage.replay.argv missing value for {flag}")
    argv[index + 1] = value


def _replay_bundle_manifest(
    *,
    bundle_lineage: dict[str, Any],
    bundle_lineage_path: Path,
    source_lineage_path: Path,
    copied_inputs: dict[str, dict[str, Any]],
    preserve_paths: bool,
) -> dict[str, Any]:
    replay = bundle_lineage.get("replay") if isinstance(bundle_lineage.get("replay"), dict) else {}
    return {
        "schema_version": REPLAY_BUNDLE_SCHEMA_VERSION,
        "lineage": bundle_lineage_path.name,
        "source_lineage": _display_path(source_lineage_path, preserve_paths),
        "input_count": len(copied_inputs),
        "inputs": [
            {
                "name": name,
                "path": _bundle_relative_path(copied),
                "sha256": copied["sha256"],
                "size_bytes": copied["size_bytes"],
                "source_path": _display_path(copied["source_path"], preserve_paths),
            }
            for name, copied in sorted(copied_inputs.items())
        ],
        "replay": {
            "argv": replay.get("argv", []),
            "command": replay.get("command", ""),
            "self_contained": replay.get("self_contained") is True,
        },
        "notes": [
            "Move this directory as a unit, then run flightrecorder replay --lineage artifact_lineage.json --out <fresh-run> from anywhere.",
            "The replay command verifies copied input hashes before regenerating artifacts.",
        ],
    }


def _bundle_relative_path(copied: dict[str, Any]) -> str:
    return str(copied["relative_path"])


def _int_score(value: Any) -> int:
    try:
        return max(0, min(100, int(value)))
    except (TypeError, ValueError):
        return 0


def _task_family(scenario_id: str) -> str:
    family = FAMILY_SUFFIX_RE.sub("", scenario_id).strip("_-")
    return family or scenario_id or "unknown"


def _regression_scenario(scenario: dict[str, Any], trace_path: Path, regression_path: Path, preserve_paths: bool) -> dict[str, Any]:
    regression = {
        key: value
        for key, value in scenario.items()
        if not key.startswith("_")
    }
    trace_ref = _display_path(trace_path, preserve_paths)
    scenario_ref = _display_path(regression_path, preserve_paths)
    regression["id"] = f"{scenario['id']}_regression"
    regression["title"] = f"Regression: {scenario['title']}"
    regression["trace"] = {"format": "auto", "path": trace_ref}
    if trace_ref.startswith("<redacted:"):
        regression["trace"]["path_redacted"] = True
    regression["rerun_command"] = (
        f"python -m flightrecorder run --scenario {shlex.quote(scenario_ref)} "
        f"--trace {shlex.quote(trace_ref)} --out {shlex.quote('runs/' + scenario['id'] + '_replay')}"
    )
    return regression


def _display_path(path: Path, preserve_paths: bool = False) -> str:
    raw = str(path)
    if preserve_paths:
        return raw
    if _is_windows_absolute(raw):
        return f"<redacted:{_basename(raw)}>"
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    try:
        return str(resolved.relative_to(cwd))
    except ValueError:
        return f"<redacted:{resolved.name}>"


def _display_path_for_output_source(path: Path, out_path: Path | None, preserve_paths: bool = False) -> str:
    if preserve_paths or out_path is None:
        return _display_path(path, preserve_paths)
    raw = str(path)
    if _is_windows_absolute(raw):
        return f"<redacted:{_basename(raw)}>"
    resolved = path.resolve()
    out_dir = out_path.parent.resolve()
    return os.path.relpath(resolved, out_dir)


def _is_windows_absolute(value: str) -> bool:
    normalized = value.replace("/", "\\")
    return (len(normalized) >= 3 and normalized[1:3] == ":\\" and normalized[0].isalpha()) or normalized.startswith("\\\\")


def _basename(value: str) -> str:
    return value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] or "path"


def _read_scorecard_ref(path: Path) -> tuple[dict[str, Any], str]:
    score_path = path / "scorecard.json" if path.is_dir() else path
    return _read_json(score_path), _display_path(score_path)


def _audit_runs(runs_dir: Path, forbidden_text: list[str]) -> dict[str, Any]:
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    if not runs_dir.is_dir():
        raise NotADirectoryError(f"Runs path is not a directory: {runs_dir}")

    scorecards: list[dict[str, Any]] = []
    leaks: list[dict[str, str]] = []
    for score_path in sorted(runs_dir.glob("*/scorecard.json")):
        scorecard = _read_json(score_path)
        scorecards.append(
            {
                "run": score_path.parent.name,
                "scenario_id": scorecard.get("scenario_id"),
                "passed": bool(scorecard.get("passed")),
                "score": scorecard.get("score"),
                "critical_failures": scorecard.get("critical_failures", []),
            }
        )

    needles = [needle for needle in forbidden_text if needle]
    if needles and runs_dir.exists():
        for path in sorted(runs_dir.rglob("*")):
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for needle in needles:
                if needle in text:
                    leaks.append({"path": str(path), "text": needle})

    passed = sum(1 for item in scorecards if item["passed"])
    failed = len(scorecards) - passed
    return {
        "runs_dir": str(runs_dir),
        "total": len(scorecards),
        "passed": passed,
        "failed": failed,
        "leaks": leaks,
        "scorecards": scorecards,
    }


OBSERVER_TEMPLATE = '''"""Read-only Hermes Flight Recorder observer plugin.

Install `hermes-flight-recorder`, set HERMES_FLIGHT_RECORDER_OUTPUT_DIR to a
restricted directory, then load this plugin through Hermes' plugin mechanism.
The collector records observer-hook JSONL only; it does not block or mutate
Hermes tools, prompts, memory, or model requests.
"""

from flightrecorder.hermes_plugin import register as register_flight_recorder


def register(ctx):
    return register_flight_recorder(ctx)
'''
