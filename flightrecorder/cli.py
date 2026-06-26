"""Command line interface for Hermes Flight Recorder."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shlex
import shutil
from pathlib import Path
from typing import Any

from .adapters import AdapterError, normalize_trace
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
from .bundle import EvidenceBundleError, build_evidence_bundle
from .calibration import ReviewCalibrationError, build_review_calibration
from .compare_gate import (
    COMPARE_GATE_POLICY_SCHEMA_VERSION,
    CompareGatePolicyError,
    evaluate_compare_gate,
    load_compare_gate_policy,
)
from .evidence import EvidenceCoverageError, build_evidence_coverage
from .lineage import REPLAY_BUNDLE_SCHEMA_VERSION, write_run_lineage
from .redaction import sanitize_trace
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
from .scenario_check import check_scenarios, discover_scenarios
from .scenario_draft import draft_scenario, safe_scenario_id, score_draft, title_from_id
from .scenario_quality import build_scenario_quality
from .scorers import score_trace
from .state import StateSnapshotError, load_state_snapshot, resolve_state_snapshot_path, sanitize_state_snapshot
from .state_capture import StateCaptureError, capture_state_snapshot
from .suite_gate import SUITE_GATE_POLICY_SCHEMA_VERSION, SuiteGatePolicyError, evaluate_suite_gate, load_gate_policy
from .trace_observability import TraceObservabilityError, build_trace_observability
from .training import TrainingExportError, export_compare_rl_dataset, export_rl_dataset
from .training_gate import (
    TRAINING_GATE_POLICY_SCHEMA_VERSION,
    TrainingGatePolicyError,
    evaluate_training_gate,
    load_training_gate_policy,
)
from .validation import validate_artifacts

RUN_SUITE_SCHEMA_VERSION = "hfr.run_suite.v1"
FAMILY_SUFFIX_RE = re.compile(r"([_-](good|bad|pass|fail|passing|failing|chosen|rejected))+$", re.IGNORECASE)


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
        StateSnapshotError,
        SuiteGatePolicyError,
        ReviewExportError,
        ReviewedGatePolicyError,
        RepairQueueError,
        TrainingExportError,
        TrainingGatePolicyError,
        CompareGatePolicyError,
        EvidenceCoverageError,
        EvidenceBundleError,
        ReviewCalibrationError,
        TraceObservabilityError,
        ReplayError,
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
    state_snapshot = load_state_snapshot(state_path) if state_path is not None else None
    scorecard = score_trace(scenario, trace, state_snapshot)
    _write_json(Path(args.out), scorecard)
    _write_score_outputs(scorecard, args)
    print(f"wrote {args.out}")
    return 0 if scorecard["passed"] else 1


def cmd_report(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario)
    trace = _read_json(Path(args.trace))
    scorecard = _read_json(Path(args.score))
    write_report(scenario, trace, scorecard, args.out)
    print(f"wrote {args.out}")
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


def cmd_run(args: argparse.Namespace) -> int:
    result = _run_scenario_artifacts(
        args.scenario,
        args.out,
        trace_override=args.trace,
        state_override=args.state,
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
    fingerprints = replay.get("input_fingerprints") if isinstance(replay.get("input_fingerprints"), dict) else {}
    _verify_replay_input("scenario", scenario_path, fingerprints)
    _verify_replay_input("source_trace", trace_path, fingerprints)
    if state_path is not None:
        _verify_replay_input("source_state_snapshot", state_path, fingerprints)

    result = _run_scenario_artifacts(
        scenario_path,
        args.out,
        trace_override=trace_path,
        state_override=state_path,
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
    fingerprints = replay.get("input_fingerprints") if isinstance(replay.get("input_fingerprints"), dict) else {}
    _verify_replay_input("scenario", scenario_path, fingerprints)
    _verify_replay_input("source_trace", trace_path, fingerprints)
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


def cmd_run_suite(args: argparse.Namespace) -> int:
    scenario_paths = discover_scenarios(Path(args.scenarios), args.pattern, args.recursive)
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
                    "state_path": _display_path(result["state_path"], args.preserve_paths) if result.get("state_path") else None,
                    "state_sha256": _lineage_input_hash(result["lineage"], "source_state_snapshot"),
                    "run_dir": _display_path(run_dir, args.preserve_paths),
                    "report": _display_path(result["paths"]["report"], args.preserve_paths),
                    "scorecard": _display_path(result["paths"]["scorecard"], args.preserve_paths),
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
        write_index(completed_run_dirs, index_path)
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
            artifacts["evidence_bundle"] = _display_path(handoff_paths["evidence_bundle"], args.preserve_paths)
        else:
            errors.append(
                {
                    "scenario_path": _display_path(Path(args.scenarios), args.preserve_paths),
                    "error": "Cannot build evidence handoff because no scenario runs completed.",
                }
            )

    validation_summary: dict[str, Any] | None = None
    validation_path = Path(args.validation_out) if args.validation_out else out_dir / "validation.json"
    if args.validate:
        artifacts["validation"] = _display_path(validation_path, args.preserve_paths)

    summary_path = Path(args.summary_out) if args.summary_out else out_dir / "suite_summary.json"
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
            preserve_paths=args.preserve_paths,
        )
        _write_json(handoff_paths["evidence_bundle"], handoff_bundle)

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


def cmd_index(args: argparse.Namespace) -> int:
    runs_dir = Path(args.runs)
    run_dirs = sorted(path for path in runs_dir.iterdir() if path.is_dir())
    write_index(run_dirs, args.out)
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
        repair_queue_paths=args.repair_queue,
        replay_bundle_paths=args.replay_bundle,
        trace_observability_paths=args.trace_observability,
        review_calibration_paths=args.review_calibration,
        scenario_quality_paths=args.scenario_quality,
        suite_summary_paths=args.suite_summary,
        suite_trend_paths=args.suite_trend,
        state_snapshot_paths=args.state_snapshot,
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
        training_export_dir=args.training_export,
        compare_export_dir=args.compare_export,
        review_export_dir=args.review_export,
        reviewed_export_dir=args.reviewed_export,
        review_calibration_path=args.review_calibration,
        gate_paths=args.gate,
        preserve_paths=args.preserve_paths,
    )
    _write_json(Path(args.out), bundle)
    print(f"wrote {args.out}")
    return 0 if bundle["passed"] else 1


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
    calibration = build_review_calibration(
        args.reviewed_export,
        min_agreement_rate=args.min_agreement_rate,
        max_disagreements=args.max_disagreements,
        max_false_positives=args.max_false_positives,
        max_false_negatives=args.max_false_negatives,
        min_comparable_labels=args.min_comparable_labels,
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
        min_trace_average_events=options["min_trace_average_events"],
        min_trace_event_type_count=options["min_trace_event_type_count"],
        min_trace_final_answer_rate=options["min_trace_final_answer_rate"],
        min_trace_tool_or_api_rate=options["min_trace_tool_or_api_rate"],
        max_trace_empty_final_answers=options["max_trace_empty_final_answers"],
        max_trace_risk_count=options["max_trace_risk_count"],
        max_quality_flags=options["max_quality_flags"],
        forbid_quality_flags=options["forbid_quality_flags"],
        forbid_quality_severities=options["forbid_quality_severities"],
        require_task_families=options["require_task_families"],
        require_trace_event_types=options["require_trace_event_types"],
        task_family_gates=options["task_family_gates"],
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
        max_needs_review=options["max_needs_review"],
        forbid_labels=options["forbid_labels"],
        require_task_families=options["require_task_families"],
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
    normalize.add_argument("--format", default="auto", choices=["auto", "trajectory_jsonl", "observer_jsonl", "atof_jsonl", "atif_json", "normalized_json"])
    normalize.add_argument("--out", required=True)
    normalize.add_argument("--secret-pattern", action="append", default=[], help="Additional regex to redact from normalized output")
    normalize.add_argument("--no-redact", action="store_true", help="Write raw normalized trace without redaction")
    normalize.set_defaults(func=cmd_normalize)

    score = subparsers.add_parser("score", help="Score a normalized trace against a scenario")
    score.add_argument("--scenario", required=True)
    score.add_argument("--trace", required=True)
    score.add_argument("--state", help="Optional JSON state snapshot for required_state assertions")
    score.add_argument("--out", required=True)
    score.add_argument("--junit-out", help="Also write a JUnit XML score report")
    score.add_argument("--markdown-out", help="Also write a Markdown score summary")
    score.set_defaults(func=cmd_score)

    report = subparsers.add_parser("report", help="Render a static HTML report")
    report.add_argument("--scenario", required=True)
    report.add_argument("--trace", required=True)
    report.add_argument("--score", required=True)
    report.add_argument("--out", required=True)
    report.set_defaults(func=cmd_report)

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

    run = subparsers.add_parser("run", help="Normalize, score, and report in one command")
    run.add_argument("--scenario", required=True)
    run.add_argument("--trace")
    run.add_argument("--state", help="Optional JSON state snapshot for required_state assertions")
    run.add_argument("--format", default="auto", choices=["auto", "trajectory_jsonl", "observer_jsonl", "atof_jsonl", "atif_json", "normalized_json"])
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
    replay.add_argument("--format", default="auto", choices=["auto", "trajectory_jsonl", "observer_jsonl", "atof_jsonl", "atif_json", "normalized_json"])
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
    run_suite.add_argument("--format", default="auto", choices=["auto", "trajectory_jsonl", "observer_jsonl", "atof_jsonl", "atif_json", "normalized_json"])
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
        help="Also write scenario quality, evidence coverage, trace observability, and evidence bundle artifacts",
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
    validate.add_argument("--repair-queue", action="append", default=[], help="Validate one repair_queue.json; may be repeated")
    validate.add_argument("--replay-bundle", action="append", default=[], help="Validate one replay-bundle directory or replay_bundle.json; may be repeated")
    validate.add_argument("--trace-observability", action="append", default=[], help="Validate one trace_observability.json; may be repeated")
    validate.add_argument("--review-calibration", action="append", default=[], help="Validate one review_calibration.json; may be repeated")
    validate.add_argument("--scenario-quality", action="append", default=[], help="Validate one scenario_quality.json; may be repeated")
    validate.add_argument("--suite-summary", action="append", default=[], help="Validate one run-suite suite_summary.json; may be repeated")
    validate.add_argument("--suite-trend", action="append", default=[], help="Validate one trend-suite suite_trend.json; may be repeated")
    validate.add_argument("--state-snapshot", action="append", default=[], help="Validate one hfr.state_snapshot.v1 JSON file; may be repeated")
    validate.add_argument("--out", help="Write validation summary JSON to this path")
    validate.add_argument("--strict", action="store_true", help="Treat warnings as validation failure")
    validate.set_defaults(func=cmd_validate)

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
    evidence_bundle.add_argument("--training-export", help="export-rl directory included in the handoff")
    evidence_bundle.add_argument("--compare-export", help="export-compare-rl directory included in the handoff")
    evidence_bundle.add_argument("--review-export", help="export-review directory included in the handoff")
    evidence_bundle.add_argument("--reviewed-export", help="apply-review directory included in the handoff")
    evidence_bundle.add_argument("--review-calibration", help="review_calibration.json included in the handoff")
    evidence_bundle.add_argument("--gate", action="append", default=[], help="Gate result JSON to require; may be repeated")
    evidence_bundle.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in the bundle summary")
    evidence_bundle.set_defaults(func=cmd_evidence_bundle)

    draft = subparsers.add_parser("draft-scenario", help="Draft a scenario JSON file from an existing run or trace")
    draft_source = draft.add_mutually_exclusive_group(required=True)
    draft_source.add_argument("--trace", help="Trace artifact to normalize and use as source evidence")
    draft_source.add_argument("--run", help="Run directory containing normalized_trace.json")
    draft.add_argument("--format", default="auto", choices=["auto", "trajectory_jsonl", "observer_jsonl", "atof_jsonl", "atif_json", "normalized_json"])
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
    gate_export.add_argument("--min-trace-average-events", type=_non_negative_float_arg, help="Minimum average normalized events per exported episode")
    gate_export.add_argument("--min-trace-event-type-count", type=_non_negative_int_arg, help="Minimum distinct normalized event types in the export")
    gate_export.add_argument("--min-trace-final-answer-rate", type=_rate_arg, help="Minimum fraction of episodes with final answers")
    gate_export.add_argument("--min-trace-tool-or-api-rate", type=_rate_arg, help="Minimum fraction of episodes with tool or API events")
    gate_export.add_argument("--max-trace-empty-final-answers", type=_non_negative_int_arg, help="Maximum exported episodes allowed to have empty final answers")
    gate_export.add_argument("--max-trace-risk-count", type=_non_negative_int_arg, help="Maximum total trace observability risks allowed")
    gate_export.add_argument("--require-trace-event-type", action="append", default=[], help="Fail unless this normalized event type appears in exported traces")
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
    gate_reviewed.add_argument("--max-needs-review", type=_non_negative_int_arg, help="Maximum allowed needs_review labels")
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
        "min_trace_average_events",
        "min_trace_event_type_count",
        "min_trace_final_answer_rate",
        "min_trace_tool_or_api_rate",
        "max_trace_empty_final_answers",
        "max_trace_risk_count",
        "max_quality_flags",
        "forbid_quality_flags",
        "forbid_quality_severities",
        "require_task_families",
        "require_trace_event_types",
        "task_family_gates",
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
        "max_needs_review": args.max_needs_review if args.max_needs_review is not None else policy.get("max_needs_review"),
        "forbid_labels": _merge_unique_strings(policy.get("forbid_labels", []), args.forbid_label),
        "require_task_families": _merge_unique_strings(policy.get("require_task_families", []), args.require_task_family),
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
        "max_needs_review",
        "forbid_labels",
        "require_task_families",
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


def _merge_gate_rule_ids(policy_values: Any, cli_values: list[str]) -> list[str]:
    return _merge_unique_strings(policy_values, cli_values)


def _merge_unique_strings(policy_values: Any, cli_values: list[str]) -> list[str]:
    merged: list[str] = []
    for value in [*(policy_values or []), *cli_values]:
        if value not in merged:
            merged.append(value)
    return merged


def _run_scenario_artifacts(
    scenario_path: str | Path,
    out_dir: str | Path,
    *,
    trace_override: str | Path | None = None,
    state_override: str | Path | None = None,
    trace_format: str = "auto",
    write_sensitive_trace: bool = False,
    preserve_paths: bool = False,
    junit_out: str | Path | None = None,
    markdown_out: str | Path | None = None,
) -> dict[str, Any]:
    scenario = load_scenario(scenario_path)
    trace_path = resolve_trace_path(scenario, trace_override)
    state_path = resolve_state_snapshot_path(scenario, state_override)
    run_dir = Path(out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_trace = normalize_trace(trace_path, trace_format)
    raw_state_snapshot = load_state_snapshot(state_path) if state_path is not None else None
    trace_label = _display_path(trace_path, preserve_paths)
    scenario.setdefault("trace", {})["path"] = trace_label
    if state_path is not None:
        scenario.setdefault("state", {})["path"] = _display_path(state_path, preserve_paths)
        scenario["state"]["format"] = "json"
    scorecard = score_trace(scenario, raw_trace, raw_state_snapshot)
    secret_patterns = scenario.get("policy", {}).get("secret_patterns") or []
    trace = sanitize_trace(raw_trace, secret_patterns)
    state_snapshot = sanitize_state_snapshot(raw_state_snapshot, secret_patterns) if raw_state_snapshot is not None else None

    normalized_path = run_dir / "normalized_trace.json"
    score_path = run_dir / "scorecard.json"
    task_completion_path = run_dir / "task_completion.json"
    state_snapshot_path = run_dir / "state_snapshot.json" if state_snapshot is not None else None
    report_path = run_dir / "report.html"
    lineage_path = run_dir / "artifact_lineage.json"
    _write_json(normalized_path, trace)
    if state_snapshot_path is not None:
        _write_json(state_snapshot_path, state_snapshot)
    _write_json(score_path, scorecard)
    _write_json(task_completion_path, scorecard["task_completion"])
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
    write_report(scenario, trace, scorecard, report_path, regression_display)
    lineage = write_run_lineage(
        scenario=scenario,
        trace=trace,
        scorecard=scorecard,
        source_trace_path=trace_path,
        source_state_snapshot_path=state_path,
        artifacts={
            "normalized_trace": normalized_path,
            "state_snapshot": state_snapshot_path,
            "scorecard": score_path,
            "task_completion": task_completion_path,
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
        "state_path": state_path,
        "scorecard": scorecard,
        "paths": {
            "run_dir": run_dir,
            "normalized_trace": normalized_path,
            "state_snapshot": state_snapshot_path,
            "scorecard": score_path,
            "task_completion": task_completion_path,
            "report": report_path,
            "lineage": lineage_path,
        },
        "lineage": lineage,
    }


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
