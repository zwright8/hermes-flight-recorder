"""Command line interface for Hermes Flight Recorder."""

from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path
from typing import Any

from .adapters import AdapterError, normalize_trace
from .artifacts import (
    ArtifactError,
    compare_scorecards,
    compare_suites,
    write_compare_report,
    write_junit,
    write_markdown_summary,
    write_suite_compare_report,
)
from .redaction import sanitize_trace
from .report import write_index, write_report
from .schema import ScenarioError, load_scenario, resolve_trace_path
from .scenario_check import check_scenarios, discover_scenarios
from .scorers import score_trace
from .training import TrainingExportError, export_rl_dataset
from .validation import validate_artifacts

RUN_SUITE_SCHEMA_VERSION = "hfr.run_suite.v1"
FAMILY_SUFFIX_RE = re.compile(r"([_-](good|bad|pass|fail|passing|failing|chosen|rejected))+$", re.IGNORECASE)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (AdapterError, ArtifactError, ScenarioError, TrainingExportError, OSError, json.JSONDecodeError) as exc:
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
    scorecard = score_trace(scenario, trace)
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


def cmd_run(args: argparse.Namespace) -> int:
    result = _run_scenario_artifacts(
        args.scenario,
        args.out,
        trace_override=args.trace,
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


def cmd_run_suite(args: argparse.Namespace) -> int:
    scenario_paths = discover_scenarios(Path(args.scenarios), args.pattern, args.recursive)
    out_dir = Path(args.out)
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
                    "trace_path": _display_path(result["trace_path"], args.preserve_paths),
                    "run_dir": _display_path(run_dir, args.preserve_paths),
                    "report": _display_path(result["paths"]["report"], args.preserve_paths),
                    "scorecard": _display_path(result["paths"]["scorecard"], args.preserve_paths),
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
            )
            artifacts["training_export"] = _display_path(training_out, args.preserve_paths)
        else:
            errors.append(
                {
                    "scenario_path": _display_path(Path(args.scenarios), args.preserve_paths),
                    "error": "Cannot export RL artifacts because no scenario runs completed.",
                }
            )

    validation_summary: dict[str, Any] | None = None
    validation_path = Path(args.validation_out) if args.validation_out else out_dir / "validation.json"
    if args.validate:
        validation_summary = validate_artifacts(
            runs_dir=out_dir,
            training_export_dir=training_out if args.export_rl else None,
            strict=args.strict,
        )
        _write_json(validation_path, validation_summary)
        artifacts["validation"] = _display_path(validation_path, args.preserve_paths)

    summary = _run_suite_summary(
        scenarios_dir=Path(args.scenarios),
        out_dir=out_dir,
        runs=runs,
        errors=errors,
        artifacts=artifacts,
        preserve_paths=args.preserve_paths,
        training_manifest=training_manifest,
        validation_summary=validation_summary,
    )
    summary_path = Path(args.summary_out) if args.summary_out else out_dir / "suite_summary.json"
    _write_json(summary_path, summary)
    print(
        f"SUITE total={summary['total']} passed={summary['passed']} failed={summary['failed']} "
        f"errors={summary['error_count']} summary={summary_path}"
    )

    if errors:
        return 1
    if args.validate and validation_summary and not validation_summary["passed"]:
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
    )
    _write_json(Path(args.out), comparison)
    if args.html_out:
        write_suite_compare_report(comparison, args.html_out)
    aggregate = comparison["aggregate"]
    print(
        f"{'REGRESSION' if comparison['regressed'] else 'NO REGRESSION'} "
        f"paired={aggregate['paired_count']} avg_score_delta={aggregate['avg_score_delta']} wrote {args.out}"
    )
    return 1 if args.fail_on_regression and comparison["regressed"] else 0


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


def cmd_validate(args: argparse.Namespace) -> int:
    summary = validate_artifacts(
        runs_dir=args.runs,
        run_dirs=args.run,
        training_export_dir=args.training_export,
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


def cmd_export_rl(args: argparse.Namespace) -> int:
    manifest = export_rl_dataset(
        args.runs,
        args.out,
        reward_scale=args.reward_scale,
        min_score_gap=args.min_score_gap,
        max_pairs_per_family=args.max_pairs_per_family,
        preserve_paths=args.preserve_paths,
    )
    print(
        "wrote RL export "
        f"episodes={manifest['episode_count']} rewards={manifest['reward_count']} "
        f"preferences={manifest['preference_count']} failure_modes={manifest['failure_mode_count']} out={args.out}"
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

    run = subparsers.add_parser("run", help="Normalize, score, and report in one command")
    run.add_argument("--scenario", required=True)
    run.add_argument("--trace")
    run.add_argument("--format", default="auto", choices=["auto", "trajectory_jsonl", "observer_jsonl", "atof_jsonl", "atif_json", "normalized_json"])
    run.add_argument("--out", required=True)
    run.add_argument("--write-sensitive-trace", action="store_true", help="Also write raw_trace.sensitive.json with unredacted evidence")
    run.add_argument("--preserve-paths", action="store_true", help="Allow absolute source paths in generated reports and regression files")
    run.add_argument("--junit-out", help="Also write a JUnit XML score report")
    run.add_argument("--markdown-out", help="Also write a Markdown score summary")
    run.add_argument("--fail-on-score", action="store_true", help="Exit nonzero when the scenario score fails")
    run.set_defaults(func=cmd_run)

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
    run_suite.add_argument("--export-rl", action="store_true", help="Also export episodes/rewards/preferences/failure modes for the completed suite")
    run_suite.add_argument("--training-export-out", help="RL export directory; defaults to <out>/training_export")
    run_suite.add_argument("--reward-scale", default="score", choices=["score", "binary", "signed"])
    run_suite.add_argument("--min-score-gap", type=int, default=1)
    run_suite.add_argument("--max-pairs-per-family", type=int, default=0)
    run_suite.add_argument("--validate", action="store_true", help="Also validate generated run and optional training artifacts")
    run_suite.add_argument("--validation-out", help="Validation JSON output path; defaults to <out>/validation.json")
    run_suite.add_argument("--strict", action="store_true", help="Treat validation warnings as validation failure")
    run_suite.add_argument("--write-sensitive-trace", action="store_true", help="Also write raw_trace.sensitive.json for each scenario")
    run_suite.add_argument("--preserve-paths", action="store_true", help="Allow absolute source paths in generated artifacts")
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
    compare_suite.add_argument("--fail-on-regression", action="store_true", help="Exit nonzero when the candidate suite regresses")
    compare_suite.set_defaults(func=cmd_compare_suite)

    check = subparsers.add_parser("check-scenarios", help="Validate scenario definitions before running them")
    check.add_argument("--scenarios", required=True, help="Directory containing scenario files")
    check.add_argument("--pattern", default="*.json", help="Scenario filename glob relative to --scenarios")
    check.add_argument("--recursive", action="store_true", help="Discover scenarios recursively with --pattern")
    check.add_argument("--out", help="Write scenario-check summary JSON to this path")
    check.add_argument("--require-traces", action="store_true", help="Fail when scenarios do not resolve to existing trace files")
    check.add_argument("--strict", action="store_true", help="Treat warnings as failure")
    check.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in generated check output")
    check.set_defaults(func=cmd_check_scenarios)

    validate = subparsers.add_parser("validate", help="Validate generated run and training artifacts")
    validate.add_argument("--run", action="append", default=[], help="Validate one run directory; may be repeated")
    validate.add_argument("--runs", help="Validate every completed run directory inside this runs directory")
    validate.add_argument("--training-export", help="Validate an export-rl output directory")
    validate.add_argument("--out", help="Write validation summary JSON to this path")
    validate.add_argument("--strict", action="store_true", help="Treat warnings as validation failure")
    validate.set_defaults(func=cmd_validate)

    export_rl = subparsers.add_parser("export-rl", help="Export completed runs as future RL training artifacts")
    export_rl.add_argument("--runs", required=True, help="Directory containing Flight Recorder run subdirectories")
    export_rl.add_argument("--out", required=True, help="Output directory for episodes/rewards/preferences/failure-mode artifacts")
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
    export_rl.set_defaults(func=cmd_export_rl)

    observer = subparsers.add_parser("observer-template", help="Print or write a read-only Hermes observer plugin template")
    observer.add_argument("--out", help="Write the template to this path instead of stdout")
    observer.add_argument("--force", action="store_true", help="Overwrite --out when it already exists")
    observer.set_defaults(func=cmd_observer_template)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_score_outputs(scorecard: dict[str, Any], args: argparse.Namespace) -> None:
    if getattr(args, "junit_out", None):
        write_junit(scorecard, args.junit_out)
    if getattr(args, "markdown_out", None):
        write_markdown_summary(scorecard, args.markdown_out)


def _run_scenario_artifacts(
    scenario_path: str | Path,
    out_dir: str | Path,
    *,
    trace_override: str | Path | None = None,
    trace_format: str = "auto",
    write_sensitive_trace: bool = False,
    preserve_paths: bool = False,
    junit_out: str | Path | None = None,
    markdown_out: str | Path | None = None,
) -> dict[str, Any]:
    scenario = load_scenario(scenario_path)
    trace_path = resolve_trace_path(scenario, trace_override)
    run_dir = Path(out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_trace = normalize_trace(trace_path, trace_format)
    trace_label = _display_path(trace_path, preserve_paths)
    scenario.setdefault("trace", {})["path"] = trace_label
    scorecard = score_trace(scenario, raw_trace)
    secret_patterns = scenario.get("policy", {}).get("secret_patterns") or []
    trace = sanitize_trace(raw_trace, secret_patterns)

    normalized_path = run_dir / "normalized_trace.json"
    score_path = run_dir / "scorecard.json"
    report_path = run_dir / "report.html"
    _write_json(normalized_path, trace)
    _write_json(score_path, scorecard)
    if write_sensitive_trace:
        _write_json(run_dir / "raw_trace.sensitive.json", raw_trace)

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

    return {
        "scenario": scenario,
        "trace_path": trace_path,
        "scorecard": scorecard,
        "paths": {
            "run_dir": run_dir,
            "normalized_trace": normalized_path,
            "scorecard": score_path,
            "report": report_path,
        },
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
    if training_manifest is not None:
        summary["training_export"] = {
            "episode_count": training_manifest.get("episode_count"),
            "reward_count": training_manifest.get("reward_count"),
            "preference_count": training_manifest.get("preference_count"),
            "failure_mode_count": training_manifest.get("failure_mode_count"),
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
