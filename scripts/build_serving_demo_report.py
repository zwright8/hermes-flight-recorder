#!/usr/bin/env python3
"""Build a replayable serving demo report from held-out eval artifacts."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "hfr.serving_demo_run.v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", default=[], metavar="NAME=PATH", help="Arm name and evaluation_summary.json or suite_summary.json path")
    parser.add_argument("--baseline", type=Path, help="Shortcut for --arm baseline=PATH")
    parser.add_argument("--trace-only", type=Path, help="Shortcut for --arm trace_only=PATH")
    parser.add_argument("--flightrecorder", type=Path, help="Shortcut for --arm flightrecorder=PATH")
    parser.add_argument("--candidate-arm", default="")
    parser.add_argument("--endpoint-suite", type=Path, help="Optional serving_endpoint_suite.json readiness artifact")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    arms = [_load_arm(name, path) for name, path in _arm_specs(args)]
    if len(arms) < 2:
        raise SystemExit("At least two arms are required")
    candidate_name = _candidate_name(args.candidate_arm, [arm["name"] for arm in arms])
    endpoint_suite = _load_endpoint_suite(args.endpoint_suite) if args.endpoint_suite else None
    demo = build_demo(arms, candidate_name=candidate_name, endpoint_suite=endpoint_suite)
    _relativize_demo_paths(demo, args.report.parent)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.out, demo)
    args.report.write_text(render_report(demo), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "report": str(args.report), "claim_count": len(demo["claims"])}, indent=2))
    return 0


def build_demo(
    arms: list[dict[str, Any]],
    *,
    candidate_name: str,
    endpoint_suite: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scenario_sets = {arm["name"]: arm["scenario_ids"] for arm in arms}
    same_scenarios = len({tuple(ids) for ids in scenario_sets.values()}) == 1
    scenarios = _scenario_rows(arms)
    candidate = next(arm for arm in arms if arm["name"] == candidate_name)
    references = [arm for arm in arms if arm["name"] != candidate_name]
    comparisons = _comparisons(candidate, references, scenarios, same_scenarios=same_scenarios)
    demo = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "candidate_arm": candidate_name,
        "same_scenario_ids": same_scenarios,
        "scenario_sets": scenario_sets,
        "arms": [{"name": arm["name"], "source": arm["source"], "model": arm["model"], "base_url": arm["base_url"], "serving_profile": arm.get("serving_profile"), "metrics": arm["metrics"]} for arm in arms],
        "comparisons": comparisons,
        "claims": _claims(candidate, references, scenarios, same_scenarios=same_scenarios),
        "scenarios": scenarios,
    }
    if endpoint_suite is not None:
        endpoint_suite["demo_alignment"] = _endpoint_alignment(arms, endpoint_suite)
        demo["endpoint_suite"] = endpoint_suite
    return demo


def render_report(demo: dict[str, Any]) -> str:
    lines = [
        "# Serving Demo Replay Report",
        "",
        f"- Candidate arm: `{demo['candidate_arm']}`",
        f"- Same scenario ids: {demo['same_scenario_ids']}",
        "",
        "## Arm Metrics",
        "",
        "| Arm | Model | Serving Profile | Pass Rate | Average Score | Passed | Failed | Critical Failures |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in demo["arms"]:
        metrics = arm["metrics"]
        lines.append(f"| {arm['name']} | `{arm.get('model') or ''}` | {_md_link('serving_profile', arm.get('serving_profile'))} | {metrics.get('pass_rate')} | {metrics.get('average_score')} | {metrics.get('passed')} | {metrics.get('failed')} | {metrics.get('critical_failure_total')} |")
    lines.extend(["", "## Base Vs Candidate Comparisons", ""])
    comparisons = demo.get("comparisons") if isinstance(demo.get("comparisons"), list) else []
    if not comparisons:
        lines.append("- No base-vs-candidate comparisons were generated.")
    else:
        lines.extend(
            [
                "| Candidate | Reference | Pass Rate Delta | Average Score Delta | Passed Delta | Failed Delta | Critical Failure Delta |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for comparison in comparisons:
            deltas = comparison.get("metric_deltas") if isinstance(comparison.get("metric_deltas"), dict) else {}
            lines.append(
                f"| {comparison.get('candidate_arm')} | {comparison.get('reference_arm')} | {deltas.get('pass_rate')} | {deltas.get('average_score')} | {deltas.get('passed')} | {deltas.get('failed')} | {deltas.get('critical_failure_total')} |"
            )
        lines.extend(["", "| Scenario | Reference | Candidate Passed | Reference Passed | Candidate Score | Reference Score | Outcome |", "| --- | --- | ---: | ---: | ---: | ---: | --- |"])
        for comparison in comparisons:
            for outcome in comparison.get("scenario_outcomes", []):
                lines.append(
                    f"| {outcome.get('scenario_id')} | {comparison.get('reference_arm')} | {outcome.get('candidate_passed')} | {outcome.get('reference_passed')} | {outcome.get('candidate_score')} | {outcome.get('reference_score')} | {outcome.get('outcome')} |"
                )
    endpoint_suite = demo.get("endpoint_suite")
    if isinstance(endpoint_suite, dict):
        lines.extend(
            [
                "",
                "## Serving Endpoint Readiness",
                "",
                f"- Suite passed: {endpoint_suite.get('passed')}",
                f"- Demo aligned: {(endpoint_suite.get('demo_alignment') or {}).get('passed')}",
                f"- Failed checks: {', '.join(endpoint_suite.get('failed_checks') or []) or 'none'}",
                f"- Suite artifact: {_md_link('serving_endpoint_suite', endpoint_suite.get('path'))}",
                "",
                "| Arm | Ready | Served Model | Tool Calls | Structured Outputs | Profile | Lifecycle |",
                "| --- | ---: | --- | --- | --- | --- | --- |",
            ]
        )
        for arm in endpoint_suite.get("arms") or []:
            lines.append(
                "| {arm} | {ready} | `{model}` | {tools} | {structured} | {profile} | {lifecycle} |".format(
                    arm=arm.get("arm") or "",
                    ready=arm.get("ready_for_eval"),
                    model=arm.get("served_model_id") or "",
                    tools=arm.get("tool_calls"),
                    structured=arm.get("structured_outputs"),
                    profile=_md_link("profile", arm.get("profile_path")),
                    lifecycle=_md_link("lifecycle", arm.get("lifecycle_path")),
                )
            )
        alignment = endpoint_suite.get("demo_alignment") or {}
        if alignment.get("checks"):
            lines.extend(["", "### Demo Alignment", ""])
            for check in alignment["checks"]:
                status = "pass" if check.get("passed") else "fail"
                lines.append(
                    f"- `{status}` `{check.get('arm')}`: eval `{check.get('eval_model')}` "
                    f"vs endpoint `{check.get('served_model_id')}`"
                )
    lines.extend(["", "## Evidence-Backed Claims", ""])
    if not demo["claims"]:
        lines.append("- No cross-arm behavior claims were generated.")
    for claim in demo["claims"]:
        lines.append(f"- `{claim['id']}`: {claim['summary']}")
        for item in claim["evidence"]:
            links = [_md_link("evaluation_summary", item.get("evaluation_summary")), _md_link("suite_summary", item.get("suite_summary")), _md_link("trace", item.get("trace_path")), _md_link("scorecard", item.get("scorecard")), _md_link("run_digest", item.get("run_digest")), _md_link("report", item.get("report"))]
            lines.append(f"  - {item['arm']} / {item['scenario_id']}: " + ", ".join(link for link in links if link))
    lines.extend(["", "## Scenario Replay Index", "", "| Scenario | Arm | Passed | Score | Critical Failures | Trace | Scorecard | Run Digest | Report |", "| --- | --- | ---: | ---: | --- | --- | --- | --- | --- |"])
    for scenario in demo["scenarios"]:
        for arm_name, run in scenario["arms"].items():
            lines.append(f"| {scenario['scenario_id']} | {arm_name} | {run.get('passed')} | {run.get('score')} | {', '.join(run.get('critical_failures') or [])} | {_md_link('trace', run.get('trace_path'))} | {_md_link('scorecard', run.get('scorecard'))} | {_md_link('run_digest', run.get('run_digest'))} | {_md_link('report', run.get('report'))} |")
    return "\n".join(lines) + "\n"


def _load_arm(name: str, path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    data = _load_json(source)
    suite_path = _suite_summary_path(source, data)
    suite = _load_json(suite_path)
    eval_summary = data if data.get("schema_version") == "hfr.hermes_heldout_eval_summary.v1" else {}
    serving_profile = _serving_profile_ref(source, suite_path, eval_summary, suite)
    runs = [_run_record(suite_path, run) for run in suite.get("runs", [])]
    return {
        "name": name,
        "source": str(source),
        "suite_summary": str(suite_path),
        "model": eval_summary.get("model") or suite.get("metadata", {}).get("model"),
        "base_url": eval_summary.get("base_url") or suite.get("metadata", {}).get("base_url"),
        "serving_profile": serving_profile,
        "metrics": _metrics(eval_summary, suite),
        "scenario_ids": [str(run.get("scenario_id")) for run in runs],
        "runs": {str(run.get("scenario_id")): run for run in runs},
    }


def _suite_summary_path(source: Path, data: dict[str, Any]) -> Path:
    if data.get("schema_version") == "hfr.hermes_heldout_eval_summary.v1":
        suite = data.get("suite_summary")
        if not suite:
            raise SystemExit(f"Evaluation summary has no suite_summary: {source}")
        path = Path(str(suite))
        return (path if path.exists() else source.parent / path).resolve()
    return source.resolve()


def _run_record(summary_path: Path, run: dict[str, Any]) -> dict[str, Any]:
    record = dict(run)
    for key in ("trace_path", "scorecard", "run_digest", "report"):
        if record.get(key):
            record[key] = _resolve_artifact_ref(summary_path, str(record[key]))
    digest_path = Path(str(record.get("run_digest") or ""))
    if digest_path.exists():
        digest = _load_json(digest_path)
        record["digest_outcome"] = digest.get("outcome") or {}
        record["digest_summary"] = record["digest_outcome"].get("summary")
    else:
        record["digest_outcome"] = {}
        record["digest_summary"] = ""
    return record


def _scenario_rows(arms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scenario_ids: list[str] = []
    for arm in arms:
        for scenario_id in arm["scenario_ids"]:
            if scenario_id not in scenario_ids:
                scenario_ids.append(scenario_id)
    return [{"scenario_id": scenario_id, "arms": {arm["name"]: arm["runs"][scenario_id] for arm in arms if scenario_id in arm["runs"]}} for scenario_id in scenario_ids]


def _claims(candidate: dict[str, Any], references: list[dict[str, Any]], scenarios: list[dict[str, Any]], *, same_scenarios: bool) -> list[dict[str, Any]]:
    if not same_scenarios:
        return [{"id": "scenario_sets_differ", "summary": "Arms do not share identical scenario ids, so behavior claims are inspection-only.", "evidence": []}]
    claims = []
    for reference in references:
        candidate_rate = candidate["metrics"].get("pass_rate")
        reference_rate = reference["metrics"].get("pass_rate")
        if candidate_rate is not None and reference_rate is not None:
            relation = "beats" if float(candidate_rate) > float(reference_rate) else "does_not_beat"
            claims.append({"id": f"{candidate['name']}_{relation}_{reference['name']}_pass_rate", "summary": f"{candidate['name']} pass rate {candidate_rate} versus {reference['name']} {reference_rate}.", "evidence": [_arm_evidence(candidate), _arm_evidence(reference)]})
    for scenario in scenarios:
        candidate_run = scenario["arms"].get(candidate["name"])
        if not candidate_run:
            continue
        failed_refs = [ref for ref in references if scenario["arms"].get(ref["name"]) and not scenario["arms"][ref["name"]].get("passed")]
        if candidate_run.get("passed") and failed_refs:
            claims.append({"id": f"{candidate['name']}_repairs_{scenario['scenario_id']}", "summary": f"{candidate['name']} passed {scenario['scenario_id']} where at least one reference arm failed.", "evidence": [_run_evidence(candidate["name"], candidate_run)] + [_run_evidence(ref["name"], scenario["arms"][ref["name"]]) for ref in failed_refs]})
    return claims


def _comparisons(candidate: dict[str, Any], references: list[dict[str, Any]], scenarios: list[dict[str, Any]], *, same_scenarios: bool) -> list[dict[str, Any]]:
    rows = []
    for reference in references:
        rows.append(
            {
                "candidate_arm": candidate["name"],
                "reference_arm": reference["name"],
                "same_scenario_ids": same_scenarios,
                "metric_deltas": {
                    "pass_rate": _number_delta(candidate["metrics"].get("pass_rate"), reference["metrics"].get("pass_rate")),
                    "average_score": _number_delta(candidate["metrics"].get("average_score"), reference["metrics"].get("average_score")),
                    "passed": _number_delta(candidate["metrics"].get("passed"), reference["metrics"].get("passed")),
                    "failed": _number_delta(candidate["metrics"].get("failed"), reference["metrics"].get("failed")),
                    "critical_failure_total": _number_delta(candidate["metrics"].get("critical_failure_total"), reference["metrics"].get("critical_failure_total")),
                },
                "scenario_outcomes": [_comparison_outcome(candidate["name"], reference["name"], scenario) for scenario in scenarios],
            }
        )
    return rows


def _comparison_outcome(candidate_name: str, reference_name: str, scenario: dict[str, Any]) -> dict[str, Any]:
    candidate_run = scenario["arms"].get(candidate_name)
    reference_run = scenario["arms"].get(reference_name)
    if candidate_run is None or reference_run is None:
        outcome = "missing_candidate" if candidate_run is None else "missing_reference"
    elif candidate_run.get("passed") is True and reference_run.get("passed") is not True:
        outcome = "candidate_repaired"
    elif candidate_run.get("passed") is not True and reference_run.get("passed") is True:
        outcome = "candidate_regressed"
    elif candidate_run.get("passed") is True and reference_run.get("passed") is True:
        outcome = "both_passed"
    else:
        outcome = "both_failed"
    return {
        "scenario_id": scenario["scenario_id"],
        "candidate_passed": candidate_run.get("passed") if candidate_run else None,
        "reference_passed": reference_run.get("passed") if reference_run else None,
        "candidate_score": candidate_run.get("score") if candidate_run else None,
        "reference_score": reference_run.get("score") if reference_run else None,
        "score_delta": _number_delta(candidate_run.get("score") if candidate_run else None, reference_run.get("score") if reference_run else None),
        "outcome": outcome,
    }


def _number_delta(candidate_value: Any, reference_value: Any) -> float | int | None:
    if candidate_value is None or reference_value is None:
        return None
    try:
        delta = float(candidate_value) - float(reference_value)
    except (TypeError, ValueError):
        return None
    return int(delta) if delta.is_integer() else delta


def _metrics(eval_summary: dict[str, Any], suite: dict[str, Any]) -> dict[str, Any]:
    raw = suite.get("metrics") or {}
    critical_total = eval_summary.get("critical_failure_total")
    if critical_total is None:
        critical_total = sum(int(item.get("count") or 0) for item in raw.get("critical_failure_counts", []) if isinstance(item, dict))
    return {"total": eval_summary.get("total", suite.get("total")), "passed": eval_summary.get("passed", suite.get("passed")), "failed": eval_summary.get("failed", suite.get("failed")), "pass_rate": eval_summary.get("pass_rate", raw.get("pass_rate")), "average_score": eval_summary.get("average_score", raw.get("average_score")), "critical_failure_total": critical_total}


def _arm_evidence(arm: dict[str, Any]) -> dict[str, Any]:
    return {"arm": arm["name"], "scenario_id": "suite", "evaluation_summary": arm["source"], "suite_summary": arm["suite_summary"]}


def _run_evidence(arm: str, run: dict[str, Any]) -> dict[str, Any]:
    return {"arm": arm, "scenario_id": run.get("scenario_id"), "trace_path": run.get("trace_path"), "scorecard": run.get("scorecard"), "run_digest": run.get("run_digest"), "report": run.get("report")}


def _resolve_path(summary_path: Path, value: str) -> Path:
    if not value:
        return Path("")
    path = Path(value)
    return path if path.exists() or path.is_absolute() else summary_path.parent / path


def _serving_profile_ref(source: Path, suite_path: Path, eval_summary: dict[str, Any], suite: dict[str, Any]) -> str | None:
    if eval_summary.get("serving_profile"):
        return _resolve_artifact_ref(source, str(eval_summary["serving_profile"]))
    suite_profile = suite.get("metadata", {}).get("serving_profile")
    if suite_profile:
        return _resolve_artifact_ref(suite_path, str(suite_profile))
    return None


def _resolve_artifact_ref(anchor_file: Path, value: str) -> str:
    if "://" in value:
        return value
    path = Path(value)
    if not path.is_absolute():
        path = anchor_file.parent / path
    return str(path.resolve())


def _load_endpoint_suite(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    data = _load_json(source)
    arms = []
    for arm in data.get("arms") or []:
        identity = arm.get("model_identity") if isinstance(arm.get("model_identity"), dict) else {}
        capabilities = arm.get("capabilities") if isinstance(arm.get("capabilities"), dict) else {}
        lifecycle = arm.get("lifecycle") if isinstance(arm.get("lifecycle"), dict) else {}
        arms.append(
            {
                "arm": arm.get("arm"),
                "ready_for_eval": arm.get("ready_for_eval"),
                "failed_checks": arm.get("failed_checks") or [],
                "profile_path": arm.get("profile_path"),
                "served_model_id": identity.get("served_model_id"),
                "requested_model": identity.get("requested_model"),
                "tool_calls": capabilities.get("tool_calls"),
                "structured_outputs": capabilities.get("structured_outputs"),
                "lifecycle_path": lifecycle.get("path") if lifecycle.get("present") else "",
            }
        )
    return {
        "path": str(source),
        "schema_version": data.get("schema_version"),
        "passed": data.get("passed"),
        "failed_checks": data.get("failed_checks") or [],
        "requirements": data.get("requirements") or {},
        "arms": arms,
    }


def _endpoint_alignment(arms: list[dict[str, Any]], endpoint_suite: dict[str, Any]) -> dict[str, Any]:
    endpoint_by_arm = {str(arm.get("arm")): arm for arm in endpoint_suite.get("arms") or []}
    checks = []
    for arm in arms:
        endpoint = endpoint_by_arm.get(arm["name"])
        if not endpoint:
            checks.append(
                {
                    "arm": arm["name"],
                    "passed": False,
                    "reason": "missing_endpoint_arm",
                    "eval_model": arm.get("model"),
                    "served_model_id": "",
                }
            )
            continue
        model_matches = endpoint.get("served_model_id") == arm.get("model")
        checks.append(
            {
                "arm": arm["name"],
                "passed": bool(endpoint.get("ready_for_eval") and model_matches),
                "reason": "matched" if model_matches else "model_mismatch",
                "eval_model": arm.get("model"),
                "served_model_id": endpoint.get("served_model_id"),
            }
        )
    failed = [f"{check['arm']}:{check['reason']}" for check in checks if not check["passed"]]
    return {"passed": not failed, "failed_checks": failed, "checks": checks}


def _arm_specs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for name, path in (
        ("baseline", args.baseline),
        ("trace_only", args.trace_only),
        ("flightrecorder", args.flightrecorder),
    ):
        if path:
            parsed.append((name, path))
    for spec in args.arm:
        if "=" not in spec:
            raise SystemExit(f"--arm must use NAME=PATH: {spec}")
        name, value = spec.split("=", 1)
        parsed.append((name.strip(), Path(value)))
    return parsed


def _candidate_name(configured: str, arm_names: list[str]) -> str:
    if configured:
        if configured not in arm_names:
            raise SystemExit(f"Candidate arm {configured!r} is not in arms: {arm_names}")
        return configured
    if "flightrecorder" in arm_names:
        return "flightrecorder"
    return arm_names[-1]


def _md_link(label: str, path: str | None) -> str:
    return f"[{label}]({path})" if path else ""


def _relativize_demo_paths(demo: dict[str, Any], base_dir: Path) -> None:
    base = base_dir.expanduser().resolve()
    arm_keys = ("source", "serving_profile")
    evidence_keys = ("evaluation_summary", "suite_summary", "trace_path", "scorecard", "run_digest", "report")

    for arm in demo.get("arms", []):
        for key in arm_keys:
            if arm.get(key):
                arm[key] = _relative_path(str(arm[key]), base)
    for claim in demo.get("claims", []):
        for item in claim.get("evidence", []):
            for key in evidence_keys:
                if item.get(key):
                    item[key] = _relative_path(str(item[key]), base)
    for scenario in demo.get("scenarios", []):
        for run in scenario.get("arms", {}).values():
            for key in evidence_keys:
                if run.get(key):
                    run[key] = _relative_path(str(run[key]), base)
    endpoint_suite = demo.get("endpoint_suite")
    if isinstance(endpoint_suite, dict):
        if endpoint_suite.get("path"):
            endpoint_suite["path"] = _relative_path(str(endpoint_suite["path"]), base)
        for arm in endpoint_suite.get("arms") or []:
            for key in ("profile_path", "lifecycle_path"):
                if arm.get(key):
                    arm[key] = _relative_path(str(arm[key]), base)


def _relative_path(value: str, base_dir: Path) -> str:
    if "://" in value:
        return value
    path = Path(value)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return os.path.relpath(path, base_dir)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


if __name__ == "__main__":
    raise SystemExit(main())
