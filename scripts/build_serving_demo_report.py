#!/usr/bin/env python3
"""Build a replayable serving demo report from held-out eval artifacts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "hfr.serving_demo_run.v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Arm name and evaluation_summary.json or suite_summary.json path. Repeat for baseline, trace_only, candidate, etc.",
    )
    parser.add_argument("--baseline", type=Path, help="Shortcut for --arm baseline=PATH")
    parser.add_argument("--trace-only", type=Path, help="Shortcut for --arm trace_only=PATH")
    parser.add_argument("--flightrecorder", type=Path, help="Shortcut for --arm flightrecorder=PATH")
    parser.add_argument("--candidate-arm", default="", help="Arm to treat as the candidate; defaults to flightrecorder or the last arm")
    parser.add_argument("--endpoint-suite", type=Path, help="Optional serving_endpoint_suite.json to link endpoint readiness.")
    parser.add_argument("--out", type=Path, required=True, help="demo_run JSON output path")
    parser.add_argument("--report", type=Path, required=True, help="Markdown report output path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    arm_specs = _arm_specs(args)
    if len(arm_specs) < 2:
        raise SystemExit("At least two arms are required")
    arms = [_load_arm(name, path) for name, path in arm_specs]
    candidate_name = _candidate_name(args.candidate_arm, [arm["name"] for arm in arms])
    endpoint_suite = _load_endpoint_suite(args.endpoint_suite) if args.endpoint_suite else None
    demo = build_demo(arms, candidate_name=candidate_name, endpoint_suite=endpoint_suite)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.out, demo)
    args.report.write_text(render_report(demo), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "report": str(args.report), "claim_count": len(demo["claims"])}, indent=2))
    return 0


def build_demo(arms: list[dict[str, Any]], *, candidate_name: str, endpoint_suite: dict[str, Any] | None = None) -> dict[str, Any]:
    scenario_sets = {arm["name"]: arm["scenario_ids"] for arm in arms}
    same_scenarios = len({tuple(ids) for ids in scenario_sets.values()}) == 1
    scenarios = _scenario_rows(arms)
    candidate = next(arm for arm in arms if arm["name"] == candidate_name)
    reference_arms = [arm for arm in arms if arm["name"] != candidate_name]
    claims = _claims(candidate, reference_arms, scenarios, same_scenarios=same_scenarios)
    if endpoint_suite:
        endpoint_suite["demo_alignment"] = _endpoint_alignment(arms, endpoint_suite)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "candidate_arm": candidate_name,
        "same_scenario_ids": same_scenarios,
        "scenario_sets": scenario_sets,
        "arms": [
            {
                "name": arm["name"],
                "source": arm["source"],
                "model": arm["model"],
                "base_url": arm["base_url"],
                "serving_profile": arm.get("serving_profile"),
                "metrics": arm["metrics"],
            }
            for arm in arms
        ],
        "endpoint_suite": endpoint_suite,
        "claims": claims,
        "scenarios": scenarios,
    }


def render_report(demo: dict[str, Any]) -> str:
    lines = [
        "# Serving Demo Replay Report",
        "",
        f"- Candidate arm: `{demo['candidate_arm']}`",
        f"- Same scenario ids: {demo['same_scenario_ids']}",
        "",
        "## Arm Metrics",
        "",
        "| Arm | Model | Pass Rate | Average Score | Passed | Failed | Critical Failures | Serving Profile |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for arm in demo["arms"]:
        metrics = arm["metrics"]
        profile = arm.get("serving_profile") or {}
        profile_link = _md_link("profile", profile.get("path")) if profile.get("path") else ""
        lines.append(
            "| {name} | `{model}` | {pass_rate} | {score} | {passed} | {failed} | {critical} | {profile} |".format(
                name=arm["name"],
                model=arm.get("model") or "",
                pass_rate=metrics.get("pass_rate"),
                score=metrics.get("average_score"),
                passed=metrics.get("passed"),
                failed=metrics.get("failed"),
                critical=metrics.get("critical_failure_total"),
                profile=profile_link,
            )
        )
    endpoint_suite = demo.get("endpoint_suite")
    if endpoint_suite:
        lines.extend(
            [
                "",
                "## Serving Endpoint Readiness",
                "",
                f"- Suite passed: {endpoint_suite.get('passed')}",
                f"- Demo aligned: {endpoint_suite.get('demo_alignment', {}).get('passed')}",
                f"- Failed checks: {', '.join(endpoint_suite.get('failed_checks') or []) if endpoint_suite.get('failed_checks') else 'none'}",
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
                lines.append(f"- `{status}` `{check.get('arm')}`: eval `{check.get('eval_model')}` vs endpoint `{check.get('served_model_id')}`")
    lines.extend(["", "## Evidence-Backed Claims", ""])
    if not demo["claims"]:
        lines.append("- No cross-arm behavior claims were generated.")
    for claim in demo["claims"]:
        lines.append(f"- `{claim['id']}`: {claim['summary']}")
        for item in claim["evidence"]:
            links = [
                _md_link("evaluation_summary", item.get("evaluation_summary")),
                _md_link("suite_summary", item.get("suite_summary")),
                _md_link("trace", item.get("trace_path")),
                _md_link("scorecard", item.get("scorecard")),
                _md_link("run_digest", item.get("run_digest")),
                _md_link("report", item.get("report")),
            ]
            lines.append(f"  - {item['arm']} / {item['scenario_id']}: " + ", ".join(link for link in links if link))
    lines.extend(
        [
            "",
            "## Scenario Replay Index",
            "",
            "| Scenario | Arm | Passed | Score | Critical Failures | Trace | Scorecard | Run Digest | Report |",
            "| --- | --- | ---: | ---: | --- | --- | --- | --- | --- |",
        ]
    )
    for scenario in demo["scenarios"]:
        for arm_name, run in scenario["arms"].items():
            lines.append(
                "| {scenario} | {arm} | {passed} | {score} | {critical} | {trace} | {scorecard} | {digest} | {report} |".format(
                    scenario=scenario["scenario_id"],
                    arm=arm_name,
                    passed=run.get("passed"),
                    score=run.get("score"),
                    critical=", ".join(run.get("critical_failures") or []),
                    trace=_md_link("trace", run.get("trace_path")),
                    scorecard=_md_link("scorecard", run.get("scorecard")),
                    digest=_md_link("run_digest", run.get("run_digest")),
                    report=_md_link("report", run.get("report")),
                )
            )
    return "\n".join(lines) + "\n"


def _load_arm(name: str, path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    data = _load_json(source)
    suite_path = _suite_summary_path(source, data)
    suite = _load_json(suite_path)
    eval_summary = data if data.get("schema_version") == "hfr.hermes_heldout_eval_summary.v1" else {}
    runs = [_run_record(suite_path, run) for run in suite.get("runs", [])]
    scenario_ids = [str(run.get("scenario_id")) for run in runs]
    metrics = _metrics(eval_summary, suite)
    return {
        "name": name,
        "source": str(source),
        "suite_summary": str(suite_path),
        "model": eval_summary.get("model") or suite.get("metadata", {}).get("model"),
        "base_url": eval_summary.get("base_url") or suite.get("metadata", {}).get("base_url"),
        "serving_profile": eval_summary.get("serving_profile") or suite.get("metadata", {}).get("serving_profile"),
        "metrics": metrics,
        "scenario_ids": scenario_ids,
        "runs": {str(run.get("scenario_id")): run for run in runs},
    }


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
            checks.append({"arm": arm["name"], "passed": False, "reason": "missing_endpoint_arm", "eval_model": arm.get("model"), "served_model_id": ""})
            continue
        checks.append(
            {
                "arm": arm["name"],
                "passed": bool(endpoint.get("ready_for_eval") and endpoint.get("served_model_id") == arm.get("model")),
                "reason": "matched" if endpoint.get("served_model_id") == arm.get("model") else "model_mismatch",
                "eval_model": arm.get("model"),
                "served_model_id": endpoint.get("served_model_id"),
            }
        )
    failed = [f"{check['arm']}:{check['reason']}" for check in checks if not check["passed"]]
    return {"passed": not failed, "failed_checks": failed, "checks": checks}


def _suite_summary_path(source: Path, data: dict[str, Any]) -> Path:
    if data.get("schema_version") == "hfr.hermes_heldout_eval_summary.v1":
        suite = data.get("suite_summary")
        if not suite:
            raise SystemExit(f"Evaluation summary has no suite_summary: {source}")
        path = Path(str(suite))
        return path if path.exists() else source.parent / path
    return source


def _run_record(summary_path: Path, run: dict[str, Any]) -> dict[str, Any]:
    record = dict(run)
    digest_path = _resolve_path(summary_path, str(record.get("run_digest") or ""))
    if digest_path.exists():
        digest = _load_json(digest_path)
        record["digest_outcome"] = digest.get("outcome") or {}
        record["digest_summary"] = (digest.get("outcome") or {}).get("summary")
        record["trace_signal"] = digest.get("trace_signal") or {}
        record["training_signals"] = digest.get("training_signals") or {}
    else:
        record["digest_outcome"] = {}
        record["digest_summary"] = ""
    return record


def _resolve_path(summary_path: Path, value: str) -> Path:
    if not value:
        return Path("")
    path = Path(value)
    if path.exists() or path.is_absolute():
        return path
    candidate = summary_path.parent / path
    return candidate


def _metrics(eval_summary: dict[str, Any], suite: dict[str, Any]) -> dict[str, Any]:
    raw_metrics = suite.get("metrics") or {}
    critical_total = eval_summary.get("critical_failure_total")
    if critical_total is None:
        critical_total = sum(int(item.get("count") or 0) for item in raw_metrics.get("critical_failure_counts", []) if isinstance(item, dict))
    return {
        "total": eval_summary.get("total", suite.get("total")),
        "passed": eval_summary.get("passed", suite.get("passed")),
        "failed": eval_summary.get("failed", suite.get("failed")),
        "pass_rate": eval_summary.get("pass_rate", raw_metrics.get("pass_rate")),
        "average_score": eval_summary.get("average_score", raw_metrics.get("average_score")),
        "critical_failure_total": critical_total,
    }


def _scenario_rows(arms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scenario_ids: list[str] = []
    for arm in arms:
        for scenario_id in arm["scenario_ids"]:
            if scenario_id not in scenario_ids:
                scenario_ids.append(scenario_id)
    rows = []
    for scenario_id in scenario_ids:
        row = {"scenario_id": scenario_id, "arms": {}}
        for arm in arms:
            if scenario_id in arm["runs"]:
                row["arms"][arm["name"]] = arm["runs"][scenario_id]
        rows.append(row)
    return rows


def _claims(
    candidate: dict[str, Any],
    references: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
    *,
    same_scenarios: bool,
) -> list[dict[str, Any]]:
    claims = []
    if not same_scenarios:
        claims.append(
            {
                "id": "scenario_sets_differ",
                "summary": "Arms do not share identical scenario ids, so cross-arm behavior claims are inspection-only.",
                "evidence": [],
            }
        )
        return claims
    for reference in references:
        candidate_rate = candidate["metrics"].get("pass_rate")
        reference_rate = reference["metrics"].get("pass_rate")
        if _number(candidate_rate) is not None and _number(reference_rate) is not None:
            relation = "beats" if float(candidate_rate) > float(reference_rate) else "does_not_beat"
            claims.append(
                {
                    "id": f"{candidate['name']}_{relation}_{reference['name']}_pass_rate",
                    "summary": f"{candidate['name']} pass rate {candidate_rate} versus {reference['name']} {reference_rate}.",
                    "evidence": [_arm_evidence(candidate), _arm_evidence(reference)],
                }
            )
    for scenario in scenarios:
        candidate_run = scenario["arms"].get(candidate["name"])
        if not candidate_run:
            continue
        failed_references = [
            reference
            for reference in references
            if scenario["arms"].get(reference["name"]) and not scenario["arms"][reference["name"]].get("passed")
        ]
        if candidate_run.get("passed") and failed_references:
            claims.append(
                {
                    "id": f"{candidate['name']}_repairs_{scenario['scenario_id']}",
                    "summary": f"{candidate['name']} passed {scenario['scenario_id']} where at least one reference arm failed.",
                    "evidence": [_run_evidence(candidate["name"], candidate_run)]
                    + [_run_evidence(reference["name"], scenario["arms"][reference["name"]]) for reference in failed_references],
                }
            )
        if not candidate_run.get("passed"):
            claims.append(
                {
                    "id": f"{candidate['name']}_remaining_gap_{scenario['scenario_id']}",
                    "summary": f"{candidate['name']} still failed {scenario['scenario_id']}.",
                    "evidence": [_run_evidence(candidate["name"], candidate_run)],
                }
            )
    return claims


def _arm_evidence(arm: dict[str, Any]) -> dict[str, Any]:
    return {
        "arm": arm["name"],
        "scenario_id": "suite",
        "suite_summary": arm["suite_summary"],
        "evaluation_summary": arm["source"],
    }


def _run_evidence(arm_name: str, run: dict[str, Any]) -> dict[str, Any]:
    return {
        "arm": arm_name,
        "scenario_id": run.get("scenario_id"),
        "trace_path": run.get("trace_path"),
        "scorecard": run.get("scorecard"),
        "run_digest": run.get("run_digest"),
        "report": run.get("report"),
        "lineage": run.get("lineage"),
    }


def _arm_specs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    specs: list[tuple[str, Path]] = []
    for name, path in (("baseline", args.baseline), ("trace_only", args.trace_only), ("flightrecorder", args.flightrecorder)):
        if path:
            specs.append((name, path))
    for raw in args.arm:
        if "=" not in raw:
            raise SystemExit(f"--arm must use NAME=PATH format: {raw}")
        name, value = raw.split("=", 1)
        specs.append((name.strip(), Path(value)))
    return specs


def _candidate_name(configured: str, arm_names: list[str]) -> str:
    if configured:
        if configured not in arm_names:
            raise SystemExit(f"Candidate arm {configured!r} is not in arms: {arm_names}")
        return configured
    if "flightrecorder" in arm_names:
        return "flightrecorder"
    return arm_names[-1]


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _md_link(label: str, value: Any) -> str:
    if not value:
        return ""
    text = str(value)
    return f"[{label}]({text})"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


if __name__ == "__main__":
    raise SystemExit(main())
