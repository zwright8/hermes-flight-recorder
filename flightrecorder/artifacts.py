"""Portable scorecard artifacts for CI and regression comparison."""

from __future__ import annotations

import html
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

COMPARE_SCHEMA_VERSION = "hfr.compare.v1"
SUITE_COMPARE_SCHEMA_VERSION = "hfr.suite_compare.v1"
SUITE_TREND_SCHEMA_VERSION = "hfr.suite_trend.v1"
CONTRACT_SCOPES = {"scenario", "scenario-and-trace"}


class ArtifactError(ValueError):
    """Raised when portable artifact generation cannot continue."""


def write_junit(scorecard: dict[str, Any], out_path: str | Path) -> None:
    """Write a minimal JUnit XML report for CI systems."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rules = scorecard.get("rules", [])
    failures = [rule for rule in rules if not rule.get("passed")]
    suite = ET.Element(
        "testsuite",
        {
            "name": f"flightrecorder.{scorecard.get('scenario_id', 'scenario')}",
            "tests": str(len(rules) or 1),
            "failures": str(len(failures)),
            "errors": "0",
            "skipped": "0",
            "time": "0",
        },
    )
    if not rules:
        ET.SubElement(suite, "testcase", {"classname": "flightrecorder", "name": "scorecard"})
    for rule in rules:
        case = ET.SubElement(
            suite,
            "testcase",
            {
                "classname": str(scorecard.get("scenario_id", "scenario")),
                "name": str(rule.get("name") or rule.get("id") or "rule"),
            },
        )
        if not rule.get("passed"):
            failure = ET.SubElement(case, "failure", {"message": str(rule.get("name") or "rule failed")})
            failure.text = "\n".join(str(item) for item in rule.get("evidence", []))
    tree = ET.ElementTree(suite)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def write_markdown_summary(scorecard: dict[str, Any], out_path: str | Path) -> None:
    """Write a concise Markdown scorecard summary."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    status = "PASS" if scorecard.get("passed") else "FAIL"
    lines = [
        "# Flight Recorder Scorecard",
        "",
        f"- Scenario: `{_md(scorecard.get('scenario_id', 'unknown'))}`",
        f"- Status: **{status}**",
        f"- Score: **{scorecard.get('score')}** / threshold `{scorecard.get('pass_threshold')}`",
        f"- Summary: {_md(scorecard.get('summary', ''))}",
        "",
        "| Rule | Status | Critical | Evidence |",
        "| --- | --- | --- | --- |",
    ]
    for rule in scorecard.get("rules", []):
        evidence = "<br>".join(_md(item) for item in rule.get("evidence", []))
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(rule.get("name") or rule.get("id") or "rule"),
                    "PASS" if rule.get("passed") else "FAIL",
                    "yes" if rule.get("critical") else "no",
                    evidence,
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_scorecards(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    baseline_label: str,
    candidate_label: str,
) -> dict[str, Any]:
    """Compare two scorecards and report regressions/fixes."""
    baseline_rules = {rule.get("id"): rule for rule in baseline.get("rules", [])}
    candidate_rules = {rule.get("id"): rule for rule in candidate.get("rules", [])}
    rule_ids = sorted({str(rule_id) for rule_id in baseline_rules | candidate_rules if rule_id})
    rule_changes: list[dict[str, Any]] = []
    regressions: list[str] = []
    fixes: list[str] = []
    for rule_id in rule_ids:
        before = baseline_rules.get(rule_id, {})
        after = candidate_rules.get(rule_id, {})
        before_passed = before.get("passed")
        after_passed = after.get("passed")
        status = "unchanged"
        if before_passed is True and after_passed is False:
            status = "regressed"
            regressions.append(rule_id)
        elif before_passed is False and after_passed is True:
            status = "fixed"
            fixes.append(rule_id)
        rule_changes.append(
            {
                "id": rule_id,
                "name": after.get("name") or before.get("name") or rule_id,
                "baseline_passed": before_passed,
                "candidate_passed": after_passed,
                "status": status,
            }
        )

    baseline_score = int(baseline.get("score", 0))
    candidate_score = int(candidate.get("score", 0))
    score_delta = candidate_score - baseline_score
    passed_regressed = bool(baseline.get("passed")) and not bool(candidate.get("passed"))
    new_critical_failures = sorted(
        set(str(item) for item in candidate.get("critical_failures", []))
        - set(str(item) for item in baseline.get("critical_failures", []))
    )
    regressed = score_delta < 0 or passed_regressed or bool(regressions) or bool(new_critical_failures)
    return {
        "schema_version": COMPARE_SCHEMA_VERSION,
        "baseline": _score_summary(baseline, baseline_label),
        "candidate": _score_summary(candidate, candidate_label),
        "score_delta": score_delta,
        "passed_regressed": passed_regressed,
        "new_critical_failures": new_critical_failures,
        "rule_changes": rule_changes,
        "regressions": regressions,
        "fixes": fixes,
        "regressed": regressed,
        "summary": _compare_summary(regressed, score_delta, regressions, fixes),
    }


def compare_suites(
    baseline_dir: str | Path,
    candidate_dir: str | Path,
    *,
    baseline_label: str | None = None,
    candidate_label: str | None = None,
    contract_scope: str = "scenario",
) -> dict[str, Any]:
    """Compare two directories of Flight Recorder run artifacts."""
    if contract_scope not in CONTRACT_SCOPES:
        raise ArtifactError(f"contract_scope must be one of {sorted(CONTRACT_SCOPES)!r}; got {contract_scope!r}")
    baseline_root = Path(baseline_dir)
    candidate_root = Path(candidate_dir)
    baseline = _suite_scorecards(baseline_root)
    candidate = _suite_scorecards(candidate_root, allow_empty=True)
    baseline_metadata = _suite_metadata(baseline_root)
    candidate_metadata = _suite_metadata(candidate_root)

    scenario_ids = sorted(set(baseline) | set(candidate))
    scenario_changes: list[dict[str, Any]] = []
    regressions: list[str] = []
    improvements: list[str] = []
    missing_in_candidate: list[str] = []
    new_in_candidate: list[str] = []
    paired_deltas: list[int] = []
    paired_baseline_scores: list[int] = []
    paired_candidate_scores: list[int] = []
    paired_baseline_passed = 0
    paired_candidate_passed = 0
    paired_baseline_scorecards: dict[str, dict[str, Any]] = {}
    paired_candidate_scorecards: dict[str, dict[str, Any]] = {}
    contract_drifts: list[dict[str, Any]] = []
    unverified_contracts: list[dict[str, Any]] = []

    for scenario_id in scenario_ids:
        before = baseline.get(scenario_id)
        after = candidate.get(scenario_id)
        if before is None:
            new_in_candidate.append(scenario_id)
            scenario_changes.append(
                {
                    "scenario_id": scenario_id,
                    "status": "new",
                    "baseline_run": None,
                    "candidate_run": after["run"],
                    "baseline_score": None,
                    "candidate_score": _score(after["scorecard"]),
                    "score_delta": None,
                    "regressed": False,
                    "summary": "Scenario exists only in candidate suite.",
                }
            )
            continue
        if after is None:
            missing_in_candidate.append(scenario_id)
            scenario_changes.append(
                {
                    "scenario_id": scenario_id,
                    "status": "missing",
                    "baseline_run": before["run"],
                    "candidate_run": None,
                    "baseline_score": _score(before["scorecard"]),
                    "candidate_score": None,
                    "score_delta": None,
                    "regressed": True,
                    "summary": "Scenario is missing from candidate suite.",
                }
            )
            continue

        comparison = compare_scorecards(
            before["scorecard"],
            after["scorecard"],
            baseline_label=before["label"],
            candidate_label=after["label"],
        )
        contract = _contract_comparison(before, after, contract_scope)
        if contract["status"] == "drifted":
            contract_drifts.append({"scenario_id": scenario_id, **contract})
        elif contract["status"] == "unverified":
            unverified_contracts.append({"scenario_id": scenario_id, **contract})
        delta = int(comparison["score_delta"])
        paired_deltas.append(delta)
        paired_baseline_scores.append(_score(before["scorecard"]))
        paired_candidate_scores.append(_score(after["scorecard"]))
        paired_baseline_passed += 1 if before["scorecard"].get("passed") else 0
        paired_candidate_passed += 1 if after["scorecard"].get("passed") else 0
        paired_baseline_scorecards[scenario_id] = before["scorecard"]
        paired_candidate_scorecards[scenario_id] = after["scorecard"]

        status = "unchanged"
        if comparison["regressed"]:
            status = "regressed"
            regressions.append(scenario_id)
        elif delta > 0 or comparison.get("fixes"):
            status = "improved"
            improvements.append(scenario_id)
        scenario_changes.append(
            {
                "scenario_id": scenario_id,
                "status": status,
                "baseline_run": before["run"],
                "candidate_run": after["run"],
                "baseline_score": comparison["baseline"]["score"],
                "candidate_score": comparison["candidate"]["score"],
                "score_delta": delta,
                "baseline_passed": comparison["baseline"]["passed"],
                "candidate_passed": comparison["candidate"]["passed"],
                "rule_regressions": comparison["regressions"],
                "rule_fixes": comparison["fixes"],
                "new_critical_failures": comparison["new_critical_failures"],
                "regressed": comparison["regressed"],
                "contract_fingerprint_status": contract["status"],
                "contract_fingerprint_scope": contract["scope"],
                "contract_fingerprint_reasons": contract["reasons"],
                "contract_fingerprints": contract["fingerprints"],
                "summary": comparison["summary"],
            }
        )

    paired_count = len(paired_deltas)
    aggregate = {
        "baseline_count": len(baseline),
        "candidate_count": len(candidate),
        "paired_count": paired_count,
        "baseline_avg_score": _mean(paired_baseline_scores),
        "candidate_avg_score": _mean(paired_candidate_scores),
        "avg_score_delta": _mean(paired_deltas),
        "total_score_delta": sum(paired_deltas),
        "baseline_pass_rate": _ratio(paired_baseline_passed, paired_count),
        "candidate_pass_rate": _ratio(paired_candidate_passed, paired_count),
        "failed_rule_deltas": _failure_deltas(paired_baseline_scorecards, paired_candidate_scorecards, "failed_rules"),
        "critical_failure_deltas": _failure_deltas(
            paired_baseline_scorecards,
            paired_candidate_scorecards,
            "critical_failures",
        ),
        "contract_drift_count": len(contract_drifts),
        "unverified_contract_count": len(unverified_contracts),
    }
    regressed = bool(regressions or missing_in_candidate)
    return {
        "schema_version": SUITE_COMPARE_SCHEMA_VERSION,
        "baseline": {
            "label": baseline_label or _display_path(baseline_root),
            "path": _display_path(baseline_root),
            "scenario_count": len(baseline),
            "metadata": baseline_metadata,
        },
        "candidate": {
            "label": candidate_label or _display_path(candidate_root),
            "path": _display_path(candidate_root),
            "scenario_count": len(candidate),
            "metadata": candidate_metadata,
        },
        "aggregate": aggregate,
        "contract_scope": contract_scope,
        "scenario_changes": scenario_changes,
        "regressions": regressions,
        "improvements": improvements,
        "missing_in_candidate": missing_in_candidate,
        "new_in_candidate": new_in_candidate,
        "contract_drifts": contract_drifts,
        "unverified_contracts": unverified_contracts,
        "regressed": regressed,
        "summary": _suite_compare_summary(regressed, regressions, missing_in_candidate, improvements, aggregate),
    }


def build_suite_trend(suite_summary_paths: list[str | Path]) -> dict[str, Any]:
    """Build a longitudinal trend over ordered run-suite summaries."""
    if not suite_summary_paths:
        raise ArtifactError("At least one --suite-summary path is required")
    points: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for index, raw_path in enumerate(suite_summary_paths):
        path = Path(raw_path)
        summary = _read_json(path)
        if not isinstance(summary, dict):
            raise ArtifactError(f"Suite summary must be a JSON object: {path}")
        point = _trend_point(summary, path, index)
        point["delta_from_previous"] = _trend_delta(previous, point) if previous is not None else None
        points.append(point)
        previous = point

    return {
        "schema_version": SUITE_TREND_SCHEMA_VERSION,
        "point_count": len(points),
        "points": points,
        "failed_rule_trends": _count_trends(points, "failed_rule_counts"),
        "critical_failure_trends": _count_trends(points, "critical_failure_counts"),
        "summary": _suite_trend_summary(points),
    }


def write_compare_report(comparison: dict[str, Any], out_path: str | Path) -> None:
    """Write a static HTML compare report."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    status = "REGRESSION" if comparison.get("regressed") else "NO REGRESSION"
    cls = "fail" if comparison.get("regressed") else "pass"
    rows = []
    for change in comparison.get("rule_changes", []):
        rows.append(
            "<tr>"
            f"<td>{_esc(change.get('name'))}</td>"
            f"<td>{_bool_label(change.get('baseline_passed'))}</td>"
            f"<td>{_bool_label(change.get('candidate_passed'))}</td>"
            f"<td class=\"{_esc(change.get('status'))}\">{_esc(change.get('status'))}</td>"
            "</tr>"
        )
    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Flight Recorder Compare</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin:32px; color:#17202a; }}
    .badge {{ display:inline-block; border-radius:6px; padding:8px 12px; font-weight:800; }}
    .pass {{ color:#147a3d; background:#d1fadf; }}
    .fail, .regressed {{ color:#b42318; background:#fee4e2; }}
    .fixed {{ color:#147a3d; }}
    .unchanged {{ color:#566573; }}
    table {{ border-collapse:collapse; width:100%; max-width:980px; margin-top:18px; }}
    th, td {{ border-bottom:1px solid #d6dbdf; text-align:left; padding:10px; }}
    code {{ background:#eef2f6; border-radius:6px; padding:2px 5px; }}
  </style>
</head>
<body>
  <h1>Flight Recorder Compare</h1>
  <p><span class="badge {cls}">{status}</span></p>
  <p>{_esc(comparison.get('summary'))}</p>
  <p>Baseline: <code>{_esc(comparison.get('baseline', {}).get('label'))}</code>
     score <strong>{comparison.get('baseline', {}).get('score')}</strong></p>
  <p>Candidate: <code>{_esc(comparison.get('candidate', {}).get('label'))}</code>
     score <strong>{comparison.get('candidate', {}).get('score')}</strong></p>
  <p>Score delta: <strong>{comparison.get('score_delta')}</strong></p>
  <table>
    <thead><tr><th>Rule</th><th>Baseline</th><th>Candidate</th><th>Status</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def write_suite_compare_report(comparison: dict[str, Any], out_path: str | Path) -> None:
    """Write a static HTML suite comparison report."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    status = "REGRESSION" if comparison.get("regressed") else "NO REGRESSION"
    cls = "fail" if comparison.get("regressed") else "pass"
    aggregate = comparison.get("aggregate", {})
    metadata = _suite_metadata_table(comparison.get("baseline", {}), comparison.get("candidate", {}))
    failure_deltas = _failure_delta_table(aggregate.get("failed_rule_deltas"), "Failed Rule Deltas")
    critical_deltas = _failure_delta_table(aggregate.get("critical_failure_deltas"), "Critical Failure Deltas")
    contract_drift_table = _contract_drift_table(comparison.get("contract_drifts"), "Contract Fingerprint Drift")
    unverified_contract_table = _contract_drift_table(comparison.get("unverified_contracts"), "Unverified Contract Fingerprints")
    rows = []
    for change in comparison.get("scenario_changes", []):
        rows.append(
            "<tr>"
            f"<td>{_esc(change.get('scenario_id'))}</td>"
            f"<td class=\"{_esc(change.get('status'))}\">{_esc(change.get('status'))}</td>"
            f"<td class=\"{_esc(change.get('contract_fingerprint_status'))}\">{_esc(change.get('contract_fingerprint_status'))}</td>"
            f"<td>{_esc(change.get('baseline_score'))}</td>"
            f"<td>{_esc(change.get('candidate_score'))}</td>"
            f"<td>{_esc(change.get('score_delta'))}</td>"
            f"<td>{_esc(change.get('summary'))}</td>"
            "</tr>"
        )
    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Flight Recorder Suite Compare</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin:32px; color:#17202a; }}
    .badge {{ display:inline-block; border-radius:6px; padding:8px 12px; font-weight:800; }}
    .pass {{ color:#147a3d; background:#d1fadf; }}
    .fail, .regressed, .missing {{ color:#b42318; background:#fee4e2; }}
    .improved, .new {{ color:#147a3d; }}
    .unchanged, .matched {{ color:#566573; }}
    .drifted, .unverified {{ color:#b42318; background:#fff1f0; }}
    .metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; max-width:980px; margin:18px 0; }}
    .metric {{ border:1px solid #d6dbdf; border-radius:8px; padding:12px; }}
    .metric strong {{ display:block; font-size:1.4rem; }}
    table {{ border-collapse:collapse; width:100%; max-width:1100px; margin-top:18px; }}
    th, td {{ border-bottom:1px solid #d6dbdf; text-align:left; padding:10px; vertical-align:top; }}
    code {{ background:#eef2f6; border-radius:6px; padding:2px 5px; }}
  </style>
</head>
<body>
  <h1>Flight Recorder Suite Compare</h1>
  <p><span class="badge {cls}">{status}</span></p>
  <p>{_esc(comparison.get('summary'))}</p>
  <p>Baseline: <code>{_esc(comparison.get('baseline', {}).get('label'))}</code></p>
  <p>Candidate: <code>{_esc(comparison.get('candidate', {}).get('label'))}</code></p>
  {metadata}
  <section class="metrics">
    <div class="metric"><span>Paired Scenarios</span><strong>{aggregate.get('paired_count')}</strong></div>
    <div class="metric"><span>Avg Score Delta</span><strong>{aggregate.get('avg_score_delta')}</strong></div>
    <div class="metric"><span>Regressions</span><strong>{len(comparison.get('regressions', []))}</strong></div>
    <div class="metric"><span>Missing Candidate</span><strong>{len(comparison.get('missing_in_candidate', []))}</strong></div>
    <div class="metric"><span>Contract Drift</span><strong>{aggregate.get('contract_drift_count')}</strong></div>
  </section>
  {failure_deltas}
  {critical_deltas}
  {contract_drift_table}
  {unverified_contract_table}
  <table>
    <thead><tr><th>Scenario</th><th>Status</th><th>Contract</th><th>Baseline</th><th>Candidate</th><th>Delta</th><th>Summary</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def write_suite_trend_report(trend: dict[str, Any], out_path: str | Path) -> None:
    """Write a static HTML suite trend report."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    point_rows = []
    for point in trend.get("points", []):
        point_rows.append(
            "<tr>"
            f"<td>{_esc(point.get('index'))}</td>"
            f"<td><code>{_esc(point.get('label'))}</code></td>"
            f"<td>{_esc(point.get('pass_rate'))}</td>"
            f"<td>{_esc(point.get('average_score'))}</td>"
            f"<td>{_esc(point.get('failed_rule_count'))}</td>"
            f"<td>{_esc(point.get('critical_failure_count'))}</td>"
            "</tr>"
        )
    failed_rows = _trend_count_table(trend.get("failed_rule_trends"), "Failed Rule Trends")
    critical_rows = _trend_count_table(trend.get("critical_failure_trends"), "Critical Failure Trends")
    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Flight Recorder Suite Trend</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin:32px; color:#17202a; }}
    table {{ border-collapse:collapse; width:100%; max-width:1100px; margin-top:18px; }}
    th, td {{ border-bottom:1px solid #d6dbdf; text-align:left; padding:10px; vertical-align:top; }}
    code {{ background:#eef2f6; border-radius:6px; padding:2px 5px; }}
    .improved {{ color:#147a3d; }}
    .regressed {{ color:#b42318; }}
    .unchanged {{ color:#566573; }}
  </style>
</head>
<body>
  <h1>Flight Recorder Suite Trend</h1>
  <p>{_esc(trend.get('summary'))}</p>
  <table>
    <thead><tr><th>#</th><th>Label</th><th>Pass Rate</th><th>Avg Score</th><th>Failed Rules</th><th>Critical Failures</th></tr></thead>
    <tbody>{''.join(point_rows)}</tbody>
  </table>
  {failed_rows}
  {critical_rows}
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def _score_summary(scorecard: dict[str, Any], label: str) -> dict[str, Any]:
    return {
        "label": label,
        "scenario_id": scorecard.get("scenario_id"),
        "passed": bool(scorecard.get("passed")),
        "score": scorecard.get("score"),
        "critical_failures": scorecard.get("critical_failures", []),
    }


def _suite_scorecards(root: Path, *, allow_empty: bool = False) -> dict[str, dict[str, Any]]:
    if not root.exists():
        raise ArtifactError(f"Suite directory not found: {root}")
    if not root.is_dir():
        raise ArtifactError(f"Suite path is not a directory: {root}")
    scorecards: dict[str, dict[str, Any]] = {}
    for score_path in sorted(root.glob("*/scorecard.json")):
        scorecard = _read_json(score_path)
        scenario_id = str(scorecard.get("scenario_id") or score_path.parent.name)
        if scenario_id in scorecards:
            raise ArtifactError(f"Duplicate scenario_id {scenario_id!r} in {root}")
        scorecards[scenario_id] = {
            "scorecard": scorecard,
            "run": score_path.parent.name,
            "label": _display_path(score_path),
            "fingerprints": _run_fingerprints(score_path.parent),
        }
    if not scorecards and not allow_empty:
        raise ArtifactError(f"No scorecard.json files found in suite directory: {root}")
    return scorecards


def _run_fingerprints(run_dir: Path) -> dict[str, Any]:
    lineage_path = run_dir / "artifact_lineage.json"
    fingerprints: dict[str, Any] = {
        "lineage_path": _display_path(lineage_path),
        "lineage_present": lineage_path.exists(),
        "inputs": {},
    }
    if not lineage_path.exists():
        return fingerprints
    try:
        lineage = _read_json(lineage_path)
    except (OSError, json.JSONDecodeError):
        fingerprints["lineage_readable"] = False
        return fingerprints
    fingerprints["lineage_readable"] = True
    inputs = lineage.get("inputs") if isinstance(lineage.get("inputs"), list) else []
    for record in inputs:
        if not isinstance(record, dict):
            continue
        name = record.get("name")
        if name not in {"scenario", "source_trace"}:
            continue
        fingerprints["inputs"][name] = {
            "path": record.get("path"),
            "sha256": record.get("sha256"),
            "exists": record.get("exists"),
        }
    return fingerprints


def _contract_comparison(before: dict[str, Any], after: dict[str, Any], contract_scope: str) -> dict[str, Any]:
    before_inputs = before.get("fingerprints", {}).get("inputs") if isinstance(before.get("fingerprints"), dict) else {}
    after_inputs = after.get("fingerprints", {}).get("inputs") if isinstance(after.get("fingerprints"), dict) else {}
    before_inputs = before_inputs if isinstance(before_inputs, dict) else {}
    after_inputs = after_inputs if isinstance(after_inputs, dict) else {}
    reasons: list[str] = []
    unknowns: list[str] = []
    for name in _contract_input_names(contract_scope):
        before_hash = _input_sha(before_inputs.get(name))
        after_hash = _input_sha(after_inputs.get(name))
        if before_hash and after_hash:
            if before_hash != after_hash:
                reasons.append(f"{name}_sha256_changed")
        else:
            unknowns.append(f"{name}_sha256_unverified")
    status = "drifted" if reasons else "unverified" if unknowns else "matched"
    return {
        "status": status,
        "scope": contract_scope,
        "reasons": reasons or unknowns,
        "fingerprints": {
            "baseline": _contract_inputs(before_inputs),
            "candidate": _contract_inputs(after_inputs),
        },
    }


def _contract_input_names(contract_scope: str) -> tuple[str, ...]:
    return ("scenario", "source_trace") if contract_scope == "scenario-and-trace" else ("scenario",)


def _input_sha(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    sha = value.get("sha256")
    return sha if isinstance(sha, str) and sha else None


def _contract_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "path": inputs.get(name, {}).get("path") if isinstance(inputs.get(name), dict) else None,
            "sha256": _input_sha(inputs.get(name)),
        }
        for name in ("scenario", "source_trace")
    }


def _suite_metadata(root: Path) -> dict[str, str]:
    summary_path = root / "suite_summary.json"
    if not summary_path.exists():
        return {}
    try:
        summary = _read_json(summary_path)
    except (OSError, json.JSONDecodeError):
        return {}
    metadata = summary.get("metadata") if isinstance(summary.get("metadata"), dict) else {}
    return {
        str(key): str(value)
        for key, value in sorted(metadata.items())
        if isinstance(key, str) and key and isinstance(value, str)
    }


def _suite_metadata_table(baseline: dict[str, Any], candidate: dict[str, Any]) -> str:
    baseline_metadata = baseline.get("metadata") if isinstance(baseline.get("metadata"), dict) else {}
    candidate_metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
    keys = sorted(set(baseline_metadata) | set(candidate_metadata))
    if not keys:
        return ""
    rows = []
    for key in keys:
        rows.append(
            "<tr>"
            f"<td><code>{_esc(key)}</code></td>"
            f"<td>{_esc(baseline_metadata.get(key, ''))}</td>"
            f"<td>{_esc(candidate_metadata.get(key, ''))}</td>"
            "</tr>"
        )
    return (
        '<section class="metadata">'
        "<h2>Experiment Metadata</h2>"
        "<table>"
        "<thead><tr><th>Key</th><th>Baseline</th><th>Candidate</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</section>"
    )


def _failure_deltas(
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    field_name: str,
) -> list[dict[str, Any]]:
    baseline_counts = _failure_scenarios(baseline, field_name)
    candidate_counts = _failure_scenarios(candidate, field_name)
    rows: list[dict[str, Any]] = []
    for rule_id in sorted(set(baseline_counts) | set(candidate_counts)):
        before = baseline_counts.get(rule_id, [])
        after = candidate_counts.get(rule_id, [])
        rows.append(
            {
                "id": rule_id,
                "baseline_count": len(before),
                "candidate_count": len(after),
                "delta": len(after) - len(before),
                "baseline_scenarios": before,
                "candidate_scenarios": after,
            }
        )
    return sorted(rows, key=lambda item: (-abs(int(item["delta"])), str(item["id"])))


def _failure_scenarios(scorecards: dict[str, dict[str, Any]], field_name: str) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {}
    for scenario_id, scorecard in scorecards.items():
        if field_name == "failed_rules":
            rule_ids = [
                str(rule.get("id"))
                for rule in scorecard.get("rules", [])
                if isinstance(rule, dict) and rule.get("id") and not rule.get("passed")
            ]
        else:
            rule_ids = scorecard.get(field_name, [])
        for rule_id in rule_ids:
            if isinstance(rule_id, str) and rule_id:
                buckets.setdefault(rule_id, []).append(scenario_id)
    return {rule_id: sorted(scenarios) for rule_id, scenarios in buckets.items()}


def _failure_delta_table(value: Any, title: str) -> str:
    rows_data = value if isinstance(value, list) else []
    if not rows_data:
        return ""
    rows = []
    for item in rows_data:
        if not isinstance(item, dict):
            continue
        delta = int(item.get("delta", 0) or 0)
        cls = "regressed" if delta > 0 else "improved" if delta < 0 else "unchanged"
        rows.append(
            "<tr>"
            f"<td><code>{_esc(item.get('id'))}</code></td>"
            f"<td>{_esc(item.get('baseline_count'))}</td>"
            f"<td>{_esc(item.get('candidate_count'))}</td>"
            f"<td class=\"{cls}\">{_esc(delta)}</td>"
            "</tr>"
        )
    if not rows:
        return ""
    return (
        '<section class="failure-deltas">'
        f"<h2>{_esc(title)}</h2>"
        "<table>"
        "<thead><tr><th>Rule</th><th>Baseline</th><th>Candidate</th><th>Delta</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</section>"
    )


def _contract_drift_table(value: Any, title: str) -> str:
    rows_data = value if isinstance(value, list) else []
    if not rows_data:
        return ""
    rows = []
    for item in rows_data:
        if not isinstance(item, dict):
            continue
        reasons = ", ".join(str(reason) for reason in item.get("reasons", []) if reason)
        fingerprints = item.get("fingerprints") if isinstance(item.get("fingerprints"), dict) else {}
        baseline = fingerprints.get("baseline") if isinstance(fingerprints.get("baseline"), dict) else {}
        candidate = fingerprints.get("candidate") if isinstance(fingerprints.get("candidate"), dict) else {}
        rows.append(
            "<tr>"
            f"<td><code>{_esc(item.get('scenario_id'))}</code></td>"
            f"<td>{_esc(item.get('status'))}</td>"
            f"<td>{_esc(reasons)}</td>"
            f"<td><code>{_esc(_short_fingerprint(baseline))}</code></td>"
            f"<td><code>{_esc(_short_fingerprint(candidate))}</code></td>"
            "</tr>"
        )
    if not rows:
        return ""
    return (
        '<section class="contract-fingerprints">'
        f"<h2>{_esc(title)}</h2>"
        "<table>"
        "<thead><tr><th>Scenario</th><th>Status</th><th>Reason</th><th>Baseline</th><th>Candidate</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</section>"
    )


def _short_fingerprint(inputs: dict[str, Any]) -> str:
    parts = []
    for name in ("scenario", "source_trace"):
        record = inputs.get(name) if isinstance(inputs.get(name), dict) else {}
        sha = record.get("sha256") if isinstance(record.get("sha256"), str) else None
        parts.append(f"{name}={sha[:12] if sha else 'missing'}")
    return "; ".join(parts)


def _trend_point(summary: dict[str, Any], path: Path, index: int) -> dict[str, Any]:
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    failed_rule_counts = _count_map(metrics.get("failed_rule_counts"))
    critical_failure_counts = _count_map(metrics.get("critical_failure_counts"))
    metadata = _metadata_map(summary.get("metadata"))
    return {
        "index": index,
        "label": _trend_label(summary, path, index, metadata),
        "path": _display_path(path),
        "metadata": metadata,
        "total": _int_value(summary.get("total")),
        "passed": _int_value(summary.get("passed")),
        "failed": _int_value(summary.get("failed")),
        "error_count": _int_value(summary.get("error_count")),
        "pass_rate": _number(metrics.get("pass_rate")),
        "average_score": _number(metrics.get("average_score")),
        "failed_rule_counts": failed_rule_counts,
        "critical_failure_counts": critical_failure_counts,
        "failed_rule_count": sum(failed_rule_counts.values()),
        "critical_failure_count": sum(critical_failure_counts.values()),
    }


def _trend_label(summary: dict[str, Any], path: Path, index: int, metadata: dict[str, str]) -> str:
    for key in ("candidate", "run", "iteration", "model"):
        if metadata.get(key):
            return metadata[key]
    label = summary.get("out_dir") or summary.get("scenarios_dir")
    if isinstance(label, str) and label:
        return label
    return f"point-{index + 1}"


def _trend_delta(previous: dict[str, Any], point: dict[str, Any]) -> dict[str, Any]:
    return {
        "pass_rate_delta": _number_delta(previous.get("pass_rate"), point.get("pass_rate")),
        "average_score_delta": _number_delta(previous.get("average_score"), point.get("average_score")),
        "failed_rule_count_delta": _int_value(point.get("failed_rule_count")) - _int_value(previous.get("failed_rule_count")),
        "critical_failure_count_delta": (
            _int_value(point.get("critical_failure_count")) - _int_value(previous.get("critical_failure_count"))
        ),
    }


def _count_trends(points: list[dict[str, Any]], field_name: str) -> list[dict[str, Any]]:
    ids = sorted(
        {
            rule_id
            for point in points
            for rule_id in (point.get(field_name) if isinstance(point.get(field_name), dict) else {})
        }
    )
    rows: list[dict[str, Any]] = []
    for rule_id in ids:
        counts = [
            {
                "index": point["index"],
                "label": point["label"],
                "count": _int_value(point.get(field_name, {}).get(rule_id)) if isinstance(point.get(field_name), dict) else 0,
            }
            for point in points
        ]
        first = counts[0]["count"] if counts else 0
        last = counts[-1]["count"] if counts else 0
        rows.append({"id": rule_id, "first_count": first, "last_count": last, "delta": last - first, "counts": counts})
    return sorted(rows, key=lambda item: (-abs(int(item["delta"])), str(item["id"])))


def _trend_count_table(value: Any, title: str) -> str:
    rows_data = value if isinstance(value, list) else []
    if not rows_data:
        return ""
    rows = []
    for item in rows_data:
        if not isinstance(item, dict):
            continue
        delta = int(item.get("delta", 0) or 0)
        cls = "regressed" if delta > 0 else "improved" if delta < 0 else "unchanged"
        counts = item.get("counts") if isinstance(item.get("counts"), list) else []
        sparkline = " -> ".join(str(count.get("count", 0)) for count in counts if isinstance(count, dict))
        rows.append(
            "<tr>"
            f"<td><code>{_esc(item.get('id'))}</code></td>"
            f"<td>{_esc(sparkline)}</td>"
            f"<td class=\"{cls}\">{_esc(delta)}</td>"
            "</tr>"
        )
    if not rows:
        return ""
    return (
        '<section class="trend-counts">'
        f"<h2>{_esc(title)}</h2>"
        "<table>"
        "<thead><tr><th>Rule</th><th>Counts</th><th>Delta</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</section>"
    )


def _suite_trend_summary(points: list[dict[str, Any]]) -> str:
    if not points:
        return "TREND: no suite summaries."
    if len(points) == 1:
        point = points[0]
        return (
            f"TREND: one point; pass rate {point.get('pass_rate')}; "
            f"average score {point.get('average_score')}."
        )
    first = points[0]
    last = points[-1]
    pass_delta = _number_delta(first.get("pass_rate"), last.get("pass_rate"))
    score_delta = _number_delta(first.get("average_score"), last.get("average_score"))
    failed_delta = _int_value(last.get("failed_rule_count")) - _int_value(first.get("failed_rule_count"))
    critical_delta = _int_value(last.get("critical_failure_count")) - _int_value(first.get("critical_failure_count"))
    return (
        f"TREND: {len(points)} points; pass_rate_delta={pass_delta}; "
        f"average_score_delta={score_delta}; failed_rule_delta={failed_delta}; "
        f"critical_failure_delta={critical_delta}."
    )


def _count_map(value: Any) -> dict[str, int]:
    if not isinstance(value, list):
        return {}
    counts: dict[str, int] = {}
    for row in value:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row.get("id"):
            continue
        counts[row["id"]] = _int_value(row.get("count"))
    return counts


def _metadata_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): value
        for key, value in sorted(value.items())
        if isinstance(key, str) and key and isinstance(value, str)
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _score(scorecard: dict[str, Any]) -> int:
    try:
        return int(scorecard.get("score", 0))
    except (TypeError, ValueError):
        return 0


def _display_path(path: Path) -> str:
    raw = str(path)
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


def _mean(values: list[int]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def _number(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _number_delta(before: Any, after: Any) -> float:
    return round(_number(after) - _number(before), 4)


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0


def _suite_compare_summary(
    regressed: bool,
    regressions: list[str],
    missing_in_candidate: list[str],
    improvements: list[str],
    aggregate: dict[str, Any],
) -> str:
    if regressed:
        parts = []
        if regressions:
            parts.append(f"scenario regressions: {', '.join(regressions)}")
        if missing_in_candidate:
            parts.append(f"missing candidate scenarios: {', '.join(missing_in_candidate)}")
        return f"REGRESSION: avg score delta {aggregate.get('avg_score_delta')}; {'; '.join(parts)}."
    if improvements:
        return f"NO REGRESSION: avg score delta {aggregate.get('avg_score_delta')}; improved scenarios: {', '.join(improvements)}."
    return f"NO REGRESSION: avg score delta {aggregate.get('avg_score_delta')}; paired scenarios unchanged."


def _compare_summary(regressed: bool, score_delta: int, regressions: list[str], fixes: list[str]) -> str:
    if regressed:
        details = ", ".join(regressions) if regressions else "score or critical-failure regression"
        return f"REGRESSION: score delta {score_delta}; regressed rules: {details}."
    if fixes:
        return f"NO REGRESSION: score delta {score_delta}; fixed rules: {', '.join(fixes)}."
    return f"NO REGRESSION: score delta {score_delta}; rule outcomes unchanged."


def _bool_label(value: Any) -> str:
    if value is True:
        return "PASS"
    if value is False:
        return "FAIL"
    return "missing"


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)
