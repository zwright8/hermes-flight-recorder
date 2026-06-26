"""Portable scorecard artifacts for CI and regression comparison."""

from __future__ import annotations

import html
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

COMPARE_SCHEMA_VERSION = "hfr.compare.v1"


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


def _score_summary(scorecard: dict[str, Any], label: str) -> dict[str, Any]:
    return {
        "label": label,
        "scenario_id": scorecard.get("scenario_id"),
        "passed": bool(scorecard.get("passed")),
        "score": scorecard.get("score"),
        "critical_failures": scorecard.get("critical_failures", []),
    }


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
