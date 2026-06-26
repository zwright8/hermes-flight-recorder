"""Portable scorecard artifacts for CI and regression comparison."""

from __future__ import annotations

import html
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

COMPARE_SCHEMA_VERSION = "hfr.compare.v1"
SUITE_COMPARE_SCHEMA_VERSION = "hfr.suite_compare.v1"


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
) -> dict[str, Any]:
    """Compare two directories of Flight Recorder run artifacts."""
    baseline_root = Path(baseline_dir)
    candidate_root = Path(candidate_dir)
    baseline = _suite_scorecards(baseline_root)
    candidate = _suite_scorecards(candidate_root, allow_empty=True)

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
        delta = int(comparison["score_delta"])
        paired_deltas.append(delta)
        paired_baseline_scores.append(_score(before["scorecard"]))
        paired_candidate_scores.append(_score(after["scorecard"]))
        paired_baseline_passed += 1 if before["scorecard"].get("passed") else 0
        paired_candidate_passed += 1 if after["scorecard"].get("passed") else 0

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
    }
    regressed = bool(regressions or missing_in_candidate)
    return {
        "schema_version": SUITE_COMPARE_SCHEMA_VERSION,
        "baseline": {
            "label": baseline_label or _display_path(baseline_root),
            "path": _display_path(baseline_root),
            "scenario_count": len(baseline),
        },
        "candidate": {
            "label": candidate_label or _display_path(candidate_root),
            "path": _display_path(candidate_root),
            "scenario_count": len(candidate),
        },
        "aggregate": aggregate,
        "scenario_changes": scenario_changes,
        "regressions": regressions,
        "improvements": improvements,
        "missing_in_candidate": missing_in_candidate,
        "new_in_candidate": new_in_candidate,
        "regressed": regressed,
        "summary": _suite_compare_summary(regressed, regressions, missing_in_candidate, improvements, aggregate),
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
    rows = []
    for change in comparison.get("scenario_changes", []):
        rows.append(
            "<tr>"
            f"<td>{_esc(change.get('scenario_id'))}</td>"
            f"<td class=\"{_esc(change.get('status'))}\">{_esc(change.get('status'))}</td>"
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
    .unchanged {{ color:#566573; }}
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
  <section class="metrics">
    <div class="metric"><span>Paired Scenarios</span><strong>{aggregate.get('paired_count')}</strong></div>
    <div class="metric"><span>Avg Score Delta</span><strong>{aggregate.get('avg_score_delta')}</strong></div>
    <div class="metric"><span>Regressions</span><strong>{len(comparison.get('regressions', []))}</strong></div>
    <div class="metric"><span>Missing Candidate</span><strong>{len(comparison.get('missing_in_candidate', []))}</strong></div>
  </section>
  <table>
    <thead><tr><th>Scenario</th><th>Status</th><th>Baseline</th><th>Candidate</th><th>Delta</th><th>Summary</th></tr></thead>
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
        }
    if not scorecards and not allow_empty:
        raise ArtifactError(f"No scorecard.json files found in suite directory: {root}")
    return scorecards


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
