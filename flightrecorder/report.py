"""Static HTML reporting."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any

from .redaction import redact_text
from .state_diff import resolve_state_diff_semantics


def write_report(
    scenario: dict[str, Any],
    trace: dict[str, Any],
    scorecard: dict[str, Any],
    out_path: str | Path,
    regression_path: str | Path | None = None,
    state_diff: dict[str, Any] | None = None,
) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(scenario, trace, scorecard, regression_path, state_diff), encoding="utf-8")


def render_report(
    scenario: dict[str, Any],
    trace: dict[str, Any],
    scorecard: dict[str, Any],
    regression_path: str | Path | None = None,
    state_diff: dict[str, Any] | None = None,
) -> str:
    secret_patterns = scenario.get("policy", {}).get("secret_patterns") or []
    status = "PASS" if scorecard.get("passed") else "FAIL"
    status_class = "pass" if scorecard.get("passed") else "fail"
    rules = "\n".join(_render_rule(rule, secret_patterns) for rule in scorecard.get("rules", []))
    task_completion = _render_task_completion(scorecard)
    state_changes = _render_state_diff(state_diff, secret_patterns)
    action_checklist = "" if task_completion else _render_action_checklist(scorecard)
    timeline = "\n".join(_render_event(i, event, secret_patterns) for i, event in enumerate(trace.get("events", [])))
    violations = "\n".join(
        _render_violation(rule, secret_patterns) for rule in scorecard.get("rules", []) if not rule.get("passed")
    )
    if not violations:
        violations = "<p>No policy violations detected.</p>"
    regression = ""
    if regression_path:
        regression = (
            "<section class=\"panel regression\">"
            "<h2>Regression Artifact</h2>"
            f"<p>Saved failing scenario: <code>{_esc(str(regression_path))}</code></p>"
            f"<pre>{_esc('python -m flightrecorder run --scenario ' + str(regression_path) + ' --out runs/replay')}</pre>"
            "</section>"
        )

    final_answer = _esc(redact_text(trace.get("final_answer") or "", secret_patterns))
    source_trace = _esc(_safe_path_label(str((scenario.get("trace") or {}).get("path") or "CLI override")))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_esc(scenario['title'])} - Hermes Flight Recorder</title>
  <style>
    :root {{ color-scheme: light; --ink:#17202a; --muted:#566573; --line:#d6dbdf; --ok:#147a3d; --bad:#b42318; --warn:#b54708; --bg:#f7f9fb; --card:#ffffff; --accent:#1f6feb; }}
    body {{ margin:0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:var(--bg); color:var(--ink); }}
    header {{ background:#101820; color:#fff; padding:28px 32px; }}
    header p {{ color:#c7d0d9; max-width:880px; }}
    main {{ max-width:1180px; margin:0 auto; padding:24px; }}
    .hero {{ display:grid; grid-template-columns: 1fr auto; gap:18px; align-items:center; }}
    .badge {{ border-radius:6px; padding:10px 14px; font-weight:800; letter-spacing:.04em; }}
    .badge.pass {{ background:#d1fadf; color:var(--ok); }}
    .badge.fail {{ background:#fee4e2; color:var(--bad); }}
    .grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap:16px; }}
    .panel, .rule, .event {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:16px; box-shadow:0 1px 2px rgba(16,24,40,.04); }}
    .panel, .rule, .event, .grid {{ min-width:0; }}
    h1, h2, h3 {{ margin:0 0 10px; }}
    .metric {{ font-size:40px; line-height:1; font-weight:800; }}
    .muted {{ color:var(--muted); }}
    .rule {{ border-left:5px solid var(--ok); margin-bottom:12px; }}
    .rule.fail {{ border-left-color:var(--bad); }}
    .rule.unknown {{ border-left-color:var(--warn); }}
    .rule h3 {{ display:flex; justify-content:space-between; gap:12px; }}
    .checklist {{ display:grid; gap:10px; }}
    .check {{ border:1px solid var(--line); border-left:5px solid var(--ok); border-radius:8px; padding:12px; background:#fff; }}
    .check.fail {{ border-left-color:var(--bad); }}
    .check strong {{ display:flex; justify-content:space-between; gap:12px; }}
    .timeline {{ display:grid; gap:10px; }}
    .diff-table {{ width:100%; border-collapse:collapse; margin-top:12px; table-layout:fixed; }}
    .diff-table th, .diff-table td {{ border-bottom:1px solid var(--line); text-align:left; padding:10px; vertical-align:top; }}
    .diff-table th:nth-child(1) {{ width:32%; }}
    .diff-table th:nth-child(2) {{ width:12%; }}
    .diff-table pre {{ margin:0; max-height:180px; }}
    .event {{ display:grid; grid-template-columns:72px 160px 1fr; gap:12px; align-items:start; }}
    code, pre {{ background:#eef2f6; border-radius:6px; padding:2px 5px; }}
    code, pre, p {{ overflow-wrap:anywhere; word-break:break-word; }}
    pre {{ overflow:auto; padding:12px; white-space:pre-wrap; }}
    ul {{ margin:8px 0 0; padding-left:20px; }}
    .violations .rule {{ border-left-color:var(--bad); }}
    @media (max-width: 720px) {{
      .hero, .event {{ grid-template-columns:1fr; }}
      header {{ padding:22px; }}
      main {{ padding:16px; }}
      .diff-table, .diff-table tbody, .diff-table tr, .diff-table td {{ display:block; width:100%; box-sizing:border-box; }}
      .diff-table thead {{ display:none; }}
      .diff-table tr {{ border:1px solid var(--line); border-radius:8px; padding:10px; margin-bottom:10px; }}
      .diff-table td {{ border-bottom:0; padding:6px 0; }}
      .diff-table td::before {{ content:attr(data-label); display:block; margin-bottom:4px; color:var(--muted); font-weight:800; font-size:12px; text-transform:uppercase; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="hero">
      <div>
        <h1>Hermes Autonomy Flight Recorder</h1>
        <p>{_esc(scenario['title'])}</p>
      </div>
      <div class="badge {status_class}">{status}</div>
    </div>
  </header>
  <main>
    <section class="grid">
      <div class="panel"><h2>Score</h2><div class="metric">{scorecard['score']}</div><p class="muted">Threshold: {scorecard['pass_threshold']}</p></div>
      <div class="panel"><h2>Scenario</h2><p><code>{_esc(scenario['id'])}</code></p><p class="muted">{_esc(scenario['prompt'])}</p></div>
      <div class="panel"><h2>Source Trace</h2><p><code>{source_trace}</code></p><p class="muted">Format: {_esc(trace.get('session', {}).get('source_format', 'unknown'))}</p></div>
    </section>
    <section class="panel"><h2>Summary</h2><p>{_esc(scorecard['summary'])}</p></section>
{task_completion}
{state_changes}
{action_checklist}
    <section class="panel"><h2>Final Answer</h2><pre>{final_answer}</pre></section>
    <section class="panel"><h2>Scorecard</h2>{rules}</section>
    <section class="panel violations"><h2>Violations</h2>{violations}</section>
{regression}
    <section class="panel"><h2>Timeline</h2><div class="timeline">{timeline}</div></section>
  </main>
</body>
</html>
"""


_INDEX_ARTIFACTS = (
    ("suite_summary.json", "Suite Summary"),
    ("scenario_quality.json", "Scenario Quality"),
    ("evidence_coverage.json", "Evidence Coverage"),
    ("trace_observability.json", "Trace Observability"),
    ("repair_queue.json", "Repair Queue"),
    ("validation.json", "Validation Summary"),
    ("evidence_bundle.json", "Evidence Bundle"),
    ("evidence_bundle_trainer.json", "Trainer Evidence Bundle"),
    ("improvement_plan.json", "Improvement Plan"),
    ("improvement_ledger.json", "Improvement Ledger"),
    ("improvement_ledger_gate.json", "Improvement Ledger Gate"),
    ("action_ledger.json", "Action Ledger"),
    ("action_ledger_gate.json", "Action Ledger Gate"),
    ("promotion_decision.json", "Promotion Decision"),
    ("promotion_ledger.json", "Promotion Ledger"),
    ("promotion_ledger_gate.json", "Promotion Ledger Gate"),
    ("promotion_archive/promotion_archive.json", "Promotion Archive"),
    ("suite_gate.json", "Suite Gate"),
    ("training_gate.json", "Training Export Gate"),
    ("compare_gate.json", "Comparison Export Gate"),
    ("reviewed_gate.json", "Reviewed Export Gate"),
    ("training_export/manifest.json", "Training Export"),
    ("compare_rl_export/manifest.json", "Comparison RL Export"),
    ("review_queue/manifest.json", "Review Queue"),
    ("reviewed_export/manifest.json", "Reviewed Export"),
    ("review_calibration.json", "Review Calibration"),
    ("trainer_preflight.json", "Trainer Preflight"),
    ("trainer_launch_check.json", "Trainer Launch Check"),
    ("trainer_archive/trainer_archive.json", "Trainer Archive"),
    ("trainer_archive_check.json", "Trainer Archive Check"),
    ("trainer_consumer_plan.json", "Trainer Consumer Plan"),
    ("trainer_wrapper_dry_run.json", "Trainer Wrapper Dry Run"),
    ("live_smoke_summary.json", "Live Smoke Summary"),
    ("suite_compare.json", "Suite Compare"),
    ("suite_trend.json", "Suite Trend"),
)


def write_index(run_dirs: list[Path], out_path: str | Path, artifacts_dir: str | Path | None = None) -> None:
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_dir in run_dirs:
        score_path = run_dir / "scorecard.json"
        if not score_path.exists():
            continue
        scorecard = json.loads(score_path.read_text(encoding="utf-8"))
        status = "PASS" if scorecard.get("passed") else "FAIL"
        cls = "pass" if scorecard.get("passed") else "fail"
        rows.append(
            f"<tr><td><a href=\"{_esc(run_dir.name)}/report.html\">{_esc(scorecard.get('scenario_title') or run_dir.name)}</a></td>"
            f"<td class=\"{cls}\">{status}</td><td>{scorecard.get('score')}</td><td>{_esc(scorecard.get('summary', ''))}</td></tr>"
        )
    artifact_rows = _index_artifact_rows(Path(artifacts_dir) if artifacts_dir is not None else output.parent, output.parent)
    artifact_section = ""
    if artifact_rows:
        artifact_section = (
            "<section><h2>Evidence Artifacts</h2>"
            "<p class=\"muted\">Generated handoff, improvement-loop, training, review, and promotion artifacts discovered for this run set.</p>"
            "<table><thead><tr><th>Artifact</th><th>Status</th><th>Key Metrics</th><th>Summary</th></tr></thead><tbody>"
            + "\n".join(artifact_rows)
            + "\n</tbody></table></section>"
        )
    html_doc = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hermes Flight Recorder Demo Runs</title>
<style>body{font-family:ui-sans-serif,system-ui;margin:32px;color:#17202a}main{max-width:1180px}section{margin-top:30px}table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid #d6dbdf;text-align:left;padding:12px;vertical-align:top}.pass{color:#147a3d;font-weight:800}.fail{color:#b42318;font-weight:800}.warn{color:#9a6700;font-weight:800}.neutral{color:#566573;font-weight:800}.muted{color:#566573;max-width:900px}code{background:#eef2f6;border-radius:6px;padding:2px 5px}a{color:#1f6feb}td{overflow-wrap:anywhere}</style>
</head><body><main><h1>Hermes Flight Recorder Demo Runs</h1><section><h2>Scenario Reports</h2><table><thead><tr><th>Scenario</th><th>Status</th><th>Score</th><th>Summary</th></tr></thead><tbody>
""" + "\n".join(rows) + f"\n</tbody></table></section>{artifact_section}</main></body></html>\n"
    output.write_text(html_doc, encoding="utf-8")


def _render_rule(rule: dict[str, Any], secret_patterns: list[str] | None = None) -> str:
    cls = "" if rule.get("passed") else " fail"
    status = "PASS" if rule.get("passed") else "FAIL"
    evidence = "".join(f"<li>{_esc(redact_text(item, secret_patterns or []))}</li>" for item in rule.get("evidence", []))
    return f"<article class=\"rule{cls}\"><h3><span>{_esc(rule['name'])}</span><span>{status}</span></h3><ul>{evidence}</ul></article>"


def _index_artifact_rows(artifacts_dir: Path, link_base: Path) -> list[str]:
    rows: list[str] = []
    for relative_path, title in _INDEX_ARTIFACTS:
        path = artifacts_dir / relative_path
        if not path.exists() or not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        state, cls = _artifact_state(payload)
        metrics = _artifact_metrics(payload)
        summary = _artifact_summary(payload)
        href = os.path.relpath(path, link_base)
        rows.append(
            "<tr>"
            f"<td><a href=\"{_esc(href)}\">{_esc(title)}</a><br><code>{_esc(relative_path)}</code></td>"
            f"<td class=\"{cls}\">{_esc(state)}</td>"
            f"<td>{_esc(metrics)}</td>"
            f"<td>{_esc(summary)}</td>"
            "</tr>"
        )
    return rows


def _artifact_state(payload: dict[str, Any]) -> tuple[str, str]:
    passed = payload.get("passed")
    if isinstance(passed, bool):
        return ("PASS" if passed else "FAIL", "pass" if passed else "fail")
    decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
    readiness = decision.get("readiness") or payload.get("readiness")
    if readiness == "blocked":
        return "BLOCKED", "fail"
    if readiness == "ready":
        return "READY", "pass"
    if payload.get("schema_version"):
        return "RECORDED", "neutral"
    return "UNKNOWN", "warn"


def _artifact_summary(payload: dict[str, Any]) -> str:
    decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
    for source in (decision, payload):
        summary = source.get("summary") if isinstance(source, dict) else None
        if isinstance(summary, str) and summary:
            return summary
    recommendation = decision.get("recommendation") or payload.get("recommendation")
    if isinstance(recommendation, str) and recommendation:
        return recommendation
    return str(payload.get("schema_version") or "No summary available.")


def _artifact_metrics(payload: dict[str, Any]) -> str:
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    candidates = (
        ("run_count", "runs"),
        ("total", "total"),
        ("passed", "passed"),
        ("failed", "failed"),
        ("pass_rate", "pass_rate"),
        ("average_score", "avg_score"),
        ("scenario_count", "scenarios"),
        ("item_count", "items"),
        ("work_item_count", "work_items"),
        ("open_work_item_count", "open"),
        ("recurring_work_item_count", "recurring"),
        ("resolved_work_item_count", "resolved"),
        ("unique_work_item_count", "unique"),
        ("episode_count", "episodes"),
        ("preference_count", "preferences"),
        ("sft_count", "sft"),
        ("dpo_count", "dpo"),
        ("candidate_win_count", "candidate_wins"),
        ("baseline_win_count", "baseline_wins"),
        ("task_completion_improvement_count", "task_improvements"),
        ("task_completion_regression_count", "task_regressions"),
        ("reviewed_label_count", "labels"),
        ("agreement_rate", "agreement"),
        ("check_count", "checks"),
        ("failed_check_count", "failed_checks"),
        ("decision_count", "decisions"),
        ("allowed_count", "allowed"),
        ("blocked_count", "blocked"),
        ("artifact_count", "artifacts"),
        ("trainer_input_count", "trainer_inputs"),
        ("path_rewrite_count", "path_rewrites"),
        ("external_command_path_count", "external_paths"),
        ("missing_external_code_count", "missing_external"),
        ("trainer_input_available_count", "inputs_available"),
        ("trainer_input_ready_count", "inputs_ready"),
        ("external_code_ready_count", "external_ready"),
        ("command_arg_count", "command_args"),
        ("missing_count", "missing"),
    )
    parts: list[str] = []
    for field_name, label in candidates:
        value = payload.get(field_name)
        if value is None:
            value = metrics.get(field_name)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float, str)):
            parts.append(f"{label}: {value}")
        if len(parts) >= 5:
            break
    return ", ".join(parts) if parts else "schema: " + str(payload.get("schema_version") or "unknown")


def _render_violation(rule: dict[str, Any], secret_patterns: list[str] | None = None) -> str:
    return _render_rule(rule, secret_patterns)


def _render_task_completion(scorecard: dict[str, Any]) -> str:
    task = scorecard.get("task_completion")
    if not isinstance(task, dict):
        return ""
    status = str(task.get("status") or "unknown")
    passed = bool(task.get("passed"))
    cls = "" if passed else " fail"
    rows = []
    for check in task.get("checks", []):
        if not isinstance(check, dict):
            continue
        check_status = "PASS" if check.get("passed") else "FAIL"
        check_cls = "" if check.get("passed") else " fail"
        rows.append(
            f"<article class=\"check{check_cls}\"><strong><span>{_esc(check.get('description') or check.get('id'))}</span>"
            f"<span>{check_status}</span></strong><p class=\"muted\">{_esc(check.get('evidence', ''))}</p></article>"
        )
    body = "".join(rows) if rows else "<p class=\"muted\">No task-completion evidence assertions configured.</p>"
    return (
        f"<section class=\"panel\"><h2>Task Completion</h2>"
        f"<article class=\"rule{cls}\"><h3><span>{_esc(status.replace('_', ' ').title())}</span>"
        f"<span>{'PASS' if passed else 'FAIL'}</span></h3><p>{_esc(task.get('summary', ''))}</p></article>"
        f"<div class=\"checklist\">{body}</div></section>"
    )


def _render_state_diff(state_diff: dict[str, Any] | None, secret_patterns: list[str]) -> str:
    if not isinstance(state_diff, dict):
        return ""
    changes = state_diff.get("changes")
    if not isinstance(changes, list):
        changes = []
    rows = []
    for change in changes:
        if not isinstance(change, dict):
            continue
        rows.append(
            "<tr>"
            f"<td data-label=\"Path\"><code>{_esc(change.get('path') or '')}</code></td>"
            f"<td data-label=\"Kind\">{_esc(change.get('kind') or '')}</td>"
            f"<td data-label=\"Before\"><pre>{_esc(_render_diff_value(change.get('before'), secret_patterns))}</pre></td>"
            f"<td data-label=\"After\"><pre>{_esc(_render_diff_value(change.get('after'), secret_patterns))}</pre></td>"
            "</tr>"
        )
    comparison_complete, change_status = resolve_state_diff_semantics(state_diff)
    body = (
        "<table class=\"diff-table\"><thead><tr><th>Path</th><th>Kind</th><th>Before</th><th>After</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        if rows
        else (
            "<p class=\"muted\">No changed state paths were emitted before the comparison became incomplete.</p>"
            if not comparison_complete
            else "<p class=\"muted\">No changed state paths were emitted.</p>"
        )
    )
    status = change_status.upper()
    status_class = " unknown" if change_status == "unknown" else ""
    return (
        f"<section class=\"panel\"><h2>State Changes</h2>"
        f"<article class=\"rule{status_class}\"><h3><span>{status}</span>"
        f"<span>{_esc(state_diff.get('change_count', 0))}</span></h3>"
        f"<p>{_esc(state_diff.get('summary', ''))}</p></article>{body}</section>"
    )


def _render_diff_value(value: Any, secret_patterns: list[str]) -> str:
    if value is None:
        rendered = "null"
    elif isinstance(value, (dict, list)):
        rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    else:
        rendered = str(value)
    return redact_text(rendered, secret_patterns)


def _render_action_checklist(scorecard: dict[str, Any]) -> str:
    evidence_rules = [
        rule
        for rule in scorecard.get("rules", [])
        if rule.get("id") in {"required_actions", "required_action_sequences", "required_event_counts"} and rule.get("items")
    ]
    if not evidence_rules:
        return ""
    rows = []
    for rule in evidence_rules:
        for item in rule.get("items", []):
            passed = bool(item.get("passed"))
            status = "PASS" if passed else "FAIL"
            cls = "" if passed else " fail"
            rows.append(
                f"<article class=\"check{cls}\"><strong><span>{_esc(item.get('description') or item.get('id'))}</span>"
                f"<span>{status}</span></strong><p class=\"muted\">{_esc(rule.get('name', 'Evidence'))}: "
                f"{_esc(item.get('evidence', ''))}</p></article>"
            )
    return f"<section class=\"panel\"><h2>Task Evidence Checklist</h2><div class=\"checklist\">{''.join(rows)}</div></section>"


def _render_event(index: int, event: dict[str, Any], secret_patterns: list[str]) -> str:
    event_type = _esc(str(event.get("type") or "event"))
    label = _esc(str(event.get("tool_name") or event.get("status") or ""))
    details = redact_text(event, secret_patterns)
    return f"<article class=\"event\"><div><strong>#{index}</strong></div><div>{event_type}<br><span class=\"muted\">{label}</span></div><pre>{_esc(details)}</pre></article>"


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _safe_path_label(value: str) -> str:
    if _is_windows_absolute(value):
        return f"<redacted:{_basename(value)}>"
    if not value.startswith("/"):
        return value
    path = Path(value)
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return f"<redacted:{path.name}>"


def _is_windows_absolute(value: str) -> bool:
    normalized = value.replace("/", "\\")
    return (len(normalized) >= 3 and normalized[1:3] == ":\\" and normalized[0].isalpha()) or normalized.startswith("\\\\")


def _basename(value: str) -> str:
    return value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] or "path"
