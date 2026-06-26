"""Static HTML reporting."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .redaction import redact_text


def write_report(
    scenario: dict[str, Any],
    trace: dict[str, Any],
    scorecard: dict[str, Any],
    out_path: str | Path,
    regression_path: str | Path | None = None,
) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(scenario, trace, scorecard, regression_path), encoding="utf-8")


def render_report(
    scenario: dict[str, Any],
    trace: dict[str, Any],
    scorecard: dict[str, Any],
    regression_path: str | Path | None = None,
) -> str:
    secret_patterns = scenario.get("policy", {}).get("secret_patterns") or []
    status = "PASS" if scorecard.get("passed") else "FAIL"
    status_class = "pass" if scorecard.get("passed") else "fail"
    rules = "\n".join(_render_rule(rule) for rule in scorecard.get("rules", []))
    task_completion = _render_task_completion(scorecard)
    action_checklist = "" if task_completion else _render_action_checklist(scorecard)
    timeline = "\n".join(_render_event(i, event, secret_patterns) for i, event in enumerate(trace.get("events", [])))
    violations = "\n".join(_render_violation(rule) for rule in scorecard.get("rules", []) if not rule.get("passed"))
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
    :root {{ color-scheme: light; --ink:#17202a; --muted:#566573; --line:#d6dbdf; --ok:#147a3d; --bad:#b42318; --bg:#f7f9fb; --card:#ffffff; --accent:#1f6feb; }}
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
    h1, h2, h3 {{ margin:0 0 10px; }}
    .metric {{ font-size:40px; line-height:1; font-weight:800; }}
    .muted {{ color:var(--muted); }}
	    .rule {{ border-left:5px solid var(--ok); margin-bottom:12px; }}
	    .rule.fail {{ border-left-color:var(--bad); }}
	    .rule h3 {{ display:flex; justify-content:space-between; gap:12px; }}
	    .checklist {{ display:grid; gap:10px; }}
	    .check {{ border:1px solid var(--line); border-left:5px solid var(--ok); border-radius:8px; padding:12px; background:#fff; }}
	    .check.fail {{ border-left-color:var(--bad); }}
	    .check strong {{ display:flex; justify-content:space-between; gap:12px; }}
	    .timeline {{ display:grid; gap:10px; }}
    .event {{ display:grid; grid-template-columns:72px 160px 1fr; gap:12px; align-items:start; }}
    code, pre {{ background:#eef2f6; border-radius:6px; padding:2px 5px; }}
    pre {{ overflow:auto; padding:12px; white-space:pre-wrap; }}
    ul {{ margin:8px 0 0; padding-left:20px; }}
    .violations .rule {{ border-left-color:var(--bad); }}
    @media (max-width: 720px) {{ .hero, .event {{ grid-template-columns:1fr; }} header {{ padding:22px; }} main {{ padding:16px; }} }}
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


def write_index(run_dirs: list[Path], out_path: str | Path) -> None:
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
    html_doc = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hermes Flight Recorder Demo Runs</title>
<style>body{font-family:ui-sans-serif,system-ui;margin:32px;color:#17202a}table{border-collapse:collapse;width:100%;max-width:1100px}th,td{border-bottom:1px solid #d6dbdf;text-align:left;padding:12px}.pass{color:#147a3d;font-weight:800}.fail{color:#b42318;font-weight:800}a{color:#1f6feb}</style>
</head><body><h1>Hermes Flight Recorder Demo Runs</h1><table><thead><tr><th>Scenario</th><th>Status</th><th>Score</th><th>Summary</th></tr></thead><tbody>
""" + "\n".join(rows) + "\n</tbody></table></body></html>\n"
    output.write_text(html_doc, encoding="utf-8")


def _render_rule(rule: dict[str, Any]) -> str:
    cls = "" if rule.get("passed") else " fail"
    status = "PASS" if rule.get("passed") else "FAIL"
    evidence = "".join(f"<li>{_esc(item)}</li>" for item in rule.get("evidence", []))
    return f"<article class=\"rule{cls}\"><h3><span>{_esc(rule['name'])}</span><span>{status}</span></h3><ul>{evidence}</ul></article>"


def _render_violation(rule: dict[str, Any]) -> str:
    return _render_rule(rule)


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
