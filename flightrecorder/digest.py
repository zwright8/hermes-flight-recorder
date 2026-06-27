"""Compact per-run evidence digests for automation and training handoffs."""

from __future__ import annotations

import json
import re
from typing import Any

from .redaction import redact_text

RUN_DIGEST_SCHEMA_VERSION = "hfr.run_digest.v1"
FAMILY_SUFFIX_RE = re.compile(r"([_-](good|bad|pass|fail|passing|failing|chosen|rejected))+$", re.IGNORECASE)


class RunDigestError(ValueError):
    """Raised when a run digest cannot be built from the supplied artifacts."""


def build_run_digest(
    scenario: dict[str, Any],
    trace: dict[str, Any],
    scorecard: dict[str, Any],
    *,
    state_diff: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact derived summary for one scored run.

    The digest intentionally duplicates only stable, bounded facts from the
    scenario, normalized trace, scorecard, task-completion verdict, and optional
    state diff. It is meant for CI bots, reviewers, repair agents, and future
    reward builders that need the run outcome without parsing the full report.
    """
    scenario_id = str(scenario.get("id") or scorecard.get("scenario_id") or "unknown")
    scenario_title = str(scenario.get("title") or scorecard.get("scenario_title") or scenario_id)
    secret_patterns = _secret_patterns(scenario)
    task_completion = _task_completion(scorecard)
    failed_rules = _failed_rules(scorecard)
    trace_signal = _trace_signal(trace)
    state_changes = _state_changes(state_diff)
    evidence = _evidence_summary(scorecard, task_completion)
    rules = _rules_summary(scorecard, failed_rules, secret_patterns)
    outcome = _outcome_summary(scorecard, task_completion)
    training_signals = _training_signals(scorecard, task_completion, state_changes, failed_rules)
    return {
        "schema_version": RUN_DIGEST_SCHEMA_VERSION,
        "scenario": {
            "id": scenario_id,
            "title": scenario_title,
            "task_family": _task_family(scenario_id),
        },
        "outcome": outcome,
        "trace_signal": trace_signal,
        "state_changes": state_changes,
        "rules": rules,
        "evidence": evidence,
        "training_signals": training_signals,
        "recommended_actions": _recommended_actions(outcome, trace_signal, state_changes, failed_rules, scenario),
    }


def render_run_digest_markdown(digest: dict[str, Any]) -> str:
    """Render a concise Markdown view of a run digest."""
    scenario = digest.get("scenario") if isinstance(digest.get("scenario"), dict) else {}
    outcome = digest.get("outcome") if isinstance(digest.get("outcome"), dict) else {}
    trace_signal = digest.get("trace_signal") if isinstance(digest.get("trace_signal"), dict) else {}
    state_changes = digest.get("state_changes") if isinstance(digest.get("state_changes"), dict) else {}
    rules = digest.get("rules") if isinstance(digest.get("rules"), dict) else {}
    training = digest.get("training_signals") if isinstance(digest.get("training_signals"), dict) else {}
    lines = [
        f"# {_md(str(scenario.get('title') or scenario.get('id') or 'Run Digest'))}",
        "",
        f"- Scenario: `{_md(str(scenario.get('id') or 'unknown'))}`",
        f"- Outcome: {'PASS' if outcome.get('passed') else 'FAIL'}",
        f"- Score: `{outcome.get('score')}` / threshold `{outcome.get('pass_threshold')}`",
        f"- Task completion: `{_md(str(outcome.get('task_completion_status') or 'unknown'))}`",
        f"- Reward hint: `{training.get('score_reward')}` score reward, `{training.get('binary_reward')}` binary reward",
        "",
        "## Trace Signal",
        "",
        f"- Events: `{trace_signal.get('event_count')}`",
        f"- Event types: `{_md(', '.join(trace_signal.get('event_types') or []))}`",
        f"- Tool calls: `{trace_signal.get('tool_call_count')}`",
        f"- API calls: `{trace_signal.get('api_call_count')}`",
        f"- Subagent starts: `{trace_signal.get('subagent_start_count')}`",
        "",
        "## State Changes",
        "",
        f"- Available: `{str(bool(state_changes.get('available'))).lower()}`",
        f"- Changed: `{str(bool(state_changes.get('changed'))).lower()}`",
        f"- Change count: `{state_changes.get('change_count')}`",
    ]
    top_changes = state_changes.get("top_changes") if isinstance(state_changes.get("top_changes"), list) else []
    if top_changes:
        lines.extend(["", "| Path | Kind |", "| --- | --- |"])
        for change in top_changes:
            if not isinstance(change, dict):
                continue
            lines.append(f"| {_md(str(change.get('path') or ''))} | `{_md(str(change.get('kind') or ''))}` |")
    lines.extend(
        [
            "",
            "## Failed Rules",
            "",
            f"- Failed: `{rules.get('failed_count')}`",
            f"- Critical failed: `{rules.get('critical_failed_count')}`",
        ]
    )
    failed = rules.get("failed") if isinstance(rules.get("failed"), list) else []
    if failed:
        lines.extend(["", "| Rule | Critical | Evidence refs |", "| --- | --- | --- |"])
        for rule in failed:
            if not isinstance(rule, dict):
                continue
            lines.append(
                f"| `{_md(str(rule.get('id') or ''))}` | "
                f"`{str(bool(rule.get('critical'))).lower()}` | `{rule.get('evidence_ref_count')}` |"
            )
    actions = digest.get("recommended_actions") if isinstance(digest.get("recommended_actions"), list) else []
    lines.extend(["", "## Recommended Actions", ""])
    if actions:
        for action in actions:
            if isinstance(action, dict):
                lines.append(f"- `{_md(str(action.get('id') or 'unknown'))}`: {_md(str(action.get('reason') or ''))}")
    else:
        lines.append("- No actions recommended.")
    return "\n".join(lines).rstrip() + "\n"


def _outcome_summary(scorecard: dict[str, Any], task_completion: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": bool(scorecard.get("passed")),
        "score": _int_between(scorecard.get("score"), 0, 100),
        "pass_threshold": _int_between(scorecard.get("pass_threshold"), 0, 100),
        "critical_failures": _string_list(scorecard.get("critical_failures")),
        "summary": str(scorecard.get("summary") or ""),
        "task_completion_status": str(task_completion.get("status") or "not_applicable"),
        "task_completion_passed": bool(task_completion.get("passed", True)),
    }


def _trace_signal(trace: dict[str, Any]) -> dict[str, Any]:
    events = trace.get("events") if isinstance(trace.get("events"), list) else []
    typed_events = [event for event in events if isinstance(event, dict)]
    event_types = sorted({str(event.get("type")) for event in typed_events if event.get("type")})
    tool_call_count = sum(1 for event in typed_events if event.get("type") == "tool_call")
    api_call_count = _api_call_count(trace, typed_events)
    subagent_start_count = sum(1 for event in typed_events if event.get("type") == "subagent_start")
    final_answer = trace.get("final_answer")
    session = trace.get("session") if isinstance(trace.get("session"), dict) else {}
    return {
        "event_count": len(typed_events),
        "event_types": event_types,
        "tool_call_count": tool_call_count,
        "tool_result_count": sum(1 for event in typed_events if event.get("type") == "tool_result"),
        "api_call_count": api_call_count,
        "subagent_start_count": subagent_start_count,
        "max_subagent_depth": _max_subagent_depth(typed_events),
        "has_final_answer": isinstance(final_answer, str) and bool(final_answer.strip()),
        "has_tool_or_api_events": tool_call_count > 0 or api_call_count > 0,
        "source_format": str(session.get("source_format") or "unknown"),
        "model": str(session.get("model") or "unknown"),
    }


def _state_changes(state_diff: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(state_diff, dict):
        return {
            "available": False,
            "changed": False,
            "change_count": 0,
            "truncated": False,
            "summary": "No state diff artifact was available for this run.",
            "top_changes": [],
        }
    changes = state_diff.get("changes") if isinstance(state_diff.get("changes"), list) else []
    top_changes: list[dict[str, str]] = []
    for change in changes[:10]:
        if isinstance(change, dict):
            top_changes.append({"path": str(change.get("path") or ""), "kind": str(change.get("kind") or "")})
    return {
        "available": True,
        "changed": bool(state_diff.get("changed")),
        "change_count": _non_negative_int(state_diff.get("change_count")),
        "truncated": bool(state_diff.get("truncated")),
        "summary": str(state_diff.get("summary") or ""),
        "top_changes": top_changes,
    }


def _rules_summary(
    scorecard: dict[str, Any],
    failed_rules: list[dict[str, Any]],
    secret_patterns: list[str],
) -> dict[str, Any]:
    rules = scorecard.get("rules") if isinstance(scorecard.get("rules"), list) else []
    failed: list[dict[str, Any]] = []
    for rule in failed_rules:
        evidence = rule.get("evidence") if isinstance(rule.get("evidence"), list) else []
        evidence_refs = rule.get("evidence_refs") if isinstance(rule.get("evidence_refs"), list) else []
        failed.append(
            {
                "id": str(rule.get("id") or "unknown"),
                "name": str(rule.get("name") or rule.get("id") or "unknown"),
                "critical": bool(rule.get("critical")),
                "penalty": _int_between(rule.get("penalty"), 0, 100),
                "evidence": [redact_text(item, secret_patterns) for item in evidence[:3]],
                "evidence_ref_count": len(evidence_refs),
                "evidence_refs": _redact_jsonable(evidence_refs[:5], secret_patterns),
            }
        )
    return {
        "total_count": len([rule for rule in rules if isinstance(rule, dict)]),
        "failed_count": len(failed_rules),
        "critical_failed_count": sum(1 for rule in failed_rules if rule.get("critical") is True),
        "failed": failed,
    }


def _evidence_summary(scorecard: dict[str, Any], task_completion: dict[str, Any]) -> dict[str, Any]:
    failed_rules = _failed_rules(scorecard)
    critical_failed_rules = [rule for rule in failed_rules if rule.get("critical") is True]
    rule_ref_count = _rule_evidence_ref_count(_rules(scorecard))
    failed_rule_ref_count = _rule_evidence_ref_count(failed_rules)
    critical_failed_rule_ref_count = _rule_evidence_ref_count(critical_failed_rules)
    task_refs = task_completion.get("evidence_refs") if isinstance(task_completion.get("evidence_refs"), list) else []
    missing_refs = (
        task_completion.get("missing_evidence_refs")
        if isinstance(task_completion.get("missing_evidence_refs"), list)
        else []
    )
    return {
        "rule_evidence_ref_count": rule_ref_count,
        "failed_rule_evidence_ref_count": failed_rule_ref_count,
        "critical_failed_rule_evidence_ref_count": critical_failed_rule_ref_count,
        "task_completion_evidence_ref_count": len(task_refs),
        "missing_evidence_ref_count": len(missing_refs),
        "total_evidence_ref_count": rule_ref_count + len(task_refs) + len(missing_refs),
    }


def _training_signals(
    scorecard: dict[str, Any],
    task_completion: dict[str, Any],
    state_changes: dict[str, Any],
    failed_rules: list[dict[str, Any]],
) -> dict[str, Any]:
    score = _int_between(scorecard.get("score"), 0, 100)
    passed = bool(scorecard.get("passed"))
    task_status = str(task_completion.get("status") or "not_applicable")
    failure_modes = [
        {
            "rule_id": str(rule.get("id") or "unknown"),
            "critical": bool(rule.get("critical")),
            "evidence_ref_count": len(rule.get("evidence_refs") if isinstance(rule.get("evidence_refs"), list) else []),
        }
        for rule in failed_rules
    ]
    return {
        "score_reward": round(score / 100.0, 4),
        "binary_reward": 1 if passed else 0,
        "task_completion_reward": 1 if task_completion.get("passed", True) else 0,
        "task_completion_status": task_status,
        "task_completion_passed": bool(task_completion.get("passed", True)),
        "state_changed": bool(state_changes.get("changed")),
        "state_change_count": _non_negative_int(state_changes.get("change_count")),
        "failure_modes": failure_modes,
    }


def _recommended_actions(
    outcome: dict[str, Any],
    trace_signal: dict[str, Any],
    state_changes: dict[str, Any],
    failed_rules: list[dict[str, Any]],
    scenario: dict[str, Any],
) -> list[dict[str, str]]:
    actions: list[dict[str, str]] = []
    if outcome.get("passed") is True:
        actions.append(
            {
                "id": "positive_training_candidate",
                "priority": "low",
                "reason": "Run passed the deterministic scorecard and can seed positive examples or regression baselines.",
            }
        )
    if failed_rules:
        actions.append(
            {
                "id": "repair_failed_rules",
                "priority": "high",
                "reason": "Run has failed deterministic rules that can be converted into repair tasks.",
            }
        )
    if any(rule.get("critical") is True for rule in failed_rules):
        actions.append(
            {
                "id": "block_promotion",
                "priority": "high",
                "reason": "At least one critical rule failed; do not promote this agent behavior without repair or review.",
            }
        )
    if outcome.get("task_completion_status") == "incomplete":
        actions.append(
            {
                "id": "fix_task_completion",
                "priority": "high",
                "reason": "The task-completion contract is configured but did not pass with observable evidence.",
            }
        )
    if trace_signal.get("event_count") == 0 or not trace_signal.get("has_tool_or_api_events"):
        actions.append(
            {
                "id": "improve_observability",
                "priority": "medium",
                "reason": "Trace has weak tool/API signal, which limits scorecard and training usefulness.",
            }
        )
    assertions = scenario.get("assertions") if isinstance(scenario.get("assertions"), dict) else {}
    if assertions.get("required_state_transitions") and not state_changes.get("available"):
        actions.append(
            {
                "id": "capture_state_diff",
                "priority": "medium",
                "reason": "Scenario checks state transitions but no state_diff artifact is available for reviewers or trainers.",
            }
        )
    if outcome.get("passed") is True and outcome.get("task_completion_status") == "complete" and state_changes.get("changed"):
        actions.append(
            {
                "id": "stateful_success_reward",
                "priority": "low",
                "reason": "Run completed the task and changed observable state, making it a strong reward-model candidate.",
            }
        )
    return actions


def _task_completion(scorecard: dict[str, Any]) -> dict[str, Any]:
    value = scorecard.get("task_completion")
    if isinstance(value, dict):
        return value
    return {
        "status": "not_applicable",
        "passed": True,
        "task_evidence_configured": False,
        "required_check_count": 0,
        "passed_check_count": 0,
        "failed_check_count": 0,
        "blocking_rule_ids": [],
        "checks": [],
        "summary": "No task-completion evidence assertions were configured.",
    }


def _rules(scorecard: dict[str, Any]) -> list[dict[str, Any]]:
    return [rule for rule in scorecard.get("rules", []) if isinstance(rule, dict)]


def _failed_rules(scorecard: dict[str, Any]) -> list[dict[str, Any]]:
    return [rule for rule in _rules(scorecard) if rule.get("passed") is False]


def _rule_evidence_ref_count(rules: list[dict[str, Any]]) -> int:
    count = 0
    for rule in rules:
        refs = rule.get("evidence_refs")
        if isinstance(refs, list):
            count += len(refs)
    return count


def _secret_patterns(scenario: dict[str, Any]) -> list[str]:
    policy = scenario.get("policy") if isinstance(scenario.get("policy"), dict) else {}
    patterns = policy.get("secret_patterns")
    return [str(pattern) for pattern in patterns] if isinstance(patterns, list) else []


def _api_call_count(trace: dict[str, Any], events: list[dict[str, Any]]) -> int:
    metadata = trace.get("metadata") if isinstance(trace.get("metadata"), dict) else {}
    api_calls = metadata.get("api_calls")
    if isinstance(api_calls, int) and not isinstance(api_calls, bool) and api_calls >= 0:
        return api_calls
    return sum(1 for event in events if event.get("type") == "api_call")


def _max_subagent_depth(events: list[dict[str, Any]]) -> int:
    parent_by_session: dict[str, str | None] = {}
    for event in events:
        if event.get("type") != "subagent_start":
            continue
        session_id = event.get("session_id")
        if isinstance(session_id, str) and session_id:
            parent = event.get("parent_session_id")
            parent_by_session[session_id] = parent if isinstance(parent, str) and parent else None
    max_depth = 0
    for session_id in parent_by_session:
        seen: set[str] = set()
        depth = 1
        parent = parent_by_session.get(session_id)
        while parent and parent not in seen:
            seen.add(parent)
            if parent in parent_by_session:
                depth += 1
                parent = parent_by_session[parent]
            else:
                break
        max_depth = max(max_depth, depth)
    return max_depth


def _task_family(scenario_id: str) -> str:
    family = FAMILY_SUFFIX_RE.sub("", scenario_id)
    return family or scenario_id


def _int_between(value: Any, minimum: int, maximum: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return max(minimum, min(maximum, value))
    return minimum


def _non_negative_int(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return 0


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _redact_jsonable(value: Any, secret_patterns: list[str]) -> Any:
    if isinstance(value, str):
        return redact_text(value, secret_patterns)
    if isinstance(value, list):
        return [_redact_jsonable(item, secret_patterns) for item in value]
    if isinstance(value, dict):
        return {str(key): _redact_jsonable(item, secret_patterns) for key, item in value.items()}
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return redact_text(json.dumps(value, sort_keys=True, ensure_ascii=False), secret_patterns)


def _md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
