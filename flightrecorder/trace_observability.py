"""Trace observability summaries for improvement-loop readiness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TRACE_OBSERVABILITY_SCHEMA_VERSION = "hfr.trace_observability.v1"


class TraceObservabilityError(ValueError):
    """Raised when trace observability cannot be summarized."""


def build_trace_observability(
    runs_dir: str | Path,
    *,
    preserve_paths: bool = False,
    min_average_events: float | None = None,
    min_event_type_count: int | None = None,
    min_tool_or_api_run_rate: float | None = None,
    max_empty_final_answers: int | None = None,
    require_event_types: list[str] | None = None,
) -> dict[str, Any]:
    """Summarize whether completed runs contain enough trace signal to learn from."""
    root = Path(runs_dir)
    records = _load_run_records(root)
    runs = [_run_row(record, preserve_paths) for record in records]
    metrics = _metrics(runs)
    checks = _checks(
        metrics,
        min_average_events=min_average_events,
        min_event_type_count=min_event_type_count,
        min_tool_or_api_run_rate=min_tool_or_api_run_rate,
        max_empty_final_answers=max_empty_final_answers,
        require_event_types=require_event_types or [],
    )
    warnings = _warnings(runs)
    return {
        "schema_version": TRACE_OBSERVABILITY_SCHEMA_VERSION,
        "runs_dir": _display_path(root, preserve_paths),
        "passed": all(check["passed"] for check in checks),
        "check_count": len(checks),
        "failed_check_count": sum(1 for check in checks if not check["passed"]),
        "checks": checks,
        "metrics": metrics,
        "warnings": warnings,
        "runs": runs,
    }


def _load_run_records(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        raise TraceObservabilityError(f"Runs directory not found: {root}")
    if not root.is_dir():
        raise TraceObservabilityError(f"Runs path is not a directory: {root}")

    records: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        trace_path = run_dir / "normalized_trace.json"
        score_path = run_dir / "scorecard.json"
        if not trace_path.exists() and not score_path.exists():
            continue
        trace = _read_object(trace_path) if trace_path.exists() else None
        scorecard = _read_object(score_path) if score_path.exists() else None
        if trace is not None and not isinstance(trace, dict):
            raise TraceObservabilityError(f"Run {run_dir} normalized_trace.json must contain a JSON object")
        if scorecard is not None and not isinstance(scorecard, dict):
            raise TraceObservabilityError(f"Run {run_dir} scorecard.json must contain a JSON object")
        records.append({"run_dir": run_dir, "trace": trace, "scorecard": scorecard})

    if not records:
        raise TraceObservabilityError(f"No completed Flight Recorder runs found in {root}")
    return records


def _run_row(record: dict[str, Any], preserve_paths: bool) -> dict[str, Any]:
    trace = record.get("trace") if isinstance(record.get("trace"), dict) else {}
    scorecard = record.get("scorecard") if isinstance(record.get("scorecard"), dict) else {}
    session = trace.get("session") if isinstance(trace.get("session"), dict) else {}
    events = trace.get("events") if isinstance(trace.get("events"), list) else []
    event_type_counts = _count_rows(_count_values(_event_type(event) for event in events if isinstance(event, dict)))
    event_types = [row["id"] for row in event_type_counts]
    final_answer = trace.get("final_answer") if isinstance(trace.get("final_answer"), str) else ""
    tool_call_count = _event_type_count(events, "tool_call")
    tool_result_count = _event_type_count(events, "tool_result")
    api_call_count = _event_type_count(events, "api_call")
    subagent_event_count = sum(1 for event in events if isinstance(event, dict) and str(event.get("type") or "").startswith("subagent_"))
    approval_event_count = _event_type_count(events, "approval")
    return {
        "run_dir": _display_path(record["run_dir"], preserve_paths),
        "scenario_id": str(scorecard.get("scenario_id") or record["run_dir"].name),
        "passed": scorecard.get("passed") if isinstance(scorecard.get("passed"), bool) else None,
        "score": _score(scorecard),
        "source_format": str(session.get("source_format") or "unknown"),
        "model": str(session.get("model") or "unknown"),
        "event_count": len(events),
        "event_type_count": len(event_type_counts),
        "event_types": event_type_counts,
        "has_final_answer": bool(final_answer.strip()),
        "final_answer_chars": len(final_answer),
        "tool_call_count": tool_call_count,
        "tool_result_count": tool_result_count,
        "api_call_count": api_call_count,
        "subagent_event_count": subagent_event_count,
        "approval_event_count": approval_event_count,
        "has_tool_or_api_events": (tool_call_count + tool_result_count + api_call_count) > 0,
        "risks": _run_risks(len(events), final_answer, tool_call_count, tool_result_count, api_call_count),
    }


def _metrics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    event_counts = [_non_negative_int(run.get("event_count")) for run in runs]
    event_type_counts: dict[str, int] = {}
    source_format_counts: dict[str, int] = {}
    model_counts: dict[str, int] = {}
    risk_counts: dict[str, int] = {}
    for run in runs:
        _merge_count_rows(event_type_counts, run.get("event_types"))
        source_format_counts[str(run.get("source_format") or "unknown")] = source_format_counts.get(str(run.get("source_format") or "unknown"), 0) + 1
        model_counts[str(run.get("model") or "unknown")] = model_counts.get(str(run.get("model") or "unknown"), 0) + 1
        for risk in run.get("risks", []):
            risk_counts[str(risk)] = risk_counts.get(str(risk), 0) + 1

    run_count = len(runs)
    runs_with_final = sum(1 for run in runs if run.get("has_final_answer") is True)
    runs_with_tool_or_api = sum(1 for run in runs if run.get("has_tool_or_api_events") is True)
    return {
        "run_count": run_count,
        "total_event_count": sum(event_counts),
        "average_event_count": round(sum(event_counts) / run_count, 2) if run_count else 0.0,
        "min_event_count": min(event_counts) if event_counts else 0,
        "max_event_count": max(event_counts) if event_counts else 0,
        "event_type_count": len(event_type_counts),
        "event_type_counts": _count_rows(event_type_counts),
        "source_format_counts": _count_rows(source_format_counts),
        "model_counts": _count_rows(model_counts),
        "runs_with_final_answer": runs_with_final,
        "empty_final_answer_count": run_count - runs_with_final,
        "final_answer_rate": _rate(runs_with_final, run_count),
        "runs_with_tool_or_api_events": runs_with_tool_or_api,
        "tool_or_api_run_rate": _rate(runs_with_tool_or_api, run_count),
        "tool_call_count": sum(_non_negative_int(run.get("tool_call_count")) for run in runs),
        "tool_result_count": sum(_non_negative_int(run.get("tool_result_count")) for run in runs),
        "api_call_count": sum(_non_negative_int(run.get("api_call_count")) for run in runs),
        "subagent_event_count": sum(_non_negative_int(run.get("subagent_event_count")) for run in runs),
        "approval_event_count": sum(_non_negative_int(run.get("approval_event_count")) for run in runs),
        "risk_counts": _count_rows(risk_counts),
    }


def _checks(
    metrics: dict[str, Any],
    *,
    min_average_events: float | None,
    min_event_type_count: int | None,
    min_tool_or_api_run_rate: float | None,
    max_empty_final_answers: int | None,
    require_event_types: list[str],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    if min_average_events is not None:
        _add_min_check(checks, "min_average_events", metrics["average_event_count"], min_average_events)
    if min_event_type_count is not None:
        _add_min_check(checks, "min_event_type_count", metrics["event_type_count"], min_event_type_count)
    if min_tool_or_api_run_rate is not None:
        _add_min_check(checks, "min_tool_or_api_run_rate", metrics["tool_or_api_run_rate"], min_tool_or_api_run_rate)
    if max_empty_final_answers is not None:
        _add_max_check(checks, "max_empty_final_answers", metrics["empty_final_answer_count"], max_empty_final_answers)
    event_counts = {row["id"]: row["count"] for row in metrics["event_type_counts"]}
    for event_type in require_event_types:
        actual = event_counts.get(event_type, 0)
        checks.append(
            {
                "id": f"require_event_type:{event_type}",
                "passed": actual > 0,
                "actual": actual,
                "expected": {"min": 1},
                "summary": f"require_event_type:{event_type}: actual={actual} min=1",
            }
        )
    return checks


def _warnings(runs: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    for run in runs:
        risks = run.get("risks", [])
        if risks:
            warnings.append(f"{run['scenario_id']} trace risks: " + ", ".join(str(risk) for risk in risks))
    return warnings


def _run_risks(event_count: int, final_answer: str, tool_call_count: int, tool_result_count: int, api_call_count: int) -> list[str]:
    risks: list[str] = []
    if event_count == 0:
        risks.append("empty_trace")
    if not final_answer.strip():
        risks.append("missing_final_answer")
    if tool_call_count == 0 and tool_result_count == 0 and api_call_count == 0:
        risks.append("no_tool_or_api_events")
    if tool_call_count != tool_result_count and tool_call_count > 0:
        risks.append("tool_call_result_imbalance")
    return risks


def _add_min_check(checks: list[dict[str, Any]], check_id: str, actual: float | int, minimum: float | int) -> None:
    checks.append(
        {
            "id": check_id,
            "passed": actual >= minimum,
            "actual": actual,
            "expected": {"min": minimum},
            "summary": f"{check_id}: actual={actual} min={minimum}",
        }
    )


def _add_max_check(checks: list[dict[str, Any]], check_id: str, actual: int, maximum: int) -> None:
    checks.append(
        {
            "id": check_id,
            "passed": actual <= maximum,
            "actual": actual,
            "expected": {"max": maximum},
            "summary": f"{check_id}: actual={actual} max={maximum}",
        }
    )


def _event_type_count(events: list[Any], event_type: str) -> int:
    return sum(1 for event in events if isinstance(event, dict) and _event_type(event) == event_type)


def _event_type(event: dict[str, Any]) -> str:
    value = event.get("type")
    return str(value) if value else "unknown"


def _merge_count_rows(counts: dict[str, int], rows: Any) -> None:
    if not isinstance(rows, list):
        return
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("id"), str) and isinstance(row.get("count"), int):
            counts[row["id"]] = counts.get(row["id"], 0) + row["count"]


def _count_values(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


def _count_rows(counts: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {"id": key, "count": count}
        for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _score(scorecard: dict[str, Any]) -> int | None:
    if "score" not in scorecard:
        return None
    try:
        return max(0, min(100, int(scorecard.get("score"))))
    except (TypeError, ValueError):
        return None


def _non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _read_object(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _display_path(path: Path, preserve_paths: bool = False) -> str:
    raw = str(path)
    if preserve_paths:
        return raw
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    try:
        return str(resolved.relative_to(cwd))
    except ValueError:
        return f"<redacted:{resolved.name}>"
