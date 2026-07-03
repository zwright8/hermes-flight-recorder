"""Repair-queue exports for failed Flight Recorder scorecard rules."""

from __future__ import annotations

import json
import hashlib
import os
import re
from pathlib import Path
from typing import Any

from .training import RunRecord, TrainingExportError, load_run_records

REPAIR_QUEUE_SCHEMA_VERSION = "hfr.repair_queue.v1"
REPAIR_ITEM_SCHEMA_VERSION = "hfr.repair_item.v1"
FAMILY_SUFFIX_RE = re.compile(r"([_-](good|bad|pass|fail|passing|failing|chosen|rejected))+$", re.IGNORECASE)


class RepairQueueError(ValueError):
    """Raised when a repair queue cannot be produced."""


def build_repair_queue(
    runs_dir: str | Path,
    *,
    preserve_paths: bool = False,
    only_critical: bool = False,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic repair queue from failed scorecard rules."""
    source = Path(runs_dir)
    repair_queue_path = Path(output_path) if output_path is not None else None
    try:
        records = load_run_records(source)
    except TrainingExportError as exc:
        raise RepairQueueError(str(exc)) from exc

    items = [
        item
        for record in records
        for item in _repair_items(record, preserve_paths, repair_queue_path)
        if not only_critical or item["critical"] is True
    ]
    metrics = _metrics(items)
    return {
        "schema_version": REPAIR_QUEUE_SCHEMA_VERSION,
        "runs_dir": _display_path(source, preserve_paths),
        "only_critical": only_critical,
        "passed": True,
        "item_count": len(items),
        "metrics": metrics,
        "items": items,
        "notes": [
            "Repair queues are derived from failed scorecard rules; they do not rescore traces or approve model updates.",
            "Use repair_item_id, rule_id, source_artifacts, and replay.command to route deterministic repair work.",
        ],
    }


def _repair_items(record: RunRecord, preserve_paths: bool, output_path: Path | None) -> list[dict[str, Any]]:
    scorecard = record.scorecard
    scenario_id = str(scorecard.get("scenario_id") or record.run_id)
    task_completion = scorecard.get("task_completion") if isinstance(scorecard.get("task_completion"), dict) else {}
    items: list[dict[str, Any]] = []
    for rule in scorecard.get("rules", []):
        if not isinstance(rule, dict) or rule.get("passed") is True:
            continue
        rule_id = str(rule.get("id") or "unknown_rule")
        critical = bool(rule.get("critical"))
        items.append(
            {
                "schema_version": REPAIR_ITEM_SCHEMA_VERSION,
                "repair_item_id": f"{record.run_id}:{rule_id}",
                "run_id": record.run_id,
                "scenario_id": scenario_id,
                "scenario_title": str(scorecard.get("scenario_title") or scenario_id),
                "task_family": _task_family(scenario_id),
                "priority": "critical" if critical else "high",
                "rule_id": rule_id,
                "rule_name": str(rule.get("name") or rule_id),
                "critical": critical,
                "penalty": _non_negative_int(rule.get("penalty")),
                "score": _score(scorecard),
                "pass_threshold": scorecard.get("pass_threshold"),
                "task_completion_status": str(task_completion.get("status") or "not_applicable"),
                "task_completion_passed": bool(task_completion.get("passed", True)),
                "summary": _summary(rule),
                "suggested_action": _suggested_action(rule_id),
                "evidence": _string_list(rule.get("evidence")),
                "evidence_refs": _evidence_refs(rule),
                "evidence_snippets": _evidence_snippets(record.trace, _evidence_refs(rule)),
                "source_artifacts": _source_artifacts(record, preserve_paths),
                "source_artifact_fingerprints": _source_artifact_fingerprints(record, output_path),
                "replay": _replay(record.lineage),
            }
        )
    return items


def _metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    scenario_ids = sorted({str(item["scenario_id"]) for item in items})
    task_families = sorted({str(item["task_family"]) for item in items})
    return {
        "item_count": len(items),
        "critical_item_count": sum(1 for item in items if item.get("critical") is True),
        "scenario_count": len(scenario_ids),
        "task_family_count": len(task_families),
        "scenarios": scenario_ids,
        "task_families": task_families,
        "priority_counts": _count_rows(str(item.get("priority") or "unknown") for item in items),
        "rule_counts": _count_rows(str(item.get("rule_id") or "unknown") for item in items),
        "critical_rule_counts": _count_rows(str(item.get("rule_id") or "unknown") for item in items if item.get("critical") is True),
        "task_completion_status_counts": _count_rows(str(item.get("task_completion_status") or "unknown") for item in items),
    }


def _source_artifacts(record: RunRecord, preserve_paths: bool) -> dict[str, str]:
    artifacts = {
        "run_dir": record.run_dir,
        "normalized_trace": record.run_dir / "normalized_trace.json",
        "scorecard": record.run_dir / "scorecard.json",
        "report": record.run_dir / "report.html",
    }
    if record.lineage_path is not None:
        artifacts["lineage"] = record.lineage_path
    regression = record.run_dir / "regression_scenario.json"
    if regression.exists():
        artifacts["regression_scenario"] = regression
    return {name: _display_path(path, preserve_paths) for name, path in artifacts.items()}


def _source_artifact_fingerprints(record: RunRecord, output_path: Path | None) -> dict[str, dict[str, Any]]:
    artifacts = {
        "normalized_trace": record.run_dir / "normalized_trace.json",
        "scorecard": record.run_dir / "scorecard.json",
        "report": record.run_dir / "report.html",
    }
    if record.lineage_path is not None:
        artifacts["lineage"] = record.lineage_path
    regression = record.run_dir / "regression_scenario.json"
    if regression.exists():
        artifacts["regression_scenario"] = regression
    return {name: _file_ref(path, output_path) for name, path in artifacts.items()}


def _file_ref(path: Path, output_path: Path | None) -> dict[str, Any]:
    exists = path.exists() and path.is_file()
    return {
        "path": _display_path_for_output_source(path, output_path),
        "exists": exists,
        "kind": "file",
        "sha256": _sha256(path) if exists else None,
        "size_bytes": path.stat().st_size if exists else None,
    }


def _replay(lineage: dict[str, Any] | None) -> dict[str, Any]:
    replay = lineage.get("replay") if isinstance(lineage, dict) and isinstance(lineage.get("replay"), dict) else {}
    return {
        "available": bool(replay),
        "self_contained": replay.get("self_contained") if isinstance(replay.get("self_contained"), bool) else None,
        "command": str(replay.get("command") or ""),
        "argv": replay.get("argv") if isinstance(replay.get("argv"), list) else [],
    }


def _summary(rule: dict[str, Any]) -> str:
    evidence = _string_list(rule.get("evidence"))
    if evidence:
        return evidence[0]
    return f"{rule.get('id') or 'rule'} failed."


def _suggested_action(rule_id: str) -> str:
    actions = {
        "forbidden_actions": "Remove forbidden tool, command, URL, or path behavior from the agent trajectory.",
        "secret_exposure": "Prevent secret-like values from appearing in tool outputs or final answers.",
        "budget": "Tighten delegation, tool-call, depth, or API-call budgeting for the scenario.",
        "required_evidence": "Add or restore observable trace evidence required by the scenario contract.",
        "required_actions": "Make the required action appear in observable tool-result evidence with the expected status.",
        "required_action_sequences": "Preserve the required action ordering in the observed trace.",
        "required_event_counts": "Adjust behavior so observed event counts satisfy the scenario contract.",
        "required_state": "Capture or repair post-run state evidence required by the scenario contract.",
        "final_answer": "Align the final answer with the scenario assertions and observed evidence.",
    }
    return actions.get(rule_id, "Inspect the failed rule evidence and repair the agent behavior or scenario contract.")


def _evidence_refs(rule: dict[str, Any]) -> list[dict[str, Any]]:
    refs = rule.get("evidence_refs")
    if not isinstance(refs, list):
        return []
    return [ref for ref in refs if isinstance(ref, dict)]


def _evidence_snippets(trace: dict[str, Any], refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    snippets: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    events = trace.get("events") if isinstance(trace.get("events"), list) else []
    for ref in refs:
        target = ref.get("target")
        event_index = ref.get("event_index")
        key = (target, event_index, ref.get("reason"))
        if key in seen:
            continue
        seen.add(key)
        if target == "event" and isinstance(event_index, int) and not isinstance(event_index, bool):
            event = events[event_index] if 0 <= event_index < len(events) and isinstance(events[event_index], dict) else {}
            snippets.append(_event_snippet(event_index, event, ref))
        elif target == "final_answer":
            snippets.append(_final_answer_snippet(trace, ref))
        elif target == "state_snapshot":
            snippets.append(_state_snapshot_snippet(ref))
        else:
            snippets.append(_episode_snippet(trace, ref))
    return snippets


def _event_snippet(event_index: int, event: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "type": event.get("type"),
        "tool_name": event.get("tool_name"),
        "status": event.get("status"),
        "args": event.get("args"),
        "text": event.get("text"),
    }
    return {
        "target": "event",
        "event_index": event_index,
        "event_type": str(event.get("type") or "unknown"),
        "tool_name": str(event.get("tool_name") or ""),
        "status": str(event.get("status") or ""),
        "reason": str(ref.get("reason") or ""),
        "text": _truncate(_json_text(payload)),
    }


def _final_answer_snippet(trace: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    return {
        "target": "final_answer",
        "reason": str(ref.get("reason") or ""),
        "text": _truncate(str(trace.get("final_answer") or "")),
    }


def _state_snapshot_snippet(ref: dict[str, Any]) -> dict[str, Any]:
    return {
        "target": "state_snapshot",
        "reason": str(ref.get("reason") or ""),
        "text": "State snapshot evidence is referenced; inspect source_artifacts and evidence_refs for the exact check.",
    }


def _episode_snippet(trace: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    events = trace.get("events") if isinstance(trace.get("events"), list) else []
    final_answer = str(trace.get("final_answer") or "")
    return {
        "target": "episode",
        "reason": str(ref.get("reason") or ""),
        "text": f"episode_summary: events={len(events)}, final_answer_chars={len(final_answer)}",
    }


def _json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _truncate(value: str, limit: int = 600) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 15] + "...[truncated]"


def _count_rows(values: Any) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return [{"id": key, "count": counts[key]} for key in sorted(counts)]


def _task_family(scenario_id: str) -> str:
    return FAMILY_SUFFIX_RE.sub("", scenario_id)


def _score(scorecard: dict[str, Any]) -> int:
    value = scorecard.get("score")
    if isinstance(value, int) and not isinstance(value, bool):
        return max(0, min(100, value))
    return 0


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int) and value >= 0:
        return value
    return 0


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _display_path(path: Path, preserve_paths: bool = False) -> str:
    if preserve_paths:
        return str(path)
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return path.name


def _display_path_for_output_source(path: Path, output_path: Path | None) -> str:
    if output_path is not None:
        try:
            source = path if path.is_absolute() else Path.cwd() / path
            output_dir = output_path.parent if output_path.is_absolute() else Path.cwd() / output_path.parent
            return os.path.relpath(source.resolve(), output_dir.resolve())
        except (OSError, ValueError):
            return str(path)
    return _display_path(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
