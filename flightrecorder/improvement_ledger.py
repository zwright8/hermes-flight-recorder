"""Longitudinal ledgers over concrete improvement-plan work items."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .improvement_plan import IMPROVEMENT_PLAN_SCHEMA_VERSION, PRIORITY_RANK
from .path_safety import path_has_symlink_component as _path_has_symlink_component

IMPROVEMENT_LEDGER_SCHEMA_VERSION = "hfr.improvement_ledger.v1"


class ImprovementLedgerError(ValueError):
    """Raised when an improvement ledger cannot be produced."""


def build_improvement_ledger(
    plan_paths: list[str | Path],
    *,
    out_path: str | Path | None = None,
    preserve_paths: bool = False,
) -> dict[str, Any]:
    """Build a deterministic ledger of improvement-plan work items across iterations."""
    if not plan_paths:
        raise ImprovementLedgerError("At least one --plan path is required.")

    plan_records: list[dict[str, Any]] = []
    grouped: dict[str, dict[str, Any]] = {}
    latest_index = len(plan_paths) - 1
    output_root = Path(out_path).parent if out_path is not None else None
    for index, raw_path in enumerate(plan_paths):
        path = Path(raw_path)
        _reject_symlinked_plan_input(path)
        plan = _read_plan(path)
        record = _plan_record(path, plan, index, preserve_paths, output_root)
        plan_records.append(record)
        for item in _plan_work_items(plan, index, record["path"]):
            entry = grouped.setdefault(item["work_key"], _ledger_entry(item))
            entry["occurrences"].append(_occurrence(item, index, record["path"]))

    entries = [_finalize_entry(entry, latest_index) for entry in grouped.values()]
    entries.sort(key=_entry_sort_key)
    metrics = _metrics(entries, plan_records)
    return {
        "schema_version": IMPROVEMENT_LEDGER_SCHEMA_VERSION,
        "ledger_path": _display_path(Path(out_path), preserve_paths) if out_path is not None else "",
        "passed": True,
        "plan_count": len(plan_records),
        "work_item_count": sum(record["work_item_count"] for record in plan_records),
        "unique_work_item_count": len(entries),
        "decision": _decision(entries, plan_records, metrics),
        "plans": plan_records,
        "metrics": metrics,
        "entries": entries,
        "notes": [
            "Improvement ledgers summarize improvement-plan work items across iterations; they do not execute repairs.",
            "Status is computed relative to the latest plan: new and recurring work is open, resolved work is absent from the latest plan.",
            "Work items are grouped by stable scenario/rule/category keys so evidence-text changes do not hide recurring repair pressure.",
        ],
    }


def stable_work_key(item: dict[str, Any]) -> str:
    """Return the stable cross-plan grouping key for one improvement-plan work item."""
    category = str(item.get("category") or "unknown")
    scenario_id = str(item.get("scenario_id") or "")
    task_family = str(item.get("task_family") or "")
    rule_id = str(item.get("rule_id") or "")
    sources = item.get("sources") if isinstance(item.get("sources"), dict) else {}
    if category == "bundle_action":
        action = sources.get("bundle_action") if isinstance(sources.get("bundle_action"), dict) else {}
        artifact = str(action.get("artifact") or "unknown")
        action_id = str(action.get("id") or "")
        return _key("bundle_action", artifact, action_id or str(item.get("summary") or "unknown"))
    if category == "repair":
        return _key("repair", scenario_id or task_family or "unknown", rule_id or "unknown_rule")
    if category == "curriculum":
        return _key("curriculum", task_family or "unknown", rule_id or "unknown_rule")
    if category == "digest_action":
        action_ids = sources.get("run_digest", {}).get("recommended_action_ids") if isinstance(sources.get("run_digest"), dict) else []
        action_key = ",".join(str(action_id) for action_id in action_ids if isinstance(action_id, str)) or "review"
        return _key("digest_action", scenario_id or task_family or "unknown", action_key)
    return _key(category, scenario_id or task_family or "unknown", rule_id or str(item.get("summary") or "unknown"))


def _read_plan(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ImprovementLedgerError(f"Improvement plan must contain a JSON object: {path}")
    if payload.get("schema_version") != IMPROVEMENT_PLAN_SCHEMA_VERSION:
        raise ImprovementLedgerError(f"Improvement plan has unsupported schema_version at {path}: {payload.get('schema_version')!r}")
    return payload


def _plan_record(path: Path, plan: dict[str, Any], index: int, preserve_paths: bool, output_root: Path | None) -> dict[str, Any]:
    decision = plan.get("decision") if isinstance(plan.get("decision"), dict) else {}
    record: dict[str, Any] = {
        "index": index,
        "path": _display_plan_path(path, output_root, preserve_paths),
        "exists": path.exists(),
        "schema_version": plan.get("schema_version"),
        "passed": plan.get("passed") is True,
        "readiness": str(plan.get("readiness") or ""),
        "recommendation": str(decision.get("recommendation") or ""),
        "work_item_count": _non_negative_int(plan.get("work_item_count")),
        "critical_or_high_count": _non_negative_int(decision.get("critical_or_high_count")),
    }
    if path.exists() and path.is_file() and not path.is_symlink() and not _path_has_symlink_component(path, include_leaf=False):
        record["size_bytes"] = path.stat().st_size
        record["sha256"] = _sha256(path)
    return record


def _reject_symlinked_plan_input(path: Path) -> None:
    if path.is_symlink() or _path_has_symlink_component(path, include_leaf=False):
        raise ImprovementLedgerError(f"improvement_ledger.plan_path must not traverse symlinked components: {path}")


def _plan_work_items(plan: dict[str, Any], plan_index: int, plan_path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in plan.get("work_items", []) if isinstance(plan.get("work_items"), list) else []:
        if not isinstance(item, dict):
            continue
        normalized = _normalized_item(item)
        normalized["plan_index"] = plan_index
        normalized["plan_path"] = plan_path
        rows.append(normalized)
    return rows


def _normalized_item(item: dict[str, Any]) -> dict[str, Any]:
    category = str(item.get("category") or "unknown")
    priority = str(item.get("priority") or "medium")
    fingerprint = item.get("fingerprint")
    if not _is_sha256(fingerprint):
        fingerprint = _fingerprint(item)
    work_key = stable_work_key(item)
    return {
        "work_key": work_key,
        "item_id": str(item.get("item_id") or f"{category}:{fingerprint[:16]}"),
        "routing_key": str(item.get("routing_key") or f"{category}:{priority}:{fingerprint[:12]}"),
        "fingerprint": fingerprint,
        "category": category,
        "priority": priority,
        "summary": str(item.get("summary") or ""),
        "suggested_action": str(item.get("suggested_action") or ""),
        "scenario_id": item.get("scenario_id") if isinstance(item.get("scenario_id"), str) else None,
        "task_family": item.get("task_family") if isinstance(item.get("task_family"), str) else None,
        "rule_id": item.get("rule_id") if isinstance(item.get("rule_id"), str) else None,
        "rule_name": item.get("rule_name") if isinstance(item.get("rule_name"), str) else None,
        "score": item.get("score") if isinstance(item.get("score"), int) and not isinstance(item.get("score"), bool) else None,
        "task_completion_status": item.get("task_completion_status") if isinstance(item.get("task_completion_status"), str) else None,
        "evidence_ref_count": len(item.get("evidence_refs", [])) if isinstance(item.get("evidence_refs"), list) else 0,
    }


def _ledger_entry(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "work_key": item["work_key"],
        "category": item["category"],
        "priority": item["priority"],
        "summary": item["summary"],
        "suggested_action": item["suggested_action"],
        "scenario_id": item["scenario_id"],
        "task_family": item["task_family"],
        "rule_id": item["rule_id"],
        "rule_name": item["rule_name"],
        "score": item["score"],
        "task_completion_status": item["task_completion_status"],
        "evidence_ref_count": item["evidence_ref_count"],
        "occurrences": [],
    }


def _occurrence(item: dict[str, Any], plan_index: int, plan_path: str) -> dict[str, Any]:
    return {
        "plan_index": plan_index,
        "plan_path": plan_path,
        "item_id": item["item_id"],
        "routing_key": item["routing_key"],
        "fingerprint": item["fingerprint"],
        "priority": item["priority"],
        "category": item["category"],
        "summary": item["summary"],
    }


def _finalize_entry(entry: dict[str, Any], latest_index: int) -> dict[str, Any]:
    occurrences = entry["occurrences"]
    plan_indexes = sorted({occurrence["plan_index"] for occurrence in occurrences})
    first_seen = plan_indexes[0]
    last_seen = plan_indexes[-1]
    open_in_latest = latest_index in plan_indexes
    latest_occurrence = occurrences[-1]
    if open_in_latest and first_seen == latest_index:
        status = "new"
    elif open_in_latest and len(plan_indexes) > 1:
        status = "recurring"
    elif open_in_latest:
        status = "open"
    else:
        status = "resolved"
    return {
        **entry,
        "status": status,
        "open": open_in_latest,
        "occurrence_count": len(occurrences),
        "plan_indexes": plan_indexes,
        "first_seen_index": first_seen,
        "last_seen_index": last_seen,
        "first_seen_path": occurrences[0]["plan_path"],
        "last_seen_path": latest_occurrence["plan_path"],
        "latest_item_id": latest_occurrence["item_id"],
        "latest_routing_key": latest_occurrence["routing_key"],
        "latest_fingerprint": latest_occurrence["fingerprint"],
    }


def _metrics(entries: list[dict[str, Any]], plans: list[dict[str, Any]]) -> dict[str, Any]:
    open_entries = [entry for entry in entries if entry["open"]]
    return {
        "plan_count": len(plans),
        "work_item_count": sum(entry["occurrence_count"] for entry in entries),
        "unique_work_item_count": len(entries),
        "open_work_item_count": len(open_entries),
        "new_work_item_count": sum(1 for entry in entries if entry["status"] == "new"),
        "recurring_work_item_count": sum(1 for entry in entries if entry["status"] == "recurring"),
        "resolved_work_item_count": sum(1 for entry in entries if entry["status"] == "resolved"),
        "critical_open_work_item_count": sum(1 for entry in open_entries if entry.get("priority") == "critical"),
        "high_open_work_item_count": sum(1 for entry in open_entries if entry.get("priority") == "high"),
        "status_counts": _count_rows(entry["status"] for entry in entries),
        "priority_counts": _count_rows(entry["priority"] for entry in entries),
        "open_priority_counts": _count_rows(entry["priority"] for entry in open_entries),
        "category_counts": _count_rows(entry["category"] for entry in entries),
        "open_category_counts": _count_rows(entry["category"] for entry in open_entries),
        "task_family_counts": _count_rows(entry["task_family"] for entry in open_entries if entry.get("task_family")),
        "rule_counts": _count_rows(entry["rule_id"] for entry in open_entries if entry.get("rule_id")),
        "plan_work_item_counts": [{"index": plan["index"], "path": plan["path"], "work_item_count": plan["work_item_count"]} for plan in plans],
    }


def _decision(entries: list[dict[str, Any]], plans: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    latest_plan = plans[-1] if plans else {}
    open_count = _non_negative_int(metrics.get("open_work_item_count"))
    critical_or_high = _non_negative_int(metrics.get("critical_open_work_item_count")) + _non_negative_int(
        metrics.get("high_open_work_item_count")
    )
    if latest_plan.get("readiness") == "blocked":
        recommendation = "fix_handoff"
        summary = "Latest improvement plan is blocked; fix handoff evidence before using longitudinal pressure."
    elif critical_or_high:
        recommendation = "continue_improvement"
        summary = f"{critical_or_high} critical/high work item(s) remain open in the latest plan."
    elif open_count:
        recommendation = "review_remaining_work"
        summary = f"{open_count} lower-priority work item(s) remain open in the latest plan."
    else:
        recommendation = "promote_or_monitor"
        summary = "No work items remain open in the latest plan."
    return {
        "readiness": "blocked" if latest_plan.get("readiness") == "blocked" else "ready",
        "recommendation": recommendation,
        "summary": summary,
        "latest_plan_index": latest_plan.get("index"),
        "latest_plan_recommendation": latest_plan.get("recommendation"),
        "open_work_item_count": open_count,
        "critical_or_high_open_count": critical_or_high,
        "resolved_work_item_count": _non_negative_int(metrics.get("resolved_work_item_count")),
        "top_open_work_items": [
            {
                "work_key": str(entry.get("work_key") or ""),
                "priority": str(entry.get("priority") or ""),
                "category": str(entry.get("category") or ""),
                "scenario_id": entry.get("scenario_id"),
                "rule_id": entry.get("rule_id"),
                "summary": str(entry.get("summary") or ""),
            }
            for entry in entries
            if entry.get("open")
        ][:5],
    }


def _entry_sort_key(entry: dict[str, Any]) -> tuple[int, int, str, str, str, str]:
    status_order = {"recurring": 0, "new": 1, "open": 2, "resolved": 3}
    return (
        status_order.get(str(entry.get("status")), 99),
        PRIORITY_RANK.get(str(entry.get("priority")), 99),
        str(entry.get("category") or ""),
        str(entry.get("task_family") or ""),
        str(entry.get("rule_id") or ""),
        str(entry.get("work_key") or ""),
    )


def _key(*parts: str) -> str:
    return ":".join(_key_part(part) for part in parts)


def _key_part(value: str) -> str:
    return str(value or "unknown").replace(":", "_").strip() or "unknown"


def _count_rows(values: Any) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for value in values:
        if not isinstance(value, str) or not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return [{"id": key, "count": counts[key]} for key in sorted(counts)]


def _fingerprint(item: dict[str, Any]) -> str:
    encoded = json.dumps(item, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and value == value.lower() and all(char in "0123456789abcdef" for char in value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path, preserve_paths: bool = False) -> str:
    raw = str(path)
    if preserve_paths:
        return raw
    if _is_windows_absolute(raw):
        return f"<redacted:{_basename(raw)}>"
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    try:
        return str(resolved.relative_to(cwd))
    except ValueError:
        return f"<redacted:{resolved.name}>"


def _display_plan_path(path: Path, output_root: Path | None, preserve_paths: bool = False) -> str:
    if preserve_paths or output_root is None:
        return _display_path(path, preserve_paths)
    raw = str(path)
    if _is_windows_absolute(raw):
        return f"<redacted:{_basename(raw)}>"
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(output_root.resolve()))
    except ValueError:
        return _display_path(path, preserve_paths)


def _is_windows_absolute(value: str) -> bool:
    normalized = value.replace("/", "\\")
    return (len(normalized) >= 3 and normalized[1:3] == ":\\" and normalized[0].isalpha()) or normalized.startswith("\\\\")


def _basename(value: str) -> str:
    return value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] or "path"


def _non_negative_int(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return max(0, value)
    return 0
