"""Longitudinal ledgers over evidence-bundle next actions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .bundle import EVIDENCE_BUNDLE_SCHEMA_VERSION

ACTION_LEDGER_SCHEMA_VERSION = "hfr.action_ledger.v1"
PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


class ActionLedgerError(ValueError):
    """Raised when an action ledger cannot be produced."""


def build_action_ledger(
    bundle_paths: list[str | Path],
    *,
    out_path: str | Path | None = None,
    preserve_paths: bool = False,
) -> dict[str, Any]:
    """Build a deterministic ledger of advisory next actions across bundles."""
    if not bundle_paths:
        raise ActionLedgerError("At least one --bundle path is required.")

    bundle_records: list[dict[str, Any]] = []
    grouped: dict[str, dict[str, Any]] = {}
    latest_index = len(bundle_paths) - 1
    for index, raw_path in enumerate(bundle_paths):
        path = Path(raw_path)
        bundle = _read_bundle(path)
        record = _bundle_record(path, bundle, index, preserve_paths)
        bundle_records.append(record)
        for action in _bundle_actions(bundle, index, record["path"]):
            entry = grouped.setdefault(action["routing_key"], _ledger_entry(action))
            entry["occurrences"].append(_occurrence(action, index, record["path"]))

    entries = [_finalize_entry(entry, latest_index) for entry in grouped.values()]
    entries.sort(key=_entry_sort_key)
    metrics = _metrics(entries, bundle_records)
    return {
        "schema_version": ACTION_LEDGER_SCHEMA_VERSION,
        "ledger_path": _display_path(Path(out_path), preserve_paths) if out_path is not None else "",
        "passed": True,
        "bundle_count": len(bundle_records),
        "action_count": sum(record["action_count"] for record in bundle_records),
        "unique_action_count": len(entries),
        "bundles": bundle_records,
        "metrics": metrics,
        "entries": entries,
        "notes": [
            "Action ledgers summarize evidence-bundle next_actions across iterations; they do not execute repairs.",
            "Status is computed relative to the latest bundle: new and recurring actions are open, resolved actions are absent from the latest bundle.",
        ],
    }


def _read_bundle(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ActionLedgerError(f"Evidence bundle must contain a JSON object: {path}")
    if payload.get("schema_version") != EVIDENCE_BUNDLE_SCHEMA_VERSION:
        raise ActionLedgerError(f"Evidence bundle has unsupported schema_version at {path}: {payload.get('schema_version')!r}")
    return payload


def _bundle_record(path: Path, bundle: dict[str, Any], index: int, preserve_paths: bool) -> dict[str, Any]:
    decision = bundle.get("decision") if isinstance(bundle.get("decision"), dict) else {}
    actions = decision.get("next_actions") if isinstance(decision.get("next_actions"), list) else []
    record: dict[str, Any] = {
        "index": index,
        "path": _display_path(path, preserve_paths),
        "exists": path.exists(),
        "schema_version": bundle.get("schema_version"),
        "passed": bundle.get("passed") is True,
        "readiness": str(bundle.get("readiness") or ""),
        "recommendation": str(decision.get("recommendation") or ""),
        "action_count": len([action for action in actions if isinstance(action, dict)]),
    }
    if path.exists() and path.is_file():
        record["size_bytes"] = path.stat().st_size
        record["sha256"] = _sha256(path)
    return record


def _bundle_actions(bundle: dict[str, Any], bundle_index: int, bundle_path: str) -> list[dict[str, Any]]:
    decision = bundle.get("decision") if isinstance(bundle.get("decision"), dict) else {}
    actions = decision.get("next_actions") if isinstance(decision.get("next_actions"), list) else []
    rows: list[dict[str, Any]] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        normalized = _normalized_action(action)
        normalized["bundle_index"] = bundle_index
        normalized["bundle_path"] = bundle_path
        rows.append(normalized)
    return rows


def _normalized_action(action: dict[str, Any]) -> dict[str, Any]:
    evidence = action.get("evidence") if isinstance(action.get("evidence"), dict) else {}
    action_id = str(action.get("id") or "unknown_action")
    priority = str(action.get("priority") or "medium")
    artifact = str(action.get("artifact") or "unknown_artifact")
    fingerprint = action.get("action_fingerprint")
    if not _is_sha256(fingerprint):
        fingerprint = _action_fingerprint(action_id, priority, artifact, evidence)
    routing_key = action.get("routing_key")
    if not isinstance(routing_key, str) or not routing_key:
        routing_key = f"{artifact}:{action_id}:{fingerprint[:12]}"
    return {
        "id": action_id,
        "priority": priority,
        "artifact": artifact,
        "summary": str(action.get("summary") or ""),
        "routing_key": routing_key,
        "action_fingerprint": fingerprint,
        "evidence": evidence,
    }


def _ledger_entry(action: dict[str, Any]) -> dict[str, Any]:
    return {
        "routing_key": action["routing_key"],
        "action_fingerprint": action["action_fingerprint"],
        "id": action["id"],
        "priority": action["priority"],
        "artifact": action["artifact"],
        "summary": action["summary"],
        "evidence": action["evidence"],
        "occurrences": [],
    }


def _occurrence(action: dict[str, Any], bundle_index: int, bundle_path: str) -> dict[str, Any]:
    return {
        "bundle_index": bundle_index,
        "bundle_path": bundle_path,
        "summary": action["summary"],
        "priority": action["priority"],
        "artifact": action["artifact"],
    }


def _finalize_entry(entry: dict[str, Any], latest_index: int) -> dict[str, Any]:
    occurrences = entry["occurrences"]
    bundle_indexes = sorted({occurrence["bundle_index"] for occurrence in occurrences})
    first_seen = bundle_indexes[0]
    last_seen = bundle_indexes[-1]
    open_in_latest = latest_index in bundle_indexes
    if open_in_latest and first_seen == latest_index:
        status = "new"
    elif open_in_latest and len(bundle_indexes) > 1:
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
        "bundle_indexes": bundle_indexes,
        "first_seen_index": first_seen,
        "last_seen_index": last_seen,
        "first_seen_path": occurrences[0]["bundle_path"],
        "last_seen_path": occurrences[-1]["bundle_path"],
    }


def _metrics(entries: list[dict[str, Any]], bundles: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "bundle_count": len(bundles),
        "action_count": sum(entry["occurrence_count"] for entry in entries),
        "unique_action_count": len(entries),
        "open_action_count": sum(1 for entry in entries if entry["open"]),
        "new_action_count": sum(1 for entry in entries if entry["status"] == "new"),
        "recurring_action_count": sum(1 for entry in entries if entry["status"] == "recurring"),
        "resolved_action_count": sum(1 for entry in entries if entry["status"] == "resolved"),
        "status_counts": _count_rows(entry["status"] for entry in entries),
        "priority_counts": _count_rows(entry["priority"] for entry in entries),
        "artifact_counts": _count_rows(entry["artifact"] for entry in entries),
        "bundle_action_counts": [{"index": bundle["index"], "path": bundle["path"], "action_count": bundle["action_count"]} for bundle in bundles],
    }


def _entry_sort_key(entry: dict[str, Any]) -> tuple[int, int, str, str]:
    status_order = {"recurring": 0, "new": 1, "open": 2, "resolved": 3}
    return (
        status_order.get(str(entry.get("status")), 99),
        PRIORITY_ORDER.get(str(entry.get("priority")), 99),
        str(entry.get("artifact") or ""),
        str(entry.get("routing_key") or ""),
    )


def _count_rows(values: Any) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return [{"id": key, "count": counts[key]} for key in sorted(counts)]


def _action_fingerprint(action_id: str, priority: str, artifact: str, evidence: dict[str, Any]) -> str:
    payload = {
        "id": action_id,
        "priority": priority,
        "artifact": artifact,
        "evidence": evidence,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
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


def _is_windows_absolute(value: str) -> bool:
    normalized = value.replace("/", "\\")
    return (len(normalized) >= 3 and normalized[1:3] == ":\\" and normalized[0].isalpha()) or normalized.startswith("\\\\")


def _basename(value: str) -> str:
    return value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] or "path"
