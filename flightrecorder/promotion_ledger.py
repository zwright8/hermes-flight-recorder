"""Longitudinal ledgers over promotion decision gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .decision_gate import DECISION_GATE_SCHEMA_VERSION

PROMOTION_LEDGER_SCHEMA_VERSION = "hfr.promotion_ledger.v1"


class PromotionLedgerError(ValueError):
    """Raised when a promotion ledger cannot be produced."""


def build_promotion_ledger(
    decision_gate_paths: list[str | Path],
    *,
    out_path: str | Path | None = None,
    preserve_paths: bool = False,
) -> dict[str, Any]:
    """Build a deterministic history of promotion decisions across iterations."""
    if not decision_gate_paths:
        raise PromotionLedgerError("At least one --decision-gate path is required.")

    records: list[dict[str, Any]] = []
    for index, raw_path in enumerate(decision_gate_paths):
        path = Path(raw_path)
        gate = _read_decision_gate(path)
        records.append(_decision_record(path, gate, index, preserve_paths))

    metrics = _metrics(records)
    return {
        "schema_version": PROMOTION_LEDGER_SCHEMA_VERSION,
        "ledger_path": _display_path(Path(out_path), preserve_paths) if out_path is not None else "",
        "passed": True,
        "decision_count": len(records),
        "records": records,
        "metrics": metrics,
        "notes": [
            "Promotion ledgers summarize decision_gate artifacts across improvement iterations; they do not launch trainers or mutate CI.",
            "Allowed means the decision gate passed and recommended allow_promotion; all other decisions are counted as blocked pressure.",
        ],
    }


def _read_decision_gate(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        raise PromotionLedgerError(f"Decision gate not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PromotionLedgerError(f"Decision gate must contain a JSON object: {path}")
    if payload.get("schema_version") != DECISION_GATE_SCHEMA_VERSION:
        raise PromotionLedgerError(f"Decision gate has unsupported schema_version at {path}: {payload.get('schema_version')!r}")
    return payload


def _decision_record(path: Path, gate: dict[str, Any], index: int, preserve_paths: bool) -> dict[str, Any]:
    source = gate.get("source_decision") if isinstance(gate.get("source_decision"), dict) else {}
    source_artifact = gate.get("source_artifact") if isinstance(gate.get("source_artifact"), dict) else {}
    record: dict[str, Any] = {
        "index": index,
        "path": _display_path(path, preserve_paths),
        "exists": path.exists(),
        "schema_version": str(gate.get("schema_version") or ""),
        "passed": gate.get("passed") is True,
        "readiness": str(gate.get("readiness") or ""),
        "recommendation": str(gate.get("recommendation") or ""),
        "expected_recommendation": str(gate.get("expected_recommendation") or ""),
        "expected_readiness": gate.get("expected_readiness") if isinstance(gate.get("expected_readiness"), str) else None,
        "require_passed": gate.get("require_passed") is True,
        "check_count": gate.get("check_count") if _is_non_negative_int(gate.get("check_count")) else 0,
        "failed_check_count": gate.get("failed_check_count") if _is_non_negative_int(gate.get("failed_check_count")) else 0,
        "source": {
            "schema_version": str(source.get("schema_version") or ""),
            "passed": source.get("passed") if isinstance(source.get("passed"), bool) else None,
            "recommendation": str(source.get("recommendation") or ""),
            "readiness": str(source.get("readiness") or ""),
            "blocking_check_count": source.get("blocking_check_count") if _is_non_negative_int(source.get("blocking_check_count")) else None,
            "artifact_path": str(source_artifact.get("path") or ""),
            "artifact_exists": source_artifact.get("exists") is True,
            "artifact_sha256": source_artifact.get("sha256") if _is_sha256(source_artifact.get("sha256")) else None,
        },
    }
    if path.exists() and path.is_file():
        record["size_bytes"] = path.stat().st_size
        record["sha256"] = _sha256(path)
    return record


def _metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    allowed_count = sum(1 for record in records if _is_allowed(record))
    blocked_count = sum(1 for record in records if _is_blocked(record))
    latest = records[-1] if records else {}
    source_artifact_keys = {
        _source_artifact_key(record)
        for record in records
        if _source_artifact_key(record)
    }
    return {
        "decision_count": len(records),
        "allowed_count": allowed_count,
        "blocked_count": blocked_count,
        "latest_recommendation": latest.get("recommendation") if records else "",
        "latest_readiness": latest.get("readiness") if records else "",
        "latest_passed": latest.get("passed") if records else None,
        "consecutive_allowed_count": _consecutive(records, _is_allowed),
        "consecutive_blocked_count": _consecutive(records, _is_blocked),
        "unique_source_artifact_count": len(source_artifact_keys),
        "recommendation_counts": _count_rows(record.get("recommendation") for record in records),
        "source_recommendation_counts": _count_rows(
            record.get("source", {}).get("recommendation") if isinstance(record.get("source"), dict) else ""
            for record in records
        ),
        "decision_gate_results": [
            {
                "index": record["index"],
                "path": record["path"],
                "passed": record["passed"],
                "recommendation": record["recommendation"],
                "source_recommendation": record["source"]["recommendation"],
                "failed_check_count": record["failed_check_count"],
            }
            for record in records
        ],
    }


def _is_allowed(record: dict[str, Any]) -> bool:
    return record.get("passed") is True and record.get("recommendation") == "allow_promotion"


def _is_blocked(record: dict[str, Any]) -> bool:
    return record.get("passed") is not True or record.get("recommendation") == "block_promotion"


def _consecutive(records: list[dict[str, Any]], predicate: Any) -> int:
    count = 0
    for record in reversed(records):
        if not predicate(record):
            break
        count += 1
    return count


def _source_artifact_key(record: dict[str, Any]) -> str:
    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    sha256 = source.get("artifact_sha256")
    if isinstance(sha256, str) and sha256:
        return sha256
    artifact_path = source.get("artifact_path")
    return artifact_path if isinstance(artifact_path, str) and artifact_path else ""


def _count_rows(values: Any) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return [{"id": key, "count": counts[key]} for key in sorted(counts)]


def _is_non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


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
