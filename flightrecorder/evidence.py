"""Suite-level evidence coverage summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EVIDENCE_COVERAGE_SCHEMA_VERSION = "hfr.evidence_coverage.v1"


class EvidenceCoverageError(ValueError):
    """Raised when evidence coverage cannot be built."""


def build_evidence_coverage(
    runs_dir: str | Path,
    *,
    preserve_paths: bool = False,
    min_failed_rule_evidence_rate: float | None = None,
    min_critical_failed_rule_evidence_rate: float | None = None,
    min_event_evidence_refs: int | None = None,
    max_failed_rules_without_evidence: int | None = None,
    max_critical_failed_rules_without_evidence: int | None = None,
    require_rule_evidence: list[str] | None = None,
) -> dict[str, Any]:
    """Summarize how well scorecard judgments are grounded in evidence refs."""
    root = Path(runs_dir)
    records = _load_run_records(root)
    run_rows: list[dict[str, Any]] = []
    rule_buckets: dict[str, dict[str, Any]] = {}
    target_counts: dict[str, int] = {}
    failed_target_counts: dict[str, int] = {}
    warnings: list[str] = []

    totals = {
        "rule_count": 0,
        "failed_rule_count": 0,
        "critical_failed_rule_count": 0,
        "evidence_ref_count": 0,
        "failed_rule_evidence_ref_count": 0,
        "critical_failed_rule_evidence_ref_count": 0,
        "failed_rules_with_evidence": 0,
        "failed_rules_without_evidence": 0,
        "critical_failed_rules_with_evidence": 0,
        "critical_failed_rules_without_evidence": 0,
        "task_evidence_ref_count": 0,
    }

    for record in records:
        row = _run_coverage(record, preserve_paths)
        run_rows.append(row)
        for key in totals:
            totals[key] += int(row.get(key, 0) or 0)
        _merge_counts(target_counts, row.get("evidence_target_counts"))
        _merge_counts(failed_target_counts, row.get("failed_rule_evidence_target_counts"))
        for rule in row["rules"]:
            _merge_rule_bucket(rule_buckets, rule)
        if row["failed_rules_without_evidence"]:
            warnings.append(
                f"{row['scenario_id']} has failed rule(s) without structured evidence refs: "
                + ", ".join(row["failed_rules_without_evidence"])
            )
        if row.get("event_count") is None:
            warnings.append(f"{row['scenario_id']} is missing normalized_trace.json; event coverage is unknown.")

    metrics = {
        "run_count": len(run_rows),
        **totals,
        "failed_rule_evidence_rate": _rate(totals["failed_rules_with_evidence"], totals["failed_rule_count"]),
        "critical_failed_rule_evidence_rate": _rate(
            totals["critical_failed_rules_with_evidence"],
            totals["critical_failed_rule_count"],
        ),
        "event_evidence_ref_count": target_counts.get("event", 0),
        "final_answer_evidence_ref_count": target_counts.get("final_answer", 0),
        "episode_evidence_ref_count": target_counts.get("episode", 0),
        "evidence_target_counts": _count_rows(target_counts),
        "failed_rule_evidence_target_counts": _count_rows(failed_target_counts),
        "rule_coverage": _public_rule_buckets(rule_buckets),
    }
    checks = _coverage_checks(
        metrics,
        min_failed_rule_evidence_rate=min_failed_rule_evidence_rate,
        min_critical_failed_rule_evidence_rate=min_critical_failed_rule_evidence_rate,
        min_event_evidence_refs=min_event_evidence_refs,
        max_failed_rules_without_evidence=max_failed_rules_without_evidence,
        max_critical_failed_rules_without_evidence=max_critical_failed_rules_without_evidence,
        require_rule_evidence=require_rule_evidence or [],
    )
    return {
        "schema_version": EVIDENCE_COVERAGE_SCHEMA_VERSION,
        "runs_dir": _display_path(root, preserve_paths),
        "passed": all(check["passed"] for check in checks),
        "check_count": len(checks),
        "failed_check_count": sum(1 for check in checks if not check["passed"]),
        "checks": checks,
        "metrics": metrics,
        "warnings": warnings,
        "runs": run_rows,
    }


def _load_run_records(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        raise EvidenceCoverageError(f"Runs directory not found: {root}")
    if not root.is_dir():
        raise EvidenceCoverageError(f"Runs path is not a directory: {root}")

    records: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        score_path = run_dir / "scorecard.json"
        if not score_path.exists():
            continue
        scorecard = _read_object(score_path)
        trace = _read_object(run_dir / "normalized_trace.json") if (run_dir / "normalized_trace.json").exists() else None
        if not isinstance(scorecard, dict):
            raise EvidenceCoverageError(f"Run {run_dir} scorecard.json must contain a JSON object")
        if trace is not None and not isinstance(trace, dict):
            raise EvidenceCoverageError(f"Run {run_dir} normalized_trace.json must contain a JSON object")
        records.append({"run_dir": run_dir, "scorecard": scorecard, "trace": trace})

    if not records:
        raise EvidenceCoverageError(f"No completed Flight Recorder runs found in {root}")
    return records


def _run_coverage(record: dict[str, Any], preserve_paths: bool) -> dict[str, Any]:
    scorecard = record["scorecard"]
    trace = record.get("trace")
    rules = [rule for rule in scorecard.get("rules", []) if isinstance(rule, dict)]
    rule_rows = [_rule_coverage(rule) for rule in rules]
    failed_rows = [rule for rule in rule_rows if rule["failed"]]
    critical_failed_rows = [rule for rule in failed_rows if rule["critical"]]
    target_counts: dict[str, int] = {}
    failed_target_counts: dict[str, int] = {}
    for rule in rule_rows:
        _merge_counts(target_counts, rule.get("evidence_target_counts"))
        if rule["failed"]:
            _merge_counts(failed_target_counts, rule.get("evidence_target_counts"))

    failed_without = [rule["rule_id"] for rule in failed_rows if rule["evidence_ref_count"] == 0]
    critical_failed_without = [rule["rule_id"] for rule in critical_failed_rows if rule["evidence_ref_count"] == 0]
    return {
        "scenario_id": str(scorecard.get("scenario_id") or record["run_dir"].name),
        "scenario_title": str(scorecard.get("scenario_title") or scorecard.get("scenario_id") or record["run_dir"].name),
        "run_dir": _display_path(record["run_dir"], preserve_paths),
        "score": _score(scorecard),
        "passed": bool(scorecard.get("passed")),
        "event_count": _event_count(trace),
        "rule_count": len(rule_rows),
        "failed_rule_count": len(failed_rows),
        "critical_failed_rule_count": len(critical_failed_rows),
        "evidence_ref_count": sum(rule["evidence_ref_count"] for rule in rule_rows),
        "failed_rule_evidence_ref_count": sum(rule["evidence_ref_count"] for rule in failed_rows),
        "critical_failed_rule_evidence_ref_count": sum(rule["evidence_ref_count"] for rule in critical_failed_rows),
        "failed_rules_with_evidence": sum(1 for rule in failed_rows if rule["evidence_ref_count"] > 0),
        "failed_rules_without_evidence": failed_without,
        "critical_failed_rules_with_evidence": sum(1 for rule in critical_failed_rows if rule["evidence_ref_count"] > 0),
        "critical_failed_rules_without_evidence": critical_failed_without,
        "task_evidence_ref_count": sum(rule["evidence_ref_count"] for rule in rule_rows if rule["rule_id"] in _TASK_RULE_IDS),
        "evidence_target_counts": _count_rows(target_counts),
        "failed_rule_evidence_target_counts": _count_rows(failed_target_counts),
        "rules": rule_rows,
    }


def _rule_coverage(rule: dict[str, Any]) -> dict[str, Any]:
    refs = _evidence_refs(rule)
    target_counts = _count_targets(refs)
    return {
        "rule_id": str(rule.get("id") or "unknown_rule"),
        "rule_name": str(rule.get("name") or rule.get("id") or "unknown_rule"),
        "passed": bool(rule.get("passed")),
        "failed": rule.get("passed") is False,
        "critical": bool(rule.get("critical")),
        "evidence_ref_count": len(refs),
        "negative_evidence_ref_count": sum(1 for ref in refs if ref.get("passed") is not True),
        "evidence_target_counts": _count_rows(target_counts),
    }


def _merge_rule_bucket(buckets: dict[str, dict[str, Any]], rule: dict[str, Any]) -> None:
    rule_id = str(rule["rule_id"])
    bucket = buckets.setdefault(
        rule_id,
        {
            "rule_id": rule_id,
            "rule_name": rule.get("rule_name", rule_id),
            "rule_count": 0,
            "passed": 0,
            "failed": 0,
            "critical_failed": 0,
            "evidence_ref_count": 0,
            "negative_evidence_ref_count": 0,
            "failed_with_evidence": 0,
            "failed_without_evidence": 0,
            "target_counts": {},
        },
    )
    bucket["rule_count"] += 1
    bucket["passed"] += 1 if rule["passed"] else 0
    bucket["failed"] += 1 if rule["failed"] else 0
    bucket["critical_failed"] += 1 if rule["failed"] and rule["critical"] else 0
    bucket["evidence_ref_count"] += int(rule["evidence_ref_count"])
    bucket["negative_evidence_ref_count"] += int(rule["negative_evidence_ref_count"])
    if rule["failed"]:
        if rule["evidence_ref_count"] > 0:
            bucket["failed_with_evidence"] += 1
        else:
            bucket["failed_without_evidence"] += 1
    _merge_counts(bucket["target_counts"], rule.get("evidence_target_counts"))
    bucket["evidence_target_counts"] = _count_rows(bucket["target_counts"])


def _public_rule_buckets(buckets: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bucket in buckets.values():
        rows.append({key: value for key, value in bucket.items() if key != "target_counts"})
    return sorted(rows, key=lambda item: str(item["rule_id"]))


def _coverage_checks(
    metrics: dict[str, Any],
    *,
    min_failed_rule_evidence_rate: float | None,
    min_critical_failed_rule_evidence_rate: float | None,
    min_event_evidence_refs: int | None,
    max_failed_rules_without_evidence: int | None,
    max_critical_failed_rules_without_evidence: int | None,
    require_rule_evidence: list[str],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    if min_failed_rule_evidence_rate is not None:
        _add_min_check(checks, "min_failed_rule_evidence_rate", metrics["failed_rule_evidence_rate"], min_failed_rule_evidence_rate)
    if min_critical_failed_rule_evidence_rate is not None:
        _add_min_check(
            checks,
            "min_critical_failed_rule_evidence_rate",
            metrics["critical_failed_rule_evidence_rate"],
            min_critical_failed_rule_evidence_rate,
        )
    if min_event_evidence_refs is not None:
        _add_min_check(checks, "min_event_evidence_refs", metrics["event_evidence_ref_count"], min_event_evidence_refs)
    if max_failed_rules_without_evidence is not None:
        _add_max_check(
            checks,
            "max_failed_rules_without_evidence",
            metrics["failed_rules_without_evidence"],
            max_failed_rules_without_evidence,
        )
    if max_critical_failed_rules_without_evidence is not None:
        _add_max_check(
            checks,
            "max_critical_failed_rules_without_evidence",
            metrics["critical_failed_rules_without_evidence"],
            max_critical_failed_rules_without_evidence,
        )
    coverage_by_rule = {row["rule_id"]: row for row in metrics["rule_coverage"]}
    for rule_id in require_rule_evidence:
        count = int(coverage_by_rule.get(rule_id, {}).get("evidence_ref_count", 0))
        _add_min_check(checks, "require_rule_evidence", count, 1, {"rule_id": rule_id})
    return checks


def _add_min_check(
    checks: list[dict[str, Any]],
    check_id: str,
    actual: float | int,
    minimum: float | int,
    scope: dict[str, str] | None = None,
) -> None:
    check = {
        "id": check_id,
        "passed": actual >= minimum,
        "actual": actual,
        "expected": {"min": minimum},
        "summary": f"{check_id}: actual={actual}, min={minimum}",
    }
    if scope:
        check["scope"] = scope
    checks.append(check)


def _add_max_check(checks: list[dict[str, Any]], check_id: str, actual: int, maximum: int) -> None:
    checks.append(
        {
            "id": check_id,
            "passed": actual <= maximum,
            "actual": actual,
            "expected": {"max": maximum},
            "summary": f"{check_id}: actual={actual}, max={maximum}",
        }
    )


def _read_object(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _evidence_refs(rule: dict[str, Any]) -> list[dict[str, Any]]:
    refs = rule.get("evidence_refs")
    if not isinstance(refs, list):
        return []
    return [ref for ref in refs if isinstance(ref, dict)]


def _count_targets(refs: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ref in refs:
        target = str(ref.get("target") or "unknown")
        counts[target] = counts.get(target, 0) + 1
    return counts


def _merge_counts(target: dict[str, int], rows: Any) -> None:
    if isinstance(rows, dict):
        iterable = rows.items()
    elif isinstance(rows, list):
        iterable = ((row.get("id"), row.get("count")) for row in rows if isinstance(row, dict))
    else:
        return
    for key, count in iterable:
        if not isinstance(key, str) or not key:
            continue
        target[key] = target.get(key, 0) + _non_negative_int(count)


def _count_rows(counts: dict[str, int]) -> list[dict[str, int | str]]:
    return [
        {"id": key, "count": count}
        for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 1.0
    return round(numerator / denominator, 4)


def _score(scorecard: dict[str, Any]) -> int:
    try:
        return max(0, min(100, int(scorecard.get("score", 0))))
    except (TypeError, ValueError):
        return 0


def _event_count(trace: Any) -> int | None:
    if not isinstance(trace, dict):
        return None
    events = trace.get("events")
    return len(events) if isinstance(events, list) else None


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int) and value > 0:
        return value
    return 0


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


_TASK_RULE_IDS = {"required_evidence", "required_actions", "required_action_sequences", "required_event_counts"}
