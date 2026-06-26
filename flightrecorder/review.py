"""Human-review exports for Flight Recorder evidence runs."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .training import RunRecord, TrainingExportError, load_run_records

REVIEW_MANIFEST_SCHEMA_VERSION = "hfr.review.manifest.v1"
REVIEW_ITEM_SCHEMA_VERSION = "hfr.review.item.v1"
REVIEW_LABEL_SCHEMA_VERSION = "hfr.review.label.v1"
REVIEW_LABELS = ("accept", "reject", "needs_review", "unsafe", "incomplete")
FAMILY_SUFFIX_RE = re.compile(r"([_-](good|bad|pass|fail|passing|failing|chosen|rejected))+$", re.IGNORECASE)


class ReviewExportError(ValueError):
    """Raised when a human-review export cannot be produced."""


def export_review_queue(
    runs_dir: str | Path,
    out_dir: str | Path,
    *,
    only_failed: bool = False,
    preserve_paths: bool = False,
) -> dict[str, Any]:
    """Export run evidence as a queue for human labeling and curation."""
    source = Path(runs_dir)
    target = Path(out_dir)
    try:
        records = load_run_records(source)
    except TrainingExportError as exc:
        raise ReviewExportError(str(exc)) from exc
    if only_failed:
        records = [record for record in records if not bool(record.scorecard.get("passed"))]
    if not records:
        raise ReviewExportError("No runs matched the requested review export filters")

    target.mkdir(parents=True, exist_ok=True)
    items = [_review_item(record, preserve_paths) for record in records]
    labels = [_label_template(item) for item in items]
    paths = {
        "review_items": target / "review_items.jsonl",
        "label_template": target / "label_template.jsonl",
        "instructions": target / "REVIEW_INSTRUCTIONS.md",
        "manifest": target / "manifest.json",
    }
    _write_jsonl(paths["review_items"], items)
    _write_jsonl(paths["label_template"], labels)

    manifest = {
        "schema_version": REVIEW_MANIFEST_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_runs_dir": _display_path(source, preserve_paths),
        "output_dir": _display_path(target, preserve_paths),
        "only_failed": only_failed,
        "item_count": len(items),
        "passed_count": sum(1 for item in items if item["scorecard"]["passed"] is True),
        "failed_count": sum(1 for item in items if item["scorecard"]["passed"] is False),
        "label_options": list(REVIEW_LABELS),
        "task_families": sorted({str(item["task_family"]) for item in items}),
        "outputs": {name: _display_path(path, preserve_paths) for name, path in paths.items()},
        "notes": [
            "Review items are derived from normalized_trace.json, scorecard.json, report.html, and optional artifact_lineage.json.",
            "Use label_template.jsonl as a starting point for human labels; do not treat suggested labels as ground truth.",
            "Human labels should be grounded in observable events, scorecard evidence, reports, and lineage.",
        ],
    }
    _write_text(paths["instructions"], _instructions(manifest))
    _write_json(paths["manifest"], manifest)
    return manifest


def _review_item(record: RunRecord, preserve_paths: bool) -> dict[str, Any]:
    scorecard = record.scorecard
    trace = record.trace
    scenario_id = str(scorecard.get("scenario_id") or record.run_id)
    passed = bool(scorecard.get("passed"))
    return {
        "schema_version": REVIEW_ITEM_SCHEMA_VERSION,
        "review_item_id": record.run_id,
        "episode_id": record.run_id,
        "scenario_id": scenario_id,
        "scenario_title": str(scorecard.get("scenario_title") or scenario_id),
        "task_family": _task_family(scenario_id),
        "source_artifacts": _source_artifacts(record, preserve_paths),
        "prompt": _prompt_from_trace(trace),
        "final_answer": str(trace.get("final_answer") or ""),
        "event_count": len(trace.get("events", [])) if isinstance(trace.get("events"), list) else 0,
        "scorecard": {
            "passed": passed,
            "score": _score(scorecard),
            "pass_threshold": scorecard.get("pass_threshold"),
            "summary": str(scorecard.get("summary") or ""),
            "critical_failures": scorecard.get("critical_failures", []),
            "failed_rules": _failed_rule_ids(scorecard),
        },
        "rule_summaries": [_rule_summary(rule) for rule in scorecard.get("rules", []) if isinstance(rule, dict)],
        "task_evidence": _task_evidence(scorecard),
        "evidence_target_counts": _evidence_target_counts(scorecard),
        "suggested_human_label": "accept" if passed else "reject",
        "label_options": list(REVIEW_LABELS),
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


def _label_template(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": REVIEW_LABEL_SCHEMA_VERSION,
        "review_item_id": item["review_item_id"],
        "episode_id": item["episode_id"],
        "scenario_id": item["scenario_id"],
        "suggested_human_label": item["suggested_human_label"],
        "human_label": None,
        "corrected_score": None,
        "reviewer": None,
        "reviewed_at": None,
        "notes": "",
        "accepted_evidence_refs": [],
        "rejected_evidence_refs": [],
    }


def _rule_summary(rule: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(rule.get("id") or ""),
        "name": str(rule.get("name") or rule.get("id") or ""),
        "passed": bool(rule.get("passed")),
        "critical": bool(rule.get("critical")),
        "penalty": int(rule.get("penalty", 0) or 0),
        "evidence": [str(item) for item in rule.get("evidence", [])],
        "evidence_ref_count": len(rule.get("evidence_refs", [])) if isinstance(rule.get("evidence_refs"), list) else 0,
    }


def _task_evidence(scorecard: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rule in scorecard.get("rules", []):
        if not isinstance(rule, dict) or rule.get("id") not in {"required_actions", "required_action_sequences", "required_event_counts"}:
            continue
        for item in rule.get("items", []):
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "rule_id": str(rule.get("id") or ""),
                    "id": str(item.get("id") or ""),
                    "description": str(item.get("description") or item.get("id") or ""),
                    "passed": bool(item.get("passed")),
                    "evidence": str(item.get("evidence") or ""),
                    "event_indices": item.get("event_indices", []),
                }
            )
    return rows


def _evidence_target_counts(scorecard: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rule in scorecard.get("rules", []):
        if not isinstance(rule, dict):
            continue
        refs = rule.get("evidence_refs")
        if not isinstance(refs, list):
            continue
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            target = str(ref.get("target") or "unknown")
            counts[target] = counts.get(target, 0) + 1
    return counts


def _instructions(manifest: dict[str, Any]) -> str:
    labels = ", ".join(f"`{label}`" for label in manifest["label_options"])
    return "\n".join(
        [
            "# Flight Recorder Review Queue",
            "",
            "This export is for human curation before evidence becomes training data.",
            "",
            f"- Items: `{manifest['item_count']}`",
            f"- Passed: `{manifest['passed_count']}`",
            f"- Failed: `{manifest['failed_count']}`",
            f"- Label options: {labels}",
            "",
            "Review `review_items.jsonl` alongside each item report and lineage file.",
            "Fill `label_template.jsonl` with `human_label`, `reviewer`, `reviewed_at`, and notes.",
            "Human labels should be grounded in observable trace events, scorecard evidence, reports, and lineage.",
            "A suggested label is only a starting point; prefer observable trace evidence over final-answer claims.",
            "",
        ]
    )


def _prompt_from_trace(trace: dict[str, Any]) -> str:
    for event in trace.get("events", []):
        if isinstance(event, dict) and event.get("type") == "user_message" and event.get("text"):
            return str(event["text"])
    return ""


def _failed_rule_ids(scorecard: dict[str, Any]) -> list[str]:
    return [
        str(rule.get("id"))
        for rule in scorecard.get("rules", [])
        if isinstance(rule, dict) and rule.get("id") and not rule.get("passed")
    ]


def _score(scorecard: dict[str, Any]) -> int:
    try:
        return max(0, min(100, int(scorecard.get("score", 0))))
    except (TypeError, ValueError):
        return 0


def _task_family(scenario_id: str) -> str:
    return FAMILY_SUFFIX_RE.sub("", scenario_id) or scenario_id


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


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
