"""Deterministic improvement plans over Flight Recorder evidence handoffs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .bundle import EVIDENCE_BUNDLE_SCHEMA_VERSION
from .digest import RUN_DIGEST_SCHEMA_VERSION
from .eval_summary import EVAL_SUMMARY_SCHEMA_VERSION
from .repair import REPAIR_QUEUE_SCHEMA_VERSION
from .training import RL_CURRICULUM_SCHEMA_VERSION

IMPROVEMENT_PLAN_SCHEMA_VERSION = "hfr.improvement_plan.v1"
PRIORITIES = ("critical", "high", "medium", "low")
PRIORITY_RANK = {priority: index for index, priority in enumerate(PRIORITIES)}
CATEGORIES = ("bundle_action", "repair", "curriculum", "digest_action")
CATEGORY_RANK = {category: index for index, category in enumerate(CATEGORIES)}


class ImprovementPlanError(ValueError):
    """Raised when an improvement plan cannot be produced."""


def build_improvement_plan(
    *,
    out_path: str | Path,
    evidence_bundle_path: str | Path,
    repair_queue_path: str | Path | None = None,
    training_export_dir: str | Path | None = None,
    runs_dir: str | Path | None = None,
    eval_summary_path: str | Path | None = None,
    preserve_paths: bool = False,
) -> dict[str, Any]:
    """Build a deterministic next-iteration plan from existing evidence artifacts."""
    bundle_path = Path(evidence_bundle_path)
    bundle = _read_json(bundle_path, "evidence_bundle")
    if bundle.get("schema_version") != EVIDENCE_BUNDLE_SCHEMA_VERSION:
        raise ImprovementPlanError(
            f"evidence_bundle schema_version must be {EVIDENCE_BUNDLE_SCHEMA_VERSION!r}; got {bundle.get('schema_version')!r}"
        )

    repair_queue: dict[str, Any] | None = None
    if repair_queue_path is not None:
        repair_queue = _read_json(Path(repair_queue_path), "repair_queue")
        if repair_queue.get("schema_version") != REPAIR_QUEUE_SCHEMA_VERSION:
            raise ImprovementPlanError(
                f"repair_queue schema_version must be {REPAIR_QUEUE_SCHEMA_VERSION!r}; got {repair_queue.get('schema_version')!r}"
            )

    curriculum: dict[str, Any] | None = None
    curriculum_path: Path | None = None
    if training_export_dir is not None:
        curriculum_path = Path(training_export_dir) / "curriculum.json"
        curriculum = _read_json(curriculum_path, "training_export curriculum")
        if curriculum.get("schema_version") != RL_CURRICULUM_SCHEMA_VERSION:
            raise ImprovementPlanError(
                f"curriculum schema_version must be {RL_CURRICULUM_SCHEMA_VERSION!r}; got {curriculum.get('schema_version')!r}"
            )

    eval_summary: dict[str, Any] | None = None
    if eval_summary_path is not None:
        eval_summary = _read_json(Path(eval_summary_path), "eval_summary")
        if eval_summary.get("schema_version") != EVAL_SUMMARY_SCHEMA_VERSION:
            raise ImprovementPlanError(
                f"eval_summary schema_version must be {EVAL_SUMMARY_SCHEMA_VERSION!r}; got {eval_summary.get('schema_version')!r}"
            )

    run_digests = _load_run_digests(Path(runs_dir), preserve_paths) if runs_dir is not None else {}
    raw_items = _build_raw_items(bundle, repair_queue, curriculum, run_digests, eval_summary)
    work_items = _finalize_work_items(raw_items)
    metrics = _metrics(work_items)
    readiness = "blocked" if bundle.get("readiness") == "blocked" else "ready"
    source_artifacts = {
        "evidence_bundle": _file_record(bundle_path, preserve_paths),
    }
    if repair_queue_path is not None:
        source_artifacts["repair_queue"] = _file_record(Path(repair_queue_path), preserve_paths)
    if training_export_dir is not None:
        source_artifacts["training_export"] = _dir_record(Path(training_export_dir), preserve_paths)
        if curriculum_path is not None:
            source_artifacts["training_export_curriculum"] = _file_record(curriculum_path, preserve_paths)
    if runs_dir is not None:
        source_artifacts["runs_dir"] = _dir_record(Path(runs_dir), preserve_paths)
    if eval_summary_path is not None:
        source_artifacts["eval_summary"] = _file_record(Path(eval_summary_path), preserve_paths)

    return {
        "schema_version": IMPROVEMENT_PLAN_SCHEMA_VERSION,
        "plan_path": _display_path(Path(out_path), preserve_paths),
        "passed": True,
        "readiness": readiness,
        "decision": _decision(readiness, bundle, metrics, work_items),
        "source_artifacts": source_artifacts,
        "metrics": metrics,
        "work_item_count": len(work_items),
        "work_items": work_items,
        "notes": [
            "Improvement plans summarize existing evidence; they do not execute repairs, train models, or approve promotion.",
            "Use item fingerprints and routing keys to deduplicate work across repeated evidence bundles.",
            "Treat curriculum and reward hints as prioritization evidence, not as an online reward function.",
        ],
    }


def work_item_fingerprint(item: dict[str, Any]) -> str:
    """Return the stable fingerprint for an improvement-plan work item."""
    payload = {
        key: item.get(key)
        for key in (
            "category",
            "priority",
            "summary",
            "suggested_action",
            "scenario_id",
            "task_family",
            "rule_id",
            "rule_name",
            "score",
            "task_completion_status",
            "sources",
            "evidence_refs",
            "evidence_snippets",
            "source_artifacts",
            "replay",
        )
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_raw_items(
    bundle: dict[str, Any],
    repair_queue: dict[str, Any] | None,
    curriculum: dict[str, Any] | None,
    run_digests: dict[str, dict[str, Any]],
    eval_summary: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    repair_items_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}

    if repair_queue is not None:
        for repair_item in repair_queue.get("items", []):
            if not isinstance(repair_item, dict):
                continue
            item = _item_from_repair(repair_item)
            items.append(item)
            key = (str(item.get("scenario_id") or ""), str(item.get("rule_id") or ""))
            repair_items_by_key.setdefault(key, []).append(item)

    for mode in _curriculum_modes(curriculum):
        matched = False
        for scenario_id in mode["scenario_ids"]:
            for item in repair_items_by_key.get((scenario_id, mode["rule_id"]), []):
                _append_source(item, "curriculum_priorities", mode)
                matched = True
        if not matched:
            items.append(_item_from_curriculum(mode))

    for digest in run_digests.values():
        attached = False
        failed_rule_ids = digest["failed_rule_ids"]
        for rule_id in failed_rule_ids:
            for item in repair_items_by_key.get((digest["scenario_id"], rule_id), []):
                item["sources"]["run_digest"] = digest
                attached = True
        if not attached and _digest_has_actionable_failure(digest):
            items.append(_item_from_digest(digest))

    decision = bundle.get("decision") if isinstance(bundle.get("decision"), dict) else {}
    for action in decision.get("next_actions", []) if isinstance(decision.get("next_actions"), list) else []:
        if isinstance(action, dict):
            items.append(_item_from_bundle_action(action))

    for item in _eval_summary_items(eval_summary):
        items.append(_item_from_eval_summary(item))

    return items


def _item_from_repair(repair_item: dict[str, Any]) -> dict[str, Any]:
    priority = _priority(str(repair_item.get("priority") or "high"))
    return {
        "category": "repair",
        "priority": priority,
        "summary": str(repair_item.get("summary") or "Repair failed scorecard rule."),
        "suggested_action": str(repair_item.get("suggested_action") or "Inspect the failed rule evidence and repair the behavior."),
        "scenario_id": str(repair_item.get("scenario_id") or ""),
        "task_family": str(repair_item.get("task_family") or "unknown"),
        "rule_id": str(repair_item.get("rule_id") or "unknown_rule"),
        "rule_name": str(repair_item.get("rule_name") or repair_item.get("rule_id") or "unknown_rule"),
        "score": _score_or_none(repair_item.get("score")),
        "task_completion_status": str(repair_item.get("task_completion_status") or "unknown"),
        "sources": {
            "bundle_action_ids": [],
            "repair_item_ids": [str(repair_item.get("repair_item_id") or "")],
            "curriculum_priorities": [],
            "run_digest": None,
        },
        "evidence_refs": _dict_list(repair_item.get("evidence_refs")),
        "evidence_snippets": _dict_list(repair_item.get("evidence_snippets")),
        "source_artifacts": repair_item.get("source_artifacts") if isinstance(repair_item.get("source_artifacts"), dict) else {},
        "replay": repair_item.get("replay") if isinstance(repair_item.get("replay"), dict) else {},
    }


def _item_from_curriculum(mode: dict[str, Any]) -> dict[str, Any]:
    priority = _priority(str(mode.get("priority_band") or "medium"))
    scenario_ids = mode.get("scenario_ids") if isinstance(mode.get("scenario_ids"), list) else []
    return {
        "category": "curriculum",
        "priority": priority,
        "summary": (
            f"Curriculum priority {mode['task_family']} / {mode['rule_id']} "
            f"appears {mode['count']} time(s)."
        ),
        "suggested_action": "Generate repair work, stronger scenarios, or review labels for this recurring failure mode.",
        "scenario_id": str(scenario_ids[0]) if scenario_ids else None,
        "task_family": str(mode.get("task_family") or "unknown"),
        "rule_id": str(mode.get("rule_id") or "unknown_rule"),
        "rule_name": str(mode.get("rule_name") or mode.get("rule_id") or "unknown_rule"),
        "score": None,
        "task_completion_status": None,
        "sources": {
            "bundle_action_ids": [],
            "repair_item_ids": [],
            "curriculum_priorities": [mode],
            "run_digest": None,
        },
        "evidence_refs": _dict_list(mode.get("example_evidence_refs")),
        "evidence_snippets": [],
        "source_artifacts": {},
        "replay": {},
    }


def _item_from_digest(digest: dict[str, Any]) -> dict[str, Any]:
    action_ids = digest.get("recommended_action_ids") if isinstance(digest.get("recommended_action_ids"), list) else []
    priority = _priority(str(digest.get("highest_action_priority") or "medium"))
    return {
        "category": "digest_action",
        "priority": priority,
        "summary": f"Run digest for {digest['scenario_id']} recommends {', '.join(action_ids) or 'review'}.",
        "suggested_action": "Inspect the compact run digest and convert its recommendation into a scenario, repair, or review task.",
        "scenario_id": str(digest.get("scenario_id") or ""),
        "task_family": str(digest.get("task_family") or "unknown"),
        "rule_id": None,
        "rule_name": None,
        "score": _score_or_none(digest.get("score")),
        "task_completion_status": str(digest.get("task_completion_status") or "unknown"),
        "sources": {
            "bundle_action_ids": [],
            "repair_item_ids": [],
            "curriculum_priorities": [],
            "run_digest": digest,
        },
        "evidence_refs": [],
        "evidence_snippets": [],
        "source_artifacts": {"run_digest": digest.get("path", "")},
        "replay": {},
    }


def _item_from_bundle_action(action: dict[str, Any]) -> dict[str, Any]:
    action_id = str(action.get("id") or "unknown_action")
    evidence = action.get("evidence") if isinstance(action.get("evidence"), dict) else {}
    return {
        "category": "bundle_action",
        "priority": _priority(str(action.get("priority") or "medium")),
        "summary": str(action.get("summary") or action_id),
        "suggested_action": "Use the referenced bundle evidence to decide the next repair, review, or handoff step.",
        "scenario_id": None,
        "task_family": None,
        "rule_id": None,
        "rule_name": None,
        "score": None,
        "task_completion_status": None,
        "sources": {
            "bundle_action_ids": [action_id],
            "repair_item_ids": [],
            "curriculum_priorities": [],
            "run_digest": None,
            "bundle_action": {
                "id": action_id,
                "artifact": str(action.get("artifact") or "unknown"),
                "routing_key": str(action.get("routing_key") or ""),
                "action_fingerprint": str(action.get("action_fingerprint") or ""),
                "evidence": evidence,
            },
        },
        "evidence_refs": [],
        "evidence_snippets": [],
        "source_artifacts": {},
        "replay": {},
    }


def _item_from_eval_summary(item: dict[str, Any]) -> dict[str, Any]:
    raw_category = str(item.get("category") or "eval_harness")
    category = raw_category if raw_category in {"repair", "curriculum"} else "bundle_action"
    eval_item_id = str(item.get("work_item_id") or item.get("reason") or "eval_summary_item")
    scenario_id = item.get("scenario_id") if isinstance(item.get("scenario_id"), str) and item.get("scenario_id") else None
    rule_id = item.get("rule_id") if isinstance(item.get("rule_id"), str) and item.get("rule_id") else None
    source_record = {
        "work_item_id": eval_item_id,
        "category": raw_category,
        "source": str(item.get("source") or "eval_summary"),
        "label": str(item.get("label") or "eval_summary"),
        "reason": str(item.get("reason") or "eval_summary_item"),
    }
    if "count" in item:
        source_record["count"] = _non_negative_int(item.get("count"))
    sources: dict[str, Any] = {
        "bundle_action_ids": [],
        "repair_item_ids": [],
        "curriculum_priorities": [],
        "run_digest": None,
        "eval_summary_items": [source_record],
    }
    if category == "bundle_action":
        sources["bundle_action_ids"] = [eval_item_id]
        sources["bundle_action"] = {
            "id": eval_item_id,
            "artifact": "eval_summary",
            "routing_key": source_record["reason"],
            "action_fingerprint": eval_item_id,
            "evidence": source_record,
        }
    return {
        "category": category,
        "priority": _priority(str(item.get("priority") or "medium")),
        "summary": str(item.get("summary") or "Review eval-summary follow-up item."),
        "suggested_action": str(item.get("suggested_action") or "Inspect eval summary and route follow-up work."),
        "scenario_id": scenario_id,
        "task_family": _eval_task_family(scenario_id, source_record),
        "rule_id": rule_id,
        "rule_name": rule_id or source_record["reason"],
        "score": None,
        "task_completion_status": "regressed" if source_record["reason"] == "task_completion_regression" else None,
        "sources": sources,
        "evidence_refs": [],
        "evidence_snippets": [],
        "source_artifacts": {},
        "replay": {},
    }


def _finalize_work_items(raw_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    finalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_items:
        item["priority"] = _priority(str(item.get("priority") or "medium"))
        item["priority_rank"] = PRIORITY_RANK[item["priority"]]
        fingerprint = work_item_fingerprint(item)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        item["fingerprint"] = fingerprint
        item["item_id"] = f"{item['category']}:{fingerprint[:16]}"
        item["routing_key"] = f"{item['category']}:{item['priority']}:{fingerprint[:12]}"
        finalized.append(item)
    finalized.sort(
        key=lambda item: (
            PRIORITY_RANK.get(str(item.get("priority")), 99),
            CATEGORY_RANK.get(str(item.get("category")), 99),
            str(item.get("task_family") or ""),
            str(item.get("scenario_id") or ""),
            str(item.get("rule_id") or ""),
            str(item.get("summary") or ""),
        )
    )
    return finalized


def _decision(
    readiness: str,
    bundle: dict[str, Any],
    metrics: dict[str, Any],
    work_items: list[dict[str, Any]],
) -> dict[str, Any]:
    critical = _count_from_rows(metrics["priority_counts"], "critical")
    high = _count_from_rows(metrics["priority_counts"], "high")
    if readiness == "blocked":
        recommendation = "fix_handoff"
        summary = "Evidence handoff is blocked; fix bundle checks or gates before using this plan for promotion."
    elif critical or high:
        recommendation = "run_improvement_iteration"
        summary = f"Route {critical + high} critical/high-priority work item(s) into the next improvement iteration."
    elif work_items:
        recommendation = "review_improvement_opportunities"
        summary = f"Review {len(work_items)} lower-priority evidence-backed improvement opportunity item(s)."
    else:
        recommendation = "promote_or_monitor"
        summary = "No work items were derived from the supplied evidence; promotion can be considered if external gates agree."
    bundle_decision = bundle.get("decision") if isinstance(bundle.get("decision"), dict) else {}
    return {
        "readiness": readiness,
        "recommendation": recommendation,
        "summary": summary,
        "source_bundle_recommendation": str(bundle_decision.get("recommendation") or ""),
        "source_bundle_next_action_count": _non_negative_int(bundle_decision.get("next_action_count")),
        "work_item_count": len(work_items),
        "critical_or_high_count": critical + high,
        "top_work_items": [
            {
                "item_id": str(item.get("item_id") or ""),
                "priority": str(item.get("priority") or ""),
                "category": str(item.get("category") or ""),
                "scenario_id": item.get("scenario_id"),
                "rule_id": item.get("rule_id"),
                "summary": str(item.get("summary") or ""),
            }
            for item in work_items[:5]
        ],
    }


def _metrics(work_items: list[dict[str, Any]]) -> dict[str, Any]:
    scenario_ids = sorted({str(item.get("scenario_id")) for item in work_items if item.get("scenario_id")})
    task_families = sorted({str(item.get("task_family")) for item in work_items if item.get("task_family")})
    rule_ids = sorted({str(item.get("rule_id")) for item in work_items if item.get("rule_id")})
    return {
        "work_item_count": len(work_items),
        "scenario_count": len(scenario_ids),
        "task_family_count": len(task_families),
        "rule_count": len(rule_ids),
        "priority_counts": _count_rows(str(item.get("priority") or "unknown") for item in work_items),
        "category_counts": _count_rows(str(item.get("category") or "unknown") for item in work_items),
        "task_family_counts": _count_rows(str(item.get("task_family")) for item in work_items if item.get("task_family")),
        "rule_counts": _count_rows(str(item.get("rule_id")) for item in work_items if item.get("rule_id")),
        "repair_backed_count": sum(1 for item in work_items if _source_list(item, "repair_item_ids")),
        "curriculum_backed_count": sum(1 for item in work_items if _source_list(item, "curriculum_priorities")),
        "digest_backed_count": sum(1 for item in work_items if _source_digest(item)),
        "bundle_action_count": sum(1 for item in work_items if item.get("category") == "bundle_action"),
        "evidence_ref_count": sum(len(item.get("evidence_refs", [])) for item in work_items if isinstance(item.get("evidence_refs"), list)),
        "scenarios": scenario_ids,
        "task_families": task_families,
        "rules": rule_ids,
    }


def _curriculum_modes(curriculum: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(curriculum, dict):
        return []
    rows: list[dict[str, Any]] = []
    families = curriculum.get("task_families") if isinstance(curriculum.get("task_families"), list) else []
    for family in families:
        if not isinstance(family, dict):
            continue
        task_family = str(family.get("task_family") or "unknown")
        modes = family.get("failure_modes") if isinstance(family.get("failure_modes"), list) else []
        for mode in modes:
            if not isinstance(mode, dict):
                continue
            rows.append(
                {
                    "task_family": task_family,
                    "rule_id": str(mode.get("rule_id") or "unknown_rule"),
                    "rule_name": str(mode.get("rule_name") or mode.get("rule_id") or "unknown_rule"),
                    "priority_score": _non_negative_int(mode.get("priority_score")),
                    "priority_band": _priority(str(mode.get("priority_band") or "medium")),
                    "count": _non_negative_int(mode.get("count")),
                    "critical_count": _non_negative_int(mode.get("critical_count")),
                    "max_penalty": _non_negative_int(mode.get("max_penalty")),
                    "scenario_ids": _string_list(mode.get("scenario_ids")),
                    "failure_ids": _string_list(mode.get("failure_ids")),
                    "example_evidence_refs": _dict_list(mode.get("example_evidence_refs")),
                }
            )
    rows.sort(key=lambda item: (-item["priority_score"], -item["count"], item["task_family"], item["rule_id"]))
    return rows


def _eval_summary_items(eval_summary: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(eval_summary, dict):
        return []
    repair_curriculum = eval_summary.get("repair_curriculum")
    if not isinstance(repair_curriculum, dict):
        return []
    items = repair_curriculum.get("items")
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _eval_task_family(scenario_id: str | None, source_record: dict[str, Any]) -> str:
    if scenario_id:
        return scenario_id
    label = str(source_record.get("label") or "")
    source = str(source_record.get("source") or "eval_summary")
    return label or source or "eval_summary"


def _load_run_digests(runs_dir: Path, preserve_paths: bool) -> dict[str, dict[str, Any]]:
    if not runs_dir.exists() or not runs_dir.is_dir():
        raise ImprovementPlanError(f"Runs directory not found: {runs_dir}")
    digests: dict[str, dict[str, Any]] = {}
    for child in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        digest_path = child / "run_digest.json"
        if not digest_path.exists():
            continue
        digest = _read_json(digest_path, f"run digest {digest_path}")
        if digest.get("schema_version") != RUN_DIGEST_SCHEMA_VERSION:
            continue
        scenario = digest.get("scenario") if isinstance(digest.get("scenario"), dict) else {}
        outcome = digest.get("outcome") if isinstance(digest.get("outcome"), dict) else {}
        rules = digest.get("rules") if isinstance(digest.get("rules"), dict) else {}
        failed = rules.get("failed") if isinstance(rules.get("failed"), list) else []
        actions = digest.get("recommended_actions") if isinstance(digest.get("recommended_actions"), list) else []
        scenario_id = str(scenario.get("id") or child.name)
        digests[scenario_id] = {
            "scenario_id": scenario_id,
            "task_family": str(scenario.get("task_family") or "unknown"),
            "path": _display_path(digest_path, preserve_paths),
            "passed": bool(outcome.get("passed")),
            "score": _score_or_none(outcome.get("score")),
            "task_completion_status": str(outcome.get("task_completion_status") or "unknown"),
            "failed_rule_ids": [str(rule.get("id")) for rule in failed if isinstance(rule, dict) and rule.get("id")],
            "recommended_action_ids": [str(action.get("id")) for action in actions if isinstance(action, dict) and action.get("id")],
            "highest_action_priority": _highest_priority(
                str(action.get("priority") or "medium") for action in actions if isinstance(action, dict)
            ),
        }
    return digests


def _digest_has_actionable_failure(digest: dict[str, Any]) -> bool:
    action_ids = set(digest.get("recommended_action_ids") if isinstance(digest.get("recommended_action_ids"), list) else [])
    return bool({"repair_failed_rules", "block_promotion", "fix_task_completion", "improve_observability"} & action_ids)


def _append_source(item: dict[str, Any], key: str, value: dict[str, Any]) -> None:
    sources = item.setdefault("sources", {})
    values = sources.setdefault(key, [])
    if isinstance(values, list) and value not in values:
        values.append(value)


def _source_list(item: dict[str, Any], key: str) -> list[Any]:
    sources = item.get("sources") if isinstance(item.get("sources"), dict) else {}
    value = sources.get(key)
    return value if isinstance(value, list) else []


def _source_digest(item: dict[str, Any]) -> dict[str, Any] | None:
    sources = item.get("sources") if isinstance(item.get("sources"), dict) else {}
    value = sources.get("run_digest")
    return value if isinstance(value, dict) else None


def _file_record(path: Path, preserve_paths: bool) -> dict[str, Any]:
    exists = path.exists() and path.is_file()
    record: dict[str, Any] = {
        "kind": "file",
        "path": _display_path(path, preserve_paths),
        "exists": exists,
    }
    if exists:
        record["size_bytes"] = path.stat().st_size
        record["sha256"] = _sha256(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            record["schema_version"] = payload.get("schema_version")
            if isinstance(payload.get("passed"), bool):
                record["passed"] = payload["passed"]
    return record


def _dir_record(path: Path, preserve_paths: bool) -> dict[str, Any]:
    exists = path.exists() and path.is_dir()
    record: dict[str, Any] = {
        "kind": "directory",
        "path": _display_path(path, preserve_paths),
        "exists": exists,
    }
    if exists:
        record["entry_count"] = len(list(path.iterdir()))
    return record


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ImprovementPlanError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ImprovementPlanError(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ImprovementPlanError(f"{label} must contain a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path, preserve_paths: bool = False) -> str:
    return str(path if preserve_paths else Path(path.name))


def _count_rows(values: Any) -> list[dict[str, int | str]]:
    counts: dict[str, int] = {}
    for value in values:
        if not isinstance(value, str) or not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return [{"id": key, "count": counts[key]} for key in sorted(counts)]


def _count_from_rows(rows: Any, row_id: str) -> int:
    if not isinstance(rows, list):
        return 0
    for row in rows:
        if isinstance(row, dict) and row.get("id") == row_id:
            count = row.get("count")
            if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                return count
    return 0


def _highest_priority(values: Any) -> str:
    priorities = [_priority(value) for value in values]
    if not priorities:
        return "medium"
    return min(priorities, key=lambda priority: PRIORITY_RANK[priority])


def _priority(value: str) -> str:
    return value if value in PRIORITY_RANK else "medium"


def _non_negative_int(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return max(0, value)
    return 0


def _score_or_none(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 100:
        return value
    return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item]


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]
