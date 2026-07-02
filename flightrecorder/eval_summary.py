"""Governance-ready summaries across held-out eval artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVAL_SUMMARY_SCHEMA_VERSION = "hfr.eval_summary.v1"
COMPARE_EXPORT_SCHEMA_VERSION = "hfr.compare_rl.manifest.v1"
COMPARE_GATE_SCHEMA_VERSION = "hfr.compare_gate.v1"
RUN_SUITE_SCHEMA_VERSION = "hfr.run_suite.v1"

_SCENARIO_BLOCKING_STATUSES = {"missing_suite_summaries", "mismatched", "empty"}


class EvalSummaryError(ValueError):
    """Raised when a governance eval summary cannot be built."""


@dataclass(frozen=True)
class LabeledPath:
    label: str
    path: Path


def build_eval_summary(
    *,
    suite_summary_specs: list[str | Path] | None = None,
    compare_export_specs: list[str | Path] | None = None,
    compare_gate_specs: list[str | Path] | None = None,
    external_adapter_plan_specs: list[str | Path] | None = None,
    preserve_paths: bool = False,
) -> dict[str, Any]:
    """Build a single artifact Governance can consume without reinterpreting raw evals."""
    suite_specs = [_labeled_path(spec) for spec in suite_summary_specs or []]
    compare_specs = [_labeled_path(spec) for spec in compare_export_specs or []]
    gate_specs = [_labeled_path(spec) for spec in compare_gate_specs or []]
    adapter_specs = [_labeled_path(spec) for spec in external_adapter_plan_specs or []]
    if not suite_specs and not compare_specs and not gate_specs and not adapter_specs:
        raise EvalSummaryError("At least one eval artifact source is required")

    arms = [_suite_arm(spec, preserve_paths) for spec in suite_specs]
    heldout = _heldout_scenario_summary(arms)
    comparisons = [_compare_export(spec, heldout, preserve_paths) for spec in compare_specs]
    gates = [_compare_gate(spec, preserve_paths) for spec in gate_specs]
    external_adapters = [_external_adapter_plan(spec, preserve_paths) for spec in adapter_specs]
    repair_curriculum = _repair_curriculum(arms, comparisons, gates, external_adapters)
    risks = _risks(arms, heldout, comparisons, gates, external_adapters)
    passed = not risks
    return {
        "schema_version": EVAL_SUMMARY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "governance_ready": passed,
        "arm_count": len(arms),
        "comparison_count": len(comparisons),
        "gate_count": len(gates),
        "external_adapter_plan_count": len(external_adapters),
        "heldout_scenarios": heldout,
        "arms": arms,
        "comparisons": comparisons,
        "compare_gates": gates,
        "external_adapter_plans": external_adapters,
        "repair_curriculum": repair_curriculum,
        "risks": risks,
        "conclusion": _conclusion(passed, risks, heldout, comparisons),
    }


def _suite_arm(spec: LabeledPath, preserve_paths: bool) -> dict[str, Any]:
    summary = _read_object(spec.path, "suite summary")
    runs = summary.get("runs") if isinstance(summary.get("runs"), list) else []
    scenario_ids = sorted({str(run.get("scenario_id")) for run in runs if isinstance(run, dict) and run.get("scenario_id")})
    duplicate_count = len([run for run in runs if isinstance(run, dict) and run.get("scenario_id")]) - len(scenario_ids)
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    validation = summary.get("validation") if isinstance(summary.get("validation"), dict) else None
    blocking_reasons: list[str] = []
    if summary.get("schema_version") != RUN_SUITE_SCHEMA_VERSION:
        blocking_reasons.append("invalid_suite_summary_schema")
    if not scenario_ids:
        blocking_reasons.append("empty_suite_summary")
    if int(summary.get("error_count", 0) or 0) > 0:
        blocking_reasons.append("suite_summary_errors")
    if validation is not None and validation.get("passed") is not True:
        blocking_reasons.append("suite_summary_validation_failed")
    if duplicate_count > 0:
        blocking_reasons.append("duplicate_scenario_ids")
    operational_metrics = _operational_metrics(summary, runs)
    return {
        "label": spec.label,
        "path": _display_path(spec.path, preserve_paths),
        "schema_version": summary.get("schema_version"),
        "scenario_count": len(scenario_ids),
        "scenario_ids": scenario_ids,
        "total": int(summary.get("total", len(runs)) or 0),
        "passed": int(summary.get("passed", 0) or 0),
        "failed": int(summary.get("failed", 0) or 0),
        "error_count": int(summary.get("error_count", 0) or 0),
        "pass_rate": metrics.get("pass_rate"),
        "average_score": metrics.get("average_score"),
        "failed_rule_counts": _count_rows(metrics.get("failed_rule_counts")),
        "critical_failure_counts": _count_rows(metrics.get("critical_failure_counts")),
        "operational_metrics": operational_metrics,
        "validation": _validation_summary(validation),
        "blocking_reasons": blocking_reasons,
    }


def _heldout_scenario_summary(arms: list[dict[str, Any]]) -> dict[str, Any]:
    if not arms:
        return {
            "status": "missing_suite_summaries",
            "identical": False,
            "cross_arm_claims_allowed": False,
            "scenario_count": 0,
            "scenario_ids": [],
            "arms": [],
            "mismatches": [],
            "blocking_reasons": ["missing_suite_summaries"],
        }

    by_arm = [{"label": arm["label"], "scenario_ids": arm["scenario_ids"], "scenario_count": arm["scenario_count"]} for arm in arms]
    reference = arms[0]["scenario_ids"]
    mismatches = []
    for arm in arms[1:]:
        current = arm["scenario_ids"]
        if current != reference:
            mismatches.append(
                {
                    "label": arm["label"],
                    "missing_from_arm": sorted(set(reference) - set(current)),
                    "extra_in_arm": sorted(set(current) - set(reference)),
                }
            )
    empty = any(not arm["scenario_ids"] for arm in arms)
    identical = len(arms) > 1 and not empty and not mismatches
    if empty:
        status = "empty"
        blocking_reasons = ["empty_heldout_scenario_set"]
    elif len(arms) == 1:
        status = "single_arm"
        blocking_reasons = []
    elif identical:
        status = "identical"
        blocking_reasons = []
    else:
        status = "mismatched"
        blocking_reasons = ["heldout_scenario_set_mismatch"]
    return {
        "status": status,
        "identical": identical,
        "cross_arm_claims_allowed": identical,
        "scenario_count": len(reference) if identical or len(arms) == 1 else len(set.intersection(*(set(arm["scenario_ids"]) for arm in arms))),
        "scenario_ids": reference if identical or len(arms) == 1 else sorted(set.intersection(*(set(arm["scenario_ids"]) for arm in arms))),
        "arms": by_arm,
        "mismatches": mismatches,
        "blocking_reasons": blocking_reasons,
    }


def _compare_export(spec: LabeledPath, heldout: dict[str, Any], preserve_paths: bool) -> dict[str, Any]:
    manifest_path = spec.path / "manifest.json" if spec.path.is_dir() else spec.path
    manifest = _read_object(manifest_path, "compare export manifest")
    missing_in_candidate = _string_list(manifest.get("missing_in_candidate"))
    new_in_candidate = _string_list(manifest.get("new_in_candidate"))
    contract_drift_count = _int_value(manifest.get("contract_drift_count"))
    unverified_contract_count = _int_value(manifest.get("unverified_contract_count"))
    pair_count = _int_value(manifest.get("pair_count"))

    claim_blockers: list[str] = []
    if manifest.get("schema_version") != COMPARE_EXPORT_SCHEMA_VERSION:
        claim_blockers.append("invalid_compare_export_schema")
    if not heldout.get("cross_arm_claims_allowed"):
        claim_blockers.append(_heldout_blocker(heldout))
    if missing_in_candidate or new_in_candidate:
        claim_blockers.append("compare_manifest_scenario_set_mismatch")
    if contract_drift_count > 0:
        claim_blockers.append("contract_fingerprint_drift")
    if unverified_contract_count > 0:
        claim_blockers.append("contract_fingerprints_unverified")
    if pair_count == 0:
        claim_blockers.append("no_comparison_pairs")

    raw_movement = {
        "pair_count": pair_count,
        "candidate_win_count": _int_value(manifest.get("candidate_win_count")),
        "baseline_win_count": _int_value(manifest.get("baseline_win_count")),
        "candidate_win_scenarios": _string_list(manifest.get("candidate_win_scenarios")),
        "baseline_win_scenarios": _string_list(manifest.get("baseline_win_scenarios")),
        "task_completion_improvement_count": _int_value(manifest.get("task_completion_improvement_count")),
        "task_completion_regression_count": _int_value(manifest.get("task_completion_regression_count")),
        "task_completion_improvement_scenarios": _string_list(manifest.get("task_completion_improvement_scenarios")),
        "task_completion_regression_scenarios": _string_list(manifest.get("task_completion_regression_scenarios")),
        "fixed_rule_counts": _count_mapping(manifest.get("fixed_rule_counts")),
        "regressed_rule_counts": _count_mapping(manifest.get("regressed_rule_counts")),
        "new_critical_failure_counts": _count_mapping(manifest.get("new_critical_failure_counts")),
        "contract_drift_count": contract_drift_count,
        "unverified_contract_count": unverified_contract_count,
        "skipped_pair_count": _int_value(manifest.get("skipped_pair_count")),
        "missing_in_candidate": missing_in_candidate,
        "new_in_candidate": new_in_candidate,
    }
    readiness_blockers = list(claim_blockers)
    if raw_movement["baseline_win_count"] > 0:
        readiness_blockers.append("baseline_wins_present")
    if raw_movement["task_completion_regression_count"] > 0:
        readiness_blockers.append("task_completion_regressions_present")
    if raw_movement["new_critical_failure_counts"]:
        readiness_blockers.append("new_critical_failures_present")

    claims_allowed = not claim_blockers
    return {
        "label": spec.label,
        "path": _display_path(spec.path, preserve_paths),
        "manifest": _display_path(manifest_path, preserve_paths),
        "schema_version": manifest.get("schema_version"),
        "claims_allowed": claims_allowed,
        "passed": not readiness_blockers,
        "blocking_reasons": readiness_blockers,
        "raw_movement": raw_movement,
        "governance_claims": _governance_claims(raw_movement, claims_allowed, claim_blockers),
    }


def _compare_gate(spec: LabeledPath, preserve_paths: bool) -> dict[str, Any]:
    gate = _read_object(spec.path, "compare gate")
    failed_checks = [
        check
        for check in gate.get("checks", [])
        if isinstance(check, dict) and check.get("passed") is not True
    ]
    blocking_reasons: list[str] = []
    if gate.get("schema_version") != COMPARE_GATE_SCHEMA_VERSION:
        blocking_reasons.append("invalid_compare_gate_schema")
    if gate.get("passed") is not True:
        blocking_reasons.append("compare_gate_failed")
    return {
        "label": spec.label,
        "path": _display_path(spec.path, preserve_paths),
        "schema_version": gate.get("schema_version"),
        "passed": gate.get("passed") is True,
        "check_count": _int_value(gate.get("check_count")),
        "failed_check_count": _int_value(gate.get("failed_check_count")),
        "failed_checks": [
            {
                "id": str(check.get("id") or "unknown"),
                "summary": str(check.get("summary") or ""),
                "scope": check.get("scope") if isinstance(check.get("scope"), dict) else None,
            }
            for check in failed_checks
        ],
        "blocking_reasons": blocking_reasons,
    }


def _external_adapter_plan(spec: LabeledPath, preserve_paths: bool) -> dict[str, Any]:
    plan = _read_object(spec.path, "external adapter plan")
    ready = plan.get("ready") is True
    return {
        "label": spec.label,
        "path": _display_path(spec.path, preserve_paths),
        "schema_version": plan.get("schema_version"),
        "ready": ready,
        "adapter_count": _int_value(plan.get("adapter_count")),
        "ready_adapter_count": _int_value(plan.get("ready_adapter_count")),
        "blocking_reasons": [] if ready else _string_list(plan.get("blocking_reasons")) or ["external_adapter_plan_not_ready"],
    }


def _repair_curriculum(
    arms: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    gates: list[dict[str, Any]],
    external_adapters: list[dict[str, Any]],
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for arm in arms:
        items.extend(_arm_work_items(arm))
    for comparison in comparisons:
        items.extend(_comparison_work_items(comparison))
    for gate in gates:
        items.extend(_gate_work_items(gate))
    for adapter in external_adapters:
        items.extend(_external_adapter_work_items(adapter))
    finalized = _finalize_work_items(items)
    return {
        "work_item_count": len(finalized),
        "critical_work_item_count": sum(1 for item in finalized if item["priority"] == "critical"),
        "priority_counts": _value_count_rows(item["priority"] for item in finalized),
        "category_counts": _value_count_rows(item["category"] for item in finalized),
        "items": finalized,
        "notes": [
            "Repair/curriculum items are derived from eval artifacts; they do not approve promotion.",
            "Use these items to route scenario repair, candidate repair, curriculum generation, or eval-harness follow-up.",
        ],
    }


def _arm_work_items(arm: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row in arm.get("critical_failure_counts", []):
        if not isinstance(row, dict):
            continue
        rule_id = str(row.get("id") or "")
        count = _int_value(row.get("count"))
        if rule_id and count > 0:
            items.append(
                _work_item(
                    category="repair",
                    priority="critical",
                    source="suite_summary",
                    label=str(arm.get("label") or "arm"),
                    reason="critical_failure",
                    rule_id=rule_id,
                    count=count,
                    summary=f"Suite arm {arm.get('label') or 'arm'} has {count} critical failure(s) for rule {rule_id}.",
                    suggested_action="Inspect failed runs for this arm and repair the model behavior or scenario contract before promotion.",
                )
            )
    critical_rules = {str(row.get("id") or "") for row in arm.get("critical_failure_counts", []) if isinstance(row, dict)}
    for row in arm.get("failed_rule_counts", []):
        if not isinstance(row, dict):
            continue
        rule_id = str(row.get("id") or "")
        count = _int_value(row.get("count"))
        if rule_id and count > 0 and rule_id not in critical_rules:
            items.append(
                _work_item(
                    category="curriculum",
                    priority="high",
                    source="suite_summary",
                    label=str(arm.get("label") or "arm"),
                    reason="failed_rule",
                    rule_id=rule_id,
                    count=count,
                    summary=f"Suite arm {arm.get('label') or 'arm'} has {count} failed rule occurrence(s) for {rule_id}.",
                    suggested_action="Prioritize curriculum or scenario repair for this repeated failed-rule pattern.",
                )
            )
    return items


def _comparison_work_items(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    raw = comparison.get("raw_movement") if isinstance(comparison.get("raw_movement"), dict) else {}
    label = str(comparison.get("label") or "comparison")
    items: list[dict[str, Any]] = []
    for scenario_id in _string_list(raw.get("baseline_win_scenarios")):
        items.append(
            _work_item(
                category="repair",
                priority="high",
                source="compare_export",
                label=label,
                reason="baseline_win",
                scenario_id=scenario_id,
                summary=f"Candidate lost to baseline on held-out scenario {scenario_id}.",
                suggested_action="Replay baseline and candidate traces, then repair the candidate behavior before using this movement for promotion.",
            )
        )
    for scenario_id in _string_list(raw.get("task_completion_regression_scenarios")):
        items.append(
            _work_item(
                category="repair",
                priority="critical",
                source="compare_export",
                label=label,
                reason="task_completion_regression",
                scenario_id=scenario_id,
                summary=f"Candidate regressed task completion on held-out scenario {scenario_id}.",
                suggested_action="Treat this as a blocking candidate repair until task completion recovers on the identical held-out scenario.",
            )
        )
    for rule_id, count in _count_mapping(raw.get("regressed_rule_counts")).items():
        if count > 0:
            items.append(
                _work_item(
                    category="curriculum",
                    priority="high",
                    source="compare_export",
                    label=label,
                    reason="regressed_rule",
                    rule_id=rule_id,
                    count=count,
                    summary=f"Rule {rule_id} regressed in {count} comparison pair(s).",
                    suggested_action="Generate repair examples or curriculum focused on this regressed rule before rerunning held-out evals.",
                )
            )
    for rule_id, count in _count_mapping(raw.get("new_critical_failure_counts")).items():
        if count > 0:
            items.append(
                _work_item(
                    category="repair",
                    priority="critical",
                    source="compare_export",
                    label=label,
                    reason="new_critical_failure",
                    rule_id=rule_id,
                    count=count,
                    summary=f"Rule {rule_id} introduced {count} new critical failure(s).",
                    suggested_action="Block promotion and repair the critical failure before rerunning the identical held-out eval set.",
                )
            )
    for reason in _string_list(comparison.get("blocking_reasons")):
        if reason in {"baseline_wins_present", "task_completion_regressions_present", "new_critical_failures_present"}:
            continue
        items.append(
            _work_item(
                category="eval_harness",
                priority=_blocking_reason_priority(reason),
                source="compare_export",
                label=label,
                reason=reason,
                summary=f"Comparison {label} is blocked by {reason}.",
                suggested_action="Resolve the comparison blocker before treating raw eval movement as a governance claim.",
            )
        )
    return items


def _gate_work_items(gate: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    label = str(gate.get("label") or "gate")
    for check in gate.get("failed_checks", []):
        if not isinstance(check, dict):
            continue
        check_id = str(check.get("id") or "unknown_check")
        items.append(
            _work_item(
                category="eval_gate",
                priority="high",
                source="compare_gate",
                label=label,
                reason=check_id,
                summary=str(check.get("summary") or f"Compare gate check {check_id} failed."),
                suggested_action="Resolve the failed compare gate check, then regenerate the eval summary for Governance.",
            )
        )
    return items


def _external_adapter_work_items(adapter: dict[str, Any]) -> list[dict[str, Any]]:
    label = str(adapter.get("label") or "external_adapter_plan")
    return [
        _work_item(
            category="eval_harness",
            priority="medium",
            source="external_adapter_plan",
            label=label,
            reason=reason,
            summary=f"External adapter plan {label} is blocked by {reason}.",
            suggested_action="Provide the missing adapter input or dependency before making external eval claims.",
        )
        for reason in _string_list(adapter.get("blocking_reasons"))
    ]


def _work_item(
    *,
    category: str,
    priority: str,
    source: str,
    label: str,
    reason: str,
    summary: str,
    suggested_action: str,
    scenario_id: str | None = None,
    rule_id: str | None = None,
    count: int | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "category": category,
        "priority": priority,
        "source": source,
        "label": label,
        "reason": reason,
        "summary": summary,
        "suggested_action": suggested_action,
    }
    if scenario_id:
        item["scenario_id"] = scenario_id
    if rule_id:
        item["rule_id"] = rule_id
    if count is not None:
        item["count"] = count
    item["work_item_id"] = _work_item_id(item)
    return item


def _work_item_id(item: dict[str, Any]) -> str:
    payload = json.dumps(item, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return "eval-" + hashlib.sha256(payload).hexdigest()[:16]


def _finalize_work_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for item in items:
        deduped.setdefault(str(item.get("work_item_id") or ""), item)
    return sorted(
        deduped.values(),
        key=lambda item: (
            {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(str(item.get("priority") or ""), 99),
            str(item.get("category") or ""),
            str(item.get("source") or ""),
            str(item.get("label") or ""),
            str(item.get("reason") or ""),
            str(item.get("scenario_id") or ""),
            str(item.get("rule_id") or ""),
        ),
    )


def _value_count_rows(values: Any) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for value in values:
        if not isinstance(value, str) or not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return [{"id": key, "count": counts[key]} for key in sorted(counts)]


def _blocking_reason_priority(reason: str) -> str:
    if reason in {
        "heldout_scenario_set_mismatch",
        "compare_manifest_scenario_set_mismatch",
        "empty_heldout_scenario_set",
        "missing_suite_summaries",
    }:
        return "critical"
    return "high"


def _governance_claims(raw_movement: dict[str, Any], claims_allowed: bool, blockers: list[str]) -> dict[str, Any]:
    if not claims_allowed:
        return {
            "candidate_win_count": 0,
            "candidate_win_scenarios": [],
            "task_completion_improvement_count": 0,
            "task_completion_improvement_scenarios": [],
            "suppressed_raw_claims": True,
            "suppression_reasons": blockers,
        }
    return {
        "candidate_win_count": raw_movement["candidate_win_count"],
        "candidate_win_scenarios": raw_movement["candidate_win_scenarios"],
        "task_completion_improvement_count": raw_movement["task_completion_improvement_count"],
        "task_completion_improvement_scenarios": raw_movement["task_completion_improvement_scenarios"],
        "suppressed_raw_claims": False,
        "suppression_reasons": [],
    }


def _risks(
    arms: list[dict[str, Any]],
    heldout: dict[str, Any],
    comparisons: list[dict[str, Any]],
    gates: list[dict[str, Any]],
    external_adapters: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    risks: list[dict[str, Any]] = []
    for arm in arms:
        for reason in arm["blocking_reasons"]:
            risks.append({"source": "suite_summary", "label": arm["label"], "reason": reason})
    if comparisons and heldout["status"] in _SCENARIO_BLOCKING_STATUSES:
        for reason in heldout["blocking_reasons"]:
            risks.append({"source": "heldout_scenarios", "reason": reason})
    for comparison in comparisons:
        for reason in comparison["blocking_reasons"]:
            risks.append({"source": "compare_export", "label": comparison["label"], "reason": reason})
    for gate in gates:
        for reason in gate["blocking_reasons"]:
            risks.append({"source": "compare_gate", "label": gate["label"], "reason": reason})
    for plan in external_adapters:
        for reason in plan["blocking_reasons"]:
            risks.append({"source": "external_adapter_plan", "label": plan["label"], "reason": reason})
    return _dedupe_risks(risks)


def _conclusion(
    passed: bool,
    risks: list[dict[str, Any]],
    heldout: dict[str, Any],
    comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    if passed:
        return {
            "status": "ready",
            "recommendation": "Governance may consume this summary directly; cross-arm claims are allowed only for the listed identical held-out scenarios.",
        }
    if comparisons and not heldout.get("cross_arm_claims_allowed"):
        recommendation = "Do not promote candidate wins or improvements until all arms use the identical held-out scenario list."
    else:
        recommendation = "Do not promote until the listed eval summary risks are resolved."
    return {"status": "blocked", "recommendation": recommendation, "risk_count": len(risks)}


def _labeled_path(spec: str | Path) -> LabeledPath:
    text = str(spec)
    if "=" in text:
        label, raw_path = text.split("=", 1)
        if label and raw_path:
            return LabeledPath(label=label, path=Path(raw_path))
    path = Path(text)
    return LabeledPath(label=_default_label(path), path=path)


def _default_label(path: Path) -> str:
    if path.name == "suite_summary.json" and path.parent.name:
        return path.parent.name
    if path.name == "manifest.json" and path.parent.name:
        return path.parent.name
    return path.stem or path.name or "eval"


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvalSummaryError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EvalSummaryError(f"Invalid JSON in {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvalSummaryError(f"{label} must be a JSON object: {path}")
    return payload


def _heldout_blocker(heldout: dict[str, Any]) -> str:
    status = str(heldout.get("status") or "unknown")
    if status == "missing_suite_summaries":
        return "missing_suite_summaries"
    if status == "single_arm":
        return "single_arm_no_cross_arm_claims"
    if status == "empty":
        return "empty_heldout_scenario_set"
    return "heldout_scenario_set_mismatch"


def _validation_summary(validation: Any) -> dict[str, Any] | None:
    if not isinstance(validation, dict):
        return None
    return {
        "passed": validation.get("passed"),
        "target_count": validation.get("target_count"),
        "error_count": validation.get("error_count"),
        "warning_count": validation.get("warning_count"),
    }


def _count_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows = []
    for item in value:
        if not isinstance(item, dict):
            continue
        row_id = item.get("id")
        count = item.get("count")
        if isinstance(row_id, str) and isinstance(count, int) and not isinstance(count, bool):
            rows.append({"id": row_id, "count": count})
    return rows


def _count_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    counts: dict[str, int] = {}
    for key, count in value.items():
        if isinstance(key, str) and isinstance(count, int) and not isinstance(count, bool):
            counts[key] = count
    return counts


def _operational_metrics(summary: dict[str, Any], runs: list[Any]) -> dict[str, Any]:
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    run_rows = [run for run in runs if isinstance(run, dict)]
    return {
        "cost": _cost_metrics(metrics, run_rows),
        "latency": _latency_metrics(metrics, run_rows),
        "tokens": _token_metrics(metrics, run_rows),
        "task_completion": _task_completion_metrics(metrics, run_rows),
    }


def _cost_metrics(metrics: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, Any]:
    costs = [_cost_usd(run) for run in runs]
    known = [value for value in costs if value is not None]
    if known:
        total = sum(known)
        source = "run_rows"
    else:
        total = _first_number(metrics, "total_cost_usd", "cost_usd")
        source = "suite_metrics" if total is not None else "missing"
    return {
        "total_usd": _round_number(total),
        "known_run_count": len(known),
        "missing_run_count": max(0, len(runs) - len(known)),
        "source": source,
    }


def _latency_metrics(metrics: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [_latency_ms(run) for run in runs]
    known = [value for value in latencies if value is not None]
    if known:
        source = "run_rows"
        average = sum(known) / len(known)
        p50 = _percentile(known, 0.50)
        p95 = _percentile(known, 0.95)
        maximum = max(known)
    else:
        source = "suite_metrics" if _first_number(metrics, "average_latency_ms", "latency_ms") is not None else "missing"
        average = _first_number(metrics, "average_latency_ms", "latency_ms")
        p50 = _first_number(metrics, "p50_latency_ms", "latency_p50_ms")
        p95 = _first_number(metrics, "p95_latency_ms", "latency_p95_ms")
        maximum = _first_number(metrics, "max_latency_ms", "latency_max_ms")
    return {
        "average_ms": _round_number(average),
        "p50_ms": _round_number(p50),
        "p95_ms": _round_number(p95),
        "max_ms": _round_number(maximum),
        "known_run_count": len(known),
        "missing_run_count": max(0, len(runs) - len(known)),
        "source": source,
    }


def _token_metrics(metrics: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, Any]:
    usages = [_token_usage(run) for run in runs]
    known = [usage for usage in usages if any(value is not None for value in usage.values())]
    source = "run_rows" if known else "suite_metrics" if _metrics_has_token_usage(metrics) else "missing"
    prompt_tokens = _sum_ints(usage["prompt_tokens"] for usage in known) if known else _first_int(metrics, "prompt_tokens", "input_tokens")
    completion_tokens = (
        _sum_ints(usage["completion_tokens"] for usage in known)
        if known
        else _first_int(metrics, "completion_tokens", "output_tokens")
    )
    total_tokens = _sum_ints(usage["total_tokens"] for usage in known) if known else _first_int(metrics, "total_tokens")
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "known_run_count": len(known),
        "missing_run_count": max(0, len(runs) - len(known)),
        "source": source,
    }


def _task_completion_metrics(metrics: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, Any]:
    task_rows = [_task_completion_record(run) for run in runs]
    known = [row for row in task_rows if row is not None]
    if known:
        statuses = [str(row.get("status") or "not_applicable") for row in known]
        passed_values = [row.get("passed") for row in known if isinstance(row.get("passed"), bool)]
        passed_count = sum(1 for value in passed_values if value is True)
        failed_count = sum(1 for value in passed_values if value is False)
        return {
            "configured_count": len(known),
            "complete_count": statuses.count("complete"),
            "incomplete_count": statuses.count("incomplete"),
            "not_applicable_count": statuses.count("not_applicable"),
            "unknown_count": max(0, len(runs) - len(known)),
            "passed_count": passed_count,
            "failed_count": failed_count,
            "pass_rate": _round_number(passed_count / len(passed_values)) if passed_values else None,
            "source": "run_rows",
        }
    metric_task = metrics.get("task_completion") if isinstance(metrics.get("task_completion"), dict) else {}
    return {
        "configured_count": _first_int(metric_task, "configured_count") or 0,
        "complete_count": _first_int(metric_task, "complete_count") or 0,
        "incomplete_count": _first_int(metric_task, "incomplete_count") or 0,
        "not_applicable_count": _first_int(metric_task, "not_applicable_count") or 0,
        "unknown_count": len(runs),
        "passed_count": _first_int(metric_task, "passed_count") or 0,
        "failed_count": _first_int(metric_task, "failed_count") or 0,
        "pass_rate": _first_number(metric_task, "pass_rate"),
        "source": "suite_metrics" if metric_task else "missing",
    }


def _cost_usd(run: dict[str, Any]) -> float | None:
    value = _first_number(run, "cost_usd", "total_cost_usd")
    if value is not None:
        return value
    for field_name in ("cost", "usage"):
        nested = run.get(field_name)
        if isinstance(nested, dict):
            value = _first_number(nested, "usd", "cost_usd", "total_cost_usd")
            if value is not None:
                return value
    return None


def _latency_ms(run: dict[str, Any]) -> float | None:
    return _first_number(run, "latency_ms", "duration_ms", "elapsed_ms", "runtime_ms")


def _token_usage(run: dict[str, Any]) -> dict[str, int | None]:
    usage = run.get("usage") if isinstance(run.get("usage"), dict) else {}
    token_usage = run.get("token_usage") if isinstance(run.get("token_usage"), dict) else {}
    sources = (run, usage, token_usage)
    return {
        "prompt_tokens": _first_int_from_sources(sources, "prompt_tokens", "input_tokens"),
        "completion_tokens": _first_int_from_sources(sources, "completion_tokens", "output_tokens"),
        "total_tokens": _first_int_from_sources(sources, "total_tokens"),
    }


def _task_completion_record(run: dict[str, Any]) -> dict[str, Any] | None:
    task = run.get("task_completion")
    if isinstance(task, dict):
        record = _task_completion_from_values(task.get("status"), task.get("passed"))
        if record is not None:
            return record
    record = _task_completion_from_values(run.get("task_completion_status"), run.get("task_completion_passed"))
    if record is not None:
        return record
    outcome = run.get("outcome") if isinstance(run.get("outcome"), dict) else {}
    return _task_completion_from_values(outcome.get("task_completion_status"), outcome.get("task_completion_passed"))


def _task_completion_from_values(status: Any, passed: Any) -> dict[str, Any] | None:
    if isinstance(status, str) or isinstance(passed, bool):
        return {
            "status": status if status in {"complete", "incomplete", "not_applicable"} else "not_applicable",
            "passed": passed if isinstance(passed, bool) else None,
        }
    return None


def _first_number(mapping: dict[str, Any], *field_names: str) -> float | None:
    for field_name in field_names:
        value = mapping.get(field_name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
            return float(value)
    return None


def _first_int(mapping: dict[str, Any], *field_names: str) -> int | None:
    for field_name in field_names:
        value = mapping.get(field_name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _first_int_from_sources(sources: tuple[dict[str, Any], ...], *field_names: str) -> int | None:
    for source in sources:
        value = _first_int(source, *field_names)
        if value is not None:
            return value
    return None


def _metrics_has_token_usage(metrics: dict[str, Any]) -> bool:
    return any(_first_int(metrics, field_name) is not None for field_name in ("prompt_tokens", "input_tokens", "completion_tokens", "output_tokens", "total_tokens"))


def _sum_ints(values: Any) -> int | None:
    known = [value for value in values if isinstance(value, int) and not isinstance(value, bool) and value >= 0]
    return sum(known) if known else None


def _round_number(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if percentile == 0.50 and len(ordered) % 2 == 0:
        middle = len(ordered) // 2
        return (ordered[middle - 1] + ordered[middle]) / 2
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0


def _display_path(path: Path, preserve_paths: bool) -> str:
    if preserve_paths:
        return str(path)
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except (OSError, ValueError):
        return str(path)


def _dedupe_risks(risks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for risk in risks:
        key = (str(risk.get("source") or ""), str(risk.get("label") or ""), str(risk.get("reason") or ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(risk)
    return deduped
