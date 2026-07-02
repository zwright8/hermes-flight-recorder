"""Governance-ready summaries across held-out eval artifacts."""

from __future__ import annotations

import json
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
