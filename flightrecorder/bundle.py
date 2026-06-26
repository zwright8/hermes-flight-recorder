"""Evidence-bundle handoff summaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

EVIDENCE_BUNDLE_SCHEMA_VERSION = "hfr.evidence_bundle.v1"


class EvidenceBundleError(ValueError):
    """Raised when an evidence bundle cannot be summarized."""


def build_evidence_bundle(
    *,
    out_path: str | Path,
    runs_dir: str | Path | None = None,
    suite_summary_path: str | Path | None = None,
    scenario_quality_path: str | Path | None = None,
    evidence_coverage_path: str | Path | None = None,
    trace_observability_path: str | Path | None = None,
    repair_queue_path: str | Path | None = None,
    validation_path: str | Path | None = None,
    training_export_dir: str | Path | None = None,
    compare_export_dir: str | Path | None = None,
    review_export_dir: str | Path | None = None,
    reviewed_export_dir: str | Path | None = None,
    review_calibration_path: str | Path | None = None,
    gate_paths: list[str | Path] | None = None,
    preserve_paths: bool = False,
) -> dict[str, Any]:
    """Build a compact handoff manifest over existing evidence artifacts."""
    artifacts: dict[str, Any] = {}
    checks: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}

    if runs_dir is not None:
        runs_path = Path(runs_dir)
        artifacts["runs_dir"] = _dir_record(runs_path, preserve_paths)
        _add_presence_check(checks, "runs_dir_exists", runs_path.exists() and runs_path.is_dir(), {"artifact": "runs_dir"})

    if suite_summary_path is not None:
        summary_path = Path(suite_summary_path)
        suite_summary = _read_json_artifact(summary_path, artifacts, "suite_summary", preserve_paths)
        _summarize_suite_summary(suite_summary, metrics, checks)

    if scenario_quality_path is not None:
        scenario_quality = _read_json_artifact(Path(scenario_quality_path), artifacts, "scenario_quality", preserve_paths)
        _summarize_boolean_artifact(
            scenario_quality,
            metrics,
            checks,
            artifact_name="scenario_quality",
            metric_prefix="scenario_quality",
            metric_fields=(
                "average_contract_score",
                "min_contract_score",
                "observable_scenario_rate",
                "weak_scenario_count",
                "final_only_scenario_count",
                "missing_trace_count",
                "missing_state_count",
                "risk_counts",
            ),
        )

    if evidence_coverage_path is not None:
        evidence_coverage = _read_json_artifact(Path(evidence_coverage_path), artifacts, "evidence_coverage", preserve_paths)
        _summarize_boolean_artifact(
            evidence_coverage,
            metrics,
            checks,
            artifact_name="evidence_coverage",
            metric_prefix="evidence_coverage",
            metric_fields=(
                "failed_rule_evidence_rate",
                "critical_failed_rule_evidence_rate",
                "failed_rules_without_evidence",
                "critical_failed_rules_without_evidence",
                "event_evidence_ref_count",
            ),
        )

    if trace_observability_path is not None:
        trace_observability = _read_json_artifact(Path(trace_observability_path), artifacts, "trace_observability", preserve_paths)
        _summarize_boolean_artifact(
            trace_observability,
            metrics,
            checks,
            artifact_name="trace_observability",
            metric_prefix="trace_observability",
            metric_fields=(
                "run_count",
                "average_event_count",
                "event_type_count",
                "final_answer_rate",
                "tool_or_api_run_rate",
                "empty_final_answer_count",
                "risk_counts",
            ),
        )

    if repair_queue_path is not None:
        repair_queue = _read_json_artifact(Path(repair_queue_path), artifacts, "repair_queue", preserve_paths)
        _summarize_boolean_artifact(
            repair_queue,
            metrics,
            checks,
            artifact_name="repair_queue",
            metric_prefix="repair_queue",
            metric_fields=(
                "item_count",
                "critical_item_count",
                "scenario_count",
                "task_family_count",
                "priority_counts",
                "rule_counts",
                "critical_rule_counts",
            ),
        )

    if validation_path is not None:
        validation = _read_json_artifact(Path(validation_path), artifacts, "validation", preserve_paths)
        _summarize_boolean_artifact(
            validation,
            metrics,
            checks,
            artifact_name="validation",
            metric_prefix="validation",
            metric_fields=("target_count", "error_count", "warning_count"),
            metrics_source="root",
        )

    if training_export_dir is not None:
        training_dir = Path(training_export_dir)
        manifest_path = training_dir / "manifest.json"
        dataset_metrics_path = training_dir / "dataset_metrics.json"
        curriculum_path = training_dir / "curriculum.json"
        artifacts["training_export"] = _dir_record(training_dir, preserve_paths)
        artifacts["training_export_manifest"] = _file_record(manifest_path, preserve_paths)
        artifacts["training_export_dataset_metrics"] = _file_record(dataset_metrics_path, preserve_paths)
        artifacts["training_export_curriculum"] = _file_record(curriculum_path, preserve_paths)
        manifest = _read_optional_json(manifest_path)
        dataset_metrics = _read_optional_json(dataset_metrics_path)
        curriculum = _read_optional_json(curriculum_path)
        _add_presence_check(checks, "training_export_exists", training_dir.exists() and training_dir.is_dir(), {"artifact": "training_export"})
        _add_presence_check(checks, "training_export_manifest_exists", manifest_path.exists() and manifest_path.is_file(), {"artifact": "training_export", "file": "manifest.json"})
        _add_presence_check(checks, "training_export_dataset_metrics_exists", dataset_metrics_path.exists() and dataset_metrics_path.is_file(), {"artifact": "training_export", "file": "dataset_metrics.json"})
        _add_presence_check(checks, "training_export_curriculum_exists", curriculum_path.exists() and curriculum_path.is_file(), {"artifact": "training_export", "file": "curriculum.json"})
        if isinstance(manifest, dict):
            metrics["training_export"] = {
                "episode_count": manifest.get("episode_count"),
                "preference_count": manifest.get("preference_count"),
                "sft_count": manifest.get("sft_count"),
                "dpo_count": manifest.get("dpo_count"),
                "reward_model_count": manifest.get("reward_model_count"),
                "quality_flag_count": manifest.get("quality_flag_count"),
            }
        if isinstance(dataset_metrics, dict):
            metrics.setdefault("training_export", {})["pass_rate"] = dataset_metrics.get("pass_rate")
            metrics.setdefault("training_export", {})["average_score"] = dataset_metrics.get("average_score")
            metrics.setdefault("training_export", {})["quality_flags"] = dataset_metrics.get("quality_flags")
        if isinstance(curriculum, dict):
            metrics.setdefault("training_export", {})["curriculum_failure_mode_count"] = curriculum.get("failure_mode_count")
            metrics.setdefault("training_export", {})["top_curriculum_priorities"] = _top_curriculum_priorities(curriculum)

    if compare_export_dir is not None:
        compare_dir = Path(compare_export_dir)
        artifacts["compare_export"] = _dir_record(compare_dir, preserve_paths)
        manifest = _read_optional_json(compare_dir / "manifest.json")
        _add_presence_check(checks, "compare_export_exists", compare_dir.exists() and compare_dir.is_dir(), {"artifact": "compare_export"})
        if isinstance(manifest, dict):
            metrics["compare_export"] = {
                "pair_count": manifest.get("pair_count"),
                "dpo_count": manifest.get("dpo_count"),
                "candidate_win_count": manifest.get("candidate_win_count"),
                "baseline_win_count": manifest.get("baseline_win_count"),
                "skipped_pair_count": manifest.get("skipped_pair_count"),
            }

    if review_export_dir is not None:
        review_dir = Path(review_export_dir)
        artifacts["review_export"] = _dir_record(review_dir, preserve_paths)
        manifest = _read_optional_json(review_dir / "manifest.json")
        _add_presence_check(checks, "review_export_exists", review_dir.exists() and review_dir.is_dir(), {"artifact": "review_export"})
        if isinstance(manifest, dict):
            metrics["review_export"] = {
                "item_count": manifest.get("item_count"),
                "failed_count": manifest.get("failed_count"),
                "passed_count": manifest.get("passed_count"),
            }

    if reviewed_export_dir is not None:
        reviewed_dir = Path(reviewed_export_dir)
        artifacts["reviewed_export"] = _dir_record(reviewed_dir, preserve_paths)
        manifest = _read_optional_json(reviewed_dir / "manifest.json")
        _add_presence_check(checks, "reviewed_export_exists", reviewed_dir.exists() and reviewed_dir.is_dir(), {"artifact": "reviewed_export"})
        if isinstance(manifest, dict):
            metrics["reviewed_export"] = {
                "reviewed_label_count": manifest.get("reviewed_label_count"),
                "sft_count": manifest.get("sft_count"),
                "dpo_count": manifest.get("dpo_count"),
                "reward_model_count": manifest.get("reward_model_count"),
            }

    if review_calibration_path is not None:
        review_calibration = _read_json_artifact(Path(review_calibration_path), artifacts, "review_calibration", preserve_paths)
        _summarize_boolean_artifact(
            review_calibration,
            metrics,
            checks,
            artifact_name="review_calibration",
            metric_prefix="review_calibration",
            metric_fields=(
                "reviewed_label_count",
                "comparable_label_count",
                "agreement_rate",
                "disagreement_count",
                "false_positive_count",
                "false_negative_count",
            ),
        )

    gate_rows: list[dict[str, Any]] = []
    for index, gate_path in enumerate(gate_paths or []):
        gate_name = f"gate_{index + 1}"
        gate = _read_json_artifact(Path(gate_path), artifacts, gate_name, preserve_paths)
        gate_id = _gate_id(gate, gate_name)
        passed = bool(gate.get("passed")) if isinstance(gate, dict) else False
        gate_rows.append({"id": gate_id, "path": artifacts[gate_name]["path"], "passed": passed})
        _add_presence_check(checks, "gate_passed", passed, {"gate": gate_id})
    if gate_rows:
        metrics["gates"] = gate_rows

    if not artifacts:
        raise EvidenceBundleError("At least one evidence artifact or directory must be provided.")

    failed_checks = sum(1 for check in checks if not check["passed"])
    passed = failed_checks == 0
    readiness = "ready" if passed else "blocked"
    bundle = {
        "schema_version": EVIDENCE_BUNDLE_SCHEMA_VERSION,
        "bundle_path": _display_path(Path(out_path), preserve_paths),
        "passed": passed,
        "readiness": readiness,
        "decision": _decision_summary(readiness, checks, artifacts, metrics),
        "check_count": len(checks),
        "failed_check_count": failed_checks,
        "checks": checks,
        "artifacts": artifacts,
        "metrics": metrics,
        "notes": [
            "Evidence bundles summarize existing artifacts; they do not rescore traces or mutate outputs.",
            "A ready bundle means included gates and readiness artifacts passed, not that the agent is sandboxed or trained.",
        ],
    }
    return bundle


def _decision_summary(
    readiness: str,
    checks: list[dict[str, Any]],
    artifacts: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    blocking_checks = [
        {
            "id": str(check.get("id") or "unknown"),
            "summary": str(check.get("summary") or ""),
            "scope": check.get("scope") if isinstance(check.get("scope"), dict) else {},
        }
        for check in checks
        if check.get("passed") is False
    ]
    gates = metrics.get("gates") if isinstance(metrics.get("gates"), list) else []
    blocking_gates = [
        {"id": str(gate.get("id") or "unknown"), "path": str(gate.get("path") or "")}
        for gate in gates
        if isinstance(gate, dict) and gate.get("passed") is False
    ]
    passed_gate_count = sum(1 for gate in gates if isinstance(gate, dict) and gate.get("passed") is True)
    recommendation = "promote_handoff" if readiness == "ready" else "block_handoff"
    next_actions = _next_actions(blocking_checks, blocking_gates, metrics)
    return {
        "readiness": readiness,
        "recommendation": recommendation,
        "summary": _decision_text(readiness, blocking_checks),
        "blocking_check_count": len(blocking_checks),
        "blocking_checks": blocking_checks,
        "blocking_gates": blocking_gates,
        "next_action_count": len(next_actions),
        "next_actions": next_actions,
        "evidence_artifacts": sorted(artifacts),
        "gate_count": len(gates),
        "passed_gate_count": passed_gate_count,
        "key_metrics": _decision_key_metrics(metrics),
    }


def _decision_text(readiness: str, blocking_checks: list[dict[str, Any]]) -> str:
    if readiness == "ready":
        return "Evidence handoff is ready: all included bundle checks and gates passed."
    if not blocking_checks:
        return "Evidence handoff is blocked."
    first = blocking_checks[0]
    return f"Evidence handoff is blocked by {len(blocking_checks)} check(s); first failure: {first['summary']}"


def _decision_key_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys: dict[str, tuple[str, ...]] = {
        "suite_summary": ("total", "passed", "failed", "pass_rate", "average_score"),
        "scenario_quality": (
            "average_contract_score",
            "min_contract_score",
            "observable_scenario_rate",
            "weak_scenario_count",
            "risk_counts",
        ),
        "evidence_coverage": (
            "failed_rule_evidence_rate",
            "critical_failed_rule_evidence_rate",
            "failed_rules_without_evidence",
            "critical_failed_rules_without_evidence",
        ),
        "trace_observability": (
            "run_count",
            "average_event_count",
            "event_type_count",
            "final_answer_rate",
            "tool_or_api_run_rate",
            "risk_counts",
        ),
        "repair_queue": ("item_count", "critical_item_count", "scenario_count", "task_family_count", "priority_counts", "rule_counts"),
        "validation": ("target_count", "error_count", "warning_count"),
        "training_export": (
            "episode_count",
            "preference_count",
            "dpo_count",
            "pass_rate",
            "average_score",
            "quality_flag_count",
            "curriculum_failure_mode_count",
            "top_curriculum_priorities",
        ),
        "compare_export": ("pair_count", "candidate_win_count", "baseline_win_count", "skipped_pair_count"),
        "review_export": ("item_count", "failed_count", "passed_count"),
        "reviewed_export": ("reviewed_label_count", "sft_count", "dpo_count", "reward_model_count"),
        "review_calibration": ("reviewed_label_count", "comparable_label_count", "agreement_rate", "disagreement_count"),
    }
    summary: dict[str, Any] = {}
    for section, fields in keys.items():
        value = metrics.get(section)
        if not isinstance(value, dict):
            continue
        section_metrics = {field: value.get(field) for field in fields if field in value}
        if section_metrics:
            summary[section] = section_metrics
    gates = metrics.get("gates")
    if isinstance(gates, list):
        summary["gates"] = {
            "total": len(gates),
            "passed": sum(1 for gate in gates if isinstance(gate, dict) and gate.get("passed") is True),
            "failed": sum(1 for gate in gates if isinstance(gate, dict) and gate.get("passed") is False),
        }
    return summary


def _next_actions(
    blocking_checks: list[dict[str, Any]],
    blocking_gates: list[dict[str, str]],
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if blocking_checks:
        actions.append(
            _action(
                "resolve_blocking_checks",
                "critical",
                "evidence_bundle",
                f"Resolve {len(blocking_checks)} blocking bundle check(s) before relying on this handoff.",
                {"blocking_check_count": len(blocking_checks), "first_blocking_check": blocking_checks[0]},
            )
        )
    if blocking_gates:
        actions.append(
            _action(
                "fix_failed_gates",
                "critical",
                "gates",
                f"Fix {len(blocking_gates)} failed gate(s) before promoting this evidence package.",
                {"blocking_gates": blocking_gates},
            )
        )

    suite = metrics.get("suite_summary") if isinstance(metrics.get("suite_summary"), dict) else {}
    failed = _non_negative_int(suite.get("failed"))
    if failed:
        actions.append(
            _action(
                "repair_failed_scenarios",
                "high",
                "suite_summary",
                f"Repair or intentionally rebaseline {failed} failing scenario(s).",
                {
                    "failed": failed,
                    "failed_rule_counts": suite.get("failed_rule_counts") if isinstance(suite.get("failed_rule_counts"), list) else [],
                },
            )
        )
    critical_total = _count_total(suite.get("critical_failure_counts"))
    if critical_total:
        actions.append(
            _action(
                "repair_critical_failures",
                "high",
                "suite_summary",
                f"Resolve {critical_total} critical failure occurrence(s) before treating the candidate as improved.",
                {"critical_failure_counts": suite.get("critical_failure_counts") if isinstance(suite.get("critical_failure_counts"), list) else []},
            )
        )

    scenario_quality = metrics.get("scenario_quality") if isinstance(metrics.get("scenario_quality"), dict) else {}
    weak_scenarios = _non_negative_int(scenario_quality.get("weak_scenario_count"))
    if weak_scenarios:
        actions.append(
            _action(
                "strengthen_weak_scenarios",
                "medium",
                "scenario_quality",
                f"Strengthen {weak_scenarios} weak scenario contract(s) before using them as durable improvement signal.",
                {"weak_scenario_count": weak_scenarios, "risk_counts": _count_rows(scenario_quality.get("risk_counts"))},
            )
        )
    scenario_risks = _count_rows(scenario_quality.get("risk_counts"))
    contract_risk_count = sum(
        scenario_risks.get(risk, 0)
        for risk in (
            "final_only_contract",
            "no_observable_assertions",
            "missing_trace_file",
            "missing_trace_path",
            "required_state_without_snapshot_path",
            "missing_state_file",
        )
    )
    if contract_risk_count:
        actions.append(
            _action(
                "ground_scenario_contracts",
                "medium",
                "scenario_quality",
                f"Ground {contract_risk_count} scenario contract risk(s) in observable trace or state evidence.",
                {"risk_counts": scenario_risks},
            )
        )

    evidence = metrics.get("evidence_coverage") if isinstance(metrics.get("evidence_coverage"), dict) else {}
    missing_failed_refs = _non_negative_int(evidence.get("failed_rules_without_evidence"))
    missing_critical_refs = _non_negative_int(evidence.get("critical_failed_rules_without_evidence"))
    if missing_failed_refs or missing_critical_refs:
        actions.append(
            _action(
                "add_structured_evidence_refs",
                "high" if missing_critical_refs else "medium",
                "evidence_coverage",
                "Add structured evidence refs for failed rules before feeding them to review or training loops.",
                {
                    "failed_rules_without_evidence": missing_failed_refs,
                    "critical_failed_rules_without_evidence": missing_critical_refs,
                    "failed_rule_evidence_rate": evidence.get("failed_rule_evidence_rate"),
                },
            )
        )

    observability = metrics.get("trace_observability") if isinstance(metrics.get("trace_observability"), dict) else {}
    trace_risks = _count_rows(observability.get("risk_counts"))
    trace_risk_count = sum(trace_risks.values())
    if trace_risk_count:
        actions.append(
            _action(
                "improve_trace_observability",
                "medium",
                "trace_observability",
                f"Improve trace capture for {trace_risk_count} trace observability risk occurrence(s).",
                {"risk_counts": trace_risks},
            )
        )

    repair_queue = metrics.get("repair_queue") if isinstance(metrics.get("repair_queue"), dict) else {}
    repair_items = _non_negative_int(repair_queue.get("item_count"))
    critical_repair_items = _non_negative_int(repair_queue.get("critical_item_count"))
    if repair_items:
        actions.append(
            _action(
                "dispatch_repair_queue",
                "critical" if critical_repair_items else "high",
                "repair_queue",
                f"Route {repair_items} failed-rule repair item(s) to the next improvement iteration.",
                {
                    "item_count": repair_items,
                    "critical_item_count": critical_repair_items,
                    "scenario_count": _non_negative_int(repair_queue.get("scenario_count")),
                    "task_family_count": _non_negative_int(repair_queue.get("task_family_count")),
                    "priority_counts": _count_rows(repair_queue.get("priority_counts")),
                    "rule_counts": _count_rows(repair_queue.get("rule_counts")),
                    "critical_rule_counts": _count_rows(repair_queue.get("critical_rule_counts")),
                },
            )
        )

    validation = metrics.get("validation") if isinstance(metrics.get("validation"), dict) else {}
    validation_errors = _non_negative_int(validation.get("error_count"))
    validation_warnings = _non_negative_int(validation.get("warning_count"))
    if validation_errors or validation_warnings:
        actions.append(
            _action(
                "fix_validation_findings",
                "critical" if validation_errors else "medium",
                "validation",
                f"Resolve validation findings before publishing this handoff: {validation_errors} error(s), {validation_warnings} warning(s).",
                {"error_count": validation_errors, "warning_count": validation_warnings},
            )
        )

    training = metrics.get("training_export") if isinstance(metrics.get("training_export"), dict) else {}
    quality_flags = [flag for flag in training.get("quality_flags", []) if isinstance(flag, dict)]
    if quality_flags:
        actions.append(
            _action(
                "review_training_quality_flags",
                "high" if any(flag.get("severity") == "error" for flag in quality_flags) else "medium",
                "training_export",
                f"Review {len(quality_flags)} training-export quality flag(s) before using rows for model updates.",
                {
                    "quality_flags": [
                        {
                            "id": str(flag.get("id") or "unknown"),
                            "severity": str(flag.get("severity") or "unknown"),
                            "message": str(flag.get("message") or ""),
                        }
                        for flag in quality_flags
                    ]
                },
            )
        )
    top_curriculum = [item for item in training.get("top_curriculum_priorities", []) if isinstance(item, dict)]
    if top_curriculum:
        actions.append(
            _action(
                "prioritize_curriculum_failures",
                _curriculum_action_priority(top_curriculum),
                "training_export",
                f"Use {len(top_curriculum)} prioritized curriculum failure mode(s) to plan repair, scenario, or data-generation work.",
                {
                    "curriculum_failure_mode_count": _non_negative_int(training.get("curriculum_failure_mode_count")),
                    "top_curriculum_priorities": top_curriculum,
                },
            )
        )
    return actions


def _action(action_id: str, priority: str, artifact: str, summary: str, evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": action_id,
        "priority": priority,
        "artifact": artifact,
        "summary": summary,
        "evidence": evidence,
    }


def _count_rows(value: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not isinstance(value, list):
        return counts
    for row in value:
        if not isinstance(row, dict):
            continue
        row_id = row.get("id")
        count = row.get("count")
        if isinstance(row_id, str) and row_id and isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            counts[row_id] = count
    return counts


def _count_total(value: Any) -> int:
    return sum(_count_rows(value).values())


def _top_curriculum_priorities(curriculum: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
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
                    "priority_band": str(mode.get("priority_band") or "unknown"),
                    "count": _non_negative_int(mode.get("count")),
                    "critical_count": _non_negative_int(mode.get("critical_count")),
                    "max_penalty": _non_negative_int(mode.get("max_penalty")),
                    "scenario_ids": _bounded_string_list(mode.get("scenario_ids"), 5),
                    "failure_ids": _bounded_string_list(mode.get("failure_ids"), 5),
                    "example_evidence_refs": _bounded_dict_list(mode.get("example_evidence_refs"), 2),
                }
            )
    rows.sort(key=lambda item: (-item["priority_score"], -item["count"], item["task_family"], item["rule_id"]))
    return rows[:limit]


def _curriculum_action_priority(top_curriculum: list[dict[str, Any]]) -> str:
    bands = {str(item.get("priority_band") or "") for item in top_curriculum}
    if "critical" in bands:
        return "critical"
    if "high" in bands:
        return "high"
    if "medium" in bands:
        return "medium"
    return "low"


def _bounded_string_list(value: Any, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item][:limit]


def _bounded_dict_list(value: Any, limit: int) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)][:limit]


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int) and value >= 0:
        return value
    return 0


def _summarize_suite_summary(suite_summary: dict[str, Any], metrics: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    suite_metrics = suite_summary.get("metrics") if isinstance(suite_summary.get("metrics"), dict) else {}
    metrics["suite_summary"] = {
        "total": suite_summary.get("total"),
        "passed": suite_summary.get("passed"),
        "failed": suite_summary.get("failed"),
        "error_count": suite_summary.get("error_count"),
        "pass_rate": suite_metrics.get("pass_rate"),
        "average_score": suite_metrics.get("average_score"),
        "critical_failure_counts": suite_metrics.get("critical_failure_counts"),
        "failed_rule_counts": suite_metrics.get("failed_rule_counts"),
    }
    _add_presence_check(checks, "suite_summary_no_errors", suite_summary.get("error_count") == 0, {"artifact": "suite_summary"})


def _summarize_boolean_artifact(
    artifact: dict[str, Any],
    metrics: dict[str, Any],
    checks: list[dict[str, Any]],
    *,
    artifact_name: str,
    metric_prefix: str,
    metric_fields: tuple[str, ...],
    metrics_source: str = "metrics",
) -> None:
    source = artifact if metrics_source == "root" else artifact.get("metrics") if isinstance(artifact.get("metrics"), dict) else {}
    metrics[metric_prefix] = {field: source.get(field) for field in metric_fields}
    _add_presence_check(checks, f"{artifact_name}_passed", artifact.get("passed") is True, {"artifact": artifact_name})


def _read_json_artifact(path: Path, artifacts: dict[str, Any], name: str, preserve_paths: bool) -> dict[str, Any]:
    artifacts[name] = _file_record(path, preserve_paths)
    if not path.exists():
        raise EvidenceBundleError(f"Evidence artifact not found: {path}")
    value = _read_optional_json(path)
    if not isinstance(value, dict):
        raise EvidenceBundleError(f"Evidence artifact must contain a JSON object: {path}")
    artifacts[name]["schema_version"] = value.get("schema_version")
    artifacts[name]["passed"] = value.get("passed") if isinstance(value.get("passed"), bool) else None
    return value


def _read_optional_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _add_presence_check(checks: list[dict[str, Any]], check_id: str, passed: bool, scope: dict[str, str]) -> None:
    checks.append(
        {
            "id": check_id,
            "passed": passed,
            "actual": {"passed": passed},
            "expected": {"passed": True},
            "scope": scope,
            "summary": f"{check_id}: passed={passed}",
        }
    )


def _gate_id(gate: dict[str, Any], fallback: str) -> str:
    schema = str(gate.get("schema_version") or fallback)
    return schema.removeprefix("hfr.").removesuffix(".v1")


def _file_record(path: Path, preserve_paths: bool) -> dict[str, Any]:
    record: dict[str, Any] = {"path": _display_path(path, preserve_paths), "exists": path.exists(), "kind": "file"}
    if path.exists() and path.is_file():
        record["size_bytes"] = path.stat().st_size
        record["sha256"] = _sha256(path)
    return record


def _dir_record(path: Path, preserve_paths: bool) -> dict[str, Any]:
    record: dict[str, Any] = {"path": _display_path(path, preserve_paths), "exists": path.exists(), "kind": "directory"}
    if path.exists() and path.is_dir():
        record["entry_count"] = sum(1 for _ in path.iterdir())
    return record


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
