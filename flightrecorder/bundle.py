"""Evidence-bundle handoff summaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .gate_contract import summarize_gate_contract
from .schema_registry import SchemaRegistryError, check_schema_file

EVIDENCE_BUNDLE_SCHEMA_VERSION = "hfr.evidence_bundle.v1"
HARNESS_RUN_MANIFEST_SCHEMA_VERSION = "hfr.harness_run_manifest.v1"
HARNESS_RUN_RESULT_SCHEMA_VERSION = "hfr.harness_run_result.v1"
_VALIDATION_REQUIRED_GATE_SCHEMAS = {
    "hfr.training_gate.v1",
    "hfr.compare_gate.v1",
    "hfr.reviewed_gate.v1",
    "hfr.review_calibration.v1",
}
_TRAINER_HANDOFF_STAGES: tuple[dict[str, str], ...] = (
    {
        "id": "trainer_preflight",
        "schema_version": "hfr.trainer_preflight.v1",
        "recommendation": "launch_allowed",
    },
    {
        "id": "trainer_launch_check",
        "schema_version": "hfr.trainer_launch_check.v1",
        "recommendation": "launch_allowed",
    },
    {
        "id": "trainer_archive",
        "schema_version": "hfr.trainer_archive.v1",
        "recommendation": "handoff_ready",
        "manifest": "trainer_archive.json",
    },
    {
        "id": "trainer_archive_check",
        "schema_version": "hfr.trainer_archive_check.v1",
        "recommendation": "consumer_ready",
    },
    {
        "id": "trainer_consumer_plan",
        "schema_version": "hfr.trainer_consumer_plan.v1",
        "recommendation": "ready_for_external_trainer",
    },
    {
        "id": "trainer_wrapper_dry_run",
        "schema_version": "hfr.example_trainer_wrapper_dry_run.v1",
        "recommendation": "dry_run_ready",
    },
)


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
    live_smoke_summary_path: str | Path | None = None,
    trainer_preflight_path: str | Path | None = None,
    trainer_launch_check_path: str | Path | None = None,
    trainer_archive_path: str | Path | None = None,
    trainer_archive_check_path: str | Path | None = None,
    trainer_consumer_plan_path: str | Path | None = None,
    trainer_wrapper_dry_run_path: str | Path | None = None,
    harness_manifest_paths: list[str | Path] | None = None,
    harness_result_paths: list[str | Path] | None = None,
    require_harness: bool = False,
    gate_paths: list[str | Path] | None = None,
    require_gate: bool = False,
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
        _summarize_run_digest_coverage(runs_path, metrics, checks, preserve_paths)

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
            metrics.setdefault("training_export", {})["trainer_view_source_fingerprint_coverage"] = dataset_metrics.get(
                "trainer_view_source_fingerprint_coverage"
            )
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
                "candidate_win_scenarios": manifest.get("candidate_win_scenarios"),
                "baseline_win_scenarios": manifest.get("baseline_win_scenarios"),
                "task_completion_improvement_count": manifest.get("task_completion_improvement_count"),
                "task_completion_regression_count": manifest.get("task_completion_regression_count"),
                "task_completion_improvement_scenarios": manifest.get("task_completion_improvement_scenarios"),
                "task_completion_regression_scenarios": manifest.get("task_completion_regression_scenarios"),
                "fixed_rule_counts": manifest.get("fixed_rule_counts"),
                "regressed_rule_counts": manifest.get("regressed_rule_counts"),
                "new_critical_failure_counts": manifest.get("new_critical_failure_counts"),
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

    if live_smoke_summary_path is not None:
        live_smoke_summary = _read_json_artifact(Path(live_smoke_summary_path), artifacts, "live_smoke_summary", preserve_paths)
        _summarize_live_smoke_summary(live_smoke_summary, metrics, checks)

    _summarize_trainer_handoff(
        artifacts,
        metrics,
        checks,
        preserve_paths=preserve_paths,
        trainer_preflight_path=trainer_preflight_path,
        trainer_launch_check_path=trainer_launch_check_path,
        trainer_archive_path=trainer_archive_path,
        trainer_archive_check_path=trainer_archive_check_path,
        trainer_consumer_plan_path=trainer_consumer_plan_path,
        trainer_wrapper_dry_run_path=trainer_wrapper_dry_run_path,
    )
    _summarize_harness_handoff(
        artifacts,
        metrics,
        checks,
        preserve_paths=preserve_paths,
        manifest_paths=harness_manifest_paths,
        result_paths=harness_result_paths,
        require_harness=require_harness,
    )

    gate_rows: list[dict[str, Any]] = []
    for index, gate_path in enumerate(gate_paths or []):
        gate_name = f"gate_{index + 1}"
        gate = _read_json_artifact(Path(gate_path), artifacts, gate_name, preserve_paths)
        gate_id = _gate_id(gate, gate_name)
        schema_version = str(gate.get("schema_version") or "")
        passed = bool(gate.get("passed")) if isinstance(gate, dict) else False
        validation = _gate_validation_metrics(gate)
        contract = summarize_gate_contract(gate)
        gate_row: dict[str, Any] = {
            "id": gate_id,
            "path": artifacts[gate_name]["path"],
            "schema_version": schema_version,
            "passed": passed,
            "contract": contract,
        }
        if validation["available"] or _gate_requires_validation(gate):
            gate_row["validation"] = validation
        gate_rows.append(gate_row)
        _add_presence_check(checks, "gate_passed", passed, {"gate": gate_id})
        _add_presence_check(
            checks,
            "gate_contract_valid",
            bool(contract["valid"]),
            {
                "gate": gate_id,
                "contract_available": str(contract["available"]).lower(),
                "contract_errors": "; ".join(contract["errors"][:3]),
            },
        )
        if _gate_requires_validation(gate):
            _add_presence_check(
                checks,
                "gate_validation_passed",
                bool(validation["available"] and validation["passed"]),
                {
                    "gate": gate_id,
                    "validation_available": str(validation["available"]).lower(),
                    "validation_error_count": str(validation["error_count"]),
                },
            )
    if gate_rows:
        metrics["gates"] = gate_rows
    if require_gate:
        _add_presence_check(checks, "gate_summary_present", bool(gate_rows), {"artifact": "gates"})

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
        "run_digest_coverage": (
            "run_count",
            "digest_count",
            "missing_digest_count",
            "invalid_digest_count",
            "digest_coverage_rate",
            "passed_digest_count",
            "failed_digest_count",
            "task_completion_status_counts",
            "recommended_action_counts",
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
            "trainer_view_source_fingerprint_coverage",
            "curriculum_failure_mode_count",
            "top_curriculum_priorities",
        ),
        "compare_export": (
            "pair_count",
            "candidate_win_count",
            "baseline_win_count",
            "task_completion_improvement_count",
            "task_completion_regression_count",
            "candidate_win_scenarios",
            "baseline_win_scenarios",
            "task_completion_improvement_scenarios",
            "task_completion_regression_scenarios",
            "fixed_rule_counts",
            "regressed_rule_counts",
            "new_critical_failure_counts",
            "skipped_pair_count",
        ),
        "review_export": ("item_count", "failed_count", "passed_count"),
        "reviewed_export": ("reviewed_label_count", "sft_count", "dpo_count", "reward_model_count"),
        "review_calibration": ("reviewed_label_count", "comparable_label_count", "agreement_rate", "disagreement_count"),
        "live_smoke_summary": (
            "passed",
            "consistent",
            "score",
            "hook_count",
            "missing_hook_count",
            "chat_completion_request_count",
            "python_version",
            "platform",
            "hermes_git_commit",
            "hermes_git_dirty",
            "flight_recorder_git_commit",
            "flight_recorder_git_dirty",
        ),
        "trainer_handoff": (
            "stage_count",
            "handoff_ready_count",
            "blocked_stage_count",
            "schema_supported_count",
            "complete_chain",
            "all_included_ready",
            "missing_stage_ids",
        ),
        "harness_handoff": (
            "manifest_count",
            "result_count",
            "pair_count",
            "passed_pair_count",
            "failed_pair_count",
            "schema_valid_pair_count",
            "consistent_pair_count",
            "missing_pair_count",
            "runners",
            "providers",
            "models",
            "trace_formats",
        ),
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

    digest_coverage = metrics.get("run_digest_coverage") if isinstance(metrics.get("run_digest_coverage"), dict) else {}
    missing_digests = _non_negative_int(digest_coverage.get("missing_digest_count"))
    invalid_digests = _non_negative_int(digest_coverage.get("invalid_digest_count"))
    if missing_digests or invalid_digests:
        actions.append(
            _action(
                "refresh_run_digests",
                "high",
                "run_digest_coverage",
                "Regenerate missing or invalid run_digest.json files before routing this bundle to CI, review, repair, or training loops.",
                {
                    "missing_digest_count": missing_digests,
                    "invalid_digest_count": invalid_digests,
                    "digest_coverage_rate": digest_coverage.get("digest_coverage_rate"),
                    "missing_digest_scenarios": (
                        digest_coverage.get("missing_digest_scenarios")
                        if isinstance(digest_coverage.get("missing_digest_scenarios"), list)
                        else []
                    ),
                    "invalid_digest_scenarios": (
                        digest_coverage.get("invalid_digest_scenarios")
                        if isinstance(digest_coverage.get("invalid_digest_scenarios"), list)
                        else []
                    ),
                },
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

    live_smoke = metrics.get("live_smoke_summary") if isinstance(metrics.get("live_smoke_summary"), dict) else {}
    if live_smoke and (live_smoke.get("passed") is not True or live_smoke.get("consistent") is not True):
        actions.append(
            _action(
                "fix_live_observer_smoke",
                "critical",
                "live_smoke_summary",
                "Fix the live Hermes observer smoke before relying on runtime integration evidence.",
                {
                    "score": live_smoke.get("score"),
                    "missing_hook_count": _non_negative_int(live_smoke.get("missing_hook_count")),
                    "missing_hooks": live_smoke.get("missing_hooks") if isinstance(live_smoke.get("missing_hooks"), list) else [],
                    "chat_completion_request_count": _non_negative_int(live_smoke.get("chat_completion_request_count")),
                },
            )
        )
    trainer = metrics.get("trainer_handoff") if isinstance(metrics.get("trainer_handoff"), dict) else {}
    blocked_trainer_stages = _non_negative_int(trainer.get("blocked_stage_count"))
    missing_trainer_stages = _bounded_string_list(trainer.get("missing_stage_ids"), len(_TRAINER_HANDOFF_STAGES))
    if blocked_trainer_stages:
        actions.append(
            _action(
                "fix_trainer_handoff",
                "critical",
                "trainer_handoff",
                f"Fix {blocked_trainer_stages} blocked trainer handoff stage(s) before external training consumes this package.",
                {
                    "blocked_stage_count": blocked_trainer_stages,
                    "stages": trainer.get("stages") if isinstance(trainer.get("stages"), list) else [],
                },
            )
        )
    if trainer and missing_trainer_stages:
        actions.append(
            _action(
                "complete_trainer_handoff_chain",
                "medium",
                "trainer_handoff",
                "Include the full trainer preflight, launch-check, archive, archive-check, consumer-plan, and wrapper receipt chain before treating the bundle as trainer-ready.",
                {
                    "missing_stage_ids": missing_trainer_stages,
                    "stage_count": _non_negative_int(trainer.get("stage_count")),
                },
            )
        )

    harness = metrics.get("harness_handoff") if isinstance(metrics.get("harness_handoff"), dict) else {}
    missing_harness_pairs = _non_negative_int(harness.get("missing_pair_count"))
    if missing_harness_pairs:
        actions.append(
            _action(
                "attach_harness_lineage",
                "critical",
                "harness_handoff",
                "Attach matched harness_manifest.json and harness_result.json artifacts before downstream Eval or Governance consumes this bundle.",
                {
                    "missing_pair_count": missing_harness_pairs,
                    "manifest_count": _non_negative_int(harness.get("manifest_count")),
                    "result_count": _non_negative_int(harness.get("result_count")),
                },
            )
        )
    failed_harness_pairs = _non_negative_int(harness.get("failed_pair_count"))
    invalid_harness_pairs = _non_negative_int(harness.get("pair_count")) - min(
        _non_negative_int(harness.get("schema_valid_pair_count")),
        _non_negative_int(harness.get("consistent_pair_count")),
    )
    if failed_harness_pairs or invalid_harness_pairs:
        actions.append(
            _action(
                "fix_harness_handoff",
                "critical",
                "harness_handoff",
                "Fix failed, malformed, or inconsistent harness handoff artifacts before relying on live-run lineage.",
                {
                    "failed_pair_count": failed_harness_pairs,
                    "invalid_pair_count": max(invalid_harness_pairs, 0),
                    "pair_count": _non_negative_int(harness.get("pair_count")),
                },
            )
        )
    return actions


def _action(action_id: str, priority: str, artifact: str, summary: str, evidence: dict[str, Any]) -> dict[str, Any]:
    fingerprint = _action_fingerprint(action_id, priority, artifact, evidence)
    return {
        "id": action_id,
        "priority": priority,
        "artifact": artifact,
        "summary": summary,
        "routing_key": f"{artifact}:{action_id}:{fingerprint[:12]}",
        "action_fingerprint": fingerprint,
        "evidence": evidence,
    }


def _action_fingerprint(action_id: str, priority: str, artifact: str, evidence: dict[str, Any]) -> str:
    payload = {
        "id": action_id,
        "priority": priority,
        "artifact": artifact,
        "evidence": evidence,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _count_map_rows(counts: dict[str, int]) -> list[dict[str, int | str]]:
    return [{"id": key, "count": counts[key]} for key in sorted(counts)]


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


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 1.0
    return round(numerator / denominator, 4)


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


def _summarize_run_digest_coverage(
    runs_path: Path,
    metrics: dict[str, Any],
    checks: list[dict[str, Any]],
    preserve_paths: bool,
) -> None:
    completed_runs = sorted(path for path in runs_path.iterdir() if path.is_dir() and (path / "scorecard.json").exists()) if runs_path.is_dir() else []
    digest_count = 0
    passed_digest_count = 0
    failed_digest_count = 0
    missing_scenarios: list[str] = []
    invalid_scenarios: list[str] = []
    task_status_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    for run_dir in completed_runs:
        digest_path = run_dir / "run_digest.json"
        scenario_id = run_dir.name
        if not digest_path.exists():
            missing_scenarios.append(scenario_id)
            continue
        try:
            digest = _read_optional_json(digest_path)
        except json.JSONDecodeError:
            invalid_scenarios.append(scenario_id)
            continue
        if not isinstance(digest, dict) or digest.get("schema_version") != "hfr.run_digest.v1":
            invalid_scenarios.append(scenario_id)
            continue
        digest_count += 1
        scenario = digest.get("scenario") if isinstance(digest.get("scenario"), dict) else {}
        scenario_id = str(scenario.get("id") or scenario_id)
        outcome = digest.get("outcome") if isinstance(digest.get("outcome"), dict) else {}
        if outcome.get("passed") is True:
            passed_digest_count += 1
        elif outcome.get("passed") is False:
            failed_digest_count += 1
        task_status = str(outcome.get("task_completion_status") or "unknown")
        task_status_counts[task_status] = task_status_counts.get(task_status, 0) + 1
        actions = digest.get("recommended_actions") if isinstance(digest.get("recommended_actions"), list) else []
        for action in actions:
            if isinstance(action, dict) and isinstance(action.get("id"), str) and action.get("id"):
                action_id = action["id"]
                action_counts[action_id] = action_counts.get(action_id, 0) + 1
    run_count = len(completed_runs)
    missing_count = len(missing_scenarios)
    invalid_count = len(invalid_scenarios)
    metrics["run_digest_coverage"] = {
        "runs_dir": _display_path(runs_path, preserve_paths),
        "run_count": run_count,
        "digest_count": digest_count,
        "missing_digest_count": missing_count,
        "invalid_digest_count": invalid_count,
        "digest_coverage_rate": _ratio(digest_count, run_count),
        "passed_digest_count": passed_digest_count,
        "failed_digest_count": failed_digest_count,
        "task_completion_status_counts": _count_map_rows(task_status_counts),
        "recommended_action_counts": _count_map_rows(action_counts),
        "missing_digest_scenarios": missing_scenarios[:20],
        "invalid_digest_scenarios": invalid_scenarios[:20],
    }
    _add_presence_check(
        checks,
        "run_digest_coverage_complete",
        missing_count == 0 and invalid_count == 0,
        {
            "artifact": "runs_dir",
            "completed_runs": str(run_count),
            "missing_digest_count": str(missing_count),
            "invalid_digest_count": str(invalid_count),
        },
    )


def _summarize_live_smoke_summary(summary: dict[str, Any], metrics: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    hooks = summary.get("hooks") if isinstance(summary.get("hooks"), list) else []
    missing_hooks = summary.get("missing_hooks") if isinstance(summary.get("missing_hooks"), list) else []
    consistent = _live_smoke_summary_consistent(summary, missing_hooks)
    metrics["live_smoke_summary"] = {
        "passed": summary.get("passed") if isinstance(summary.get("passed"), bool) else False,
        "consistent": consistent,
        "score": summary.get("score"),
        "hook_count": len([item for item in hooks if isinstance(item, str)]),
        "missing_hook_count": len([item for item in missing_hooks if isinstance(item, str)]),
        "missing_hooks": [item for item in missing_hooks if isinstance(item, str)],
        "chat_completion_request_count": summary.get("chat_completion_request_count"),
        **_live_smoke_environment_metrics(summary.get("environment")),
    }
    _add_presence_check(checks, "live_smoke_summary_passed", summary.get("passed") is True, {"artifact": "live_smoke_summary"})
    _add_presence_check(checks, "live_smoke_summary_consistent", consistent, {"artifact": "live_smoke_summary"})
    _add_presence_check(checks, "live_smoke_summary_no_missing_hooks", not missing_hooks, {"artifact": "live_smoke_summary"})


def _live_smoke_summary_consistent(summary: dict[str, Any], missing_hooks: list[Any]) -> bool:
    score = summary.get("score")
    return (
        summary.get("passed") is True
        and summary.get("hermes_exit_code") == 0
        and isinstance(score, int)
        and not isinstance(score, bool)
        and score >= 90
        and not missing_hooks
        and _non_negative_int(summary.get("chat_completion_request_count")) > 0
    )


def _live_smoke_environment_metrics(environment: Any) -> dict[str, Any]:
    if not isinstance(environment, dict):
        return {}
    fields = (
        "python_version",
        "python_implementation",
        "platform",
        "hermes_root",
        "hermes_git_commit",
        "hermes_git_dirty",
        "flight_recorder_root",
        "flight_recorder_git_commit",
        "flight_recorder_git_dirty",
    )
    return {field: environment.get(field) for field in fields if field in environment}


def _summarize_harness_handoff(
    artifacts: dict[str, Any],
    metrics: dict[str, Any],
    checks: list[dict[str, Any]],
    *,
    preserve_paths: bool,
    manifest_paths: list[str | Path] | None,
    result_paths: list[str | Path] | None,
    require_harness: bool,
) -> None:
    manifests = [Path(path) for path in manifest_paths or []]
    results = [Path(path) for path in result_paths or []]
    if not manifests and not results:
        if require_harness:
            metrics["harness_handoff"] = _empty_harness_metrics()
            _add_presence_check(checks, "harness_handoff_present", False, {"artifact": "harness_handoff"})
        return

    _add_presence_check(
        checks,
        "harness_handoff_pair_count",
        len(manifests) == len(results) and bool(manifests),
        {
            "artifact": "harness_handoff",
            "manifest_count": str(len(manifests)),
            "result_count": str(len(results)),
        },
    )

    rows: list[dict[str, Any]] = []
    runner_counts: dict[str, int] = {}
    provider_counts: dict[str, int] = {}
    model_counts: dict[str, int] = {}
    trace_format_counts: dict[str, int] = {}
    for index, (manifest_path, result_path) in enumerate(zip(manifests, results), start=1):
        manifest_key = f"harness_manifest_{index}"
        result_key = f"harness_result_{index}"
        manifest = _read_json_artifact(manifest_path, artifacts, manifest_key, preserve_paths)
        result = _read_json_artifact(result_path, artifacts, result_key, preserve_paths)
        row = _harness_pair_metrics(manifest_path, manifest, result_path, result, preserve_paths)
        rows.append(row)
        for counts, value in (
            (runner_counts, row["runner"]),
            (provider_counts, row["provider"]),
            (model_counts, row["model"]),
            (trace_format_counts, row["trace_format"]),
        ):
            if value:
                counts[value] = counts.get(value, 0) + 1

        scope = {
            "artifact": "harness_handoff",
            "scenario_id": row["scenario_id"],
            "runner": row["runner"],
            "provider": row["provider"],
        }
        _add_presence_check(checks, "harness_pair_schema_valid", bool(row["schema_valid"]), scope)
        _add_presence_check(checks, "harness_pair_consistent", bool(row["consistent"]), scope)
        _add_presence_check(checks, "harness_result_passed", bool(row["passed"]), scope)

    missing_pair_count = abs(len(manifests) - len(results))
    metrics["harness_handoff"] = {
        "manifest_count": len(manifests),
        "result_count": len(results),
        "pair_count": len(rows),
        "passed_pair_count": sum(1 for row in rows if row["passed"]),
        "failed_pair_count": sum(1 for row in rows if not row["passed"]),
        "schema_valid_pair_count": sum(1 for row in rows if row["schema_valid"]),
        "consistent_pair_count": sum(1 for row in rows if row["consistent"]),
        "missing_pair_count": missing_pair_count,
        "runners": _count_map_rows(runner_counts),
        "providers": _count_map_rows(provider_counts),
        "models": _count_map_rows(model_counts),
        "trace_formats": _count_map_rows(trace_format_counts),
        "runs": rows,
    }
    if require_harness:
        _add_presence_check(checks, "harness_handoff_present", bool(rows), {"artifact": "harness_handoff"})


def _empty_harness_metrics() -> dict[str, Any]:
    return {
        "manifest_count": 0,
        "result_count": 0,
        "pair_count": 0,
        "passed_pair_count": 0,
        "failed_pair_count": 0,
        "schema_valid_pair_count": 0,
        "consistent_pair_count": 0,
        "missing_pair_count": 1,
        "runners": [],
        "providers": [],
        "models": [],
        "trace_formats": [],
        "runs": [],
    }


def _harness_pair_metrics(
    manifest_path: Path,
    manifest: dict[str, Any],
    result_path: Path,
    result: dict[str, Any],
    preserve_paths: bool,
) -> dict[str, Any]:
    model = manifest.get("model") if isinstance(manifest.get("model"), dict) else {}
    result_model = result.get("model") if isinstance(result.get("model"), dict) else {}
    scenario = manifest.get("scenario") if isinstance(manifest.get("scenario"), dict) else {}
    scorecard = result.get("scorecard") if isinstance(result.get("scorecard"), dict) else {}
    trace = result.get("trace") if isinstance(result.get("trace"), dict) else {}
    replay = result.get("replay") if isinstance(result.get("replay"), dict) else {}
    schema_errors: list[str] = []
    for schema_name, path in (("harness_run_manifest", manifest_path), ("harness_run_result", result_path)):
        check = _schema_check_summary(path, schema_name)
        if not check["passed"]:
            schema_errors.extend(f"{schema_name}: {error}" for error in check["errors"])
    consistency_errors = _harness_consistency_errors(manifest_path, manifest, result_path, result)
    return {
        "id": str(result.get("scenario_id") or scenario.get("id") or manifest_path.parent.name or "unknown"),
        "scenario_id": str(result.get("scenario_id") or scenario.get("id") or ""),
        "runner": str(result.get("runner") or manifest.get("runner") or ""),
        "provider": str(result.get("provider") or manifest.get("provider") or ""),
        "model": str(result_model.get("id") or model.get("id") or ""),
        "manifest_path": _display_path(manifest_path, preserve_paths),
        "result_path": _display_path(result_path, preserve_paths),
        "trace_format": str(trace.get("format") or ""),
        "trace_path": str(trace.get("path") or ""),
        "score": scorecard.get("score"),
        "passed": scorecard.get("passed") is True,
        "schema_valid": not schema_errors,
        "consistent": not consistency_errors,
        "schema_errors": schema_errors[:5],
        "consistency_errors": consistency_errors[:5],
        "replay_lineage": str(replay.get("lineage") or ""),
        "replay_self_contained": replay.get("self_contained") if isinstance(replay.get("self_contained"), bool) else None,
    }


def _schema_check_summary(path: Path, schema_name: str) -> dict[str, Any]:
    try:
        result = check_schema_file(path, schema_name)
    except (OSError, json.JSONDecodeError, SchemaRegistryError) as exc:
        return {"passed": False, "errors": [str(exc)]}
    errors = result.get("errors") if isinstance(result.get("errors"), list) else []
    return {"passed": bool(result.get("passed")), "errors": [str(error) for error in errors[:5]]}


def _harness_consistency_errors(manifest_path: Path, manifest: dict[str, Any], result_path: Path, result: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field_name in ("runner", "provider"):
        if manifest.get(field_name) != result.get(field_name):
            errors.append(f"{field_name} mismatch")
    manifest_model = manifest.get("model") if isinstance(manifest.get("model"), dict) else {}
    result_model = result.get("model") if isinstance(result.get("model"), dict) else {}
    if manifest_model.get("id") != result_model.get("id"):
        errors.append("model.id mismatch")
    scenario = manifest.get("scenario") if isinstance(manifest.get("scenario"), dict) else {}
    if scenario.get("id") != result.get("scenario_id"):
        errors.append("scenario id mismatch")
    outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), dict) else {}
    output_result = outputs.get("result")
    if isinstance(output_result, str) and output_result:
        expected_result = _resolve_harness_path(manifest_path.parent, output_result)
        if expected_result != result_path.resolve():
            errors.append("manifest.outputs.result does not point at harness result")
    if result.get("schema_version") != HARNESS_RUN_RESULT_SCHEMA_VERSION:
        errors.append("result schema_version mismatch")
    if manifest.get("schema_version") != HARNESS_RUN_MANIFEST_SCHEMA_VERSION:
        errors.append("manifest schema_version mismatch")
    return errors


def _resolve_harness_path(base_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _summarize_trainer_handoff(
    artifacts: dict[str, Any],
    metrics: dict[str, Any],
    checks: list[dict[str, Any]],
    *,
    preserve_paths: bool,
    trainer_preflight_path: str | Path | None,
    trainer_launch_check_path: str | Path | None,
    trainer_archive_path: str | Path | None,
    trainer_archive_check_path: str | Path | None,
    trainer_consumer_plan_path: str | Path | None,
    trainer_wrapper_dry_run_path: str | Path | None,
) -> None:
    paths = {
        "trainer_preflight": trainer_preflight_path,
        "trainer_launch_check": trainer_launch_check_path,
        "trainer_archive": trainer_archive_path,
        "trainer_archive_check": trainer_archive_check_path,
        "trainer_consumer_plan": trainer_consumer_plan_path,
        "trainer_wrapper_dry_run": trainer_wrapper_dry_run_path,
    }
    stages: list[dict[str, Any]] = []
    for spec in _TRAINER_HANDOFF_STAGES:
        stage_id = spec["id"]
        raw_path = paths.get(stage_id)
        if raw_path is None:
            continue
        path = Path(raw_path)
        if "manifest" in spec:
            artifact = _read_json_manifest_artifact(path, artifacts, stage_id, spec["manifest"], preserve_paths)
        else:
            artifact = _read_json_artifact(path, artifacts, stage_id, preserve_paths)
        stage = _trainer_stage_metrics(stage_id, artifact, artifacts[stage_id], spec)
        stages.append(stage)
        _add_presence_check(
            checks,
            f"{stage_id}_schema_supported",
            bool(stage["schema_supported"]),
            {
                "artifact": stage_id,
                "expected_schema_version": stage["expected_schema_version"],
                "schema_version": stage["schema_version"],
            },
        )
        _add_presence_check(
            checks,
            f"{stage_id}_ready",
            bool(stage["handoff_ready"]),
            {
                "artifact": stage_id,
                "readiness": stage["readiness"],
                "recommendation": stage["recommendation"],
                "expected_recommendation": stage["expected_recommendation"],
            },
        )
    if not stages:
        return

    included_stage_ids = {stage["id"] for stage in stages}
    expected_stage_ids = [spec["id"] for spec in _TRAINER_HANDOFF_STAGES]
    missing_stage_ids = [stage_id for stage_id in expected_stage_ids if stage_id not in included_stage_ids]
    metrics["trainer_handoff"] = {
        "stage_count": len(stages),
        "handoff_ready_count": sum(1 for stage in stages if stage["handoff_ready"] is True),
        "blocked_stage_count": sum(1 for stage in stages if stage["handoff_ready"] is False),
        "schema_supported_count": sum(1 for stage in stages if stage["schema_supported"] is True),
        "complete_chain": not missing_stage_ids,
        "all_included_ready": all(stage["handoff_ready"] is True for stage in stages),
        "missing_stage_ids": missing_stage_ids,
        "stages": stages,
    }


def _trainer_stage_metrics(
    stage_id: str,
    artifact: dict[str, Any],
    record: dict[str, Any],
    spec: dict[str, str],
) -> dict[str, Any]:
    schema_version = str(artifact.get("schema_version") or "")
    readiness = str(artifact.get("readiness") or "")
    recommendation = str(artifact.get("recommendation") or "")
    expected_schema = spec["schema_version"]
    expected_recommendation = spec["recommendation"]
    schema_supported = schema_version == expected_schema
    handoff_ready = (
        schema_supported
        and artifact.get("passed") is True
        and readiness == "ready"
        and recommendation == expected_recommendation
    )
    metrics = artifact.get("metrics") if isinstance(artifact.get("metrics"), dict) else {}
    stage: dict[str, Any] = {
        "id": stage_id,
        "path": str(record.get("path") or ""),
        "schema_version": schema_version,
        "expected_schema_version": expected_schema,
        "schema_supported": schema_supported,
        "passed": artifact.get("passed") is True,
        "readiness": readiness,
        "recommendation": recommendation,
        "expected_recommendation": expected_recommendation,
        "handoff_ready": handoff_ready,
        "check_count": _non_negative_int(artifact.get("check_count")),
        "failed_check_count": _non_negative_int(artifact.get("failed_check_count")),
    }
    for field_name in (
        "gate_count",
        "passed_gate_count",
        "trainer_input_count",
        "trainer_input_ready_count",
        "trainer_input_available_count",
        "external_code_file_count",
        "external_code_ready_count",
        "missing_external_code_count",
        "missing_trainer_input_count",
        "command_arg_count",
        "artifact_count",
        "missing_count",
        "path_rewrite_count",
    ):
        value = artifact.get(field_name)
        if value is None and isinstance(metrics, dict):
            value = metrics.get(field_name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            stage[field_name] = value
    return stage


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


def _read_json_manifest_artifact(
    path: Path,
    artifacts: dict[str, Any],
    name: str,
    manifest_name: str,
    preserve_paths: bool,
) -> dict[str, Any]:
    if not path.is_dir():
        return _read_json_artifact(path, artifacts, name, preserve_paths)
    artifacts[name] = _dir_record(path, preserve_paths)
    manifest_path = path / manifest_name
    value = _read_json_artifact(manifest_path, artifacts, f"{name}_manifest", preserve_paths)
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


def _gate_requires_validation(gate: dict[str, Any]) -> bool:
    return gate.get("schema_version") in _VALIDATION_REQUIRED_GATE_SCHEMAS


def _gate_validation_metrics(gate: dict[str, Any]) -> dict[str, Any]:
    metrics = gate.get("metrics") if isinstance(gate.get("metrics"), dict) else {}
    validation = metrics.get("validation") if isinstance(metrics.get("validation"), dict) else {}
    if not validation:
        return {
            "available": False,
            "passed": False,
            "strict": False,
            "target_count": 0,
            "error_count": 0,
            "warning_count": 0,
        }
    available = validation.get("available") is True
    return {
        "available": available,
        "passed": bool(available and validation.get("passed")),
        "strict": bool(validation.get("strict")),
        "target_count": _non_negative_int(validation.get("target_count")),
        "error_count": _non_negative_int(validation.get("error_count")),
        "warning_count": _non_negative_int(validation.get("warning_count")),
    }


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
