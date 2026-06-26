"""Policy gates over training-export dataset metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TRAINING_GATE_SCHEMA_VERSION = "hfr.training_gate.v1"
TRAINING_GATE_POLICY_SCHEMA_VERSION = "hfr.training_gate.policy.v1"

_COUNT_POLICY_FIELDS = {
    "min_episodes",
    "min_preferences",
    "min_sft",
    "min_dpo",
    "min_reward_model",
    "min_step_rewards",
    "min_task_completion_configured",
    "min_task_completion_complete",
    "max_quality_flags",
    "max_task_completion_incomplete",
    "max_unverified_source_fingerprints",
    "max_unverified_trainer_view_source_fingerprints",
    "min_trace_event_type_count",
    "max_trace_empty_final_answers",
    "max_trace_risk_count",
}
_RATE_POLICY_FIELDS = {
    "min_pass_rate",
    "min_source_fingerprint_rate",
    "min_trainer_view_source_fingerprint_rate",
    "min_task_completion_check_pass_rate",
    "min_trace_final_answer_rate",
    "min_trace_tool_or_api_rate",
}
_FLOAT_POLICY_FIELDS = {"min_trace_average_events"}
_SCALAR_POLICY_FIELDS = {*_RATE_POLICY_FIELDS, *_FLOAT_POLICY_FIELDS, "min_average_score", *_COUNT_POLICY_FIELDS}
_LIST_POLICY_FIELDS = {"forbid_quality_flags", "forbid_quality_severities", "require_task_families", "require_trace_event_types"}
_TASK_FAMILY_SCALAR_POLICY_FIELDS = {
    "min_episodes",
    "min_pass_rate",
    "min_average_score",
    "min_step_rewards",
    "min_failure_modes",
    "min_sft",
    "min_dpo",
    "min_reward_model",
    "min_task_completion_complete",
    "max_task_completion_incomplete",
    "max_failed",
}
_TASK_FAMILY_GATE_FIELDS = {"task_family", *_TASK_FAMILY_SCALAR_POLICY_FIELDS}
_POLICY_FIELDS = {"schema_version", "description", "task_family_gates", *_SCALAR_POLICY_FIELDS, *_LIST_POLICY_FIELDS}
_QUALITY_SEVERITIES = {"info", "warning", "error"}


class TrainingGatePolicyError(ValueError):
    """Raised when a training gate policy file is malformed."""


def load_training_gate_policy(path: str | Path) -> dict[str, Any]:
    """Load and validate a versioned training gate policy JSON file."""
    policy_path = Path(path)
    try:
        raw = json.loads(policy_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TrainingGatePolicyError(f"Invalid JSON in training gate policy {policy_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise TrainingGatePolicyError(f"Training gate policy must be a JSON object: {policy_path}")

    version = raw.get("schema_version")
    if version != TRAINING_GATE_POLICY_SCHEMA_VERSION:
        raise TrainingGatePolicyError(
            f"training gate policy schema_version must be {TRAINING_GATE_POLICY_SCHEMA_VERSION!r}; got {version!r}"
        )

    unknown = sorted(set(raw) - _POLICY_FIELDS)
    if unknown:
        raise TrainingGatePolicyError(f"Unknown training gate policy field(s): {', '.join(unknown)}")

    policy: dict[str, Any] = {"schema_version": TRAINING_GATE_POLICY_SCHEMA_VERSION}
    if "description" in raw:
        if not isinstance(raw["description"], str):
            raise TrainingGatePolicyError("training gate policy field description must be a string")
        policy["description"] = raw["description"]

    for field in _SCALAR_POLICY_FIELDS:
        if field not in raw or raw[field] is None:
            continue
        if field in _RATE_POLICY_FIELDS:
            policy[field] = _policy_float(field, raw[field], 0.0, 1.0)
        elif field in _FLOAT_POLICY_FIELDS:
            policy[field] = _policy_float(field, raw[field], 0.0, None)
        elif field == "min_average_score":
            policy[field] = _policy_float(field, raw[field], 0.0, 100.0)
        else:
            policy[field] = _policy_non_negative_int(field, raw[field])

    for field in _LIST_POLICY_FIELDS:
        if field not in raw or raw[field] is None:
            continue
        values = _policy_string_list(field, raw[field])
        if field == "forbid_quality_severities":
            unknown_severities = sorted(set(values) - _QUALITY_SEVERITIES)
            if unknown_severities:
                raise TrainingGatePolicyError(
                    f"training gate policy field {field} has invalid severity value(s): {', '.join(unknown_severities)}"
                )
        policy[field] = values

    if "task_family_gates" in raw and raw["task_family_gates"] is not None:
        policy["task_family_gates"] = _policy_task_family_gates(raw["task_family_gates"])

    return policy


def evaluate_training_gate(
    dataset_metrics: dict[str, Any],
    *,
    training_export_path: str | Path,
    min_episodes: int | None = None,
    min_pass_rate: float | None = None,
    min_average_score: float | None = None,
    min_preferences: int | None = None,
    min_sft: int | None = None,
    min_dpo: int | None = None,
    min_reward_model: int | None = None,
    min_step_rewards: int | None = None,
    min_task_completion_configured: int | None = None,
    min_task_completion_complete: int | None = None,
    max_task_completion_incomplete: int | None = None,
    min_task_completion_check_pass_rate: float | None = None,
    min_source_fingerprint_rate: float | None = None,
    max_unverified_source_fingerprints: int | None = None,
    min_trainer_view_source_fingerprint_rate: float | None = None,
    max_unverified_trainer_view_source_fingerprints: int | None = None,
    min_trace_average_events: float | None = None,
    min_trace_event_type_count: int | None = None,
    min_trace_final_answer_rate: float | None = None,
    min_trace_tool_or_api_rate: float | None = None,
    max_trace_empty_final_answers: int | None = None,
    max_trace_risk_count: int | None = None,
    max_quality_flags: int | None = None,
    forbid_quality_flags: list[str] | None = None,
    forbid_quality_severities: list[str] | None = None,
    require_task_families: list[str] | None = None,
    require_trace_event_types: list[str] | None = None,
    task_family_gates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate readiness checks against dataset_metrics.json."""
    artifact_counts = dataset_metrics.get("artifact_counts") if isinstance(dataset_metrics.get("artifact_counts"), dict) else {}
    source_fingerprint_coverage = _source_fingerprint_coverage(dataset_metrics)
    trainer_view_source_fingerprint_coverage = _trainer_view_source_fingerprint_coverage(dataset_metrics)
    task_completion = _task_completion_metrics(dataset_metrics)
    trace_signal = _trace_signal_metrics(dataset_metrics)
    checks: list[dict[str, Any]] = []

    if min_episodes is not None:
        _add_min_check(checks, "min_episodes", _int_value(dataset_metrics.get("episode_count")), min_episodes)
    if min_pass_rate is not None:
        _add_min_check(checks, "min_pass_rate", _number(dataset_metrics.get("pass_rate")), min_pass_rate)
    if min_average_score is not None:
        _add_min_check(checks, "min_average_score", _number(dataset_metrics.get("average_score")), min_average_score)
    if min_preferences is not None:
        _add_min_check(checks, "min_preferences", _int_value(artifact_counts.get("preferences")), min_preferences)
    if min_sft is not None:
        _add_min_check(checks, "min_sft", _int_value(artifact_counts.get("sft")), min_sft)
    if min_dpo is not None:
        _add_min_check(checks, "min_dpo", _int_value(artifact_counts.get("dpo")), min_dpo)
    if min_reward_model is not None:
        _add_min_check(checks, "min_reward_model", _int_value(artifact_counts.get("reward_model")), min_reward_model)
    if min_step_rewards is not None:
        _add_min_check(checks, "min_step_rewards", _int_value(artifact_counts.get("step_rewards")), min_step_rewards)
    if min_task_completion_configured is not None:
        _add_min_check(
            checks,
            "min_task_completion_configured",
            _int_value(task_completion.get("configured_count")),
            min_task_completion_configured,
        )
    if min_task_completion_complete is not None:
        _add_min_check(
            checks,
            "min_task_completion_complete",
            _int_value(task_completion.get("complete_count")),
            min_task_completion_complete,
        )
    if max_task_completion_incomplete is not None:
        _add_max_check(
            checks,
            "max_task_completion_incomplete",
            _int_value(task_completion.get("incomplete_count")),
            max_task_completion_incomplete,
        )
    if min_task_completion_check_pass_rate is not None:
        _add_min_check(
            checks,
            "min_task_completion_check_pass_rate",
            _number(task_completion.get("check_pass_rate")),
            min_task_completion_check_pass_rate,
        )
    if min_source_fingerprint_rate is not None:
        _add_min_check(
            checks,
            "min_source_fingerprint_rate",
            _number(source_fingerprint_coverage.get("rate")),
            min_source_fingerprint_rate,
        )
    if max_unverified_source_fingerprints is not None:
        _add_max_check(
            checks,
            "max_unverified_source_fingerprints",
            _int_value(source_fingerprint_coverage.get("unverified")),
            max_unverified_source_fingerprints,
        )
    if min_trainer_view_source_fingerprint_rate is not None:
        _add_min_check(
            checks,
            "min_trainer_view_source_fingerprint_rate",
            _number(trainer_view_source_fingerprint_coverage.get("fully_verified_rate")),
            min_trainer_view_source_fingerprint_rate,
        )
    if max_unverified_trainer_view_source_fingerprints is not None:
        _add_max_check(
            checks,
            "max_unverified_trainer_view_source_fingerprints",
            _int_value(trainer_view_source_fingerprint_coverage.get("unverified")),
            max_unverified_trainer_view_source_fingerprints,
        )
    if min_trace_average_events is not None:
        _add_min_check(checks, "min_trace_average_events", _number(trace_signal.get("average_event_count")), min_trace_average_events)
    if min_trace_event_type_count is not None:
        _add_min_check(checks, "min_trace_event_type_count", _int_value(trace_signal.get("event_type_count")), min_trace_event_type_count)
    if min_trace_final_answer_rate is not None:
        _add_min_check(checks, "min_trace_final_answer_rate", _number(trace_signal.get("final_answer_rate")), min_trace_final_answer_rate)
    if min_trace_tool_or_api_rate is not None:
        _add_min_check(checks, "min_trace_tool_or_api_rate", _number(trace_signal.get("tool_or_api_episode_rate")), min_trace_tool_or_api_rate)
    if max_trace_empty_final_answers is not None:
        _add_max_check(
            checks,
            "max_trace_empty_final_answers",
            _int_value(trace_signal.get("empty_final_answer_count")),
            max_trace_empty_final_answers,
        )
    if max_trace_risk_count is not None:
        _add_max_check(checks, "max_trace_risk_count", _int_value(trace_signal.get("risk_count")), max_trace_risk_count)
    event_type_counts = _count_rows(trace_signal.get("event_type_counts"))
    for event_type in require_trace_event_types or []:
        _add_min_check(
            checks,
            "require_trace_event_type",
            event_type_counts.get(event_type, 0),
            1,
            {"event_type": event_type},
        )

    quality_flags = _quality_flags(dataset_metrics.get("quality_flags"))
    if max_quality_flags is not None:
        _add_max_check(checks, "max_quality_flags", len(quality_flags), max_quality_flags)
    flag_ids = _count_values(flag.get("id") for flag in quality_flags)
    severities = _count_values(flag.get("severity") for flag in quality_flags)
    for flag_id in forbid_quality_flags or []:
        _add_absence_check(checks, "forbid_quality_flag", flag_id, flag_ids.get(flag_id, 0))
    for severity in forbid_quality_severities or []:
        _add_absence_check(checks, "forbid_quality_severity", severity, severities.get(severity, 0))

    family_rows = _task_family_rows(dataset_metrics.get("task_families"))
    for family in require_task_families or []:
        _add_presence_check(checks, "require_task_family", family in family_rows, {"task_family": family})
    for gate in task_family_gates or []:
        _evaluate_task_family_gate(checks, gate, family_rows)

    passed = all(check["passed"] for check in checks)
    return {
        "schema_version": TRAINING_GATE_SCHEMA_VERSION,
        "training_export": str(training_export_path),
        "passed": passed,
        "check_count": len(checks),
        "failed_check_count": sum(1 for check in checks if not check["passed"]),
        "checks": checks,
        "metrics": {
            "episode_count": dataset_metrics.get("episode_count"),
            "pass_rate": dataset_metrics.get("pass_rate"),
            "average_score": dataset_metrics.get("average_score"),
            "quality_flag_count": len(quality_flags),
            "source_fingerprint_coverage": source_fingerprint_coverage,
            "trainer_view_source_fingerprint_coverage": trainer_view_source_fingerprint_coverage,
            "task_completion": task_completion,
            "trace_signal": trace_signal,
            "artifact_counts": artifact_counts,
        },
    }


def _evaluate_task_family_gate(checks: list[dict[str, Any]], gate: dict[str, Any], family_rows: dict[str, dict[str, Any]]) -> None:
    family = str(gate["task_family"])
    row = family_rows.get(family)
    scope = {"task_family": family}
    if row is None:
        _add_presence_check(checks, "task_family_present", False, scope)
        return
    _add_presence_check(checks, "task_family_present", True, scope)
    field_checks = {
        "min_episodes": ("task_family_min_episodes", "episode_count"),
        "min_pass_rate": ("task_family_min_pass_rate", "pass_rate"),
        "min_average_score": ("task_family_min_average_score", "average_score"),
        "min_step_rewards": ("task_family_min_step_rewards", "step_reward_count"),
        "min_failure_modes": ("task_family_min_failure_modes", "failure_mode_count"),
        "min_sft": ("task_family_min_sft", "sft_count"),
        "min_dpo": ("task_family_min_dpo", "dpo_count"),
        "min_reward_model": ("task_family_min_reward_model", "reward_model_count"),
        "min_task_completion_complete": ("task_family_min_task_completion_complete", "task_completion_complete"),
    }
    for policy_field, (check_id, row_field) in field_checks.items():
        if gate.get(policy_field) is None:
            continue
        actual = _number(row.get(row_field)) if "rate" in policy_field or "score" in policy_field else _int_value(row.get(row_field))
        _add_min_check(checks, check_id, actual, gate[policy_field], scope)
    if gate.get("max_failed") is not None:
        _add_max_check(checks, "task_family_max_failed", _int_value(row.get("failed")), gate["max_failed"], scope)
    if gate.get("max_task_completion_incomplete") is not None:
        _add_max_check(
            checks,
            "task_family_max_task_completion_incomplete",
            _int_value(row.get("task_completion_incomplete")),
            gate["max_task_completion_incomplete"],
            scope,
        )


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
    _append_check(checks, check, scope)


def _add_max_check(
    checks: list[dict[str, Any]],
    check_id: str,
    actual: int,
    maximum: int,
    scope: dict[str, str] | None = None,
) -> None:
    check = {
        "id": check_id,
        "passed": actual <= maximum,
        "actual": actual,
        "expected": {"max": maximum},
        "summary": f"{check_id}: actual={actual}, max={maximum}",
    }
    _append_check(checks, check, scope)


def _add_absence_check(
    checks: list[dict[str, Any]],
    check_id: str,
    item_id: str,
    count: int,
    scope: dict[str, str] | None = None,
) -> None:
    check = {
        "id": check_id,
        "passed": count == 0,
        "actual": {"id": item_id, "count": count},
        "expected": {"count": 0},
        "summary": f"{check_id}: id={item_id}, count={count}",
    }
    _append_check(checks, check, scope)


def _add_presence_check(checks: list[dict[str, Any]], check_id: str, present: bool, scope: dict[str, str]) -> None:
    check = {
        "id": check_id,
        "passed": present,
        "actual": {"present": present},
        "expected": {"present": True},
        "summary": f"{check_id}: present={present}",
    }
    _append_check(checks, check, scope)


def _append_check(checks: list[dict[str, Any]], check: dict[str, Any], scope: dict[str, str] | None = None) -> None:
    if scope:
        check["scope"] = scope
    checks.append(check)


def _quality_flags(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [flag for flag in value if isinstance(flag, dict)]


def _task_family_rows(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for row in value:
        if not isinstance(row, dict) or not isinstance(row.get("task_family"), str):
            continue
        rows[row["task_family"]] = row
    return rows


def _source_fingerprint_coverage(dataset_metrics: dict[str, Any]) -> dict[str, Any]:
    coverage = dataset_metrics.get("source_fingerprint_coverage")
    if not isinstance(coverage, dict):
        episode_count = _int_value(dataset_metrics.get("episode_count"))
        return {
            "episodes": episode_count,
            "fully_verified": 0,
            "unverified": episode_count,
            "rate": 0.0,
        }
    episodes = _int_value(coverage.get("episodes"))
    fully_verified = _int_value(coverage.get("fully_verified"))
    return {
        "episodes": episodes,
        "fully_verified": fully_verified,
        "unverified": _int_value(coverage.get("unverified")),
        "with_scenario_sha256": _int_value(coverage.get("with_scenario_sha256")),
        "with_source_trace_sha256": _int_value(coverage.get("with_source_trace_sha256")),
        "rate": round(fully_verified / episodes, 4) if episodes else 0.0,
    }


def _trainer_view_source_fingerprint_coverage(dataset_metrics: dict[str, Any]) -> dict[str, Any]:
    coverage = dataset_metrics.get("trainer_view_source_fingerprint_coverage")
    if not isinstance(coverage, dict):
        artifact_counts = dataset_metrics.get("artifact_counts") if isinstance(dataset_metrics.get("artifact_counts"), dict) else {}
        sft_rows = _int_value(artifact_counts.get("sft"))
        dpo_rows = _int_value(artifact_counts.get("dpo"))
        reward_model_rows = _int_value(artifact_counts.get("reward_model"))
        row_count = sft_rows + dpo_rows + reward_model_rows
        return {
            "rows": row_count,
            "sft_rows": sft_rows,
            "dpo_rows": dpo_rows,
            "reward_model_rows": reward_model_rows,
            "fully_verified": 0,
            "unverified": row_count,
            "fully_verified_rate": 0.0,
        }
    rows = _int_value(coverage.get("rows"))
    fully_verified = _int_value(coverage.get("fully_verified"))
    return {
        "rows": rows,
        "sft_rows": _int_value(coverage.get("sft_rows")),
        "dpo_rows": _int_value(coverage.get("dpo_rows")),
        "reward_model_rows": _int_value(coverage.get("reward_model_rows")),
        "fully_verified": fully_verified,
        "unverified": _int_value(coverage.get("unverified")),
        "fully_verified_rate": _number(coverage.get("fully_verified_rate"))
        if "fully_verified_rate" in coverage
        else round(fully_verified / rows, 4)
        if rows
        else 0.0,
    }


def _task_completion_metrics(dataset_metrics: dict[str, Any]) -> dict[str, Any]:
    metrics = dataset_metrics.get("task_completion")
    if not isinstance(metrics, dict):
        return {
            "episode_count": _int_value(dataset_metrics.get("episode_count")),
            "configured_count": 0,
            "complete_count": 0,
            "incomplete_count": 0,
            "not_applicable_count": 0,
            "unknown_count": 0,
            "required_check_count": 0,
            "passed_check_count": 0,
            "check_pass_rate": 0.0,
        }
    return {
        "episode_count": _int_value(metrics.get("episode_count")),
        "configured_count": _int_value(metrics.get("configured_count")),
        "complete_count": _int_value(metrics.get("complete_count")),
        "incomplete_count": _int_value(metrics.get("incomplete_count")),
        "not_applicable_count": _int_value(metrics.get("not_applicable_count")),
        "unknown_count": _int_value(metrics.get("unknown_count")),
        "required_check_count": _int_value(metrics.get("required_check_count")),
        "passed_check_count": _int_value(metrics.get("passed_check_count")),
        "check_pass_rate": _number(metrics.get("check_pass_rate")),
    }


def _trace_signal_metrics(dataset_metrics: dict[str, Any]) -> dict[str, Any]:
    metrics = dataset_metrics.get("trace_signal")
    if not isinstance(metrics, dict):
        episode_count = _int_value(dataset_metrics.get("episode_count"))
        return {
            "episode_count": episode_count,
            "total_event_count": 0,
            "average_event_count": 0.0,
            "min_event_count": 0,
            "max_event_count": 0,
            "event_type_count": 0,
            "event_type_counts": [],
            "episodes_with_final_answer": 0,
            "empty_final_answer_count": episode_count,
            "final_answer_rate": 0.0,
            "episodes_with_tool_or_api_events": 0,
            "tool_or_api_episode_rate": 0.0,
            "tool_call_count": 0,
            "tool_result_count": 0,
            "api_call_count": 0,
            "subagent_event_count": 0,
            "approval_event_count": 0,
            "risk_count": 0,
            "risk_counts": [],
        }
    return {
        "episode_count": _int_value(metrics.get("episode_count")),
        "total_event_count": _int_value(metrics.get("total_event_count")),
        "average_event_count": _number(metrics.get("average_event_count")),
        "min_event_count": _int_value(metrics.get("min_event_count")),
        "max_event_count": _int_value(metrics.get("max_event_count")),
        "event_type_count": _int_value(metrics.get("event_type_count")),
        "event_type_counts": metrics.get("event_type_counts") if isinstance(metrics.get("event_type_counts"), list) else [],
        "episodes_with_final_answer": _int_value(metrics.get("episodes_with_final_answer")),
        "empty_final_answer_count": _int_value(metrics.get("empty_final_answer_count")),
        "final_answer_rate": _number(metrics.get("final_answer_rate")),
        "episodes_with_tool_or_api_events": _int_value(metrics.get("episodes_with_tool_or_api_events")),
        "tool_or_api_episode_rate": _number(metrics.get("tool_or_api_episode_rate")),
        "tool_call_count": _int_value(metrics.get("tool_call_count")),
        "tool_result_count": _int_value(metrics.get("tool_result_count")),
        "api_call_count": _int_value(metrics.get("api_call_count")),
        "subagent_event_count": _int_value(metrics.get("subagent_event_count")),
        "approval_event_count": _int_value(metrics.get("approval_event_count")),
        "risk_count": _int_value(metrics.get("risk_count")),
        "risk_counts": metrics.get("risk_counts") if isinstance(metrics.get("risk_counts"), list) else [],
    }


def _count_rows(value: Any) -> dict[str, int]:
    if not isinstance(value, list):
        return {}
    counts: dict[str, int] = {}
    for row in value:
        if isinstance(row, dict) and isinstance(row.get("id"), str) and isinstance(row.get("count"), int):
            counts[row["id"]] = max(0, row["count"])
    return counts


def _count_values(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if not isinstance(value, str) or not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return counts


def _number(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0


def _policy_float(field: str, value: Any, minimum: float, maximum: float | None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingGatePolicyError(f"training gate policy field {field} must be a number")
    parsed = float(value)
    if parsed < minimum or (maximum is not None and parsed > maximum):
        if maximum is None:
            raise TrainingGatePolicyError(f"training gate policy field {field} must be at least {minimum}")
        raise TrainingGatePolicyError(f"training gate policy field {field} must be between {minimum} and {maximum}")
    return parsed


def _policy_non_negative_int(field: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TrainingGatePolicyError(f"training gate policy field {field} must be a non-negative integer")
    return value


def _policy_string_list(field: str, value: Any) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise TrainingGatePolicyError(f"training gate policy field {field} must be a list of non-empty strings")
    return list(dict.fromkeys(value))


def _policy_task_family_gates(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TrainingGatePolicyError("training gate policy field task_family_gates must be a list")
    gates: list[dict[str, Any]] = []
    for index, raw_gate in enumerate(value):
        if not isinstance(raw_gate, dict):
            raise TrainingGatePolicyError(f"training gate policy task_family_gates[{index}] must be an object")
        unknown = sorted(set(raw_gate) - _TASK_FAMILY_GATE_FIELDS)
        if unknown:
            raise TrainingGatePolicyError(
                f"Unknown training gate task_family_gates[{index}] field(s): {', '.join(unknown)}"
            )
        task_family = raw_gate.get("task_family")
        if not isinstance(task_family, str) or not task_family:
            raise TrainingGatePolicyError(f"training gate policy task_family_gates[{index}].task_family must be a non-empty string")
        gate: dict[str, Any] = {"task_family": task_family}
        for field in _TASK_FAMILY_SCALAR_POLICY_FIELDS:
            if field not in raw_gate or raw_gate[field] is None:
                continue
            if field == "min_pass_rate":
                gate[field] = _policy_float(f"task_family_gates[{index}].{field}", raw_gate[field], 0.0, 1.0)
            elif field == "min_average_score":
                gate[field] = _policy_float(f"task_family_gates[{index}].{field}", raw_gate[field], 0.0, 100.0)
            else:
                gate[field] = _policy_non_negative_int(f"task_family_gates[{index}].{field}", raw_gate[field])
        gates.append(gate)
    return gates
