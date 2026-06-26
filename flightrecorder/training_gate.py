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
    "max_quality_flags",
}
_SCALAR_POLICY_FIELDS = {"min_pass_rate", "min_average_score", *_COUNT_POLICY_FIELDS}
_LIST_POLICY_FIELDS = {"forbid_quality_flags", "forbid_quality_severities", "require_task_families"}
_TASK_FAMILY_SCALAR_POLICY_FIELDS = {
    "min_episodes",
    "min_pass_rate",
    "min_average_score",
    "min_step_rewards",
    "min_failure_modes",
    "min_sft",
    "min_dpo",
    "min_reward_model",
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
        if field == "min_pass_rate":
            policy[field] = _policy_float(field, raw[field], 0.0, 1.0)
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
    max_quality_flags: int | None = None,
    forbid_quality_flags: list[str] | None = None,
    forbid_quality_severities: list[str] | None = None,
    require_task_families: list[str] | None = None,
    task_family_gates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate readiness checks against dataset_metrics.json."""
    artifact_counts = dataset_metrics.get("artifact_counts") if isinstance(dataset_metrics.get("artifact_counts"), dict) else {}
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
    }
    for policy_field, (check_id, row_field) in field_checks.items():
        if gate.get(policy_field) is None:
            continue
        actual = _number(row.get(row_field)) if "rate" in policy_field or "score" in policy_field else _int_value(row.get(row_field))
        _add_min_check(checks, check_id, actual, gate[policy_field], scope)
    if gate.get("max_failed") is not None:
        _add_max_check(checks, "task_family_max_failed", _int_value(row.get("failed")), gate["max_failed"], scope)


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


def _policy_float(field: str, value: Any, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingGatePolicyError(f"training gate policy field {field} must be a number")
    parsed = float(value)
    if not minimum <= parsed <= maximum:
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
