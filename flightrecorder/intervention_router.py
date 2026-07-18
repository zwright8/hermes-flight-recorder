"""Deterministic failure-to-intervention routing for the complete agent stack."""

from __future__ import annotations

import hashlib
import json
from typing import Any

INTERVENTION_ROUTE_SCHEMA_VERSION = "hfr.intervention_route.v1"

_ROUTES = (
    "prompt_policy",
    "tool_schema",
    "parser_runtime",
    "planner_routing",
    "memory_retrieval",
    "guardrail_sandbox",
    "dataset",
    "evaluation",
    "model_training",
    "human_review",
)
_COST_RANK = {route: index for index, route in enumerate(_ROUTES)}

_FAILURE_ROUTES = {
    "ambiguous_instruction": "prompt_policy",
    "instruction_conflict": "prompt_policy",
    "system_prompt_gap": "prompt_policy",
    "low_final_answer_quality": "prompt_policy",
    "wrong_tool_selection": "tool_schema",
    "invalid_tool_arguments": "tool_schema",
    "missing_tool_description": "tool_schema",
    "tool_schema_drift": "tool_schema",
    "tool_result_parse_error": "parser_runtime",
    "serialization_error": "parser_runtime",
    "runtime_exception": "parser_runtime",
    "adapter_normalization_error": "parser_runtime",
    "planning_loop": "planner_routing",
    "excessive_retries": "planner_routing",
    "handoff_routing_error": "planner_routing",
    "subagent_coordination_error": "planner_routing",
    "retrieval_miss": "memory_retrieval",
    "stale_memory": "memory_retrieval",
    "context_selection_error": "memory_retrieval",
    "prompt_injection": "guardrail_sandbox",
    "forbidden_action": "guardrail_sandbox",
    "unsafe_side_effect": "guardrail_sandbox",
    "sandbox_escape": "guardrail_sandbox",
    "training_data_contamination": "dataset",
    "pii_exposure": "dataset",
    "label_error": "dataset",
    "duplicate_amplification": "dataset",
    "insufficient_eval_repeats": "evaluation",
    "eval_identity_mismatch": "evaluation",
    "benchmark_leakage": "evaluation",
    "grader_disagreement": "human_review",
    "model_capability_shortfall": "model_training",
    "capacity_shortfall": "model_training",
    "tool_calling_capability_shortfall": "model_training",
}

_ACCEPTANCE_METRICS = {
    "prompt_policy": ["target failure rate decreases on unchanged held-out scenarios", "no critical policy regression"],
    "tool_schema": ["tool selection and argument-schema validity improve", "no tool compatibility regression"],
    "parser_runtime": ["replay succeeds with zero parse/runtime errors", "unchanged valid traces remain equivalent"],
    "planner_routing": ["completion improves within step/retry budget", "handoff and loop rates do not regress"],
    "memory_retrieval": ["required evidence recall improves", "unsupported-answer rate does not regress"],
    "guardrail_sandbox": ["critical unsafe actions remain zero", "safe task completion remains within policy threshold"],
    "dataset": ["governance and contamination gates pass", "affected lineage is rebuilt and fingerprinted"],
    "evaluation": ["identical-arm repeated evidence is complete", "confidence and identity gates pass"],
    "model_training": ["held-out lower confidence bound exceeds minimum effect", "safety/cost/latency gates do not regress"],
    "human_review": ["reviewer resolves failure class and intervention", "decision carries evidence and confidence"],
}


class InterventionRouterError(ValueError):
    """Raised when a failure cluster is malformed."""


def route_failure_cluster(cluster: dict[str, Any]) -> dict[str, Any]:
    """Choose the least-cost adequate intervention and explain alternatives."""

    if not isinstance(cluster, dict):
        raise InterventionRouterError("failure cluster must be an object")
    cluster_id = _required_string(cluster, "cluster_id")
    failure_modes = sorted(
        {
            value.strip()
            for value in cluster.get("failure_modes", [])
            if isinstance(value, str) and value.strip()
        }
    )
    confidence = _bounded_number(cluster.get("confidence"), 0.0, 1.0)
    severity = str(cluster.get("severity") or "medium").casefold()
    if severity not in {"critical", "high", "medium", "low"}:
        raise InterventionRouterError("severity must be critical, high, medium, or low")
    normalized = {
        "cluster_id": cluster_id,
        "failure_modes": failure_modes,
        "severity": severity,
        "confidence": confidence,
        "frequency": _non_negative_int(cluster.get("frequency")),
        "affected_task_families": _string_list(cluster.get("affected_task_families")),
        "affected_tools": _string_list(cluster.get("affected_tools")),
        "affected_policies": _string_list(cluster.get("affected_policies")),
        "evidence_refs": _evidence_refs(cluster.get("evidence_refs")),
    }
    matched = sorted(
        {_FAILURE_ROUTES[mode] for mode in failure_modes if mode in _FAILURE_ROUTES},
        key=lambda route: (_COST_RANK[route], route),
    )
    routing_reasons: list[str] = []
    forced_review_reasons: list[str] = []
    if severity == "critical":
        forced_review_reasons.append("high_impact_requires_human_review")
    if "grader_disagreement" in failure_modes:
        forced_review_reasons.append("grader_disagreement_requires_human_review")
    if confidence < 0.6:
        forced_review_reasons.append("low_confidence")
    if forced_review_reasons:
        selected = "human_review"
        routing_reasons.extend(forced_review_reasons)
    elif not matched:
        selected = "human_review"
        routing_reasons.append("unknown_failure_class")
    else:
        selected = matched[0]
        routing_reasons.extend(f"matched:{mode}->{_FAILURE_ROUTES[mode]}" for mode in failure_modes if mode in _FAILURE_ROUTES)
        if len(matched) > 1:
            routing_reasons.append("least_cost_adequate_intervention_selected")
        if selected == "model_training" and not normalized["evidence_refs"]:
            selected = "human_review"
            routing_reasons.append("model_training_requires_capability_evidence")

    rejected = []
    for route in _ROUTES:
        if route == selected:
            continue
        if route == "model_training" and selected != "model_training":
            reason = "weight updates are higher cost and require explicit capability evidence"
        elif route in matched:
            reason = f"adequate but higher estimated cost/risk than {selected}"
        elif route == "human_review":
            reason = "deterministic routing confidence is sufficient"
        else:
            reason = "failure evidence does not match this intervention class"
        rejected.append(
            {
                "intervention": route,
                "reason": reason,
                "cost_rank": _COST_RANK[route],
            }
        )
    rejected.sort(key=lambda row: (row["cost_rank"], row["intervention"]))

    identity = {
        "cluster": normalized,
        "selected": selected,
        "routing_reasons": routing_reasons,
        "rejected": rejected,
    }
    fingerprint = _canonical_sha256(identity)
    return {
        "schema_version": INTERVENTION_ROUTE_SCHEMA_VERSION,
        "route_id": f"hfrroute-{fingerprint[:16]}",
        "routing_fingerprint": fingerprint,
        "cluster": normalized,
        "selected_intervention": selected,
        "routing_reasons": routing_reasons,
        "estimated_cost": _estimated_cost(selected),
        "estimated_risk": _estimated_risk(selected, severity),
        "rejected_alternatives": rejected,
        "work_item": {
            "work_item_id": f"{selected}:{fingerprint[:16]}",
            "intervention": selected,
            "priority": severity,
            "summary": _summary(selected, failure_modes),
            "evidence_refs": normalized["evidence_refs"],
            "affected_task_families": normalized["affected_task_families"],
            "affected_tools": normalized["affected_tools"],
            "affected_policies": normalized["affected_policies"],
            "acceptance_metrics": list(_ACCEPTANCE_METRICS[selected]),
            "execution_boundary": {
                "production_mutation_started": False,
                "requires_reviewed_change": True,
                "requires_heldout_evaluation": True,
                "requires_promotion_gate": selected != "human_review",
            },
        },
    }


def cluster_from_improvement_item(item: dict[str, Any]) -> dict[str, Any]:
    """Normalize a legacy improvement-plan item into router input."""

    text = " ".join(
        str(item.get(field) or "")
        for field in ("category", "summary", "suggested_action", "rule_id", "rule_name")
    ).casefold()
    inferred: list[str] = []
    keyword_modes = (
        (("prompt", "instruction"), "ambiguous_instruction"),
        (("tool schema", "tool argument", "wrong tool"), "invalid_tool_arguments"),
        (("parse", "runtime", "serialization"), "tool_result_parse_error"),
        (("loop", "retry", "routing", "handoff"), "planning_loop"),
        (("retrieval", "memory", "context"), "retrieval_miss"),
        (("unsafe", "forbidden", "injection", "sandbox"), "prompt_injection"),
        (("dataset", "label", "contamination", "redaction", "pii"), "training_data_contamination"),
        (("eval", "benchmark", "grader", "confidence"), "insufficient_eval_repeats"),
        (("model", "capacity", "fine-tun", "training"), "model_capability_shortfall"),
    )
    for keywords, mode in keyword_modes:
        if any(keyword in text for keyword in keywords):
            inferred.append(mode)
    confidence = 0.8 if inferred else 0.4
    return {
        "cluster_id": str(item.get("item_id") or item.get("fingerprint") or "unknown-item"),
        "failure_modes": inferred or ["unexpected_behavior"],
        "severity": str(item.get("priority") or "medium"),
        "confidence": confidence,
        "frequency": 1,
        "affected_task_families": [str(item["task_family"])] if item.get("task_family") else [],
        "affected_tools": [],
        "affected_policies": [],
        "evidence_refs": item.get("evidence_refs") if isinstance(item.get("evidence_refs"), list) else [],
    }


def _estimated_cost(route: str) -> dict[str, Any]:
    return {
        "rank": _COST_RANK[route],
        "class": "low" if _COST_RANK[route] <= 2 else "medium" if _COST_RANK[route] <= 7 else "high",
    }


def _estimated_risk(route: str, severity: str) -> dict[str, Any]:
    production_risk = "high" if route in {"guardrail_sandbox", "model_training"} else "medium" if severity in {"critical", "high"} else "low"
    return {"class": production_risk, "requires_canary": route != "human_review", "requires_rollback_target": route != "human_review"}


def _summary(route: str, modes: list[str]) -> str:
    rendered = ", ".join(modes) if modes else "unclassified behavior"
    return f"Apply {route.replace('_', ' ')} intervention for: {rendered}."


def _required_string(value: dict[str, Any], field: str) -> str:
    rendered = value.get(field)
    if not isinstance(rendered, str) or not rendered.strip():
        raise InterventionRouterError(f"{field} must be a non-empty string")
    return rendered.strip()


def _bounded_number(value: Any, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise InterventionRouterError("confidence must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise InterventionRouterError("confidence must be numeric") from exc
    if number < minimum or number > maximum:
        raise InterventionRouterError(f"confidence must be from {minimum} to {maximum}")
    return number


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted({item.strip() for item in value if isinstance(item, str) and item.strip()})


def _evidence_refs(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows = [row for row in value if isinstance(row, dict)]
    return sorted(rows, key=lambda row: json.dumps(row, sort_keys=True, default=str))


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
