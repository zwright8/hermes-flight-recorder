"""Policy gate over Flight Recorder decision recommendations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

DECISION_GATE_SCHEMA_VERSION = "hfr.decision_gate.v1"


class DecisionGateError(ValueError):
    """Raised when a decision gate cannot be evaluated."""


def evaluate_decision_gate(
    artifact: dict[str, Any],
    *,
    artifact_path: str | Path,
    expect_recommendation: str,
    expect_readiness: str | None = None,
    require_passed: bool = False,
) -> dict[str, Any]:
    """Evaluate a stable decision recommendation from another artifact."""
    if not isinstance(artifact, dict):
        raise DecisionGateError("Decision gate input must be a JSON object.")
    if not expect_recommendation:
        raise DecisionGateError("--expect-recommendation must be non-empty.")
    decision = artifact.get("decision") if isinstance(artifact.get("decision"), dict) else {}
    source_passed = artifact.get("passed") if isinstance(artifact.get("passed"), bool) else None
    source = {
        "schema_version": str(artifact.get("schema_version") or ""),
        "passed": source_passed,
        "recommendation": str(decision.get("recommendation") or ""),
        "readiness": str(decision.get("readiness") or ""),
        "summary": str(decision.get("summary") or ""),
        "blocking_check_count": decision.get("blocking_check_count")
        if isinstance(decision.get("blocking_check_count"), int) and not isinstance(decision.get("blocking_check_count"), bool)
        else None,
        "key_metrics": decision.get("key_metrics") if isinstance(decision.get("key_metrics"), dict) else {},
    }
    checks: list[dict[str, Any]] = [
        {
            "id": "recommendation_matches",
            "passed": source["recommendation"] == expect_recommendation,
            "actual": {"recommendation": source["recommendation"]},
            "expected": {"recommendation": expect_recommendation},
            "summary": f"recommendation_matches: actual={source['recommendation']!r}, expected={expect_recommendation!r}",
        }
    ]
    if expect_readiness is not None:
        checks.append(
            {
                "id": "readiness_matches",
                "passed": source["readiness"] == expect_readiness,
                "actual": {"readiness": source["readiness"]},
                "expected": {"readiness": expect_readiness},
                "summary": f"readiness_matches: actual={source['readiness']!r}, expected={expect_readiness!r}",
            }
        )
    if require_passed:
        checks.append(
            {
                "id": "source_artifact_passed",
                "passed": source_passed is True,
                "actual": {"passed": source_passed},
                "expected": {"passed": True},
                "summary": f"source_artifact_passed: actual={source_passed!r}, expected=True",
            }
        )

    failed_check_count = sum(1 for check in checks if not check["passed"])
    passed = failed_check_count == 0
    return {
        "schema_version": DECISION_GATE_SCHEMA_VERSION,
        "artifact": str(artifact_path),
        "passed": passed,
        "readiness": "ready" if passed else "blocked",
        "recommendation": "allow_promotion" if passed else "block_promotion",
        "expected_recommendation": expect_recommendation,
        "expected_readiness": expect_readiness,
        "require_passed": require_passed,
        "check_count": len(checks),
        "failed_check_count": failed_check_count,
        "checks": checks,
        "source_decision": source,
        "notes": [
            "Decision gates consume Flight Recorder decision recommendations for CI and orchestration.",
            "They do not rerun evals, repair failures, train models, or mutate the source artifact.",
        ],
    }
