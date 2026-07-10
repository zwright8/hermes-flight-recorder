"""Policy gate over Flight Recorder decision recommendations."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .path_safety import path_has_symlink_component as _path_has_symlink_component
from .schema_registry import SchemaRegistryError, check_schema_contract

DECISION_GATE_SCHEMA_VERSION = "hfr.decision_gate.v1"
DECISION_GATE_SOURCE_SCHEMAS = {
    "hfr.action_ledger_gate.v1": "action_ledger_gate",
    "hfr.agentic_loop_governance_receipt.v1": "agentic_loop_governance_receipt",
    "hfr.agentic_loop_ledger.v1": "agentic_loop_ledger",
    "hfr.compare_gate.v1": "compare_gate",
    "hfr.evidence_bundle.v1": "evidence_bundle",
    "hfr.improvement_ledger.v1": "improvement_ledger",
    "hfr.improvement_ledger_gate.v1": "improvement_ledger_gate",
    "hfr.improvement_plan.v1": "improvement_plan",
    "hfr.promotion_decision.v1": "promotion_decision",
    "hfr.promotion_ledger_gate.v1": "promotion_ledger_gate",
    "hfr.reviewed_gate.v1": "reviewed_gate",
    "hfr.suite_gate.v1": "suite_gate",
    "hfr.training_gate.v1": "training_gate",
}
_SOURCE_CONTRACT_ERROR_LIMIT = 8
_SOURCE_CONTRACT_ERROR_LENGTH = 320


class DecisionGateError(ValueError):
    """Raised when a decision gate cannot be evaluated."""


def evaluate_decision_gate(
    artifact: dict[str, Any],
    *,
    artifact_path: str | Path,
    artifact_display_path: str | None = None,
    expect_recommendation: str,
    expect_readiness: str | None = None,
    require_passed: bool = False,
    preserve_paths: bool = False,
) -> dict[str, Any]:
    """Evaluate a stable decision recommendation from another artifact."""
    if not isinstance(artifact, dict):
        raise DecisionGateError("Decision gate input must be a JSON object.")
    if not expect_recommendation:
        raise DecisionGateError("--expect-recommendation must be non-empty.")
    source_contract_errors = decision_gate_source_contract_errors(artifact)
    if source_contract_errors:
        raise DecisionGateError("Decision gate source artifact is not eligible: " + "; ".join(source_contract_errors))
    reject_symlinked_decision_artifact_input(Path(artifact_path))
    source_artifact = _source_artifact_record(Path(artifact_path), preserve_paths, artifact_display_path)
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
        "artifact": source_artifact["path"],
        "source_artifact": source_artifact,
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


def decision_gate_source_contract_errors(artifact: Any) -> list[str]:
    """Return bounded errors when an artifact is not an eligible gate source."""
    if not isinstance(artifact, dict):
        return ["source artifact must be a JSON object"]
    schema_version = artifact.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version.strip():
        return ["source artifact must declare a supported non-empty schema_version"]
    schema_name = DECISION_GATE_SOURCE_SCHEMAS.get(schema_version)
    if schema_name is None:
        supported = ", ".join(sorted(DECISION_GATE_SOURCE_SCHEMAS))
        displayed_version = _bounded_error(repr(schema_version))
        return [f"source artifact schema_version {displayed_version} is not supported; expected one of: {supported}"]
    try:
        result = check_schema_contract(artifact, name_or_id=schema_name)
    except SchemaRegistryError as exc:
        return [f"source artifact schema contract could not be resolved: {_bounded_error(str(exc))}"]
    errors = result.get("errors") if isinstance(result.get("errors"), list) else []
    bounded = [
        f"source artifact does not satisfy {schema_version!r}: {_bounded_error(str(error))}"
        for error in errors[:_SOURCE_CONTRACT_ERROR_LIMIT]
    ]
    if len(errors) > _SOURCE_CONTRACT_ERROR_LIMIT:
        bounded.append(f"source artifact has {len(errors) - _SOURCE_CONTRACT_ERROR_LIMIT} additional schema error(s)")
    return bounded


def _bounded_error(message: str) -> str:
    if len(message) <= _SOURCE_CONTRACT_ERROR_LENGTH:
        return message
    return message[: _SOURCE_CONTRACT_ERROR_LENGTH - 3] + "..."


def _source_artifact_record(path: Path, preserve_paths: bool, display_path: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": display_path if display_path is not None else _display_path(path, preserve_paths),
        "kind": "file",
        "exists": path.exists(),
    }
    if path.exists() and path.is_file() and not path.is_symlink() and not _path_has_symlink_component(path, include_leaf=False):
        record["size_bytes"] = path.stat().st_size
        record["sha256"] = _sha256(path)
    return record


def reject_symlinked_decision_artifact_input(path: Path) -> None:
    if path.is_symlink() or _path_has_symlink_component(path, include_leaf=False):
        raise DecisionGateError(f"decision_gate.artifact_path must not traverse symlinked components: {path}")


def _display_path(path: Path, preserve_paths: bool) -> str:
    if preserve_paths or not path.is_absolute():
        return str(path)
    return f"<redacted:{path.name}>"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
