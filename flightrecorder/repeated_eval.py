"""Canonical repeated three-arm evaluation and promotion evidence.

The module is dependency-free and side-effect-free apart from explicit JSON
writes performed by callers.  It turns one immutable, per-repeat suite result
into an observation and combines observations into a paired promotion receipt.
Promotion evidence is deliberately rebuildable from its source observations so
hash or semantic drift fails validation.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .schema_registry import SchemaRegistryError, check_schema_contract


ARM_IDENTITY_SCHEMA_VERSION = "hfr.eval_arm_identity.v1"
OBSERVATION_SCHEMA_VERSION = "hfr.agentic_eval_observation.v1"
PROMOTION_EVIDENCE_SCHEMA_VERSION = "hfr.agentic_eval_promotion_evidence.v1"
REQUEST_ATTESTATION_SCHEMA_VERSION = "hfr.eval_request_attestation.v1"

ARMS = ("baseline", "trace_only", "flightrecorder")
POOL_TYPES = ("frozen", "rolling", "adversarial")
PRIMARY_METRICS = ("pass_rate", "score")

DEFAULT_POLICY: dict[str, Any] = {
    "primary_metric": "pass_rate",
    "minimum_effect": 0.0,
    "confidence_level": 0.95,
    "bootstrap_samples": 2000,
    "bootstrap_seed": 1729,
    "minimum_repeats": 3,
    "required_pools": list(POOL_TYPES),
    "max_cost_increase_ratio": 0.0,
    "max_latency_increase_ratio": 0.0,
    "cost_absolute_tolerance_usd": 0.0,
    "latency_absolute_tolerance_seconds": 0.0,
    "family_non_regression_tolerance": 0.0,
    "risk_non_regression_tolerance": 0.0,
}


class RepeatedEvalError(ValueError):
    """Raised when repeated evaluation evidence cannot be constructed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RepeatedEvalError(f"Expected a JSON object: {path}")
    return value


def load_arm_identity(path: str | Path, *, expected_arm: str | None = None) -> dict[str, Any]:
    identity = load_json(path)
    schema = check_schema_contract(identity, name_or_id="eval_arm_identity")
    errors = list(schema["errors"])
    errors.extend(_identity_errors(identity, expected_arm=expected_arm))
    if errors:
        raise RepeatedEvalError("Invalid immutable arm identity: " + "; ".join(errors))
    return identity


def build_observation(
    *,
    arm_identity_path: str | Path,
    evaluation_summary_path: str | Path,
    suite_summary_path: str | Path,
    request_attestation_path: str | Path | None,
    serving_profile_path: str | Path | None,
    repeat_index: int,
    seed: int,
    decoding: dict[str, Any],
    pool_type: str,
    pool_id: str,
    risk_tier: str,
    out_path: str | Path | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build one immutable observation from a single suite repeat."""
    identity_path = Path(arm_identity_path)
    evaluation_path = Path(evaluation_summary_path)
    suite_path = Path(suite_summary_path)
    request_path = Path(request_attestation_path) if request_attestation_path is not None else None
    serving_path = Path(serving_profile_path) if serving_profile_path is not None else None
    output_path = Path(out_path) if out_path is not None else None
    identity = load_arm_identity(identity_path)
    evaluation = load_json(evaluation_path)
    suite = load_json(suite_path)

    if repeat_index < 0:
        raise RepeatedEvalError("repeat_index must be non-negative")
    if pool_type not in POOL_TYPES:
        raise RepeatedEvalError(f"pool_type must be one of {list(POOL_TYPES)!r}")
    if not pool_id:
        raise RepeatedEvalError("pool_id must be non-empty")
    if not risk_tier:
        raise RepeatedEvalError("risk_tier must be non-empty")
    decoding_record = _decoding_record(decoding)
    request_attestation = load_json(request_path) if request_path is not None and request_path.is_file() else None
    serving_profile = load_json(serving_path) if serving_path is not None and serving_path.is_file() else None

    runs = suite.get("runs") if isinstance(suite.get("runs"), list) else []
    cases = [_case_record(run, risk_tier=risk_tier) for run in runs if isinstance(run, dict)]
    cases.sort(key=lambda case: case["scenario_id"])
    scenario_ids = [case["scenario_id"] for case in cases]
    fingerprints = {case["scenario_id"]: case["scenario_sha256"] for case in cases}
    blocking_reasons: list[str] = []
    suite_schema = check_schema_contract(suite, name_or_id="run_suite")
    if not suite_schema["passed"]:
        blocking_reasons.append("suite_schema_invalid")
    if evaluation.get("schema_version") != "hfr.hermes_heldout_eval_summary.v1":
        blocking_reasons.append("evaluation_summary_schema_invalid")
    if not cases:
        blocking_reasons.append("empty_scenario_set")
    if len(set(scenario_ids)) != len(scenario_ids):
        blocking_reasons.append("duplicate_scenario_ids")
    if any(not _is_sha256(value) for value in fingerprints.values()):
        blocking_reasons.append("missing_scenario_fingerprints")
    if int(suite.get("error_count") or 0) != 0:
        blocking_reasons.append("suite_errors")
    if int(suite.get("total") or 0) != len(cases):
        blocking_reasons.append("suite_count_mismatch")
    if evaluation.get("arm") not in {None, identity["arm"]}:
        blocking_reasons.append("evaluation_arm_identity_mismatch")
    base_model = str(identity["model"]["id"])
    if str(evaluation.get("model") or "") != base_model:
        blocking_reasons.append("evaluation_model_identity_mismatch")

    request_reasons, request_summary = _request_attestation_status(
        request_attestation,
        expected_model=base_model,
        expected_base_url=str(evaluation.get("base_url") or ""),
        seed=seed,
        decoding=decoding_record,
    )
    blocking_reasons.extend(request_reasons)
    serving_reasons, adapter_attested = _serving_attestation_status(
        serving_profile,
        identity=identity,
        expected_base_url=str(evaluation.get("base_url") or ""),
    )
    blocking_reasons.extend(serving_reasons)

    source_base = output_path.parent if output_path is not None else None
    source_artifacts = {
        "arm_identity": _artifact_ref(identity_path, source_base),
        "evaluation_summary": _artifact_ref(evaluation_path, source_base),
        "suite_summary": _artifact_ref(suite_path, source_base),
        "request_attestation": _artifact_ref(request_path, source_base) if request_path is not None and request_path.is_file() else None,
        "serving_profile": _artifact_ref(serving_path, source_base) if serving_path is not None and serving_path.is_file() else None,
    }
    if not all(ref["public_safe"] for ref in source_artifacts.values() if isinstance(ref, dict)):
        blocking_reasons.append("source_paths_not_public_safe")

    observation = {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "created_at": created_at or utc_now(),
        "arm": identity["arm"],
        "repeat_index": repeat_index,
        "seed": seed,
        "decoding": decoding_record,
        "pool": {"type": pool_type, "id": pool_id, "risk_tier": risk_tier},
        "identity": {
            "manifest_sha256": source_artifacts["arm_identity"]["sha256"],
            "projection_sha256": canonical_sha256(identity),
            **identity,
        },
        "scenario_set": {
            "scenario_count": len(scenario_ids),
            "scenario_ids": scenario_ids,
            "scenario_fingerprints": dict(sorted(fingerprints.items())),
            "scenario_set_sha256": canonical_sha256(dict(sorted(fingerprints.items()))),
        },
        "execution_attestation": {
            **request_summary,
            "adapter_attested": adapter_attested,
            "serving_profile_sha256": source_artifacts["serving_profile"]["sha256"]
            if isinstance(source_artifacts["serving_profile"], dict)
            else None,
        },
        "case_count": len(cases),
        "cases": cases,
        "source_artifacts": source_artifacts,
        "passed": not blocking_reasons,
        "readiness": "ready_for_paired_comparison" if not blocking_reasons else "blocked",
        "blocking_reasons": sorted(set(blocking_reasons)),
    }
    schema = check_schema_contract(observation, name_or_id="agentic_eval_observation")
    if not schema["passed"]:
        raise RepeatedEvalError("Observation schema violation: " + "; ".join(schema["errors"]))
    return observation


def validate_observation(path: str | Path) -> dict[str, Any]:
    """Replay an observation against current source files and report drift."""
    observation_path = Path(path)
    errors: list[str] = []
    try:
        observation = load_json(observation_path)
        schema = check_schema_contract(observation, name_or_id="agentic_eval_observation")
        errors.extend(schema["errors"])
        if errors:
            return _validation_result(OBSERVATION_SCHEMA_VERSION, errors)
        sources = observation["source_artifacts"]
        resolved = {
            name: _verify_and_resolve_ref(ref, observation_path.parent, errors, name)
            if isinstance(ref, dict)
            else None
            for name, ref in sources.items()
        }
        if errors:
            return _validation_result(OBSERVATION_SCHEMA_VERSION, errors)
        rebuilt = build_observation(
            arm_identity_path=resolved["arm_identity"],
            evaluation_summary_path=resolved["evaluation_summary"],
            suite_summary_path=resolved["suite_summary"],
            request_attestation_path=resolved.get("request_attestation"),
            serving_profile_path=resolved.get("serving_profile"),
            repeat_index=observation["repeat_index"],
            seed=observation["seed"],
            decoding=observation["decoding"],
            pool_type=observation["pool"]["type"],
            pool_id=observation["pool"]["id"],
            risk_tier=observation["pool"]["risk_tier"],
            out_path=observation_path,
            created_at=observation["created_at"],
        )
        if rebuilt != observation:
            errors.append("observation does not match deterministic replay of its source artifacts")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError, SchemaRegistryError) as exc:
        errors.append(str(exc))
    return _validation_result(OBSERVATION_SCHEMA_VERSION, errors)


def paired_bootstrap(
    candidate: Iterable[float],
    reference: Iterable[float],
    *,
    confidence_level: float = 0.95,
    samples: int = 2000,
    seed: int = 1729,
    cluster_ids: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Return a paired scenario/pool-cluster bootstrap interval and effect.

    Repeated seeds for one scenario are averaged inside their cluster before
    resampling. This avoids treating correlated repeats as independent samples.
    Callers that omit ``cluster_ids`` retain pair-level behavior by assigning
    every pair to its own cluster.
    """
    candidate_values = [float(value) for value in candidate]
    reference_values = [float(value) for value in reference]
    if not candidate_values or len(candidate_values) != len(reference_values):
        raise RepeatedEvalError("paired bootstrap requires equal non-empty candidate/reference samples")
    if not 0.0 < confidence_level < 1.0:
        raise RepeatedEvalError("confidence_level must be between zero and one")
    if samples < 100:
        raise RepeatedEvalError("bootstrap_samples must be at least 100")
    differences = [candidate_value - reference_value for candidate_value, reference_value in zip(candidate_values, reference_values)]
    raw_cluster_ids = list(cluster_ids) if cluster_ids is not None else list(range(len(differences)))
    if len(raw_cluster_ids) != len(differences):
        raise RepeatedEvalError("cluster_ids must have exactly one entry per paired sample")
    grouped: dict[str, dict[str, list[float]]] = {}
    for cluster_id, candidate_value, reference_value, difference in zip(
        raw_cluster_ids, candidate_values, reference_values, differences
    ):
        key = canonical_sha256({"cluster_id": cluster_id})
        row = grouped.setdefault(key, {"candidate": [], "reference": [], "difference": []})
        row["candidate"].append(candidate_value)
        row["reference"].append(reference_value)
        row["difference"].append(difference)
    cluster_rows = [
        {
            "candidate": statistics.fmean(row["candidate"]),
            "reference": statistics.fmean(row["reference"]),
            "difference": statistics.fmean(row["difference"]),
            "pair_count": len(row["difference"]),
        }
        for _, row in sorted(grouped.items())
    ]
    rng = random.Random(seed)
    n = len(differences)
    cluster_count = len(cluster_rows)
    boot = [
        statistics.fmean(cluster_rows[rng.randrange(cluster_count)]["difference"] for _ in range(cluster_count))
        for _ in range(samples)
    ]
    boot.sort()
    alpha = 1.0 - confidence_level
    lower_index = max(0, min(samples - 1, math.floor((alpha / 2.0) * (samples - 1))))
    upper_index = max(0, min(samples - 1, math.ceil((1.0 - alpha / 2.0) * (samples - 1))))
    cluster_differences = [row["difference"] for row in cluster_rows]
    mean_difference = statistics.fmean(cluster_differences)
    paired_sd = statistics.stdev(cluster_differences) if cluster_count > 1 else 0.0
    standardized = mean_difference / paired_sd if paired_sd > 0 else None
    return {
        "pair_count": n,
        "cluster_count": cluster_count,
        "effective_sample_count": cluster_count,
        "resampling_unit": "scenario_pool_cluster",
        "pairs_per_cluster": {
            "min": min(row["pair_count"] for row in cluster_rows),
            "max": max(row["pair_count"] for row in cluster_rows),
        },
        "candidate_mean": statistics.fmean(row["candidate"] for row in cluster_rows),
        "reference_mean": statistics.fmean(row["reference"] for row in cluster_rows),
        "mean_difference": mean_difference,
        "paired_standardized_effect": standardized,
        "confidence_level": confidence_level,
        "confidence_interval": {"lower": boot[lower_index], "upper": boot[upper_index]},
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }


def build_promotion_evidence(
    *,
    observation_paths: dict[str, list[str | Path]],
    policy: dict[str, Any] | None = None,
    out_path: str | Path | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build canonical, uncertainty-aware promotion evidence for three arms."""
    effective_policy = _effective_policy(policy)
    output_path = Path(out_path) if out_path is not None else None
    output_base = output_path.parent if output_path is not None else None
    observations: dict[str, list[dict[str, Any]]] = {}
    source_artifacts: dict[str, list[dict[str, Any]]] = {}
    source_errors: list[str] = []
    for arm in ARMS:
        paths = [Path(path) for path in observation_paths.get(arm, [])]
        observations[arm] = []
        source_artifacts[arm] = []
        for path in paths:
            validation = validate_observation(path)
            if not validation["passed"]:
                source_errors.extend(f"{arm}:{path}:{error}" for error in validation["errors"])
            observation = load_json(path)
            observations[arm].append(observation)
            source_artifacts[arm].append(_artifact_ref(path, output_base))

    checks: list[dict[str, Any]] = []
    _add_check(checks, "three_required_arms_present", all(observations[arm] for arm in ARMS), {arm: len(observations[arm]) for arm in ARMS}, ">=1 observation per arm", "coverage")
    _add_check(checks, "observation_sources_replay", not source_errors, source_errors, [], "integrity")
    _add_check(
        checks,
        "source_paths_public_safe",
        all(ref["public_safe"] for refs in source_artifacts.values() for ref in refs),
        {arm: [ref["path"] for ref in refs if not ref["public_safe"]] for arm, refs in source_artifacts.items()},
        "safe relative source paths",
        "integrity",
    )

    identity_summary, identity_errors = _identity_summary(observations)
    _add_check(checks, "immutable_arm_identity", not identity_errors, identity_errors, [], "identity")

    case_maps, case_errors = _case_maps(observations)
    keys_by_arm = {arm: sorted(case_maps[arm]) for arm in ARMS}
    paired = bool(keys_by_arm["baseline"]) and keys_by_arm["baseline"] == keys_by_arm["trace_only"] == keys_by_arm["flightrecorder"]
    _add_check(
        checks,
        "identical_paired_observations",
        paired and not case_errors,
        {"case_errors": case_errors, "pair_counts": {arm: len(keys) for arm, keys in keys_by_arm.items()}},
        "identical non-empty pool/scenario/repeat/seed keys",
        "comparability",
    )

    coverage = _coverage_summary(observations, effective_policy)
    _add_check(checks, "required_pool_coverage", coverage["passed"], coverage, {"required_pools": effective_policy["required_pools"]}, "coverage")
    _add_check(
        checks,
        "minimum_repeats_per_scenario",
        coverage["minimum_repeats_passed"],
        coverage["minimum_observed_repeats"],
        {">=": effective_policy["minimum_repeats"]},
        "coverage",
    )

    effects: dict[str, Any] = {}
    if paired:
        candidate_values = _metric_values(case_maps["flightrecorder"], keys_by_arm["flightrecorder"], effective_policy["primary_metric"])
        for reference_arm in ("baseline", "trace_only"):
            reference_values = _metric_values(case_maps[reference_arm], keys_by_arm[reference_arm], effective_policy["primary_metric"])
            effects[reference_arm] = paired_bootstrap(
                candidate_values,
                reference_values,
                cluster_ids=[key[:4] for key in keys_by_arm["flightrecorder"]],
                confidence_level=effective_policy["confidence_level"],
                samples=effective_policy["bootstrap_samples"],
                seed=effective_policy["bootstrap_seed"],
            )
    for reference_arm in ("baseline", "trace_only"):
        effect = effects.get(reference_arm)
        passed = bool(
            effect
            and effect["mean_difference"] > 0
            and effect["confidence_interval"]["lower"] >= effective_policy["minimum_effect"]
        )
        _add_check(
            checks,
            f"primary_effect_vs_{reference_arm}",
            passed,
            effect,
            {"mean_difference": ">0", "confidence_lower": {">=": effective_policy["minimum_effect"]}},
            "improvement",
        )

    non_regression = _non_regression_summary(case_maps, paired, effective_policy)
    for item in non_regression["checks"]:
        checks.append(item)

    failed_checks = [item for item in checks if not item["passed"]]
    evidence = {
        "schema_version": PROMOTION_EVIDENCE_SCHEMA_VERSION,
        "created_at": created_at or utc_now(),
        "passed": not failed_checks,
        "promotion_ready": not failed_checks,
        "readiness": "ready_for_governance" if not failed_checks else "blocked",
        "recommendation": "send_to_governance" if not failed_checks else "block_promotion_and_repair",
        "policy": effective_policy,
        "arms": identity_summary,
        "source_artifacts": source_artifacts,
        "paired_observation_count": len(keys_by_arm["flightrecorder"]) if paired else 0,
        "pairing": {
            "passed": paired and not case_errors,
            "key_fields": [
                "pool_type",
                "pool_id",
                "scenario_id",
                "scenario_sha256",
                "repeat_index",
                "seed",
                "decoding_config_sha256",
            ],
            "pair_counts": {arm: len(keys) for arm, keys in keys_by_arm.items()},
            "errors": case_errors,
        },
        "pool_coverage": coverage,
        "effects": effects,
        "non_regression": {key: value for key, value in non_regression.items() if key != "checks"},
        "check_count": len(checks),
        "failed_check_count": len(failed_checks),
        "checks": checks,
        "blocking_reasons": [item["id"] for item in failed_checks],
    }
    schema = check_schema_contract(evidence, name_or_id="agentic_eval_promotion_evidence")
    if not schema["passed"]:
        raise RepeatedEvalError("Promotion evidence schema violation: " + "; ".join(schema["errors"]))
    return evidence


def validate_promotion_evidence(path: str | Path) -> dict[str, Any]:
    """Strictly replay promotion evidence from all bound observations."""
    evidence_path = Path(path)
    errors: list[str] = []
    try:
        evidence = load_json(evidence_path)
        schema = check_schema_contract(evidence, name_or_id="agentic_eval_promotion_evidence")
        errors.extend(schema["errors"])
        if errors:
            return _validation_result(PROMOTION_EVIDENCE_SCHEMA_VERSION, errors)
        observation_paths: dict[str, list[Path]] = {}
        for arm in ARMS:
            observation_paths[arm] = [
                _verify_and_resolve_ref(ref, evidence_path.parent, errors, f"{arm}[{index}]")
                for index, ref in enumerate(evidence["source_artifacts"].get(arm, []))
            ]
        if errors:
            return _validation_result(PROMOTION_EVIDENCE_SCHEMA_VERSION, errors)
        rebuilt = build_promotion_evidence(
            observation_paths=observation_paths,
            policy=evidence["policy"],
            out_path=evidence_path,
            created_at=evidence["created_at"],
        )
        if rebuilt != evidence:
            errors.append("promotion evidence does not match deterministic replay of its observations")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError, SchemaRegistryError) as exc:
        errors.append(str(exc))
    return _validation_result(PROMOTION_EVIDENCE_SCHEMA_VERSION, errors)


def _case_record(run: dict[str, Any], *, risk_tier: str) -> dict[str, Any]:
    critical = sorted({str(value) for value in run.get("critical_failures", []) if value})
    tool_valid = run.get("tool_schema_valid")
    if not isinstance(tool_valid, bool):
        failed = {str(value) for value in run.get("failed_rules", [])}
        tool_valid = not bool(failed & {"tool_schema", "tool_call_schema", "structured_tool_call"})
    cost = _optional_float(run.get("cost_usd"))
    latency = _optional_float(run.get("latency_seconds"))
    if latency is None:
        latency_ms = _optional_float(run.get("latency_ms"))
        latency = latency_ms / 1000.0 if latency_ms is not None else None
    return {
        "scenario_id": str(run.get("scenario_id") or ""),
        "scenario_sha256": str(run.get("scenario_sha256") or ""),
        "task_family": str(run.get("task_family") or "unknown"),
        "risk_tier": str(run.get("risk_tier") or risk_tier),
        "passed": run.get("passed") is True,
        "score": float(run.get("score") or 0.0),
        "critical_failures": critical,
        "tool_schema_valid": tool_valid,
        "cost_usd": cost,
        "latency_seconds": latency,
    }


def _decoding_record(value: dict[str, Any]) -> dict[str, Any]:
    temperature = float(value.get("temperature", 0.0))
    top_p = float(value.get("top_p", 1.0))
    max_tokens = int(value.get("max_tokens", 0))
    if not math.isfinite(temperature) or not math.isfinite(top_p) or temperature < 0 or not 0 < top_p <= 1 or max_tokens <= 0:
        raise RepeatedEvalError("decoding requires temperature>=0, 0<top_p<=1, and max_tokens>0")
    return {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "deterministic": temperature == 0.0,
        "config_sha256": canonical_sha256({"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}),
    }


def _request_attestation_status(
    attestation: dict[str, Any] | None,
    *,
    expected_model: str,
    expected_base_url: str,
    seed: int,
    decoding: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    expected = {
        "seed": seed,
        "temperature": decoding["temperature"],
        "top_p": decoding["top_p"],
        "max_tokens": decoding["max_tokens"],
    }
    expected_sha256 = canonical_sha256(expected)
    empty_summary = {
        "request_count": 0,
        "matching_request_count": 0,
        "request_config_sha256": expected_sha256,
        "observed_models": [],
        "endpoint_base_url": expected_base_url,
    }
    if not isinstance(attestation, dict):
        return ["request_attestation_missing"], empty_summary

    reasons: set[str] = set()
    if attestation.get("schema_version") != REQUEST_ATTESTATION_SCHEMA_VERSION:
        reasons.add("request_attestation_invalid")
    configured = attestation.get("configured") if isinstance(attestation.get("configured"), dict) else {}
    configured_projection = {key: configured.get(key) for key in expected}
    if configured_projection != expected or configured.get("config_sha256") != expected_sha256:
        reasons.add("request_configuration_mismatch")
    endpoint = str(attestation.get("endpoint_base_url") or "").rstrip("/")
    if not expected_base_url or endpoint != expected_base_url.rstrip("/"):
        reasons.add("request_endpoint_mismatch")

    requests = attestation.get("requests") if isinstance(attestation.get("requests"), list) else []
    matching = 0
    observed_models: set[str] = set()
    for row in requests:
        if not isinstance(row, dict):
            reasons.add("request_attestation_invalid")
            continue
        projection = {key: row.get(key) for key in expected}
        model = str(row.get("model") or "")
        if model:
            observed_models.add(model)
        config_matches = projection == expected and row.get("config_sha256") == expected_sha256
        model_matches = model == expected_model
        integrity_present = _is_sha256(row.get("body_sha256"))
        recorded_match = row.get("matched") is True
        if config_matches and model_matches and integrity_present and recorded_match:
            matching += 1
        if not config_matches or not recorded_match:
            reasons.add("request_configuration_mismatch")
        if not model_matches:
            reasons.add("request_model_identity_mismatch")
        if not integrity_present:
            reasons.add("request_attestation_invalid")
    if not requests:
        reasons.add("request_attestation_empty")
    if attestation.get("request_count") != len(requests):
        reasons.add("request_attestation_invalid")
    if attestation.get("matching_request_count") != matching or matching != len(requests):
        reasons.add("request_configuration_mismatch")
    declared_models = attestation.get("observed_models")
    if not isinstance(declared_models, list):
        reasons.add("request_attestation_invalid")
        declared_models = []
    if sorted(str(value) for value in declared_models if value) != sorted(observed_models):
        reasons.add("request_attestation_invalid")
    if attestation.get("passed") is not True or attestation.get("blocking_reasons") not in ([], None):
        reasons.add("request_attestation_blocked")
    return sorted(reasons), {
        "request_count": len(requests),
        "matching_request_count": matching,
        "request_config_sha256": expected_sha256,
        "observed_models": sorted(observed_models),
        "endpoint_base_url": endpoint,
    }


def _serving_attestation_status(
    profile: dict[str, Any] | None,
    *,
    identity: dict[str, Any],
    expected_base_url: str,
) -> tuple[list[str], bool]:
    arm = str(identity.get("arm") or "")
    tuned = arm in {"trace_only", "flightrecorder"}
    if not isinstance(profile, dict):
        return (["serving_profile_required_for_tuned_arm"] if tuned else []), False

    reasons: set[str] = set()
    if profile.get("schema_version") != "hfr.serving_profile.v1":
        reasons.add("serving_profile_invalid")
    if profile.get("arm") != arm:
        reasons.add("serving_profile_arm_mismatch")
    endpoint = profile.get("endpoint") if isinstance(profile.get("endpoint"), dict) else {}
    if not expected_base_url or str(endpoint.get("base_url") or "").rstrip("/") != expected_base_url.rstrip("/"):
        reasons.add("serving_profile_endpoint_mismatch")
    preflight = profile.get("eval_preflight") if isinstance(profile.get("eval_preflight"), dict) else {}
    if preflight.get("ready") is not True or preflight.get("failed_checks") not in ([], None):
        reasons.add("serving_profile_not_ready")

    model_identity = profile.get("model_identity") if isinstance(profile.get("model_identity"), dict) else {}
    adapter = model_identity.get("adapter") if isinstance(model_identity.get("adapter"), dict) else {}
    expected_adapter = identity.get("adapter") if isinstance(identity.get("adapter"), dict) else None
    if tuned:
        observed_adapter = {key: adapter.get(key) for key in ("id", "revision", "sha256")}
        if adapter.get("present") is not True or observed_adapter != expected_adapter:
            reasons.add("serving_adapter_identity_mismatch")
        base_model = str((identity.get("model") or {}).get("id") or "")
        expected_adapter_id = str((expected_adapter or {}).get("id") or "")
        observed_model_ids = model_identity.get("observed_model_ids")
        if not isinstance(observed_model_ids, list):
            observed_model_ids = []
            reasons.add("serving_profile_invalid")
        observed_models = {
            str(value)
            for value in (
                model_identity.get("served_model_id"),
                model_identity.get("metadata_model"),
                model_identity.get("chat_response_model"),
                *observed_model_ids,
            )
            if value
        }
        if not any(
            value == expected_adapter_id
            or value == f"{base_model}+{expected_adapter_id}"
            or value.startswith(f"{base_model}+") and expected_adapter_id in value
            for value in observed_models
        ):
            reasons.add("serving_adapter_not_observed")
    elif adapter.get("present") is True:
        reasons.add("baseline_serving_profile_declares_adapter")
    return sorted(reasons), tuned and not reasons


def _identity_errors(identity: dict[str, Any], *, expected_arm: str | None) -> list[str]:
    errors: list[str] = []
    arm = identity.get("arm")
    if expected_arm is not None and arm != expected_arm:
        errors.append(f"arm expected {expected_arm!r}, got {arm!r}")
    for label in ("model", "runtime", "tools", "environment"):
        record = identity.get(label)
        if not isinstance(record, dict) or not _is_sha256(record.get("sha256")):
            errors.append(f"{label}.sha256 must be an immutable lowercase SHA-256")
    model = identity.get("model") if isinstance(identity.get("model"), dict) else {}
    if str(model.get("revision") or "").lower() in {"", "main", "master", "latest", "head"}:
        errors.append("model.revision must be immutable, not a mutable alias")
    adapter = identity.get("adapter")
    if arm == "baseline" and adapter is not None:
        errors.append("baseline identity must not declare an adapter")
    if arm in {"trace_only", "flightrecorder"}:
        if not isinstance(adapter, dict):
            errors.append(f"{arm} identity requires an adapter")
        else:
            if not _is_sha256(adapter.get("sha256")):
                errors.append("adapter.sha256 must be an immutable lowercase SHA-256")
            if str(adapter.get("revision") or "").lower() in {"", "main", "master", "latest", "head"}:
                errors.append("adapter.revision must be immutable, not a mutable alias")
    return errors


def _effective_policy(policy: dict[str, Any] | None) -> dict[str, Any]:
    value = {**DEFAULT_POLICY, **(policy or {})}
    if value["primary_metric"] not in PRIMARY_METRICS:
        raise RepeatedEvalError(f"primary_metric must be one of {list(PRIMARY_METRICS)!r}")
    if int(value["minimum_repeats"]) < 3:
        raise RepeatedEvalError("minimum_repeats must be at least 3")
    if int(value["bootstrap_samples"]) < 100:
        raise RepeatedEvalError("bootstrap_samples must be at least 100")
    required_pools = list(value["required_pools"])
    if not required_pools or any(pool not in POOL_TYPES for pool in required_pools):
        raise RepeatedEvalError(f"required_pools must be a non-empty subset of {list(POOL_TYPES)!r}")
    value["required_pools"] = sorted(set(required_pools))
    for key in (
        "minimum_effect",
        "max_cost_increase_ratio",
        "max_latency_increase_ratio",
        "cost_absolute_tolerance_usd",
        "latency_absolute_tolerance_seconds",
        "family_non_regression_tolerance",
        "risk_non_regression_tolerance",
    ):
        value[key] = float(value[key])
        if value[key] < 0:
            raise RepeatedEvalError(f"{key} must be non-negative")
    value["confidence_level"] = float(value["confidence_level"])
    if not 0.0 < value["confidence_level"] < 1.0:
        raise RepeatedEvalError("confidence_level must be between zero and one")
    value["bootstrap_samples"] = int(value["bootstrap_samples"])
    value["bootstrap_seed"] = int(value["bootstrap_seed"])
    value["minimum_repeats"] = int(value["minimum_repeats"])
    return value


def _identity_summary(observations: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, Any], list[str]]:
    summary: dict[str, Any] = {}
    errors: list[str] = []
    for arm in ARMS:
        identities = [observation.get("identity") for observation in observations[arm]]
        projections = [identity.get("projection_sha256") for identity in identities if isinstance(identity, dict)]
        if not projections:
            summary[arm] = {"observation_count": 0, "identity": None}
            errors.append(f"{arm}:missing_identity")
            continue
        if len(set(projections)) != 1:
            errors.append(f"{arm}:identity_changed_across_repeats")
        identity = dict(identities[0])
        summary[arm] = {"observation_count": len(observations[arm]), "identity": identity}
        errors.extend(f"{arm}:{error}" for error in _identity_errors(identity, expected_arm=arm))
    if all(summary.get(arm, {}).get("identity") for arm in ARMS):
        models = [summary[arm]["identity"]["model"] for arm in ARMS]
        model_keys = {(model["id"], model["revision"], model["sha256"]) for model in models}
        if len(model_keys) != 1:
            errors.append("base_model_identity_differs_across_arms")
        for component in ("runtime", "tools", "environment"):
            component_keys = {canonical_sha256(summary[arm]["identity"][component]) for arm in ARMS}
            if len(component_keys) != 1:
                errors.append(f"{component}_identity_differs_across_arms")
        trace_adapter = summary["trace_only"]["identity"].get("adapter") or {}
        fr_adapter = summary["flightrecorder"]["identity"].get("adapter") or {}
        if trace_adapter.get("sha256") == fr_adapter.get("sha256"):
            errors.append("trace_only_and_flightrecorder_adapters_are_identical")
    return summary, sorted(set(errors))


def _case_maps(observations: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, dict[tuple[Any, ...], dict[str, Any]]], list[str]]:
    maps: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = {arm: {} for arm in ARMS}
    errors: list[str] = []
    for arm in ARMS:
        for observation in observations[arm]:
            if observation.get("passed") is not True:
                errors.append(f"{arm}:blocked_observation")
            pool = observation.get("pool") or {}
            for case in observation.get("cases", []):
                key = (
                    pool.get("type"),
                    pool.get("id"),
                    case.get("scenario_id"),
                    case.get("scenario_sha256"),
                    observation.get("repeat_index"),
                    observation.get("seed"),
                    observation.get("decoding", {}).get("config_sha256"),
                )
                if key in maps[arm]:
                    errors.append(f"{arm}:duplicate_pair_key:{key!r}")
                maps[arm][key] = case
    return maps, sorted(set(errors))


def _coverage_summary(observations: dict[str, list[dict[str, Any]]], policy: dict[str, Any]) -> dict[str, Any]:
    arm_pools = {
        arm: sorted({str(observation.get("pool", {}).get("type") or "") for observation in rows if observation.get("pool")})
        for arm, rows in observations.items()
    }
    required = set(policy["required_pools"])
    missing = {arm: sorted(required - set(pools)) for arm, pools in arm_pools.items()}
    repeat_counts: dict[str, dict[str, int]] = {arm: {} for arm in ARMS}
    all_counts: list[int] = []
    for arm in ARMS:
        grouped: dict[tuple[str, str, str], set[tuple[int, int]]] = {}
        for observation in observations[arm]:
            pool = observation.get("pool") or {}
            for case in observation.get("cases", []):
                group = (str(pool.get("type") or ""), str(pool.get("id") or ""), str(case.get("scenario_id") or ""))
                grouped.setdefault(group, set()).add((int(observation.get("repeat_index") or 0), int(observation.get("seed") or 0)))
        for group, repeats in grouped.items():
            label = "/".join(group)
            repeat_counts[arm][label] = len(repeats)
            all_counts.append(len(repeats))
    minimum = min(all_counts) if all_counts else 0
    return {
        "passed": all(not values for values in missing.values()),
        "required_pools": policy["required_pools"],
        "arm_pools": arm_pools,
        "missing_pools": missing,
        "repeat_counts": repeat_counts,
        "minimum_observed_repeats": minimum,
        "minimum_repeats_passed": bool(all_counts) and minimum >= policy["minimum_repeats"],
    }


def _metric_values(case_map: dict[tuple[Any, ...], dict[str, Any]], keys: list[tuple[Any, ...]], metric: str) -> list[float]:
    if metric == "pass_rate":
        return [1.0 if case_map[key]["passed"] else 0.0 for key in keys]
    return [float(case_map[key]["score"]) / 100.0 for key in keys]


def _non_regression_summary(
    case_maps: dict[str, dict[tuple[Any, ...], dict[str, Any]]],
    paired: bool,
    policy: dict[str, Any],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    if not paired:
        for check_id in ("critical_safety", "tool_schema", "cost", "latency", "family", "risk"):
            _add_check(checks, f"{check_id}_non_regression", False, None, "paired observations required", "non_regression")
        return {"checks": checks, "metrics": metrics}

    keys = sorted(case_maps["flightrecorder"])
    candidate = [case_maps["flightrecorder"][key] for key in keys]
    for reference_arm in ("baseline", "trace_only"):
        reference = [case_maps[reference_arm][key] for key in keys]
        prefix = f"vs_{reference_arm}"
        candidate_critical = [failure for case in candidate for failure in case["critical_failures"]]
        reference_critical = [failure for case in reference for failure in case["critical_failures"]]
        new_critical_by_pair = [
            {"pair_key": list(key), "new_ids": sorted(set(candidate_case["critical_failures"]) - set(reference_case["critical_failures"]))}
            for key, candidate_case, reference_case in zip(keys, candidate, reference)
            if set(candidate_case["critical_failures"]) - set(reference_case["critical_failures"])
        ]
        new_critical = sorted({failure for row in new_critical_by_pair for failure in row["new_ids"]})
        critical_passed = len(candidate_critical) <= len(reference_critical) and not new_critical_by_pair
        _add_check(
            checks,
            f"critical_safety_non_regression_{prefix}",
            critical_passed,
            {
                "candidate": len(candidate_critical),
                "reference": len(reference_critical),
                "new_ids": new_critical,
                "new_by_pair": new_critical_by_pair,
            },
            {"candidate_count": {"<=": len(reference_critical)}, "new_ids": [], "new_by_pair": []},
            "non_regression",
        )

        candidate_tool_invalid = sum(1 for case in candidate if case["tool_schema_valid"] is not True)
        reference_tool_invalid = sum(1 for case in reference if case["tool_schema_valid"] is not True)
        new_tool_invalid = [
            list(key)
            for key, candidate_case, reference_case in zip(keys, candidate, reference)
            if candidate_case["tool_schema_valid"] is not True and reference_case["tool_schema_valid"] is True
        ]
        _add_check(
            checks,
            f"tool_schema_non_regression_{prefix}",
            candidate_tool_invalid <= reference_tool_invalid and not new_tool_invalid,
            {
                "candidate_invalid": candidate_tool_invalid,
                "reference_invalid": reference_tool_invalid,
                "new_invalid_pair_keys": new_tool_invalid,
            },
            {"candidate_invalid": {"<=": reference_tool_invalid}, "new_invalid_pair_keys": []},
            "non_regression",
        )

        cost = _operational_non_regression(candidate, reference, "cost_usd", policy["max_cost_increase_ratio"], policy["cost_absolute_tolerance_usd"])
        latency = _operational_non_regression(candidate, reference, "latency_seconds", policy["max_latency_increase_ratio"], policy["latency_absolute_tolerance_seconds"])
        _add_check(checks, f"cost_non_regression_{prefix}", cost["passed"], cost, {"maximum_allowed": cost["maximum_allowed"]}, "non_regression")
        _add_check(checks, f"latency_non_regression_{prefix}", latency["passed"], latency, {"maximum_allowed": latency["maximum_allowed"]}, "non_regression")

        family = _group_non_regression(candidate, reference, "task_family", policy["family_non_regression_tolerance"])
        risk = _group_non_regression(candidate, reference, "risk_tier", policy["risk_non_regression_tolerance"])
        _add_check(checks, f"family_non_regression_{prefix}", family["passed"], family["groups"], "candidate rate >= reference rate - tolerance", "non_regression")
        _add_check(checks, f"risk_non_regression_{prefix}", risk["passed"], risk["groups"], "candidate rate >= reference rate - tolerance", "non_regression")
        metrics[prefix] = {
            "critical": {
                "candidate": len(candidate_critical),
                "reference": len(reference_critical),
                "new_ids": new_critical,
                "new_by_pair": new_critical_by_pair,
            },
            "tool_schema": {
                "candidate_invalid": candidate_tool_invalid,
                "reference_invalid": reference_tool_invalid,
                "new_invalid_pair_keys": new_tool_invalid,
            },
            "cost": cost,
            "latency": latency,
            "family": family,
            "risk": risk,
        }
    return {"checks": checks, "metrics": metrics}


def _operational_non_regression(
    candidate: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    field: str,
    ratio: float,
    absolute_tolerance: float,
) -> dict[str, Any]:
    candidate_values = [case[field] for case in candidate if case[field] is not None]
    reference_values = [case[field] for case in reference if case[field] is not None]
    complete = len(candidate_values) == len(candidate) and len(reference_values) == len(reference)
    candidate_mean = statistics.fmean(candidate_values) if candidate_values else None
    reference_mean = statistics.fmean(reference_values) if reference_values else None
    maximum = reference_mean * (1.0 + ratio) + absolute_tolerance if reference_mean is not None else None
    return {
        "passed": bool(complete and candidate_mean is not None and maximum is not None and candidate_mean <= maximum),
        "complete": complete,
        "candidate_mean": candidate_mean,
        "reference_mean": reference_mean,
        "maximum_allowed": maximum,
    }


def _group_non_regression(
    candidate: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    field: str,
    tolerance: float,
) -> dict[str, Any]:
    groups = sorted({str(case[field]) for case in candidate + reference})
    rows = []
    for group in groups:
        candidate_values = [1.0 if case["passed"] else 0.0 for case in candidate if str(case[field]) == group]
        reference_values = [1.0 if case["passed"] else 0.0 for case in reference if str(case[field]) == group]
        candidate_rate = statistics.fmean(candidate_values) if candidate_values else None
        reference_rate = statistics.fmean(reference_values) if reference_values else None
        passed = bool(candidate_rate is not None and reference_rate is not None and candidate_rate + tolerance >= reference_rate)
        rows.append({"group": group, "candidate_rate": candidate_rate, "reference_rate": reference_rate, "tolerance": tolerance, "passed": passed})
    return {"passed": bool(rows) and all(row["passed"] for row in rows), "groups": rows}


def _add_check(
    checks: list[dict[str, Any]],
    check_id: str,
    passed: bool,
    actual: Any,
    expected: Any,
    semantics: str,
) -> None:
    checks.append({"id": check_id, "passed": bool(passed), "semantics": semantics, "actual": actual, "expected": expected})


def _artifact_ref(path: Path, output_base: Path | None) -> dict[str, Any]:
    if path.is_symlink():
        raise RepeatedEvalError(f"Source artifact must not be a symlink: {path}")
    path = path.resolve(strict=True)
    public_safe = False
    display = str(path)
    if output_base is not None:
        try:
            relative = path.relative_to(output_base.resolve(strict=False))
            display = relative.as_posix()
            public_safe = _is_safe_relative(display)
        except ValueError:
            public_safe = False
    return {
        "path": display,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "schema_version": str(load_json(path).get("schema_version") or ""),
        "public_safe": public_safe,
    }


def _verify_and_resolve_ref(ref: dict[str, Any], base: Path, errors: list[str], label: str) -> Path:
    path_value = ref.get("path")
    if not isinstance(path_value, str) or not _is_safe_relative(path_value):
        errors.append(f"{label}: source path is not a safe relative path")
        return base / "__invalid__"
    candidate = base / path_value
    if candidate.is_symlink():
        errors.append(f"{label}: source artifact must not be a symlink")
    path = candidate.resolve(strict=False)
    try:
        path.relative_to(base.resolve(strict=False))
    except ValueError:
        errors.append(f"{label}: source path escapes evidence directory")
        return path
    if not path.exists() or not path.is_file() or path.is_symlink():
        errors.append(f"{label}: source artifact is missing or not a regular file")
        return path
    if path.stat().st_size != ref.get("size_bytes"):
        errors.append(f"{label}: source artifact size changed")
    if sha256_file(path) != ref.get("sha256"):
        errors.append(f"{label}: source artifact SHA-256 changed")
    return path


def _is_safe_relative(value: str) -> bool:
    if not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts and "." not in path.parts


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) and converted >= 0 else None


def _validation_result(schema_version: str, errors: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "hfr.agentic_eval_semantic_validation.v1",
        "artifact_schema_version": schema_version,
        "passed": not errors,
        "error_count": len(errors),
        "errors": errors,
    }
