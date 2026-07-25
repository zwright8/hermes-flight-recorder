"""Tau-3 promotion decision and publication preflight builder.

The builder consumes public-safe evidence artifacts, binds them by content
hash, and emits a create-once decision. It deliberately does not open sealed
payloads or copy local source paths into the output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .atomic_json import atomic_write_json_cas
from .path_safety import path_has_symlink_component
from .schema_registry import SchemaRegistryError, check_schema_contract

TAU3_PROMOTION_PREFLIGHT_SCHEMA_VERSION = "hfr.tau3_promotion_publication_preflight.v1"
TAU3_POST_PUBLICATION_RECORD_SCHEMA_VERSION = "hfr.tau3_post_publication_record.v1"

KNOWN_SCHEMA_BY_INPUT = {
    "sealed_public_evaluation_report": "tau3_evaluation",
    "sealed_grid_completeness": "tau3_sealed_grid_completeness",
    "sealed_authorization": "tau3_sealed_authorization",
    "candidate_lock": "tau3_candidate_lock",
    "postlock_attempt_ledger": "tau3_candidate_attempt_ledger",
}

REQUIRED_EVALUATION_CHECK_IDS = (
    "source_results_valid",
    "identical_harness",
    "unique_paired_results",
    "safety_non_inferiority_vs_base",
)
EVIDENCE_INPUT_IDS = (
    "sealed_public_evaluation_report",
    "sealed_grid_completeness",
    "sealed_authorization",
    "candidate_lock",
    "postlock_attempt_ledger",
    "protocol_lineage_attestation",
    "readiness_validation",
    "budget_evidence",
    "license_evidence",
    "contamination_evidence",
    "redaction_evidence",
)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
HF_REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")
PRIVATE_TEXT_RE = re.compile(r"(^/|[A-Za-z]:[\\/]|\\\\|/Users/|/home/|/tmp/|localhost|127\.0\.0\.1|0\.0\.0\.0)")
RAW_SEALED_KEYS = {
    "expected_actions",
    "evaluation_criteria",
    "global_simulation_guidelines",
    "grader",
    "grader_secret",
    "messages",
    "persona_config",
    "policy",
    "prompt",
    "prompts",
    "raw_data",
    "raw_payload",
    "review",
    "reviews",
    "sealed_prompt",
    "sealed_task",
    "target_state",
    "tasks",
    "tool_defs",
    "user_scenario",
}
PRIVATE_IDENTIFIER_KEYS = {
    "api_base",
    "api_key",
    "device_id",
    "device_identifier",
    "endpoint_url",
    "hostname",
    "local_path",
    "private_path",
    "serial_number",
}


class Tau3PromotionPreflightError(ValueError):
    """Raised when promotion/publication preflight cannot be built safely."""


@dataclass(frozen=True)
class _JsonArtifact:
    label: str
    payload: dict[str, Any]
    sha256: str
    size: int
    schema_passed: bool
    schema_errors: tuple[str, ...]


def build_tau3_promotion_preflight(
    *,
    sealed_public_evaluation_report: str | Path,
    sealed_grid_completeness: str | Path,
    sealed_authorization: str | Path,
    candidate_lock: str | Path,
    postlock_attempt_ledger: str | Path,
    protocol_lineage_attestation: str | Path,
    readiness_validation: str | Path,
    budget_evidence: str | Path,
    license_evidence: str | Path,
    contamination_evidence: str | Path,
    redaction_evidence: str | Path,
    out: str | Path,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Create a hash-bound fail-closed Tau-3 promotion/publication decision."""

    target = Path(out)
    if target.exists():
        raise Tau3PromotionPreflightError(f"promotion preflight output already exists: {target}")

    artifacts = _load_inputs(
        {
            "sealed_public_evaluation_report": sealed_public_evaluation_report,
            "sealed_grid_completeness": sealed_grid_completeness,
            "sealed_authorization": sealed_authorization,
            "candidate_lock": candidate_lock,
            "postlock_attempt_ledger": postlock_attempt_ledger,
            "protocol_lineage_attestation": protocol_lineage_attestation,
            "readiness_validation": readiness_validation,
            "budget_evidence": budget_evidence,
            "license_evidence": license_evidence,
            "contamination_evidence": contamination_evidence,
            "redaction_evidence": redaction_evidence,
        }
    )

    raw_violations = _public_safety_violations([artifact.payload for artifact in artifacts.values()])
    if raw_violations:
        raise Tau3PromotionPreflightError(
            "promotion preflight input contains forbidden sealed/private material: "
            + "; ".join(raw_violations[:20])
        )

    predicates = _predicate_results(artifacts)
    blocking = [predicate for predicate, passed in predicates.items() if not passed]
    allowed = not blocking

    payload: dict[str, Any] = {
        "schema_version": TAU3_PROMOTION_PREFLIGHT_SCHEMA_VERSION,
        "created_at": created_at or _now_utc(),
        "allowed": allowed,
        "publication_status": _publication_status(allowed, artifacts["sealed_public_evaluation_report"].payload),
        "hf_revision": None,
        "hashes_only": True,
        "local_paths_included": False,
        "raw_payload_included": False,
        "private_identifiers_included": False,
        "evidence_bindings": {
            label: _artifact_binding(artifacts[label])
            for label in EVIDENCE_INPUT_IDS
        },
        "promotion_predicates": predicates,
        "failed_predicate_count": len(blocking),
        "blocking_reasons": blocking,
        "negative_result_withheld_honestly": (not allowed and _evaluation_is_valid_negative(artifacts["sealed_public_evaluation_report"].payload)),
        "notes": [
            "allowed is true only when every frozen promotion and publication predicate is proven before upload",
            "hf_revision is recorded only by the post-publication record after upload",
            "allowed=false with no Hugging Face revision is a valid result when sealed evidence is negative or inconclusive",
        ],
    }
    payload["decision_sha256"] = _canonical_sha256(payload)
    _assert_output_public_safe(payload)
    schema = check_schema_contract(payload, name_or_id="tau3_promotion_publication_preflight")
    if schema.get("passed") is not True:
        raise Tau3PromotionPreflightError("promotion preflight violates schema: " + "; ".join(str(error) for error in schema.get("errors", [])))
    atomic_write_json_cas(target, payload, expected_sha256=None, new_file_mode=0o444)
    return payload


def build_tau3_post_publication_record(
    *,
    preflight: str | Path,
    hf_revision: str,
    out: str | Path,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Create a post-upload record binding an allowed preflight to the HF revision."""

    target = Path(out)
    if target.exists():
        raise Tau3PromotionPreflightError(f"post-publication output already exists: {target}")
    if not HF_REVISION_RE.fullmatch(hf_revision):
        raise Tau3PromotionPreflightError("hf_revision must be a 40-64 character hex revision")
    preflight_artifact = _read_json_artifact(Path(preflight), "preflight")
    schema = check_schema_contract(preflight_artifact.payload, name_or_id="tau3_promotion_publication_preflight")
    if schema.get("passed") is not True:
        raise Tau3PromotionPreflightError("preflight violates schema: " + "; ".join(str(error) for error in schema.get("errors", [])))
    _assert_output_public_safe(preflight_artifact.payload)
    if preflight_artifact.payload.get("allowed") is not True:
        raise Tau3PromotionPreflightError("post-publication record requires an allowed preflight")
    if preflight_artifact.payload.get("publication_status") != "ready_for_publication":
        raise Tau3PromotionPreflightError("post-publication record requires a preflight ready_for_publication")
    if preflight_artifact.payload.get("hf_revision") is not None:
        raise Tau3PromotionPreflightError("preflight hf_revision must be null; revisions are recorded post-publication")

    record: dict[str, Any] = {
        "schema_version": TAU3_POST_PUBLICATION_RECORD_SCHEMA_VERSION,
        "created_at": created_at or _now_utc(),
        "status": "published",
        "hashes_only": True,
        "local_paths_included": False,
        "raw_payload_included": False,
        "private_identifiers_included": False,
        "preflight": {
            "sha256": preflight_artifact.sha256,
            "size": preflight_artifact.size,
            "schema_version": str(preflight_artifact.payload.get("schema_version") or ""),
            "allowed": True,
            "publication_status": "ready_for_publication",
            "decision_sha256": str(preflight_artifact.payload.get("decision_sha256") or ""),
        },
        "huggingface": {
            "revision": hf_revision,
            "revision_sha256": hashlib.sha256(hf_revision.encode("utf-8")).hexdigest(),
            "revision_format": "hex_40_to_64",
        },
    }
    record["record_sha256"] = _canonical_sha256(record)
    _assert_output_public_safe(record)
    schema = check_schema_contract(record, name_or_id="tau3_post_publication_record")
    if schema.get("passed") is not True:
        raise Tau3PromotionPreflightError("post-publication record violates schema: " + "; ".join(str(error) for error in schema.get("errors", [])))
    atomic_write_json_cas(target, record, expected_sha256=None, new_file_mode=0o444)
    return record


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed-public-evaluation-report", type=Path, required=True)
    parser.add_argument("--sealed-grid-completeness", type=Path, required=True)
    parser.add_argument("--sealed-authorization", type=Path, required=True)
    parser.add_argument("--candidate-lock", type=Path, required=True)
    parser.add_argument("--postlock-attempt-ledger", type=Path, required=True)
    parser.add_argument("--protocol-lineage-attestation", type=Path, required=True)
    parser.add_argument("--readiness-validation", type=Path, required=True)
    parser.add_argument("--budget-evidence", type=Path, required=True)
    parser.add_argument("--license-evidence", type=Path, required=True)
    parser.add_argument("--contamination-evidence", type=Path, required=True)
    parser.add_argument("--redaction-evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--created-at")
    return parser


def build_post_publication_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Tau-3 post-publication record.")
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--hf-revision", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--created-at")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        decision = build_tau3_promotion_preflight(
            sealed_public_evaluation_report=args.sealed_public_evaluation_report,
            sealed_grid_completeness=args.sealed_grid_completeness,
            sealed_authorization=args.sealed_authorization,
            candidate_lock=args.candidate_lock,
            postlock_attempt_ledger=args.postlock_attempt_ledger,
            protocol_lineage_attestation=args.protocol_lineage_attestation,
            readiness_validation=args.readiness_validation,
            budget_evidence=args.budget_evidence,
            license_evidence=args.license_evidence,
            contamination_evidence=args.contamination_evidence,
            redaction_evidence=args.redaction_evidence,
            out=args.out,
            created_at=args.created_at,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps({"allowed": decision["allowed"], "publication_status": decision["publication_status"], "decision_sha256": decision["decision_sha256"]}, indent=2, sort_keys=True))
    return 0


def post_publication_main(argv: list[str] | None = None) -> int:
    args = build_post_publication_arg_parser().parse_args(argv)
    try:
        record = build_tau3_post_publication_record(
            preflight=args.preflight,
            hf_revision=args.hf_revision,
            out=args.out,
            created_at=args.created_at,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps({"status": record["status"], "hf_revision": record["huggingface"]["revision"], "record_sha256": record["record_sha256"]}, indent=2, sort_keys=True))
    return 0


def _load_inputs(paths: dict[str, str | Path]) -> dict[str, _JsonArtifact]:
    return {label: _read_json_artifact(Path(path), label) for label, path in paths.items()}


def _read_json_artifact(path: Path, label: str) -> _JsonArtifact:
    if path_has_symlink_component(path, include_leaf=True):
        raise Tau3PromotionPreflightError(f"{label} must not contain symlink components")
    if not path.is_file():
        raise Tau3PromotionPreflightError(f"{label} does not exist")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if before.st_dev != after.st_dev or before.st_ino != after.st_ino or before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise Tau3PromotionPreflightError(f"{label} changed while being read")
    raw = b"".join(chunks)
    if len(raw) != before.st_size:
        raise Tau3PromotionPreflightError(f"{label} read size mismatch")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise Tau3PromotionPreflightError(f"{label} must contain a JSON object")
    schema_passed = True
    schema_errors: tuple[str, ...] = ()
    schema_name = KNOWN_SCHEMA_BY_INPUT.get(label)
    if schema_name is not None:
        try:
            result = check_schema_contract(payload, name_or_id=schema_name)
        except SchemaRegistryError as exc:
            schema_passed = False
            schema_errors = (str(exc),)
        else:
            schema_passed = result.get("passed") is True
            schema_errors = tuple(str(error) for error in result.get("errors", []))
    return _JsonArtifact(
        label=label,
        payload=payload,
        sha256=hashlib.sha256(raw).hexdigest(),
        size=len(raw),
        schema_passed=schema_passed,
        schema_errors=schema_errors,
    )


def _predicate_results(artifacts: dict[str, _JsonArtifact]) -> dict[str, bool]:
    evaluation = artifacts["sealed_public_evaluation_report"].payload
    grid = artifacts["sealed_grid_completeness"].payload
    authorization = artifacts["sealed_authorization"].payload
    lock = artifacts["candidate_lock"].payload
    ledger = artifacts["postlock_attempt_ledger"].payload
    lineage = artifacts["protocol_lineage_attestation"].payload
    readiness = artifacts["readiness_validation"].payload
    budget = artifacts["budget_evidence"].payload
    license_evidence = artifacts["license_evidence"].payload
    contamination = artifacts["contamination_evidence"].payload
    redaction = artifacts["redaction_evidence"].payload

    schema_ok = {label: artifact.schema_passed for label, artifact in artifacts.items()}
    base_predicates = {
        "schema_contracts_passed": all(schema_ok.values()),
        "sealed_public_evaluation_report_valid": schema_ok["sealed_public_evaluation_report"] and evaluation.get("mode") == "sealed" and evaluation.get("public_payload_scan", {}).get("passed") is True,
        "sealed_grid_complete": schema_ok["sealed_grid_completeness"] and grid.get("passed") is True and grid.get("status") == "complete",
        "sealed_authorization_valid": schema_ok["sealed_authorization"] and authorization.get("authorized") is True,
        "candidate_lock_bound": schema_ok["candidate_lock"] and lock.get("sealed_access_authorized") is True,
        "postlock_attempt_ledger_bound": schema_ok["postlock_attempt_ledger"] and _ledger_binds_lock(ledger, artifacts["candidate_lock"].sha256),
        "protocol_lineage_attestation_passed": lineage.get("schema_version") == "hfr.tau3_protocol_lineage_attestation.v1" and lineage.get("passed") is True,
        "readiness_validation_passed": _generic_passed(readiness, "readiness_validation"),
        "budget_passed": _generic_passed(budget, "budget"),
        "license_passed": _generic_passed(license_evidence, "license"),
        "contamination_passed": _contamination_passed(contamination),
        "redaction_passed": _redaction_passed(redaction),
        "hash_bindings_replay": _hash_bindings_replay(artifacts),
        "no_raw_sealed_payloads": True,
        "no_private_identifiers": True,
    }
    return {
        **base_predicates,
        "candidate_beats_strongest_comparator": _candidate_beats_strongest_comparator(evaluation),
        "comparator_bootstrap_ci_excludes_zero": _strongest_comparator_ci_excludes_zero(evaluation),
        "candidate_beats_base": _candidate_beats_base(evaluation),
        "safety_non_inferiority_passed": _evaluation_check(evaluation, "safety_non_inferiority_vs_base") and _all_effect_flag(evaluation, "per_domain_non_inferiority_passed"),
        "per_domain_non_inferiority_passed": _all_effect_flag(evaluation, "per_domain_non_inferiority_passed"),
        "harness_equivalence_passed": evaluation.get("harness", {}).get("passed") is True,
        "required_evaluation_checks_passed": evaluation.get("passed") is True and evaluation.get("promotion_ready") is True and all(_promotion_predicate(evaluation, predicate) for predicate in REQUIRED_EVALUATION_CHECK_IDS),
    }


def _hash_bindings_replay(artifacts: dict[str, _JsonArtifact]) -> bool:
    grid = artifacts["sealed_grid_completeness"].payload.get("bindings", {})
    auth = artifacts["sealed_authorization"].payload
    ledger = artifacts["postlock_attempt_ledger"].payload
    return (
        isinstance(grid, dict)
        and grid.get("authorization_sha256") == artifacts["sealed_authorization"].sha256
        and grid.get("candidate_lock_sha256") == artifacts["candidate_lock"].sha256
        and auth.get("candidate_lock", {}).get("sha256") == artifacts["candidate_lock"].sha256
        and ledger.get("lock", {}).get("sha256") == artifacts["candidate_lock"].sha256
    )


def _ledger_binds_lock(ledger: dict[str, Any], lock_sha256: str) -> bool:
    lock = ledger.get("lock")
    if not isinstance(lock, dict) or lock.get("sha256") != lock_sha256:
        return False
    lock_created_at = _parse_time(lock.get("created_at"))
    ledger_created_at = _parse_time(ledger.get("created_at"))
    if lock_created_at is None or ledger_created_at is None or ledger_created_at <= lock_created_at:
        return False
    for attempt in ledger.get("attempts", []):
        if not isinstance(attempt, dict):
            return False
        for ref_key in ("intent", "outcome", "training_receipt"):
            ref = attempt.get(ref_key)
            if ref is not None and not isinstance(ref, dict):
                return False
        for binding_key, lock_key in (
            ("protocol_sha256", "protocol_sha256"),
            ("protocol_signature", "protocol_signature"),
            ("dataset_manifest_sha256", "dataset_manifest_sha256"),
            ("dataset_files_sha256", "dataset_files_sha256"),
            ("adapter_tree_sha256", "adapter_tree_sha256"),
        ):
            value = attempt.get("bindings", {}).get(binding_key)
            if value is not None and lock.get(lock_key) is not None and value != lock.get(lock_key):
                return False
    return True


def _candidate_beats_base(evaluation: dict[str, Any]) -> bool:
    macro = evaluation.get("metrics", {}).get("macro_pass1", {})
    return _number(macro.get("adapter")) is not None and _number(macro.get("base")) is not None and macro["adapter"] > macro["base"]


def _candidate_beats_strongest_comparator(evaluation: dict[str, Any]) -> bool:
    macro = evaluation.get("metrics", {}).get("macro_pass1", {})
    adapter = _number(macro.get("adapter"))
    comparators = [_number(macro.get("comparator_1")), _number(macro.get("comparator_2"))]
    if adapter is None or any(value is None for value in comparators):
        return False
    return adapter > max(value for value in comparators if value is not None)


def _strongest_comparator_ci_excludes_zero(evaluation: dict[str, Any]) -> bool:
    macro = evaluation.get("metrics", {}).get("macro_pass1", {})
    comparator_values = {arm: _number(macro.get(arm)) for arm in ("comparator_1", "comparator_2")}
    if any(value is None for value in comparator_values.values()):
        return False
    strongest = max(comparator_values, key=lambda arm: comparator_values[arm] if comparator_values[arm] is not None else -1.0)
    effect = evaluation.get("effects", {}).get(strongest, {}).get("domain_stratified_macro_pass1", {})
    interval = effect.get("confidence_interval", {})
    return _number(effect.get("mean_difference")) is not None and effect["mean_difference"] > 0 and _number(interval.get("lower")) is not None and interval["lower"] > 0


def _all_effect_flag(evaluation: dict[str, Any], key: str) -> bool:
    effects = evaluation.get("effects")
    if not isinstance(effects, dict):
        return False
    for arm in ("base", "comparator_1", "comparator_2"):
        if not isinstance(effects.get(arm), dict) or effects[arm].get(key) is not True:
            return False
    return True


def _evaluation_check(evaluation: dict[str, Any], check_id: str) -> bool:
    for check in evaluation.get("checks", []):
        if isinstance(check, dict) and check.get("id") == check_id:
            return check.get("passed") is True
    return False


def _promotion_predicate(evaluation: dict[str, Any], predicate: str) -> bool:
    checks = {check.get("id"): check.get("passed") for check in evaluation.get("checks", []) if isinstance(check, dict)}
    return checks.get(predicate) is True


def _generic_passed(payload: dict[str, Any], kind: str) -> bool:
    if payload.get("passed") is True:
        return True
    if payload.get("valid") is True:
        return True
    if payload.get(f"{kind}_passed") is True:
        return True
    status = payload.get("status") or payload.get("readiness")
    return status in {"approved", "complete", "passed", "ready", "ready_for_publication_review", "ready_for_review"}


def _contamination_passed(payload: dict[str, Any]) -> bool:
    return _generic_passed(payload, "contamination") and payload.get("unresolved_leakage") is not True and payload.get("leakage_found") is not True


def _redaction_passed(payload: dict[str, Any]) -> bool:
    return _generic_passed(payload, "redaction") and payload.get("secrets_found") is not True and payload.get("unredacted_sensitive_data") is not True


def _evaluation_is_valid_negative(evaluation: dict[str, Any]) -> bool:
    return evaluation.get("schema_version") == "hfr.tau3_evaluation.v1" and evaluation.get("mode") == "sealed" and evaluation.get("promotion_ready") is False


def _publication_status(allowed: bool, evaluation: dict[str, Any]) -> str:
    if allowed:
        return "ready_for_publication"
    if _evaluation_is_valid_negative(evaluation):
        return "withheld_negative_result"
    return "blocked"


def _artifact_binding(artifact: _JsonArtifact) -> dict[str, Any]:
    return {
        "sha256": artifact.sha256,
        "size": artifact.size,
        "schema_version": str(artifact.payload.get("schema_version") or ""),
        "schema_passed": artifact.schema_passed,
    }


def _public_safety_violations(values: list[Any]) -> list[str]:
    violations: list[str] = []
    for index, value in enumerate(values):
        _collect_public_safety_violations(value, f"$[{index}]", violations)
    return violations


def _collect_public_safety_violations(value: Any, location: str, violations: list[str]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            key_text = str(key)
            if key_text in RAW_SEALED_KEYS:
                violations.append(f"{location}.{key_text}")
            if key_text in PRIVATE_IDENTIFIER_KEYS and not _is_public_safe_identifier_value(nested):
                violations.append(f"{location}.{key_text}")
            _collect_public_safety_violations(nested, f"{location}.{key_text}", violations)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _collect_public_safety_violations(nested, f"{location}[{index}]", violations)
    elif isinstance(value, str) and PRIVATE_TEXT_RE.search(value):
        violations.append(location)


def _is_public_safe_identifier_value(value: Any) -> bool:
    if value is None or value is False:
        return True
    if isinstance(value, str) and SHA256_RE.fullmatch(value):
        return True
    return False


def _assert_output_public_safe(payload: dict[str, Any]) -> None:
    violations = _public_safety_violations([payload])
    if violations:
        raise Tau3PromotionPreflightError("promotion preflight output contains forbidden material: " + "; ".join(violations[:20]))


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
