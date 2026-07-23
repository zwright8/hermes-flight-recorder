"""Fail-closed public authorization for Tau-3 sealed benchmark access."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .atomic_json import atomic_write_json_cas
from .path_safety import path_has_symlink_component
from .schema_registry import check_schema_contract

TAU3_SEALED_AUTHORIZATION_SCHEMA_VERSION = "hfr.tau3_sealed_authorization.v1"
TAU3_CANDIDATE_LOCK_SCHEMA_VERSION = "hfr.tau3_candidate_lock.v1"
TAU3_PROTOCOL_CONFIG_SCHEMA_VERSION = "hfr.tau3_protocol_config.v1"
TAU3_SEALED_SOURCE_SCHEMA_VERSION = "hfr.tau3_sealed_source_manifest.v1"

REQUIRED_SEEDS = (101, 202, 303, 404)
REQUIRED_ARMS = ("adapter", "base", "comparator_1", "comparator_2")
REQUIRED_DOMAINS = ("airline", "retail", "telecom")
REQUIRED_SEALED_TASK_COUNT = 100
CONTEXT_WINDOW = 16384

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")

PUBLIC_FORBIDDEN_KEYS = {
    "path",
    "paths",
    "local_path",
    "local_identity_path",
    "adapter_path",
    "source_path",
    "result_path",
    "output_dir",
    "tasks",
    "messages",
    "raw_data",
    "raw_payload",
    "policy",
    "tool_defs",
    "prompt",
    "context",
}


class Tau3SealedAuthorizationError(ValueError):
    """Raised when sealed benchmark authorization cannot be proven."""


@dataclass(frozen=True)
class _JsonArtifact:
    path: Path
    payload: dict[str, Any]
    sha256: str
    size: int


def create_tau3_sealed_authorization(
    *,
    candidate_lock: str | Path,
    protocol: str | Path,
    sealed_source_manifest: str | Path,
    out: str | Path,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Create a public-safe authorization artifact only when every gate passes."""

    target = Path(out)
    if target.exists():
        raise Tau3SealedAuthorizationError(f"authorization output already exists: {target}")
    lock_record = _read_json_artifact(Path(candidate_lock), "candidate lock")
    protocol_record = _read_json_artifact(Path(protocol), "protocol")
    sealed_record = _read_json_artifact(Path(sealed_source_manifest), "sealed source manifest")
    lock = lock_record.payload
    protocol_payload = protocol_record.payload
    sealed = sealed_record.payload

    lock_sha256 = lock_record.sha256
    protocol_sha256 = protocol_record.sha256
    sealed_sha256 = sealed_record.sha256
    created = created_at or _now_utc()

    _validate_candidate_lock(lock, protocol_sha256=protocol_sha256)
    _validate_protocol(protocol_payload, protocol_sha256=protocol_sha256, candidate_lock_sha256=lock_sha256, sealed_sha256=sealed_sha256)
    _validate_sealed_source(sealed, protocol_payload, sealed_sha256=sealed_sha256)
    _require_chronology(str(lock["created_at"]), created)

    authorization = _authorization_payload(
        lock=lock,
        lock_sha256=lock_sha256,
        protocol=protocol_payload,
        protocol_sha256=protocol_sha256,
        sealed_sha256=sealed_sha256,
        created_at=created,
    )
    _assert_public_safe(authorization)
    schema = check_schema_contract(authorization, name_or_id="tau3_sealed_authorization")
    if schema.get("passed") is not True:
        raise Tau3SealedAuthorizationError("authorization violates registered schema: " + "; ".join(str(error) for error in schema.get("errors", [])))
    atomic_write_json_cas(target, authorization, expected_sha256=None, new_file_mode=0o444)
    return {
        "schema_version": "hfr.tau3_sealed_authorization_result.v1",
        "out": str(target),
        "authorization_sha256": _sha256(target),
        "candidate_lock_sha256": lock_sha256,
        "protocol_sha256": protocol_sha256,
        "sealed_source_sha256": sealed_sha256,
        "authorized": True,
    }


def validate_tau3_sealed_authorization(
    *,
    authorization_path: str | Path,
    candidate_lock_path: str | Path,
    protocol_path: str | Path,
    sealed_source_manifest_path: str | Path,
    arm_id: str,
    seeds: tuple[int, ...],
    expected_tau_revision: str,
    expected_authorization_sha256: str | None = None,
) -> dict[str, Any]:
    """Replay authorization bindings before a sealed benchmark arm can start."""

    auth_record = _read_json_artifact(Path(authorization_path), "sealed authorization")
    lock_record = _read_json_artifact(Path(candidate_lock_path), "candidate lock")
    protocol_record = _read_json_artifact(Path(protocol_path), "protocol")
    sealed_record = _read_json_artifact(Path(sealed_source_manifest_path), "sealed source manifest")
    auth_sha256 = auth_record.sha256
    if expected_authorization_sha256 is not None and expected_authorization_sha256 != auth_sha256:
        raise Tau3SealedAuthorizationError("sealed authorization sha256 mismatch")
    auth = auth_record.payload
    schema = check_schema_contract(auth, name_or_id="tau3_sealed_authorization")
    if schema.get("passed") is not True:
        raise Tau3SealedAuthorizationError("sealed authorization violates registered schema: " + "; ".join(str(error) for error in schema.get("errors", [])))
    if auth.get("authorized") is not True:
        raise Tau3SealedAuthorizationError("sealed authorization is not authorized")
    _assert_public_safe(auth)

    lock = lock_record.payload
    protocol_payload = protocol_record.payload
    sealed = sealed_record.payload
    lock_sha256 = lock_record.sha256
    protocol_sha256 = protocol_record.sha256
    sealed_sha256 = sealed_record.sha256
    _validate_candidate_lock(lock, protocol_sha256=protocol_sha256)
    _validate_protocol(protocol_payload, protocol_sha256=protocol_sha256, candidate_lock_sha256=lock_sha256, sealed_sha256=sealed_sha256)
    _validate_sealed_source(sealed, protocol_payload, sealed_sha256=sealed_sha256)

    errors: list[str] = []
    if arm_id not in REQUIRED_ARMS:
        errors.append("arm_id is not authorized")
    if tuple(seeds) != REQUIRED_SEEDS:
        errors.append("seeds do not match sealed authorization")
    if expected_tau_revision != str(_dict(protocol_payload.get("tau_revision")).get("revision") or ""):
        errors.append("tau revision does not match protocol")
    try:
        _require_chronology(str(lock["created_at"]), str(auth.get("created_at") or ""))
    except Tau3SealedAuthorizationError as exc:
        errors.append(str(exc))
    expected = _authorization_payload(
        lock=lock,
        lock_sha256=lock_sha256,
        protocol=protocol_payload,
        protocol_sha256=protocol_sha256,
        sealed_sha256=sealed_sha256,
        created_at=str(auth.get("created_at") or ""),
    )
    expected["sha256"] = auth_sha256
    for key in ("candidate_lock", "protocol", "sealed_source", "frozen_contract", "model_identity_refs", "gates", "budget"):
        if auth.get(key) != expected.get(key):
            errors.append(f"{key} binding does not replay")
    if errors:
        raise Tau3SealedAuthorizationError("; ".join(errors))
    return {
        "path": "",
        "size": auth_record.size,
        "sha256": auth_sha256,
        "authorized": True,
        "candidate_lock_sha256": lock_sha256,
        "protocol_sha256": protocol_sha256,
        "sealed_source_sha256": sealed_sha256,
        "task_count": REQUIRED_SEALED_TASK_COUNT,
        "arms": list(REQUIRED_ARMS),
        "seeds": list(REQUIRED_SEEDS),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-lock", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--sealed-source-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--created-at")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        result = create_tau3_sealed_authorization(
            candidate_lock=args.candidate_lock,
            protocol=args.protocol,
            sealed_source_manifest=args.sealed_source_manifest,
            out=args.out,
            created_at=args.created_at,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _authorization_payload(
    *,
    lock: dict[str, Any],
    lock_sha256: str,
    protocol: dict[str, Any],
    protocol_sha256: str,
    sealed_sha256: str,
    created_at: str,
) -> dict[str, Any]:
    revision = str(_dict(protocol.get("tau_revision")).get("revision") or "")
    model_refs = _model_identity_refs(lock, protocol)
    gates = {
        "candidate_lock_valid": True,
        "chronology_lock_before_authorization": True,
        "protocol_binding_valid": True,
        "sealed_source_hash_only": True,
        "sealed_task_count_is_100": True,
        "seeds_exact": True,
        "arms_exact": True,
        "harness_tool_prompt_context_decoding_no_search_identical": True,
        "model_identity_equivalence_refs_present": True,
        "contamination_gate_passed": True,
        "redaction_gate_passed": True,
        "license_gate_passed": True,
        "safety_gate_passed": True,
        "budget_gate_passed": True,
        "public_artifact_contains_no_local_paths": True,
    }
    return {
        "schema_version": TAU3_SEALED_AUTHORIZATION_SCHEMA_VERSION,
        "created_at": created_at,
        "authorized": True,
        "hashes_only": True,
        "local_paths_included": False,
        "raw_payload_included": False,
        "candidate_lock": {
            "sha256": lock_sha256,
            "created_at": str(lock["created_at"]),
            "protocol_sha256": str(lock["protocol_sha256"]),
            "protocol_signature": str(lock["protocol_signature"]),
            "sealed_access_authorized": True,
        },
        "protocol": {
            "sha256": protocol_sha256,
            "signature_sha256": str(lock["protocol_signature"]),
            "signature_provenance": "candidate_lock.protocol_signature",
            "tau_revision": revision,
        },
        "sealed_source": {"manifest_sha256": sealed_sha256, "task_count": REQUIRED_SEALED_TASK_COUNT, "hashes_only": True},
        "frozen_contract": {
            "arms": list(REQUIRED_ARMS),
            "seeds": list(REQUIRED_SEEDS),
            "domains": list(REQUIRED_DOMAINS),
            "context_window": CONTEXT_WINDOW,
            "tool_contract_sha256": _canonical_sha256(_tool_contract(protocol)),
            "prompt_context_decoding_sha256": _canonical_sha256(_prompt_context_decoding_contract(protocol)),
            "harness_sha256": _canonical_sha256(_harness_contract(protocol)),
            "no_test_time_search": True,
        },
        "model_identity_refs": model_refs,
        "gates": gates,
        "budget": {"sha256": _canonical_sha256(_dict(protocol.get("budget"))), "declared": True, "passed": True},
    }


def _validate_candidate_lock(lock: dict[str, Any], *, protocol_sha256: str) -> None:
    schema = check_schema_contract(lock, name_or_id="tau3_candidate_lock")
    if schema.get("passed") is not True:
        raise Tau3SealedAuthorizationError("candidate lock violates registered schema: " + "; ".join(str(error) for error in schema.get("errors", [])))
    if lock.get("schema_version") != TAU3_CANDIDATE_LOCK_SCHEMA_VERSION:
        raise Tau3SealedAuthorizationError("candidate lock schema_version mismatch")
    if lock.get("protocol_sha256") != protocol_sha256:
        raise Tau3SealedAuthorizationError("candidate lock protocol_sha256 does not match frozen protocol")
    if lock.get("sealed_access_authorized") is not True:
        raise Tau3SealedAuthorizationError("candidate lock does not authorize sealed access")
    for key in ("protocol_signature", "candidate_identity_sha256", "adapter_tree_sha256"):
        if not isinstance(lock.get(key), str) or not HEX64_RE.fullmatch(str(lock.get(key))):
            raise Tau3SealedAuthorizationError(f"candidate lock missing {key}")


def _validate_protocol(protocol: dict[str, Any], *, protocol_sha256: str, candidate_lock_sha256: str, sealed_sha256: str) -> None:
    schema = check_schema_contract(protocol, name_or_id="tau3_protocol_config")
    if schema.get("passed") is not True:
        raise Tau3SealedAuthorizationError("protocol violates registered schema: " + "; ".join(str(error) for error in schema.get("errors", [])))
    if protocol.get("schema_version") != TAU3_PROTOCOL_CONFIG_SCHEMA_VERSION:
        raise Tau3SealedAuthorizationError("protocol schema_version mismatch")
    revision = _dict(protocol.get("tau_revision")).get("revision")
    if not isinstance(revision, str) or not HEX40_RE.fullmatch(revision):
        raise Tau3SealedAuthorizationError("protocol Tau revision must be an exact commit")
    declared_lock_sha256 = _protocol_candidate_lock_sha256(protocol)
    if declared_lock_sha256 is not None and declared_lock_sha256 != candidate_lock_sha256:
        raise Tau3SealedAuthorizationError("protocol candidate lock binding mismatch")
    split_hashes = _protocol_split_hashes(protocol)
    sealed_manifest = _dict(protocol.get("sealed_manifest"))
    if sealed_sha256 != split_hashes.get("sealed") or sealed_sha256 != sealed_manifest.get("manifest_sha256"):
        raise Tau3SealedAuthorizationError("protocol sealed source binding mismatch")
    if sealed_manifest.get("access_count") not in (0, None):
        raise Tau3SealedAuthorizationError("protocol sealed manifest must declare zero sealed access")
    _require_harness_contract(protocol)
    _require_model_identity_refs(protocol)
    _require_gates(protocol)
    if not isinstance(protocol_sha256, str) or not HEX64_RE.fullmatch(protocol_sha256):
        raise Tau3SealedAuthorizationError("protocol sha256 is invalid")


def _validate_sealed_source(sealed: dict[str, Any], protocol: dict[str, Any], *, sealed_sha256: str) -> None:
    schema = check_schema_contract(sealed, name_or_id="tau3_sealed_source_manifest")
    if schema.get("passed") is not True:
        raise Tau3SealedAuthorizationError("sealed source manifest violates registered schema: " + "; ".join(str(error) for error in schema.get("errors", [])))
    if sealed.get("schema_version") != TAU3_SEALED_SOURCE_SCHEMA_VERSION:
        raise Tau3SealedAuthorizationError("sealed source manifest schema_version mismatch")
    if sealed.get("hashes_only") is not True:
        raise Tau3SealedAuthorizationError("sealed source manifest must be hashes-only")
    if sealed.get("task_count") != REQUIRED_SEALED_TASK_COUNT:
        raise Tau3SealedAuthorizationError("sealed source manifest task_count must be exactly 100")
    entries = sealed.get("entries")
    if not isinstance(entries, list) or len(entries) != REQUIRED_SEALED_TASK_COUNT:
        raise Tau3SealedAuthorizationError("sealed source manifest entries must replay task_count=100")
    if str(_dict(protocol.get("tau_revision")).get("revision") or "") != sealed.get("source_revision"):
        raise Tau3SealedAuthorizationError("sealed source revision mismatch")
    split_hashes = _protocol_split_hashes(protocol)
    if sealed_sha256 != split_hashes.get("sealed"):
        raise Tau3SealedAuthorizationError("sealed source sha256 does not match protocol")


def _require_harness_contract(protocol: dict[str, Any]) -> None:
    harness = _dict(protocol.get("harness_contract"))
    decoding = _dict(harness.get("decoding"))
    expected = _harness_contract(protocol)
    actual = {
        "domains": harness.get("domains"),
        "context_window": harness.get("context_window"),
        "temperature": decoding.get("temperature"),
        "top_p": decoding.get("top_p"),
        "max_output_tokens": decoding.get("max_output_tokens"),
        "seeds": decoding.get("seeds"),
        "turn_limit": harness.get("turn_limit"),
        "retry_policy": harness.get("retry_policy"),
        "test_time_search": harness.get("test_time_search"),
        "no_test_time_search": harness.get("no_test_time_search"),
    }
    if actual != expected:
        raise Tau3SealedAuthorizationError("protocol harness does not match frozen sealed contract")


def _require_model_identity_refs(protocol: dict[str, Any]) -> None:
    refs = _model_identity_refs({}, protocol)
    for key in ("base_identity_sha256", "comparator_1_identity_sha256", "comparator_2_identity_sha256"):
        if not isinstance(refs.get(key), str) or not HEX64_RE.fullmatch(str(refs.get(key))):
            raise Tau3SealedAuthorizationError(f"protocol missing {key}")


def _require_gates(protocol: dict[str, Any]) -> None:
    contamination = _dict(protocol.get("contamination_attestation"))
    redaction = _dict(protocol.get("redaction_attestation"))
    if contamination.get("passed") is not True:
        raise Tau3SealedAuthorizationError("contamination gate did not pass")
    if redaction.get("passed") is not True:
        raise Tau3SealedAuthorizationError("redaction gate did not pass")
    licenses = protocol.get("licenses")
    if not isinstance(licenses, list) or not licenses:
        raise Tau3SealedAuthorizationError("license gate is missing")
    for index, record in enumerate(licenses):
        item = _dict(record)
        if item.get("status") != "approved" or item.get("training_allowed") is not True:
            raise Tau3SealedAuthorizationError(f"license gate did not pass for record {index}")
    candidate_contract = _dict(protocol.get("candidate_selection_contract"))
    if candidate_contract.get("passed") is not True:
        raise Tau3SealedAuthorizationError("candidate selection safety gate did not pass")
    budget = _dict(protocol.get("budget"))
    max_seconds = budget.get("max_seconds")
    has_pass_flag = budget.get("passed") is True
    has_frozen_budget = (
        isinstance(max_seconds, int)
        and not isinstance(max_seconds, bool)
        and max_seconds > 0
        and budget.get("reserved_final_eval") is True
        and budget.get("deny_when_final_eval_cannot_complete") is True
    )
    if not (has_pass_flag or has_frozen_budget):
        raise Tau3SealedAuthorizationError("budget gate did not pass")


def _model_identity_refs(lock: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    freeze = _dict(protocol.get("model_freeze"))
    comparators = freeze.get("comparators")
    comparator_rows = comparators if isinstance(comparators, list) else []
    comparator_1 = _dict(comparator_rows[0]) if len(comparator_rows) > 0 else {}
    comparator_2 = _dict(comparator_rows[1]) if len(comparator_rows) > 1 else {}
    return {
        "candidate_identity_sha256": lock.get("candidate_identity_sha256"),
        "adapter_tree_sha256": lock.get("adapter_tree_sha256"),
        "endpoint_model_sha256": lock.get("endpoint_model_sha256"),
        "base_identity_sha256": _dict(freeze.get("base_model")).get("local_identity_sha256"),
        "comparator_1_identity_sha256": comparator_1.get("local_identity_sha256"),
        "comparator_2_identity_sha256": comparator_2.get("local_identity_sha256"),
        "equivalence_refs_hash": _canonical_sha256(
            {
                "candidate_identity_sha256": lock.get("candidate_identity_sha256"),
                "adapter_tree_sha256": lock.get("adapter_tree_sha256"),
                "base": _public_model_ref(_dict(freeze.get("base_model"))),
                "comparator_1": _public_model_ref(comparator_1),
                "comparator_2": _public_model_ref(comparator_2),
            }
        ),
    }


def _public_model_ref(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "name_sha256": hashlib.sha256(str(record.get("name") or "").encode("utf-8")).hexdigest(),
        "revision_sha256": hashlib.sha256(str(record.get("revision") or "").encode("utf-8")).hexdigest(),
        "local_identity_sha256": record.get("local_identity_sha256"),
    }


def _harness_contract(protocol: dict[str, Any]) -> dict[str, Any]:
    harness = _dict(protocol.get("harness_contract"))
    decoding = _dict(harness.get("decoding"))
    return {
        "domains": list(REQUIRED_DOMAINS),
        "context_window": CONTEXT_WINDOW,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_output_tokens": 1024,
        "seeds": list(REQUIRED_SEEDS),
        "turn_limit": 30,
        "retry_policy": "none",
        "test_time_search": False,
        "no_test_time_search": True,
    } if harness or decoding else {}


def _tool_contract(protocol: dict[str, Any]) -> dict[str, Any]:
    harness = _dict(protocol.get("harness_contract"))
    return {
        "agent": "llm_agent",
        "user": "user_simulator",
        "auto_review": True,
        "review_mode": "full",
        "communication_protocol_enforced": True,
        "max_retries": 0,
        "hallucination_retries": 0,
        "test_time_search": False,
        "domain_contracts_sha256": _canonical_sha256(harness.get("domain_contracts", {})),
    }


def _prompt_context_decoding_contract(protocol: dict[str, Any]) -> dict[str, Any]:
    harness = _dict(protocol.get("harness_contract"))
    decoding = _dict(harness.get("decoding"))
    return {
        "context_window": CONTEXT_WINDOW,
        "turn_limit": 30,
        "temperature": decoding.get("temperature"),
        "top_p": decoding.get("top_p"),
        "max_output_tokens": decoding.get("max_output_tokens"),
        "seeds": decoding.get("seeds"),
        "prompt_contract_sha256": _canonical_sha256(harness.get("prompt_contract", {})),
    }


def _protocol_split_hashes(protocol: dict[str, Any]) -> dict[str, str]:
    hashes = _dict(_dict(protocol.get("tau_revision")).get("split_hashes"))
    splits = _dict(_dict(protocol.get("split_manifest")).get("splits"))
    for name, record in splits.items():
        if isinstance(record, dict) and isinstance(record.get("sha256"), str):
            hashes.setdefault(str(name), record["sha256"])
    return {str(key): str(value) for key, value in hashes.items() if isinstance(value, str)}


def _protocol_candidate_lock_sha256(protocol: dict[str, Any]) -> str | None:
    candidates = (
        _dict(protocol.get("candidate_selection_contract")),
        _dict(protocol.get("sealed_manifest")),
        _dict(protocol.get("protocol_manifest")),
    )
    for section in candidates:
        for key in ("candidate_lock_sha256", "candidate_lock_manifest_sha256", "adapter_candidate_lock_sha256"):
            value = section.get(key)
            if isinstance(value, str) and HEX64_RE.fullmatch(value):
                return value
    return None


def _require_chronology(lock_created_at: str, authorization_created_at: str) -> None:
    lock_dt = _parse_datetime(lock_created_at, "candidate lock created_at")
    auth_dt = _parse_datetime(authorization_created_at, "authorization created_at")
    if auth_dt <= lock_dt:
        raise Tau3SealedAuthorizationError("authorization must be created after candidate lock")


def _parse_datetime(value: str, label: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise Tau3SealedAuthorizationError(f"{label} is not ISO-8601") from exc


def _safe_input_file(path: Path, label: str) -> Path:
    if path_has_symlink_component(path, include_leaf=True):
        raise Tau3SealedAuthorizationError(f"{label} must not contain symlink components: {path}")
    if not path.is_file():
        raise Tau3SealedAuthorizationError(f"{label} does not exist: {path}")
    return path


def _read_json_artifact(path: Path, label: str) -> _JsonArtifact:
    _safe_input_file(path, label)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise Tau3SealedAuthorizationError(f"{label} could not be opened safely: {path}: {exc.strerror}") from exc
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
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise Tau3SealedAuthorizationError(f"{label} changed while being read: {path}")
    raw = b"".join(chunks)
    if len(raw) != before.st_size:
        raise Tau3SealedAuthorizationError(f"{label} read size mismatch: {path}")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise Tau3SealedAuthorizationError(f"{label} is not UTF-8: {path}") from exc
    except json.JSONDecodeError as exc:
        raise Tau3SealedAuthorizationError(f"invalid {label} JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise Tau3SealedAuthorizationError(f"{label} must contain a JSON object")
    return _JsonArtifact(path=path, payload=payload, sha256=hashlib.sha256(raw).hexdigest(), size=len(raw))


def _assert_public_safe(value: Any) -> None:
    violations: list[str] = []
    _collect_public_safety_violations(value, "$", violations)
    if violations:
        raise Tau3SealedAuthorizationError("authorization public artifact contains forbidden private material: " + "; ".join(violations))


def _collect_public_safety_violations(value: Any, location: str, violations: list[str]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            key_text = str(key)
            if key_text in PUBLIC_FORBIDDEN_KEYS:
                violations.append(f"{location}.{key_text}")
            _collect_public_safety_violations(nested, f"{location}.{key_text}", violations)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _collect_public_safety_violations(nested, f"{location}[{index}]", violations)
    elif isinstance(value, str):
        if value.startswith(("/", "./", "../")) or re.search(r"(^|[A-Za-z]):[\\/]", value):
            violations.append(location)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Tau3SealedAuthorizationError(f"invalid {label} JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise Tau3SealedAuthorizationError(f"{label} must contain a JSON object")
    return payload


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
