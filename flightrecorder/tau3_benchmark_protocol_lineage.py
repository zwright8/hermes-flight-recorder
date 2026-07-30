"""Blind custody and sealed-only protocol rotation for Tau-3 benchmarks."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .atomic_json import atomic_write_json_cas
from .path_safety import path_has_symlink_component
from .repeated_eval import canonical_sha256
from .schema_registry import check_schema_contract

TAU3_BLIND_GENERATOR_VALIDATION_SCHEMA_VERSION = "hfr.tau3_blind_generator_validation.v1"
TAU3_FRESH_CONTAMINATION_REPLAY_SCHEMA_VERSION = "hfr.tau3_fresh_contamination_replay.v1"
TAU3_BLIND_CUSTODY_RECEIPT_SCHEMA_VERSION = "hfr.tau3_blind_custody_receipt.v1"
TAU3_BENCHMARK_PROTOCOL_LINEAGE_SCHEMA_VERSION = "hfr.tau3_benchmark_protocol_lineage.v1"
TAU3_PROTOCOL_CONFIG_SCHEMA_VERSION = "hfr.tau3_protocol_config.v1"
TAU3_SEALED_SOURCE_MANIFEST_SCHEMA_VERSION = "hfr.tau3_sealed_source_manifest.v1"

REQUIRED_TASK_COUNT = 100
REQUIRED_DOMAINS = ("airline", "retail", "telecom")
ALLOWED_PROTOCOL_DELTA_PATHS = (
    "sealed_manifest",
    "split_manifest.source_manifest",
    "split_manifest.splits.sealed",
    "tau_revision.split_hashes.sealed",
)

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")


class Tau3BenchmarkProtocolLineageError(ValueError):
    """Raised when blind custody or protocol rotation cannot be proven."""


@dataclass(frozen=True)
class _JsonArtifact:
    path: Path
    payload: dict[str, Any]
    sha256: str
    size: int


def create_tau3_blind_custody_receipt(
    *,
    custody_id: str,
    sealed_source_manifest: str | Path,
    generator_validation: str | Path,
    fresh_contamination_replay: str | Path,
    retired_source_incident_sha256: str,
    out: str | Path,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Create one hash-only receipt after generator and disjointness replay."""

    target = Path(out)
    if target.exists():
        raise Tau3BenchmarkProtocolLineageError(f"custody receipt already exists: {target}")
    if not custody_id:
        raise Tau3BenchmarkProtocolLineageError("custody_id must be nonempty")
    _require_sha256(retired_source_incident_sha256, "retired source incident")

    sealed = _read_json_artifact(Path(sealed_source_manifest), "sealed source manifest")
    generator = _read_json_artifact(Path(generator_validation), "blind generator validation")
    contamination = _read_json_artifact(Path(fresh_contamination_replay), "fresh contamination replay")
    sealed_summary = _validate_sealed_source_manifest(sealed)
    generator_summary = _validate_generator_validation(generator, sealed=sealed)
    _validate_fresh_contamination(contamination, sealed=sealed)
    if generator_summary["domain_counts"] != sealed_summary["domain_counts"]:
        raise Tau3BenchmarkProtocolLineageError(
            "generator validation domain counts do not match the sealed source manifest"
        )

    receipt = {
        "schema_version": TAU3_BLIND_CUSTODY_RECEIPT_SCHEMA_VERSION,
        "created_at": created_at or _now_utc(),
        "custody_id_hash": hashlib.sha256(custody_id.encode("utf-8")).hexdigest(),
        "source_revision": sealed_summary["source_revision"],
        "sealed_source_manifest_sha256": sealed.sha256,
        "entries_sha256": sealed_summary["entries_sha256"],
        "task_count": REQUIRED_TASK_COUNT,
        "domain_counts": sealed_summary["domain_counts"],
        "generator_validation_sha256": generator.sha256,
        "fresh_contamination_replay_sha256": contamination.sha256,
        "retired_source_incident_sha256": retired_source_incident_sha256,
        "custody": {
            "hashes_only": True,
            "local_paths_included": False,
            "raw_payload_included": False,
            "searchable_plaintext_persisted": False,
            "retired_source_used": False,
            "payload_lifetime": "blocking_custodian_memory_only",
            "one_shot_prepared": True,
            "consumed": False,
        },
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    _check_schema(receipt, "tau3_blind_custody_receipt", "blind custody receipt")
    _assert_hash_only_public_artifact(receipt, "blind custody receipt")
    atomic_write_json_cas(target, receipt, expected_sha256=None, new_file_mode=0o444)
    return receipt


def validate_tau3_blind_custody_receipt(
    *,
    custody_receipt: str | Path,
    sealed_source_manifest: str | Path,
    generator_validation: str | Path,
    fresh_contamination_replay: str | Path,
    expected_retired_source_incident_sha256: str | None = None,
) -> dict[str, Any]:
    """Replay a blind custody receipt against all hash-only source evidence."""

    receipt = _read_json_artifact(Path(custody_receipt), "blind custody receipt")
    sealed = _read_json_artifact(Path(sealed_source_manifest), "sealed source manifest")
    generator = _read_json_artifact(Path(generator_validation), "blind generator validation")
    contamination = _read_json_artifact(Path(fresh_contamination_replay), "fresh contamination replay")

    _check_schema(receipt.payload, "tau3_blind_custody_receipt", "blind custody receipt")
    _assert_hash_only_public_artifact(receipt.payload, "blind custody receipt")
    _validate_receipt_self_seal(receipt.payload, field="receipt_sha256", label="blind custody receipt")
    sealed_summary = _validate_sealed_source_manifest(sealed)
    generator_summary = _validate_generator_validation(generator, sealed=sealed)
    _validate_fresh_contamination(contamination, sealed=sealed)

    expected = {
        "source_revision": sealed_summary["source_revision"],
        "sealed_source_manifest_sha256": sealed.sha256,
        "entries_sha256": sealed_summary["entries_sha256"],
        "task_count": REQUIRED_TASK_COUNT,
        "domain_counts": sealed_summary["domain_counts"],
        "generator_validation_sha256": generator.sha256,
        "fresh_contamination_replay_sha256": contamination.sha256,
    }
    for key, value in expected.items():
        if receipt.payload.get(key) != value:
            raise Tau3BenchmarkProtocolLineageError(f"blind custody receipt {key} mismatch")
    if generator_summary["domain_counts"] != sealed_summary["domain_counts"]:
        raise Tau3BenchmarkProtocolLineageError(
            "generator validation domain counts do not match the sealed source manifest"
        )
    if (
        expected_retired_source_incident_sha256 is not None
        and receipt.payload.get("retired_source_incident_sha256")
        != expected_retired_source_incident_sha256
    ):
        raise Tau3BenchmarkProtocolLineageError(
            "blind custody receipt retired source incident mismatch"
        )
    custody = _dict(receipt.payload.get("custody"))
    if custody != {
        "hashes_only": True,
        "local_paths_included": False,
        "raw_payload_included": False,
        "searchable_plaintext_persisted": False,
        "retired_source_used": False,
        "payload_lifetime": "blocking_custodian_memory_only",
        "one_shot_prepared": True,
        "consumed": False,
    }:
        raise Tau3BenchmarkProtocolLineageError("blind custody receipt custody gates changed")

    return {
        "schema_version": "hfr.tau3_blind_custody_validation.v1",
        "passed": True,
        "custody_receipt_sha256": receipt.sha256,
        "sealed_source_manifest_sha256": sealed.sha256,
        "generator_validation_sha256": generator.sha256,
        "fresh_contamination_replay_sha256": contamination.sha256,
        "task_count": REQUIRED_TASK_COUNT,
        "domain_counts": sealed_summary["domain_counts"],
    }


def create_tau3_benchmark_protocol_lineage(
    *,
    training_protocol: str | Path,
    benchmark_protocol: str | Path,
    custody_receipt: str | Path,
    sealed_source_manifest: str | Path,
    generator_validation: str | Path,
    fresh_contamination_replay: str | Path,
    retired_source_incident_sha256: str,
    out: str | Path,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Create a hash-only proof for one exact sealed-only protocol rotation."""

    target = Path(out)
    if target.exists():
        raise Tau3BenchmarkProtocolLineageError(f"benchmark protocol lineage already exists: {target}")
    evidence = _replay_benchmark_protocol_lineage(
        training_protocol=training_protocol,
        benchmark_protocol=benchmark_protocol,
        custody_receipt=custody_receipt,
        sealed_source_manifest=sealed_source_manifest,
        generator_validation=generator_validation,
        fresh_contamination_replay=fresh_contamination_replay,
        retired_source_incident_sha256=retired_source_incident_sha256,
    )
    lineage = {
        "schema_version": TAU3_BENCHMARK_PROTOCOL_LINEAGE_SCHEMA_VERSION,
        "created_at": created_at or _now_utc(),
        "passed": True,
        **evidence,
        "gates": {
            "training_protocol_schema_passed": True,
            "benchmark_protocol_schema_passed": True,
            "exact_sealed_only_delta": True,
            "fresh_source_bound_everywhere": True,
            "custody_receipt_replayed": True,
            "fresh_contamination_replay_passed": True,
            "retired_source_not_reused": True,
        },
        "hashes_only": True,
        "local_paths_included": False,
        "raw_payload_included": False,
    }
    lineage["lineage_sha256"] = canonical_sha256(lineage)
    _check_schema(lineage, "tau3_benchmark_protocol_lineage", "benchmark protocol lineage")
    _assert_hash_only_public_artifact(lineage, "benchmark protocol lineage")
    atomic_write_json_cas(target, lineage, expected_sha256=None, new_file_mode=0o444)
    return lineage


def validate_tau3_benchmark_protocol_lineage(
    *,
    lineage: str | Path,
    training_protocol: str | Path,
    benchmark_protocol: str | Path,
    custody_receipt: str | Path,
    sealed_source_manifest: str | Path,
    generator_validation: str | Path,
    fresh_contamination_replay: str | Path,
    retired_source_incident_sha256: str,
) -> dict[str, Any]:
    """Replay a stored benchmark-protocol lineage against every bound artifact."""

    lineage_record = _validated_lineage_record(Path(lineage))
    evidence = _replay_benchmark_protocol_lineage(
        training_protocol=training_protocol,
        benchmark_protocol=benchmark_protocol,
        custody_receipt=custody_receipt,
        sealed_source_manifest=sealed_source_manifest,
        generator_validation=generator_validation,
        fresh_contamination_replay=fresh_contamination_replay,
        retired_source_incident_sha256=retired_source_incident_sha256,
    )
    for key, value in evidence.items():
        if lineage_record.payload.get(key) != value:
            raise Tau3BenchmarkProtocolLineageError(
                f"benchmark protocol lineage {key} does not replay"
            )
    return {
        "schema_version": "hfr.tau3_benchmark_protocol_lineage_validation.v1",
        "passed": True,
        "lineage_sha256": lineage_record.sha256,
        "training_protocol_sha256": evidence["training_protocol_sha256"],
        "benchmark_protocol_sha256": evidence["benchmark_protocol_sha256"],
        "blind_custody_receipt_sha256": evidence["fresh_bindings"][
            "blind_custody_receipt_sha256"
        ],
        "sealed_source_manifest_sha256": evidence["fresh_bindings"][
            "sealed_source_manifest_sha256"
        ],
    }


def inspect_tau3_benchmark_protocol_lineage(
    *,
    lineage: str | Path,
) -> dict[str, Any]:
    """Validate the immutable public lineage record without authorizing access."""

    lineage_record = _validated_lineage_record(Path(lineage))
    return {
        "schema_version": "hfr.tau3_benchmark_protocol_lineage_inspection.v1",
        "passed": True,
        "sha256": lineage_record.sha256,
        "training_protocol_sha256": lineage_record.payload["training_protocol_sha256"],
        "benchmark_protocol_sha256": lineage_record.payload["benchmark_protocol_sha256"],
    }


def _validated_lineage_record(path: Path) -> _JsonArtifact:
    lineage_record = _read_json_artifact(path, "benchmark protocol lineage")
    _check_schema(
        lineage_record.payload,
        "tau3_benchmark_protocol_lineage",
        "benchmark protocol lineage",
    )
    _assert_hash_only_public_artifact(lineage_record.payload, "benchmark protocol lineage")
    _validate_receipt_self_seal(
        lineage_record.payload,
        field="lineage_sha256",
        label="benchmark protocol lineage",
    )
    if any(value is not True for value in _dict(lineage_record.payload.get("gates")).values()):
        raise Tau3BenchmarkProtocolLineageError("benchmark protocol lineage gate is not true")
    return lineage_record


def _replay_benchmark_protocol_lineage(
    *,
    training_protocol: str | Path,
    benchmark_protocol: str | Path,
    custody_receipt: str | Path,
    sealed_source_manifest: str | Path,
    generator_validation: str | Path,
    fresh_contamination_replay: str | Path,
    retired_source_incident_sha256: str,
) -> dict[str, Any]:
    _require_sha256(retired_source_incident_sha256, "retired source incident")
    training = _read_json_artifact(Path(training_protocol), "training protocol")
    benchmark = _read_json_artifact(Path(benchmark_protocol), "benchmark protocol")
    sealed = _read_json_artifact(Path(sealed_source_manifest), "sealed source manifest")
    contamination = _read_json_artifact(
        Path(fresh_contamination_replay),
        "fresh contamination replay",
    )
    custody = _read_json_artifact(Path(custody_receipt), "blind custody receipt")

    _check_protocol(training, "training protocol")
    _check_protocol(benchmark, "benchmark protocol")
    if training.sha256 == benchmark.sha256:
        raise Tau3BenchmarkProtocolLineageError(
            "training and benchmark protocols must have different file hashes"
        )
    custody_validation = validate_tau3_blind_custody_receipt(
        custody_receipt=custody.path,
        sealed_source_manifest=sealed.path,
        generator_validation=generator_validation,
        fresh_contamination_replay=contamination.path,
        expected_retired_source_incident_sha256=retired_source_incident_sha256,
    )
    if custody_validation.get("passed") is not True:
        raise Tau3BenchmarkProtocolLineageError("blind custody receipt replay failed")

    normalized_training = _without_allowed_delta(training.payload)
    normalized_benchmark = _without_allowed_delta(benchmark.payload)
    if normalized_training != normalized_benchmark:
        raise Tau3BenchmarkProtocolLineageError(
            "benchmark protocol changes fields outside the sealed-only allowlist"
        )
    changes: list[dict[str, Any]] = []
    for path in ALLOWED_PROTOCOL_DELTA_PATHS:
        before = _nested_path(training.payload, path)
        after = _nested_path(benchmark.payload, path)
        if before == after:
            raise Tau3BenchmarkProtocolLineageError(
                f"benchmark protocol required sealed-source binding did not rotate: {path}"
            )
        changes.append(
            {
                "path": path,
                "before_sha256": canonical_sha256(before),
                "after_sha256": canonical_sha256(after),
            }
        )
    _validate_benchmark_fresh_bindings(
        training=training.payload,
        benchmark=benchmark.payload,
        sealed=sealed,
        contamination=contamination,
    )

    return {
        "training_protocol_sha256": training.sha256,
        "benchmark_protocol_sha256": benchmark.sha256,
        "frozen_fields_sha256": canonical_sha256(normalized_training),
        "allowed_delta": {
            "paths": list(ALLOWED_PROTOCOL_DELTA_PATHS),
            "change_count": len(changes),
            "changes_sha256": canonical_sha256(changes),
        },
        "fresh_bindings": {
            "blind_custody_receipt_sha256": custody.sha256,
            "sealed_source_manifest_sha256": sealed.sha256,
            "fresh_contamination_replay_sha256": contamination.sha256,
            "retired_source_incident_sha256": retired_source_incident_sha256,
        },
    }


def _validate_sealed_source_manifest(sealed: _JsonArtifact) -> dict[str, Any]:
    _check_schema(sealed.payload, "tau3_sealed_source_manifest", "sealed source manifest")
    if sealed.payload.get("schema_version") != TAU3_SEALED_SOURCE_MANIFEST_SCHEMA_VERSION:
        raise Tau3BenchmarkProtocolLineageError("sealed source manifest schema_version mismatch")
    if sealed.payload.get("hashes_only") is not True:
        raise Tau3BenchmarkProtocolLineageError("sealed source manifest is not hashes-only")
    if sealed.payload.get("task_count") != REQUIRED_TASK_COUNT:
        raise Tau3BenchmarkProtocolLineageError("sealed source manifest must contain exactly 100 tasks")
    revision = sealed.payload.get("source_revision")
    if not isinstance(revision, str) or not HEX40_RE.fullmatch(revision):
        raise Tau3BenchmarkProtocolLineageError("sealed source manifest source revision is invalid")
    entries = sealed.payload.get("entries")
    if not isinstance(entries, list) or len(entries) != REQUIRED_TASK_COUNT:
        raise Tau3BenchmarkProtocolLineageError("sealed source manifest entries must contain exactly 100 rows")
    for key in ("task_id_sha256", "prompt_sha256", "task_sha256"):
        values = [item.get(key) for item in entries if isinstance(item, dict)]
        if len(values) != REQUIRED_TASK_COUNT or len(set(values)) != REQUIRED_TASK_COUNT:
            raise Tau3BenchmarkProtocolLineageError(
                f"sealed source manifest {key} values must be 100 unique hashes"
            )
        for value in values:
            _require_sha256(value, f"sealed source manifest {key}")
    domain_counts = _dict(sealed.payload.get("domain_counts"))
    if not domain_counts:
        raise Tau3BenchmarkProtocolLineageError(
            "sealed source manifest must declare public-safe domain_counts"
        )
    _validate_domain_counts(domain_counts)
    derived_domain_counts = {
        domain: sum(
            1
            for entry in entries
            if isinstance(entry, dict) and entry.get("domain") == domain
        )
        for domain in REQUIRED_DOMAINS
    }
    if derived_domain_counts != domain_counts:
        raise Tau3BenchmarkProtocolLineageError(
            "sealed source manifest domain_counts do not replay from its hash-only entries"
        )
    return {
        "source_revision": revision,
        "entries_sha256": canonical_sha256(entries),
        "domain_counts": domain_counts,
    }


def _validate_generator_validation(
    generator: _JsonArtifact,
    *,
    sealed: _JsonArtifact,
) -> dict[str, Any]:
    _check_schema(
        generator.payload,
        "tau3_blind_generator_validation",
        "blind generator validation",
    )
    _assert_hash_only_public_artifact(generator.payload, "blind generator validation")
    if generator.payload.get("schema_version") != TAU3_BLIND_GENERATOR_VALIDATION_SCHEMA_VERSION:
        raise Tau3BenchmarkProtocolLineageError("blind generator validation schema_version mismatch")
    if generator.payload.get("sealed_source_manifest_sha256") != sealed.sha256:
        raise Tau3BenchmarkProtocolLineageError(
            "blind generator validation sealed source hash mismatch"
        )
    if generator.payload.get("source_revision") != sealed.payload.get("source_revision"):
        raise Tau3BenchmarkProtocolLineageError(
            "blind generator validation source revision mismatch"
        )
    domain_counts = _dict(generator.payload.get("domain_counts"))
    _validate_domain_counts(domain_counts)
    if _dict(generator.payload.get("golden_replay")) != {
        "passed": True,
        "replayed_task_count": REQUIRED_TASK_COUNT,
        "passed_task_count": REQUIRED_TASK_COUNT,
        "failed_task_count": 0,
        "state_check_failure_count": 0,
    }:
        raise Tau3BenchmarkProtocolLineageError("blind generator golden replay did not pass exactly")
    return {"domain_counts": domain_counts}


def _validate_fresh_contamination(
    contamination: _JsonArtifact,
    *,
    sealed: _JsonArtifact,
) -> None:
    _check_schema(
        contamination.payload,
        "tau3_fresh_contamination_replay",
        "fresh contamination replay",
    )
    _assert_hash_only_public_artifact(contamination.payload, "fresh contamination replay")
    if contamination.payload.get("schema_version") != TAU3_FRESH_CONTAMINATION_REPLAY_SCHEMA_VERSION:
        raise Tau3BenchmarkProtocolLineageError("fresh contamination replay schema_version mismatch")
    if contamination.payload.get("fresh_sealed_source_manifest_sha256") != sealed.sha256:
        raise Tau3BenchmarkProtocolLineageError(
            "fresh contamination replay sealed source hash mismatch"
        )
    overlaps = _dict(contamination.payload.get("overlaps"))
    if len(overlaps) != 12 or any(value != 0 for value in overlaps.values()):
        raise Tau3BenchmarkProtocolLineageError(
            "fresh contamination replay contains a train, development, or retired-source overlap"
        )


def _validate_benchmark_fresh_bindings(
    *,
    training: dict[str, Any],
    benchmark: dict[str, Any],
    sealed: _JsonArtifact,
    contamination: _JsonArtifact,
) -> None:
    fresh_sha = sealed.sha256
    retired_sha = _nested_path(training, "tau_revision.split_hashes.sealed")
    if not isinstance(retired_sha, str) or not HEX64_RE.fullmatch(retired_sha):
        raise Tau3BenchmarkProtocolLineageError("training protocol retired sealed hash is invalid")
    if fresh_sha == retired_sha:
        raise Tau3BenchmarkProtocolLineageError("benchmark protocol reused the retired sealed source")
    if _nested_path(benchmark, "tau_revision.split_hashes.sealed") != fresh_sha:
        raise Tau3BenchmarkProtocolLineageError(
            "benchmark protocol Tau revision does not bind the fresh sealed source"
        )
    benchmark_split = _dict(_nested_path(benchmark, "split_manifest.splits.sealed"))
    if benchmark_split.get("sealed") is not True or benchmark_split.get("sha256") != fresh_sha:
        raise Tau3BenchmarkProtocolLineageError(
            "benchmark protocol sealed split does not bind the fresh sealed source"
        )
    source_manifest = _dict(_nested_path(benchmark, "split_manifest.source_manifest"))
    _require_sha256(source_manifest.get("sha256"), "benchmark split source manifest")
    sealed_manifest = _dict(benchmark.get("sealed_manifest"))
    if sealed_manifest.get("manifest_sha256") != fresh_sha:
        raise Tau3BenchmarkProtocolLineageError(
            "benchmark protocol sealed manifest does not bind the fresh sealed source"
        )
    if sealed_manifest.get("access_count") != 0:
        raise Tau3BenchmarkProtocolLineageError(
            "benchmark protocol must declare zero sealed access before authorization"
        )
    entries = sealed.payload["entries"]
    expected_prompts = {str(item["prompt_sha256"]) for item in entries}
    expected_blocking = {
        str(item[key])
        for item in entries
        for key in ("task_id_sha256", "prompt_sha256", "task_sha256")
    }
    if set(sealed_manifest.get("prompt_template_hashes") or []) != expected_prompts:
        raise Tau3BenchmarkProtocolLineageError(
            "benchmark protocol prompt-template hashes do not match the fresh sealed source"
        )
    if not expected_blocking.issubset(set(sealed_manifest.get("leakage_blocking_hashes") or [])):
        raise Tau3BenchmarkProtocolLineageError(
            "benchmark protocol leakage-blocking hashes omit fresh sealed identities"
        )
    if (
        contamination.payload.get("retired_sealed_source_manifest_sha256") != retired_sha
        or contamination.payload.get("fresh_sealed_source_manifest_sha256") != fresh_sha
    ):
        raise Tau3BenchmarkProtocolLineageError(
            "fresh contamination replay does not bridge retired and fresh sealed sources"
        )


def _check_protocol(protocol: _JsonArtifact, label: str) -> None:
    _check_schema(protocol.payload, "tau3_protocol_config", label)
    if protocol.payload.get("schema_version") != TAU3_PROTOCOL_CONFIG_SCHEMA_VERSION:
        raise Tau3BenchmarkProtocolLineageError(f"{label} schema_version mismatch")


def _without_allowed_delta(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(payload)
    for path in ALLOWED_PROTOCOL_DELTA_PATHS:
        _delete_nested_path(normalized, path)
    return normalized


def _delete_nested_path(payload: dict[str, Any], path: str) -> None:
    parts = path.split(".")
    current: dict[str, Any] = payload
    for part in parts[:-1]:
        value = current.get(part)
        if not isinstance(value, dict):
            raise Tau3BenchmarkProtocolLineageError(f"protocol is missing required path: {path}")
        current = value
    if parts[-1] not in current:
        raise Tau3BenchmarkProtocolLineageError(f"protocol is missing required path: {path}")
    del current[parts[-1]]


def _nested_path(payload: dict[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise Tau3BenchmarkProtocolLineageError(f"artifact is missing required path: {path}")
        value = value[part]
    return value


def _validate_domain_counts(value: dict[str, Any]) -> None:
    if set(value) != set(REQUIRED_DOMAINS):
        raise Tau3BenchmarkProtocolLineageError("domain_counts must contain airline, retail, and telecom")
    counts = [value[domain] for domain in REQUIRED_DOMAINS]
    if any(not isinstance(count, int) or isinstance(count, bool) or count <= 0 for count in counts):
        raise Tau3BenchmarkProtocolLineageError("domain_counts values must be positive integers")
    if sum(counts) != REQUIRED_TASK_COUNT:
        raise Tau3BenchmarkProtocolLineageError("domain_counts must sum to exactly 100")
    if max(counts) - min(counts) > 1:
        raise Tau3BenchmarkProtocolLineageError(
            "domain_counts must be balanced to within one task"
        )


def _validate_receipt_self_seal(
    payload: dict[str, Any],
    *,
    field: str,
    label: str,
) -> None:
    expected = payload.get(field)
    unsealed = {key: value for key, value in payload.items() if key != field}
    if expected != canonical_sha256(unsealed):
        raise Tau3BenchmarkProtocolLineageError(f"{label} self-seal mismatch")


def _check_schema(payload: dict[str, Any], schema_name: str, label: str) -> None:
    result = check_schema_contract(payload, name_or_id=schema_name)
    if result.get("passed") is not True:
        raise Tau3BenchmarkProtocolLineageError(
            f"{label} violates registered schema: "
            + "; ".join(str(error) for error in result.get("errors", []))
        )


def _read_json_artifact(path: Path, label: str) -> _JsonArtifact:
    if path_has_symlink_component(path, include_leaf=True):
        raise Tau3BenchmarkProtocolLineageError(f"{label} path contains a symlink: {path}")
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise Tau3BenchmarkProtocolLineageError(
                    f"{label} must be a singly linked regular file: {path}"
                )
            raw = handle.read()
            after = os.fstat(handle.fileno())
        path_state = os.lstat(path)
    except OSError as exc:
        raise Tau3BenchmarkProtocolLineageError(f"{label} is not a regular file: {path}") from exc
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if (
        any(getattr(before, field) != getattr(after, field) for field in stable_fields)
        or len(raw) != after.st_size
        or not stat.S_ISREG(path_state.st_mode)
        or path_state.st_nlink != 1
        or (path_state.st_dev, path_state.st_ino) != (after.st_dev, after.st_ino)
        or path_has_symlink_component(path, include_leaf=True)
    ):
        raise Tau3BenchmarkProtocolLineageError(f"{label} changed while it was read: {path}")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Tau3BenchmarkProtocolLineageError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise Tau3BenchmarkProtocolLineageError(f"{label} must contain a JSON object: {path}")
    return _JsonArtifact(
        path=path,
        payload=payload,
        sha256=hashlib.sha256(raw).hexdigest(),
        size=len(raw),
    )


def _assert_hash_only_public_artifact(payload: dict[str, Any], label: str) -> None:
    forbidden_keys = {
        "path",
        "local_path",
        "source_path",
        "tasks",
        "messages",
        "prompt",
        "raw_payload",
        "raw_data",
        "policy",
        "tool_defs",
    }
    strings: list[tuple[str | None, str]] = []

    def walk(value: Any, key: str | None = None) -> None:
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                if child_key in forbidden_keys:
                    raise Tau3BenchmarkProtocolLineageError(
                        f"{label} contains forbidden public key: {child_key}"
                    )
                walk(child_value, child_key)
        elif isinstance(value, list):
            for child_value in value:
                walk(child_value, key)
        elif isinstance(value, str):
            strings.append((key, value))

    walk(payload)
    for key, value in strings:
        if key == "created_at":
            continue
        if value.startswith(("/", "~", "file:")) or re.match(r"^[A-Za-z]:[\\/]", value):
            raise Tau3BenchmarkProtocolLineageError(
                f"{label} contains a local path-like value"
            )


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not HEX64_RE.fullmatch(value):
        raise Tau3BenchmarkProtocolLineageError(f"{label} must be a lowercase SHA-256")
    return value


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
