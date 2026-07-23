"""Fail-closed public-safe Tau-3 sealed-grid completeness artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .atomic_json import atomic_write_json_cas
from .path_safety import path_has_symlink_component
from .schema_registry import check_schema_contract
from .tau3_sealed_authorization import Tau3SealedAuthorizationError, validate_tau3_sealed_authorization

TAU3_SEALED_GRID_COMPLETENESS_SCHEMA_VERSION = "hfr.tau3_sealed_grid_completeness.v1"
TAU3_BENCHMARK_RUN_SCHEMA_VERSION = "hfr.tau3_benchmark_run.v1"
TAU3_SEALED_SOURCE_SCHEMA_VERSION = "hfr.tau3_sealed_source_manifest.v1"
TAU3_SEALED_AUTHORIZATION_SCHEMA_VERSION = "hfr.tau3_sealed_authorization.v1"

REQUIRED_ARMS = ("adapter", "base", "comparator_1", "comparator_2")
REQUIRED_SEEDS = (101, 202, 303, 404)
REQUIRED_DOMAINS = ("airline", "retail", "telecom")
REQUIRED_SEALED_TASK_COUNT = 100
REQUIRED_EPISODES_PER_ARM = REQUIRED_SEALED_TASK_COUNT * len(REQUIRED_SEEDS)
REQUIRED_TOTAL_EPISODES = REQUIRED_EPISODES_PER_ARM * len(REQUIRED_ARMS)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

FORBIDDEN_PUBLIC_KEYS = {
    "path",
    "paths",
    "local_path",
    "result_path",
    "source_path",
    "output_dir",
    "tasks",
    "messages",
    "prompt",
    "raw_data",
    "raw_payload",
    "scores",
    "score",
    "reward",
}
PRIVATE_PATH_RE = re.compile(r"(^/|[A-Za-z]:[\\/]|\\\\|/Users/|/home/|/tmp/|localhost|127\.0\.0\.1)")


class Tau3SealedGridCompletenessError(ValueError):
    """Raised when sealed benchmark completeness cannot be proven."""


@dataclass(frozen=True)
class _JsonArtifact:
    path: Path
    payload: dict[str, Any]
    sha256: str
    size: int


@dataclass(frozen=True)
class _ArmEvidence:
    arm_id: str
    manifest: _JsonArtifact
    task_hashes_by_seed: dict[int, set[str]]
    episode_count_by_seed: dict[int, int]
    protocol_sha256: str
    candidate_lock_sha256: str
    sealed_source_sha256: str
    authorization_sha256: str
    harness_hash: str
    model_identity_hash: str


def build_tau3_sealed_grid_completeness(
    *,
    arm_manifests: list[str | Path],
    candidate_lock: str | Path,
    protocol: str | Path,
    sealed_source_manifest: str | Path,
    sealed_authorization: str | Path,
    expected_tau_revision: str,
    out: str | Path,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Write a public-safe completeness artifact for the final sealed 4x4 Tau-3 grid."""

    target = Path(out)
    if target.exists():
        raise Tau3SealedGridCompletenessError(f"sealed-grid completeness output already exists: {target}")
    sealed = _read_json_artifact(Path(sealed_source_manifest), "sealed source manifest")
    authorization = _read_json_artifact(Path(sealed_authorization), "sealed authorization")
    _validate_sealed_source(sealed)
    _validate_authorization(authorization, sealed)
    authorization_replay = _replay_authorization(
        authorization_path=Path(sealed_authorization),
        candidate_lock_path=Path(candidate_lock),
        protocol_path=Path(protocol),
        sealed_source_manifest_path=Path(sealed_source_manifest),
        authorization=authorization,
        expected_tau_revision=expected_tau_revision,
    )
    candidate_lock_artifact = _read_json_artifact(Path(candidate_lock), "candidate lock")

    if len(arm_manifests) != len(REQUIRED_ARMS):
        raise Tau3SealedGridCompletenessError("exactly four final sealed arm manifests are required")
    arms = [
        _load_arm(
            Path(path),
            sealed=sealed,
            authorization=authorization,
            authorization_replay=authorization_replay,
            candidate_lock=candidate_lock_artifact,
        )
        for path in arm_manifests
    ]
    by_arm = {arm.arm_id: arm for arm in arms}
    if tuple(sorted(by_arm)) != REQUIRED_ARMS or len(by_arm) != len(REQUIRED_ARMS):
        raise Tau3SealedGridCompletenessError("sealed arms must be exactly adapter/base/comparator_1/comparator_2")

    expected_task_hashes = _sealed_task_hashes(sealed.payload)
    expected_seed_hashes = {seed: expected_task_hashes for seed in REQUIRED_SEEDS}
    for arm in by_arm.values():
        if arm.task_hashes_by_seed != expected_seed_hashes:
            raise Tau3SealedGridCompletenessError(f"{arm.arm_id} sealed task coverage does not exactly match sealed source manifest")
        if any(count != REQUIRED_SEALED_TASK_COUNT for count in arm.episode_count_by_seed.values()):
            raise Tau3SealedGridCompletenessError(f"{arm.arm_id} sealed episode counts do not replay 100 tasks per seed")
    coverage_fingerprint = _canonical_sha256(
        {
            arm_id: {str(seed): sorted(by_arm[arm_id].task_hashes_by_seed[seed]) for seed in REQUIRED_SEEDS}
            for arm_id in REQUIRED_ARMS
        }
    )

    protocol_hashes = {arm.protocol_sha256 for arm in by_arm.values()}
    lock_hashes = {arm.candidate_lock_sha256 for arm in by_arm.values()}
    source_hashes = {arm.sealed_source_sha256 for arm in by_arm.values()}
    auth_hashes = {arm.authorization_sha256 for arm in by_arm.values()}
    if protocol_hashes != {str(authorization.payload["protocol"]["sha256"])}:
        raise Tau3SealedGridCompletenessError("sealed arm protocol hashes do not exactly match authorization")
    if lock_hashes != {str(authorization.payload["candidate_lock"]["sha256"])}:
        raise Tau3SealedGridCompletenessError("sealed arm candidate-lock hashes do not exactly match authorization")
    if source_hashes != {sealed.sha256}:
        raise Tau3SealedGridCompletenessError("sealed arm sealed-source hashes do not exactly match source manifest")
    if auth_hashes != {authorization.sha256}:
        raise Tau3SealedGridCompletenessError("sealed arm authorization hashes do not exactly match authorization artifact")

    payload = {
        "schema_version": TAU3_SEALED_GRID_COMPLETENESS_SCHEMA_VERSION,
        "created_at": created_at or _now_utc(),
        "passed": True,
        "status": "complete",
        "hashes_only": True,
        "local_paths_included": False,
        "raw_payload_included": False,
        "scores_included": False,
        "bindings": {
            "authorization_sha256": authorization.sha256,
            "candidate_lock_sha256": str(authorization.payload["candidate_lock"]["sha256"]),
            "protocol_sha256": str(authorization.payload["protocol"]["sha256"]),
            "sealed_source_sha256": sealed.sha256,
            "coverage_fingerprint_sha256": coverage_fingerprint,
            "arm_manifest_sha256": {arm_id: by_arm[arm_id].manifest.sha256 for arm_id in REQUIRED_ARMS},
            "model_identity_sha256": {arm_id: by_arm[arm_id].model_identity_hash for arm_id in REQUIRED_ARMS},
            "harness_equivalence_sha256": _canonical_sha256({arm_id: by_arm[arm_id].harness_hash for arm_id in REQUIRED_ARMS}),
        },
        "counts": {
            "arm_count": len(REQUIRED_ARMS),
            "seed_count": len(REQUIRED_SEEDS),
            "domain_count": len(REQUIRED_DOMAINS),
            "sealed_task_count": REQUIRED_SEALED_TASK_COUNT,
            "episodes_per_arm": REQUIRED_EPISODES_PER_ARM,
            "total_episodes": REQUIRED_TOTAL_EPISODES,
            "per_arm_seed_task_count": {
                arm_id: {str(seed): len(by_arm[arm_id].task_hashes_by_seed[seed]) for seed in REQUIRED_SEEDS}
                for arm_id in REQUIRED_ARMS
            },
        },
        "gates": {
            "arms_exact": True,
            "seeds_exact": True,
            "domains_exact": True,
            "sealed_task_count_exact": True,
            "no_duplicate_task_rows": True,
            "no_missing_task_rows": True,
            "no_extra_task_rows": True,
            "same_task_coverage_across_arms": True,
            "same_task_coverage_across_seeds": True,
            "result_hashes_replayed": True,
            "authorization_binding_replayed": True,
            "candidate_lock_binding_replayed": True,
            "protocol_binding_replayed": True,
            "sealed_source_binding_replayed": True,
            "harness_equivalence_bound": True,
            "model_identities_bound": True,
            "public_payload_safe": True,
        },
    }
    _assert_public_safe(payload)
    schema = check_schema_contract(payload, name_or_id="tau3_sealed_grid_completeness")
    if schema.get("passed") is not True:
        raise Tau3SealedGridCompletenessError("sealed-grid completeness violates schema: " + "; ".join(schema["errors"]))
    atomic_write_json_cas(target, payload, expected_sha256=None, new_file_mode=0o444)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-manifest", type=Path, action="append", required=True, help="Final sealed Tau-3 arm manifest; pass exactly four times")
    parser.add_argument("--candidate-lock", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--sealed-source-manifest", type=Path, required=True)
    parser.add_argument("--sealed-authorization", type=Path, required=True)
    parser.add_argument("--expected-tau-revision", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--created-at")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        artifact = build_tau3_sealed_grid_completeness(
            arm_manifests=args.arm_manifest,
            candidate_lock=args.candidate_lock,
            protocol=args.protocol,
            sealed_source_manifest=args.sealed_source_manifest,
            sealed_authorization=args.sealed_authorization,
            expected_tau_revision=args.expected_tau_revision,
            out=args.out,
            created_at=args.created_at,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps(_stable_cli_error(exc), sort_keys=True), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "passed": artifact["passed"],
                "status": artifact["status"],
                "sealed_source_sha256": artifact["bindings"]["sealed_source_sha256"],
                "coverage_fingerprint_sha256": artifact["bindings"]["coverage_fingerprint_sha256"],
                "total_episodes": artifact["counts"]["total_episodes"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _load_arm(
    path: Path,
    *,
    sealed: _JsonArtifact,
    authorization: _JsonArtifact,
    authorization_replay: dict[str, Any],
    candidate_lock: _JsonArtifact,
) -> _ArmEvidence:
    manifest = _read_json_artifact(path, "sealed arm manifest")
    payload = manifest.payload
    check = check_schema_contract(payload, name_or_id="tau3_benchmark_run")
    if check.get("passed") is not True:
        raise Tau3SealedGridCompletenessError("sealed arm manifest violates schema: " + "; ".join(check["errors"]))
    if payload.get("schema_version") != TAU3_BENCHMARK_RUN_SCHEMA_VERSION or payload.get("phase") != "final":
        raise Tau3SealedGridCompletenessError("arm manifest must be a final Tau-3 benchmark run")
    arm_id = str(payload.get("arm_id") or "")
    if arm_id not in REQUIRED_ARMS:
        raise Tau3SealedGridCompletenessError("arm manifest has unauthorized arm_id")
    if payload.get("mode") != "sealed":
        raise Tau3SealedGridCompletenessError(f"{arm_id} arm manifest is not sealed mode")
    if payload.get("run_count") != 12 or payload.get("success_count") != 12 or payload.get("failure_count") != 0:
        raise Tau3SealedGridCompletenessError(f"{arm_id} arm must contain exactly 12 successful domain/seed runs")
    if payload.get("source") is not None or payload.get("sealed_payload_accessed") is not False or payload.get("sealed_task_ids_materialized") is not False:
        raise Tau3SealedGridCompletenessError(f"{arm_id} arm leaked or materialized sealed source state")

    protocol_sha = _require_ref_sha(payload.get("protocol"), "protocol")
    lock_sha = _require_ref_sha(payload.get("candidate_lock"), "candidate lock")
    sealed_source_sha = _sealed_source_ref_sha(payload)
    auth_sha = _sealed_authorization_ref_sha(payload)
    if protocol_sha != payload.get("protocol_sha256"):
        raise Tau3SealedGridCompletenessError(f"{arm_id} protocol reference does not match arm protocol_sha256")
    if sealed_source_sha != sealed.sha256:
        raise Tau3SealedGridCompletenessError(f"{arm_id} sealed source hash does not match sealed source manifest")
    if auth_sha != authorization.sha256:
        raise Tau3SealedGridCompletenessError(f"{arm_id} authorization hash does not match authorization artifact")
    _validate_arm_authorization_ref(payload, authorization_replay)
    _validate_arm_harness(payload)
    _validate_arm_model_identity(payload, authorization.payload)

    base = manifest.path.parent
    _read_json_ref(base, payload.get("protocol"), protocol_sha, "protocol")
    staged_lock = _read_json_ref(base, payload.get("candidate_lock"), lock_sha, "candidate lock")
    _read_json_ref(base, payload.get("sealed_authorization"), auth_sha, "sealed authorization")
    _read_json_ref(base, payload.get("sealed_task_count_manifest"), sealed_source_sha, "sealed source manifest")
    if staged_lock.payload != candidate_lock.payload:
        raise Tau3SealedGridCompletenessError(f"{arm_id} staged candidate lock payload does not match replayed candidate lock")
    lock_time = _parse_time(candidate_lock.payload.get("created_at"))
    final_time = _parse_time(payload.get("created_at"))
    if lock_time is None or final_time is None or final_time <= lock_time:
        raise Tau3SealedGridCompletenessError(f"{arm_id} final manifest chronology does not replay")
    prelaunch = _read_json_ref(base, payload.get("prelaunch_receipt"), _dict(payload.get("prelaunch_receipt")).get("sha256"), "prelaunch receipt")
    prelaunch_time = _parse_time(prelaunch.payload.get("created_at"))
    if prelaunch_time is None or prelaunch_time <= lock_time:
        raise Tau3SealedGridCompletenessError(f"{arm_id} prelaunch chronology does not replay")

    receipts = _list_of_dicts(payload.get("run_receipts"))
    expected_grid = {(domain, seed) for domain in REQUIRED_DOMAINS for seed in REQUIRED_SEEDS}
    actual_grid = {(row.get("domain"), row.get("seed")) for row in receipts}
    if actual_grid != expected_grid or len(receipts) != len(expected_grid):
        raise Tau3SealedGridCompletenessError(f"{arm_id} run receipts do not contain the exact 3x4 domain/seed grid")

    task_hashes_by_seed: dict[int, set[str]] = {seed: set() for seed in REQUIRED_SEEDS}
    episode_count_by_seed: dict[int, int] = {seed: 0 for seed in REQUIRED_SEEDS}
    for record in receipts:
        domain = str(record["domain"])
        seed = int(record["seed"])
        receipt_artifact = _read_json_ref(base, {"path": record.get("path")}, record.get("receipt_sha256"), "run receipt")
        receipt = receipt_artifact.payload
        if receipt.get("terminal_status") != "completed" or receipt.get("mode") != "sealed":
            raise Tau3SealedGridCompletenessError(f"{arm_id} {domain}/{seed} run receipt did not replay sealed completion")
        if receipt.get("arm_id") != arm_id:
            raise Tau3SealedGridCompletenessError(f"{arm_id} {domain}/{seed} run receipt arm binding mismatch")
        if receipt.get("domain") != domain or receipt.get("seed") != seed:
            raise Tau3SealedGridCompletenessError(f"{arm_id} run receipt domain/seed binding mismatch")
        if receipt.get("protocol_sha256") != protocol_sha:
            raise Tau3SealedGridCompletenessError(f"{arm_id} run receipt protocol binding mismatch")
        receipt_time = _parse_time(receipt.get("created_at"))
        if receipt_time is None or receipt_time <= lock_time:
            raise Tau3SealedGridCompletenessError(f"{arm_id} {domain}/{seed} run receipt chronology does not replay")
        result_rel = receipt.get("result_path")
        result_artifact = _read_json_ref(receipt_artifact.path.parent, {"path": result_rel}, receipt.get("result_sha256"), "raw result")
        if result_artifact.sha256 != record.get("result_sha256"):
            raise Tau3SealedGridCompletenessError(f"{arm_id} {domain}/{seed} raw result hash mismatch")
        task_hashes = _task_hashes_from_result(result_artifact, expected_domain=domain, expected_seed=seed)
        if task_hashes & task_hashes_by_seed[seed]:
            raise Tau3SealedGridCompletenessError(f"{arm_id} seed {seed} contains duplicate sealed task rows")
        task_hashes_by_seed[seed].update(task_hashes)
        episode_count_by_seed[seed] += len(task_hashes)

    return _ArmEvidence(
        arm_id=arm_id,
        manifest=manifest,
        task_hashes_by_seed=task_hashes_by_seed,
        episode_count_by_seed=episode_count_by_seed,
        protocol_sha256=protocol_sha,
        candidate_lock_sha256=lock_sha,
        sealed_source_sha256=sealed_source_sha,
        authorization_sha256=auth_sha,
        harness_hash=_canonical_sha256({"config": payload.get("config"), "agent": payload.get("agent"), "user_simulator": payload.get("user_simulator"), "reviewer": payload.get("reviewer")}),
        model_identity_hash=_canonical_sha256(payload.get("arm_identity")),
    )


def _task_hashes_from_result(artifact: _JsonArtifact, *, expected_domain: str, expected_seed: int) -> set[str]:
    payload = artifact.payload
    simulations = payload.get("simulations")
    if not isinstance(simulations, list) or not simulations:
        raise Tau3SealedGridCompletenessError("raw result simulations must be a non-empty list")
    task_hashes: set[str] = set()
    for index, simulation in enumerate(simulations):
        if not isinstance(simulation, dict):
            raise Tau3SealedGridCompletenessError("raw result simulation row must be an object")
        if simulation.get("seed") not in (expected_seed, None):
            raise Tau3SealedGridCompletenessError("raw result simulation seed mismatch")
        task_id = simulation.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise Tau3SealedGridCompletenessError("raw result simulation missing task_id")
        digest = hashlib.sha256(f"{expected_domain}:{task_id}".encode("utf-8")).hexdigest()
        if digest in task_hashes:
            raise Tau3SealedGridCompletenessError(f"duplicate task row in raw result at index {index}")
        task_hashes.add(digest)
    return task_hashes


def _replay_authorization(
    *,
    authorization_path: Path,
    candidate_lock_path: Path,
    protocol_path: Path,
    sealed_source_manifest_path: Path,
    authorization: _JsonArtifact,
    expected_tau_revision: str,
) -> dict[str, Any]:
    last_record: dict[str, Any] | None = None
    for arm_id in REQUIRED_ARMS:
        try:
            record = validate_tau3_sealed_authorization(
                authorization_path=authorization_path,
                candidate_lock_path=candidate_lock_path,
                protocol_path=protocol_path,
                sealed_source_manifest_path=sealed_source_manifest_path,
                arm_id=arm_id,
                seeds=REQUIRED_SEEDS,
                expected_tau_revision=expected_tau_revision,
                expected_authorization_sha256=authorization.sha256,
            )
        except Tau3SealedAuthorizationError as exc:
            raise Tau3SealedGridCompletenessError("sealed authorization replay failed") from exc
        _compare_authorization_replay(authorization.payload, record)
        last_record = record
    if last_record is None:
        raise Tau3SealedGridCompletenessError("sealed authorization replay failed")
    return last_record


def _compare_authorization_replay(auth: dict[str, Any], record: dict[str, Any]) -> None:
    expected = {
        "authorized": True,
        "candidate_lock_sha256": _dict(auth.get("candidate_lock")).get("sha256"),
        "protocol_sha256": _dict(auth.get("protocol")).get("sha256"),
        "sealed_source_sha256": _dict(auth.get("sealed_source")).get("manifest_sha256"),
        "task_count": REQUIRED_SEALED_TASK_COUNT,
        "arms": list(REQUIRED_ARMS),
        "seeds": list(REQUIRED_SEEDS),
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise Tau3SealedGridCompletenessError("sealed authorization replay binding mismatch")


def _validate_arm_authorization_ref(payload: dict[str, Any], replay: dict[str, Any]) -> None:
    ref = _dict(payload.get("sealed_authorization"))
    expected = {
        "sha256": replay.get("sha256"),
        "authorized": True,
        "candidate_lock_sha256": replay.get("candidate_lock_sha256"),
        "protocol_sha256": replay.get("protocol_sha256"),
        "sealed_source_sha256": replay.get("sealed_source_sha256"),
        "task_count": REQUIRED_SEALED_TASK_COUNT,
        "arms": list(REQUIRED_ARMS),
        "seeds": list(REQUIRED_SEEDS),
    }
    for key, value in expected.items():
        if ref.get(key) != value:
            raise Tau3SealedGridCompletenessError("arm sealed authorization reference binding mismatch")


def _validate_arm_harness(payload: dict[str, Any]) -> None:
    cfg = _dict(payload.get("config"))
    expected = {
        "domains": list(REQUIRED_DOMAINS),
        "seeds": list(REQUIRED_SEEDS),
        "context_window": 16384,
        "max_steps": 30,
        "test_time_search": False,
        "max_retries": 0,
        "hallucination_retries": 0,
        "auto_review": True,
        "review_mode": "full",
        "agent": "llm_agent",
        "user": "user_simulator",
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise Tau3SealedGridCompletenessError("arm harness binding mismatch")
    for section in ("agent", "user_simulator", "reviewer"):
        endpoint = _dict(payload.get(section))
        if (
            endpoint.get("context_window") != 16384
            or endpoint.get("max_tokens") != 1024
            or endpoint.get("temperature") != 0.0
            or endpoint.get("top_p") != 1.0
        ):
            raise Tau3SealedGridCompletenessError("arm harness endpoint binding mismatch")


def _validate_arm_model_identity(payload: dict[str, Any], authorization: dict[str, Any]) -> None:
    arm_id = str(payload.get("arm_id") or "")
    identity = _dict(payload.get("arm_identity"))
    refs = _dict(authorization.get("model_identity_refs"))
    if arm_id == "adapter":
        adapter = _dict(identity.get("adapter"))
        expected = {
            "candidate_lock_sha256": _dict(authorization.get("candidate_lock")).get("sha256"),
            "candidate_identity_sha256": refs.get("candidate_identity_sha256"),
            "endpoint_model_sha256": refs.get("endpoint_model_sha256"),
        }
        for key, value in expected.items():
            if identity.get(key) != value:
                raise Tau3SealedGridCompletenessError("adapter model identity binding mismatch")
        if adapter.get("tree_sha256") != refs.get("adapter_tree_sha256"):
            raise Tau3SealedGridCompletenessError("adapter model identity binding mismatch")
        return
    key_by_arm = {
        "base": "base_identity_sha256",
        "comparator_1": "comparator_1_identity_sha256",
        "comparator_2": "comparator_2_identity_sha256",
    }
    expected_key = key_by_arm.get(arm_id)
    if expected_key is None or identity.get("model_identity_sha256") != refs.get(expected_key):
        raise Tau3SealedGridCompletenessError("model identity binding mismatch")


def _validate_sealed_source(sealed: _JsonArtifact) -> None:
    check = check_schema_contract(sealed.payload, name_or_id="tau3_sealed_source_manifest")
    if check.get("passed") is not True:
        raise Tau3SealedGridCompletenessError("sealed source manifest violates schema: " + "; ".join(check["errors"]))
    if sealed.payload.get("schema_version") != TAU3_SEALED_SOURCE_SCHEMA_VERSION:
        raise Tau3SealedGridCompletenessError("sealed source manifest schema_version mismatch")
    if sealed.payload.get("hashes_only") is not True or sealed.payload.get("task_count") != REQUIRED_SEALED_TASK_COUNT:
        raise Tau3SealedGridCompletenessError("sealed source manifest must be hashes-only with exactly 100 tasks")
    hashes = _sealed_task_hashes(sealed.payload)
    if len(hashes) != REQUIRED_SEALED_TASK_COUNT:
        raise Tau3SealedGridCompletenessError("sealed source manifest task hashes must be unique and exactly 100")


def _validate_authorization(authorization: _JsonArtifact, sealed: _JsonArtifact) -> None:
    check = check_schema_contract(authorization.payload, name_or_id="tau3_sealed_authorization")
    if check.get("passed") is not True:
        raise Tau3SealedGridCompletenessError("sealed authorization violates schema: " + "; ".join(check["errors"]))
    payload = authorization.payload
    if payload.get("schema_version") != TAU3_SEALED_AUTHORIZATION_SCHEMA_VERSION or payload.get("authorized") is not True:
        raise Tau3SealedGridCompletenessError("sealed authorization is not authorized")
    frozen = _dict(payload.get("frozen_contract"))
    if frozen.get("arms") != list(REQUIRED_ARMS) or frozen.get("seeds") != list(REQUIRED_SEEDS) or frozen.get("domains") != list(REQUIRED_DOMAINS):
        raise Tau3SealedGridCompletenessError("sealed authorization frozen grid does not match required arms/seeds/domains")
    sealed_source = _dict(payload.get("sealed_source"))
    if sealed_source.get("manifest_sha256") != sealed.sha256 or sealed_source.get("task_count") != REQUIRED_SEALED_TASK_COUNT:
        raise Tau3SealedGridCompletenessError("sealed authorization sealed-source binding mismatch")


def _sealed_task_hashes(payload: dict[str, Any]) -> set[str]:
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return set()
    hashes = {str(_dict(entry).get("task_id_sha256") or "") for entry in entries}
    if not all(SHA256_RE.fullmatch(item) for item in hashes):
        raise Tau3SealedGridCompletenessError("sealed source task_id_sha256 entries must be SHA-256 digests")
    return hashes


def _sealed_source_ref_sha(payload: dict[str, Any]) -> str:
    ref = _dict(payload.get("sealed_task_count_manifest"))
    sha = ref.get("sha256") or ref.get("manifest_sha256")
    if not isinstance(sha, str) or not SHA256_RE.fullmatch(sha):
        raise Tau3SealedGridCompletenessError("sealed arm missing sealed source manifest hash")
    if ref.get("task_count") != REQUIRED_SEALED_TASK_COUNT:
        raise Tau3SealedGridCompletenessError("sealed arm sealed source task_count must be exactly 100")
    if ref.get("hashes_only") is not True:
        raise Tau3SealedGridCompletenessError("sealed arm sealed source binding must be hashes-only")
    return sha


def _sealed_authorization_ref_sha(payload: dict[str, Any]) -> str:
    ref = _dict(payload.get("sealed_authorization"))
    sha = ref.get("sha256") or ref.get("authorization_sha256")
    if not isinstance(sha, str) or not SHA256_RE.fullmatch(sha):
        raise Tau3SealedGridCompletenessError("sealed arm missing sealed authorization hash")
    return sha


def _require_ref_sha(record: Any, label: str) -> str:
    sha = _dict(record).get("sha256")
    if not isinstance(sha, str) or not SHA256_RE.fullmatch(sha):
        raise Tau3SealedGridCompletenessError(f"{label} reference missing valid sha256")
    return sha


def _read_json_ref(base: Path, record: Any, expected_sha256: Any, label: str) -> _JsonArtifact:
    if not isinstance(expected_sha256, str) or not SHA256_RE.fullmatch(expected_sha256):
        raise Tau3SealedGridCompletenessError(f"{label} reference missing valid sha256")
    rel = _dict(record).get("path")
    if not isinstance(rel, str):
        raise Tau3SealedGridCompletenessError(f"{label} reference path missing")
    path = _resolve_local_ref(base, rel, label)
    artifact = _read_json_artifact(path, label)
    if artifact.sha256 != expected_sha256:
        raise Tau3SealedGridCompletenessError(f"{label} file hash does not replay")
    return artifact


def _resolve_local_ref(base: Path, rel: Any, label: str) -> Path:
    if not isinstance(rel, str) or not rel:
        raise Tau3SealedGridCompletenessError(f"{label} reference path missing")
    if _is_unsafe_relative_path(rel):
        raise Tau3SealedGridCompletenessError(f"{label} reference path must be relative and local")
    unresolved = base / rel
    if path_has_symlink_component(unresolved, include_leaf=True):
        raise Tau3SealedGridCompletenessError(f"{label} reference contains a symlink component")
    path = unresolved.resolve()
    try:
        path.relative_to(base.resolve())
    except ValueError as exc:
        raise Tau3SealedGridCompletenessError(f"{label} reference escapes artifact root") from exc
    if not path.is_file():
        raise Tau3SealedGridCompletenessError(f"{label} referenced file does not exist")
    return path


def _read_json_artifact(path: Path, label: str) -> _JsonArtifact:
    if path_has_symlink_component(path, include_leaf=True):
        raise Tau3SealedGridCompletenessError(f"{label} must not contain symlink components")
    if not path.is_file():
        raise Tau3SealedGridCompletenessError(f"{label} does not exist")
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
        raise Tau3SealedGridCompletenessError(f"{label} changed while being read")
    raw = b"".join(chunks)
    if len(raw) != before.st_size:
        raise Tau3SealedGridCompletenessError(f"{label} read size mismatch")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise Tau3SealedGridCompletenessError(f"{label} must contain a JSON object")
    return _JsonArtifact(path=path, payload=payload, sha256=hashlib.sha256(raw).hexdigest(), size=len(raw))


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _is_unsafe_relative_path(value: str) -> bool:
    pure = PurePosixPath(value)
    win = PureWindowsPath(value)
    return pure.is_absolute() or win.is_absolute() or "\\" in value or any(part in {"", ".", ".."} for part in pure.parts)


def _assert_public_safe(value: Any) -> None:
    violations: list[str] = []
    _collect_public_safety_violations(value, "$", violations)
    if violations:
        raise Tau3SealedGridCompletenessError("sealed-grid completeness public artifact contains forbidden material: " + "; ".join(violations))


def _collect_public_safety_violations(value: Any, location: str, violations: list[str]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            key_text = str(key)
            if key_text in FORBIDDEN_PUBLIC_KEYS:
                violations.append(f"{location}.{key_text}")
            _collect_public_safety_violations(nested, f"{location}.{key_text}", violations)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _collect_public_safety_violations(nested, f"{location}[{index}]", violations)
    elif isinstance(value, str) and PRIVATE_PATH_RE.search(value):
        violations.append(location)


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _stable_cli_error(exc: BaseException) -> dict[str, str]:
    text = str(exc).lower()
    if "authorization" in text:
        code = "authorization_replay_failed"
    elif "hash" in text:
        code = "hash_replay_failed"
    elif "symlink" in text:
        code = "unsafe_path"
    elif "coverage" in text or "episode" in text or "task" in text:
        code = "sealed_coverage_failed"
    elif "schema" in text:
        code = "schema_validation_failed"
    else:
        code = "sealed_grid_completeness_failed"
    return {"error_code": code, "message": "sealed grid completeness validation failed"}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
