#!/usr/bin/env python3
"""Build a create-once Tau-3 parent-to-child protocol lineage attestation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.atomic_json import AtomicJsonError, atomic_write_json_cas  # noqa: E402
from flightrecorder.path_safety import path_has_symlink_component  # noqa: E402
from flightrecorder.schema_registry import check_schema_contract  # noqa: E402
from flightrecorder.tau3_policy_complete_dataset import (  # noqa: E402
    TAU3_POLICY_COMPLETE_DATASET_SCHEMA_VERSION,
)


SCHEMA_VERSION = "hfr.tau3_protocol_lineage_attestation.v1"
TEACHER_ROLE = "teacher_generation_and_review_only"
PUBLIC_PATH_KEYS = {
    "local_path",
    "source_path",
    "model_path",
    "tokenizer_path",
    "license_path",
    "manifest_path",
    "cache_path",
    "local_identity_path",
}
PARENT_PROTOCOL_ROLES = {
    "protocol_manifest": ("protocol", "protocol_manifest.json"),
    "tau_revision": ("protocol", "tau_revision.json"),
    "split_manifest": ("protocol", "split_manifest.json"),
    "harness_contract": ("protocol", "harness_contract.json"),
    "model_freeze": ("protocol", "model_freeze.json"),
    "budget": ("protocol", "budget.json"),
    "sealed_manifest": ("sealed", "sealed_manifest.json"),
    "mlx_qlora_plan": ("training", "mlx_qlora_plan.json"),
    "recipe_space": ("training", "recipe_space.json"),
    "candidate_selection_contract": ("training", "candidate_selection_contract.json"),
    "contamination_report": ("generation", "contamination_report.json"),
    "redaction_report": ("generation", "redaction_report.json"),
    "license_report": ("generation", "license_report.json"),
}
STRIPPED_PROTOCOL_MANIFEST_KEYS = {
    "created_at",
    "environment",
    "frozen",
    "lineage_rule",
    "schema_version",
    "signature",
    "signature_algorithm",
    "signed",
}


class Tau3ProtocolLineageError(ValueError):
    """Raised when the child protocol is not a constrained lineage extension."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-bundle", type=Path, default=Path("runs/tau3_core_training_artifacts"))
    parser.add_argument("--child-protocol", type=Path, default=Path("local/tau3/protocol-teacher-v1.json"))
    parser.add_argument("--corpus", type=Path, required=True, help="Canonical teacher/data-generation corpus file")
    parser.add_argument("--mixture", type=Path, required=True, help="Canonical teacher/data-generation mixture file")
    parser.add_argument("--out", type=Path, required=True, help="New attestation JSON path")
    parser.add_argument("--created-at", default="2026-07-24T00:00:00+00:00")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        attestation = build_tau3_protocol_lineage_attestation(
            parent_bundle=args.parent_bundle,
            child_protocol=args.child_protocol,
            corpus=args.corpus,
            mixture=args.mixture,
            out=args.out,
            created_at=args.created_at,
        )
    except (OSError, Tau3ProtocolLineageError, AtomicJsonError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(attestation, indent=2, sort_keys=True))
    return 0


def build_tau3_protocol_lineage_attestation(
    *,
    parent_bundle: str | Path,
    child_protocol: str | Path,
    corpus: str | Path,
    mixture: str | Path,
    out: str | Path,
    created_at: str = "2026-07-24T00:00:00+00:00",
) -> dict[str, Any]:
    """Validate and write a create-once parent-to-teacher-protocol attestation."""

    target = Path(out)
    if target.exists():
        raise Tau3ProtocolLineageError(f"attestation output already exists: {target}")

    parent_root = Path(parent_bundle)
    child_path = Path(child_protocol)
    corpus_ref = _file_identity(Path(corpus), "corpus")
    mixture_ref = _file_identity(Path(mixture), "mixture")
    parent_protocol, parent_artifacts = _load_parent_protocol(parent_root)
    child_raw = _read_json_object(child_path, "child protocol")

    child_checks = _validate_child_protocol(
        parent_protocol,
        child_raw,
        corpus_ref=corpus_ref,
    )
    child_checks.extend(
        _validate_bound_mixture(
            Path(mixture),
            child_protocol_sha256=_sha256_file(child_path),
        )
    )
    parent_normalized = _normalize_protocol(parent_protocol)
    child_normalized = _normalize_protocol(child_raw)
    parent_hash = _canonical_sha256(parent_normalized)
    child_hash = _canonical_sha256(child_normalized)

    attestation = {
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at,
        "passed": True,
        "lineage": {
            "kind": "tau3_parent_child_protocol",
            "rule": "child may only add the predeclared teacher/data-generation extension; all frozen parent fields must replay after public canonicalization",
        },
        "parent": {
            "bundle_path": _public_path(parent_root),
            "protocol_canonical_sha256": parent_hash,
            "bundle_manifest_sha256": _sha256_file(parent_root / "manifest.json"),
            "artifacts": parent_artifacts,
        },
        "child": {
            "path": _public_path(child_path),
            "file_sha256": _sha256_file(child_path),
            "protocol_canonical_sha256": child_hash,
        },
        "bindings": {
            "corpus": corpus_ref,
            "mixture": mixture_ref,
            "child_training_captures_sha256": child_raw["split_manifest"]["training_captures"]["sha256"],
        },
        "allowed_delta": {
            "model_freeze.teachers": "exactly one pinned teacher with role teacher_generation_and_review_only and comparator_eligible=false",
            "model_freeze.teacher_policy": "teacher-only generation/review policy text may be added",
            "licenses": "one matching teacher license row may be appended",
            "sealed_manifest.quarantined_at": "timestamp drift is ignored; sealed access_count and hash lists remain frozen",
            "split_manifest.training_captures.local_path": "machine-local path is ignored; corpus hash/counts remain frozen and bound",
        },
        "frozen_field_hashes": _frozen_field_hashes(parent_normalized, child_normalized),
        "checks": child_checks,
    }
    attestation["attestation_sha256"] = _canonical_sha256(attestation)
    atomic_write_json_cas(target, attestation, expected_sha256=None, new_file_mode=0o600)
    return attestation


def _validate_bound_mixture(
    mixture_path: Path,
    *,
    child_protocol_sha256: str,
) -> list[dict[str, Any]]:
    payload = _read_json_object(mixture_path, "mixture")
    if (
        payload.get("schema_version")
        != TAU3_POLICY_COMPLETE_DATASET_SCHEMA_VERSION
    ):
        return [
            {
                "id": "mixture_file_bound",
                "passed": True,
                "expected": "create-once file identity",
                "actual": _sha256_file(mixture_path),
            }
        ]
    schema = check_schema_contract(
        payload,
        name_or_id=TAU3_POLICY_COMPLETE_DATASET_SCHEMA_VERSION,
    )
    if schema.get("passed") is not True:
        raise Tau3ProtocolLineageError(
            "policy-complete mixture schema failed: "
            + "; ".join(schema.get("errors") or [])
        )
    replayed = _canonical_sha256(
        {
            key: value
            for key, value in payload.items()
            if key != "manifest_sha256"
        }
    )
    parent = payload.get("parent_protocol")
    sealed = payload.get("sealed")
    if (
        payload.get("manifest_sha256") != replayed
        or not isinstance(parent, dict)
        or parent.get("sha256") != child_protocol_sha256
        or payload.get("passed") is not True
        or payload.get("training_started") is not False
        or not isinstance(sealed, dict)
        or sealed.get("access_count") != 0
        or sealed.get("payload_accessed") is not False
    ):
        raise Tau3ProtocolLineageError(
            "policy-complete mixture does not replay its seal, protocol, or sealed-isolation gates"
        )
    return [
        {
            "id": "policy_complete_mixture_schema",
            "passed": True,
            "expected": TAU3_POLICY_COMPLETE_DATASET_SCHEMA_VERSION,
            "actual": payload.get("schema_version"),
        },
        {
            "id": "policy_complete_mixture_self_seal",
            "passed": True,
            "expected": replayed,
            "actual": payload.get("manifest_sha256"),
        },
        {
            "id": "policy_complete_mixture_parent_protocol",
            "passed": True,
            "expected": child_protocol_sha256,
            "actual": parent.get("sha256"),
        },
        {
            "id": "policy_complete_mixture_sealed_access_zero",
            "passed": True,
            "expected": 0,
            "actual": sealed.get("access_count"),
        },
    ]


def _load_parent_protocol(parent_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if path_has_symlink_component(parent_root, include_leaf=True):
        raise Tau3ProtocolLineageError(f"parent bundle path must not contain symlink components: {parent_root}")
    manifest_path = parent_root / "manifest.json"
    manifest = _read_json_object(manifest_path, "parent bundle manifest")
    if manifest.get("ready_for_training") is not True:
        raise Tau3ProtocolLineageError("parent bundle is not marked ready_for_training")
    records = {
        str(record.get("role")): record
        for record in manifest.get("artifacts", [])
        if isinstance(record, dict)
    }
    missing = sorted(set(PARENT_PROTOCOL_ROLES) - set(records))
    if missing:
        raise Tau3ProtocolLineageError("parent bundle missing required artifact role(s): " + ", ".join(missing))

    loaded: dict[str, Any] = {}
    artifact_records: list[dict[str, Any]] = []
    for role in sorted(PARENT_PROTOCOL_ROLES):
        record = records[role]
        rel_path = _safe_relative_path(record.get("path"), role)
        path = parent_root / rel_path
        digest = _sha256_file(path)
        expected = record.get("sha256")
        if digest != expected:
            raise Tau3ProtocolLineageError(f"parent artifact hash mismatch for {role}: expected {expected}, got {digest}")
        size = path.stat().st_size
        if record.get("size") != size:
            raise Tau3ProtocolLineageError(f"parent artifact size mismatch for {role}: expected {record.get('size')}, got {size}")
        loaded[role] = _read_json_object(path, f"parent {role}")
        artifact_records.append({"role": role, "path": str(rel_path), "sha256": digest, "size": size})

    protocol_manifest = loaded["protocol_manifest"]
    protocol: dict[str, Any] = {
        "schema_version": "hfr.tau3_protocol_config.v1",
        "protocol_manifest": protocol_manifest,
        "tau_revision": loaded["tau_revision"],
        "split_manifest": loaded["split_manifest"],
        "harness_contract": loaded["harness_contract"],
        "model_freeze": loaded["model_freeze"],
        "budget": loaded["budget"],
        "sealed_manifest": loaded["sealed_manifest"],
        "mlx_qlora_plan": loaded["mlx_qlora_plan"],
        "recipe_space": loaded["recipe_space"],
        "candidate_selection_contract": loaded["candidate_selection_contract"],
        "contamination_attestation": loaded["contamination_report"].get("attestation", {}),
        "redaction_attestation": loaded["redaction_report"].get("attestation", {}),
        "licenses": loaded["license_report"].get("sources", []),
        "environment_manifest": protocol_manifest.get("environment", {}),
    }
    return protocol, artifact_records


def _validate_child_protocol(
    parent_protocol: dict[str, Any],
    child_raw: dict[str, Any],
    *,
    corpus_ref: dict[str, Any],
) -> list[dict[str, Any]]:
    if child_raw.get("schema_version") != "hfr.tau3_protocol_config.v1":
        raise Tau3ProtocolLineageError("child protocol schema_version must be hfr.tau3_protocol_config.v1")
    child_training_captures = child_raw.get("split_manifest", {}).get("training_captures")
    if not isinstance(child_training_captures, dict):
        raise Tau3ProtocolLineageError("child split_manifest.training_captures must be an object")
    if child_training_captures.get("sha256") != corpus_ref["sha256"]:
        raise Tau3ProtocolLineageError("child training_captures sha256 does not match derived corpus file sha256")

    parent_normalized = _normalize_protocol(parent_protocol)
    child_normalized = _normalize_protocol(child_raw)
    expected_child = copy.deepcopy(parent_normalized)
    _apply_allowed_teacher_delta(expected_child, child_normalized)
    if expected_child != child_normalized:
        diffs = _diff_paths(expected_child, child_normalized)
        raise Tau3ProtocolLineageError("child protocol changes forbidden frozen field(s): " + ", ".join(diffs[:20]))

    return [
        {"id": "parent_bundle_ready", "passed": True, "expected": True, "actual": True},
        {"id": "frozen_fields_replay", "passed": True, "expected": "no forbidden diffs", "actual": "no forbidden diffs"},
        {"id": "teacher_extension_predeclared", "passed": True, "expected": TEACHER_ROLE, "actual": TEACHER_ROLE},
        {"id": "corpus_hash_bound", "passed": True, "expected": corpus_ref["sha256"], "actual": child_training_captures["sha256"]},
    ]


def _apply_allowed_teacher_delta(expected_child: dict[str, Any], child_normalized: dict[str, Any]) -> None:
    child_model_freeze = child_normalized.get("model_freeze")
    if not isinstance(child_model_freeze, dict):
        raise Tau3ProtocolLineageError("child model_freeze must be an object")
    teachers = child_model_freeze.get("teachers")
    if not isinstance(teachers, list) or len(teachers) != 1 or not isinstance(teachers[0], dict):
        raise Tau3ProtocolLineageError("child model_freeze.teachers must contain exactly one teacher object")
    teacher = teachers[0]
    if teacher.get("role") != TEACHER_ROLE:
        raise Tau3ProtocolLineageError("child teacher role must be teacher_generation_and_review_only")
    eligibility = teacher.get("pre_run_eligibility")
    if not isinstance(eligibility, dict):
        raise Tau3ProtocolLineageError("child teacher pre_run_eligibility must be an object")
    if eligibility.get("role") != TEACHER_ROLE:
        raise Tau3ProtocolLineageError("child teacher pre_run_eligibility.role must be teacher_generation_and_review_only")
    if eligibility.get("comparator_eligible") is not False:
        raise Tau3ProtocolLineageError("child teacher must not be comparator eligible")
    if eligibility.get("excluded_from_comparator_rule") is not True:
        raise Tau3ProtocolLineageError("child teacher must be excluded from comparator rule")
    if teacher.get("name") != "mlx-community/Qwen3.6-35B-A3B-4bit":
        raise Tau3ProtocolLineageError("child teacher must be the predeclared Qwen3.6 35B A3B teacher")
    if teacher.get("revision") != "38740b847e4cb78f352aba30aa41c76e08e6eb46":
        raise Tau3ProtocolLineageError("child teacher revision is not the predeclared immutable revision")
    if teacher.get("license") != "Apache-2.0":
        raise Tau3ProtocolLineageError("child teacher license must be Apache-2.0")

    expected_model_freeze = expected_child["model_freeze"]
    if expected_model_freeze.get("teachers") not in ([], None):
        raise Tau3ProtocolLineageError("parent protocol already has teachers; refusing non-create-once teacher extension")
    expected_model_freeze["teachers"] = teachers
    if "teacher_policy" in child_model_freeze:
        if child_model_freeze["teacher_policy"] != (
            "Teachers are pinned for local generation and review evidence only. "
            "They are excluded from the 7-9B dense comparator eligibility rule and from benchmark superiority claims."
        ):
            raise Tau3ProtocolLineageError("child teacher_policy is not the predeclared teacher-only policy")
        expected_model_freeze["teacher_policy"] = child_model_freeze["teacher_policy"]

    parent_licenses = expected_child.get("licenses")
    child_licenses = child_normalized.get("licenses")
    if not isinstance(parent_licenses, list) or not isinstance(child_licenses, list):
        raise Tau3ProtocolLineageError("parent and child licenses must be arrays")
    if child_licenses[: len(parent_licenses)] != parent_licenses:
        raise Tau3ProtocolLineageError("child licenses must preserve all parent license rows")
    appended = child_licenses[len(parent_licenses) :]
    expected_license = {
        "id": teacher["name"],
        "license": "Apache-2.0",
        "status": "approved",
        "training_allowed": True,
        "usage": TEACHER_ROLE,
    }
    if appended != [expected_license]:
        raise Tau3ProtocolLineageError("child must append exactly one matching teacher license row")
    expected_child["licenses"] = child_licenses


def _normalize_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    normalized = _public_contract(protocol)
    protocol_manifest = normalized.get("protocol_manifest")
    if isinstance(protocol_manifest, dict):
        for key in STRIPPED_PROTOCOL_MANIFEST_KEYS:
            protocol_manifest.pop(key, None)
    mlx_plan = normalized.get("mlx_qlora_plan")
    if isinstance(mlx_plan, dict):
        mlx_plan.pop("tokenizer_compatibility", None)
    sealed_manifest = normalized.get("sealed_manifest")
    if isinstance(sealed_manifest, dict):
        sealed_manifest.pop("quarantined_at", None)
    return normalized


def _public_contract(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _public_contract(item)
            for key, item in value.items()
            if str(key) not in PUBLIC_PATH_KEYS
        }
    if isinstance(value, list):
        return [_public_contract(item) for item in value]
    return value


def _frozen_field_hashes(parent_normalized: dict[str, Any], child_normalized: dict[str, Any]) -> list[dict[str, str]]:
    fields = [
        "protocol_manifest",
        "tau_revision",
        "split_manifest",
        "harness_contract",
        "budget",
        "sealed_manifest",
        "mlx_qlora_plan",
        "recipe_space",
        "candidate_selection_contract",
        "contamination_attestation",
        "redaction_attestation",
    ]
    rows: list[dict[str, str]] = []
    for field in fields:
        rows.append({
            "field": field,
            "parent_sha256": _canonical_sha256(parent_normalized[field]),
            "child_sha256": _canonical_sha256(child_normalized[field]),
        })
    parent_model = copy.deepcopy(parent_normalized["model_freeze"])
    child_model = copy.deepcopy(child_normalized["model_freeze"])
    parent_model.pop("teachers", None)
    child_model.pop("teachers", None)
    child_model.pop("teacher_policy", None)
    rows.append({
        "field": "model_freeze_without_teacher_extension",
        "parent_sha256": _canonical_sha256(parent_model),
        "child_sha256": _canonical_sha256(child_model),
    })
    rows.append({
        "field": "parent_license_prefix",
        "parent_sha256": _canonical_sha256(parent_normalized["licenses"]),
        "child_sha256": _canonical_sha256(child_normalized["licenses"][: len(parent_normalized["licenses"])]),
    })
    return rows


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if path_has_symlink_component(path, include_leaf=True):
        raise Tau3ProtocolLineageError(f"{label} path must not contain symlink components: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Tau3ProtocolLineageError(f"{label} must contain a JSON object: {path}")
    return payload


def _file_identity(path: Path, label: str) -> dict[str, Any]:
    if path_has_symlink_component(path, include_leaf=True):
        raise Tau3ProtocolLineageError(f"{label} path must not contain symlink components: {path}")
    if not path.is_file():
        raise Tau3ProtocolLineageError(f"{label} must be a regular file: {path}")
    return {
        "path": _public_path(path),
        "sha256": _sha256_file(path),
        "size": path.stat().st_size,
    }


def _public_path(path: Path) -> str:
    raw = path
    try:
        resolved = raw.resolve(strict=False)
        return str(resolved.relative_to(ROOT))
    except ValueError:
        pass
    if not raw.is_absolute() and ".." not in raw.parts:
        return str(raw)
    return raw.name


def _safe_relative_path(value: Any, role: str) -> Path:
    if not isinstance(value, str) or not value:
        raise Tau3ProtocolLineageError(f"parent artifact {role} path must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise Tau3ProtocolLineageError(f"parent artifact {role} path is unsafe: {value}")
    return path


def _sha256_file(path: Path) -> str:
    if path_has_symlink_component(path, include_leaf=True):
        raise Tau3ProtocolLineageError(f"hash input must not contain symlink components: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _diff_paths(left: Any, right: Any, path: str = "$") -> list[str]:
    if type(left) is not type(right):
        return [path]
    if isinstance(left, dict):
        diffs: list[str] = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                diffs.append(f"{path}.{key}")
            else:
                diffs.extend(_diff_paths(left[key], right[key], f"{path}.{key}"))
        return diffs
    if isinstance(left, list):
        diffs = [] if len(left) == len(right) else [path]
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            diffs.extend(_diff_paths(left_item, right_item, f"{path}[{index}]"))
        return diffs
    return [] if left == right else [path]


if __name__ == "__main__":
    raise SystemExit(main())
