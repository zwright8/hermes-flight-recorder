"""Governed Tau-3 candidate identity emitter."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .atomic_json import AtomicJsonError, atomic_write_json_cas
from .path_safety import path_has_symlink_component
from .schema_registry import check_schema_contract

TAU3_CANDIDATE_IDENTITY_SCHEMA_VERSION = "hfr.tau3_candidate_identity.v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CANDIDATE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


class Tau3CandidateIdentityError(ValueError):
    """Raised when a candidate identity cannot be emitted safely."""


def build_tau3_candidate_identity(
    *,
    candidate_id: str,
    training_receipt_path: str | Path,
    endpoint_model: str,
    output_path: str | Path,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Validate a final Tau-3 MLX receipt and write a public-safe identity."""

    candidate = candidate_id.strip()
    if not candidate:
        raise Tau3CandidateIdentityError("candidate_id must be non-empty")
    if not CANDIDATE_ID_RE.fullmatch(candidate) or candidate.startswith(("sk-", "hf_")):
        raise Tau3CandidateIdentityError("candidate_id must be a lowercase public-safe slug using only a-z, 0-9, '.', '_', or '-'")
    endpoint = str(endpoint_model)
    if not endpoint:
        raise Tau3CandidateIdentityError("endpoint_model must be non-empty")

    receipt_path = Path(training_receipt_path)
    out = Path(output_path)
    if out.exists():
        raise Tau3CandidateIdentityError(f"candidate identity already exists: {out}")
    if path_has_symlink_component(out, include_leaf=True):
        raise Tau3CandidateIdentityError(f"candidate identity output must not contain symlink components: {out}")

    receipt = _load_receipt(receipt_path)
    receipt_sha256 = _sha256_file(receipt_path)
    adapter = _verified_adapter(receipt, receipt_path=receipt_path)
    _reject_output_inside_adapter(out, adapter["_adapter_dir"])
    binding = _binding_hashes(receipt)
    endpoint_model_sha256 = hashlib.sha256(endpoint.encode("utf-8")).hexdigest()

    identity = {
        "schema_version": TAU3_CANDIDATE_IDENTITY_SCHEMA_VERSION,
        "created_at": created_at or _now_utc(),
        "candidate_id": candidate,
        "training_receipt_sha256": receipt_sha256,
        "final_training_receipt_sha256": receipt_sha256,
        "adapter_tree_sha256": adapter["receipt_tree_sha256"],
        "endpoint_model_sha256": endpoint_model_sha256,
        "training_binding": binding,
        "adapter_identity": {
            "adapter_tree_sha256": adapter["receipt_tree_sha256"],
            "tree_sha256": adapter["receipt_tree_sha256"],
            "file_count": adapter["file_count"],
            "adapter_weight_file_count": adapter["adapter_weight_file_count"],
            "declared_file_set_sha256": adapter["declared_file_set_sha256"],
            "replayed_file_set_sha256": adapter["replayed_file_set_sha256"],
        },
        "governance": {
            "training_receipt_schema_checked": True,
            "training_receipt_final": True,
            "training_receipt_success": True,
            "training_weights_updated": True,
            "adapter_files_replayed": True,
            "endpoint_model_hash_only": True,
            "hashes_only": True,
            "local_paths_included": False,
            "absolute_paths_included": False,
            "raw_endpoint_model_included": False,
            "raw_training_receipt_included": False,
            "public_safe": True,
            "private_material_included": False,
            "sealed_access_authorized": False,
        },
        "schema_checked": True,
        "read_only": True,
    }
    schema = check_schema_contract(identity, name_or_id="tau3_candidate_identity")
    if schema["passed"] is not True:
        raise Tau3CandidateIdentityError("candidate identity violates schema: " + "; ".join(schema["errors"]))
    digest = atomic_write_json_cas(out, identity, expected_sha256=None, new_file_mode=0o444)
    out.chmod(0o444)
    return {**identity, "identity_file_sha256": digest}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--endpoint-model", required=True, help="Exact endpoint model string; only its SHA-256 is emitted")
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        identity = build_tau3_candidate_identity(
            candidate_id=args.candidate_id,
            training_receipt_path=args.training_receipt,
            endpoint_model=args.endpoint_model,
            output_path=args.out,
        )
    except (AtomicJsonError, OSError, Tau3CandidateIdentityError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps({
        "adapter_tree_sha256": identity["adapter_tree_sha256"],
        "candidate_id": identity["candidate_id"],
        "endpoint_model_sha256": identity["endpoint_model_sha256"],
        "identity_file": str(args.out),
        "identity_file_sha256": identity["identity_file_sha256"],
        "training_receipt_sha256": identity["training_receipt_sha256"],
    }, indent=2, sort_keys=True))
    return 0


def _load_receipt(path: Path) -> dict[str, Any]:
    if path_has_symlink_component(path, include_leaf=True):
        raise Tau3CandidateIdentityError(f"training receipt must not contain symlink components: {path}")
    payload = _load_json_object(path)
    schema = check_schema_contract(payload, name_or_id="tau3_mlx_training_run")
    if schema["passed"] is not True:
        raise Tau3CandidateIdentityError(f"{path}: training receipt schema failed: " + "; ".join(schema["errors"]))
    failures = []
    if payload.get("phase") != "final":
        failures.append("phase")
    if payload.get("terminal_status") != "success":
        failures.append("terminal_status")
    if payload.get("weights_updated") is not True:
        failures.append("weights_updated")
    if payload.get("schema_checked") is not True:
        failures.append("schema_checked")
    if int(payload.get("adapter_weight_file_count") or 0) <= 0:
        failures.append("adapter_weight_file_count")
    if failures:
        raise Tau3CandidateIdentityError(f"{path}: training receipt is not final successful weighted output: {', '.join(failures)}")
    return payload


def _verified_adapter(receipt: dict[str, Any], *, receipt_path: Path) -> dict[str, Any]:
    raw_adapter = receipt.get("adapter")
    if not isinstance(raw_adapter, dict):
        raise Tau3CandidateIdentityError("training receipt missing adapter record")
    rel = raw_adapter.get("path")
    if not isinstance(rel, str) or not rel:
        raise Tau3CandidateIdentityError("training receipt adapter.path must be a portable relative path")
    adapter_dir = _resolve_receipt_relative_dir(receipt_path.parent, rel, label="adapter.path")
    declared_files = raw_adapter.get("files")
    if not isinstance(declared_files, list) or not declared_files:
        raise Tau3CandidateIdentityError("training receipt adapter.files must be non-empty")

    replayed_files = _fingerprint_training_tree(adapter_dir)
    declared_by_path: dict[str, dict[str, Any]] = {}
    for item in declared_files:
        if not isinstance(item, dict):
            raise Tau3CandidateIdentityError("training receipt adapter.files contains a non-object")
        rel_path = _portable_relative_path(item.get("path"), label="adapter file path")
        record = {
            "path": rel_path,
            "size": _int_field(item.get("size"), label="adapter file size"),
            "sha256": _sha_field(item.get("sha256"), label="adapter file sha256"),
            "kind": str(item.get("kind") or ""),
        }
        if record["kind"] not in {"adapter", "artifact", "checkpoint", "config"}:
            raise Tau3CandidateIdentityError(f"training receipt adapter file has unsupported kind: {record['kind']!r}")
        if rel_path in declared_by_path:
            raise Tau3CandidateIdentityError(f"duplicate declared adapter file: {rel_path}")
        declared_by_path[rel_path] = record

    if set(declared_by_path) != set(replayed_files):
        raise Tau3CandidateIdentityError("adapter file set does not replay")
    for rel_path, declared in declared_by_path.items():
        if declared != replayed_files[rel_path]:
            raise Tau3CandidateIdentityError(f"adapter file hash does not replay: {rel_path}")

    declared_tree_sha256 = _sha_field(raw_adapter.get("tree_sha256"), label="adapter.tree_sha256")
    replayed_tree_sha256 = _training_tree_sha256(list(replayed_files.values()))
    if declared_tree_sha256 != replayed_tree_sha256:
        raise Tau3CandidateIdentityError("adapter tree_sha256 does not replay")
    weight_count = sum(1 for record in replayed_files.values() if record["kind"] == "adapter" and record["size"] > 0)
    if weight_count <= 0:
        raise Tau3CandidateIdentityError("adapter tree contains no non-empty adapter weight file")
    if int(receipt.get("adapter_weight_file_count") or 0) != weight_count:
        raise Tau3CandidateIdentityError("adapter_weight_file_count does not replay")

    declared_order = [declared_by_path[path] for path in sorted(declared_by_path)]
    return {
        "receipt_tree_sha256": declared_tree_sha256,
        "file_count": len(declared_order),
        "adapter_weight_file_count": weight_count,
        "declared_file_set_sha256": _canonical_sha256(declared_order),
        "replayed_file_set_sha256": _canonical_sha256(list(replayed_files[path] for path in sorted(replayed_files))),
        "_adapter_dir": adapter_dir,
    }


def _binding_hashes(receipt: dict[str, Any]) -> dict[str, Any]:
    raw = receipt.get("training_binding")
    if not isinstance(raw, dict):
        raise Tau3CandidateIdentityError("training receipt missing training_binding")
    binding = {
        "protocol_sha256": _nested_sha(raw, "protocol", "sha256"),
        "protocol_signature": _nested_sha(raw, "protocol", "protocol_signature"),
        "model_freeze_sha256": _nested_sha(raw, "protocol", "model_freeze_sha256"),
        "recipe_space_sha256": _nested_sha(raw, "protocol", "recipe_space_sha256"),
        "mlx_qlora_plan_sha256": _nested_sha(raw, "protocol", "mlx_qlora_plan_sha256"),
        "base_identity_sha256": _nested_sha(raw, "model", "identity_sha256"),
        "base_tree_sha256": _nested_sha(raw, "model", "tree_sha256"),
        "dataset_manifest_sha256": _nested_sha(raw, "dataset", "manifest_sha256"),
        "dataset_files_sha256": _nested_sha(raw, "dataset", "files_sha256"),
        "source_binding_sha256": _nested_sha(raw, "dataset", "source_binding_sha256"),
        "recipe_sha256": _nested_sha(raw, "recipe", "recipe_sha256"),
    }
    missing = [key for key, value in binding.items() if value is None]
    if missing:
        raise Tau3CandidateIdentityError("training receipt missing required hash binding(s): " + ", ".join(missing))
    return binding


def _resolve_receipt_relative_dir(parent: Path, value: str, *, label: str) -> Path:
    rel = Path(_portable_relative_path(value, label=label))
    path = parent / rel
    if path_has_symlink_component(path, include_leaf=True):
        raise Tau3CandidateIdentityError(f"{label} must not contain symlink components")
    resolved_parent = parent.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(resolved_parent):
        raise Tau3CandidateIdentityError(f"{label} escapes the receipt directory")
    if not resolved.is_dir():
        raise Tau3CandidateIdentityError(f"{label} does not resolve to a directory")
    return resolved


def _fingerprint_training_tree(root: Path) -> dict[str, dict[str, Any]]:
    files = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path_has_symlink_component(path, include_leaf=True):
            raise Tau3CandidateIdentityError(f"adapter file must not contain symlink components: {path}")
        rel = path.relative_to(root).as_posix()
        files[rel] = {"path": rel, "size": path.stat().st_size, "sha256": _sha256_file(path), "kind": _fingerprint_kind(rel)}
    return files


def _training_tree_sha256(files: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(files, key=lambda item: item["path"]):
        digest.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def _fingerprint_kind(rel: str) -> str:
    name = Path(rel).name
    if name in {"adapter_config.json", "config.json"}:
        return "config"
    if "checkpoint" in rel.lower():
        return "checkpoint"
    if Path(rel).suffix in {".safetensors", ".npz", ".bin"}:
        return "adapter"
    return "artifact"


def _portable_relative_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise Tau3CandidateIdentityError(f"{label} must be a non-empty string")
    path = Path(value)
    if value == "." or path.is_absolute() or "\\" in value or ".." in path.parts:
        raise Tau3CandidateIdentityError(f"{label} must be a portable relative path")
    return path.as_posix()


def _reject_output_inside_adapter(out: Path, adapter_dir: Path) -> None:
    parent = out.parent.resolve(strict=True) if out.parent.exists() else out.parent.resolve()
    target = parent / out.name
    if target.resolve(strict=False).is_relative_to(adapter_dir.resolve(strict=True)):
        raise Tau3CandidateIdentityError("candidate identity output must be outside the adapter directory")


def _nested_sha(payload: dict[str, Any], *keys: str) -> str | None:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value if isinstance(value, str) and SHA256_RE.fullmatch(value) else None


def _sha_field(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise Tau3CandidateIdentityError(f"{label} must be a SHA-256 hex digest")
    return value


def _int_field(value: Any, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise Tau3CandidateIdentityError(f"{label} must be a non-negative integer")
    return value


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Tau3CandidateIdentityError(f"JSON file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Tau3CandidateIdentityError(f"JSON file must contain an object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
