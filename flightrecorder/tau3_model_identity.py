"""Content-addressed identity for local Tau-3 study model directories."""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path, PurePosixPath
from typing import Any

from .path_safety import path_has_symlink_component


TAU3_MODEL_IDENTITY_SCHEMA_VERSION = "hfr.tau3_model_identity.v1"
_MUTABLE_REVISIONS = {"", "main", "master", "head", "latest", "unknown"}
_WEIGHT_SUFFIXES = {".safetensors", ".npz", ".gguf", ".bin"}
_TOKENIZER_PAYLOAD_NAMES = {
    "tokenizer.json",
    "tokenizer.model",
    "sentencepiece.bpe.model",
    "spiece.model",
}


class Tau3ModelIdentityError(ValueError):
    """Raised when a local model tree cannot produce a trustworthy identity."""


def build_tau3_model_identity(
    model_path: str | Path,
    *,
    model_id: str,
    revision: str,
) -> dict[str, Any]:
    """Hash every regular file in a local model tree into a portable identity."""

    root = Path(model_path)
    _validate_declared_identity(model_id, revision)
    errors = _model_tree_errors(root)
    if errors:
        raise Tau3ModelIdentityError("; ".join(errors))
    records = [_file_record(path, root) for path in _regular_files(root)]
    requirement_errors = _required_model_file_errors(records)
    if requirement_errors:
        raise Tau3ModelIdentityError("; ".join(requirement_errors))
    total_size = sum(int(record["size"]) for record in records)
    return {
        "schema_version": TAU3_MODEL_IDENTITY_SCHEMA_VERSION,
        "model_id": model_id,
        "revision": revision,
        "file_count": len(records),
        "total_size": total_size,
        "tree_sha256": _canonical_sha256(records),
        "files": records,
    }


def validate_tau3_model_identity(
    identity: Any,
    model_path: str | Path,
    *,
    expected_model_id: str,
    expected_revision: str,
) -> list[str]:
    """Replay an identity against the complete current local model tree."""

    if not isinstance(identity, dict):
        return ["model identity must be a JSON object"]
    errors: list[str] = []
    root = Path(model_path)
    try:
        _validate_declared_identity(expected_model_id, expected_revision)
    except Tau3ModelIdentityError as exc:
        errors.append(str(exc))
    if identity.get("schema_version") != TAU3_MODEL_IDENTITY_SCHEMA_VERSION:
        errors.append(f"schema_version must be {TAU3_MODEL_IDENTITY_SCHEMA_VERSION!r}")
    if identity.get("model_id") != expected_model_id:
        errors.append("model_id does not match the frozen model entry")
    if identity.get("revision") != expected_revision:
        errors.append("revision does not match the frozen model entry")
    errors.extend(_model_tree_errors(root))
    rows = identity.get("files")
    if not isinstance(rows, list) or not rows:
        errors.append("files must be a non-empty array")
        return errors
    records: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"files[{index}] must be an object")
            continue
        rel = _safe_relative_path(row.get("path"))
        if rel is None:
            errors.append(f"files[{index}].path must be a safe POSIX-relative path")
            continue
        key = rel.as_posix()
        if key in records:
            errors.append(f"duplicate model file record: {key}")
            continue
        records[key] = row
    if errors:
        return errors
    actual_files = {path.relative_to(root).as_posix(): path for path in _regular_files(root)}
    missing_records = sorted(set(actual_files) - set(records))
    stale_records = sorted(set(records) - set(actual_files))
    if missing_records:
        errors.append("identity omits model files: " + ", ".join(missing_records))
    if stale_records:
        errors.append("identity references missing model files: " + ", ".join(stale_records))
    normalized: list[dict[str, Any]] = []
    for rel_path in sorted(set(records) & set(actual_files)):
        row = records[rel_path]
        path = actual_files[rel_path]
        try:
            size, digest = _stable_file_fingerprint(path)
        except (OSError, Tau3ModelIdentityError) as exc:
            errors.append(f"model file could not be replayed safely: {rel_path}: {exc}")
            continue
        if row.get("size") != size:
            errors.append(f"model file size does not replay: {rel_path}")
        if row.get("sha256") != digest:
            errors.append(f"model file SHA-256 does not replay: {rel_path}")
        normalized.append({"path": rel_path, "size": size, "sha256": digest})
    errors.extend(_required_model_file_errors(normalized))
    if identity.get("file_count") != len(actual_files):
        errors.append("file_count does not replay")
    actual_total = sum(int(record["size"]) for record in normalized)
    if identity.get("total_size") != actual_total:
        errors.append("total_size does not replay")
    if identity.get("tree_sha256") != _canonical_sha256(normalized):
        errors.append("tree_sha256 does not replay")
    return errors


def _validate_declared_identity(model_id: str, revision: str) -> None:
    if not isinstance(model_id, str) or not model_id.strip():
        raise Tau3ModelIdentityError("model_id must be a non-empty string")
    if not isinstance(revision, str) or revision.strip().lower() in _MUTABLE_REVISIONS:
        raise Tau3ModelIdentityError("revision must be immutable and must not be main/latest/unknown")


def _model_tree_errors(root: Path) -> list[str]:
    if path_has_symlink_component(root, include_leaf=True):
        return ["model_path must not contain symlink components"]
    if not root.is_dir():
        return ["model_path must be an existing directory"]
    errors = []
    for path in root.rglob("*"):
        if path.is_symlink():
            errors.append(f"model tree must not contain symlinks: {path.relative_to(root).as_posix()}")
    if not any(path.is_file() for path in root.rglob("*")):
        errors.append("model tree contains no regular files")
    return errors


def _regular_files(root: Path) -> list[Path]:
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and not path.is_symlink()),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    size, digest = _stable_file_fingerprint(path)
    return {
        "path": path.relative_to(root).as_posix(),
        "size": size,
        "sha256": digest,
    }


def _required_model_file_errors(records: list[dict[str, Any]]) -> list[str]:
    names = {str(record.get("path") or "") for record in records}
    basenames = {PurePosixPath(name).name for name in names}
    errors = []
    if "config.json" not in basenames:
        errors.append("model identity must include config.json")
    has_tokenizer_payload = bool(_TOKENIZER_PAYLOAD_NAMES & basenames) or {
        "vocab.json",
        "merges.txt",
    }.issubset(basenames)
    if not has_tokenizer_payload:
        errors.append("model identity must include an actual tokenizer payload")
    if not any(PurePosixPath(name).suffix.lower() in _WEIGHT_SUFFIXES for name in names):
        errors.append("model identity must include at least one local weight file")
    return errors


def _safe_relative_path(value: Any) -> PurePosixPath | None:
    if not isinstance(value, str) or not value or "\\" in value:
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return None
    return path


def _stable_file_fingerprint(path: Path) -> tuple[int, str]:
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise Tau3ModelIdentityError("path is not a regular file")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    after = path.stat(follow_symlinks=False)
    stable_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
    if size != before.st_size or any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise Tau3ModelIdentityError("file changed while it was being hashed")
    return size, digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
