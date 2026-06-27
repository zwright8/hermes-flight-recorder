"""Portable archives for promotion-history evidence."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .decision_gate import DECISION_GATE_SCHEMA_VERSION
from .promotion_gate import PROMOTION_LEDGER_GATE_SCHEMA_VERSION
from .promotion_ledger import PROMOTION_LEDGER_SCHEMA_VERSION

PROMOTION_ARCHIVE_SCHEMA_VERSION = "hfr.promotion_archive.v1"


class PromotionArchiveError(ValueError):
    """Raised when a promotion archive cannot be produced."""


def build_promotion_archive(
    *,
    out_dir: str | Path,
    promotion_ledger_path: str | Path,
    promotion_ledger_gate_path: str | Path | None = None,
    decision_gate_paths: list[str | Path] | None = None,
    require_self_contained: bool = False,
    force: bool = False,
    preserve_paths: bool = False,
) -> dict[str, Any]:
    """Copy promotion evidence into a portable directory with a hash manifest."""
    target = Path(out_dir)
    artifacts_dir = target / "artifacts"
    _prepare_archive_dir(target, force)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    ledger_path = Path(promotion_ledger_path)
    ledger = _read_json_artifact(ledger_path, PROMOTION_LEDGER_SCHEMA_VERSION, "promotion ledger")
    ledger_record = _copy_artifact(
        "promotion_ledger",
        "promotion_ledger",
        ledger_path,
        artifacts_dir / "promotion_ledger.json",
        target,
        preserve_paths,
    )
    artifacts = [ledger_record]
    missing: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []

    gate_record: dict[str, Any] | None = None
    if promotion_ledger_gate_path is not None:
        gate_path = Path(promotion_ledger_gate_path)
        _read_json_artifact(gate_path, PROMOTION_LEDGER_GATE_SCHEMA_VERSION, "promotion ledger gate")
        gate_record = _copy_artifact(
            "promotion_ledger_gate",
            "promotion_ledger_gate",
            gate_path,
            artifacts_dir / "promotion_ledger_gate.json",
            target,
            preserve_paths,
        )
        artifacts.append(gate_record)
        relationships.append({"from": gate_record["name"], "to": ledger_record["name"], "type": "gates"})

    decision_sources = _decision_gate_sources(ledger, ledger_path, decision_gate_paths or [])
    copied_decisions: list[tuple[dict[str, Any], dict[str, Any], Path]] = []
    seen_decision_hashes: set[str] = set()
    for decision_index, (source_path, missing_reason) in enumerate(decision_sources):
        if source_path is None:
            missing.append(_missing("decision_gate", decision_index, missing_reason or "path is redacted or unavailable"))
            continue
        source_error = _copyable_file_error(source_path)
        if source_error is not None:
            missing.append(_missing("decision_gate", decision_index, source_error))
            continue
        decision_gate = _read_json_artifact(source_path, DECISION_GATE_SCHEMA_VERSION, "decision gate")
        digest = _sha256(source_path)
        if digest in seen_decision_hashes:
            continue
        seen_decision_hashes.add(digest)
        archive_name = f"decision_gate_{len(copied_decisions):03d}.json"
        record = _copy_artifact(
            f"decision_gate_{len(copied_decisions):03d}",
            "decision_gate",
            source_path,
            artifacts_dir / archive_name,
            target,
            preserve_paths,
        )
        artifacts.append(record)
        relationships.append({"from": ledger_record["name"], "to": record["name"], "type": "summarizes"})
        copied_decisions.append((record, decision_gate, source_path))

    seen_source_hashes: set[str] = set()
    for artifact_index, artifact in enumerate(artifacts):
        artifact["index"] = artifact_index
    for record, decision_gate, decision_path in copied_decisions:
        source_path, missing_reason = _source_artifact_path(decision_gate, decision_path)
        if source_path is None:
            missing.append(_missing("source_artifact", record["index"], missing_reason or "source artifact path is redacted or unavailable"))
            continue
        source_error = _copyable_file_error(source_path)
        if source_error is not None:
            missing.append(_missing("source_artifact", record["index"], source_error))
            continue
        digest = _sha256(source_path)
        if digest in seen_source_hashes:
            continue
        seen_source_hashes.add(digest)
        archive_name = f"source_artifact_{len(seen_source_hashes) - 1:03d}.json"
        source_record = _copy_artifact(
            f"source_artifact_{len(seen_source_hashes) - 1:03d}",
            "source_artifact",
            source_path,
            artifacts_dir / archive_name,
            target,
            preserve_paths,
        )
        artifacts.append(source_record)
        relationships.append({"from": record["name"], "to": source_record["name"], "type": "source_artifact"})

    self_contained = not missing
    for artifact_index, artifact in enumerate(artifacts):
        artifact["index"] = artifact_index
    archive = {
        "schema_version": PROMOTION_ARCHIVE_SCHEMA_VERSION,
        "archive_path": _display_path(target, preserve_paths),
        "manifest_path": "promotion_archive.json",
        "passed": self_contained or not require_self_contained,
        "self_contained": self_contained,
        "require_self_contained": require_self_contained,
        "artifacts": artifacts,
        "missing": missing,
        "relationships": relationships,
        "metrics": _metrics(artifacts, missing),
        "notes": [
            "Promotion archives copy promotion-history evidence into a portable directory; they do not rerun gates, train models, or mutate CI.",
            "Archive validation checks copied artifact hashes, so original local source paths are not required after the archive is built.",
        ],
    }
    _write_json(target / "promotion_archive.json", archive)
    return archive


def _decision_gate_sources(
    ledger: dict[str, Any],
    ledger_path: Path,
    explicit_paths: list[str | Path],
) -> list[tuple[Path | None, str | None]]:
    paths: list[tuple[Path | None, str | None]] = [(Path(path), None) for path in explicit_paths]
    for record in ledger.get("records", []):
        if not isinstance(record, dict):
            continue
        paths.append(_resolve_recorded_path(record.get("path"), ledger_path.parent))
    return paths


def _prepare_archive_dir(target: Path, force: bool) -> None:
    if target.exists() and not target.is_dir():
        raise PromotionArchiveError(f"promotion archive output is not a directory: {target}")
    if not target.exists() or not any(target.iterdir()):
        return
    if not force:
        raise PromotionArchiveError(f"promotion archive output is not empty: {target}; pass --force to replace it")
    if not _is_existing_promotion_archive(target):
        raise PromotionArchiveError(
            f"refusing to replace non-archive directory: {target}; choose an empty output directory or an existing promotion archive"
        )
    shutil.rmtree(target)


def _is_existing_promotion_archive(target: Path) -> bool:
    manifest_path = target / "promotion_archive.json"
    if not manifest_path.is_file():
        return False
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(value, dict) and value.get("schema_version") == PROMOTION_ARCHIVE_SCHEMA_VERSION


def _source_artifact_path(decision_gate: dict[str, Any], decision_path: Path) -> tuple[Path | None, str | None]:
    source = decision_gate.get("source_artifact") if isinstance(decision_gate.get("source_artifact"), dict) else {}
    return _resolve_recorded_path(source.get("path"), decision_path.parent)


def _copy_artifact(
    name: str,
    role: str,
    source_path: Path,
    archive_path: Path,
    archive_root: Path,
    preserve_paths: bool,
) -> dict[str, Any]:
    source_error = _copyable_file_error(source_path)
    if source_error is not None:
        raise PromotionArchiveError(f"{role} source {source_error}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, archive_path)
    value = json.loads(archive_path.read_text(encoding="utf-8"))
    record = {
        "index": 0,
        "name": name,
        "role": role,
        "path": str(archive_path.relative_to(archive_root)),
        "original_path": _display_path(source_path, preserve_paths),
        "exists": True,
        "schema_version": value.get("schema_version") if isinstance(value, dict) else None,
        "size_bytes": archive_path.stat().st_size,
        "sha256": _sha256(archive_path),
    }
    return record


def _missing(role: str, index: int, reason: str) -> dict[str, Any]:
    return {"role": role, "index": index, "reason": reason}


def _metrics(artifacts: list[dict[str, Any]], missing: list[dict[str, Any]]) -> dict[str, Any]:
    role_counts = _count_rows(record.get("role") for record in artifacts)
    missing_role_counts = _count_rows(record.get("role") for record in missing)
    return {
        "artifact_count": len(artifacts),
        "decision_gate_count": sum(1 for record in artifacts if record.get("role") == "decision_gate"),
        "source_artifact_count": sum(1 for record in artifacts if record.get("role") == "source_artifact"),
        "missing_count": len(missing),
        "role_counts": role_counts,
        "missing_role_counts": missing_role_counts,
        "unique_sha256_count": len({record.get("sha256") for record in artifacts if isinstance(record.get("sha256"), str)}),
    }


def _read_json_artifact(path: Path, schema_version: str, label: str) -> dict[str, Any]:
    source_error = _copyable_file_error(path)
    if source_error is not None:
        raise PromotionArchiveError(f"{label} {source_error}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PromotionArchiveError(f"{label} must contain a JSON object: {path}")
    if value.get("schema_version") != schema_version:
        raise PromotionArchiveError(f"{label} schema_version must be {schema_version!r}; got {value.get('schema_version')!r}")
    return value


def _resolve_recorded_path(value: Any, base_dir: Path) -> tuple[Path | None, str | None]:
    if not isinstance(value, str) or not value or value.startswith("<redacted:") or value.startswith("<missing-"):
        return None, "path is redacted or unavailable"
    raw = Path(value)
    if raw.is_absolute() or _is_windows_absolute(value):
        return None, "absolute recorded paths are not archived"
    if ".." in raw.parts:
        return None, "parent traversal is not allowed in recorded paths"

    candidates: list[Path] = []
    unsafe_reasons: list[str] = []
    for root in _unique_paths([Path.cwd(), base_dir]):
        candidate = root / raw
        if _path_resolves_inside(candidate, root):
            candidates.append(candidate)
        else:
            unsafe_reasons.append(f"path resolves outside allowed root: {root}")

    copy_errors: list[str] = []
    for candidate in candidates:
        source_error = _copyable_file_error(candidate)
        if source_error is None:
            return candidate, None
        if candidate.exists() or candidate.is_symlink():
            copy_errors.append(source_error)
    if copy_errors:
        return None, "; ".join(copy_errors)
    if candidates:
        return candidates[0], None
    return None, "; ".join(unsafe_reasons) or "recorded path could not be resolved safely"


def _copyable_file_error(path: Path) -> str | None:
    if path.is_symlink():
        return f"file is a symlink: {path}"
    if not path.exists() or not path.is_file():
        return f"file not found: {path}"
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        return f"file could not be resolved: {path}: {exc}"
    if not resolved.is_file():
        return f"path is not a regular file: {path}"
    return None


def _path_resolves_inside(path: Path, root: Path) -> bool:
    try:
        root_resolved = root.resolve()
        path_resolved = path.resolve(strict=False)
    except OSError:
        return False
    return path_resolved == root_resolved or path_resolved.is_relative_to(root_resolved)


def _unique_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        try:
            key = str(path.resolve(strict=False))
        except OSError:
            key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _count_rows(values: Any) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return [{"id": key, "count": counts[key]} for key in sorted(counts)]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path, preserve_paths: bool = False) -> str:
    raw = str(path)
    if preserve_paths:
        return raw
    if _is_windows_absolute(raw):
        return f"<redacted:{_basename(raw)}>"
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    try:
        return str(resolved.relative_to(cwd))
    except ValueError:
        return f"<redacted:{resolved.name}>"


def _is_windows_absolute(value: str) -> bool:
    normalized = value.replace("/", "\\")
    return (len(normalized) >= 3 and normalized[1:3] == ":\\" and normalized[0].isalpha()) or normalized.startswith("\\\\")


def _basename(value: str) -> str:
    return value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] or "path"
