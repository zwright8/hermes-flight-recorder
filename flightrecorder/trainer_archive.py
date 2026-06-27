"""Portable archives for trainer handoff evidence."""

from __future__ import annotations

import hashlib
import json
import shlex
import shutil
from pathlib import Path
from typing import Any

from .preflight import TRAINER_LAUNCH_CHECK_SCHEMA_VERSION, TRAINER_PREFLIGHT_SCHEMA_VERSION

TRAINER_ARCHIVE_SCHEMA_VERSION = "hfr.trainer_archive.v1"
_ARCHIVE_MANIFEST = "trainer_archive.json"
_TREE_HASH_ALGORITHM = "sha256(sorted-relative-path-size-file-sha256)"


class TrainerArchiveError(ValueError):
    """Raised when a trainer handoff archive cannot be produced."""


def build_trainer_archive(
    *,
    out_dir: str | Path,
    preflight_path: str | Path,
    launch_check_path: str | Path | None = None,
    require_self_contained: bool = False,
    force: bool = False,
    preserve_paths: bool = False,
) -> dict[str, Any]:
    """Copy trainer-launch evidence into a portable hash-checked directory."""
    target = Path(out_dir)
    artifacts_dir = target / "artifacts"
    _prepare_archive_dir(target, force)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    source_preflight_path = Path(preflight_path)
    preflight = _read_json_artifact(source_preflight_path, TRAINER_PREFLIGHT_SCHEMA_VERSION, "trainer preflight")
    artifacts: list[dict[str, Any]] = [
        _copy_file_artifact(
            "trainer_preflight",
            "trainer_preflight",
            source_preflight_path,
            artifacts_dir / "trainer_preflight.json",
            target,
            preserve_paths,
        )
    ]
    missing: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []

    launch_check: dict[str, Any] | None = None
    launch_included = False
    launch_passed = False
    if launch_check_path is None:
        missing.append(_missing("trainer_launch_check", 0, "launch check path was not provided"))
    else:
        source_launch_path = Path(launch_check_path)
        launch_check = _read_json_artifact(source_launch_path, TRAINER_LAUNCH_CHECK_SCHEMA_VERSION, "trainer launch check")
        launch_record = _copy_file_artifact(
            "trainer_launch_check",
            "trainer_launch_check",
            source_launch_path,
            artifacts_dir / "trainer_launch_check.json",
            target,
            preserve_paths,
        )
        artifacts.append(launch_record)
        relationships.append({"from": "trainer_launch_check", "to": "trainer_preflight", "type": "validates"})
        launch_included = True
        launch_passed = launch_check.get("passed") is True

    artifacts.extend(
        _copy_preflight_paths(
            preflight.get("gates"),
            role="gate",
            base_dir=source_preflight_path.parent,
            archive_dir=artifacts_dir / "gates",
            archive_root=target,
            missing=missing,
            preserve_paths=preserve_paths,
        )
    )
    artifacts.extend(
        _copy_preflight_paths(
            preflight.get("validation_summaries"),
            role="validation_summary",
            base_dir=source_preflight_path.parent,
            archive_dir=artifacts_dir / "validation_summaries",
            archive_root=target,
            missing=missing,
            preserve_paths=preserve_paths,
        )
    )
    artifacts.extend(
        _copy_preflight_mapping(
            preflight.get("artifacts"),
            role="trainer_artifact",
            base_dir=source_preflight_path.parent,
            archive_dir=artifacts_dir / "trainer_artifacts",
            archive_root=target,
            missing=missing,
            preserve_paths=preserve_paths,
        )
    )
    artifacts.extend(
        _copy_preflight_mapping(
            preflight.get("schema_contracts"),
            role="schema_contract",
            base_dir=source_preflight_path.parent,
            archive_dir=artifacts_dir / "schema_contracts",
            archive_root=target,
            missing=missing,
            preserve_paths=preserve_paths,
        )
    )

    for index, artifact in enumerate(artifacts):
        artifact["index"] = index

    approved_command = _approved_command_record(launch_check)
    trainer_inputs = _trainer_inputs(artifacts)
    path_rewrites = _path_rewrites(trainer_inputs)
    portable_command = _portable_command_record(approved_command, path_rewrites)
    self_contained = not missing
    ready_for_training = preflight.get("passed") is True and launch_included and launch_passed
    passed = ready_for_training and (self_contained or not require_self_contained)
    archive = {
        "schema_version": TRAINER_ARCHIVE_SCHEMA_VERSION,
        "archive_path": _display_path(target, preserve_paths),
        "manifest_path": _ARCHIVE_MANIFEST,
        "passed": passed,
        "readiness": "ready" if passed else "blocked",
        "recommendation": "handoff_ready" if passed else "block_handoff",
        "self_contained": self_contained,
        "require_self_contained": require_self_contained,
        "ready_for_training": ready_for_training,
        "launch_check_included": launch_included,
        "approved_command": approved_command,
        "trainer_inputs": trainer_inputs,
        "path_rewrites": path_rewrites,
        "portable_command": portable_command,
        "artifacts": artifacts,
        "missing": missing,
        "relationships": relationships,
        "metrics": _metrics(artifacts, missing, trainer_inputs, path_rewrites),
        "notes": [
            "Trainer archives copy trainer handoff evidence into a portable directory; they do not train models or execute the trainer command.",
            "The portable command is advisory: it rewrites known trainer-input paths to archive-local paths but does not include trainer code or execute anything.",
            "Archive validation checks copied file hashes and directory tree hashes, so original local source paths are not required after the archive is built.",
        ],
    }
    _write_json(target / _ARCHIVE_MANIFEST, archive)
    return archive


def _copy_preflight_paths(
    value: Any,
    *,
    role: str,
    base_dir: Path,
    archive_dir: Path,
    archive_root: Path,
    missing: list[dict[str, Any]],
    preserve_paths: bool,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    records: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            missing.append(_missing(role, index, "preflight record is not an object"))
            continue
        name = _record_name(role, item, index)
        source_path, reason = _resolve_recorded_path(item.get("path"), base_dir)
        if source_path is None:
            missing.append(_missing(role, index, reason or "path is unavailable", name=name))
            continue
        try:
            records.append(
                _copy_path_artifact(
                    name,
                    role,
                    source_path,
                    archive_dir / f"{index:03d}_{_slug(name)}",
                    archive_root,
                    preserve_paths,
                )
            )
        except TrainerArchiveError as exc:
            missing.append(_missing(role, index, str(exc), name=name))
    return records


def _copy_preflight_mapping(
    value: Any,
    *,
    role: str,
    base_dir: Path,
    archive_dir: Path,
    archive_root: Path,
    missing: list[dict[str, Any]],
    preserve_paths: bool,
) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    records: list[dict[str, Any]] = []
    for index, (name, item) in enumerate(sorted(value.items())):
        clean_name = str(name or f"{role}_{index}")
        if not isinstance(item, dict):
            missing.append(_missing(role, index, "preflight record is not an object", name=clean_name))
            continue
        source_path, reason = _resolve_recorded_path(item.get("path"), base_dir)
        if source_path is None:
            missing.append(_missing(role, index, reason or "path is unavailable", name=clean_name))
            continue
        try:
            records.append(
                _copy_path_artifact(
                    clean_name,
                    role,
                    source_path,
                    archive_dir / f"{index:03d}_{_slug(clean_name)}",
                    archive_root,
                    preserve_paths,
                )
            )
        except TrainerArchiveError as exc:
            missing.append(_missing(role, index, str(exc), name=clean_name))
    return records


def _copy_path_artifact(
    name: str,
    role: str,
    source_path: Path,
    archive_path: Path,
    archive_root: Path,
    preserve_paths: bool,
) -> dict[str, Any]:
    if source_path.is_dir() and not source_path.is_symlink():
        return _copy_directory_artifact(name, role, source_path, archive_path, archive_root, preserve_paths)
    return _copy_file_artifact(name, role, source_path, archive_path.with_suffix(_suffix_for(source_path)), archive_root, preserve_paths)


def _copy_file_artifact(
    name: str,
    role: str,
    source_path: Path,
    archive_path: Path,
    archive_root: Path,
    preserve_paths: bool,
) -> dict[str, Any]:
    source_error = _copyable_file_error(source_path)
    if source_error is not None:
        raise TrainerArchiveError(f"{role} source {source_error}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, archive_path)
    payload = _read_json_optional(archive_path)
    return {
        "index": 0,
        "name": name,
        "role": role,
        "kind": "file",
        "path": str(archive_path.relative_to(archive_root)),
        "original_path": _display_path(source_path, preserve_paths),
        "exists": True,
        "schema_version": payload.get("schema_version") if isinstance(payload, dict) else None,
        "source_passed": payload.get("passed") if isinstance(payload, dict) and isinstance(payload.get("passed"), bool) else None,
        "size_bytes": archive_path.stat().st_size,
        "sha256": _sha256(archive_path),
    }


def _copy_directory_artifact(
    name: str,
    role: str,
    source_path: Path,
    archive_path: Path,
    archive_root: Path,
    preserve_paths: bool,
) -> dict[str, Any]:
    source_error = _copyable_directory_error(source_path)
    if source_error is not None:
        raise TrainerArchiveError(f"{role} source {source_error}")
    if _path_resolves_inside(archive_path, source_path):
        raise TrainerArchiveError(f"archive path must not be inside source directory: {archive_path}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        shutil.rmtree(archive_path)
    _copy_regular_tree(source_path, archive_path)
    tree = _tree_fingerprint(archive_path)
    return {
        "index": 0,
        "name": name,
        "role": role,
        "kind": "directory",
        "path": str(archive_path.relative_to(archive_root)),
        "original_path": _display_path(source_path, preserve_paths),
        "exists": True,
        "schema_version": None,
        "source_passed": None,
        "size_bytes": tree["size_bytes"],
        "file_count": tree["file_count"],
        "sha256": tree["sha256"],
        "tree_hash_algorithm": _TREE_HASH_ALGORITHM,
    }


def _prepare_archive_dir(target: Path, force: bool) -> None:
    if target.exists() and not target.is_dir():
        raise TrainerArchiveError(f"trainer archive output is not a directory: {target}")
    if not target.exists() or not any(target.iterdir()):
        return
    if not force:
        raise TrainerArchiveError(f"trainer archive output is not empty: {target}; pass --force to replace it")
    if not _is_existing_trainer_archive(target):
        raise TrainerArchiveError(
            f"refusing to replace non-archive directory: {target}; choose an empty output directory or an existing trainer archive"
        )
    shutil.rmtree(target)


def _is_existing_trainer_archive(target: Path) -> bool:
    manifest_path = target / _ARCHIVE_MANIFEST
    if not manifest_path.is_file():
        return False
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(value, dict) and value.get("schema_version") == TRAINER_ARCHIVE_SCHEMA_VERSION


def _read_json_artifact(path: Path, schema_version: str, label: str) -> dict[str, Any]:
    source_error = _copyable_file_error(path)
    if source_error is not None:
        raise TrainerArchiveError(f"{label} {source_error}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TrainerArchiveError(f"{label} must contain a JSON object: {path}")
    if value.get("schema_version") != schema_version:
        raise TrainerArchiveError(f"{label} schema_version must be {schema_version!r}; got {value.get('schema_version')!r}")
    return value


def _read_json_optional(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _resolve_recorded_path(value: Any, base_dir: Path) -> tuple[Path | None, str | None]:
    if not isinstance(value, str) or not value or value.startswith("<redacted:") or value.startswith("<missing-"):
        return None, "path is redacted or unavailable"
    raw = Path(value)
    if ".." in raw.parts:
        return None, "parent traversal is not allowed in recorded paths"
    candidates = [raw] if raw.is_absolute() or _is_windows_absolute(value) else _unique_paths([Path.cwd() / raw, base_dir / raw])
    copy_errors: list[str] = []
    for candidate in candidates:
        source_error = _copyable_path_error(candidate)
        if source_error is None:
            return candidate, None
        if candidate.exists() or candidate.is_symlink():
            copy_errors.append(source_error)
    if copy_errors:
        return None, "; ".join(copy_errors)
    return None, f"path could not be resolved: {value}"


def _copyable_path_error(path: Path) -> str | None:
    if path.is_symlink():
        return f"path is a symlink: {path}"
    if path.is_file():
        return _copyable_file_error(path)
    if path.is_dir():
        return _copyable_directory_error(path)
    return f"path not found: {path}"


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


def _copyable_directory_error(path: Path) -> str | None:
    if path.is_symlink():
        return f"directory is a symlink: {path}"
    if not path.exists() or not path.is_dir():
        return f"directory not found: {path}"
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        return f"directory could not be resolved: {path}: {exc}"
    if not resolved.is_dir():
        return f"path is not a regular directory: {path}"
    for child in path.rglob("*"):
        if child.is_symlink():
            return f"directory contains a symlink: {child}"
        if not child.is_file() and not child.is_dir():
            return f"directory contains a non-regular path: {child}"
    return None


def _copy_regular_tree(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for child in sorted(source.rglob("*")):
        relative = child.relative_to(source)
        destination = target / relative
        if child.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif child.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, destination)
        else:
            raise TrainerArchiveError(f"cannot archive non-regular path: {child}")


def _tree_fingerprint(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    file_count = 0
    size_bytes = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        file_hash = _sha256(path)
        size = path.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\0")
        file_count += 1
        size_bytes += size
    return {"sha256": digest.hexdigest(), "file_count": file_count, "size_bytes": size_bytes}


def _record_name(role: str, item: dict[str, Any], index: int) -> str:
    for key in ("id", "schema_name", "path"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value if key != "path" else f"{role}_{Path(value).name or index}"
    return f"{role}_{index}"


def _missing(role: str, index: int, reason: str, *, name: str | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {"role": role, "index": index, "reason": reason}
    if name:
        item["name"] = name
    return item


def _approved_command_record(launch_check: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(launch_check, dict):
        return {"approved": False, "provided": False, "raw": "", "argv": [], "parseable": False, "shell": ""}
    command = launch_check.get("approved_command")
    if not isinstance(command, dict):
        return {"approved": False, "provided": False, "raw": "", "argv": [], "parseable": False, "shell": ""}
    argv = command.get("argv") if isinstance(command.get("argv"), list) else []
    clean_argv = [str(item) for item in argv if isinstance(item, str)]
    raw = command.get("raw") if isinstance(command.get("raw"), str) else ""
    shell = command.get("shell") if isinstance(command.get("shell"), str) else ""
    return {
        "approved": command.get("approved") is True,
        "provided": command.get("provided") is True,
        "raw": raw,
        "argv": clean_argv,
        "parseable": command.get("parseable") is True,
        "shell": shell if shell else (shlex.join(clean_argv) if clean_argv else raw),
    }


def _trainer_inputs(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    inputs: list[dict[str, Any]] = []
    for artifact in artifacts:
        if artifact.get("role") != "trainer_artifact":
            continue
        record: dict[str, Any] = {
            "artifact_index": artifact.get("index"),
            "artifact_name": artifact.get("name"),
            "kind": artifact.get("kind"),
            "original_path": artifact.get("original_path"),
            "archive_path": artifact.get("path"),
            "size_bytes": artifact.get("size_bytes"),
            "sha256": artifact.get("sha256"),
        }
        if artifact.get("kind") == "directory":
            record["file_count"] = artifact.get("file_count")
            record["tree_hash_algorithm"] = artifact.get("tree_hash_algorithm")
        inputs.append(record)
    return inputs


def _path_rewrites(trainer_inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rewrites: list[dict[str, Any]] = []
    for item in trainer_inputs:
        original = item.get("original_path")
        archive_path = item.get("archive_path")
        if not isinstance(original, str) or not original or original.startswith("<"):
            continue
        if not isinstance(archive_path, str) or not archive_path:
            continue
        rewrites.append(
            {
                "artifact_name": str(item.get("artifact_name") or ""),
                "kind": str(item.get("kind") or ""),
                "original_path": original,
                "archive_path": archive_path,
            }
        )
    return rewrites


def _portable_command_record(approved_command: dict[str, Any], path_rewrites: list[dict[str, Any]]) -> dict[str, Any]:
    argv = approved_command.get("argv") if isinstance(approved_command.get("argv"), list) else []
    clean_argv = [item for item in argv if isinstance(item, str)]
    rewritten_argv, rewrite_count = _rewrite_command_argv(clean_argv, path_rewrites)
    return {
        "approved": approved_command.get("approved") is True,
        "available": bool(rewritten_argv),
        "rewritten": rewrite_count > 0,
        "path_rewrite_count": rewrite_count,
        "argv": rewritten_argv,
        "shell": shlex.join(rewritten_argv) if rewritten_argv else "",
        "notes": [
            "Archive-local command is advisory and rewrites only recognized trainer-input paths.",
            "Run it from the trainer archive root or resolve archive_path entries explicitly in your launcher.",
        ],
    }


def _rewrite_command_argv(argv: list[str], path_rewrites: list[dict[str, Any]]) -> tuple[list[str], int]:
    rewritten: list[str] = []
    rewrite_count = 0
    ordered = sorted(path_rewrites, key=lambda item: len(str(item.get("original_path") or "")), reverse=True)
    for token in argv:
        new_token = _rewrite_command_token(token, ordered)
        if new_token != token:
            rewrite_count += 1
        rewritten.append(new_token)
    return rewritten, rewrite_count


def _rewrite_command_token(token: str, path_rewrites: list[dict[str, Any]]) -> str:
    for item in path_rewrites:
        original = item.get("original_path")
        archive_path = item.get("archive_path")
        if not isinstance(original, str) or not original:
            continue
        if not isinstance(archive_path, str) or not archive_path:
            continue
        replacement = _replace_path_value(token, original, archive_path)
        if replacement != token:
            return replacement
        if "=" in token:
            key, value = token.split("=", 1)
            rewritten_value = _replace_path_value(value, original, archive_path)
            if rewritten_value != value:
                return f"{key}={rewritten_value}"
    return token


def _replace_path_value(value: str, original: str, archive_path: str) -> str:
    if value == original:
        return archive_path
    prefix = original.rstrip("/") + "/"
    if value.startswith(prefix):
        return archive_path.rstrip("/") + "/" + value[len(prefix) :]
    return value


def _metrics(
    artifacts: list[dict[str, Any]],
    missing: list[dict[str, Any]],
    trainer_inputs: list[dict[str, Any]],
    path_rewrites: list[dict[str, Any]],
) -> dict[str, Any]:
    role_counts = _count_rows(record.get("role") for record in artifacts)
    missing_role_counts = _count_rows(record.get("role") for record in missing)
    return {
        "artifact_count": len(artifacts),
        "file_artifact_count": sum(1 for record in artifacts if record.get("kind") == "file"),
        "directory_artifact_count": sum(1 for record in artifacts if record.get("kind") == "directory"),
        "trainer_input_count": len(trainer_inputs),
        "path_rewrite_count": len(path_rewrites),
        "missing_count": len(missing),
        "total_size_bytes": sum(_int_value(record.get("size_bytes")) for record in artifacts),
        "role_counts": role_counts,
        "missing_role_counts": missing_role_counts,
        "unique_sha256_count": len({record.get("sha256") for record in artifacts if isinstance(record.get("sha256"), str)}),
    }


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


def _suffix_for(path: Path) -> str:
    suffix = path.suffix
    return suffix if suffix else ".artifact"


def _slug(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in value)
    collapsed = "_".join(part for part in cleaned.split("_") if part)
    return collapsed[:120] or "artifact"


def _int_value(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _is_windows_absolute(value: str) -> bool:
    normalized = value.replace("/", "\\")
    return (len(normalized) >= 3 and normalized[1:3] == ":\\" and normalized[0].isalpha()) or normalized.startswith("\\\\")


def _basename(value: str) -> str:
    return value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] or "path"
