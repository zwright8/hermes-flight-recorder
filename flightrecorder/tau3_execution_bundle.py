"""Assemble private Tau-3 training/evaluation execution bundles.

The bundle is intentionally private and schema-less: it is the durable local
evidence package consumed by :mod:`flightrecorder.tau3_execution_validation`.
All copied artifacts are output-relative, hash-addressed, and copied into a
fresh directory so later validators can replay evidence without depending on
the original run paths.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence

from .path_safety import path_has_symlink_component

EXECUTION_BUNDLE_SCHEMA_VERSION = "hfr.tau3_execution_bundle.v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class Tau3ExecutionBundleError(ValueError):
    """Raised when a Tau-3 execution bundle cannot be assembled safely."""


@dataclass(frozen=True)
class CandidateInput:
    candidate_id: str
    directory: Path


@dataclass(frozen=True)
class ArmInput:
    arm_id: str
    directory: Path


def build_tau3_execution_bundle(
    *,
    out_dir: str | Path,
    flight_recorder_git_commit: str,
    tracked_worktree_clean: bool,
    protocol: str | Path,
    selected_candidate_id: str,
    candidate_dirs: Sequence[CandidateInput],
    candidate_selection_report: str | Path,
    candidate_lock: str | Path,
    development_arm_dirs: Sequence[ArmInput],
    sealed_arm_dirs: Sequence[ArmInput],
    public_report: str | Path,
    expected_source_hashes: Mapping[str, str] | None = None,
    make_read_only: bool = True,
) -> dict[str, Any]:
    """Copy portable Tau-3 evidence into a fresh execution bundle directory."""

    if not GIT_COMMIT_RE.fullmatch(flight_recorder_git_commit):
        raise Tau3ExecutionBundleError("flight_recorder_git_commit must be a clean 40-character lowercase hex revision")
    if tracked_worktree_clean is not True:
        raise Tau3ExecutionBundleError("tracked_worktree_clean must be explicitly true")
    if not _safe_id(selected_candidate_id):
        raise Tau3ExecutionBundleError("selected_candidate_id must be a safe portable identifier")

    out = Path(out_dir)
    _require_fresh_output(out)
    expected = dict(expected_source_hashes or {})
    _check_expected_hash("protocol", Path(protocol), expected)
    _check_expected_hash("candidate_selection_report", Path(candidate_selection_report), expected)
    _check_expected_hash("candidate_lock", Path(candidate_lock), expected)
    _check_expected_hash("public_report", Path(public_report), expected)

    candidate_ids = [candidate.candidate_id for candidate in candidate_dirs]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise Tau3ExecutionBundleError("duplicate candidate id")
    if selected_candidate_id not in set(candidate_ids):
        raise Tau3ExecutionBundleError("selected candidate is not present in candidate_dirs")
    dev_arms = [arm.arm_id for arm in development_arm_dirs]
    sealed_arms = [arm.arm_id for arm in sealed_arm_dirs]
    if len(dev_arms) != len(set(dev_arms)):
        raise Tau3ExecutionBundleError("duplicate development arm id")
    if len(sealed_arms) != len(set(sealed_arms)):
        raise Tau3ExecutionBundleError("duplicate sealed arm id")

    out.mkdir(mode=0o700)
    try:
        protocol_ref = _copy_file(Path(protocol), out / "protocol.json", out, expected_sha=expected.get("protocol"))
        training_dir = out / "training"
        candidate_receipts: list[dict[str, Any]] = []
        selected_receipt: dict[str, Any] | None = None
        for candidate in candidate_dirs:
            if not _safe_id(candidate.candidate_id):
                raise Tau3ExecutionBundleError(f"candidate id is unsafe: {candidate.candidate_id!r}")
            label = f"candidate:{candidate.candidate_id}"
            _check_expected_hash(label, candidate.directory, expected)
            copied = _copy_dir(candidate.directory, training_dir / candidate.candidate_id, expected_sha=expected.get(label))
            receipt_path = copied / "training_receipt.json"
            if not receipt_path.is_file():
                raise Tau3ExecutionBundleError(f"candidate {candidate.candidate_id!r} is missing training_receipt.json")
            receipt_ref = _file_ref(receipt_path, out)
            receipt_ref["candidate_id"] = candidate.candidate_id
            candidate_receipts.append(receipt_ref)
            if candidate.candidate_id == selected_candidate_id:
                selected_receipt = _file_ref(receipt_path, out)
        if selected_receipt is None:
            raise Tau3ExecutionBundleError("selected receipt was not copied")

        selection_ref = _copy_file(
            Path(candidate_selection_report),
            out / "candidate-selection-report.json",
            out,
            expected_sha=expected.get("candidate_selection_report"),
        )
        lock_ref = _copy_file(Path(candidate_lock), out / "candidate-lock.json", out, expected_sha=expected.get("candidate_lock"))

        development_refs = _copy_arms(development_arm_dirs, out / "benchmark" / "development", out, expected, "development")
        sealed_refs = _copy_arms(sealed_arm_dirs, out / "benchmark" / "sealed", out, expected, "sealed")
        report_ref = _copy_file(
            Path(public_report),
            out / "public-evaluation-report.json",
            out,
            expected_sha=expected.get("public_report"),
        )

        manifest = {
            "schema_version": EXECUTION_BUNDLE_SCHEMA_VERSION,
            "code_revision": {
                "flight_recorder_git_commit": flight_recorder_git_commit,
                "tracked_worktree_clean": True,
            },
            "protocol": protocol_ref,
            "training": {
                "selected_candidate_id": selected_candidate_id,
                "selected_receipt": selected_receipt,
                "candidate_receipts": candidate_receipts,
                "candidate_selection_report": selection_ref,
                "candidate_locks": [lock_ref],
            },
            "benchmark": {
                "development_arms": development_refs,
                "sealed_arms": sealed_refs,
                "public_report": report_ref,
            },
        }
        _reject_private_manifest_paths(manifest)
        _write_manifest(out / "manifest.json", manifest)
        _verify_manifest_refs(out, manifest)
        if make_read_only:
            _make_tree_read_only(out)
        return manifest
    except Exception:
        if out.exists():
            shutil.rmtree(out)
        raise


def parse_candidate_arg(value: str) -> CandidateInput:
    key, path = _split_binding(value, "candidate")
    return CandidateInput(candidate_id=key, directory=Path(path))


def parse_arm_arg(value: str) -> ArmInput:
    key, path = _split_binding(value, "arm")
    return ArmInput(arm_id=key, directory=Path(path))


def parse_expected_source_hash(value: str) -> tuple[str, str]:
    key, digest = _split_binding(value, "expected source hash")
    if not SHA256_RE.fullmatch(digest):
        raise Tau3ExecutionBundleError(f"expected source hash for {key!r} must be SHA-256")
    return key, digest


def _copy_arms(
    arms: Sequence[ArmInput],
    dest_root: Path,
    bundle_root: Path,
    expected: Mapping[str, str],
    mode: str,
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for arm in arms:
        if not _safe_id(arm.arm_id):
            raise Tau3ExecutionBundleError(f"{mode} arm id is unsafe: {arm.arm_id!r}")
        label = f"{mode}:{arm.arm_id}"
        _check_expected_hash(label, arm.directory, expected)
        copied = _copy_dir(arm.directory, dest_root / arm.arm_id, expected_sha=expected.get(label))
        manifest_path = copied / "manifest.json"
        if not manifest_path.is_file():
            raise Tau3ExecutionBundleError(f"{mode} arm {arm.arm_id!r} is missing manifest.json")
        ref = _file_ref(manifest_path, bundle_root)
        ref["arm_id"] = arm.arm_id
        refs.append(ref)
    return refs


def _copy_file(src: Path, dst: Path, root: Path, *, expected_sha: str | None = None) -> dict[str, Any]:
    source = _safe_source_file(src)
    if expected_sha is not None and _sha256_file(source) != expected_sha:
        raise Tau3ExecutionBundleError(f"source hash mismatch for {src}")
    _require_output_path(dst, root)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dst)
    if _sha256_file(source) != _sha256_file(dst):
        raise Tau3ExecutionBundleError(f"post-copy hash drift for {src}")
    return _file_ref(dst, root)


def _copy_dir(src: Path, dst: Path, *, expected_sha: str | None = None) -> Path:
    source = _safe_source_dir(src)
    if expected_sha is not None and _tree_sha256(source) != expected_sha:
        raise Tau3ExecutionBundleError(f"source tree hash mismatch for {src}")
    if dst.exists():
        raise Tau3ExecutionBundleError(f"copy destination already exists: {dst}")
    _require_no_symlinks(source)
    shutil.copytree(source, dst, symlinks=False, copy_function=shutil.copy2)
    if _tree_sha256(source) != _tree_sha256(dst):
        raise Tau3ExecutionBundleError(f"post-copy tree hash drift for {src}")
    return dst


def _safe_source_file(path: Path) -> Path:
    if path.is_symlink() or path_has_symlink_component(path, include_leaf=True):
        raise Tau3ExecutionBundleError(f"source file contains a symlink component: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise Tau3ExecutionBundleError(f"source is not a file: {path}")
    if path_has_symlink_component(resolved, include_leaf=True):
        raise Tau3ExecutionBundleError(f"source file contains a symlink component: {path}")
    return resolved


def _safe_source_dir(path: Path) -> Path:
    if path.is_symlink() or path_has_symlink_component(path, include_leaf=True):
        raise Tau3ExecutionBundleError(f"source directory contains a symlink component: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise Tau3ExecutionBundleError(f"source is not a directory: {path}")
    if path_has_symlink_component(resolved, include_leaf=True):
        raise Tau3ExecutionBundleError(f"source directory contains a symlink component: {path}")
    return resolved


def _require_no_symlinks(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink() or path_has_symlink_component(path, include_leaf=True):
            raise Tau3ExecutionBundleError(f"source tree contains a symlink component: {path}")


def _require_fresh_output(out: Path) -> None:
    if out.exists():
        if not out.is_dir():
            raise Tau3ExecutionBundleError(f"output exists and is not a directory: {out}")
        try:
            next(out.iterdir())
        except StopIteration:
            return
        raise Tau3ExecutionBundleError(f"output directory must be empty: {out}")
    parent = out.parent.resolve(strict=True) if out.parent.exists() else out.parent.resolve()
    if path_has_symlink_component(parent, include_leaf=True):
        raise Tau3ExecutionBundleError(f"output parent contains a symlink component: {out.parent}")


def _require_output_path(path: Path, root: Path) -> None:
    resolved_root = root.resolve()
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise Tau3ExecutionBundleError(f"output path escapes bundle root: {path}") from exc


def _file_ref(path: Path, root: Path) -> dict[str, Any]:
    rel = _relative_path(path, root)
    return {"path": rel, "sha256": _sha256_file(path), "size": path.stat().st_size}


def _relative_path(path: Path, root: Path) -> str:
    try:
        rel = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise Tau3ExecutionBundleError(f"path is outside bundle root: {path}") from exc
    if _unsafe_relative(rel):
        raise Tau3ExecutionBundleError(f"unsafe output-relative path: {rel}")
    return rel


def _verify_manifest_refs(root: Path, manifest: dict[str, Any]) -> None:
    refs = [
        manifest["protocol"],
        manifest["training"]["selected_receipt"],
        manifest["training"]["candidate_selection_report"],
        *manifest["training"]["candidate_locks"],
        *manifest["training"]["candidate_receipts"],
        *manifest["benchmark"]["development_arms"],
        *manifest["benchmark"]["sealed_arms"],
        manifest["benchmark"]["public_report"],
    ]
    for ref in refs:
        path = root / ref["path"]
        if not path.is_file():
            raise Tau3ExecutionBundleError(f"manifest ref missing after copy: {ref['path']}")
        if _sha256_file(path) != ref["sha256"]:
            raise Tau3ExecutionBundleError(f"manifest ref hash drift after copy: {ref['path']}")


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _make_tree_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_file():
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        elif path.is_dir():
            path.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    root.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)


def _check_expected_hash(label: str, path: Path, expected: Mapping[str, str]) -> None:
    digest = expected.get(label)
    if digest is None:
        return
    if not SHA256_RE.fullmatch(digest):
        raise Tau3ExecutionBundleError(f"expected source hash for {label!r} must be SHA-256")
    actual = _tree_sha256(path) if path.is_dir() else _sha256_file(path)
    if actual != digest:
        raise Tau3ExecutionBundleError(f"source hash mismatch for {label!r}")


def _tree_sha256(root: Path) -> str:
    source = root.resolve(strict=True)
    records = []
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        if path.is_symlink() or path_has_symlink_component(path, include_leaf=True):
            raise Tau3ExecutionBundleError(f"source tree contains a symlink component: {path}")
        rel = path.relative_to(source).as_posix()
        records.append({"path": rel, "size": path.stat().st_size, "sha256": _sha256_file(path)})
    digest = hashlib.sha256()
    for record in records:
        digest.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_id(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value))


def _unsafe_relative(value: str) -> bool:
    pure = PurePosixPath(value)
    win = PureWindowsPath(value)
    return (
        not value
        or pure.is_absolute()
        or win.is_absolute()
        or "\\" in value
        or any(part in {"", ".", ".."} for part in pure.parts)
    )


def _split_binding(value: str, label: str) -> tuple[str, str]:
    if "=" not in value:
        raise Tau3ExecutionBundleError(f"{label} must use id=path")
    key, path = value.split("=", 1)
    if not key or not path:
        raise Tau3ExecutionBundleError(f"{label} must use non-empty id=path")
    return key, path


def _reject_private_manifest_paths(value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _reject_private_manifest_paths(item)
    elif isinstance(value, list):
        for item in value:
            _reject_private_manifest_paths(item)
    elif isinstance(value, str) and (value.startswith("/") or "://" in value or "\\" in value):
        raise Tau3ExecutionBundleError("execution bundle manifest contains a private or absolute path")
