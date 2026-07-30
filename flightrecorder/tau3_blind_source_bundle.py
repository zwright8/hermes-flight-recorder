"""Fail-closed validation for a hash-only fresh Tau-3 custody bundle."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from .path_safety import path_has_symlink_component
from .schema_registry import check_schema_contract

DOMAINS = ("airline", "retail", "telecom")
EXPECTED_DOMAIN_COUNTS = {"airline": 34, "retail": 33, "telecom": 33}
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


class Tau3BlindSourceBundleError(ValueError):
    """Raised when fresh blind-source evidence is incomplete or inconsistent."""


def validate_tau3_blind_source_bundle(
    *,
    sealed_source_manifest: str | Path,
    generator_validation: str | Path,
    fresh_contamination_replay: str | Path,
    generator_script: str | Path,
    tau_repo: str | Path,
    training_dataset: str | Path,
    development_source: str | Path,
    retired_sealed_source: str | Path,
    expected_source_revision: str,
    expected_generator_commit: str,
) -> dict[str, Any]:
    """Validate schemas, hashes, privacy, permissions, and cross-artifact bindings."""

    if HEX40_RE.fullmatch(expected_source_revision) is None:
        raise Tau3BlindSourceBundleError(
            "expected source revision must be lowercase 40-hex"
        )
    if HEX40_RE.fullmatch(expected_generator_commit) is None:
        raise Tau3BlindSourceBundleError(
            "expected generator commit must be lowercase 40-hex"
        )

    sealed_path = _regular_private_file(
        Path(sealed_source_manifest), "sealed source manifest"
    )
    validation_path = _regular_private_file(
        Path(generator_validation), "generator validation"
    )
    contamination_path = _regular_private_file(
        Path(fresh_contamination_replay), "fresh contamination replay"
    )
    script_path = _regular_file(Path(generator_script), "generator script")
    tau_repo_path = Path(tau_repo)
    _require_clean_revision(tau_repo_path, expected_source_revision)
    training_path = _regular_file(Path(training_dataset), "training dataset")
    development_path = _regular_file(Path(development_source), "development source")
    retired_path = _regular_file(
        Path(retired_sealed_source), "retired sealed source manifest"
    )
    if not os.access(script_path, os.X_OK):
        raise Tau3BlindSourceBundleError("generator script must be executable")
    _require_script_commit_binding(script_path, expected_generator_commit)

    sealed = _read_json(sealed_path, "sealed source manifest")
    validation = _read_json(validation_path, "generator validation")
    contamination = _read_json(contamination_path, "fresh contamination replay")
    _check_schema(sealed, "tau3_sealed_source_manifest", "sealed source manifest")
    _check_schema(
        validation,
        "tau3_blind_generator_validation",
        "generator validation",
    )
    _check_schema(
        contamination,
        "tau3_fresh_contamination_replay",
        "fresh contamination replay",
    )

    sealed_sha256 = _sha256(sealed_path)
    if (
        sealed.get("source_revision") != expected_source_revision
        or validation.get("source_revision") != expected_source_revision
    ):
        raise Tau3BlindSourceBundleError("fresh source revision binding mismatch")
    if (
        sealed.get("task_count") != 100
        or validation.get("task_count") != 100
        or sealed.get("domain_counts") != EXPECTED_DOMAIN_COUNTS
        or validation.get("domain_counts") != EXPECTED_DOMAIN_COUNTS
    ):
        raise Tau3BlindSourceBundleError(
            "fresh source must contain balanced 34/33/33 coverage"
        )
    if validation.get("sealed_source_manifest_sha256") != sealed_sha256:
        raise Tau3BlindSourceBundleError(
            "generator validation sealed source hash mismatch"
        )
    source = validation.get("generator_source")
    if not isinstance(source, dict):
        raise Tau3BlindSourceBundleError("generator validation lacks source binding")
    if source.get("commit_sha") != expected_generator_commit:
        raise Tau3BlindSourceBundleError("generator commit binding mismatch")
    if source.get("script_sha256") != _sha256(script_path):
        raise Tau3BlindSourceBundleError("generator script hash mismatch")

    entries = sealed.get("entries")
    if not isinstance(entries, list) or len(entries) != 100:
        raise Tau3BlindSourceBundleError(
            "sealed source manifest must contain exactly 100 entries"
        )
    replayed_counts = Counter(
        str(entry.get("domain")) for entry in entries if isinstance(entry, dict)
    )
    if {domain: replayed_counts[domain] for domain in DOMAINS} != (
        EXPECTED_DOMAIN_COUNTS
    ):
        raise Tau3BlindSourceBundleError(
            "sealed source domain counts do not replay from entries"
        )
    triples: set[tuple[str, str, str]] = set()
    task_hashes: set[str] = set()
    prompt_hashes: set[str] = set()
    task_id_hashes: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise Tau3BlindSourceBundleError(
                f"sealed source entry {index} is not an object"
            )
        triple = (
            str(entry.get("task_id_sha256") or ""),
            str(entry.get("prompt_sha256") or ""),
            str(entry.get("task_sha256") or ""),
        )
        if any(HEX64_RE.fullmatch(value) is None for value in triple):
            raise Tau3BlindSourceBundleError(
                f"sealed source entry {index} has an invalid hash"
            )
        if triple in triples:
            raise Tau3BlindSourceBundleError(
                "sealed source contains a duplicate hash triple"
            )
        triples.add(triple)
        task_id_hashes.add(triple[0])
        prompt_hashes.add(triple[1])
        task_hashes.add(triple[2])
    if min(len(task_id_hashes), len(prompt_hashes), len(task_hashes)) != 100:
        raise Tau3BlindSourceBundleError(
            "fresh task, prompt, and task-id hashes must each be unique"
        )

    golden = validation.get("golden_replay")
    if (
        not isinstance(golden, dict)
        or golden.get("passed") is not True
        or golden.get("replayed_task_count") != 100
        or golden.get("passed_task_count") != 100
        or golden.get("failed_task_count") != 0
        or golden.get("state_check_failure_count") != 0
    ):
        raise Tau3BlindSourceBundleError("generator golden replay is incomplete")
    if any(
        validation.get(key) is not True
        for key in (
            "passed",
            "schema_validation_passed",
            "task_hashes_unique",
            "prompt_hashes_unique",
            "hashes_only",
        )
    ):
        raise Tau3BlindSourceBundleError("generator validation did not pass")

    expected_contamination_hashes = {
        "training_dataset_sha256": _sha256(training_path),
        "development_source_sha256": _sha256(development_path),
        "retired_sealed_source_manifest_sha256": _sha256(retired_path),
        "fresh_sealed_source_manifest_sha256": sealed_sha256,
    }
    for key, expected in expected_contamination_hashes.items():
        if contamination.get(key) != expected:
            raise Tau3BlindSourceBundleError(
                f"fresh contamination {key} binding mismatch"
            )
    overlaps = contamination.get("overlaps")
    if not isinstance(overlaps, dict) or any(
        not isinstance(value, int) or isinstance(value, bool) or value != 0
        for value in overlaps.values()
    ):
        raise Tau3BlindSourceBundleError(
            "fresh contamination replay contains a nonzero overlap"
        )
    if contamination.get("passed") is not True:
        raise Tau3BlindSourceBundleError("fresh contamination replay did not pass")

    for label, payload in (
        ("sealed source manifest", sealed),
        ("generator validation", validation),
        ("fresh contamination replay", contamination),
    ):
        _require_hash_only(payload, label)

    result = {
        "schema_version": "hfr.tau3_blind_source_bundle_validation.v1",
        "passed": True,
        "source_revision": expected_source_revision,
        "generator_commit_sha": expected_generator_commit,
        "task_count": 100,
        "domain_counts": EXPECTED_DOMAIN_COUNTS,
        "sealed_source_manifest_sha256": sealed_sha256,
        "generator_validation_sha256": _sha256(validation_path),
        "fresh_contamination_replay_sha256": _sha256(contamination_path),
        "generator_script_sha256": _sha256(script_path),
        "hashes_only": True,
        "local_paths_included": False,
        "raw_payload_included": False,
    }
    _check_schema(
        result,
        "tau3_blind_source_bundle_validation",
        "blind source bundle validation",
    )
    return result


def _regular_file(path: Path, label: str) -> Path:
    if path_has_symlink_component(path, include_leaf=True):
        raise Tau3BlindSourceBundleError(f"{label} must not contain symlink components")
    if not path.is_file():
        raise Tau3BlindSourceBundleError(f"{label} is not a regular file")
    return path


def _require_clean_revision(repo: Path, expected_revision: str) -> None:
    if path_has_symlink_component(repo, include_leaf=True):
        raise Tau3BlindSourceBundleError(
            "Tau repository must not contain symlink components"
        )
    if not repo.is_dir():
        raise Tau3BlindSourceBundleError("Tau repository is not a directory")
    revision = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if revision.returncode != 0 or status.returncode != 0:
        raise Tau3BlindSourceBundleError("could not inspect Tau repository")
    if revision.stdout.strip() != expected_revision or status.stdout.strip():
        raise Tau3BlindSourceBundleError(
            "Tau repository is not the expected clean revision"
        )


def _require_script_commit_binding(script: Path, commit_sha: str) -> None:
    root_result = subprocess.run(
        ["git", "-C", str(script.parent), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if root_result.returncode != 0:
        raise Tau3BlindSourceBundleError(
            "generator script is not inside a Git repository"
        )
    root = Path(root_result.stdout.strip()).resolve(strict=True)
    try:
        relative = script.resolve(strict=True).relative_to(root).as_posix()
    except ValueError as exc:
        raise Tau3BlindSourceBundleError(
            "generator script escapes its Git repository"
        ) from exc
    resolved = subprocess.run(
        ["git", "-C", str(root), "rev-parse", f"{commit_sha}^{{commit}}"],
        check=False,
        capture_output=True,
        text=True,
    )
    committed = subprocess.run(
        ["git", "-C", str(root), "show", f"{commit_sha}:{relative}"],
        check=False,
        capture_output=True,
    )
    if (
        resolved.returncode != 0
        or resolved.stdout.strip() != commit_sha
        or committed.returncode != 0
        or hashlib.sha256(committed.stdout).hexdigest() != _sha256(script)
    ):
        raise Tau3BlindSourceBundleError(
            "generator commit does not contain the exact executable"
        )


def _regular_private_file(path: Path, label: str) -> Path:
    path = _regular_file(path, label)
    if path.stat().st_mode & 0o077:
        raise Tau3BlindSourceBundleError(f"{label} must not be group/world accessible")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Tau3BlindSourceBundleError(f"{label} is invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise Tau3BlindSourceBundleError(f"{label} must contain an object")
    return payload


def _check_schema(payload: dict[str, Any], schema: str, label: str) -> None:
    result = check_schema_contract(payload, name_or_id=schema)
    if result.get("passed") is not True:
        detail = "; ".join(str(item) for item in result.get("errors", []))
        raise Tau3BlindSourceBundleError(f"{label} schema validation failed: {detail}")


def _require_hash_only(payload: dict[str, Any], label: str) -> None:
    forbidden_keys = {
        "task",
        "tasks",
        "prompt",
        "prompts",
        "messages",
        "policy",
        "tool_calls",
        "tool_results",
        "local_path",
        "path",
    }

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key.lower() in forbidden_keys:
                    raise Tau3BlindSourceBundleError(
                        f"{label} contains forbidden raw field {key!r}"
                    )
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str):
            lowered = value.lower()
            if value.startswith("/") or value.startswith("~") or "file://" in lowered:
                raise Tau3BlindSourceBundleError(f"{label} contains a local path")

    if payload.get("hashes_only") is not True:
        raise Tau3BlindSourceBundleError(f"{label} is not marked hash-only")
    if payload.get("local_paths_included") not in (None, False):
        raise Tau3BlindSourceBundleError(f"{label} includes local paths")
    if payload.get("raw_payload_included") not in (None, False):
        raise Tau3BlindSourceBundleError(f"{label} includes raw payload")
    visit(payload)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
