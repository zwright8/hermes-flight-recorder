"""Assemble qualified Tau-3 competitive-v3 training evidence by reference."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .schema_registry import SchemaRegistryError, check_schema_contract
from .tau3_competitive_v3 import (
    TRAINING_SCHEMA_VERSION,
    _Target,
    _validate_training_evidence,
)


CANDIDATE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
FORBIDDEN_REF_SEGMENTS = {
    "final",
    "logs",
    "private",
    "publication",
    "raw",
    "raw-logs",
    "raw_logs",
    "sealed",
}
OUTPUT_NAME = "training-evidence.json"


class Tau3CompetitiveV3TrainingEvidenceError(ValueError):
    """Raised when qualified training evidence cannot be assembled safely."""


def build_tau3_competitive_v3_training_evidence(
    bundle: str | Path,
    *,
    candidate_ids: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Validate conventional candidate artifacts and write one reference wrapper."""

    root = Path(bundle).resolve()
    if not root.is_dir():
        raise Tau3CompetitiveV3TrainingEvidenceError(
            f"bundle directory is missing: {root}"
        )
    output = root / OUTPUT_NAME
    if output.exists():
        raise Tau3CompetitiveV3TrainingEvidenceError(
            f"refusing to overwrite existing training evidence: {output}"
        )
    identifiers = list(candidate_ids)
    if len(identifiers) < 2:
        raise Tau3CompetitiveV3TrainingEvidenceError(
            "at least two --candidate identifiers are required"
        )
    if len(set(identifiers)) != len(identifiers):
        raise Tau3CompetitiveV3TrainingEvidenceError(
            "candidate identifiers must be distinct"
        )
    for candidate_id in identifiers:
        if CANDIDATE_ID_RE.fullmatch(candidate_id) is None:
            raise Tau3CompetitiveV3TrainingEvidenceError(
                "candidate identifiers must be safe lowercase identifiers"
            )

    shared = {
        "train": _require_file(
            root,
            root / "dataset" / "train.jsonl",
            "training dataset",
        ),
        "valid": _require_file(
            root,
            root / "dataset" / "valid.jsonl",
            "validation dataset",
        ),
        "protocol": _require_file(
            root,
            root / "evidence" / "protocol.json",
            "protocol",
        ),
        "identity": _require_file(
            root,
            root / "evidence" / "base-model-identity.json",
            "base model identity",
        ),
    }
    candidates = [
        _candidate_evidence(root, candidate_id, shared)
        for candidate_id in identifiers
    ]
    evidence = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "budgets": {"separate_candidate_and_infra_budgets": True},
        "qualified_candidates": candidates,
    }
    _replay_evidence(root, evidence, output)

    if output.exists():
        raise Tau3CompetitiveV3TrainingEvidenceError(
            f"refusing to overwrite existing training evidence: {output}"
        )
    payload = (
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary: Path | None = None
    linked = False
    linked_identity: tuple[int, int] | None = None
    try:
        descriptor, temporary_text = tempfile.mkstemp(
            prefix=".training-evidence.",
            suffix=".tmp",
            dir=root,
        )
        temporary = Path(temporary_text)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_stat = os.lstat(temporary)
        linked_identity = (temporary_stat.st_dev, temporary_stat.st_ino)
        try:
            os.link(temporary, output)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                raise Tau3CompetitiveV3TrainingEvidenceError(
                    f"refusing to overwrite existing training evidence: {output}"
                ) from exc
            raise
        linked = True
        try:
            output_stat = os.lstat(output)
        except OSError as exc:
            raise Tau3CompetitiveV3TrainingEvidenceError(
                "training evidence output disappeared during atomic publication"
            ) from exc
        if (output_stat.st_dev, output_stat.st_ino) != linked_identity:
            raise Tau3CompetitiveV3TrainingEvidenceError(
                "training evidence output was replaced during atomic publication"
            )
        persisted = _read_json(output, "written training evidence")
        _replay_evidence(root, persisted, output)
    except Exception as original:
        if linked and linked_identity is not None:
            try:
                current = os.lstat(output)
                if (current.st_dev, current.st_ino) == linked_identity:
                    output.unlink()
            except FileNotFoundError:
                pass
            except OSError as cleanup_error:
                raise Tau3CompetitiveV3TrainingEvidenceError(
                    "training evidence publication failed and owned-output "
                    f"cleanup also failed: {cleanup_error}"
                ) from original
        raise
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    return {
        "schema_version": "hfr.tau3_competitive_v3_training_evidence_build.v1",
        "passed": True,
        "candidate_ids": identifiers,
        "candidate_count": len(identifiers),
        "path": str(output),
        "sha256": _sha256_file(output),
    }


def _candidate_evidence(
    root: Path,
    candidate_id: str,
    shared: dict[str, Path],
) -> dict[str, Any]:
    candidate_root = root / "training" / "candidates" / candidate_id
    _require_directory(root, candidate_root, f"{candidate_id} directory")
    for directory in (
        candidate_root / "run",
        candidate_root / "exposure",
        candidate_root / "internal-validation",
        candidate_root / "development",
        candidate_root / "behavior",
    ):
        _require_safe_tree(root, directory)

    receipt_path = _require_file(
        root,
        candidate_root / "run" / "training_receipt.json",
        f"{candidate_id} training receipt",
    )
    receipt = _read_json(receipt_path, f"{candidate_id} training receipt")
    _reject_unsafe_refs(receipt, f"{candidate_id} training receipt")
    recipe = _dict(_dict(receipt.get("training_binding")).get("recipe"))
    exposure_objective = _dict(
        _dict(_dict(receipt.get("training_binding")).get("exposure")).get(
            "objective"
        )
    )
    full_gradient = (
        recipe.get("full_gradient_objective") is True
        and exposure_objective.get("full_gradient") is True
        and exposure_objective.get("detached_prefix") is not True
    )
    detached_prefix = (
        recipe.get("full_gradient_objective") is False
        and recipe.get("prefix_cache_training") is True
        and recipe.get("prefix_equivalence_required") is True
        and recipe.get("prefix_equivalence_passed") is True
        and exposure_objective.get("full_gradient") is False
        and exposure_objective.get("detached_prefix") is True
    )
    if not (full_gradient or detached_prefix):
        raise Tau3CompetitiveV3TrainingEvidenceError(
            f"{candidate_id} training receipt has no qualified objective"
        )

    prefix_path = candidate_root / "prefix-equivalence.json"
    if full_gradient and os.path.lexists(prefix_path):
        raise Tau3CompetitiveV3TrainingEvidenceError(
            f"{candidate_id} full-gradient candidate must not include prefix equivalence"
        )
    if detached_prefix:
        prefix_path = _require_file(
            root,
            prefix_path,
            f"{candidate_id} prefix equivalence",
        )
        _reject_unsafe_refs(
            _read_json(prefix_path, f"{candidate_id} prefix equivalence"),
            f"{candidate_id} prefix equivalence",
        )

    paths = {
        "exposure_receipt": _require_file(
            root,
            candidate_root / "exposure" / "training_exposure_receipt.json",
            f"{candidate_id} exposure receipt",
        ),
        "exposure_ledger": _require_file(
            root,
            candidate_root / "exposure" / "training_exposure_ledger.jsonl",
            f"{candidate_id} exposure ledger",
        ),
        "exposure_validation": _require_file(
            root,
            candidate_root / "exposure" / "validation.json",
            f"{candidate_id} exposure validation",
        ),
        "internal": _require_file(
            root,
            candidate_root / "internal-validation" / "internal-validation.json",
            f"{candidate_id} internal validation",
        ),
        "scorecard": _require_file(
            root,
            candidate_root / "development" / "development-scorecard.json",
            f"{candidate_id} development scorecard",
        ),
        "probes": _require_file(
            root,
            candidate_root / "behavior" / "behavior-probes.json",
            f"{candidate_id} behavior probes",
        ),
    }
    for label, path in paths.items():
        if path.suffix == ".json":
            _reject_unsafe_refs(
                _read_json(path, f"{candidate_id} {label}"),
                f"{candidate_id} {label}",
            )

    result: dict[str, Any] = {
        "candidate_id": candidate_id,
        "exposure": {
            "dataset": _ref(root, shared["train"]),
            "receipt": _ref(root, paths["exposure_receipt"]),
            "ledger": _ref(root, paths["exposure_ledger"]),
            "validation": _ref(root, paths["exposure_validation"]),
        },
        "training_receipt": _ref(root, receipt_path),
        "internal_validation": {
            "artifact": _ref(root, paths["internal"]),
            "dataset": _ref(root, shared["valid"]),
            "protocol": _ref(root, shared["protocol"]),
            "model_identity": _ref(root, shared["identity"]),
        },
        "development_scorecard": _ref(root, paths["scorecard"]),
        "behavior_probes": _ref(root, paths["probes"]),
    }
    if detached_prefix:
        result["prefix_equivalence"] = _ref(root, prefix_path)
    return result


def _replay_evidence(root: Path, evidence: dict[str, Any], path: Path) -> None:
    try:
        schema = check_schema_contract(
            evidence,
            name_or_id="tau3_competitive_v3_training_evidence",
        )
    except SchemaRegistryError as exc:
        raise Tau3CompetitiveV3TrainingEvidenceError(
            f"training evidence schema is unavailable: {exc}"
        ) from exc
    if schema.get("passed") is not True:
        raise Tau3CompetitiveV3TrainingEvidenceError(
            f"training evidence schema check failed: {schema.get('errors')}"
        )
    target = _Target("competitive_v3_training", str(path))
    try:
        _validate_training_evidence(root, target, evidence)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise Tau3CompetitiveV3TrainingEvidenceError(
            f"training evidence replay failed: {exc}"
        ) from exc
    if target.errors:
        raise Tau3CompetitiveV3TrainingEvidenceError(
            "training evidence replay failed: " + "; ".join(target.errors)
        )


def _require_safe_tree(root: Path, directory: Path) -> None:
    _require_directory(root, directory, f"candidate evidence directory {directory.name}")
    for current, directories, files in os.walk(directory, followlinks=False):
        current_path = Path(current)
        for name in [*directories, *files]:
            path = current_path / name
            if path.is_symlink():
                raise Tau3CompetitiveV3TrainingEvidenceError(
                    f"candidate evidence must not contain symlinks: {path}"
                )
            _require_inside(root, path.resolve(), "candidate evidence")


def _require_directory(root: Path, path: Path, label: str) -> Path:
    _require_no_symlink(root, path, label)
    if not path.is_dir():
        raise Tau3CompetitiveV3TrainingEvidenceError(
            f"{label} directory is missing: {path}"
        )
    _require_inside(root, path.resolve(), label)
    return path


def _require_file(root: Path, path: Path, label: str) -> Path:
    _require_no_symlink(root, path, label)
    if not path.is_file():
        raise Tau3CompetitiveV3TrainingEvidenceError(
            f"{label} file is missing: {path}"
        )
    _require_inside(root, path.resolve(), label)
    return path


def _require_no_symlink(root: Path, path: Path, label: str) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise Tau3CompetitiveV3TrainingEvidenceError(
            f"{label} must resolve inside the bundle"
        ) from exc
    current = root
    if current.is_symlink():
        raise Tau3CompetitiveV3TrainingEvidenceError(
            f"{label} must not traverse symlinks"
        )
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise Tau3CompetitiveV3TrainingEvidenceError(
                f"{label} must not traverse symlinks: {current}"
            )


def _require_inside(root: Path, path: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise Tau3CompetitiveV3TrainingEvidenceError(
            f"{label} must resolve inside the bundle"
        ) from exc


def _reject_unsafe_refs(value: Any, label: str) -> None:
    if isinstance(value, dict):
        path = value.get("path")
        if isinstance(path, str):
            windows = PureWindowsPath(path)
            posix = PurePosixPath(path)
            lowered = {part.lower() for part in posix.parts}
            if (
                not path
                or "\\" in path
                or posix.is_absolute()
                or windows.is_absolute()
                or bool(windows.drive)
                or ".." in posix.parts
                or "~" in posix.parts
                or lowered.intersection(FORBIDDEN_REF_SEGMENTS)
                or any(part.lower().endswith(".log") for part in posix.parts)
            ):
                raise Tau3CompetitiveV3TrainingEvidenceError(
                    f"{label} contains an unsafe/private/sealed/raw-log reference"
                )
        for child in value.values():
            _reject_unsafe_refs(child, label)
    elif isinstance(value, list):
        for child in value:
            _reject_unsafe_refs(child, label)


def _ref(root: Path, path: Path) -> dict[str, str]:
    relative = path.relative_to(root).as_posix()
    _reject_unsafe_refs({"path": relative}, "training evidence")
    return {"path": relative, "sha256": _sha256_file(path)}


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Tau3CompetitiveV3TrainingEvidenceError(
            f"{label} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise Tau3CompetitiveV3TrainingEvidenceError(
            f"{label} must be a JSON object"
        )
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
