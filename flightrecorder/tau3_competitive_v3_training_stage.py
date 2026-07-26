"""Fail-closed staging for completed Tau-3 v3 MLX training runs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .schema_registry import SchemaRegistryError, check_schema_contract
from .tau3_mlx_training import validate_tau3_process_segments


STAGING_SCHEMA_VERSION = "hfr.tau3_competitive_v3_training_run_stage.v1"
TRAINING_RECEIPT_SCHEMA_VERSION = "hfr.tau3_mlx_training_run.v1"
CANDIDATE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
SUPPORT_RECORDS = ("prelaunch_receipt", "mlx_lora_config", "telemetry")


class Tau3CompetitiveV3TrainingStageError(ValueError):
    """Raised when a completed training run cannot be staged safely."""


def stage_tau3_competitive_v3_training_run(
    bundle: str | Path,
    *,
    candidate_id: str,
    training_run: str | Path,
) -> dict[str, Any]:
    """Copy only governed receipt, telemetry, config, and adapter artifacts."""

    if CANDIDATE_ID_RE.fullmatch(candidate_id) is None:
        raise Tau3CompetitiveV3TrainingStageError(
            "candidate_id must be a safe lowercase identifier"
        )
    root = Path(bundle).resolve()
    if not root.is_dir():
        raise Tau3CompetitiveV3TrainingStageError(
            f"bundle directory is missing: {root}"
        )
    source = Path(training_run).resolve()
    if not source.is_dir():
        raise Tau3CompetitiveV3TrainingStageError(
            f"training run directory is missing: {source}"
        )
    receipt_path = source / "training_receipt.json"
    _require_inside(source, receipt_path.resolve(), "training receipt")
    receipt = _read_json(receipt_path, "training receipt")
    _validate_completed_receipt(source, receipt)

    destination = (
        root / "training" / "candidates" / candidate_id / "run"
    )
    _require_inside(root, destination, "training staging destination")
    if destination.exists():
        raise Tau3CompetitiveV3TrainingStageError(
            f"refusing to overwrite existing staged run: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{candidate_id}.run.",
            dir=destination.parent,
        )
    )
    try:
        for relative in _selected_source_directories(receipt):
            (temporary / relative).mkdir(parents=True, exist_ok=True)
        selected = _selected_source_files(source, receipt_path, receipt)
        records: list[dict[str, Any]] = []
        for kind, relative, source_path in selected:
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target)
            records.append(
                {
                    "kind": kind,
                    "path": relative.as_posix(),
                    "sha256": _sha256_file(target),
                    "size": target.stat().st_size,
                }
            )
        records.sort(key=lambda item: str(item["path"]))
        staged_tree_sha256 = _records_sha256(records)
        staging_receipt = {
            "schema_version": STAGING_SCHEMA_VERSION,
            "schema_checked": True,
            "passed": True,
            "candidate_id": candidate_id,
            "raw_logs_included": False,
            "source_training_receipt_sha256": _sha256_file(receipt_path),
            "staged_file_count": len(records),
            "staged_files": records,
            "staged_tree_sha256": staged_tree_sha256,
        }
        try:
            staging_schema_result = check_schema_contract(
                staging_receipt,
                name_or_id="tau3_competitive_v3_training_run_stage",
            )
        except SchemaRegistryError as exc:
            raise Tau3CompetitiveV3TrainingStageError(
                f"training staging schema is unavailable: {exc}"
            ) from exc
        if staging_schema_result.get("passed") is not True:
            raise Tau3CompetitiveV3TrainingStageError(
                "training staging receipt schema check failed: "
                f"{staging_schema_result.get('errors')}"
            )
        _write_json(temporary / "staging_receipt.json", staging_receipt)
        _freeze_staged_training_trees(temporary, receipt)
        _validate_completed_receipt(temporary, receipt)
        if destination.exists():
            raise Tau3CompetitiveV3TrainingStageError(
                f"refusing to overwrite existing staged run: {destination}"
            )
        os.replace(temporary, destination)
    except Exception:
        _make_owned_tree_writable(temporary)
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    training_receipt = destination / "training_receipt.json"
    staging_receipt_path = destination / "staging_receipt.json"
    return {
        **staging_receipt,
        "training_receipt": _bundle_ref(root, training_receipt),
        "staging_receipt": _bundle_ref(root, staging_receipt_path),
    }


def _validate_completed_receipt(
    source: Path,
    receipt: dict[str, Any],
) -> None:
    if receipt.get("schema_version") != TRAINING_RECEIPT_SCHEMA_VERSION:
        raise Tau3CompetitiveV3TrainingStageError(
            f"training receipt schema_version must be {TRAINING_RECEIPT_SCHEMA_VERSION}"
        )
    try:
        schema_result = check_schema_contract(
            receipt,
            name_or_id="tau3_mlx_training_run",
        )
    except SchemaRegistryError as exc:
        raise Tau3CompetitiveV3TrainingStageError(
            f"training receipt schema is unavailable: {exc}"
        ) from exc
    if schema_result.get("passed") is not True:
        raise Tau3CompetitiveV3TrainingStageError(
            f"training receipt schema check failed: {schema_result.get('errors')}"
        )
    if (
        receipt.get("phase") != "final"
        or receipt.get("terminal_status") != "success"
        or receipt.get("weights_updated") is not True
        or receipt.get("schema_checked") is not True
    ):
        raise Tau3CompetitiveV3TrainingStageError(
            "training receipt must be final, successful, schema-checked, and weight-updating"
        )
    checks = receipt.get("checks")
    if (
        not isinstance(checks, list)
        or not checks
        or any(
            not isinstance(check, dict) or check.get("passed") is not True
            for check in checks
        )
    ):
        raise Tau3CompetitiveV3TrainingStageError(
            "training receipt must contain only passing checks"
        )
    if not isinstance(receipt.get("training_binding"), dict):
        raise Tau3CompetitiveV3TrainingStageError(
            "training receipt must contain a governed training_binding"
        )
    support_paths = {
        field: _validate_file_record(source, receipt.get(field), field)
        for field in SUPPORT_RECORDS
    }
    _validate_prelaunch_receipt(support_paths["prelaunch_receipt"])
    _validate_telemetry(
        support_paths["telemetry"],
        receipt["telemetry"],
    )
    adapter_weight_count = _validate_adapter_tree(
        source,
        receipt.get("adapter"),
    )
    if receipt.get("adapter_weight_file_count") != adapter_weight_count:
        raise Tau3CompetitiveV3TrainingStageError(
            "adapter_weight_file_count does not replay"
        )
    _validate_process_segment_evidence(source, receipt)


def _validate_process_segment_evidence(
    source: Path,
    receipt: dict[str, Any],
) -> None:
    config = receipt.get("config")
    segmented = (
        isinstance(config, dict)
        and config.get("process_segment_iters") is not None
    )
    evidence = receipt.get("process_segments")
    if segmented != isinstance(evidence, dict):
        raise Tau3CompetitiveV3TrainingStageError(
            "process_segments evidence must be present exactly for segmented runs"
        )
    if not segmented:
        return
    assert isinstance(evidence, dict)
    validation = validate_tau3_process_segments(
        evidence,
        output_dir=source,
        expected_config=config,
    )
    if validation.get("passed") is not True:
        raise Tau3CompetitiveV3TrainingStageError(
            "process segment chain does not replay: "
            + json.dumps(validation.get("errors") or [], sort_keys=True)
        )


def _validate_file_record(source: Path, value: Any, label: str) -> Path:
    if not isinstance(value, dict):
        raise Tau3CompetitiveV3TrainingStageError(
            f"{label} must be a file record"
        )
    path = _resolve_source_ref(source, value.get("path"), f"{label}.path")
    if value.get("sha256") != _sha256_file(path):
        raise Tau3CompetitiveV3TrainingStageError(
            f"{label}.sha256 does not replay"
        )
    return path


def _validate_prelaunch_receipt(path: Path) -> None:
    receipt = _read_json(path, "prelaunch receipt")
    if receipt.get("schema_version") != TRAINING_RECEIPT_SCHEMA_VERSION:
        raise Tau3CompetitiveV3TrainingStageError(
            f"prelaunch receipt schema_version must be {TRAINING_RECEIPT_SCHEMA_VERSION}"
        )
    try:
        schema_result = check_schema_contract(
            receipt,
            name_or_id="tau3_mlx_training_run",
        )
    except SchemaRegistryError as exc:
        raise Tau3CompetitiveV3TrainingStageError(
            f"prelaunch receipt schema is unavailable: {exc}"
        ) from exc
    if schema_result.get("passed") is not True:
        raise Tau3CompetitiveV3TrainingStageError(
            f"prelaunch receipt schema check failed: {schema_result.get('errors')}"
        )
    checks = receipt.get("checks")
    if (
        receipt.get("phase") != "prelaunch"
        or receipt.get("terminal_status") != "prelaunch"
        or receipt.get("weights_updated") is not False
        or not isinstance(checks, list)
        or not checks
        or any(
            not isinstance(check, dict) or check.get("passed") is not True
            for check in checks
        )
    ):
        raise Tau3CompetitiveV3TrainingStageError(
            "prelaunch receipt must be prelaunch-only with passing checks"
        )


def _validate_telemetry(path: Path, record: dict[str, Any]) -> None:
    event_count = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                event_count += 1
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise Tau3CompetitiveV3TrainingStageError(
                        f"telemetry line {line_number} must be a JSON object"
                    )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Tau3CompetitiveV3TrainingStageError(
            f"telemetry must be valid JSONL: {exc}"
        ) from exc
    if record.get("event_count") != event_count:
        raise Tau3CompetitiveV3TrainingStageError(
            "telemetry.event_count does not replay"
        )


def _validate_adapter_tree(source: Path, value: Any) -> int:
    if not isinstance(value, dict):
        raise Tau3CompetitiveV3TrainingStageError(
            "adapter must be an artifact tree record"
        )
    adapter_root = _resolve_source_ref(
        source,
        value.get("path"),
        "adapter.path",
        require_file=False,
    )
    if not adapter_root.is_dir():
        raise Tau3CompetitiveV3TrainingStageError(
            "adapter.path must reference a directory"
        )
    files = value.get("files")
    if not isinstance(files, list) or not files:
        raise Tau3CompetitiveV3TrainingStageError(
            "adapter.files must be a non-empty list"
        )
    records: list[dict[str, Any]] = []
    weight_count = 0
    seen: set[str] = set()
    for index, record in enumerate(files):
        if not isinstance(record, dict):
            raise Tau3CompetitiveV3TrainingStageError(
                f"adapter.files[{index}] must be an object"
            )
        relative = _safe_relative_path(
            record.get("path"),
            f"adapter.files[{index}].path",
        )
        relative_text = relative.as_posix()
        if relative_text in seen:
            raise Tau3CompetitiveV3TrainingStageError(
                "adapter.files paths must be unique"
            )
        seen.add(relative_text)
        path = _resolve_source_ref(
            adapter_root,
            relative_text,
            f"adapter.files[{index}].path",
        )
        if (
            record.get("sha256") != _sha256_file(path)
            or record.get("size") != path.stat().st_size
        ):
            raise Tau3CompetitiveV3TrainingStageError(
                f"adapter.files[{index}] fingerprint does not replay"
            )
        if record.get("kind") == "adapter" and path.stat().st_size > 0:
            weight_count += 1
        records.append(dict(record))
    if value.get("file_count") != len(records):
        raise Tau3CompetitiveV3TrainingStageError(
            "adapter.file_count does not replay"
        )
    if weight_count < 1:
        raise Tau3CompetitiveV3TrainingStageError(
            "adapter tree must contain non-empty weight files"
        )
    if value.get("tree_sha256") != _records_sha256(records):
        raise Tau3CompetitiveV3TrainingStageError(
            "adapter.tree_sha256 does not replay"
        )
    return weight_count


def _selected_source_files(
    source: Path,
    receipt_path: Path,
    receipt: dict[str, Any],
) -> list[tuple[str, Path, Path]]:
    selected: list[tuple[str, Path, Path]] = [
        ("training_receipt", Path("training_receipt.json"), receipt_path)
    ]
    seen = {"training_receipt.json"}
    for field in SUPPORT_RECORDS:
        record = receipt[field]
        relative = _safe_relative_path(record["path"], f"{field}.path")
        if relative.as_posix() in seen:
            raise Tau3CompetitiveV3TrainingStageError(
                "staged source file paths must be unique"
            )
        seen.add(relative.as_posix())
        selected.append(
            (
                field,
                relative,
                _resolve_source_ref(
                    source,
                    relative.as_posix(),
                    field,
                ),
            )
        )
    adapter = receipt["adapter"]
    adapter_relative = _safe_relative_path(
        adapter["path"],
        "adapter.path",
    )
    adapter_root = _resolve_source_ref(
        source,
        adapter_relative.as_posix(),
        "adapter.path",
        require_file=False,
    )
    for index, record in enumerate(adapter["files"]):
        file_relative = _safe_relative_path(
            record["path"],
            f"adapter.files[{index}].path",
        )
        relative = adapter_relative / file_relative
        if relative.as_posix() in seen:
            raise Tau3CompetitiveV3TrainingStageError(
                "staged source file paths must be unique"
            )
        seen.add(relative.as_posix())
        selected.append(
            (
                str(record.get("kind") or "adapter_artifact"),
                relative,
                _resolve_source_ref(
                    adapter_root,
                    file_relative.as_posix(),
                    f"adapter.files[{index}].path",
                ),
            )
        )
    process_segments = receipt.get("process_segments")
    if isinstance(process_segments, dict):
        for kind, relative, source_path in _process_segment_source_files(
            source,
            process_segments,
        ):
            relative_text = relative.as_posix()
            if relative_text in seen:
                continue
            seen.add(relative_text)
            selected.append((kind, relative, source_path))
    return selected


def _process_segment_source_files(
    source: Path,
    evidence: dict[str, Any],
) -> list[tuple[str, Path, Path]]:
    selected: list[tuple[str, Path, Path]] = []
    for label in ("plan", "manifest"):
        record = evidence.get(label)
        if not isinstance(record, dict):
            raise Tau3CompetitiveV3TrainingStageError(
                f"process_segments.{label} must be a file record"
            )
        relative = _safe_relative_path(
            record.get("path"),
            f"process_segments.{label}.path",
        )
        selected.append(
            (
                f"process_segment_{label}",
                relative,
                _resolve_source_ref(source, relative.as_posix(), label),
            )
        )

    trees: list[tuple[str, dict[str, Any]]] = []
    artifact_tree = evidence.get("artifact_tree")
    if isinstance(artifact_tree, dict):
        trees.append(("artifact_tree", artifact_tree))
    else:
        raise Tau3CompetitiveV3TrainingStageError(
            "process_segments.artifact_tree must be an artifact tree record"
        )
    recovery = evidence.get("recovery")
    if not isinstance(recovery, dict):
        raise Tau3CompetitiveV3TrainingStageError(
            "process_segments.recovery must be an object"
        )
    partials = recovery.get("preserved_partial_artifact_trees")
    if not isinstance(partials, list):
        raise Tau3CompetitiveV3TrainingStageError(
            "process_segments recovery partial trees must be an array"
        )
    for index, tree in enumerate(partials):
        if not isinstance(tree, dict):
            raise Tau3CompetitiveV3TrainingStageError(
                f"process_segments recovery partial tree {index} must be an object"
            )
        trees.append((f"recovery.partial[{index}]", tree))
    failed = recovery.get("preserved_failed_artifact_tree")
    if failed is not None:
        if not isinstance(failed, dict):
            raise Tau3CompetitiveV3TrainingStageError(
                "process_segments recovery failed tree must be an object"
            )
        trees.append(("recovery.failed", failed))
    for label, tree in trees:
        selected.extend(
            _artifact_tree_source_files(
                source,
                tree,
                label=label,
            )
        )
    return selected


def _artifact_tree_source_files(
    source: Path,
    artifact_tree: dict[str, Any],
    *,
    label: str,
) -> list[tuple[str, Path, Path]]:
    selected: list[tuple[str, Path, Path]] = []
    root_relative = _safe_relative_path(
        artifact_tree.get("path"),
        f"process_segments.{label}.path",
    )
    artifact_root = _resolve_source_ref(
        source,
        root_relative.as_posix(),
        f"process_segments.{label}.path",
        require_file=False,
    )
    files = artifact_tree.get("files")
    if not isinstance(files, list):
        raise Tau3CompetitiveV3TrainingStageError(
            f"process_segments.{label}.files must be an array"
        )
    for index, record in enumerate(files):
        if not isinstance(record, dict):
            raise Tau3CompetitiveV3TrainingStageError(
                f"process_segments.{label}.files[{index}] must be an object"
            )
        file_relative = _safe_relative_path(
            record.get("path"),
            f"process_segments.{label}.files[{index}].path",
        )
        relative = root_relative / file_relative
        selected.append(
            (
                "process_segment_artifact",
                relative,
                _resolve_source_ref(
                    artifact_root,
                    file_relative.as_posix(),
                    f"process_segments.{label}.files[{index}].path",
                ),
            )
        )
    return selected


def _selected_source_directories(
    receipt: dict[str, Any],
) -> list[Path]:
    evidence = receipt.get("process_segments")
    if not isinstance(evidence, dict):
        return []
    records: list[Any] = [
        evidence.get("artifact_tree"),
        evidence.get("final_adapter"),
    ]
    recovery = evidence.get("recovery")
    if isinstance(recovery, dict):
        partials = recovery.get("preserved_partial_artifact_trees")
        if isinstance(partials, list):
            records.extend(partials)
        records.append(recovery.get("preserved_failed_artifact_tree"))
    directories: set[Path] = set()
    for index, record in enumerate(records):
        if record is None:
            continue
        if not isinstance(record, dict):
            raise Tau3CompetitiveV3TrainingStageError(
                f"process segment directory record {index} must be an object"
            )
        directories.add(
            _safe_relative_path(
                record.get("path"),
                f"process segment directory record {index}.path",
            )
        )
    return sorted(directories)


def _freeze_staged_training_trees(
    root: Path,
    receipt: dict[str, Any],
) -> None:
    adapter = receipt.get("adapter")
    if isinstance(adapter, dict):
        _make_tree_readonly(
            root
            / _safe_relative_path(
                adapter.get("path"),
                "adapter.path",
            )
        )
    if isinstance(receipt.get("process_segments"), dict):
        _make_tree_readonly(root / "process_segments")


def _make_tree_readonly(root: Path) -> None:
    if not root.is_dir():
        raise Tau3CompetitiveV3TrainingStageError(
            f"staged governed tree is missing: {root}"
        )
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            raise Tau3CompetitiveV3TrainingStageError(
                f"staged governed tree contains a symlink: {path}"
            )
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
        else:
            raise Tau3CompetitiveV3TrainingStageError(
                f"staged governed tree contains a non-regular node: {path}"
            )
    root.chmod(0o555)


def _make_owned_tree_writable(root: Path) -> None:
    if not root.exists() or root.is_symlink():
        return
    for path in sorted(root.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            path.chmod(0o755)
        elif path.is_file() and not path.is_symlink():
            path.chmod(0o644)
    root.chmod(0o755)


def _resolve_source_ref(
    root: Path,
    value: Any,
    label: str,
    *,
    require_file: bool = True,
) -> Path:
    relative = _safe_relative_path(value, label)
    path = (root / relative).resolve()
    _require_inside(root.resolve(), path, label)
    if require_file and not path.is_file():
        raise Tau3CompetitiveV3TrainingStageError(
            f"{label} does not reference a file"
        )
    return path


def _safe_relative_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise Tau3CompetitiveV3TrainingStageError(
            f"{label} must be a non-empty relative path"
        )
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        value.startswith(("/", "\\"))
        or posix.is_absolute()
        or windows.is_absolute()
        or any(part in ("", ".", "..") for part in posix.parts)
        or any(part == ".." for part in windows.parts)
    ):
        raise Tau3CompetitiveV3TrainingStageError(
            f"{label} must be a safe relative path"
        )
    return Path(*posix.parts)


def _require_inside(root: Path, path: Path, label: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise Tau3CompetitiveV3TrainingStageError(
            f"{label} escapes its allowed root"
        ) from exc


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise Tau3CompetitiveV3TrainingStageError(
            f"{label} is missing: {path}"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Tau3CompetitiveV3TrainingStageError(
            f"{label} must contain valid JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise Tau3CompetitiveV3TrainingStageError(
            f"{label} must be a JSON object"
        )
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _records_sha256(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        )
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_ref(root: Path, path: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256_file(path),
    }
