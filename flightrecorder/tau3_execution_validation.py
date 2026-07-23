"""Post-hoc validators for private Tau-3 training execution bundles.

The validators in this module are a thin mission-critic layer over artifacts
already emitted by the Tau-3 training and benchmark runners. They intentionally
do not add a public schema. Instead, they fail closed over this private local
manifest layout:

```
{
  "schema_version": "hfr.tau3_execution_bundle.v1",
  "code_revision": {"flight_recorder_git_commit": "...40 hex...", "tracked_worktree_clean": true},
  "protocol": {"path": "protocol.json", "sha256": "..."},
  "training": {
    "selected_candidate_id": "candidate-a",
    "selected_receipt": {"path": "training/candidate-a/training_receipt.json", "sha256": "..."},
    "candidate_receipts": [{"path": "...", "sha256": "..."}],
    "candidate_locks": [{"path": "candidate-lock.json", "sha256": "..."}]
  },
  "benchmark": {
    "development_arms": [{"arm_id": "adapter", "path": ".../manifest.json", "sha256": "..."}],
    "sealed_arms": [{"arm_id": "adapter", "path": ".../manifest.json", "sha256": "..."}],
    "public_report": {"path": "public-evaluation-report.json", "sha256": "..."}
  }
}
```

All manifest file references must be relative paths below the bundle root and
must include SHA-256 digests. Candidate locks use the registered public-safe
``hfr.tau3_candidate_lock.v1`` schema: they are hash-only selector outputs that
bind one selected final training receipt to the adapter tree, recipe, base
model, dataset, protocol, candidate identity, endpoint model, and development
selection evidence before sealed evidence exists. The validator links the
top-level selected receipt reference to ``training_receipt_sha256``.

```
{
  "schema_version": "hfr.tau3_candidate_lock.v1",
  "created_at": "2026-07-23T00:00:00Z",
  "selected_candidate_id_hash": "...",
  "training_receipt_sha256": "...",
  "candidate_identity_sha256": "...",
  "development_selection_report_sha256": "...",
  "development_benchmark_manifest_sha256": "...",
  "endpoint_model_sha256": "...",
  "protocol_sha256": "...",
  "protocol_signature": "...",
  "adapter_tree_sha256": "...",
  "recipe_sha256": "...",
  "base_identity_sha256": "...",
  "base_tree_sha256": "...",
  "dataset_manifest_sha256": "...",
  "dataset_files_sha256": "...",
  "source_binding_sha256": "...",
  "hashes_only": true,
  "local_paths_included": false,
  "raw_payload_included": false
}
```
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .path_safety import path_has_symlink_component
from .schema_registry import check_schema_contract
from .tau3_evaluation import analyze_tau3_evaluation

EXECUTION_BUNDLE_SCHEMA_VERSION = "hfr.tau3_execution_bundle.v1"
CANDIDATE_LOCK_SCHEMA_VERSION = "hfr.tau3_candidate_lock.v1"
VALIDATION_SCHEMA_VERSION = "hfr.validation.v1"
TAU3_MLX_TRAINING_RUN_SCHEMA_VERSION = "hfr.tau3_mlx_training_run.v1"
TAU3_BENCHMARK_RUN_SCHEMA_VERSION = "hfr.tau3_benchmark_run.v1"
TAU3_EVALUATION_SCHEMA_VERSION = "hfr.tau3_evaluation.v1"
DOMAINS = ("airline", "retail", "telecom")
SEEDS = (101, 202, 303, 404)
ARMS = ("adapter", "base", "comparator_1", "comparator_2")
PRIVATE_PATH_RE = re.compile(r"(^/|[A-Za-z]:[\\/]|\\\\|/Users/|/home/|/tmp/|localhost|127\.0\.0\.1)")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass
class _Target:
    type: str
    path: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "path": self.path,
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "details": self.details,
        }


def validate_tau3_training_result_bundle(bundle: str | Path, *, strict: bool = False) -> dict[str, Any]:
    """Validate real Tau-3 MLX training evidence and exactly one candidate lock."""

    root, manifest, manifest_target = _load_bundle(bundle)
    targets = [manifest_target]
    protocol = _load_ref_target(root, manifest.get("protocol"), "protocol")
    targets.append(protocol.target)
    training = manifest.get("training") if isinstance(manifest.get("training"), dict) else {}
    if not training:
        targets.append(_missing_target(root, "training", "manifest missing training object"))
        return _summary(strict, targets)

    selected = _load_ref_target(root, training.get("selected_receipt"), "selected_training_receipt")
    targets.append(selected.target)
    selected_payload = selected.payload if isinstance(selected.payload, dict) else {}
    targets.extend(_validate_training_receipt_artifacts(root, selected, "selected"))
    _require(selected.target, selected_payload.get("terminal_status") == "success", "selected training receipt did not finish with terminal_status=success")
    _require(selected.target, selected_payload.get("weights_updated") is True, "selected training receipt does not prove weights_updated=true")
    _require(selected.target, int(selected_payload.get("adapter_weight_file_count") or 0) > 0, "selected receipt has no adapter weight files")

    candidate_hashes: set[str] = set()
    for record in _list_of_refs(training.get("candidate_receipts")):
        candidate = _load_ref_target(root, record, "candidate_training_receipt")
        targets.append(candidate.target)
        if isinstance(candidate.payload, dict):
            targets.extend(_validate_training_receipt_artifacts(root, candidate, "candidate"))
        if candidate.sha256 is not None:
            candidate_hashes.add(candidate.sha256)
    _require(manifest_target, selected.sha256 in candidate_hashes, "selected receipt must also be listed in candidate_receipts")

    selection = _load_candidate_selection_report(root, training.get("candidate_selection_report"), manifest_target)
    targets.append(selection.target)

    lock_refs = _list_of_refs(training.get("candidate_locks"))
    _require(manifest_target, len(lock_refs) == 1, f"expected exactly one candidate lock, found {len(lock_refs)}")
    lock = _load_ref_target(root, lock_refs[0] if lock_refs else None, "candidate_lock")
    targets.append(lock.target)
    selected_candidate_id = training.get("selected_candidate_id")
    if isinstance(lock.payload, dict):
        _validate_candidate_lock(lock.target, lock.payload, selected=selected, protocol=protocol, selected_candidate_id=selected_candidate_id)
        if isinstance(selected_candidate_id, str) and selected_candidate_id:
            _require(lock.target, lock.payload.get("selected_candidate_id_hash") == _canonical_sha256(selected_candidate_id), "candidate lock selected_candidate_id_hash does not match selected_candidate_id")
        if isinstance(selection.payload, dict):
            _validate_candidate_selection(
                selection.target,
                selection.payload,
                selection=selection,
                lock=lock.payload,
                selected=selected,
                selected_candidate_id=selected_candidate_id,
            )

    if strict:
        _require(selected.target, selected_payload.get("schema_checked") is True, "selected receipt must set schema_checked=true")
        losses = _dict(selected_payload.get("losses"))
        _require(selected.target, bool(losses.get("train")), "strict mode requires observed training loss telemetry")

    return _summary(strict, targets)


def _validate_training_receipt_artifacts(root: Path, loaded: _Loaded, label: str) -> list[_Target]:
    targets: list[_Target] = []
    payload = loaded.payload if isinstance(loaded.payload, dict) else {}
    _check_schema(loaded.target, payload, "tau3_mlx_training_run")
    _require(loaded.target, payload.get("schema_version") == TAU3_MLX_TRAINING_RUN_SCHEMA_VERSION, f"{label} receipt is not hfr.tau3_mlx_training_run.v1")
    _require(loaded.target, payload.get("phase") == "final", f"{label} receipt is not final")
    _require(loaded.target, isinstance(payload.get("config"), dict) and bool(payload.get("config")), f"{label} receipt missing config")
    _require(loaded.target, bool(_dict(payload.get("adapter")).get("tree_sha256")), f"{label} receipt missing adapter.tree_sha256")
    _require(loaded.target, isinstance(payload.get("training_binding"), dict), f"{label} receipt missing training_binding; current receipt cannot prove protocol/base/dataset/recipe binding")
    _require(loaded.target, isinstance(payload.get("telemetry"), dict) and payload["telemetry"].get("event_count", 0) > 0, f"{label} receipt telemetry has no events")

    base = loaded.path.parent if loaded.path is not None else root
    _require(loaded.target, payload.get("output_dir") == ".", f"{label} receipt output_dir must be output-local '.'")
    prelaunch = _receipt_nested_file(root, base, payload, "prelaunch_receipt", f"{label}_training_prelaunch_receipt")
    telemetry = _receipt_nested_file(root, base, payload, "telemetry", f"{label}_training_telemetry")
    mlx_config = _receipt_nested_file(root, base, payload, "mlx_lora_config", f"{label}_mlx_lora_config")
    adapter = _validate_adapter_tree(root, base, payload, label)
    targets.extend([prelaunch.target, telemetry.target, mlx_config.target, adapter])
    if isinstance(prelaunch.payload, dict):
        _check_schema(prelaunch.target, prelaunch.payload, "tau3_mlx_training_run")
        _require(prelaunch.target, prelaunch.payload.get("phase") == "prelaunch", f"{label} prelaunch receipt phase mismatch")
        _require(prelaunch.target, prelaunch.payload.get("terminal_status") == "prelaunch", f"{label} prelaunch receipt terminal_status mismatch")
        _require(prelaunch.target, prelaunch.payload.get("config") == payload.get("config"), f"{label} prelaunch config does not match final config")
        _require(prelaunch.target, prelaunch.payload.get("training_binding") == payload.get("training_binding"), f"{label} prelaunch training_binding does not match final binding")
        _require(prelaunch.target, prelaunch.payload.get("output_dir") == ".", f"{label} prelaunch output_dir must be output-local '.'")
        prelaunch_base = prelaunch.path.parent if prelaunch.path is not None else base
        prelaunch_mlx = _receipt_nested_file(root, prelaunch_base, prelaunch.payload, "mlx_lora_config", f"{label}_prelaunch_mlx_lora_config")
        targets.append(prelaunch_mlx.target)
        _require(prelaunch.target, _dict(prelaunch.payload.get("mlx_lora_config")).get("sha256") == _dict(payload.get("mlx_lora_config")).get("sha256"), f"{label} prelaunch MLX config hash does not match final")
        _require(prelaunch_mlx.target, _dict(prelaunch.payload.get("mlx_lora_config")).get("read_only") is True, f"{label} prelaunch MLX config file record must be read_only")
    if isinstance(mlx_config.payload, dict):
        _require(mlx_config.target, _dict(payload.get("mlx_lora_config")).get("read_only") is True, f"{label} MLX config file record must be read_only")
        _validate_mlx_config(mlx_config.target, mlx_config.payload, payload)
    if telemetry.path is not None:
        _validate_training_telemetry(telemetry.target, telemetry.path, payload)
    return targets


def validate_tau3_benchmark_result_bundle(bundle: str | Path, *, strict: bool = False) -> dict[str, Any]:
    """Validate development and one-shot sealed Tau-3 benchmark evidence."""

    root, manifest, manifest_target = _load_bundle(bundle)
    targets = [manifest_target]
    training_result = validate_tau3_training_result_bundle(root, strict=strict)
    for item in training_result["targets"]:
        target = _Target(
            type=f"training.{item['type']}",
            path=item["path"],
            errors=list(item["errors"]),
            warnings=list(item["warnings"]),
            details=dict(item["details"]),
        )
        targets.append(target)
    if training_result.get("passed") is not True:
        targets.append(_Target("training_result", str(root), errors=["training result validation failed; benchmark evidence cannot stand alone"]))

    benchmark = manifest.get("benchmark") if isinstance(manifest.get("benchmark"), dict) else {}
    if not benchmark:
        targets.append(_missing_target(root, "benchmark", "manifest missing benchmark object"))
        return _summary(strict, targets)

    selected_lock = _selected_lock(root, manifest)
    targets.append(selected_lock.target)
    lock_payload = selected_lock.payload if isinstance(selected_lock.payload, dict) else {}
    dev_manifests = _load_arm_manifests(root, benchmark.get("development_arms"), "development_arm", targets)
    sealed_manifests = _load_arm_manifests(root, benchmark.get("sealed_arms"), "sealed_arm", targets)
    _validate_development_grid(manifest_target, dev_manifests, manifest, lock_payload)
    _validate_sealed_grid(manifest_target, sealed_manifests)
    _validate_common_protocol_and_lock(manifest_target, dev_manifests, sealed_manifests, manifest)
    _validate_sealed_predates(manifest_target, manifest, sealed_manifests)

    for loaded in [*dev_manifests, *sealed_manifests]:
        targets.extend(_validate_run_receipts(root, loaded))

    report = _load_ref_target(root, benchmark.get("public_report"), "public_evaluation_report")
    targets.append(report.target)
    if isinstance(report.payload, dict):
        _validate_public_report(root, report.target, report.payload, sealed_manifests)

    return _summary(strict, targets)


@dataclass(frozen=True)
class _Loaded:
    target: _Target
    payload: Any = None
    path: Path | None = None
    sha256: str | None = None


def _load_bundle(bundle: str | Path) -> tuple[Path, dict[str, Any], _Target]:
    root = Path(bundle).resolve()
    target = _Target("manifest", "manifest.json")
    target.details["root"] = str(root)
    if not root.is_dir():
        target.errors.append(f"bundle is not a directory: {bundle}")
        return root, {}, target
    path = root / "manifest.json"
    if not path.is_file():
        target.errors.append("missing manifest.json")
        return root, {}, target
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        target.errors.append(f"manifest is not readable JSON: {exc}")
        return root, {}, target
    _require(target, manifest.get("schema_version") == EXECUTION_BUNDLE_SCHEMA_VERSION, f"manifest schema_version must be {EXECUTION_BUNDLE_SCHEMA_VERSION}")
    _require(target, not _private_path_hits(manifest), "manifest must not contain absolute/private paths")
    code_revision = _dict(manifest.get("code_revision"))
    commit = code_revision.get("flight_recorder_git_commit")
    _require(target, isinstance(commit, str) and bool(re.fullmatch(r"[0-9a-f]{40}", commit)), "manifest code_revision.flight_recorder_git_commit must be a 40-character git commit")
    _require(target, code_revision.get("tracked_worktree_clean") is True, "manifest code_revision.tracked_worktree_clean must be true")
    return root, manifest, target


def _load_ref_target(root: Path, record: Any, target_type: str) -> _Loaded:
    path_text = record.get("path") if isinstance(record, dict) else None
    target = _Target(target_type, path_text if isinstance(path_text, str) else "<missing>")
    path = _resolve_ref(root, record, target)
    if path is None:
        return _Loaded(target=target)
    digest = _sha256_file(path)
    expected = record.get("sha256")
    _require(target, isinstance(expected, str) and bool(SHA256_RE.fullmatch(expected)), "reference missing valid sha256")
    _require(target, expected == digest, "reference sha256 mismatch")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        payload = None
        if not target_type.endswith("training_telemetry"):
            target.errors.append("referenced file is not JSON")
    except OSError as exc:
        payload = None
        target.errors.append(f"referenced file is unreadable: {exc}")
    target.details.update({"sha256": digest, "size": path.stat().st_size})
    return _Loaded(target=target, payload=payload, path=path, sha256=digest)


def _load_json_ref_target(root: Path, record: Any, target_type: str) -> _Loaded:
    loaded = _load_ref_target(root, record, target_type)
    if loaded.path is not None and not isinstance(loaded.payload, dict):
        loaded.target.errors.append("referenced file must be a JSON object")
    return loaded


def _load_local_ref_target(root: Path, base: Path, record: Any, target_type: str) -> _Loaded:
    path_text = record.get("path") if isinstance(record, dict) else None
    target = _Target(target_type, path_text if isinstance(path_text, str) else "<missing>")
    path = _resolve_local_ref(root, base, record, target)
    if path is None:
        return _Loaded(target=target)
    digest = _sha256_file(path)
    expected = record.get("sha256")
    _require(target, isinstance(expected, str) and bool(SHA256_RE.fullmatch(expected)), "reference missing valid sha256")
    _require(target, expected == digest, "reference sha256 mismatch")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        payload = None
        if not target_type.endswith("training_telemetry"):
            target.errors.append("referenced file is not JSON")
    except OSError as exc:
        payload = None
        target.errors.append(f"referenced file is unreadable: {exc}")
    target.details.update({"sha256": digest, "size": path.stat().st_size})
    return _Loaded(target=target, payload=payload, path=path, sha256=digest)


def _selected_lock(root: Path, manifest: dict[str, Any]) -> _Loaded:
    lock_refs = _list_of_refs(_dict(manifest.get("training")).get("candidate_locks"))
    if len(lock_refs) != 1:
        return _Loaded(_Target("selected_candidate_lock", "<missing>", errors=[f"expected exactly one selected candidate lock, found {len(lock_refs)}"]))
    return _load_json_ref_target(root, lock_refs[0], "selected_candidate_lock")


def _resolve_ref(root: Path, record: Any, target: _Target) -> Path | None:
    if not isinstance(record, dict):
        target.errors.append("reference must be an object with path and sha256")
        return None
    rel = record.get("path")
    if not isinstance(rel, str) or not rel:
        target.errors.append("reference path missing")
        return None
    if _is_unsafe_relative_path(rel):
        target.errors.append("reference path must be relative and stay below bundle root")
        return None
    path = (root / rel).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        target.errors.append("reference resolves outside bundle root")
        return None
    if path_has_symlink_component(path, include_leaf=True):
        target.errors.append("reference path contains a symlink component")
        return None
    if not path.is_file():
        target.errors.append("referenced file does not exist")
        return None
    return path


def _resolve_local_ref(root: Path, base: Path, record: Any, target: _Target) -> Path | None:
    if not isinstance(record, dict):
        target.errors.append("reference must be an object with path and sha256")
        return None
    rel = record.get("path")
    if not isinstance(rel, str) or not rel:
        target.errors.append("reference path missing")
        return None
    if _is_unsafe_relative_path(rel):
        target.errors.append("reference path must be output-local and stay below its artifact root")
        return None
    path = (base / rel).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        target.errors.append("reference resolves outside bundle root")
        return None
    if path_has_symlink_component(path, include_leaf=True):
        target.errors.append("reference path contains a symlink component")
        return None
    if not path.is_file():
        target.errors.append("referenced file does not exist")
        return None
    return path


def _is_unsafe_relative_path(value: str) -> bool:
    pure = PurePosixPath(value)
    win = PureWindowsPath(value)
    return (
        pure.is_absolute()
        or win.is_absolute()
        or "\\" in value
        or any(part in {"", ".", ".."} for part in pure.parts)
    )


def _receipt_nested_file(root: Path, base: Path, receipt: dict[str, Any], key: str, target_type: str) -> _Loaded:
    record = receipt.get(key)
    target = _Target(target_type, f"<receipt:{key}>")
    if not isinstance(record, dict):
        target.errors.append(f"receipt missing {key} record")
        return _Loaded(target)
    path_text = record.get("path")
    sha = record.get("sha256")
    if not isinstance(path_text, str) or not path_text:
        target.errors.append(f"receipt {key} missing path")
        return _Loaded(target)
    if _is_unsafe_relative_path(path_text):
        target.errors.append(f"receipt {key} path must be copied output-local")
        return _Loaded(target)
    return _load_local_ref_target(root, base, {"path": path_text, "sha256": sha}, target_type)


def _validate_adapter_tree(root: Path, base: Path, receipt: dict[str, Any], label: str) -> _Target:
    adapter = _dict(receipt.get("adapter"))
    path_text = adapter.get("path")
    target = _Target(f"{label}_adapter_tree", str(path_text or "<missing>"))
    if not isinstance(path_text, str) or not path_text:
        target.errors.append(f"{label} adapter missing path")
        return target
    if _is_unsafe_relative_path(path_text):
        target.errors.append(f"{label} adapter path must be copied output-local")
        return target
    adapter_root = (base / path_text).resolve()
    try:
        adapter_root.relative_to(root)
    except ValueError:
        target.errors.append(f"{label} adapter path resolves outside bundle root")
        return target
    if not adapter_root.is_dir():
        target.errors.append(f"{label} adapter path is not a directory")
        return target
    if path_has_symlink_component(adapter_root, include_leaf=True):
        target.errors.append(f"{label} adapter path contains a symlink component")
        return target
    declared_files = _list_of_dicts(adapter.get("files"))
    actual_records: list[dict[str, Any]] = []
    for record in sorted(declared_files, key=lambda item: str(item.get("path"))):
        rel = record.get("path")
        if not isinstance(rel, str) or _is_unsafe_relative_path(rel):
            target.errors.append(f"{label} adapter file path is unsafe")
            continue
        file_path = (adapter_root / rel).resolve()
        try:
            file_path.relative_to(adapter_root)
        except ValueError:
            target.errors.append(f"{label} adapter file escapes adapter root: {rel}")
            continue
        if not file_path.is_file():
            target.errors.append(f"{label} adapter file missing: {rel}")
            continue
        actual = {
            "path": rel,
            "size": file_path.stat().st_size,
            "sha256": _sha256_file(file_path),
            "kind": str(record.get("kind") or _adapter_file_kind(rel)),
        }
        _require(target, record.get("size") == actual["size"], f"{label} adapter file size mismatch: {rel}")
        _require(target, record.get("sha256") == actual["sha256"], f"{label} adapter file sha256 mismatch: {rel}")
        _require(target, record.get("kind") == actual["kind"], f"{label} adapter file kind mismatch: {rel}")
        actual_records.append(actual)
    digest = hashlib.sha256()
    for record in actual_records:
        digest.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    tree_sha = digest.hexdigest() if actual_records else None
    _require(target, adapter.get("file_count") == len(actual_records), f"{label} adapter file_count mismatch")
    _require(target, adapter.get("tree_sha256") == tree_sha, f"{label} adapter tree_sha256 mismatch")
    _require(target, any(record.get("kind") == "adapter" and int(record.get("size") or 0) > 0 for record in actual_records), f"{label} adapter tree has no non-empty adapter weight file")
    target.details.update({"file_count": len(actual_records), "tree_sha256": tree_sha})
    return target


def _adapter_file_kind(rel: str) -> str:
    name = Path(rel).name
    if name in {"adapter_config.json", "config.json"}:
        return "config"
    if "checkpoint" in rel.lower():
        return "checkpoint"
    if Path(rel).suffix in {".safetensors", ".npz", ".bin"}:
        return "adapter"
    return "artifact"


def _validate_mlx_config(target: _Target, mlx_config: dict[str, Any], receipt: dict[str, Any]) -> None:
    cfg = _dict(receipt.get("config"))
    adapter = _dict(receipt.get("adapter"))
    _require(target, mlx_config.get("train") is True, "MLX config must train")
    _require(target, mlx_config.get("fine_tune_type") == "lora", "MLX config must use LoRA")
    _require(target, mlx_config.get("test") is False, "MLX config must not enable test mode")
    _require(target, mlx_config.get("report_to") is None, "MLX config must not report externally")
    adapter_path = mlx_config.get("adapter_path")
    _require(target, isinstance(adapter_path, str) and not _is_unsafe_relative_path(adapter_path), "MLX config adapter_path must be output-relative")
    _require(target, adapter_path == adapter.get("path"), "MLX config adapter_path must exactly match copied adapter path")
    key_map = {
        "iters": "iters",
        "learning_rate": "learning_rate",
        "num_layers": "num_layers",
        "batch_size": "batch_size",
        "max_seq_length": "max_seq_length",
        "seed": "seed",
        "mask_prompt": "mask_prompt",
        "grad_checkpoint": "grad_checkpoint",
    }
    for config_key, receipt_key in key_map.items():
        if receipt_key in cfg:
            _require(target, mlx_config.get(config_key) == cfg.get(receipt_key), f"MLX config {config_key} does not match receipt config")
    lora = _dict(mlx_config.get("lora_parameters"))
    for key in ("rank", "scale", "dropout"):
        if key in cfg:
            _require(target, lora.get(key) == cfg.get(key), f"MLX lora_parameters.{key} does not match receipt config")


def _validate_training_telemetry(target: _Target, path: Path, receipt: dict[str, Any]) -> None:
    event_count = 0
    observed_train_losses: list[float] = []
    observed_validation_losses: list[float] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        target.errors.append(f"telemetry unreadable: {exc}")
        return
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            target.errors.append(f"telemetry line {line_number} is not JSON: {exc.msg}")
            continue
        if not isinstance(event, dict):
            target.errors.append(f"telemetry line {line_number} must be a JSON object")
            continue
        event_count += 1
        _require(target, isinstance(event.get("time"), str) and bool(event.get("time")), f"telemetry line {line_number} missing time")
        _require(target, isinstance(event.get("stream"), str) and bool(event.get("stream")), f"telemetry line {line_number} missing stream")
        text = str(event.get("text") or "")
        for match in re.finditer(r"\b(?P<kind>train|training|valid|validation|val)(?:[_ -]?loss)?\b[^0-9+-]*(?P<loss>[+-]?(?:\d+(?:\.\d*)?|\.\d+))", text, re.IGNORECASE):
            value = float(match.group("loss"))
            kind = match.group("kind").lower()
            if kind in {"valid", "validation", "val"}:
                observed_validation_losses.append(value)
            else:
                observed_train_losses.append(value)
    telemetry = _dict(receipt.get("telemetry"))
    _require(target, event_count == telemetry.get("event_count"), "telemetry event_count does not match JSONL line count")
    losses = _dict(receipt.get("losses"))
    _require(target, observed_train_losses == list(losses.get("train") or []), "telemetry train losses do not replay final receipt losses")
    _require(target, observed_validation_losses == list(losses.get("validation") or []), "telemetry validation losses do not replay final receipt losses")
    _require(target, bool(observed_train_losses), "telemetry JSONL does not contain an observed train loss")
    target.details.update({"event_count": event_count, "train_loss_count": len(observed_train_losses), "validation_loss_count": len(observed_validation_losses)})


def _validate_candidate_lock(target: _Target, lock: dict[str, Any], *, selected: _Loaded, protocol: _Loaded, selected_candidate_id: Any) -> None:
    _check_schema(target, lock, "tau3_candidate_lock")
    _require(target, lock.get("schema_version") == CANDIDATE_LOCK_SCHEMA_VERSION, f"candidate lock schema_version must be {CANDIDATE_LOCK_SCHEMA_VERSION}")
    _require(target, not _private_path_hits(lock), "candidate lock must not contain absolute/private paths")
    _require(target, _parse_time(lock.get("created_at")) is not None, "candidate lock missing parseable created_at")
    _require(target, lock.get("hashes_only") is True, "candidate lock must be hashes_only")
    _require(target, lock.get("local_paths_included") is False, "candidate lock must not include local paths")
    _require(target, lock.get("raw_payload_included") is False, "candidate lock must not include raw payloads")
    selected_payload = selected.payload if isinstance(selected.payload, dict) else {}
    binding = _dict(selected_payload.get("training_binding"))
    adapter = _dict(selected_payload.get("adapter"))
    model = _dict(binding.get("model"))
    dataset = _dict(binding.get("dataset"))
    recipe = _dict(binding.get("recipe"))
    expected = {
        "protocol_sha256": protocol.sha256 or _nested(binding, "protocol", "sha256"),
        "adapter_tree_sha256": adapter.get("tree_sha256"),
        "recipe_sha256": recipe.get("recipe_sha256"),
        "base_identity_sha256": model.get("identity_sha256"),
        "base_tree_sha256": model.get("tree_sha256"),
        "dataset_manifest_sha256": dataset.get("manifest_sha256"),
        "dataset_files_sha256": dataset.get("files_sha256"),
        "source_binding_sha256": dataset.get("source_binding_sha256"),
        "protocol_signature": _nested(binding, "protocol", "protocol_signature"),
    }
    _require(target, lock.get("training_receipt_sha256") == selected.sha256, "candidate lock training_receipt_sha256 does not match selected receipt")
    if isinstance(selected_candidate_id, str) and selected_candidate_id:
        _require(target, lock.get("selected_candidate_id_hash") == _canonical_sha256(selected_candidate_id), "candidate lock selected_candidate_id_hash mismatch")
    for key, value in expected.items():
        _require(target, isinstance(value, str) and bool(SHA256_RE.fullmatch(value)), f"selected receipt missing {key} source field")
        _require(target, lock.get(key) == value, f"candidate lock {key} mismatch")
    target.details["selected_candidate_id_hash"] = lock.get("selected_candidate_id_hash")


def _load_candidate_selection_report(root: Path, record: Any, manifest_target: _Target) -> _Loaded:
    if isinstance(record, list):
        _require(manifest_target, len(record) == 1, f"expected exactly one candidate_selection_report, found {len(record)}")
        selected = record[0] if record else None
    elif isinstance(record, dict):
        selected = record
    else:
        _require(manifest_target, False, "training must include exactly one candidate_selection_report reference")
        selected = None
    return _load_json_ref_target(root, selected, "candidate_selection_report")


def _validate_candidate_selection(
    target: _Target,
    report: dict[str, Any],
    *,
    selection: _Loaded,
    lock: dict[str, Any],
    selected: _Loaded,
    selected_candidate_id: Any,
) -> None:
    _check_schema(target, report, "tau3_candidate_selection")
    _require(target, not _private_path_hits(report), "candidate selection report must not contain absolute/private paths")
    _require(target, report.get("passed") is True, "candidate selection report must be passed")
    _require(target, report.get("schema_checked") is True, "candidate selection report must set schema_checked=true")
    _require(target, report.get("selected_candidate_id") == selected_candidate_id, "candidate selection selected_candidate_id does not match manifest")
    _require(target, lock.get("development_selection_report_sha256") == selection.sha256, "candidate lock development_selection_report_sha256 does not match selection report")
    chosen = _dict(report.get("selection"))
    _require(target, chosen.get("candidate_id") == selected_candidate_id, "selection candidate_id does not match selected candidate")
    _require(target, chosen.get("candidate_identity_sha256") == lock.get("candidate_identity_sha256"), "selection candidate_identity_sha256 does not match lock")
    candidates = _list_of_dicts(report.get("candidates"))
    selected_rows = [row for row in candidates if row.get("candidate_id") == selected_candidate_id]
    _require(target, len(selected_rows) == 1, f"selection report must contain exactly one selected candidate row, found {len(selected_rows)}")
    if selected_rows:
        row = selected_rows[0]
        artifacts = _dict(row.get("artifacts"))
        training_receipt = _dict(artifacts.get("training_receipt"))
        development_manifest = _dict(artifacts.get("development_manifest"))
        candidate_identity = _dict(row.get("candidate_identity"))
        _require(target, training_receipt.get("sha256") == selected.sha256, "selection training receipt binding does not match selected receipt")
        _require(target, training_receipt.get("sha256") == lock.get("training_receipt_sha256"), "selection training receipt binding does not match lock")
        _require(target, development_manifest.get("sha256") == lock.get("development_benchmark_manifest_sha256"), "selection development manifest binding does not match lock")
        _require(target, candidate_identity.get("sha256") == lock.get("candidate_identity_sha256"), "selection candidate identity binding does not match lock")
        _require(target, candidate_identity.get("endpoint_model_sha256") == lock.get("endpoint_model_sha256"), "selection endpoint model binding does not match lock")


def _load_arm_manifests(root: Path, refs: Any, target_type: str, targets: list[_Target]) -> list[_Loaded]:
    loaded: list[_Loaded] = []
    for record in _list_of_refs(refs):
        item = _load_ref_target(root, record, target_type)
        targets.append(item.target)
        if isinstance(item.payload, dict):
            _check_schema(item.target, item.payload, "tau3_benchmark_run")
            _require(item.target, item.payload.get("schema_version") == TAU3_BENCHMARK_RUN_SCHEMA_VERSION, "arm manifest is not hfr.tau3_benchmark_run.v1")
            _require(item.target, item.payload.get("phase") == "final", "arm manifest is not final")
            _require(item.target, item.payload.get("run_count") == 12, "arm manifest must contain exactly 12 domain/seed runs")
            _require(item.target, item.payload.get("success_count") == 12 and item.payload.get("failure_count") == 0, "all 12 arm runs must complete successfully")
            targets.extend(_validate_benchmark_arm_refs(root, item))
            loaded.append(item)
    return loaded


def _validate_benchmark_arm_refs(root: Path, loaded: _Loaded) -> list[_Target]:
    payload = loaded.payload if isinstance(loaded.payload, dict) else {}
    base = loaded.path.parent if loaded.path is not None else root
    targets: list[_Target] = []
    prelaunch = _receipt_nested_file(root, base, payload, "prelaunch_receipt", "benchmark_prelaunch_receipt")
    targets.append(prelaunch.target)
    if isinstance(prelaunch.payload, dict):
        _check_schema(prelaunch.target, prelaunch.payload, "tau3_benchmark_run")
        _require(prelaunch.target, prelaunch.payload.get("phase") == "prelaunch", "benchmark prelaunch phase mismatch")
        for key in ("protocol_sha256", "mode", "arm_id", "config", "candidate_lock", "candidate_identity"):
            _require(prelaunch.target, prelaunch.payload.get(key) == payload.get(key), f"benchmark prelaunch {key} does not match final manifest")
    if payload.get("mode") == "development":
        source = _receipt_nested_file(root, base, payload, "source", "development_source")
        targets.append(source.target)
        if payload.get("arm_id") == "adapter":
            candidate_identity = _receipt_nested_file(root, base, payload, "candidate_identity", "development_candidate_identity")
            targets.append(candidate_identity.target)
            if isinstance(candidate_identity.payload, dict):
                _validate_development_candidate_identity(candidate_identity.target, payload, candidate_identity.payload)
    if payload.get("mode") == "sealed":
        candidate_lock = _receipt_nested_file(root, base, payload, "candidate_lock", "sealed_candidate_lock")
        targets.append(candidate_lock.target)
        if isinstance(candidate_lock.payload, dict) and payload.get("arm_id") == "adapter":
            _validate_sealed_adapter_identity(candidate_lock.target, payload, candidate_lock.payload)
    return targets


def _validate_development_candidate_identity(target: _Target, arm: dict[str, Any], identity: dict[str, Any]) -> None:
    arm_identity = _dict(arm.get("arm_identity"))
    adapter = _dict(arm_identity.get("adapter"))
    _require(target, bool(adapter), "development adapter arm identity missing adapter request evidence")
    _require(target, arm_identity.get("candidate_identity_sha256") == _dict(arm.get("candidate_identity")).get("sha256"), "development arm candidate identity hash does not match ref")
    _require(target, adapter.get("tree_sha256") == _identity_field(identity, "adapter_tree_sha256", "tree_sha256"), "development adapter tree_sha256 does not match candidate identity")
    _require(target, arm_identity.get("endpoint_model_sha256") == _identity_field(identity, "endpoint_model_sha256"), "development endpoint model does not match candidate identity")


def _validate_sealed_adapter_identity(target: _Target, arm: dict[str, Any], lock: dict[str, Any]) -> None:
    arm_identity = _dict(arm.get("arm_identity"))
    adapter = _dict(arm_identity.get("adapter"))
    _require(target, bool(adapter), "sealed adapter arm identity missing adapter request evidence")
    _require(target, arm_identity.get("candidate_lock_sha256") == _dict(arm.get("candidate_lock")).get("sha256"), "sealed adapter arm lock hash does not match ref")
    _require(target, arm_identity.get("candidate_identity_sha256") == lock.get("candidate_identity_sha256"), "sealed adapter candidate identity does not match lock")
    _require(target, arm_identity.get("endpoint_model_sha256") == lock.get("endpoint_model_sha256"), "sealed adapter endpoint model does not match lock")
    _require(target, adapter.get("tree_sha256") == lock.get("adapter_tree_sha256"), "sealed adapter tree_sha256 does not match lock")


def _validate_development_grid(target: _Target, arms: list[_Loaded], manifest: dict[str, Any], lock: dict[str, Any]) -> None:
    payloads = [item.payload for item in arms if isinstance(item.payload, dict)]
    _require(target, len(payloads) >= 2, "development benchmark must contain one base and at least one adapter candidate")
    base_count = sum(1 for item in payloads if item.get("arm_id") == "base")
    adapter_payloads = [item for item in payloads if item.get("arm_id") == "adapter"]
    other = sorted({str(item.get("arm_id")) for item in payloads if item.get("arm_id") not in {"base", "adapter"}})
    _require(target, base_count == 1, f"development benchmark must contain exactly one base arm, found {base_count}")
    _require(target, bool(adapter_payloads), "development benchmark must contain at least one adapter candidate arm")
    _require(target, not other, f"development benchmark must not contain comparator arms: {other}")
    selected_candidate_id = _dict(manifest.get("training")).get("selected_candidate_id")
    candidate_ids = _training_candidate_ids(manifest)
    seen_identities: set[str] = set()
    selected_lock_matches = 0
    for payload in payloads:
        _require(target, payload.get("mode") == "development", f"development {payload.get('arm_id')} arm mode must be development")
        combo = {(row.get("domain"), row.get("seed")) for row in _list_of_dicts(payload.get("run_receipts"))}
        expected = {(domain, seed) for domain in DOMAINS for seed in SEEDS}
        _require(target, combo == expected, f"development {payload.get('arm_id')} does not contain the exact 3x4 domain/seed grid")
        _require(target, payload.get("candidate_lock") is None, "development arms must not use a sealed candidate lock")
        if payload.get("arm_id") == "adapter":
            identity = _dict(payload.get("candidate_identity"))
            identity_sha = identity.get("sha256")
            candidate_id = _dict(payload.get("arm_identity")).get("candidate_id") or identity.get("candidate_id")
            _require(target, isinstance(identity_sha, str) and bool(SHA256_RE.fullmatch(identity_sha)), "development adapter arm must bind candidate_identity.sha256")
            _require(target, identity_sha not in seen_identities, "development adapter candidate identities must be unique")
            if isinstance(identity_sha, str):
                seen_identities.add(identity_sha)
            if candidate_ids:
                _require(target, isinstance(candidate_id, str) and candidate_id in candidate_ids, "development adapter candidate_id must match a training candidate when exposed")
            if lock:
                loaded = next((item for item in arms if item.payload is payload), None)
                manifest_sha = loaded.sha256 if loaded is not None else None
                identity_matches = identity_sha == lock.get("candidate_identity_sha256")
                manifest_matches = manifest_sha == lock.get("development_benchmark_manifest_sha256")
                if identity_matches and manifest_matches:
                    selected_lock_matches += 1
                    endpoint = _dict(payload.get("arm_identity")).get("endpoint_model_sha256")
                    if endpoint is not None:
                        _require(target, endpoint == lock.get("endpoint_model_sha256"), "development selected adapter endpoint_model_sha256 does not match lock")
    if isinstance(selected_candidate_id, str) and selected_candidate_id and candidate_ids:
        _require(target, selected_candidate_id in candidate_ids, "selected_candidate_id must match an exposed training candidate")
    if lock:
        _require(target, selected_lock_matches == 1, f"expected exactly one development adapter to match selected lock identity/manifest, found {selected_lock_matches}")


def _training_candidate_ids(manifest: dict[str, Any]) -> set[str]:
    training = _dict(manifest.get("training"))
    ids: set[str] = set()
    selected = training.get("selected_candidate_id")
    if isinstance(selected, str) and selected:
        ids.add(selected)
    for record in _list_of_refs(training.get("candidate_receipts")):
        candidate_id = record.get("candidate_id")
        if isinstance(candidate_id, str) and candidate_id:
            ids.add(candidate_id)
    return ids


def _validate_sealed_grid(target: _Target, arms: list[_Loaded]) -> None:
    payloads = [item.payload for item in arms if isinstance(item.payload, dict)]
    _require(target, len(payloads) == 4, "sealed benchmark must contain exactly four arm manifests")
    _require(target, sorted(str(item.get("arm_id")) for item in payloads) == list(ARMS), "sealed benchmark arms must be adapter/base/comparator_1/comparator_2")
    for payload in payloads:
        _require(target, payload.get("mode") == "sealed", f"{payload.get('arm_id')} arm mode must be sealed")
        combo = {(row.get("domain"), row.get("seed")) for row in _list_of_dicts(payload.get("run_receipts"))}
        expected = {(domain, seed) for domain in DOMAINS for seed in SEEDS}
        _require(target, combo == expected, f"sealed {payload.get('arm_id')} does not contain the exact 3x4 domain/seed grid")
        _require(target, payload.get("source") is None, "sealed arm must not bind a source split")
        _require(target, _dict(payload.get("task_selection")).get("task_ids_in_command") is False, "sealed arm must not materialize task IDs")
        _require(target, payload.get("sealed_payload_accessed") is False, "sealed arm must report sealed_payload_accessed=false")
        _require(target, payload.get("sealed_task_ids_materialized") is False, "sealed arm must report sealed_task_ids_materialized=false")


def _validate_common_protocol_and_lock(target: _Target, dev: list[_Loaded], sealed: list[_Loaded], manifest: dict[str, Any]) -> None:
    payloads = [item.payload for item in [*dev, *sealed] if isinstance(item.payload, dict)]
    protocols = {item.get("protocol_sha256") for item in payloads}
    _require(target, len(protocols) == 1 and None not in protocols, "all benchmark arms must bind one identical protocol hash")
    sealed_locks = {_dict(item.get("candidate_lock")).get("sha256") for item in payloads if item.get("mode") == "sealed"}
    _require(target, len(sealed_locks) == 1 and None not in sealed_locks, "all sealed arms must bind one identical candidate lock")
    lock_refs = _list_of_refs(_dict(manifest.get("training")).get("candidate_locks"))
    selected_lock_sha = lock_refs[0].get("sha256") if len(lock_refs) == 1 else None
    sealed_lock_sha = next(iter(sealed_locks)) if sealed_locks else None
    _require(target, sealed_lock_sha == selected_lock_sha, "sealed arms must bind the exact selected candidate lock")
    for payload in payloads:
        if payload.get("mode") == "development":
            _require(target, payload.get("candidate_lock") is None, "development arms must not use a sealed candidate lock")
        _require(target, _dict(payload.get("config")).get("test_time_search") is False, "benchmark config must disable test-time search")


def _validate_sealed_predates(target: _Target, manifest: dict[str, Any], sealed: list[_Loaded]) -> None:
    lock_refs = _list_of_refs(_dict(manifest.get("training")).get("candidate_locks"))
    if len(lock_refs) != 1:
        return
    root = Path(target.details.get("root", "")) if target.details.get("root") else None
    lock_time: datetime | None = None
    if root is not None:
        lock = _load_ref_target(root, lock_refs[0], "candidate_lock_for_time")
        lock_time = _parse_time(_dict(lock.payload).get("created_at")) if isinstance(lock.payload, dict) else None
    if root is None or lock_time is None:
        target.warnings.append("candidate lock time unavailable for sealed chronology replay")
        return
    root_path = root
    for item in sealed:
        payload = item.payload if isinstance(item.payload, dict) else {}
        base = item.path.parent if item.path is not None else root_path
        created = _parse_time(payload.get("created_at"))
        _require(target, created is not None and lock_time < created, "candidate lock must predate sealed final evidence")
        prelaunch = _load_local_ref_target(root_path, base, _dict(payload.get("prelaunch_receipt")), "sealed_prelaunch_for_time")
        prelaunch_time = _parse_time(_dict(prelaunch.payload).get("created_at")) if isinstance(prelaunch.payload, dict) else None
        _require(target, prelaunch_time is not None and lock_time < prelaunch_time, "candidate lock must predate sealed prelaunch evidence")
        for run_time_target in _validate_run_receipt_times(root_path, item, lock_time):
            target.errors.extend(run_time_target.errors)


def _validate_run_receipts(root: Path, loaded: _Loaded) -> list[_Target]:
    payload = loaded.payload if isinstance(loaded.payload, dict) else {}
    base = loaded.path.parent if loaded.path is not None else root
    targets: list[_Target] = []
    for record in _list_of_dicts(payload.get("run_receipts")):
        ref = {"path": record.get("path"), "sha256": _run_receipt_expected_sha(record)}
        receipt_target = _Target("benchmark_run_receipt", ref["path"])
        _require(receipt_target, isinstance(record.get("receipt_sha256"), str) and bool(SHA256_RE.fullmatch(str(record.get("receipt_sha256")))), "manifest run receipt missing receipt_sha256")
        _require(receipt_target, isinstance(record.get("result_path"), str) and bool(record.get("result_path")), "manifest run receipt missing copied result_path")
        path = _resolve_local_ref(root, base, ref, receipt_target)
        if path is None:
            targets.append(receipt_target)
            continue
        digest = _sha256_file(path)
        _require(receipt_target, ref["sha256"] == digest, "manifest run receipt sha256 mismatch")
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            receipt_target.errors.append(f"run receipt JSON invalid: {exc}")
            targets.append(receipt_target)
            continue
        _require(receipt_target, receipt.get("result_sha256") == record.get("result_sha256"), "manifest result_sha256 does not match run receipt")
        _require(receipt_target, receipt.get("terminal_status") == "completed", "run receipt did not complete")
        _require(receipt_target, receipt.get("mode") == payload.get("mode"), "run receipt mode mismatch")
        _require(receipt_target, receipt.get("arm_id") == payload.get("arm_id"), "run receipt arm mismatch")
        _require(receipt_target, receipt.get("protocol_sha256") == payload.get("protocol_sha256"), "run receipt protocol mismatch")
        _require(receipt_target, receipt.get("result_sha256") is not None, "run receipt missing raw result hash")
        _require(receipt_target, receipt.get("result_path") == record.get("result_path"), "manifest copied result_path does not match run receipt")
        _validate_raw_result(root, path.parent, receipt_target, receipt)
        receipt_target.details.update({"sha256": digest, "result_sha256": receipt.get("result_sha256")})
        targets.append(receipt_target)
    return targets


def _validate_run_receipt_times(root: Path, loaded: _Loaded, lock_time: datetime) -> list[_Target]:
    payload = loaded.payload if isinstance(loaded.payload, dict) else {}
    base = loaded.path.parent if loaded.path is not None else root
    targets: list[_Target] = []
    for record in _list_of_dicts(payload.get("run_receipts")):
        rel = record.get("path")
        target = _Target("sealed_run_receipt_time", str(rel or "<missing>"))
        if not isinstance(rel, str):
            target.errors.append("sealed run receipt missing path")
            targets.append(target)
            continue
        ref = {"path": rel, "sha256": _run_receipt_expected_sha(record)}
        path = _resolve_local_ref(root, base, ref, target)
        if path is None:
            targets.append(target)
            continue
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            target.errors.append(f"sealed run receipt JSON invalid: {exc}")
            targets.append(target)
            continue
        created = _parse_time(_dict(receipt).get("created_at"))
        _require(target, created is not None and lock_time < created, "candidate lock must predate sealed run receipt evidence")
        targets.append(target)
    return targets


def _run_receipt_expected_sha(record: dict[str, Any]) -> Any:
    return record.get("receipt_sha256")


def _validate_raw_result(root: Path, base: Path, target: _Target, receipt: dict[str, Any]) -> None:
    result_path = receipt.get("result_path")
    if not isinstance(result_path, str) or not result_path:
        target.errors.append("run receipt missing result_path")
        return
    path = Path(result_path)
    if path.is_absolute():
        try:
            resolved = path.resolve()
            resolved.relative_to(root)
        except (OSError, ValueError):
            target.errors.append("run receipt result_path must stay under bundle root")
            return
    else:
        if _is_unsafe_relative_path(result_path):
            target.errors.append("run receipt result_path must be output-local below receipt directory")
            return
        resolved = (base / result_path).resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            target.errors.append("run receipt result_path resolves outside bundle root")
            return
    if not resolved.is_file():
        target.errors.append("raw result file does not exist")
        return
    if path_has_symlink_component(resolved, include_leaf=True):
        target.errors.append("raw result path contains a symlink component")
        return
    _require(target, _sha256_file(resolved) == receipt.get("result_sha256"), "raw result sha256 mismatch")
    target.details["result_path"] = str(resolved)


def _validate_public_report(root: Path, target: _Target, report: dict[str, Any], sealed: list[_Loaded]) -> None:
    _require(target, report.get("schema_version") == TAU3_EVALUATION_SCHEMA_VERSION, "public evaluation report must be hfr.tau3_evaluation.v1")
    _check_schema(target, report, "tau3_evaluation")
    _require(target, not _private_path_hits(report), "public evaluation report contains private paths or loopback endpoints")
    sealed_by_arm = {str(item.payload.get("arm_id")): item for item in sealed if isinstance(item.payload, dict)}
    _require(target, sorted(sealed_by_arm) == list(ARMS), "public report replay requires sealed adapter/base/comparator_1/comparator_2 arms")
    raw_paths = _sealed_raw_result_paths(root, sealed_by_arm, target)
    if sorted(raw_paths) == list(ARMS) and all(raw_paths[arm] for arm in ARMS):
        try:
            expected = analyze_tau3_evaluation(
                arm_result_paths=raw_paths,
                mode="sealed",
                expected_tau_revision=str(report.get("tau_revision") or ""),
                created_at=str(report.get("created_at") or ""),
                bootstrap_samples=int(_dict(report.get("analysis_config")).get("bootstrap_samples") or 100),
                bootstrap_seed=int(_dict(report.get("analysis_config")).get("bootstrap_seed") or 0),
                non_inferiority_margin=float(_dict(report.get("analysis_config")).get("non_inferiority_margin") or 0.0),
            )
        except (OSError, ValueError) as exc:
            target.errors.append(f"public evaluation replay failed: {exc}")
            return
        for key in (
            "metrics",
            "effects",
            "per_task_hashed",
            "pairing",
            "harness",
            "tau_revision",
            "mode",
            "analysis_config",
            "checks",
            "failed_check_count",
            "blocking_reasons",
            "passed",
            "promotion_ready",
            "readiness",
        ):
            _require(target, report.get(key) == expected.get(key), f"public evaluation {key} does not replay from sealed evidence")
    failed_checks = sum(1 for check in _list_of_dicts(report.get("checks")) if check.get("passed") is not True)
    _require(target, report.get("failed_check_count") == failed_checks, "public evaluation failed_check_count does not match checks")
    expected_passed = failed_checks == 0
    _require(target, report.get("passed") == expected_passed, "public evaluation passed flag is inconsistent with checks")
    expected_promotion_ready = expected_passed and report.get("mode") == "sealed" and report.get("readiness") == "ready_for_publication_review" and not report.get("blocking_reasons")
    _require(target, report.get("promotion_ready") == expected_promotion_ready, "public evaluation promotion_ready flag is inconsistent")
    scan = _dict(report.get("public_payload_scan"))
    declared_report_sha256 = scan.get("report_sha256")
    report_without_scan_hash = json.loads(json.dumps(report))
    _dict(report_without_scan_hash.get("public_payload_scan")).pop("report_sha256", None)
    _require(
        target,
        declared_report_sha256 == _canonical_sha256(report_without_scan_hash),
        "public evaluation public_payload_scan.report_sha256 does not replay",
    )
    _validate_report_source_artifacts(root, target, report, raw_paths)


def _sealed_raw_result_paths(root: Path, sealed_by_arm: dict[str, _Loaded], target: _Target) -> dict[str, list[Path]]:
    result_paths: dict[str, list[Path]] = {arm: [] for arm in ARMS}
    for arm, loaded in sealed_by_arm.items():
        payload = loaded.payload if isinstance(loaded.payload, dict) else {}
        base = loaded.path.parent if loaded.path is not None else root
        for record in _list_of_dicts(payload.get("run_receipts")):
            receipt_path = _resolve_local_ref(root, base, {"path": record.get("path"), "sha256": _run_receipt_expected_sha(record)}, target)
            if receipt_path is None:
                continue
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                target.errors.append(f"cannot replay sealed run receipt for public report: {exc}")
                continue
            rel = receipt.get("result_path")
            if not isinstance(rel, str) or _is_unsafe_relative_path(rel):
                target.errors.append("sealed receipt result_path is not output-local")
                continue
            result_path = (receipt_path.parent / rel).resolve()
            try:
                result_path.relative_to(root)
            except ValueError:
                target.errors.append("sealed raw result path escapes bundle root")
                continue
            result_paths[arm].append(result_path)
    return result_paths


def _validate_report_source_artifacts(
    root: Path,
    target: _Target,
    report: dict[str, Any],
    raw_paths: dict[str, list[Path]],
) -> None:
    source_artifacts = _dict(report.get("source_artifacts"))
    for arm in ARMS:
        artifacts = _list_of_dicts(source_artifacts.get(arm))
        expected_hashes = sorted(_sha256_file(path) for path in raw_paths.get(arm, []))
        actual_hashes = sorted(str(item.get("sha256") or "") for item in artifacts)
        _require(
            target,
            bool(expected_hashes) and actual_hashes == expected_hashes,
            f"public report source_artifacts.{arm} must exactly bind sealed raw result hashes",
        )
        for item in artifacts:
            _require(target, item.get("public_safe") is True, f"public report source artifact must be public_safe for {arm}")
            path_text = item.get("path")
            if not isinstance(path_text, str) or _is_unsafe_relative_path(path_text):
                target.errors.append(f"public report source artifact path is unsafe for {arm}")
                continue
            resolved = (root / path_text).resolve()
            try:
                resolved.relative_to(root)
            except ValueError:
                target.errors.append(f"public report source artifact escapes bundle root for {arm}")
                continue
            if resolved.is_file():
                _require(target, _sha256_file(resolved) == item.get("sha256"), f"public report source artifact sha256 mismatch for {arm}")


def _identity_field(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    for nested_key in ("candidate", "identity", "model_identity", "adapter_identity", "adapter"):
        nested = record.get(nested_key)
        if isinstance(nested, dict):
            value = _identity_field(nested, *keys)
            if value is not None:
                return value
    return None


def _check_schema(target: _Target, payload: dict[str, Any], name: str) -> None:
    check = check_schema_contract(payload, name_or_id=name)
    _require(target, check.get("passed") is True, f"{name} schema check failed: {check.get('errors')}")


def _summary(strict: bool, targets: list[_Target]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "passed": all(target.passed for target in targets),
        "strict": strict,
        "target_count": len(targets),
        "error_count": sum(len(target.errors) for target in targets),
        "warning_count": sum(len(target.warnings) for target in targets),
        "targets": [target.as_dict() for target in targets],
    }
    check = check_schema_contract(payload, name_or_id="validation")
    if check.get("passed") is not True:
        payload["passed"] = False
        payload["error_count"] += 1
        payload["targets"].append(
            {
                "type": "validation_result_schema",
                "path": "<result>",
                "passed": False,
                "errors": [str(check.get("errors"))],
                "warnings": [],
                "details": {},
            }
        )
    return payload


def _missing_target(root: Path, target_type: str, message: str) -> _Target:
    target = _Target(target_type, str(root))
    target.errors.append(message)
    return target


def _list_of_refs(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _nested(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        current = _dict(current).get(key)
    return current


def _require(target: _Target, condition: bool, error: str) -> None:
    if not condition:
        target.errors.append(error)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _private_path_hits(value: Any) -> list[str]:
    hits: list[str] = []
    if isinstance(value, dict):
        for item in value.values():
            hits.extend(_private_path_hits(item))
    elif isinstance(value, list):
        for item in value:
            hits.extend(_private_path_hits(item))
    elif isinstance(value, str) and PRIVATE_PATH_RE.search(value):
        hits.append(value)
    return hits


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
