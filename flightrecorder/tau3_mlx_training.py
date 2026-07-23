"""Fail-closed local MLX-LM QLoRA runner for governed Tau-3 mixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .path_safety import path_has_symlink_component
from .schema_registry import check_schema_contract
from .tau3_model_identity import validate_tau3_model_identity
from .tau3_training_artifacts import REQUIRED_ARTIFACT_MAP, validate_tau3_training_bundle
from .tau3_training_mixture import TAU3_TRAINING_MIXTURE_SCHEMA_VERSION

TAU3_MLX_TRAINING_RUN_SCHEMA_VERSION = "hfr.tau3_mlx_training_run.v1"
MAX_TIMEOUT_SECONDS = 604_800
MAX_ITERS = 2_000_000
MAX_RANK = 256
MAX_BATCH_SIZE = 64
MAX_GRAD_ACCUMULATION = 512
MAX_SEQ_LENGTH = 65_536
LOSS_RE = re.compile(r"\b(?P<kind>train|training|valid|validation|val)(?:[_ -]?loss)?\b[^0-9+-]*(?P<loss>[+-]?(?:\d+(?:\.\d*)?|\.\d+))", re.IGNORECASE)
SECRET_RE = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{8,}|hf_[A-Za-z0-9]{8,})\b")
FORBIDDEN_TOKEN_FRAGMENTS = (
    "--push-to-hub",
    "--report-to",
    "--wandb",
    "--use-dora",
    "--dora",
    "--full",
    "--fine-tune-type=full",
    "--fine_tune_type=full",
    "--allow-network",
    "--hub-token",
    "--api-key",
)
FORBIDDEN_DATA_FRAGMENTS = (
    "known info",
    "task instructions",
    "evaluation criteria",
    "user_scenario",
    "user scenario",
    "hidden user",
    "user simulator",
    "you are simulating",
    "known_info",
    "evaluation_criteria",
    "invented_tau_tool",
    "meta final",
    "synthetic final",
    "final_response",
    "check that the agent",
    "check whether the agent",
)


class Tau3MlxTrainingError(ValueError):
    """Raised when local Tau-3 MLX training cannot be launched safely."""


@dataclass(frozen=True)
class Tau3MlxTrainingConfig:
    """Bounded hyperparameters for the local MLX-LM LoRA subprocess."""

    iters: int = 100
    learning_rate: float = 1e-5
    rank: int = 16
    scale: float = 20.0
    dropout: float = 0.0
    num_layers: int = 16
    max_seq_length: int = 8192
    batch_size: int = 1
    grad_accumulation: int = 1
    seed: int = 17
    save_every: int = 50
    report_every: int = 10
    eval_every: int = 50
    val_batches: int = -1
    mask_prompt: bool = True
    grad_checkpoint: bool = True
    clear_cache_threshold: int = 0
    timeout_seconds: int = 172_800


def run_tau3_mlx_training(
    *,
    bundle_dir: str | Path | None = None,
    mixture_dir: str | Path | None = None,
    model_identity_path: str | Path | None = None,
    model_path: str | Path | None = None,
    output_dir: str | Path,
    config: Tau3MlxTrainingConfig | None = None,
    workspace_root: str | Path | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Validate a governed Tau-3 dataset source and run local ``mlx_lm lora``.

    Preferred input is one ``tau3_training_mixture`` variant directory plus a
    replayable local model identity JSON. Legacy production bundles are still
    accepted, but they must pass the same direct semantic scan; attestation
    alone cannot authorize training.
    """

    cfg = config or Tau3MlxTrainingConfig()
    root = _resolve_workspace_root(workspace_root)
    if (bundle_dir is None) == (mixture_dir is None):
        raise Tau3MlxTrainingError("provide exactly one of bundle_dir or mixture_dir")
    source_kind = "bundle" if bundle_dir is not None else "mixture"
    source_path = _require_local_directory(Path(bundle_dir or mixture_dir), root, source_kind)
    output = _require_local_output(Path(output_dir), root)
    output.mkdir(parents=True, exist_ok=True)
    adapter_dir = output / "adapter"
    telemetry_path = output / "telemetry.jsonl"
    prelaunch_path = output / "prelaunch_receipt.json"
    final_path = output / "training_receipt.json"

    checks: list[dict[str, Any]] = []
    if source_kind == "bundle":
        validation = validate_tau3_training_bundle(source_path, strict=True)
        _add_check(checks, "strict_bundle_validation_passed", validation.get("passed") is True, validation.get("summary"), "passed")
        if validation.get("passed") is not True:
            raise Tau3MlxTrainingError("strict production bundle validation failed: " + json.dumps(_failed_ids(validation), sort_keys=True))
        payloads = _load_required_payloads(source_path)
        model_ref = _model_ref(payloads)
        data_dir = _resolve_bundle_relative_dir(source_path / "training", _mlx_dataset_path(payloads), "mlx data")
        _check_launch_readiness(source_path, payloads, cfg, root, checks)
    else:
        if model_identity_path is None or model_path is None:
            raise Tau3MlxTrainingError("mixture training requires model_identity_path and model_path")
        model_dir = _require_local_directory(Path(model_path), root, "model")
        identity_file = _require_local_file(Path(model_identity_path), root, "model identity")
        identity = json.loads(identity_file.read_text(encoding="utf-8"))
        model_ref = str(model_dir)
        data_dir = source_path
        _check_mixture_launch_readiness(source_path, identity_file, identity, model_dir, cfg, root, checks)
    if any(not check["passed"] for check in checks):
        raise Tau3MlxTrainingError("prelaunch checks failed: " + json.dumps([c["id"] for c in checks if not c["passed"]], sort_keys=True))

    python = _require_local_venv_python(root)
    _require_local_directory(data_dir, root, "mlx data")
    adapter_dir.mkdir()
    lora_config_path = output / "mlx_lora_config.json"
    _write_new_json_readonly(lora_config_path, _mlx_lora_config(model_ref, data_dir, adapter_dir, cfg))
    command = _build_command(python, model_ref, data_dir, adapter_dir, lora_config_path, cfg)
    _reject_forbidden_tokens(command)

    prelaunch = {
        "schema_version": TAU3_MLX_TRAINING_RUN_SCHEMA_VERSION,
        "phase": "prelaunch",
        "created_at": created_at or _now_utc(),
        "bundle": {"kind": source_kind, **_path_record(source_path)},
        "output_dir": str(output),
        "command": _redact_command(command),
        "config": _config_record(cfg),
        "mlx_lora_config": {"path": str(lora_config_path), "sha256": _sha256_file(lora_config_path), "read_only": not bool(lora_config_path.stat().st_mode & 0o222)},
        "checks": checks,
        "weights_updated": False,
        "terminal_status": "prelaunch",
    }
    _write_new_json_readonly(prelaunch_path, prelaunch)

    status = "crash"
    exit_code: int | None = None
    interrupted = False
    timed_out = False
    losses: dict[str, list[float]] = {"train": [], "validation": []}
    started = time.monotonic()
    telemetry_count = 0
    peak_rss_kb = 0
    try:
        exit_code, timed_out, telemetry_count, peak_rss_kb = _run_child(
            command=command,
            cwd=root,
            telemetry_path=telemetry_path,
            timeout_seconds=cfg.timeout_seconds,
            losses=losses,
        )
        status = _classify(exit_code, timed_out, telemetry_path)
    except KeyboardInterrupt:
        interrupted = True
        status = "interrupted"
    elapsed = time.monotonic() - started
    fingerprints = _fingerprint_tree(adapter_dir)
    adapter_weight_files = [
        record
        for record in fingerprints["files"]
        if record.get("kind") == "adapter" and int(record.get("size") or 0) > 0
    ]
    if status == "success" and not adapter_weight_files:
        status = "no_output"
    weights_updated = status == "success" and bool(adapter_weight_files)

    final = {
        "schema_version": TAU3_MLX_TRAINING_RUN_SCHEMA_VERSION,
        "phase": "final",
        "created_at": created_at or _now_utc(),
        "bundle": {"kind": source_kind, **_path_record(source_path)},
        "output_dir": str(output),
        "prelaunch_receipt": {
            "path": str(prelaunch_path),
            "sha256": _sha256_file(prelaunch_path),
            "read_only": not bool(prelaunch_path.stat().st_mode & 0o222),
        },
        "telemetry": {
            "path": str(telemetry_path),
            "sha256": _sha256_file(telemetry_path) if telemetry_path.exists() else None,
            "event_count": telemetry_count,
        },
        "command": _redact_command(command),
        "config": _config_record(cfg),
        "mlx_lora_config": {"path": str(lora_config_path), "sha256": _sha256_file(lora_config_path), "read_only": not bool(lora_config_path.stat().st_mode & 0o222)},
        "checks": checks,
        "terminal_status": status,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "interrupted": interrupted,
        "elapsed_seconds": round(elapsed, 6),
        "peak_child_rss_kb": peak_rss_kb,
        "losses": {
            "train": losses["train"],
            "validation": losses["validation"],
            "last_train": losses["train"][-1] if losses["train"] else None,
            "last_validation": losses["validation"][-1] if losses["validation"] else None,
        },
        "adapter": fingerprints,
        "adapter_weight_file_count": len(adapter_weight_files),
        "weights_updated": weights_updated,
        "schema_checked": True,
    }
    schema_check = check_schema_contract(final, name_or_id="tau3_mlx_training_run")
    if schema_check["passed"] is not True:
        raise Tau3MlxTrainingError("final receipt violates schema: " + json.dumps(schema_check["errors"], sort_keys=True))
    _write_new_json_readonly(final_path, final)
    return final


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bundle", type=Path)
    source.add_argument("--mixture-dir", type=Path)
    parser.add_argument("--model-identity", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=Tau3MlxTrainingConfig.iters)
    parser.add_argument("--lr", type=float, default=Tau3MlxTrainingConfig.learning_rate)
    parser.add_argument("--rank", type=int, default=Tau3MlxTrainingConfig.rank)
    parser.add_argument("--scale", type=float, default=Tau3MlxTrainingConfig.scale)
    parser.add_argument("--dropout", type=float, default=Tau3MlxTrainingConfig.dropout)
    parser.add_argument("--num-layers", type=int, default=Tau3MlxTrainingConfig.num_layers)
    parser.add_argument("--max-seq-length", type=int, default=Tau3MlxTrainingConfig.max_seq_length)
    parser.add_argument("--batch-size", type=int, default=Tau3MlxTrainingConfig.batch_size)
    parser.add_argument("--grad-accumulation", type=int, default=Tau3MlxTrainingConfig.grad_accumulation)
    parser.add_argument("--seed", type=int, default=Tau3MlxTrainingConfig.seed)
    parser.add_argument("--save-every", type=int, default=Tau3MlxTrainingConfig.save_every)
    parser.add_argument("--report-every", type=int, default=Tau3MlxTrainingConfig.report_every)
    parser.add_argument("--eval-every", type=int, default=Tau3MlxTrainingConfig.eval_every)
    parser.add_argument("--val-batches", type=int, default=Tau3MlxTrainingConfig.val_batches)
    parser.add_argument("--clear-cache-threshold", type=int, default=Tau3MlxTrainingConfig.clear_cache_threshold)
    grad = parser.add_mutually_exclusive_group()
    grad.add_argument("--grad-checkpoint", dest="grad_checkpoint", action="store_true", default=True)
    grad.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false")
    parser.add_argument("--timeout-seconds", type=int, default=Tau3MlxTrainingConfig.timeout_seconds)
    mask = parser.add_mutually_exclusive_group()
    mask.add_argument("--mask-prompt", dest="mask_prompt", action="store_true", default=True)
    mask.add_argument("--no-mask-prompt", dest="mask_prompt", action="store_false")
    return parser


def config_from_args(args: argparse.Namespace) -> Tau3MlxTrainingConfig:
    return Tau3MlxTrainingConfig(
        iters=args.iters,
        learning_rate=args.lr,
        rank=args.rank,
        scale=args.scale,
        dropout=args.dropout,
        num_layers=args.num_layers,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        grad_accumulation=args.grad_accumulation,
        seed=args.seed,
        save_every=args.save_every,
        report_every=args.report_every,
        eval_every=args.eval_every,
        val_batches=args.val_batches,
        mask_prompt=args.mask_prompt,
        grad_checkpoint=args.grad_checkpoint,
        clear_cache_threshold=args.clear_cache_threshold,
        timeout_seconds=args.timeout_seconds,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        receipt = run_tau3_mlx_training(
            bundle_dir=args.bundle,
            mixture_dir=args.mixture_dir,
            model_identity_path=args.model_identity,
            model_path=args.model_path,
            output_dir=args.out,
            config=config_from_args(args),
        )
    except (OSError, Tau3MlxTrainingError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps({"receipt": str(Path(args.out) / "training_receipt.json"), "terminal_status": receipt["terminal_status"], "weights_updated": receipt["weights_updated"]}, indent=2, sort_keys=True))
    return 0 if receipt["weights_updated"] else 1


def _resolve_workspace_root(root: str | Path | None) -> Path:
    path = Path(root) if root is not None else Path.cwd()
    resolved = path.resolve(strict=True)
    if path_has_symlink_component(resolved, include_leaf=True):
        raise Tau3MlxTrainingError(f"workspace root must not contain symlink components: {path}")
    return resolved


def _require_local_directory(path: Path, root: Path, label: str) -> Path:
    unresolved = path if path.is_absolute() else root / path
    if path_has_symlink_component(unresolved, include_leaf=True):
        raise Tau3MlxTrainingError(f"{label} must not contain symlink components: {path}")
    resolved = _resolve_under_root(path, root, label, must_exist=True)
    if not resolved.is_dir():
        raise Tau3MlxTrainingError(f"{label} must be a directory: {path}")
    if path_has_symlink_component(resolved, include_leaf=True):
        raise Tau3MlxTrainingError(f"{label} must not contain symlink components: {path}")
    return resolved


def _require_local_file(path: Path, root: Path, label: str) -> Path:
    unresolved = path if path.is_absolute() else root / path
    if path_has_symlink_component(unresolved, include_leaf=True):
        raise Tau3MlxTrainingError(f"{label} must not contain symlink components: {path}")
    resolved = _resolve_under_root(path, root, label, must_exist=True)
    if not resolved.is_file():
        raise Tau3MlxTrainingError(f"{label} must be a file: {path}")
    if path_has_symlink_component(resolved, include_leaf=True):
        raise Tau3MlxTrainingError(f"{label} must not contain symlink components: {path}")
    return resolved


def _require_local_output(path: Path, root: Path) -> Path:
    resolved = _resolve_under_root(path, root, "output", must_exist=False)
    if path_has_symlink_component(resolved, include_leaf=True):
        raise Tau3MlxTrainingError(f"output must not contain symlink components: {path}")
    if resolved.exists() and (not resolved.is_dir() or any(resolved.iterdir())):
        raise Tau3MlxTrainingError(f"output must be missing or an empty directory: {path}")
    return resolved


def _resolve_under_root(path: Path, root: Path, label: str, *, must_exist: bool) -> Path:
    candidate = path if path.is_absolute() else root / path
    try:
        resolved = candidate.resolve(strict=must_exist)
    except FileNotFoundError as exc:
        raise Tau3MlxTrainingError(f"{label} does not exist: {path}") from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise Tau3MlxTrainingError(f"could not resolve {label}: {path}: {exc}") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise Tau3MlxTrainingError(f"{label} must resolve under workspace root: {path}") from exc
    return resolved


def _require_local_venv_python(root: Path) -> Path:
    python = root / ".venv" / "bin" / "python"
    if path_has_symlink_component(python.parent, include_leaf=True):
        raise Tau3MlxTrainingError(f"local virtual-environment directory must not traverse symlinks: {python.parent}")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise Tau3MlxTrainingError(f"local MLX training requires executable {python}")
    try:
        resolved = python.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise Tau3MlxTrainingError(f"local Python symlink could not be resolved safely: {python}") from exc
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise Tau3MlxTrainingError(f"resolved local Python is not executable: {resolved}")
    return resolved


def _load_required_payloads(bundle: Path) -> dict[str, Any]:
    payloads: dict[str, Any] = {}
    for role, rel_path in REQUIRED_ARTIFACT_MAP.items():
        path = bundle / rel_path
        if path_has_symlink_component(path, include_leaf=True):
            raise Tau3MlxTrainingError(f"required artifact must not be symlinked: {rel_path}")
        if path.suffix == ".json":
            payloads[role] = json.loads(path.read_text(encoding="utf-8"))
    payloads["manifest"] = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    return payloads


def _check_launch_readiness(bundle: Path, payloads: dict[str, Any], cfg: Tau3MlxTrainingConfig, root: Path, checks: list[dict[str, Any]]) -> None:
    manifest = payloads["manifest"]
    _add_check(checks, "bundle_mode_is_production", manifest.get("bundle_mode") == "production", manifest.get("bundle_mode"), "production")
    _add_check(checks, "bundle_ready_for_training", manifest.get("ready_for_training") is True, manifest.get("ready_for_training"), True)
    model_freeze = payloads["model_freeze"]
    model_manifest = payloads["model_manifest"]
    dataset_manifest = payloads["dataset_manifest"]
    mlx_plan = payloads["mlx_qlora_plan"]
    launch = payloads["trainer_launch_check"]
    budget = payloads["budget"]
    base = model_freeze.get("base_model") if isinstance(model_freeze.get("base_model"), dict) else {}
    base_name = str(base.get("name") or "")
    base_revision = str(base.get("revision") or "")
    _add_check(checks, "base_identity_matches_protocol", str(model_manifest.get("base_model") or model_manifest.get("model_id") or "") == base_name and str(model_manifest.get("revision") or "") == base_revision, {"model_manifest": model_manifest, "protocol_base": base}, "same base model and revision")
    _add_check(checks, "dataset_manifest_local_only", dataset_manifest.get("local_only") is True or not _truthy(dataset_manifest, "allow_network", "network"), _summary(dataset_manifest), "local only")
    _add_check(checks, "dataset_views_train_valid_only", _dataset_views_train_valid_only(dataset_manifest), _summary(dataset_manifest.get("views")), "mlx train/valid only for trainer")
    _add_check(checks, "mlx_dataset_hashes_replay", _mlx_dataset_hashes_replay(bundle / "training", dataset_manifest), _summary(dataset_manifest.get("mlx_dataset_manifest")), "current MLX dataset hashes")
    direct_scan = _scan_mlx_data_dir(bundle / "training" / _mlx_dataset_path(payloads))
    _add_check(checks, "training_target_quality_direct_semantic_scan", direct_scan["passed"], direct_scan, "no evaluator/meta/tool leakage in train/valid rows")
    _add_check(checks, "training_target_quality_no_eval_criteria_exposure", _training_target_quality_passed(bundle, payloads), _training_target_quality_summary(payloads), "computed no exposure")
    method_text = json.dumps({"mlx": mlx_plan, "launch": launch}, sort_keys=True).lower()
    _add_check(checks, "qlora_lora_only_no_full_or_dora", "dora" not in method_text and "full" not in method_text and "qlora" in method_text and "lora" in method_text, _summary(mlx_plan), "QLoRA/LoRA without full or DoRA")
    _add_check(checks, "plan_uses_development_not_sealed", not _truthy(mlx_plan, "sealed_used", "test_used") and not _truthy(payloads["candidate_selection_contract"], "sealed_used", "test_used"), _summary(mlx_plan), "no sealed/test trainer use")
    planned_command = mlx_plan.get("command_argv") if isinstance(mlx_plan.get("command_argv"), list) else []
    launch_command = _extract_launch_command(launch)
    _add_check(checks, "frozen_launch_command_has_no_forbidden_flags", not _contains_forbidden(planned_command) and not _contains_forbidden(launch_command), {"plan": planned_command, "launch": launch_command}, "no network/report/push/full/dora flags")
    max_seconds = _number(budget.get("max_seconds"))
    training_budget = _number((budget.get("stages") or {}).get("training")) if isinstance(budget.get("stages"), dict) else None
    allowed_seconds = min(value for value in (max_seconds, training_budget, MAX_TIMEOUT_SECONDS) if value is not None)
    _add_check(checks, "timeout_within_budget", cfg.timeout_seconds <= allowed_seconds, cfg.timeout_seconds, f"<= {allowed_seconds}")
    _add_check(checks, "hyperparameters_within_bounds", _config_within_bounds(cfg), _config_record(cfg), "bounded local training hyperparameters")
    for role, rel_path in REQUIRED_ARTIFACT_MAP.items():
        artifact = bundle / rel_path
        _add_check(checks, f"artifact_local_regular:{role}", artifact.is_file() and not path_has_symlink_component(artifact, include_leaf=True) and artifact.resolve().is_relative_to(root), str(artifact), "regular local file under workspace")


def _check_mixture_launch_readiness(mixture: Path, identity_path: Path, identity: dict[str, Any], model_dir: Path, cfg: Tau3MlxTrainingConfig, root: Path, checks: list[dict[str, Any]]) -> None:
    manifest_path = mixture / "manifest.json"
    if not manifest_path.is_file():
        _add_check(checks, "mixture_manifest_present", False, str(manifest_path), "manifest.json")
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = check_schema_contract(manifest, name_or_id=TAU3_TRAINING_MIXTURE_SCHEMA_VERSION)
    _add_check(checks, "mixture_manifest_schema_passed", schema.get("passed") is True, schema.get("errors"), "passed")
    _add_check(checks, "mixture_variant_not_root_set", manifest.get("variant") != "mixture_set", manifest.get("variant"), "single trainable variant")
    _add_check(checks, "mixture_passed", manifest.get("passed") is True, manifest.get("passed"), True)
    _add_check(checks, "mixture_no_sealed_or_test_rows", manifest.get("sealed_rows") == 0 and manifest.get("test_rows") == 0, {"sealed": manifest.get("sealed_rows"), "test": manifest.get("test_rows")}, {"sealed": 0, "test": 0})
    _add_check(checks, "mixture_not_already_training_started", manifest.get("training_started") is False, manifest.get("training_started"), False)
    _add_check(checks, "mixture_files_replay", _mixture_files_replay(mixture, manifest), _summary(manifest.get("files")), "current train/valid hashes")
    _add_check(checks, "mixture_source_hashes_replay", _mixture_source_hashes_replay(mixture, manifest), _summary(manifest.get("source_binding")), "current source hashes")
    direct_scan = _scan_mlx_data_dir(mixture)
    _add_check(checks, "training_target_quality_direct_semantic_scan", direct_scan["passed"], direct_scan, "no evaluator/meta/tool leakage in train/valid rows")
    model_id = str(identity.get("model_id") or "")
    revision = str(identity.get("revision") or "")
    identity_errors = validate_tau3_model_identity(identity, model_dir, expected_model_id=model_id, expected_revision=revision)
    _add_check(checks, "model_identity_replays", not identity_errors, {"path": str(identity_path), "errors": identity_errors, "model_id": model_id, "revision": revision}, "identity fully replays local model tree")
    _add_check(checks, "model_local_regular_under_workspace", model_dir.is_dir() and model_dir.resolve().is_relative_to(root), str(model_dir), "local model directory under workspace")
    _add_check(checks, "timeout_within_budget", cfg.timeout_seconds <= MAX_TIMEOUT_SECONDS, cfg.timeout_seconds, f"<= {MAX_TIMEOUT_SECONDS}")
    _add_check(checks, "hyperparameters_within_bounds", _config_within_bounds(cfg), _config_record(cfg), "bounded local training hyperparameters")


def _mixture_files_replay(mixture: Path, manifest: dict[str, Any]) -> bool:
    files = manifest.get("files")
    if not isinstance(files, dict):
        return False
    for name in ("train", "valid"):
        record = files.get(name)
        if not isinstance(record, dict) or record.get("path") != f"{name}.jsonl":
            return False
        path = mixture / f"{name}.jsonl"
        if not path.is_file() or path_has_symlink_component(path, include_leaf=True):
            return False
        if record.get("size") != path.stat().st_size or record.get("sha256") != _sha256_file(path):
            return False
    return True


def _mixture_source_hashes_replay(mixture: Path, manifest: dict[str, Any]) -> bool:
    binding = manifest.get("source_binding")
    if not isinstance(binding, dict):
        return False
    source_dir_value = binding.get("source_dir")
    source_root = Path(source_dir_value) if isinstance(source_dir_value, str) and source_dir_value else mixture.parent.parent
    for name in ("train", "valid"):
        record = binding.get(name)
        if not isinstance(record, dict) or record.get("path") != f"{name}.jsonl":
            return False
        path = source_root / f"{name}.jsonl"
        if not path.is_file() or path_has_symlink_component(path, include_leaf=True):
            return False
        if record.get("sha256") != _sha256_file(path):
            return False
    return True


def _scan_mlx_data_dir(data_dir: Path) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    row_count = 0
    for split in ("train", "valid"):
        path = data_dir / f"{split}.jsonl"
        if not path.is_file():
            findings.append({"split": split, "line": 0, "reason": "missing split file"})
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row_count += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                findings.append({"split": split, "line": line_number, "reason": f"invalid JSON: {exc.msg}"})
                continue
            for hit in _semantic_leak_hits(row):
                findings.append({"split": split, "line": line_number, **hit})
    return {"passed": not findings and row_count > 0, "row_count": row_count, "finding_count": len(findings), "findings": findings[:25]}


def _semantic_leak_hits(value: Any, path: str = "$") -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            lowered_key = key_text.lower()
            for fragment in FORBIDDEN_DATA_FRAGMENTS:
                if fragment in lowered_key:
                    hits.append({"path": f"{path}.{key_text}", "reason": f"forbidden key fragment: {fragment}"})
            hits.extend(_semantic_leak_hits(item, f"{path}.{key_text}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            hits.extend(_semantic_leak_hits(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        for fragment in FORBIDDEN_DATA_FRAGMENTS:
            if fragment in lowered:
                hits.append({"path": path, "reason": f"forbidden text fragment: {fragment}"})
    return hits


def _dataset_views_train_valid_only(dataset_manifest: dict[str, Any]) -> bool:
    views = dataset_manifest.get("views")
    if not isinstance(views, dict):
        return False
    for forbidden in ("sealed", "test", "mlx_test", "mlx_sealed"):
        if forbidden in views:
            return False
    return all(name in views for name in ("mlx_train", "mlx_valid"))


def _mlx_dataset_hashes_replay(training_dir: Path, dataset_manifest: dict[str, Any]) -> bool:
    manifest_ref = dataset_manifest.get("mlx_dataset_manifest") if isinstance(dataset_manifest.get("mlx_dataset_manifest"), dict) else {}
    rel = manifest_ref.get("path")
    if not isinstance(rel, str):
        return False
    manifest_path = _resolve_bundle_relative_file(training_dir, rel, "MLX dataset manifest")
    if manifest_ref.get("sha256") != _sha256_file(manifest_path):
        return False
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("sealed_rows") != 0 or payload.get("test_file_present") is not False:
        return False
    files = payload.get("files")
    if not isinstance(files, dict):
        return False
    base = manifest_path.parent
    for name in ("train", "valid"):
        record = files.get(name)
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            return False
        path = _resolve_bundle_relative_file(base, record["path"], f"MLX {name} file")
        if record.get("sha256") != _sha256_file(path):
            return False
    return True


def _training_target_quality_summary(payloads: dict[str, Any]) -> Any:
    for payload in (payloads.get("dataset_manifest"), payloads.get("mlx_qlora_plan"), payloads.get("trainer_preflight")):
        if isinstance(payload, dict) and isinstance(payload.get("training_target_quality"), dict):
            return payload["training_target_quality"]
    return "missing training_target_quality attestation and computable local source envelopes"


def _training_target_quality_passed(bundle: Path, payloads: dict[str, Any]) -> bool:
    source_paths = _source_envelope_paths(bundle, payloads)
    if not source_paths:
        return _scan_mlx_data_dir(bundle / "training" / _mlx_dataset_path(payloads))["passed"]
    criteria = _evaluation_criteria_from_sources(source_paths)
    if not criteria:
        return False
    targets = _assistant_targets_from_mlx_views(bundle, payloads)
    return not _targets_expose_criteria(targets, criteria)


def _quality_attestation_passed(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return (
        value.get("passed") is True
        and value.get("evaluation_criteria_exposure") is False
        and int(value.get("exact_match_count") or 0) == 0
        and int(value.get("substantial_exposure_count") or 0) == 0
    )


def _source_envelope_paths(bundle: Path, payloads: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for payload in (payloads.get("protocol_manifest"), payloads.get("split_manifest"), payloads.get("dataset_manifest")):
        for rel in _find_source_path_values(payload):
            try:
                path = _resolve_bundle_relative_file(bundle, rel, "training source envelope")
            except Tau3MlxTrainingError:
                continue
            if path.name in {"train_tasks.jsonl", "development_tasks.jsonl"} or "training_source" in path.as_posix():
                paths.append(path)
    unique: dict[str, Path] = {}
    for path in paths:
        unique[str(path)] = path
    return list(unique.values())


def _find_source_path_values(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower()
            if (
                isinstance(item, str)
                and (
                    key_text in {"source_path", "train_tasks", "development_tasks", "train_source", "development_source"}
                    or key_text.endswith("_source_path")
                    or key_text.endswith("_tasks_path")
                    or item.endswith(("train_tasks.jsonl", "development_tasks.jsonl"))
                )
            ):
                found.append(item)
            else:
                found.extend(_find_source_path_values(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_find_source_path_values(item))
    return found


def _evaluation_criteria_from_sources(paths: list[Path]) -> list[str]:
    criteria: list[str] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            criteria.extend(_criteria_strings(row))
    return [item for item in criteria if item.strip()]


def _criteria_strings(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key) == "evaluation_criteria":
                found.extend(_string_leaves(item))
            else:
                found.extend(_criteria_strings(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_criteria_strings(item))
    return found


def _string_leaves(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        items: list[str] = []
        for item in value.values():
            items.extend(_string_leaves(item))
        return items
    if isinstance(value, list):
        items = []
        for item in value:
            items.extend(_string_leaves(item))
        return items
    return []


def _assistant_targets_from_mlx_views(bundle: Path, payloads: dict[str, Any]) -> list[str]:
    data_dir = _resolve_bundle_relative_dir(bundle / "training", _mlx_dataset_path(payloads), "mlx data")
    targets: list[str] = []
    for name in ("train.jsonl", "valid.jsonl"):
        path = data_dir / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages")
            if isinstance(messages, list):
                for message in messages:
                    if isinstance(message, dict) and message.get("role") == "assistant":
                        content = message.get("content")
                        if isinstance(content, str):
                            targets.append(content)
    return targets


def _targets_expose_criteria(targets: list[str], criteria: list[str]) -> bool:
    normalized_targets = [_normalize_text(target) for target in targets]
    for criterion in criteria:
        norm_criterion = _normalize_text(criterion)
        if not norm_criterion:
            continue
        criterion_tokens = set(norm_criterion.split())
        for target in normalized_targets:
            if target == norm_criterion or (len(norm_criterion) >= 40 and norm_criterion in target):
                return True
            target_tokens = set(target.split())
            if len(criterion_tokens) >= 8 and len(criterion_tokens & target_tokens) / len(criterion_tokens) >= 0.8:
                return True
    return False


def _normalize_text(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())


def _mlx_dataset_path(payloads: dict[str, Any]) -> str:
    record = payloads["dataset_manifest"].get("mlx_dataset_manifest")
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise Tau3MlxTrainingError("dataset_manifest must bind mlx_dataset_manifest.path")
    manifest_path = record["path"]
    parent = Path(manifest_path).parent.as_posix()
    return parent if parent != "." else ""


def _resolve_bundle_relative_file(base: Path, rel: str, label: str) -> Path:
    path = _resolve_bundle_relative_path(base, rel, label)
    if not path.is_file():
        raise Tau3MlxTrainingError(f"{label} must be a file: {rel}")
    return path


def _resolve_bundle_relative_dir(base: Path, rel: str, label: str) -> Path:
    path = _resolve_bundle_relative_path(base, rel, label)
    if not path.is_dir():
        raise Tau3MlxTrainingError(f"{label} must be a directory: {rel}")
    return path


def _resolve_bundle_relative_path(base: Path, rel: str, label: str) -> Path:
    raw = Path(rel)
    if raw.is_absolute() or ".." in raw.parts or not rel:
        if rel:
            raise Tau3MlxTrainingError(f"{label} must be a relative path below its manifest root: {rel}")
    path = (base / raw).resolve(strict=True)
    try:
        path.relative_to(base.resolve(strict=True))
    except ValueError as exc:
        raise Tau3MlxTrainingError(f"{label} escapes its manifest root: {rel}") from exc
    if path_has_symlink_component(path, include_leaf=True):
        raise Tau3MlxTrainingError(f"{label} must not contain symlink components: {rel}")
    return path


def _model_ref(payloads: dict[str, Any]) -> str:
    model_manifest = payloads["model_manifest"]
    return str(model_manifest.get("model_id") or model_manifest.get("base_model") or "")


def _mlx_lora_config(model: str, data_dir: Path, adapter_dir: Path, cfg: Tau3MlxTrainingConfig) -> dict[str, Any]:
    return {
        "model": model,
        "train": True,
        "fine_tune_type": "lora",
        "data": str(data_dir),
        "adapter_path": str(adapter_dir),
        "iters": cfg.iters,
        "learning_rate": cfg.learning_rate,
        "num_layers": cfg.num_layers,
        "batch_size": cfg.batch_size,
        "grad_accumulation_steps": cfg.grad_accumulation,
        "steps_per_report": cfg.report_every,
        "steps_per_eval": cfg.eval_every,
        "val_batches": cfg.val_batches,
        "save_every": cfg.save_every,
        "max_seq_length": cfg.max_seq_length,
        "seed": cfg.seed,
        "mask_prompt": cfg.mask_prompt,
        "grad_checkpoint": cfg.grad_checkpoint,
        "clear_cache_threshold": cfg.clear_cache_threshold,
        "report_to": None,
        "test": False,
        "lora_parameters": {
            "rank": cfg.rank,
            "scale": cfg.scale,
            "dropout": cfg.dropout,
        },
    }


def _build_command(python: Path, model: str, data_dir: Path, adapter_dir: Path, config_path: Path, cfg: Tau3MlxTrainingConfig) -> list[str]:
    command = [
        str(python),
        "-m",
        "mlx_lm",
        "lora",
        "--config",
        str(config_path),
        "--model",
        model,
        "--train",
        "--data",
        str(data_dir),
        "--adapter-path",
        str(adapter_dir),
        "--fine-tune-type",
        "lora",
        "--iters",
        str(cfg.iters),
        "--learning-rate",
        str(cfg.learning_rate),
        "--num-layers",
        str(cfg.num_layers),
        "--max-seq-length",
        str(cfg.max_seq_length),
        "--batch-size",
        str(cfg.batch_size),
        "--grad-accumulation-steps",
        str(cfg.grad_accumulation),
        "--steps-per-report",
        str(cfg.report_every),
        "--steps-per-eval",
        str(cfg.eval_every),
        "--val-batches",
        str(cfg.val_batches),
        "--seed",
        str(cfg.seed),
        "--save-every",
        str(cfg.save_every),
        "--clear-cache-threshold",
        str(cfg.clear_cache_threshold),
    ]
    if cfg.mask_prompt:
        command.append("--mask-prompt")
    if cfg.grad_checkpoint:
        command.append("--grad-checkpoint")
    return command


def _run_child(*, command: list[str], cwd: Path, telemetry_path: Path, timeout_seconds: int, losses: dict[str, list[float]]) -> tuple[int | None, bool, int, int]:
    proc = subprocess.Popen(command, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    events: queue.Queue[tuple[str, str | None]] = queue.Queue()
    for stream_name, stream in (("stdout", proc.stdout), ("stderr", proc.stderr)):
        thread = threading.Thread(target=_reader, args=(stream_name, stream, events), daemon=True)
        thread.start()
    deadline = time.monotonic() + timeout_seconds
    count = 0
    timed_out = False
    peak_rss_kb = 0
    next_rss_sample = 0.0
    with telemetry_path.open("x", encoding="utf-8") as telemetry:
        while True:
            now = time.monotonic()
            if now >= next_rss_sample:
                peak_rss_kb = max(peak_rss_kb, _process_rss_kb(proc.pid))
                next_rss_sample = now + 1.0
            if proc.poll() is not None:
                while not events.empty():
                    item = events.get()
                    if item[1] is not None:
                        count += _write_telemetry(telemetry, item[0], item[1] or "", losses)
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                _terminate_process(proc)
                break
            try:
                stream_name, line = events.get(timeout=min(0.05, remaining))
            except queue.Empty:
                continue
            if line is not None:
                count += _write_telemetry(telemetry, stream_name, line, losses)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        while not events.empty():
            stream_name, line = events.get()
            if line is not None:
                count += _write_telemetry(telemetry, stream_name, line, losses)
        if timed_out:
            telemetry.write(json.dumps({"time": _now_utc(), "stream": "system", "text": "training subprocess timed out"}, sort_keys=True) + "\n")
            count += 1
    for stream in (proc.stdout, proc.stderr):
        if stream is not None:
            stream.close()
    return proc.returncode, timed_out, count, peak_rss_kb


def _reader(stream_name: str, stream: Any, events: queue.Queue[tuple[str, str | None]]) -> None:
    if stream is None:
        return
    for line in stream:
        events.put((stream_name, line.rstrip("\n")))
    events.put((stream_name, None))


def _write_telemetry(handle: Any, stream_name: str, line: str, losses: dict[str, list[float]]) -> int:
    text = _redact_text(line)
    for match in LOSS_RE.finditer(text):
        value = float(match.group("loss"))
        kind = match.group("kind").lower()
        losses["validation" if kind in {"valid", "validation", "val"} else "train"].append(value)
    handle.write(json.dumps({"time": _now_utc(), "stream": stream_name, "text": text}, sort_keys=True) + "\n")
    return 1


def _terminate_process(proc: subprocess.Popen[str]) -> None:
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
    except ProcessLookupError:
        return


def _classify(exit_code: int | None, timed_out: bool, telemetry_path: Path) -> str:
    if timed_out:
        return "timeout"
    text = telemetry_path.read_text(encoding="utf-8") if telemetry_path.exists() else ""
    if "out of memory" in text.lower() or "oom" in text.lower():
        return "oom"
    if exit_code == 0:
        return "success"
    return "crash"


def _fingerprint_tree(root: Path) -> dict[str, Any]:
    files = []
    if root.is_dir():
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            rel = path.relative_to(root).as_posix()
            files.append({"path": rel, "size": path.stat().st_size, "sha256": _sha256_file(path), "kind": _fingerprint_kind(rel)})
    digest = hashlib.sha256()
    for record in files:
        digest.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return {"path": str(root), "file_count": len(files), "files": files, "tree_sha256": digest.hexdigest() if files else None}


def _fingerprint_kind(rel: str) -> str:
    name = Path(rel).name
    if name in {"adapter_config.json", "config.json"}:
        return "config"
    if "checkpoint" in rel.lower():
        return "checkpoint"
    if Path(rel).suffix in {".safetensors", ".npz", ".bin"}:
        return "adapter"
    return "artifact"


def _write_new_json_readonly(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    path.chmod(0o444)


def _config_within_bounds(cfg: Tau3MlxTrainingConfig) -> bool:
    return (
        1 <= cfg.iters <= MAX_ITERS
        and 0 < cfg.learning_rate <= 1
        and 1 <= cfg.rank <= MAX_RANK
        and cfg.scale > 0
        and 0 <= cfg.dropout < 1
        and 1 <= cfg.num_layers <= 256
        and 1 <= cfg.max_seq_length <= MAX_SEQ_LENGTH
        and 1 <= cfg.batch_size <= MAX_BATCH_SIZE
        and 1 <= cfg.grad_accumulation <= MAX_GRAD_ACCUMULATION
        and 1 <= cfg.save_every <= MAX_ITERS
        and 1 <= cfg.report_every <= MAX_ITERS
        and 1 <= cfg.eval_every <= MAX_ITERS
        and (-1 <= cfg.val_batches <= MAX_ITERS)
        and cfg.clear_cache_threshold >= 0
        and 1 <= cfg.timeout_seconds <= MAX_TIMEOUT_SECONDS
    )


def _config_record(cfg: Tau3MlxTrainingConfig) -> dict[str, Any]:
    return {
        "iters": cfg.iters,
        "learning_rate": cfg.learning_rate,
        "rank": cfg.rank,
        "scale": cfg.scale,
        "dropout": cfg.dropout,
        "num_layers": cfg.num_layers,
        "max_seq_length": cfg.max_seq_length,
        "batch_size": cfg.batch_size,
        "grad_accumulation": cfg.grad_accumulation,
        "seed": cfg.seed,
        "save_every": cfg.save_every,
        "report_every": cfg.report_every,
        "eval_every": cfg.eval_every,
        "val_batches": cfg.val_batches,
        "mask_prompt": cfg.mask_prompt,
        "grad_checkpoint": cfg.grad_checkpoint,
        "clear_cache_threshold": cfg.clear_cache_threshold,
        "timeout_seconds": cfg.timeout_seconds,
    }


def _extract_launch_command(payload: dict[str, Any]) -> list[str]:
    for key in ("command_argv", "argv"):
        value = payload.get(key)
        if isinstance(value, list):
            return [str(item) for item in value]
    value = payload.get("command")
    if isinstance(value, str):
        return value.split()
    return []


def _reject_forbidden_tokens(tokens: list[str]) -> None:
    if _contains_forbidden(tokens):
        raise Tau3MlxTrainingError("subprocess command contains forbidden MLX/network/reporting flag")


def _contains_forbidden(tokens: list[Any]) -> bool:
    lowered = " ".join(str(token).lower() for token in tokens)
    return any(fragment in lowered for fragment in FORBIDDEN_TOKEN_FRAGMENTS) or any(endpoint in lowered for endpoint in ("http://", "https://", "wandb", "huggingface.co"))


def _add_check(checks: list[dict[str, Any]], check_id: str, passed: bool, actual: Any, expected: Any) -> None:
    checks.append({"id": check_id, "passed": bool(passed), "actual": _json_safe(actual), "expected": _json_safe(expected)})


def _failed_ids(validation: dict[str, Any]) -> list[str]:
    return [str(check.get("id")) for check in validation.get("checks", []) if isinstance(check, dict) and check.get("passed") is not True]


def _path_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": _sha256_tree(path) if path.is_dir() else _sha256_file(path)}


def _sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(child for child in path.rglob("*") if child.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(_sha256_file(item).encode("ascii"))
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _redact_command(command: list[str]) -> list[str]:
    return [_redact_text(value) for value in command]


def _redact_text(value: str) -> str:
    return SECRET_RE.sub("<redacted>", value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _truthy(payload: dict[str, Any], *keys: str) -> bool:
    return any(payload.get(key) is True for key in keys)


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _summary(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: value[key] for key in sorted(value)[:8]}
    return value


def _process_rss_kb(pid: int) -> int:
    try:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return int(result.stdout.strip()) if result.returncode == 0 and result.stdout.strip().isdigit() else 0
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


if __name__ == "__main__":
    raise SystemExit(main())
