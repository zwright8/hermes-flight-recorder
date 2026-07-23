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
LOSS_RE = re.compile(
    r"\b(?P<kind>train|training|valid|validation|val)[_ -]*loss\b\s*[:=]?\s*(?P<loss>[+-]?(?:\d+(?:\.\d*)?|\.\d+))",
    re.IGNORECASE,
)
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
    protocol_path: str | Path | None = None,
    model_identity_path: str | Path | None = None,
    model_path: str | Path | None = None,
    output_dir: str | Path,
    config: Tau3MlxTrainingConfig | None = None,
    resume_receipt_path: str | Path | None = None,
    resume_adapter_file: str | Path | None = None,
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
    if bundle_dir is not None:
        source_kind = "bundle"
        raw_source_path: str | Path = bundle_dir
    else:
        source_kind = "mixture"
        assert mixture_dir is not None
        raw_source_path = mixture_dir
    source_path = _require_local_directory(Path(raw_source_path), root, source_kind)
    output = _require_local_output(Path(output_dir), root)
    output.mkdir(parents=True, exist_ok=True)
    adapter_dir = output / "adapter"
    telemetry_path = output / "telemetry.jsonl"
    prelaunch_path = output / "prelaunch_receipt.json"
    final_path = output / "training_receipt.json"

    checks: list[dict[str, Any]] = []
    training_binding: dict[str, Any] | None = None
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
        if protocol_path is None or model_identity_path is None or model_path is None:
            raise Tau3MlxTrainingError("mixture training requires protocol_path, model_identity_path, and model_path")
        protocol_file = _require_local_file(Path(protocol_path), root, "protocol")
        protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
        model_dir = _require_local_directory(Path(model_path), root, "model")
        identity_file = _require_local_file(Path(model_identity_path), root, "model identity")
        identity = json.loads(identity_file.read_text(encoding="utf-8"))
        model_ref = str(model_dir)
        data_dir = source_path
        training_binding = _check_mixture_launch_readiness(source_path, protocol_file, protocol, identity_file, identity, model_dir, cfg, root, checks)
    if any(not check["passed"] for check in checks):
        raise Tau3MlxTrainingError("prelaunch checks failed: " + json.dumps([c["id"] for c in checks if not c["passed"]], sort_keys=True))
    resume = _validate_resume_binding(
        resume_receipt_path=resume_receipt_path,
        resume_adapter_file=resume_adapter_file,
        root=root,
        source_kind=source_kind,
        source_path=source_path,
        cfg=cfg,
        training_binding=training_binding,
        checks=checks,
    )
    if training_binding is not None and resume is not None:
        training_binding = {**training_binding, "resume": resume}
    if any(not check["passed"] for check in checks):
        raise Tau3MlxTrainingError("prelaunch checks failed: " + json.dumps([c["id"] for c in checks if not c["passed"]], sort_keys=True))

    python = _require_local_venv_python(root)
    _require_local_directory(data_dir, root, "mlx data")
    adapter_dir.mkdir()
    lora_config_path = output / "mlx_lora_config.json"
    _write_new_json_readonly(lora_config_path, _mlx_lora_config(model_ref, data_dir, _relative_output_path(adapter_dir, output), cfg))
    command = _build_command(python, model_ref, data_dir, adapter_dir, lora_config_path, cfg, resume_adapter_file=Path(resume["adapter_file"]["path"]) if resume else None)
    _reject_forbidden_tokens(command)

    prelaunch = {
        "schema_version": TAU3_MLX_TRAINING_RUN_SCHEMA_VERSION,
        "phase": "prelaunch",
        "created_at": created_at or _now_utc(),
        "bundle": {"kind": source_kind, **_path_record(source_path)},
        "output_dir": ".",
        "command": _redact_command(command),
        "config": _config_record(cfg, resume=resume),
        "mlx_lora_config": _output_file_record(lora_config_path, output),
        "training_binding": training_binding,
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
        "output_dir": ".",
        "prelaunch_receipt": _output_file_record(prelaunch_path, output),
        "telemetry": {
            "path": _relative_output_path(telemetry_path, output),
            "sha256": _sha256_file(telemetry_path) if telemetry_path.exists() else None,
            "event_count": telemetry_count,
        },
        "command": _redact_command(command),
        "config": _config_record(cfg, resume=resume),
        "mlx_lora_config": _output_file_record(lora_config_path, output),
        "training_binding": training_binding,
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
        "adapter": {**fingerprints, "path": _relative_output_path(adapter_dir, output)},
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
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--model-identity", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume-receipt", type=Path)
    parser.add_argument("--resume-adapter-file", type=Path)
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
            protocol_path=args.protocol,
            model_identity_path=args.model_identity,
            model_path=args.model_path,
            output_dir=args.out,
            config=config_from_args(args),
            resume_receipt_path=args.resume_receipt,
            resume_adapter_file=args.resume_adapter_file,
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
    # Invoke through the virtual-environment entry point. Executing the resolved
    # base interpreter bypasses pyvenv.cfg and drops the environment's packages.
    return python


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


def _check_mixture_launch_readiness(
    mixture: Path,
    protocol_path: Path,
    protocol: dict[str, Any],
    identity_path: Path,
    identity: dict[str, Any],
    model_dir: Path,
    cfg: Tau3MlxTrainingConfig,
    root: Path,
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest_path = mixture / "manifest.json"
    if not manifest_path.is_file():
        _add_check(checks, "mixture_manifest_present", False, str(manifest_path), "manifest.json")
        return _empty_mixture_binding(protocol_path, identity_path, cfg)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol_sha256 = _sha256_file(protocol_path)
    identity_sha256 = _sha256_file(identity_path)
    protocol_schema = check_schema_contract(protocol, name_or_id="tau3_protocol_config")
    _add_check(checks, "protocol_schema_passed", protocol_schema.get("passed") is True, protocol_schema.get("errors"), "registered tau3_protocol_config schema")
    base = _protocol_base_model(protocol)
    frozen_model_id = str(base.get("name") or "")
    frozen_revision = str(base.get("revision") or "")
    frozen_identity_sha256 = str(base.get("local_identity_sha256") or "")
    frozen_tree_sha256 = str(base.get("local_tree_sha256") or "")
    _add_check(
        checks,
        "protocol_base_identity_declared",
        bool(frozen_model_id and frozen_revision and frozen_identity_sha256 and frozen_tree_sha256),
        {key: base.get(key) for key in ("name", "revision", "local_identity_sha256", "local_tree_sha256")},
        "name, revision, local_identity_sha256, and local_tree_sha256",
    )
    _add_check(
        checks,
        "protocol_base_identity_matches_local_identity",
        identity.get("model_id") == frozen_model_id
        and identity.get("revision") == frozen_revision
        and identity.get("tree_sha256") == frozen_tree_sha256
        and identity_sha256 == frozen_identity_sha256,
        {
            "protocol": {
                "model_id": frozen_model_id,
                "revision": frozen_revision,
                "identity_sha256": frozen_identity_sha256,
                "tree_sha256": frozen_tree_sha256,
            },
            "local_identity": {
                "model_id": identity.get("model_id"),
                "revision": identity.get("revision"),
                "identity_sha256": identity_sha256,
                "tree_sha256": identity.get("tree_sha256"),
            },
        },
        "exact frozen base model, revision, identity hash, and tree hash",
    )
    manifest_protocol_sha = _extract_protocol_sha256(manifest)
    _add_check(
        checks,
        "mixture_manifest_protocol_sha_matches",
        manifest_protocol_sha == protocol_sha256,
        manifest_protocol_sha or "missing protocol SHA provenance",
        protocol_sha256,
    )
    recipe = _recipe_record(cfg)
    recipe_sha256 = _canonical_sha256(recipe)
    recipe_id = f"tau3-mlx-recipe-{recipe_sha256[:16]}"
    recipe_check = _recipe_within_protocol(protocol, cfg, recipe)
    _add_check(checks, "recipe_within_protocol_recipe_space", recipe_check["passed"], recipe_check, "recipe inside frozen recipe_space")
    plan_check = _protocol_mlx_plan_allows_local_adapter_4bit(protocol)
    _add_check(checks, "protocol_mlx_plan_local_4bit_adapter_only", plan_check["passed"], plan_check, "local-only 4-bit adapter-only MLX plan")
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
    identity_errors = validate_tau3_model_identity(identity, model_dir, expected_model_id=frozen_model_id, expected_revision=frozen_revision)
    _add_check(checks, "model_identity_replays", not identity_errors, {"path": str(identity_path), "errors": identity_errors, "model_id": frozen_model_id, "revision": frozen_revision}, "identity fully replays local model tree")
    _add_check(checks, "model_local_regular_under_workspace", model_dir.is_dir() and model_dir.resolve().is_relative_to(root), str(model_dir), "local model directory under workspace")
    _add_check(checks, "timeout_within_budget", cfg.timeout_seconds <= MAX_TIMEOUT_SECONDS, cfg.timeout_seconds, f"<= {MAX_TIMEOUT_SECONDS}")
    _add_check(checks, "hyperparameters_within_bounds", _config_within_bounds(cfg), _config_record(cfg), "bounded local training hyperparameters")
    protocol_signature = _protocol_signature_binding(protocol, protocol_sha256)
    _add_check(
        checks,
        "protocol_signature_binding_is_sha256",
        bool(protocol_signature["protocol_signature"]),
        protocol_signature,
        "64 hex protocol signature or protocol-file content seal",
    )
    return {
        "protocol": {
            "path": str(protocol_path),
            "sha256": protocol_sha256,
            "schema_version": protocol.get("schema_version"),
            **protocol_signature,
            "model_freeze_sha256": _canonical_sha256(protocol.get("model_freeze")),
            "recipe_space_sha256": _canonical_sha256(protocol.get("recipe_space")),
            "mlx_qlora_plan_sha256": _canonical_sha256(protocol.get("mlx_qlora_plan")),
        },
        "model": {
            "path": str(model_dir),
            "identity_path": str(identity_path),
            "identity_sha256": identity_sha256,
            "model_id": identity.get("model_id"),
            "revision": identity.get("revision"),
            "tree_sha256": identity.get("tree_sha256"),
        },
        "dataset": {
            "path": str(mixture),
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
            "files_sha256": _canonical_sha256(manifest.get("files")),
            "source_binding_sha256": _canonical_sha256(manifest.get("source_binding")),
            "declared_protocol_sha256": manifest_protocol_sha,
        },
        "recipe": {
            **recipe,
            "recipe_sha256": recipe_sha256,
            "recipe_id": recipe_id,
        },
    }


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
    source_manifest = binding.get("source_manifest")
    if not isinstance(source_manifest, dict) or source_manifest.get("path") != "manifest.json":
        return False
    source_manifest_path = source_root / "manifest.json"
    if (
        not source_manifest_path.is_file()
        or path_has_symlink_component(source_manifest_path, include_leaf=True)
        or source_manifest.get("sha256") != _sha256_file(source_manifest_path)
    ):
        return False
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


def _empty_mixture_binding(protocol_path: Path, identity_path: Path, cfg: Tau3MlxTrainingConfig) -> dict[str, Any]:
    recipe = _recipe_record(cfg)
    recipe_sha256 = _canonical_sha256(recipe)
    return {
        "protocol": {"path": str(protocol_path), "sha256": _sha256_file(protocol_path) if protocol_path.is_file() else None},
        "model": {"identity_path": str(identity_path), "identity_sha256": _sha256_file(identity_path) if identity_path.is_file() else None},
        "dataset": None,
        "recipe": {**recipe, "recipe_sha256": recipe_sha256, "recipe_id": f"tau3-mlx-recipe-{recipe_sha256[:16]}"},
    }


def _validate_resume_binding(
    *,
    resume_receipt_path: str | Path | None,
    resume_adapter_file: str | Path | None,
    root: Path,
    source_kind: str,
    source_path: Path,
    cfg: Tau3MlxTrainingConfig,
    training_binding: dict[str, Any] | None,
    checks: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if resume_receipt_path is None and resume_adapter_file is None:
        return None
    if resume_receipt_path is None or resume_adapter_file is None:
        _add_check(checks, "resume_inputs_complete", False, {"receipt": resume_receipt_path, "adapter_file": resume_adapter_file}, "both resume receipt and adapter file")
        return None

    receipt_path = _require_local_file(Path(resume_receipt_path), root, "resume receipt")
    adapter_file = _require_local_file(Path(resume_adapter_file), root, "resume adapter file")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    schema = check_schema_contract(receipt, name_or_id="tau3_mlx_training_run")
    _add_check(checks, "resume_receipt_schema_passed", schema.get("passed") is True, schema.get("errors"), "registered tau3_mlx_training_run schema")
    _add_check(
        checks,
        "resume_receipt_final_success",
        receipt.get("phase") == "final"
        and receipt.get("terminal_status") == "success"
        and receipt.get("weights_updated") is True
        and int(receipt.get("adapter_weight_file_count") or 0) > 0,
        {
            "phase": receipt.get("phase"),
            "terminal_status": receipt.get("terminal_status"),
            "weights_updated": receipt.get("weights_updated"),
            "adapter_weight_file_count": receipt.get("adapter_weight_file_count"),
        },
        {"phase": "final", "terminal_status": "success", "weights_updated": True, "adapter_weight_file_count": "> 0"},
    )

    prior_adapter = receipt.get("adapter") if isinstance(receipt.get("adapter"), dict) else {}
    prior_adapter_path_value = prior_adapter.get("path")
    if isinstance(prior_adapter_path_value, str) and prior_adapter_path_value:
        try:
            prior_adapter_candidate = _resolve_receipt_local_path(prior_adapter_path_value, receipt_path)
            prior_adapter_dir = _require_local_directory(prior_adapter_candidate, root, "resume prior adapter")
        except Tau3MlxTrainingError as exc:
            _add_check(checks, "resume_prior_adapter_local", False, str(exc), "local adapter directory under workspace")
            prior_adapter_dir = None
    else:
        _add_check(checks, "resume_prior_adapter_local", False, prior_adapter_path_value, "adapter.path")
        prior_adapter_dir = None

    file_record = None
    current_tree = None
    if prior_adapter_dir is not None:
        current_tree = _fingerprint_tree(prior_adapter_dir)
        _add_check(
            checks,
            "resume_adapter_tree_fingerprint_replays",
            current_tree.get("tree_sha256") == prior_adapter.get("tree_sha256"),
            {"current": current_tree.get("tree_sha256"), "receipt": prior_adapter.get("tree_sha256")},
            "same adapter tree fingerprint",
        )
        try:
            rel = adapter_file.relative_to(prior_adapter_dir).as_posix()
        except ValueError:
            rel = None
        files = prior_adapter.get("files")
        if isinstance(files, list) and rel is not None:
            for record in files:
                if isinstance(record, dict) and record.get("path") == rel:
                    file_record = record
                    break
        _add_check(
            checks,
            "resume_adapter_file_bound_to_prior_fingerprint",
            file_record is not None
            and file_record.get("sha256") == _sha256_file(adapter_file)
            and file_record.get("kind") in {"adapter", "checkpoint"},
            {"relative_path": rel, "receipt_record": file_record, "sha256": _sha256_file(adapter_file)},
            "adapter/checkpoint file listed in prior adapter fingerprint with matching sha256",
        )

    prior_binding = receipt.get("training_binding") if isinstance(receipt.get("training_binding"), dict) else None
    prior_config = receipt.get("config") if isinstance(receipt.get("config"), dict) else {}
    current_config = _config_record(cfg)
    if training_binding is not None:
        _add_check(checks, "resume_training_binding_present", prior_binding is not None, prior_binding, "prior training_binding")
        binding_match = prior_binding is not None and _resume_binding_matches(prior_binding, training_binding)
        _add_check(checks, "resume_protocol_model_dataset_match", binding_match, _resume_binding_summary(prior_binding, training_binding), "same protocol, model, and dataset binding")
    else:
        prior_bundle = receipt.get("bundle") if isinstance(receipt.get("bundle"), dict) else {}
        current_bundle = {"kind": source_kind, **_path_record(source_path)}
        _add_check(checks, "resume_bundle_binding_match", prior_bundle == current_bundle, {"prior": prior_bundle, "current": current_bundle}, "same bundle binding")
    config_match = _resume_config_matches(prior_config, current_config)
    _add_check(checks, "resume_hyperparameters_match", config_match["passed"], config_match, "same hyperparameters except increased iters")

    receipt_sha256 = _sha256_file(receipt_path)
    adapter_sha256 = _sha256_file(adapter_file)
    return {
        "enabled": True,
        "receipt": {
            "path": str(receipt_path),
            "sha256": receipt_sha256,
            "created_at": receipt.get("created_at"),
            "terminal_status": receipt.get("terminal_status"),
        },
        "adapter_file": {
            "path": str(adapter_file),
            "relative_path": file_record.get("path") if isinstance(file_record, dict) else None,
            "sha256": adapter_sha256,
            "kind": file_record.get("kind") if isinstance(file_record, dict) else None,
            "size": adapter_file.stat().st_size,
        },
        "prior_adapter": {
            "path": str(prior_adapter_dir) if prior_adapter_dir is not None else prior_adapter_path_value,
            "tree_sha256": prior_adapter.get("tree_sha256"),
            "verified_tree_sha256": current_tree.get("tree_sha256") if isinstance(current_tree, dict) else None,
        },
        "prior_config_sha256": _canonical_sha256(_resume_comparable_config(prior_config)),
        "current_config_sha256": _canonical_sha256(_resume_comparable_config(current_config)),
    }


def _resume_binding_matches(prior: dict[str, Any], current: dict[str, Any]) -> bool:
    for section in ("protocol", "model", "dataset"):
        if prior.get(section) != current.get(section):
            return False
    prior_recipe_value = prior.get("recipe")
    current_recipe_value = current.get("recipe")
    prior_recipe: dict[str, Any] = prior_recipe_value if isinstance(prior_recipe_value, dict) else {}
    current_recipe: dict[str, Any] = current_recipe_value if isinstance(current_recipe_value, dict) else {}
    return _resume_comparable_recipe(prior_recipe) == _resume_comparable_recipe(current_recipe)


def _resume_binding_summary(prior: dict[str, Any] | None, current: dict[str, Any]) -> dict[str, Any]:
    prior = prior or {}
    return {
        "prior": {
            "protocol_sha256": (prior.get("protocol") or {}).get("sha256") if isinstance(prior.get("protocol"), dict) else None,
            "model_identity_sha256": (prior.get("model") or {}).get("identity_sha256") if isinstance(prior.get("model"), dict) else None,
            "dataset_manifest_sha256": (prior.get("dataset") or {}).get("manifest_sha256") if isinstance(prior.get("dataset"), dict) else None,
            "recipe_sha256": (prior.get("recipe") or {}).get("recipe_sha256") if isinstance(prior.get("recipe"), dict) else None,
        },
        "current": {
            "protocol_sha256": (current.get("protocol") or {}).get("sha256") if isinstance(current.get("protocol"), dict) else None,
            "model_identity_sha256": (current.get("model") or {}).get("identity_sha256") if isinstance(current.get("model"), dict) else None,
            "dataset_manifest_sha256": (current.get("dataset") or {}).get("manifest_sha256") if isinstance(current.get("dataset"), dict) else None,
            "recipe_sha256": (current.get("recipe") or {}).get("recipe_sha256") if isinstance(current.get("recipe"), dict) else None,
        },
    }


def _resume_config_matches(prior: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    failures = []
    comparable_prior = _resume_comparable_config(prior)
    comparable_current = _resume_comparable_config(current)
    for key, current_value in comparable_current.items():
        if comparable_prior.get(key) != current_value:
            failures.append({"field": key, "prior": comparable_prior.get(key), "current": current_value})
    prior_iters = prior.get("iters")
    current_iters = current.get("iters")
    if not isinstance(prior_iters, int) or not isinstance(current_iters, int) or current_iters <= prior_iters:
        failures.append({"field": "iters", "prior": prior_iters, "current": current_iters, "expected": "current iters greater than prior iters"})
    return {"passed": not failures, "failures": failures}


def _resume_comparable_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key not in {"iters", "resume"}}


def _resume_comparable_recipe(recipe: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in recipe.items() if key not in {"iters", "recipe_sha256", "recipe_id"}}


def _protocol_base_model(protocol: dict[str, Any]) -> dict[str, Any]:
    freeze = protocol.get("model_freeze")
    if not isinstance(freeze, dict):
        return {}
    base = freeze.get("base_model")
    return base if isinstance(base, dict) else {}


def _protocol_signature_binding(protocol: dict[str, Any], protocol_sha256: str) -> dict[str, Any]:
    signature, source = _protocol_signature(protocol)
    if signature is not None:
        return {
            "protocol_signature": signature if _is_sha256(signature) else None,
            "protocol_signature_provenance": {
                "source": source,
                "algorithm": "sha256",
            },
        }
    return {
        "protocol_signature": protocol_sha256,
        "protocol_signature_provenance": {
            "source": "protocol_file_sha256_content_seal",
            "algorithm": "sha256",
        },
    }


def _protocol_signature(protocol: dict[str, Any]) -> tuple[str | None, str | None]:
    manifest = protocol.get("protocol_manifest")
    if isinstance(manifest, dict) and isinstance(manifest.get("signature"), str):
        return manifest["signature"], "protocol_manifest.signature"
    if isinstance(protocol.get("signature"), str):
        return str(protocol["signature"]), "protocol.signature"
    return None, None


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _extract_protocol_sha256(value: Any) -> str | None:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if isinstance(item, str) and "protocol" in lowered and ("sha256" in lowered or lowered.endswith("_sha")):
                return item
            if "protocol" in lowered and isinstance(item, dict):
                for nested_key in ("sha256", "protocol_sha256", "config_sha256"):
                    nested = item.get(nested_key)
                    if isinstance(nested, str):
                        return nested
            found = _extract_protocol_sha256(item)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _extract_protocol_sha256(item)
            if found is not None:
                return found
    return None


def _recipe_record(cfg: Tau3MlxTrainingConfig) -> dict[str, Any]:
    return {
        "backend": "mlx-lm",
        "fine_tune_type": "lora",
        "quantization": "4-bit",
        "adapter_only": True,
        "rank": cfg.rank,
        "scale": cfg.scale,
        "dropout": cfg.dropout,
        "learning_rate": cfg.learning_rate,
        "num_layers": cfg.num_layers,
        "max_seq_length": cfg.max_seq_length,
        "batch_size": cfg.batch_size,
        "grad_accumulation": cfg.grad_accumulation,
        "iters": cfg.iters,
        "seed": cfg.seed,
        "mask_prompt": cfg.mask_prompt,
        "grad_checkpoint": cfg.grad_checkpoint,
    }


def _recipe_within_protocol(protocol: dict[str, Any], cfg: Tau3MlxTrainingConfig, recipe: dict[str, Any]) -> dict[str, Any]:
    space = protocol.get("recipe_space")
    if not isinstance(space, dict):
        return {"passed": False, "reason": "missing recipe_space"}
    failures: list[dict[str, Any]] = []
    if space.get("bounded") is not True:
        failures.append({"field": "bounded", "actual": space.get("bounded"), "expected": True})
    if space.get("development_only") is not True:
        failures.append({"field": "development_only", "actual": space.get("development_only"), "expected": True})
    if space.get("sealed_used") is not False:
        failures.append({"field": "sealed_used", "actual": space.get("sealed_used"), "expected": False})
    bounds = space.get("bounds")
    if not isinstance(bounds, dict):
        failures.append({"field": "bounds", "actual": bounds, "expected": "object"})
        bounds = {}
    field_values = {
        "rank": cfg.rank,
        "alpha": cfg.scale,
        "scale": cfg.scale,
        "learning_rate": cfg.learning_rate,
        "sequence_length": cfg.max_seq_length,
        "max_seq_length": cfg.max_seq_length,
        "steps": cfg.iters,
        "iters": cfg.iters,
        "batch_size": cfg.batch_size,
        "grad_accumulation": cfg.grad_accumulation,
        "num_layers": cfg.num_layers,
        "dropout": cfg.dropout,
    }
    for field, value in field_values.items():
        if field in bounds and not _value_allowed_by_bound(value, bounds[field]):
            failures.append({"field": field, "actual": value, "expected": bounds[field]})
    required_groups = (("rank",), ("learning_rate",), ("sequence_length", "max_seq_length"), ("steps", "iters"))
    for names in required_groups:
        if not any(name in bounds for name in names):
            failures.append({"field": "/".join(names), "actual": "missing", "expected": "frozen bound"})
    return {"passed": not failures, "recipe": recipe, "bounds": bounds, "failures": failures}


def _value_allowed_by_bound(value: int | float, bound: Any) -> bool:
    if isinstance(bound, list) and bound:
        numbers = [item for item in bound if isinstance(item, (int, float)) and not isinstance(item, bool)]
        if len(numbers) != len(bound):
            return False
        if len(numbers) == 2:
            return float(numbers[0]) <= float(value) <= float(numbers[1])
        return any(float(value) == float(item) for item in numbers)
    if isinstance(bound, dict):
        minimum = bound.get("min")
        maximum = bound.get("max")
        choices = bound.get("values")
        if isinstance(choices, list):
            return any(float(value) == float(item) for item in choices if isinstance(item, (int, float)) and not isinstance(item, bool))
        if isinstance(minimum, (int, float)) and float(value) < float(minimum):
            return False
        if isinstance(maximum, (int, float)) and float(value) > float(maximum):
            return False
        return isinstance(minimum, (int, float)) or isinstance(maximum, (int, float))
    return False


def _protocol_mlx_plan_allows_local_adapter_4bit(protocol: dict[str, Any]) -> dict[str, Any]:
    plan = protocol.get("mlx_qlora_plan")
    if not isinstance(plan, dict):
        return {"passed": False, "reason": "missing mlx_qlora_plan"}
    output_value = plan.get("output_contract")
    output: dict[str, Any] = output_value if isinstance(output_value, dict) else {}
    text = json.dumps(plan, sort_keys=True).lower()
    failures = []
    if plan.get("local_only") is not True:
        failures.append("local_only must be true")
    if _truthy(plan, "network", "allow_network"):
        failures.append("network must be false")
    if "mlx" not in text:
        failures.append("plan must name MLX")
    if "4-bit" not in text and "4bit" not in text:
        failures.append("plan must require 4-bit")
    if "lora" not in text or "qlora" not in text:
        failures.append("plan must require QLoRA/LoRA")
    if output.get("adapter_only") is not True:
        failures.append("output_contract.adapter_only must be true")
    if "dora" in text or "full" in text or _contains_forbidden(_extract_launch_command(plan)):
        failures.append("plan must not permit full, DoRA, network, reporting, or push flags")
    return {"passed": not failures, "failures": failures, "plan": _summary(plan)}


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
    manifest_ref_value = dataset_manifest.get("mlx_dataset_manifest")
    manifest_ref: dict[str, Any] = manifest_ref_value if isinstance(manifest_ref_value, dict) else {}
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


def _mlx_lora_config(model: str, data_dir: Path, adapter_path: str, cfg: Tau3MlxTrainingConfig) -> dict[str, Any]:
    return {
        "model": model,
        "train": True,
        "fine_tune_type": "lora",
        "data": str(data_dir),
        "adapter_path": adapter_path,
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


def _build_command(
    python: Path,
    model: str,
    data_dir: Path,
    adapter_dir: Path,
    config_path: Path,
    cfg: Tau3MlxTrainingConfig,
    *,
    resume_adapter_file: Path | None = None,
) -> list[str]:
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
    if resume_adapter_file is not None:
        command.extend(["--resume-adapter-file", str(resume_adapter_file)])
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
    handle.flush()
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


def _output_file_record(path: Path, output_root: Path) -> dict[str, Any]:
    return {
        "path": _relative_output_path(path, output_root),
        "sha256": _sha256_file(path),
        "read_only": not bool(path.stat().st_mode & 0o222),
    }


def _relative_output_path(path: Path, output_root: Path) -> str:
    try:
        rel = path.relative_to(output_root).as_posix()
    except ValueError as exc:
        raise Tau3MlxTrainingError(f"generated artifact is outside output directory: {path}") from exc
    if rel in {"", "."} or rel.startswith("../") or "/../" in rel or Path(rel).is_absolute():
        raise Tau3MlxTrainingError(f"generated artifact has unsafe relative path: {rel}")
    return rel


def _resolve_receipt_local_path(value: str, receipt_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if value in {"", "."} or ".." in path.parts:
        raise Tau3MlxTrainingError(f"receipt-local path is unsafe: {value}")
    return receipt_path.parent / path


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


def _config_record(cfg: Tau3MlxTrainingConfig, *, resume: dict[str, Any] | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
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
    if resume is not None:
        record["resume"] = resume
    return record


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


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
