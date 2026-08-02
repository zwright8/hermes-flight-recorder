"""Fail-closed local MLX-LM QLoRA runner for governed Tau-3 mixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import re
import signal
import shutil
import stat
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .path_safety import path_has_symlink_component
from .schema_registry import check_schema_contract
from .tau3_competitive_dataset import (
    TAU3_COMPETITIVE_DATASET_SCHEMA_VERSION,
    TOKENIZER_ASSET_FILENAMES,
    _load_local_tokenizer,
    validate_tau3_competitive_dataset_bundle,
)
from .tau3_exposure import Tau3ExposureError, validate_tau3_exposure_ledger
from .tau3_model_identity import validate_tau3_model_identity
from .tau3_policy_complete_dataset import (
    TAU3_POLICY_COMPLETE_DATASET_SCHEMA_VERSION,
    TAU3_POLICY_COMPLETE_ROW_SCHEMA_VERSION,
    USER_SIMULATOR_PRIVATE_MARKERS,
)
from .tau3_prefix_equivalence import validate_tau3_prefix_equivalence
from .tau3_training_artifacts import REQUIRED_ARTIFACT_MAP, validate_tau3_training_bundle
from .tau3_training_mixture import TAU3_TRAINING_MIXTURE_SCHEMA_VERSION

TAU3_MLX_TRAINING_RUN_SCHEMA_VERSION = "hfr.tau3_mlx_training_run.v1"
TAU3_MLX_PROCESS_SEGMENTS_SCHEMA_VERSION = "hfr.tau3_mlx_process_segments.v1"
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
NONFINITE_LOSS_RE = re.compile(
    r"\b(?:train|training|valid|validation|val)[_ -]*loss\b\s*[:=]?\s*"
    r"[+-]?(?:nan|inf(?:inity)?)\b",
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
    disable_compile: bool = False
    fixed_shape_padding: bool = False
    prefix_cache_training: bool = False
    exposure_ledger_training: bool = False
    clear_cache_threshold: int = 0
    process_segment_iters: int | None = None
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
    exposure_dataset_path: str | Path | None = None,
    exposure_receipt_path: str | Path | None = None,
    exposure_ledger_path: str | Path | None = None,
    prefix_equivalence_path: str | Path | None = None,
    grounded_validator_python: str | Path | None = None,
    workspace_root: str | Path | None = None,
    created_at: str | None = None,
    resume_process_segments: bool = False,
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
    output = _require_local_output(
        Path(output_dir),
        root,
        resume_process_segments=resume_process_segments,
    )
    output.mkdir(
        parents=True,
        exist_ok=resume_process_segments,
    )
    adapter_dir = output / "adapter"
    telemetry_path = output / "telemetry.jsonl"
    prelaunch_path = output / "prelaunch_receipt.json"
    final_path = output / "training_receipt.json"

    checks: list[dict[str, Any]] = []
    training_binding: dict[str, Any] | None = None
    exposure_binding: dict[str, Any] | None = None
    prefix_equivalence_binding: dict[str, Any] | None = None
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
        exposure_binding = _validate_exposure_training_binding(
            dataset_path=exposure_dataset_path,
            receipt_path=exposure_receipt_path,
            ledger_path=exposure_ledger_path,
            training_data_dir=data_dir,
            root=root,
            cfg=cfg,
            checks=checks,
        )
        prefix_equivalence_binding = _validate_prefix_equivalence_binding(
            equivalence_path=prefix_equivalence_path,
            training_data_dir=data_dir,
            protocol_path=protocol_file,
            model_identity_path=identity_file,
            root=root,
            cfg=cfg,
            checks=checks,
        )
        training_binding = _check_mixture_launch_readiness(
            source_path,
            protocol_file,
            protocol,
            identity_file,
            identity,
            model_dir,
            cfg,
            root,
            checks,
            exposure_binding=exposure_binding,
            grounded_validator_python=grounded_validator_python,
        )
    if any(not check["passed"] for check in checks):
        raise Tau3MlxTrainingError("prelaunch checks failed: " + json.dumps([c["id"] for c in checks if not c["passed"]], sort_keys=True))
    if source_kind == "bundle" and (
        prefix_equivalence_path is not None
        or (cfg.prefix_cache_training and cfg.exposure_ledger_training)
    ):
        raise Tau3MlxTrainingError(
            "prefix equivalence is supported only for protocol-bound mixture training"
        )
    if exposure_binding is None:
        exposure_binding = _validate_exposure_training_binding(
            dataset_path=exposure_dataset_path,
            receipt_path=exposure_receipt_path,
            ledger_path=exposure_ledger_path,
            training_data_dir=data_dir,
            root=root,
            cfg=cfg,
            checks=checks,
        )
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
    if training_binding is not None and exposure_binding is not None:
        training_binding = {**training_binding, "exposure": exposure_binding}
    if training_binding is not None and prefix_equivalence_binding is not None:
        training_binding = {
            **training_binding,
            "prefix_equivalence": prefix_equivalence_binding,
        }
    if training_binding is not None and resume is not None:
        training_binding = {**training_binding, "resume": resume}
    if any(not check["passed"] for check in checks):
        raise Tau3MlxTrainingError("prelaunch checks failed: " + json.dumps([c["id"] for c in checks if not c["passed"]], sort_keys=True))

    python = _require_local_venv_python(root)
    _require_local_directory(data_dir, root, "mlx data")
    if resume_process_segments and cfg.process_segment_iters is None:
        raise Tau3MlxTrainingError(
            "resume_process_segments requires process_segment_iters"
        )
    if cfg.process_segment_iters is None:
        adapter_dir.mkdir()
    lora_config_path = output / "mlx_lora_config.json"
    lora_config = _mlx_lora_config(
        model_ref,
        data_dir,
        _relative_output_path(adapter_dir, output),
        cfg,
        exposure_binding=exposure_binding,
        prefix_equivalence_binding=prefix_equivalence_binding,
    )
    if resume_process_segments:
        _require_existing_json_equal(
            lora_config_path,
            lora_config,
            "segmented-resume MLX config",
        )
    else:
        _write_new_json_readonly(lora_config_path, lora_config)
    command = _build_command(
        python,
        model_ref,
        data_dir,
        adapter_dir,
        lora_config_path,
        cfg,
        resume_adapter_file=Path(resume["adapter_file"]["path"]) if resume else None,
        exposure_binding=exposure_binding,
        prefix_equivalence_binding=prefix_equivalence_binding,
    )
    _reject_forbidden_tokens(command)
    if cfg.process_segment_iters is not None and resume is not None:
        raise Tau3MlxTrainingError(
            "process-segmented training cannot use adapter-only resume evidence; "
            "exact optimizer state is required"
        )

    prelaunch = {
        "schema_version": TAU3_MLX_TRAINING_RUN_SCHEMA_VERSION,
        "phase": "prelaunch",
        "created_at": created_at or _now_utc(),
        "bundle": {"kind": source_kind, **_path_record(source_path)},
        "output_dir": ".",
        "command": _redact_command(command),
        "config": _config_record(
            cfg,
            resume=resume,
            exposure_binding=exposure_binding,
            prefix_equivalence_binding=prefix_equivalence_binding,
        ),
        "mlx_lora_config": _output_file_record(lora_config_path, output),
        "training_binding": training_binding,
        "checks": checks,
        "weights_updated": False,
        "terminal_status": "prelaunch",
    }
    if resume_process_segments:
        _validate_segment_resume_prelaunch(
            prelaunch_path,
            prelaunch,
            output,
        )
    else:
        _write_new_json_readonly(prelaunch_path, prelaunch)

    status = "crash"
    exit_code: int | None = None
    interrupted = False
    timed_out = False
    losses: dict[str, list[float]] = {"train": [], "validation": []}
    started = time.monotonic()
    telemetry_count = 0
    peak_rss_kb = 0
    process_segments: dict[str, Any] | None = None
    try:
        if cfg.process_segment_iters is None:
            exit_code, timed_out, telemetry_count, peak_rss_kb = _run_child(
                command=command,
                cwd=root,
                telemetry_path=telemetry_path,
                timeout_seconds=cfg.timeout_seconds,
                losses=losses,
                disable_compile=cfg.disable_compile,
            )
            status = _classify(exit_code, timed_out, telemetry_path)
        else:
            segmented = _run_process_segments(
                command=command,
                cwd=root,
                output_dir=output,
                final_adapter_dir=adapter_dir,
                aggregate_telemetry_path=telemetry_path,
                cfg=cfg,
                losses=losses,
                resume=resume_process_segments,
            )
            status = str(segmented["terminal_status"])
            exit_code = segmented["exit_code"]
            timed_out = bool(segmented["timed_out"])
            telemetry_count = int(segmented["telemetry_event_count"])
            peak_rss_kb = int(segmented["peak_child_rss_kb"])
            process_segments = segmented["process_segments"]
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
        "config": _config_record(
            cfg,
            resume=resume,
            exposure_binding=exposure_binding,
            prefix_equivalence_binding=prefix_equivalence_binding,
        ),
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
    if process_segments is not None:
        final["process_segments"] = process_segments
    schema_check = check_schema_contract(final, name_or_id="tau3_mlx_training_run")
    if schema_check["passed"] is not True:
        raise Tau3MlxTrainingError("final receipt violates schema: " + json.dumps(schema_check["errors"], sort_keys=True))
    if resume_process_segments:
        _freeze_json_publication_partials(final_path)
    if resume_process_segments and os.path.lexists(final_path):
        return _validate_existing_final_training_receipt(
            final_path,
            final,
        )
    _publish_new_json_readonly(final_path, final)
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
    parser.add_argument("--exposure-dataset", type=Path)
    parser.add_argument("--exposure-receipt", type=Path)
    parser.add_argument("--exposure-ledger", type=Path)
    parser.add_argument(
        "--prefix-equivalence",
        type=Path,
        help=(
            "Passing bounded full-gradient versus detached-prefix evidence; "
            "required when prefix-cache and exposure-ledger training are combined"
        ),
    )
    parser.add_argument(
        "--grounded-validator-python",
        type=Path,
        help="Python executable used to replay v3 grounded-generation evidence before MLX training launch",
    )
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
    parser.add_argument(
        "--process-segment-iters",
        type=int,
        default=Tau3MlxTrainingConfig.process_segment_iters,
        help=(
            "Recycle the governed MLX child after this many microbatches while "
            "preserving exact adapter and optimizer state. Supported only by "
            "combined exposure-ledger detached-prefix training."
        ),
    )
    parser.add_argument(
        "--resume-process-segments",
        action="store_true",
        help=(
            "Resume an interrupted process-segmented run in the same output "
            "directory after replaying its immutable prelaunch, plan, and "
            "committed adapter/optimizer chain."
        ),
    )
    grad = parser.add_mutually_exclusive_group()
    grad.add_argument("--grad-checkpoint", dest="grad_checkpoint", action="store_true", default=True)
    grad.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false")
    compile_mode = parser.add_mutually_exclusive_group()
    compile_mode.add_argument("--disable-compile", dest="disable_compile", action="store_true", default=False)
    compile_mode.add_argument("--enable-compile", dest="disable_compile", action="store_false")
    padding_mode = parser.add_mutually_exclusive_group()
    padding_mode.add_argument(
        "--fixed-shape-padding",
        dest="fixed_shape_padding",
        action="store_true",
        default=False,
        help="Pad every batch to max-seq-length so MLX can reuse one compiled graph.",
    )
    padding_mode.add_argument(
        "--dynamic-shape-padding",
        dest="fixed_shape_padding",
        action="store_false",
        help="Pad each batch only to its longest sequence (MLX-LM default).",
    )
    training_objective = parser.add_mutually_exclusive_group()
    training_objective.add_argument(
        "--prefix-cache-training",
        dest="prefix_cache_training",
        action="store_true",
        default=False,
        help=(
            "Materialize the complete masked prompt into a detached cache and "
            "backpropagate only through the supervised assistant suffix."
        ),
    )
    training_objective.add_argument(
        "--full-sequence-training",
        dest="prefix_cache_training",
        action="store_false",
        help="Backpropagate through the complete sequence (MLX-LM default).",
    )
    parser.add_argument(
        "--exposure-ledger-training",
        dest="exposure_ledger_training",
        action="store_true",
        default=False,
        help=(
            "Replay MLX-LM LoRA batches from a validated Tau-3 exposure "
            "ledger; full-gradient by default, or qualified detached-prefix "
            "when combined with --prefix-cache-training."
        ),
    )
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
        disable_compile=args.disable_compile,
        fixed_shape_padding=args.fixed_shape_padding,
        prefix_cache_training=args.prefix_cache_training,
        exposure_ledger_training=args.exposure_ledger_training,
        clear_cache_threshold=args.clear_cache_threshold,
        process_segment_iters=args.process_segment_iters,
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
            exposure_dataset_path=args.exposure_dataset,
            exposure_receipt_path=args.exposure_receipt,
            exposure_ledger_path=args.exposure_ledger,
            prefix_equivalence_path=args.prefix_equivalence,
            grounded_validator_python=args.grounded_validator_python,
            resume_process_segments=args.resume_process_segments,
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


def _require_local_output(
    path: Path,
    root: Path,
    *,
    resume_process_segments: bool = False,
) -> Path:
    resolved = _resolve_under_root(path, root, "output", must_exist=False)
    if path_has_symlink_component(resolved, include_leaf=True):
        raise Tau3MlxTrainingError(f"output must not contain symlink components: {path}")
    if resume_process_segments:
        if not resolved.is_dir():
            raise Tau3MlxTrainingError(
                f"segmented-resume output must be an existing directory: {path}"
            )
        required = (
            resolved / "prelaunch_receipt.json",
            resolved / "mlx_lora_config.json",
            resolved / "process_segments" / "plan.json",
            resolved / "process_segments" / "segments",
        )
        if any(not item.exists() for item in required):
            raise Tau3MlxTrainingError(
                "segmented-resume output is missing immutable recovery artifacts"
            )
        final_receipt = resolved / "training_receipt.json"
        if os.path.lexists(final_receipt) and not _is_immutable_regular_file(
            final_receipt
        ):
            raise Tau3MlxTrainingError(
                "segmented-resume terminal training receipt is mutable or unsafe"
            )
        return resolved
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
    *,
    exposure_binding: dict[str, Any] | None = None,
    grounded_validator_python: str | Path | None = None,
) -> dict[str, Any]:
    manifest_path = mixture / "manifest.json"
    if not manifest_path.is_file():
        _add_check(checks, "mixture_manifest_present", False, str(manifest_path), "manifest.json")
        return _empty_mixture_binding(protocol_path, identity_path, cfg)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    policy_complete = (
        manifest.get("schema_version")
        == TAU3_POLICY_COMPLETE_DATASET_SCHEMA_VERSION
    )
    competitive_v3 = (
        manifest.get("schema_version")
        == TAU3_COMPETITIVE_DATASET_SCHEMA_VERSION
    )
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
        manifest_protocol_sha == protocol_sha256
        or (competitive_v3 and manifest_protocol_sha is None),
        manifest_protocol_sha or "missing protocol SHA provenance",
        protocol_sha256,
    )
    recipe = _recipe_record(cfg, exposure_binding=exposure_binding)
    recipe_sha256 = _canonical_sha256(recipe)
    recipe_id = f"tau3-mlx-recipe-{recipe_sha256[:16]}"
    recipe_check = _recipe_within_protocol(
        protocol,
        cfg,
        recipe,
        exposure_binding=exposure_binding,
    )
    _add_check(checks, "recipe_within_protocol_recipe_space", recipe_check["passed"], recipe_check, "recipe inside frozen recipe_space")
    plan_check = _protocol_mlx_plan_allows_local_adapter_4bit(protocol)
    _add_check(checks, "protocol_mlx_plan_local_4bit_adapter_only", plan_check["passed"], plan_check, "local-only 4-bit adapter-only MLX plan")
    if competitive_v3:
        v3_validation = validate_tau3_competitive_dataset_bundle(
            mixture,
            strict=True,
            grounded_validator_python=grounded_validator_python,
        )
        grounded_validation_binding = _grounded_validation_binding(
            grounded_validator_python,
            root,
        )
        _add_check(
            checks,
            "competitive_v3_dataset_validation_passed",
            v3_validation.get("passed") is True,
            {
                "errors": v3_validation.get("errors"),
                "coverage": v3_validation.get("coverage"),
            },
            "strict v3 dataset validation passed",
        )
    else:
        v3_validation = None
        grounded_validation_binding = None
    schema_name = (
        TAU3_COMPETITIVE_DATASET_SCHEMA_VERSION
        if competitive_v3
        else TAU3_POLICY_COMPLETE_DATASET_SCHEMA_VERSION
        if policy_complete
        else TAU3_TRAINING_MIXTURE_SCHEMA_VERSION
    )
    schema = check_schema_contract(manifest, name_or_id=schema_name)
    _add_check(checks, "mixture_manifest_schema_passed", schema.get("passed") is True, schema.get("errors"), "passed")
    _add_check(
        checks,
        "mixture_variant_not_root_set",
        policy_complete or competitive_v3 or manifest.get("variant") != "mixture_set",
        manifest.get("lineage_id")
        if policy_complete or competitive_v3
        else manifest.get("variant"),
        "single trainable dataset",
    )
    _add_check(checks, "mixture_passed", manifest.get("passed") is True, manifest.get("passed"), True)
    if competitive_v3:
        tokenizer_record = manifest.get("tokenizer_config") if isinstance(manifest.get("tokenizer_config"), dict) else {}
        _add_check(
            checks,
            "competitive_v3_manifest_status_passed",
            manifest.get("status") == "passed",
            manifest.get("status"),
            "passed",
        )
        _add_check(
            checks,
            "competitive_v3_files_replay",
            _competitive_v3_files_replay(mixture, manifest),
            _summary(manifest.get("files")),
            "current train/valid hashes and byte counts",
        )
        _add_check(
            checks,
            "competitive_v3_tokenizer_exact_recorded",
            bool(
                tokenizer_record.get("tokenizer_id")
                and tokenizer_record.get("tokenizer_revision")
                and tokenizer_record.get("chat_template_sha256")
                and tokenizer_record.get("tokenizer_json_sha256")
                and tokenizer_record.get("tokenizer_config_sha256")
            ),
            tokenizer_record,
            "pinned tokenizer identity and chat-template hashes",
        )
        tokenizer_model_check = _competitive_v3_tokenizer_matches_model(
            mixture,
            tokenizer_record,
            model_dir,
        )
        _add_check(
            checks,
            "competitive_v3_tokenizer_matches_model",
            tokenizer_model_check["passed"],
            tokenizer_model_check,
            "dataset tokenizer assets and chat template equal local model tokenizer",
        )
    sealed_source = manifest.get("sealed_access") if competitive_v3 else manifest.get("sealed")
    sealed_record = sealed_source if isinstance(sealed_source, dict) else {}
    sealed_ok = (
        sealed_record.get("access_count") == 0
        and (
            sealed_record.get("payload_accessed") is False
            or sealed_record.get("raw_sealed_payload_read") is False
        )
        and not (mixture / "test.jsonl").exists()
        if policy_complete or competitive_v3
        else manifest.get("sealed_rows") == 0 and manifest.get("test_rows") == 0
    )
    _add_check(
        checks,
        "mixture_no_sealed_or_test_rows",
        sealed_ok,
        sealed_record
        if policy_complete or competitive_v3
        else {
            "sealed": manifest.get("sealed_rows"),
            "test": manifest.get("test_rows"),
        },
        {"sealed_access": 0, "test_file": False}
        if policy_complete or competitive_v3
        else {"sealed": 0, "test": 0},
    )
    _add_check(
        checks,
        "mixture_not_already_training_started",
        manifest.get("training_started") is False
        or (competitive_v3 and "training_started" not in manifest),
        manifest.get("training_started"),
        False,
    )
    _add_check(
        checks,
        "mixture_files_replay",
        _competitive_v3_files_replay(mixture, manifest)
        if competitive_v3
        else _mixture_files_replay(mixture, manifest),
        _summary(manifest.get("files")),
        "current train/valid hashes",
    )
    source_hashes_replay = (
        _policy_complete_manifest_replays(
            manifest,
            protocol_sha256=protocol_sha256,
        )
        if policy_complete
        else v3_validation is not None and v3_validation.get("passed") is True
        if competitive_v3
        else _mixture_source_hashes_replay(mixture, manifest)
    )
    _add_check(
        checks,
        "mixture_source_hashes_replay",
        source_hashes_replay,
        _summary(manifest.get("inputs"))
        if policy_complete
        else _summary(v3_validation)
        if competitive_v3
        else _summary(manifest.get("source_binding")),
        "current policy-complete seal and parent protocol"
        if policy_complete
        else "strict v3 dataset replay"
        if competitive_v3
        else "current source hashes",
    )
    if policy_complete:
        tokenizer_record = (
            manifest.get("tokenizer")
            if isinstance(manifest.get("tokenizer"), dict)
            else {}
        )
        supervision = (
            manifest.get("supervision")
            if isinstance(manifest.get("supervision"), dict)
            else {}
        )
        _add_check(
            checks,
            "policy_complete_mask_prompt_enforced",
            supervision.get("mask_prompt_required") is True and cfg.mask_prompt,
            {
                "manifest_required": supervision.get("mask_prompt_required"),
                "recipe_mask_prompt": cfg.mask_prompt,
            },
            {"manifest_required": True, "recipe_mask_prompt": True},
        )
        _add_check(
            checks,
            "policy_complete_sequence_budget_matches",
            tokenizer_record.get("max_seq_length") == cfg.max_seq_length
            and int(tokenizer_record.get("max_rendered_tokens") or 0)
            <= cfg.max_seq_length,
            {
                "dataset_max_seq_length": tokenizer_record.get("max_seq_length"),
                "dataset_max_rendered_tokens": tokenizer_record.get(
                    "max_rendered_tokens"
                ),
                "recipe_max_seq_length": cfg.max_seq_length,
            },
            "recipe max_seq_length equals the tokenizer-audited dataset budget",
        )
    direct_scan = _scan_mlx_data_dir(
        mixture,
        policy_complete=policy_complete,
    )
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
            "source_binding_sha256": _canonical_sha256(
                {
                    "inputs": manifest.get("inputs"),
                    "partition": manifest.get("partition"),
                    "balance": manifest.get("balance"),
                    "supervision": manifest.get("supervision"),
                }
                if policy_complete
                else {
                    "files": manifest.get("files"),
                    "source_dataset": manifest.get("source_dataset"),
                    "tokenizer_config": manifest.get("tokenizer_config"),
                    "manifest_sha256": manifest.get("manifest_sha256"),
                }
                if competitive_v3
                else manifest.get("source_binding")
            ),
            "declared_protocol_sha256": manifest_protocol_sha,
            **(
                {"grounded_validation": grounded_validation_binding}
                if grounded_validation_binding is not None
                else {}
            ),
        },
        "recipe": {
            **recipe,
            "recipe_sha256": recipe_sha256,
            "recipe_id": recipe_id,
        },
    }


def _grounded_validation_binding(
    grounded_validator_python: str | Path | None,
    root: Path,
) -> dict[str, Any]:
    script = Path(__file__).resolve().parents[1] / "scripts" / "validate_tau3_grounded_generation.py"
    binding: dict[str, Any] = {
        "strict_dataset_validation": True,
        "validator": "validate_tau3_competitive_dataset_bundle",
        "grounded_replay": {
            "mode": "direct_in_process"
            if grounded_validator_python is None
            else "external_subprocess",
        },
        "script": {
            "path": str(script),
            "sha256": _sha256_file(script) if script.is_file() else None,
        },
    }
    if grounded_validator_python is None:
        return binding
    raw = Path(grounded_validator_python).expanduser()
    resolved = raw.resolve(strict=False)
    interpreter: dict[str, Any] = {
        "path": str(resolved),
        "provided_path": str(grounded_validator_python),
    }
    try:
        if resolved.is_file() and resolved.resolve().is_relative_to(root.resolve()):
            interpreter["sha256"] = _sha256_file(resolved)
            interpreter["sha256_policy"] = "workspace_executable_content_hash"
        else:
            interpreter["sha256"] = None
            interpreter["sha256_policy"] = "not_hashed_outside_workspace"
    except OSError as exc:
        interpreter["sha256"] = None
        interpreter["sha256_policy"] = f"not_hashed_unresolved:{type(exc).__name__}"
    binding["grounded_replay"]["interpreter"] = interpreter
    return binding


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


def _competitive_v3_files_replay(mixture: Path, manifest: dict[str, Any]) -> bool:
    files = manifest.get("files")
    if not isinstance(files, dict):
        return False
    for split in ("train", "valid"):
        record = files.get(split)
        if not isinstance(record, dict) or record.get("path") != f"{split}.jsonl":
            return False
        path = mixture / f"{split}.jsonl"
        if not path.is_file() or path_has_symlink_component(path, include_leaf=True):
            return False
        if record.get("sha256") != _sha256_file(path):
            return False
        size = record.get("bytes", record.get("size"))
        if size != path.stat().st_size:
            return False
    return True


def _competitive_v3_tokenizer_matches_model(
    mixture: Path,
    record: dict[str, Any],
    model_dir: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    copied: dict[str, Any] = {}
    model: dict[str, Any] = {}
    path_leaf = str(record.get("path_leaf") or "")
    dataset_tokenizer = mixture / path_leaf
    if (
        not path_leaf
        or Path(path_leaf).is_absolute()
        or ".." in Path(path_leaf).parts
        or path_has_symlink_component(dataset_tokenizer, include_leaf=True)
        or not dataset_tokenizer.resolve().is_relative_to(mixture.resolve())
        or not dataset_tokenizer.is_dir()
    ):
        errors.append("tokenizer_config.path_leaf is unsafe or missing")
        return {"passed": False, "errors": errors, "dataset": copied, "model": model}

    asset_records = record.get("copied_assets")
    if not isinstance(asset_records, dict):
        asset_records = {}
    for filename in TOKENIZER_ASSET_FILENAMES:
        dataset_asset = dataset_tokenizer / filename
        model_asset = model_dir / filename
        expected_hash = asset_records.get(filename)
        dataset_hash = _sha256_file(dataset_asset) if dataset_asset.is_file() else None
        model_hash = _sha256_file(model_asset) if model_asset.is_file() else None
        copied[filename] = dataset_hash
        model[filename] = model_hash
        if filename in {"tokenizer.json", "tokenizer_config.json"}:
            if dataset_hash is None:
                errors.append(f"dataset tokenizer asset {filename} is missing")
            if model_hash is None:
                errors.append(f"model tokenizer asset {filename} is missing")
            if dataset_hash is not None and expected_hash != dataset_hash:
                errors.append(f"dataset copied asset {filename} hash does not replay")
            if dataset_hash is not None and model_hash is not None and dataset_hash != model_hash:
                errors.append(f"dataset tokenizer asset {filename} does not match model")
        elif expected_hash is not None or dataset_hash is not None or model_hash is not None:
            if dataset_hash is None:
                errors.append(f"dataset tokenizer asset {filename} is missing")
            if model_hash is None:
                errors.append(f"model tokenizer asset {filename} is missing")
            if dataset_hash is not None and expected_hash != dataset_hash:
                errors.append(f"dataset copied asset {filename} hash does not replay")
            if dataset_hash is not None and model_hash is not None and dataset_hash != model_hash:
                errors.append(f"dataset tokenizer asset {filename} does not match model")

    for field, filename in (
        ("tokenizer_json_sha256", "tokenizer.json"),
        ("tokenizer_config_sha256", "tokenizer_config.json"),
        ("chat_template_file_sha256", "chat_template.jinja"),
    ):
        expected = record.get(field)
        if expected and copied.get(filename) != expected:
            errors.append(f"{field} does not match copied tokenizer asset")
        if expected and model.get(filename) != expected:
            errors.append(f"{field} does not match model tokenizer asset")

    try:
        tokenizer = _load_local_tokenizer(model_dir)
        model_chat_template_sha = _canonical_sha256(str(getattr(tokenizer, "chat_template", "") or ""))
    except Exception as exc:
        model_chat_template_sha = None
        errors.append(f"model chat_template cannot be replayed: {type(exc).__name__}")
    copied["chat_template_sha256"] = record.get("chat_template_sha256")
    model["chat_template_sha256"] = model_chat_template_sha
    if record.get("chat_template_sha256") != model_chat_template_sha:
        errors.append("chat_template_sha256 does not match model tokenizer")

    return {
        "passed": not errors,
        "errors": errors,
        "dataset": copied,
        "model": model,
        "path_leaf": path_leaf,
    }


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


def _policy_complete_manifest_replays(
    manifest: dict[str, Any],
    *,
    protocol_sha256: str,
) -> bool:
    declared_seal = manifest.get("manifest_sha256")
    replayed_seal = _canonical_sha256(
        {
            key: value
            for key, value in manifest.items()
            if key != "manifest_sha256"
        }
    )
    parent = manifest.get("parent_protocol")
    coverage = manifest.get("coverage")
    contamination = manifest.get("contamination")
    balance = manifest.get("balance")
    supervision = manifest.get("supervision")
    sealed = manifest.get("sealed")
    return (
        isinstance(declared_seal, str)
        and declared_seal == replayed_seal
        and isinstance(parent, dict)
        and parent.get("sha256") == protocol_sha256
        and isinstance(coverage, dict)
        and coverage.get("passed") is True
        and isinstance(contamination, dict)
        and contamination.get("passed") is True
        and isinstance(balance, dict)
        and balance.get("passed") is True
        and isinstance(supervision, dict)
        and supervision.get("mask_prompt_required") is True
        and supervision.get("negative_actions_are_context_only") is True
        and isinstance(sealed, dict)
        and sealed.get("access_count") == 0
        and sealed.get("payload_accessed") is False
        and manifest.get("training_started") is False
    )


def _empty_mixture_binding(protocol_path: Path, identity_path: Path, cfg: Tau3MlxTrainingConfig) -> dict[str, Any]:
    recipe = _recipe_record(cfg)
    recipe_sha256 = _canonical_sha256(recipe)
    return {
        "protocol": {"path": str(protocol_path), "sha256": _sha256_file(protocol_path) if protocol_path.is_file() else None},
        "model": {"identity_path": str(identity_path), "identity_sha256": _sha256_file(identity_path) if identity_path.is_file() else None},
        "dataset": None,
        "recipe": {**recipe, "recipe_sha256": recipe_sha256, "recipe_id": f"tau3-mlx-recipe-{recipe_sha256[:16]}"},
    }


def _validate_exposure_training_binding(
    *,
    dataset_path: str | Path | None,
    receipt_path: str | Path | None,
    ledger_path: str | Path | None,
    training_data_dir: Path,
    root: Path,
    cfg: Tau3MlxTrainingConfig,
    checks: list[dict[str, Any]],
) -> dict[str, Any] | None:
    provided = [dataset_path is not None, receipt_path is not None, ledger_path is not None]
    if not cfg.exposure_ledger_training:
        _add_check(
            checks,
            "exposure_ledger_training_disabled_without_inputs",
            not any(provided),
            {
                "exposure_dataset": dataset_path is not None,
                "exposure_receipt": receipt_path is not None,
                "exposure_ledger": ledger_path is not None,
            },
            "no exposure artifacts unless --exposure-ledger-training is enabled",
        )
        return None
    if not all(provided):
        _add_check(
            checks,
            "exposure_inputs_complete",
            False,
            {
                "exposure_dataset": dataset_path is not None,
                "exposure_receipt": receipt_path is not None,
                "exposure_ledger": ledger_path is not None,
            },
            "dataset, receipt, and ledger",
        )
        return None
    assert dataset_path is not None and receipt_path is not None and ledger_path is not None
    dataset_file = _require_local_file(Path(dataset_path), root, "exposure dataset")
    receipt_file = _require_local_file(Path(receipt_path), root, "exposure receipt")
    ledger_file = _require_local_file(Path(ledger_path), root, "exposure ledger")
    expected_dataset = _require_local_file(
        training_data_dir / "train.jsonl",
        root,
        "training data train split",
    )
    try:
        validation = validate_tau3_exposure_ledger(
            dataset_file,
            receipt_file,
            ledger_file,
        )
        receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    except (OSError, ValueError, Tau3ExposureError, json.JSONDecodeError) as exc:
        _add_check(
            checks,
            "exposure_ledger_replays",
            False,
            str(exc),
            "receipt and ledger replay from dataset",
        )
        return None
    sampler = receipt.get("sampler_config") if isinstance(receipt.get("sampler_config"), dict) else {}
    coverage = receipt.get("coverage") if isinstance(receipt.get("coverage"), dict) else {}
    optimizer_steps = coverage.get("complete_optimizer_step_count")
    microbatch_iterations = (
        int(optimizer_steps) * cfg.grad_accumulation
        if isinstance(optimizer_steps, int)
        else None
    )
    _add_check(
        checks,
        "exposure_ledger_replays",
        validation.get("passed") is True,
        validation,
        "receipt and ledger replay from dataset",
    )
    _add_check(
        checks,
        "exposure_dataset_matches_training_data",
        dataset_file == expected_dataset
        and _sha256_file(dataset_file) == _sha256_file(expected_dataset),
        {
            "exposure_dataset": str(dataset_file),
            "training_dataset": str(expected_dataset),
            "exposure_sha256": _sha256_file(dataset_file),
            "training_sha256": _sha256_file(expected_dataset),
        },
        "exact resolved data_dir/train.jsonl path and content",
    )
    _add_check(
        checks,
        "exposure_recipe_matches_sampler",
        sampler.get("batch_size") == cfg.batch_size
        and sampler.get("gradient_accumulation_steps") == cfg.grad_accumulation
        and microbatch_iterations == cfg.iters,
        {
            "recipe": {
                "batch_size": cfg.batch_size,
                "grad_accumulation": cfg.grad_accumulation,
                "microbatch_iterations": cfg.iters,
            },
            "sampler": sampler,
            "optimizer_steps": optimizer_steps,
            "expected_microbatch_iterations": microbatch_iterations,
        },
        "batch_size, grad_accumulation, and microbatch iters equal exposure receipt",
    )
    _add_check(
        checks,
        "exposure_full_row_multi_epoch_complete",
        receipt.get("passed") is True
        and coverage.get("all_rows_seen") is True
        and coverage.get("full_epoch_replay") is True
        and coverage.get("complete_optimizer_steps") is True
        and float(coverage.get("effective_epochs") or 0.0) >= 2.0,
        {
            "passed": receipt.get("passed"),
            "all_rows_seen": coverage.get("all_rows_seen"),
            "full_epoch_replay": coverage.get("full_epoch_replay"),
            "complete_optimizer_steps": coverage.get("complete_optimizer_steps"),
            "effective_epochs": coverage.get("effective_epochs"),
        },
        "candidate-eligible full-row multi-epoch exposure",
    )
    return {
        "mode": (
            "deterministic_exposure_ledger_detached_prefix"
            if cfg.prefix_cache_training
            else "deterministic_exposure_ledger_full_gradient"
        ),
        "dataset": {
            "path": str(dataset_file),
            "sha256": _sha256_file(dataset_file),
            "content_sha256": (receipt.get("dataset") or {}).get("content_sha256")
            if isinstance(receipt.get("dataset"), dict)
            else None,
            "row_count": (receipt.get("dataset") or {}).get("row_count")
            if isinstance(receipt.get("dataset"), dict)
            else None,
        },
        "receipt": {
            "path": str(receipt_file),
            "sha256": _sha256_file(receipt_file),
            "schema_version": receipt.get("schema_version"),
        },
        "ledger": {
            "path": str(ledger_file),
            "sha256": _sha256_file(ledger_file),
            "optimizer_steps": optimizer_steps,
            "microbatch_iterations": microbatch_iterations,
        },
        "sampler_config": sampler,
        "coverage": {
            "all_rows_seen": coverage.get("all_rows_seen"),
            "effective_epochs": coverage.get("effective_epochs"),
            "complete_optimizer_steps": coverage.get("complete_optimizer_steps"),
            "complete_optimizer_step_count": coverage.get("complete_optimizer_step_count"),
            "optimizer_steps": optimizer_steps,
            "microbatch_iterations": microbatch_iterations,
        },
        "objective": {
            "trainer": (
                "mlx_lm_detached_complete_prompt_cache_lora"
                if cfg.prefix_cache_training
                else "mlx_lm_standard_lora"
            ),
            "iterator_patch_only": not cfg.prefix_cache_training,
            "full_gradient": not cfg.prefix_cache_training,
            "detached_prefix": cfg.prefix_cache_training,
            "mask_prompt": cfg.mask_prompt,
            "deterministic_exposure_replay": True,
            "mlx_lm_iters_are_microbatches": True,
        },
    }


def _validate_prefix_equivalence_binding(
    *,
    equivalence_path: str | Path | None,
    training_data_dir: Path,
    protocol_path: Path,
    model_identity_path: Path,
    root: Path,
    cfg: Tau3MlxTrainingConfig,
    checks: list[dict[str, Any]],
) -> dict[str, Any] | None:
    required = cfg.prefix_cache_training and cfg.exposure_ledger_training
    if not required:
        _add_check(
            checks,
            "prefix_equivalence_absent_unless_exposure_prefix_training",
            equivalence_path is None,
            {"provided": equivalence_path is not None, "required": required},
            "no equivalence artifact outside combined exposure-prefix training",
        )
        return None
    if equivalence_path is None:
        raise Tau3MlxTrainingError(
            "prefix equivalence artifact is required for combined "
            "exposure-ledger prefix-cache training"
        )
    artifact_path = _require_local_file(
        Path(equivalence_path),
        root,
        "prefix equivalence",
    )
    dataset_path = _require_local_file(
        training_data_dir / "train.jsonl",
        root,
        "training data train split",
    )
    expected_bindings = {
        "dataset_file_sha256": _sha256_file(dataset_path),
        "protocol_file_sha256": _sha256_file(protocol_path),
        "model_identity_file_sha256": _sha256_file(model_identity_path),
        "recipe": {
            "rank": cfg.rank,
            "scale": cfg.scale,
            "learning_rate": cfg.learning_rate,
            "num_layers": cfg.num_layers,
            "max_seq_length": cfg.max_seq_length,
            "batch_size": cfg.batch_size,
            "grad_accumulation": cfg.grad_accumulation,
            "mask_prompt": cfg.mask_prompt,
            "seed": cfg.seed,
        },
    }
    validation = validate_tau3_prefix_equivalence(
        artifact_path,
        expected_bindings=expected_bindings,
    )
    membership = _prefix_equivalence_sample_membership(
        artifact_path,
        dataset_path,
    )
    _add_check(
        checks,
        "prefix_equivalence_replays_and_matches_launch",
        validation.get("passed") is True,
        validation,
        "passing full-gradient A/B bound to dataset, protocol, model, and recipe",
    )
    _add_check(
        checks,
        "prefix_equivalence_sample_is_candidate_dataset_subset",
        membership.get("passed") is True,
        membership,
        "every bounded A/B source row belongs to the candidate training dataset",
    )
    if (
        validation.get("passed") is not True
        or membership.get("passed") is not True
    ):
        return None
    return {
        "path": str(artifact_path),
        "sha256": _sha256_file(artifact_path),
        "schema_version": "hfr.tau3_prefix_equivalence.v1",
        "validation_passed": True,
        "validation_schema_version": validation.get("schema_version"),
        "bindings": expected_bindings,
        "sample_membership": membership,
    }


def _prefix_equivalence_sample_membership(
    artifact_path: Path,
    dataset_path: Path,
) -> dict[str, Any]:
    """Replay that every smoke source row is an exact candidate row."""

    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"passed": False, "error": str(exc)}
    sample = artifact.get("sample") if isinstance(artifact, dict) else None
    sample_hashes = (
        sample.get("row_hashes") if isinstance(sample, dict) else None
    )
    if (
        not isinstance(sample_hashes, list)
        or not sample_hashes
        or any(not isinstance(value, str) for value in sample_hashes)
    ):
        return {
            "passed": False,
            "error": "equivalence sample row hashes are unavailable",
        }
    candidate_hashes: set[str] = set()
    try:
        lines = dataset_path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                return {
                    "passed": False,
                    "error": (
                        f"candidate dataset line {line_number} "
                        "must be an object"
                    ),
                }
            candidate_hashes.add(_canonical_sha256(row))
    except (OSError, json.JSONDecodeError) as exc:
        return {"passed": False, "error": str(exc)}
    missing = [
        value for value in sample_hashes if value not in candidate_hashes
    ]
    return {
        "passed": not missing,
        "sample_row_count": len(sample_hashes),
        "candidate_row_count": len(candidate_hashes),
        "missing_row_hashes": missing,
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


def _recipe_record(
    cfg: Tau3MlxTrainingConfig,
    *,
    exposure_binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    optimizer_steps = _exposure_optimizer_steps(exposure_binding)
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
        "microbatch_iterations": cfg.iters if cfg.exposure_ledger_training else None,
        "optimizer_steps": optimizer_steps if cfg.exposure_ledger_training else None,
        "seed": cfg.seed,
        "mask_prompt": cfg.mask_prompt,
        "grad_checkpoint": cfg.grad_checkpoint,
        "disable_compile": cfg.disable_compile,
        "fixed_shape_padding": cfg.fixed_shape_padding,
        "prefix_cache_training": cfg.prefix_cache_training,
        "exposure_ledger_training": cfg.exposure_ledger_training,
        "full_gradient_objective": not cfg.prefix_cache_training,
        "prefix_equivalence_required": (
            cfg.prefix_cache_training and cfg.exposure_ledger_training
        ),
        "prefix_equivalence_passed": (
            cfg.prefix_cache_training and cfg.exposure_ledger_training
        ),
    }


def _recipe_within_protocol(
    protocol: dict[str, Any],
    cfg: Tau3MlxTrainingConfig,
    recipe: dict[str, Any],
    *,
    exposure_binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
    if cfg.exposure_ledger_training:
        optimizer_steps = _exposure_optimizer_steps(exposure_binding)
        if optimizer_steps is None:
            failures.append(
                {
                    "field": "steps",
                    "actual": None,
                    "expected": "validated exposure optimizer step count",
                }
            )
        if cfg.grad_accumulation <= 0 or cfg.iters % cfg.grad_accumulation != 0:
            failures.append(
                {
                    "field": "iters/grad_accumulation",
                    "actual": {
                        "iters": cfg.iters,
                        "grad_accumulation": cfg.grad_accumulation,
                    },
                    "expected": "microbatch iterations divisible into complete optimizer steps",
                }
            )
    else:
        optimizer_steps = cfg.iters
    field_values = {
        "rank": cfg.rank,
        "alpha": cfg.scale,
        "scale": cfg.scale,
        "learning_rate": cfg.learning_rate,
        "sequence_length": cfg.max_seq_length,
        "max_seq_length": cfg.max_seq_length,
        "steps": optimizer_steps,
        "iters": cfg.iters,
        "batch_size": cfg.batch_size,
        "grad_accumulation": cfg.grad_accumulation,
        "num_layers": cfg.num_layers,
        "dropout": cfg.dropout,
    }
    for field, value in field_values.items():
        if value is None:
            continue
        if field in bounds and not _value_allowed_by_bound(value, bounds[field]):
            failures.append({"field": field, "actual": value, "expected": bounds[field]})
    required_groups = (("rank",), ("learning_rate",), ("sequence_length", "max_seq_length"), ("steps", "iters"))
    for names in required_groups:
        if not any(name in bounds for name in names):
            failures.append({"field": "/".join(names), "actual": "missing", "expected": "frozen bound"})
    return {"passed": not failures, "recipe": recipe, "bounds": bounds, "failures": failures}


def _exposure_optimizer_steps(exposure_binding: dict[str, Any] | None) -> int | None:
    if not isinstance(exposure_binding, dict):
        return None
    ledger = exposure_binding.get("ledger") if isinstance(exposure_binding.get("ledger"), dict) else {}
    coverage = exposure_binding.get("coverage") if isinstance(exposure_binding.get("coverage"), dict) else {}
    for source in (ledger, coverage):
        value = source.get("optimizer_steps") or source.get("complete_optimizer_step_count")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


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


def _scan_mlx_data_dir(
    data_dir: Path,
    *,
    policy_complete: bool = False,
) -> dict[str, Any]:
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
            ignored = (
                frozenset({"invented_tau_tool"})
                if policy_complete
                else frozenset()
            )
            for hit in _semantic_leak_hits(row, ignored_fragments=ignored):
                findings.append({"split": split, "line": line_number, **hit})
            if policy_complete:
                for hit in _policy_complete_row_hits(row, split=split):
                    findings.append(
                        {"split": split, "line": line_number, **hit}
                    )
    return {"passed": not findings and row_count > 0, "row_count": row_count, "finding_count": len(findings), "findings": findings[:25]}


def _semantic_leak_hits(
    value: Any,
    path: str = "$",
    *,
    ignored_fragments: frozenset[str] = frozenset(),
) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            lowered_key = key_text.lower()
            for fragment in FORBIDDEN_DATA_FRAGMENTS:
                if fragment in ignored_fragments:
                    continue
                if fragment in lowered_key:
                    hits.append({"path": f"{path}.{key_text}", "reason": f"forbidden key fragment: {fragment}"})
            hits.extend(
                _semantic_leak_hits(
                    item,
                    f"{path}.{key_text}",
                    ignored_fragments=ignored_fragments,
                )
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            hits.extend(
                _semantic_leak_hits(
                    item,
                    f"{path}[{index}]",
                    ignored_fragments=ignored_fragments,
                )
            )
    elif isinstance(value, str):
        lowered = value.lower()
        for fragment in FORBIDDEN_DATA_FRAGMENTS:
            if fragment in ignored_fragments:
                continue
            if fragment in lowered:
                hits.append({"path": path, "reason": f"forbidden text fragment: {fragment}"})
    return hits


def _policy_complete_row_hits(
    row: Any,
    *,
    split: str,
) -> list[dict[str, str]]:
    if not isinstance(row, dict):
        return [{"path": "$", "reason": "policy-complete row must be an object"}]
    messages = row.get("messages")
    tools = row.get("tools")
    metadata = row.get("metadata")
    if not isinstance(messages, list) or not messages:
        return [{"path": "$.messages", "reason": "missing policy-complete messages"}]
    if not isinstance(tools, list) or not tools:
        return [{"path": "$.tools", "reason": "missing full ordered tool catalog"}]
    if not isinstance(metadata, dict):
        return [{"path": "$.metadata", "reason": "missing policy-complete metadata"}]
    hits: list[dict[str, str]] = []
    if metadata.get("schema_version") != TAU3_POLICY_COMPLETE_ROW_SCHEMA_VERSION:
        hits.append(
            {
                "path": "$.metadata.schema_version",
                "reason": "invalid policy-complete row schema",
            }
        )
    if metadata.get("split") != split:
        hits.append(
            {
                "path": "$.metadata.split",
                "reason": "row split does not match source file",
            }
        )
    if metadata.get("mask_prompt_required") is not True:
        hits.append(
            {
                "path": "$.metadata.mask_prompt_required",
                "reason": "policy-complete row must require prompt masking",
            }
        )
    if messages[0].get("role") != "system":
        hits.append(
            {
                "path": "$.messages[0]",
                "reason": "policy-complete row must begin with system prompt",
            }
        )
    if messages[-1].get("role") != "assistant":
        hits.append(
            {
                "path": f"$.messages[{len(messages) - 1}]",
                "reason": "policy-complete supervised target must be assistant",
            }
        )
    tool_names = {
        str((tool.get("function") or {}).get("name") or tool.get("name") or "")
        for tool in tools
        if isinstance(tool, dict)
    }
    invented_indices: list[int] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            hits.append(
                {
                    "path": f"$.messages[{index}]",
                    "reason": "message must be an object",
                }
            )
            continue
        content = str(message.get("content") or "").lower()
        if index > 0:
            for marker in USER_SIMULATOR_PRIVATE_MARKERS:
                if marker in content:
                    hits.append(
                        {
                            "path": f"$.messages[{index}].content",
                            "reason": f"user-simulator private marker: {marker}",
                        }
                    )
        for call in message.get("tool_calls") or []:
            function = call.get("function") if isinstance(call, dict) else None
            name = (
                str(function.get("name") or "")
                if isinstance(function, dict)
                else ""
            )
            if name == "invented_tau_tool":
                invented_indices.append(index)
            elif name not in tool_names:
                hits.append(
                    {
                        "path": f"$.messages[{index}].tool_calls",
                        "reason": f"tool call is absent from ordered catalog: {name}",
                    }
                )
    if invented_indices:
        if metadata.get("negative_prefix") is not True:
            hits.append(
                {
                    "path": "$.metadata.negative_prefix",
                    "reason": "invented tool requires masked negative-prefix evidence",
                }
            )
        for index in invented_indices:
            if index >= len(messages) - 1:
                hits.append(
                    {
                        "path": f"$.messages[{index}].tool_calls",
                        "reason": "invented tool cannot be the supervised target",
                    }
                )
                continue
            call_id = str(
                (messages[index].get("tool_calls") or [{}])[0].get("id") or ""
            )
            result = messages[index + 1] if index + 1 < len(messages) else {}
            if (
                result.get("role") != "tool"
                or result.get("tool_call_id") != call_id
                or "error" not in str(result.get("content") or "").lower()
            ):
                hits.append(
                    {
                        "path": f"$.messages[{index}]",
                        "reason": "invented tool prefix must be followed by bound error evidence",
                    }
                )
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


def _mlx_lora_config(
    model: str,
    data_dir: Path,
    adapter_path: str,
    cfg: Tau3MlxTrainingConfig,
    *,
    exposure_binding: dict[str, Any] | None = None,
    prefix_equivalence_binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "train": True,
        "fine_tune_type": "lora",
        "data": str(data_dir),
        "adapter_path": adapter_path,
        "iters": cfg.iters,
        "microbatch_iterations": cfg.iters if cfg.exposure_ledger_training else None,
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
        "fixed_shape_padding": cfg.fixed_shape_padding,
        "prefix_cache_training": cfg.prefix_cache_training,
        "exposure_ledger_training": cfg.exposure_ledger_training,
        "clear_cache_threshold": cfg.clear_cache_threshold,
        "process_segment_iters": cfg.process_segment_iters,
        "report_to": None,
        "test": False,
        "lora_parameters": {
            "rank": cfg.rank,
            "scale": cfg.scale,
            "dropout": cfg.dropout,
        },
    }
    if exposure_binding is not None:
        payload["exposure"] = exposure_binding
    if prefix_equivalence_binding is not None:
        payload["prefix_equivalence"] = prefix_equivalence_binding
    return payload


def _build_command(
    python: Path,
    model: str,
    data_dir: Path,
    adapter_dir: Path,
    config_path: Path,
    cfg: Tau3MlxTrainingConfig,
    *,
    resume_adapter_file: Path | None = None,
    exposure_binding: dict[str, Any] | None = None,
    prefix_equivalence_binding: dict[str, Any] | None = None,
) -> list[str]:
    if cfg.exposure_ledger_training and cfg.prefix_cache_training:
        module = "flightrecorder.mlx_exposure_prefix_cache_lora"
    elif cfg.exposure_ledger_training:
        module = "flightrecorder.mlx_exposure_lora"
    elif cfg.prefix_cache_training:
        module = "flightrecorder.mlx_prefix_cache_lora"
    elif cfg.fixed_shape_padding:
        module = "flightrecorder.mlx_fixed_shape_lora"
    else:
        module = "mlx_lm"
    command = [
        str(python),
        "-m",
        module,
    ]
    if not (cfg.fixed_shape_padding or cfg.prefix_cache_training or cfg.exposure_ledger_training):
        command.append("lora")
    command.extend(
        [
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
    )
    if cfg.mask_prompt:
        command.append("--mask-prompt")
    if cfg.grad_checkpoint:
        command.append("--grad-checkpoint")
    if resume_adapter_file is not None:
        command.extend(["--resume-adapter-file", str(resume_adapter_file)])
    if exposure_binding is not None:
        command.extend(
            [
                "--exposure-dataset",
                str(exposure_binding["dataset"]["path"]),
                "--exposure-receipt",
                str(exposure_binding["receipt"]["path"]),
                "--exposure-ledger",
                str(exposure_binding["ledger"]["path"]),
            ]
        )
    if prefix_equivalence_binding is not None:
        command.extend(
            [
                "--prefix-equivalence",
                str(prefix_equivalence_binding["path"]),
            ]
        )
    return command


def build_tau3_process_segment_plan(cfg: Tau3MlxTrainingConfig) -> dict[str, Any]:
    """Build the deterministic, hash-chained child-process execution plan."""

    cadence = cfg.process_segment_iters
    if cadence is None:
        raise Tau3MlxTrainingError("process_segment_iters is required")
    if not _config_within_bounds(cfg):
        raise Tau3MlxTrainingError(
            "process-segment configuration is outside governed bounds"
        )
    policy = {
        "schema_version": TAU3_MLX_PROCESS_SEGMENTS_SCHEMA_VERSION,
        "execution": "strictly_sequential_child_processes",
        "boundary_semantics": "half_open_microbatch_iterations",
        "state_continuity": "adapter_and_optimizer_sha256",
        "partial_directory_policy": "fresh_then_atomic_rename",
        "aggregate_telemetry_policy": "byte_concatenation_in_segment_order",
        "total_iters": cfg.iters,
        "process_segment_iters": cadence,
        "gradient_accumulation": cfg.grad_accumulation,
        "report_every": cfg.report_every,
        "dropout": cfg.dropout,
    }
    policy_sha256 = _canonical_sha256(policy)
    segments: list[dict[str, Any]] = []
    previous: str | None = None
    start = 0
    index = 0
    while start < cfg.iters:
        end = min(start + cadence, cfg.iters)
        record: dict[str, Any] = {
            "index": index,
            "segment_id": f"segment-{index + 1:04d}",
            "start_iter": start,
            "end_iter": end,
            "iteration_count": end - start,
            "previous_plan_record_sha256": previous,
        }
        record["plan_record_sha256"] = _canonical_sha256(record)
        previous = record["plan_record_sha256"]
        segments.append(record)
        start = end
        index += 1
    plan: dict[str, Any] = {
        "schema_version": TAU3_MLX_PROCESS_SEGMENTS_SCHEMA_VERSION,
        "policy": policy,
        "policy_sha256": policy_sha256,
        "segments": segments,
    }
    plan["plan_sha256"] = _canonical_sha256(plan)
    return plan


def validate_tau3_process_segments(
    process_segments: dict[str, Any],
    *,
    output_dir: str | Path,
    expected_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Independently replay a successful segmented-run evidence chain."""

    errors: list[str] = []
    output = Path(output_dir)
    try:
        output = output.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        return {"passed": False, "errors": [f"output_dir is unavailable: {exc}"]}
    if not isinstance(process_segments, dict):
        return {"passed": False, "errors": ["process_segments must be an object"]}

    policy = process_segments.get("policy")
    if not isinstance(policy, dict):
        errors.append("policy must be an object")
        policy = {}
    _validate_process_segment_policy(policy, errors)
    _validate_process_segment_config_binding(
        policy,
        expected_config,
        errors,
    )
    policy_sha256 = process_segments.get("policy_sha256")
    if policy_sha256 != _canonical_sha256(policy):
        errors.append("policy_sha256 mismatch")

    plan_binding = process_segments.get("plan")
    plan = _validate_bound_json_file(
        plan_binding,
        output,
        "plan",
        errors,
    )
    if plan is not None:
        expected_plan_sha = plan.get("plan_sha256")
        replay_plan = dict(plan)
        replay_plan.pop("plan_sha256", None)
        if expected_plan_sha != _canonical_sha256(replay_plan):
            errors.append("plan content hash mismatch")
        if process_segments.get("plan_sha256") != expected_plan_sha:
            errors.append("receipt plan_sha256 mismatch")
        if plan.get("policy") != policy:
            errors.append("plan policy differs from receipt policy")
        if plan.get("policy_sha256") != policy_sha256:
            errors.append("plan policy_sha256 mismatch")
        _validate_process_segment_plan_records(plan, errors)

    manifest_binding = process_segments.get("manifest")
    manifest = _validate_bound_json_file(
        manifest_binding,
        output,
        "manifest",
        errors,
    )
    if manifest is not None:
        receipt_manifest_view = {
            key: value
            for key, value in process_segments.items()
            if key not in {"manifest", "validation"}
        }
        if receipt_manifest_view != manifest:
            errors.append(
                "receipt process_segments fields differ from bound manifest"
            )
        expected_manifest_sha = manifest.get("manifest_sha256")
        replay_manifest = dict(manifest)
        replay_manifest.pop("manifest_sha256", None)
        if expected_manifest_sha != _canonical_sha256(replay_manifest):
            errors.append("manifest content hash mismatch")
        if process_segments.get("manifest_sha256") != expected_manifest_sha:
            errors.append("receipt manifest_sha256 mismatch")
        if manifest.get("policy") != policy:
            errors.append("manifest policy differs from receipt policy")
        if manifest.get("policy_sha256") != policy_sha256:
            errors.append("manifest policy_sha256 mismatch")
        if manifest.get("plan_sha256") != process_segments.get("plan_sha256"):
            errors.append("manifest plan_sha256 mismatch")
        if manifest.get("segments") != process_segments.get("segments"):
            errors.append("manifest segment records differ from receipt")
        if manifest.get("terminal_status") != "success":
            errors.append("segmented run terminal_status is not success")

    entries = process_segments.get("segments")
    if not isinstance(entries, list) or not entries:
        errors.append("segments must be a non-empty array")
        entries = []
    total_iters = policy.get("total_iters")
    expected_start = 0
    previous_record_sha: str | None = None
    previous_adapter_sha: str | None = None
    previous_optimizer_sha: str | None = None
    previous_adapter_path: str | None = None
    previous_optimizer_path: str | None = None
    telemetry_paths: list[Path] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"segment entry {index} must be an object")
            continue
        record = entry.get("record")
        if not isinstance(record, dict):
            errors.append(f"segment entry {index} record must be an object")
            continue
        record_sha = record.get("segment_record_sha256")
        replay_record = dict(record)
        replay_record.pop("segment_record_sha256", None)
        if record_sha != _canonical_sha256(replay_record):
            errors.append(f"segment {index} record hash mismatch")
        if record.get("index") != index:
            errors.append(f"segment {index} index mismatch")
        if record.get("start_iter") != expected_start:
            errors.append(f"segment {index} is not contiguous")
        start_iter = record.get("start_iter")
        end_iter = record.get("end_iter")
        if (
            not isinstance(start_iter, int)
            or not isinstance(end_iter, int)
            or end_iter <= start_iter
        ):
            errors.append(f"segment {index} has invalid boundaries")
        else:
            expected_start = end_iter
        if record.get("previous_segment_record_sha256") != previous_record_sha:
            errors.append(f"segment {index} record chain mismatch")
        adapter_input = record.get("adapter_input")
        optimizer_input = record.get("optimizer_state_input")
        if index == 0:
            if adapter_input is not None or optimizer_input is not None:
                errors.append("first segment must not have state inputs")
        else:
            if not isinstance(adapter_input, dict) or adapter_input.get(
                "sha256"
            ) != previous_adapter_sha or adapter_input.get(
                "path"
            ) != previous_adapter_path:
                errors.append(f"segment {index} adapter input chain mismatch")
            if not isinstance(optimizer_input, dict) or optimizer_input.get(
                "sha256"
            ) != previous_optimizer_sha or optimizer_input.get(
                "path"
            ) != previous_optimizer_path:
                errors.append(f"segment {index} optimizer input chain mismatch")
        if record.get("terminal_status") != "success":
            errors.append(f"segment {index} terminal_status is not success")

        record_file = _validate_bound_json_file(
            entry.get("record_file"),
            output,
            f"segment {index} record_file",
            errors,
        )
        if record_file is not None and record_file != record:
            errors.append(f"segment {index} record_file content mismatch")
        telemetry = _validate_bound_file_record(
            record.get("telemetry"),
            output,
            f"segment {index} telemetry",
            errors,
        )
        if telemetry is not None:
            if _telemetry_has_nonfinite_loss(telemetry):
                errors.append(
                    f"segment {index} telemetry contains non-finite loss"
                )
            telemetry_paths.append(telemetry)
        adapter_output = _validate_bound_file_record(
            record.get("adapter_output"),
            output,
            f"segment {index} adapter_output",
            errors,
        )
        optimizer_output = _validate_bound_file_record(
            record.get("optimizer_state_output"),
            output,
            f"segment {index} optimizer_state_output",
            errors,
        )
        adapter_tree = record.get("adapter_tree")
        if isinstance(adapter_tree, dict):
            adapter_root = _resolve_output_relative_path(
                adapter_tree.get("path"),
                output,
                f"segment {index} adapter_tree",
                errors,
            )
            if adapter_root is not None:
                replay_tree = _relative_fingerprint_tree(
                    _fingerprint_tree(adapter_root),
                    output,
                )
                if replay_tree != adapter_tree:
                    errors.append(f"segment {index} adapter tree mismatch")
                if not _tree_is_readonly_regular(adapter_root):
                    errors.append(
                        f"segment {index} adapter tree is mutable or unsafe"
                    )
                if (
                    optimizer_output is not None
                    and optimizer_output.is_relative_to(adapter_root)
                ):
                    errors.append(
                        f"segment {index} optimizer state is inside adapter tree"
                    )
        else:
            errors.append(f"segment {index} adapter_tree must be an object")
        previous_record_sha = (
            record_sha if isinstance(record_sha, str) else previous_record_sha
        )
        previous_adapter_sha = (
            _sha256_file(adapter_output) if adapter_output is not None else None
        )
        previous_adapter_path = (
            record["adapter_output"].get("path")
            if isinstance(record.get("adapter_output"), dict)
            else None
        )
        previous_optimizer_sha = (
            _sha256_file(optimizer_output)
            if optimizer_output is not None
            else None
        )
        previous_optimizer_path = (
            record["optimizer_state_output"].get("path")
            if isinstance(record.get("optimizer_state_output"), dict)
            else None
        )

    if isinstance(total_iters, int) and expected_start != total_iters:
        errors.append("segment execution does not cover total_iters")
    completed_count = sum(
        isinstance(entry, dict)
        and isinstance(entry.get("record"), dict)
        and entry["record"].get("terminal_status") == "success"
        for entry in entries
    )
    if process_segments.get("completed_segment_count") != completed_count:
        errors.append("completed_segment_count does not replay")
    if plan is not None:
        planned = plan.get("segments")
        planned_count = len(planned) if isinstance(planned, list) else 0
        if process_segments.get("planned_segment_count") != planned_count:
            errors.append("planned_segment_count does not replay")
        executed = [
            {
                key: entry["record"].get(key)
                for key in (
                    "index",
                    "segment_id",
                    "start_iter",
                    "end_iter",
                    "iteration_count",
                )
            }
            for entry in entries
            if isinstance(entry, dict) and isinstance(entry.get("record"), dict)
        ]
        expected = [
            {
                key: item.get(key)
                for key in (
                    "index",
                    "segment_id",
                    "start_iter",
                    "end_iter",
                    "iteration_count",
                )
            }
            for item in planned
        ] if isinstance(planned, list) else []
        if executed != expected:
            errors.append("executed segments differ from plan")

    aggregate = _validate_bound_file_record(
        process_segments.get("aggregate_telemetry"),
        output,
        "aggregate_telemetry",
        errors,
    )
    if aggregate is not None:
        expected_bytes = b"".join(path.read_bytes() for path in telemetry_paths)
        if aggregate.read_bytes() != expected_bytes:
            errors.append("aggregate telemetry is not exact ordered concatenation")

    final_adapter = process_segments.get("final_adapter")
    artifact_tree = process_segments.get("artifact_tree")
    if entries and isinstance(final_adapter, dict):
        final_adapter_root = _resolve_output_relative_path(
            final_adapter.get("path"),
            output,
            "final_adapter",
            errors,
        )
        if final_adapter_root is not None:
            replay_final = _relative_fingerprint_tree(
                _fingerprint_tree(final_adapter_root),
                output,
            )
            if replay_final != final_adapter:
                errors.append("final adapter tree mismatch")
            if not _tree_is_readonly_regular(final_adapter_root):
                errors.append("final adapter tree is mutable or unsafe")
            expected_files = _expected_assembled_adapter_files(
                entries,
                errors,
            )
            if replay_final.get("files") != expected_files:
                errors.append(
                    "final adapter differs from the ordered segment assembly"
                )
    else:
        errors.append("final_adapter must be an object")
    if isinstance(artifact_tree, dict):
        segment_root = _resolve_output_relative_path(
            artifact_tree.get("path"),
            output,
            "artifact_tree",
            errors,
        )
        if segment_root is not None:
            replay_artifacts = _relative_fingerprint_tree(
                _fingerprint_tree(segment_root),
                output,
            )
            if replay_artifacts != artifact_tree:
                errors.append("segment artifact tree mismatch")
    else:
        errors.append("artifact_tree must be an object")
    _validate_process_segment_recovery_artifacts(
        process_segments.get("recovery"),
        output,
        errors,
    )

    return {
        "passed": not errors,
        "errors": errors,
        "segment_count": len(entries),
        "covered_iters": expected_start,
    }


def _run_process_segments(
    *,
    command: list[str],
    cwd: Path,
    output_dir: Path,
    final_adapter_dir: Path,
    aggregate_telemetry_path: Path,
    cfg: Tau3MlxTrainingConfig,
    losses: dict[str, list[float]],
    resume: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.resolve(strict=True)
    final_adapter_dir = final_adapter_dir.resolve(strict=False)
    aggregate_telemetry_path = aggregate_telemetry_path.resolve(strict=False)
    plan = build_tau3_process_segment_plan(cfg)
    process_root = output_dir / "process_segments"
    committed_root = process_root / "segments"
    failed_root = process_root / "failed"
    plan_path = process_root / "plan.json"
    if resume:
        _require_existing_json_equal(
            plan_path,
            plan,
            "segmented-resume process plan",
        )
        manifest_path = process_root / "manifest.json"
        if os.path.lexists(manifest_path):
            _freeze_process_segment_partials(process_root)
            return _load_completed_process_segments(
                manifest_path=manifest_path,
                output_dir=output_dir,
                losses=losses,
                cfg=cfg,
            )
        if failed_root.is_dir() and any(failed_root.iterdir()):
            raise Tau3MlxTrainingError(
                "segmented-resume found committed child-failure evidence; "
                "it is preserved and cannot be retried as host-loss recovery"
            )
    else:
        process_root.mkdir()
        committed_root.mkdir()
        _write_new_json_readonly(plan_path, plan)

    deadline = time.monotonic() + cfg.timeout_seconds
    recovery = _load_committed_process_segments(
        committed_root=committed_root,
        output_dir=output_dir,
        plan=plan,
    )
    entries: list[dict[str, Any]] = recovery["entries"]
    previous_record_sha: str | None = recovery["previous_record_sha256"]
    previous_adapter_file: Path | None = recovery["previous_adapter_file"]
    previous_optimizer_file: Path | None = recovery["previous_optimizer_file"]
    terminal_status = "success"
    exit_code: int | None = 0
    timed_out = False
    peak_rss_kb = int(recovery["peak_child_rss_kb"])
    telemetry_count = int(recovery["telemetry_event_count"])
    telemetry_sources: list[Path] = recovery["telemetry_paths"]
    preserved_partial_paths = _freeze_process_segment_partials(process_root)
    for path in telemetry_sources:
        _replay_telemetry_losses(path, losses)

    for planned in plan["segments"][len(entries) :]:
        segment_id = str(planned["segment_id"])
        partial = _fresh_process_segment_partial(process_root, segment_id)
        partial.mkdir()
        segment_adapter_dir = partial / "adapter"
        segment_adapter_dir.mkdir()
        segment_optimizer_path = partial / "optimizer_state.safetensors"
        segment_telemetry_path = partial / "telemetry.jsonl"
        segment_command = _replace_command_option(
            command,
            "--adapter-path",
            str(segment_adapter_dir),
        )
        segment_command.extend(
            [
                "--hfr-child-segment-start",
                str(planned["start_iter"]),
                "--hfr-child-segment-end",
                str(planned["end_iter"]),
                "--hfr-child-segment-optimizer-state-output",
                str(segment_optimizer_path),
            ]
        )
        adapter_input: dict[str, Any] | None = None
        optimizer_input: dict[str, Any] | None = None
        if previous_adapter_file is not None:
            adapter_sha = _sha256_file(previous_adapter_file)
            adapter_input = {
                "path": _relative_output_path(previous_adapter_file, output_dir),
                "sha256": adapter_sha,
            }
            segment_command.extend(
                [
                    "--hfr-child-segment-adapter-input",
                    str(previous_adapter_file),
                    "--hfr-child-segment-adapter-sha256",
                    adapter_sha,
                ]
            )
        if previous_optimizer_file is not None:
            optimizer_sha = _sha256_file(previous_optimizer_file)
            optimizer_input = {
                "path": _relative_output_path(previous_optimizer_file, output_dir),
                "sha256": optimizer_sha,
            }
            segment_command.extend(
                [
                    "--hfr-child-segment-optimizer-state-input",
                    str(previous_optimizer_file),
                    "--hfr-child-segment-optimizer-state-sha256",
                    optimizer_sha,
                ]
            )
        _reject_forbidden_tokens(segment_command)
        remaining = max(1, int(deadline - time.monotonic()))
        child_exit, child_timed_out, child_count, child_peak = _run_child(
            command=segment_command,
            cwd=cwd,
            telemetry_path=segment_telemetry_path,
            timeout_seconds=remaining,
            losses=losses,
            disable_compile=cfg.disable_compile,
        )
        _require_safe_regular_tree(
            partial,
            f"{segment_id} child output",
        )
        child_status = _classify(
            child_exit,
            child_timed_out,
            segment_telemetry_path,
        )
        adapter_output = _select_segment_adapter_file(segment_adapter_dir)
        if (
            child_status == "success"
            and (
                adapter_output is None
                or not segment_optimizer_path.is_file()
                or segment_optimizer_path.stat().st_size <= 0
            )
        ):
            child_status = "no_output"
        successful = child_status == "success"
        destination_root = committed_root
        if not successful:
            failed_root.mkdir(exist_ok=True)
            destination_root = failed_root
        destination = destination_root / segment_id
        destination_adapter = destination / "adapter"
        destination_telemetry = destination / "telemetry.jsonl"
        destination_optimizer = destination / "optimizer_state.safetensors"
        destination_adapter_output = (
            destination_adapter / adapter_output.relative_to(segment_adapter_dir)
            if adapter_output is not None
            else None
        )
        adapter_tree = _relative_fingerprint_tree(
            _fingerprint_tree(segment_adapter_dir),
            partial,
        )
        adapter_tree["path"] = _relative_output_path(
            destination_adapter,
            output_dir,
        )
        adapter_output_record: dict[str, Any] | None = None
        if adapter_output is not None:
            assert destination_adapter_output is not None
            adapter_output_record = {
                "path": _relative_output_path(
                    destination_adapter_output,
                    output_dir,
                ),
                "sha256": _sha256_file(adapter_output),
                "read_only": True,
            }
        record: dict[str, Any] = {
            "index": planned["index"],
            "segment_id": segment_id,
            "start_iter": planned["start_iter"],
            "end_iter": planned["end_iter"],
            "iteration_count": planned["iteration_count"],
            "previous_segment_record_sha256": previous_record_sha,
            "adapter_input": adapter_input,
            "optimizer_state_input": optimizer_input,
            "command": _redact_command(segment_command),
            "telemetry": {
                "path": _relative_output_path(
                    destination_telemetry,
                    output_dir,
                ),
                "sha256": _sha256_file(segment_telemetry_path),
                "event_count": child_count,
                "read_only": True,
            },
            "adapter_output": adapter_output_record,
            "adapter_tree": adapter_tree,
            "optimizer_state_output": (
                {
                    "path": _relative_output_path(
                        destination_optimizer,
                        output_dir,
                    ),
                    "sha256": _sha256_file(segment_optimizer_path),
                    "read_only": True,
                }
                if segment_optimizer_path.is_file()
                and segment_optimizer_path.stat().st_size > 0
                else None
            ),
            "terminal_status": child_status,
            "exit_code": child_exit,
            "timed_out": child_timed_out,
            "telemetry_event_count": child_count,
            "peak_child_rss_kb": child_peak,
        }
        record["segment_record_sha256"] = _canonical_sha256(record)
        record_path = partial / "segment_record.json"
        _write_new_json_readonly(record_path, record)
        _commit_readonly_directory(partial, destination)
        record_file = _output_file_record(
            destination / "segment_record.json",
            output_dir,
        )
        entry = {"record": record, "record_file": record_file}
        entries.append(entry)
        telemetry_sources.append(destination_telemetry)
        telemetry_count += child_count
        peak_rss_kb = max(peak_rss_kb, child_peak)
        exit_code = child_exit
        timed_out = child_timed_out
        if not successful:
            terminal_status = child_status
            break
        previous_record_sha = str(record["segment_record_sha256"])
        assert destination_adapter_output is not None
        previous_adapter_file = destination_adapter_output
        previous_optimizer_file = destination_optimizer

    expected_telemetry = b"".join(
        path.read_bytes() for path in telemetry_sources
    )
    if os.path.lexists(aggregate_telemetry_path):
        aggregate_mode = os.lstat(aggregate_telemetry_path).st_mode
        if (
            not resume
            or not stat.S_ISREG(aggregate_mode)
            or path_has_symlink_component(
                aggregate_telemetry_path,
                include_leaf=True,
            )
            or bool(aggregate_mode & 0o222)
            or aggregate_telemetry_path.read_bytes() != expected_telemetry
        ):
            raise Tau3MlxTrainingError(
                "existing aggregate telemetry is not an immutable exact "
                "segment concatenation"
            )
    else:
        aggregate_partial = _fresh_process_segment_partial(
            process_root,
            "aggregate-telemetry",
        )
        _commit_readonly_file(
            aggregate_partial,
            aggregate_telemetry_path,
            expected_telemetry,
        )
    if terminal_status == "success" and previous_adapter_file is not None:
        if final_adapter_dir.exists():
            errors: list[str] = []
            expected_files = _expected_assembled_adapter_files(
                entries,
                errors,
            )
            actual_tree = _relative_fingerprint_tree(
                _fingerprint_tree(final_adapter_dir),
                output_dir,
            )
            if (
                not resume
                or errors
                or actual_tree.get("files") != expected_files
                or not _tree_is_readonly_regular(final_adapter_dir)
            ):
                raise Tau3MlxTrainingError(
                    "existing final adapter is not the immutable ordered "
                    "segment assembly: "
                    + json.dumps(errors, sort_keys=True)
                )
        else:
            adapter_partial = _fresh_process_segment_partial(
                process_root,
                "final-adapter",
            )
            _assemble_final_segment_adapter(
                entries=entries,
                output_dir=output_dir,
                final_adapter_dir=adapter_partial,
            )
            _commit_readonly_directory(
                adapter_partial,
                final_adapter_dir,
            )

    if failed_root.is_dir():
        _make_tree_readonly(failed_root)
        _fsync_tree(failed_root)
    artifact_tree = _relative_fingerprint_tree(
        _fingerprint_tree(committed_root),
        output_dir,
    )
    final_adapter = (
        _relative_fingerprint_tree(
            _fingerprint_tree(final_adapter_dir),
            output_dir,
        )
        if final_adapter_dir.is_dir()
        else None
    )
    manifest: dict[str, Any] = {
        "schema_version": TAU3_MLX_PROCESS_SEGMENTS_SCHEMA_VERSION,
        "policy": plan["policy"],
        "policy_sha256": plan["policy_sha256"],
        "plan": _output_file_record(plan_path, output_dir),
        "plan_sha256": plan["plan_sha256"],
        "segments": entries,
        "aggregate_telemetry": _output_file_record(
            aggregate_telemetry_path,
            output_dir,
        ),
        "artifact_tree": artifact_tree,
        "final_adapter": final_adapter,
        "terminal_status": terminal_status,
        "completed_segment_count": sum(
            entry["record"]["terminal_status"] == "success" for entry in entries
        ),
        "planned_segment_count": len(plan["segments"]),
        "recovery": {
            "resumed": resume,
            "accepted_segment_count": int(
                recovery["accepted_segment_count"]
            ),
            "preserved_partial_artifact_trees": [
                _preserved_partial_artifact_record(path, output_dir)
                for path in preserved_partial_paths
            ],
            "preserved_failed_artifact_tree": (
                _relative_fingerprint_tree(
                    _fingerprint_tree(failed_root),
                    output_dir,
                )
                if failed_root.is_dir()
                else None
            ),
        },
    }
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    manifest_path = process_root / "manifest.json"
    _publish_new_json_readonly(manifest_path, manifest)
    process_record = {
        **manifest,
        "manifest": _output_file_record(manifest_path, output_dir),
    }
    validation = validate_tau3_process_segments(
        process_record,
        output_dir=output_dir,
        expected_config=_process_segment_config_binding(cfg),
    )
    process_record["validation"] = validation
    if terminal_status == "success" and validation["passed"] is not True:
        raise Tau3MlxTrainingError(
            "process segment chain failed self-validation: "
            + json.dumps(validation["errors"], sort_keys=True)
        )
    return {
        "terminal_status": terminal_status,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "telemetry_event_count": telemetry_count,
        "peak_child_rss_kb": peak_rss_kb,
        "process_segments": process_record,
    }


def _assemble_final_segment_adapter(
    *,
    entries: list[dict[str, Any]],
    output_dir: Path,
    final_adapter_dir: Path,
) -> None:
    if not entries:
        raise Tau3MlxTrainingError(
            "cannot assemble a final adapter without committed segments"
        )
    final_adapter_dir.mkdir()
    last_index = len(entries) - 1
    for index, entry in enumerate(entries):
        record = entry.get("record")
        adapter_tree = (
            record.get("adapter_tree") if isinstance(record, dict) else None
        )
        if not isinstance(adapter_tree, dict):
            raise Tau3MlxTrainingError(
                f"segment {index} has no adapter tree for final assembly"
            )
        errors: list[str] = []
        source_root = _resolve_output_relative_path(
            adapter_tree.get("path"),
            output_dir,
            f"segment {index} adapter tree",
            errors,
        )
        if errors or source_root is None or not source_root.is_dir():
            raise Tau3MlxTrainingError(
                "segment adapter tree cannot be assembled: "
                + json.dumps(errors, sort_keys=True)
            )
        _require_safe_regular_tree(
            source_root,
            f"segment {index} adapter tree",
        )
        for source in sorted(
            path for path in source_root.rglob("*") if path.is_file()
        ):
            relative = source.relative_to(source_root)
            if index != last_index and relative.as_posix() in {
                "adapters.safetensors",
                "adapter_config.json",
                "config.json",
            }:
                continue
            destination = final_adapter_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if (
                    not destination.is_file()
                    or _sha256_file(destination) != _sha256_file(source)
                ):
                    raise Tau3MlxTrainingError(
                        "segment adapter assembly collision differs: "
                        + relative.as_posix()
                    )
                continue
            shutil.copy2(source, destination)


def _load_completed_process_segments(
    *,
    manifest_path: Path,
    output_dir: Path,
    losses: dict[str, list[float]],
    cfg: Tau3MlxTrainingConfig,
) -> dict[str, Any]:
    if (
        not manifest_path.is_file()
        or path_has_symlink_component(manifest_path, include_leaf=True)
        or bool(manifest_path.stat().st_mode & 0o222)
    ):
        raise Tau3MlxTrainingError(
            "completed process manifest must be an immutable regular file"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Tau3MlxTrainingError(
            f"completed process manifest is invalid: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise Tau3MlxTrainingError(
            "completed process manifest must be an object"
        )
    process_record = {
        **manifest,
        "manifest": _output_file_record(manifest_path, output_dir),
    }
    validation = validate_tau3_process_segments(
        process_record,
        output_dir=output_dir,
        expected_config=_process_segment_config_binding(cfg),
    )
    process_record["validation"] = validation
    if validation.get("passed") is not True:
        raise Tau3MlxTrainingError(
            "completed process manifest failed validation: "
            + json.dumps(validation.get("errors"), sort_keys=True)
        )
    entries = process_record["segments"]
    telemetry_count = 0
    peak_rss_kb = 0
    exit_code: int | None = 0
    for entry in entries:
        record = entry["record"]
        telemetry_path = _resolve_output_relative_path(
            record["telemetry"]["path"],
            output_dir,
            "completed segment telemetry",
            [],
        )
        assert telemetry_path is not None
        _replay_telemetry_losses(telemetry_path, losses)
        telemetry_count += int(record.get("telemetry_event_count") or 0)
        peak_rss_kb = max(
            peak_rss_kb,
            int(record.get("peak_child_rss_kb") or 0),
        )
        exit_code = record.get("exit_code")
    return {
        "terminal_status": "success",
        "exit_code": exit_code,
        "timed_out": False,
        "telemetry_event_count": telemetry_count,
        "peak_child_rss_kb": peak_rss_kb,
        "process_segments": process_record,
    }


def _load_committed_process_segments(
    *,
    committed_root: Path,
    output_dir: Path,
    plan: dict[str, Any],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    telemetry_paths: list[Path] = []
    previous_record_sha: str | None = None
    previous_adapter_file: Path | None = None
    previous_optimizer_file: Path | None = None
    telemetry_count = 0
    peak_rss_kb = 0
    errors: list[str] = []
    planned = plan["segments"]
    children = sorted(committed_root.iterdir())
    if any(not child.is_dir() for child in children):
        raise Tau3MlxTrainingError(
            "committed segment root contains a non-directory artifact"
        )
    if len(children) > len(planned):
        raise Tau3MlxTrainingError(
            "committed segment count exceeds the frozen process plan"
        )
    for index, segment_dir in enumerate(children):
        expected = planned[index]
        if segment_dir.name != expected["segment_id"]:
            errors.append(
                f"committed segment {index} is not the next planned segment"
            )
            continue
        record_path = segment_dir / "segment_record.json"
        if (
            not record_path.is_file()
            or path_has_symlink_component(record_path, include_leaf=True)
            or bool(record_path.stat().st_mode & 0o222)
        ):
            errors.append(
                f"committed segment {index} record is unavailable or mutable"
            )
            continue
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(f"committed segment {index} record is invalid: {exc}")
            continue
        if not isinstance(record, dict):
            errors.append(f"committed segment {index} record is not an object")
            continue
        for key in (
            "index",
            "segment_id",
            "start_iter",
            "end_iter",
            "iteration_count",
        ):
            if record.get(key) != expected.get(key):
                errors.append(
                    f"committed segment {index} differs from plan field {key}"
                )
        record_sha = record.get("segment_record_sha256")
        replay_record = dict(record)
        replay_record.pop("segment_record_sha256", None)
        if record_sha != _canonical_sha256(replay_record):
            errors.append(f"committed segment {index} record hash mismatch")
        if record.get("previous_segment_record_sha256") != previous_record_sha:
            errors.append(f"committed segment {index} record chain mismatch")
        if record.get("terminal_status") != "success":
            errors.append(f"committed segment {index} is not successful")

        adapter_input = record.get("adapter_input")
        optimizer_input = record.get("optimizer_state_input")
        if index == 0:
            if adapter_input is not None or optimizer_input is not None:
                errors.append(
                    "first committed segment unexpectedly has state inputs"
                )
        else:
            if (
                not isinstance(adapter_input, dict)
                or previous_adapter_file is None
                or adapter_input.get("sha256")
                != _sha256_file(previous_adapter_file)
            ):
                errors.append(
                    f"committed segment {index} adapter chain mismatch"
                )
            if (
                not isinstance(optimizer_input, dict)
                or previous_optimizer_file is None
                or optimizer_input.get("sha256")
                != _sha256_file(previous_optimizer_file)
            ):
                errors.append(
                    f"committed segment {index} optimizer chain mismatch"
                )

        telemetry = _validate_bound_file_record(
            record.get("telemetry"),
            output_dir,
            f"committed segment {index} telemetry",
            errors,
        )
        adapter = _validate_bound_file_record(
            record.get("adapter_output"),
            output_dir,
            f"committed segment {index} adapter",
            errors,
        )
        optimizer = _validate_bound_file_record(
            record.get("optimizer_state_output"),
            output_dir,
            f"committed segment {index} optimizer",
            errors,
        )
        expected_adapter_root = segment_dir / "adapter"
        if adapter is not None and adapter.parent != expected_adapter_root:
            errors.append(
                f"committed segment {index} adapter is outside its adapter tree"
            )
        if optimizer is not None and optimizer != (
            segment_dir / "optimizer_state.safetensors"
        ):
            errors.append(
                f"committed segment {index} optimizer path is not canonical"
            )
        adapter_tree = record.get("adapter_tree")
        if isinstance(adapter_tree, dict):
            replay_tree = _relative_fingerprint_tree(
                _fingerprint_tree(expected_adapter_root),
                output_dir,
            )
            if replay_tree != adapter_tree:
                errors.append(
                    f"committed segment {index} adapter tree mismatch"
                )
            if not _tree_is_readonly_regular(expected_adapter_root):
                errors.append(
                    f"committed segment {index} adapter tree is mutable or unsafe"
                )
        else:
            errors.append(
                f"committed segment {index} adapter tree is unavailable"
            )
        if telemetry is not None:
            if _telemetry_has_nonfinite_loss(telemetry):
                errors.append(
                    f"committed segment {index} telemetry contains "
                    "non-finite loss"
                )
            telemetry_paths.append(telemetry)
        telemetry_count += int(record.get("telemetry_event_count") or 0)
        peak_rss_kb = max(
            peak_rss_kb,
            int(record.get("peak_child_rss_kb") or 0),
        )
        entries.append(
            {
                "record": record,
                "record_file": _output_file_record(
                    record_path,
                    output_dir,
                ),
            }
        )
        previous_record_sha = (
            str(record_sha) if isinstance(record_sha, str) else None
        )
        previous_adapter_file = adapter
        previous_optimizer_file = optimizer
    if errors:
        raise Tau3MlxTrainingError(
            "committed process segment chain failed validation: "
            + json.dumps(errors, sort_keys=True)
        )
    return {
        "entries": entries,
        "previous_record_sha256": previous_record_sha,
        "previous_adapter_file": previous_adapter_file,
        "previous_optimizer_file": previous_optimizer_file,
        "telemetry_paths": telemetry_paths,
        "telemetry_event_count": telemetry_count,
        "peak_child_rss_kb": peak_rss_kb,
        "accepted_segment_count": len(entries),
    }


def _fresh_process_segment_partial(
    process_root: Path,
    segment_id: str,
) -> Path:
    canonical = process_root / f".{segment_id}.partial"
    if not os.path.lexists(canonical):
        return canonical
    for attempt in range(1, 10_000):
        candidate = process_root / (
            f".{segment_id}.partial-retry-{attempt:04d}"
        )
        if not os.path.lexists(candidate):
            return candidate
    raise Tau3MlxTrainingError(
        f"no fresh partial directory name remains for {segment_id}"
    )


def _replay_telemetry_losses(
    path: Path,
    losses: dict[str, list[float]],
) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise Tau3MlxTrainingError(
            f"committed telemetry cannot be replayed: {path}: {exc}"
        ) from exc
    for line_number, line in enumerate(lines, start=1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Tau3MlxTrainingError(
                "committed telemetry is invalid JSON at "
                f"{path}:{line_number}: {exc}"
            ) from exc
        text = event.get("text") if isinstance(event, dict) else None
        if not isinstance(text, str):
            continue
        if NONFINITE_LOSS_RE.search(text):
            raise Tau3MlxTrainingError(
                "committed telemetry contains non-finite loss at "
                f"{path}:{line_number}"
            )
        for match in LOSS_RE.finditer(text):
            value = float(match.group("loss"))
            kind = match.group("kind").lower()
            losses[
                "validation"
                if kind in {"valid", "validation", "val"}
                else "train"
            ].append(value)


def _require_existing_json_equal(
    path: Path,
    expected: dict[str, Any],
    label: str,
) -> None:
    if (
        not path.is_file()
        or path_has_symlink_component(path, include_leaf=True)
        or bool(path.stat().st_mode & 0o222)
    ):
        raise Tau3MlxTrainingError(
            f"{label} must be an immutable regular file"
        )
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Tau3MlxTrainingError(f"{label} is invalid: {exc}") from exc
    if actual != expected:
        raise Tau3MlxTrainingError(
            f"{label} differs from the fresh invocation"
        )


def _validate_segment_resume_prelaunch(
    path: Path,
    expected: dict[str, Any],
    output_dir: Path,
) -> None:
    if (
        not path.is_file()
        or path_has_symlink_component(path, include_leaf=True)
        or bool(path.stat().st_mode & 0o222)
    ):
        raise Tau3MlxTrainingError(
            "segmented-resume prelaunch receipt must be immutable"
        )
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Tau3MlxTrainingError(
            f"segmented-resume prelaunch receipt is invalid: {exc}"
        ) from exc
    if not isinstance(actual, dict):
        raise Tau3MlxTrainingError(
            "segmented-resume prelaunch receipt must be an object"
        )
    replay = dict(expected)
    replay["created_at"] = actual.get("created_at")
    if actual != replay:
        raise Tau3MlxTrainingError(
            "segmented-resume prelaunch receipt differs from fresh preflight"
        )
    schema = check_schema_contract(
        actual,
        name_or_id="tau3_mlx_training_run",
    )
    if schema.get("passed") is not True:
        raise Tau3MlxTrainingError(
            "segmented-resume prelaunch receipt violates schema: "
            + json.dumps(schema.get("errors"), sort_keys=True)
        )
    mlx_binding = actual.get("mlx_lora_config")
    errors: list[str] = []
    _validate_bound_file_record(
        mlx_binding,
        output_dir,
        "segmented-resume MLX config",
        errors,
    )
    if errors:
        raise Tau3MlxTrainingError(
            "segmented-resume prelaunch artifact binding failed: "
            + json.dumps(errors, sort_keys=True)
        )


def _validate_process_segment_plan_records(
    plan: dict[str, Any],
    errors: list[str],
) -> None:
    records = plan.get("segments")
    if not isinstance(records, list) or not records:
        errors.append("plan segments must be a non-empty array")
        return
    total_iters = (plan.get("policy") or {}).get("total_iters")
    expected_start = 0
    previous: str | None = None
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"plan segment {index} must be an object")
            continue
        digest = record.get("plan_record_sha256")
        replay = dict(record)
        replay.pop("plan_record_sha256", None)
        if digest != _canonical_sha256(replay):
            errors.append(f"plan segment {index} hash mismatch")
        if record.get("index") != index:
            errors.append(f"plan segment {index} index mismatch")
        if record.get("previous_plan_record_sha256") != previous:
            errors.append(f"plan segment {index} chain mismatch")
        if record.get("start_iter") != expected_start:
            errors.append(f"plan segment {index} is not contiguous")
        end = record.get("end_iter")
        start = record.get("start_iter")
        if not isinstance(start, int) or not isinstance(end, int) or end <= start:
            errors.append(f"plan segment {index} has invalid boundaries")
        else:
            if record.get("iteration_count") != end - start:
                errors.append(f"plan segment {index} iteration_count mismatch")
            expected_start = end
        previous = digest if isinstance(digest, str) else previous
    if isinstance(total_iters, int) and expected_start != total_iters:
        errors.append("plan does not cover total_iters")


def _validate_process_segment_policy(
    policy: dict[str, Any],
    errors: list[str],
) -> None:
    exact = {
        "schema_version": TAU3_MLX_PROCESS_SEGMENTS_SCHEMA_VERSION,
        "execution": "strictly_sequential_child_processes",
        "boundary_semantics": "half_open_microbatch_iterations",
        "state_continuity": "adapter_and_optimizer_sha256",
        "partial_directory_policy": "fresh_then_atomic_rename",
        "aggregate_telemetry_policy": (
            "byte_concatenation_in_segment_order"
        ),
        "dropout": 0.0,
    }
    for field, expected in exact.items():
        if policy.get(field) != expected:
            errors.append(f"policy field {field} mismatch")
    total = policy.get("total_iters")
    cadence = policy.get("process_segment_iters")
    accumulation = policy.get("gradient_accumulation")
    report = policy.get("report_every")
    if (
        not isinstance(total, int)
        or not isinstance(cadence, int)
        or not isinstance(accumulation, int)
        or not isinstance(report, int)
        or min(total, cadence, accumulation, report) <= 0
        or cadence > total
        or total % accumulation
        or cadence % accumulation
        or cadence % report
    ):
        errors.append("policy segment/alignment values are invalid")


def _process_segment_config_binding(
    cfg: Tau3MlxTrainingConfig,
) -> dict[str, Any]:
    return {
        "iters": cfg.iters,
        "process_segment_iters": cfg.process_segment_iters,
        "grad_accumulation": cfg.grad_accumulation,
        "report_every": cfg.report_every,
        "dropout": cfg.dropout,
    }


def _validate_process_segment_config_binding(
    policy: dict[str, Any],
    config: dict[str, Any] | None,
    errors: list[str],
) -> None:
    if config is None:
        return
    expected = {
        "total_iters": config.get("iters"),
        "process_segment_iters": config.get("process_segment_iters"),
        "gradient_accumulation": config.get("grad_accumulation"),
        "report_every": config.get("report_every"),
        "dropout": config.get("dropout"),
    }
    for field, value in expected.items():
        if policy.get(field) != value:
            errors.append(
                f"policy field {field} does not match training config"
            )


def _validate_process_segment_recovery_artifacts(
    recovery: Any,
    output: Path,
    errors: list[str],
) -> None:
    if not isinstance(recovery, dict):
        errors.append("recovery must be an object")
        return
    partials = recovery.get("preserved_partial_artifact_trees")
    if not isinstance(partials, list):
        errors.append(
            "recovery preserved_partial_artifact_trees must be an array"
        )
        partials = []
    for index, tree in enumerate(partials):
        if not isinstance(tree, dict):
            errors.append(f"recovery partial tree {index} is invalid")
            continue
        if tree.get("artifact_kind") == "regular_file":
            path = _validate_bound_file_record(
                tree,
                output,
                f"recovery partial file {index}",
                errors,
            )
            if (
                path is not None
                and tree.get("size") != path.stat().st_size
            ):
                errors.append(
                    f"recovery partial file {index} size mismatch"
                )
            continue
        root = _resolve_output_relative_path(
            tree.get("path"),
            output,
            f"recovery partial tree {index}",
            errors,
        )
        if root is not None:
            if not _tree_is_readonly_regular(root):
                errors.append(
                    f"recovery partial tree {index} is mutable or unsafe"
                )
            else:
                replay = _relative_fingerprint_tree(
                    _fingerprint_tree(root),
                    output,
                )
                expected_tree = dict(tree)
                expected_tree.pop("artifact_kind", None)
                if replay != expected_tree:
                    errors.append(f"recovery partial tree {index} mismatch")
    failed = recovery.get("preserved_failed_artifact_tree")
    if failed is not None:
        if not isinstance(failed, dict):
            errors.append("recovery failed artifact tree is invalid")
        else:
            root = _resolve_output_relative_path(
                failed.get("path"),
                output,
                "recovery failed artifact tree",
                errors,
            )
            if root is not None:
                if not _tree_is_readonly_regular(root):
                    errors.append(
                        "recovery failed artifact tree is mutable or unsafe"
                    )
                else:
                    replay = _relative_fingerprint_tree(
                        _fingerprint_tree(root),
                        output,
                    )
                    if replay != failed:
                        errors.append("recovery failed artifact tree mismatch")


def _validate_bound_json_file(
    binding: Any,
    output: Path,
    label: str,
    errors: list[str],
) -> dict[str, Any] | None:
    path = _validate_bound_file_record(binding, output, label, errors)
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"{label} is not valid JSON: {exc}")
        return None
    if not isinstance(payload, dict):
        errors.append(f"{label} JSON must be an object")
        return None
    return payload


def _validate_bound_file_record(
    binding: Any,
    output: Path,
    label: str,
    errors: list[str],
) -> Path | None:
    if not isinstance(binding, dict):
        errors.append(f"{label} binding must be an object")
        return None
    path = _resolve_output_relative_path(binding.get("path"), output, label, errors)
    if path is None:
        return None
    if not path.is_file() or path_has_symlink_component(path, include_leaf=True):
        errors.append(f"{label} must be a regular non-symlink file")
        return None
    expected = binding.get("sha256")
    if expected != _sha256_file(path):
        errors.append(f"{label} sha256 mismatch")
    if binding.get("read_only") is not True or bool(path.stat().st_mode & 0o222):
        errors.append(f"{label} must be read-only")
    return path


def _resolve_output_relative_path(
    value: Any,
    output: Path,
    label: str,
    errors: list[str],
) -> Path | None:
    if not isinstance(value, str) or not value:
        errors.append(f"{label} path must be a non-empty string")
        return None
    raw = Path(value)
    if raw.is_absolute() or ".." in raw.parts:
        errors.append(f"{label} path must be output-relative")
        return None
    try:
        path = (output / raw).resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        errors.append(f"{label} path is unavailable: {exc}")
        return None
    if not path.is_relative_to(output):
        errors.append(f"{label} path escapes output_dir")
        return None
    return path


def _replace_command_option(
    command: list[str],
    option: str,
    value: str,
) -> list[str]:
    result = list(command)
    try:
        index = result.index(option)
    except ValueError as exc:
        raise Tau3MlxTrainingError(f"child command is missing {option}") from exc
    if index + 1 >= len(result):
        raise Tau3MlxTrainingError(f"child command has no value for {option}")
    result[index + 1] = value
    return result


def _select_segment_adapter_file(adapter_dir: Path) -> Path | None:
    canonical = adapter_dir / "adapters.safetensors"
    if canonical.is_file() and canonical.stat().st_size > 0:
        return canonical
    candidates = sorted(
        path
        for path in adapter_dir.iterdir()
        if path.is_file()
        and path.stat().st_size > 0
        and path.suffix in {".safetensors", ".npz", ".bin"}
    )
    if len(candidates) != 1:
        return None
    return candidates[0]


def _relative_fingerprint_tree(
    fingerprint: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    return {
        **fingerprint,
        "path": _relative_output_path(Path(fingerprint["path"]), output_root),
    }


def _preserved_partial_artifact_record(
    path: Path,
    output_root: Path,
) -> dict[str, Any]:
    mode = os.lstat(path).st_mode
    if stat.S_ISDIR(mode):
        return {
            **_relative_fingerprint_tree(
                _fingerprint_tree(path),
                output_root,
            ),
            "artifact_kind": "directory",
        }
    if stat.S_ISREG(mode):
        return {
            "artifact_kind": "regular_file",
            "path": _relative_output_path(path, output_root),
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
            "read_only": not bool(mode & 0o222),
        }
    raise Tau3MlxTrainingError(
        f"preserved partial has unsupported file type: {path}"
    )


def _expected_assembled_adapter_files(
    entries: list[dict[str, Any]],
    errors: list[str],
) -> list[dict[str, Any]]:
    assembled: dict[str, dict[str, Any]] = {}
    last_index = len(entries) - 1
    for index, entry in enumerate(entries):
        record = entry.get("record")
        tree = record.get("adapter_tree") if isinstance(record, dict) else None
        files = tree.get("files") if isinstance(tree, dict) else None
        if not isinstance(files, list):
            errors.append(f"segment {index} adapter files are unavailable")
            continue
        for file_record in files:
            if not isinstance(file_record, dict):
                errors.append(
                    f"segment {index} adapter file record is invalid"
                )
                continue
            relative = file_record.get("path")
            if not isinstance(relative, str):
                errors.append(
                    f"segment {index} adapter file path is invalid"
                )
                continue
            if index != last_index and relative in {
                "adapters.safetensors",
                "adapter_config.json",
                "config.json",
            }:
                continue
            previous = assembled.get(relative)
            if previous is not None and previous != file_record:
                errors.append(
                    f"segment adapter assembly collision differs: {relative}"
                )
                continue
            assembled[relative] = file_record
    return [assembled[path] for path in sorted(assembled)]


def _make_tree_readonly(root: Path) -> None:
    directories, files = _regular_tree_nodes(root, "read-only tree")
    for path in files:
        path.chmod(0o444)
    for path in sorted(directories, reverse=True):
        path.chmod(0o555)


def _make_files_readonly(root: Path) -> None:
    _, files = _regular_tree_nodes(root, "read-only files")
    for path in files:
        path.chmod(0o444)


def _commit_readonly_directory(source: Path, destination: Path) -> None:
    _require_safe_regular_tree(source, "directory commit source")
    _fsync_tree(source)
    _make_files_readonly(source)
    _fsync_tree(source)
    os.replace(source, destination)
    _fsync_directory(destination.parent)
    _make_tree_readonly(destination)
    _fsync_tree(destination)
    _fsync_directory(destination.parent)


def _commit_readonly_file(
    source: Path,
    destination: Path,
    content: bytes,
) -> None:
    _publish_new_readonly_file(source, destination, content)


def _publish_new_readonly_file(
    source: Path,
    destination: Path,
    content: bytes,
) -> None:
    with source.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        os.fchmod(handle.fileno(), 0o444)
        os.fsync(handle.fileno())
    if not stat.S_ISREG(os.lstat(source).st_mode):
        raise Tau3MlxTrainingError(
            f"publication partial is not a regular file: {source}"
        )
    try:
        os.link(
            source,
            destination,
            follow_symlinks=False,
        )
    except FileExistsError as exc:
        raise Tau3MlxTrainingError(
            f"create-once publication destination already exists: {destination}"
        ) from exc
    _fsync_directory(destination.parent)
    os.unlink(source)
    _fsync_directory(destination.parent)


def _fsync_tree(root: Path) -> None:
    directories, files = _regular_tree_nodes(root, "fsync tree")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    for path in files:
        descriptor = os.open(path, os.O_RDONLY | nofollow)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise Tau3MlxTrainingError(
                    f"fsync tree file changed type: {path}"
                )
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    for path in sorted(directories[1:], reverse=True):
        _fsync_directory(path)
    _fsync_directory(root)


def _fsync_directory(path: Path) -> None:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise Tau3MlxTrainingError(
                f"fsync directory changed type: {path}"
            )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _freeze_regular_file_nofollow(path: Path, label: str) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise Tau3MlxTrainingError(
            f"{label} cannot be opened safely: {path}: {exc}"
        ) from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise Tau3MlxTrainingError(
                f"{label} must be a regular file: {path}"
            )
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _tree_is_readonly_regular(root: Path) -> bool:
    try:
        directories, files = _regular_tree_nodes(
            root,
            "read-only validation tree",
        )
    except (OSError, Tau3MlxTrainingError):
        return False
    return all(
        not bool(os.lstat(path).st_mode & 0o222)
        for path in [*directories, *files]
    )


def _require_safe_regular_tree(root: Path, label: str) -> None:
    _regular_tree_nodes(root, label)


def _regular_tree_nodes(
    root: Path,
    label: str,
) -> tuple[list[Path], list[Path]]:
    if path_has_symlink_component(root, include_leaf=True):
        raise Tau3MlxTrainingError(
            f"{label} must not contain symlink components: {root}"
        )
    try:
        root_status = os.lstat(root)
    except OSError as exc:
        raise Tau3MlxTrainingError(
            f"{label} is unavailable: {root}: {exc}"
        ) from exc
    if not stat.S_ISDIR(root_status.st_mode):
        raise Tau3MlxTrainingError(
            f"{label} root must be a regular directory: {root}"
        )
    directories = [root]
    files: list[Path] = []
    pending = [root]
    while pending:
        current = pending.pop()
        try:
            with os.scandir(current) as iterator:
                entries = sorted(
                    iterator,
                    key=lambda entry: entry.name,
                )
        except OSError as exc:
            raise Tau3MlxTrainingError(
                f"{label} cannot be traversed safely: {current}: {exc}"
            ) from exc
        for entry in entries:
            path = Path(entry.path)
            try:
                mode = entry.stat(follow_symlinks=False).st_mode
            except OSError as exc:
                raise Tau3MlxTrainingError(
                    f"{label} node is unavailable: {path}: {exc}"
                ) from exc
            if stat.S_ISLNK(mode):
                raise Tau3MlxTrainingError(
                    f"{label} contains a symlink: {path}"
                )
            if stat.S_ISDIR(mode):
                directories.append(path)
                pending.append(path)
            elif stat.S_ISREG(mode):
                files.append(path)
            else:
                raise Tau3MlxTrainingError(
                    f"{label} contains a non-regular node: {path}"
                )
    directories.sort()
    files.sort()
    return directories, files


def _freeze_process_segment_partials(process_root: Path) -> list[Path]:
    partials = sorted(process_root.glob(".*.partial*"))
    for path in partials:
        mode = os.lstat(path).st_mode
        if stat.S_ISDIR(mode):
            _require_safe_regular_tree(path, "preserved process partial")
            _fsync_tree(path)
            _make_tree_readonly(path)
            _fsync_tree(path)
        elif stat.S_ISREG(mode):
            _freeze_regular_file_nofollow(
                path,
                "preserved process partial",
            )
        else:
            raise Tau3MlxTrainingError(
                f"preserved process partial must be a regular file or "
                f"directory: {path}"
            )
    return partials


def _run_child(
    *,
    command: list[str],
    cwd: Path,
    telemetry_path: Path,
    timeout_seconds: int,
    losses: dict[str, list[float]],
    disable_compile: bool,
) -> tuple[int | None, bool, int, int]:
    environment = os.environ.copy()
    if disable_compile:
        environment["MLX_DISABLE_COMPILE"] = "1"
    else:
        environment.pop("MLX_DISABLE_COMPILE", None)
    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
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
    if NONFINITE_LOSS_RE.search(text):
        return "crash"
    if exit_code == 0:
        return "success"
    return "crash"


def _telemetry_has_nonfinite_loss(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return False
    return NONFINITE_LOSS_RE.search(text) is not None


def _fingerprint_tree(root: Path) -> dict[str, Any]:
    files = []
    if os.path.lexists(root):
        _, regular_files = _regular_tree_nodes(root, "fingerprint tree")
        for path in regular_files:
            rel = path.relative_to(root).as_posix()
            files.append(
                {
                    "path": rel,
                    "size": os.lstat(path).st_size,
                    "sha256": _sha256_file(path),
                    "kind": _fingerprint_kind(rel),
                }
            )
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


def _publish_new_json_readonly(
    path: Path,
    payload: dict[str, Any],
) -> None:
    partial = _fresh_process_segment_partial(path.parent, path.name)
    content = (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _publish_new_readonly_file(partial, path, content)


def _freeze_json_publication_partials(destination: Path) -> list[Path]:
    partials = sorted(
        destination.parent.glob(f".{destination.name}.partial*")
    )
    for path in partials:
        mode = os.lstat(path).st_mode
        if not stat.S_ISREG(mode):
            raise Tau3MlxTrainingError(
                "JSON publication partial must be a regular file: "
                f"{path}"
            )
        _freeze_regular_file_nofollow(path, "JSON publication partial")
    return partials


def _is_immutable_regular_file(path: Path) -> bool:
    try:
        mode = os.lstat(path).st_mode
    except OSError:
        return False
    return (
        stat.S_ISREG(mode)
        and not path_has_symlink_component(path, include_leaf=True)
        and not bool(mode & 0o222)
    )


def _validate_existing_final_training_receipt(
    path: Path,
    expected: dict[str, Any],
) -> dict[str, Any]:
    if not _is_immutable_regular_file(path):
        raise Tau3MlxTrainingError(
            "existing final training receipt is mutable or unsafe"
        )
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Tau3MlxTrainingError(
            f"existing final training receipt is invalid: {exc}"
        ) from exc
    if not isinstance(actual, dict):
        raise Tau3MlxTrainingError(
            "existing final training receipt must be an object"
        )
    replay = dict(expected)
    for field in ("created_at", "elapsed_seconds"):
        replay[field] = actual.get(field)
    if actual != replay:
        raise Tau3MlxTrainingError(
            "existing final training receipt differs from recovered evidence"
        )
    schema = check_schema_contract(
        actual,
        name_or_id="tau3_mlx_training_run",
    )
    if schema.get("passed") is not True:
        raise Tau3MlxTrainingError(
            "existing final training receipt violates schema: "
            + json.dumps(schema.get("errors"), sort_keys=True)
        )
    return actual


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
        and (
            cfg.process_segment_iters is None
            or (
                cfg.prefix_cache_training
                and cfg.exposure_ledger_training
                and cfg.dropout == 0
                and cfg.process_segment_iters >= cfg.grad_accumulation
                and cfg.process_segment_iters <= cfg.iters
                and cfg.iters % cfg.grad_accumulation == 0
                and cfg.process_segment_iters % cfg.grad_accumulation == 0
                and cfg.process_segment_iters % cfg.report_every == 0
            )
        )
        and 1 <= cfg.timeout_seconds <= MAX_TIMEOUT_SECONDS
        and (
            not cfg.prefix_cache_training
            or (
                cfg.batch_size == 1
                and (
                    cfg.grad_accumulation == 1
                    or cfg.exposure_ledger_training
                )
                and cfg.mask_prompt
                and not cfg.grad_checkpoint
                and cfg.disable_compile
                and not cfg.fixed_shape_padding
            )
        )
        and not (cfg.exposure_ledger_training and cfg.fixed_shape_padding)
        and not (cfg.exposure_ledger_training and not cfg.mask_prompt)
    )


def _config_record(
    cfg: Tau3MlxTrainingConfig,
    *,
    resume: dict[str, Any] | None = None,
    exposure_binding: dict[str, Any] | None = None,
    prefix_equivalence_binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
        "disable_compile": cfg.disable_compile,
        "fixed_shape_padding": cfg.fixed_shape_padding,
        "prefix_cache_training": cfg.prefix_cache_training,
        "exposure_ledger_training": cfg.exposure_ledger_training,
        "clear_cache_threshold": cfg.clear_cache_threshold,
        "process_segment_iters": cfg.process_segment_iters,
        "timeout_seconds": cfg.timeout_seconds,
    }
    if resume is not None:
        record["resume"] = resume
    if exposure_binding is not None:
        ledger = exposure_binding.get("ledger") if isinstance(exposure_binding.get("ledger"), dict) else {}
        coverage = exposure_binding.get("coverage") if isinstance(exposure_binding.get("coverage"), dict) else {}
        record["exposure_schedule"] = {
            "microbatch_iterations": ledger.get("microbatch_iterations"),
            "optimizer_steps": ledger.get("optimizer_steps")
            or coverage.get("optimizer_steps"),
            "gradient_accumulation_steps": cfg.grad_accumulation,
            "mlx_lm_iters_are_microbatches": True,
        }
    if prefix_equivalence_binding is not None:
        record["prefix_equivalence"] = {
            "sha256": prefix_equivalence_binding.get("sha256"),
            "validation_passed": prefix_equivalence_binding.get(
                "validation_passed"
            ),
        }
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
