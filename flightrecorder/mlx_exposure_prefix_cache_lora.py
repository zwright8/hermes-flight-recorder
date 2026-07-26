"""Replay a Tau-3 exposure ledger with qualified detached-prefix MLX LoRA.

This trainer combines two governed mechanisms:

* every row is selected in the exact optimizer-step/microbatch order from a
  validated Tau-3 exposure ledger; and
* every complete prompt is materialized into a fresh detached inference cache
  before target-only gradients are computed over the assistant suffix.

The method is an approximation to full-sequence SFT. It is therefore available
only when a passing, hash-bound Tau-3 prefix-equivalence artifact is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .mlx_exposure_lora import (
    ExposureLoraError,
    _assert_dataset_matches_receipt,
    _dataset_item,
    _parse_exposure_args,
    load_exposure_schedule,
)
from .mlx_prefix_cache_lora import (
    PrefixCacheTrainingError,
    _evaluate,
    _materialize_prompt_cache,
    _target_value_and_grad,
    split_supervised_tokens,
)
from .tau3_exposure import Tau3ExposureError
from .tau3_prefix_equivalence import validate_tau3_prefix_equivalence


class ExposurePrefixCacheLoraError(ValueError):
    """Raised when governed exposure-prefix training cannot run safely."""


_RUNTIME_SCHEDULE: dict[str, Any] | None = None


def train_with_exposure_prefix_cache(
    model: Any,
    optimizer: Any,
    train_dataset: Any,
    val_dataset: Any = None,
    args: Any = None,
    loss: Any = None,
    iterate_batches: Any = None,
    training_callback: Any = None,
) -> None:
    """Train target suffixes in the exact exposure-ledger optimizer schedule."""

    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_map

    del loss, iterate_batches
    if args is None:
        raise ExposurePrefixCacheLoraError("training arguments are required")
    schedule = _require_schedule()
    if args.batch_size != 1:
        raise ExposurePrefixCacheLoraError(
            "exposure-prefix training requires batch_size=1"
        )
    if args.grad_accumulation_steps < 1:
        raise ExposurePrefixCacheLoraError(
            "grad_accumulation_steps must be at least one"
        )
    if args.grad_checkpoint:
        raise ExposurePrefixCacheLoraError(
            "exposure-prefix training does not use gradient checkpointing"
        )
    if args.iters != schedule["microbatch_iterations"]:
        raise ExposurePrefixCacheLoraError(
            "training iters do not match the exposure microbatch count"
        )
    sampler = schedule["receipt"].get("sampler_config", {})
    if args.grad_accumulation_steps != sampler.get(
        "gradient_accumulation_steps"
    ):
        raise ExposurePrefixCacheLoraError(
            "gradient accumulation does not match the exposure receipt"
        )
    world = mx.distributed.init()
    if world.size() != 1:
        raise ExposurePrefixCacheLoraError(
            "exposure-prefix training requires one local process"
        )
    _assert_dataset_matches_receipt(train_dataset, schedule["receipt"])
    if mx.metal.is_available():
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])

    row_order = _flatten_row_schedule(schedule)
    token_counts = _flatten_token_counts(schedule)
    if len(row_order) != args.iters:
        raise ExposurePrefixCacheLoraError(
            "flattened exposure order does not match training iters"
        )
    if len(token_counts) != args.iters:
        raise ExposurePrefixCacheLoraError(
            "flattened exposure token counts do not match training iters"
        )
    segment = _prepare_segment(
        schedule,
        args,
        model=model,
        optimizer=optimizer,
        mx=mx,
        tree_flatten=tree_flatten,
    )
    segment_start = int(segment["start"])
    segment_end = int(segment["end"])
    segment_rows = row_order[segment_start:segment_end]
    segment_token_counts = token_counts[segment_start:segment_end]
    if segment["enabled"]:
        print(
            "Starting exposure-ledger prefix-cache training..., "
            f"global microbatches: [{segment_start}, {segment_end}) "
            f"of {args.iters}, "
            f"global optimizer steps: {schedule['optimizer_steps']}",
            flush=True,
        )
    else:
        print(
            "Starting exposure-ledger prefix-cache training..., "
            f"microbatches: {args.iters}, "
            f"optimizer steps: {schedule['optimizer_steps']}",
            flush=True,
        )
    model.train()
    accumulated_gradients: Any = None
    train_losses: list[float] = []
    target_tokens = 0
    prompt_tokens = 0
    trained_target_tokens = sum(
        int(counts["supervised_tokens"])
        for counts in token_counts[:segment_start]
    )
    trained_prompt_tokens = sum(
        int(counts["prompt_tokens"])
        for counts in token_counts[:segment_start]
    )
    report_started = time.perf_counter()

    for iteration, (row_index, expected_tokens) in enumerate(
        zip(segment_rows, segment_token_counts, strict=True),
        start=segment_start + 1,
    ):
        if val_dataset is not None and (
            iteration == 1
            or iteration % args.steps_per_eval == 0
            or iteration == args.iters
        ):
            validation_started = time.perf_counter()
            validation_loss = _evaluate(
                model,
                val_dataset,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
            )
            validation_seconds = time.perf_counter() - validation_started
            print(
                f"Iter {iteration}: Val loss {validation_loss:.3f}, "
                f"Val took {validation_seconds:.3f}s",
                flush=True,
            )
            if training_callback is not None:
                training_callback.on_val_loss_report(
                    {
                        "iteration": iteration - 1,
                        "val_loss": validation_loss,
                        "val_time": validation_seconds,
                    }
                )
            report_started = time.perf_counter()

        tokens, prompt_offset = _dataset_item(train_dataset, row_index)
        if int(prompt_offset) != int(expected_tokens["prompt_tokens"]):
            raise ExposurePrefixCacheLoraError(
                "runtime prompt tokens differ from the exposure ledger at "
                f"global microbatch {iteration - 1}"
            )
        prefix, suffix_inputs, targets = split_supervised_tokens(
            tokens,
            int(prompt_offset),
            args.max_seq_length,
        )
        cache = _materialize_prompt_cache(model, prefix)
        loss_value, supervised_count, gradients = _target_value_and_grad(
            model,
            suffix_inputs,
            targets,
            cache,
        )
        current_targets = int(supervised_count.item())
        if current_targets != int(expected_tokens["supervised_tokens"]):
            raise ExposurePrefixCacheLoraError(
                "runtime supervised tokens differ from the exposure ledger at "
                f"global microbatch {iteration - 1}"
            )
        if accumulated_gradients is None:
            accumulated_gradients = gradients
        else:
            accumulated_gradients = tree_map(
                lambda current, previous: current + previous,
                gradients,
                accumulated_gradients,
            )
        update_boundary = iteration % args.grad_accumulation_steps == 0
        if update_boundary:
            if args.grad_accumulation_steps > 1:
                accumulated_gradients = tree_map(
                    lambda value: value / args.grad_accumulation_steps,
                    accumulated_gradients,
                )
            optimizer.update(model, accumulated_gradients)
            accumulated_gradients = None
            mx.eval(model.trainable_parameters(), optimizer.state)
        else:
            mx.eval(loss_value, supervised_count, accumulated_gradients)

        train_losses.append(float(loss_value.item()))
        target_tokens += current_targets
        prompt_tokens += int(prompt_offset)
        del cache, gradients, loss_value, supervised_count
        mx.clear_cache()

        if iteration % args.steps_per_report == 0 or iteration == args.iters:
            elapsed = time.perf_counter() - report_started
            report_steps = len(train_losses)
            average_loss = sum(train_losses) / report_steps
            trained_target_tokens += target_tokens
            trained_prompt_tokens += prompt_tokens
            learning_rate = float(optimizer.learning_rate.item())
            iterations_per_second = report_steps / elapsed
            tokens_per_second = target_tokens / elapsed
            peak_memory = mx.get_peak_memory() / 1e9
            print(
                f"Iter {iteration}: Train loss {average_loss:.3f}, "
                f"Learning Rate {learning_rate:.3e}, "
                f"It/sec {iterations_per_second:.3f}, "
                f"Tokens/sec {tokens_per_second:.3f}, "
                f"Trained Tokens {trained_target_tokens}, "
                f"Prompt Tokens {trained_prompt_tokens}, "
                f"Peak mem {peak_memory:.3f} GB",
                flush=True,
            )
            if training_callback is not None:
                training_callback.on_train_loss_report(
                    {
                        "iteration": iteration,
                        "train_loss": average_loss,
                        "learning_rate": learning_rate,
                        "iterations_per_second": iterations_per_second,
                        "tokens_per_second": tokens_per_second,
                        "trained_tokens": trained_target_tokens,
                        "peak_memory": peak_memory,
                    }
                )
            train_losses = []
            target_tokens = 0
            prompt_tokens = 0
            report_started = time.perf_counter()

        if (
            iteration % args.steps_per_save == 0
            and update_boundary
        ):
            _save_adapter(
                model,
                args.adapter_file,
                iteration,
                tree_flatten,
                mx,
                update_current=not bool(segment["enabled"]),
                create_once=bool(segment["enabled"]),
            )

    if accumulated_gradients is not None:
        raise ExposurePrefixCacheLoraError(
            "exposure schedule ended with a partial optimizer step"
        )
    expected_prompt_tokens = sum(
        int(counts["prompt_tokens"])
        for counts in token_counts[:segment_end]
    )
    expected_target_tokens = sum(
        int(counts["supervised_tokens"])
        for counts in token_counts[:segment_end]
    )
    if trained_prompt_tokens != expected_prompt_tokens:
        raise ExposurePrefixCacheLoraError(
            "cumulative prompt-token counter differs from the exposure ledger"
        )
    if trained_target_tokens != expected_target_tokens:
        raise ExposurePrefixCacheLoraError(
            "cumulative supervised-token counter differs from the exposure ledger"
        )
    if segment["enabled"]:
        actual_optimizer_step = int(optimizer.step.item())
        expected_optimizer_step = (
            segment_end // args.grad_accumulation_steps
        )
        if actual_optimizer_step != expected_optimizer_step:
            raise ExposurePrefixCacheLoraError(
                "segment optimizer step does not match its global microbatch end: "
                f"actual={actual_optimizer_step}, "
                f"expected={expected_optimizer_step}"
            )
    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    if segment["enabled"]:
        _atomic_save_safetensors(
            Path(args.adapter_file),
            adapter_weights,
            mx,
            metadata={
                "schema_version": "hfr.mlx_segment_adapter.v1",
                "global_microbatch_end": str(segment_end),
            },
        )
        _atomic_save_safetensors(
            Path(segment["optimizer_state_output"]),
            dict(tree_flatten(optimizer.state)),
            mx,
            metadata={
                "schema_version": "hfr.mlx_optimizer_state.v1",
                "global_microbatch_end": str(segment_end),
                "optimizer_step": str(
                    segment_end // args.grad_accumulation_steps
                ),
            },
        )
    else:
        mx.save_safetensors(str(args.adapter_file), adapter_weights)
    print(f"Saved final weights to {args.adapter_file}.", flush=True)


def _save_adapter(
    model: Any,
    adapter_file: str | Path,
    iteration: int,
    tree_flatten: Any,
    mx: Any,
    *,
    update_current: bool = True,
    create_once: bool = False,
) -> None:
    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    if update_current:
        if create_once:
            _atomic_save_safetensors(Path(adapter_file), adapter_weights, mx)
        else:
            mx.save_safetensors(str(adapter_file), adapter_weights)
    checkpoint = (
        Path(adapter_file).parent
        / f"{iteration:07d}_adapters.safetensors"
    )
    if create_once:
        _atomic_save_safetensors(checkpoint, adapter_weights, mx)
    else:
        mx.save_safetensors(str(checkpoint), adapter_weights)
    destinations = (
        f"{adapter_file} and {checkpoint}"
        if update_current
        else str(checkpoint)
    )
    print(
        f"Iter {iteration}: Saved adapter weights to {destinations}.",
        flush=True,
    )


def _flatten_row_schedule(schedule: dict[str, Any]) -> list[int]:
    order: list[int] = []
    for optimizer_step in schedule.get("steps", []):
        for microbatch in optimizer_step:
            if not isinstance(microbatch, list) or len(microbatch) != 1:
                raise ExposurePrefixCacheLoraError(
                    "exposure-prefix training requires one row per microbatch"
                )
            order.append(int(microbatch[0]))
    return order


def _flatten_token_counts(schedule: dict[str, Any]) -> list[dict[str, int]]:
    counts: list[dict[str, int]] = []
    parallel = schedule.get("microbatch_token_counts")
    steps = schedule.get("steps")
    if not isinstance(parallel, list) or not isinstance(steps, list):
        raise ExposurePrefixCacheLoraError(
            "exposure schedule token accounting is unavailable"
        )
    if len(parallel) != len(steps):
        raise ExposurePrefixCacheLoraError(
            "exposure schedule token accounting is not parallel to its steps"
        )
    for step, step_counts in zip(steps, parallel, strict=True):
        if not isinstance(step, list) or not isinstance(step_counts, list):
            raise ExposurePrefixCacheLoraError(
                "exposure schedule token accounting step is invalid"
            )
        if len(step_counts) != len(step):
            raise ExposurePrefixCacheLoraError(
                "exposure schedule token accounting microbatches are misaligned"
            )
        for item in step_counts:
            if not isinstance(item, dict):
                raise ExposurePrefixCacheLoraError(
                    "exposure schedule token accounting entry is invalid"
                )
            prompt = item.get("prompt_tokens")
            supervised = item.get("supervised_tokens")
            if (
                not isinstance(prompt, int)
                or prompt < 1
                or not isinstance(supervised, int)
                or supervised < 1
            ):
                raise ExposurePrefixCacheLoraError(
                    "exposure schedule token accounting values are invalid"
                )
            counts.append(
                {
                    "prompt_tokens": prompt,
                    "supervised_tokens": supervised,
                }
            )
    return counts


def _prepare_segment(
    schedule: dict[str, Any],
    args: Any,
    *,
    model: Any,
    optimizer: Any,
    mx: Any,
    tree_flatten: Any,
) -> dict[str, Any]:
    segment = schedule.get("segment")
    if not isinstance(segment, dict):
        segment = {
            "enabled": False,
            "start": 0,
            "end": int(schedule["microbatch_iterations"]),
        }
    if not segment.get("enabled"):
        return segment

    start = int(segment["start"])
    end = int(segment["end"])
    total = int(schedule["microbatch_iterations"])
    accumulation = int(args.grad_accumulation_steps)
    if not 0 <= start < end <= total:
        raise ExposurePrefixCacheLoraError(
            "child segment must satisfy 0 <= start < end <= global iters"
        )
    if start % accumulation or end % accumulation:
        raise ExposurePrefixCacheLoraError(
            "child segment boundaries must align with optimizer steps"
        )
    report_cadence = int(args.steps_per_report)
    if report_cadence < 1:
        raise ExposurePrefixCacheLoraError(
            "global report cadence must be at least one"
        )
    if start % report_cadence:
        raise ExposurePrefixCacheLoraError(
            "child segment start must align with the global report cadence"
        )
    if end != total and end % report_cadence:
        raise ExposurePrefixCacheLoraError(
            "nonfinal child segment end must align with the global report cadence"
        )

    adapter_output = Path(args.adapter_file)
    optimizer_output = Path(segment["optimizer_state_output"])
    if adapter_output.resolve(strict=False) == optimizer_output.resolve(strict=False):
        raise ExposurePrefixCacheLoraError(
            "segment adapter and optimizer-state outputs must be different files"
        )
    _require_create_once_destination(adapter_output, "segment adapter output")
    _require_create_once_destination(
        optimizer_output,
        "segment optimizer-state output",
    )

    adapter_input = segment.get("adapter_input")
    if adapter_input is not None:
        adapter_weights = _load_hash_bound_safetensors(
            Path(adapter_input),
            str(segment["adapter_input_sha256"]),
            mx,
            "segment adapter input",
        )
        expected_weights = dict(tree_flatten(model.trainable_parameters()))
        _assert_tensor_mapping_matches(
            adapter_weights,
            expected_weights,
            "segment adapter input",
        )
        model.load_weights(list(adapter_weights.items()), strict=False)
        mx.eval(model.trainable_parameters())

    optimizer.init(model.trainable_parameters())
    optimizer_input = segment.get("optimizer_state_input")
    if optimizer_input is not None:
        optimizer_state = _load_hash_bound_safetensors(
            Path(optimizer_input),
            str(segment["optimizer_state_input_sha256"]),
            mx,
            "segment optimizer-state input",
        )
        expected_state = dict(tree_flatten(optimizer.state))
        _assert_tensor_mapping_matches(
            optimizer_state,
            expected_state,
            "segment optimizer-state input",
        )
        from mlx.utils import tree_unflatten

        optimizer.state = tree_unflatten(list(optimizer_state.items()))
        mx.eval(optimizer.state)
    optimizer_step = int(optimizer.step.item())
    expected_step = start // accumulation
    if optimizer_step != expected_step:
        raise ExposurePrefixCacheLoraError(
            "segment optimizer step does not match its global microbatch start: "
            f"actual={optimizer_step}, expected={expected_step}"
        )
    return segment


def _assert_tensor_mapping_matches(
    actual: dict[str, Any],
    expected: dict[str, Any],
    label: str,
) -> None:
    if set(actual) != set(expected):
        raise ExposurePrefixCacheLoraError(
            f"{label} tensor names do not match the current training state"
        )
    for name, expected_value in expected.items():
        actual_value = actual[name]
        if (
            actual_value.shape != expected_value.shape
            or actual_value.dtype != expected_value.dtype
        ):
            raise ExposurePrefixCacheLoraError(
                f"{label} tensor {name!r} has an incompatible shape or dtype"
            )


def _load_hash_bound_safetensors(
    path: Path,
    expected_sha256: str,
    mx: Any,
    label: str,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ExposurePrefixCacheLoraError(
            f"{label} must be a regular non-symlink file"
        )
    actual_sha256 = _file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ExposurePrefixCacheLoraError(
            f"{label} SHA-256 mismatch: actual={actual_sha256}, "
            f"expected={expected_sha256}"
        )
    try:
        payload = mx.load(str(path))
    except Exception as exc:
        raise ExposurePrefixCacheLoraError(
            f"{label} is not a readable safetensors artifact"
        ) from exc
    if not isinstance(payload, dict) or not payload:
        raise ExposurePrefixCacheLoraError(
            f"{label} must contain at least one tensor"
        )
    return payload


def _require_create_once_destination(path: Path, label: str) -> None:
    if os.path.lexists(path):
        raise ExposurePrefixCacheLoraError(f"{label} already exists: {path}")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise ExposurePrefixCacheLoraError(
            f"{label} parent must be an existing non-symlink directory: "
            f"{path.parent}"
        )


def _atomic_save_safetensors(
    path: Path,
    tensors: dict[str, Any],
    mx: Any,
    *,
    metadata: dict[str, str] | None = None,
) -> None:
    _require_create_once_destination(path, "create-once safetensors output")
    temporary_path: Path | None = None
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp.safetensors",
            dir=path.parent,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(file_descriptor, "wb") as artifact:
            mx.save_safetensors(
                artifact,
                tensors,
                metadata=metadata,
            )
            artifact.flush()
            os.fsync(artifact.fileno())
        os.chmod(temporary_path, 0o600)
        os.link(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except FileExistsError as exc:
        raise ExposurePrefixCacheLoraError(
            f"create-once safetensors output already exists: {path}"
        ) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for block in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_schedule() -> dict[str, Any]:
    if _RUNTIME_SCHEDULE is None:
        raise ExposurePrefixCacheLoraError(
            "exposure-prefix schedule is not installed"
        )
    return _RUNTIME_SCHEDULE


def _parse_args(
    argv: list[str],
) -> tuple[argparse.Namespace, list[str]]:
    exposure_args, without_exposure = _parse_exposure_args(argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--prefix-equivalence", type=Path, required=True)
    parser.add_argument("--hfr-child-segment-start", type=int)
    parser.add_argument("--hfr-child-segment-end", type=int)
    parser.add_argument("--hfr-child-segment-adapter-input", type=Path)
    parser.add_argument("--hfr-child-segment-adapter-sha256")
    parser.add_argument("--hfr-child-segment-optimizer-state-input", type=Path)
    parser.add_argument("--hfr-child-segment-optimizer-state-sha256")
    parser.add_argument("--hfr-child-segment-optimizer-state-output", type=Path)
    equivalence_args, _ = parser.parse_known_args(without_exposure)
    custom_options = {
        "--prefix-equivalence",
        "--hfr-child-segment-start",
        "--hfr-child-segment-end",
        "--hfr-child-segment-adapter-input",
        "--hfr-child-segment-adapter-sha256",
        "--hfr-child-segment-optimizer-state-input",
        "--hfr-child-segment-optimizer-state-sha256",
        "--hfr-child-segment-optimizer-state-output",
    }
    passthrough = _strip_value_options(without_exposure, custom_options)
    exposure_args.prefix_equivalence = equivalence_args.prefix_equivalence
    for field in (
        "hfr_child_segment_start",
        "hfr_child_segment_end",
        "hfr_child_segment_adapter_input",
        "hfr_child_segment_adapter_sha256",
        "hfr_child_segment_optimizer_state_input",
        "hfr_child_segment_optimizer_state_sha256",
        "hfr_child_segment_optimizer_state_output",
    ):
        setattr(exposure_args, field, getattr(equivalence_args, field))
    exposure_args.hfr_child_segment_enabled = any(
        token in custom_options - {"--prefix-equivalence"}
        or any(
            token.startswith(f"{option}=")
            for option in custom_options - {"--prefix-equivalence"}
        )
        for token in without_exposure
    )
    return exposure_args, passthrough


def _strip_value_options(argv: list[str], options: set[str]) -> list[str]:
    passthrough: list[str] = []
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token in options:
            skip_next = True
            continue
        if any(token.startswith(f"{option}=") for option in options):
            continue
        passthrough.append(token)
    return passthrough


def _build_segment_config(
    args: argparse.Namespace,
    *,
    total_microbatches: int,
) -> dict[str, Any]:
    if not args.hfr_child_segment_enabled:
        return {
            "enabled": False,
            "start": 0,
            "end": total_microbatches,
        }

    start = (
        0
        if args.hfr_child_segment_start is None
        else int(args.hfr_child_segment_start)
    )
    end = (
        total_microbatches
        if args.hfr_child_segment_end is None
        else int(args.hfr_child_segment_end)
    )
    if not 0 <= start < end <= total_microbatches:
        raise ExposurePrefixCacheLoraError(
            "child segment must satisfy 0 <= start < end <= global iters"
        )
    adapter_input = args.hfr_child_segment_adapter_input
    adapter_sha256 = args.hfr_child_segment_adapter_sha256
    optimizer_input = args.hfr_child_segment_optimizer_state_input
    optimizer_sha256 = args.hfr_child_segment_optimizer_state_sha256
    optimizer_output = args.hfr_child_segment_optimizer_state_output
    _require_path_sha_pair(
        adapter_input,
        adapter_sha256,
        "child segment adapter input",
    )
    _require_path_sha_pair(
        optimizer_input,
        optimizer_sha256,
        "child segment optimizer-state input",
    )
    if optimizer_output is None:
        raise ExposurePrefixCacheLoraError(
            "segmented mode requires --hfr-child-segment-optimizer-state-output"
        )
    if start > 0 and (adapter_input is None or optimizer_input is None):
        raise ExposurePrefixCacheLoraError(
            "a resumed child segment requires both adapter and optimizer-state inputs"
        )
    return {
        "enabled": True,
        "start": start,
        "end": end,
        "adapter_input": adapter_input,
        "adapter_input_sha256": adapter_sha256,
        "optimizer_state_input": optimizer_input,
        "optimizer_state_input_sha256": optimizer_sha256,
        "optimizer_state_output": optimizer_output,
    }


def _require_path_sha_pair(
    path: Path | None,
    sha256: str | None,
    label: str,
) -> None:
    if (path is None) != (sha256 is None):
        raise ExposurePrefixCacheLoraError(
            f"{label} and its SHA-256 must be supplied together"
        )
    if sha256 is not None and (
        len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise ExposurePrefixCacheLoraError(
            f"{label} SHA-256 must be 64 lowercase hexadecimal characters"
        )


def main(argv: list[str] | None = None) -> None:
    """Install governed evidence and run MLX-LM's normal LoRA setup."""

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    raw_argv = sys.argv[1:] if argv is None else argv
    args, passthrough = _parse_args(raw_argv)
    equivalence = validate_tau3_prefix_equivalence(args.prefix_equivalence)
    if equivalence.get("passed") is not True:
        raise ExposurePrefixCacheLoraError(
            "prefix-equivalence artifact failed validation: "
            + str(equivalence.get("errors"))
        )
    schedule = load_exposure_schedule(
        dataset_jsonl=args.exposure_dataset,
        receipt_path=args.exposure_receipt,
        ledger_path=args.exposure_ledger,
        batch_size=args.batch_size,
        grad_accumulation_steps=args.grad_accumulation_steps,
        iters=args.iters,
    )
    schedule["segment"] = _build_segment_config(
        args,
        total_microbatches=int(schedule["microbatch_iterations"]),
    )
    global _RUNTIME_SCHEDULE
    _RUNTIME_SCHEDULE = schedule

    import mlx_lm.lora as lora

    lora.train = train_with_exposure_prefix_cache
    sys.argv = [sys.argv[0], *passthrough]
    lora.main()


if __name__ == "__main__":
    try:
        main()
    except (
        ExposureLoraError,
        ExposurePrefixCacheLoraError,
        PrefixCacheTrainingError,
        Tau3ExposureError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
