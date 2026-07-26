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
import os
import sys
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
    if len(row_order) != args.iters:
        raise ExposurePrefixCacheLoraError(
            "flattened exposure order does not match training iters"
        )
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
    trained_target_tokens = 0
    trained_prompt_tokens = 0
    report_started = time.perf_counter()

    for iteration, row_index in enumerate(row_order, start=1):
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
        current_targets = int(supervised_count.item())
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
            _save_adapter(model, args.adapter_file, iteration, tree_flatten, mx)

    if accumulated_gradients is not None:
        raise ExposurePrefixCacheLoraError(
            "exposure schedule ended with a partial optimizer step"
        )
    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(args.adapter_file), adapter_weights)
    print(f"Saved final weights to {args.adapter_file}.", flush=True)


def _save_adapter(
    model: Any,
    adapter_file: str | Path,
    iteration: int,
    tree_flatten: Any,
    mx: Any,
) -> None:
    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(adapter_file), adapter_weights)
    checkpoint = (
        Path(adapter_file).parent
        / f"{iteration:07d}_adapters.safetensors"
    )
    mx.save_safetensors(str(checkpoint), adapter_weights)
    print(
        f"Iter {iteration}: Saved adapter weights to "
        f"{adapter_file} and {checkpoint}.",
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
    equivalence_args, _ = parser.parse_known_args(without_exposure)
    passthrough: list[str] = []
    skip_next = False
    for token in without_exposure:
        if skip_next:
            skip_next = False
            continue
        if token == "--prefix-equivalence":
            skip_next = True
            continue
        passthrough.append(token)
    exposure_args.prefix_equivalence = equivalence_args.prefix_equivalence
    return exposure_args, passthrough


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
