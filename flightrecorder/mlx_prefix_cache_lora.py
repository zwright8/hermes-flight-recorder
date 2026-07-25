"""Memory-bounded MLX-LM LoRA training over complete governed prompts.

Every dataset row still contains and executes its complete Tau system prompt,
ordered tool catalog, conversation, and final assistant target. The prompt is
materialized into a fresh inference cache for the current adapter weights, then
treated as a constant while gradients flow through the supervised assistant
suffix. This keeps target-only supervision and full-context conditioning while
avoiding an 8K-12K-token backward graph on memory-constrained Apple Silicon.

The detached prompt cache is an explicit approximation to full-sequence SFT.
Callers must record ``prefix_cache_training`` in their governed recipe.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any


class PrefixCacheTrainingError(ValueError):
    """Raised when a row cannot support safe target-suffix training."""


def split_supervised_tokens(
    tokens: Sequence[int],
    prompt_offset: int,
    max_seq_length: int,
) -> tuple[list[int], list[int], list[int]]:
    """Split one chat rendering without losing the first target prediction.

    The cache ends one token before ``prompt_offset``. The last prompt token is
    the first suffix input, so it predicts the first assistant target exactly
    as it does in ordinary next-token training.
    """

    token_list = [int(token) for token in tokens]
    if len(token_list) > max_seq_length:
        raise PrefixCacheTrainingError(
            f"rendered row has {len(token_list)} tokens, exceeding "
            f"max_seq_length={max_seq_length}; truncation is forbidden"
        )
    if prompt_offset < 2:
        raise PrefixCacheTrainingError(
            "prompt_offset must leave at least one cached prompt token"
        )
    if prompt_offset >= len(token_list):
        raise PrefixCacheTrainingError(
            "prompt_offset must precede at least one supervised target token"
        )
    prefix = token_list[: prompt_offset - 1]
    suffix_inputs = token_list[prompt_offset - 1 : -1]
    targets = token_list[prompt_offset:]
    if len(suffix_inputs) != len(targets):
        raise PrefixCacheTrainingError(
            "suffix inputs and supervised targets must have equal length"
        )
    return prefix, suffix_inputs, targets


def _materialize_prompt_cache(
    model: Any,
    prefix_tokens: Sequence[int],
) -> Any:
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    model.eval()
    prefix = mx.array([prefix_tokens])
    output = model(prefix, cache=cache)
    mx.eval(output, [entry.state for entry in cache])
    del output, prefix
    mx.clear_cache()
    return cache


def _target_value_and_grad(
    model: Any,
    suffix_inputs: Sequence[int],
    targets: Sequence[int],
    cache: Any,
) -> tuple[Any, Any, Any]:
    import mlx.core as mx
    import mlx.nn as nn

    inputs = mx.array([suffix_inputs])
    target_array = mx.array([targets])
    model.train()

    def loss_fn(active_model: Any) -> tuple[Any, Any]:
        logits = active_model(inputs, cache=cache)
        loss = nn.losses.cross_entropy(logits, target_array)
        return loss.astype(mx.float32).mean(), mx.array(target_array.size)

    (loss, target_count), gradients = nn.value_and_grad(model, loss_fn)(model)
    mx.eval(loss, target_count, gradients)
    return loss, target_count, gradients


def _evaluate(
    model: Any,
    dataset: Any,
    *,
    num_batches: int,
    max_seq_length: int,
) -> float:
    import mlx.core as mx
    import mlx.nn as nn
    if num_batches == -1:
        count = len(dataset)
    else:
        count = min(len(dataset), num_batches)
    if count < 1:
        raise PrefixCacheTrainingError("validation requires at least one row")

    losses: list[float] = []
    for index in _validation_indices(dataset, count):
        tokens, prompt_offset = dataset[int(index)]
        prefix, suffix_inputs, targets = split_supervised_tokens(
            tokens,
            int(prompt_offset),
            max_seq_length,
        )
        cache = _materialize_prompt_cache(model, prefix)
        model.eval()
        logits = model(mx.array([suffix_inputs]), cache=cache)
        target_array = mx.array([targets])
        loss = (
            nn.losses.cross_entropy(logits, target_array)
            .astype(mx.float32)
            .mean()
        )
        mx.eval(loss)
        losses.append(float(loss.item()))
        del logits, loss, cache
        mx.clear_cache()
    return sum(losses) / len(losses)


def _validation_indices(dataset: Any, count: int) -> list[int]:
    """Choose a stable, domain-balanced validation slice when metadata exists."""

    raw_dataset = getattr(dataset, "_data", None)
    raw_rows = getattr(raw_dataset, "_data", None)
    domains: dict[str, list[int]] = {}
    if isinstance(raw_rows, list) and len(raw_rows) == len(dataset):
        for index, row in enumerate(raw_rows):
            metadata = row.get("metadata") if isinstance(row, dict) else None
            domain = metadata.get("domain") if isinstance(metadata, dict) else None
            if isinstance(domain, str) and domain:
                domains.setdefault(domain, []).append(index)
    if len(domains) >= 2:
        selected: list[int] = []
        cursors = {domain: 0 for domain in domains}
        while len(selected) < count:
            progressed = False
            for domain in sorted(domains):
                cursor = cursors[domain]
                if cursor < len(domains[domain]):
                    selected.append(domains[domain][cursor])
                    cursors[domain] = cursor + 1
                    progressed = True
                    if len(selected) == count:
                        break
            if not progressed:
                break
        return selected
    if count == len(dataset):
        return list(range(len(dataset)))
    if count == 1:
        return [0]
    return [
        round(position * (len(dataset) - 1) / (count - 1))
        for position in range(count)
    ]


def train_with_prefix_cache(
    model: Any,
    optimizer: Any,
    train_dataset: Any,
    val_dataset: Any = None,
    args: Any = None,
    loss: Any = None,
    iterate_batches: Any = None,
    training_callback: Any = None,
) -> None:
    """MLX-LM-compatible trainer using a detached complete-prompt cache."""

    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten

    del loss, iterate_batches
    if args is None:
        raise PrefixCacheTrainingError("training arguments are required")
    if args.batch_size != 1:
        raise PrefixCacheTrainingError("prefix-cache training requires batch_size=1")
    if args.grad_accumulation_steps != 1:
        raise PrefixCacheTrainingError(
            "prefix-cache training requires grad_accumulation_steps=1"
        )
    if args.grad_checkpoint:
        raise PrefixCacheTrainingError(
            "prefix-cache training does not use gradient checkpointing"
        )
    world = mx.distributed.init()
    if world.size() != 1:
        raise PrefixCacheTrainingError(
            "prefix-cache training currently requires one local process"
        )
    if mx.metal.is_available():
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])

    print(f"Starting prefix-cache training..., iters: {args.iters}", flush=True)
    train_losses: list[float] = []
    target_tokens = 0
    prompt_tokens = 0
    trained_target_tokens = 0
    trained_prompt_tokens = 0
    report_started = time.perf_counter()
    order: list[int] = []
    order_cursor = 0

    for iteration in range(1, args.iters + 1):
        if not order or order_cursor >= len(order):
            order = [int(index) for index in np.random.permutation(len(train_dataset))]
            order_cursor = 0
        row_index = order[order_cursor]
        order_cursor += 1

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

        tokens, prompt_offset = train_dataset[row_index]
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
        optimizer.update(model, gradients)
        mx.eval(model.trainable_parameters(), optimizer.state)

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

        if iteration % args.steps_per_save == 0:
            adapter_weights = dict(tree_flatten(model.trainable_parameters()))
            mx.save_safetensors(str(args.adapter_file), adapter_weights)
            checkpoint = (
                Path(args.adapter_file).parent
                / f"{iteration:07d}_adapters.safetensors"
            )
            mx.save_safetensors(str(checkpoint), adapter_weights)
            print(
                f"Iter {iteration}: Saved adapter weights to "
                f"{args.adapter_file} and {checkpoint}.",
                flush=True,
            )

    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(args.adapter_file), adapter_weights)
    print(f"Saved final weights to {args.adapter_file}.", flush=True)


def main() -> None:
    """Patch MLX-LM's trainer entry point and run its normal LoRA setup."""

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    import mlx_lm.lora as lora

    lora.train = train_with_prefix_cache
    lora.main()


if __name__ == "__main__":
    main()
