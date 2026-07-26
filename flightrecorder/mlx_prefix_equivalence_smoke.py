"""Instrumented MLX LoRA smoke runner for Tau-3 prefix equivalence.

This entry point is deliberately not candidate-eligible. It exists only to
collect bounded, same-sample raw measurements for the registered
``tau3_prefix_equivalence`` artifact. Qualified candidate launches use
``mlx_exposure_prefix_cache_lora`` and require that artifact in advance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Sequence

from .mlx_exposure_lora import (
    ExposureLoraError,
    _assert_dataset_matches_receipt,
    _canonical_sha256,
    _dataset_item,
    _parse_exposure_args,
    _supported_raw_rows,
    load_exposure_schedule,
)
from .mlx_prefix_cache_lora import (
    PrefixCacheTrainingError,
    _materialize_prompt_cache,
    _target_value_and_grad,
    split_supervised_tokens,
)
from .tau3_exposure import Tau3ExposureError
from .tau3_prefix_equivalence import (
    REQUIRED_DOMAINS,
    REQUIRED_EQUIVALENCE_PROBE_FAMILIES,
)

MEASUREMENT_SCHEMA_VERSION = "hfr.tau3_prefix_equivalence_run.v1"


class PrefixEquivalenceSmokeError(ValueError):
    """Raised when a bounded equivalence smoke cannot run safely."""


_RUNTIME: dict[str, Any] | None = None


def target_accounting_row(
    *,
    tokens: Sequence[int],
    prompt_offset: int,
    row_sha256: str,
) -> dict[str, Any]:
    """Return replayable target-boundary evidence for one rendered row."""

    token_list = [int(token) for token in tokens]
    if prompt_offset < 1 or prompt_offset >= len(token_list):
        raise PrefixEquivalenceSmokeError(
            "prompt_offset must precede at least one supervised token"
        )
    target_tokens = token_list[prompt_offset:]
    mask_shape = {
        "sequence_length": len(token_list),
        "supervised_positions": list(range(prompt_offset, len(token_list))),
    }
    return {
        "row_sha256": row_sha256,
        "prompt_offset": prompt_offset,
        "target_start": prompt_offset,
        "target_end": len(token_list),
        "supervised_token_count": len(target_tokens),
        "loss_mask_sha256": _canonical_json_sha256(mask_shape),
        "target_tokens_sha256": _canonical_json_sha256(target_tokens),
    }


def supervised_target_preserving_prompt_tail(
    tokens: Sequence[int],
    prompt_offset: int,
    prompt_tail_token_limit: int,
) -> tuple[list[int], int]:
    """Bound the proxy prefix without truncating any supervised target."""

    token_list = [int(token) for token in tokens]
    if prompt_tail_token_limit < 1:
        raise PrefixEquivalenceSmokeError(
            "prompt_tail_token_limit must be positive"
        )
    if prompt_offset < 1 or prompt_offset >= len(token_list):
        raise PrefixEquivalenceSmokeError(
            "prompt_offset must precede at least one supervised token"
        )
    start = max(0, prompt_offset - prompt_tail_token_limit)
    effective = token_list[start:]
    effective_prompt_offset = prompt_offset - start
    if effective[effective_prompt_offset:] != token_list[prompt_offset:]:
        raise PrefixEquivalenceSmokeError(
            "bounded proxy changed supervised target tokens"
        )
    return effective, effective_prompt_offset


def _effective_row(
    dataset: Any,
    row_index: int,
) -> tuple[list[int], int]:
    tokens, prompt_offset = _dataset_item(dataset, row_index)
    runtime = _require_runtime()
    return supervised_target_preserving_prompt_tail(
        tokens,
        int(prompt_offset),
        int(runtime["prompt_tail_token_limit"]),
    )


def _single_row_batch(
    tokens: Sequence[int],
    prompt_offset: int,
    max_seq_length: int,
) -> tuple[Any, Any]:
    """Build the exact MLX-LM batch shape for one bounded token row."""

    token_list = [int(token) for token in tokens]
    if len(token_list) > max_seq_length:
        raise PrefixEquivalenceSmokeError(
            "bounded equivalence row exceeds max_seq_length"
        )
    if prompt_offset < 1 or prompt_offset >= len(token_list):
        raise PrefixEquivalenceSmokeError(
            "bounded equivalence prompt offset is invalid"
        )
    import mlx.core as mx
    import numpy as np

    pad_to = 32
    width = 1 + pad_to * ((len(token_list) + pad_to - 1) // pad_to)
    width = min(width, max_seq_length)
    batch = np.zeros((1, width), dtype=np.int32)
    batch[0, : len(token_list)] = token_list
    return mx.array(batch), mx.array(
        [(prompt_offset, len(token_list) - 1)]
    )


def train_equivalence_smoke(
    model: Any,
    optimizer: Any,
    train_dataset: Any,
    val_dataset: Any = None,
    args: Any = None,
    loss: Any = None,
    iterate_batches: Any = None,
    training_callback: Any = None,
) -> None:
    """Collect one raw full-gradient or detached-prefix measurement run."""

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.tuner.trainer import default_loss
    from mlx.utils import tree_flatten, tree_map

    del val_dataset, loss, iterate_batches, training_callback
    runtime = _require_runtime()
    schedule = runtime["schedule"]
    if args is None:
        raise PrefixEquivalenceSmokeError("training arguments are required")
    if args.batch_size != 1:
        raise PrefixEquivalenceSmokeError(
            "prefix-equivalence smoke requires batch_size=1"
        )
    if args.grad_accumulation_steps < 1:
        raise PrefixEquivalenceSmokeError(
            "gradient accumulation must be at least one"
        )
    if args.grad_checkpoint and runtime["arm"] != "full_gradient":
        raise PrefixEquivalenceSmokeError(
            "detached-prefix smoke does not support gradient checkpointing"
        )
    if args.iters != schedule["microbatch_iterations"]:
        raise PrefixEquivalenceSmokeError(
            "iters do not match the exposure smoke schedule"
        )
    if args.grad_accumulation_steps != schedule["receipt"][
        "sampler_config"
    ]["gradient_accumulation_steps"]:
        raise PrefixEquivalenceSmokeError(
            "gradient accumulation does not match the exposure smoke receipt"
        )
    world = mx.distributed.init()
    if world.size() != 1:
        raise PrefixEquivalenceSmokeError(
            "prefix-equivalence smoke requires one local process"
        )
    _assert_dataset_matches_receipt(train_dataset, schedule["receipt"])
    if mx.metal.is_available():
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])

    row_order = _flatten_schedule(schedule)
    raw_rows = _supported_raw_rows(train_dataset)
    row_hashes = [_canonical_sha256(row) for row in raw_rows]
    target_rows_by_index: dict[int, dict[str, Any]] = {}
    gradient_squared_norms: dict[str, float] = {}
    losses: list[float] = []
    numerical_failures = 0
    accumulated_gradients: Any = None
    model.train()
    if args.grad_checkpoint:
        from mlx_lm.tuner.trainer import grad_checkpoint

        grad_checkpoint(model.layers[0])
    full_value_and_grad = nn.value_and_grad(model, default_loss)
    compiled_full_value_and_grad: Any = None
    if runtime["compiled_full_gradient"]:
        compile_state = [model.state, mx.random.state]

        @partial(
            mx.compile,
            inputs=compile_state,
            outputs=compile_state,
        )
        def compiled_full_value_and_grad(batch: Any) -> Any:
            return full_value_and_grad(model, *batch)

    print(
        "Starting prefix-equivalence smoke..., "
        f"arm: {runtime['arm']}, microbatches: {len(row_order)}",
        flush=True,
    )
    for iteration, row_index in enumerate(row_order, start=1):
        tokens, prompt_offset = _effective_row(train_dataset, row_index)
        if row_index not in target_rows_by_index:
            target_rows_by_index[row_index] = target_accounting_row(
                tokens=tokens,
                prompt_offset=int(prompt_offset),
                row_sha256=row_hashes[row_index],
            )
        if runtime["arm"] == "full_gradient":
            batch = _single_row_batch(
                tokens,
                int(prompt_offset),
                args.max_seq_length,
            )
            if compiled_full_value_and_grad is not None:
                (loss_value, supervised_count), gradients = (
                    compiled_full_value_and_grad(batch)
                )
            else:
                (loss_value, supervised_count), gradients = (
                    full_value_and_grad(model, *batch)
                )
            mx.eval(loss_value, supervised_count, gradients)
        else:
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
            del cache

        loss_float = float(loss_value.item())
        if not math.isfinite(loss_float):
            numerical_failures += 1
        losses.append(loss_float)
        for name, gradient in tree_flatten(gradients):
            norm_value = mx.sqrt(
                mx.sum(mx.square(gradient.astype(mx.float32)))
            )
            mx.eval(norm_value)
            norm = float(norm_value.item())
            if not math.isfinite(norm):
                numerical_failures += 1
                continue
            gradient_squared_norms[name] = (
                gradient_squared_norms.get(name, 0.0) + norm * norm
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
            mx.eval(accumulated_gradients)
        del gradients, loss_value, supervised_count
        mx.clear_cache()
        print(
            f"Iter {iteration}: {runtime['arm']} loss {loss_float:.6f}, "
            f"Peak mem {mx.get_peak_memory() / 1e9:.3f} GB",
            flush=True,
        )

    if accumulated_gradients is not None:
        raise PrefixEquivalenceSmokeError(
            "smoke schedule ended with a partial optimizer step"
        )
    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(args.adapter_file), adapter_weights)
    target_rows = [
        target_rows_by_index[index]
        for index in sorted(target_rows_by_index)
    ]
    measurement = {
        "schema_version": MEASUREMENT_SCHEMA_VERSION,
        "arm": runtime["arm"],
        "run_id": runtime["run_id"],
        "bindings": runtime["bindings"],
        "sample": runtime["sample"],
        "execution": {
            "grad_checkpoint": bool(args.grad_checkpoint),
            "compile_disabled": os.environ.get("MLX_DISABLE_COMPILE") == "1",
            "compiled_full_gradient": runtime["compiled_full_gradient"],
            "microbatch_iterations": args.iters,
            "gradient_accumulation_steps": args.grad_accumulation_steps,
            "gradient_evidence_kind": "per_microbatch_gradient_l2",
        },
        "target_rows": target_rows,
        "gradient_modules": [
            {
                "name": name,
                "l2_norm": math.sqrt(gradient_squared_norms[name]),
            }
            for name in sorted(gradient_squared_norms)
        ],
        "losses": losses,
        "peak_memory_bytes": int(mx.get_peak_memory()),
        "numerical_failure_count": numerical_failures,
    }
    path = runtime["measurement_out"]
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(measurement, indent=2, sort_keys=True) + "\n")
    path.chmod(0o444)
    print(f"Saved raw equivalence measurement to {path}.", flush=True)
    print(f"Saved final weights to {args.adapter_file}.", flush=True)


def make_standard_full_gradient_measurement_train(
    standard_iterate_batches: Any,
) -> Any:
    """Instrument MLX-LM's compiled step with scalar gradient reductions."""

    def instrumented_standard_train(
        model: Any,
        optimizer: Any,
        train_dataset: Any,
        val_dataset: Any = None,
        args: Any = None,
        loss: Any = None,
        iterate_batches: Any = None,
        training_callback: Any = None,
    ) -> Any:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm.tuner.trainer import (
            _clear_cache,
            average_gradients,
            default_loss,
            evaluate,
            grad_checkpoint,
        )
        from mlx.utils import tree_flatten
        from mlx.utils import tree_map

        runtime = _require_runtime()
        if args is None:
            raise PrefixEquivalenceSmokeError(
                "training arguments are required"
            )
        schedule = runtime["schedule"]
        _assert_dataset_matches_receipt(
            train_dataset,
            schedule["receipt"],
        )
        if args.batch_size != 1:
            raise PrefixEquivalenceSmokeError(
                "standard equivalence smoke requires batch_size=1"
            )
        if args.iters != schedule["microbatch_iterations"]:
            raise PrefixEquivalenceSmokeError(
                "iters do not match the exposure smoke schedule"
            )
        if args.grad_accumulation_steps != schedule["receipt"][
            "sampler_config"
        ]["gradient_accumulation_steps"]:
            raise PrefixEquivalenceSmokeError(
                "gradient accumulation does not match the exposure smoke receipt"
            )
        world = mx.distributed.init()
        if world.size() != 1:
            raise PrefixEquivalenceSmokeError(
                "prefix-equivalence smoke requires one local process"
            )
        if mx.metal.is_available():
            mx.set_wired_limit(
                mx.device_info()["max_recommended_working_set_size"]
            )
        if args.grad_checkpoint:
            grad_checkpoint(model.layers[0])
        if loss is None:
            loss = default_loss
        if iterate_batches is None:
            iterate_batches = standard_iterate_batches

        raw_rows = _supported_raw_rows(train_dataset)
        row_hashes = [_canonical_sha256(row) for row in raw_rows]
        target_rows_by_index: dict[int, dict[str, Any]] = {}
        for row_index in _flatten_schedule(schedule):
            if row_index in target_rows_by_index:
                continue
            tokens, prompt_offset = _effective_row(
                train_dataset,
                row_index,
            )
            target_rows_by_index[row_index] = target_accounting_row(
                tokens=tokens,
                prompt_offset=int(prompt_offset),
                row_sha256=row_hashes[row_index],
            )
        module_names = [
            name for name, _ in tree_flatten(model.trainable_parameters())
        ]
        gradient_squared_norms = {name: 0.0 for name in module_names}
        loss_value_and_grad = nn.value_and_grad(model, loss)
        state = [model.state, optimizer.state, mx.random.state]
        grad_accumulation_steps = args.grad_accumulation_steps

        @partial(mx.compile, inputs=state, outputs=state)
        def step(
            batch: Any,
            previous_gradients: Any,
            do_update: bool,
        ) -> Any:
            (loss_value, token_count), gradients = loss_value_and_grad(
                model,
                *batch,
            )
            if previous_gradients is not None:
                gradients = tree_map(
                    lambda current, previous: current + previous,
                    gradients,
                    previous_gradients,
                )
            gradient_norms: tuple[Any, ...] = ()
            if do_update:
                gradients = average_gradients(gradients)
                if grad_accumulation_steps > 1:
                    gradients = tree_map(
                        lambda value: value / grad_accumulation_steps,
                        gradients,
                    )
                gradient_items = tree_flatten(gradients)
                if [name for name, _ in gradient_items] != module_names:
                    raise PrefixEquivalenceSmokeError(
                        "compiled gradient modules differ from trainable modules"
                    )
                gradient_norms = tuple(
                    mx.sqrt(
                        mx.sum(mx.square(value.astype(mx.float32)))
                    )
                    for _, value in gradient_items
                )
                optimizer.update(model, gradients)
                gradients = None
            return loss_value, token_count, gradients, gradient_norms

        model.train()
        losses: list[float] = []
        numerical_failures = 0
        accumulated_gradients: Any = None
        report_loss: Any = 0
        report_tokens: Any = 0
        report_steps = 0
        trained_tokens = 0
        train_time = 0.0
        row_order = _flatten_schedule(schedule)
        for iteration, row_index in enumerate(row_order, start=1):
            tokens, prompt_offset = _effective_row(
                train_dataset,
                row_index,
            )
            batch = _single_row_batch(
                tokens,
                prompt_offset,
                args.max_seq_length,
            )
            tic = time.perf_counter()
            if val_dataset and (
                iteration == 1
                or iteration % args.steps_per_eval == 0
                or iteration == args.iters
            ):
                val_tic = time.perf_counter()
                val_loss = evaluate(
                    model=model,
                    dataset=val_dataset,
                    loss=loss,
                    batch_size=args.batch_size,
                    num_batches=args.val_batches,
                    max_seq_length=args.max_seq_length,
                    iterate_batches=iterate_batches,
                )
                model.train()
                val_time = time.perf_counter() - val_tic
                print(
                    f"Iter {iteration}: Val loss {val_loss:.3f}, "
                    f"Val took {val_time:.3f}s",
                    flush=True,
                )
                if training_callback is not None:
                    training_callback.on_val_loss_report(
                        {
                            "iteration": iteration - 1,
                            "val_loss": val_loss,
                            "val_time": val_time,
                        }
                    )
                tic = time.perf_counter()

            (
                loss_value,
                token_count,
                accumulated_gradients,
                update_gradient_norms,
            ) = step(
                batch,
                accumulated_gradients,
                iteration % grad_accumulation_steps == 0,
            )
            report_loss += loss_value
            report_tokens += token_count
            report_steps += 1
            mx.eval(
                state,
                report_loss,
                report_tokens,
                accumulated_gradients,
                update_gradient_norms,
            )
            loss_float = float(loss_value.item())
            losses.append(loss_float)
            if not math.isfinite(loss_float):
                numerical_failures += 1
            if update_gradient_norms:
                for name, norm_value in zip(
                    module_names,
                    update_gradient_norms,
                    strict=True,
                ):
                    norm = float(norm_value.item())
                    if not math.isfinite(norm):
                        numerical_failures += 1
                        continue
                    gradient_squared_norms[name] += norm * norm
            _clear_cache(args.clear_cache_threshold)
            train_time += time.perf_counter() - tic

            if (
                iteration % args.steps_per_report == 0
                or iteration == args.iters
            ):
                train_loss = mx.distributed.all_sum(
                    report_loss,
                    stream=mx.cpu,
                ).item()
                train_loss /= report_steps
                report_token_count = mx.distributed.all_sum(
                    report_tokens,
                    stream=mx.cpu,
                ).item()
                learning_rate = optimizer.learning_rate.item()
                iterations_per_second = (
                    args.steps_per_report / train_time
                )
                tokens_per_second = (
                    float(report_token_count) / train_time
                )
                trained_tokens += report_token_count
                peak_memory_gb = mx.get_peak_memory() / 1e9
                print(
                    f"Iter {iteration}: Train loss {train_loss:.3f}, "
                    f"Learning Rate {learning_rate:.3e}, "
                    f"It/sec {iterations_per_second:.3f}, "
                    f"Tokens/sec {tokens_per_second:.3f}, "
                    f"Trained Tokens {trained_tokens}, "
                    f"Peak mem {peak_memory_gb:.3f} GB",
                    flush=True,
                )
                if training_callback is not None:
                    training_callback.on_train_loss_report(
                        {
                            "iteration": iteration,
                            "train_loss": train_loss,
                            "learning_rate": learning_rate,
                            "iterations_per_second": iterations_per_second,
                            "tokens_per_second": tokens_per_second,
                            "trained_tokens": trained_tokens,
                            "peak_memory": peak_memory_gb,
                        }
                    )
                report_loss = 0
                report_tokens = 0
                report_steps = 0
                train_time = 0.0

            if (
                iteration % args.steps_per_save == 0
                and world.rank() == 0
            ):
                adapter_weights = dict(
                    tree_flatten(model.trainable_parameters())
                )
                mx.save_safetensors(
                    str(args.adapter_file),
                    adapter_weights,
                )
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

        if accumulated_gradients is not None:
            raise PrefixEquivalenceSmokeError(
                "smoke schedule ended with a partial optimizer step"
            )
        adapter_weights = dict(tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(str(args.adapter_file), adapter_weights)
        gradient_modules = [
            {
                "name": name,
                "l2_norm": math.sqrt(gradient_squared_norms[name]),
            }
            for name in module_names
        ]
        measurement = {
            "schema_version": MEASUREMENT_SCHEMA_VERSION,
            "arm": runtime["arm"],
            "run_id": runtime["run_id"],
            "bindings": runtime["bindings"],
            "sample": runtime["sample"],
            "execution": {
                "grad_checkpoint": bool(args.grad_checkpoint),
                "compile_disabled": False,
                "compiled_full_gradient": True,
                "gradient_evidence_kind": (
                    "pre_optimizer_accumulated_gradient_l2"
                ),
                "microbatch_iterations": args.iters,
                "gradient_accumulation_steps": args.grad_accumulation_steps,
            },
            "target_rows": [
                target_rows_by_index[index]
                for index in sorted(target_rows_by_index)
            ],
            "gradient_modules": gradient_modules,
            "losses": losses,
            "peak_memory_bytes": int(mx.get_peak_memory()),
            "numerical_failure_count": numerical_failures,
        }
        path = runtime["measurement_out"]
        with path.open("x", encoding="utf-8") as handle:
            handle.write(
                json.dumps(measurement, indent=2, sort_keys=True) + "\n"
            )
        path.chmod(0o444)
        print(
            f"Saved standard full-gradient measurement to {path}.",
            flush=True,
        )
        print(f"Saved final weights to {args.adapter_file}.", flush=True)

    return instrumented_standard_train


def _flatten_schedule(schedule: dict[str, Any]) -> list[int]:
    order: list[int] = []
    for step in schedule.get("steps", []):
        for microbatch in step:
            if not isinstance(microbatch, list) or len(microbatch) != 1:
                raise PrefixEquivalenceSmokeError(
                    "equivalence smoke requires one row per microbatch"
                )
            order.append(int(microbatch[0]))
    return order


def _parse_smoke_args(
    argv: list[str],
) -> tuple[argparse.Namespace, list[str]]:
    exposure_args, without_exposure = _parse_exposure_args(argv)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--equivalence-arm",
        choices=("full_gradient", "detached_prefix"),
        required=True,
    )
    parser.add_argument("--measurement-out", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model-identity", type=Path, required=True)
    parser.add_argument("--candidate-dataset", type=Path, required=True)
    parser.add_argument(
        "--prompt-tail-token-limit",
        type=int,
        required=True,
    )
    parser.add_argument("--recipe-binding", type=Path, required=False)
    parser.add_argument("--run-id", required=False, default="smoke-run")
    parser.add_argument(
        "--compiled-full-gradient",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--standard-full-gradient",
        action="store_true",
        default=False,
    )
    parser.add_argument("--sample-domains", required=True)
    parser.add_argument("--sample-probe-families", required=True)
    smoke_args, _ = parser.parse_known_args(without_exposure)
    for key, value in vars(exposure_args).items():
        setattr(smoke_args, key, value)
    strip = {
        "--equivalence-arm",
        "--measurement-out",
        "--protocol",
        "--model-identity",
        "--candidate-dataset",
        "--prompt-tail-token-limit",
        "--recipe-binding",
        "--run-id",
        "--sample-domains",
        "--sample-probe-families",
    }
    strip_flags = {
        "--compiled-full-gradient",
        "--standard-full-gradient",
    }
    passthrough: list[str] = []
    skip_next = False
    for token in without_exposure:
        if skip_next:
            skip_next = False
            continue
        if token in strip:
            skip_next = True
            continue
        if token in strip_flags:
            continue
        passthrough.append(token)
    return smoke_args, passthrough


def _require_runtime() -> dict[str, Any]:
    if _RUNTIME is None:
        raise PrefixEquivalenceSmokeError(
            "prefix-equivalence smoke runtime is not installed"
        )
    return _RUNTIME


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrefixEquivalenceSmokeError(
            f"{label} is unreadable: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise PrefixEquivalenceSmokeError(f"{label} must be a JSON object")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_jsonl_objects(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PrefixEquivalenceSmokeError(
            f"{label} is unreadable: {exc}"
        ) from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PrefixEquivalenceSmokeError(
                f"{label} line {line_number} is invalid JSON: {exc.msg}"
            ) from exc
        if not isinstance(row, dict):
            raise PrefixEquivalenceSmokeError(
                f"{label} line {line_number} must be an object"
            )
        rows.append(row)
    if not rows:
        raise PrefixEquivalenceSmokeError(f"{label} contains no rows")
    return rows


def main(argv: list[str] | None = None) -> None:
    """Install a bounded measurement runtime and enter MLX-LM LoRA setup."""

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    raw_argv = sys.argv[1:] if argv is None else argv
    args, passthrough = _parse_smoke_args(raw_argv)
    for path, label in (
        (args.exposure_dataset, "exposure dataset"),
        (args.exposure_receipt, "exposure receipt"),
        (args.exposure_ledger, "exposure ledger"),
        (args.protocol, "protocol"),
        (args.model_identity, "model identity"),
        (args.candidate_dataset, "candidate dataset"),
    ):
        if not path.is_file():
            raise PrefixEquivalenceSmokeError(f"{label} does not exist: {path}")
    if args.measurement_out.exists():
        raise PrefixEquivalenceSmokeError(
            f"measurement output already exists: {args.measurement_out}"
        )
    args.measurement_out.parent.mkdir(parents=True, exist_ok=True)
    domains = tuple(
        value.strip() for value in args.sample_domains.split(",") if value.strip()
    )
    families = tuple(
        value.strip()
        for value in args.sample_probe_families.split(",")
        if value.strip()
    )
    if set(domains) != set(REQUIRED_DOMAINS):
        raise PrefixEquivalenceSmokeError(
            "sample domains must include airline, retail, and telecom"
        )
    if set(families) != set(REQUIRED_EQUIVALENCE_PROBE_FAMILIES):
        raise PrefixEquivalenceSmokeError(
            "sample probe families must include every required family"
        )
    if args.prompt_tail_token_limit < 1:
        raise PrefixEquivalenceSmokeError(
            "prompt-tail-token-limit must be positive"
        )
    sample_rows = _read_jsonl_objects(
        args.exposure_dataset,
        "exposure dataset",
    )
    candidate_rows = _read_jsonl_objects(
        args.candidate_dataset,
        "candidate dataset",
    )
    sample_row_hashes = [_canonical_sha256(row) for row in sample_rows]
    candidate_row_hashes = {
        _canonical_sha256(row) for row in candidate_rows
    }
    missing_sample_rows = [
        row_sha256
        for row_sha256 in sample_row_hashes
        if row_sha256 not in candidate_row_hashes
    ]
    if missing_sample_rows:
        raise PrefixEquivalenceSmokeError(
            "equivalence sample contains rows outside the candidate dataset: "
            + ",".join(missing_sample_rows)
        )
    recipe = (
        _load_json_object(args.recipe_binding, "recipe binding")
        if args.recipe_binding is not None
        else {
            "rank": 16,
            "scale": 20.0,
            "learning_rate": 1e-5,
            "num_layers": 16,
            "max_seq_length": 8192,
            "batch_size": args.batch_size,
            "grad_accumulation": args.grad_accumulation_steps,
            "mask_prompt": True,
            "seed": 101,
        }
    )
    schedule = load_exposure_schedule(
        dataset_jsonl=args.exposure_dataset,
        receipt_path=args.exposure_receipt,
        ledger_path=args.exposure_ledger,
        batch_size=args.batch_size,
        grad_accumulation_steps=args.grad_accumulation_steps,
        iters=args.iters,
        bounded_smoke=True,
    )
    global _RUNTIME
    _RUNTIME = {
        "arm": args.equivalence_arm,
        "run_id": args.run_id,
        "measurement_out": args.measurement_out,
        "compiled_full_gradient": bool(args.compiled_full_gradient),
        "standard_full_gradient": bool(args.standard_full_gradient),
        "prompt_tail_token_limit": args.prompt_tail_token_limit,
        "schedule": schedule,
        "bindings": {
            "dataset_file_sha256": _sha256_file(args.candidate_dataset),
            "protocol_file_sha256": _sha256_file(args.protocol),
            "model_identity_file_sha256": _sha256_file(args.model_identity),
            "recipe": recipe,
        },
        "sample": {
            "domains": list(domains),
            "probe_families": list(families),
            "stratified": True,
            "source_sample_file_sha256": _sha256_file(
                args.exposure_dataset
            ),
            "derivation": {
                "method": (
                    "supervised_target_preserving_prompt_tail_v1"
                ),
                "prompt_tail_token_limit": args.prompt_tail_token_limit,
                "proxy_only": True,
                "candidate_training_uses_full_prompt": True,
            },
        },
    }

    import mlx_lm.lora as lora

    if args.standard_full_gradient:
        if args.equivalence_arm != "full_gradient":
            raise PrefixEquivalenceSmokeError(
                "standard-full-gradient is valid only for the full-gradient arm"
            )
        from mlx_lm.tuner.trainer import (
            iterate_batches as upstream_iterate_batches,
        )

        lora.train = make_standard_full_gradient_measurement_train(
            upstream_iterate_batches,
        )
    else:
        lora.train = train_equivalence_smoke
    sys.argv = [sys.argv[0], *passthrough]
    lora.main()


if __name__ == "__main__":
    try:
        main()
    except (
        ExposureLoraError,
        PrefixCacheTrainingError,
        PrefixEquivalenceSmokeError,
        Tau3ExposureError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
