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
from pathlib import Path
from typing import Any, Sequence

from .mlx_exposure_lora import (
    ExposureLoraError,
    _assert_dataset_matches_receipt,
    _batch_from_indices,
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
    if args.grad_checkpoint:
        raise PrefixEquivalenceSmokeError(
            "equivalence arms must both disable gradient checkpointing"
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
    full_value_and_grad = nn.value_and_grad(model, default_loss)

    print(
        "Starting prefix-equivalence smoke..., "
        f"arm: {runtime['arm']}, microbatches: {len(row_order)}",
        flush=True,
    )
    for iteration, row_index in enumerate(row_order, start=1):
        tokens, prompt_offset = _dataset_item(train_dataset, row_index)
        if row_index not in target_rows_by_index:
            target_rows_by_index[row_index] = target_accounting_row(
                tokens=tokens,
                prompt_offset=int(prompt_offset),
                row_sha256=row_hashes[row_index],
            )
        if runtime["arm"] == "full_gradient":
            batch = _batch_from_indices(
                train_dataset,
                [row_index],
                batch_size=1,
                max_seq_length=args.max_seq_length,
            )
            (loss_value, supervised_count), gradients = full_value_and_grad(
                model,
                *batch,
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
    parser.add_argument("--recipe-binding", type=Path, required=False)
    parser.add_argument("--run-id", required=False, default="smoke-run")
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
        "--recipe-binding",
        "--run-id",
        "--sample-domains",
        "--sample-probe-families",
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
        "schedule": schedule,
        "bindings": {
            "dataset_file_sha256": _sha256_file(args.exposure_dataset),
            "protocol_file_sha256": _sha256_file(args.protocol),
            "model_identity_file_sha256": _sha256_file(args.model_identity),
            "recipe": recipe,
        },
        "sample": {
            "domains": list(domains),
            "probe_families": list(families),
            "stratified": True,
        },
    }

    import mlx_lm.lora as lora

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
