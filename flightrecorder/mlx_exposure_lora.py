"""Run MLX-LM LoRA training in deterministic exposure-ledger order.

This entry point leaves MLX-LM's model setup, optimizer, objective, masking,
checkpointing, and LoRA implementation in place. It only replaces the training
batch iterator with the row_index/microbatch sequence from a validated Tau-3
exposure ledger, so full-gradient SFT can be replayed exactly from the sealed
dataset receipt.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .tau3_exposure import Tau3ExposureError, validate_tau3_exposure_ledger


class ExposureLoraError(ValueError):
    """Raised when exposure-governed MLX-LM training cannot replay safely."""


def load_exposure_schedule(
    *,
    dataset_jsonl: str | Path,
    receipt_path: str | Path,
    ledger_path: str | Path,
    batch_size: int,
    grad_accumulation_steps: int,
    iters: int,
    bounded_smoke: bool = False,
) -> dict[str, Any]:
    """Validate and return the deterministic microbatch schedule."""

    validation = validate_tau3_exposure_ledger(
        dataset_jsonl,
        receipt_path,
        ledger_path,
    )
    receipt = _load_json(Path(receipt_path))
    ledger = _load_jsonl(Path(ledger_path))
    sampler = receipt.get("sampler_config")
    coverage = receipt.get("coverage")
    if not isinstance(sampler, dict):
        raise ExposureLoraError("exposure receipt sampler_config must be an object")
    if not isinstance(coverage, dict):
        raise ExposureLoraError("exposure receipt coverage must be an object")
    optimizer_steps = coverage.get("complete_optimizer_step_count")
    if not isinstance(optimizer_steps, int) or optimizer_steps < 1:
        raise ExposureLoraError("exposure receipt has invalid optimizer step count")
    ledger_microbatch_iterations = sum(
        len(step.get("microbatches", [])) for step in ledger
    )
    expected_microbatch_iterations = optimizer_steps * grad_accumulation_steps
    expected = {
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accumulation_steps,
        "microbatch_iterations": iters,
        "optimizer_steps": optimizer_steps,
    }
    actual = {
        "batch_size": sampler.get("batch_size"),
        "gradient_accumulation_steps": sampler.get(
            "gradient_accumulation_steps"
        ),
        "microbatch_iterations": ledger_microbatch_iterations,
        "optimizer_steps": optimizer_steps,
    }
    if actual != expected:
        raise ExposureLoraError(
            "training recipe does not match exposure ledger: "
            + json.dumps({"actual": actual, "expected": expected}, sort_keys=True)
        )
    if ledger_microbatch_iterations != expected_microbatch_iterations:
        raise ExposureLoraError(
            "exposure microbatch iterations do not equal optimizer_steps * gradient_accumulation_steps"
        )
    if coverage.get("complete_optimizer_steps") is not True:
        raise ExposureLoraError("exposure ledger contains a partial optimizer step")
    if coverage.get("all_rows_seen") is not True:
        raise ExposureLoraError("exposure ledger does not expose every row")
    if receipt.get("passed") is not True:
        if not bounded_smoke or not _bounded_smoke_exposure_is_safe(receipt):
            raise ExposureLoraError("exposure receipt is not candidate eligible")
    steps = []
    for step in ledger:
        microbatches = []
        for microbatch in step.get("microbatches", []):
            row_hashes = microbatch.get("row_hashes")
            if not isinstance(row_hashes, list) or len(row_hashes) != batch_size:
                raise ExposureLoraError("exposure microbatch size mismatch")
            row_indices = []
            for row_hash in row_hashes:
                matches = [
                    row
                    for row in step.get("rows", [])
                    if row.get("row_sha256") == row_hash
                ]
                matching_indices = {
                    int(row["row_index"])
                    for row in matches
                    if isinstance(row.get("row_index"), int)
                }
                if len(matching_indices) != 1:
                    raise ExposureLoraError(
                        "exposure row hash does not map to one dataset row"
                    )
                row_indices.append(next(iter(matching_indices)))
            microbatches.append(row_indices)
        if len(microbatches) != grad_accumulation_steps:
            raise ExposureLoraError("gradient accumulation schedule mismatch")
        steps.append(microbatches)
    return {
        "validation": validation,
        "receipt": receipt,
        "steps": steps,
        "microbatch_iterations": ledger_microbatch_iterations,
        "optimizer_steps": optimizer_steps,
    }


def _bounded_smoke_exposure_is_safe(receipt: dict[str, Any]) -> bool:
    """Allow a non-candidate sample only when behavior completeness alone fails."""

    eligibility = receipt.get("candidate_eligibility")
    if not isinstance(eligibility, dict):
        return False
    checks = eligibility.get("checks")
    if not isinstance(checks, list) or not checks:
        return False
    failed_ids = {
        check.get("id")
        for check in checks
        if isinstance(check, dict) and check.get("passed") is not True
    }
    required_passes = {
        "every_row_seen",
        "at_least_two_effective_epochs",
        "full_epoch_replay",
        "complete_optimizer_steps",
        "effective_batch_at_least_four",
        "competitive_dataset_row_schema",
        "required_domains_exact",
        "recovery_strata_nonzero",
        "stopping_strata_nonzero",
        "clarification_strata_nonzero",
        "telecom_strata_nonzero",
        "state_mutation_strata_nonzero",
        "action_class_strata_nonzero",
        "result_class_strata_nonzero",
    }
    passed_ids = {
        check.get("id")
        for check in checks
        if isinstance(check, dict) and check.get("passed") is True
    }
    return (
        failed_ids == {"required_behaviors_exact"}
        and required_passes.issubset(passed_ids)
    )


def exposure_iterate_batches(
    dataset: Any,
    batch_size: int,
    max_seq_length: int,
    loop: bool = False,
    seed: int | None = None,
    comm_group: Any = None,
) -> Iterator[tuple[Any, Any]]:
    """Yield MLX-LM-compatible batches in the validated exposure order."""

    del seed
    if comm_group is not None and comm_group.size() != 1:
        raise ExposureLoraError("exposure-ledger training requires one local worker")
    state = _require_runtime_state()
    if int(batch_size) != state["batch_size"]:
        raise ExposureLoraError("MLX-LM batch_size does not match exposure receipt")
    _assert_dataset_matches_receipt(dataset, state["receipt"])
    yielded = 0
    while True:
        for step in state["steps"]:
            for row_indices in step:
                yielded += 1
                yield _batch_from_indices(
                    dataset,
                    row_indices,
                    batch_size=batch_size,
                    max_seq_length=max_seq_length,
                )
        if not loop:
            break
        if yielded >= state["expected_microbatches"]:
            break


def install_exposure_schedule(schedule: dict[str, Any]) -> None:
    """Install process-local schedule state before patching MLX-LM."""

    global _RUNTIME_STATE
    sampler = schedule["receipt"]["sampler_config"]
    _RUNTIME_STATE = {
        "receipt": schedule["receipt"],
        "steps": schedule["steps"],
        "batch_size": int(sampler["batch_size"]),
        "expected_microbatches": int(schedule["microbatch_iterations"]),
    }


_RUNTIME_STATE: dict[str, Any] | None = None


def main(argv: list[str] | None = None) -> None:
    args, passthrough = _parse_exposure_args(sys.argv[1:] if argv is None else argv)
    schedule = load_exposure_schedule(
        dataset_jsonl=args.exposure_dataset,
        receipt_path=args.exposure_receipt,
        ledger_path=args.exposure_ledger,
        batch_size=args.batch_size,
        grad_accumulation_steps=args.grad_accumulation_steps,
        iters=args.iters,
    )
    install_exposure_schedule(schedule)

    import mlx_lm.lora as lora
    from mlx_lm.tuner.trainer import (
        iterate_batches as upstream_iterate_batches,
        train as upstream_train,
    )

    lora.train = make_exposure_train(upstream_train, upstream_iterate_batches)
    sys.argv = [sys.argv[0], *passthrough]
    lora.main()


def make_exposure_train(upstream_train: Any, standard_iterate_batches: Any) -> Any:
    """Return an MLX-LM train wrapper that governs only the train dataset."""

    def exposure_train(*train_args: Any, **kwargs: Any) -> Any:
        train_dataset = _extract_train_dataset(train_args, kwargs)
        state = _require_runtime_state()
        _assert_dataset_matches_receipt(train_dataset, state["receipt"])

        def governed_iterate_batches(dataset: Any, *args: Any, **kwargs: Any) -> Any:
            if dataset is train_dataset:
                return exposure_iterate_batches(dataset, *args, **kwargs)
            return standard_iterate_batches(dataset, *args, **kwargs)

        kwargs["iterate_batches"] = governed_iterate_batches
        return upstream_train(*train_args, **kwargs)

    return exposure_train


def _parse_exposure_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--exposure-dataset", type=Path, required=True)
    parser.add_argument("--exposure-receipt", type=Path, required=True)
    parser.add_argument("--exposure-ledger", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--grad-accumulation-steps", type=int, required=True)
    parser.add_argument("--iters", type=int, required=True)
    args, _ = parser.parse_known_args(argv)
    passthrough: list[str] = []
    skip_next = False
    strip = {
        "--exposure-dataset",
        "--exposure-receipt",
        "--exposure-ledger",
    }
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token in strip:
            skip_next = True
            continue
        passthrough.append(token)
    return args, passthrough


def _require_runtime_state() -> dict[str, Any]:
    if _RUNTIME_STATE is None:
        raise ExposureLoraError("exposure schedule is not installed")
    return _RUNTIME_STATE


def _extract_train_dataset(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if len(args) >= 3:
        return args[2]
    if "train_dataset" in kwargs:
        return kwargs["train_dataset"]
    raise ExposureLoraError("MLX-LM train_dataset argument is unavailable")


def _assert_dataset_matches_receipt(dataset: Any, receipt: dict[str, Any]) -> None:
    raw_rows = _supported_raw_rows(dataset)
    row_hashes = [_canonical_sha256(row) for row in raw_rows]
    content_sha256 = _canonical_sha256(
        {
            "row_count": len(row_hashes),
            "row_hashes_in_file_order": row_hashes,
        }
    )
    expected = receipt.get("dataset", {}).get("content_sha256")
    if content_sha256 != expected:
        raise ExposureLoraError("MLX-LM dataset order/content differs from exposure receipt")


def _supported_raw_rows(dataset: Any) -> list[dict[str, Any]]:
    try:
        from mlx_lm.tuner.datasets import (
            CacheDataset,
            ChatDataset,
            ConcatenatedDataset,
        )
    except Exception as exc:  # pragma: no cover - exercised only without MLX-LM.
        raise ExposureLoraError("MLX-LM dataset classes are unavailable") from exc
    if isinstance(dataset, ConcatenatedDataset):
        raise ExposureLoraError("ConcatenatedDataset is not supported for exposure-ledger training")
    if isinstance(dataset, CacheDataset):
        inner = getattr(dataset, "_data", None)
        if isinstance(inner, ConcatenatedDataset):
            raise ExposureLoraError("ConcatenatedDataset is not supported for exposure-ledger training")
        if isinstance(inner, ChatDataset):
            raw_rows = getattr(inner, "_data", None)
        else:
            raise ExposureLoraError("unsupported CacheDataset inner dataset for exposure-ledger training")
    elif isinstance(dataset, ChatDataset):
        raw_rows = getattr(dataset, "_data", None)
    else:
        raise ExposureLoraError("unsupported MLX-LM dataset wrapper for exposure-ledger training")
    if not isinstance(raw_rows, list) or not all(isinstance(row, dict) for row in raw_rows):
        raise ExposureLoraError("MLX-LM ChatDataset raw row order is unavailable")
    return raw_rows


def _batch_from_indices(
    dataset: Any,
    indices: list[int],
    *,
    batch_size: int,
    max_seq_length: int,
) -> tuple[Any, Any]:
    if len(indices) != batch_size:
        raise ExposureLoraError("exposure microbatch does not match MLX batch size")
    batch = [_dataset_item(dataset, index) for index in indices]
    if len(batch[0]) == 2:
        tokens, offsets = zip(*batch)
    else:
        tokens = batch
        offsets = [0] * len(tokens)
    true_lengths = [len(row) for row in tokens]
    if any(length > max_seq_length for length in true_lengths):
        raise ExposureLoraError("exposure row exceeds max_seq_length; truncation is forbidden")
    for prompt_offset, true_length in zip(offsets, true_lengths):
        if int(prompt_offset) < 0 or int(prompt_offset) >= true_length:
            raise ExposureLoraError("exposure row prompt offset is outside the sequence budget")
    import mlx.core as mx
    import numpy as np

    lengths = true_lengths
    pad_to = 32
    width = 1 + pad_to * ((max(lengths) + pad_to - 1) // pad_to)
    width = min(width, max_seq_length)
    batch_array = np.zeros((batch_size, width), dtype=np.int32)
    for row_index, row_tokens in enumerate(tokens):
        batch_array[row_index, : lengths[row_index]] = row_tokens[: lengths[row_index]]
    return mx.array(batch_array), mx.array(list(zip(offsets, lengths)))


def _dataset_item(dataset: Any, index: int) -> Any:
    try:
        from mlx_lm.tuner.datasets import ChatDataset
    except Exception:
        ChatDataset = ()  # type: ignore[assignment]
    if isinstance(dataset, ChatDataset):
        return dataset.process(dataset[index])
    return dataset[index]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExposureLoraError(f"invalid JSON in {path}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ExposureLoraError(f"JSON artifact must be an object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ExposureLoraError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ExposureLoraError(f"line {line_number}: ledger step must be an object")
        rows.append(payload)
    if not rows:
        raise ExposureLoraError("exposure ledger must contain at least one step")
    return rows


def _canonical_sha256(value: Any) -> str:
    import hashlib

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":
    try:
        main()
    except (ExposureLoraError, Tau3ExposureError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
