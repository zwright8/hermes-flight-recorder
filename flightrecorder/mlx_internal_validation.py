"""Run coverage-complete target-only loss evaluation for one MLX adapter."""

from __future__ import annotations

import argparse
import json
import math
import os
import stat
import sys
import time
from pathlib import Path
from typing import Any

from .mlx_prefix_cache_lora import (
    _materialize_prompt_cache,
    split_supervised_tokens,
)
from .path_safety import path_has_symlink_component
from .tau3_candidate_identity import (
    Tau3CandidateIdentityError,
    _load_receipt,
    _verified_adapter,
)
from .tau3_internal_validation import (
    INTERNAL_VALIDATION_METHOD,
    Tau3InternalValidationError,
    _canonical_sha256,
    _dict,
    _expected_run_binding,
    _load_json_object,
    _read_jsonl_objects,
    _sha256_file,
    build_tau3_internal_validation,
    validate_tau3_internal_validation,
)
from .tau3_model_identity import validate_tau3_model_identity

ARTIFACT_FILENAME = "internal-validation.json"
MEASUREMENTS_FILENAME = "measurements.jsonl"
RUN_BINDING_FILENAME = "run-binding.json"


class MlxInternalValidationError(ValueError):
    """Raised when local adapter loss evaluation cannot run safely."""


def run_mlx_internal_validation(
    *,
    dataset_path: str | Path,
    training_receipt_path: str | Path,
    protocol_path: str | Path,
    model_identity_path: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    max_seq_length: int,
    progress_every: int = 10,
) -> dict[str, Any]:
    """Evaluate every validation row and emit independently replayable evidence."""

    if max_seq_length < 1:
        raise MlxInternalValidationError(
            "max_seq_length must be positive"
        )
    if progress_every < 1:
        raise MlxInternalValidationError(
            "progress_every must be positive"
        )
    dataset_file = Path(dataset_path)
    dataset_manifest_file = dataset_file.parent / "manifest.json"
    receipt_file = Path(training_receipt_path)
    protocol_file = Path(protocol_path)
    identity_file = Path(model_identity_path)
    model_dir = Path(model_path)
    out = Path(output_dir)
    for label, path in (
        ("dataset", dataset_file),
        ("dataset manifest", dataset_manifest_file),
        ("training receipt", receipt_file),
        ("protocol", protocol_file),
        ("model identity", identity_file),
        ("model", model_dir),
        ("output", out),
    ):
        if path_has_symlink_component(path, include_leaf=True):
            raise MlxInternalValidationError(
                f"{label} path must not contain symlink components: {path}"
            )

    artifact_file = out / ARTIFACT_FILENAME
    measurements_file = out / MEASUREMENTS_FILENAME
    run_binding_file = out / RUN_BINDING_FILENAME
    if artifact_file.exists():
        result = validate_tau3_internal_validation(
            artifact_file,
            dataset_path=dataset_file,
            training_receipt_path=receipt_file,
            protocol_path=protocol_file,
            model_identity_path=identity_file,
        )
        if result.get("passed") is not True:
            raise MlxInternalValidationError(
                "existing internal-validation artifact does not replay: "
                + "; ".join(result.get("errors") or [])
            )
        return _load_json_object(
            artifact_file,
            "internal-validation artifact",
        )

    receipt = _load_receipt(receipt_file)
    adapter = _verified_adapter(receipt, receipt_path=receipt_file)
    adapter_dir = adapter["_adapter_dir"]
    _reject_output_inside_immutable_tree(out, adapter_dir, "adapter")
    _reject_output_inside_immutable_tree(out, model_dir, "base model")
    out.mkdir(parents=True, exist_ok=True)
    unexpected = sorted(
        path.name
        for path in out.iterdir()
        if path.name not in {
            MEASUREMENTS_FILENAME,
            RUN_BINDING_FILENAME,
        }
    )
    if unexpected:
        raise MlxInternalValidationError(
            "output directory contains unexpected files: "
            + ", ".join(unexpected)
        )
    protocol = _load_json_object(protocol_file, "protocol")
    identity = _load_json_object(identity_file, "model identity")
    if protocol.get("schema_version") != "hfr.tau3_protocol_config.v1":
        raise MlxInternalValidationError(
            "protocol must be hfr.tau3_protocol_config.v1"
        )
    if _nested(receipt, "training_binding", "protocol", "sha256") != (
        _sha256_file(protocol_file)
    ):
        raise MlxInternalValidationError(
            "training receipt does not bind the protocol"
        )
    if _nested(receipt, "training_binding", "model", "identity_sha256") != (
        _sha256_file(identity_file)
    ):
        raise MlxInternalValidationError(
            "training receipt does not bind the model identity"
        )
    if _nested(
        receipt,
        "training_binding",
        "dataset",
        "manifest_sha256",
    ) != _sha256_file(dataset_manifest_file):
        raise MlxInternalValidationError(
            "training receipt does not bind the dataset manifest"
        )
    recipe_max_seq_length = _nested(
        receipt,
        "training_binding",
        "recipe",
        "max_seq_length",
    )
    if recipe_max_seq_length != max_seq_length:
        raise MlxInternalValidationError(
            "max_seq_length must exactly match the training recipe"
        )
    identity_errors = validate_tau3_model_identity(
        identity,
        model_dir,
        expected_model_id=str(identity.get("model_id") or ""),
        expected_revision=str(identity.get("revision") or ""),
    )
    if identity_errors:
        raise MlxInternalValidationError(
            "model identity does not replay: " + "; ".join(identity_errors)
        )

    run_binding = _expected_run_binding(
        dataset_file=dataset_file,
        dataset_manifest_file=dataset_manifest_file,
        receipt_file=receipt_file,
        adapter_tree_sha256=adapter["receipt_tree_sha256"],
        protocol_file=protocol_file,
        identity_file=identity_file,
        max_seq_length=max_seq_length,
    )
    _create_or_verify_run_binding(run_binding_file, run_binding)
    dataset_rows = _read_jsonl_objects(
        dataset_file,
        "internal-validation dataset",
    )
    completed = _load_resume_measurements(
        measurements_file,
        dataset_rows,
        max_seq_length=max_seq_length,
    )
    if len(completed) < len(dataset_rows):
        _require_writable_measurements(measurements_file)

    receipt_sha256_before = _sha256_file(receipt_file)
    started = time.perf_counter()
    if len(completed) < len(dataset_rows):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from mlx_lm import load

        model, tokenizer = load(
            str(model_dir.resolve(strict=True)),
            adapter_path=str(adapter_dir.resolve(strict=True)),
            lazy=False,
        )
        del tokenizer
        for index in range(len(completed), len(dataset_rows)):
            measurement = _evaluate_row(
                model,
                dataset_rows[index],
                row_index=index,
                max_seq_length=max_seq_length,
            )
            _append_jsonl_durable(measurements_file, measurement)
            completed.append(measurement)
            if (
                len(completed) % progress_every == 0
                or len(completed) == len(dataset_rows)
            ):
                elapsed = time.perf_counter() - started
                print(
                    "Internal validation: "
                    f"{len(completed)}/{len(dataset_rows)} rows, "
                    f"elapsed {elapsed:.1f}s",
                    flush=True,
                )
        del model

    if _sha256_file(receipt_file) != receipt_sha256_before:
        raise MlxInternalValidationError(
            "training receipt changed during internal validation"
        )
    adapter_after = _verified_adapter(receipt, receipt_path=receipt_file)
    if adapter_after["receipt_tree_sha256"] != (
        adapter["receipt_tree_sha256"]
    ):
        raise MlxInternalValidationError(
            "adapter tree changed during internal validation"
        )
    measurements_file.chmod(0o400)
    artifact = build_tau3_internal_validation(
        dataset_path=dataset_file,
        measurements_path=measurements_file,
        run_binding_path=run_binding_file,
        training_receipt_path=receipt_file,
        protocol_path=protocol_file,
        model_identity_path=identity_file,
        output_path=artifact_file,
        max_seq_length=max_seq_length,
    )
    validation = validate_tau3_internal_validation(
        artifact_file,
        dataset_path=dataset_file,
        training_receipt_path=receipt_file,
        protocol_path=protocol_file,
        model_identity_path=identity_file,
    )
    if validation.get("passed") is not True:
        raise MlxInternalValidationError(
            "emitted internal-validation artifact does not replay: "
            + "; ".join(validation.get("errors") or [])
        )
    return artifact


def _evaluate_row(
    model: Any,
    row: dict[str, Any],
    *,
    row_index: int,
    max_seq_length: int,
) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn

    metadata = _dict(row.get("metadata"))
    token_counts = _dict(metadata.get("token_counts"))
    tokens = token_counts.get("input_token_ids")
    prompt_tokens = token_counts.get("prompt_tokens")
    if not isinstance(tokens, list) or not isinstance(prompt_tokens, int):
        raise MlxInternalValidationError(
            f"row {row_index} lacks governed token metadata"
        )
    prefix, suffix_inputs, targets = split_supervised_tokens(
        tokens,
        prompt_tokens,
        max_seq_length,
    )
    cache = _materialize_prompt_cache(model, prefix)
    model.eval()
    logits = model(mx.array([suffix_inputs]), cache=cache)
    target_array = mx.array([targets])
    losses = nn.losses.cross_entropy(
        logits,
        target_array,
    ).astype(mx.float32)
    mean_loss_value = losses.mean()
    mx.eval(mean_loss_value)
    mean_loss = float(mean_loss_value.item())
    loss_sum = mean_loss * len(targets)
    finite = math.isfinite(loss_sum) and math.isfinite(mean_loss)
    measurement = {
        "row_index": row_index,
        "row_sha256": _canonical_sha256(row),
        "domain": metadata.get("domain"),
        "behavior": metadata.get("behavior"),
        "prompt_tokens": prompt_tokens,
        "supervised_tokens": len(targets),
        "input_token_ids_sha256": _canonical_sha256(tokens),
        "target_tokens_sha256": _canonical_sha256(targets),
        "mean_loss": mean_loss,
        "loss_sum": loss_sum,
        "finite": finite,
    }
    del logits, losses, mean_loss_value, cache
    mx.clear_cache()
    if not finite:
        raise MlxInternalValidationError(
            f"row {row_index} produced a non-finite loss"
        )
    return measurement


def _load_resume_measurements(
    path: Path,
    dataset_rows: list[dict[str, Any]],
    *,
    max_seq_length: int,
) -> list[dict[str, Any]]:
    if not path.exists():
        with path.open("x", encoding="utf-8"):
            pass
        path.chmod(0o600)
        return []
    rows = _read_jsonl_allow_empty(path)
    if len(rows) > len(dataset_rows):
        raise MlxInternalValidationError(
            "measurements contain more rows than the validation dataset"
        )
    for index, measurement in enumerate(rows):
        errors = _resume_measurement_errors(
            dataset_rows[index],
            measurement,
            row_index=index,
            max_seq_length=max_seq_length,
        )
        if errors:
            raise MlxInternalValidationError(
                f"measurement row {index} cannot resume: "
                + "; ".join(errors)
            )
    return rows


def _resume_measurement_errors(
    dataset_row: dict[str, Any],
    measurement: dict[str, Any],
    *,
    row_index: int,
    max_seq_length: int,
) -> list[str]:
    errors: list[str] = []
    metadata = _dict(dataset_row.get("metadata"))
    token_counts = _dict(metadata.get("token_counts"))
    tokens = token_counts.get("input_token_ids")
    prompt_tokens = token_counts.get("prompt_tokens")
    supervised_tokens = token_counts.get("supervised_tokens")
    if not isinstance(tokens, list) or len(tokens) > max_seq_length:
        return ["dataset token ids are invalid or exceed max_seq_length"]
    if not isinstance(prompt_tokens, int) or not isinstance(
        supervised_tokens,
        int,
    ):
        return ["dataset prompt/target accounting is invalid"]
    target_tokens = tokens[prompt_tokens:]
    expected = {
        "row_index": row_index,
        "row_sha256": _canonical_sha256(dataset_row),
        "domain": metadata.get("domain"),
        "behavior": metadata.get("behavior"),
        "prompt_tokens": prompt_tokens,
        "supervised_tokens": supervised_tokens,
        "input_token_ids_sha256": _canonical_sha256(tokens),
        "target_tokens_sha256": _canonical_sha256(target_tokens),
        "finite": True,
    }
    for field, value in expected.items():
        if measurement.get(field) != value:
            errors.append(f"{field} mismatch")
    mean_loss = measurement.get("mean_loss")
    loss_sum = measurement.get("loss_sum")
    if (
        not _finite_nonnegative(mean_loss)
        or not _finite_nonnegative(loss_sum)
    ):
        errors.append("loss is not finite and nonnegative")
    elif not math.isclose(
        float(loss_sum),
        float(mean_loss) * supervised_tokens,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        errors.append("loss_sum does not replay")
    return errors


def _create_or_verify_run_binding(
    path: Path,
    expected: dict[str, Any],
) -> None:
    if path.exists():
        actual = _load_json_object(
            path,
            "internal-validation run binding",
        )
        if actual != expected:
            raise MlxInternalValidationError(
                "existing run binding does not match current sources"
            )
        return
    with path.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(expected, indent=2, sort_keys=True) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o400)


def _reject_output_inside_immutable_tree(
    output: Path,
    immutable_root: Path,
    label: str,
) -> None:
    resolved_output = output.resolve(strict=False)
    resolved_root = immutable_root.resolve(strict=True)
    if resolved_output == resolved_root or resolved_output.is_relative_to(
        resolved_root
    ):
        raise MlxInternalValidationError(
            f"output directory must not be inside the immutable {label} tree"
        )


def _require_writable_measurements(path: Path) -> None:
    if not path.is_file():
        raise MlxInternalValidationError(
            "measurements path must be a regular file"
        )
    mode = stat.S_IMODE(path.stat().st_mode)
    if not mode & stat.S_IWUSR:
        raise MlxInternalValidationError(
            "incomplete measurements file is read-only"
        )


def _append_jsonl_durable(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl_allow_empty(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MlxInternalValidationError(
                f"measurements line {line_number} is invalid JSON: {exc.msg}"
            ) from exc
        if not isinstance(row, dict):
            raise MlxInternalValidationError(
                f"measurements line {line_number} must be an object"
            )
        rows.append(row)
    return rows


def _finite_nonnegative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _nested(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model-identity", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, required=True)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        artifact = run_mlx_internal_validation(
            dataset_path=args.dataset,
            training_receipt_path=args.training_receipt,
            protocol_path=args.protocol,
            model_identity_path=args.model_identity,
            model_path=args.model,
            output_dir=args.out,
            max_seq_length=args.max_seq_length,
            progress_every=args.progress_every,
        )
    except (
        MlxInternalValidationError,
        Tau3CandidateIdentityError,
        Tau3InternalValidationError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "artifact": str(args.out / ARTIFACT_FILENAME),
                "passed": artifact["passed"],
                "row_count": artifact["coverage"]["row_count"],
                "weighted_mean_loss": artifact["aggregate"][
                    "weighted_mean_loss"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
