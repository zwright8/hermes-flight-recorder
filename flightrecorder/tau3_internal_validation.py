"""Coverage-complete Tau-3 internal-validation loss evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schema_registry import check_schema_contract
from .tau3_exposure import REQUIRED_BEHAVIORS, REQUIRED_DOMAINS

TAU3_INTERNAL_VALIDATION_SCHEMA_VERSION = (
    "hfr.tau3_internal_validation.v1"
)
TAU3_INTERNAL_VALIDATION_CHECK_SCHEMA_VERSION = (
    "hfr.tau3_internal_validation_check.v1"
)
INTERNAL_VALIDATION_METHOD = (
    "detached_complete_prompt_target_loss_v1"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class Tau3InternalValidationError(ValueError):
    """Raised when internal-validation evidence cannot be built safely."""


def build_tau3_internal_validation(
    *,
    dataset_path: str | Path,
    measurements_path: str | Path,
    run_binding_path: str | Path,
    training_receipt_path: str | Path,
    protocol_path: str | Path,
    model_identity_path: str | Path,
    output_path: str | Path,
    max_seq_length: int,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a hash-bound summary from complete row-level measurements."""

    dataset_file = Path(dataset_path)
    dataset_manifest_file = dataset_file.parent / "manifest.json"
    measurement_file = Path(measurements_path)
    run_binding_file = Path(run_binding_path)
    receipt_file = Path(training_receipt_path)
    protocol_file = Path(protocol_path)
    identity_file = Path(model_identity_path)
    out = Path(output_path)
    if out.exists():
        raise Tau3InternalValidationError(
            f"internal-validation artifact already exists: {out}"
        )
    if max_seq_length < 1:
        raise Tau3InternalValidationError(
            "max_seq_length must be positive"
        )

    dataset_rows = _read_jsonl_objects(
        dataset_file,
        "internal-validation dataset",
    )
    dataset_manifest = _load_json_object(
        dataset_manifest_file,
        "dataset manifest",
    )
    _require_internal_validation_dataset_binding(
        dataset_file,
        dataset_manifest,
    )
    measurements = _read_jsonl_objects(
        measurement_file,
        "internal-validation measurements",
    )
    run_binding = _load_json_object(
        run_binding_file,
        "internal-validation run binding",
    )
    receipt = _load_json_object(receipt_file, "training receipt")
    protocol = _load_json_object(protocol_file, "protocol")
    identity = _load_json_object(model_identity_path, "model identity")
    _require_successful_training_receipt(receipt)
    if _nested(
        receipt,
        "training_binding",
        "recipe",
        "max_seq_length",
    ) != max_seq_length:
        raise Tau3InternalValidationError(
            "max_seq_length does not match the training recipe"
        )
    if _nested(
        receipt,
        "training_binding",
        "dataset",
        "manifest_sha256",
    ) != _sha256_file(dataset_manifest_file):
        raise Tau3InternalValidationError(
            "dataset manifest does not match the training receipt"
        )

    adapter_tree_sha256 = _nested(
        receipt,
        "adapter",
        "tree_sha256",
    )
    protocol_sha256 = _sha256_file(protocol_file)
    identity_sha256 = _sha256_file(identity_file)
    if _nested(receipt, "training_binding", "protocol", "sha256") != (
        protocol_sha256
    ):
        raise Tau3InternalValidationError(
            "training receipt protocol binding does not match protocol file"
        )
    if _nested(receipt, "training_binding", "model", "identity_sha256") != (
        identity_sha256
    ):
        raise Tau3InternalValidationError(
            "training receipt model identity binding does not match identity file"
        )
    if protocol.get("schema_version") != "hfr.tau3_protocol_config.v1":
        raise Tau3InternalValidationError(
            "protocol must be hfr.tau3_protocol_config.v1"
        )
    if not isinstance(identity, dict) or not _is_sha256(adapter_tree_sha256):
        raise Tau3InternalValidationError(
            "training receipt lacks a valid adapter tree hash"
        )
    expected_run_binding = _expected_run_binding(
        dataset_file=dataset_file,
        dataset_manifest_file=dataset_manifest_file,
        receipt_file=receipt_file,
        adapter_tree_sha256=adapter_tree_sha256,
        protocol_file=protocol_file,
        identity_file=identity_file,
        max_seq_length=max_seq_length,
    )
    if run_binding != expected_run_binding:
        raise Tau3InternalValidationError(
            "run binding does not match the evaluated sources"
        )

    errors, replay = _replay_rows(
        dataset_rows,
        measurements,
        max_seq_length=max_seq_length,
    )
    if errors:
        raise Tau3InternalValidationError("; ".join(errors))

    rows_ref = {
        "path": _portable_relative_ref(measurement_file, out.parent),
        "sha256": _sha256_file(measurement_file),
        "size": measurement_file.stat().st_size,
    }
    run_binding_ref = {
        "path": _portable_relative_ref(run_binding_file, out.parent),
        "sha256": _sha256_file(run_binding_file),
        "size": run_binding_file.stat().st_size,
    }
    artifact = {
        "schema_version": TAU3_INTERNAL_VALIDATION_SCHEMA_VERSION,
        "created_at": created_at or _now_utc(),
        "passed": True,
        "method": {
            "name": INTERNAL_VALIDATION_METHOD,
            "full_prompt_conditioning": True,
            "target_only_loss": True,
            "prompt_tokens_masked": True,
            "candidate_weights_frozen": True,
            "batch_size": 1,
            "max_seq_length": max_seq_length,
        },
        "bindings": {
            "dataset_file_sha256": _sha256_file(dataset_file),
            "dataset_manifest_sha256": _sha256_file(
                dataset_manifest_file
            ),
            "training_receipt_sha256": _sha256_file(receipt_file),
            "adapter_tree_sha256": adapter_tree_sha256,
            "protocol_sha256": protocol_sha256,
            "model_identity_sha256": identity_sha256,
        },
        "coverage": replay["coverage"],
        "aggregate": replay["aggregate"],
        "measurements": rows_ref,
        "run_binding": run_binding_ref,
        "schema_checked": True,
    }
    schema = check_schema_contract(
        artifact,
        name_or_id="tau3_internal_validation",
    )
    if schema.get("passed") is not True:
        raise Tau3InternalValidationError(
            "internal-validation artifact violates schema: "
            + "; ".join(schema.get("errors") or [])
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    out.chmod(0o444)
    return artifact


def validate_tau3_internal_validation(
    artifact_path: str | Path,
    *,
    dataset_path: str | Path,
    training_receipt_path: str | Path,
    protocol_path: str | Path,
    model_identity_path: str | Path,
) -> dict[str, Any]:
    """Independently replay one internal-validation loss artifact."""

    artifact_file = Path(artifact_path)
    errors: list[str] = []
    try:
        artifact = _load_json_object(
            artifact_file,
            "internal-validation artifact",
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _check_result(artifact_file, [str(exc)])

    schema = check_schema_contract(
        artifact,
        name_or_id="tau3_internal_validation",
    )
    errors.extend(str(error) for error in schema.get("errors") or [])
    if artifact.get("passed") is not True:
        errors.append("artifact.passed must be true")
    method = _dict(artifact.get("method"))
    if method.get("name") != INTERNAL_VALIDATION_METHOD:
        errors.append("method.name is invalid")
    for field in (
        "full_prompt_conditioning",
        "target_only_loss",
        "prompt_tokens_masked",
        "candidate_weights_frozen",
    ):
        if method.get(field) is not True:
            errors.append(f"method.{field} must be true")
    if method.get("batch_size") != 1:
        errors.append("method.batch_size must be 1")
    max_seq_length = method.get("max_seq_length")
    if (
        not isinstance(max_seq_length, int)
        or isinstance(max_seq_length, bool)
        or max_seq_length < 1
    ):
        errors.append("method.max_seq_length must be positive")
        max_seq_length = 0

    dataset_file = Path(dataset_path)
    dataset_manifest_file = dataset_file.parent / "manifest.json"
    receipt_file = Path(training_receipt_path)
    protocol_file = Path(protocol_path)
    identity_file = Path(model_identity_path)
    bindings = _dict(artifact.get("bindings"))
    for field, path in (
        ("dataset_file_sha256", dataset_file),
        ("dataset_manifest_sha256", dataset_manifest_file),
        ("training_receipt_sha256", receipt_file),
        ("protocol_sha256", protocol_file),
        ("model_identity_sha256", identity_file),
    ):
        try:
            actual = _sha256_file(path)
        except OSError as exc:
            errors.append(f"{field} source is unreadable: {exc}")
            continue
        if bindings.get(field) != actual:
            errors.append(f"bindings.{field} does not match its source")
    try:
        dataset_manifest = _load_json_object(
            dataset_manifest_file,
            "dataset manifest",
        )
        _require_internal_validation_dataset_binding(
            dataset_file,
            dataset_manifest,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(str(exc))

    receipt: dict[str, Any] = {}
    try:
        receipt = _load_json_object(receipt_file, "training receipt")
        _require_successful_training_receipt(receipt)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(str(exc))
    if bindings.get("adapter_tree_sha256") != _nested(
        receipt,
        "adapter",
        "tree_sha256",
    ):
        errors.append(
            "bindings.adapter_tree_sha256 does not match training receipt"
        )
    if _nested(
        receipt,
        "training_binding",
        "recipe",
        "max_seq_length",
    ) != max_seq_length:
        errors.append("max_seq_length does not match the training recipe")
    if _nested(
        receipt,
        "training_binding",
        "dataset",
        "manifest_sha256",
    ) != bindings.get("dataset_manifest_sha256"):
        errors.append("dataset manifest does not match the training receipt")
    if _nested(receipt, "training_binding", "protocol", "sha256") != (
        bindings.get("protocol_sha256")
    ):
        errors.append("training receipt protocol binding mismatch")
    if _nested(receipt, "training_binding", "model", "identity_sha256") != (
        bindings.get("model_identity_sha256")
    ):
        errors.append("training receipt model identity binding mismatch")

    measurements_ref = _dict(artifact.get("measurements"))
    run_binding_ref = _dict(artifact.get("run_binding"))
    measurement_file: Path | None = None
    try:
        measurement_file = _resolve_relative_ref(
            artifact_file.parent,
            measurements_ref.get("path"),
        )
        if measurements_ref.get("sha256") != _sha256_file(
            measurement_file
        ):
            errors.append("measurements.sha256 does not replay")
        if measurements_ref.get("size") != measurement_file.stat().st_size:
            errors.append("measurements.size does not replay")
    except (OSError, ValueError) as exc:
        errors.append(f"measurements reference is invalid: {exc}")
    run_binding_file: Path | None = None
    try:
        run_binding_file = _resolve_relative_ref(
            artifact_file.parent,
            run_binding_ref.get("path"),
        )
        if run_binding_ref.get("sha256") != _sha256_file(
            run_binding_file
        ):
            errors.append("run_binding.sha256 does not replay")
        if run_binding_ref.get("size") != run_binding_file.stat().st_size:
            errors.append("run_binding.size does not replay")
    except (OSError, ValueError) as exc:
        errors.append(f"run binding reference is invalid: {exc}")

    if run_binding_file is not None and isinstance(
        bindings.get("adapter_tree_sha256"),
        str,
    ):
        try:
            run_binding = _load_json_object(
                run_binding_file,
                "internal-validation run binding",
            )
            expected_run_binding = _expected_run_binding(
                dataset_file=dataset_file,
                dataset_manifest_file=dataset_manifest_file,
                receipt_file=receipt_file,
                adapter_tree_sha256=bindings["adapter_tree_sha256"],
                protocol_file=protocol_file,
                identity_file=identity_file,
                max_seq_length=max_seq_length,
            )
            if run_binding != expected_run_binding:
                errors.append(
                    "run binding does not match the evaluated sources"
                )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(str(exc))

    replay: dict[str, Any] | None = None
    if measurement_file is not None and max_seq_length:
        try:
            dataset_rows = _read_jsonl_objects(
                dataset_file,
                "internal-validation dataset",
            )
            measurements = _read_jsonl_objects(
                measurement_file,
                "internal-validation measurements",
            )
            row_errors, replay = _replay_rows(
                dataset_rows,
                measurements,
                max_seq_length=max_seq_length,
            )
            errors.extend(row_errors)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(str(exc))
    if replay is not None:
        if artifact.get("coverage") != replay["coverage"]:
            errors.append("coverage does not replay from measurements")
        if not _json_numbers_close(
            artifact.get("aggregate"),
            replay["aggregate"],
        ):
            errors.append("aggregate does not replay from measurements")

    return _check_result(artifact_file, errors)


def _replay_rows(
    dataset_rows: list[dict[str, Any]],
    measurements: list[dict[str, Any]],
    *,
    max_seq_length: int,
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    if len(measurements) != len(dataset_rows):
        errors.append(
            "measurements must contain exactly one row per dataset row"
        )
    domain_counts: Counter[str] = Counter()
    behavior_counts: Counter[str] = Counter()
    row_hashes: list[str] = []
    total_supervised_tokens = 0
    total_loss_sum = 0.0
    numerical_failures = 0
    for index, dataset_row in enumerate(dataset_rows):
        if index >= len(measurements):
            break
        measurement = measurements[index]
        metadata = _dict(dataset_row.get("metadata"))
        token_counts = _dict(metadata.get("token_counts"))
        expected_hash = _canonical_sha256(dataset_row)
        domain = metadata.get("domain")
        behavior = metadata.get("behavior")
        input_token_ids = token_counts.get("input_token_ids")
        prompt_tokens = token_counts.get("prompt_tokens")
        supervised_tokens = token_counts.get("supervised_tokens")
        total_tokens = token_counts.get("total_tokens")
        label = f"row {index}"
        if measurement.get("row_index") != index:
            errors.append(f"{label} row_index mismatch")
        if measurement.get("row_sha256") != expected_hash:
            errors.append(f"{label} row_sha256 mismatch")
        if not isinstance(domain, str) or not domain:
            errors.append(f"{label} dataset domain is invalid")
            domain = ""
        if not isinstance(behavior, str) or not behavior:
            errors.append(f"{label} dataset behavior is invalid")
            behavior = ""
        if measurement.get("domain") != domain:
            errors.append(f"{label} domain mismatch")
        if measurement.get("behavior") != behavior:
            errors.append(f"{label} behavior mismatch")
        if not isinstance(input_token_ids, list) or not all(
            isinstance(token, int) and not isinstance(token, bool)
            for token in input_token_ids
        ):
            errors.append(f"{label} input_token_ids are invalid")
            input_token_ids = []
        if (
            not isinstance(prompt_tokens, int)
            or isinstance(prompt_tokens, bool)
            or prompt_tokens < 2
        ):
            errors.append(f"{label} prompt_tokens are invalid")
            prompt_tokens = 0
        if (
            not isinstance(supervised_tokens, int)
            or isinstance(supervised_tokens, bool)
            or supervised_tokens < 1
        ):
            errors.append(f"{label} supervised_tokens are invalid")
            supervised_tokens = 0
        if total_tokens != len(input_token_ids):
            errors.append(f"{label} total_tokens do not match token ids")
        if prompt_tokens + supervised_tokens != len(input_token_ids):
            errors.append(f"{label} prompt/target accounting mismatch")
        if len(input_token_ids) > max_seq_length:
            errors.append(f"{label} exceeds max_seq_length")
        if token_counts.get("input_token_ids_sha256") != (
            _canonical_sha256(input_token_ids)
        ):
            errors.append(f"{label} input_token_ids_sha256 mismatch")
        target_tokens = input_token_ids[prompt_tokens:]
        if measurement.get("target_tokens_sha256") != (
            _canonical_sha256(target_tokens)
        ):
            errors.append(f"{label} target_tokens_sha256 mismatch")
        if measurement.get("input_token_ids_sha256") != (
            _canonical_sha256(input_token_ids)
        ):
            errors.append(f"{label} input_token_ids_sha256 mismatch")
        for field, expected in (
            ("prompt_tokens", prompt_tokens),
            ("supervised_tokens", supervised_tokens),
        ):
            if measurement.get(field) != expected:
                errors.append(f"{label} {field} mismatch")
        mean_loss = measurement.get("mean_loss")
        loss_sum = measurement.get("loss_sum")
        finite = (
            isinstance(mean_loss, (int, float))
            and not isinstance(mean_loss, bool)
            and math.isfinite(float(mean_loss))
            and float(mean_loss) >= 0.0
            and isinstance(loss_sum, (int, float))
            and not isinstance(loss_sum, bool)
            and math.isfinite(float(loss_sum))
            and float(loss_sum) >= 0.0
        )
        if measurement.get("finite") is not True or not finite:
            numerical_failures += 1
            errors.append(f"{label} loss is not finite")
            continue
        expected_loss_sum = float(mean_loss) * supervised_tokens
        if not math.isclose(
            float(loss_sum),
            expected_loss_sum,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            errors.append(f"{label} loss_sum does not replay")
        row_hashes.append(expected_hash)
        domain_counts[domain] += 1
        behavior_counts[behavior] += 1
        total_supervised_tokens += supervised_tokens
        total_loss_sum += float(loss_sum)

    if set(domain_counts) != set(REQUIRED_DOMAINS):
        errors.append("coverage must include every required domain")
    if set(behavior_counts) != set(REQUIRED_BEHAVIORS):
        errors.append("coverage must include every required behavior")
    weighted_mean_loss = (
        total_loss_sum / total_supervised_tokens
        if total_supervised_tokens
        else math.inf
    )
    aggregate = {
        "total_supervised_tokens": total_supervised_tokens,
        "weighted_mean_loss": weighted_mean_loss,
        "perplexity": (
            math.exp(weighted_mean_loss)
            if math.isfinite(weighted_mean_loss)
            and weighted_mean_loss < 700
            else math.inf
        ),
        "numerical_failure_count": numerical_failures,
    }
    coverage = {
        "row_count": len(dataset_rows),
        "evaluated_row_count": len(measurements),
        "every_row_evaluated": len(measurements) == len(dataset_rows),
        "row_hashes_sha256": _canonical_sha256(row_hashes),
        "domains": [
            {"name": name, "row_count": domain_counts[name]}
            for name in sorted(domain_counts)
        ],
        "behaviors": [
            {"name": name, "row_count": behavior_counts[name]}
            for name in sorted(behavior_counts)
        ],
        "required_domains_exact": set(domain_counts) == set(REQUIRED_DOMAINS),
        "required_behaviors_exact": (
            set(behavior_counts) == set(REQUIRED_BEHAVIORS)
        ),
    }
    return errors, {"coverage": coverage, "aggregate": aggregate}


def _require_successful_training_receipt(receipt: dict[str, Any]) -> None:
    schema = check_schema_contract(
        receipt,
        name_or_id="tau3_mlx_training_run",
    )
    if schema.get("passed") is not True:
        raise Tau3InternalValidationError(
            "training receipt schema failed: "
            + "; ".join(schema.get("errors") or [])
        )
    if (
        receipt.get("phase") != "final"
        or receipt.get("terminal_status") != "success"
        or receipt.get("weights_updated") is not True
    ):
        raise Tau3InternalValidationError(
            "training receipt must be final, successful, and weights-updated"
        )


def _require_internal_validation_dataset_binding(
    dataset_file: Path,
    manifest: dict[str, Any],
) -> None:
    if manifest.get("schema_version") != (
        "hfr.tau3_competitive_dataset.v1"
    ):
        raise Tau3InternalValidationError(
            "dataset manifest must be hfr.tau3_competitive_dataset.v1"
        )
    valid = _nested(manifest, "files", "valid")
    if (
        not isinstance(valid, dict)
        or valid.get("path") != dataset_file.name
        or valid.get("sha256") != _sha256_file(dataset_file)
        or valid.get("bytes") != dataset_file.stat().st_size
    ):
        raise Tau3InternalValidationError(
            "dataset manifest valid split does not bind the validation file"
        )


def _expected_run_binding(
    *,
    dataset_file: Path,
    dataset_manifest_file: Path,
    receipt_file: Path,
    adapter_tree_sha256: str,
    protocol_file: Path,
    identity_file: Path,
    max_seq_length: int,
) -> dict[str, Any]:
    return {
        "schema_version": "hfr.tau3_internal_validation_run_binding.v1",
        "method": INTERNAL_VALIDATION_METHOD,
        "candidate_weights_frozen": True,
        "max_seq_length": max_seq_length,
        "bindings": {
            "dataset_file_sha256": _sha256_file(dataset_file),
            "dataset_manifest_sha256": _sha256_file(
                dataset_manifest_file
            ),
            "training_receipt_sha256": _sha256_file(receipt_file),
            "adapter_tree_sha256": adapter_tree_sha256,
            "protocol_sha256": _sha256_file(protocol_file),
            "model_identity_sha256": _sha256_file(identity_file),
        },
    }


def _check_result(path: Path, errors: list[str]) -> dict[str, Any]:
    return {
        "schema_version": TAU3_INTERNAL_VALIDATION_CHECK_SCHEMA_VERSION,
        "artifact": {
            "path": str(path),
            "sha256": _sha256_file(path) if path.is_file() else None,
        },
        "passed": not errors,
        "error_count": len(errors),
        "errors": errors,
    }


def _read_jsonl_objects(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise Tau3InternalValidationError(
            f"{label} is unreadable: {exc}"
        ) from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Tau3InternalValidationError(
                f"{label} line {line_number} is invalid JSON: {exc.msg}"
            ) from exc
        if not isinstance(row, dict):
            raise Tau3InternalValidationError(
                f"{label} line {line_number} must be an object"
            )
        rows.append(row)
    if not rows:
        raise Tau3InternalValidationError(f"{label} contains no rows")
    return rows


def _load_json_object(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Tau3InternalValidationError(
            f"{label} is unreadable: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise Tau3InternalValidationError(f"{label} must be a JSON object")
    return payload


def _portable_relative_ref(path: Path, parent: Path) -> str:
    resolved = path.resolve(strict=True)
    base = parent.resolve(strict=True)
    try:
        relative = resolved.relative_to(base)
    except ValueError as exc:
        raise Tau3InternalValidationError(
            "measurements must be stored beneath the artifact directory"
        ) from exc
    if relative == Path(".") or ".." in relative.parts:
        raise Tau3InternalValidationError(
            "measurements path must be a portable relative file"
        )
    return relative.as_posix()


def _resolve_relative_ref(parent: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise Tau3InternalValidationError(
            "measurements.path must be a non-empty string"
        )
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or "\\" in value:
        raise Tau3InternalValidationError(
            "measurements.path must be portable and relative"
        )
    base = parent.resolve(strict=True)
    resolved = (base / relative).resolve(strict=True)
    if not resolved.is_relative_to(base) or not resolved.is_file():
        raise Tau3InternalValidationError(
            "measurements.path escapes the artifact directory"
        )
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _nested(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _json_numbers_close(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _json_numbers_close(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _json_numbers_close(a, b) for a, b in zip(left, right, strict=True)
        )
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        return math.isclose(
            float(left),
            float(right),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    return left == right


def _now_utc() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )
