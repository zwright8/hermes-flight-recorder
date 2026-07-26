"""Fail-closed validation for Tau-3 detached-prefix qualification evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from .schema_registry import SchemaRegistryError, check_schema_contract

TAU3_PREFIX_EQUIVALENCE_SCHEMA_VERSION = "hfr.tau3_prefix_equivalence.v1"
TAU3_PREFIX_EQUIVALENCE_VALIDATION_SCHEMA_VERSION = (
    "hfr.tau3_prefix_equivalence_validation.v1"
)
REQUIRED_EQUIVALENCE_PROBE_FAMILIES = (
    "tool_choice",
    "clarification",
    "recovery",
    "stopping",
    "state_transition",
)
REQUIRED_DOMAINS = ("airline", "retail", "telecom")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class Tau3PrefixEquivalenceError(ValueError):
    """Raised when prefix-equivalence evidence cannot be read safely."""


def validate_tau3_prefix_equivalence(
    artifact_or_path: dict[str, Any] | str | Path,
    *,
    expected_bindings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay a bounded full-gradient versus detached-prefix A/B artifact."""

    path: Path | None = None
    if isinstance(artifact_or_path, dict):
        artifact = artifact_or_path
    else:
        path = Path(artifact_or_path)
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return _result(path, [f"artifact is unreadable: {exc}"])
        if not isinstance(artifact, dict):
            return _result(path, ["artifact must be a JSON object"])

    errors = _schema_errors(artifact)
    method = _dict(artifact.get("method"))
    if method.get("reference") != "standard_full_gradient_mlx_lora":
        errors.append("method.reference must be standard_full_gradient_mlx_lora")
    if (
        method.get("candidate")
        != "detached_complete_prompt_cache_mlx_lora"
    ):
        errors.append(
            "method.candidate must be detached_complete_prompt_cache_mlx_lora"
        )
    max_probe_drop = _finite_number(
        method.get("max_material_probe_drop"),
        errors,
        "method.max_material_probe_drop",
    )
    if max_probe_drop is not None and not 0 <= max_probe_drop <= 0.05:
        errors.append("method.max_material_probe_drop must be between 0 and 0.05")
    max_loss_delta = _finite_number(
        method.get("max_loss_replay_delta"),
        errors,
        "method.max_loss_replay_delta",
    )
    if max_loss_delta is not None and not 0 <= max_loss_delta <= 0.0001:
        errors.append("method.max_loss_replay_delta must be between 0 and 0.0001")

    bindings = _dict(artifact.get("bindings"))
    for field in (
        "dataset_file_sha256",
        "protocol_file_sha256",
        "model_identity_file_sha256",
    ):
        if not _is_sha256(bindings.get(field)):
            errors.append(f"bindings.{field} must be a sha256")
    recipe = _dict(bindings.get("recipe"))
    _validate_recipe(recipe, errors)
    if expected_bindings is not None:
        _validate_expected_bindings(bindings, expected_bindings, errors)

    sample = _dict(artifact.get("sample"))
    row_hashes = sample.get("row_hashes")
    if not isinstance(row_hashes, list) or not row_hashes:
        errors.append("sample.row_hashes must be a non-empty list")
        row_hashes = []
    elif (
        any(not _is_sha256(value) for value in row_hashes)
        or len(set(row_hashes)) != len(row_hashes)
    ):
        errors.append("sample.row_hashes must contain unique sha256 values")
    if sample.get("row_count") != len(row_hashes):
        errors.append("sample.row_count does not replay from row_hashes")
    if sample.get("row_hashes_sha256") != _canonical_sha256(row_hashes):
        errors.append("sample.row_hashes_sha256 does not replay")
    if sample.get("stratified") is not True:
        errors.append("sample.stratified must be true")
    if set(_string_list(sample.get("domains"))) != set(REQUIRED_DOMAINS):
        errors.append("sample.domains must include airline, retail, and telecom")
    if set(_string_list(sample.get("probe_families"))) != set(
        REQUIRED_EQUIVALENCE_PROBE_FAMILIES
    ):
        errors.append("sample.probe_families must include every required family")

    target = _dict(artifact.get("target_accounting"))
    full_target = _dict(target.get("full_gradient"))
    prefix_target = _dict(target.get("detached_prefix"))
    target_fields = (
        "sample_row_count",
        "supervised_token_count",
        "target_boundaries_sha256",
        "loss_mask_sha256",
        "target_tokens_sha256",
    )
    if any(full_target.get(field) != prefix_target.get(field) for field in target_fields):
        errors.append(
            "target accounting must match between full-gradient and detached-prefix arms"
        )
    if full_target.get("sample_row_count") != len(row_hashes):
        errors.append("target accounting sample_row_count must match the sample")
    if not isinstance(full_target.get("supervised_token_count"), int) or int(
        full_target.get("supervised_token_count") or 0
    ) < 1:
        errors.append("target accounting supervised_token_count must be positive")
    for arm_name, arm in (
        ("full_gradient", full_target),
        ("detached_prefix", prefix_target),
    ):
        target_rows = arm.get("rows")
        if not isinstance(target_rows, list):
            errors.append(f"target_accounting.{arm_name}.rows must be a list")
            target_rows = []
        _replay_target_accounting(
            arm_name,
            arm,
            target_rows,
            row_hashes,
            errors,
        )
        for field in (
            "target_boundaries_sha256",
            "loss_mask_sha256",
            "target_tokens_sha256",
        ):
            if not _is_sha256(arm.get(field)):
                errors.append(f"target_accounting.{arm_name}.{field} must be a sha256")
    if full_target.get("rows") != prefix_target.get("rows"):
        errors.append(
            "target accounting row evidence must match between full-gradient "
            "and detached-prefix arms"
        )

    gradients = _dict(artifact.get("gradient_evidence"))
    modules = gradients.get("intended_modules")
    if (
        not isinstance(modules, list)
        or not modules
        or any(not isinstance(item, str) or not item for item in modules)
        or len(set(modules)) != len(modules)
    ):
        errors.append("gradient_evidence.intended_modules must be unique names")
        modules = []
    if gradients.get("intended_modules_sha256") != _canonical_sha256(modules):
        errors.append("gradient_evidence.intended_modules_sha256 does not replay")
    for arm_name in ("full_gradient", "detached_prefix"):
        arm = _dict(gradients.get(arm_name))
        module_records = arm.get("modules")
        if not isinstance(module_records, list):
            errors.append(f"gradient_evidence.{arm_name}.modules must be a list")
            module_records = []
        module_names: list[str] = []
        squared_norm = 0.0
        invalid_module = False
        for index, module_record in enumerate(module_records):
            if not isinstance(module_record, dict):
                errors.append(
                    f"gradient_evidence.{arm_name}.modules[{index}] must be an object"
                )
                invalid_module = True
                continue
            name = module_record.get("name")
            norm = _finite_number(
                module_record.get("l2_norm"),
                errors,
                f"gradient_evidence.{arm_name}.modules[{index}].l2_norm",
            )
            if not isinstance(name, str) or not name:
                errors.append(
                    f"gradient_evidence.{arm_name}.modules[{index}].name must be nonempty"
                )
                invalid_module = True
            else:
                module_names.append(name)
            if norm is None or norm <= 0:
                errors.append(
                    f"gradient_evidence.{arm_name}.modules[{index}].l2_norm "
                    "must be positive"
                )
                invalid_module = True
            else:
                squared_norm += norm * norm
        if module_names != modules:
            errors.append(
                f"gradient_evidence.{arm_name}.modules must exactly match "
                "intended_modules"
            )
        if arm.get("finite") is not True:
            errors.append(f"gradient_evidence.{arm_name}.finite must be true")
        if arm.get("nonzero_module_count") != len(modules) or not modules:
            errors.append(
                f"gradient_evidence.{arm_name} must have nonzero gradients "
                "for every intended adapter module"
            )
        norm = _finite_number(
            arm.get("gradient_l2_norm"),
            errors,
            f"gradient_evidence.{arm_name}.gradient_l2_norm",
        )
        if norm is not None and norm <= 0:
            errors.append(
                f"gradient_evidence.{arm_name}.gradient_l2_norm must be positive"
            )
        if (
            norm is not None
            and not invalid_module
            and not math.isclose(
                norm,
                math.sqrt(squared_norm),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            errors.append(
                f"gradient_evidence.{arm_name}.gradient_l2_norm does not "
                "replay from module evidence"
            )

    probes = _dict(artifact.get("behavior_probes"))
    if set(_string_list(probes.get("required_families"))) != set(
        REQUIRED_EQUIVALENCE_PROBE_FAMILIES
    ):
        errors.append(
            "behavior_probes.required_families must include every required family"
        )
    family_results = _dict(probes.get("family_results"))
    for family in REQUIRED_EQUIVALENCE_PROBE_FAMILIES:
        item = _dict(family_results.get(family))
        trials = item.get("trials")
        if not isinstance(trials, list) or not trials:
            errors.append(f"behavior_probes.{family}.trials must be non-empty")
            trials = []
        seen_probe_ids: set[str] = set()
        full_passes = 0
        prefix_passes = 0
        for index, trial in enumerate(trials):
            if not isinstance(trial, dict):
                errors.append(
                    f"behavior_probes.{family}.trials[{index}] must be an object"
                )
                continue
            probe_id = trial.get("probe_id")
            if (
                not isinstance(probe_id, str)
                or not probe_id
                or probe_id in seen_probe_ids
            ):
                errors.append(
                    f"behavior_probes.{family}.trials[{index}].probe_id "
                    "must be unique and nonempty"
                )
            else:
                seen_probe_ids.add(probe_id)
            for arm_field in (
                "full_gradient_passed",
                "detached_prefix_passed",
            ):
                if not isinstance(trial.get(arm_field), bool):
                    errors.append(
                        f"behavior_probes.{family}.trials[{index}].{arm_field} "
                        "must be boolean"
                    )
            full_passes += trial.get("full_gradient_passed") is True
            prefix_passes += trial.get("detached_prefix_passed") is True
        if not isinstance(item.get("trial_count"), int) or int(
            item.get("trial_count") or 0
        ) < 1:
            errors.append(f"behavior_probes.{family}.trial_count must be positive")
        elif item.get("trial_count") != len(trials):
            errors.append(
                f"behavior_probes.{family}.trial_count does not replay"
            )
        full_rate = _rate(
            item.get("full_gradient_pass_rate"),
            errors,
            f"behavior_probes.{family}.full_gradient_pass_rate",
        )
        prefix_rate = _rate(
            item.get("detached_prefix_pass_rate"),
            errors,
            f"behavior_probes.{family}.detached_prefix_pass_rate",
        )
        replayed_full_rate = (
            full_passes / len(trials) if trials else None
        )
        replayed_prefix_rate = (
            prefix_passes / len(trials) if trials else None
        )
        if (
            full_rate is not None
            and replayed_full_rate is not None
            and not math.isclose(
                full_rate,
                replayed_full_rate,
                rel_tol=0,
                abs_tol=1e-12,
            )
        ):
            errors.append(
                f"behavior_probes.{family}.full_gradient_pass_rate does not replay"
            )
        if (
            prefix_rate is not None
            and replayed_prefix_rate is not None
            and not math.isclose(
                prefix_rate,
                replayed_prefix_rate,
                rel_tol=0,
                abs_tol=1e-12,
            )
        ):
            errors.append(
                f"behavior_probes.{family}.detached_prefix_pass_rate does not replay"
            )
        if (
            replayed_full_rate is not None
            and replayed_prefix_rate is not None
            and max_probe_drop is not None
            and replayed_prefix_rate + max_probe_drop < replayed_full_rate
        ):
            errors.append(
                f"behavior_probes.{family} shows material degradation "
                "for detached-prefix training"
            )

    stability = _dict(artifact.get("stability"))
    for arm_name in ("full_gradient", "detached_prefix"):
        arm = _dict(stability.get(arm_name))
        replays = arm.get("replays")
        if not isinstance(replays, list):
            errors.append(f"stability.{arm_name}.replays must be a list")
            replays = []
        replay_losses: list[list[float]] = []
        replay_peak_memory: list[int] = []
        replay_numerical_failures = 0
        for replay_index, replay in enumerate(replays):
            if not isinstance(replay, dict):
                errors.append(
                    f"stability.{arm_name}.replays[{replay_index}] must be an object"
                )
                continue
            losses = replay.get("losses")
            if not isinstance(losses, list) or not losses:
                errors.append(
                    f"stability.{arm_name}.replays[{replay_index}].losses "
                    "must be non-empty"
                )
                losses = []
            numeric_losses: list[float] = []
            for loss_index, loss in enumerate(losses):
                value = _finite_number(
                    loss,
                    errors,
                    f"stability.{arm_name}.replays[{replay_index}]"
                    f".losses[{loss_index}]",
                )
                if value is not None:
                    numeric_losses.append(value)
            replay_losses.append(numeric_losses)
            peak = replay.get("peak_memory_bytes")
            if not isinstance(peak, int) or isinstance(peak, bool) or peak < 1:
                errors.append(
                    f"stability.{arm_name}.replays[{replay_index}]"
                    ".peak_memory_bytes must be positive"
                )
            else:
                replay_peak_memory.append(peak)
            failures_count = replay.get("numerical_failure_count")
            if (
                not isinstance(failures_count, int)
                or isinstance(failures_count, bool)
                or failures_count < 0
            ):
                errors.append(
                    f"stability.{arm_name}.replays[{replay_index}]"
                    ".numerical_failure_count must be nonnegative"
                )
            else:
                replay_numerical_failures += failures_count
        replay_count = arm.get("replay_count")
        loss_count = arm.get("loss_count")
        if not isinstance(replay_count, int) or replay_count < 2:
            errors.append(f"stability.{arm_name}.replay_count must be at least two")
        elif replay_count != len(replays):
            errors.append(f"stability.{arm_name}.replay_count does not replay")
        if not isinstance(loss_count, int) or loss_count < 1:
            errors.append(f"stability.{arm_name}.loss_count must be positive")
        replayed_loss_count = sum(len(losses) for losses in replay_losses)
        if loss_count != replayed_loss_count:
            errors.append(f"stability.{arm_name}.loss_count does not replay")
        if arm.get("finite_loss_count") != replayed_loss_count:
            errors.append(
                f"stability.{arm_name}.finite_loss_count does not replay"
            )
        if (
            arm.get("numerical_failure_count") != 0
            or replay_numerical_failures != 0
        ):
            errors.append(
                f"stability.{arm_name}.numerical_failure_count must be zero"
            )
        if (
            not isinstance(arm.get("peak_memory_bytes"), int)
            or int(arm.get("peak_memory_bytes") or 0) < 1
            or (
                replay_peak_memory
                and arm.get("peak_memory_bytes") != max(replay_peak_memory)
            )
        ):
            errors.append(f"stability.{arm_name}.peak_memory_bytes must be positive")
        replayed_delta = _max_replay_delta(replay_losses)
        delta = _finite_number(
            arm.get("loss_replay_max_abs_delta"),
            errors,
            f"stability.{arm_name}.loss_replay_max_abs_delta",
        )
        if (
            delta is not None
            and replayed_delta is not None
            and not math.isclose(
                delta,
                replayed_delta,
                rel_tol=0,
                abs_tol=1e-12,
            )
        ):
            errors.append(
                f"stability.{arm_name}.loss_replay_max_abs_delta does not replay"
            )
        if (
            delta is not None
            and max_loss_delta is not None
            and delta > max_loss_delta
        ):
            errors.append(
                f"stability.{arm_name}.loss replay exceeds the frozen tolerance"
            )
        if not _is_sha256(arm.get("replay_sha256")):
            errors.append(f"stability.{arm_name}.replay_sha256 must be a sha256")
        elif arm.get("replay_sha256") != _canonical_sha256(replays):
            errors.append(f"stability.{arm_name}.replay_sha256 does not replay")

    failures = artifact.get("failures")
    if failures != []:
        errors.append("failures must be an empty list for qualification")
    if artifact.get("passed") is not True:
        errors.append("artifact.passed must be true for qualification")
    return _result(path, errors)


def _validate_recipe(recipe: dict[str, Any], errors: list[str]) -> None:
    positive_ints = (
        "rank",
        "num_layers",
        "max_seq_length",
        "batch_size",
        "grad_accumulation",
    )
    for field in positive_ints:
        if not isinstance(recipe.get(field), int) or isinstance(
            recipe.get(field), bool
        ) or int(recipe.get(field) or 0) < 1:
            errors.append(f"bindings.recipe.{field} must be a positive integer")
    for field in ("scale", "learning_rate"):
        value = _finite_number(
            recipe.get(field),
            errors,
            f"bindings.recipe.{field}",
        )
        if value is not None and value <= 0:
            errors.append(f"bindings.recipe.{field} must be positive")
    if recipe.get("mask_prompt") is not True:
        errors.append("bindings.recipe.mask_prompt must be true")
    seeds = recipe.get("allowed_seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(
            not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
            for seed in seeds
        )
        or len(set(seeds)) != len(seeds)
    ):
        errors.append("bindings.recipe.allowed_seeds must be unique integers")


def _replay_target_accounting(
    arm_name: str,
    arm: dict[str, Any],
    rows: list[Any],
    sample_row_hashes: list[Any],
    errors: list[str],
) -> None:
    boundaries: list[dict[str, Any]] = []
    mask_records: list[dict[str, Any]] = []
    target_records: list[dict[str, Any]] = []
    replayed_row_hashes: list[str] = []
    supervised_tokens = 0
    for index, row in enumerate(rows):
        label = f"target_accounting.{arm_name}.rows[{index}]"
        if not isinstance(row, dict):
            errors.append(f"{label} must be an object")
            continue
        row_sha256 = row.get("row_sha256")
        if not _is_sha256(row_sha256):
            errors.append(f"{label}.row_sha256 must be a sha256")
            continue
        replayed_row_hashes.append(row_sha256)
        prompt_offset = row.get("prompt_offset")
        target_start = row.get("target_start")
        target_end = row.get("target_end")
        token_count = row.get("supervised_token_count")
        for field, value in (
            ("prompt_offset", prompt_offset),
            ("target_start", target_start),
            ("target_end", target_end),
            ("supervised_token_count", token_count),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
            ):
                errors.append(f"{label}.{field} must be a positive integer")
        if (
            isinstance(prompt_offset, int)
            and isinstance(target_start, int)
            and target_start != prompt_offset
        ):
            errors.append(f"{label}.target_start must equal prompt_offset")
        if (
            isinstance(target_start, int)
            and isinstance(target_end, int)
            and isinstance(token_count, int)
            and target_end - target_start != token_count
        ):
            errors.append(
                f"{label}.target_end-target_start must equal "
                "supervised_token_count"
            )
        loss_mask_sha256 = row.get("loss_mask_sha256")
        target_tokens_sha256 = row.get("target_tokens_sha256")
        if not _is_sha256(loss_mask_sha256):
            errors.append(f"{label}.loss_mask_sha256 must be a sha256")
        if not _is_sha256(target_tokens_sha256):
            errors.append(f"{label}.target_tokens_sha256 must be a sha256")
        if isinstance(token_count, int) and not isinstance(token_count, bool):
            supervised_tokens += token_count
        boundaries.append(
            {
                "row_sha256": row_sha256,
                "prompt_offset": prompt_offset,
                "target_start": target_start,
                "target_end": target_end,
                "supervised_token_count": token_count,
            }
        )
        mask_records.append(
            {
                "row_sha256": row_sha256,
                "loss_mask_sha256": loss_mask_sha256,
            }
        )
        target_records.append(
            {
                "row_sha256": row_sha256,
                "target_tokens_sha256": target_tokens_sha256,
            }
        )
    if replayed_row_hashes != sample_row_hashes:
        errors.append(
            f"target_accounting.{arm_name}.rows must match sample row order"
        )
    if arm.get("sample_row_count") != len(rows):
        errors.append(
            f"target_accounting.{arm_name}.sample_row_count does not replay"
        )
    if arm.get("supervised_token_count") != supervised_tokens:
        errors.append(
            f"target_accounting.{arm_name}.supervised_token_count does not replay"
        )
    if arm.get("target_boundaries_sha256") != _canonical_sha256(boundaries):
        errors.append(
            f"target_accounting.{arm_name}.target_boundaries_sha256 does not replay"
        )
    if arm.get("loss_mask_sha256") != _canonical_sha256(mask_records):
        errors.append(
            f"target_accounting.{arm_name}.loss_mask_sha256 does not replay"
        )
    if arm.get("target_tokens_sha256") != _canonical_sha256(target_records):
        errors.append(
            f"target_accounting.{arm_name}.target_tokens_sha256 does not replay"
        )


def _max_replay_delta(replay_losses: list[list[float]]) -> float | None:
    if len(replay_losses) < 2 or not replay_losses[0]:
        return None
    reference = replay_losses[0]
    if any(len(losses) != len(reference) for losses in replay_losses[1:]):
        return math.inf
    return max(
        abs(value - reference[index])
        for losses in replay_losses[1:]
        for index, value in enumerate(losses)
    )


def _validate_expected_bindings(
    recorded: dict[str, Any],
    expected: dict[str, Any],
    errors: list[str],
) -> None:
    for field in (
        "dataset_file_sha256",
        "protocol_file_sha256",
        "model_identity_file_sha256",
    ):
        if recorded.get(field) != expected.get(field):
            errors.append(
                f"bindings.{field} does not match the candidate launch binding"
            )
    recorded_recipe = _dict(recorded.get("recipe"))
    expected_recipe = _dict(expected.get("recipe"))
    for field in (
        "rank",
        "scale",
        "learning_rate",
        "num_layers",
        "max_seq_length",
        "batch_size",
        "grad_accumulation",
        "mask_prompt",
    ):
        if recorded_recipe.get(field) != expected_recipe.get(field):
            errors.append(
                f"bindings.recipe.{field} does not match the candidate recipe"
            )
    seed = expected_recipe.get("seed")
    allowed = recorded_recipe.get("allowed_seeds")
    if not isinstance(allowed, list) or seed not in allowed:
        errors.append("bindings.recipe.seed is not qualified by allowed_seeds")


def _schema_errors(artifact: dict[str, Any]) -> list[str]:
    try:
        result = check_schema_contract(
            artifact,
            name_or_id="tau3_prefix_equivalence",
        )
    except SchemaRegistryError as exc:
        return [f"tau3_prefix_equivalence schema is unavailable: {exc}"]
    return [f"schema: {error}" for error in result.get("errors", [])]


def _result(path: Path | None, errors: list[str]) -> dict[str, Any]:
    return {
        "schema_version": TAU3_PREFIX_EQUIVALENCE_VALIDATION_SCHEMA_VERSION,
        "artifact": {
            "path": str(path) if path is not None else None,
            "sha256": _sha256_file(path) if path is not None and path.is_file() else None,
        },
        "passed": not errors,
        "error_count": len(errors),
        "errors": errors,
    }


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _finite_number(
    value: Any,
    errors: list[str],
    label: str,
) -> float | None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        errors.append(f"{label} must be a finite number")
        return None
    return float(value)


def _rate(value: Any, errors: list[str], label: str) -> float | None:
    number = _finite_number(value, errors, label)
    if number is not None and not 0 <= number <= 1:
        errors.append(f"{label} must be between zero and one")
        return None
    return number


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
