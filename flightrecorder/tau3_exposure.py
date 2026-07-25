"""Deterministic Tau-3 training exposure ledgers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

TAU3_TRAINING_EXPOSURE_SCHEMA_VERSION = "hfr.tau3_training_exposure.v1"
TAU3_COMPETITIVE_ROW_SCHEMA_VERSION = "hfr.tau3_competitive_dataset_row.v1"
LEGACY_EXPOSURE_ROW_SCHEMA_VERSION = "hfr.tau3_training_exposure_legacy_fixture.v1"
ASSISTANT_MESSAGE_TARGET = "assistant_message"

REQUIRED_LEGACY_ROW_METADATA = (
    "domain",
    "behavior",
    "target_tool",
    "action_class",
    "result_class",
    "length_bucket",
    "source_family",
    "source_provenance",
    "prompt_tokens",
    "supervised_tokens",
)
REQUIRED_COMPETITIVE_ROW_METADATA = (
    "domain",
    "behavior",
    "target_tool_name",
    "target_action_class",
    "preceding_result_class",
    "source_family_id",
    "source_provenance",
    "token_counts",
)
REQUIRED_DOMAINS = ("airline", "retail", "telecom")
REQUIRED_BEHAVIORS = (
    "successful_completion",
    "clarification_refusal",
    "authentication",
    "confirmation_before_mutation",
    "later_task_completion_actions",
    "safe_stopping",
    "transfer_handoff",
    "empty_result_recovery",
    "error_result_recovery",
    "repeated_call_recovery",
    "hallucinated_tool_correction",
    "harmful_mutation_correction",
    "premature_completion_correction",
)
RECOVERY_BEHAVIORS = {
    "empty_result_recovery",
    "error_result_recovery",
    "repeated_call_recovery",
}
MUTATION_BEHAVIORS = {
    "confirmation_before_mutation",
    "harmful_mutation_correction",
}


class Tau3ExposureError(ValueError):
    """Raised when exposure-ledger construction or validation fails closed."""


@dataclass(frozen=True)
class ExposureRow:
    index: int
    row: dict[str, Any]
    row_hash: str
    metadata: dict[str, Any]

    @property
    def stratum(self) -> tuple[str, ...]:
        return (
            str(self.metadata["domain"]),
            str(self.metadata["behavior"]),
            str(self.metadata["target_tool"]),
            str(self.metadata["action_class"]),
            str(self.metadata["result_class"]),
            str(self.metadata["length_bucket"]),
            str(self.metadata["source_family"]),
        )


def build_tau3_exposure_ledger(
    dataset_jsonl: str | Path,
    out_dir: str | Path,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int = 1,
    sampler_name: str = "deterministic_stratified_round_robin_v1",
    split: str = "train",
) -> dict[str, Any]:
    """Build a replayable full-row exposure ledger for Tau-3 training."""

    source = Path(dataset_jsonl)
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise Tau3ExposureError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    rows = load_exposure_rows(source)
    config = normalize_sampler_config(
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        sampler_name=sampler_name,
        split=split,
    )
    plan = plan_exposures(rows, config)
    ledger_path = out / "training_exposure_ledger.jsonl"
    _write_jsonl(ledger_path, plan["ledger"])

    receipt = build_exposure_receipt(
        rows,
        config,
        dataset_path=source,
        ledger_path=ledger_path,
        ledger=plan["ledger"],
        coverage=plan["coverage"],
    )
    receipt_path = out / "training_exposure_receipt.json"
    _write_json(receipt_path, receipt)
    return {**receipt, "receipt_path": str(receipt_path)}


def validate_tau3_exposure_ledger(
    dataset_jsonl: str | Path,
    receipt_path: str | Path,
    ledger_path: str | Path | None = None,
) -> dict[str, Any]:
    """Recompute and validate a Tau-3 exposure ledger without trusting it."""

    source = Path(dataset_jsonl)
    receipt_file = Path(receipt_path)
    receipt = _load_json(receipt_file)
    if receipt.get("schema_version") != TAU3_TRAINING_EXPOSURE_SCHEMA_VERSION:
        raise Tau3ExposureError("receipt schema_version is not hfr.tau3_training_exposure.v1")
    expected_ledger_path = Path(ledger_path) if ledger_path is not None else receipt_file.parent / str(
        receipt.get("files", {}).get("ledger", {}).get("path", "")
    )
    if expected_ledger_path.name != "training_exposure_ledger.jsonl":
        raise Tau3ExposureError("ledger path must end in training_exposure_ledger.jsonl")

    rows = load_exposure_rows(source)
    config = _normalize_receipt_sampler_config(receipt["sampler_config"])
    recomputed = plan_exposures(rows, config)
    emitted_ledger = _load_jsonl(expected_ledger_path)
    if emitted_ledger != recomputed["ledger"]:
        raise Tau3ExposureError("ledger does not replay from dataset content, sampler config, and seed")
    recomputed_receipt = build_exposure_receipt(
        rows,
        config,
        dataset_path=source,
        ledger_path=expected_ledger_path,
        ledger=recomputed["ledger"],
        coverage=recomputed["coverage"],
    )
    comparable_receipt = dict(receipt)
    comparable_receipt.pop("receipt_path", None)
    if comparable_receipt != recomputed_receipt:
        raise Tau3ExposureError("receipt does not replay from dataset and ledger")
    return {
        "schema_version": "hfr.tau3_training_exposure_validation.v1",
        "passed": True,
        "dataset_label": source.name,
        "receipt_path": str(receipt_file),
        "ledger_path": str(expected_ledger_path),
        "row_count": len(rows),
        "step_count": len(emitted_ledger),
        "candidate_eligible": bool(receipt["candidate_eligibility"]["passed"]),
        "receipt_sha256": _sha256_file(receipt_file),
        "ledger_sha256": _sha256_file(expected_ledger_path),
    }


def load_exposure_rows(path: str | Path) -> list[ExposureRow]:
    rows: list[ExposureRow] = []
    for line_number, payload in _iter_jsonl(Path(path)):
        if not isinstance(payload, dict):
            raise Tau3ExposureError(f"line {line_number}: row must be a JSON object")
        rows.append(
            ExposureRow(
                index=len(rows),
                row=payload,
                row_hash=_canonical_sha256(payload),
                metadata=_normalize_row_metadata(payload.get("metadata"), line_number),
            )
        )
    if not rows:
        raise Tau3ExposureError("dataset must contain at least one row")
    hashes = [row.row_hash for row in rows]
    if len(set(hashes)) != len(hashes):
        raise Tau3ExposureError("dataset contains duplicate canonical rows")
    return rows


def _normalize_row_metadata(raw: Any, line_number: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise Tau3ExposureError(f"line {line_number}: metadata must be an object")
    if raw.get("schema_version") == TAU3_COMPETITIVE_ROW_SCHEMA_VERSION:
        return _normalize_competitive_row_metadata(raw, line_number)
    return _normalize_legacy_row_metadata(raw, line_number)


def _normalize_competitive_row_metadata(raw: dict[str, Any], line_number: int) -> dict[str, Any]:
    missing = [field for field in REQUIRED_COMPETITIVE_ROW_METADATA if field not in raw]
    if missing:
        raise Tau3ExposureError(f"line {line_number}: missing metadata.{missing[0]}")
    domain = _required_string(raw, "domain", line_number)
    behavior = _required_string(raw, "behavior", line_number)
    action_class = _required_string(raw, "target_action_class", line_number)
    result_class = _required_string(raw, "preceding_result_class", line_number)
    source_family = _required_string(raw, "source_family_id", line_number)
    target_tool = _normalize_target_tool(raw, action_class, line_number)
    source_provenance = raw["source_provenance"]
    if not isinstance(source_provenance, dict):
        raise Tau3ExposureError(f"line {line_number}: metadata.source_provenance must be an object")
    method = source_provenance.get("method")
    if not isinstance(method, str) or not method:
        raise Tau3ExposureError(f"line {line_number}: metadata.source_provenance.method must be a non-empty string")
    token_counts = raw["token_counts"]
    if not isinstance(token_counts, dict):
        raise Tau3ExposureError(f"line {line_number}: metadata.token_counts must be an object")
    if token_counts.get("exact") is not True:
        raise Tau3ExposureError(f"line {line_number}: metadata.token_counts.exact must be true")
    if token_counts.get("chat_template_aware") is not True:
        raise Tau3ExposureError(f"line {line_number}: metadata.token_counts.chat_template_aware must be true")
    method_name = str(token_counts.get("method") or "")
    if "estimate" in method_name or method_name == "deterministic_json_char4_estimate":
        raise Tau3ExposureError(f"line {line_number}: metadata.token_counts.method must not be estimated")
    prompt_tokens = _required_positive_int(token_counts, "prompt_tokens", line_number, prefix="metadata.token_counts")
    supervised_tokens = _required_positive_int(token_counts, "supervised_tokens", line_number, prefix="metadata.token_counts")
    derived_length_bucket = _length_bucket(prompt_tokens + supervised_tokens)
    explicit_length_bucket = raw.get("length_bucket")
    if explicit_length_bucket is not None:
        if not isinstance(explicit_length_bucket, str) or not explicit_length_bucket:
            raise Tau3ExposureError(f"line {line_number}: metadata.length_bucket must be a non-empty string")
        if explicit_length_bucket != derived_length_bucket:
            raise Tau3ExposureError(f"line {line_number}: metadata.length_bucket does not match exact token total")
    return {
        "domain": domain,
        "behavior": behavior,
        "target_tool": target_tool,
        "action_class": action_class,
        "result_class": result_class,
        "length_bucket": derived_length_bucket,
        "source_family": source_family,
        "source_provenance": method,
        "prompt_tokens": prompt_tokens,
        "supervised_tokens": supervised_tokens,
        "_row_schema_version": TAU3_COMPETITIVE_ROW_SCHEMA_VERSION,
    }


def _normalize_legacy_row_metadata(raw: dict[str, Any], line_number: int) -> dict[str, Any]:
    missing = [field for field in REQUIRED_LEGACY_ROW_METADATA if field not in raw]
    if missing:
        raise Tau3ExposureError(f"line {line_number}: missing metadata.{missing[0]}")
    for token_field in ("prompt_tokens", "supervised_tokens"):
        _required_positive_int(raw, token_field, line_number, prefix="metadata")
    normalized: dict[str, Any] = {
        "prompt_tokens": raw["prompt_tokens"],
        "supervised_tokens": raw["supervised_tokens"],
        "_row_schema_version": LEGACY_EXPOSURE_ROW_SCHEMA_VERSION,
    }
    for field in REQUIRED_LEGACY_ROW_METADATA:
        if field in ("prompt_tokens", "supervised_tokens"):
            continue
        value = raw[field]
        if not isinstance(value, str) or not value:
            raise Tau3ExposureError(f"line {line_number}: metadata.{field} must be a non-empty string")
        normalized[field] = value
    return normalized


def _normalize_target_tool(raw: dict[str, Any], action_class: str, line_number: int) -> str:
    value = raw["target_tool_name"]
    if not isinstance(value, str):
        raise Tau3ExposureError(f"line {line_number}: metadata.target_tool_name must be a string")
    if action_class == ASSISTANT_MESSAGE_TARGET:
        if value != ASSISTANT_MESSAGE_TARGET:
            raise Tau3ExposureError(
                f"line {line_number}: metadata.target_tool_name must be assistant_message for non-tool targets"
            )
        return ASSISTANT_MESSAGE_TARGET
    if not value:
        raise Tau3ExposureError(f"line {line_number}: metadata.target_tool_name must be a non-empty string")
    return value


def _required_string(raw: dict[str, Any], field: str, line_number: int) -> str:
    value = raw[field]
    if not isinstance(value, str) or not value:
        raise Tau3ExposureError(f"line {line_number}: metadata.{field} must be a non-empty string")
    return value


def _required_positive_int(raw: dict[str, Any], field: str, line_number: int, *, prefix: str) -> int:
    value = raw[field]
    if not isinstance(value, int) or value < 1:
        raise Tau3ExposureError(f"line {line_number}: {prefix}.{field} must be a positive integer")
    return value


def _length_bucket(total_tokens: int) -> str:
    if total_tokens <= 512:
        return "short"
    if total_tokens <= 1024:
        return "medium"
    if total_tokens <= 2048:
        return "long"
    return "extra_long"


def normalize_sampler_config(
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int = 1,
    sampler_name: str = "deterministic_stratified_round_robin_v1",
    split: str = "train",
) -> dict[str, Any]:
    if sampler_name != "deterministic_stratified_round_robin_v1":
        raise Tau3ExposureError("unsupported sampler_name")
    for name, value in (
        ("seed", seed),
        ("epochs", epochs),
        ("batch_size", batch_size),
        ("gradient_accumulation_steps", gradient_accumulation_steps),
    ):
        if not isinstance(value, int) or value < 1:
            raise Tau3ExposureError(f"{name} must be a positive integer")
    if not isinstance(split, str) or not split:
        raise Tau3ExposureError("split must be a non-empty string")
    return {
        "sampler_name": sampler_name,
        "split": split,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
    }


def _normalize_receipt_sampler_config(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise Tau3ExposureError("sampler_config must be an object")
    required = {
        "sampler_name",
        "split",
        "seed",
        "epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "effective_batch_size",
    }
    extra = sorted(set(raw) - required)
    missing = sorted(required - set(raw))
    if missing:
        raise Tau3ExposureError(f"sampler_config missing {missing[0]}")
    if extra:
        raise Tau3ExposureError(f"sampler_config has unexpected field {extra[0]}")
    config = normalize_sampler_config(
        sampler_name=raw["sampler_name"],
        split=raw["split"],
        seed=raw["seed"],
        epochs=raw["epochs"],
        batch_size=raw["batch_size"],
        gradient_accumulation_steps=raw["gradient_accumulation_steps"],
    )
    if raw["effective_batch_size"] != config["effective_batch_size"]:
        raise Tau3ExposureError("sampler_config effective_batch_size does not replay")
    return config


def plan_exposures(rows: list[ExposureRow], config: dict[str, Any]) -> dict[str, Any]:
    if not rows:
        raise Tau3ExposureError("cannot plan exposures for an empty dataset")
    dataset_content_sha256 = _dataset_content_sha256(rows)
    config_sha256 = _canonical_sha256(config)
    ordered: list[ExposureRow] = []
    for epoch in range(int(config["epochs"])):
        ordered.extend(_epoch_order(rows, epoch=epoch, dataset_content_sha256=dataset_content_sha256, config=config))

    cumulative = {row.row_hash: 0 for row in rows}
    ledger: list[dict[str, Any]] = []
    microbatch_size = int(config["batch_size"])
    effective_batch_size = int(config["effective_batch_size"])
    for step_index, offset in enumerate(range(0, len(ordered), effective_batch_size)):
        effective_batch = ordered[offset : offset + effective_batch_size]
        for row in effective_batch:
            cumulative[row.row_hash] += 1
        step_rows = [
            {
                "row_index": row.index,
                "row_sha256": row.row_hash,
                "domain": row.metadata["domain"],
                "behavior": row.metadata["behavior"],
                "target_tool": row.metadata["target_tool"],
                "action_class": row.metadata["action_class"],
                "result_class": row.metadata["result_class"],
                "length_bucket": row.metadata["length_bucket"],
                "source_family": row.metadata["source_family"],
                "source_provenance": row.metadata["source_provenance"],
                "prompt_tokens": row.metadata["prompt_tokens"],
                "supervised_tokens": row.metadata["supervised_tokens"],
                "cumulative_exposure": cumulative[row.row_hash],
            }
            for row in effective_batch
        ]
        microbatches = [
            {
                "microbatch_index": microbatch_index,
                "row_hashes": [row["row_sha256"] for row in step_rows[start : start + microbatch_size]],
            }
            for microbatch_index, start in enumerate(range(0, len(step_rows), microbatch_size))
        ]
        ledger.append(
            {
                "schema_version": "hfr.tau3_training_exposure_step.v1",
                "step_index": step_index,
                "epoch_index": offset // len(rows),
                "sampler_name": config["sampler_name"],
                "dataset_content_sha256": dataset_content_sha256,
                "sampler_config_sha256": config_sha256,
                "microbatch_size": microbatch_size,
                "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
                "effective_batch_size": effective_batch_size,
                "effective_batch_row_count": len(step_rows),
                "microbatch_count": len(microbatches),
                "microbatches": microbatches,
                "rows": step_rows,
                "step_sha256": _canonical_sha256(
                    {
                        "step_index": step_index,
                        "dataset_content_sha256": dataset_content_sha256,
                        "sampler_config_sha256": config_sha256,
                        "row_hashes": [row["row_sha256"] for row in step_rows],
                        "microbatches": microbatches,
                    }
                ),
            }
        )
    return {
        "ledger": ledger,
        "coverage": _coverage(rows, ledger, config),
    }


def build_exposure_receipt(
    rows: list[ExposureRow],
    config: dict[str, Any],
    *,
    dataset_path: Path,
    ledger_path: Path,
    ledger: list[dict[str, Any]],
    coverage: dict[str, Any],
) -> dict[str, Any]:
    candidate_checks = _candidate_checks(rows, config, coverage)
    return {
        "schema_version": TAU3_TRAINING_EXPOSURE_SCHEMA_VERSION,
        "artifact_type": "tau3_training_exposure_receipt",
        "passed": all(check["passed"] for check in candidate_checks),
        "dataset": {
            "label": dataset_path.name,
            "row_count": len(rows),
            "content_sha256": _dataset_content_sha256(rows),
            "file_sha256": _sha256_file(dataset_path),
        },
        "sampler_config": config,
        "sampler_config_sha256": _canonical_sha256(config),
        "files": {
            "ledger": {
                "path": ledger_path.name,
                "sha256": _sha256_file(ledger_path),
                "step_count": len(ledger),
            }
        },
        "coverage": coverage,
        "candidate_eligibility": {
            "passed": all(check["passed"] for check in candidate_checks),
            "checks": candidate_checks,
        },
    }


def _epoch_order(
    rows: list[ExposureRow],
    *,
    epoch: int,
    dataset_content_sha256: str,
    config: dict[str, Any],
) -> list[ExposureRow]:
    grouped: dict[tuple[str, ...], list[ExposureRow]] = {}
    for row in rows:
        grouped.setdefault(row.stratum, []).append(row)
    stratum_order = sorted(
        grouped,
        key=lambda stratum: _canonical_sha256(
            {
                "seed": config["seed"],
                "epoch": epoch,
                "dataset_content_sha256": dataset_content_sha256,
                "stratum": stratum,
            }
        ),
    )
    shuffled = {
        stratum: sorted(
            grouped[stratum],
            key=lambda row: _canonical_sha256(
                {
                    "seed": config["seed"],
                    "epoch": epoch,
                    "dataset_content_sha256": dataset_content_sha256,
                    "stratum": stratum,
                    "row_sha256": row.row_hash,
                }
            ),
        )
        for stratum in stratum_order
    }
    output: list[ExposureRow] = []
    while any(shuffled.values()):
        for stratum in stratum_order:
            if shuffled[stratum]:
                output.append(shuffled[stratum].pop(0))
    return output


def _coverage(rows: list[ExposureRow], ledger: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    observed: dict[str, int] = {row.row_hash: 0 for row in rows}
    domain_counts: dict[str, int] = {}
    behavior_counts: dict[str, int] = {}
    target_tools: set[str] = set()
    action_classes: set[str] = set()
    result_classes: set[str] = set()
    length_buckets: set[str] = set()
    source_families: set[str] = set()
    source_provenance: set[str] = set()
    row_schema_versions: set[str] = set()
    total_prompt = 0
    total_supervised = 0
    complete_optimizer_steps = 0
    for step in ledger:
        if (
            step.get("effective_batch_row_count") == config["effective_batch_size"]
            and step.get("microbatch_count") == config["gradient_accumulation_steps"]
        ):
            complete_optimizer_steps += 1
        for item in step["rows"]:
            observed[item["row_sha256"]] += 1
            domain_counts[item["domain"]] = domain_counts.get(item["domain"], 0) + 1
            behavior_counts[item["behavior"]] = behavior_counts.get(item["behavior"], 0) + 1
            target_tools.add(item["target_tool"])
            action_classes.add(item["action_class"])
            result_classes.add(item["result_class"])
            length_buckets.add(item["length_bucket"])
            source_families.add(item["source_family"])
            source_provenance.add(item["source_provenance"])
            row_schema_versions.add(str(rows[item["row_index"]].metadata["_row_schema_version"]))
            total_prompt += int(item["prompt_tokens"])
            total_supervised += int(item["supervised_tokens"])
    min_exposure = min(observed.values())
    max_exposure = max(observed.values())
    row_count = len(rows)
    total_row_exposures = sum(observed.values())
    return {
        "row_count": row_count,
        "total_row_exposures": total_row_exposures,
        "effective_epochs": total_row_exposures / row_count,
        "min_row_exposure": min_exposure,
        "max_row_exposure": max_exposure,
        "all_rows_seen": min_exposure >= 1,
        "full_epoch_replay": total_row_exposures == row_count * int(config["epochs"]),
        "complete_optimizer_steps": complete_optimizer_steps == len(ledger),
        "complete_optimizer_step_count": complete_optimizer_steps,
        "optimizer_step_count": len(ledger),
        "domains": sorted(domain_counts),
        "behaviors": sorted(behavior_counts),
        "domain_exposure_counts": {key: domain_counts[key] for key in sorted(domain_counts)},
        "behavior_exposure_counts": {key: behavior_counts[key] for key in sorted(behavior_counts)},
        "target_tools": sorted(target_tools),
        "action_classes": sorted(action_classes),
        "result_classes": sorted(result_classes),
        "length_buckets": sorted(length_buckets),
        "source_families": sorted(source_families),
        "source_provenance": sorted(source_provenance),
        "row_schema_versions": sorted(row_schema_versions),
        "required_recovery_exposures": sum(behavior_counts.get(behavior, 0) for behavior in RECOVERY_BEHAVIORS),
        "required_stopping_exposures": behavior_counts.get("safe_stopping", 0),
        "required_clarification_exposures": behavior_counts.get("clarification_refusal", 0),
        "required_telecom_exposures": domain_counts.get("telecom", 0),
        "required_state_mutation_exposures": sum(behavior_counts.get(behavior, 0) for behavior in MUTATION_BEHAVIORS),
        "prompt_tokens_exposed": total_prompt,
        "supervised_tokens_exposed": total_supervised,
    }


def _candidate_checks(rows: list[ExposureRow], config: dict[str, Any], coverage: dict[str, Any]) -> list[dict[str, Any]]:
    dataset_domains = sorted({str(row.metadata["domain"]) for row in rows})
    dataset_behaviors = sorted({str(row.metadata["behavior"]) for row in rows})
    row_schema_versions = sorted({str(row.metadata["_row_schema_version"]) for row in rows})
    missing_domains = sorted(set(REQUIRED_DOMAINS) - set(dataset_domains))
    extra_domains = sorted(set(dataset_domains) - set(REQUIRED_DOMAINS))
    missing_behaviors = sorted(set(REQUIRED_BEHAVIORS) - set(dataset_behaviors))
    extra_behaviors = sorted(set(dataset_behaviors) - set(REQUIRED_BEHAVIORS))
    checks = [
        _check("every_row_seen", coverage["all_rows_seen"], True, coverage["min_row_exposure"]),
        _check("at_least_two_effective_epochs", coverage["effective_epochs"] >= 2, ">=2", coverage["effective_epochs"]),
        _check("full_epoch_replay", coverage["full_epoch_replay"], True, coverage["total_row_exposures"]),
        _check(
            "complete_optimizer_steps",
            coverage["complete_optimizer_steps"],
            True,
            {
                "complete_optimizer_step_count": coverage["complete_optimizer_step_count"],
                "optimizer_step_count": coverage["optimizer_step_count"],
            },
        ),
        _check("effective_batch_at_least_four", config["effective_batch_size"] >= 4, ">=4", config["effective_batch_size"]),
        _check(
            "competitive_dataset_row_schema",
            row_schema_versions == [TAU3_COMPETITIVE_ROW_SCHEMA_VERSION],
            [TAU3_COMPETITIVE_ROW_SCHEMA_VERSION],
            row_schema_versions,
        ),
        _check(
            "required_domains_exact",
            not missing_domains and not extra_domains,
            list(REQUIRED_DOMAINS),
            {"present": dataset_domains, "missing": missing_domains, "extra": extra_domains},
        ),
        _check(
            "required_behaviors_exact",
            not missing_behaviors and not extra_behaviors,
            list(REQUIRED_BEHAVIORS),
            {"present": dataset_behaviors, "missing": missing_behaviors, "extra": extra_behaviors},
        ),
        _check("recovery_strata_nonzero", coverage["required_recovery_exposures"] > 0, ">0", coverage["required_recovery_exposures"]),
        _check("stopping_strata_nonzero", coverage["required_stopping_exposures"] > 0, ">0", coverage["required_stopping_exposures"]),
        _check("clarification_strata_nonzero", coverage["required_clarification_exposures"] > 0, ">0", coverage["required_clarification_exposures"]),
        _check("telecom_strata_nonzero", coverage["required_telecom_exposures"] > 0, ">0", coverage["required_telecom_exposures"]),
        _check("state_mutation_strata_nonzero", coverage["required_state_mutation_exposures"] > 0, ">0", coverage["required_state_mutation_exposures"]),
        _check("action_class_strata_nonzero", len(coverage["action_classes"]) >= 1, ">=1", coverage["action_classes"]),
        _check("result_class_strata_nonzero", len(coverage["result_classes"]) >= 1, ">=1", coverage["result_classes"]),
    ]
    return checks


def _check(check_id: str, passed: bool, expected: Any, actual: Any) -> dict[str, Any]:
    return {"id": check_id, "passed": bool(passed), "expected": expected, "actual": actual}


def _dataset_content_sha256(rows: list[ExposureRow]) -> str:
    return _canonical_sha256(
        {
            "row_count": len(rows),
            "row_hashes_in_file_order": [row.row_hash for row in rows],
        }
    )


def _iter_jsonl(path: Path) -> Iterable[tuple[int, Any]]:
    if not path.exists():
        raise Tau3ExposureError(f"dataset does not exist: {path}")
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            yield line_number, json.loads(line)
        except json.JSONDecodeError as exc:
            raise Tau3ExposureError(f"line {line_number}: invalid JSON: {exc.msg}") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, payload in _iter_jsonl(path):
        if not isinstance(payload, dict):
            raise Tau3ExposureError(f"line {line_number}: ledger step must be an object")
        rows.append(payload)
    if not rows:
        raise Tau3ExposureError("ledger must contain at least one step")
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Tau3ExposureError(f"invalid JSON in {path}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise Tau3ExposureError(f"JSON artifact must be an object: {path}")
    return payload


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
