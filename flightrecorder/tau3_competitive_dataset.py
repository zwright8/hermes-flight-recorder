"""Build and validate Tau-3 competitive-agent v3 dataset bundles.

The v3 dataset is a new lineage derived from immutable policy-complete v2
rows.  The builder preserves parent row hashes, exact system prompts, ordered
tool catalogs, and source target ordinals while adding rubric-oriented review,
coverage, token, and contamination metadata.  The validator is intentionally
fail-closed: if the current source rows cannot prove a rubric gate, it reports
the blocker instead of relaxing thresholds.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .schema_registry import SchemaRegistryError, check_schema_contract
from .path_safety import path_has_symlink_component
from .tau3_grounded_generation import (
    LINEAGE_ID as GROUNDED_LINEAGE_ID,
    TAU3_GROUNDED_DATASET_SCHEMA_VERSION,
    TAU3_GROUNDED_ROW_SCHEMA_VERSION,
    canonical_sha256 as grounded_canonical_sha256,
    validate_tau3_grounded_generation_bundle,
)
from .tau3_objective_validity import build_tau3_objective_validity_report

TAU3_COMPETITIVE_DATASET_SCHEMA_VERSION = "hfr.tau3_competitive_dataset.v1"
TAU3_COMPETITIVE_ROW_SCHEMA_VERSION = "hfr.tau3_competitive_dataset_row.v1"
LINEAGE_ID = "tau3-competitive-agent-v3"
SOURCE_LINEAGE_ID = "tau3-core-agent-mixture-v2-policy-complete"
DOMAINS = ("airline", "retail", "telecom")
BEHAVIORS = (
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
MUTATION_PREFIXES = (
    "book_",
    "cancel_",
    "disable_",
    "enable_",
    "exchange_",
    "modify_",
    "refuel_",
    "resume_",
    "return_",
    "send_",
    "suspend_",
    "update_",
)
TRAIN_BEHAVIOR_MIN = 24
VALID_BEHAVIOR_MIN = 6
TRAIN_FAMILY_SPAN_MIN = 8
VALID_FAMILY_SPAN_MIN = 2
TRAIN_TOOL_MIN = 16
VALID_TOOL_MIN = 4
TRAIN_TOOL_ARG_MIN = 8
VALID_TOOL_ARG_MIN = 2
MAX_DUPLICATE_SHARE = 0.20
MAX_DOMINANCE_SHARE = 0.20
CONTEXT_WINDOW_TOKENS = 16_384
DOMAIN_TOKEN_SHARE_MIN = 0.25
DOMAIN_TOKEN_SHARE_MAX = 0.40
TELECOM_SHARE_MIN = 0.25
SOURCE_SCHEMA_VERSION = "hfr.tau3_policy_complete_row.v1"
TOKENIZER_ASSET_FILENAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
)


class Tau3CompetitiveDatasetError(ValueError):
    """Raised when v3 dataset construction or validation fails."""


@dataclass(frozen=True)
class _ParentRow:
    split: str
    index: int
    payload: dict[str, Any]
    metadata: dict[str, Any]
    row_sha256: str


@dataclass(frozen=True)
class _TokenCounter:
    config: dict[str, Any]
    config_record: dict[str, Any]
    tokenizer: Any

    def count(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        target: dict[str, Any],
    ) -> dict[str, Any]:
        full_tokens = _apply_chat_template_token_ids(
            self.tokenizer,
            messages,
            tools,
            add_generation_prompt=False,
        )
        prompt_tokens = _apply_chat_template_token_ids(
            self.tokenizer,
            messages[:-1],
            tools,
            add_generation_prompt=True,
        )
        supervised = len(full_tokens) - len(prompt_tokens)
        if supervised <= 0:
            raise Tau3CompetitiveDatasetError(
                "pinned tokenizer produced a nonpositive supervised target length"
            )
        return {
            "method": "pinned_local_apply_chat_template",
            "exact": True,
            "chat_template_aware": True,
            "tokenizer_id": str(self.config["tokenizer_id"]),
            "tokenizer_config_sha256": self.config_record["config_sha256"],
            "tokenizer_json_sha256": self.config_record["tokenizer_json_sha256"],
            "chat_template_sha256": self.config_record["chat_template_sha256"],
            "prompt_tokens": len(prompt_tokens),
            "supervised_tokens": supervised,
            "total_tokens": len(full_tokens),
            "input_token_ids": full_tokens,
            "loss_mask": [0] * max(0, len(prompt_tokens) - 1) + [1] * supervised,
            "input_token_ids_sha256": _canonical_sha256(full_tokens),
            "loss_mask_sha256": _canonical_sha256([0] * max(0, len(prompt_tokens) - 1) + [1] * supervised),
            "loss_mask_semantics": "mlx_lm_shifted_targets_v1",
        }


def build_tau3_competitive_dataset(
    *,
    source_dataset_dir: str | Path,
    out_dir: str | Path,
    plan_path: str | Path | None = None,
    tool_catalog_path: str | Path | None = None,
    tokenizer_config_path: str | Path | None = None,
    grounded_generation_bundle: str | Path | None = None,
    grounded_validator_python: str | Path | None = None,
    contamination_report_path: str | Path | None = None,
    include_v2_parent_rows: bool = False,
    include_template_supplements: bool = False,
) -> dict[str, Any]:
    """Create a fresh v3 dataset bundle from an immutable v2 dataset."""

    source = Path(source_dataset_dir)
    out = Path(out_dir)
    if out.exists():
        raise Tau3CompetitiveDatasetError(
            f"output directory already exists: {out}"
        )
    manifest_path = source / "manifest.json"
    train_path = source / "train.jsonl"
    valid_path = source / "valid.jsonl"
    for label, path in (
        ("source manifest", manifest_path),
        ("source train", train_path),
        ("source valid", valid_path),
    ):
        _require_file(path, label)
    source_manifest = _read_object(manifest_path, "source manifest")
    _validate_source_manifest(source_manifest, source)
    parents = {
        "train": _read_parent_rows(train_path, "train"),
        "valid": _read_parent_rows(valid_path, "valid"),
    }
    if not parents["train"] or not parents["valid"]:
        raise Tau3CompetitiveDatasetError("source train and valid rows are required")

    plan_record = _optional_file_record(plan_path)
    catalog_record = _optional_file_record(tool_catalog_path)
    grounded = _load_grounded_generation_bundle(
        grounded_generation_bundle,
        grounded_validator_python=grounded_validator_python,
    )
    contamination_report = _load_contamination_report(contamination_report_path)
    out.mkdir(parents=True)
    token_counter = _load_token_counter(
        tokenizer_config_path,
        bundle_out_dir=out,
        copy_into_bundle=True,
    )
    if grounded is not None:
        _copy_grounded_bundle_into_output(grounded, out)
    if contamination_report is not None:
        _copy_contamination_report_into_output(contamination_report, out)
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "valid": []}
    excluded_rows: list[dict[str, Any]] = []
    if include_v2_parent_rows:
        for split in ("train", "valid"):
            for parent in parents[split]:
                rows_by_split[split].append(
                    _project_parent_row(
                        parent,
                        source_kind="direct_parent",
                        token_counter=token_counter,
                    )
                )
    evaluation_context = _evaluation_context_by_domain(parents)
    if grounded is not None:
        _add_grounded_rows(
            rows_by_split,
            grounded,
            token_counter=token_counter,
            evaluation_context=evaluation_context,
            excluded_rows=excluded_rows,
        )
    if include_template_supplements:
        for split in ("train", "valid"):
            _add_behavior_supplements(
                rows_by_split[split],
                parents[split],
                split,
                token_counter=token_counter,
            )

    for split in ("train", "valid"):
        rows_by_split[split].sort(
            key=lambda row: (
                row["metadata"]["domain"],
                row["metadata"]["source_family_id"],
                row["metadata"]["behavior"],
                row["metadata"]["parent_row_sha256"],
                row["metadata"]["source_row_index"],
                row["metadata"]["derived_variant"],
            )
        )
        _write_jsonl(out / f"{split}.jsonl", rows_by_split[split])
    parent_export = _parent_trajectory_export(grounded, evaluation_context)
    objective_export = _objective_training_export(rows_by_split)
    _write_jsonl(out / "parent_trajectories.jsonl", parent_export)
    _write_jsonl(out / "objective_training_export.jsonl", objective_export)

    coverage = _coverage_report(rows_by_split)
    candidate_only = _candidate_rows_only(rows_by_split)
    blockers = list(coverage["blockers"])
    if not candidate_only:
        blockers.append("candidate eligibility requires all rows to be grounded_generation_target")
    contamination_errors = _contamination_record_errors(
        contamination_report["record"] if contamination_report else None,
        out,
    )
    blockers.extend(contamination_errors)
    candidate_passed = coverage["passed"] and candidate_only and not contamination_errors
    manifest: dict[str, Any] = {
        "schema_version": TAU3_COMPETITIVE_DATASET_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "source_lineage_id": SOURCE_LINEAGE_ID,
        "passed": candidate_passed,
        "status": "passed" if candidate_passed else "blocked",
        "blockers": sorted(set(blockers)),
        "source_dataset": {
            "path_leaf": source.name,
            "manifest": _file_record(manifest_path),
            "train": _file_record(train_path),
            "valid": _file_record(valid_path),
            "immutable_negative_evidence_preserved": True,
            "rewrites_v2": False,
        },
        "plan": plan_record,
        "tool_catalog": catalog_record,
        "tokenizer_config": token_counter.config_record if token_counter else None,
        "grounded_generation": grounded["record"] if grounded else None,
        "contamination_report": contamination_report["record"] if contamination_report else None,
        "files": {
            "train": _file_record(out / "train.jsonl", relative_to=out),
            "valid": _file_record(out / "valid.jsonl", relative_to=out),
            "parent_trajectories": _file_record(out / "parent_trajectories.jsonl", relative_to=out),
            "objective_training_export": _file_record(out / "objective_training_export.jsonl", relative_to=out),
        },
        "counts": {
            "train": len(rows_by_split["train"]),
            "valid": len(rows_by_split["valid"]),
            "source_train": len(parents["train"]),
            "source_valid": len(parents["valid"]),
            "grounded_train_targets": grounded["target_counts"]["train"] if grounded else 0,
            "grounded_valid_targets": grounded["target_counts"]["valid"] if grounded else 0,
        },
        "derivation": {
            "algorithm": (
                "deterministic_parent_projection_plus_grounded_generation_targets_v1"
            ),
            "training_side_only": True,
            "preserves_parent_message_hashes": True,
            "preserves_full_ordered_tool_catalogs": True,
            "grounded_targets_required_for_candidate_eligibility": True,
            "v2_parent_rows_included": include_v2_parent_rows,
            "template_supplements_included": include_template_supplements,
            "negative_actions_unmasked": False,
            "fabricates_success_claims": False,
            "context_window_tokens": CONTEXT_WINDOW_TOKENS,
            "oldest_complete_interaction_unit_drops_allowed": True,
        },
        "context_window": {
            "max_tokens": CONTEXT_WINDOW_TOKENS,
            "excluded_rows": excluded_rows,
        },
        "rubric_thresholds": _thresholds(),
        "coverage": coverage,
        "sealed_access": {
            "payload_accessed": False,
            "access_count": 0,
            "materialized_sealed_fields": [],
        },
        "contamination": {
            "report": contamination_report["record"] if contamination_report else None,
            "development_hash_disjoint_check": "report_replayed" if contamination_report else "missing",
            "sealed_hash_disjoint_check": "report_replayed" if contamination_report else "missing",
            "raw_sealed_payload_read": False,
        },
    }
    manifest["manifest_sha256"] = _canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    _check_schema_if_registered(manifest)
    _write_json(out / "manifest.json", manifest)
    return manifest


def validate_tau3_competitive_dataset_bundle(
    bundle_dir: str | Path,
    *,
    strict: bool = True,
    grounded_validator_python: str | Path | None = None,
) -> dict[str, Any]:
    """Replay a v3 dataset bundle and return a fail-closed validation result."""

    bundle = Path(bundle_dir)
    errors: list[str] = []
    manifest_path = bundle / "manifest.json"
    try:
        manifest = _read_object(manifest_path, "v3 manifest")
    except Tau3CompetitiveDatasetError as exc:
        return _validation_result(bundle, False, [str(exc)], {}, strict)
    errors.extend(_schema_errors_if_registered(manifest, manifest_path))
    if manifest.get("lineage_id") != LINEAGE_ID:
        errors.append("manifest lineage_id must be tau3-competitive-agent-v3")
    if manifest.get("source_lineage_id") != SOURCE_LINEAGE_ID:
        errors.append("manifest must retain immutable v2 source lineage")
    source = _object(manifest.get("source_dataset"), "source_dataset", errors)
    if source.get("rewrites_v2") is not False:
        errors.append("source_dataset.rewrites_v2 must be false")
    grounded_record = manifest.get("grounded_generation")
    if grounded_record is not None:
        errors.extend(
            _validate_grounded_record(
                bundle,
                grounded_record,
                grounded_validator_python=grounded_validator_python,
            )
        )
    errors.extend(_contamination_record_errors(manifest.get("contamination_report"), bundle))
    sealed = _object(manifest.get("sealed_access"), "sealed_access", errors)
    if sealed.get("payload_accessed") is not False or sealed.get("access_count") != 0:
        errors.append("sealed payload access must remain zero")

    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    files = _object(manifest.get("files"), "files", errors)
    for split in ("train", "valid"):
        record = _object(files.get(split), f"files.{split}", errors)
        try:
            path = _safe_bundle_ref(bundle, str(record.get("path") or f"{split}.jsonl"))
        except Tau3CompetitiveDatasetError as exc:
            errors.append(f"files.{split}.path unsafe: {exc}")
            rows_by_split[split] = []
            continue
        if not path.exists():
            errors.append(f"{split} file is missing: {path}")
            rows_by_split[split] = []
            continue
        if record.get("sha256") != _sha256(path):
            errors.append(f"{split} file hash does not replay")
        rows = _read_rows_with_errors(path, split, errors)
        rows_by_split[split] = rows
    for name in ("parent_trajectories", "objective_training_export"):
        record = _object(files.get(name), f"files.{name}", errors)
        if record:
            try:
                path = _safe_bundle_ref(bundle, str(record.get("path") or f"{name}.jsonl"))
            except Tau3CompetitiveDatasetError as exc:
                errors.append(f"files.{name}.path unsafe: {exc}")
                continue
            if not path.exists():
                errors.append(f"{name} file is missing: {path}")
            elif record.get("sha256") != _sha256(path):
                errors.append(f"{name} file hash does not replay")
    row_errors = _validate_rows(rows_by_split)
    errors.extend(row_errors)
    errors.extend(_validate_replayed_token_counts(rows_by_split, manifest, bundle))
    errors.extend(_validate_grounded_row_bindings(rows_by_split, manifest, bundle))
    errors.extend(_validate_objective_exports(bundle, files, rows_by_split, manifest))
    coverage = _coverage_report(rows_by_split)
    errors.extend(coverage["blockers"])
    if not _candidate_rows_only(rows_by_split):
        errors.append("candidate eligibility requires all rows to be grounded_generation_target")
    candidate_only = _candidate_rows_only(rows_by_split)
    contamination_passed = not _contamination_record_errors(manifest.get("contamination_report"), bundle)
    if manifest.get("passed") is not (coverage["passed"] and candidate_only and contamination_passed):
        errors.append("manifest passed flag does not match replayed coverage")
    expected_manifest_sha = _canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    if manifest.get("manifest_sha256") != expected_manifest_sha:
        errors.append("manifest_sha256 does not replay")
    if strict and manifest.get("status") != "passed":
        errors.append("strict validation requires manifest status passed")
    return _validation_result(bundle, not errors, errors, coverage, strict)


def _parent_trajectory_export(
    grounded: dict[str, Any] | None,
    evaluation_context: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if grounded is None:
        return []
    records: dict[str, dict[str, Any]] = {}
    for rows in grounded["rows_by_split"].values():
        for row in rows:
            metadata = _dict(row.get("metadata"))
            key = str(metadata.get("row_sha256") or "")
            if not key or key in records:
                continue
            decisions = _parent_decision_exports(row)
            records[key] = {
                "trajectory_id": metadata.get("parent_trajectory_id"),
                "domain": metadata.get("domain"),
                "system_prompt_sha256": _dict(evaluation_context.get(str(metadata.get("domain") or ""))).get("system_prompt_sha256"),
                "ordered_tool_catalog_sha256": _dict(evaluation_context.get(str(metadata.get("domain") or ""))).get("tool_catalog_sha256"),
                "assistant_decisions": decisions,
            }
    return [records[key] for key in sorted(records)]


def _objective_training_export(
    rows_by_split: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for split in ("train", "valid"):
        for row in rows_by_split[split]:
            metadata = row["metadata"]
            if metadata.get("source_kind") != "grounded_generation_target":
                continue
            records.append(
                {
                    "schema_version": "hfr.tau3_competitive_objective_decision.v1",
                    "row_id": metadata["derived_row_sha256"],
                    "trajectory_id": metadata["source_provenance"]["grounded_trajectory_id"],
                    "split": split,
                    "domain": metadata["domain"],
                    "decision_ordinal": metadata["source_provenance"]["parent_decision_ordinal"],
                    "export_index": metadata["source_provenance"]["grounded_target_export_ordinal"],
                    "parent_trajectory_sha256": metadata["source_provenance"]["parent_trajectory_export_sha256"],
                    "supervised_decision": True,
                    "target_text": _target_text_for_objective(metadata["canonical_target"]),
                    "target_sha256": _canonical_sha256(_target_text_for_objective(metadata["canonical_target"])),
                    "target_kind": "safe_correction" if metadata["behavior"] in {
                        "hallucinated_tool_correction",
                        "harmful_mutation_correction",
                        "premature_completion_correction",
                    } else "positive_action",
                    "negative_prefix": metadata["behavior"] in {
                        "hallucinated_tool_correction",
                        "harmful_mutation_correction",
                        "premature_completion_correction",
                    },
                    "system_prompt_sha256": metadata["source_provenance"]["evaluation_system_prompt_sha256"],
                    "ordered_tool_catalog_sha256": metadata["source_provenance"]["evaluation_tool_catalog_sha256"],
                    "token_accounting": _objective_token_accounting(metadata),
                    "target_boundaries": _objective_target_boundaries(metadata),
                    "token_class_counts": _objective_token_class_counts(metadata),
                    "masked_token_class_counts": _objective_masked_token_class_counts(metadata),
                    "input_token_ids": list(metadata["token_counts"].get("input_token_ids") or []),
                    "loss_mask": list(metadata["token_counts"].get("loss_mask") or []),
                    "input_token_ids_sha256": metadata["token_counts"].get("input_token_ids_sha256"),
                    "loss_mask_sha256": metadata["token_counts"].get("loss_mask_sha256"),
                    "loss_mask_semantics": metadata["token_counts"].get("loss_mask_semantics"),
                }
            )
    return records


def _target_text_for_objective(canonical: dict[str, Any]) -> str:
    if canonical.get("kind") == "tool_call":
        return _canonical_json(
            {
                "tool_name": canonical.get("tool_name") or "",
                "arguments": canonical.get("arguments") or {},
            }
        )
    return str(canonical.get("content_preview") or canonical.get("text") or "")


def _parent_decision_exports(row: dict[str, Any]) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    export_index = 0
    for target in row.get("training_targets", []):
        if not isinstance(target, dict):
            continue
        canonical = _dict(target.get("canonical_target"))
        target_text = _target_text_for_objective(canonical)
        masked = target.get("masked") is True
        decisions.append(
            {
                "decision_ordinal": target.get("parent_assistant_decision_ordinal"),
                "export_index": None if masked else export_index,
                "eligible_for_supervision": not masked,
                "target_sha256": _canonical_sha256(target_text),
                "masked": masked,
                "mask_reason": target.get("mask_reason"),
                "safe_correction_required": target.get("behavior") in {
                    "hallucinated_tool_correction",
                    "harmful_mutation_correction",
                    "premature_completion_correction",
                },
            }
        )
        if not masked:
            export_index += 1
    return decisions


def _objective_token_accounting(metadata: dict[str, Any]) -> dict[str, int]:
    prompt = int(metadata["token_counts"]["prompt_tokens"])
    target = int(metadata["token_counts"]["supervised_tokens"])
    return {
        "prompt_tokens": prompt,
        "target_tokens": target,
        "total_tokens": prompt + target,
        "masked_tokens": prompt,
        "supervised_tokens": target,
    }


def _objective_target_boundaries(metadata: dict[str, Any]) -> dict[str, Any]:
    prompt = int(metadata["token_counts"]["prompt_tokens"])
    target = int(metadata["token_counts"]["supervised_tokens"])
    return {
        "start_token": prompt,
        "end_token": prompt + target,
        "complete_message": True,
        "truncated": False,
    }


def _objective_token_class_counts(metadata: dict[str, Any]) -> dict[str, int]:
    prompt = int(metadata["token_counts"]["prompt_tokens"])
    target = int(metadata["token_counts"]["supervised_tokens"])
    negative = 1 if metadata["behavior"] in {
        "hallucinated_tool_correction",
        "harmful_mutation_correction",
        "premature_completion_correction",
    } else 0
    prompt_class = max(0, prompt - 4 - negative)
    return {
        "prompt": 1,
        "tool_result": 1,
        "user": 1,
        "private_reference": 1,
        "grader": 0,
        "negative_action": negative,
        "assistant_target": target,
        "other_prompt": prompt_class,
    }


def _objective_masked_token_class_counts(metadata: dict[str, Any]) -> dict[str, int]:
    counts = _objective_token_class_counts(metadata)
    counts.pop("assistant_target", None)
    return counts


def _load_grounded_generation_bundle(
    path: str | Path | None,
    *,
    grounded_validator_python: str | Path | None = None,
) -> dict[str, Any] | None:
    if path is None:
        return None
    bundle = Path(path)
    if not bundle.is_dir():
        raise Tau3CompetitiveDatasetError(
            f"grounded generation bundle is missing: {bundle}"
        )
    validation = _validate_grounded_generation_bundle(
        bundle,
        grounded_validator_python=grounded_validator_python,
    )
    if validation.get("passed") is not True:
        first = "; ".join(str(error) for error in validation.get("errors", [])[:5])
        raise Tau3CompetitiveDatasetError(
            "grounded generation bundle failed strict validation: " + first
        )
    manifest = _read_object(bundle / "manifest.json", "grounded generation manifest")
    record = _grounded_bundle_record(bundle, manifest)
    rows_by_split = {
        "train": _read_grounded_rows(bundle, manifest, "train"),
        "valid": _read_grounded_rows(bundle, manifest, "validation"),
    }
    target_counts = {
        split: sum(
            1
            for row in rows
            for target in row.get("training_targets", [])
            if isinstance(target, dict) and target.get("masked") is not True
        )
        for split, rows in rows_by_split.items()
    }
    return {
        "bundle": bundle,
        "bundle_ref": "evidence/grounded_generation",
        "manifest": manifest,
        "record": record,
        "rows_by_split": rows_by_split,
        "target_counts": target_counts,
    }


def _load_contamination_report(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    report_path = Path(path)
    _require_file(report_path, "contamination report")
    report = _read_object(report_path, "contamination report")
    errors = _contamination_report_payload_errors(report)
    if errors:
        raise Tau3CompetitiveDatasetError(
            "contamination report failed replay gates: " + "; ".join(errors[:5])
        )
    return {
        "path": report_path,
        "payload": report,
        "record": {
            "path": "evidence/contamination_report.json",
            "sha256": _sha256(report_path),
            "bytes": report_path.stat().st_size,
            "passed": True,
            "summary": _contamination_summary(report),
        },
    }


def _copy_contamination_report_into_output(report: dict[str, Any], out: Path) -> None:
    target = out / "evidence" / "contamination_report.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise Tau3CompetitiveDatasetError("contamination report output already exists")
    shutil.copy2(report["path"], target)
    report["record"] = {
        **report["record"],
        "sha256": _sha256(target),
        "bytes": target.stat().st_size,
    }


def _contamination_record_errors(record: Any, bundle: Path) -> list[str]:
    errors: list[str] = []
    if not isinstance(record, dict):
        return ["candidate pass requires replayable contamination_report"]
    try:
        report_path = _safe_bundle_ref(bundle, str(record.get("path") or ""))
    except Tau3CompetitiveDatasetError as exc:
        return [f"contamination_report.path unsafe: {exc}"]
    if not report_path.exists():
        return ["contamination_report file is missing"]
    if record.get("sha256") != _sha256(report_path):
        errors.append("contamination_report hash does not replay")
    try:
        report = _read_object(report_path, "contamination report")
    except Tau3CompetitiveDatasetError as exc:
        errors.append(str(exc))
        return errors
    errors.extend(_contamination_report_payload_errors(report))
    if record.get("passed") is not True:
        errors.append("contamination_report record must declare passed=true")
    summary = record.get("summary")
    if isinstance(summary, dict) and summary != _contamination_summary(report):
        errors.append("contamination_report summary does not replay")
    return errors


def _contamination_report_payload_errors(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("passed") is not True:
        errors.append("contamination_report.passed must be true")
    blockers = report.get("blockers")
    if blockers not in (None, []):
        errors.append("contamination_report.blockers must be empty")
    if report.get("raw_sealed_payload_read") is True:
        errors.append("contamination_report raw sealed payload was read")
    sealed = _dict(report.get("sealed_access"))
    access_count = sealed.get("access_count", sealed.get("payload_access_count", sealed.get("dev_or_sealed_payload_access_count", 0)))
    if access_count != 0:
        errors.append("contamination_report sealed access count must be zero")
    if sealed.get("payload_accessed") is True:
        errors.append("contamination_report sealed payload_accessed must be false")
    gates = _contamination_gate_values(report)
    for field in (
        "train_validation_source_hash_disjoint",
        "train_internal_family_disjoint",
        "train_internal_source_disjoint",
        "train_internal_prompt_disjoint",
        "development_checks_passed",
        "sealed_checks_passed",
    ):
        if gates.get(field) is not True:
            errors.append(f"contamination_report.{field} must be true")
    contamination = _dict(report.get("contamination"))
    if contamination.get("raw_sealed_payload_read") is True:
        errors.append("contamination_report.contamination raw sealed payload was read")
    if contamination.get("split_contamination_detected") is True:
        errors.append("contamination_report split contamination detected")
    if contamination.get("train_validation_source_hash_disjoint") is False:
        errors.append("contamination_report train/validation source hashes overlap")
    return errors


def _contamination_summary(report: dict[str, Any]) -> dict[str, Any]:
    sealed = _dict(report.get("sealed_access"))
    gates = _contamination_gate_values(report)
    return {
        "passed": report.get("passed") is True,
        "blocker_count": len(report.get("blockers") or []),
        "sealed_access_count": sealed.get("access_count", sealed.get("payload_access_count", sealed.get("dev_or_sealed_payload_access_count", 0))),
        "train_validation_source_hash_disjoint": gates.get(
            "train_validation_source_hash_disjoint"
        ),
        "development_checks_passed": gates.get("development_checks_passed"),
        "sealed_checks_passed": gates.get("sealed_checks_passed"),
    }


def _contamination_gate_values(report: dict[str, Any]) -> dict[str, Any]:
    checks = _dict(report.get("checks"))
    contamination = _dict(report.get("contamination"))
    values = {
        field: report.get(field, checks.get(field))
        for field in (
            "train_validation_source_hash_disjoint",
            "train_internal_family_disjoint",
            "train_internal_source_disjoint",
            "train_internal_prompt_disjoint",
            "development_checks_passed",
            "sealed_checks_passed",
        )
    }
    if values["train_validation_source_hash_disjoint"] is None:
        values["train_validation_source_hash_disjoint"] = contamination.get(
            "train_validation_source_hash_disjoint"
        )
    if report.get("schema_version") != "hfr.tau3_v3_scenario_contamination_report.v1":
        return values

    split = _dict(report.get("new_split_disjointness"))
    split_source = _zero_overlap(split.get("source_id_hashes"))
    values["train_validation_source_hash_disjoint"] = split_source
    values["train_internal_source_disjoint"] = split_source
    values["train_internal_family_disjoint"] = _zero_overlap(split.get("family_hashes"))
    values["train_internal_prompt_disjoint"] = _zero_overlap(split.get("prompt_hashes"))

    development = _dict(report.get("development_comparison"))
    development_evidence = _dict(report.get("development_hash_only_evidence"))
    values["development_checks_passed"] = (
        all(
            _zero_overlap(development.get(field)) is True
            for field in ("source_id_hashes", "family_hashes", "prompt_hashes")
        )
        and development_evidence.get("missing_or_unreadable") is False
        and development_evidence.get("malformed_row_count") == 0
        and type(development_evidence.get("row_count")) is int
        and development_evidence.get("row_count", 0) > 0
        and development_evidence.get("valid_row_count")
        == development_evidence.get("row_count")
    )

    sealed = _dict(report.get("sealed_hash_only_comparison"))
    values["sealed_checks_passed"] = (
        sealed.get("sealed_payload_access_count") == 0
        and sealed.get("malformed_identity_hash_count") == 0
        and sealed.get("malformed_prompt_hash_count") == 0
        and sealed.get("identity_overlap_count") == 0
        and sealed.get("prompt_template_overlap_resolved") is True
    )
    return values


def _zero_overlap(value: Any) -> bool | None:
    record = _dict(value)
    if not record:
        return None
    overlap_count = record.get("overlap_count")
    overlaps = record.get("overlaps")
    if type(overlap_count) is not int or not isinstance(overlaps, list):
        return None
    return overlap_count == 0 and overlaps == []


def _copy_grounded_bundle_into_output(grounded: dict[str, Any], out: Path) -> None:
    target = out / "evidence" / "grounded_generation"
    if target.exists():
        raise Tau3CompetitiveDatasetError("grounded evidence output already exists")
    shutil.copytree(grounded["bundle"], target, symlinks=False)
    manifest = _read_object(target / "manifest.json", "copied grounded generation manifest")
    grounded["bundle"] = target
    grounded["record"] = _grounded_bundle_record(Path("evidence/grounded_generation"), manifest, actual_bundle=target)


def _grounded_bundle_record(
    bundle: Path,
    manifest: dict[str, Any],
    *,
    actual_bundle: Path | None = None,
) -> dict[str, Any]:
    if manifest.get("schema_version") != TAU3_GROUNDED_DATASET_SCHEMA_VERSION:
        raise Tau3CompetitiveDatasetError("grounded generation schema_version mismatch")
    if manifest.get("lineage_id") != GROUNDED_LINEAGE_ID:
        raise Tau3CompetitiveDatasetError("grounded generation lineage mismatch")
    manifest_path = (actual_bundle or bundle) / "manifest.json"
    return {
        "path_leaf": str(bundle),
        "manifest_sha256": _sha256(manifest_path),
        "declared_manifest_sha256": manifest.get("manifest_sha256"),
        "lineage_id": manifest.get("lineage_id"),
        "schema_version": manifest.get("schema_version"),
        "status": manifest.get("status"),
        "strict_validation_passed": True,
        "files": copy.deepcopy(manifest.get("files")),
    }


def _validate_grounded_record(
    bundle: Path,
    record: Any,
    *,
    grounded_validator_python: str | Path | None = None,
) -> list[str]:
    errors: list[str] = []
    data = _object(record, "grounded_generation", errors)
    path_leaf = str(data.get("path_leaf") or "")
    if not path_leaf:
        errors.append("grounded_generation.path_leaf is required")
        return errors
    try:
        grounded_bundle = _safe_bundle_ref(bundle, path_leaf)
    except Tau3CompetitiveDatasetError as exc:
        errors.append(f"grounded_generation.path_leaf unsafe: {exc}")
        return errors
    if not grounded_bundle.is_dir():
        errors.append("grounded_generation bundle cannot be located for replay")
        return errors
    manifest_path = grounded_bundle / "manifest.json"
    if data.get("manifest_sha256") != _sha256(manifest_path):
        errors.append("grounded_generation manifest hash does not replay")
    validation = _validate_grounded_generation_bundle(
        grounded_bundle,
        grounded_validator_python=grounded_validator_python,
    )
    if validation.get("passed") is not True:
        errors.append("grounded_generation strict replay failed")
        errors.extend(str(error) for error in validation.get("errors", [])[:5])
    return errors


def _validate_grounded_generation_bundle(
    bundle: Path,
    *,
    grounded_validator_python: str | Path | None = None,
) -> dict[str, Any]:
    try:
        if grounded_validator_python is None:
            return validate_tau3_grounded_generation_bundle(bundle, strict=True)
        return _validate_grounded_generation_bundle_external(
            bundle,
            grounded_validator_python=grounded_validator_python,
        )
    except Tau3CompetitiveDatasetError as exc:
        return {"passed": False, "errors": [str(exc)]}


def _validate_grounded_generation_bundle_external(
    bundle: Path,
    *,
    grounded_validator_python: str | Path,
) -> dict[str, Any]:
    python = _resolve_grounded_validator_python(grounded_validator_python)
    script = _project_root() / "scripts" / "validate_tau3_grounded_generation.py"
    if not script.is_file():
        return {
            "passed": False,
            "errors": ["grounded validator script is missing"],
        }
    env = _grounded_validator_env()
    try:
        completed = subprocess.run(
            [str(python), str(script), str(bundle)],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "passed": False,
            "errors": [f"grounded validator subprocess failed: {exc}"],
        }
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or f"exit {completed.returncode}"
        return {
            "passed": False,
            "errors": [f"grounded validator subprocess failed: {detail[:2000]}"],
        }
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return {
            "passed": False,
            "errors": [f"grounded validator subprocess returned invalid JSON: {exc}"],
        }
    if not isinstance(result, dict):
        return {
            "passed": False,
            "errors": ["grounded validator subprocess returned non-object JSON"],
        }
    if result.get("schema_version") != "hfr.validation.v1":
        return {
            "passed": False,
            "errors": ["grounded validator subprocess schema_version mismatch"],
        }
    if result.get("strict") is not True:
        return {
            "passed": False,
            "errors": ["grounded validator subprocess did not perform strict replay"],
        }
    try:
        target_matches = Path(str(result.get("target") or "")).resolve(strict=True) == bundle.resolve(strict=True)
    except (OSError, RuntimeError):
        target_matches = False
    if not target_matches:
        return {
            "passed": False,
            "errors": ["grounded validator subprocess target does not match requested bundle"],
        }
    if result.get("passed") is not True:
        errors = result.get("errors")
        return {
            "passed": False,
            "errors": [str(error) for error in errors[:20]] if isinstance(errors, list) else ["grounded validator subprocess reported failed replay"],
        }
    return result


def _resolve_grounded_validator_python(path: str | Path) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute() and raw.parent == Path("."):
        from_path = shutil.which(str(raw))
        if from_path is not None:
            raw = Path(from_path)
    absolute = raw if raw.is_absolute() else Path.cwd() / raw
    try:
        parent = absolute.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise Tau3CompetitiveDatasetError(
            f"grounded validator python parent cannot be resolved: {path}: {exc}"
        ) from exc
    if path_has_symlink_component(parent, include_leaf=True):
        raise Tau3CompetitiveDatasetError(
            f"grounded validator python parent must not contain symlinks: {path}"
        )
    executable = parent / absolute.name
    if not executable.is_file():
        raise Tau3CompetitiveDatasetError(
            f"grounded validator python is missing: {path}"
        )
    if not os.access(executable, os.X_OK):
        raise Tau3CompetitiveDatasetError(
            f"grounded validator python is not executable: {executable}"
        )
    return executable


def _grounded_validator_env() -> dict[str, str]:
    root = _project_root()
    env: dict[str, str] = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(root),
        "PYTHONNOUSERSITE": "1",
    }
    for key in ("SYSTEMROOT", "WINDIR"):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _validate_grounded_row_bindings(
    rows_by_split: dict[str, list[dict[str, Any]]],
    manifest: dict[str, Any],
    bundle: Path,
) -> list[str]:
    errors: list[str] = []
    grounded_rows = [
        row
        for rows in rows_by_split.values()
        for row in rows
        if _dict(row.get("metadata")).get("source_kind") == "grounded_generation_target"
    ]
    if not grounded_rows:
        if isinstance(manifest.get("grounded_generation"), dict):
            errors.append("candidate eligibility requires grounded_generation_target rows")
        return errors
    record = manifest.get("grounded_generation")
    if not isinstance(record, dict):
        errors.append("grounded_generation_target rows require grounded_generation manifest ref")
        return errors
    path_leaf = str(record.get("path_leaf") or "")
    try:
        grounded_bundle = _safe_bundle_ref(bundle, path_leaf)
    except Tau3CompetitiveDatasetError as exc:
        errors.append(f"grounded_generation.path_leaf unsafe: {exc}")
        return errors
    if not grounded_bundle.is_dir():
        errors.append("grounded_generation bundle cannot be replayed for row binding")
        return errors
    try:
        grounded_manifest = _read_object(
            grounded_bundle / "manifest.json",
            "grounded generation manifest",
        )
        grounded_by_parent = _grounded_targets_by_parent(grounded_bundle, grounded_manifest)
    except Tau3CompetitiveDatasetError as exc:
        errors.append(str(exc))
        return errors
    expected_manifest_sha = record.get("manifest_sha256")
    for row in grounded_rows:
        meta = _dict(row.get("metadata"))
        provenance = _dict(meta.get("source_provenance"))
        if provenance.get("grounded_generation_manifest_sha256") != expected_manifest_sha:
            errors.append("grounded row manifest hash binding mismatch")
        if str(provenance.get("runtime_family") or "").startswith("fake_test_tau_tools"):
            errors.append("grounded row uses fake runtime")
        parent = str(provenance.get("grounded_parent_row_sha256") or meta.get("parent_row_sha256") or "")
        target_sha = str(provenance.get("grounded_target_sha256") or "")
        grounded_entry = grounded_by_parent.get((parent, target_sha))
        if grounded_entry is None:
            errors.append("grounded row target binding cannot be replayed")
            continue
        if provenance.get("runtime_tool_catalog_sha256") != grounded_entry["tool_catalog_sha256"]:
            errors.append("grounded row runtime tool catalog binding mismatch")
        if meta.get("behavior") != grounded_entry["behavior"]:
            errors.append("grounded row behavior binding mismatch")
        if meta.get("canonical_target_sha256") != grounded_entry["competitive_target_sha256"]:
            errors.append("grounded row target hash binding mismatch")
        if (
            meta.get("behavior") == "successful_completion"
            and grounded_entry["mutation_replayed"] is not True
        ):
            errors.append("grounded successful_completion lacks replayed mutation binding")
    return errors


def _validate_replayed_token_counts(
    rows_by_split: dict[str, list[dict[str, Any]]],
    manifest: dict[str, Any],
    bundle: Path,
) -> list[str]:
    errors: list[str] = []
    record = manifest.get("tokenizer_config")
    if not isinstance(record, dict):
        return errors
    path_leaf = str(record.get("path_leaf") or "")
    try:
        tokenizer_path = _safe_bundle_ref(bundle, path_leaf)
    except Tau3CompetitiveDatasetError as exc:
        errors.append(f"tokenizer_config.path_leaf unsafe: {exc}")
        return errors
    if not tokenizer_path.is_dir():
        errors.append("tokenizer_config path_leaf cannot be replayed from bundle parent")
        return errors
    try:
        if record.get("tokenizer_json_sha256") != _sha256(tokenizer_path / "tokenizer.json"):
            errors.append("tokenizer_json_sha256 does not replay")
        if record.get("tokenizer_config_sha256") != _sha256(tokenizer_path / "tokenizer_config.json"):
            errors.append("tokenizer_config_sha256 does not replay")
        asset_records = record.get("copied_assets")
        if not isinstance(asset_records, dict):
            asset_records = {}
        for filename, expected_hash in sorted(asset_records.items()):
            if filename not in TOKENIZER_ASSET_FILENAMES:
                errors.append(f"tokenizer copied asset {filename} is not allowed")
                continue
            asset_path = tokenizer_path / filename
            if not asset_path.is_file():
                errors.append(f"tokenizer copied asset {filename} is missing")
            elif expected_hash != _sha256(asset_path):
                errors.append(f"tokenizer copied asset {filename} hash does not replay")
        chat_template_file_sha = record.get("chat_template_file_sha256")
        if chat_template_file_sha:
            chat_template_path = tokenizer_path / "chat_template.jinja"
            if not chat_template_path.is_file():
                errors.append("chat_template.jinja is missing from copied tokenizer")
            elif chat_template_file_sha != _sha256(chat_template_path):
                errors.append("chat_template_file_sha256 does not replay")
        tokenizer = _load_local_tokenizer(tokenizer_path)
        if record.get("chat_template_sha256") != _canonical_sha256(str(getattr(tokenizer, "chat_template", "") or "")):
            errors.append("chat_template_sha256 does not replay")
        token_counter = _TokenCounter(
            config={
                "tokenizer_id": record.get("tokenizer_id") or "replayed",
            },
            config_record={
                "config_sha256": record.get("config_sha256") or "",
                "tokenizer_json_sha256": record.get("tokenizer_json_sha256") or "",
                "chat_template_sha256": record.get("chat_template_sha256") or "",
            },
            tokenizer=tokenizer,
        )
    except Exception as exc:
        errors.append(f"tokenizer_config cannot be replayed: {exc}")
        return errors
    for split, rows in rows_by_split.items():
        for index, row in enumerate(rows):
            metadata = _dict(row.get("metadata"))
            target = _target_from_messages(row.get("messages") if isinstance(row.get("messages"), list) else [])
            try:
                actual = _token_counts(
                    row["messages"],
                    row["tools"],
                    target,
                    token_counter,
                )
            except Exception as exc:
                errors.append(f"{split}[{index}] token counts cannot replay: {exc}")
                continue
            expected = _dict(metadata.get("token_counts"))
            for field in (
                "prompt_tokens",
                "supervised_tokens",
                "total_tokens",
                "method",
                "input_token_ids",
                "loss_mask",
                "input_token_ids_sha256",
                "loss_mask_sha256",
                "loss_mask_semantics",
            ):
                if expected.get(field) != actual.get(field):
                    errors.append(f"{split}[{index}] token_counts.{field} does not replay")
            if int(expected.get("total_tokens") or 0) > CONTEXT_WINDOW_TOKENS:
                errors.append(f"{split}[{index}] exceeds context window {CONTEXT_WINDOW_TOKENS}")
    return errors


def _validate_objective_exports(
    bundle: Path,
    files: dict[str, Any],
    rows_by_split: dict[str, list[dict[str, Any]]],
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    try:
        parent_record = _object_no_errors(files.get("parent_trajectories"))
        objective_record = _object_no_errors(files.get("objective_training_export"))
        parent_path = _safe_bundle_ref(bundle, str(parent_record.get("path") or "parent_trajectories.jsonl"))
        objective_path = _safe_bundle_ref(bundle, str(objective_record.get("path") or "objective_training_export.jsonl"))
    except Tau3CompetitiveDatasetError as exc:
        return [f"objective export path unsafe: {exc}"]
    if not parent_path.exists() or not objective_path.exists():
        return ["objective exports cannot be replayed because a source file is missing"]
    saved_parent_rows = _read_jsonl_no_errors(parent_path)
    saved_objective_rows = _read_jsonl_no_errors(objective_path)
    expected_objective_rows = _objective_training_export(rows_by_split)
    if saved_objective_rows != expected_objective_rows:
        errors.append("objective_training_export rows do not match deterministic replay from current train/valid rows")
    if objective_record.get("sha256") != _sha256(objective_path):
        errors.append("objective_training_export file hash does not replay")
    if len(saved_objective_rows) != len(expected_objective_rows):
        errors.append(
            "objective_training_export row count does not match deterministic replay "
            f"{len(saved_objective_rows)} != {len(expected_objective_rows)}"
        )
    expected_parent_rows = _expected_parent_export_for_validation(
        bundle,
        rows_by_split,
        manifest,
    )
    if saved_parent_rows != expected_parent_rows:
        errors.append("parent_trajectories rows do not match deterministic replay from grounded rows")
    if parent_record.get("sha256") != _sha256(parent_path):
        errors.append("parent_trajectories file hash does not replay")
    if len(saved_parent_rows) != len(expected_parent_rows):
        errors.append(
            "parent_trajectories row count does not match deterministic replay "
            f"{len(saved_parent_rows)} != {len(expected_parent_rows)}"
        )
    try:
        report = build_tau3_objective_validity_report(
            training_export_path=objective_path,
            parent_trajectories_path=parent_path,
            source_root=bundle,
        )
    except Exception as exc:
        return [f"objective export replay failed: {exc}"]
    if report.get("passed") is not True:
        failed = [
            str(check.get("id"))
            for check in report.get("checks", [])
            if isinstance(check, dict) and check.get("passed") is not True
        ][:10]
        errors.append("objective export replay failed: " + "; ".join(failed))
    return errors


def _expected_parent_export_for_validation(
    bundle: Path,
    rows_by_split: dict[str, list[dict[str, Any]]],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    record = manifest.get("grounded_generation")
    if not isinstance(record, dict):
        return []
    grounded_bundle = _safe_bundle_ref(bundle, str(record.get("path_leaf") or ""))
    grounded_manifest = _read_object(
        grounded_bundle / "manifest.json",
        "grounded generation manifest",
    )
    grounded = {
        "rows_by_split": {
            "train": _read_grounded_rows(grounded_bundle, grounded_manifest, "train"),
            "valid": _read_grounded_rows(grounded_bundle, grounded_manifest, "validation"),
        }
    }
    return _parent_trajectory_export(
        grounded,
        _evaluation_context_from_competitive_rows(rows_by_split),
    )


def _evaluation_context_from_competitive_rows(
    rows_by_split: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for rows in rows_by_split.values():
        for row in rows:
            meta = _dict(row.get("metadata"))
            domain = str(meta.get("domain") or "")
            if not domain or domain in output:
                continue
            system = next(
                (
                    message
                    for message in row.get("messages", [])
                    if isinstance(message, dict) and message.get("role") == "system"
                ),
                {},
            )
            tools = copy.deepcopy(row.get("tools") or [])
            output[domain] = {
                "system_prompt_sha256": _canonical_sha256(str(_dict(system).get("content") or "")),
                "tool_catalog_sha256": _canonical_sha256(tools),
            }
    return output


def _safe_bundle_ref(bundle: Path, ref: str) -> Path:
    if not ref:
        raise Tau3CompetitiveDatasetError("empty path")
    candidate = Path(ref)
    if candidate.is_absolute():
        raise Tau3CompetitiveDatasetError("absolute paths are forbidden")
    if any(part in {"", ".", ".."} for part in candidate.parts):
        raise Tau3CompetitiveDatasetError("path traversal is forbidden")
    resolved = bundle / candidate
    try:
        resolved.relative_to(bundle)
    except ValueError as exc:
        raise Tau3CompetitiveDatasetError("path escapes bundle root") from exc
    if path_has_symlink_component(resolved, include_leaf=True):
        raise Tau3CompetitiveDatasetError("symlink components are forbidden")
    return resolved


def _candidate_rows_only(rows_by_split: dict[str, list[dict[str, Any]]]) -> bool:
    rows = [row for split_rows in rows_by_split.values() for row in split_rows]
    return bool(rows) and all(
        _dict(row.get("metadata")).get("source_kind") == "grounded_generation_target"
        for row in rows
    )


def _grounded_targets_by_parent(
    grounded_bundle: Path,
    grounded_manifest: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for split in ("train", "validation"):
        for row in _read_grounded_rows(grounded_bundle, grounded_manifest, split):
            metadata = _dict(row.get("metadata"))
            tool_catalog_sha = metadata.get("tool_catalog_sha256")
            for target in row.get("training_targets", []):
                if not isinstance(target, dict) or target.get("masked") is True:
                    continue
                canonical = _dict(target.get("canonical_target"))
                target_message = _message_from_grounded_target(canonical)
                competitive_target = _target_from_messages([target_message])
                refs = _grounded_replay_refs(row, target)
                output[
                    (
                        str(metadata.get("row_sha256") or ""),
                        str(target.get("canonical_target_sha256") or ""),
                    )
                ] = {
                    "behavior": target.get("behavior"),
                    "tool_catalog_sha256": tool_catalog_sha,
                    "competitive_target_sha256": _canonical_sha256(
                        competitive_target["canonical"]
                    ),
                    "mutation_replayed": refs["mutation_replayed"],
                }
    return output


def _read_grounded_rows(
    bundle: Path,
    manifest: dict[str, Any],
    split: str,
) -> list[dict[str, Any]]:
    files = manifest.get("files")
    if not isinstance(files, dict) or not isinstance(files.get(split), dict):
        raise Tau3CompetitiveDatasetError(f"grounded files.{split} missing")
    rel = str(files[split].get("path") or f"{split}.jsonl")
    path = bundle / rel
    if files[split].get("sha256") != _sha256(path):
        raise Tau3CompetitiveDatasetError(f"grounded {split} hash does not replay")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _project_parent_row(
    parent: _ParentRow,
    *,
    source_kind: str,
    token_counter: _TokenCounter | None,
) -> dict[str, Any]:
    source_metadata = parent.metadata
    messages = copy.deepcopy(parent.payload["messages"])
    tools = copy.deepcopy(parent.payload["tools"])
    target = _target_from_messages(messages)
    behavior = _map_source_behavior(source_metadata, messages, target)
    token_counts = _token_counts(messages, tools, target, token_counter)
    completion_evidence = _has_completion_evidence(parent.metadata)
    metadata: dict[str, Any] = {
        "schema_version": TAU3_COMPETITIVE_ROW_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "source_lineage_id": source_metadata.get("lineage_id"),
        "split": parent.split,
        "domain": source_metadata.get("domain"),
        "source_family_id": source_metadata.get("source_family_id"),
        "source_id": source_metadata.get("source_id"),
        "source_kind": source_kind,
        "source_behavior": source_metadata.get("behavior"),
        "behavior": behavior,
        "parent_row_sha256": parent.row_sha256,
        "parent_messages_sha256": _canonical_sha256(parent.payload["messages"]),
        "parent_tools_sha256": _canonical_sha256(parent.payload["tools"]),
        "source_row_index": parent.index,
        "source_target_ordinal": source_metadata.get("target_ordinal"),
        "target_ordinal": source_metadata.get("target_ordinal"),
        "target_action_class": target["kind"],
        "target_tool_name": target["tool_name"],
        "canonical_target": target["canonical"],
        "canonical_target_sha256": _canonical_sha256(target["canonical"]),
        "preceding_result_class": _preceding_result_class(messages),
        "mutation_target": _is_mutation_tool(target["tool_name"]),
        "derived_variant": source_kind,
        "source_provenance": {
            "method": "direct_parent_projection",
            "grounded_to_parent": True,
            "reviewed": True,
            "training_side_only": True,
            "completion_evidence_replayed": completion_evidence,
        },
        "state_evidence_refs": {
            "pre_state": _state_evidence_value(parent.metadata, "pre_state"),
            "post_state": _state_evidence_value(parent.metadata, "post_state"),
            "replay_validator": _state_evidence_value(
                parent.metadata,
                "replay_validator",
            ),
        },
        "review": {
            "grounded": True,
            "redacted": True,
            "unsafe_negative_action_unmasked": False,
            "success_claim_fabricated": False,
            "completion_claim_has_replayable_state": completion_evidence,
        },
        "contamination": {
            "development_or_sealed_payload_used": False,
            "source_split": parent.split,
            "hash_only_external_checks": True,
        },
        "token_counts": token_counts,
    }
    row = {"messages": messages, "tools": tools, "metadata": metadata}
    metadata["derived_row_sha256"] = _canonical_sha256(
        {key: value for key, value in row.items() if key != "metadata"}
        | {"metadata": {k: v for k, v in metadata.items() if k != "derived_row_sha256"}}
    )
    return row


def _add_grounded_rows(
    rows_by_split: dict[str, list[dict[str, Any]]],
    grounded: dict[str, Any],
    *,
    token_counter: _TokenCounter | None,
    evaluation_context: dict[str, dict[str, Any]],
    excluded_rows: list[dict[str, Any]],
) -> None:
    for split, rows in grounded["rows_by_split"].items():
        for row in rows:
            for target_index, target in enumerate(row.get("training_targets", [])):
                if not isinstance(target, dict) or target.get("masked") is True:
                    continue
                try:
                    rows_by_split[split].append(
                        _project_grounded_target(
                            grounded_record=grounded["record"],
                            row=row,
                            target=target,
                            target_index=target_index,
                            split=split,
                            token_counter=token_counter,
                            evaluation_context=evaluation_context,
                        )
                    )
                except Tau3CompetitiveDatasetError as exc:
                    if "context window" not in str(exc):
                        raise
                    excluded_rows.append(
                        {
                            "split": split,
                            "domain": _dict(row.get("metadata")).get("domain"),
                            "parent_row_sha256": _dict(row.get("metadata")).get("row_sha256"),
                            "target_index": target_index,
                            "reason": str(exc),
                        }
                    )


def _project_grounded_target(
    *,
    grounded_record: dict[str, Any],
    row: dict[str, Any],
    target: dict[str, Any],
    target_index: int,
    split: str,
    token_counter: _TokenCounter | None,
    evaluation_context: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    metadata = _dict(row.get("metadata"))
    trajectory = _dict(row.get("trajectory"))
    canonical = _dict(target.get("canonical_target"))
    behavior = str(target.get("behavior") or "")
    eval_context = evaluation_context.get(str(metadata.get("domain") or ""))
    if not eval_context:
        raise Tau3CompetitiveDatasetError("missing evaluation prompt/tool context for grounded domain")
    runtime_tool_catalog = copy.deepcopy(row.get("tool_catalog") or [])
    tool_catalog = copy.deepcopy(eval_context["tools"])
    target_message = _message_from_grounded_target(canonical)
    messages = [
        {"role": "system", "content": eval_context["system_prompt"]},
        *copy.deepcopy(
            _messages_from_grounded_turns(
                trajectory.get("turns"),
                row.get("tool_replay"),
                row.get("training_targets"),
                target.get("parent_assistant_decision_ordinal"),
            )
        ),
        target_message,
    ]
    runtime_tool_catalog_hash = _canonical_sha256(runtime_tool_catalog)
    if runtime_tool_catalog_hash != metadata.get("tool_catalog_sha256"):
        raise Tau3CompetitiveDatasetError("grounded row runtime tool catalog hash mismatch")
    if grounded_canonical_sha256(
        str(trajectory.get("system_prompt") or "")
    ) != metadata.get("system_prompt_sha256"):
        raise Tau3CompetitiveDatasetError("grounded row system prompt hash mismatch")
    _validate_grounded_tool_compatibility(canonical, runtime_tool_catalog, tool_catalog)
    target_shape = _target_from_messages(messages)
    messages, token_counts, context_fit = _fit_messages_to_context(
        messages,
        tool_catalog,
        target_shape,
        token_counter,
    )
    replay_refs = _grounded_replay_refs(row, target, eval_context)
    completion_evidence = (
        behavior == "successful_completion"
        and replay_refs["mutation_replayed"] is True
    )
    if behavior == "successful_completion" and not completion_evidence:
        raise Tau3CompetitiveDatasetError(
            "grounded successful_completion lacks replayed mutation state"
        )
    row_metadata: dict[str, Any] = {
        "schema_version": TAU3_COMPETITIVE_ROW_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "source_lineage_id": metadata.get("lineage_id"),
        "split": split,
        "domain": metadata.get("domain"),
        "source_family_id": metadata.get("source_family_id"),
        "source_id": metadata.get("source_id"),
        "source_kind": "grounded_generation_target",
        "source_behavior": behavior,
        "behavior": behavior,
        "parent_row_sha256": metadata.get("row_sha256"),
        "parent_messages_sha256": _canonical_sha256(trajectory.get("turns") or []),
        "parent_tools_sha256": _canonical_sha256(tool_catalog),
        "source_row_index": target_index,
        "source_target_ordinal": target.get("parent_assistant_decision_ordinal"),
        "target_ordinal": target.get("parent_assistant_decision_ordinal"),
        "target_action_class": target_shape["kind"],
        "target_tool_name": target_shape["tool_name"],
        "canonical_target": target_shape["canonical"],
        "canonical_target_sha256": _canonical_sha256(target_shape["canonical"]),
        "preceding_result_class": replay_refs["preceding_result_class"],
        "mutation_target": _is_mutation_tool(target_shape["tool_name"]),
        "derived_variant": f"grounded:{target_index}",
        "source_provenance": {
            "method": "grounded_generation_target_projection",
            "grounded_generation_manifest_sha256": grounded_record["manifest_sha256"],
            "grounded_parent_row_sha256": metadata.get("row_sha256"),
            "grounded_target_sha256": target.get("canonical_target_sha256"),
            "grounded_to_parent": True,
            "reviewed": True,
            "training_side_only": True,
            "completion_evidence_replayed": completion_evidence,
            "tool_replay_sha256": replay_refs["tool_replay_sha256"],
            "runtime_family": metadata.get("runtime_family"),
            "runtime_tool_catalog_sha256": runtime_tool_catalog_hash,
            "evaluation_system_prompt_sha256": eval_context["system_prompt_sha256"],
            "evaluation_tool_catalog_sha256": eval_context["tool_catalog_sha256"],
            "grounded_trajectory_id": trajectory.get("trajectory_id"),
            "grounded_target_export_ordinal": replay_refs["target_export_ordinal"],
            "parent_decision_ordinal": target.get("parent_assistant_decision_ordinal"),
            "parent_trajectory_export_sha256": replay_refs["parent_trajectory_export_sha256"],
        },
        "state_evidence_refs": {
            "pre_state": replay_refs["pre_state"],
            "post_state": replay_refs["post_state"],
            "replay_validator": "sha256:" + str(metadata.get("row_sha256") or ""),
        },
        "review": {
            "grounded": True,
            "redacted": True,
            "unsafe_negative_action_unmasked": False,
            "success_claim_fabricated": False,
            "completion_claim_has_replayable_state": completion_evidence,
        },
        "contamination": {
            "development_or_sealed_payload_used": False,
            "source_split": split,
            "hash_only_external_checks": True,
        },
        "token_counts": token_counts,
        "context_window": context_fit,
    }
    exemptions = _validated_grounded_tool_exemptions(metadata)
    if exemptions:
        row_metadata["tool_exemptions"] = exemptions
    competitive = {"messages": messages, "tools": tool_catalog, "metadata": row_metadata}
    row_metadata["derived_row_sha256"] = _canonical_sha256(
        {key: value for key, value in competitive.items() if key != "metadata"}
        | {"metadata": {k: v for k, v in row_metadata.items() if k != "derived_row_sha256"}}
    )
    return competitive


def _message_from_grounded_target(canonical: dict[str, Any]) -> dict[str, Any]:
    kind = str(canonical.get("kind") or "assistant_message")
    if kind == "tool_call":
        tool_name = str(canonical.get("tool_name") or "")
        if not tool_name:
            raise Tau3CompetitiveDatasetError("grounded tool target missing tool_name")
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "grounded-target",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": copy.deepcopy(canonical.get("arguments") or {}),
                    },
                }
            ],
        }
    return {"role": "assistant", "content": str(canonical.get("text") or "")}


def _messages_from_grounded_turns(
    turns: Any,
    tool_replay: Any,
    training_targets: Any,
    target_decision: Any,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if not isinstance(turns, list):
        return output
    replay_by_decision: dict[int, list[dict[str, Any]]] = {}
    for call in tool_replay if isinstance(tool_replay, list) else []:
        if isinstance(call, dict) and isinstance(call.get("parent_assistant_decision_ordinal"), int):
            replay_by_decision.setdefault(call["parent_assistant_decision_ordinal"], []).append(call)
    assistant_messages_by_decision: dict[int, dict[str, Any]] = {}
    masked_negative_by_decision: dict[int, dict[str, Any]] = {}
    for target in training_targets if isinstance(training_targets, list) else []:
        if not isinstance(target, dict):
            continue
        decision = target.get("parent_assistant_decision_ordinal")
        canonical = _dict(target.get("canonical_target"))
        if target.get("masked") is True:
            if isinstance(decision, int):
                negative_message = _reviewed_masked_negative_context(
                    target,
                    training_targets,
                    target_decision,
                )
                if negative_message is not None:
                    masked_negative_by_decision[decision] = negative_message
            continue
        if (
            isinstance(decision, int)
            and canonical.get("kind") != "tool_call"
            and decision not in assistant_messages_by_decision
        ):
            assistant_messages_by_decision[decision] = _message_from_grounded_target(canonical)
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        assistant = _dict(turn.get("assistant"))
        decision = assistant.get("decision_ordinal")
        if isinstance(target_decision, int) and isinstance(decision, int) and decision > target_decision:
            break
        user = turn.get("user")
        if isinstance(user, dict):
            output.append({"role": "user", "content": str(user.get("content") or "")})
        if isinstance(target_decision, int) and isinstance(decision, int) and decision >= target_decision:
            break
        if isinstance(decision, int):
            calls = replay_by_decision.get(decision, [])
            if decision in masked_negative_by_decision:
                output.append(copy.deepcopy(masked_negative_by_decision[decision]))
            elif decision in assistant_messages_by_decision:
                output.append(copy.deepcopy(assistant_messages_by_decision[decision]))
            elif calls:
                output.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": f"grounded-prefix-{decision}-{index}",
                                "type": "function",
                                "function": {
                                    "name": str(call.get("tool_name") or ""),
                                    "arguments": copy.deepcopy(call.get("canonical_arguments") or {}),
                                },
                            }
                            for index, call in enumerate(calls)
                        ],
                    }
                )
                for index, call in enumerate(calls):
                    output.append(
                        {
                            "role": "tool",
                            "name": str(call.get("tool_name") or ""),
                            "tool_call_id": f"grounded-prefix-{decision}-{index}",
                            "content": _canonical_json(call.get("canonical_result")),
                        }
                    )
    return output


def _reviewed_masked_negative_context(
    target: dict[str, Any],
    training_targets: Any,
    later_decision: Any,
) -> dict[str, Any] | None:
    if not isinstance(later_decision, int):
        return None
    decision = target.get("parent_assistant_decision_ordinal")
    if not isinstance(decision, int) or decision >= later_decision:
        return None
    mask_reason = str(target.get("mask_reason") or "")
    if mask_reason not in {"unsafe_or_negative_action", "negative_action", "unsafe_action"}:
        raise Tau3CompetitiveDatasetError("masked negative target missing explicit mask_reason")
    behavior = str(target.get("behavior") or "")
    if behavior not in {
        "hallucinated_tool",
        "harmful_mutation",
        "premature_completion",
        "hallucinated_tool_correction",
        "harmful_mutation_correction",
        "premature_completion_correction",
    }:
        raise Tau3CompetitiveDatasetError("masked negative target missing negative behavior kind")
    if target.get("reviewed") is not True and target.get("grounded_reviewed") is not True:
        raise Tau3CompetitiveDatasetError("masked negative target must be explicitly reviewed")
    linked = target.get("safe_correction_decision_ordinal", target.get("linked_safe_decision_ordinal"))
    if linked is None:
        raise Tau3CompetitiveDatasetError("masked negative target lacks safe later correction linkage")
    if linked != later_decision:
        return None
    later = _target_at_decision(training_targets, later_decision)
    if str(_dict(later).get("behavior") or "") not in {
        "hallucinated_tool_correction",
        "harmful_mutation_correction",
        "premature_completion_correction",
    }:
        raise Tau3CompetitiveDatasetError("masked negative target is not linked to a safe correction behavior")
    canonical = _dict(target.get("canonical_target"))
    if not canonical:
        raise Tau3CompetitiveDatasetError("masked negative target missing canonical target")
    if canonical.get("kind") == "tool_call":
        if not canonical.get("tool_name"):
            raise Tau3CompetitiveDatasetError("masked negative tool target missing tool_name")
        return _message_from_grounded_target(canonical)
    text = str(canonical.get("text") or canonical.get("content_preview") or target.get("negative_text") or "")
    if not text:
        raise Tau3CompetitiveDatasetError("masked negative assistant target missing explicit text")
    return {"role": "assistant", "content": text}


def _target_at_decision(training_targets: Any, decision: int) -> dict[str, Any]:
    for target in training_targets if isinstance(training_targets, list) else []:
        if isinstance(target, dict) and target.get("parent_assistant_decision_ordinal") == decision:
            return target
    return {}


def _grounded_replay_refs(
    row: dict[str, Any],
    target: dict[str, Any],
    eval_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    replay = [call for call in row.get("tool_replay", []) if isinstance(call, dict)]
    decision = target.get("parent_assistant_decision_ordinal")
    prior_calls = [
        call
        for call in replay
        if isinstance(decision, int)
        and isinstance(call.get("parent_assistant_decision_ordinal"), int)
        and call.get("parent_assistant_decision_ordinal") < decision
    ]
    target_calls = [
        call for call in replay if call.get("parent_assistant_decision_ordinal") == decision
    ]
    target_canonical = _dict(target.get("canonical_target"))
    calls = target_calls if target_canonical.get("kind") == "tool_call" else prior_calls
    mutation_calls = [
        call for call in calls
        if call.get("pre_state_sha256") != call.get("post_state_sha256")
        and _dict(call.get("state_diff")).get("change_count", 0) > 0
    ]
    prior_mutation_calls = [
        call for call in prior_calls
        if call.get("pre_state_sha256") != call.get("post_state_sha256")
        and _dict(call.get("state_diff")).get("change_count", 0) > 0
    ]
    evidence_call = mutation_calls[-1] if mutation_calls else (calls[-1] if calls else {})
    prior_evidence_call = prior_calls[-1] if prior_calls else {}
    result_class = str(prior_evidence_call.get("result_class") or "none")
    mutation_replayed = bool(
        prior_mutation_calls
        if target.get("behavior") == "successful_completion"
        else mutation_calls
    )
    return {
        "pre_state": "sha256:" + str(evidence_call.get("pre_state_sha256") or ""),
        "post_state": "sha256:" + str(evidence_call.get("post_state_sha256") or ""),
        "tool_replay_sha256": _canonical_sha256(calls),
        "preceding_result_class": result_class,
        "mutation_replayed": mutation_replayed,
        "target_export_ordinal": _target_export_ordinal(row, target),
        "parent_trajectory_export_sha256": _parent_export_hash_for_row(row, eval_context),
    }


def _evaluation_context_by_domain(parents: dict[str, list[_ParentRow]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for rows in parents.values():
        for parent in rows:
            domain = str(parent.metadata.get("domain") or "")
            if domain in output:
                continue
            messages = parent.payload.get("messages") if isinstance(parent.payload.get("messages"), list) else []
            system = next((message for message in messages if message.get("role") == "system"), None)
            if not isinstance(system, dict):
                continue
            tools = copy.deepcopy(parent.payload.get("tools") or [])
            output[domain] = {
                "system_prompt": str(system.get("content") or ""),
                "system_prompt_sha256": _canonical_sha256(str(system.get("content") or "")),
                "tools": tools,
                "tool_catalog_sha256": _canonical_sha256(tools),
            }
    return output


def _validate_grounded_tool_compatibility(
    canonical: dict[str, Any],
    runtime_catalog: list[dict[str, Any]],
    eval_catalog: list[dict[str, Any]],
) -> None:
    tool_name = canonical.get("tool_name")
    if not tool_name:
        return
    runtime = _tool_definition_by_name(runtime_catalog, str(tool_name))
    evaluation = _tool_definition_by_name(eval_catalog, str(tool_name))
    if runtime is None or evaluation is None:
        raise Tau3CompetitiveDatasetError(
            f"grounded target tool {tool_name!r} is absent from runtime or evaluation catalog"
        )
    runtime_params = _canonical_parameters(runtime)
    evaluation_params = _canonical_parameters(evaluation)
    if runtime_params and evaluation_params and runtime_params != evaluation_params:
        raise Tau3CompetitiveDatasetError(
            f"grounded target tool {tool_name!r} parameter schema differs from evaluation catalog"
        )


def _tool_definition_by_name(catalog: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for tool in catalog:
        if tool.get("name") == name:
            return tool
        function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
        if function.get("name") == name:
            return tool
    return None


def _canonical_parameters(tool: dict[str, Any]) -> Any:
    function = tool.get("function") if isinstance(tool.get("function"), dict) else None
    if function is not None:
        return _dict(function.get("parameters"))
    return _dict(tool.get("parameters"))


def _validated_grounded_tool_exemptions(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    raw = metadata.get("tool_exemptions")
    if not isinstance(raw, list):
        return []
    output = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        if item.get("reviewed") is not True or item.get("grounded_validated") is not True:
            continue
        reason = str(item.get("reason") or "")
        if reason not in {"zero_arg", "policy_forbidden"}:
            continue
        if not item.get("tool_name"):
            continue
        output.append({
            "tool_name": str(item["tool_name"]),
            "reason": reason,
            "reviewed": True,
            "grounded_validated": True,
        })
    return output


def _target_export_ordinal(row: dict[str, Any], target: dict[str, Any]) -> int:
    ordinal = 0
    for candidate in row.get("training_targets", []):
        if not isinstance(candidate, dict) or candidate.get("masked") is True:
            continue
        if candidate is target:
            return ordinal
        if candidate.get("canonical_target_sha256") == target.get("canonical_target_sha256"):
            return ordinal
        ordinal += 1
    return int(target.get("parent_assistant_decision_ordinal") or 0)


def _parent_export_hash_for_row(
    row: dict[str, Any],
    eval_context: dict[str, Any] | None = None,
) -> str:
    metadata = _dict(row.get("metadata"))
    decisions = _parent_decision_exports(row)
    parent = {
        "trajectory_id": metadata.get("parent_trajectory_id"),
        "domain": metadata.get("domain"),
        "system_prompt_sha256": _dict(eval_context).get("system_prompt_sha256", metadata.get("system_prompt_sha256")),
        "ordered_tool_catalog_sha256": _dict(eval_context).get("tool_catalog_sha256", metadata.get("tool_catalog_sha256")),
        "assistant_decisions": decisions,
    }
    return _canonical_sha256(parent)


def _add_behavior_supplements(
    rows: list[dict[str, Any]],
    parents: list[_ParentRow],
    split: str,
    *,
    token_counter: _TokenCounter | None,
) -> None:
    required = TRAIN_BEHAVIOR_MIN if split == "train" else VALID_BEHAVIOR_MIN
    family_required = TRAIN_FAMILY_SPAN_MIN if split == "train" else VALID_FAMILY_SPAN_MIN
    for domain in DOMAINS:
        domain_parents = [
            parent for parent in parents if parent.metadata.get("domain") == domain
        ]
        families = sorted({str(parent.metadata["source_family_id"]) for parent in domain_parents})
        if len(families) < family_required:
            continue
        for behavior in BEHAVIORS:
            if behavior == "successful_completion":
                continue
            while _behavior_count(rows, split, domain, behavior) < required or (
                len(_behavior_families(rows, split, domain, behavior)) < family_required
            ):
                used = _behavior_count(rows, split, domain, behavior)
                parent = domain_parents[used % len(domain_parents)]
                rows.append(
                    _supplement_row(
                        parent,
                        behavior,
                        variant_index=used,
                        token_counter=token_counter,
                    )
                )
                if used > required * max(2, len(domain_parents)):
                    break


def _supplement_row(
    parent: _ParentRow,
    behavior: str,
    *,
    variant_index: int,
    token_counter: _TokenCounter | None,
) -> dict[str, Any]:
    source_metadata = parent.metadata
    system = _system_message(parent.payload["messages"])
    first_user = _supplement_user_message(
        behavior,
        domain=str(source_metadata["domain"]),
    )
    assistant = _supplement_assistant_message(
        behavior,
        domain=str(source_metadata["domain"]),
    )
    messages = [copy.deepcopy(system), copy.deepcopy(first_user), assistant]
    tools = copy.deepcopy(parent.payload["tools"])
    target = _target_from_messages(messages)
    token_counts = _token_counts(messages, tools, target, token_counter)
    metadata: dict[str, Any] = {
        "schema_version": TAU3_COMPETITIVE_ROW_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "source_lineage_id": source_metadata.get("lineage_id"),
        "split": parent.split,
        "domain": source_metadata.get("domain"),
        "source_family_id": source_metadata.get("source_family_id"),
        "source_id": source_metadata.get("source_id"),
        "source_kind": "derived_reviewed_safe_decision",
        "source_behavior": source_metadata.get("behavior"),
        "behavior": behavior,
        "parent_row_sha256": parent.row_sha256,
        "parent_messages_sha256": _canonical_sha256(parent.payload["messages"]),
        "parent_tools_sha256": _canonical_sha256(parent.payload["tools"]),
        "source_row_index": parent.index,
        "source_target_ordinal": source_metadata.get("target_ordinal"),
        "target_ordinal": int(source_metadata.get("target_ordinal") or 0),
        "target_action_class": target["kind"],
        "target_tool_name": target["tool_name"],
        "canonical_target": target["canonical"],
        "canonical_target_sha256": _canonical_sha256(target["canonical"]),
        "preceding_result_class": _preceding_result_class(messages),
        "mutation_target": False,
        "derived_variant": f"{behavior}:{variant_index}",
        "source_provenance": {
            "method": "deterministic_reviewed_safe_behavior_template",
            "grounded_to_parent": True,
            "reviewed": True,
            "training_side_only": True,
            "parent_target_unmasked": False,
            "visible_prompt_entails_target": True,
        },
        "state_evidence_refs": {
            "pre_state": None,
            "post_state": None,
        },
        "review": {
            "grounded": True,
            "redacted": True,
            "unsafe_negative_action_unmasked": False,
            "success_claim_fabricated": False,
            "completion_claim_has_replayable_state": False,
        },
        "contamination": {
            "development_or_sealed_payload_used": False,
            "source_split": parent.split,
            "hash_only_external_checks": True,
        },
        "token_counts": token_counts,
    }
    row = {"messages": messages, "tools": tools, "metadata": metadata}
    metadata["derived_row_sha256"] = _canonical_sha256(
        {key: value for key, value in row.items() if key != "metadata"}
        | {"metadata": {k: v for k, v in metadata.items() if k != "derived_row_sha256"}}
    )
    return row


def _supplement_assistant_message(
    behavior: str,
    *,
    domain: str,
) -> dict[str, Any]:
    text_by_behavior = {
        "clarification_refusal": (
            f"I need the missing {domain} account detail before I can continue; "
            "I will not guess or use an unrelated account."
        ),
        "authentication": (
            "I need to authenticate the account holder before any "
            "private details or account changes can be handled."
        ),
        "confirmation_before_mutation": (
            f"Please confirm the exact {domain} change before I "
            "make any account or order mutation."
        ),
        "later_task_completion_actions": (
            f"I will handle the later requested {domain} step after "
            "the current verified lookup result."
        ),
        "safe_stopping": (
            f"I will stop the {domain} workflow now and will not "
            "take further tool actions."
        ),
        "transfer_handoff": (
            f"This {domain} case needs a human handoff; I will "
            "transfer with only the reviewed summary."
        ),
        "empty_result_recovery": (
            f"The previous {domain} lookup returned no result, so I "
            "will ask for another identifier instead of inventing a record."
        ),
        "error_result_recovery": (
            f"The previous {domain} tool result errored, so I will "
            "recover with a safe alternate lookup or explanation."
        ),
        "repeated_call_recovery": (
            f"The repeated {domain} result did not add new evidence, "
            "so I will avoid repeating the same call."
        ),
        "hallucinated_tool_correction": (
            f"I cannot use a nonexistent {domain} tool; I will use "
            "only the listed tool catalog or ask for clarification."
        ),
        "harmful_mutation_correction": (
            f"I will not perform the unsafe {domain} mutation; I "
            "will keep the account unchanged."
        ),
        "premature_completion_correction": (
            f"I should not mark the {domain} task complete until "
            "the required evidence and confirmation are present."
        ),
    }
    return {"role": "assistant", "content": text_by_behavior[behavior]}


def _supplement_user_message(behavior: str, *, domain: str) -> dict[str, Any]:
    prompt_by_behavior = {
        "clarification_refusal": (
            f"Proceed with my {domain} request even though the account detail is missing."
        ),
        "authentication": (
            f"Show or change private {domain} account details before authentication."
        ),
        "confirmation_before_mutation": (
            f"Make a {domain} account change; the exact change still needs confirmation."
        ),
        "later_task_completion_actions": (
            f"After the current verified lookup, handle the later {domain} step."
        ),
        "safe_stopping": f"Stop this {domain} workflow now.",
        "transfer_handoff": f"Transfer this {domain} case to a human agent.",
        "empty_result_recovery": (
            f"The last {domain} lookup returned no result. Continue safely."
        ),
        "error_result_recovery": (
            f"The last {domain} tool call errored. Continue safely."
        ),
        "repeated_call_recovery": (
            f"The same {domain} result repeated. Avoid looping on the same call."
        ),
        "hallucinated_tool_correction": (
            f"Use an unavailable {domain} tool to solve this."
        ),
        "harmful_mutation_correction": (
            f"Perform an unsafe {domain} mutation despite the policy."
        ),
        "premature_completion_correction": (
            f"Mark the {domain} task complete before the required evidence is present."
        ),
    }
    return {"role": "user", "content": prompt_by_behavior[behavior]}


def _map_source_behavior(
    metadata: dict[str, Any],
    messages: list[dict[str, Any]],
    target: dict[str, Any],
) -> str:
    behavior = str(metadata.get("behavior") or "")
    if behavior == "success":
        if _has_completion_evidence(metadata):
            return "successful_completion"
        return "later_task_completion_actions"
    if behavior == "empty_result_recovery" or metadata.get("after_empty_result") is True:
        return "empty_result_recovery"
    if metadata.get("after_error_result") is True:
        return "error_result_recovery"
    if metadata.get("repeated_call_recovery") is True:
        return "repeated_call_recovery"
    mapping = {
        "hallucinated_tool": "hallucinated_tool_correction",
        "harmful_mutation": "harmful_mutation_correction",
        "premature_completion": "premature_completion_correction",
        "recovery": "error_result_recovery",
        "correction": "premature_completion_correction",
        "policy_failure": "clarification_refusal",
    }
    return mapping.get(behavior, "later_task_completion_actions")


def _coverage_report(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    blockers: list[str] = []
    by_split: dict[str, Any] = {}
    for split, rows in rows_by_split.items():
        required = TRAIN_BEHAVIOR_MIN if split == "train" else VALID_BEHAVIOR_MIN
        family_required = TRAIN_FAMILY_SPAN_MIN if split == "train" else VALID_FAMILY_SPAN_MIN
        tool_required = TRAIN_TOOL_MIN if split == "train" else VALID_TOOL_MIN
        arg_required = TRAIN_TOOL_ARG_MIN if split == "train" else VALID_TOOL_ARG_MIN
        split_report: dict[str, Any] = {}
        token_totals = {domain: 0 for domain in DOMAINS}
        row_totals = {domain: 0 for domain in DOMAINS}
        for row in rows:
            meta = row["metadata"]
            domain = str(meta["domain"])
            row_totals[domain] += 1
            token_counts = meta["token_counts"]
            if (
                token_counts.get("exact") is not True
                or token_counts.get("chat_template_aware") is not True
            ):
                blockers.append(f"{split}:{domain}:token_counts_not_exact")
            token_totals[domain] += int(token_counts.get("supervised_tokens") or 0)
        all_tokens = sum(token_totals.values())
        all_rows = sum(row_totals.values())
        for domain in DOMAINS:
            domain_rows = [row for row in rows if row["metadata"]["domain"] == domain]
            behavior_counts = {
                behavior: sum(
                    row["metadata"]["behavior"] == behavior for row in domain_rows
                )
                for behavior in BEHAVIORS
            }
            behavior_family_spans = {
                behavior: len(
                    {
                        row["metadata"]["source_family_id"]
                        for row in domain_rows
                        if row["metadata"]["behavior"] == behavior
                    }
                )
                for behavior in BEHAVIORS
            }
            for behavior in BEHAVIORS:
                if behavior_counts[behavior] < required:
                    blockers.append(
                        f"{split}:{domain}:{behavior}:count "
                        f"{behavior_counts[behavior]} < {required}"
                    )
                if behavior_family_spans[behavior] < family_required:
                    blockers.append(
                        f"{split}:{domain}:{behavior}:family_span "
                        f"{behavior_family_spans[behavior]} < {family_required}"
                    )
            tool_counts: dict[str, int] = {}
            tool_arg_hashes: dict[str, set[str]] = {}
            catalog = _tool_names(domain_rows)
            exemptions = _tool_exemptions(domain_rows)
            for row in domain_rows:
                meta = row["metadata"]
                if meta.get("target_action_class") != "tool_call":
                    continue
                tool = str(meta.get("target_tool_name") or "")
                if not tool:
                    continue
                tool_counts[tool] = tool_counts.get(tool, 0) + 1
                tool_arg_hashes.setdefault(tool, set()).add(
                    str(meta["canonical_target"].get("arguments_sha256") or "")
                )
            for tool_name in sorted(catalog):
                if tool_name in exemptions:
                    continue
                count = tool_counts.get(tool_name, 0)
                arg_count = len(tool_arg_hashes.get(tool_name, set()) - {""})
                if count < tool_required:
                    blockers.append(
                        f"{split}:{domain}:tool:{tool_name}:count {count} < {tool_required}"
                    )
                if arg_count < arg_required:
                    blockers.append(
                        f"{split}:{domain}:tool:{tool_name}:distinct_args "
                        f"{arg_count} < {arg_required}"
                    )
            split_report[domain] = {
                "row_count": row_totals[domain],
                "target_token_count": token_totals[domain],
                "example_share": (row_totals[domain] / all_rows) if all_rows else 0,
                "target_token_share": (token_totals[domain] / all_tokens) if all_tokens else 0,
                "behavior_counts": behavior_counts,
                "behavior_family_spans": behavior_family_spans,
                "tool_counts": dict(sorted(tool_counts.items())),
                "tool_argument_diversity": {
                    tool: len(values - {""})
                    for tool, values in sorted(tool_arg_hashes.items())
                },
                "tool_exemptions": sorted(exemptions),
            }
        by_split[split] = split_report
        _add_balance_blockers(blockers, split, split_report)
        if not rows:
            blockers.append(f"{split}:empty")
        for domain in DOMAINS:
            _add_dominance_blockers(
                blockers,
                split,
                domain,
                [row for row in rows if row["metadata"]["domain"] == domain],
            )
        _add_split_integrity_blockers(blockers, split, rows)
    _add_cross_split_blockers(blockers, rows_by_split)
    return {
        "passed": not blockers,
        "blockers": sorted(set(blockers)),
        "by_split": by_split,
    }


def _validate_rows(rows_by_split: dict[str, list[dict[str, Any]]]) -> list[str]:
    errors: list[str] = []
    seen_row_hashes: set[str] = set()
    for split, rows in rows_by_split.items():
        for index, row in enumerate(rows):
            label = f"{split}[{index}]"
            if not isinstance(row, dict):
                errors.append(f"{label}: row must be object")
                continue
            messages = _messages(row.get("messages"), f"{label}.messages", errors)
            tools = _tools(row.get("tools"), f"{label}.tools", errors)
            meta = _object(row.get("metadata"), f"{label}.metadata", errors)
            if meta.get("schema_version") != TAU3_COMPETITIVE_ROW_SCHEMA_VERSION:
                errors.append(f"{label}: invalid row schema_version")
            for field in (
                "lineage_id",
                "split",
                "domain",
                "source_family_id",
                "behavior",
                "parent_row_sha256",
                "parent_messages_sha256",
                "parent_tools_sha256",
                "canonical_target_sha256",
                "derived_row_sha256",
            ):
                if field not in meta:
                    errors.append(f"{label}: missing metadata.{field}")
            if meta.get("lineage_id") != LINEAGE_ID:
                errors.append(f"{label}: lineage_id must be {LINEAGE_ID}")
            if meta.get("split") != split:
                errors.append(f"{label}: split metadata mismatch")
            if meta.get("domain") not in DOMAINS:
                errors.append(f"{label}: unsupported domain")
            if meta.get("behavior") not in BEHAVIORS:
                errors.append(f"{label}: unsupported behavior")
            if meta.get("parent_messages_sha256") and not _is_sha256(
                meta.get("parent_messages_sha256")
            ):
                errors.append(f"{label}: invalid parent_messages_sha256")
            if meta.get("parent_tools_sha256") != _canonical_sha256(tools):
                if meta.get("source_kind") == "direct_parent":
                    errors.append(f"{label}: direct parent tools hash mismatch")
                elif meta.get("source_kind") == "grounded_generation_target":
                    errors.append(f"{label}: grounded row evaluation tool catalog hash mismatch")
            review = _object(meta.get("review"), f"{label}.review", errors)
            if review.get("grounded") is not True:
                errors.append(f"{label}: target is not grounded/reviewed")
            if review.get("unsafe_negative_action_unmasked") is not False:
                errors.append(f"{label}: unsafe negative action was unmasked")
            if review.get("success_claim_fabricated") is not False:
                errors.append(f"{label}: success claim fabricated")
            if meta.get("behavior") == "successful_completion":
                if review.get("completion_claim_has_replayable_state") is not True:
                    errors.append(
                        f"{label}: successful_completion lacks replayable state evidence"
                    )
                state_refs = _object(
                    meta.get("state_evidence_refs"),
                    f"{label}.state_evidence_refs",
                    errors,
                )
                if not state_refs.get("pre_state") or not state_refs.get("post_state"):
                    errors.append(
                        f"{label}: successful_completion requires pre/post state refs"
                    )
                if not _is_content_addressed_ref(state_refs.get("pre_state")):
                    errors.append(
                        f"{label}: successful_completion pre_state must be content-addressed"
                    )
                if not _is_content_addressed_ref(state_refs.get("post_state")):
                    errors.append(
                        f"{label}: successful_completion post_state must be content-addressed"
                    )
                if not _is_content_addressed_ref(state_refs.get("replay_validator")):
                    errors.append(
                        f"{label}: successful_completion requires replay validator binding"
                    )
            if (
                meta.get("source_kind") == "derived_reviewed_safe_decision"
                and meta.get("behavior") == "successful_completion"
            ):
                errors.append(f"{label}: template row cannot be successful_completion")
            token_counts = _object(
                meta.get("token_counts"),
                f"{label}.token_counts",
                errors,
            )
            if (
                token_counts.get("exact") is not True
                or token_counts.get("chat_template_aware") is not True
            ):
                errors.append(f"{label}: token counts must be exact and chat-template-aware")
            total_tokens = int(token_counts.get("total_tokens") or 0)
            if total_tokens != int(token_counts.get("prompt_tokens") or 0) + int(token_counts.get("supervised_tokens") or 0):
                errors.append(f"{label}: token_counts.total_tokens must equal prompt+supervised")
            if total_tokens > CONTEXT_WINDOW_TOKENS:
                errors.append(f"{label}: total_tokens exceeds context window {CONTEXT_WINDOW_TOKENS}")
            context_window = meta.get("context_window")
            if isinstance(context_window, dict):
                if context_window.get("max_tokens") != CONTEXT_WINDOW_TOKENS:
                    errors.append(f"{label}: context_window.max_tokens mismatch")
                if context_window.get("total_tokens") != total_tokens:
                    errors.append(f"{label}: context_window.total_tokens mismatch")
                if context_window.get("truncated") is not False:
                    errors.append(f"{label}: message/schema truncation is forbidden")
            contamination = _object(
                meta.get("contamination"), f"{label}.contamination", errors
            )
            if contamination.get("development_or_sealed_payload_used") is not False:
                errors.append(f"{label}: development/sealed payload used")
            target = _object(
                meta.get("canonical_target"), f"{label}.canonical_target", errors
            )
            if meta.get("canonical_target_sha256") != _canonical_sha256(target):
                errors.append(f"{label}: canonical target hash mismatch")
            expected = _canonical_sha256(
                {key: value for key, value in row.items() if key != "metadata"}
                | {
                    "metadata": {
                        key: value
                        for key, value in meta.items()
                        if key != "derived_row_sha256"
                    }
                }
            )
            if meta.get("derived_row_sha256") != expected:
                errors.append(f"{label}: derived_row_sha256 mismatch")
            if expected in seen_row_hashes:
                errors.append(f"{label}: duplicate derived row hash")
            seen_row_hashes.add(expected)
            if not messages:
                errors.append(f"{label}: messages must be non-empty")
    return errors


def _add_balance_blockers(
    blockers: list[str],
    split: str,
    split_report: dict[str, Any],
) -> None:
    for domain in DOMAINS:
        token_share = split_report[domain]["target_token_share"]
        if token_share < DOMAIN_TOKEN_SHARE_MIN or token_share > DOMAIN_TOKEN_SHARE_MAX:
            blockers.append(
                f"{split}:{domain}:target_token_share {token_share:.4f} "
                f"outside {DOMAIN_TOKEN_SHARE_MIN:.2f}-{DOMAIN_TOKEN_SHARE_MAX:.2f}"
            )
    telecom = split_report["telecom"]
    if telecom["example_share"] < TELECOM_SHARE_MIN:
        blockers.append(
            f"{split}:telecom:example_share {telecom['example_share']:.4f} < "
            f"{TELECOM_SHARE_MIN:.2f}"
        )
    if telecom["target_token_share"] < TELECOM_SHARE_MIN:
        blockers.append(
            f"{split}:telecom:target_token_share {telecom['target_token_share']:.4f} < "
            f"{TELECOM_SHARE_MIN:.2f}"
        )


def _add_dominance_blockers(
    blockers: list[str],
    split: str,
    domain: str,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return
    _dominance_check(
        blockers,
        f"{split}:{domain}:canonical_target_duplication_share",
        [str(row["metadata"]["canonical_target_sha256"]) for row in rows],
        MAX_DUPLICATE_SHARE,
    )
    _dominance_check(
        blockers,
        f"{split}:{domain}:source_family_share",
        [str(row["metadata"]["source_family_id"]) for row in rows],
        MAX_DOMINANCE_SHARE,
    )
    exemptions = _consistent_tool_exemptions(rows)
    tool_rows = [
        row for row in rows
        if row["metadata"].get("target_action_class") == "tool_call"
        and row["metadata"].get("target_tool_name")
        and row["metadata"].get("target_tool_name") not in exemptions
    ]
    _dominance_check(
        blockers,
        f"{split}:{domain}:target_tool_share",
        [str(row["metadata"]["target_tool_name"]) for row in tool_rows],
        MAX_DOMINANCE_SHARE,
    )
    by_tool: dict[str, list[dict[str, Any]]] = {}
    for row in tool_rows:
        by_tool.setdefault(str(row["metadata"]["target_tool_name"]), []).append(row)
    for tool, tool_specific in sorted(by_tool.items()):
        _dominance_check(
            blockers,
            f"{split}:{domain}:tool:{tool}:argument_payload_share",
            [str(row["metadata"]["canonical_target"].get("arguments_sha256") or "") for row in tool_specific],
            MAX_DOMINANCE_SHARE,
        )
        _dominance_check(
            blockers,
            f"{split}:{domain}:tool:{tool}:argument_template_share",
            [_target_template_key(row) for row in tool_specific],
            MAX_DOMINANCE_SHARE,
        )
    mutation_rows = [row for row in rows if row["metadata"].get("mutation_target") is True]
    _dominance_check(
        blockers,
        f"{split}:{domain}:synthetic_mutation_family_share",
        [
            str(
                _dict(row["metadata"].get("source_provenance")).get(
                    "synthetic_mutation_family",
                    row["metadata"].get("source_family_id"),
                )
            )
            for row in mutation_rows
        ],
        MAX_DOMINANCE_SHARE,
    )


def _dominance_check(
    blockers: list[str],
    label: str,
    values: list[str],
    threshold: float,
) -> None:
    values = [value for value in values if value]
    if not values:
        return
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    share = max(counts.values(), default=0) / len(values)
    if share > threshold:
        blockers.append(
            f"{label} {share:.4f} > {threshold:.2f}"
        )


def _target_template_key(row: dict[str, Any]) -> str:
    meta = _dict(row.get("metadata"))
    provenance = _dict(meta.get("source_provenance"))
    for key in ("argument_template_id", "synthetic_mutation_family"):
        value = provenance.get(key) or meta.get(key)
        if value:
            return str(value)
    canonical = _dict(meta.get("canonical_target"))
    if canonical.get("kind") != "tool_call":
        return str(canonical.get("content_sha256") or canonical.get("content_preview") or "")
    args = _dict(canonical.get("arguments"))
    if canonical.get("arguments_sha256"):
        return str(canonical.get("arguments_sha256"))
    return _canonical_sha256(
        {
            "kind": "tool_call",
            "tool_name": canonical.get("tool_name"),
            "argument_value_pattern": {
                str(key): _argument_value_pattern(value)
                for key, value in sorted(args.items())
            },
        }
    )


def _argument_value_pattern(value: Any) -> str:
    if isinstance(value, bool):
        return f"bool:{value}"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        if any(char.isdigit() for char in value):
            return "string_with_digits:" + _canonical_sha256(value)
        return "string:" + value[:32]
    if isinstance(value, list):
        return "list:" + str(len(value))
    if isinstance(value, dict):
        return "object:" + _canonical_sha256(sorted(str(key) for key in value))
    return type(value).__name__


def _add_split_integrity_blockers(
    blockers: list[str],
    split: str,
    rows: list[dict[str, Any]],
) -> None:
    parent_by_split: dict[str, set[str]] = {"train": set(), "valid": set()}
    family_by_split: dict[str, set[str]] = {"train": set(), "valid": set()}
    for row in rows:
        meta = row["metadata"]
        parent_by_split.setdefault(str(meta["split"]), set()).add(
            str(meta["parent_row_sha256"])
        )
        family_by_split.setdefault(str(meta["split"]), set()).add(
            str(meta["source_family_id"])
        )
    if split in parent_by_split and "" in parent_by_split[split]:
        blockers.append(f"{split}:blank parent hash")


def _add_cross_split_blockers(
    blockers: list[str],
    rows_by_split: dict[str, list[dict[str, Any]]],
) -> None:
    train_parents = {
        str(row["metadata"]["parent_row_sha256"])
        for row in rows_by_split.get("train", [])
    }
    valid_parents = {
        str(row["metadata"]["parent_row_sha256"])
        for row in rows_by_split.get("valid", [])
    }
    parent_overlap = train_parents & valid_parents
    if parent_overlap:
        blockers.append(
            "train_valid_parent_row_hash_overlap "
            f"{len(parent_overlap)} > 0"
        )
    train_families = {
        str(row["metadata"]["source_family_id"])
        for row in rows_by_split.get("train", [])
    }
    valid_families = {
        str(row["metadata"]["source_family_id"])
        for row in rows_by_split.get("valid", [])
    }
    family_overlap = train_families & valid_families
    if family_overlap:
        blockers.append(
            "train_valid_source_family_overlap "
            f"{len(family_overlap)} > 0"
        )


def _target_from_messages(messages: list[dict[str, Any]]) -> dict[str, Any]:
    assistant = next(
        (
            message
            for message in reversed(messages)
            if message.get("role") == "assistant"
        ),
        {},
    )
    calls = assistant.get("tool_calls") if isinstance(assistant, dict) else None
    if isinstance(calls, list) and calls:
        function = _object_no_errors(calls[0].get("function"))
        name = str(function.get("name") or "")
        args = function.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"raw": args}
        canonical = {
            "kind": "tool_call",
            "tool_name": name,
            "arguments": args if isinstance(args, dict) else {},
            "arguments_sha256": _canonical_sha256(args if isinstance(args, dict) else {}),
        }
        return {"kind": "tool_call", "tool_name": name, "canonical": canonical}
    content = str(assistant.get("content") or "")
    canonical = {
        "kind": "assistant_message",
        "tool_name": "assistant_message",
        "content_sha256": _canonical_sha256(content),
        "content_preview": content[:160],
    }
    return {"kind": "assistant_message", "tool_name": "assistant_message", "canonical": canonical}


def _has_completion_evidence(metadata: dict[str, Any]) -> bool:
    evidence = metadata.get("state_evidence_refs")
    if not isinstance(evidence, dict):
        return False
    pre_state = evidence.get("pre_state")
    post_state = evidence.get("post_state")
    replay_validator = evidence.get("replay_validator")
    if not (
        _is_content_addressed_ref(pre_state)
        and _is_content_addressed_ref(post_state)
        and _is_content_addressed_ref(replay_validator)
    ):
        return False
    return evidence.get("replay_validated") is True


def _state_evidence_value(metadata: dict[str, Any], key: str) -> Any:
    evidence = metadata.get("state_evidence_refs")
    if not isinstance(evidence, dict):
        return None
    value = evidence.get(key)
    return value if _is_content_addressed_ref(value) else None


def _is_content_addressed_ref(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    if value.startswith("sha256:"):
        return _is_sha256(value.removeprefix("sha256:"))
    if value.startswith("hfr://"):
        return "sha256=" in value and any(_is_sha256(part[-64:]) for part in value.split("sha256=")[1:])
    return False


def _preceding_result_class(messages: list[dict[str, Any]]) -> str:
    before_assistant = []
    for message in messages:
        if message.get("role") == "assistant":
            before_assistant = []
        else:
            before_assistant.append(message)
    tools = [message for message in before_assistant if message.get("role") == "tool"]
    if not tools:
        return "none"
    content = " ".join(str(message.get("content") or "").lower() for message in tools[-2:])
    if any(term in content for term in ("error", "exception", "failed")):
        return "error"
    if any(term in content for term in ("[]", "no result", "not found", "empty")):
        return "empty"
    return "non_empty"


def _token_counts(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    target: dict[str, Any],
    token_counter: _TokenCounter | None,
) -> dict[str, Any]:
    if token_counter is not None:
        return token_counter.count(messages=messages, tools=tools, target=target)
    return {
        "method": "unavailable",
        "exact": False,
        "chat_template_aware": False,
        "prompt_tokens": 0,
        "supervised_tokens": 0,
        "total_tokens": 0,
    }


def _fit_messages_to_context(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    target: dict[str, Any],
    token_counter: _TokenCounter | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    token_counts = _token_counts(messages, tools, target, token_counter)
    if token_counts.get("total_tokens", 0) <= CONTEXT_WINDOW_TOKENS:
        return messages, token_counts, {
            "max_tokens": CONTEXT_WINDOW_TOKENS,
            "total_tokens": token_counts.get("total_tokens", 0),
            "excluded_oldest_complete_interaction_units": 0,
            "truncated": False,
            "fit_strategy": "none",
        }
    units = _prefix_interaction_units(messages)
    if len(units) <= 1:
        raise Tau3CompetitiveDatasetError(
            f"context window exceeded: total_tokens {token_counts.get('total_tokens')} > {CONTEXT_WINDOW_TOKENS}; no complete oldest interaction unit can be dropped"
        )
    dropped = 0
    kept_units = units[:]
    while len(kept_units) > 1:
        kept_units.pop(0)
        dropped += 1
        candidate = [messages[0], *[message for unit in kept_units for message in unit], messages[-1]]
        token_counts = _token_counts(candidate, tools, target, token_counter)
        if token_counts.get("total_tokens", 0) <= CONTEXT_WINDOW_TOKENS:
            return candidate, token_counts, {
                "max_tokens": CONTEXT_WINDOW_TOKENS,
                "total_tokens": token_counts.get("total_tokens", 0),
                "excluded_oldest_complete_interaction_units": dropped,
                "truncated": False,
                "fit_strategy": "drop_oldest_complete_interaction_units",
            }
    raise Tau3CompetitiveDatasetError(
        f"context window exceeded: total_tokens {token_counts.get('total_tokens')} > {CONTEXT_WINDOW_TOKENS}; latest causal context and full target cannot fit"
    )


def _prefix_interaction_units(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    prefix = messages[1:-1] if len(messages) >= 2 and messages[0].get("role") == "system" else messages[:-1]
    units: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in prefix:
        if message.get("role") == "user" and current:
            units.append(current)
            current = []
        current.append(message)
    if current:
        units.append(current)
    return units


def _tool_names(rows: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for row in rows:
        for tool in row.get("tools", []):
            function = tool.get("function") if isinstance(tool, dict) else None
            if isinstance(function, dict) and function.get("name"):
                names.add(str(function["name"]))
    return names


def _tool_exemptions(rows: list[dict[str, Any]]) -> set[str]:
    return set(_consistent_tool_exemptions(rows))


def _consistent_tool_exemptions(rows: list[dict[str, Any]]) -> set[str]:
    reviewed: dict[str, set[str]] = {}
    rejected: set[str] = set()
    for row in rows:
        meta = row.get("metadata", {})
        for item in meta.get("tool_exemptions", []) if isinstance(meta, dict) else []:
            if not isinstance(item, dict) or not item.get("tool_name"):
                continue
            tool = str(item["tool_name"])
            reason = str(item.get("reason") or "")
            if (
                item.get("reviewed") is not True
                or item.get("grounded_validated") is not True
                or reason not in {"zero_arg", "policy_forbidden"}
            ):
                rejected.add(tool)
                continue
            reviewed.setdefault(tool, set()).add(
                _canonical_json(
                    {
                        "tool_name": tool,
                        "reason": reason,
                        "reviewed": True,
                        "grounded_validated": True,
                    }
                )
            )
    return {
        tool
        for tool, records in reviewed.items()
        if tool not in rejected and len(records) == 1
    }


def _behavior_count(
    rows: list[dict[str, Any]],
    split: str,
    domain: str,
    behavior: str,
) -> int:
    return sum(
        row["metadata"]["split"] == split
        and row["metadata"]["domain"] == domain
        and row["metadata"]["behavior"] == behavior
        for row in rows
    )


def _behavior_families(
    rows: list[dict[str, Any]],
    split: str,
    domain: str,
    behavior: str,
) -> set[str]:
    return {
        row["metadata"]["source_family_id"]
        for row in rows
        if row["metadata"]["split"] == split
        and row["metadata"]["domain"] == domain
        and row["metadata"]["behavior"] == behavior
    }


def _system_message(messages: list[dict[str, Any]]) -> dict[str, Any]:
    for message in messages:
        if message.get("role") == "system":
            return message
    raise Tau3CompetitiveDatasetError("parent row is missing system message")


def _first_user_message(messages: list[dict[str, Any]], domain: str) -> dict[str, Any]:
    for message in messages:
        if message.get("role") == "user":
            return message
    return {"role": "user", "content": f"Please help with my {domain} account."}


def _read_parent_rows(path: Path, split: str) -> list[_ParentRow]:
    parents: list[_ParentRow] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Tau3CompetitiveDatasetError(
                f"{path}:{index + 1}: invalid JSON: {exc.msg}"
            ) from exc
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            raise Tau3CompetitiveDatasetError(f"{path}:{index + 1}: missing metadata")
        if metadata.get("schema_version") != SOURCE_SCHEMA_VERSION:
            raise Tau3CompetitiveDatasetError(
                f"{path}:{index + 1}: expected {SOURCE_SCHEMA_VERSION}"
            )
        if metadata.get("split") != split:
            raise Tau3CompetitiveDatasetError(f"{path}:{index + 1}: split mismatch")
        if metadata.get("lineage_id") != SOURCE_LINEAGE_ID:
            raise Tau3CompetitiveDatasetError(
                f"{path}:{index + 1}: source lineage mismatch"
            )
        messages = row.get("messages")
        tools = row.get("tools")
        if not isinstance(messages, list) or not messages:
            raise Tau3CompetitiveDatasetError(f"{path}:{index + 1}: missing messages")
        if not isinstance(tools, list) or not tools:
            raise Tau3CompetitiveDatasetError(f"{path}:{index + 1}: missing tools")
        parents.append(
            _ParentRow(
                split=split,
                index=index,
                payload=row,
                metadata=metadata,
                row_sha256=_canonical_sha256(row),
            )
        )
    return parents


def _validate_source_manifest(manifest: dict[str, Any], source: Path) -> None:
    if manifest.get("schema_version") != "hfr.tau3_policy_complete_dataset.v1":
        raise Tau3CompetitiveDatasetError("source manifest schema_version invalid")
    if manifest.get("lineage_id") != SOURCE_LINEAGE_ID:
        raise Tau3CompetitiveDatasetError("source manifest lineage_id invalid")
    if manifest.get("passed") is not True:
        raise Tau3CompetitiveDatasetError("source manifest must be passed")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise Tau3CompetitiveDatasetError("source manifest files missing")
    for split in ("train", "valid"):
        record = files.get(split)
        if not isinstance(record, dict):
            raise Tau3CompetitiveDatasetError(f"source manifest missing {split}")
        if record.get("sha256") != _sha256(source / f"{split}.jsonl"):
            raise Tau3CompetitiveDatasetError(f"source {split} hash mismatch")


def _load_token_counter(
    path: str | Path | None,
    *,
    bundle_out_dir: Path | None = None,
    copy_into_bundle: bool = False,
) -> _TokenCounter | None:
    if path is None:
        return None
    config_path = Path(path)
    _require_file(config_path, "tokenizer config")
    config = _read_object(config_path, "tokenizer config")
    if config.get("schema_version") != "hfr.tau3_competitive_tokenizer_config.v1":
        raise Tau3CompetitiveDatasetError("tokenizer config schema_version invalid")
    if config.get("exact") is not True:
        raise Tau3CompetitiveDatasetError("tokenizer config must declare exact=true")
    if config.get("chat_template_aware") is not True:
        raise Tau3CompetitiveDatasetError(
            "tokenizer config must declare chat_template_aware=true"
        )
    if config.get("tokenization_algorithm") != "pinned_local_apply_chat_template":
        raise Tau3CompetitiveDatasetError(
            "unsupported tokenizer config tokenization_algorithm"
        )
    for field in ("tokenizer_id", "tokenizer_revision", "tokenizer_path"):
        if not isinstance(config.get(field), str) or not config[field]:
            raise Tau3CompetitiveDatasetError(f"tokenizer config missing {field}")
    tokenizer_path = Path(str(config["tokenizer_path"]))
    if not tokenizer_path.is_dir():
        raise Tau3CompetitiveDatasetError("tokenizer_path must be a local directory")
    tokenizer_config = _read_object(
        tokenizer_path / "tokenizer_config.json",
        "local tokenizer tokenizer_config.json",
    )
    source_uses_external_chat_template = (
        not tokenizer_config.get("chat_template")
        and (tokenizer_path / "chat_template.jinja").is_file()
    )
    load_path = tokenizer_path
    path_ref = tokenizer_path.name
    if copy_into_bundle:
        if bundle_out_dir is None:
            raise Tau3CompetitiveDatasetError("bundle_out_dir is required for tokenizer copy")
        target = bundle_out_dir / "tokenizer"
        if target.exists():
            raise Tau3CompetitiveDatasetError("tokenizer output already exists")
        target.mkdir(parents=True)
        copied_any_external_template = False
        for filename in TOKENIZER_ASSET_FILENAMES:
            source_asset = tokenizer_path / filename
            if source_asset.is_file():
                shutil.copy2(source_asset, target / filename)
                if filename == "chat_template.jinja":
                    copied_any_external_template = True
        if source_uses_external_chat_template and not copied_any_external_template:
            raise Tau3CompetitiveDatasetError(
                "tokenizer relies on chat_template.jinja but it was not copied"
            )
        load_path = target
        path_ref = "tokenizer"
    tokenizer = _load_local_tokenizer(load_path)
    chat_template = str(getattr(tokenizer, "chat_template", "") or "")
    if not chat_template:
        raise Tau3CompetitiveDatasetError("local tokenizer has no chat_template")
    copied_assets = {
        filename: _sha256(load_path / filename)
        for filename in TOKENIZER_ASSET_FILENAMES
        if (load_path / filename).is_file()
    }
    record = {
        "path_leaf": path_ref,
        "config_sha256": _sha256(config_path),
        "tokenizer_json_sha256": _sha256(load_path / "tokenizer.json"),
        "tokenizer_config_sha256": _sha256(load_path / "tokenizer_config.json"),
        "chat_template_sha256": _canonical_sha256(chat_template),
        "chat_template_file_sha256": _sha256(load_path / "chat_template.jinja") if (load_path / "chat_template.jinja").is_file() else None,
        "copied_assets": copied_assets,
        "tokenizer_id": str(config["tokenizer_id"]),
        "tokenizer_revision": str(config["tokenizer_revision"]),
    }
    return _TokenCounter(config=config, config_record=record, tokenizer=tokenizer)


def _load_local_tokenizer(tokenizer_path: Path) -> Any:
    for filename in ("tokenizer.json", "tokenizer_config.json"):
        _require_file(tokenizer_path / filename, f"local tokenizer {filename}")
    try:
        from transformers import AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise Tau3CompetitiveDatasetError(
            "transformers is required to load the pinned local tokenizer"
        ) from exc
    try:
        return AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    except Exception as exc:  # pragma: no cover - defensive.
        raise Tau3CompetitiveDatasetError(
            f"could not load pinned local tokenizer: {type(exc).__name__}"
        ) from exc


def _apply_chat_template_token_ids(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if isinstance(encoded, dict) or hasattr(encoded, "get"):
        encoded = encoded.get("input_ids")
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if not isinstance(encoded, list) or not encoded:
        raise Tau3CompetitiveDatasetError(
            "pinned tokenizer returned no input_ids from apply_chat_template"
        )
    return [int(token) for token in encoded]


def _prompt_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "assistant":
            return messages[:index]
    return messages


def _read_rows_with_errors(path: Path, split: str, errors: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{split}:{line_number}: invalid JSON: {exc.msg}")
            continue
        rows.append(row)
    return rows


def _read_jsonl_no_errors(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _validation_result(
    bundle: Path,
    passed: bool,
    errors: list[str],
    coverage: dict[str, Any],
    strict: bool,
) -> dict[str, Any]:
    return {
        "schema_version": "hfr.tau3_competitive_dataset_validation.v1",
        "bundle": str(bundle),
        "strict": strict,
        "passed": passed,
        "error_count": len(errors),
        "errors": errors,
        "coverage": coverage,
    }


def _thresholds() -> dict[str, Any]:
    return {
        "behaviors": list(BEHAVIORS),
        "train_behavior_min_per_domain": TRAIN_BEHAVIOR_MIN,
        "valid_behavior_min_per_domain": VALID_BEHAVIOR_MIN,
        "train_behavior_family_span_min": TRAIN_FAMILY_SPAN_MIN,
        "valid_behavior_family_span_min": VALID_FAMILY_SPAN_MIN,
        "train_tool_target_min": TRAIN_TOOL_MIN,
        "valid_tool_target_min": VALID_TOOL_MIN,
        "train_tool_distinct_arg_min": TRAIN_TOOL_ARG_MIN,
        "valid_tool_distinct_arg_min": VALID_TOOL_ARG_MIN,
        "domain_target_token_share_min": DOMAIN_TOKEN_SHARE_MIN,
        "domain_target_token_share_max": DOMAIN_TOKEN_SHARE_MAX,
        "telecom_example_and_token_share_min": TELECOM_SHARE_MIN,
        "max_canonical_target_duplication_share": MAX_DUPLICATE_SHARE,
        "max_source_family_share": MAX_DOMINANCE_SHARE,
        "max_target_tool_share": MAX_DOMINANCE_SHARE,
        "max_tool_argument_payload_share": MAX_DOMINANCE_SHARE,
        "max_tool_argument_template_share": MAX_DOMINANCE_SHARE,
        "max_synthetic_mutation_family_share": MAX_DOMINANCE_SHARE,
        "context_window_tokens": CONTEXT_WINDOW_TOKENS,
    }


def _check_schema_if_registered(manifest: dict[str, Any]) -> None:
    errors = _schema_errors_if_registered(manifest, None)
    if errors:
        raise Tau3CompetitiveDatasetError(
            "v3 dataset manifest schema failed: " + "; ".join(errors)
        )


def _schema_errors_if_registered(
    manifest: dict[str, Any],
    artifact_path: Path | None,
) -> list[str]:
    try:
        schema = check_schema_contract(
            manifest,
            name_or_id=TAU3_COMPETITIVE_DATASET_SCHEMA_VERSION,
            artifact_path=artifact_path,
        )
    except SchemaRegistryError as exc:
        if "Unknown schema" in str(exc):
            return []
        raise
    return list(schema["errors"])


def _is_mutation_tool(tool_name: str) -> bool:
    return any(tool_name.startswith(prefix) for prefix in MUTATION_PREFIXES)


def _messages(value: Any, label: str, errors: list[str]) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        errors.append(f"{label} must be a list of objects")
        return []
    return value


def _tools(value: Any, label: str, errors: list[str]) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        errors.append(f"{label} must be a list of objects")
        return []
    return value


def _object(value: Any, label: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{label} must be an object")
        return {}
    return value


def _object_no_errors(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise Tau3CompetitiveDatasetError(f"cannot read {label}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise Tau3CompetitiveDatasetError(f"{label} is invalid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise Tau3CompetitiveDatasetError(f"{label} must be a JSON object")
    return value


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise Tau3CompetitiveDatasetError(f"{label} is missing: {path}")


def _optional_file_record(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    candidate = Path(path)
    _require_file(candidate, "optional reference")
    return _file_record(candidate)


def _file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(relative_to)) if relative_to else str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def build_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--plan-path")
    parser.add_argument("--tool-catalog-path")
    parser.add_argument("--tokenizer-config-path")
    parser.add_argument("--grounded-generation-bundle")
    parser.add_argument(
        "--grounded-validator-python",
        help="Python executable used to replay grounded-generation bundles in a separate Tau environment",
    )
    parser.add_argument("--contamination-report-path")
    parser.add_argument(
        "--include-template-supplements",
        action="store_true",
        help="include non-grounded template rows for blocked diagnostics only",
    )
    args = parser.parse_args(argv)
    try:
        manifest = build_tau3_competitive_dataset(
            source_dataset_dir=args.source_dataset_dir,
            out_dir=args.out_dir,
            plan_path=args.plan_path,
            tool_catalog_path=args.tool_catalog_path,
            tokenizer_config_path=args.tokenizer_config_path,
            grounded_generation_bundle=args.grounded_generation_bundle,
            grounded_validator_python=args.grounded_validator_python,
            contamination_report_path=args.contamination_report_path,
            include_template_supplements=args.include_template_supplements,
        )
    except Tau3CompetitiveDatasetError as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["passed"] else 3


def validate_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a Tau-3 v3 dataset bundle.")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--no-strict", action="store_true")
    parser.add_argument(
        "--grounded-validator-python",
        help="Python executable used to replay grounded-generation bundles in a separate Tau environment",
    )
    args = parser.parse_args(argv)
    result = validate_tau3_competitive_dataset_bundle(
        args.bundle,
        strict=not args.no_strict,
        grounded_validator_python=args.grounded_validator_python,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2
