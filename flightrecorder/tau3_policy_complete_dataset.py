"""Build the contamination-safe Tau-3 policy-complete MLX dataset.

The legacy compact mixture intentionally removed the system prompt, reduced the
tool catalog, and dropped rows that exceeded a small token budget.  This module
is a breaking, create-once replacement.  It keeps the exact Tau system prompt
and ordered tool catalog, splits training-side task families before projection,
and trims only complete historical interaction groups around a final assistant
target.

Rejected actions are useful as context, not targets.  Rows that contain a
negative action therefore end in a safe correction and require MLX-LM
``mask_prompt=true`` so only the final assistant message receives loss.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .path_safety import path_has_symlink_component
from .schema_registry import check_schema_contract
from .tau3_capture import validate_tau3_capture

TAU3_POLICY_COMPLETE_DATASET_SCHEMA_VERSION = "hfr.tau3_policy_complete_dataset.v1"
TAU3_POLICY_COMPLETE_ROW_SCHEMA_VERSION = "hfr.tau3_policy_complete_row.v1"
LINEAGE_ID = "tau3-core-agent-mixture-v2-policy-complete"
DOMAINS = ("airline", "retail", "telecom")
BEHAVIORS = (
    "clarification_refusal",
    "correction",
    "hallucinated_tool",
    "harmful_mutation",
    "policy_failure",
    "premature_completion",
    "recovery",
    "success",
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
LOOKUP_PREFIXES = (
    "find_",
    "get_",
    "list_",
    "search_",
)
CONTEXT_POLICY = (
    "pin_exact_system_full_ordered_tools_target_latest_user_immediate_result_"
    "and_negative_evidence_then_trim_oldest_complete_interaction_groups"
)
PARTITION_ALGORITHM = "sha256_ranked_domain_stratified_task_or_scenario_family_holdout_v2"
NEAR_DUPLICATE_THRESHOLD = 0.98
MAX_TRAIN_DOMAIN_SHARE = 0.50
USER_SIMULATOR_PRIVATE_MARKERS = (
    "\nevaluation criteria:",
    "\nknown info:",
    "\ntask instructions:",
    "\nuser scenario:",
    "check that agent ",
    "reference tau tool trajectory completed.",
)


class Tau3PolicyCompleteDatasetError(ValueError):
    """Raised when v2 dataset construction cannot prove its invariants."""


@dataclass(frozen=True)
class _Target:
    source_kind: str
    source_id: str
    source_sha256: str
    family_id: str
    domain: str
    behavior: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    target_index: int
    target_kind: str
    target_tool_name: str
    target_ordinal: int
    after_empty_result: bool
    after_error_result: bool
    repeated_call_recovery: bool
    negative_prefix: bool
    pinned_message_indices: frozenset[int]


def build_tau3_policy_complete_dataset(
    *,
    teacher_corpus_dir: str | Path,
    captures_path: str | Path,
    train_split_path: str | Path,
    development_split_path: str | Path,
    development_tasks_path: str | Path,
    parent_protocol_path: str | Path,
    tokenizer_path: str | Path,
    out_dir: str | Path,
    max_seq_length: int = 16384,
    context_window: int = 16384,
    validation_fraction: float = 0.20,
    partition_salt: str = "tau3-policy-complete-v2",
) -> dict[str, Any]:
    """Create a new-only MLX dataset and strict evidence manifest."""

    corpus_dir = Path(teacher_corpus_dir)
    captures_file = Path(captures_path)
    train_split_file = Path(train_split_path)
    development_split_file = Path(development_split_path)
    development_tasks_file = Path(development_tasks_path)
    protocol_file = Path(parent_protocol_path)
    tokenizer_root = Path(tokenizer_path)
    out = Path(out_dir)
    for label, path in (
        ("teacher corpus", corpus_dir),
        ("captures", captures_file),
        ("train split", train_split_file),
        ("development split", development_split_file),
        ("development tasks", development_tasks_file),
        ("parent protocol", protocol_file),
        ("tokenizer", tokenizer_root),
        ("output", out),
    ):
        _reject_symlink_path(path, label)
    _require_file(corpus_dir / "manifest.json", "teacher corpus manifest")
    _require_file(corpus_dir / "train.jsonl", "teacher train corpus")
    _require_file(corpus_dir / "valid.jsonl", "teacher development corpus")
    for label, path in (
        ("captures", captures_file),
        ("train split", train_split_file),
        ("development split", development_split_file),
        ("development tasks", development_tasks_file),
        ("parent protocol", protocol_file),
    ):
        _require_file(path, label)
    _require_new_output(out)
    if max_seq_length <= 0 or context_window <= 0:
        raise Tau3PolicyCompleteDatasetError("sequence and context budgets must be positive")
    if max_seq_length > context_window:
        raise Tau3PolicyCompleteDatasetError("max_seq_length must not exceed context_window")
    if not 0.0 < validation_fraction < 0.5:
        raise Tau3PolicyCompleteDatasetError("validation_fraction must be between 0 and 0.5")
    if not partition_salt:
        raise Tau3PolicyCompleteDatasetError("partition_salt must be non-empty")

    protocol = _read_object(protocol_file, "parent protocol")
    _validate_parent_protocol(protocol, protocol_file, train_split_file, development_split_file)
    train_split = _read_source_split(train_split_file, expected_split="train")
    development_split = _read_source_split(
        development_split_file,
        expected_split="development",
    )
    train_family_ids = {str(value) for value in train_split["family_ids"]}
    development_family_ids = {str(value) for value in development_split["family_ids"]}
    family_overlap = sorted(train_family_ids & development_family_ids)
    if family_overlap:
        raise Tau3PolicyCompleteDatasetError(
            "official train/development family overlap: " + family_overlap[0]
        )

    corpus_manifest = _read_object(corpus_dir / "manifest.json", "teacher corpus manifest")
    _validate_corpus_manifest(corpus_dir, corpus_manifest)
    teacher_rows = _read_jsonl(corpus_dir / "train.jsonl", "teacher train corpus")
    if any(str(row.get("metadata", {}).get("split") or "") != "train" for row in teacher_rows):
        raise Tau3PolicyCompleteDatasetError(
            "teacher train corpus contains a non-training-side row"
        )
    system_prompts = _extract_system_prompts(teacher_rows)

    captures = _read_jsonl(captures_file, "Tau captures")
    capture_errors = {
        str(row.get("trajectory_id") or index): errors
        for index, row in enumerate(captures)
        if (errors := validate_tau3_capture(row))
    }
    if capture_errors:
        first = sorted(capture_errors)[0]
        raise Tau3PolicyCompleteDatasetError(
            f"invalid capture {first}: {capture_errors[first][0]}"
        )
    train_captures = [row for row in captures if row.get("split") == "train"]
    excluded_development_captures = [
        row for row in captures if row.get("split") == "development"
    ]
    if not train_captures or not excluded_development_captures:
        raise Tau3PolicyCompleteDatasetError(
            "captures must contain training-side rows and excluded development evidence"
        )
    _validate_capture_evidence_coverage(train_captures)
    tool_catalogs = _extract_tool_catalogs(train_captures)
    _validate_training_side_families(teacher_rows, train_captures, train_family_ids)

    development_tasks = _read_jsonl(
        development_tasks_file,
        "development source tasks",
    )
    contamination = _contamination_report(
        teacher_rows=teacher_rows,
        development_split=development_split,
        development_tasks=development_tasks,
        protocol=protocol,
    )
    if contamination["passed"] is not True:
        failed = next(check for check in contamination["checks"] if not check["passed"])
        raise Tau3PolicyCompleteDatasetError(
            f"contamination check failed: {failed['id']}"
        )

    available_families_by_domain = _available_families_by_domain(teacher_rows)
    qualified_families_by_domain = _coverage_qualified_teacher_families(
        teacher_rows
    )
    partition = _partition_families(
        available_families_by_domain,
        qualified_families_by_domain,
        validation_fraction=validation_fraction,
        salt=partition_salt,
    )
    valid_families = {
        family
        for domain in DOMAINS
        for family in partition["by_domain"][domain]["internal_valid_families"]
    }

    teacher_targets = _teacher_targets(teacher_rows, system_prompts, tool_catalogs)
    counterfactual_targets, counterfactual_exclusions = (
        _teacher_counterfactual_targets(teacher_targets)
    )
    targets = [*teacher_targets, *counterfactual_targets]
    capture_exclusions = _capture_projection_exclusions(train_captures)
    if not targets:
        raise Tau3PolicyCompleteDatasetError("no supervised targets were derived")

    tokenizer = _load_tokenizer(tokenizer_root)
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "valid": []}
    for target in targets:
        split = "valid" if target.family_id in valid_families else "train"
        row, rendered_tokens = _project_target(
            target,
            split=split,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            context_window=context_window,
        )
        rows_by_split[split].append(row)
        if rendered_tokens != row["metadata"]["rendered_tokens"]:
            raise Tau3PolicyCompleteDatasetError(
                f"{target.source_id}: rendered token count was not retained"
            )
    rows_by_split["train"], balance, balance_exclusions = _balance_training_rows(
        rows_by_split["train"]
    )
    for split in ("train", "valid"):
        rows_by_split[split].sort(
            key=lambda row: (
                str(row["metadata"]["domain"]),
                str(row["metadata"]["source_family_id"]),
                str(row["metadata"]["source_id"]),
                int(row["metadata"]["target_ordinal"]),
            )
        )
        if not rows_by_split[split]:
            raise Tau3PolicyCompleteDatasetError(f"{split} projection is empty")
    lengths = [
        (
            int(row["metadata"]["rendered_tokens"]),
            str(row["metadata"]["derived_row_sha256"]),
        )
        for rows in rows_by_split.values()
        for row in rows
    ]

    coverage = _coverage_report(rows_by_split, system_prompts, tool_catalogs)
    if coverage["passed"] is not True:
        failed = next(check for check in coverage["checks"] if not check["passed"])
        raise Tau3PolicyCompleteDatasetError(f"coverage check failed: {failed['id']}")

    out.mkdir(parents=True)
    for split in ("train", "valid"):
        _write_jsonl(out / f"{split}.jsonl", rows_by_split[split])
    longest_length, longest_id = max(lengths, default=(0, ""))
    input_records = {
        "teacher_corpus_manifest": _file_record(corpus_dir / "manifest.json"),
        "teacher_train": _file_record(corpus_dir / "train.jsonl"),
        "teacher_development_excluded": _file_record(corpus_dir / "valid.jsonl"),
        "captures": _file_record(captures_file),
        "train_split": _file_record(train_split_file),
        "development_split_hashes_only": _file_record(development_split_file),
        "development_tasks_contamination_only": _file_record(development_tasks_file),
        "parent_protocol": _file_record(protocol_file),
    }
    manifest: dict[str, Any] = {
        "schema_version": TAU3_POLICY_COMPLETE_DATASET_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "format": "mlx-chat-jsonl",
        "passed": True,
        "parent_protocol": input_records["parent_protocol"],
        "inputs": input_records,
        "counts": {
            "train": len(rows_by_split["train"]),
            "valid": len(rows_by_split["valid"]),
            "teacher_training_rows": len(teacher_rows),
            "teacher_success_targets": len(teacher_targets),
            "teacher_success_targets_admitted": sum(
                row["metadata"]["source_kind"] == "teacher_success"
                for rows in rows_by_split.values()
                for row in rows
            ),
            "teacher_counterfactual_targets": len(counterfactual_targets),
            "teacher_counterfactual_targets_admitted": sum(
                str(row["metadata"]["source_kind"]).startswith(
                    "teacher_counterfactual_"
                )
                for rows in rows_by_split.values()
                for row in rows
            ),
            "teacher_counterfactual_family_exclusions": len(
                counterfactual_exclusions
            ),
            "teacher_development_rows_excluded": int(
                corpus_manifest.get("counts", {}).get("valid") or 0
            ),
            "capture_training_rows": len(train_captures),
            "capture_training_rows_projected": 0,
            "capture_development_rows_excluded": len(excluded_development_captures),
        },
        "exclusions": {
            "reason_coded": True,
            "count": (
                len(capture_exclusions)
                + len(counterfactual_exclusions)
                + len(balance_exclusions)
            ),
            "records_sha256": _canonical_sha256(
                [
                    *capture_exclusions,
                    *counterfactual_exclusions,
                    *balance_exclusions,
                ]
            ),
            "reason_counts": _count_by_key(
                [
                    *capture_exclusions,
                    *counterfactual_exclusions,
                    *balance_exclusions,
                ],
                "reason",
            ),
            "records": [
                *capture_exclusions,
                *counterfactual_exclusions,
                *balance_exclusions,
            ],
        },
        "files": {
            split: _file_record(out / f"{split}.jsonl", relative_to=out)
            for split in ("train", "valid")
        },
        "partition": partition,
        "balance": balance,
        "supervision": {
            "mask_prompt_required": True,
            "last_assistant_message_only": True,
            "negative_actions_are_context_only": True,
            "capture_content_policy": (
                "raw_capture_user_simulator_and_reference-only messages excluded; "
                "capture tools and behavior evidence retained"
            ),
            "negative_prefix_unmasked_target_count": 0,
            "negative_prefix_target_count": sum(
                bool(row["metadata"]["negative_prefix"])
                for rows in rows_by_split.values()
                for row in rows
            ),
        },
        "context_projection": {
            "policy": CONTEXT_POLICY,
            "full_tool_catalog_required": True,
            "exact_system_prompt_required": True,
            "content_truncation_allowed": False,
            "complete_interaction_group_removal_only": True,
        },
        "tokenizer": {
            "checked": True,
            "method": "pinned_base_apply_chat_template",
            "path_leaf": tokenizer_root.name,
            "chat_template_sha256": _canonical_sha256(
                str(getattr(tokenizer, "chat_template", "") or "")
            ),
            "row_count": len(lengths),
            "min_rendered_tokens": min(length for length, _ in lengths),
            "max_rendered_tokens": longest_length,
            "max_seq_length": max_seq_length,
            "context_window": context_window,
            "over_max_seq_length_count": 0,
            "over_context_window_count": 0,
            "longest_row_id": longest_id,
        },
        "coverage": coverage,
        "contamination": contamination,
        "sealed": {
            "manifest_hash_only": True,
            "access_count": 0,
            "payload_accessed": False,
            "task_ids_materialized": False,
        },
        "training_started": False,
    }
    manifest["manifest_sha256"] = _canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    schema = check_schema_contract(
        manifest,
        name_or_id=TAU3_POLICY_COMPLETE_DATASET_SCHEMA_VERSION,
    )
    if schema["passed"] is not True:
        raise Tau3PolicyCompleteDatasetError(
            "policy-complete manifest schema failed: " + "; ".join(schema["errors"])
        )
    _write_json(out / "manifest.json", manifest)
    return manifest


def _validate_parent_protocol(
    protocol: dict[str, Any],
    protocol_path: Path,
    train_split_path: Path,
    development_split_path: Path,
) -> None:
    if protocol.get("schema_version") != "hfr.tau3_protocol_config.v1":
        raise Tau3PolicyCompleteDatasetError("parent protocol schema_version is invalid")
    sealed = _object(protocol.get("sealed_manifest"), "parent sealed manifest")
    if sealed.get("access_count") != 0:
        raise Tau3PolicyCompleteDatasetError("parent sealed access_count must remain zero")
    recipe = _object(protocol.get("recipe_space"), "parent recipe space")
    if recipe.get("sealed_used") is not False:
        raise Tau3PolicyCompleteDatasetError("parent recipe space records sealed use")
    selection = _object(
        protocol.get("candidate_selection_contract"),
        "parent candidate selection contract",
    )
    if selection.get("sealed_used") is not False:
        raise Tau3PolicyCompleteDatasetError(
            "parent candidate selection contract records sealed use"
        )
    split_manifest = _object(protocol.get("split_manifest"), "parent split manifest")
    splits = _object(split_manifest.get("splits"), "parent split manifest splits")
    expected = {
        "train": (train_split_path, splits.get("train")),
        "development": (development_split_path, splits.get("development")),
    }
    for label, (path, raw_record) in expected.items():
        record = _object(raw_record, f"parent {label} split binding")
        if record.get("sha256") != _sha256(path):
            raise Tau3PolicyCompleteDatasetError(
                f"parent {label} split hash does not replay"
            )
    if _sha256(protocol_path) == "0" * 64:  # pragma: no cover - defensive.
        raise Tau3PolicyCompleteDatasetError("parent protocol hash is invalid")


def _validate_corpus_manifest(corpus_dir: Path, manifest: dict[str, Any]) -> None:
    if (
        manifest.get("schema_version") != "hfr.tau3_conversation_import.v1"
        or manifest.get("passed") is not True
    ):
        raise Tau3PolicyCompleteDatasetError(
            "teacher corpus manifest is not a passed conversation import"
        )
    files = _object(manifest.get("files"), "teacher corpus files")
    for split in ("train", "valid"):
        record = _object(files.get(split), f"teacher corpus {split} binding")
        if record.get("sha256") != _sha256(corpus_dir / f"{split}.jsonl"):
            raise Tau3PolicyCompleteDatasetError(
                f"teacher corpus {split} hash does not replay"
            )
    provenance = _object(
        manifest.get("generation_provenance"),
        "teacher generation provenance",
    )
    if not _is_sha256(provenance.get("protocol_sha256")):
        raise Tau3PolicyCompleteDatasetError(
            "teacher corpus protocol hash is invalid"
        )


def _extract_system_prompts(
    teacher_rows: list[dict[str, Any]],
) -> dict[str, str]:
    prompts: dict[str, set[str]] = {domain: set() for domain in DOMAINS}
    for row in teacher_rows:
        metadata = _object(row.get("metadata"), "teacher metadata")
        domain = str(metadata.get("domain") or "")
        if domain not in prompts:
            raise Tau3PolicyCompleteDatasetError(
                f"teacher row has unsupported domain: {domain!r}"
            )
        messages = _messages(row.get("messages"), "teacher messages")
        system = [message for message in messages if message.get("role") == "system"]
        if len(system) != 1 or messages[0].get("role") != "system":
            raise Tau3PolicyCompleteDatasetError(
                f"teacher {domain} row must begin with exactly one system prompt"
            )
        content = str(system[0].get("content") or "")
        if not content:
            raise Tau3PolicyCompleteDatasetError(
                f"teacher {domain} system prompt is empty"
            )
        declared = metadata.get("system_prompt_sha256")
        if declared != _canonical_sha256(content):
            raise Tau3PolicyCompleteDatasetError(
                f"teacher {domain} system prompt hash does not replay"
            )
        prompts[domain].add(content)
    output: dict[str, str] = {}
    for domain, values in prompts.items():
        if len(values) != 1:
            raise Tau3PolicyCompleteDatasetError(
                f"teacher {domain} must bind exactly one system prompt"
            )
        output[domain] = next(iter(values))
    return output


def _extract_tool_catalogs(
    captures: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    catalogs: dict[str, dict[str, list[dict[str, Any]]]] = {
        domain: {} for domain in DOMAINS
    }
    for capture in captures:
        domain = str(capture.get("domain") or "")
        tools = _tools(capture.get("tools"), f"capture {domain} tools")
        catalogs[domain][_canonical_sha256(tools)] = tools
    output: dict[str, list[dict[str, Any]]] = {}
    for domain, values in catalogs.items():
        if len(values) != 1:
            raise Tau3PolicyCompleteDatasetError(
                f"capture {domain} must bind exactly one ordered tool catalog"
            )
        output[domain] = next(iter(values.values()))
    return output


def _validate_capture_evidence_coverage(
    captures: list[dict[str, Any]],
) -> None:
    observed = {
        (str(capture["domain"]), str(capture["behavior"]))
        for capture in captures
    }
    missing = [
        f"{domain}:{behavior}"
        for domain in DOMAINS
        for behavior in BEHAVIORS
        if (domain, behavior) not in observed
    ]
    if missing:
        raise Tau3PolicyCompleteDatasetError(
            "capture behavior evidence is incomplete: " + missing[0]
        )


def _validate_training_side_families(
    teacher_rows: list[dict[str, Any]],
    captures: list[dict[str, Any]],
    train_family_ids: set[str],
) -> None:
    observed = {
        str(row["metadata"].get("task_family") or "")
        for row in teacher_rows
    } | {str(row.get("task_family") or "") for row in captures}
    missing = sorted(family for family in observed if family not in train_family_ids)
    if missing:
        raise Tau3PolicyCompleteDatasetError(
            "training input family is absent from official train split: " + missing[0]
        )


def _available_families_by_domain(
    teacher_rows: list[dict[str, Any]],
) -> dict[str, list[str]]:
    values: dict[str, set[str]] = {domain: set() for domain in DOMAINS}
    for row in teacher_rows:
        metadata = row["metadata"]
        domain = str(metadata["domain"])
        values[domain].add(
            _v2_family_id(
                domain,
                str(metadata["task_family"]),
                str(metadata["task_id"]),
            )
        )
    return {domain: sorted(values[domain]) for domain in DOMAINS}


def _coverage_qualified_teacher_families(
    teacher_rows: list[dict[str, Any]],
) -> dict[str, list[str]]:
    flags: dict[tuple[str, str], dict[str, bool]] = {}
    for row in teacher_rows:
        metadata = row["metadata"]
        domain = str(metadata["domain"])
        family = _v2_family_id(
            domain,
            str(metadata["task_family"]),
            str(metadata["task_id"]),
        )
        current = flags.setdefault(
            (domain, family),
            {
                "lookup": False,
                "mutation": False,
                "later_action": False,
            },
        )
        seen_user = False
        ordinal = 0
        for message in row["messages"]:
            if message.get("role") == "user":
                seen_user = True
                continue
            if message.get("role") != "assistant" or not seen_user:
                continue
            tool_name = _message_tool_name(message)
            if tool_name:
                current["lookup"] = current["lookup"] or tool_name.startswith(
                    LOOKUP_PREFIXES
                )
                current["mutation"] = current["mutation"] or tool_name.startswith(
                    MUTATION_PREFIXES
                )
                current["later_action"] = current["later_action"] or ordinal > 0
            ordinal += 1
    output = {
        domain: sorted(
            family
            for (candidate_domain, family), observed in flags.items()
            if candidate_domain == domain and all(observed.values())
        )
        for domain in DOMAINS
    }
    missing = [domain for domain, families in output.items() if len(families) < 2]
    if missing:
        raise Tau3PolicyCompleteDatasetError(
            "teacher corpus requires at least two policy-complete families for "
            + missing[0]
        )
    return output


def _partition_families(
    available: dict[str, list[str]],
    qualified_families: dict[str, list[str]],
    *,
    validation_fraction: float,
    salt: str,
) -> dict[str, Any]:
    by_domain: dict[str, Any] = {}
    all_train: set[str] = set()
    all_valid: set[str] = set()
    for domain in DOMAINS:
        families = available[domain]
        if len(families) < 2:
            raise Tau3PolicyCompleteDatasetError(
                f"{domain} requires at least two training-side families"
            )
        desired = max(1, math.ceil(len(families) * validation_fraction))
        desired = min(desired, len(families) - 1)
        ranked_qualified = sorted(
            qualified_families[domain],
            key=lambda family: _canonical_sha256(f"{salt}:{domain}:{family}"),
        )
        if len(ranked_qualified) < 2:
            raise Tau3PolicyCompleteDatasetError(
                f"{domain} requires policy-complete families in both fit and internal validation"
            )
        valid = ranked_qualified[:1]
        remaining = sorted(
            (family for family in families if family not in valid),
            key=lambda family: _canonical_sha256(f"{salt}:{domain}:{family}"),
        )
        valid.extend(remaining[: max(0, desired - len(valid))])
        valid_set = set(valid)
        fit = [family for family in families if family not in valid_set]
        if not set(ranked_qualified) & set(fit):
            raise Tau3PolicyCompleteDatasetError(
                f"{domain} internal validation consumed every policy-complete family"
            )
        by_domain[domain] = {
            "fit_families": fit,
            "internal_valid_families": sorted(valid_set),
            "coverage_qualified_fit_family_count": len(
                set(ranked_qualified) & set(fit)
            ),
            "coverage_qualified_internal_valid_family_count": len(
                set(ranked_qualified) & valid_set
            ),
        }
        all_train.update(fit)
        all_valid.update(valid_set)
    overlap = sorted(all_train & all_valid)
    if overlap:
        raise Tau3PolicyCompleteDatasetError(
            "fit/internal-valid family overlap: " + overlap[0]
        )
    return {
        "algorithm": PARTITION_ALGORITHM,
        "family_key_policy": {
            "airline": "upstream_task_family",
            "retail": "upstream_task_family",
            "telecom": "issue_type_plus_hidden_state_condition_set_persona_excluded",
        },
        "salt_sha256": _canonical_sha256(salt),
        "validation_fraction": validation_fraction,
        "family_overlap_count": 0,
        "fit_family_count": len(all_train),
        "internal_valid_family_count": len(all_valid),
        "by_domain": by_domain,
    }


def _teacher_targets(
    teacher_rows: list[dict[str, Any]],
    system_prompts: dict[str, str],
    tool_catalogs: dict[str, list[dict[str, Any]]],
) -> list[_Target]:
    targets: list[_Target] = []
    for row in teacher_rows:
        metadata = row["metadata"]
        domain = str(metadata["domain"])
        family = _v2_family_id(
            domain,
            str(metadata["task_family"]),
            str(metadata["task_id"]),
        )
        source_id = str(metadata["episode_id"])
        messages = _normalize_messages(
            row["messages"],
            f"teacher {source_id}",
        )
        messages[0] = {"role": "system", "content": system_prompts[domain]}
        seen_user = False
        ordinal = 0
        for index, message in enumerate(messages):
            if message.get("role") == "user":
                seen_user = True
                continue
            if message.get("role") != "assistant" or not seen_user:
                continue
            tool_name = _message_tool_name(message)
            targets.append(
                _Target(
                    source_kind="teacher_success",
                    source_id=source_id,
                    source_sha256=_canonical_sha256(row),
                    family_id=family,
                    domain=domain,
                    behavior="success",
                    messages=messages,
                    tools=tool_catalogs[domain],
                    target_index=index,
                    target_kind="tool_call" if tool_name else "assistant_message",
                    target_tool_name=tool_name,
                    target_ordinal=ordinal,
                    after_empty_result=_prior_result_matches(
                        messages,
                        index,
                        _is_empty_result_message,
                    ),
                    after_error_result=_prior_result_matches(
                        messages,
                        index,
                        _is_error_result_message,
                    ),
                    repeated_call_recovery=False,
                    negative_prefix=False,
                    pinned_message_indices=frozenset(),
                )
            )
            ordinal += 1
    return targets


def _teacher_counterfactual_targets(
    teacher_targets: list[_Target],
) -> tuple[list[_Target], list[dict[str, Any]]]:
    """Derive safe recovery/correction rows only from agent-visible successes.

    Capture-v1 remains the immutable source of the exact ordered tool catalogs
    and behavior taxonomy. Its first user event is a private user-simulator
    instruction envelope and its terminal assistant text can be grader or
    reference-only prose, so neither is projected into agent SFT.
    """

    grouped: dict[tuple[str, str], list[_Target]] = {}
    for target in teacher_targets:
        grouped.setdefault((target.domain, target.family_id), []).append(target)
    derived: list[_Target] = []
    exclusions: list[dict[str, Any]] = []
    for (domain, family), family_targets in sorted(grouped.items()):
        action = _select_counterfactual_action_target(family_targets)
        lookup = next(
            (
                target
                for target in sorted(family_targets, key=_target_sort_key)
                if target.target_kind == "tool_call"
                and target.target_tool_name.startswith(LOOKUP_PREFIXES)
            ),
            None,
        )
        if action is None:
            exclusions.append(
                {
                    "source_id": family,
                    "source_sha256": _canonical_sha256(
                        [target.source_sha256 for target in family_targets]
                    ),
                    "domain": domain,
                    "behavior": "counterfactual_family",
                    "family_id": family,
                    "reason": "no_agent_visible_teacher_action_target",
                }
            )
            continue
        for behavior in (
            "recovery",
            "correction",
            "hallucinated_tool",
            "harmful_mutation",
            "policy_failure",
            "premature_completion",
        ):
            derived.append(_counterfactual_action_target(action, behavior))
        if lookup is None:
            exclusions.append(
                {
                    "source_id": family,
                    "source_sha256": _canonical_sha256(
                        [target.source_sha256 for target in family_targets]
                    ),
                    "domain": domain,
                    "behavior": "empty_result_recovery",
                    "family_id": family,
                    "reason": "no_agent_visible_teacher_lookup_target",
                }
            )
        else:
            derived.append(_counterfactual_empty_target(lookup))
    return derived, exclusions


def _select_counterfactual_action_target(
    targets: list[_Target],
) -> _Target | None:
    candidates = [target for target in targets if target.target_kind == "tool_call"]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda target: (
            not target.target_tool_name.startswith(MUTATION_PREFIXES),
            target.target_ordinal <= 0,
            *_target_sort_key(target),
        ),
    )


def _target_sort_key(target: _Target) -> tuple[Any, ...]:
    return (
        target.source_id,
        target.target_ordinal,
        target.target_tool_name,
        target.source_sha256,
    )


def _counterfactual_action_target(base: _Target, behavior: str) -> _Target:
    messages = _copy_messages(base.messages[: base.target_index])
    original_user = max(
        (
            index
            for index, message in enumerate(messages)
            if message.get("role") == "user"
        ),
        default=-1,
    )
    pinned = {original_user} if original_user >= 0 else set()
    after_error = behavior in {"recovery", "hallucinated_tool"}
    if after_error:
        call_id = f"counterfactual-{behavior}-invalid"
        messages.extend(
            [
                _tool_call_message(call_id, "invented_tau_tool", {}),
                {
                    "role": "tool",
                    "name": "invented_tau_tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(
                        {
                            "error": "unknown_tool",
                            "message": "Tool is not present in the provided catalog.",
                        },
                        sort_keys=True,
                    ),
                },
            ]
        )
        pinned.update({len(messages) - 2, len(messages) - 1})
    else:
        rejected, feedback = _counterfactual_rejected_exchange(behavior)
        messages.extend(
            [
                {"role": "assistant", "content": rejected},
                {"role": "user", "content": feedback},
            ]
        )
        pinned.update({len(messages) - 2, len(messages) - 1})
    messages.append(_copy_message(base.messages[base.target_index]))
    source_id = f"{base.source_id}:counterfactual:{behavior}"
    return _Target(
        source_kind=f"teacher_counterfactual_{behavior}",
        source_id=source_id,
        source_sha256=_canonical_sha256(
            {
                "teacher_source_sha256": base.source_sha256,
                "teacher_target_ordinal": base.target_ordinal,
                "behavior": behavior,
                "derivation": "agent_visible_prefix_then_rejected_context_then_gold_action",
            }
        ),
        family_id=base.family_id,
        domain=base.domain,
        behavior=behavior,
        messages=messages,
        tools=base.tools,
        target_index=len(messages) - 1,
        target_kind=base.target_kind,
        target_tool_name=base.target_tool_name,
        target_ordinal=base.target_ordinal,
        after_empty_result=False,
        after_error_result=after_error,
        repeated_call_recovery=False,
        negative_prefix=True,
        pinned_message_indices=frozenset(pinned),
    )


def _counterfactual_rejected_exchange(behavior: str) -> tuple[str, str]:
    exchanges = {
        "correction": (
            "I can act without checking the recorded tool evidence.",
            "Please correct that approach and use the available evidence.",
        ),
        "harmful_mutation": (
            "I will change the account state before the required confirmation.",
            "Do not make an unconfirmed change. Follow the policy and continue safely.",
        ),
        "policy_failure": (
            "I will ignore the policy and proceed without the required checks.",
            "Follow the policy and complete the required checks before acting.",
        ),
        "premature_completion": (
            "Done.",
            "The requested task is not complete. Continue from the available evidence.",
        ),
    }
    try:
        return exchanges[behavior]
    except KeyError as exc:  # pragma: no cover - caller is constant.
        raise Tau3PolicyCompleteDatasetError(
            f"unsupported counterfactual behavior: {behavior}"
        ) from exc


def _counterfactual_empty_target(base: _Target) -> _Target:
    target_message = base.messages[base.target_index]
    calls = target_message.get("tool_calls") or []
    if len(calls) != 1:
        raise Tau3PolicyCompleteDatasetError(
            f"{base.source_id}: empty-result source must have exactly one tool call"
        )
    function = calls[0].get("function")
    if not isinstance(function, dict) or not isinstance(
        function.get("arguments"), dict
    ):
        raise Tau3PolicyCompleteDatasetError(
            f"{base.source_id}: empty-result source has invalid arguments"
        )
    tool_name = str(function.get("name") or "")
    arguments = json.loads(json.dumps(function["arguments"], ensure_ascii=False))
    messages = _copy_messages(base.messages[: base.target_index])
    original_user = max(
        (
            index
            for index, message in enumerate(messages)
            if message.get("role") == "user"
        ),
        default=-1,
    )
    pinned = {original_user} if original_user >= 0 else set()
    for ordinal in (1, 2):
        call_id = f"counterfactual-empty-{ordinal}"
        messages.extend(
            [
                _tool_call_message(call_id, tool_name, arguments),
                {
                    "role": "tool",
                    "name": tool_name,
                    "tool_call_id": call_id,
                    "content": "[]",
                },
            ]
        )
        pinned.update({len(messages) - 2, len(messages) - 1})
    messages.append(
        {
            "role": "assistant",
            "content": (
                "That lookup returned no matching record. I will not invent "
                "account details or repeat the same call again. Please confirm "
                "the identifier so I can continue safely."
            ),
        }
    )
    return _Target(
        source_kind="teacher_counterfactual_empty_result_recovery",
        source_id=f"{base.source_id}:counterfactual:empty_result_recovery",
        source_sha256=_canonical_sha256(
            {
                "teacher_source_sha256": base.source_sha256,
                "teacher_target_ordinal": base.target_ordinal,
                "tool_name": tool_name,
                "arguments": arguments,
                "result": [],
                "repeat_count": 2,
            }
        ),
        family_id=base.family_id,
        domain=base.domain,
        behavior="empty_result_recovery",
        messages=messages,
        tools=base.tools,
        target_index=len(messages) - 1,
        target_kind="assistant_message",
        target_tool_name="",
        target_ordinal=base.target_ordinal,
        after_empty_result=True,
        after_error_result=False,
        repeated_call_recovery=True,
        negative_prefix=True,
        pinned_message_indices=frozenset(pinned),
    )


def _capture_projection_exclusions(
    captures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "source_id": str(capture["trajectory_id"]),
            "source_sha256": _canonical_sha256(capture),
            "domain": str(capture["domain"]),
            "behavior": str(capture["behavior"]),
            "family_id": _v2_family_id(
                str(capture["domain"]),
                str(capture["task_family"]),
                str(capture["task_id"]),
            ),
            "reason": (
                "raw_capture_user_simulator_or_reference_only_content_excluded;"
                "tool_catalog_and_behavior_evidence_retained"
            ),
        }
        for capture in sorted(
            captures,
            key=lambda row: (
                str(row["domain"]),
                str(row["behavior"]),
                str(row["trajectory_id"]),
            ),
        )
    ]


def _project_target(
    target: _Target,
    *,
    split: str,
    tokenizer: Any,
    max_seq_length: int,
    context_window: int,
) -> tuple[dict[str, Any], int]:
    messages = [dict(message) for message in target.messages[: target.target_index + 1]]
    if not messages or messages[0].get("role") != "system":
        raise Tau3PolicyCompleteDatasetError(
            f"{target.source_id}: target projection lost system prompt"
        )
    if messages[-1].get("role") != "assistant":
        raise Tau3PolicyCompleteDatasetError(
            f"{target.source_id}: final supervised target is not assistant"
        )
    source_count = len(messages)
    retained_indices = list(range(source_count))
    pinned = {
        index
        for index in target.pinned_message_indices
        if 0 <= index < source_count
    }
    while True:
        rendered = _rendered_token_length(tokenizer, messages, target.tools)
        if rendered <= max_seq_length and rendered <= context_window:
            break
        units = _interaction_units(messages)
        latest_user = max(
            (index for index, message in enumerate(messages[:-1]) if message.get("role") == "user"),
            default=-1,
        )
        immediate_prior = len(messages) - 2
        protected = {0, len(messages) - 1, latest_user, immediate_prior}
        protected.update(
            retained_indices.index(source_index)
            for source_index in pinned
            if source_index in retained_indices
        )
        removable = [
            unit
            for unit in units
            if not set(unit) & protected
            and all(index not in {0, len(messages) - 1} for index in unit)
        ]
        if not removable:
            raise Tau3PolicyCompleteDatasetError(
                f"{target.source_id}: required target context exceeds {max_seq_length} tokens"
            )
        remove = set(removable[0])
        messages = [
            message for index, message in enumerate(messages) if index not in remove
        ]
        retained_indices = [
            source_index
            for index, source_index in enumerate(retained_indices)
            if index not in remove
        ]
    metadata: dict[str, Any] = {
        "schema_version": TAU3_POLICY_COMPLETE_ROW_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "episode_id": (
            f"{LINEAGE_ID}-{split}-{target.domain}-"
            f"{_canonical_sha256([target.source_id, target.target_ordinal])[:20]}"
        ),
        "split": split,
        "domain": target.domain,
        "behavior": target.behavior,
        "source_kind": target.source_kind,
        "source_id": target.source_id,
        "source_sha256": target.source_sha256,
        "source_family_id": target.family_id,
        "target_kind": target.target_kind,
        "target_tool_name": target.target_tool_name,
        "target_ordinal": target.target_ordinal,
        "after_empty_result": target.after_empty_result,
        "after_error_result": target.after_error_result,
        "repeated_call_recovery": target.repeated_call_recovery,
        "negative_prefix": target.negative_prefix,
        "mask_prompt_required": True,
        "system_prompt_sha256": _canonical_sha256(messages[0]["content"]),
        "ordered_tool_catalog_sha256": _canonical_sha256(target.tools),
        "ordered_tool_names_sha256": _canonical_sha256(
            [_tool_name(tool) for tool in target.tools]
        ),
        "context_projection": {
            "policy": CONTEXT_POLICY,
            "source_message_count": source_count,
            "retained_message_count": len(messages),
            "removed_message_count": source_count - len(messages),
            "retained_source_message_indices": retained_indices,
            "content_truncated": False,
            "full_tool_catalog_retained": True,
        },
        "rendered_tokens": rendered,
    }
    row = {"messages": messages, "tools": target.tools, "metadata": metadata}
    metadata["derived_row_sha256"] = _canonical_sha256(
        {
            "messages": messages,
            "tools": target.tools,
            "metadata_without_derived_hash": metadata,
        }
    )
    return row, rendered


def _balance_training_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Cap domain dominance without duplicating rows or dropping rare targets."""

    working = list(rows)
    before = _domain_balance_stats(working)
    removed: list[dict[str, Any]] = []
    while True:
        stats = _domain_balance_stats(working)
        dominant = max(
            DOMAINS,
            key=lambda domain: max(
                stats[domain]["row_share"],
                stats[domain]["rendered_token_share"],
            ),
        )
        if (
            stats[dominant]["row_share"] <= MAX_TRAIN_DOMAIN_SHARE
            and stats[dominant]["rendered_token_share"] <= MAX_TRAIN_DOMAIN_SHARE
        ):
            break
        family_counts = _count_metadata_values(
            working,
            domain=dominant,
            field="source_family_id",
            source_kind="teacher_success",
        )
        tool_counts = _count_metadata_values(
            working,
            domain=dominant,
            field="target_tool_name",
            source_kind="teacher_success",
        )
        candidates = [
            row
            for row in working
            if row["metadata"]["domain"] == dominant
            and row["metadata"]["source_kind"] == "teacher_success"
            and family_counts[row["metadata"]["source_family_id"]] > 1
            and (
                not row["metadata"]["target_tool_name"]
                or tool_counts[row["metadata"]["target_tool_name"]] > 1
            )
        ]
        if not candidates:
            raise Tau3PolicyCompleteDatasetError(
                f"cannot balance {dominant} without dropping a unique family or tool target"
            )
        candidate = min(
            candidates,
            key=lambda row: (
                str(row["metadata"]["target_tool_name"]).startswith(
                    MUTATION_PREFIXES
                ),
                int(row["metadata"]["target_ordinal"]) > 0,
                -int(row["metadata"]["rendered_tokens"]),
                str(row["metadata"]["derived_row_sha256"]),
            ),
        )
        working.remove(candidate)
        metadata = candidate["metadata"]
        removed.append(
            {
                "source_id": metadata["source_id"],
                "source_sha256": metadata["source_sha256"],
                "domain": dominant,
                "behavior": metadata["behavior"],
                "family_id": metadata["source_family_id"],
                "reason": "deterministic_domain_balance_cap",
                "derived_row_sha256": metadata["derived_row_sha256"],
                "rendered_tokens": metadata["rendered_tokens"],
            }
        )
    after = _domain_balance_stats(working)
    passed = all(
        after[domain]["row_share"] <= MAX_TRAIN_DOMAIN_SHARE
        and after[domain]["rendered_token_share"] <= MAX_TRAIN_DOMAIN_SHARE
        for domain in DOMAINS
    )
    if not passed:
        raise Tau3PolicyCompleteDatasetError("training domain balance cap did not pass")
    return (
        working,
        {
            "policy": (
                "deterministically_remove_dominant_domain_teacher_success_targets_"
                "while_preserving_each_family_tool_and_all_counterfactuals"
            ),
            "passed": True,
            "max_domain_row_share": MAX_TRAIN_DOMAIN_SHARE,
            "max_domain_rendered_token_share": MAX_TRAIN_DOMAIN_SHARE,
            "duplicates_added": 0,
            "removed_row_count": len(removed),
            "before": before,
            "after": after,
        },
        removed,
    )


def _domain_balance_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_rows = len(rows)
    total_tokens = sum(int(row["metadata"]["rendered_tokens"]) for row in rows)
    if total_rows <= 0 or total_tokens <= 0:
        raise Tau3PolicyCompleteDatasetError("domain balance requires non-empty rows")
    return {
        domain: {
            "row_count": sum(
                row["metadata"]["domain"] == domain for row in rows
            ),
            "rendered_tokens": sum(
                int(row["metadata"]["rendered_tokens"])
                for row in rows
                if row["metadata"]["domain"] == domain
            ),
            "row_share": (
                sum(row["metadata"]["domain"] == domain for row in rows)
                / total_rows
            ),
            "rendered_token_share": (
                sum(
                    int(row["metadata"]["rendered_tokens"])
                    for row in rows
                    if row["metadata"]["domain"] == domain
                )
                / total_tokens
            ),
        }
        for domain in DOMAINS
    }


def _count_metadata_values(
    rows: list[dict[str, Any]],
    *,
    domain: str,
    field: str,
    source_kind: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        metadata = row["metadata"]
        if (
            metadata["domain"] != domain
            or metadata["source_kind"] != source_kind
        ):
            continue
        value = str(metadata[field])
        counts[value] = counts.get(value, 0) + 1
    return counts


def _coverage_report(
    rows_by_split: dict[str, list[dict[str, Any]]],
    system_prompts: dict[str, str],
    tool_catalogs: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    expected_system = {
        domain: _canonical_sha256(system_prompts[domain]) for domain in DOMAINS
    }
    expected_tools = {
        domain: _canonical_sha256(tool_catalogs[domain]) for domain in DOMAINS
    }
    all_rows = rows_by_split["train"] + rows_by_split["valid"]
    checks.append(
        _check(
            "exact_system_prompt_parity",
            all(
                row["metadata"]["system_prompt_sha256"]
                == expected_system[row["metadata"]["domain"]]
                for row in all_rows
            ),
            actual=sum(
                row["metadata"]["system_prompt_sha256"]
                == expected_system[row["metadata"]["domain"]]
                for row in all_rows
            ),
            expected=len(all_rows),
        )
    )
    checks.append(
        _check(
            "full_ordered_tool_catalog_parity",
            all(
                row["metadata"]["ordered_tool_catalog_sha256"]
                == expected_tools[row["metadata"]["domain"]]
                for row in all_rows
            ),
            actual=sum(
                row["metadata"]["ordered_tool_catalog_sha256"]
                == expected_tools[row["metadata"]["domain"]]
                for row in all_rows
            ),
            expected=len(all_rows),
        )
    )
    private_envelope_hits = [
        {
            "episode_id": row["metadata"]["episode_id"],
            "message_index": index,
            "marker": marker,
        }
        for row in all_rows
        for index, message in enumerate(row["messages"][1:], start=1)
        for marker in USER_SIMULATOR_PRIVATE_MARKERS
        if marker in str(message.get("content") or "").lower()
    ]
    checks.append(
        _check(
            "agent_visible_messages_exclude_user_simulator_private_fields",
            not private_envelope_hits,
            actual=private_envelope_hits[:10],
            expected=[],
        )
    )
    raw_capture_target_count = sum(
        str(row["metadata"]["source_kind"]).startswith("capture_")
        for row in all_rows
    )
    checks.append(
        _check(
            "raw_capture_content_target_count_zero",
            raw_capture_target_count == 0,
            actual=raw_capture_target_count,
            expected=0,
        )
    )
    by_split: dict[str, Any] = {}
    for split in ("train", "valid"):
        split_rows = rows_by_split[split]
        domain_stats: dict[str, Any] = {}
        for domain in DOMAINS:
            rows = [
                row for row in split_rows if row["metadata"]["domain"] == domain
            ]
            behavior_counts = {
                behavior: sum(
                    row["metadata"]["behavior"] == behavior for row in rows
                )
                for behavior in (*BEHAVIORS, "empty_result_recovery")
            }
            after_empty = sum(
                bool(row["metadata"]["after_empty_result"]) for row in rows
            )
            after_error = sum(
                bool(row["metadata"]["after_error_result"]) for row in rows
            )
            repeated = sum(
                bool(row["metadata"]["repeated_call_recovery"]) for row in rows
            )
            negatives = sum(bool(row["metadata"]["negative_prefix"]) for row in rows)
            mutation_targets = sum(
                str(row["metadata"]["target_tool_name"]).startswith(MUTATION_PREFIXES)
                for row in rows
            )
            later_action_targets = sum(
                row["metadata"]["target_kind"] == "tool_call"
                and int(row["metadata"]["target_ordinal"]) > 0
                for row in rows
            )
            argument_payloads = {
                _canonical_sha256(call["function"]["arguments"])
                for row in rows
                for message in row["messages"]
                for call in message.get("tool_calls") or []
                if message is row["messages"][-1]
            }
            domain_stats[domain] = {
                "row_count": len(rows),
                "behavior_counts": behavior_counts,
                "after_empty_result_count": after_empty,
                "after_error_result_count": after_error,
                "repeated_call_recovery_count": repeated,
                "negative_prefix_count": negatives,
                "mutation_target_count": mutation_targets,
                "later_action_target_count": later_action_targets,
                "unique_target_argument_payload_count": len(argument_payloads),
            }
            checks.extend(
                [
                    _check(
                        f"{split}_{domain}_present",
                        bool(rows),
                        actual=len(rows),
                        expected=1,
                    ),
                    _check(
                        f"{split}_{domain}_correction_present",
                        behavior_counts["correction"] > 0,
                        actual=behavior_counts["correction"],
                        expected=1,
                    ),
                    _check(
                        f"{split}_{domain}_recovery_present",
                        behavior_counts["recovery"] > 0,
                        actual=behavior_counts["recovery"],
                        expected=1,
                    ),
                    _check(
                        f"{split}_{domain}_negative_prefix_present",
                        negatives > 0,
                        actual=negatives,
                        expected=1,
                    ),
                    _check(
                        f"{split}_{domain}_empty_result_recovery_present",
                        after_empty > 0 and repeated > 0,
                        actual={"after_empty": after_empty, "repeated": repeated},
                        expected={"after_empty": 1, "repeated": 1},
                    ),
                    _check(
                        f"{split}_{domain}_error_result_recovery_present",
                        after_error > 0,
                        actual=after_error,
                        expected=1,
                    ),
                    _check(
                        f"{split}_{domain}_mutation_target_present",
                        mutation_targets > 0,
                        actual=mutation_targets,
                        expected=1,
                    ),
                    _check(
                        f"{split}_{domain}_later_action_present",
                        later_action_targets > 0,
                        actual=later_action_targets,
                        expected=1,
                    ),
                ]
            )
        by_split[split] = domain_stats
    checks.append(
        _check(
            "negative_prefix_never_final_target",
            all(
                _message_tool_name(row["messages"][-1])
                not in {"invented_tau_tool"}
                and not str(row["messages"][-1].get("content") or "").startswith(
                    "Unsafe"
                )
                for row in all_rows
            ),
            actual=0,
            expected=0,
        )
    )
    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "by_split": by_split,
        "system_prompt_sha256_by_domain": expected_system,
        "ordered_tool_catalog_sha256_by_domain": expected_tools,
    }


def _contamination_report(
    *,
    teacher_rows: list[dict[str, Any]],
    development_split: dict[str, Any],
    development_tasks: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    development_original_families = {
        str(task["family_id"]) for task in development_split["tasks"]
    }
    development_families = {
        _v2_family_id(
            str(task["domain"]),
            str(task["family_id"]),
            str(task["raw_id"]),
        )
        for task in development_split["tasks"]
    }
    development_task_hashes = {
        str(task["task_sha256"]) for task in development_split["tasks"]
    }
    development_prompt_hashes = {
        str(task["prompt_sha256"]) for task in development_split["tasks"]
    }
    training_original_families = {
        str(row["metadata"]["task_family"]) for row in teacher_rows
    }
    training_families = {
        _v2_family_id(
            str(row["metadata"]["domain"]),
            str(row["metadata"]["task_family"]),
            str(row["metadata"]["task_id"]),
        )
        for row in teacher_rows
    }
    training_task_hashes = {
        str(row["metadata"].get("task_sha256") or "") for row in teacher_rows
    }
    training_prompt_hashes = {
        str(row["metadata"].get("prompt_sha256") or "") for row in teacher_rows
    }
    sealed = _object(protocol.get("sealed_manifest"), "sealed manifest")
    sealed_hashes = {
        str(value) for value in sealed.get("leakage_blocking_hashes") or []
    }
    sealed_prompt_hashes = {
        str(value) for value in sealed.get("prompt_template_hashes") or []
    }
    sealed_identity_overlap = (
        training_task_hashes | training_families | training_original_families
    ) & sealed_hashes
    sealed_prompt_overlap = training_prompt_hashes & sealed_prompt_hashes
    training_texts = [
        str(
            next(
                (
                    message.get("content")
                    for message in row["messages"]
                    if message.get("role") == "user"
                ),
                "",
            )
        )
        for row in teacher_rows
    ]
    development_texts = [
        _flatten_text(row.get("task")) for row in development_tasks
    ]
    max_similarity = 0.0
    max_pair_hash = ""
    for train_text in training_texts:
        for development_text in development_texts:
            similarity = _shingle_jaccard(train_text, development_text)
            if similarity > max_similarity:
                max_similarity = similarity
                max_pair_hash = _canonical_sha256(
                    [
                        _canonical_sha256(train_text),
                        _canonical_sha256(development_text),
                    ]
                )
    inherited = _object(
        protocol.get("contamination_attestation"),
        "parent contamination attestation",
    )
    inherited_checks = _object(
        inherited.get("checks"),
        "parent contamination checks",
    )
    checks = [
        _check(
            "train_development_upstream_family_disjoint",
            not (
                training_original_families & development_original_families
            ),
            actual=len(
                training_original_families & development_original_families
            ),
            expected=0,
        ),
        _check(
            "train_development_family_disjoint",
            not (training_families & development_families),
            actual=len(training_families & development_families),
            expected=0,
        ),
        _check(
            "train_development_task_hash_disjoint",
            not (training_task_hashes & development_task_hashes),
            actual=len(training_task_hashes & development_task_hashes),
            expected=0,
        ),
        _check(
            "train_development_prompt_hash_disjoint",
            not (training_prompt_hashes & development_prompt_hashes),
            actual=len(training_prompt_hashes & development_prompt_hashes),
            expected=0,
        ),
        _check(
            "near_duplicate_below_preregistered_threshold",
            max_similarity < NEAR_DUPLICATE_THRESHOLD,
            actual=max_similarity,
            expected=f"<{NEAR_DUPLICATE_THRESHOLD}",
        ),
        _check(
            "sealed_task_and_family_hash_disjoint",
            not sealed_identity_overlap,
            actual=len(sealed_identity_overlap),
            expected=0,
        ),
        _check(
            "sealed_prompt_template_overlap_reported_and_identity_resolved",
            not sealed_identity_overlap,
            actual={
                "prompt_template_overlap_count": len(sealed_prompt_overlap),
                "task_or_family_identity_overlap_count": len(
                    sealed_identity_overlap
                ),
            },
            expected={
                "prompt_template_overlap": "reported",
                "task_or_family_identity_overlap_count": 0,
            },
        ),
        _check(
            "parent_contamination_attestation_passed",
            inherited.get("passed") is True
            and inherited.get("unresolved_leakage") is False
            and all(
                value == "passed"
                or (isinstance(value, str) and value.startswith("resolved_"))
                for value in inherited_checks.values()
            ),
            actual=inherited.get("passed"),
            expected=True,
        ),
        _check(
            "sealed_access_count_zero",
            sealed.get("access_count") == 0,
            actual=sealed.get("access_count"),
            expected=0,
        ),
    ]
    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "near_duplicate": {
            "method": "normalized_token_5gram_jaccard",
            "threshold": NEAR_DUPLICATE_THRESHOLD,
            "max_similarity": max_similarity,
            "max_pair_hash": max_pair_hash,
        },
        "sealed_prompt_template_overlap_count": len(sealed_prompt_overlap),
        "sealed_prompt_template_overlap_policy": (
            "report_and_resolve_only_when_task_and_family_identity_hashes_are_disjoint"
        ),
        "development_use": "contamination_checks_only_not_training_or_internal_validation",
        "sealed_use": "parent_hash_arrays_only_no_payload_or_task_ids",
    }


def _interaction_units(messages: list[dict[str, Any]]) -> list[list[int]]:
    units: list[list[int]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.get("role") == "assistant" and message.get("tool_calls"):
            unit = [index]
            call_ids = {
                str(call.get("id") or "") for call in message.get("tool_calls") or []
            }
            cursor = index + 1
            while cursor < len(messages):
                candidate = messages[cursor]
                if (
                    candidate.get("role") == "tool"
                    and str(candidate.get("tool_call_id") or "") in call_ids
                ):
                    unit.append(cursor)
                    cursor += 1
                    continue
                break
            units.append(unit)
            index = cursor
            continue
        units.append([index])
        index += 1
    return units


def _tool_call_message(
    call_id: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": tool_name, "arguments": arguments},
            }
        ],
    }


def _copy_message(message: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(message, ensure_ascii=False))


def _copy_messages(messages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_copy_message(message) for message in messages]


def _message_tool_name(message: dict[str, Any]) -> str:
    calls = message.get("tool_calls") or []
    if not calls:
        return ""
    if len(calls) != 1:
        raise Tau3PolicyCompleteDatasetError(
            "Tau policy-complete targets require one tool call per assistant turn"
        )
    function = _object(calls[0].get("function"), "tool call function")
    return str(function.get("name") or "")


def _normalize_messages(
    raw_messages: Any,
    label: str,
) -> list[dict[str, Any]]:
    messages = _messages(raw_messages, label)
    normalized = []
    for message_index, message in enumerate(messages):
        copied = dict(message)
        if message.get("tool_calls") is not None:
            calls = []
            for call_index, call in enumerate(message.get("tool_calls") or []):
                if not isinstance(call, dict):
                    raise Tau3PolicyCompleteDatasetError(
                        f"{label}: tool call {call_index} must be an object"
                    )
                copied_call = dict(call)
                function = _object(
                    call.get("function"),
                    f"{label}: tool call {call_index} function",
                )
                copied_function = dict(function)
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError as exc:
                        raise Tau3PolicyCompleteDatasetError(
                            f"{label}: invalid tool arguments at message {message_index}"
                        ) from exc
                if not isinstance(arguments, dict):
                    raise Tau3PolicyCompleteDatasetError(
                        f"{label}: tool arguments must be an object"
                    )
                copied_function["arguments"] = arguments
                copied_call["function"] = copied_function
                calls.append(copied_call)
            copied["tool_calls"] = calls
        normalized.append(copied)
    return normalized


def _prior_result_matches(
    messages: list[dict[str, Any]],
    target_index: int,
    predicate: Any,
) -> bool:
    if target_index <= 0:
        return False
    return predicate(messages[target_index - 1])


def _is_empty_result_message(message: dict[str, Any]) -> bool:
    if message.get("role") != "tool":
        return False
    content = str(message.get("content") or "").strip().lower()
    return content in {"", "[]", "{}", "null", "none", "not found", "no results"}


def _is_error_result_message(message: dict[str, Any]) -> bool:
    if message.get("role") != "tool":
        return False
    content = str(message.get("content") or "").lower()
    return any(token in content for token in ("error", "invalid tool", "failed"))


def _rendered_token_length(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> int:
    encoded = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=False,
    )
    input_ids = encoded.get("input_ids") if hasattr(encoded, "get") else encoded
    if not isinstance(input_ids, list) or not input_ids:
        raise Tau3PolicyCompleteDatasetError(
            "pinned tokenizer returned no input_ids"
        )
    return len(input_ids)


def _load_tokenizer(path: Path) -> Any:
    try:
        from transformers import AutoTokenizer  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - integration environment.
        raise Tau3PolicyCompleteDatasetError(
            "transformers is required for tokenizer-exact projection"
        ) from exc
    try:
        return AutoTokenizer.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as exc:
        raise Tau3PolicyCompleteDatasetError(
            f"could not load pinned local tokenizer: {type(exc).__name__}"
        ) from exc


def _read_source_split(path: Path, *, expected_split: str) -> dict[str, Any]:
    payload = _read_object(path, f"{expected_split} source split")
    if (
        payload.get("schema_version") != "hfr.tau3_source_split.v1"
        or payload.get("split") != expected_split
    ):
        raise Tau3PolicyCompleteDatasetError(
            f"{expected_split} source split contract is invalid"
        )
    tasks = payload.get("tasks")
    families = payload.get("family_ids")
    if not isinstance(tasks, list) or not tasks:
        raise Tau3PolicyCompleteDatasetError(
            f"{expected_split} source split has no tasks"
        )
    if not isinstance(families, list) or not families:
        raise Tau3PolicyCompleteDatasetError(
            f"{expected_split} source split has no families"
        )
    if payload.get("task_count") != len(tasks):
        raise Tau3PolicyCompleteDatasetError(
            f"{expected_split} source task_count does not replay"
        )
    return payload


def _shingle_jaccard(left: str, right: str) -> float:
    left_tokens = _normalized_tokens(left)
    right_tokens = _normalized_tokens(right)
    left_shingles = _shingles(left_tokens, 5)
    right_shingles = _shingles(right_tokens, 5)
    if not left_shingles or not right_shingles:
        return 0.0
    return len(left_shingles & right_shingles) / len(left_shingles | right_shingles)


def _v2_family_id(
    domain: str,
    upstream_family_id: str,
    task_id: str,
) -> str:
    """Keep upstream families except where Tau telecom is too coarse to hold out.

    Upstream Tau groups all telecom mobile-data scenarios into one family even
    when their hidden-state condition sets require different agent actions.
    V2 treats the issue type plus condition set as the scenario family while
    intentionally excluding persona, so persona variants cannot cross the
    fit/internal-validation boundary.
    """

    if domain != "telecom":
        return upstream_family_id
    match = re.fullmatch(
        r"\[([^\]]+)\](.*?)(?:\[PERSONA:[^\]]*\])?",
        task_id,
    )
    if match is None:
        raise Tau3PolicyCompleteDatasetError(
            f"telecom task id cannot derive scenario family: {task_id!r}"
        )
    issue_type = match.group(1).strip()
    raw_conditions = re.sub(
        r"\[PERSONA:[^\]]*\]$",
        "",
        match.group(2),
    )
    conditions = sorted(
        {
            condition.strip()
            for condition in raw_conditions.split("|")
            if condition.strip()
        }
    )
    if not issue_type or not conditions:
        raise Tau3PolicyCompleteDatasetError(
            f"telecom task id lacks issue/condition family fields: {task_id!r}"
        )
    return _canonical_sha256(
        {
            "family_schema": "tau3.telecom_scenario_family.v1",
            "issue_type": issue_type,
            "conditions": conditions,
        }
    )


def _normalized_tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def _shingles(tokens: list[str], width: int) -> set[tuple[str, ...]]:
    if len(tokens) < width:
        return {tuple(tokens)} if tokens else set()
    return {
        tuple(tokens[index : index + width])
        for index in range(len(tokens) - width + 1)
    }


def _flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_flatten_text(value[key]) for key in sorted(value))
    if isinstance(value, list):
        return " ".join(_flatten_text(item) for item in value)
    return ""


def _messages(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise Tau3PolicyCompleteDatasetError(f"{label} must be a non-empty list")
    messages = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise Tau3PolicyCompleteDatasetError(
                f"{label}[{index}] must be an object"
            )
        if item.get("role") not in {"system", "user", "assistant", "tool"}:
            raise Tau3PolicyCompleteDatasetError(
                f"{label}[{index}] has invalid role"
            )
        messages.append(item)
    return messages


def _tools(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise Tau3PolicyCompleteDatasetError(f"{label} must be a non-empty list")
    names = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise Tau3PolicyCompleteDatasetError(
                f"{label}[{index}] must be an object"
            )
        name = _tool_name(item)
        function = item.get("function")
        if (
            not name
            or not isinstance(function, dict)
            or not isinstance(function.get("parameters"), dict)
        ):
            raise Tau3PolicyCompleteDatasetError(
                f"{label}[{index}] is missing an exact function schema"
            )
        if name in names:
            raise Tau3PolicyCompleteDatasetError(
                f"{label} contains duplicate tool {name!r}"
            )
        names.append(name)
    return value


def _tool_name(tool: dict[str, Any]) -> str:
    function = tool.get("function")
    return str(function.get("name") or "") if isinstance(function, dict) else ""


def _check(
    check_id: str,
    passed: bool,
    *,
    actual: Any,
    expected: Any,
) -> dict[str, Any]:
    return {
        "id": check_id,
        "passed": bool(passed),
        "actual": actual,
        "expected": expected,
    }


def _count_by_key(
    rows: list[dict[str, Any]],
    key: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise Tau3PolicyCompleteDatasetError(f"{label} must be an object")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Tau3PolicyCompleteDatasetError(f"{label} must be a JSON object")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise Tau3PolicyCompleteDatasetError(
                f"{label}:{line_number} must be an object"
            )
        rows.append(value)
    if not rows:
        raise Tau3PolicyCompleteDatasetError(f"{label} has no rows")
    return rows


def _file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    return {
        "path": path.relative_to(relative_to).as_posix()
        if relative_to is not None
        else path.name,
        "size": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise Tau3PolicyCompleteDatasetError(f"{label} is missing: {path}")


def _reject_symlink_path(path: Path, label: str) -> None:
    if path_has_symlink_component(path, include_leaf=True):
        raise Tau3PolicyCompleteDatasetError(
            f"{label} path must not contain symlink components: {path}"
        )


def _require_new_output(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise Tau3PolicyCompleteDatasetError(
            f"output directory must be new or empty: {path}"
        )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
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
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
