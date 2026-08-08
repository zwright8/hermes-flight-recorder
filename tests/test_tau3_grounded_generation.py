from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import unittest
from datetime import date, datetime, time
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

from flightrecorder.schema_registry import check_schema_contract
from flightrecorder.tau3_grounded_generation import (
    BEHAVIORS,
    CONTENT_ADDRESSED_FAMILY_SEMANTICS,
    GENERATED_FAMILY_SCHEMA_VERSION,
    GENERATION_STRATUM_SCHEMA_VERSION,
    LEGACY_COVERAGE_PROFILE_ID,
    LINEAGE_ID,
    SCALED_COVERAGE_PROFILE_ID,
    SCALED_RANK_TIE_BREAKER_CONTRACT,
    SCALED_SELECTION_ALGORITHM,
    SCALED_SELECTION_RECEIPT_SCHEMA_VERSION,
    SELECTION_STRATUM_SCHEMA_VERSION,
    TAU3_GROUNDED_DATASET_SCHEMA_VERSION,
    TOOL_EXEMPTION_SCHEMA_VERSION,
    TOOL_EXEMPTION_REVIEW_INFERENCE_SCHEMA_VERSION,
    TOOL_EXEMPTION_REVIEW_SCHEMA_VERSION,
    TRAINING_HANDOFF_SCHEMA_VERSION,
    Tau3GroundedGenerationError,
    _FakeTestTauRuntime,
    _VendoredTauRuntime,
    _argument_grounding_evidence,
    _confirmation_detail_grounding,
    _confirmation_rule,
    _confirmed_argument_receipt,
    _coverage,
    _coverage_profile_record,
    _dominance_record,
    _fraction_at_least,
    _fraction_at_most,
    _generation_stratum_definition,
    _generation_provenance_errors,
    _install_tau_import_shims,
    _meets_minimum,
    _manifest,
    _model_to_json,
    _ordered_initial_sync_evidence,
    _policy_call_errors,
    _ratio_at_most,
    _scaled_selection_claim_errors,
    _scaled_tool_exemption_errors,
    _selection_profile_errors,
    _selection_receipt,
    _selection_receipt_errors,
    _state_diff,
    _target_for_decision,
    _tool_result_class,
    _validation_result,
    build_tau3_grounded_generation_dataset,
    canonical_sha256,
    validate_tau3_grounded_generation_bundle,
    write_build_validate_tau3_grounded_generation_candidates,
)


TAU_REVISION = "a" * 40


class _UnitEnum(Enum):
    ACTIVE = "Active"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _test_reviewer_artifact(
    *,
    candidate: bool = False,
    generator_inference_receipt_sha256: str | None = None,
    policy_sha256: str | None = None,
) -> dict[str, Any]:
    inference_receipt = {
        "schema_version": "hfr.tau3_policy_review_inference_receipt.v1",
        "inference_origin": "native_codex" if candidate else "synthetic_test",
        "model": "gpt-5.6-sol" if candidate else "none",
        "reasoning_effort": "xhigh" if candidate else "none",
        "native_codex_inference_calls": 1 if candidate else 0,
        "provider_accessed": candidate,
        "network_accessed": candidate,
        "prohibited_external_model_provider_calls": 0,
        "prohibited_external_network_calls": 0,
    }
    return {
        "schema_version": "hfr.tau3_policy_reviewer_artifact.v1",
        "inference_receipt": inference_receipt,
        "reviewer_inference_receipt_sha256": canonical_sha256(inference_receipt),
        "generator_inference_receipt_sha256": (
            generator_inference_receipt_sha256
            or canonical_sha256("synthetic-generator-inference")
        ),
        "policy_sha256": policy_sha256 or canonical_sha256("fake-test-policy"),
        "independent_review_pass": True,
    }


def _test_reviewer_record(
    targets: list[dict[str, Any]],
    *,
    candidate: bool = False,
    generator_inference_receipt_sha256: str | None = None,
    policy_sha256: str | None = None,
) -> dict[str, Any]:
    artifact = _test_reviewer_artifact(
        candidate=candidate,
        generator_inference_receipt_sha256=generator_inference_receipt_sha256,
        policy_sha256=policy_sha256,
    )
    reviews = [
        copy.deepcopy(target["policy_review"])
        for target in targets
        if isinstance(target.get("policy_review"), dict)
    ]
    return {
        "id": "unit-reviewer",
        "sha256": canonical_sha256(artifact),
        "artifact": artifact,
        "review_set_sha256": canonical_sha256(reviews),
    }


def _attach_policy_review(
    target: dict[str, Any],
    *,
    turn_ordinal: int,
    decision_ordinal: int,
    allowed: bool,
    reason_id: str,
) -> None:
    canonical_target = {
        "kind": str(target.get("kind") or "assistant_message"),
        "text": str(target.get("text") or ""),
        "tool_name": target.get("tool_name")
        if isinstance(target.get("tool_name"), str)
        else None,
        "arguments": copy.deepcopy(target.get("arguments") or {}),
    }
    review = {
        "schema_version": "hfr.tau3_policy_review.v1",
        "reviewer": "unit-reviewer",
        "reviewer_artifact_sha256": canonical_sha256(_test_reviewer_artifact()),
        "policy_sha256": canonical_sha256("fake-test-policy"),
        "canonical_target_sha256": canonical_sha256(canonical_target),
        "parent_turn_ordinal": turn_ordinal,
        "parent_assistant_decision_ordinal": decision_ordinal,
        "allowed": allowed,
        "reason_id": reason_id,
    }
    review["review_receipt_sha256"] = canonical_sha256(review)
    target["policy_review"] = review


def _promote_candidate_reviewer(
    row: dict[str, Any],
    *,
    policy_sha256: str,
) -> None:
    generator_receipt_sha256 = row["generation_provenance"]["inference_receipt_sha256"]
    artifact = _test_reviewer_artifact(
        candidate=True,
        generator_inference_receipt_sha256=generator_receipt_sha256,
        policy_sha256=policy_sha256,
    )
    artifact_sha256 = canonical_sha256(artifact)
    targets: list[dict[str, Any]] = []
    for turn in row["turns"]:
        assistant = turn.get("assistant")
        if not isinstance(assistant, dict):
            continue
        target = assistant.get("safe_corrected_target")
        if not isinstance(target, dict):
            continue
        review = target.get("policy_review")
        if isinstance(review, dict):
            review["reviewer_artifact_sha256"] = artifact_sha256
            review["review_receipt_sha256"] = canonical_sha256(
                {
                    key: value
                    for key, value in review.items()
                    if key != "review_receipt_sha256"
                }
            )
        targets.append(target)
    row["reviewer"] = _test_reviewer_record(
        targets,
        candidate=True,
        generator_inference_receipt_sha256=generator_receipt_sha256,
        policy_sha256=policy_sha256,
    )


def _scenario(
    *,
    split: str,
    domain: str,
    family_index: int,
    behavior: str,
    source_id: str | None = None,
    unsafe: bool = False,
    fabricated_success: bool = False,
) -> dict[str, Any]:
    family_id = f"{split}-{domain}-family-{family_index}"
    row_id = source_id or f"{split}-{domain}-{family_index}-{behavior}"
    record_id = f"{domain}-{family_index}"
    visible_request = (
        f"Use record {record_id}. The requested literal values are updated, unsafe, and boom."
    )
    tool_calls = [{"tool_name": "get_record", "arguments": {"id": record_id}}]
    target_text = f"Handled {behavior} for {domain}."
    target: dict[str, Any] = {
        "behavior": behavior,
        "kind": "assistant_message",
        "text": target_text,
    }
    if behavior == "successful_completion":
        if fabricated_success:
            target["text"] = "Completed the account update successfully."
        else:
            tool_calls = [
                {
                    "tool_name": "update_record",
                    "arguments": {"id": record_id, "patch": {"status": "updated"}},
                }
            ]
            target = {
                "behavior": "later_task_completion_actions",
                "kind": "tool_call",
                "text": "",
                "tool_name": "update_record",
                "arguments": {"id": record_id, "patch": {"status": "updated"}},
            }
    elif behavior == "empty_result_recovery":
        tool_calls = [{"tool_name": "empty_search", "arguments": {}}]
    elif behavior == "error_result_recovery":
        tool_calls = [{"tool_name": "raise_tool_exception", "arguments": {"message": "boom"}}]
    elif behavior == "repeated_call_recovery":
        tool_calls = [
            {"tool_name": "get_record", "arguments": {"id": record_id}},
            {"tool_name": "get_record", "arguments": {"id": record_id}},
        ]
    elif behavior in {"harmful_mutation_correction", "premature_completion_correction"}:
        target.update(
            {
                "tool_name": "update_record",
                "arguments": {"id": record_id, "patch": {"status": "unsafe"}},
                "requires_confirmation": True,
            }
        )
        if unsafe:
            if behavior == "harmful_mutation_correction":
                target = {
                    "behavior": behavior,
                    "negative_behavior": "harmful_mutation",
                    "kind": "tool_call",
                    "text": "",
                    "tool_name": "update_record",
                    "arguments": {"id": record_id, "patch": {"status": "unsafe"}},
                    "unsafe": True,
                    "mask_reason": "unsafe_or_negative_action",
                    "reviewed": True,
                    "safe_correction_decision_ordinal": 1,
                }
            else:
                target = {
                    "behavior": behavior,
                    "negative_behavior": "premature_completion",
                    "kind": "assistant_message",
                    "text": "Done before verifying the required evidence.",
                    "unsafe": True,
                    "mask_reason": "unsafe_or_negative_action",
                    "reviewed": True,
                    "safe_correction_decision_ordinal": 1,
                }
            tool_calls = []
    if behavior in {
        "empty_result_recovery",
        "error_result_recovery",
        "repeated_call_recovery",
    }:
        first_call = copy.deepcopy(tool_calls[0])
        turns = [
            {
                "user": {"content": f"Please handle {behavior}."},
                "assistant": {
                    "decision_ordinal": 0,
                    "tool_calls": [first_call],
                },
            }
        ]
        if behavior == "repeated_call_recovery":
            turns.append(
                {
                    "assistant": {
                        "decision_ordinal": 1,
                        "tool_calls": [copy.deepcopy(first_call)],
                    },
                }
            )
        turns.append(
            {
                "assistant": {
                    "decision_ordinal": 2 if behavior == "repeated_call_recovery" else 1,
                    "tool_calls": [],
                    "safe_corrected_target": target,
                },
            }
        )
    else:
        turns = [
            {
                "user": {"content": f"Please handle {behavior}."},
                "assistant": {
                    "decision_ordinal": 0,
                    "tool_calls": tool_calls,
                    "safe_corrected_target": target,
                },
            }
        ]
    if unsafe:
        turns.append(
            {
                "user": {"content": "Do not take that action; use the safe correction."},
                "assistant": {
                    "decision_ordinal": 1,
                    "tool_calls": [],
                    "safe_corrected_target": {
                        "behavior": behavior,
                        "kind": "assistant_message",
                        "text": f"Safely corrected {behavior} for {domain}.",
                    },
                },
            }
        )
    elif behavior == "successful_completion" and not fabricated_success:
        turns.append(
            {
                "user": {"content": "The tool confirmed the requested state change."},
                "assistant": {
                    "decision_ordinal": 1,
                    "tool_calls": [],
                    "safe_corrected_target": {
                        "behavior": "successful_completion",
                        "kind": "assistant_message",
                        "text": "Completed the account update successfully.",
                    },
                },
            }
        )
    turns[0]["user"]["content"] = visible_request
    if behavior == "successful_completion" and not fabricated_success:
        mutation_turn = turns[0]
        mutation_turn["assistant"]["decision_ordinal"] = 1
        mutation_turn["user"] = {"content": "Yes, I confirm."}
        mutation_turn["user"]["content"] = "Yes."
        mutation_call = mutation_turn["assistant"]["tool_calls"][0]
        mutation_call["confirmation"] = {
            "policy_rule": "test.update_record.explicit_confirmation",
            "policy_sha256": canonical_sha256("fake-test-policy"),
            "request_decision_ordinal": 0,
            "arguments_sha256": canonical_sha256(mutation_call["arguments"]),
            "action_label": "update record",
            "required_details": [],
        }
        turns.insert(
            0,
            {
                "user": {"content": visible_request},
                "assistant": {
                    "decision_ordinal": 0,
                    "tool_calls": [],
                    "safe_corrected_target": {
                        "behavior": "confirmation_before_mutation",
                        "kind": "assistant_message",
                        "text": (
                            f"Please confirm update record: id {record_id}; "
                            "status updated."
                        ),
                    },
                },
            },
        )
        turns[-1]["assistant"]["decision_ordinal"] = 2
    for turn_ordinal, turn in enumerate(turns):
        assistant = turn.get("assistant")
        if not isinstance(assistant, dict):
            continue
        reviewed_target = assistant.get("safe_corrected_target")
        if not isinstance(reviewed_target, dict):
            continue
        decision_ordinal = int(assistant["decision_ordinal"])
        if reviewed_target.get("unsafe") is True:
            _attach_policy_review(
                reviewed_target,
                turn_ordinal=turn_ordinal,
                decision_ordinal=decision_ordinal,
                allowed=False,
                reason_id="synthetic_unsafe_action",
            )
        elif (
            reviewed_target.get("kind") == "tool_call"
            or (unsafe and turn_ordinal > 0)
        ):
            _attach_policy_review(
                reviewed_target,
                turn_ordinal=turn_ordinal,
                decision_ordinal=decision_ordinal,
                allowed=True,
                reason_id=(
                    "synthetic_safe_correction"
                    if unsafe
                    else "synthetic_allowed_mutation"
                ),
            )
    reviewer_targets = [
        reviewed_target
        for reviewed_turn in turns
        for reviewed_assistant in [reviewed_turn.get("assistant")]
        if isinstance(reviewed_assistant, dict)
        for reviewed_target in [reviewed_assistant.get("safe_corrected_target")]
        if isinstance(reviewed_target, dict)
    ]
    return {
        "trajectory_id": f"traj-{row_id}",
        "domain": domain,
        "split": split,
        "source_family": "reviewed_synthetic",
        "source_family_id": family_id,
        "source_id": row_id,
        "tau_revision": TAU_REVISION,
        "runtime_family": "fake_test_tau_tools",
        "system_prompt": f"Tau {domain} policy prompt.",
        "initial_state": {
            "records": {record_id: {"id": record_id, "status": "open"}},
            "notes": [],
        },
        "turns": turns,
        "recipe": {"id": "unit-recipe", "sha256": canonical_sha256("recipe")},
        "teacher": {"id": "unit-teacher", "sha256": canonical_sha256("teacher")},
        "reviewer": _test_reviewer_record(reviewer_targets),
        "redaction": {"passed": True, "method": "synthetic-no-pii"},
        "contamination": {
            "source_split": split,
            "raw_sealed_payload_read": False,
            "sealed_hash_only": True,
        },
    }


def _complete_source(path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for domain in ("airline", "retail", "telecom"):
        for split, family_count in (("train", 8), ("validation", 2)):
            for family_index in range(family_count):
                for behavior in BEHAVIORS:
                    rows.append(
                        _scenario(
                            split=split,
                            domain=domain,
                            family_index=family_index,
                            behavior=behavior,
                            unsafe=behavior in {"harmful_mutation_correction", "premature_completion_correction"},
                        )
                    )
    _write_jsonl(path, rows)


def _rewrite_bundle_manifest(bundle: Path) -> None:
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    for split in ("train", "validation"):
        path = bundle / f"{split}.jsonl"
        manifest["files"][split]["sha256"] = _sha256(path)
        manifest["files"][split]["bytes"] = path.stat().st_size
    manifest["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    _write_json(bundle / "manifest.json", manifest)


def _candidate_generation_provenance(
    row: dict[str, Any],
    selection_receipt: dict[str, Any],
) -> dict[str, Any]:
    inference_receipt = {
        "schema_version": "hfr.tau3_inference_receipt.v1",
        "inference_origin": "native_codex",
        "generator_model": "gpt-5.6-sol",
        "reasoning_effort": "xhigh",
        "native_codex_inference_calls": 1,
        "provider_accessed": True,
        "network_accessed": True,
        "prohibited_external_model_provider_calls": 0,
        "prohibited_external_network_calls": 0,
        "tau_revision": row["tau_revision"],
        "system_prompt_sha256": canonical_sha256(row["system_prompt"]),
        "tool_catalog_sha256": canonical_sha256(row["tool_catalog"]),
    }
    controller_receipt = {
        "schema_version": "hfr.tau3_generation_controller_receipt.v1",
        "controller_origin": "native_codex",
        "controller_model": "gpt-5.6-sol",
        "controller_reasoning_effort": "xhigh",
        "generator_inference_receipt_sha256": canonical_sha256(inference_receipt),
        "selection_receipt_sha256": selection_receipt["receipt_sha256"],
        "training_started": False,
        "scaled_generation_started": False,
        "test_split_payload_accessed": False,
        "prohibited_external_model_provider_calls": 0,
        "prohibited_external_network_calls": 0,
        "source_task_sha256": selection_receipt["task_sha256"],
    }
    return {
        "schema_version": "hfr.tau3_generation_provenance.v2",
        "inference_origin": "native_codex",
        "generator_model": "gpt-5.6-sol",
        "reasoning_effort": "xhigh",
        "native_codex_inference_used": True,
        "native_codex_inference_calls": 1,
        "provider_accessed": True,
        "network_accessed": True,
        "prohibited_external_model_provider_calls": 0,
        "prohibited_external_network_calls": 0,
        "inference_receipt": inference_receipt,
        "controller_receipt": controller_receipt,
        "inference_receipt_sha256": canonical_sha256(inference_receipt),
        "controller_receipt_sha256": canonical_sha256(controller_receipt),
        "generated_artifacts_redacted": True,
        "raw_session_publishable": False,
        "raw_session_owner_only": True,
        "raw_session_disposition": "not_retained",
    }


def _candidate_selection_receipt(row: dict[str, Any]) -> dict[str, Any]:
    family_id = row["source_family_id"]
    salt = "synthetic-selection-salt"
    salt_sha256 = hashlib.sha256(salt.encode("utf-8")).hexdigest()
    source_name = "train" if row["split"] == "train" else "development"
    source_file_path = (
        "local/tau3/source-v1/training_source/train_tasks.jsonl"
        if source_name == "train"
        else "local/tau3/source-v1/training_source/development_tasks.jsonl"
    )
    source_file_sha256 = canonical_sha256("synthetic-source-file")
    task = {
        "id": "synthetic-task-id",
        "initial_state": row["initial_state"]["task_initialization"],
        "user_scenario": {"instructions": "synthetic-instructions"},
    }
    task_sha256 = canonical_sha256(task)
    source_row = {
        "schema_version": "synthetic-source.v1",
        "domain": row["domain"],
        "split": source_name,
        "source_revision": row["tau_revision"],
        "task_family": family_id,
        "task": task,
        "task_sha256": task_sha256,
        "prompt_sha256": canonical_sha256(
            task["user_scenario"]["instructions"]
        ),
    }
    receipt = {
        "schema_version": "hfr.tau3_selection_receipt.v1",
        "algorithm": "sha256_ranked_deterministic_per_domain_split",
        "family_identifier": family_id,
        "family_identifier_semantics": "content_addressed_sha256",
        "source": source_name,
        "mapped_grounded_split": row["split"],
        "source_file_path": source_file_path,
        "source_file_sha256": source_file_sha256,
        "source_line_number": 1,
        "canonical_source_row_sha256": canonical_sha256(source_row),
        "task_id_sha256": hashlib.sha256(
            task["id"].encode("utf-8")
        ).hexdigest(),
        "prompt_sha256": source_row["prompt_sha256"],
        "source_family_sha256": family_id,
        "source_revision": row["tau_revision"],
        "selection_salt": salt,
        "selection_salt_sha256": salt_sha256,
        "task_sha256": task_sha256,
        "task_initial_state_sha256": canonical_sha256(row["initial_state"]["task_initialization"]),
        "selected_family_sha256": family_id,
    }
    canonical = json.dumps(
        source_row,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    receipt["selection_rank_sha256"] = hashlib.sha256(
        salt.encode("utf-8") + b"\0" + canonical
    ).hexdigest()
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def _candidate_selection_source(
    receipt: dict[str, Any],
    row: dict[str, Any],
) -> tuple[dict[str, dict[str, str]], tuple[tuple[int, dict[str, Any]], ...], str]:
    task = {
        "id": "synthetic-task-id",
        "initial_state": row["initial_state"]["task_initialization"],
        "user_scenario": {"instructions": "synthetic-instructions"},
    }
    source_row = {
        "schema_version": "synthetic-source.v1",
        "domain": row["domain"],
        "split": receipt["source"],
        "source_revision": row["tau_revision"],
        "task_family": row["source_family_id"],
        "task": task,
        "task_sha256": canonical_sha256(task),
        "prompt_sha256": canonical_sha256(
            task["user_scenario"]["instructions"]
        ),
    }
    specs = {
        row["split"]: {
            "source_split": receipt["source"],
            "path": receipt["source_file_path"],
            "sha256": receipt["source_file_sha256"],
        }
    }
    return specs, ((1, source_row),), hashlib.sha256(
        receipt["selection_salt"].encode("utf-8")
    ).hexdigest()


def _scaled_training_handoff() -> dict[str, Any]:
    tokenizer = {
        "identifier": "unit-tokenizer",
        "revision": "unit-revision",
    }
    tokenizer["identity_sha256"] = canonical_sha256(tokenizer)
    handoff = {
        "schema_version": TRAINING_HANDOFF_SCHEMA_VERSION,
        "tokenizer": tokenizer,
        "chat_template": {
            "identifier": "unit-chat-template",
            "sha256": canonical_sha256("unit-chat-template-content"),
        },
        "context_budget": 32768,
    }
    handoff["handoff_sha256"] = canonical_sha256(handoff)
    return handoff


def _scaled_source_task_row(
    *,
    domain: str,
    source: str,
    family_sha256: str,
    task_index: int,
    initial_state: Any,
    revision: str = TAU_REVISION,
) -> dict[str, Any]:
    task = {
        "id": f"task-{domain}-{task_index}",
        "initial_state": copy.deepcopy(initial_state),
        "user_scenario": {"instructions": f"prompt-{domain}-{task_index}"},
    }
    return {
        "schema_version": "synthetic-source.v1",
        "domain": domain,
        "split": source,
        "source_revision": revision,
        "task_family": family_sha256,
        "task": task,
        "task_sha256": canonical_sha256(task),
        "prompt_sha256": canonical_sha256(task["user_scenario"]["instructions"]),
    }


def _scaled_selection_receipt_fixture(
    *,
    row: dict[str, Any],
    numbered_source_rows: tuple[tuple[int, dict[str, Any]], ...],
    selected_line: int,
    generation_stratum: dict[str, Any],
    recipe: dict[str, Any],
    generation_variant_ordinal: int = 0,
    family_scoped_stratum: bool = False,
) -> tuple[dict[str, Any], dict[str, dict[str, str]], str]:
    source = "train" if row["split"] == "train" else "development"
    source_path = f"local/tau3/source-v1/training_source/{source}_synthetic.jsonl"
    source_file_sha256 = canonical_sha256(
        [source_row for _, source_row in numbered_source_rows]
    )
    campaign_salt = "scaled-unit-campaign-salt"
    campaign_salt_sha256 = hashlib.sha256(campaign_salt.encode("utf-8")).hexdigest()
    selection_stratum = {
        "schema_version": SELECTION_STRATUM_SCHEMA_VERSION,
        "source": source,
        "mapped_grounded_split": row["split"],
        "domain": row["domain"],
        "eligible_source_family_sha256": (
            row["source_family_id"] if family_scoped_stratum else None
        ),
    }
    selection_stratum_sha256 = canonical_sha256(selection_stratum)
    eligible = [
        (line, source_row)
        for line, source_row in numbered_source_rows
        if source_row.get("domain") == row["domain"]
        and source_row.get("split") == source
        and (
            selection_stratum["eligible_source_family_sha256"] is None
            or source_row.get("task_family")
            == selection_stratum["eligible_source_family_sha256"]
        )
    ]

    def rank(source_row: dict[str, Any]) -> str:
        canonical = json.dumps(
            source_row,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(
            campaign_salt.encode("utf-8")
            + b"\0"
            + selection_stratum_sha256.encode("utf-8")
            + b"\0"
            + canonical
        ).hexdigest()

    ordered = sorted(
        eligible,
        key=lambda item: (
            rank(item[1]),
            str(item[1].get("task_sha256") or ""),
            canonical_sha256(item[1]),
            item[0],
        ),
    )
    rank_ordinal = next(
        index for index, (line, _) in enumerate(ordered) if line == selected_line
    )
    source_row = next(
        source_row for line, source_row in numbered_source_rows if line == selected_line
    )
    task = source_row["task"]
    generation_stratum_sha256 = canonical_sha256(generation_stratum)
    generation_recipe_sha256 = canonical_sha256(recipe)
    receipt: dict[str, Any] = {
        "schema_version": SCALED_SELECTION_RECEIPT_SCHEMA_VERSION,
        "algorithm": SCALED_SELECTION_ALGORITHM,
        "rank_tie_breaker_contract": SCALED_RANK_TIE_BREAKER_CONTRACT,
        "family_identifier": row["source_family_id"],
        "family_identifier_semantics": CONTENT_ADDRESSED_FAMILY_SEMANTICS,
        "source": source,
        "mapped_grounded_split": row["split"],
        "source_file_path": source_path,
        "source_file_sha256": source_file_sha256,
        "source_line_number": selected_line,
        "canonical_source_row_sha256": canonical_sha256(source_row),
        "task_id_sha256": hashlib.sha256(task["id"].encode("utf-8")).hexdigest(),
        "prompt_sha256": source_row["prompt_sha256"],
        "source_family_sha256": source_row["task_family"],
        "source_revision": source_row["source_revision"],
        "campaign_salt": campaign_salt,
        "campaign_salt_sha256": campaign_salt_sha256,
        "selection_rank_sha256": rank(source_row),
        "task_sha256": source_row["task_sha256"],
        "task_initial_state_sha256": canonical_sha256(task["initial_state"]),
        "selection_stratum_definition": selection_stratum,
        "selection_stratum_sha256": selection_stratum_sha256,
        "rank_ordinal": rank_ordinal,
        "generation_variant_ordinal": generation_variant_ordinal,
        "generation_stratum_definition": generation_stratum,
        "generation_stratum_sha256": generation_stratum_sha256,
        "generation_recipe": recipe,
        "generation_recipe_sha256": generation_recipe_sha256,
        "selected_family_sha256": source_row["task_family"],
    }
    family_binding = {
        "schema_version": GENERATED_FAMILY_SCHEMA_VERSION,
        "source_family_sha256": receipt["source_family_sha256"],
        "source": source,
        "mapped_grounded_split": row["split"],
        "domain": row["domain"],
        "generation_stratum_sha256": generation_stratum_sha256,
        "generation_recipe_sha256": generation_recipe_sha256,
        "generation_variant_ordinal": generation_variant_ordinal,
    }
    receipt["generated_family_identifier"] = canonical_sha256(family_binding)
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    specs = {
        row["split"]: {
            "source_split": source,
            "path": source_path,
            "sha256": source_file_sha256,
        }
    }
    return receipt, specs, campaign_salt_sha256


def _coverage_policy_review(
    target: dict[str, Any],
    *,
    allowed: bool,
) -> dict[str, Any]:
    review = {
        "schema_version": "hfr.tau3_policy_review.v1",
        "reviewer": "unit-aggregate-reviewer",
        "reviewer_artifact_sha256": canonical_sha256("unit-aggregate-reviewer"),
        "policy_sha256": canonical_sha256("unit-aggregate-policy"),
        "canonical_target_sha256": target["canonical_target_sha256"],
        "parent_turn_ordinal": target["parent_turn_ordinal"],
        "parent_assistant_decision_ordinal": target[
            "parent_assistant_decision_ordinal"
        ],
        "allowed": allowed,
        "reason_id": "aggregate-safe" if allowed else "aggregate-negative",
    }
    review["review_receipt_sha256"] = canonical_sha256(review)
    return review


def _scaled_aggregate_row(
    *,
    split: str,
    domain: str,
    family_index: int,
) -> dict[str, Any]:
    domain_code = {"airline": "airx", "retail": "retx", "telecom": "telx"}[domain]
    split_code = "trn" if split == "train" else "val"
    family_count = 8 if split == "train" else 6
    repetitions = 3 if split == "train" else 1
    targets_per_row = len(BEHAVIORS) * repetitions
    tool_target_limit = 300 if split == "train" else 75
    targets: list[dict[str, Any]] = []
    for behavior_index, behavior in enumerate(BEHAVIORS):
        for repetition in range(repetitions):
            local_index = behavior_index * repetitions + repetition
            global_index = family_index * targets_per_row + local_index
            decision = global_index * 2 + 1
            unique = f"{split_code}-{domain_code}-{global_index:04d}"
            if global_index < tool_target_limit:
                canonical_target = {
                    "kind": "tool_call",
                    "text": "",
                    "tool_name": f"lookup_{global_index % 5}",
                    "arguments": {"value": unique},
                }
            else:
                canonical_target = {
                    "kind": "assistant_message",
                    "text": f"message-{unique}",
                    "tool_name": None,
                    "arguments": {},
                }
            safe_target = {
                "parent_turn_ordinal": decision,
                "parent_assistant_decision_ordinal": decision,
                "behavior": behavior,
                "masked": False,
                "mask_reason": None,
                "policy_review": None,
                "canonical_target": canonical_target,
                "canonical_target_sha256": canonical_sha256(canonical_target),
            }
            if behavior in {
                "hallucinated_tool_correction",
                "harmful_mutation_correction",
                "premature_completion_correction",
            }:
                safe_target["policy_review"] = _coverage_policy_review(
                    safe_target,
                    allowed=True,
                )
                negative_kind = {
                    "hallucinated_tool_correction": "hallucinated_tool",
                    "harmful_mutation_correction": "harmful_mutation",
                    "premature_completion_correction": "premature_completion",
                }[behavior]
                negative_canonical = {
                    "kind": "assistant_message",
                    "text": f"negative-{unique}",
                    "tool_name": None,
                    "arguments": {},
                }
                negative_target = {
                    "parent_turn_ordinal": decision - 1,
                    "parent_assistant_decision_ordinal": decision - 1,
                    "behavior": behavior,
                    "negative_behavior": negative_kind,
                    "masked": True,
                    "mask_reason": "reviewed_negative_context",
                    "reviewed": True,
                    "safe_correction_decision_ordinal": decision,
                    "policy_review": None,
                    "canonical_target": negative_canonical,
                    "canonical_target_sha256": canonical_sha256(negative_canonical),
                }
                negative_target["policy_review"] = _coverage_policy_review(
                    negative_target,
                    allowed=False,
                )
                targets.append(negative_target)
            targets.append(safe_target)
    family_id = canonical_sha256(
        {"split": split, "domain": domain, "family_index": family_index}
    )
    generated_id = canonical_sha256(
        {"generated": family_id, "split": split, "domain": domain}
    )
    selection_stratum_sha = canonical_sha256(
        {"selection": family_id, "split": split, "domain": domain}
    )
    generation_stratum_sha = canonical_sha256(
        {"generation": family_id, "split": split, "domain": domain}
    )
    recipe_sha = canonical_sha256("aggregate-recipe")
    task_sha = canonical_sha256(
        {"task": family_id, "split": split, "domain": domain}
    )
    prompt_sha = canonical_sha256(
        {"prompt": family_id, "split": split, "domain": domain}
    )
    catalog = [
        {
            "type": "function",
            "function": {
                "name": f"lookup_{index}",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            },
        }
        for index in range(5)
    ]
    row_id = f"row-{split_code}-{domain_code}-{family_index:02d}"
    return {
        "trajectory": {
            "trajectory_id": row_id,
            "system_prompt": "unit scaled prompt",
            "turns": [{"row_id": row_id}],
        },
        "tool_catalog": catalog,
        "training_targets": targets,
        "metadata": {
            "domain": domain,
            "split": split,
            "runtime_family": "vendored_tau_tools@" + TAU_REVISION,
            "source_family": "reviewed_synthetic",
            "source_family_id": family_id,
            "tool_catalog_sha256": canonical_sha256(catalog),
            "policy_sha256": canonical_sha256("unit-aggregate-policy"),
            "tool_exemptions": [],
            "selection_receipt": {
                "schema_version": SCALED_SELECTION_RECEIPT_SCHEMA_VERSION,
                "selection_stratum_sha256": selection_stratum_sha,
                "rank_ordinal": 0,
                "generation_stratum_sha256": generation_stratum_sha,
                "generation_recipe_sha256": recipe_sha,
                "generation_variant_ordinal": 0,
                "generated_family_identifier": generated_id,
                "source_family_sha256": family_id,
                "source": "train" if split == "train" else "development",
                "mapped_grounded_split": split,
                "task_sha256": task_sha,
                "prompt_sha256": prompt_sha,
            },
        },
    }


def _scaled_coverage_fixture() -> dict[str, list[dict[str, Any]]]:
    return {
        split: [
            _scaled_aggregate_row(
                split=split,
                domain=domain,
                family_index=family_index,
            )
            for domain in ("airline", "retail", "telecom")
            for family_index in range(8 if split == "train" else 6)
        ]
        for split in ("train", "validation")
    }


class Tau3GroundedGenerationTests(unittest.TestCase):
    def test_tool_result_class_recognizes_serialized_empty_containers(self) -> None:
        for value in (None, [], {}, "null", "[]", "{}", "  {}  "):
            self.assertEqual(_tool_result_class(value), "empty")
        self.assertEqual(_tool_result_class('{"error":"not_found"}'), "error")
        self.assertEqual(_tool_result_class("ordinary non-empty text"), "success")

    def test_model_to_json_normalizes_scalar_dates_times_and_enums(self) -> None:
        payload = {
            "date": date(2026, 7, 25),
            "datetime": datetime(2026, 7, 25, 12, 3, 4),
            "time": time(12, 3, 4),
            "enum": _UnitEnum.ACTIVE,
        }

        normalized = _model_to_json(payload)

        self.assertEqual(
            normalized,
            {
                "date": "2026-07-25",
                "datetime": "2026-07-25T12:03:04",
                "time": "12:03:04",
                "enum": "Active",
            },
        )
        self.assertIsInstance(canonical_sha256(normalized), str)

    def test_complete_fake_fixture_builds_blocked_not_candidate_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            _complete_source(source)

            manifest = build_tau3_grounded_generation_dataset(
                source=source,
                out_dir=root / "out",
                strict_coverage=False,
            )
            result = validate_tau3_grounded_generation_bundle(root / "out", strict=False)

            self.assertEqual(manifest["schema_version"], TAU3_GROUNDED_DATASET_SCHEMA_VERSION)
            self.assertEqual(manifest["lineage_id"], LINEAGE_ID)
            self.assertFalse(manifest["passed"])
            self.assertFalse(result["passed"])
            self.assertTrue(
                any("candidate-eligible vendored Tau replay" in error for error in result["errors"]),
                result["errors"][:5],
            )
            self.assertEqual(result["coverage"]["by_split"]["train"]["airline"]["source_family_count"], 8)
            self.assertEqual(result["coverage"]["by_split"]["validation"]["airline"]["source_family_count"], 2)
            row = json.loads((root / "out" / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertNotIn("initial_state", row)
            self.assertIn("initial_state_ref", row)

    def test_shared_initial_states_are_deduplicated_into_one_blob(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            first = _scenario(
                split="train",
                domain="airline",
                family_index=0,
                behavior="authentication",
            )
            second = _scenario(
                split="train",
                domain="airline",
                family_index=1,
                behavior="clarification_refusal",
            )
            second["initial_state"] = json.loads(json.dumps(first["initial_state"]))
            _write_jsonl(source, [first, second])

            build_tau3_grounded_generation_dataset(
                source=source,
                out_dir=root / "out",
                strict_coverage=False,
            )

            state_blobs = list((root / "out" / "states").glob("*.json"))
            rows = [
                json.loads(line)
                for line in (root / "out" / "train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(state_blobs), 1)
            self.assertEqual(rows[0]["initial_state_ref"], rows[1]["initial_state_ref"])

    def test_validation_rejects_tampered_and_traversing_state_refs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            _write_jsonl(
                source,
                [
                    _scenario(
                        split="train",
                        domain="airline",
                        family_index=0,
                        behavior="authentication",
                    )
                ],
            )
            out = root / "out"
            build_tau3_grounded_generation_dataset(
                source=source,
                out_dir=out,
                strict_coverage=False,
            )
            state_path = next((out / "states").glob("*.json"))
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["records"]["tampered"] = {"id": "tampered"}
            _write_json(state_path, state)

            result = validate_tau3_grounded_generation_bundle(out, strict=False)

            self.assertFalse(result["passed"])
            self.assertTrue(
                any("initial_state_ref.sha256 does not replay" in error for error in result["errors"]),
                result["errors"],
            )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            _write_jsonl(
                source,
                [
                    _scenario(
                        split="train",
                        domain="airline",
                        family_index=0,
                        behavior="authentication",
                    )
                ],
            )
            out = root / "out"
            build_tau3_grounded_generation_dataset(
                source=source,
                out_dir=out,
                strict_coverage=False,
            )
            result = validate_tau3_grounded_generation_bundle(out, strict=False)

            self.assertFalse(result["passed"])
            self.assertFalse(
                any("runtime cannot be instantiated" in error for error in result["errors"]),
                result["errors"],
            )
            self.assertFalse(
                any("tool_replay" in error for error in result["errors"]),
                result["errors"],
            )
            rows = [
                json.loads(line)
                for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["initial_state_ref"]["path"] = "../escaped.json"
            rows[0]["metadata"]["row_sha256"] = canonical_sha256(
                {
                    key: value
                    for key, value in rows[0].items()
                    if key != "metadata"
                }
                | {
                    "metadata": {
                        key: value
                        for key, value in rows[0]["metadata"].items()
                        if key != "row_sha256"
                    }
                }
            )
            _write_jsonl(out / "train.jsonl", rows)
            _rewrite_bundle_manifest(out)

            result = validate_tau3_grounded_generation_bundle(out, strict=False)

            self.assertFalse(result["passed"])
            self.assertTrue(
                any("initial_state_ref.path must be a safe relative path" in error for error in result["errors"]),
                result["errors"],
            )

    def test_validation_replays_and_rejects_tampered_state_result_tool_and_args(self) -> None:
        tamper_cases = (
            ("arguments_sha256", lambda call: call.update({"canonical_arguments": {"id": "missing"}})),
            ("canonical_result", lambda call: call.update({"canonical_result": {"id": "tampered"}})),
            ("tool_definition_sha256", lambda call: call.update({"tool_definition_sha256": "b" * 64})),
            ("post_state_sha256", lambda call: call.update({"post_state_sha256": "c" * 64})),
        )
        for label, mutate in tamper_cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = root / "source.jsonl"
                _complete_source(source)
                out = root / "out"
                build_tau3_grounded_generation_dataset(
                    source=source,
                    out_dir=out,
                    strict_coverage=False,
                )
                rows = [
                    json.loads(line)
                    for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
                ]
                mutate(rows[0]["tool_replay"][0])
                rows[0]["metadata"]["row_sha256"] = canonical_sha256(
                    {
                        key: value
                        for key, value in rows[0].items()
                        if key != "metadata"
                    }
                    | {
                        "metadata": {
                            key: value
                            for key, value in rows[0]["metadata"].items()
                            if key != "row_sha256"
                        }
                    }
                )
                _write_jsonl(out / "train.jsonl", rows)
                _rewrite_bundle_manifest(out)

                result = validate_tau3_grounded_generation_bundle(out)

                self.assertFalse(result["passed"])
                self.assertTrue(
                    any(label in error for error in result["errors"]),
                    result["errors"][:10],
                )

    def test_source_expected_result_hash_and_class_are_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            row = _scenario(
                split="train",
                domain="airline",
                family_index=0,
                behavior="authentication",
            )
            expected = {
                "id": "airline-0",
                "status": "open",
            }
            row["turns"][0]["assistant"]["tool_calls"][0]["expected_result_sha256"] = canonical_sha256(expected)
            row["turns"][0]["assistant"]["tool_calls"][0]["expected_result_class"] = "success"
            _write_jsonl(source, [row])
            out = root / "out"

            build_tau3_grounded_generation_dataset(
                source=source,
                out_dir=out,
                strict_coverage=False,
            )

            exported = json.loads((out / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
            evidence = exported["tool_replay"][0]
            self.assertTrue(evidence["source_expected_result_verified"])
            self.assertEqual(evidence["source_expected_result_sha256"], canonical_sha256(expected))
            self.assertEqual(evidence["source_expected_result_class"], "success")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            row = _scenario(
                split="train",
                domain="airline",
                family_index=0,
                behavior="authentication",
            )
            row["turns"][0]["assistant"]["tool_calls"][0]["expected_result_sha256"] = "0" * 64
            _write_jsonl(source, [row])

            with self.assertRaises(Tau3GroundedGenerationError):
                build_tau3_grounded_generation_dataset(
                    source=source,
                    out_dir=root / "out",
                    strict_coverage=False,
                )

    def test_recovery_targets_require_the_claimed_immediate_tool_condition(self) -> None:
        cases = (
            ("empty_result_recovery", "success after empty claim"),
            ("error_result_recovery", "success after error claim"),
            ("repeated_call_recovery", "single call after repeated claim"),
        )
        for behavior, label in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = root / "source.jsonl"
                row = _scenario(
                    split="train",
                    domain="airline",
                    family_index=0,
                    behavior=behavior,
                )
                if behavior in {"empty_result_recovery", "error_result_recovery"}:
                    row["turns"][0]["assistant"]["tool_calls"] = [
                        {
                            "tool_name": "get_record",
                            "arguments": {"id": "airline-0"},
                        }
                    ]
                else:
                    recovery = row["turns"][2]
                    recovery["assistant"]["decision_ordinal"] = 1
                    row["turns"] = [row["turns"][0], recovery]
                _write_jsonl(source, [row])

                with self.assertRaisesRegex(
                    Tau3GroundedGenerationError,
                    "empty-result|error-result|repeated-call",
                ):
                    build_tau3_grounded_generation_dataset(
                        source=source,
                        out_dir=root / "out",
                        strict_coverage=False,
                    )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            row = _scenario(
                split="train",
                domain="airline",
                family_index=0,
                behavior="empty_result_recovery",
            )
            row["turns"][0]["assistant"]["tool_calls"][0]["expected_result_class"] = "success"
            _write_jsonl(source, [row])

            with self.assertRaises(Tau3GroundedGenerationError):
                build_tau3_grounded_generation_dataset(
                    source=source,
                    out_dir=root / "out",
                    strict_coverage=False,
                )

    def test_validator_rejects_tampered_source_expected_result_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            row = _scenario(
                split="train",
                domain="airline",
                family_index=0,
                behavior="authentication",
            )
            expected = {"id": "airline-0", "status": "open"}
            row["turns"][0]["assistant"]["tool_calls"][0]["expected_result_sha256"] = canonical_sha256(expected)
            _write_jsonl(source, [row])
            out = root / "out"
            build_tau3_grounded_generation_dataset(
                source=source,
                out_dir=out,
                strict_coverage=False,
            )
            rows = [
                json.loads(line)
                for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["tool_replay"][0]["source_expected_result_sha256"] = "1" * 64
            rows[0]["metadata"]["row_sha256"] = canonical_sha256(
                {
                    key: value
                    for key, value in rows[0].items()
                    if key != "metadata"
                }
                | {
                    "metadata": {
                        key: value
                        for key, value in rows[0]["metadata"].items()
                        if key != "row_sha256"
                    }
                }
            )
            _write_jsonl(out / "train.jsonl", rows)
            _rewrite_bundle_manifest(out)

            result = validate_tau3_grounded_generation_bundle(out, strict=False)

            self.assertFalse(result["passed"])
            self.assertTrue(
                any("expected_result_sha256 does not match replayed result" in error for error in result["errors"]),
                result["errors"],
            )

    def test_builder_rejects_fabricated_success_without_replayed_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            _write_jsonl(
                source,
                [
                    _scenario(
                        split="train",
                        domain="airline",
                        family_index=0,
                        behavior="successful_completion",
                        fabricated_success=True,
                    )
                ],
            )
            with self.assertRaisesRegex(
                Tau3GroundedGenerationError,
                "fabricates completion",
            ):
                build_tau3_grounded_generation_dataset(
                    source=source,
                    out_dir=root / "out",
                    strict_coverage=False,
                )

    def test_completion_requires_mutation_on_an_earlier_decision(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            row = _scenario(
                split="train",
                domain="airline",
                family_index=0,
                behavior="successful_completion",
            )
            row["turns"] = row["turns"][:1]
            row["turns"][0]["assistant"]["safe_corrected_target"] = {
                "behavior": "successful_completion",
                "kind": "assistant_message",
                "text": "Completed the account update successfully.",
            }
            row["reviewer"] = _test_reviewer_record(
                [row["turns"][0]["assistant"]["safe_corrected_target"]]
            )
            _write_jsonl(source, [row])

            with self.assertRaisesRegex(
                Tau3GroundedGenerationError,
                "without prior replayed post-state mutation",
            ):
                build_tau3_grounded_generation_dataset(
                    source=source,
                    out_dir=root / "out",
                    strict_coverage=False,
                )

    def test_validation_reports_split_contamination(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            shared_source_id = "shared-source"
            _write_jsonl(
                source,
                [
                    _scenario(
                        split="train",
                        domain="airline",
                        family_index=0,
                        behavior="successful_completion",
                        source_id=shared_source_id,
                    ),
                    _scenario(
                        split="validation",
                        domain="airline",
                        family_index=0,
                        behavior="successful_completion",
                        source_id=shared_source_id,
                    ),
                ],
            )
            out = root / "out"
            build_tau3_grounded_generation_dataset(
                source=source,
                out_dir=out,
                strict_coverage=False,
            )

            result = validate_tau3_grounded_generation_bundle(out, strict=False)

            self.assertFalse(result["passed"])
            self.assertTrue(
                any("source_id crosses splits" in error for error in result["errors"]),
                result["errors"],
            )

    def test_builder_requires_explicit_complete_matching_contamination_metadata(self) -> None:
        cases = (
            ("missing", lambda row: row.pop("contamination")),
            ("partial", lambda row: row.update({"contamination": {"raw_sealed_payload_read": False}})),
            (
                "sealed_payload",
                lambda row: row.update(
                    {
                        "contamination": {
                            "source_split": "train",
                            "raw_sealed_payload_read": True,
                            "sealed_hash_only": True,
                        }
                    }
                ),
            ),
            (
                "not_hash_only",
                lambda row: row.update(
                    {
                        "contamination": {
                            "source_split": "train",
                            "raw_sealed_payload_read": False,
                            "sealed_hash_only": False,
                        }
                    }
                ),
            ),
            (
                "split_mismatch",
                lambda row: row.update(
                    {
                        "contamination": {
                            "source_split": "validation",
                            "raw_sealed_payload_read": False,
                            "sealed_hash_only": True,
                        }
                    }
                ),
            ),
        )
        for label, mutate in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = root / "source.jsonl"
                row = _scenario(
                    split="train",
                    domain="airline",
                    family_index=0,
                    behavior="authentication",
                )
                mutate(row)
                _write_jsonl(source, [row])

                with self.assertRaises(Tau3GroundedGenerationError):
                    build_tau3_grounded_generation_dataset(
                        source=source,
                        out_dir=root / "out",
                        strict_coverage=False,
                    )

    def test_validator_replays_row_contamination_metadata(self) -> None:
        tamper_cases = (
            (
                "raw_sealed_payload_read must be false",
                lambda contamination: contamination.update({"raw_sealed_payload_read": True}),
            ),
            (
                "sealed_hash_only must be true",
                lambda contamination: contamination.update({"sealed_hash_only": False}),
            ),
            (
                "source_split must match row split",
                lambda contamination: contamination.update({"source_split": "validation"}),
            ),
        )
        for expected, mutate in tamper_cases:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = root / "source.jsonl"
                _write_jsonl(
                    source,
                    [
                        _scenario(
                            split="train",
                            domain="airline",
                            family_index=0,
                            behavior="authentication",
                        )
                    ],
                )
                out = root / "out"
                build_tau3_grounded_generation_dataset(
                    source=source,
                    out_dir=out,
                    strict_coverage=False,
                )
                rows = [
                    json.loads(line)
                    for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
                ]
                mutate(rows[0]["metadata"]["contamination"])
                rows[0]["metadata"]["row_sha256"] = canonical_sha256(
                    {
                        key: value
                        for key, value in rows[0].items()
                        if key != "metadata"
                    }
                    | {
                        "metadata": {
                            key: value
                            for key, value in rows[0]["metadata"].items()
                            if key != "row_sha256"
                        }
                    }
                )
                _write_jsonl(out / "train.jsonl", rows)
                _rewrite_bundle_manifest(out)

                result = validate_tau3_grounded_generation_bundle(out, strict=False)

                self.assertFalse(result["passed"])
                self.assertTrue(
                    any(expected in error for error in result["errors"]),
                    result["errors"],
                )

    def test_builder_rejects_unmasked_unsafe_corrected_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            _write_jsonl(
                source,
                [
                    _scenario(
                        split="train",
                        domain="airline",
                        family_index=0,
                        behavior="harmful_mutation_correction",
                        unsafe=False,
                    )
                ],
            )

            with self.assertRaises(Tau3GroundedGenerationError):
                build_tau3_grounded_generation_dataset(
                    source=source,
                    out_dir=root / "out",
                    strict_coverage=False,
                )

    def test_reviewed_negative_action_is_retained_and_linked_to_safe_correction(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            _write_jsonl(
                source,
                [
                    _scenario(
                        split="train",
                        domain="airline",
                        family_index=0,
                        behavior="harmful_mutation_correction",
                        unsafe=True,
                    )
                ],
            )
            out = root / "out"

            build_tau3_grounded_generation_dataset(
                source=source,
                out_dir=out,
                strict_coverage=False,
            )

            exported = json.loads(
                (out / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            negative, correction = exported["training_targets"]
            self.assertTrue(negative["masked"])
            self.assertTrue(negative["reviewed"])
            self.assertEqual(negative["negative_behavior"], "harmful_mutation")
            self.assertEqual(negative["safe_correction_decision_ordinal"], 1)
            self.assertEqual(negative["canonical_target"]["kind"], "tool_call")
            self.assertEqual(negative["canonical_target"]["tool_name"], "update_record")
            self.assertFalse(correction["masked"])
            self.assertEqual(correction["parent_assistant_decision_ordinal"], 1)

    def test_masked_negative_requires_review_and_valid_forward_link(self) -> None:
        for field in ("reviewed", "safe_correction_decision_ordinal"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = root / "source.jsonl"
                row = _scenario(
                    split="train",
                    domain="airline",
                    family_index=0,
                    behavior="harmful_mutation_correction",
                    unsafe=True,
                )
                row["turns"][0]["assistant"]["safe_corrected_target"].pop(field)
                _write_jsonl(source, [row])

                with self.assertRaises(Tau3GroundedGenerationError):
                    build_tau3_grounded_generation_dataset(
                        source=source,
                        out_dir=root / "out",
                        strict_coverage=False,
                    )

    def test_builder_rejects_unbound_unmasked_tool_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            row = _scenario(
                split="train",
                domain="airline",
                family_index=0,
                behavior="authentication",
            )
            row["turns"][0]["assistant"]["safe_corrected_target"].update(
                {
                    "tool_name": "get_record",
                    "arguments": {"id": "different-record"},
                    "safe_precondition": "forged bypass must not admit this target",
                }
            )
            _write_jsonl(source, [row])

            with self.assertRaises(Tau3GroundedGenerationError):
                build_tau3_grounded_generation_dataset(
                    source=source,
                    out_dir=root / "out",
                    strict_coverage=False,
                )

    def test_builder_rejects_malformed_canonical_target_kind_and_tool_fields(self) -> None:
        cases = (
            ("bad_kind", {"kind": "other"}),
            ("message_with_tool", {"kind": "assistant_message", "tool_name": "get_record"}),
            ("tool_call_without_args", {"kind": "tool_call", "tool_name": "get_record", "arguments": {}}),
        )
        for label, update in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = root / "source.jsonl"
                row = _scenario(
                    split="train",
                    domain="airline",
                    family_index=0,
                    behavior="authentication",
                )
                row["turns"][0]["assistant"]["safe_corrected_target"].update(update)
                _write_jsonl(source, [row])

                with self.assertRaises(Tau3GroundedGenerationError):
                    build_tau3_grounded_generation_dataset(
                        source=source,
                        out_dir=root / "out",
                        strict_coverage=False,
                    )

    def test_zero_argument_tool_target_is_grounded_by_exact_catalog_and_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            row = _scenario(
                split="train",
                domain="airline",
                family_index=0,
                behavior="empty_result_recovery",
            )
            row["turns"][1]["assistant"]["safe_corrected_target"].update(
                {
                    "kind": "tool_call",
                    "tool_name": "empty_search",
                    "arguments": {},
                }
            )
            row["turns"][1]["assistant"]["tool_calls"] = [
                {"tool_name": "empty_search", "arguments": {}}
            ]
            _write_jsonl(source, [row])

            build_tau3_grounded_generation_dataset(
                source=source,
                out_dir=root / "out",
                strict_coverage=False,
            )

            result = validate_tau3_grounded_generation_bundle(
                root / "out",
                strict=False,
            )
            self.assertFalse(
                any("empty target arguments" in error for error in result["errors"]),
                result["errors"],
            )
            exported = json.loads(
                (root / "out" / "train.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertEqual(
                exported["training_targets"][0]["canonical_target"]["arguments"],
                {},
            )

    def test_tool_exemptions_export_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            row = _scenario(
                split="train",
                domain="airline",
                family_index=0,
                behavior="empty_result_recovery",
            )
            row["tool_exemptions"] = [
                {
                    "tool_name": "empty_search",
                    "reason": "zero_arg",
                    "reviewer": "unit-reviewer",
                },
                {
                    "tool_name": "get_record",
                    "reason": "policy_forbidden",
                    "reviewer": "unit-reviewer",
                    "policy_hash": canonical_sha256("policy"),
                    "citation": "policy:do-not-use-get-record-for-this-case",
                },
            ]
            _write_jsonl(source, [row])

            build_tau3_grounded_generation_dataset(
                source=source,
                out_dir=root / "out",
                strict_coverage=False,
            )

            exported = json.loads((root / "out" / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(exported["metadata"]["tool_exemptions"], row["tool_exemptions"])

        bad_cases = (
            (
                "zero_arg_on_required_tool",
                [
                    {
                        "tool_name": "get_record",
                        "reason": "zero_arg",
                        "reviewer": "unit-reviewer",
                    }
                ],
            ),
            (
                "policy_missing_evidence",
                [
                    {
                        "tool_name": "get_record",
                        "reason": "policy_forbidden",
                        "reviewer": "unit-reviewer",
                    }
                ],
            ),
            (
                "not_in_catalog",
                [
                    {
                        "tool_name": "missing_tool",
                        "reason": "zero_arg",
                        "reviewer": "unit-reviewer",
                    }
                ],
            ),
            (
                "bad_reason",
                [
                    {
                        "tool_name": "empty_search",
                        "reason": "reviewed",
                        "reviewer": "unit-reviewer",
                    }
                ],
            ),
        )
        for label, exemptions in bad_cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = root / "source.jsonl"
                row = _scenario(
                    split="train",
                    domain="airline",
                    family_index=0,
                    behavior="authentication",
                )
                row["tool_exemptions"] = exemptions
                _write_jsonl(source, [row])

                with self.assertRaises(Tau3GroundedGenerationError):
                    build_tau3_grounded_generation_dataset(
                        source=source,
                        out_dir=root / "out",
                        strict_coverage=False,
                    )

    def test_validator_replays_tool_exemptions(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            _write_jsonl(
                source,
                [
                    _scenario(
                        split="train",
                        domain="airline",
                        family_index=0,
                        behavior="authentication",
                    )
                ],
            )
            out = root / "out"
            build_tau3_grounded_generation_dataset(
                source=source,
                out_dir=out,
                strict_coverage=False,
            )
            rows = [
                json.loads(line)
                for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["metadata"]["tool_exemptions"] = [
                {
                    "tool_name": "missing_tool",
                    "reason": "zero_arg",
                    "reviewer": "unit-reviewer",
                }
            ]
            rows[0]["metadata"]["row_sha256"] = canonical_sha256(
                {
                    key: value
                    for key, value in rows[0].items()
                    if key != "metadata"
                }
                | {
                    "metadata": {
                        key: value
                        for key, value in rows[0]["metadata"].items()
                        if key != "row_sha256"
                    }
                }
            )
            _write_jsonl(out / "train.jsonl", rows)
            _rewrite_bundle_manifest(out)

            result = validate_tau3_grounded_generation_bundle(out, strict=False)

            self.assertFalse(result["passed"])
            self.assertTrue(
                any("tool_name is not in exact catalog" in error for error in result["errors"]),
                result["errors"],
            )

    def test_validator_rejects_forged_safe_precondition_on_unexecuted_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            _write_jsonl(
                source,
                [
                    _scenario(
                        split="train",
                        domain="airline",
                        family_index=0,
                        behavior="authentication",
                    )
                ],
            )
            out = root / "out"
            build_tau3_grounded_generation_dataset(
                source=source,
                out_dir=out,
                strict_coverage=False,
            )
            rows = [
                json.loads(line)
                for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["training_targets"][0]["canonical_target"]["tool_name"] = "get_record"
            rows[0]["training_targets"][0]["canonical_target"]["arguments"] = {
                "id": "not-executed"
            }
            rows[0]["training_targets"][0]["canonical_target_sha256"] = canonical_sha256(
                rows[0]["training_targets"][0]["canonical_target"]
            )
            rows[0]["training_targets"][0]["safe_precondition"] = "forged bypass"
            rows[0]["metadata"]["row_sha256"] = canonical_sha256(
                {
                    key: value
                    for key, value in rows[0].items()
                    if key != "metadata"
                }
                | {
                    "metadata": {
                        key: value
                        for key, value in rows[0]["metadata"].items()
                        if key != "row_sha256"
                    }
                }
            )
            _write_jsonl(out / "train.jsonl", rows)
            _rewrite_bundle_manifest(out)

            result = validate_tau3_grounded_generation_bundle(out, strict=False)

            self.assertFalse(result["passed"])
            self.assertTrue(
                any("target tool call is not exactly bound" in error for error in result["errors"]),
                result["errors"],
            )

    def test_vendored_tau_adapter_replays_actual_airline_tool_when_available(self) -> None:
        repo = Path(__file__).resolve().parents[1] / "local" / "tau3" / "repository"
        if not repo.is_dir():
            self.skipTest("local/tau3/repository is absent")
        try:
            import sys

            sys.path.insert(0, str(repo / "src"))
            _install_tau_import_shims()
            from tau2.domains.airline.data_model import FlightDB
            from tau2.domains.airline.utils import AIRLINE_DB_PATH
        except Exception as exc:
            self.skipTest(f"vendored Tau toolkit dependencies unavailable: {exc}")
        state = FlightDB.load(AIRLINE_DB_PATH).model_dump(mode="json")
        user_id = next(iter(state["users"]))
        import subprocess

        revision = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            row = _scenario(
                split="train",
                domain="airline",
                family_index=0,
                behavior="authentication",
            )
            row["tau_revision"] = revision
            row["runtime_family"] = f"vendored_tau_tools@{revision}"
            row["tau_repo"] = repo.relative_to(Path(__file__).resolve().parents[1]).as_posix()
            full_state = {
                "task_initialization": None,
                "pre_sync": {"agent_db": state, "user_db": None},
                "post_sync": {"agent_db": state, "user_db": None},
            }
            row["initial_state"] = full_state
            row["source_family_id"] = canonical_sha256("synthetic-airline-family")
            runtime = _VendoredTauRuntime(
                domain="airline",
                revision=revision,
                state=full_state,
                repo=row["tau_repo"],
            )
            row["system_prompt"] = runtime.system_prompt()
            row["tool_catalog"] = runtime.tool_catalog()
            row["selection_receipt"] = _candidate_selection_receipt(row)
            row["generation_provenance"] = _candidate_generation_provenance(
                row,
                row["selection_receipt"],
            )
            _promote_candidate_reviewer(
                row,
                policy_sha256=runtime.policy_sha256(),
            )
            specs, selection_rows, salt_sha256 = _candidate_selection_source(
                row["selection_receipt"],
                row,
            )
            expected_user = state["users"][user_id]
            row["turns"][0]["user"]["content"] = f"Use customer {user_id}."
            row["turns"][0]["assistant"]["tool_calls"] = [
                {
                    "tool_name": "get_user_details",
                    "arguments": {"user_id": user_id},
                    "expected_result_sha256": canonical_sha256(expected_user),
                    "expected_result_class": "success",
                }
            ]
            row["turns"][0]["assistant"]["safe_corrected_target"] = {
                "behavior": "authentication",
                "kind": "assistant_message",
                "text": "I verified the user details with the Tau airline tool.",
            }
            _write_jsonl(source, [row])

            out = root / "out"
            with (
                patch(
                    "flightrecorder.tau3_grounded_generation.PERMITTED_SELECTION_SOURCES",
                    specs,
                ),
                patch(
                    "flightrecorder.tau3_grounded_generation.CAMPAIGN_SELECTION_SALT_SHA256",
                    salt_sha256,
                ),
                patch(
                    "flightrecorder.tau3_grounded_generation._load_permitted_selection_rows",
                    return_value=selection_rows,
                ),
            ):
                build_tau3_grounded_generation_dataset(
                    source=source,
                    out_dir=out,
                    strict_coverage=False,
                )
                clean = validate_tau3_grounded_generation_bundle(out, strict=False)
            self.assertFalse(clean["passed"])
            self.assertFalse(
                any("runtime cannot be instantiated" in error for error in clean["errors"]),
                clean["errors"],
            )
            self.assertFalse(
                any("tool_replay" in error for error in clean["errors"]),
                clean["errors"],
            )
            rows = [
                json.loads(line)
                for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(rows[0]["tool_replay"][0]["source_expected_result_verified"])
            self.assertEqual(
                rows[0]["tool_replay"][0]["source_expected_result_sha256"],
                canonical_sha256(expected_user),
            )
            rows[0]["metadata"]["tau_repo"]["tree_sha256"] = "f" * 64
            rows[0]["metadata"]["row_sha256"] = canonical_sha256(
                {
                    key: value
                    for key, value in rows[0].items()
                    if key != "metadata"
                }
                | {
                    "metadata": {
                        key: value
                        for key, value in rows[0]["metadata"].items()
                        if key != "row_sha256"
                    }
                }
            )
            _write_jsonl(out / "train.jsonl", rows)
            _rewrite_bundle_manifest(out)

            tampered = validate_tau3_grounded_generation_bundle(out, strict=False)

            self.assertFalse(tampered["passed"])
            self.assertTrue(
                any("tau_repo.tree_sha256 does not replay" in error for error in tampered["errors"]),
                tampered["errors"],
            )

    def test_vendored_tau_telecom_date_result_is_canonical_json_when_available(self) -> None:
        repo = Path(__file__).resolve().parents[1] / "local" / "tau3" / "repository"
        if not repo.is_dir():
            self.skipTest("local/tau3/repository is absent")
        try:
            import subprocess
            import sys

            sys.path.insert(0, str(repo / "src"))
            _install_tau_import_shims()
            from tau2.domains.telecom.data_model import TelecomDB
            from tau2.domains.telecom.environment import get_environment
            from tau2.domains.telecom.utils import TELECOM_DB_PATH
            from flightrecorder.tau3_grounded_generation import _VendoredTauRuntime
        except Exception as exc:
            self.skipTest(f"vendored Tau toolkit dependencies unavailable: {exc}")
        revision = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        TelecomDB.load(TELECOM_DB_PATH)
        environment = get_environment()
        pre_sync = {
            "agent_db": _model_to_json(environment.tools.db),
            "user_db": _model_to_json(environment.user_tools.db),
        }
        environment.sync_tools()
        post_sync = {
            "agent_db": _model_to_json(environment.tools.db),
            "user_db": _model_to_json(environment.user_tools.db),
        }
        state = {
            "task_initialization": None,
            "pre_sync": pre_sync,
            "post_sync": post_sync,
        }
        customer = post_sync["agent_db"]["customers"][0]
        line_id = customer["line_ids"][0]
        runtime = _VendoredTauRuntime(
            domain="telecom",
            revision=revision,
            state=state,
            repo=repo.relative_to(Path(__file__).resolve().parents[1]).as_posix(),
        )

        result = runtime.call(
            "get_data_usage",
            {"customer_id": customer["customer_id"], "line_id": line_id},
        )

        self.assertIsInstance(result["cycle_end_date"], str)
        self.assertRegex(result["cycle_end_date"], r"^2025-\d{2}-\d{2}$")
        self.assertIsInstance(canonical_sha256(result), str)

    def test_vendored_runtime_derives_prompt_catalog_and_full_sync_state(self) -> None:
        class FakeDB:
            def __init__(self, payload: dict[str, Any]) -> None:
                self.payload = copy.deepcopy(payload)

            @classmethod
            def model_validate(cls, payload: dict[str, Any]) -> "FakeDB":
                return cls(payload)

            def model_dump(self, *, mode: str) -> dict[str, Any]:
                if mode != "json":
                    raise AssertionError("unexpected serialization mode")
                return copy.deepcopy(self.payload)

        class FakeAgentDB(FakeDB):
            pass

        class FakeUserDB(FakeDB):
            pass

        class FakeToolkit:
            def __init__(self, db: FakeDB) -> None:
                self.db = db
                self._db_type = type(db)

            def update_db(self, payload: dict[str, Any]) -> None:
                self.db = self._db_type.model_validate(payload)

            @staticmethod
            def tool_mutates_state(tool_name: str) -> bool:
                return tool_name == "write_quota"

        catalog = [
            {
                "type": "function",
                "function": {
                    "name": "write_quota",
                    "description": "Synthetic write.",
                    "parameters": {
                        "type": "object",
                        "properties": {"quota": {"type": "integer"}},
                        "required": ["quota"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_quota",
                    "description": "Synthetic read.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
        ]
        tools = [SimpleNamespace(openai_schema=copy.deepcopy(item)) for item in catalog]

        class FakeEnvironment:
            def __init__(
                self,
                agent_db: FakeAgentDB | None = None,
                user_db: FakeUserDB | None = None,
            ) -> None:
                self.tools = FakeToolkit(agent_db or FakeAgentDB({"quota": 1}))
                self.user_tools = FakeToolkit(
                    user_db or FakeUserDB({"mirrored_quota": 0})
                )
                self.get_tools_calls = 0
                self.set_state_inputs: list[dict[str, Any]] = []

            def get_tools(self) -> list[Any]:
                self.get_tools_calls += 1
                return tools if self.get_tools_calls == 1 else list(reversed(tools))

            @staticmethod
            def get_policy() -> str:
                return "synthetic-policy"

            @staticmethod
            def get_domain_name() -> str:
                return "telecom"

            def sync_tools(self) -> None:
                self.user_tools.db.payload["mirrored_quota"] = self.tools.db.payload["quota"]

            def set_state(
                self,
                initialization_data: Any,
                initialization_actions: Any,
                message_history: list[Any],
                *,
                strict: bool,
            ) -> None:
                self.set_state_inputs.append(
                    {
                        "initialization_data": (
                            None
                            if initialization_data is None
                            else _model_to_json(initialization_data)
                        ),
                        "initialization_actions": copy.deepcopy(initialization_actions),
                        "message_history": copy.deepcopy(message_history),
                        "strict": strict,
                    }
                )
                if strict is not True:
                    raise AssertionError("unexpected synthetic replay controls")
                if initialization_actions is None:
                    if initialization_data is not None or message_history:
                        raise AssertionError("unexpected typed fixed-point inputs")
                    self.sync_tools()
                    self.sync_tools()
                    return
                if (
                    not isinstance(initialization_data, FakeInitializationData)
                    or initialization_data.model_dump(mode="json")
                    != {
                        "agent_data": {"quota": 1},
                        "user_data": {"mirrored_quota": 0},
                    }
                ):
                    raise AssertionError("official initialization_data changed")
                if initialization_actions != ["sync_then_mutate"]:
                    raise AssertionError("unexpected synthetic initialization actions")
                if message_history != [
                    {"role": "assistant", "content": "official-history"}
                ]:
                    raise AssertionError("official message_history changed")
                # Match the pinned Environment.set_state behavior. The
                # production telecom replay guard must ignore these two
                # cross-type alias assignments while retaining each toolkit's
                # own correctly typed update_db result.
                self.tools.update_db(initialization_data.agent_data)
                self.user_tools.db = self.tools.db
                self.user_tools.update_db(initialization_data.user_data)
                self.tools.db = self.user_tools.db
                self.sync_tools()
                self.tools.db.payload["quota"] = 2
                self.sync_tools()

            def make_tool_call(self, tool_name: str, *, requestor: str, **kwargs: Any) -> dict[str, Any]:
                if tool_name != "write_quota" or requestor != "assistant":
                    raise AssertionError("unexpected synthetic tool call")
                self.tools.db.payload["quota"] = kwargs["quota"]
                return {"updated": True}

        class FakeLLMAgent:
            def __init__(self, *, tools: list[Any], domain_policy: str, llm: str) -> None:
                if not tools or not domain_policy or llm != "offline-contract-validation":
                    raise AssertionError("runtime derivation inputs were not preserved")
                self.system_prompt = "runtime-derived-system-prompt"

        environments: list[FakeEnvironment] = []

        def get_environment(
            *,
            db: FakeAgentDB | None = None,
            user_db: FakeUserDB | None = None,
            policy_type: str | None = None,
        ) -> FakeEnvironment:
            if (db is None) != (user_db is None):
                raise AssertionError("assistant and user DBs must remain independently typed")
            if db is not None and not isinstance(db, FakeAgentDB):
                raise AssertionError("assistant DB type was not preserved")
            if user_db is not None and not isinstance(user_db, FakeUserDB):
                raise AssertionError("user DB type was not preserved")
            if db is None and policy_type is not None:
                raise AssertionError("default environment received a policy type")
            if db is not None and policy_type != "manual":
                raise AssertionError("typed telecom factory policy type was not preserved")
            environment = FakeEnvironment(db, user_db)
            environments.append(environment)
            return environment

        environment_module = SimpleNamespace(get_environment=get_environment)
        llm_agent_module = SimpleNamespace(LLMAgent=FakeLLMAgent)

        class FakeInitializationData:
            def __init__(self, value: dict[str, Any]) -> None:
                self.agent_data = copy.deepcopy(value["agent_data"])
                self.user_data = copy.deepcopy(value["user_data"])

            def model_dump(self, *, mode: str) -> dict[str, Any]:
                if mode != "json":
                    raise AssertionError(
                        "unexpected InitializationData serialization mode"
                    )
                return {
                    "agent_data": copy.deepcopy(self.agent_data),
                    "user_data": copy.deepcopy(self.user_data),
                }

        class FakeInitialState:
            validated_values: list[dict[str, Any]] = []

            def __init__(self, value: dict[str, Any]) -> None:
                self._value = copy.deepcopy(value)
                self.initialization_data = FakeInitializationData(
                    value["initialization_data"]
                )
                self.initialization_actions = copy.deepcopy(
                    value["initialization_actions"]
                )
                self.message_history = copy.deepcopy(value["message_history"])

            @classmethod
            def model_validate(cls, value: dict[str, Any]) -> "FakeInitialState":
                cls.validated_values.append(copy.deepcopy(value))
                return cls(value)

            def model_dump(self, *, mode: str) -> dict[str, Any]:
                if mode != "json":
                    raise AssertionError("unexpected InitialState serialization mode")
                return copy.deepcopy(self._value)

        def import_module(name: str) -> Any:
            if name.endswith(".environment"):
                return environment_module
            if name == "tau2.agent.llm_agent":
                return llm_agent_module
            if name == "tau2.data_model.tasks":
                return SimpleNamespace(InitialState=FakeInitialState)
            if name == "tau2.domains.telecom.data_model":
                return SimpleNamespace(TelecomDB=FakeAgentDB)
            if name == "tau2.domains.telecom.user_data_model":
                return SimpleNamespace(TelecomUserDB=FakeUserDB)
            raise AssertionError("unexpected import")

        pre_sync = {"agent_db": {"quota": 1}, "user_db": {"mirrored_quota": 0}}
        post_sync = {"agent_db": {"quota": 2}, "user_db": {"mirrored_quota": 2}}
        task_initialization = {
            "initialization_data": {
                "agent_data": {"quota": 1},
                "user_data": {"mirrored_quota": 0},
            },
            "initialization_actions": ["sync_then_mutate"],
            "message_history": [
                {"role": "assistant", "content": "official-history"}
            ],
        }
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            with (
                patch("flightrecorder.tau3_grounded_generation._resolve_tau_repo", return_value=repo),
                patch("flightrecorder.tau3_grounded_generation._git", return_value="a" * 40),
                patch("flightrecorder.tau3_grounded_generation._install_tau_import_shims"),
                patch("flightrecorder.tau3_grounded_generation._assert_vendored_tau_module_origins"),
                patch("flightrecorder.tau3_grounded_generation.importlib.import_module", side_effect=import_module),
            ):
                runtime = _VendoredTauRuntime(
                    domain="telecom",
                    revision="a" * 40,
                    state={
                        "task_initialization": task_initialization,
                        "pre_sync": pre_sync,
                        "post_sync": post_sync,
                    },
                    repo=repo,
                )

        self.assertEqual(runtime.system_prompt(), "runtime-derived-system-prompt")
        self.assertEqual(runtime.tool_catalog(), catalog)
        environment = environments[0]
        self.assertEqual(environment.get_tools_calls, 1)
        self.assertNotEqual(
            [item["function"]["name"] for item in catalog],
            sorted(item["function"]["name"] for item in catalog),
        )
        self.assertTrue(runtime.initial_sync_evidence["performed"])
        self.assertEqual(runtime.initial_sync_evidence["sync_count"], 2)
        self.assertEqual(
            [step["ordinal"] for step in runtime.initial_sync_evidence["steps"]],
            [0, 1],
        )
        self.assertEqual(runtime.initial_sync_evidence["pre_state"]["user_db"], pre_sync["user_db"])
        self.assertEqual(runtime.initial_sync_evidence["post_state"]["user_db"], post_sync["user_db"])
        self.assertEqual(
            runtime.initial_sync_evidence["steps"][0]["post_state"]["agent_db"],
            {"quota": 1},
        )
        self.assertEqual(
            runtime.initial_sync_evidence["steps"][1]["pre_state"]["agent_db"],
            {"quota": 2},
        )
        self.assertEqual(
            runtime.task_initialization_sha256,
            canonical_sha256(task_initialization),
        )
        self.assertEqual(
            FakeInitialState.validated_values,
            [task_initialization, task_initialization],
        )
        for replay_environment in environments[:2]:
            self.assertEqual(
                replay_environment.set_state_inputs,
                [
                    {
                        "initialization_data": task_initialization[
                            "initialization_data"
                        ],
                        "initialization_actions": task_initialization[
                            "initialization_actions"
                        ],
                        "message_history": task_initialization["message_history"],
                        "strict": True,
                    }
                ],
            )
            self.assertIsInstance(replay_environment.tools.db, FakeAgentDB)
            self.assertIsInstance(replay_environment.user_tools.db, FakeUserDB)
            self.assertIsNot(
                replay_environment.tools.db,
                replay_environment.user_tools.db,
            )
        self.assertEqual(
            runtime.original_initial_state_replay_evidence,
            runtime.initial_sync_evidence,
        )
        self.assertEqual(
            runtime.initial_sync_evidence["sequence_sha256"],
            canonical_sha256(runtime.initial_sync_evidence["steps"]),
        )
        self.assertEqual(
            runtime.initial_sync_evidence["steps"][1]["previous_step_sha256"],
            runtime.initial_sync_evidence["steps"][0]["step_sha256"],
        )
        self.assertEqual(
            runtime.materialized_constructor_sync_evidence["sync_count"],
            0,
        )
        self.assertFalse(runtime.materialized_constructor_sync_evidence["performed"])
        self.assertEqual(
            runtime.materialized_constructor_sync_evidence["sequence_sha256"],
            canonical_sha256([]),
        )
        self.assertEqual(
            runtime.materialized_constructor_sync_evidence["pre_state"],
            post_sync,
        )
        self.assertEqual(
            runtime.materialized_constructor_sync_evidence["post_state"],
            post_sync,
        )
        self.assertEqual(runtime.materialized_state_replay_evidence["sync_count"], 2)
        self.assertTrue(
            all(
                step["pre_state"] == post_sync
                and step["post_state"] == post_sync
                for step in runtime.materialized_state_replay_evidence["steps"]
            )
        )
        self.assertEqual(
            runtime.materialized_state_replay_evidence["pre_state"],
            post_sync,
        )
        self.assertEqual(
            runtime.materialized_state_replay_evidence["post_state"],
            post_sync,
        )
        runtime.call("write_quota", {"quota": 3})
        self.assertTrue(runtime.last_sync_evidence["performed"])
        self.assertEqual(runtime.last_sync_evidence["pre_state"]["user_db"]["mirrored_quota"], 2)
        self.assertEqual(runtime.last_sync_evidence["post_state"]["user_db"]["mirrored_quota"], 3)

    def test_initial_sync_sequence_fails_closed_at_zero_and_accepts_one_or_multiple(self) -> None:
        """Candidate replay requires >=1 physical sync; valid ordered chains are not singleton-only."""

        first_pre = {"agent_db": {"quota": 1}, "user_db": {"quota": 0}}
        first_post = {"agent_db": {"quota": 1}, "user_db": {"quota": 1}}
        second_pre = {"agent_db": {"quota": 2}, "user_db": {"quota": 1}}
        second_post = {"agent_db": {"quota": 2}, "user_db": {"quota": 2}}

        with self.assertRaisesRegex(
            Tau3GroundedGenerationError,
            "at least one sync_tools call",
        ):
            _ordered_initial_sync_evidence([])

        one = _ordered_initial_sync_evidence([(first_pre, first_post)])
        self.assertEqual(one["sync_count"], 1)
        self.assertEqual(one["steps"][0]["ordinal"], 0)
        self.assertIsNone(one["steps"][0]["previous_step_sha256"])
        self.assertEqual(one["sequence_sha256"], canonical_sha256(one["steps"]))

        multiple = _ordered_initial_sync_evidence(
            [(first_pre, first_post), (second_pre, second_post)]
        )
        self.assertEqual(multiple["sync_count"], 2)
        self.assertEqual([step["ordinal"] for step in multiple["steps"]], [0, 1])
        self.assertEqual(multiple["pre_state"], first_pre)
        self.assertEqual(multiple["post_state"], second_post)
        self.assertEqual(
            multiple["steps"][1]["previous_step_sha256"],
            multiple["steps"][0]["step_sha256"],
        )
        self.assertEqual(
            multiple["sequence_sha256"],
            canonical_sha256(multiple["steps"]),
        )

    def test_builder_rejects_nonruntime_prompt_and_nonexact_tool_catalog(self) -> None:
        runtime = _FakeTestTauRuntime({"records": {}, "notes": []})
        exact_catalog = runtime.tool_catalog()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            row = _scenario(split="train", domain="airline", family_index=0, behavior="clarification_refusal")
            row["tool_catalog"] = copy.deepcopy(exact_catalog)
            _write_jsonl(source, [row])
            build_tau3_grounded_generation_dataset(source=source, out_dir=root / "exact", strict_coverage=False)

        alternatives = [
            sorted(exact_catalog, key=lambda item: item["function"]["name"]),
            [copy.deepcopy(item["function"]) for item in exact_catalog],
        ]
        for index, alternative in enumerate(alternatives):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = root / "source.jsonl"
                row = _scenario(split="train", domain="airline", family_index=0, behavior="clarification_refusal")
                row["tool_catalog"] = alternative
                _write_jsonl(source, [row])
                with self.assertRaisesRegex(Tau3GroundedGenerationError, "exact runtime-derived Tau catalog"):
                    build_tau3_grounded_generation_dataset(
                        source=source,
                        out_dir=root / "rejected",
                        strict_coverage=False,
                    )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            row = _scenario(split="train", domain="airline", family_index=0, behavior="clarification_refusal")
            row["runtime_family"] = "vendored_tau_tools@" + "a" * 40
            row["system_prompt"] = "source-supplied-system-prompt"
            runtime.system_prompt = lambda: "runtime-derived-system-prompt"  # type: ignore[method-assign]
            _write_jsonl(source, [row])
            with (
                patch("flightrecorder.tau3_grounded_generation._runtime_for_scenario", return_value=runtime),
                self.assertRaisesRegex(Tau3GroundedGenerationError, "exact Tau LLMAgent runtime prompt"),
            ):
                build_tau3_grounded_generation_dataset(
                    source=source,
                    out_dir=root / "rejected",
                    strict_coverage=False,
                )

    def test_argument_grounding_is_visible_and_strictly_chronological(self) -> None:
        turns = [
            {
                "user": {"content": "Use account acct-7."},
                "assistant": {"decision_ordinal": 0, "tool_calls": []},
            },
            {
                "user": {"content": "Proceed."},
                "assistant": {"decision_ordinal": 1, "tool_calls": []},
            },
        ]
        visible = _argument_grounding_evidence({"account_id": "acct-7"}, 0, turns, [])
        self.assertEqual(visible[0]["source"], "visible_user")

        prior_result = {"account_id": "acct-8"}
        prior_call = {
            "evidence_replayed": True,
            "parent_assistant_decision_ordinal": 0,
            "canonical_result": prior_result,
            "result_sha256": canonical_sha256(prior_result),
        }
        replayed = _argument_grounding_evidence({"account_id": "acct-8"}, 1, turns, [prior_call])
        self.assertEqual(replayed[0]["source"], "prior_tool_result")
        with self.assertRaisesRegex(Tau3GroundedGenerationError, "not grounded"):
            _argument_grounding_evidence({"account_id": "invented-9"}, 1, turns, [prior_call])

        same_decision = copy.deepcopy(prior_call)
        same_decision["parent_assistant_decision_ordinal"] = 1
        with self.assertRaisesRegex(Tau3GroundedGenerationError, "not grounded"):
            _argument_grounding_evidence({"account_id": "acct-8"}, 1, turns, [same_decision])

    def test_turn_decisions_must_match_physical_order(self) -> None:
        def duplicate_decision(row: dict[str, Any]) -> None:
            row["turns"][2]["assistant"]["decision_ordinal"] = 1

        def reverse_decisions(row: dict[str, Any]) -> None:
            row["turns"][0]["assistant"]["decision_ordinal"] = 1
            row["turns"][1]["assistant"]["decision_ordinal"] = 0

        for mutate in (duplicate_decision, reverse_decisions):
            with self.subTest(case=mutate.__name__), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = root / "source.jsonl"
                row = _scenario(split="train", domain="airline", family_index=0, behavior="successful_completion")
                mutate(row)
                _write_jsonl(source, [row])
                with self.assertRaisesRegex(Tau3GroundedGenerationError, "physical turn ordinal"):
                    build_tau3_grounded_generation_dataset(
                        source=source,
                        out_dir=root / "rejected",
                        strict_coverage=False,
                    )

    def test_confirmation_financial_details_require_prior_grounding(self) -> None:
        turns = [
            {
                "user": {"content": "Check the current total."},
                "assistant": {"decision_ordinal": 0, "tool_calls": []},
            },
            {
                "user": {"content": "Show me the confirmation."},
                "assistant": {"decision_ordinal": 1, "tool_calls": []},
            },
        ]
        confirmation = {"required_details": [{"kind": "price", "value": "USD-25"}]}
        prior_result = {"quoted_price": "USD-25"}
        prior_call = {
            "evidence_replayed": True,
            "parent_assistant_decision_ordinal": 0,
            "canonical_result": prior_result,
            "result_sha256": canonical_sha256(prior_result),
        }
        grounded = _confirmation_detail_grounding(confirmation, 1, turns, [prior_call])
        self.assertEqual(grounded[0]["source"], "prior_tool_result")
        invented = {"required_details": [{"kind": "price", "value": "USD-0"}]}
        with self.assertRaisesRegex(Tau3GroundedGenerationError, "not grounded"):
            _confirmation_detail_grounding(invented, 1, turns, [prior_call])

    def test_every_pinned_mutation_has_an_exact_confirmation_rule(self) -> None:
        expected = {
            "airline": {
                "book_reservation",
                "cancel_reservation",
                "send_certificate",
                "update_reservation_baggages",
                "update_reservation_flights",
                "update_reservation_passengers",
            },
            "retail": {
                "cancel_pending_order",
                "exchange_delivered_order_items",
                "modify_pending_order_address",
                "modify_pending_order_items",
                "modify_pending_order_payment",
                "modify_user_address",
                "return_delivered_order_items",
            },
            "telecom": {
                "disable_roaming",
                "enable_roaming",
                "refuel_data",
                "resume_line",
                "send_payment_request",
                "suspend_line",
            },
        }
        expected_detail_kinds = {
            ("airline", "book_reservation"): "total",
            ("airline", "cancel_reservation"): "refund",
            ("airline", "update_reservation_baggages"): "price_difference",
            ("airline", "update_reservation_flights"): "price_difference",
            ("retail", "cancel_pending_order"): "refund",
            ("retail", "exchange_delivered_order_items"): "price_difference",
            ("retail", "modify_pending_order_items"): "price_difference",
            ("retail", "return_delivered_order_items"): "refund",
            ("telecom", "refuel_data"): "price",
        }
        for domain, tool_names in expected.items():
            for tool_name in tool_names:
                with self.subTest(domain=domain, tool_name=tool_name):
                    detail_kind = expected_detail_kinds.get((domain, tool_name))
                    suffix = (
                        "financial_confirmation"
                        if detail_kind is not None
                        else "detail_confirmation"
                    )
                    self.assertEqual(
                        _confirmation_rule(domain, tool_name),
                        (f"{domain}.{tool_name}.{suffix}", detail_kind),
                    )
        self.assertIsNone(_confirmation_rule("telecom", "unregistered_mutation"))
        self.assertEqual(
            _confirmation_rule("telecom", "update_record"),
            ("test.update_record.explicit_confirmation", None),
        )

    def test_nonfinancial_telecom_mutation_requires_adjacent_exact_confirmation(
        self,
    ) -> None:
        def fixture(*, intervening_turn: bool = False) -> tuple[
            list[dict[str, Any]],
            list[dict[str, Any]],
            list[dict[str, Any]],
            dict[str, Any],
            str,
        ]:
            policy_sha256 = canonical_sha256("fake-test-policy")
            arguments = {"phone_number": "+1-202-555-0199"}
            request_target = {
                "behavior": "confirmation_before_mutation",
                "kind": "assistant_message",
                "text": (
                    "Please confirm enable roaming: "
                    "phone number +1-202-555-0199."
                ),
            }
            mutation_turn_ordinal = 2 if intervening_turn else 1
            mutation_target = {
                "behavior": "later_task_completion_actions",
                "kind": "tool_call",
                "text": "",
                "tool_name": "enable_roaming",
                "arguments": copy.deepcopy(arguments),
            }
            _attach_policy_review(
                mutation_target,
                turn_ordinal=mutation_turn_ordinal,
                decision_ordinal=1,
                allowed=True,
                reason_id="synthetic_allowed_mutation",
            )
            turns = [
                {
                    "user": {"content": "Enable roaming for this phone number."},
                    "assistant": {
                        "decision_ordinal": 0,
                        "tool_calls": [],
                        "safe_corrected_target": request_target,
                    },
                }
            ]
            if intervening_turn:
                turns.append({"user": {"content": "Let me check first."}})
            turns.append(
                {
                    "user": {"content": "Yes."},
                    "assistant": {
                        "decision_ordinal": 1,
                        "tool_calls": [],
                        "safe_corrected_target": mutation_target,
                    },
                }
            )
            targets = [
                _target_for_decision({}, turns[0]["assistant"], 0, 0),
                _target_for_decision(
                    {},
                    turns[mutation_turn_ordinal]["assistant"],
                    1,
                    mutation_turn_ordinal,
                ),
            ]
            confirmation = {
                "policy_rule": "telecom.enable_roaming.detail_confirmation",
                "policy_sha256": policy_sha256,
                "request_decision_ordinal": 0,
                "arguments_sha256": canonical_sha256(arguments),
                "action_label": "enable roaming",
                "confirmed_arguments": _confirmed_argument_receipt(arguments),
                "required_details": [],
                "detail_grounding": [],
            }
            replay = [
                {
                    "parent_turn_ordinal": mutation_turn_ordinal,
                    "parent_assistant_decision_ordinal": 1,
                    "tool_name": "enable_roaming",
                    "canonical_arguments": arguments,
                    "arguments_sha256": canonical_sha256(arguments),
                    "tool_mutates_state": True,
                    "evidence_replayed": True,
                    "confirmation": confirmation,
                }
            ]
            return (
                turns,
                targets,
                replay,
                _test_reviewer_record(targets),
                policy_sha256,
            )

        def errors(
            evidence: tuple[
                list[dict[str, Any]],
                list[dict[str, Any]],
                list[dict[str, Any]],
                dict[str, Any],
                str,
            ],
        ) -> list[str]:
            turns, targets, replay, reviewer, policy_sha256 = evidence
            return _policy_call_errors(
                domain="telecom",
                turns=turns,
                targets=targets,
                replay=replay,
                policy_sha256=policy_sha256,
                reviewer_record=reviewer,
                candidate=False,
                context="test",
            )

        adjacent = fixture()
        self.assertEqual(errors(adjacent), [])
        self.assertEqual(adjacent[2][0]["confirmation"]["required_details"], [])

        missing = fixture()
        missing[2][0].pop("confirmation")
        self.assertTrue(
            any("lacks confirmation evidence" in error for error in errors(missing))
        )

        stale = fixture()
        stale[2][0]["confirmation"]["request_decision_ordinal"] = -1
        self.assertTrue(
            any("must immediately precede" in error for error in errors(stale))
        )

        nonadjacent = fixture(intervening_turn=True)
        nonadjacent_errors = errors(nonadjacent)
        self.assertTrue(
            any("not physically adjacent" in error for error in nonadjacent_errors),
            nonadjacent_errors,
        )

        mismatched = fixture()
        mismatched[2][0]["confirmation"]["arguments_sha256"] = canonical_sha256(
            {"phone_number": "+1-202-555-0100"}
        )
        self.assertTrue(
            any("arguments changed after confirmation" in error for error in errors(mismatched))
        )

    def test_policy_rejects_wrong_financial_detail_kind_even_when_value_is_visible(self) -> None:
        policy_sha256 = canonical_sha256("fake-test-policy")
        arguments = {"reservation_id": "R-1"}
        request_target = {
            "behavior": "confirmation_before_mutation",
            "kind": "assistant_message",
            "text": (
                "Confirm cancel reservation: reservation id R-1; "
                "price USD-10; refund USD-10."
            ),
        }
        mutation_target = {
            "behavior": "later_task_completion_actions",
            "kind": "tool_call",
            "text": "",
            "tool_name": "cancel_reservation",
            "arguments": copy.deepcopy(arguments),
        }
        _attach_policy_review(
            mutation_target,
            turn_ordinal=1,
            decision_ordinal=1,
            allowed=True,
            reason_id="synthetic_allowed_mutation",
        )
        turns = [
            {
                "user": {"content": "The quoted refund is USD-10."},
                "assistant": {
                    "decision_ordinal": 0,
                    "tool_calls": [],
                    "safe_corrected_target": request_target,
                },
            },
            {
                "user": {"content": "Yes."},
                "assistant": {
                    "decision_ordinal": 1,
                    "tool_calls": [],
                    "safe_corrected_target": mutation_target,
                },
            },
        ]
        targets = [
            _target_for_decision({}, turns[index]["assistant"], index, index)
            for index in range(2)
        ]
        confirmation = {
            "policy_rule": "airline.cancel_reservation.financial_confirmation",
            "policy_sha256": policy_sha256,
            "request_decision_ordinal": 0,
            "arguments_sha256": canonical_sha256(arguments),
            "action_label": "cancel reservation",
            "confirmed_arguments": _confirmed_argument_receipt(arguments),
            "required_details": [{"kind": "refund", "value": "USD-10"}],
        }
        confirmation["detail_grounding"] = _confirmation_detail_grounding(
            confirmation,
            0,
            turns,
            [],
        )
        replay = [
            {
                "parent_turn_ordinal": 1,
                "parent_assistant_decision_ordinal": 1,
                "tool_name": "cancel_reservation",
                "canonical_arguments": arguments,
                "arguments_sha256": canonical_sha256(arguments),
                "tool_mutates_state": True,
                "evidence_replayed": True,
                "confirmation": confirmation,
            }
        ]
        reviewer = _test_reviewer_record(targets)
        self.assertEqual(
            _policy_call_errors(
                domain="airline",
                turns=turns,
                targets=targets,
                replay=replay,
                policy_sha256=policy_sha256,
                reviewer_record=reviewer,
                candidate=False,
                context="test",
            ),
            [],
        )
        replay[0]["confirmation"]["required_details"][0]["kind"] = "price"
        errors = _policy_call_errors(
            domain="airline",
            turns=turns,
            targets=targets,
            replay=replay,
            policy_sha256=policy_sha256,
            reviewer_record=reviewer,
            candidate=False,
            context="test",
        )
        self.assertTrue(any("requires retained refund" in error for error in errors))

    def test_state_sync_references_and_candidate_artifacts_are_owner_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            row = _scenario(split="train", domain="airline", family_index=0, behavior="successful_completion")
            row["initial_state"]["agent_db"] = {"synthetic": True}
            row["initial_state"]["user_db"] = {"synthetic": True}
            _write_jsonl(source, [row])
            out = root / "bundle"
            build_tau3_grounded_generation_dataset(source=source, out_dir=out, strict_coverage=False)
            exported = json.loads((out / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
            refs = [
                exported["initial_state_ref"],
                exported["final_state_ref"],
                exported["initial_sync"]["pre_state_ref"],
                exported["initial_sync"]["post_state_ref"],
            ]
            for call in exported["tool_replay"]:
                refs.extend(
                    [
                        call["pre_state_ref"],
                        call["pre_sync_state_ref"],
                        call["post_state_ref"],
                        call["sync_evidence"]["pre_state_ref"],
                        call["sync_evidence"]["post_state_ref"],
                    ]
                )
            initial_state = json.loads((out / exported["initial_state_ref"]["path"]).read_text(encoding="utf-8"))
            self.assertIn("agent_db", initial_state)
            self.assertIn("user_db", initial_state)
            self.assertEqual(out.stat().st_mode & 0o777, 0o700)
            self.assertEqual((out / "states").stat().st_mode & 0o777, 0o700)
            for path in (out / "manifest.json", out / "train.jsonl", out / "validation.jsonl"):
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            for state_ref in refs:
                self.assertEqual((out / state_ref["path"]).stat().st_mode & 0o777, 0o600)

            (out / "train.jsonl").chmod(0o644)
            result = validate_tau3_grounded_generation_bundle(out, strict=False)
            self.assertFalse(result["passed"])
            self.assertTrue(
                any(
                    "owner-only" in error or "mode" in error or "broader" in error
                    for error in result["errors"]
                )
            )

            train_path = out / "train.jsonl"
            train_path.chmod(0o700)
            result = validate_tau3_grounded_generation_bundle(out, strict=False)
            self.assertFalse(result["passed"])
            self.assertTrue(any("broader than 0600" in error for error in result["errors"]))

            train_path.chmod(0o600)
            backing_path = out / "train.backing"
            train_path.rename(backing_path)
            train_path.symlink_to(backing_path.name)
            result = validate_tau3_grounded_generation_bundle(out, strict=False)
            self.assertFalse(result["passed"])
            self.assertTrue(any("symbolic link" in error for error in result["errors"]))

    def test_validator_rejects_intermediate_symlink_before_reading_split(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            _write_jsonl(
                source,
                [
                    _scenario(
                        split="train",
                        domain="airline",
                        family_index=0,
                        behavior="authentication",
                    )
                ],
            )
            out = root / "bundle"
            build_tau3_grounded_generation_dataset(
                source=source,
                out_dir=out,
                strict_coverage=False,
            )
            actual = out / "actual"
            actual.mkdir(mode=0o700)
            actual.chmod(0o700)
            (out / "train.jsonl").rename(actual / "train.jsonl")
            (out / "nested").symlink_to(actual.name, target_is_directory=True)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            manifest["files"]["train"]["path"] = "nested/train.jsonl"
            manifest["manifest_sha256"] = canonical_sha256(
                {
                    key: value
                    for key, value in manifest.items()
                    if key != "manifest_sha256"
                }
            )
            _write_json(out / "manifest.json", manifest)
            result = validate_tau3_grounded_generation_bundle(out, strict=False)
            self.assertFalse(result["passed"])
            self.assertTrue(any("symbolic link" in error for error in result["errors"]))

    def test_validator_rejects_tampered_sync_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            _write_jsonl(
                source,
                [_scenario(split="train", domain="airline", family_index=0, behavior="successful_completion")],
            )
            out = root / "bundle"
            build_tau3_grounded_generation_dataset(source=source, out_dir=out, strict_coverage=False)
            rows = [json.loads(line) for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()]
            rows[0]["initial_sync"]["post_state_sha256"] = "0" * 64
            rows[0]["metadata"]["row_sha256"] = canonical_sha256(
                {key: value for key, value in rows[0].items() if key != "metadata"}
                | {
                    "metadata": {
                        key: value
                        for key, value in rows[0]["metadata"].items()
                        if key != "row_sha256"
                    }
                }
            )
            _write_jsonl(out / "train.jsonl", rows)
            _rewrite_bundle_manifest(out)
            result = validate_tau3_grounded_generation_bundle(out, strict=False)
            self.assertFalse(result["passed"])
            self.assertTrue(any("initial_sync" in error for error in result["errors"]))

    def test_mutations_require_detailed_confirmation_and_affirmative_reply(self) -> None:
        def remove_confirmation(row: dict[str, Any]) -> None:
            row["turns"][1]["assistant"]["tool_calls"][0].pop("confirmation")

        def make_request_vague(row: dict[str, Any]) -> None:
            row["turns"][0]["assistant"]["safe_corrected_target"]["text"] = "Please confirm."

        def remove_argument_labels(row: dict[str, Any]) -> None:
            call = row["turns"][1]["assistant"]["tool_calls"][0]
            record_id = call["arguments"]["id"]
            row["turns"][0]["assistant"]["safe_corrected_target"]["text"] = (
                f"Please confirm update record {record_id} updated."
            )

        def remove_affirmative_reply(row: dict[str, Any]) -> None:
            row["turns"][1]["user"]["content"] = "No."

        def qualify_affirmative_with_negation(row: dict[str, Any]) -> None:
            row["turns"][1]["user"]["content"] = "Yes, but do not proceed."

        def change_arguments_after_confirmation(row: dict[str, Any]) -> None:
            row["turns"][1]["assistant"]["tool_calls"][0]["confirmation"]["arguments_sha256"] = "0" * 64

        def remove_action_label(row: dict[str, Any]) -> None:
            row["turns"][1]["assistant"]["tool_calls"][0]["confirmation"]["action_label"] = ""

        def remove_policy_review(row: dict[str, Any]) -> None:
            row["turns"][1]["assistant"]["safe_corrected_target"].pop("policy_review")

        def deny_policy_review(row: dict[str, Any]) -> None:
            row["turns"][1]["assistant"]["safe_corrected_target"]["policy_review"]["allowed"] = False

        def forge_target_bound_review(row: dict[str, Any]) -> None:
            review = row["turns"][1]["assistant"]["safe_corrected_target"]["policy_review"]
            review["canonical_target_sha256"] = "0" * 64
            review["review_receipt_sha256"] = canonical_sha256(
                {
                    key: value
                    for key, value in review.items()
                    if key != "review_receipt_sha256"
                }
            )

        cases = (
            remove_confirmation,
            make_request_vague,
            remove_argument_labels,
            remove_affirmative_reply,
            qualify_affirmative_with_negation,
            change_arguments_after_confirmation,
            remove_action_label,
            remove_policy_review,
            deny_policy_review,
            forge_target_bound_review,
        )
        for mutate in cases:
            with self.subTest(case=mutate.__name__), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = root / "source.jsonl"
                row = _scenario(split="train", domain="airline", family_index=0, behavior="successful_completion")
                mutate(row)
                _write_jsonl(source, [row])
                with self.assertRaises(Tau3GroundedGenerationError):
                    build_tau3_grounded_generation_dataset(
                        source=source,
                        out_dir=root / "rejected",
                        strict_coverage=False,
                    )

    def test_masked_action_requires_independently_reviewed_safe_correction(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            row = _scenario(
                split="train",
                domain="airline",
                family_index=0,
                behavior="harmful_mutation_correction",
                unsafe=True,
            )
            row["turns"][1]["assistant"]["safe_corrected_target"].pop("policy_review")
            _write_jsonl(source, [row])
            with self.assertRaisesRegex(Tau3GroundedGenerationError, "safe[_ ]correction"):
                build_tau3_grounded_generation_dataset(
                    source=source,
                    out_dir=root / "rejected",
                    strict_coverage=False,
                )

    def test_policy_reviewer_artifact_cannot_self_attest_independence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            row = _scenario(
                split="train",
                domain="airline",
                family_index=0,
                behavior="successful_completion",
            )
            artifact = row["reviewer"]["artifact"]
            artifact["independent_review_pass"] = False
            row["reviewer"]["sha256"] = canonical_sha256(artifact)
            target = row["turns"][1]["assistant"]["safe_corrected_target"]
            review = target["policy_review"]
            review["reviewer_artifact_sha256"] = row["reviewer"]["sha256"]
            review["review_receipt_sha256"] = canonical_sha256(
                {
                    key: value
                    for key, value in review.items()
                    if key != "review_receipt_sha256"
                }
            )
            row["reviewer"]["review_set_sha256"] = canonical_sha256([review])
            _write_jsonl(source, [row])
            with self.assertRaisesRegex(
                Tau3GroundedGenerationError,
                "independent review pass",
            ):
                build_tau3_grounded_generation_dataset(
                    source=source,
                    out_dir=root / "rejected",
                    strict_coverage=False,
                )

    def test_executed_masked_unsafe_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            row = _scenario(
                split="train",
                domain="airline",
                family_index=0,
                behavior="harmful_mutation_correction",
                unsafe=True,
            )
            first_turn = row["turns"][0]
            target = first_turn["assistant"]["safe_corrected_target"]
            arguments = copy.deepcopy(target["arguments"])
            first_turn["user"]["content"] = json.dumps(arguments, sort_keys=True)
            first_turn["assistant"]["tool_calls"] = [
                {"tool_name": target["tool_name"], "arguments": arguments}
            ]
            _write_jsonl(source, [row])
            with self.assertRaisesRegex(Tau3GroundedGenerationError, "masked unsafe action was executed"):
                build_tau3_grounded_generation_dataset(
                    source=source,
                    out_dir=root / "rejected",
                    strict_coverage=False,
                )

    def test_codex_provenance_and_content_addressed_selection_are_fail_closed(self) -> None:
        row = {
            "runtime_family": "vendored_tau_tools@" + "a" * 40,
            "source_family_id": canonical_sha256({"synthetic_family": 1}),
            "domain": "airline",
            "split": "train",
            "tau_revision": "a" * 40,
            "system_prompt": "synthetic-runtime-prompt",
            "tool_catalog": [{"type": "function", "function": {"name": "lookup"}}],
            "initial_state": {
                "task_initialization": None,
                "pre_sync": {"agent_db": {}, "user_db": None},
                "post_sync": {"agent_db": {}, "user_db": None},
            },
        }
        receipt = _candidate_selection_receipt(row)
        provenance = _candidate_generation_provenance(row, receipt)
        self.assertEqual(_generation_provenance_errors(provenance, True, "test"), [])
        scaled_provenance = copy.deepcopy(provenance)
        scaled_controller = scaled_provenance["controller_receipt"]
        scaled_controller.update(
            {
                "schema_version": "hfr.tau3_generation_controller_receipt.v2",
                "scaled_generation_started": True,
                "coverage_profile_id": SCALED_COVERAGE_PROFILE_ID,
                "generated_family_identifier": canonical_sha256(
                    "scaled-generated-family"
                ),
                "generation_variant_ordinal": 0,
            }
        )
        scaled_provenance["controller_receipt_sha256"] = canonical_sha256(
            scaled_controller
        )
        self.assertEqual(
            _generation_provenance_errors(
                scaled_provenance,
                True,
                "test",
                scaled_generation=True,
            ),
            [],
        )
        dishonest_scaled = copy.deepcopy(scaled_provenance)
        dishonest_scaled["controller_receipt"]["scaled_generation_started"] = False
        dishonest_scaled["controller_receipt_sha256"] = canonical_sha256(
            dishonest_scaled["controller_receipt"]
        )
        self.assertTrue(
            _generation_provenance_errors(
                dishonest_scaled,
                True,
                "test",
                scaled_generation=True,
            )
        )
        for field in ("provider_accessed", "network_accessed"):
            contradictory = copy.deepcopy(provenance)
            contradictory[field] = False
            self.assertTrue(_generation_provenance_errors(contradictory, True, "test"))
        contradictory = copy.deepcopy(provenance)
        contradictory["native_codex_inference_calls"] = 0
        self.assertTrue(_generation_provenance_errors(contradictory, True, "test"))
        legacy_ambiguous = copy.deepcopy(provenance)
        legacy_ambiguous["external_network_calls"] = 0
        self.assertTrue(_generation_provenance_errors(legacy_ambiguous, True, "test"))
        nested_extra = copy.deepcopy(provenance)
        nested_extra["inference_receipt"]["provider_absence_attested"] = True
        nested_extra["inference_receipt_sha256"] = canonical_sha256(
            nested_extra["inference_receipt"]
        )
        nested_extra["controller_receipt"][
            "generator_inference_receipt_sha256"
        ] = nested_extra["inference_receipt_sha256"]
        nested_extra["controller_receipt_sha256"] = canonical_sha256(
            nested_extra["controller_receipt"]
        )
        self.assertTrue(_generation_provenance_errors(nested_extra, True, "test"))
        wrong_controller = copy.deepcopy(provenance)
        wrong_controller["controller_receipt"]["controller_model"] = "different-model"
        wrong_controller["controller_receipt_sha256"] = canonical_sha256(
            wrong_controller["controller_receipt"]
        )
        self.assertTrue(_generation_provenance_errors(wrong_controller, True, "test"))
        tampered_nested = copy.deepcopy(provenance)
        tampered_nested["controller_receipt"]["training_started"] = True
        tampered_nested["controller_receipt_sha256"] = canonical_sha256(
            tampered_nested["controller_receipt"]
        )
        self.assertTrue(_generation_provenance_errors(tampered_nested, True, "test"))

        family_id = row["source_family_id"]
        specs, selection_rows, salt_sha256 = _candidate_selection_source(receipt, row)
        with (
            patch(
                "flightrecorder.tau3_grounded_generation.PERMITTED_SELECTION_SOURCES",
                specs,
            ),
            patch(
                "flightrecorder.tau3_grounded_generation.CAMPAIGN_SELECTION_SALT_SHA256",
                salt_sha256,
            ),
            patch(
                "flightrecorder.tau3_grounded_generation._load_permitted_selection_rows",
                return_value=selection_rows,
            ),
        ):
            self.assertEqual(
                _selection_receipt_errors(
                    receipt,
                    family_id,
                    True,
                    "test",
                    domain="airline",
                    split="train",
                ),
                [],
            )
        double_hashed = copy.deepcopy(receipt)
        double_hashed["selected_family_sha256"] = canonical_sha256(family_id)
        double_hashed["receipt_sha256"] = canonical_sha256(
            {key: value for key, value in double_hashed.items() if key != "receipt_sha256"}
        )
        with (
            patch(
                "flightrecorder.tau3_grounded_generation.PERMITTED_SELECTION_SOURCES",
                specs,
            ),
            patch(
                "flightrecorder.tau3_grounded_generation.CAMPAIGN_SELECTION_SALT_SHA256",
                salt_sha256,
            ),
            patch(
                "flightrecorder.tau3_grounded_generation._load_permitted_selection_rows",
                return_value=selection_rows,
            ),
        ):
            self.assertTrue(
                _selection_receipt_errors(
                    double_hashed,
                    family_id,
                    True,
                    "test",
                    domain="airline",
                    split="train",
                )
            )

    def test_selection_receipt_replays_source_payload_and_deterministic_minimum(self) -> None:
        row = {
            "runtime_family": "vendored_tau_tools@" + "a" * 40,
            "source_family_id": canonical_sha256("selected-family"),
            "domain": "airline",
            "split": "train",
            "tau_revision": "a" * 40,
            "initial_state": {
                "task_initialization": None,
                "pre_sync": {"agent_db": {}, "user_db": None},
                "post_sync": {"agent_db": {}, "user_db": None},
            },
        }
        receipt = _candidate_selection_receipt(row)
        specs, selection_rows, salt_sha256 = _candidate_selection_source(receipt, row)
        selected_row = selection_rows[0][1]
        selected_rank = receipt["selection_rank_sha256"]
        competitor = None
        for suffix in range(4096):
            candidate = copy.deepcopy(selected_row)
            candidate["task"]["id"] = f"synthetic-competitor-{suffix}"
            candidate["task_sha256"] = canonical_sha256(candidate["task"])
            canonical = json.dumps(
                candidate,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            rank = hashlib.sha256(
                receipt["selection_salt"].encode("utf-8") + b"\0" + canonical
            ).hexdigest()
            if rank < selected_rank:
                competitor = candidate
                break
        self.assertIsNotNone(competitor)
        with (
            patch(
                "flightrecorder.tau3_grounded_generation.PERMITTED_SELECTION_SOURCES",
                specs,
            ),
            patch(
                "flightrecorder.tau3_grounded_generation.CAMPAIGN_SELECTION_SALT_SHA256",
                salt_sha256,
            ),
            patch(
                "flightrecorder.tau3_grounded_generation._load_permitted_selection_rows",
                return_value=((1, selected_row), (2, competitor)),
            ),
        ):
            errors = _selection_receipt_errors(
                receipt,
                row["source_family_id"],
                True,
                "test",
                domain="airline",
                split="train",
            )
        self.assertTrue(any("deterministic minimum" in error for error in errors))

        tampered_source_row = copy.deepcopy(selected_row)
        tampered_source_row["task"]["initial_state"] = {"synthetic": True}
        with (
            patch(
                "flightrecorder.tau3_grounded_generation.PERMITTED_SELECTION_SOURCES",
                specs,
            ),
            patch(
                "flightrecorder.tau3_grounded_generation.CAMPAIGN_SELECTION_SALT_SHA256",
                salt_sha256,
            ),
            patch(
                "flightrecorder.tau3_grounded_generation._load_permitted_selection_rows",
                return_value=((1, tampered_source_row),),
            ),
        ):
            errors = _selection_receipt_errors(
                receipt,
                row["source_family_id"],
                True,
                "test",
                domain="airline",
                split="train",
            )
        self.assertTrue(any("task_initial_state_sha256" in error for error in errors))
        wrong_algorithm = copy.deepcopy(receipt)
        wrong_algorithm["algorithm"] = "unregistered"
        wrong_algorithm["receipt_sha256"] = canonical_sha256(
            {key: value for key, value in wrong_algorithm.items() if key != "receipt_sha256"}
        )
        with (
            patch(
                "flightrecorder.tau3_grounded_generation.PERMITTED_SELECTION_SOURCES",
                specs,
            ),
            patch(
                "flightrecorder.tau3_grounded_generation.CAMPAIGN_SELECTION_SALT_SHA256",
                salt_sha256,
            ),
            patch(
                "flightrecorder.tau3_grounded_generation._load_permitted_selection_rows",
                return_value=selection_rows,
            ),
        ):
            self.assertTrue(
                _selection_receipt_errors(
                    wrong_algorithm,
                    row["source_family_id"],
                    True,
                    "test",
                    domain="airline",
                    split="train",
                )
            )

    def test_candidate_write_path_is_create_only_and_rejects_empty_input(self) -> None:
        candidate = _scenario(
            split="train",
            domain="airline",
            family_index=0,
            behavior="authentication",
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "existing-source.jsonl"
            source.write_bytes(b"preserve-existing-source\n")
            source.chmod(0o600)
            out = root / "new-bundle"

            with self.assertRaisesRegex(
                Tau3GroundedGenerationError,
                "candidate source already exists",
            ):
                write_build_validate_tau3_grounded_generation_candidates(
                    candidates=[candidate],
                    source=source,
                    out_dir=out,
                    strict_coverage=False,
                )
            self.assertEqual(source.read_bytes(), b"preserve-existing-source\n")
            self.assertFalse(out.exists())

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "new-source.jsonl"
            out = root / "existing-bundle"
            out.mkdir(mode=0o700)
            marker = out / "preserve.txt"
            marker.write_bytes(b"preserve-existing-bundle\n")
            marker.chmod(0o600)

            with self.assertRaisesRegex(
                Tau3GroundedGenerationError,
                "output directory already exists",
            ):
                write_build_validate_tau3_grounded_generation_candidates(
                    candidates=[candidate],
                    source=source,
                    out_dir=out,
                    strict_coverage=False,
                )
            self.assertFalse(source.exists())
            self.assertEqual(marker.read_bytes(), b"preserve-existing-bundle\n")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "empty-source.jsonl"
            out = root / "empty-bundle"
            with self.assertRaisesRegex(
                Tau3GroundedGenerationError,
                "at least one row",
            ):
                write_build_validate_tau3_grounded_generation_candidates(
                    candidates=[],
                    source=source,
                    out_dir=out,
                    strict_coverage=False,
                )
            self.assertFalse(source.exists())
            self.assertFalse(out.exists())

    def test_candidate_contract_replays_end_to_end_with_only_coverage_blockers(self) -> None:
        class CandidateRuntime:
            def __init__(self, initial_state: dict[str, Any]) -> None:
                self._state = copy.deepcopy(initial_state["post_sync"])
                pre_sync = copy.deepcopy(initial_state["pre_sync"])
                post_sync = copy.deepcopy(initial_state["post_sync"])
                intermediate = copy.deepcopy(pre_sync)
                intermediate["user_db"] = {"synthetic": "intermediate"}
                step_states = (
                    (pre_sync, intermediate),
                    (intermediate, post_sync),
                )
                self.initial_sync_evidence = _ordered_initial_sync_evidence(
                    list(step_states)
                )
                self.last_sync_evidence: dict[str, Any] | None = None

            @property
            def state(self) -> dict[str, Any]:
                return self._state

            def tool_catalog(self) -> list[dict[str, Any]]:
                return _FakeTestTauRuntime({}).tool_catalog()

            @staticmethod
            def system_prompt() -> str:
                return "runtime-derived-system-prompt"

            @staticmethod
            def policy_sha256() -> str:
                return canonical_sha256("fake-test-policy")

            @staticmethod
            def tool_mutates_state(tool_name: str) -> bool:
                return tool_name == "update_record"

            def call(self, tool_name: str, arguments: dict[str, Any]) -> Any:
                fake = _FakeTestTauRuntime(self._state["agent_db"])
                result = fake.call(tool_name, arguments)
                self._state["agent_db"] = fake.state
                pre_sync = copy.deepcopy(self._state)
                post_sync = copy.deepcopy(self._state)
                self.last_sync_evidence = {
                    "performed": True,
                    "pre_state": pre_sync,
                    "post_state": post_sync,
                    "pre_state_sha256": canonical_sha256(pre_sync),
                    "post_state_sha256": canonical_sha256(post_sync),
                    "state_diff": _state_diff(pre_sync, post_sync),
                }
                return result

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            row = _scenario(split="train", domain="telecom", family_index=0, behavior="successful_completion")
            agent_state = copy.deepcopy(row["initial_state"])
            full_state = {
                "task_initialization": None,
                "pre_sync": {"agent_db": agent_state, "user_db": {"synthetic": "pre"}},
                "post_sync": {"agent_db": agent_state, "user_db": {"synthetic": "post"}},
            }
            row["initial_state"] = full_state
            row["source_family_id"] = canonical_sha256("synthetic-candidate-family")
            row["tau_revision"] = "1d244f5dca42944b67a379b44bfeb9f5748f189d"
            row["runtime_family"] = "vendored_tau_tools@" + row["tau_revision"]
            runtime = CandidateRuntime(full_state)
            row["system_prompt"] = runtime.system_prompt()
            row["tool_catalog"] = runtime.tool_catalog()
            row["selection_receipt"] = _candidate_selection_receipt(row)
            row["generation_provenance"] = _candidate_generation_provenance(
                row,
                row["selection_receipt"],
            )
            _promote_candidate_reviewer(
                row,
                policy_sha256=runtime.policy_sha256(),
            )
            specs, selection_rows, salt_sha256 = _candidate_selection_source(
                row["selection_receipt"],
                row,
            )
            out = root / "bundle"
            with (
                patch(
                    "flightrecorder.tau3_grounded_generation._runtime_for_scenario",
                    side_effect=lambda payload: CandidateRuntime(payload["initial_state"]),
                ),
                patch(
                    "flightrecorder.tau3_grounded_generation.PERMITTED_SELECTION_SOURCES",
                    specs,
                ),
                patch(
                    "flightrecorder.tau3_grounded_generation.CAMPAIGN_SELECTION_SALT_SHA256",
                    salt_sha256,
                ),
                patch(
                    "flightrecorder.tau3_grounded_generation._load_permitted_selection_rows",
                    return_value=selection_rows,
                ),
            ):
                write_result = write_build_validate_tau3_grounded_generation_candidates(
                    candidates=[row],
                    source=source,
                    out_dir=out,
                    strict_coverage=False,
                )
                result = write_result["validation"]
                baseline_rows = [
                    json.loads(line)
                    for line in (out / "train.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]
                tamper_errors: dict[str, list[str]] = {}
                for label in (
                    "zero_sequence",
                    "sequence_hash",
                    "step_hash",
                    "state_hash",
                    "state_diff",
                    "predecessor",
                    "step_order",
                    "missing_step",
                ):
                    rows = copy.deepcopy(baseline_rows)
                    if label == "zero_sequence":
                        rows[0]["initial_sync"]["performed"] = False
                        rows[0]["initial_sync"]["sync_count"] = 0
                        rows[0]["initial_sync"]["steps"] = []
                        rows[0]["initial_sync"]["sequence_sha256"] = canonical_sha256([])
                    elif label == "sequence_hash":
                        rows[0]["initial_sync"]["sequence_sha256"] = "0" * 64
                    elif label == "step_hash":
                        rows[0]["initial_sync"]["steps"][0]["step_sha256"] = "0" * 64
                    elif label == "state_hash":
                        rows[0]["initial_sync"]["steps"][0][
                            "post_state_sha256"
                        ] = "0" * 64
                    elif label == "state_diff":
                        rows[0]["initial_sync"]["steps"][0]["state_diff"] = {
                            "tampered": True
                        }
                    elif label == "predecessor":
                        rows[0]["initial_sync"]["steps"][1][
                            "previous_step_sha256"
                        ] = "0" * 64
                    elif label == "step_order":
                        rows[0]["initial_sync"]["steps"].reverse()
                    else:
                        rows[0]["initial_sync"]["steps"].pop(0)
                    rows[0]["metadata"]["row_sha256"] = canonical_sha256(
                        {
                            key: value
                            for key, value in rows[0].items()
                            if key != "metadata"
                        }
                        | {
                            "metadata": {
                                key: value
                                for key, value in rows[0]["metadata"].items()
                                if key != "row_sha256"
                            }
                        }
                    )
                    _write_jsonl(out / "train.jsonl", rows)
                    _rewrite_bundle_manifest(out)
                    tampered = validate_tau3_grounded_generation_bundle(
                        out,
                        strict=False,
                    )
                    tamper_errors[label] = tampered["errors"]

                _write_jsonl(out / "train.jsonl", baseline_rows)
                _rewrite_bundle_manifest(out)
                snapshot_ref = baseline_rows[0]["initial_sync"]["steps"][0][
                    "pre_state_ref"
                ]
                snapshot_path = out / snapshot_ref["path"]
                original_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
                for missing_field in ("agent_db", "user_db"):
                    incomplete = copy.deepcopy(original_snapshot)
                    incomplete[missing_field] = None
                    _write_json(snapshot_path, incomplete)
                    tampered = validate_tau3_grounded_generation_bundle(
                        out,
                        strict=False,
                    )
                    tamper_errors[f"incomplete_{missing_field}"] = tampered["errors"]
                    _write_json(snapshot_path, original_snapshot)
            coverage_blockers = set(result["coverage"]["blockers"])
            noncoverage = [error for error in result["errors"] if error not in coverage_blockers]
            self.assertTrue(write_result["contract_validated"])
            self.assertEqual(write_result["noncoverage_error_count"], 0)
            self.assertGreater(source.stat().st_size, 0)
            self.assertEqual(source.stat().st_mode & 0o777, 0o600)
            self.assertEqual(out.stat().st_mode & 0o777, 0o700)
            self.assertEqual(noncoverage, [])
            self.assertTrue(
                any(
                    "at least one physical sync step" in error
                    for error in tamper_errors["zero_sequence"]
                ),
                tamper_errors["zero_sequence"],
            )
            self.assertTrue(
                any(
                    "sequence_sha256 does not replay ordered steps" in error
                    for error in tamper_errors["sequence_hash"]
                ),
                tamper_errors["sequence_hash"],
            )
            self.assertTrue(
                any(
                    "steps[0].step_sha256 does not replay" in error
                    for error in tamper_errors["step_hash"]
                ),
                tamper_errors["step_hash"],
            )
            self.assertTrue(
                any(
                    "steps[0].post_state_sha256 does not replay" in error
                    for error in tamper_errors["state_hash"]
                ),
                tamper_errors["state_hash"],
            )
            self.assertTrue(
                any(
                    "steps[0].state_diff does not replay" in error
                    for error in tamper_errors["state_diff"]
                ),
                tamper_errors["state_diff"],
            )
            self.assertTrue(
                any(
                    "previous_step_sha256 breaks the physical hash chain" in error
                    for error in tamper_errors["predecessor"]
                ),
                tamper_errors["predecessor"],
            )
            self.assertTrue(
                any(
                    "ordinal must equal its physical order" in error
                    for error in tamper_errors["step_order"]
                ),
                tamper_errors["step_order"],
            )
            self.assertTrue(
                any(
                    "sync_count does not match ordered steps" in error
                    for error in tamper_errors["missing_step"]
                ),
                tamper_errors["missing_step"],
            )
            self.assertTrue(
                any(
                    "agent_db must be a complete object" in error
                    for error in tamper_errors["incomplete_agent_db"]
                ),
                tamper_errors["incomplete_agent_db"],
            )
            self.assertTrue(
                any(
                    "user_db must be a complete telecom object" in error
                    for error in tamper_errors["incomplete_user_db"]
                ),
                tamper_errors["incomplete_user_db"],
            )

    def test_scaled_selection_accepts_exact_stratum_ordinal_and_variant(self) -> None:
        initial_state = None
        family_a = canonical_sha256("scaled-family-a")
        family_b = canonical_sha256("scaled-family-b")
        source_rows = (
            (
                1,
                _scaled_source_task_row(
                    domain="airline",
                    source="train",
                    family_sha256=family_a,
                    task_index=1,
                    initial_state=initial_state,
                ),
            ),
            (
                2,
                _scaled_source_task_row(
                    domain="airline",
                    source="train",
                    family_sha256=family_b,
                    task_index=2,
                    initial_state=initial_state,
                ),
            ),
        )
        canonical_target = {
            "kind": "assistant_message",
            "text": "unit scaled target",
            "tool_name": None,
            "arguments": {},
        }
        target = {
            "behavior": "authentication",
            "masked": False,
            "canonical_target": canonical_target,
            "canonical_target_sha256": canonical_sha256(canonical_target),
        }
        generation_stratum = _generation_stratum_definition(
            source_provenance="reviewed_synthetic",
            targets=[target],
            tool_history=[],
            turns=[{"assistant": {"decision_ordinal": 0}}],
        )
        recipe = {
            "id": "scaled-unit-recipe",
            "version": "v1",
            "sha256": canonical_sha256(
                {"id": "scaled-unit-recipe", "version": "v1"}
            ),
        }
        row = {
            "runtime_family": "vendored_tau_tools@" + TAU_REVISION,
            "source_family_id": family_b,
            "domain": "airline",
            "split": "train",
            "tau_revision": TAU_REVISION,
            "initial_state": {
                "task_initialization": initial_state,
                "pre_sync": {"agent_db": {}, "user_db": None},
                "post_sync": {"agent_db": {}, "user_db": None},
            },
        }
        receipt, specs, salt_sha256 = _scaled_selection_receipt_fixture(
            row=row,
            numbered_source_rows=source_rows,
            selected_line=2,
            generation_stratum=generation_stratum,
            recipe=recipe,
            generation_variant_ordinal=0,
        )
        with (
            patch(
                "flightrecorder.tau3_grounded_generation.PERMITTED_SELECTION_SOURCES",
                specs,
            ),
            patch(
                "flightrecorder.tau3_grounded_generation.CAMPAIGN_SELECTION_SALT_SHA256",
                salt_sha256,
            ),
            patch(
                "flightrecorder.tau3_grounded_generation._load_permitted_selection_rows",
                return_value=source_rows,
            ),
        ):
            errors = _selection_receipt_errors(
                receipt,
                family_b,
                True,
                "scaled",
                domain="airline",
                split="train",
                tau_revision=TAU_REVISION,
                expected_generation_stratum=generation_stratum,
                expected_recipe=recipe,
            )
        self.assertEqual(errors, [])
        self.assertIn(receipt["rank_ordinal"], {0, 1})
        self.assertEqual(receipt["generation_variant_ordinal"], 0)
        self.assertEqual(
            receipt["schema_version"],
            SCALED_SELECTION_RECEIPT_SCHEMA_VERSION,
        )

    def test_scaled_selection_rejects_forged_bound_claims(self) -> None:
        initial_state = {"seed": "bound"}
        family = canonical_sha256("scaled-forgery-family")
        source_rows = (
            (
                7,
                _scaled_source_task_row(
                    domain="airline",
                    source="train",
                    family_sha256=family,
                    task_index=7,
                    initial_state=initial_state,
                ),
            ),
            (
                9,
                _scaled_source_task_row(
                    domain="airline",
                    source="train",
                    family_sha256=canonical_sha256("other-family"),
                    task_index=9,
                    initial_state={"seed": "other"},
                ),
            ),
        )
        generation_stratum = _generation_stratum_definition(
            source_provenance="official_train_derived",
            targets=[],
            tool_history=[],
            turns=[{}],
        )
        recipe = {
            "id": "scaled-forgery-recipe",
            "version": "v1",
            "sha256": canonical_sha256(
                {"id": "scaled-forgery-recipe", "version": "v1"}
            ),
        }
        row = {
            "runtime_family": "vendored_tau_tools@" + TAU_REVISION,
            "source_family_id": family,
            "domain": "airline",
            "split": "train",
            "tau_revision": TAU_REVISION,
            "initial_state": {
                "task_initialization": initial_state,
                "pre_sync": {"agent_db": {}, "user_db": None},
                "post_sync": {"agent_db": {}, "user_db": None},
            },
        }
        receipt, specs, salt_sha256 = _scaled_selection_receipt_fixture(
            row=row,
            numbered_source_rows=source_rows,
            selected_line=7,
            generation_stratum=generation_stratum,
            recipe=recipe,
        )

        def resign(value: dict[str, Any]) -> dict[str, Any]:
            value["receipt_sha256"] = canonical_sha256(
                {key: item for key, item in value.items() if key != "receipt_sha256"}
            )
            return value

        forgeries: dict[str, dict[str, Any]] = {}
        forged = copy.deepcopy(receipt)
        forged["rank_ordinal"] = 1 - int(receipt["rank_ordinal"])
        forgeries["ordinal"] = resign(forged)
        forged = copy.deepcopy(receipt)
        forged["selection_stratum_definition"]["domain"] = "retail"
        forged["selection_stratum_sha256"] = canonical_sha256(
            forged["selection_stratum_definition"]
        )
        forgeries["stratum"] = resign(forged)
        forged = copy.deepcopy(receipt)
        forged["task_sha256"] = "0" * 64
        forgeries["source_task"] = resign(forged)
        forged = copy.deepcopy(receipt)
        forged["source_family_sha256"] = "1" * 64
        forgeries["source_family"] = resign(forged)
        forged = copy.deepcopy(receipt)
        forged["generation_variant_ordinal"] = 1
        forgeries["variant"] = resign(forged)
        forged = copy.deepcopy(receipt)
        forged["task_initial_state_sha256"] = "2" * 64
        forgeries["task_initialization"] = resign(forged)
        forged = copy.deepcopy(receipt)
        forged["prompt_sha256"] = "3" * 64
        forgeries["prompt"] = resign(forged)
        forged = copy.deepcopy(receipt)
        forged["mapped_grounded_split"] = "validation"
        forged["selection_stratum_definition"]["mapped_grounded_split"] = "validation"
        forged["selection_stratum_sha256"] = canonical_sha256(
            forged["selection_stratum_definition"]
        )
        forgeries["cross_split"] = resign(forged)

        with (
            patch(
                "flightrecorder.tau3_grounded_generation.PERMITTED_SELECTION_SOURCES",
                specs,
            ),
            patch(
                "flightrecorder.tau3_grounded_generation.CAMPAIGN_SELECTION_SALT_SHA256",
                salt_sha256,
            ),
            patch(
                "flightrecorder.tau3_grounded_generation._load_permitted_selection_rows",
                return_value=source_rows,
            ),
        ):
            for label, forged_receipt in forgeries.items():
                with self.subTest(label=label):
                    self.assertTrue(
                        _selection_receipt_errors(
                            forged_receipt,
                            family,
                            True,
                            "scaled",
                            domain="airline",
                            split="train",
                            tau_revision=TAU_REVISION,
                            expected_generation_stratum=generation_stratum,
                            expected_recipe=recipe,
                        )
                    )

    def test_scaled_selection_rejects_skips_duplicates_and_family_reuse(self) -> None:
        rows = _scaled_coverage_fixture()
        baseline = _scaled_selection_claim_errors(rows)
        self.assertEqual(baseline, [])

        duplicate_rows = {"train": [copy.deepcopy(rows["train"][0])], "validation": []}
        duplicate_rows["train"].append(copy.deepcopy(duplicate_rows["train"][0]))
        duplicate_errors = _scaled_selection_claim_errors(duplicate_rows)
        self.assertIn("E_SCALE_SELECTION_DUPLICATE_ORDINAL", duplicate_errors)

        skipped_rows = {
            "train": [copy.deepcopy(rows["train"][0]), copy.deepcopy(rows["train"][1])],
            "validation": [],
        }
        first = skipped_rows["train"][0]["metadata"]["selection_receipt"]
        second = skipped_rows["train"][1]["metadata"]["selection_receipt"]
        second["selection_stratum_sha256"] = first["selection_stratum_sha256"]
        second["rank_ordinal"] = 2
        skipped_errors = _scaled_selection_claim_errors(skipped_rows)
        self.assertIn("E_SCALE_SELECTION_SKIPPED_ORDINAL", skipped_errors)

        reused_rows = {
            "train": [copy.deepcopy(rows["train"][0]), copy.deepcopy(rows["train"][1])],
            "validation": [],
        }
        reused_rows["train"][1]["metadata"]["selection_receipt"][
            "generated_family_identifier"
        ] = reused_rows["train"][0]["metadata"]["selection_receipt"][
            "generated_family_identifier"
        ]
        reuse_errors = _scaled_selection_claim_errors(reused_rows)
        self.assertIn("E_SCALE_GENERATED_FAMILY_REUSE", reuse_errors)

    def test_scaled_receipts_cannot_downgrade_to_implicit_pilot_profile(self) -> None:
        rows = _scaled_coverage_fixture()
        self.assertEqual(
            _selection_profile_errors(rows, LEGACY_COVERAGE_PROFILE_ID),
            ["E_SCALE_COVERAGE_PROFILE_DOWNGRADE"],
        )
        self.assertEqual(
            _selection_profile_errors(rows, SCALED_COVERAGE_PROFILE_ID),
            [],
        )

    def test_selection_receipt_binds_official_task_initialization(self) -> None:
        family_id = canonical_sha256("synthetic-family")
        row = {
            "runtime_family": "vendored_tau_tools@" + "a" * 40,
            "source_family_id": family_id,
            "domain": "airline",
            "split": "train",
            "tau_revision": "a" * 40,
            "initial_state": {
                "task_initialization": None,
                "pre_sync": {"agent_db": {}, "user_db": None},
                "post_sync": {"agent_db": {}, "user_db": None},
            },
        }
        row["selection_receipt"] = _candidate_selection_receipt(row)
        specs, selection_rows, salt_sha256 = _candidate_selection_source(
            row["selection_receipt"],
            row,
        )
        with (
            patch(
                "flightrecorder.tau3_grounded_generation.PERMITTED_SELECTION_SOURCES",
                specs,
            ),
            patch(
                "flightrecorder.tau3_grounded_generation.CAMPAIGN_SELECTION_SALT_SHA256",
                salt_sha256,
            ),
            patch(
                "flightrecorder.tau3_grounded_generation._load_permitted_selection_rows",
                return_value=selection_rows,
            ),
        ):
            self.assertEqual(_selection_receipt(row), row["selection_receipt"])
            original_revision = row["tau_revision"]
            row["tau_revision"] = "b" * 40
            with self.assertRaisesRegex(Tau3GroundedGenerationError, "source_revision"):
                _selection_receipt(row)
            row["tau_revision"] = original_revision
            row["initial_state"]["task_initialization"] = {
                "initialization_data": None,
                "initialization_actions": None,
                "message_history": None,
            }
            with self.assertRaisesRegex(Tau3GroundedGenerationError, "task initialization"):
                _selection_receipt(row)

    def test_scaled_full_rubric_aggregate_passes_from_admitted_targets(self) -> None:
        coverage = _coverage(
            _scaled_coverage_fixture(),
            profile_id=SCALED_COVERAGE_PROFILE_ID,
            training_handoff=_scaled_training_handoff(),
        )
        self.assertTrue(coverage["passed"], coverage["blocker_records"])
        self.assertEqual(coverage["blockers"], [])
        self.assertEqual(
            coverage["coverage_profile"],
            _coverage_profile_record(SCALED_COVERAGE_PROFILE_ID),
        )
        self.assertEqual(
            coverage["by_split"]["train"]["airline"]["behaviors"][
                "authentication"
            ]["admitted_target_count"],
            24,
        )
        self.assertEqual(
            coverage["by_split"]["validation"]["telecom"]["behaviors"][
                "authentication"
            ]["admitted_target_count"],
            6,
        )
        replayed_sha = canonical_sha256(
            {key: value for key, value in coverage.items() if key != "coverage_sha256"}
        )
        self.assertEqual(coverage["coverage_sha256"], replayed_sha)

    def test_scaled_manifest_and_validation_result_expose_bound_aggregate_evidence(self) -> None:
        rows = _scaled_coverage_fixture()
        handoff = _scaled_training_handoff()
        coverage = _coverage(
            rows,
            profile_id=SCALED_COVERAGE_PROFILE_ID,
            training_handoff=handoff,
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            source.write_text("{}\n", encoding="utf-8")
            out = root / "bundle"
            out.mkdir()
            for split in ("train", "validation"):
                split_path = out / f"{split}.jsonl"
                split_path.write_text("", encoding="utf-8")
                split_path.chmod(0o600)
            manifest = _manifest(
                source,
                out,
                rows,
                coverage,
                profile_id=SCALED_COVERAGE_PROFILE_ID,
                training_handoff=handoff,
            )
        self.assertEqual(
            manifest["coverage_profile"],
            _coverage_profile_record(SCALED_COVERAGE_PROFILE_ID),
        )
        self.assertEqual(manifest["coverage"], coverage)
        self.assertEqual(
            manifest["hash_disjointness"],
            coverage["hash_disjointness"],
        )
        self.assertEqual(manifest["training_handoff"], handoff)
        self.assertEqual(
            manifest["manifest_sha256"],
            canonical_sha256(
                {
                    key: value
                    for key, value in manifest.items()
                    if key != "manifest_sha256"
                }
            ),
        )
        schema_result = check_schema_contract(
            manifest,
            name_or_id="tau3_grounded_generation",
        )
        self.assertTrue(schema_result["passed"], schema_result["errors"])
        result = _validation_result(
            Path("bounded-scaled-bundle"),
            True,
            [],
            coverage,
            True,
            profile_id=SCALED_COVERAGE_PROFILE_ID,
        )
        self.assertEqual(result["coverage"], coverage)
        self.assertEqual(
            result["coverage_profile"],
            _coverage_profile_record(SCALED_COVERAGE_PROFILE_ID),
        )
        self.assertEqual(
            result["validation_sha256"],
            canonical_sha256(
                {
                    key: value
                    for key, value in result.items()
                    if key != "validation_sha256"
                }
            ),
        )

    def test_scaled_quantitative_boundaries_pass_at_equality_and_fail_below(self) -> None:
        for minimum in (16, 4, 8, 2, 24, 6, 8, 2):
            with self.subTest(minimum=minimum):
                self.assertTrue(_meets_minimum(minimum, minimum))
                self.assertFalse(_meets_minimum(minimum - 1, minimum))
        self.assertTrue(_fraction_at_least(25, 100, 1, 4))
        self.assertFalse(_fraction_at_least(24, 100, 1, 4))
        self.assertTrue(_fraction_at_most(40, 100, 2, 5))
        self.assertFalse(_fraction_at_most(41, 100, 2, 5))
        self.assertTrue(_ratio_at_most(160, 100, 8, 5))
        self.assertFalse(_ratio_at_most(161, 100, 8, 5))
        self.assertTrue(_fraction_at_most(20, 100, 1, 5))
        self.assertFalse(_fraction_at_most(21, 100, 1, 5))
        self.assertTrue(_fraction_at_most(1, 100, 1, 100))
        self.assertFalse(_fraction_at_most(2, 100, 1, 100))
        equality = _dominance_record(
            __import__("collections").Counter(
                {canonical_sha256(index): 1 for index in range(5)}
            )
        )
        one_over = _dominance_record(
            __import__("collections").Counter(
                {
                    canonical_sha256(0): 2,
                    canonical_sha256(1): 1,
                    canonical_sha256(2): 1,
                    canonical_sha256(3): 1,
                    canonical_sha256(4): 1,
                }
            )
        )
        self.assertTrue(equality["passed"])
        self.assertFalse(one_over["passed"])

    def test_scaled_behavior_counts_ignore_metadata_and_masked_targets(self) -> None:
        rows = _scaled_coverage_fixture()
        for row in rows["train"]:
            if row["metadata"]["domain"] != "airline":
                continue
            row["metadata"]["behaviors"] = list(BEHAVIORS)
            for target in row["training_targets"]:
                if target.get("behavior") == "authentication":
                    target["masked"] = True
        coverage = _coverage(
            rows,
            profile_id=SCALED_COVERAGE_PROFILE_ID,
            training_handoff=_scaled_training_handoff(),
        )
        metric = coverage["by_split"]["train"]["airline"]["behaviors"][
            "authentication"
        ]
        self.assertEqual(metric["admitted_target_count"], 0)
        self.assertEqual(metric["generated_family_count"], 0)
        self.assertIn("E_SCALE_BEHAVIOR_TARGET_COUNT", coverage["blockers"])
        self.assertIn("E_SCALE_BEHAVIOR_FAMILY_SPAN", coverage["blockers"])

    def test_scaled_correction_coverage_requires_reviewed_masked_context(self) -> None:
        rows = _scaled_coverage_fixture()
        airline_row = next(
            row
            for row in rows["train"]
            if row["metadata"]["domain"] == "airline"
        )
        removed = False
        retained: list[dict[str, Any]] = []
        for target in airline_row["training_targets"]:
            if (
                not removed
                and target.get("masked") is True
                and target.get("behavior") == "harmful_mutation_correction"
            ):
                removed = True
                continue
            retained.append(target)
        airline_row["training_targets"] = retained
        coverage = _coverage(
            rows,
            profile_id=SCALED_COVERAGE_PROFILE_ID,
            training_handoff=_scaled_training_handoff(),
        )
        metric = coverage["by_split"]["train"]["airline"]["behaviors"][
            "harmful_mutation_correction"
        ]
        self.assertEqual(metric["admitted_target_count"], 24)
        self.assertEqual(metric["reviewed_correction_pair_count"], 23)
        self.assertIn("E_SCALE_CORRECTION_CONTEXT_COUNT", coverage["blockers"])

    def test_scaled_zero_argument_exemption_is_exact_and_tamper_evident(self) -> None:
        row = _scaled_aggregate_row(
            split="train",
            domain="airline",
            family_index=0,
        )
        zero_tool = {
            "type": "function",
            "function": {
                "name": "zero_lookup",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }
        row["tool_catalog"].append(zero_tool)
        row["metadata"]["tool_catalog_sha256"] = canonical_sha256(row["tool_catalog"])
        row["metadata"]["generation_provenance"] = {
            "inference_receipt_sha256": canonical_sha256("generator")
        }
        exemption = {
            "schema_version": TOOL_EXEMPTION_SCHEMA_VERSION,
            "domain": "airline",
            "tool_name": "zero_lookup",
            "reason": "zero_arg",
            "scope": [
                "canonical_argument_payload_count",
                "supervised_target_count",
            ],
            "reviewer": "independent-ultra-reviewer",
            "reviewer_artifact_sha256": canonical_sha256("reviewer-artifact"),
            "reviewer_inference_receipt_sha256": canonical_sha256(
                "reviewer-inference"
            ),
            "reviewer_model": "gpt-5.6-sol",
            "reviewer_reasoning_effort": "ultra",
            "independent_review": True,
            "review_artifact": {},
            "tool_catalog_sha256": row["metadata"]["tool_catalog_sha256"],
            "policy_hash": row["metadata"]["policy_sha256"],
            "evidence_sha256": canonical_sha256("zero-argument-evidence"),
            "citation": "runtime-schema-required-arguments-empty",
        }
        review_inference = {
            "schema_version": TOOL_EXEMPTION_REVIEW_INFERENCE_SCHEMA_VERSION,
            "inference_origin": "native_codex",
            "model": exemption["reviewer_model"],
            "reasoning_effort": exemption["reviewer_reasoning_effort"],
            "native_codex_inference_calls": 1,
            "provider_accessed": True,
            "network_accessed": True,
            "prohibited_external_model_provider_calls": 0,
            "prohibited_external_network_calls": 0,
        }
        exemption["reviewer_inference_receipt_sha256"] = canonical_sha256(
            review_inference
        )
        exemption["review_artifact"] = {
            "schema_version": TOOL_EXEMPTION_REVIEW_SCHEMA_VERSION,
            "reviewer": exemption["reviewer"],
            "inference_receipt": review_inference,
            "review_scope": {
                key: copy.deepcopy(exemption[key])
                for key in (
                    "domain",
                    "tool_name",
                    "reason",
                    "scope",
                    "tool_catalog_sha256",
                    "policy_hash",
                    "evidence_sha256",
                    "citation",
                )
            },
            "independent_review_pass": True,
        }
        exemption["reviewer_artifact_sha256"] = canonical_sha256(
            exemption["review_artifact"]
        )
        exemption["receipt_sha256"] = canonical_sha256(exemption)
        self.assertEqual(
            _scaled_tool_exemption_errors(
                exemption,
                row=row,
                tool_name="zero_lookup",
            ),
            [],
        )
        for field, value in (
            ("reviewer_reasoning_effort", "xhigh"),
            ("tool_catalog_sha256", "0" * 64),
            ("receipt_sha256", "1" * 64),
        ):
            tampered = copy.deepcopy(exemption)
            tampered[field] = value
            self.assertTrue(
                _scaled_tool_exemption_errors(
                    tampered,
                    row=row,
                    tool_name="zero_lookup",
                ),
                field,
            )
        nonzero = copy.deepcopy(row)
        nonzero["tool_catalog"][-1]["function"]["parameters"]["required"] = [
            "value"
        ]
        nonzero["metadata"]["tool_catalog_sha256"] = canonical_sha256(
            nonzero["tool_catalog"]
        )
        rebound = copy.deepcopy(exemption)
        rebound["tool_catalog_sha256"] = nonzero["metadata"]["tool_catalog_sha256"]
        rebound["receipt_sha256"] = canonical_sha256(
            {key: value for key, value in rebound.items() if key != "receipt_sha256"}
        )
        self.assertIn(
            "E_SCALE_TOOL_EXEMPTION_ZERO_ARG_FALSE",
            _scaled_tool_exemption_errors(
                rebound,
                row=nonzero,
                tool_name="zero_lookup",
            ),
        )
        optional = copy.deepcopy(row)
        optional["tool_catalog"][-1]["function"]["parameters"]["properties"] = {
            "optional": {"type": "string"}
        }
        optional["metadata"]["tool_catalog_sha256"] = canonical_sha256(
            optional["tool_catalog"]
        )
        optional_exemption = copy.deepcopy(exemption)
        optional_exemption["tool_catalog_sha256"] = optional["metadata"][
            "tool_catalog_sha256"
        ]
        optional_exemption["review_artifact"]["review_scope"][
            "tool_catalog_sha256"
        ] = optional_exemption["tool_catalog_sha256"]
        optional_exemption["reviewer_artifact_sha256"] = canonical_sha256(
            optional_exemption["review_artifact"]
        )
        optional_exemption["receipt_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in optional_exemption.items()
                if key != "receipt_sha256"
            }
        )
        self.assertIn(
            "E_SCALE_TOOL_EXEMPTION_ZERO_ARG_FALSE",
            _scaled_tool_exemption_errors(
                optional_exemption,
                row=optional,
                tool_name="zero_lookup",
            ),
        )

    def test_scaled_token_balance_duplication_dominance_and_overlap_fail_closed(self) -> None:
        rows = _scaled_coverage_fixture()
        first_train = rows["train"][0]
        first_validation = rows["validation"][0]
        first_validation["metadata"]["selection_receipt"][
            "task_sha256"
        ] = first_train["metadata"]["selection_receipt"]["task_sha256"]

        telecom_targets = [
            target
            for row in rows["train"]
            if row["metadata"]["domain"] == "telecom"
            for target in row["training_targets"]
            if target.get("masked") is False
        ]
        for target in telecom_targets:
            target["canonical_target"]["text"] = "x" * 128
            target["canonical_target_sha256"] = canonical_sha256(
                target["canonical_target"]
            )
            if isinstance(target.get("policy_review"), dict):
                target["policy_review"]["canonical_target_sha256"] = target[
                    "canonical_target_sha256"
                ]
                target["policy_review"]["review_receipt_sha256"] = canonical_sha256(
                    {
                        key: value
                        for key, value in target["policy_review"].items()
                        if key != "review_receipt_sha256"
                    }
                )

        duplicate_canonical = {
            "kind": "assistant_message",
            "text": "duplicated-positive-target",
            "tool_name": None,
            "arguments": {},
        }
        changed = 0
        for row in rows["train"]:
            if row["metadata"]["domain"] != "airline":
                continue
            for target in row["training_targets"]:
                if target.get("masked") is not False or changed >= 100:
                    continue
                target["canonical_target"] = copy.deepcopy(duplicate_canonical)
                target["canonical_target_sha256"] = canonical_sha256(
                    duplicate_canonical
                )
                if isinstance(target.get("policy_review"), dict):
                    target["policy_review"]["canonical_target_sha256"] = target[
                        "canonical_target_sha256"
                    ]
                    target["policy_review"]["review_receipt_sha256"] = canonical_sha256(
                        {
                            key: value
                            for key, value in target["policy_review"].items()
                            if key != "review_receipt_sha256"
                        }
                    )
                changed += 1
        skewed = 0
        for row in rows["train"]:
            if row["metadata"]["domain"] != "retail":
                continue
            for target in row["training_targets"]:
                canonical = target.get("canonical_target")
                if (
                    not isinstance(canonical, dict)
                    or canonical.get("kind") != "tool_call"
                    or canonical.get("tool_name") == "lookup_0"
                    or skewed >= 40
                ):
                    continue
                canonical["tool_name"] = "lookup_0"
                target["canonical_target_sha256"] = canonical_sha256(canonical)
                if isinstance(target.get("policy_review"), dict):
                    target["policy_review"]["canonical_target_sha256"] = target[
                        "canonical_target_sha256"
                    ]
                    target["policy_review"]["review_receipt_sha256"] = canonical_sha256(
                        {
                            key: value
                            for key, value in target["policy_review"].items()
                            if key != "review_receipt_sha256"
                        }
                    )
                skewed += 1
        duplicate_row = copy.deepcopy(rows["train"][0])
        duplicate_receipt = duplicate_row["metadata"]["selection_receipt"]
        duplicate_receipt["selection_stratum_sha256"] = canonical_sha256(
            "duplicate-row-selection"
        )
        duplicate_receipt["task_sha256"] = canonical_sha256("duplicate-row-task")
        duplicate_receipt["prompt_sha256"] = canonical_sha256("duplicate-row-prompt")
        duplicate_receipt["source_family_sha256"] = canonical_sha256(
            "duplicate-row-family"
        )
        duplicate_receipt["generated_family_identifier"] = canonical_sha256(
            "duplicate-row-generated-family"
        )
        rows["train"].append(duplicate_row)

        coverage = _coverage(
            rows,
            profile_id=SCALED_COVERAGE_PROFILE_ID,
            training_handoff=_scaled_training_handoff(),
        )
        self.assertIn("E_SCALE_TASK_HASH_OVERLAP", coverage["blockers"])
        self.assertIn("E_SCALE_DOMAIN_TOKEN_SHARE", coverage["blockers"])
        self.assertIn("E_SCALE_DOMAIN_TOKEN_RATIO", coverage["blockers"])
        self.assertIn("E_SCALE_TARGET_DUPLICATION", coverage["blockers"])
        self.assertIn(
            "E_SCALE_TRAINING_EXAMPLE_DUPLICATION",
            coverage["blockers"],
        )
        self.assertIn("E_SCALE_TARGET_TOOL_DOMINANCE", coverage["blockers"])

    def test_scaled_hash_evidence_is_complete_and_all_overlap_dimensions_fail(self) -> None:
        overlap_errors = {
            "task_sha256": "E_SCALE_TASK_HASH_OVERLAP",
            "source_family_sha256": "E_SCALE_SOURCE_FAMILY_HASH_OVERLAP",
            "prompt_sha256": "E_SCALE_PROMPT_HASH_OVERLAP",
        }
        for field, error_id in overlap_errors.items():
            with self.subTest(field=field):
                rows = _scaled_coverage_fixture()
                rows["validation"][0]["metadata"]["selection_receipt"][field] = (
                    rows["train"][0]["metadata"]["selection_receipt"][field]
                )
                if field == "source_family_sha256":
                    rows["validation"][0]["metadata"]["source_family_id"] = rows[
                        "train"
                    ][0]["metadata"]["source_family_id"]
                coverage = _coverage(
                    rows,
                    profile_id=SCALED_COVERAGE_PROFILE_ID,
                    training_handoff=_scaled_training_handoff(),
                )
                self.assertIn(error_id, coverage["blockers"])

        rows = _scaled_coverage_fixture()
        rows["train"][0]["metadata"]["selection_receipt"].pop("task_sha256")
        coverage = _coverage(
            rows,
            profile_id=SCALED_COVERAGE_PROFILE_ID,
            training_handoff=_scaled_training_handoff(),
        )
        self.assertIn(
            "E_SCALE_TASK_HASH_EVIDENCE_INCOMPLETE",
            coverage["blockers"],
        )
        dimension = coverage["hash_disjointness"]["dimensions"]["task_sha256"]
        self.assertFalse(dimension["complete"])
        self.assertEqual(
            dimension["train_valid_claim_count"],
            dimension["train_row_count"] - 1,
        )

    def test_legacy_pilot_bundle_is_preserved_as_exact_confirmation_negative_evidence(
        self,
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        pilot = (
            root
            / "local/tau3/gpt56-sol-v3-data-campaign/pilot-repair-006/generated/grounded-bundle"
        )
        if not pilot.is_dir():
            self.skipTest("immutable pilot-006 bundle is absent")
        try:
            __import__("pydantic")
        except ModuleNotFoundError as exc:
            self.skipTest(f"vendored Tau dependencies unavailable: {exc}")
        try:
            result = validate_tau3_grounded_generation_bundle(pilot, strict=False)
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"vendored Tau dependencies unavailable: {exc}")
        blockers = set(result["coverage"]["blockers"])
        noncoverage = [error for error in result["errors"] if error not in blockers]
        self.assertEqual(
            result["coverage_profile"]["profile_id"],
            LEGACY_COVERAGE_PROFILE_ID,
        )
        self.assertEqual(len(blockers), 12)
        self.assertEqual(
            noncoverage,
            [
                "train[2].tool_replay[5] mutation lacks confirmation evidence",
                "validation[2].tool_replay[3] mutation lacks confirmation evidence",
            ],
        )
        self.assertEqual(
            _sha256(pilot / "manifest.json"),
            "5e3fee6e51b78b7af43f20adcfb550e0b2b940f462c26dd89b3f1820c5b2f279",
        )

    def test_schema_file_is_valid_json(self) -> None:
        schema_path = (
            Path(__file__).resolve().parents[1]
            / "flightrecorder"
            / "schemas"
            / "tau3_grounded_generation.v1.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(
            schema["properties"]["schema_version"]["const"],
            TAU3_GROUNDED_DATASET_SCHEMA_VERSION,
        )
        self.assertIn("artifact_security", schema["required"])
        self.assertIn("row_contract", schema["required"])
        self.assertIn("scaledToolCoverageExemption", schema["$defs"])
        self.assertIn("toolCoverageExemptionReview", schema["$defs"])
        hash_dimension = schema["$defs"]["hashDisjointDimension"]
        self.assertIn("train_valid_claim_count", hash_dimension["required"])
        self.assertIn("complete", hash_dimension["required"])
        self.assertIn("mode", schema["$defs"]["file"]["required"])
        self.assertIn("initial_sync", schema["$defs"]["candidateRowEvidence"]["required"])
        initial_sync_schema = schema["$defs"]["initialSyncEvidence"]
        self.assertIn("sync_count", initial_sync_schema["required"])
        self.assertIn("sequence_sha256", initial_sync_schema["required"])
        self.assertIn("steps", initial_sync_schema["required"])
        self.assertEqual(
            initial_sync_schema["properties"]["steps"]["minItems"],
            1,
        )
        sync_step_schema = initial_sync_schema["properties"]["steps"]["items"][
            "allOf"
        ][1]
        self.assertIn("previous_step_sha256", sync_step_schema["required"])
        self.assertIn("step_sha256", sync_step_schema["required"])
        metadata_schema = schema["$defs"]["candidateRowEvidence"]["properties"]["metadata"]
        self.assertIn("generation_provenance", metadata_schema["required"])
        self.assertIn("selection_receipt", metadata_schema["required"])
        self.assertIn("reviewer", metadata_schema["required"])
        row_contract = schema["properties"]["row_contract"]
        self.assertIn("task_initialization_receipt_required", row_contract["required"])
        self.assertIn("policy_target_review_required", row_contract["required"])
        self.assertIn(
            "ordered_initial_sync_sequence_required",
            row_contract["required"],
        )
        self.assertIn(
            "initial_sync_sequence_sha256_required",
            row_contract["required"],
        )
        provenance_schema = schema["$defs"]["generationProvenance"]
        self.assertEqual(
            provenance_schema["properties"]["schema_version"]["const"],
            "hfr.tau3_generation_provenance.v2",
        )
        self.assertIn("native_codex_inference_calls", provenance_schema["required"])
        self.assertIn("inference_receipt", provenance_schema["required"])
        self.assertIn("controller_receipt", provenance_schema["required"])
        controller_schema = schema["$defs"]["controllerReceiptV1"]
        self.assertIn("generator_inference_receipt_sha256", controller_schema["required"])
        self.assertIn("selection_receipt_sha256", controller_schema["required"])
        scaled_controller_schema = schema["$defs"]["controllerReceiptV2"]
        self.assertEqual(
            scaled_controller_schema["properties"]["scaled_generation_started"][
                "const"
            ],
            True,
        )
        self.assertIn("generated_family_identifier", scaled_controller_schema["required"])
        reviewer_schema = schema["$defs"]["policyReviewer"]
        self.assertIn("review_set_sha256", reviewer_schema["required"])
        confirmation_schema = schema["$defs"]["confirmationEvidence"]
        self.assertIn("confirmed_arguments", confirmation_schema["required"])
        selection_schema = schema["$defs"]["selectionReceiptV1"]
        self.assertEqual(
            selection_schema["properties"]["algorithm"]["const"],
            "sha256_ranked_deterministic_per_domain_split",
        )
        scaled_selection_schema = schema["$defs"]["selectionReceiptV2"]
        self.assertEqual(
            scaled_selection_schema["properties"]["algorithm"]["const"],
            SCALED_SELECTION_ALGORITHM,
        )
        self.assertIn("rank_ordinal", scaled_selection_schema["required"])
        self.assertIn(
            "generation_variant_ordinal",
            scaled_selection_schema["required"],
        )
        self.assertIn(
            "generated_family_identifier",
            scaled_selection_schema["required"],
        )
        self.assertIn("coverageProfile", schema["$defs"])
        self.assertIn("trainingHandoff", schema["$defs"])
        self.assertIn("hashDisjointness", schema["$defs"])

        registry_path = (
            Path(__file__).resolve().parents[1] / "flightrecorder" / "schemas" / "manifest.json"
        )
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        record = next(
            item for item in registry["schemas"] if item.get("name") == "tau3_grounded_generation"
        )
        description = record["description"].casefold()
        self.assertIn("full environment", description)
        self.assertIn("owner-only", description)
        self.assertIn("provenance", description)
        self.assertIn("selection", description)


if __name__ == "__main__":
    unittest.main()
