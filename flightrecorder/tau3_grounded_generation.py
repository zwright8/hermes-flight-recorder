"""Grounded Tau-3 trajectory generation for training-side evidence.

This module is dependency-free apart from an explicitly pinned Tau checkout and
fails closed. Candidate rows derive the LLMAgent prompt and ordered OpenAI tool
schemas from the runtime, replay assistant and user-side environment state
across every sync boundary, prove chronological argument and confirmation
grounding, and are written owner-only. Validation independently replays those
contracts; source claims and recorded hashes are evidence, not authority.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import hashlib
import importlib
import importlib.util
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import types
from collections import Counter
from functools import lru_cache
from dataclasses import dataclass
from datetime import date, datetime, time
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .path_safety import path_has_symlink_component

TAU3_GROUNDED_DATASET_SCHEMA_VERSION = "hfr.tau3_grounded_generation.v1"
TAU3_GROUNDED_ROW_SCHEMA_VERSION = "hfr.tau3_grounded_generation_row.v1"
LINEAGE_ID = "tau3-grounded-generation-v1"
DOMAINS = ("airline", "retail", "telecom")
SPLITS = ("train", "validation")
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
TRAIN_FAMILY_MIN = 8
VALIDATION_FAMILY_MIN = 2
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_SOURCE_FAMILIES = {"official_train_derived", "reviewed_synthetic"}
FAKE_TEST_RUNTIME_FAMILY = "fake_test_tau_tools"
VENDORED_RUNTIME_PREFIX = "vendored_tau_tools@"
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
OWNER_DIRECTORY_MODE = 0o700
OWNER_FILE_MODE = 0o600
CONTENT_ADDRESSED_FAMILY_SEMANTICS = "content_addressed_sha256"
CONTENT_ADDRESSED_SELECTION_ALGORITHM = "sha256_ranked_deterministic_per_domain_split"
SCALED_SELECTION_RECEIPT_SCHEMA_VERSION = "hfr.tau3_selection_receipt.v2"
SCALED_SELECTION_ALGORITHM = "sha256_ranked_deterministic_stratified_ordinal_v1"
SCALED_RANK_TIE_BREAKER_CONTRACT = (
    "selection_rank_sha256,task_sha256,canonical_source_row_sha256,source_line_number"
)
SELECTION_STRATUM_SCHEMA_VERSION = "hfr.tau3_selection_stratum.v1"
GENERATION_STRATUM_SCHEMA_VERSION = "hfr.tau3_generation_stratum.v1"
GENERATED_FAMILY_SCHEMA_VERSION = "hfr.tau3_generated_family.v1"
COVERAGE_PROFILE_SCHEMA_VERSION = "hfr.tau3_coverage_profile.v1"
LEGACY_COVERAGE_PROFILE_ID = "tau3_pilot_legacy_v1"
SCALED_COVERAGE_PROFILE_ID = "tau3_scaled_full_rubric_v1"
SCALED_COVERAGE_SCHEMA_VERSION = "hfr.tau3_scaled_coverage.v1"
TRAINING_HANDOFF_SCHEMA_VERSION = "hfr.tau3_training_handoff.v1"
TOOL_EXEMPTION_SCHEMA_VERSION = "hfr.tau3_tool_coverage_exemption.v1"
TOOL_EXEMPTION_REVIEW_SCHEMA_VERSION = (
    "hfr.tau3_tool_coverage_exemption_review.v1"
)
TOOL_EXEMPTION_REVIEW_INFERENCE_SCHEMA_VERSION = (
    "hfr.tau3_tool_coverage_exemption_review_inference.v1"
)
TOKEN_COUNT_SCHEMA_VERSION = "hfr.tau3_supervised_target_token_count.v1"
TOKEN_COUNT_ALGORITHM = "canonical_json_utf8_byte_tokens_v1"
MAX_MACHINE_COUNT = (1 << 53) - 1
SCALED_TOOL_TARGET_MINIMUMS = {"train": 16, "validation": 4}
SCALED_TOOL_ARGUMENT_MINIMUMS = {"train": 8, "validation": 2}
SCALED_BEHAVIOR_TARGET_MINIMUMS = {"train": 24, "validation": 6}
SCALED_BEHAVIOR_FAMILY_MINIMUMS = {"train": 8, "validation": 2}
NEGATIVE_CORRECTION_BEHAVIORS = {
    "hallucinated_tool_correction": "hallucinated_tool",
    "harmful_mutation_correction": "harmful_mutation",
    "premature_completion_correction": "premature_completion",
}
CAMPAIGN_SELECTION_SALT_SHA256 = (
    "80730fa6e6066c64f3e6231ba37fded807d537ffc770a8688c46b269c312dbbb"
)
NATIVE_CODEX_ORIGIN = "native_codex"
AFFIRMATIVE_REPLY_RE = re.compile(
    r"\s*(?:yes(?:,?\s+please)?|confirmed?|i confirm|i agree|please proceed|proceed|go ahead|do it)[.!]?\s*",
    re.IGNORECASE,
)
REGISTERED_MUTATING_TOOLS = {
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
CONFIRMATION_REQUIRED_TOOLS = {
    "airline": set(REGISTERED_MUTATING_TOOLS["airline"]),
    "retail": set(REGISTERED_MUTATING_TOOLS["retail"]),
    "telecom": set(REGISTERED_MUTATING_TOOLS["telecom"]),
}
REQUIRED_CONFIRMATION_DETAIL_KIND = {
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
PERMITTED_SELECTION_SOURCES: dict[str, dict[str, str]] = {
    "train": {
        "source_split": "train",
        "path": "local/tau3/source-v1/training_source/train_tasks.jsonl",
        "sha256": "debddcbf6ad27a59aa9ff7d93e7a3e2f3e126008c2ed55d51ca46e979e117916",
    },
    "validation": {
        "source_split": "development",
        "path": "local/tau3/source-v1/training_source/development_tasks.jsonl",
        "sha256": "0d431cb5d6b1e5b12606c9eb20d1b2e02b212d8c6cf835a0c04fe0ecdd574b32",
    },
}


class Tau3GroundedGenerationError(ValueError):
    """Raised when grounded generation or validation fails closed."""


@dataclass(frozen=True)
class _Scenario:
    index: int
    payload: dict[str, Any]
    row_sha256: str


def build_tau3_grounded_generation_dataset(
    *,
    source: str | Path,
    out_dir: str | Path,
    strict_coverage: bool = True,
    coverage_profile: str = LEGACY_COVERAGE_PROFILE_ID,
    training_handoff: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a grounded JSONL bundle from replayable train-side scenarios."""

    profile_id = _normalize_coverage_profile_id(coverage_profile)
    source_path = Path(source)
    out = Path(out_dir)
    _require_input_file(source_path, "source")
    _require_new_output_dir(out)
    scenarios = _read_scenarios(source_path)
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    for scenario in scenarios:
        row = _build_row(scenario)
        rows_by_split[row["metadata"]["split"]].append(row)
    if not any(rows_by_split.values()):
        raise Tau3GroundedGenerationError("source produced no replayable rows")
    profile_errors = _selection_profile_errors(rows_by_split, profile_id)
    if profile_errors:
        raise Tau3GroundedGenerationError("; ".join(profile_errors))

    coverage = _coverage(
        rows_by_split,
        profile_id=profile_id,
        training_handoff=training_handoff,
    )
    if strict_coverage and not coverage["passed"]:
        raise Tau3GroundedGenerationError(
            "coverage is incomplete: " + "; ".join(coverage["blockers"])
        )

    staging = _staging_dir(out)
    if staging.exists() or staging.is_symlink():
        raise Tau3GroundedGenerationError(f"staging directory already exists: {staging}")
    _secure_mkdir(staging)
    try:
        _externalize_state_snapshots(rows_by_split, staging)
        for split in SPLITS:
            rows_by_split[split].sort(
                key=lambda row: (
                    row["metadata"]["domain"],
                    row["metadata"]["source_family_id"],
                    row["metadata"]["parent_trajectory_id"],
                )
            )
            _write_jsonl(staging / f"{split}.jsonl", rows_by_split[split])
        manifest = _manifest(
            source_path,
            staging,
            rows_by_split,
            coverage,
            profile_id=profile_id,
            training_handoff=training_handoff,
        )
        _write_json(staging / "manifest.json", manifest)
        _publish_directory_no_replace(staging, out)
        _fsync_directory(out.parent)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def validate_tau3_grounded_generation_bundle(
    bundle_dir: str | Path,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Replay and validate a grounded generation bundle."""

    bundle = Path(bundle_dir)
    errors: list[str] = []
    if _path_has_prohibited_basename(bundle):
        return _validation_result(
            bundle,
            False,
            ["bundle path contains a prohibited basename"],
            {},
            strict,
        )
    if path_has_symlink_component(bundle, include_leaf=True):
        return _validation_result(
            bundle,
            False,
            ["bundle path must not contain symlink components"],
            {},
            strict,
        )
    manifest_path = bundle / "manifest.json"
    preflight_errors = _owner_only_path_errors(bundle, "bundle", directory=True)
    preflight_errors.extend(
        _owner_only_path_errors(manifest_path, "manifest", directory=False)
    )
    if preflight_errors:
        return _validation_result(bundle, False, preflight_errors, {}, strict)
    try:
        manifest = _read_json(manifest_path, "manifest")
    except Tau3GroundedGenerationError as exc:
        return _validation_result(bundle, False, [str(exc)], {}, strict)
    if manifest.get("schema_version") != TAU3_GROUNDED_DATASET_SCHEMA_VERSION:
        errors.append("manifest schema_version mismatch")
    if manifest.get("lineage_id") != LINEAGE_ID:
        errors.append("manifest lineage_id mismatch")
    profile_id = _coverage_profile_from_manifest(manifest, errors)
    artifact_security = _object(manifest.get("artifact_security"), "artifact_security", errors)
    if artifact_security != {
        "classification": "sensitive",
        "owner_only_required": True,
        "directory_mode": "0700",
        "file_mode": "0600_or_stricter",
        "raw_session_publishable": False,
    }:
        errors.append("artifact_security contract mismatch")
    row_contract = _object(manifest.get("row_contract"), "row_contract", errors)
    required_row_contract = {
        "schema_version": TAU3_GROUNDED_ROW_SCHEMA_VERSION,
        "state_refs_owner_only": True,
        "initial_sync_evidence_required": True,
        "ordered_initial_sync_sequence_required": True,
        "initial_sync_sequence_sha256_required": True,
        "per_call_sync_evidence_required": True,
        "generation_provenance_required": True,
        "selection_receipt_required": True,
        "task_initialization_receipt_required": True,
        "policy_target_review_required": True,
        "confirmation_detail_grounding_required": True,
    }
    if profile_id == SCALED_COVERAGE_PROFILE_ID:
        required_row_contract["scaled_selection_receipt_required"] = True
        required_row_contract["generated_family_binding_required"] = True
    if row_contract != required_row_contract:
        errors.append("row_contract mismatch")
    derivation = _object(manifest.get("derivation"), "derivation", errors)
    for field in (
        "runtime_derived_system_prompt",
        "ordered_openai_tool_schema_catalog",
        "full_environment_state_replayed",
        "chronological_argument_grounding",
        "policy_confirmation_replayed",
        "policy_mutation_review_replayed",
        "confirmation_detail_grounding_replayed",
        "task_initialization_replayed",
        "ordered_initial_sync_sequence_replayed",
        "initial_sync_sequence_sha256_bound",
        "single_ordered_tool_sequence",
        "scoped_codex_provenance",
        "registered_selection_algorithm",
        "selection_hash_semantics_replayed",
        "negative_or_unsafe_targets_masked",
    ):
        if derivation.get(field) is not True:
            errors.append(f"derivation.{field} must be true")
    if profile_id == SCALED_COVERAGE_PROFILE_ID:
        for field in (
            "deterministic_stratified_ordinal_selection",
            "full_rubric_aggregate_replayed",
            "supervised_target_token_count_dependency_free",
        ):
            if derivation.get(field) is not True:
                errors.append(f"derivation.{field} must be true")
    sealed = _object(manifest.get("sealed_access"), "sealed_access", errors)
    if sealed.get("payload_accessed") is not False or sealed.get("access_count") != 0:
        errors.append("sealed_access must prove zero payload access")
    contamination = _object(manifest.get("contamination"), "contamination", errors)
    if contamination.get("raw_sealed_payload_read") is not False:
        errors.append("contamination.raw_sealed_payload_read must be false")
    if contamination.get("split_contamination_detected") is not False:
        errors.append("contamination.split_contamination_detected must be false")

    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    files = _object(manifest.get("files"), "files", errors)
    for split in SPLITS:
        record = _object(files.get(split), f"files.{split}", errors)
        rel = str(record.get("path") or f"{split}.jsonl")
        if not _safe_relative_path(rel):
            errors.append(f"files.{split}.path must be a safe relative path")
            continue
        path = bundle / rel
        path_errors = _bundle_artifact_path_errors(
            bundle,
            path,
            f"files.{split}",
            directory=False,
        )
        if path_errors:
            errors.extend(path_errors)
            continue
        if record.get("sha256") != _sha256(path):
            errors.append(f"{split} file hash does not replay")
        errors.extend(_file_record_errors(path, record, f"files.{split}"))
        rows = _read_jsonl(path, split, errors)
        rows_by_split[split] = rows
        for index, row in enumerate(rows):
            errors.extend(_validate_row(row, f"{split}[{index}]", bundle))

    errors.extend(_split_contamination_errors(rows_by_split))
    errors.extend(_selection_profile_errors(rows_by_split, profile_id))
    if profile_id == SCALED_COVERAGE_PROFILE_ID:
        errors.extend(_scaled_selection_claim_errors(rows_by_split))
        errors.extend(_scaled_hash_disjointness_errors(rows_by_split))
        errors.extend(
            _training_handoff_errors(
                manifest.get("training_handoff"),
                "manifest.training_handoff",
            )
        )
    coverage = _coverage(
        rows_by_split,
        profile_id=profile_id,
        training_handoff=(
            manifest.get("training_handoff")
            if isinstance(manifest.get("training_handoff"), dict)
            else None
        ),
    )
    errors.extend(coverage["blockers"])
    if profile_id == SCALED_COVERAGE_PROFILE_ID:
        if manifest.get("coverage") != coverage:
            errors.append("E_SCALE_MANIFEST_COVERAGE_BINDING")
        if manifest.get("blockers") != coverage["blockers"]:
            errors.append("E_SCALE_MANIFEST_BLOCKER_BINDING")
        expected_disjointness = _split_hash_evidence(rows_by_split)
        if manifest.get("hash_disjointness") != expected_disjointness:
            errors.append("E_SCALE_MANIFEST_DISJOINTNESS_BINDING")
        expected_counts = {split: len(rows_by_split[split]) for split in SPLITS}
        if manifest.get("counts") != expected_counts:
            errors.append("E_SCALE_MANIFEST_COUNT_BINDING")
    if manifest.get("passed") is not coverage["passed"]:
        errors.append("manifest passed flag does not match replayed coverage")
    if manifest.get("status") != ("passed" if coverage["passed"] else "blocked"):
        errors.append("manifest status does not match replayed coverage")
    expected_manifest_sha = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    if manifest.get("manifest_sha256") != expected_manifest_sha:
        errors.append("manifest_sha256 does not replay")
    if strict and manifest.get("status") != "passed":
        errors.append("strict validation requires status=passed")
    return _validation_result(
        bundle,
        not errors,
        errors,
        coverage,
        strict,
        profile_id=profile_id,
    )


def write_build_validate_tau3_grounded_generation_candidates(
    *,
    candidates: Iterable[dict[str, Any]],
    source: str | Path,
    out_dir: str | Path,
    strict_coverage: bool = True,
    coverage_profile: str = LEGACY_COVERAGE_PROFILE_ID,
    training_handoff: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically publish candidates, then build and validate their bundle.

    The source and bundle are create-only.  In bounded-pilot mode
    (``strict_coverage=False``), expected aggregate coverage blockers may remain,
    but every noncoverage validation error still fails closed.
    """

    source_path = Path(source)
    out = Path(out_dir)
    _require_new_output_file(source_path, "candidate source")
    _require_new_output_dir(out)
    if source_path == out:
        raise Tau3GroundedGenerationError(
            "candidate source and output bundle must be distinct"
        )
    _require_owner_only_directory(source_path.parent, "candidate source parent")
    _require_owner_only_directory(out.parent, "output bundle parent")

    source_record = _write_new_owner_only_jsonl_atomically(source_path, candidates)
    source_sha256 = source_record["sha256"]
    manifest = build_tau3_grounded_generation_dataset(
        source=source_path,
        out_dir=out,
        strict_coverage=strict_coverage,
        coverage_profile=coverage_profile,
        training_handoff=training_handoff,
    )
    validation = validate_tau3_grounded_generation_bundle(
        out,
        strict=strict_coverage,
    )
    coverage_blockers = set(
        _dict(validation.get("coverage")).get("blockers", [])
    )
    nonwaivable_blockers = set(
        _dict(validation.get("coverage")).get("nonwaivable_blockers", [])
    )
    validation_errors = validation.get("errors")
    if not isinstance(validation_errors, list):
        validation_errors = ["validator did not return an errors list"]
    noncoverage_errors = [
        error
        for error in validation_errors
        if error not in coverage_blockers or error in nonwaivable_blockers
    ]
    contract_validated = not noncoverage_errors and (
        validation.get("passed") is True or not strict_coverage
    )
    if _sha256(source_path) != source_sha256:
        raise Tau3GroundedGenerationError(
            "candidate source changed during bundle build or validation"
        )
    if not contract_validated:
        raise Tau3GroundedGenerationError(
            "built candidate bundle failed validation: "
            + "; ".join(str(error) for error in noncoverage_errors[:8])
        )
    return {
        "source": source_record,
        "manifest": manifest,
        "validation": validation,
        "contract_validated": True,
        "coverage_passed": _dict(validation.get("coverage")).get("passed") is True,
        "noncoverage_error_count": 0,
    }


def build_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        help="training-side scenario JSONL (new target with --write-source-from-stdin)",
    )
    parser.add_argument("--out-dir", required=True, help="new output bundle directory")
    parser.add_argument(
        "--write-source-from-stdin",
        action="store_true",
        help=(
            "atomically create --source from candidate JSONL on stdin, then build "
            "and validate the bundle"
        ),
    )
    parser.add_argument(
        "--allow-incomplete-coverage",
        action="store_true",
        help="write a blocked manifest instead of failing on missing family/behavior coverage",
    )
    parser.add_argument(
        "--coverage-profile",
        choices=(LEGACY_COVERAGE_PROFILE_ID, SCALED_COVERAGE_PROFILE_ID),
        default=LEGACY_COVERAGE_PROFILE_ID,
        help="versioned aggregate validation profile",
    )
    parser.add_argument(
        "--training-handoff-json",
        help="owner-controlled JSON identity receipt required by the scaled profile",
    )
    args = parser.parse_args(argv)
    training_handoff = None
    if args.training_handoff_json:
        try:
            training_handoff = _read_json(
                Path(args.training_handoff_json),
                "training handoff",
            )
        except Tau3GroundedGenerationError as exc:
            parser.exit(1, f"error: {exc}\n")
    try:
        if args.write_source_from_stdin:
            result = write_build_validate_tau3_grounded_generation_candidates(
                candidates=_candidate_rows_from_stdin(),
                source=args.source,
                out_dir=args.out_dir,
                strict_coverage=not args.allow_incomplete_coverage,
                coverage_profile=args.coverage_profile,
                training_handoff=training_handoff,
            )
            print(
                json.dumps(
                    {
                        "contract_validated": result["contract_validated"],
                        "coverage_passed": result["coverage_passed"],
                        "out_dir": args.out_dir,
                        "source_bytes": result["source"]["bytes"],
                        "source_mode": result["source"]["mode"],
                        "source_row_count": result["source"]["row_count"],
                        "source_sha256": result["source"]["sha256"],
                    },
                    sort_keys=True,
                )
            )
            return 0
        manifest = build_tau3_grounded_generation_dataset(
            source=args.source,
            out_dir=args.out_dir,
            strict_coverage=not args.allow_incomplete_coverage,
            coverage_profile=args.coverage_profile,
            training_handoff=training_handoff,
        )
    except Tau3GroundedGenerationError as exc:
        parser.exit(1, f"error: {exc}\n")
    print(json.dumps({"out_dir": args.out_dir, "passed": manifest["passed"]}, sort_keys=True))
    return 0


def _candidate_rows_from_stdin() -> Iterable[dict[str, Any]]:
    for line_number, line in enumerate(sys.stdin, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Tau3GroundedGenerationError(
                f"stdin candidate line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise Tau3GroundedGenerationError(
                f"stdin candidate line {line_number} must be an object"
            )
        yield value


def validate_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a Tau-3 grounded generation bundle.")
    parser.add_argument("bundle_dir")
    parser.add_argument("--no-strict", action="store_true")
    args = parser.parse_args(argv)
    result = validate_tau3_grounded_generation_bundle(
        args.bundle_dir,
        strict=not args.no_strict,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


def _build_row(scenario: _Scenario) -> dict[str, Any]:
    payload = scenario.payload
    _validate_scenario(payload, scenario.index)
    runtime = _runtime_for_scenario(payload)
    tool_catalog = runtime.tool_catalog()
    if "tool_catalog" in payload and _canonical_value(payload["tool_catalog"]) != tool_catalog:
        raise Tau3GroundedGenerationError(
            "source tool_catalog does not match exact runtime-derived Tau catalog"
        )
    tool_catalog_hash = canonical_sha256(tool_catalog)
    system_prompt = str(payload["system_prompt"])
    runtime_prompt = runtime.system_prompt()
    if _runtime_is_vendored(payload["runtime_family"]):
        if not isinstance(runtime_prompt, str) or system_prompt != runtime_prompt:
            raise Tau3GroundedGenerationError(
                "source system_prompt does not match the exact Tau LLMAgent runtime prompt"
            )
        system_prompt = runtime_prompt
    parent_turns = copy.deepcopy(payload["turns"])
    decision_errors = _decision_order_errors(parent_turns, "source.turns")
    if decision_errors:
        raise Tau3GroundedGenerationError("; ".join(decision_errors))
    tool_history: list[dict[str, Any]] = []
    training_targets: list[dict[str, Any]] = []
    for turn_index, turn in enumerate(parent_turns):
        assistant = _object_required(turn.get("assistant"), f"turns[{turn_index}].assistant")
        decision_ordinal = assistant["decision_ordinal"]
        safe_target = _target_for_decision(payload, assistant, decision_ordinal, turn_index)
        if safe_target is not None:
            training_targets.append(safe_target)
        for call_ordinal, raw_call in enumerate(_list_required(assistant.get("tool_calls"), f"turns[{turn_index}].assistant.tool_calls")):
            replayed = _replay_call(
                runtime,
                raw_call,
                tool_catalog=tool_catalog,
                tool_catalog_hash=tool_catalog_hash,
                parent_turn_ordinal=turn_index,
                assistant_decision_ordinal=decision_ordinal,
                tool_call_ordinal=call_ordinal,
                prior_calls=tool_history,
            )
            replayed["argument_grounding"] = _argument_grounding_evidence(
                replayed["canonical_arguments"],
                decision_ordinal,
                parent_turns,
                tool_history,
            )
            replayed["confirmation"] = _canonical_value(raw_call.get("confirmation"))
            confirmation = replayed["confirmation"]
            if replayed["tool_mutates_state"] is True and isinstance(confirmation, dict):
                request_decision = confirmation.get("request_decision_ordinal")
                if type(request_decision) is int:
                    confirmation["detail_grounding"] = _confirmation_detail_grounding(
                        confirmation,
                        request_decision,
                        parent_turns,
                        tool_history,
                    )
                confirmation["confirmed_arguments"] = _confirmed_argument_receipt(
                    replayed["canonical_arguments"]
                )
            tool_history.append(replayed)
    _assert_training_targets_grounded(
        training_targets,
        tool_catalog,
        tool_history,
        parent_turns,
    )
    reviewer_record = _review_record(payload, "reviewer")
    policy_errors = _policy_call_errors(
        domain=str(payload["domain"]),
        turns=parent_turns,
        targets=training_targets,
        replay=tool_history,
        policy_sha256=runtime.policy_sha256(),
        reviewer_record=reviewer_record,
        candidate=_runtime_is_vendored(str(payload.get("runtime_family") or "")),
        context="source",
    )
    if policy_errors:
        raise Tau3GroundedGenerationError("; ".join(policy_errors))
    recovery_errors = _recovery_context_errors(
        training_targets,
        tool_history,
        parent_turns,
        "source",
    )
    if recovery_errors:
        raise Tau3GroundedGenerationError("; ".join(recovery_errors))
    tool_exemptions = _tool_exemptions(payload, tool_catalog)
    tau_repo = _tau_repo_record(payload)
    recipe_record = _review_record(payload, "recipe")
    selection_receipt = _selection_receipt(
        payload,
        training_targets=training_targets,
        tool_history=tool_history,
        turns=parent_turns,
        recipe=recipe_record,
    )
    generation_provenance = _generation_provenance(
        payload,
        tau_revision=str(payload["tau_revision"]),
        system_prompt_sha256=canonical_sha256(system_prompt),
        tool_catalog_sha256=tool_catalog_hash,
        source_task_sha256=str(
            selection_receipt.get("task_sha256") or scenario.row_sha256
        ),
        selection_receipt_sha256=str(selection_receipt.get("receipt_sha256") or ""),
    )
    reviewer_binding_errors = _reviewer_generation_binding_errors(
        reviewer_record,
        generation_provenance,
        candidate=_runtime_is_vendored(str(payload.get("runtime_family") or "")),
        context="source.reviewer",
    )
    if reviewer_binding_errors:
        raise Tau3GroundedGenerationError("; ".join(reviewer_binding_errors))
    metadata = {
        "schema_version": TAU3_GROUNDED_ROW_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "training_side_only": True,
        "domain": payload["domain"],
        "split": payload["split"],
        "source_family": payload["source_family"],
        "source_family_id": payload["source_family_id"],
        "source_id": payload["source_id"],
        "source_sha256": scenario.row_sha256,
        "parent_trajectory_id": payload["trajectory_id"],
        "tau_revision": payload["tau_revision"],
        "runtime_family": payload["runtime_family"],
        "tau_repo": tau_repo,
        "system_prompt_sha256": canonical_sha256(system_prompt),
        "policy_sha256": runtime.policy_sha256(),
        "tool_catalog_sha256": tool_catalog_hash,
        "initial_state_sha256": canonical_sha256(payload["initial_state"]),
        "final_state_sha256": canonical_sha256(runtime.state),
        "full_environment_state": _runtime_is_vendored(payload["runtime_family"]),
        "runtime_prompt_derived": _runtime_is_vendored(payload["runtime_family"]),
        "openai_tool_catalog_derived": _runtime_is_vendored(payload["runtime_family"]),
        "behaviors": sorted({target["behavior"] for target in training_targets}),
        "recipe": recipe_record,
        "teacher": _review_record(payload, "teacher"),
        "reviewer": reviewer_record,
        "redaction": _redaction(payload),
        "contamination": _contamination(payload),
        "tool_exemptions": tool_exemptions,
        "generation_provenance": generation_provenance,
        "selection_receipt": selection_receipt,
    }
    if selection_receipt.get("schema_version") == SCALED_SELECTION_RECEIPT_SCHEMA_VERSION:
        metadata["generated_family_id"] = selection_receipt[
            "generated_family_identifier"
        ]
    row = {
        "schema_version": TAU3_GROUNDED_ROW_SCHEMA_VERSION,
        "trajectory": {
            "trajectory_id": payload["trajectory_id"],
            "domain": payload["domain"],
            "split": payload["split"],
            "system_prompt": system_prompt,
            "turns": parent_turns,
        },
        "tool_catalog": tool_catalog,
        "initial_state": copy.deepcopy(payload["initial_state"]),
        "initial_sync": copy.deepcopy(runtime.initial_sync_evidence),
        "final_state": copy.deepcopy(runtime.state),
        "tool_replay": tool_history,
        "training_targets": training_targets,
        "metadata": metadata,
    }
    completion_errors = _completion_claim_errors(row, "source")
    if completion_errors:
        raise Tau3GroundedGenerationError("; ".join(completion_errors))
    metadata["row_sha256"] = canonical_sha256(_without_row_sha(row))
    return row


def _validate_scenario(payload: dict[str, Any], index: int) -> None:
    context = f"source[{index}]"
    for field in (
        "trajectory_id",
        "domain",
        "split",
        "source_family",
        "source_family_id",
        "source_id",
        "tau_revision",
        "runtime_family",
        "system_prompt",
    ):
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            raise Tau3GroundedGenerationError(f"{context}.{field} must be a non-empty string")
    if payload["domain"] not in DOMAINS:
        raise Tau3GroundedGenerationError(f"{context}.domain must be one of {', '.join(DOMAINS)}")
    if payload["split"] not in SPLITS:
        raise Tau3GroundedGenerationError(f"{context}.split must be train or validation; sealed/test splits are forbidden")
    if payload["source_family"] not in ALLOWED_SOURCE_FAMILIES:
        raise Tau3GroundedGenerationError(f"{context}.source_family must be official_train_derived or reviewed_synthetic")
    if not _is_replayable_runtime_family(payload["runtime_family"]):
        raise Tau3GroundedGenerationError(
            f"{context}.runtime_family must be vendored_tau_tools@<40hex> for eligibility; "
            f"{FAKE_TEST_RUNTIME_FAMILY!r} is accepted only for blocked validator-mechanics tests"
        )
    if _runtime_is_vendored(payload["runtime_family"]) and payload["runtime_family"] != f"{VENDORED_RUNTIME_PREFIX}{payload['tau_revision']}":
        raise Tau3GroundedGenerationError(f"{context}.runtime_family must bind the exact tau_revision")
    if not HEX40_RE.fullmatch(payload["tau_revision"]):
        raise Tau3GroundedGenerationError(f"{context}.tau_revision must be an immutable 40-hex revision")
    if not isinstance(payload.get("initial_state"), dict):
        raise Tau3GroundedGenerationError(f"{context}.initial_state must be an object")
    if "tool_catalog" in payload and (
        not isinstance(payload.get("tool_catalog"), list) or not payload["tool_catalog"]
    ):
        raise Tau3GroundedGenerationError(f"{context}.tool_catalog must be a non-empty list when provided")
    if not isinstance(payload.get("turns"), list) or not payload["turns"]:
        raise Tau3GroundedGenerationError(f"{context}.turns must be a non-empty list")
    contamination_errors = _contamination_errors(payload.get("contamination"), payload["split"], context)
    if contamination_errors:
        raise Tau3GroundedGenerationError("; ".join(contamination_errors))
    if _redaction(payload).get("passed") is not True:
        raise Tau3GroundedGenerationError(f"{context} redaction metadata must pass before export")


def _target_for_decision(
    scenario: dict[str, Any],
    assistant: dict[str, Any],
    decision_ordinal: int,
    turn_index: int,
) -> dict[str, Any] | None:
    target = assistant.get("safe_corrected_target")
    if target is None:
        return None
    if not isinstance(target, dict):
        raise Tau3GroundedGenerationError(f"turns[{turn_index}].assistant.safe_corrected_target must be an object")
    behavior = str(target.get("behavior") or "")
    if behavior not in BEHAVIORS:
        raise Tau3GroundedGenerationError(f"turns[{turn_index}] safe target behavior is not in the rubric")
    unsafe = target.get("unsafe") is True or target.get("mask") is True
    tool_name = target.get("tool_name")
    if unsafe:
        if target.get("reviewed") is not True:
            raise Tau3GroundedGenerationError(
                f"turns[{turn_index}] masked negative target must be explicitly reviewed"
            )
        safe_decision = target.get("safe_correction_decision_ordinal")
        if (
            type(safe_decision) is not int
            or safe_decision <= decision_ordinal
        ):
            raise Tau3GroundedGenerationError(
                f"turns[{turn_index}] masked negative target must link to a later safe correction"
            )
        kind = str(target.get("kind") or "assistant_message")
        canonical = {
            "kind": kind,
            "text": str(target.get("text") or ""),
            "tool_name": tool_name if isinstance(tool_name, str) else None,
            "arguments": _canonical_value(target.get("arguments") or {}),
        }
        _raise_target_shape_errors(
            canonical,
            f"turns[{turn_index}].assistant.safe_corrected_target",
        )
        if kind == "assistant_message" and not canonical["text"]:
            raise Tau3GroundedGenerationError(
                f"turns[{turn_index}] masked negative assistant target must carry explicit text"
            )
        return {
            "parent_turn_ordinal": turn_index,
            "parent_assistant_decision_ordinal": decision_ordinal,
            "behavior": behavior,
            "negative_behavior": str(target.get("negative_behavior") or ""),
            "masked": True,
            "mask_reason": str(target.get("mask_reason") or "unsafe_or_negative_action"),
            "reviewed": True,
            "policy_review": _canonical_value(target.get("policy_review")),
            "safe_correction_decision_ordinal": safe_decision,
            "canonical_target": canonical,
            "canonical_target_sha256": canonical_sha256(canonical),
        }
    kind = str(target.get("kind") or "assistant_message")
    arguments = _canonical_value(target.get("arguments") or {})
    canonical = {
        "kind": kind,
        "text": str(target.get("text") or ""),
        "tool_name": tool_name if isinstance(tool_name, str) else None,
        "arguments": arguments,
    }
    _raise_target_shape_errors(canonical, f"turns[{turn_index}].assistant.safe_corrected_target")
    return {
        "parent_turn_ordinal": turn_index,
        "parent_assistant_decision_ordinal": decision_ordinal,
        "behavior": behavior,
        "masked": False,
        "mask_reason": None,
        "policy_review": _canonical_value(target.get("policy_review")),
        "canonical_target": canonical,
        "canonical_target_sha256": canonical_sha256(canonical),
    }


def _assert_training_targets_grounded(
    targets: list[dict[str, Any]],
    tool_catalog: list[dict[str, Any]],
    tool_history: list[dict[str, Any]],
    turns: list[dict[str, Any]],
) -> None:
    for target in targets:
        canonical = _dict(target.get("canonical_target"))
        if target.get("masked") is not True and canonical.get("kind") == "tool_call":
            target["argument_grounding"] = _argument_grounding_evidence(
                _dict(canonical.get("arguments")),
                int(target["parent_assistant_decision_ordinal"]),
                turns,
                tool_history,
            )
    errors = [
        error
        for index, target in enumerate(targets)
        for error in _validate_target(target, f"source.training_targets[{index}]")
    ]
    errors.extend(_masked_correction_link_errors(targets, "source"))
    errors.extend(_target_binding_errors(targets, tool_catalog, tool_history, "source"))
    if errors:
        raise Tau3GroundedGenerationError("; ".join(errors))


def _raise_target_shape_errors(canonical: dict[str, Any], context: str) -> None:
    errors = _target_shape_errors(canonical, context)
    if errors:
        raise Tau3GroundedGenerationError("; ".join(errors))


def _target_shape_errors(canonical: dict[str, Any], context: str) -> list[str]:
    errors: list[str] = []
    kind = canonical.get("kind")
    tool_name = canonical.get("tool_name")
    arguments = canonical.get("arguments")
    if kind not in {"assistant_message", "tool_call"}:
        errors.append(f"{context}.kind must be assistant_message or tool_call")
    if kind == "assistant_message" and tool_name is not None:
        errors.append(f"{context}.assistant_message must not carry tool_name")
    if kind == "tool_call":
        if not isinstance(tool_name, str) or not tool_name:
            errors.append(f"{context}.tool_call must carry non-empty tool_name")
        if not isinstance(arguments, dict):
            errors.append(f"{context}.tool_call arguments must be an object")
    return errors


def _target_binding_errors(
    targets: Any,
    tool_catalog: list[dict[str, Any]],
    replay: list[dict[str, Any]],
    context: str,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(targets, list):
        return errors
    for index, target in enumerate(targets):
        if not isinstance(target, dict) or target.get("masked") is True:
            continue
        canonical = target.get("canonical_target")
        if not isinstance(canonical, dict):
            continue
        tool_name = canonical.get("tool_name")
        if tool_name is None:
            continue
        if not isinstance(tool_name, str) or not tool_name:
            errors.append(f"{context}.training_targets[{index}] tool_name must be a non-empty string")
            continue
        try:
            tool_def = _find_tool(tool_catalog, tool_name)
        except Tau3GroundedGenerationError as exc:
            errors.append(f"{context}.training_targets[{index}] target tool is absent from exact catalog: {exc}")
            continue
        args = canonical.get("arguments")
        if not isinstance(args, dict):
            errors.append(f"{context}.training_targets[{index}] target arguments must be an object")
            continue
        if not args and _required_arg_count(tool_def) > 0:
            errors.append(
                f"{context}.training_targets[{index}] empty target arguments require a zero-argument catalog tool"
            )
            continue
        decision = target.get("parent_assistant_decision_ordinal")
        bound_calls = [
            call
            for call in replay
            if (
            isinstance(call, dict)
            and call.get("tool_name") == tool_name
            and call.get("canonical_arguments") == args
            and call.get("parent_assistant_decision_ordinal") == decision
            and call.get("parent_turn_ordinal") == target.get("parent_turn_ordinal")
            and call.get("evidence_replayed") is True
            )
        ]
        if len(bound_calls) != 1:
            errors.append(
                f"{context}.training_targets[{index}] target tool call is not exactly bound to replayed evidence at the same decision"
            )
        elif target.get("argument_grounding") != bound_calls[0].get("argument_grounding"):
            errors.append(
                f"{context}.training_targets[{index}] argument grounding does not match replayed call evidence"
            )
    return errors


def _decision_order_errors(turns: Any, context: str) -> list[str]:
    if not isinstance(turns, list):
        return [f"{context} must be a list"]
    errors: list[str] = []
    for turn_ordinal, turn in enumerate(turns):
        if not isinstance(turn, dict):
            errors.append(f"{context}[{turn_ordinal}] must be an object")
            continue
        assistant = turn.get("assistant")
        if not isinstance(assistant, dict):
            errors.append(f"{context}[{turn_ordinal}].assistant must be an object")
            continue
        if assistant.get("decision_ordinal") != turn_ordinal:
            errors.append(
                f"{context}[{turn_ordinal}].assistant.decision_ordinal must equal its physical turn ordinal"
            )
    return errors


def _argument_grounding_evidence(
    arguments: dict[str, Any],
    decision_ordinal: int,
    turns: list[dict[str, Any]],
    prior_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    visible_users: list[tuple[int, str]] = []
    matching_turns = [
        turn_ordinal
        for turn_ordinal, turn in enumerate(turns)
        if isinstance(turn, dict)
        and _dict(turn.get("assistant")).get("decision_ordinal") == decision_ordinal
    ]
    if len(matching_turns) != 1:
        raise Tau3GroundedGenerationError(
            "argument grounding requires exactly one physical turn for the assistant decision"
        )
    current_turn_ordinal = matching_turns[0]
    for turn_ordinal, turn in enumerate(turns[: current_turn_ordinal + 1]):
        user = _dict(turn.get("user"))
        content = user.get("content")
        if isinstance(content, str) and content:
            visible_users.append((turn_ordinal, content))
    visible_results = [
        (index, call)
        for index, call in enumerate(prior_calls)
        if (
            isinstance(call, dict)
            and call.get("evidence_replayed") is True
            and type(call.get("parent_assistant_decision_ordinal")) is int
            and call["parent_assistant_decision_ordinal"] < decision_ordinal
        )
    ]
    for pointer, value in _scalar_leaves(arguments):
        value_hash = canonical_sha256(value)
        matched = False
        needle = _visible_scalar(value)
        if needle:
            for turn_ordinal, content in visible_users:
                if _text_exposes_scalar(content, value):
                    evidence.append(
                        {
                            "argument_pointer": pointer,
                            "value_sha256": value_hash,
                            "source": "visible_user",
                            "source_turn_ordinal": turn_ordinal,
                            "source_content_sha256": canonical_sha256(content),
                        }
                    )
                    matched = True
                    break
        if matched:
            continue
        for call_index, call in visible_results:
            result_pointer = _find_scalar_pointer(call.get("canonical_result"), value)
            if result_pointer is None:
                continue
            evidence.append(
                {
                    "argument_pointer": pointer,
                    "value_sha256": value_hash,
                    "source": "prior_tool_result",
                    "source_call_ordinal": call_index,
                    "source_decision_ordinal": call["parent_assistant_decision_ordinal"],
                    "source_result_pointer": result_pointer,
                    "source_result_sha256": str(call.get("result_sha256") or ""),
                }
            )
            matched = True
            break
        if not matched:
            raise Tau3GroundedGenerationError(
                f"tool argument {pointer or '/'} is not grounded in visible user content or a prior replayed result"
            )
    return evidence


def _scalar_leaves(value: Any, pointer: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        leaves: list[tuple[str, Any]] = []
        for key in sorted(value):
            leaves.extend(
                _scalar_leaves(
                    value[key],
                    f"{pointer}/{_escape_json_pointer(str(key))}",
                )
            )
        return leaves
    if isinstance(value, list):
        leaves = []
        for index, item in enumerate(value):
            leaves.extend(_scalar_leaves(item, f"{pointer}/{index}"))
        return leaves
    return [(pointer or "/", _canonical_value(value))]


def _visible_scalar(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return json.dumps(_canonical_value(value), sort_keys=True, separators=(",", ":"))


def _text_exposes_scalar(text: str, value: Any) -> bool:
    needle = _visible_scalar(value)
    if not needle:
        return False
    escaped = re.escape(needle)
    if needle[0].isalnum() and needle[-1].isalnum():
        return re.search(rf"(?<!\w){escaped}(?!\w)", text, re.IGNORECASE) is not None
    return needle.casefold() in text.casefold()


def _find_scalar_pointer(value: Any, expected: Any, pointer: str = "") -> str | None:
    for candidate_pointer, candidate in _scalar_leaves(value, pointer):
        if candidate == _canonical_value(expected) and type(candidate) is type(_canonical_value(expected)):
            return candidate_pointer
    return None


def _policy_call_errors(
    *,
    domain: str,
    turns: Any,
    targets: Any,
    replay: Any,
    policy_sha256: str,
    reviewer_record: Any,
    candidate: bool,
    context: str,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(turns, list) or not isinstance(targets, list) or not isinstance(replay, list):
        return errors
    target_list = [target for target in targets if isinstance(target, dict)]
    errors.extend(
        _policy_reviewer_record_errors(
            reviewer_record,
            target_list,
            policy_sha256=policy_sha256,
            candidate=candidate,
            context=f"{context}.reviewer",
        )
    )
    turns_by_decision = {
        assistant.get("decision_ordinal"): (turn_ordinal, turn)
        for turn_ordinal, turn in enumerate(turns)
        if isinstance(turn, dict)
        for assistant in [_dict(turn.get("assistant"))]
        if type(assistant.get("decision_ordinal")) is int
    }
    for index, target in enumerate(target_list):
        if target.get("masked") is not True:
            continue
        review = _dict(target.get("policy_review"))
        errors.extend(
            _policy_review_receipt_errors(
                review,
                target,
                policy_sha256=policy_sha256,
                reviewer_record=reviewer_record,
                expected_allowed=False,
                context=f"{context}.training_targets[{index}]",
            )
        )
        canonical = _dict(target.get("canonical_target"))
        if canonical.get("kind") == "tool_call":
            executed = any(
                isinstance(call, dict)
                and call.get("parent_assistant_decision_ordinal")
                == target.get("parent_assistant_decision_ordinal")
                and call.get("tool_name") == canonical.get("tool_name")
                and call.get("canonical_arguments") == canonical.get("arguments")
                for call in replay
            )
            if executed:
                errors.append(
                    f"{context}.training_targets[{index}] masked unsafe action was executed"
                )
        safe_decision = target.get("safe_correction_decision_ordinal")
        correction_targets = [
            item
            for item in target_list
            if item.get("parent_assistant_decision_ordinal") == safe_decision
            and type(item.get("parent_turn_ordinal")) is int
            and type(target.get("parent_turn_ordinal")) is int
            and item["parent_turn_ordinal"] > target["parent_turn_ordinal"]
            and item.get("masked") is not True
            and item.get("behavior") == target.get("behavior")
        ]
        if len(correction_targets) == 1:
            correction_review = _dict(correction_targets[0].get("policy_review"))
            errors.extend(
                _policy_review_receipt_errors(
                    correction_review,
                    correction_targets[0],
                    policy_sha256=policy_sha256,
                    reviewer_record=reviewer_record,
                    expected_allowed=True,
                    context=f"{context}.training_targets[{index}].safe_correction",
                )
            )
    for index, call in enumerate(replay):
        if not isinstance(call, dict):
            continue
        decision = call.get("parent_assistant_decision_ordinal")
        if type(decision) is not int:
            continue
        exact_targets = [
            target
            for target in target_list
            if target.get("parent_assistant_decision_ordinal") == decision
            and target.get("parent_turn_ordinal") == call.get("parent_turn_ordinal")
            and _dict(target.get("canonical_target")).get("kind") == "tool_call"
            and _dict(target.get("canonical_target")).get("tool_name") == call.get("tool_name")
            and _dict(target.get("canonical_target")).get("arguments")
            == call.get("canonical_arguments")
        ]
        if call.get("tool_mutates_state") is not True:
            continue
        safe_targets = [target for target in exact_targets if target.get("masked") is not True]
        if len(safe_targets) != 1:
            errors.append(
                f"{context}.tool_replay[{index}] mutation must bind exactly one unmasked reviewed target"
            )
            continue
        safe_target = safe_targets[0]
        safe_review = _dict(safe_target.get("policy_review"))
        errors.extend(
            _policy_review_receipt_errors(
                safe_review,
                safe_target,
                policy_sha256=policy_sha256,
                reviewer_record=reviewer_record,
                expected_allowed=True,
                context=f"{context}.tool_replay[{index}]",
            )
        )
        tool_name = str(call.get("tool_name") or "")
        if tool_name != "update_record" and tool_name not in REGISTERED_MUTATING_TOOLS.get(
            domain, set()
        ):
            errors.append(
                f"{context}.tool_replay[{index}] mutating tool has no registered policy rule"
            )
            continue
        rule_record = _confirmation_rule(domain, tool_name)
        if rule_record is None:
            if call.get("confirmation") not in (None, {}):
                errors.append(
                    f"{context}.tool_replay[{index}] retains confirmation for a policy that does not require it"
                )
            continue
        rule, required_detail_kind = rule_record
        confirmation = call.get("confirmation")
        if not isinstance(confirmation, dict):
            errors.append(f"{context}.tool_replay[{index}] mutation lacks confirmation evidence")
            continue
        if confirmation.get("policy_rule") != rule:
            errors.append(f"{context}.tool_replay[{index}] confirmation policy_rule mismatch")
        if confirmation.get("policy_sha256") != policy_sha256:
            errors.append(f"{context}.tool_replay[{index}] confirmation policy hash mismatch")
        if confirmation.get("arguments_sha256") != call.get("arguments_sha256"):
            errors.append(f"{context}.tool_replay[{index}] mutation arguments changed after confirmation")
        request_decision = confirmation.get("request_decision_ordinal")
        if type(request_decision) is not int or request_decision != decision - 1:
            errors.append(
                f"{context}.tool_replay[{index}] confirmation request must immediately precede mutation"
            )
            continue
        request_turn = turns_by_decision.get(request_decision)
        if request_turn is None:
            errors.append(f"{context}.tool_replay[{index}] confirmation request turn is missing")
            continue
        request_matches = [
            target
            for target in target_list
            if target.get("parent_assistant_decision_ordinal") == request_decision
            and target.get("masked") is not True
            and target.get("behavior") == "confirmation_before_mutation"
            and target.get("parent_turn_ordinal") == request_turn[0]
            and _dict(target.get("canonical_target")).get("kind") == "assistant_message"
        ]
        if len(request_matches) != 1:
            errors.append(
                f"{context}.tool_replay[{index}] confirmation request target is missing or ambiguous"
            )
            continue
        request_text = str(_dict(request_matches[0].get("canonical_target")).get("text") or "")
        action_label = confirmation.get("action_label")
        expected_action_label = str(call.get("tool_name") or "").replace("_", " ")
        if (
            not isinstance(action_label, str)
            or not action_label.strip()
            or action_label.strip().casefold() != expected_action_label.casefold()
            or action_label.casefold() not in request_text.casefold()
        ):
            errors.append(
                f"{context}.tool_replay[{index}] confirmation action label is not tool-bound"
            )
        if confirmation.get("confirmed_arguments") != _confirmed_argument_receipt(
            call.get("canonical_arguments")
        ):
            errors.append(
                f"{context}.tool_replay[{index}] confirmed argument pointer receipt does not replay"
            )
        for pointer, value in _scalar_leaves(call.get("canonical_arguments")):
            if not _text_exposes_labeled_scalar(
                request_text,
                _argument_pointer_label(pointer),
                value,
            ):
                errors.append(
                    f"{context}.tool_replay[{index}] confirmation omits labeled argument {pointer}"
                )
        details = confirmation.get("required_details")
        if not isinstance(details, list):
            errors.append(f"{context}.tool_replay[{index}] confirmation required_details must be a list")
            details = []
        financial_kinds: list[str] = []
        for detail_index, detail in enumerate(details):
            if not isinstance(detail, dict):
                errors.append(
                    f"{context}.tool_replay[{index}].confirmation.required_details[{detail_index}] must be an object"
                )
                continue
            kind = detail.get("kind")
            value = detail.get("value")
            if kind in {"price", "total", "price_difference", "refund"}:
                financial_kinds.append(str(kind))
            if (
                not isinstance(kind, str)
                or not kind
                or not _text_exposes_labeled_scalar(request_text, kind, value)
            ):
                errors.append(
                    f"{context}.tool_replay[{index}].confirmation.required_details[{detail_index}] is not exposed in the request"
                )
        if required_detail_kind is not None and financial_kinds.count(required_detail_kind) != 1:
            errors.append(
                f"{context}.tool_replay[{index}] policy requires retained {required_detail_kind} detail"
            )
        if any(kind != required_detail_kind for kind in financial_kinds):
            errors.append(
                f"{context}.tool_replay[{index}] retained financial detail kind is not policy-specific"
            )
        try:
            expected_detail_grounding = _confirmation_detail_grounding(
                confirmation,
                request_decision,
                turns,
                replay,
            )
        except Tau3GroundedGenerationError as exc:
            errors.append(
                f"{context}.tool_replay[{index}] confirmation detail is ungrounded: {exc}"
            )
        else:
            if confirmation.get("detail_grounding") != expected_detail_grounding:
                errors.append(
                    f"{context}.tool_replay[{index}] confirmation detail grounding does not replay"
                )
        mutation_turn = turns_by_decision.get(decision)
        if mutation_turn is None:
            errors.append(f"{context}.tool_replay[{index}] mutation turn is missing")
            continue
        if (
            mutation_turn[0] != call.get("parent_turn_ordinal")
            or request_turn[0] != mutation_turn[0] - 1
        ):
            errors.append(
                f"{context}.tool_replay[{index}] confirmation request is not physically adjacent to mutation"
            )
        reply = str(_dict(mutation_turn[1].get("user")).get("content") or "").strip()
        if AFFIRMATIVE_REPLY_RE.fullmatch(reply) is None:
            errors.append(
                f"{context}.tool_replay[{index}] mutation lacks an explicit affirmative user reply"
            )
    return errors


def _policy_review_receipt_errors(
    review: dict[str, Any],
    target: dict[str, Any],
    *,
    policy_sha256: str,
    reviewer_record: Any,
    expected_allowed: bool,
    context: str,
) -> list[str]:
    errors: list[str] = []
    reviewer = _dict(reviewer_record)
    expected_keys = {
        "schema_version",
        "reviewer",
        "reviewer_artifact_sha256",
        "policy_sha256",
        "canonical_target_sha256",
        "parent_turn_ordinal",
        "parent_assistant_decision_ordinal",
        "allowed",
        "reason_id",
        "review_receipt_sha256",
    }
    if set(review) != expected_keys:
        errors.append(f"{context}.policy_review fields do not match the receipt contract")
    if review.get("schema_version") != "hfr.tau3_policy_review.v1":
        errors.append(f"{context}.policy_review.schema_version mismatch")
    reviewer_id = reviewer.get("id")
    reviewer_sha256 = reviewer.get("sha256")
    if not isinstance(reviewer_id, str) or not reviewer_id:
        errors.append(f"{context}.reviewer artifact id is missing")
    if not SHA256_RE.fullmatch(str(reviewer_sha256 or "")):
        errors.append(f"{context}.reviewer artifact hash is missing")
    if review.get("reviewer") != reviewer_id:
        errors.append(f"{context}.policy_review reviewer does not bind metadata reviewer")
    if review.get("reviewer_artifact_sha256") != reviewer_sha256:
        errors.append(f"{context}.policy_review reviewer artifact hash does not bind")
    if review.get("policy_sha256") != policy_sha256:
        errors.append(f"{context}.policy_review policy hash is not runtime-derived")
    if review.get("canonical_target_sha256") != target.get("canonical_target_sha256"):
        errors.append(f"{context}.policy_review target hash does not bind")
    for field in ("parent_turn_ordinal", "parent_assistant_decision_ordinal"):
        if review.get(field) != target.get(field):
            errors.append(f"{context}.policy_review {field} does not bind")
    if review.get("allowed") is not expected_allowed:
        errors.append(f"{context}.policy_review allowed verdict mismatch")
    if not isinstance(review.get("reason_id"), str) or not review["reason_id"]:
        errors.append(f"{context}.policy_review reason_id is missing")
    expected_receipt_sha256 = canonical_sha256(
        {key: item for key, item in review.items() if key != "review_receipt_sha256"}
    )
    if review.get("review_receipt_sha256") != expected_receipt_sha256:
        errors.append(f"{context}.policy_review receipt hash does not replay")
    return errors


def _policy_reviewer_record_errors(
    value: Any,
    targets: list[dict[str, Any]],
    *,
    policy_sha256: str,
    candidate: bool,
    context: str,
) -> list[str]:
    if not isinstance(value, dict):
        return [f"{context} must be an object"]
    errors: list[str] = []
    expected_keys = {"id", "sha256", "artifact", "review_set_sha256"}
    if set(value) != expected_keys:
        errors.append(f"{context} fields do not match the reviewer contract")
    reviewer_id = value.get("id")
    if not isinstance(reviewer_id, str) or not reviewer_id:
        errors.append(f"{context}.id must be a non-empty string")
    artifact = value.get("artifact")
    if not isinstance(artifact, dict):
        errors.append(f"{context}.artifact must be an object")
        artifact = {}
    artifact_keys = {
        "schema_version",
        "inference_receipt",
        "reviewer_inference_receipt_sha256",
        "generator_inference_receipt_sha256",
        "policy_sha256",
        "independent_review_pass",
    }
    if set(artifact) != artifact_keys:
        errors.append(f"{context}.artifact fields do not match the reviewer contract")
    if artifact.get("schema_version") != "hfr.tau3_policy_reviewer_artifact.v1":
        errors.append(f"{context}.artifact.schema_version mismatch")
    inference_receipt = artifact.get("inference_receipt")
    if not isinstance(inference_receipt, dict):
        errors.append(f"{context}.artifact.inference_receipt must be an object")
        inference_receipt = {}
    inference_keys = {
        "schema_version",
        "inference_origin",
        "model",
        "reasoning_effort",
        "native_codex_inference_calls",
        "provider_accessed",
        "network_accessed",
        "prohibited_external_model_provider_calls",
        "prohibited_external_network_calls",
    }
    if set(inference_receipt) != inference_keys:
        errors.append(
            f"{context}.artifact.inference_receipt fields do not match the contract"
        )
    if (
        inference_receipt.get("schema_version")
        != "hfr.tau3_policy_review_inference_receipt.v1"
    ):
        errors.append(f"{context}.artifact.inference_receipt.schema_version mismatch")
    reviewer_inference_sha256 = canonical_sha256(inference_receipt)
    if artifact.get("reviewer_inference_receipt_sha256") != reviewer_inference_sha256:
        errors.append(f"{context}.artifact reviewer inference receipt does not replay")
    if value.get("sha256") != canonical_sha256(artifact):
        errors.append(f"{context}.sha256 does not replay reviewer artifact")
    if artifact.get("policy_sha256") != policy_sha256:
        errors.append(f"{context}.artifact policy hash is not runtime-derived")
    if artifact.get("independent_review_pass") is not True:
        errors.append(f"{context}.artifact must attest an independent review pass")
    generator_inference_sha256 = str(
        artifact.get("generator_inference_receipt_sha256") or ""
    )
    if not SHA256_RE.fullmatch(generator_inference_sha256):
        errors.append(f"{context}.artifact generator inference receipt must be sha256")
    if generator_inference_sha256 == reviewer_inference_sha256:
        errors.append(f"{context}.artifact reviewer and generator receipts must differ")
    review_set = [
        copy.deepcopy(target.get("policy_review"))
        for target in targets
        if isinstance(target.get("policy_review"), dict)
    ]
    if value.get("review_set_sha256") != canonical_sha256(review_set):
        errors.append(f"{context}.review_set_sha256 does not replay target reviews")
    if candidate:
        if inference_receipt.get("inference_origin") != NATIVE_CODEX_ORIGIN:
            errors.append(f"{context}.artifact reviewer origin must be native_codex")
        if inference_receipt.get("model") != "gpt-5.6-sol":
            errors.append(f"{context}.artifact reviewer model must be gpt-5.6-sol")
        if inference_receipt.get("reasoning_effort") not in {"xhigh", "ultra"}:
            errors.append(f"{context}.artifact reviewer effort must be xhigh or ultra")
        native_calls = inference_receipt.get("native_codex_inference_calls")
        if type(native_calls) is not int or native_calls < 1:
            errors.append(f"{context}.artifact reviewer native calls must be positive")
        if inference_receipt.get("provider_accessed") is not True:
            errors.append(f"{context}.artifact reviewer provider_accessed must be true")
        if inference_receipt.get("network_accessed") is not True:
            errors.append(f"{context}.artifact reviewer network_accessed must be true")
    for field in (
        "prohibited_external_model_provider_calls",
        "prohibited_external_network_calls",
    ):
        if inference_receipt.get(field) != 0:
            errors.append(f"{context}.artifact.inference_receipt.{field} must be zero")
    return errors


def _reviewer_generation_binding_errors(
    reviewer_record: Any,
    generation_provenance: Any,
    *,
    candidate: bool,
    context: str,
) -> list[str]:
    if not candidate:
        return []
    reviewer = _dict(reviewer_record)
    artifact = _dict(reviewer.get("artifact"))
    provenance = _dict(generation_provenance)
    if (
        artifact.get("generator_inference_receipt_sha256")
        != provenance.get("inference_receipt_sha256")
    ):
        return [f"{context} does not bind the generator inference receipt"]
    return []


def _confirmation_rule(domain: str, tool_name: str) -> tuple[str, str | None] | None:
    if tool_name == "update_record":
        return "test.update_record.explicit_confirmation", None
    if tool_name not in CONFIRMATION_REQUIRED_TOOLS.get(domain, set()):
        return None
    detail_kind = REQUIRED_CONFIRMATION_DETAIL_KIND.get((domain, tool_name))
    suffix = "financial_confirmation" if detail_kind is not None else "detail_confirmation"
    return f"{domain}.{tool_name}.{suffix}", detail_kind


def _confirmation_detail_grounding(
    confirmation: dict[str, Any],
    request_decision_ordinal: int,
    turns: list[dict[str, Any]],
    prior_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    details = confirmation.get("required_details")
    if not isinstance(details, list):
        return []
    values = {
        str(index): detail["value"]
        for index, detail in enumerate(details)
        if isinstance(detail, dict) and "value" in detail
    }
    if not values:
        return []
    return _argument_grounding_evidence(
        values,
        request_decision_ordinal,
        turns,
        prior_calls,
    )


def _confirmed_argument_receipt(arguments: Any) -> list[dict[str, Any]]:
    return [
        {
            "argument_pointer": pointer,
            "argument_label": _argument_pointer_label(pointer),
            "value_sha256": canonical_sha256(value),
        }
        for pointer, value in _scalar_leaves(arguments)
    ]


def _argument_pointer_label(pointer: str) -> str:
    parts = [
        part.replace("~1", "/").replace("~0", "~")
        for part in pointer.split("/")
        if part
    ]
    label = next((part for part in reversed(parts) if not part.isdigit()), "value")
    return re.sub(r"[_-]+", " ", label).strip()


def _text_exposes_labeled_scalar(text: str, label: str, value: Any) -> bool:
    normalized_label = re.sub(r"[_-]+", " ", label).strip()
    needle = _visible_scalar(value)
    if not normalized_label or not needle:
        return False
    for label_match in re.finditer(
        rf"(?<!\w){re.escape(normalized_label)}(?!\w)",
        text,
        re.IGNORECASE,
    ):
        bounded_suffix = text[label_match.end() : label_match.end() + 192]
        if _text_exposes_scalar(bounded_suffix, value):
            return True
    return False


def _generation_provenance(
    payload: dict[str, Any],
    *,
    tau_revision: str,
    system_prompt_sha256: str,
    tool_catalog_sha256: str,
    source_task_sha256: str,
    selection_receipt_sha256: str,
) -> dict[str, Any]:
    candidate = _runtime_is_vendored(str(payload.get("runtime_family") or ""))
    selection = _dict(payload.get("selection_receipt"))
    scaled_generation = (
        selection.get("schema_version") == SCALED_SELECTION_RECEIPT_SCHEMA_VERSION
    )
    value = payload.get("generation_provenance")
    if value is None and not candidate:
        inference_receipt = {
            "schema_version": "hfr.tau3_inference_receipt.v1",
            "inference_origin": "synthetic_test",
            "generator_model": "none",
            "reasoning_effort": "none",
            "native_codex_inference_calls": 0,
            "provider_accessed": False,
            "network_accessed": False,
            "prohibited_external_model_provider_calls": 0,
            "prohibited_external_network_calls": 0,
            "tau_revision": tau_revision,
            "system_prompt_sha256": system_prompt_sha256,
            "tool_catalog_sha256": tool_catalog_sha256,
        }
        inference_receipt_sha256 = canonical_sha256(inference_receipt)
        controller_receipt = {
            "schema_version": "hfr.tau3_generation_controller_receipt.v1",
            "controller_origin": "synthetic_test",
            "controller_model": "none",
            "controller_reasoning_effort": "none",
            "generator_inference_receipt_sha256": inference_receipt_sha256,
            "selection_receipt_sha256": selection_receipt_sha256,
            "training_started": False,
            "scaled_generation_started": False,
            "test_split_payload_accessed": False,
            "prohibited_external_model_provider_calls": 0,
            "prohibited_external_network_calls": 0,
            "source_task_sha256": source_task_sha256,
        }
        value = {
            "schema_version": "hfr.tau3_generation_provenance.v2",
            "inference_origin": "synthetic_test",
            "generator_model": "none",
            "reasoning_effort": "none",
            "native_codex_inference_used": False,
            "native_codex_inference_calls": 0,
            "provider_accessed": False,
            "network_accessed": False,
            "prohibited_external_model_provider_calls": 0,
            "prohibited_external_network_calls": 0,
            "inference_receipt": inference_receipt,
            "controller_receipt": controller_receipt,
            "inference_receipt_sha256": inference_receipt_sha256,
            "controller_receipt_sha256": canonical_sha256(controller_receipt),
            "generated_artifacts_redacted": True,
            "raw_session_publishable": False,
            "raw_session_owner_only": True,
            "raw_session_disposition": "not_retained",
        }
    errors = _generation_provenance_errors(
        value,
        candidate,
        "source.generation_provenance",
        scaled_generation=scaled_generation,
    )
    if errors:
        raise Tau3GroundedGenerationError("; ".join(errors))
    binding_errors = _generation_provenance_binding_errors(
        value,
        tau_revision=tau_revision,
        system_prompt_sha256=system_prompt_sha256,
        tool_catalog_sha256=tool_catalog_sha256,
        source_task_sha256=source_task_sha256,
        selection_receipt_sha256=selection_receipt_sha256,
        context="source.generation_provenance",
        scaled_selection_receipt=selection if scaled_generation else None,
    )
    if binding_errors:
        raise Tau3GroundedGenerationError("; ".join(binding_errors))
    return copy.deepcopy(value)


def _generation_provenance_errors(
    value: Any,
    candidate: bool,
    context: str,
    *,
    scaled_generation: bool | None = None,
) -> list[str]:
    if not isinstance(value, dict):
        return [f"{context} must be an object"]
    errors: list[str] = []
    expected_provenance_keys = {
        "schema_version",
        "inference_origin",
        "generator_model",
        "reasoning_effort",
        "native_codex_inference_used",
        "native_codex_inference_calls",
        "provider_accessed",
        "network_accessed",
        "prohibited_external_model_provider_calls",
        "prohibited_external_network_calls",
        "inference_receipt",
        "controller_receipt",
        "inference_receipt_sha256",
        "controller_receipt_sha256",
        "generated_artifacts_redacted",
        "raw_session_publishable",
        "raw_session_owner_only",
        "raw_session_disposition",
    }
    if set(value) != expected_provenance_keys:
        errors.append(f"{context} fields do not match the provenance contract")
    if value.get("schema_version") != "hfr.tau3_generation_provenance.v2":
        errors.append(f"{context}.schema_version mismatch")
    origin = value.get("inference_origin")
    model = value.get("generator_model")
    codex_generated = origin == NATIVE_CODEX_ORIGIN or (
        isinstance(model, str) and model.startswith("gpt-")
    )
    if candidate and origin != NATIVE_CODEX_ORIGIN:
        errors.append(f"{context}.inference_origin must disclose native_codex")
    if candidate and value.get("native_codex_inference_used") is not True:
        errors.append(f"{context}.native_codex_inference_used must be true")
    native_calls = value.get("native_codex_inference_calls")
    if candidate and (type(native_calls) is not int or native_calls < 1):
        errors.append(f"{context}.native_codex_inference_calls must be positive")
    if not candidate and type(native_calls) is not int:
        errors.append(f"{context}.native_codex_inference_calls must be an integer")
    if codex_generated and value.get("provider_accessed") is not True:
        errors.append(f"{context}.provider_accessed contradicts Codex generation")
    if codex_generated and value.get("network_accessed") is not True:
        errors.append(f"{context}.network_accessed contradicts Codex generation")
    for legacy_field in ("external_model_provider_calls", "external_network_calls"):
        if legacy_field in value:
            errors.append(f"{context}.{legacy_field} is ambiguous and prohibited")
    for field in (
        "prohibited_external_model_provider_calls",
        "prohibited_external_network_calls",
    ):
        if value.get(field) != 0:
            errors.append(f"{context}.{field} must prove zero prohibited external calls")
    for field in ("inference_receipt_sha256", "controller_receipt_sha256"):
        if not SHA256_RE.fullmatch(str(value.get(field) or "")):
            errors.append(f"{context}.{field} must be sha256")
    inference_receipt = value.get("inference_receipt")
    controller_receipt = value.get("controller_receipt")
    if not isinstance(inference_receipt, dict):
        errors.append(f"{context}.inference_receipt must be an object")
        inference_receipt = {}
    if not isinstance(controller_receipt, dict):
        errors.append(f"{context}.controller_receipt must be an object")
        controller_receipt = {}
    expected_inference_keys = {
        "schema_version",
        "inference_origin",
        "generator_model",
        "reasoning_effort",
        "native_codex_inference_calls",
        "provider_accessed",
        "network_accessed",
        "prohibited_external_model_provider_calls",
        "prohibited_external_network_calls",
        "tau_revision",
        "system_prompt_sha256",
        "tool_catalog_sha256",
    }
    legacy_controller_keys = {
        "schema_version",
        "controller_origin",
        "controller_model",
        "controller_reasoning_effort",
        "generator_inference_receipt_sha256",
        "selection_receipt_sha256",
        "training_started",
        "scaled_generation_started",
        "test_split_payload_accessed",
        "prohibited_external_model_provider_calls",
        "prohibited_external_network_calls",
        "source_task_sha256",
    }
    scaled_controller_keys = legacy_controller_keys | {
        "coverage_profile_id",
        "generated_family_identifier",
        "generation_variant_ordinal",
    }
    if set(inference_receipt) != expected_inference_keys:
        errors.append(f"{context}.inference_receipt fields do not match the contract")
    controller_version = controller_receipt.get("schema_version")
    if scaled_generation is None:
        scaled_generation = (
            controller_version == "hfr.tau3_generation_controller_receipt.v2"
        )
    expected_controller_keys = (
        scaled_controller_keys if scaled_generation else legacy_controller_keys
    )
    if set(controller_receipt) != expected_controller_keys:
        errors.append(f"{context}.controller_receipt fields do not match the contract")
    if value.get("inference_receipt_sha256") != canonical_sha256(inference_receipt):
        errors.append(f"{context}.inference_receipt_sha256 does not replay")
    if value.get("controller_receipt_sha256") != canonical_sha256(controller_receipt):
        errors.append(f"{context}.controller_receipt_sha256 does not replay")
    if inference_receipt.get("schema_version") != "hfr.tau3_inference_receipt.v1":
        errors.append(f"{context}.inference_receipt.schema_version mismatch")
    for field in (
        "inference_origin",
        "generator_model",
        "reasoning_effort",
        "native_codex_inference_calls",
        "provider_accessed",
        "network_accessed",
        "prohibited_external_model_provider_calls",
        "prohibited_external_network_calls",
    ):
        if inference_receipt.get(field) != value.get(field):
            errors.append(f"{context}.inference_receipt.{field} does not bind provenance")
    expected_controller_version = (
        "hfr.tau3_generation_controller_receipt.v2"
        if scaled_generation
        else "hfr.tau3_generation_controller_receipt.v1"
    )
    if controller_receipt.get("schema_version") != expected_controller_version:
        errors.append(f"{context}.controller_receipt.schema_version mismatch")
    for field, top_field in (
        ("controller_origin", "inference_origin"),
        ("controller_model", "generator_model"),
        ("controller_reasoning_effort", "reasoning_effort"),
        ("generator_inference_receipt_sha256", "inference_receipt_sha256"),
    ):
        if controller_receipt.get(field) != value.get(top_field):
            errors.append(f"{context}.controller_receipt.{field} does not bind provenance")
    for field in ("training_started", "test_split_payload_accessed"):
        if controller_receipt.get(field) is not False:
            errors.append(f"{context}.controller_receipt.{field} must be false")
    if controller_receipt.get("scaled_generation_started") is not scaled_generation:
        errors.append(
            f"{context}.controller_receipt.scaled_generation_started does not bind selection profile"
        )
    if scaled_generation:
        if controller_receipt.get("coverage_profile_id") != SCALED_COVERAGE_PROFILE_ID:
            errors.append(f"{context}.controller_receipt.coverage_profile_id mismatch")
        if not SHA256_RE.fullmatch(
            str(controller_receipt.get("generated_family_identifier") or "")
        ):
            errors.append(
                f"{context}.controller_receipt.generated_family_identifier must be sha256"
            )
        variant = controller_receipt.get("generation_variant_ordinal")
        if type(variant) is not int or variant < 0:
            errors.append(
                f"{context}.controller_receipt.generation_variant_ordinal must be non-negative"
            )
    for field in (
        "prohibited_external_model_provider_calls",
        "prohibited_external_network_calls",
    ):
        if controller_receipt.get(field) != value.get(field):
            errors.append(f"{context}.controller_receipt.{field} does not bind provenance")
    if value.get("generated_artifacts_redacted") is not True:
        errors.append(f"{context}.generated_artifacts_redacted must be true")
    if value.get("raw_session_publishable") is not False:
        errors.append(f"{context}.raw_session_publishable must be false")
    if value.get("raw_session_owner_only") is not True:
        errors.append(f"{context}.raw_session_owner_only must be true")
    if value.get("raw_session_disposition") not in {"not_retained", "retained_owner_only"}:
        errors.append(f"{context}.raw_session_disposition is invalid")
    if not isinstance(model, str) or not model:
        errors.append(f"{context}.generator_model must be a non-empty string")
    if not isinstance(value.get("reasoning_effort"), str) or not value["reasoning_effort"]:
        errors.append(f"{context}.reasoning_effort must be a non-empty string")
    if candidate and model != "gpt-5.6-sol":
        errors.append(f"{context}.generator_model must be gpt-5.6-sol")
    if candidate and value.get("reasoning_effort") != "xhigh":
        errors.append(f"{context}.reasoning_effort must be xhigh")
    return errors


def _generation_provenance_binding_errors(
    value: Any,
    *,
    tau_revision: str,
    system_prompt_sha256: str,
    tool_catalog_sha256: str,
    source_task_sha256: str,
    selection_receipt_sha256: str,
    context: str,
    scaled_selection_receipt: dict[str, Any] | None = None,
) -> list[str]:
    if not isinstance(value, dict):
        return [f"{context} must be an object"]
    inference_receipt = _dict(value.get("inference_receipt"))
    controller_receipt = _dict(value.get("controller_receipt"))
    errors: list[str] = []
    for field, expected in (
        ("tau_revision", tau_revision),
        ("system_prompt_sha256", system_prompt_sha256),
        ("tool_catalog_sha256", tool_catalog_sha256),
    ):
        if inference_receipt.get(field) != expected:
            errors.append(f"{context}.inference_receipt.{field} does not bind row evidence")
    if controller_receipt.get("source_task_sha256") != source_task_sha256:
        errors.append(
            f"{context}.controller_receipt.source_task_sha256 does not bind selection"
        )
    if controller_receipt.get("selection_receipt_sha256") != selection_receipt_sha256:
        errors.append(
            f"{context}.controller_receipt.selection_receipt_sha256 does not bind selection"
        )
    if scaled_selection_receipt is not None:
        if controller_receipt.get("coverage_profile_id") != SCALED_COVERAGE_PROFILE_ID:
            errors.append(
                f"{context}.controller_receipt.coverage_profile_id does not bind scaled profile"
            )
        if controller_receipt.get(
            "generated_family_identifier"
        ) != scaled_selection_receipt.get("generated_family_identifier"):
            errors.append(
                f"{context}.controller_receipt.generated_family_identifier does not bind selection"
            )
        if controller_receipt.get(
            "generation_variant_ordinal"
        ) != scaled_selection_receipt.get("generation_variant_ordinal"):
            errors.append(
                f"{context}.controller_receipt.generation_variant_ordinal does not bind selection"
            )
    return errors


def _selection_receipt(
    payload: dict[str, Any],
    *,
    training_targets: list[dict[str, Any]] | None = None,
    tool_history: list[dict[str, Any]] | None = None,
    turns: list[dict[str, Any]] | None = None,
    recipe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidate = _runtime_is_vendored(str(payload.get("runtime_family") or ""))
    family_id = str(payload.get("source_family_id") or "")
    value = payload.get("selection_receipt")
    if value is None and not candidate:
        value = {
            "schema_version": "hfr.tau3_selection_receipt.v1",
            "algorithm": "synthetic_test_fixture",
            "family_identifier": family_id,
            "family_identifier_semantics": "opaque_test_identifier",
            "selected_family_sha256": canonical_sha256(family_id),
        }
        value["receipt_sha256"] = canonical_sha256(value)
    errors = _selection_receipt_errors(
        value,
        family_id,
        candidate,
        "source.selection_receipt",
        domain=str(payload.get("domain") or ""),
        split=str(payload.get("split") or ""),
        tau_revision=str(payload.get("tau_revision") or ""),
        expected_generation_stratum=(
            _generation_stratum_definition(
                source_provenance=str(payload.get("source_family") or ""),
                targets=training_targets or [],
                tool_history=tool_history or [],
                turns=turns or [],
            )
            if isinstance(value, dict)
            and value.get("schema_version") == SCALED_SELECTION_RECEIPT_SCHEMA_VERSION
            else None
        ),
        expected_recipe=recipe,
    )
    if errors:
        raise Tau3GroundedGenerationError("; ".join(errors))
    if candidate and value["task_initial_state_sha256"] != canonical_sha256(
        _dict(payload.get("initial_state")).get("task_initialization")
    ):
        raise Tau3GroundedGenerationError(
            "source.selection_receipt.task_initial_state_sha256 does not bind task initialization"
        )
    return copy.deepcopy(value)


def _selection_receipt_errors(
    value: Any,
    family_id: str,
    candidate: bool,
    context: str,
    *,
    domain: str | None = None,
    split: str | None = None,
    tau_revision: str | None = None,
    expected_generation_stratum: dict[str, Any] | None = None,
    expected_recipe: dict[str, Any] | None = None,
) -> list[str]:
    if not isinstance(value, dict):
        return [f"{context} must be an object"]
    if candidate and value.get("schema_version") == SCALED_SELECTION_RECEIPT_SCHEMA_VERSION:
        return _scaled_selection_receipt_errors(
            value,
            family_id,
            context,
            domain=str(domain or ""),
            split=str(split or ""),
            tau_revision=str(tau_revision or ""),
            expected_generation_stratum=expected_generation_stratum,
            expected_recipe=expected_recipe,
        )
    errors: list[str] = []
    if value.get("schema_version") != "hfr.tau3_selection_receipt.v1":
        errors.append(f"{context}.schema_version mismatch")
    if not isinstance(value.get("algorithm"), str) or not value["algorithm"]:
        errors.append(f"{context}.algorithm must be a non-empty string")
    if value.get("family_identifier") != family_id:
        errors.append(f"{context}.family_identifier does not bind source_family_id")
    semantics = value.get("family_identifier_semantics")
    selected_digest = value.get("selected_family_sha256")
    if candidate:
        if value.get("algorithm") != CONTENT_ADDRESSED_SELECTION_ALGORITHM:
            errors.append(f"{context}.algorithm is not the registered campaign algorithm")
        if not SHA256_RE.fullmatch(family_id):
            errors.append(f"{context}.family_identifier must already be a content address")
        if semantics != CONTENT_ADDRESSED_FAMILY_SEMANTICS:
            errors.append(f"{context}.family_identifier_semantics mismatch")
        if selected_digest != family_id:
            errors.append(
                f"{context}.selected_family_sha256 must equal the content-addressed identifier without rehashing"
            )
        if tau_revision is not None and value.get("source_revision") != tau_revision:
            errors.append(f"{context}.source_revision does not bind row tau_revision")
        salt_sha256 = str(value.get("selection_salt_sha256") or "")
        task_sha256 = str(value.get("task_sha256") or "")
        task_initial_state_sha256 = str(value.get("task_initial_state_sha256") or "")
        for field, digest in (
            ("selection_salt_sha256", salt_sha256),
            ("task_sha256", task_sha256),
            ("task_initial_state_sha256", task_initial_state_sha256),
        ):
            if not SHA256_RE.fullmatch(digest):
                errors.append(f"{context}.{field} must be sha256")
        errors.extend(
            _selection_source_evidence_errors(
                value,
                family_id,
                context,
                domain=str(domain or ""),
                split=str(split or ""),
            )
        )
    elif semantics != "opaque_test_identifier" or selected_digest != canonical_sha256(family_id):
        errors.append(f"{context} synthetic identifier semantics mismatch")
    expected_receipt_sha = canonical_sha256(
        {key: item for key, item in value.items() if key != "receipt_sha256"}
    )
    if value.get("receipt_sha256") != expected_receipt_sha:
        errors.append(f"{context}.receipt_sha256 does not replay")
    return errors


def _generation_stratum_definition(
    *,
    source_provenance: str,
    targets: list[Any],
    tool_history: list[Any],
    turns: list[Any],
) -> dict[str, Any]:
    behaviors: set[str] = set()
    target_tools: set[str] = set()
    target_action_kinds: set[str] = set()
    action_classes: set[str] = set()
    for target in targets:
        if not isinstance(target, dict):
            continue
        behavior = target.get("behavior")
        if isinstance(behavior, str) and behavior in BEHAVIORS:
            behaviors.add(behavior)
        canonical = _dict(target.get("canonical_target"))
        kind = canonical.get("kind")
        if kind in {"assistant_message", "tool_call"}:
            target_action_kinds.add(str(kind))
        tool_name = canonical.get("tool_name")
        if isinstance(tool_name, str) and tool_name:
            target_tools.add(tool_name)
            action_classes.add(
                "mutation" if _is_mutation_tool(tool_name) else "lookup"
            )
        elif kind == "assistant_message":
            action_classes.add("communication")
    result_contexts: set[str] = set()
    for call in tool_history:
        if not isinstance(call, dict):
            continue
        result_class = call.get("result_class")
        if isinstance(result_class, str) and result_class:
            result_contexts.add(result_class)
        if _dict(call.get("context")).get("repeated_call") is True:
            result_contexts.add("repeated")
    turn_count = len(turns)
    if turn_count <= 1:
        sequence_bucket = "one"
    elif turn_count <= 4:
        sequence_bucket = "two_to_four"
    elif turn_count <= 8:
        sequence_bucket = "five_to_eight"
    else:
        sequence_bucket = "nine_or_more"
    return {
        "schema_version": GENERATION_STRATUM_SCHEMA_VERSION,
        "behaviors": sorted(behaviors),
        "target_tools": sorted(target_tools),
        "target_action_kinds": sorted(target_action_kinds),
        "action_classes": sorted(action_classes),
        "result_context_classes": sorted(result_contexts),
        "sequence_length_bucket": sequence_bucket,
        "source_provenance": source_provenance,
    }


def _scaled_selection_receipt_errors(
    value: dict[str, Any],
    family_id: str,
    context: str,
    *,
    domain: str,
    split: str,
    tau_revision: str,
    expected_generation_stratum: dict[str, Any] | None,
    expected_recipe: dict[str, Any] | None,
) -> list[str]:
    errors: list[str] = []
    expected_keys = {
        "schema_version",
        "algorithm",
        "rank_tie_breaker_contract",
        "family_identifier",
        "family_identifier_semantics",
        "source",
        "mapped_grounded_split",
        "source_file_path",
        "source_file_sha256",
        "source_line_number",
        "canonical_source_row_sha256",
        "task_id_sha256",
        "prompt_sha256",
        "source_family_sha256",
        "source_revision",
        "campaign_salt",
        "campaign_salt_sha256",
        "selection_rank_sha256",
        "task_sha256",
        "task_initial_state_sha256",
        "selection_stratum_definition",
        "selection_stratum_sha256",
        "rank_ordinal",
        "generation_variant_ordinal",
        "generation_stratum_definition",
        "generation_stratum_sha256",
        "generation_recipe",
        "generation_recipe_sha256",
        "generated_family_identifier",
        "selected_family_sha256",
        "receipt_sha256",
    }
    if set(value) != expected_keys:
        errors.append(f"{context} fields do not match the scaled selection contract")
    if value.get("algorithm") != SCALED_SELECTION_ALGORITHM:
        errors.append(f"{context}.algorithm is not the scaled registered algorithm")
    if value.get("rank_tie_breaker_contract") != SCALED_RANK_TIE_BREAKER_CONTRACT:
        errors.append(f"{context}.rank_tie_breaker_contract mismatch")
    if value.get("family_identifier") != family_id:
        errors.append(f"{context}.family_identifier does not bind source_family_id")
    if not SHA256_RE.fullmatch(family_id):
        errors.append(f"{context}.family_identifier must already be a content address")
    if value.get("family_identifier_semantics") != CONTENT_ADDRESSED_FAMILY_SEMANTICS:
        errors.append(f"{context}.family_identifier_semantics mismatch")
    if value.get("selected_family_sha256") != family_id:
        errors.append(f"{context}.selected_family_sha256 must preserve source-family binding")
    if value.get("source_family_sha256") != family_id:
        errors.append(f"{context}.source_family_sha256 does not bind source_family_id")
    if value.get("source_revision") != tau_revision:
        errors.append(f"{context}.source_revision does not bind row tau_revision")
    for field in (
        "source_file_sha256",
        "canonical_source_row_sha256",
        "task_id_sha256",
        "prompt_sha256",
        "source_family_sha256",
        "campaign_salt_sha256",
        "selection_rank_sha256",
        "task_sha256",
        "task_initial_state_sha256",
        "selection_stratum_sha256",
        "generation_stratum_sha256",
        "generation_recipe_sha256",
        "generated_family_identifier",
        "selected_family_sha256",
    ):
        if not SHA256_RE.fullmatch(str(value.get(field) or "")):
            errors.append(f"{context}.{field} must be sha256")
    for field in ("rank_ordinal", "generation_variant_ordinal"):
        if type(value.get(field)) is not int or value[field] < 0:
            errors.append(f"{context}.{field} must be a non-negative integer")

    stratum = value.get("selection_stratum_definition")
    expected_stratum_keys = {
        "schema_version",
        "source",
        "mapped_grounded_split",
        "domain",
        "eligible_source_family_sha256",
    }
    if not isinstance(stratum, dict):
        errors.append(f"{context}.selection_stratum_definition must be an object")
        stratum = {}
    elif set(stratum) != expected_stratum_keys:
        errors.append(f"{context}.selection_stratum_definition fields mismatch")
    if stratum.get("schema_version") != SELECTION_STRATUM_SCHEMA_VERSION:
        errors.append(f"{context}.selection_stratum_definition.schema_version mismatch")
    if stratum.get("source") != value.get("source"):
        errors.append(f"{context}.selection stratum source does not bind receipt")
    if stratum.get("mapped_grounded_split") != split:
        errors.append(f"{context}.selection stratum split does not bind row")
    if stratum.get("domain") != domain:
        errors.append(f"{context}.selection stratum domain does not bind row")
    eligible_family = stratum.get("eligible_source_family_sha256")
    if eligible_family is not None and eligible_family != family_id:
        errors.append(f"{context}.selection stratum family does not bind selected family")
    if eligible_family is not None and not SHA256_RE.fullmatch(str(eligible_family)):
        errors.append(f"{context}.selection stratum family must be null or sha256")
    if value.get("selection_stratum_sha256") != canonical_sha256(stratum):
        errors.append(f"{context}.selection_stratum_sha256 does not replay")

    generation_stratum = value.get("generation_stratum_definition")
    if not isinstance(generation_stratum, dict):
        errors.append(f"{context}.generation_stratum_definition must be an object")
        generation_stratum = {}
    if generation_stratum.get("schema_version") != GENERATION_STRATUM_SCHEMA_VERSION:
        errors.append(f"{context}.generation_stratum_definition.schema_version mismatch")
    if value.get("generation_stratum_sha256") != canonical_sha256(generation_stratum):
        errors.append(f"{context}.generation_stratum_sha256 does not replay")
    if (
        expected_generation_stratum is not None
        and generation_stratum != expected_generation_stratum
    ):
        errors.append(f"{context}.generation_stratum_definition does not bind generated content")

    recipe = value.get("generation_recipe")
    if not isinstance(recipe, dict):
        errors.append(f"{context}.generation_recipe must be an object")
        recipe = {}
    if set(recipe) != {"id", "version", "sha256"}:
        errors.append(f"{context}.generation_recipe fields mismatch")
    for field in ("id", "version"):
        if not isinstance(recipe.get(field), str) or not recipe[field]:
            errors.append(f"{context}.generation_recipe.{field} must be non-empty")
    expected_recipe_sha = canonical_sha256(
        {"id": recipe.get("id"), "version": recipe.get("version")}
    )
    if recipe.get("sha256") != expected_recipe_sha:
        errors.append(f"{context}.generation_recipe.sha256 does not replay id/version")
    if value.get("generation_recipe_sha256") != canonical_sha256(recipe):
        errors.append(f"{context}.generation_recipe_sha256 does not replay")
    if expected_recipe is not None and recipe != expected_recipe:
        errors.append(f"{context}.generation_recipe does not bind row recipe")

    generated_family_binding = {
        "schema_version": GENERATED_FAMILY_SCHEMA_VERSION,
        "source_family_sha256": value.get("source_family_sha256"),
        "source": value.get("source"),
        "mapped_grounded_split": value.get("mapped_grounded_split"),
        "domain": domain,
        "generation_stratum_sha256": value.get("generation_stratum_sha256"),
        "generation_recipe_sha256": value.get("generation_recipe_sha256"),
        "generation_variant_ordinal": value.get("generation_variant_ordinal"),
    }
    if value.get("generated_family_identifier") != canonical_sha256(
        generated_family_binding
    ):
        errors.append(f"{context}.generated_family_identifier does not replay composite binding")
    errors.extend(
        _scaled_selection_source_evidence_errors(
            value,
            family_id,
            context,
            domain=domain,
            split=split,
            stratum=stratum,
        )
    )
    expected_receipt_sha = canonical_sha256(
        {key: item for key, item in value.items() if key != "receipt_sha256"}
    )
    if value.get("receipt_sha256") != expected_receipt_sha:
        errors.append(f"{context}.receipt_sha256 does not replay")
    return errors


def _scaled_selection_source_evidence_errors(
    value: dict[str, Any],
    family_id: str,
    context: str,
    *,
    domain: str,
    split: str,
    stratum: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    spec = PERMITTED_SELECTION_SOURCES.get(split)
    if spec is None:
        return [f"{context}.mapped_grounded_split has no permitted source"]
    source_name = str(spec.get("source_split") or "")
    source_path = str(spec.get("path") or "")
    source_file_sha256 = str(spec.get("sha256") or "")
    if value.get("source") != source_name:
        errors.append(f"{context}.source does not match the registered source split")
    if value.get("mapped_grounded_split") != split:
        errors.append(f"{context}.mapped_grounded_split does not match row split")
    if split == "validation" and value.get("source") != "development":
        errors.append(f"{context}.validation must map only from development")
    if split == "train" and value.get("source") != "train":
        errors.append(f"{context}.train must map only from train")
    if value.get("source_file_path") != source_path:
        errors.append(f"{context}.source_file_path is not the registered source")
    if value.get("source_file_sha256") != source_file_sha256:
        errors.append(f"{context}.source_file_sha256 is not the pinned source hash")
    salt = value.get("campaign_salt")
    if not isinstance(salt, str) or not salt:
        errors.append(f"{context}.campaign_salt must be a non-empty string")
        salt = ""
    actual_salt_sha256 = hashlib.sha256(salt.encode("utf-8")).hexdigest()
    if value.get("campaign_salt_sha256") != actual_salt_sha256:
        errors.append(f"{context}.campaign_salt_sha256 does not bind campaign_salt")
    if actual_salt_sha256 != CAMPAIGN_SELECTION_SALT_SHA256:
        errors.append(f"{context}.campaign_salt is not the pinned campaign salt")
    line_number = value.get("source_line_number")
    if type(line_number) is not int or line_number < 1:
        errors.append(f"{context}.source_line_number must be a positive integer")
        return errors
    try:
        rows = _load_permitted_selection_rows(source_path, source_file_sha256)
    except Tau3GroundedGenerationError as exc:
        errors.append(f"{context}.source replay failed: {exc}")
        return errors
    selected = [(number, row) for number, row in rows if number == line_number]
    if len(selected) != 1:
        errors.append(f"{context}.source_line_number does not identify one source row")
        return errors
    source_row = selected[0][1]
    task = source_row.get("task")
    if not isinstance(task, dict):
        errors.append(f"{context}.selected source task is not an object")
        return errors
    if source_row.get("domain") != domain:
        errors.append(f"{context}.selected source domain does not match row domain")
    if source_row.get("split") != source_name:
        errors.append(f"{context}.selected source split does not replay")
    if source_row.get("task_family") != family_id:
        errors.append(f"{context}.selected source family does not bind source_family_id")
    if value.get("source_family_sha256") != source_row.get("task_family"):
        errors.append(f"{context}.source_family_sha256 does not replay")
    if value.get("source_revision") != source_row.get("source_revision"):
        errors.append(f"{context}.source_revision does not replay")
    canonical_source_row_sha256 = canonical_sha256(source_row)
    if value.get("canonical_source_row_sha256") != canonical_source_row_sha256:
        errors.append(f"{context}.canonical_source_row_sha256 does not replay")
    task_sha256 = canonical_sha256(task)
    if source_row.get("task_sha256") != task_sha256:
        errors.append(f"{context}.selected source task_sha256 does not replay")
    if value.get("task_sha256") != task_sha256:
        errors.append(f"{context}.task_sha256 does not bind the selected source task")
    task_id_sha256 = hashlib.sha256(str(task.get("id") or "").encode("utf-8")).hexdigest()
    if value.get("task_id_sha256") != task_id_sha256:
        errors.append(f"{context}.task_id_sha256 does not replay")
    prompt_sha256 = canonical_sha256(_dict(task.get("user_scenario")).get("instructions"))
    if source_row.get("prompt_sha256") != prompt_sha256:
        errors.append(f"{context}.selected source prompt_sha256 does not replay")
    if value.get("prompt_sha256") != prompt_sha256:
        errors.append(f"{context}.prompt_sha256 does not bind the selected source task")
    initial_state_sha256 = canonical_sha256(task.get("initial_state"))
    if value.get("task_initial_state_sha256") != initial_state_sha256:
        errors.append(f"{context}.task_initial_state_sha256 does not bind the selected task")
    expected_rank = _campaign_scaled_selection_rank(
        salt,
        str(value.get("selection_stratum_sha256") or ""),
        source_row,
    )
    if value.get("selection_rank_sha256") != expected_rank:
        errors.append(f"{context}.selection_rank_sha256 does not replay")
    eligible_family = stratum.get("eligible_source_family_sha256")
    candidates = [
        (number, row)
        for number, row in rows
        if row.get("domain") == domain
        and row.get("split") == source_name
        and (
            eligible_family is None
            or row.get("task_family") == eligible_family
        )
    ]
    ordered = sorted(
        candidates,
        key=lambda item: (
            _campaign_scaled_selection_rank(
                salt,
                str(value.get("selection_stratum_sha256") or ""),
                item[1],
            ),
            str(item[1].get("task_sha256") or ""),
            canonical_sha256(item[1]),
            item[0],
        ),
    )
    rank_ordinal = value.get("rank_ordinal")
    if type(rank_ordinal) is not int or rank_ordinal < 0 or rank_ordinal >= len(ordered):
        errors.append(f"{context}.rank_ordinal is out of range for the declared stratum")
        return errors
    expected_number, expected_row = ordered[rank_ordinal]
    if (
        expected_number != line_number
        or canonical_sha256(expected_row) != canonical_source_row_sha256
    ):
        errors.append(f"{context}.source row does not match the declared deterministic ordinal")
    return errors


def _selection_source_evidence_errors(
    value: dict[str, Any],
    family_id: str,
    context: str,
    *,
    domain: str,
    split: str,
) -> list[str]:
    errors: list[str] = []
    spec = PERMITTED_SELECTION_SOURCES.get(split)
    if spec is None:
        return [f"{context}.mapped_grounded_split has no permitted source"]
    source_name = str(spec.get("source_split") or "")
    source_path = str(spec.get("path") or "")
    source_file_sha256 = str(spec.get("sha256") or "")
    if value.get("source") != source_name:
        errors.append(f"{context}.source does not match the registered source split")
    if value.get("mapped_grounded_split") != split:
        errors.append(f"{context}.mapped_grounded_split does not match row split")
    if value.get("source_file_path") != source_path:
        errors.append(f"{context}.source_file_path is not the registered source")
    if value.get("source_file_sha256") != source_file_sha256:
        errors.append(f"{context}.source_file_sha256 is not the pinned source hash")
    salt = value.get("selection_salt")
    if not isinstance(salt, str) or not salt:
        errors.append(f"{context}.selection_salt must be a non-empty string")
        salt = ""
    actual_salt_sha256 = hashlib.sha256(salt.encode("utf-8")).hexdigest()
    if value.get("selection_salt_sha256") != actual_salt_sha256:
        errors.append(f"{context}.selection_salt_sha256 does not bind selection_salt")
    if actual_salt_sha256 != CAMPAIGN_SELECTION_SALT_SHA256:
        errors.append(f"{context}.selection_salt is not the pinned campaign salt")
    line_number = value.get("source_line_number")
    if type(line_number) is not int or line_number < 1:
        errors.append(f"{context}.source_line_number must be a positive integer")
        return errors
    try:
        rows = _load_permitted_selection_rows(source_path, source_file_sha256)
    except Tau3GroundedGenerationError as exc:
        errors.append(f"{context}.source replay failed: {exc}")
        return errors
    selected_matches = [row for number, row in rows if number == line_number]
    if len(selected_matches) != 1:
        errors.append(f"{context}.source_line_number does not identify one source row")
        return errors
    source_row = selected_matches[0]
    task = source_row.get("task")
    if not isinstance(task, dict):
        errors.append(f"{context}.selected source task is not an object")
        return errors
    if source_row.get("domain") != domain:
        errors.append(f"{context}.selected source domain does not match row domain")
    if source_row.get("split") != source_name:
        errors.append(f"{context}.selected source split does not replay")
    if source_row.get("task_family") != family_id:
        errors.append(f"{context}.selected source family does not bind source_family_id")
    if value.get("source_family_sha256") != source_row.get("task_family"):
        errors.append(f"{context}.source_family_sha256 does not replay")
    source_revision = source_row.get("source_revision")
    if value.get("source_revision") != source_revision:
        errors.append(f"{context}.source_revision does not replay")
    canonical_source_row_sha256 = canonical_sha256(source_row)
    if value.get("canonical_source_row_sha256") != canonical_source_row_sha256:
        errors.append(f"{context}.canonical_source_row_sha256 does not replay")
    actual_task_sha256 = canonical_sha256(task)
    if source_row.get("task_sha256") != actual_task_sha256:
        errors.append(f"{context}.selected source task_sha256 does not replay")
    if value.get("task_sha256") != actual_task_sha256:
        errors.append(f"{context}.task_sha256 does not bind the selected source task")
    task_id_sha256 = hashlib.sha256(str(task.get("id") or "").encode("utf-8")).hexdigest()
    if value.get("task_id_sha256") != task_id_sha256:
        errors.append(f"{context}.task_id_sha256 does not replay")
    instructions = _dict(task.get("user_scenario")).get("instructions")
    prompt_sha256 = canonical_sha256(instructions)
    if source_row.get("prompt_sha256") != prompt_sha256:
        errors.append(f"{context}.selected source prompt_sha256 does not replay")
    if value.get("prompt_sha256") != prompt_sha256:
        errors.append(f"{context}.prompt_sha256 does not bind the selected source task")
    task_initial_state_sha256 = canonical_sha256(task.get("initial_state"))
    if value.get("task_initial_state_sha256") != task_initial_state_sha256:
        errors.append(f"{context}.task_initial_state_sha256 does not bind the selected task")
    expected_rank = _campaign_selection_rank(salt, source_row)
    if value.get("selection_rank_sha256") != expected_rank:
        errors.append(f"{context}.selection_rank_sha256 does not replay")
    candidates = [
        (number, row)
        for number, row in rows
        if row.get("domain") == domain
    ]
    if not candidates:
        errors.append(f"{context}.registered source has no candidate for row domain")
        return errors
    expected_number, expected_row = min(
        candidates,
        key=lambda item: (
            _campaign_selection_rank(salt, item[1]),
            str(item[1].get("task_sha256") or ""),
            item[0],
        ),
    )
    if (
        line_number != expected_number
        or canonical_source_row_sha256 != canonical_sha256(expected_row)
    ):
        errors.append(
            f"{context}.selected source row is not the deterministic minimum for domain/split"
        )
    return errors


def _campaign_selection_rank(salt: str, source_row: dict[str, Any]) -> str:
    canonical = json.dumps(
        _canonical_value(source_row),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(salt.encode("utf-8") + b"\0" + canonical).hexdigest()


def _campaign_scaled_selection_rank(
    campaign_salt: str,
    selection_stratum_sha256: str,
    source_row: dict[str, Any],
) -> str:
    canonical = json.dumps(
        _canonical_value(source_row),
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


@lru_cache(maxsize=8)
def _load_permitted_selection_rows(
    source_path: str,
    expected_sha256: str,
) -> tuple[tuple[int, dict[str, Any]], ...]:
    if not _safe_relative_path(source_path):
        raise Tau3GroundedGenerationError("registered selection source path is unsafe")
    project_root = Path(__file__).resolve().parents[1]
    path = project_root / source_path
    if path_has_symlink_component(path, include_leaf=True):
        raise Tau3GroundedGenerationError(
            "registered selection source path contains a symlink component"
        )
    try:
        resolved = path.resolve(strict=True)
        root_resolved = project_root.resolve(strict=True)
    except OSError as exc:
        raise Tau3GroundedGenerationError(
            f"registered selection source cannot be resolved: {exc}"
        ) from exc
    if root_resolved not in resolved.parents or not resolved.is_file():
        raise Tau3GroundedGenerationError(
            "registered selection source must be a repository file"
        )
    if not SHA256_RE.fullmatch(expected_sha256) or _sha256(path) != expected_sha256:
        raise Tau3GroundedGenerationError("registered selection source hash mismatch")
    rows: list[tuple[int, dict[str, Any]]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise Tau3GroundedGenerationError(
            f"registered selection source cannot be read: {exc}"
        ) from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Tau3GroundedGenerationError(
                f"registered selection source has invalid JSON at line {line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise Tau3GroundedGenerationError(
                f"registered selection source row {line_number} is not an object"
            )
        rows.append((line_number, _canonical_value(value)))
    if not rows:
        raise Tau3GroundedGenerationError("registered selection source is empty")
    return tuple(rows)


def _tool_exemptions(payload: dict[str, Any], tool_catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw = payload.get("tool_exemptions", [])
    errors = _tool_exemption_errors(raw, tool_catalog, "source.tool_exemptions")
    if errors:
        raise Tau3GroundedGenerationError("; ".join(errors))
    return copy.deepcopy(raw)


def _tool_exemption_errors(value: Any, tool_catalog: list[dict[str, Any]], context: str) -> list[str]:
    errors: list[str] = []
    if value is None:
        return errors
    if not isinstance(value, list):
        return [f"{context} must be a list"]
    seen: set[str] = set()
    for index, record in enumerate(value):
        label = f"{context}[{index}]"
        if not isinstance(record, dict):
            errors.append(f"{label} must be an object")
            continue
        tool_name = record.get("tool_name")
        if not isinstance(tool_name, str) or not tool_name:
            errors.append(f"{label}.tool_name must be a non-empty string")
            continue
        if tool_name in seen:
            errors.append(f"{label}.tool_name duplicates an earlier exemption")
        seen.add(tool_name)
        try:
            tool_def = _find_tool(tool_catalog, tool_name)
        except Tau3GroundedGenerationError as exc:
            errors.append(f"{label}.tool_name is not in exact catalog: {exc}")
            continue
        reason = record.get("reason")
        if reason not in {"zero_arg", "policy_forbidden"}:
            errors.append(f"{label}.reason must be zero_arg or policy_forbidden")
            continue
        if not isinstance(record.get("reviewer"), str) or not record["reviewer"].strip():
            errors.append(f"{label}.reviewer must be a non-empty string")
        if reason == "zero_arg" and not _is_zero_argument_tool(tool_def):
            errors.append(
                f"{label}.reason zero_arg requires a runtime tool with no declared arguments"
            )
        if reason == "policy_forbidden":
            policy_hash = record.get("policy_hash")
            citation = record.get("citation")
            if not isinstance(policy_hash, str) or not SHA256_RE.fullmatch(policy_hash):
                errors.append(f"{label}.policy_hash must be a non-empty sha256 for policy_forbidden")
            if not isinstance(citation, str) or not citation.strip():
                errors.append(f"{label}.citation must be non-empty for policy_forbidden")
    return errors


def _required_arg_count(tool_def: dict[str, Any]) -> int:
    params = tool_def.get("parameters")
    if not isinstance(params, dict):
        params = tool_def.get("params")
    if not isinstance(params, dict):
        function = tool_def.get("function")
        if isinstance(function, dict):
            params = function.get("parameters")
    if not isinstance(params, dict):
        return 0
    required = params.get("required")
    if isinstance(required, list):
        return len(required)
    return 0


def _is_zero_argument_tool(tool_def: dict[str, Any]) -> bool:
    params = tool_def.get("parameters")
    if not isinstance(params, dict):
        params = tool_def.get("params")
    if not isinstance(params, dict):
        function = tool_def.get("function")
        if isinstance(function, dict):
            params = function.get("parameters")
    if not isinstance(params, dict):
        return False
    properties = params.get("properties")
    required = params.get("required", [])
    return properties == {} and required == []


def _replay_call(
    runtime: Any,
    raw_call: Any,
    *,
    tool_catalog: list[dict[str, Any]],
    tool_catalog_hash: str,
    parent_turn_ordinal: int,
    assistant_decision_ordinal: int,
    tool_call_ordinal: int,
    prior_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(raw_call, dict):
        raise Tau3GroundedGenerationError("tool call must be an object")
    tool_name = raw_call.get("tool_name")
    if not isinstance(tool_name, str) or not tool_name:
        raise Tau3GroundedGenerationError("tool_call.tool_name must be a non-empty string")
    args = _canonical_value(raw_call.get("arguments") or {})
    if not isinstance(args, dict):
        raise Tau3GroundedGenerationError("tool_call.arguments must be an object")
    tool_def = _find_tool(tool_catalog, tool_name)
    tool_mutates_state = bool(runtime.tool_mutates_state(tool_name))
    pre_state = copy.deepcopy(runtime.state)
    pre_hash = canonical_sha256(pre_state)
    result_class = "success"
    exception = None
    try:
        result = runtime.call(tool_name, copy.deepcopy(args))
        result_class = _tool_result_class(result)
    except Exception as exc:  # deliberate: tool exceptions are evidence.
        result = {"error": exc.__class__.__name__, "message": str(exc)}
        result_class = "exception"
        exception = {
            "type": exc.__class__.__name__,
            "message": str(exc),
        }
    canonical_result = _canonical_value(result)
    result_sha256 = canonical_sha256(canonical_result)
    expected_result_sha256 = raw_call.get("expected_result_sha256")
    expected_result_class = raw_call.get("expected_result_class")
    expected_verified = expected_result_sha256 is not None or expected_result_class is not None
    if expected_result_sha256 is not None:
        if not isinstance(expected_result_sha256, str) or not SHA256_RE.fullmatch(expected_result_sha256):
            raise Tau3GroundedGenerationError("tool_call.expected_result_sha256 must be a sha256")
        if expected_result_sha256 != result_sha256:
            raise Tau3GroundedGenerationError("tool_call.expected_result_sha256 does not match replayed result")
    if expected_result_class is not None:
        if expected_result_class not in {"success", "empty", "error", "exception"}:
            raise Tau3GroundedGenerationError("tool_call.expected_result_class is invalid")
        if expected_result_class != result_class:
            raise Tau3GroundedGenerationError("tool_call.expected_result_class does not match replayed result class")
    post_state = copy.deepcopy(runtime.state)
    post_hash = canonical_sha256(post_state)
    state_diff = _state_diff(pre_state, post_state)
    if state_diff["change_count"] > 0 and pre_hash == post_hash:
        raise Tau3GroundedGenerationError("state diff claims mutation but state hash did not change")
    repeated_prior = sum(
        1
        for call in prior_calls
        if call["tool_name"] == tool_name and call["canonical_arguments"] == args
    )
    sync_evidence = copy.deepcopy(getattr(runtime, "last_sync_evidence", None))
    if not isinstance(sync_evidence, dict):
        sync_evidence = {
            "performed": False,
            "pre_state": copy.deepcopy(post_state),
            "post_state": copy.deepcopy(post_state),
            "pre_state_sha256": post_hash,
            "post_state_sha256": post_hash,
            "state_diff": _state_diff(post_state, post_state),
        }
    if _canonical_value(sync_evidence.get("post_state")) != post_state:
        raise Tau3GroundedGenerationError("sync evidence post-state is not the runtime post-state")
    return {
        "parent_turn_ordinal": parent_turn_ordinal,
        "parent_assistant_decision_ordinal": assistant_decision_ordinal,
        "tool_call_ordinal": tool_call_ordinal,
        "tool_name": tool_name,
        "tool_definition_sha256": canonical_sha256(tool_def),
        "tool_catalog_sha256": tool_catalog_hash,
        "tool_mutates_state": tool_mutates_state,
        "canonical_arguments": args,
        "arguments_sha256": canonical_sha256(args),
        "result_class": result_class,
        "canonical_result": canonical_result,
        "result_sha256": result_sha256,
        "source_expected_result_sha256": expected_result_sha256,
        "source_expected_result_class": expected_result_class,
        "source_expected_result_verified": expected_verified,
        "exception": exception,
        "pre_state_sha256": pre_hash,
        "post_state_sha256": post_hash,
        "state_diff": state_diff,
        "pre_state": pre_state,
        "pre_sync_state": copy.deepcopy(sync_evidence["pre_state"]),
        "post_state": post_state,
        "sync_evidence": sync_evidence,
        "context": {
            "empty_result": result_class == "empty",
            "error_result": result_class in {"error", "exception"},
            "repeated_call_prior_count": repeated_prior,
            "repeated_call": repeated_prior > 0,
        },
        "evidence_replayed": True,
    }


def _tool_result_class(result: Any) -> str:
    """Classify native and JSON-serialized Tau tool results consistently."""

    candidate = result
    if isinstance(result, str):
        try:
            candidate = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            candidate = result
    if candidate is None or candidate == [] or candidate == {}:
        return "empty"
    if isinstance(candidate, dict) and candidate.get("error"):
        return "error"
    return "success"


def _validate_row(row: Any, context: str, bundle: Path) -> list[str]:
    errors: list[str] = []
    if not isinstance(row, dict):
        return [f"{context} must be an object"]
    metadata = _object(row.get("metadata"), f"{context}.metadata", errors)
    if row.get("schema_version") != TAU3_GROUNDED_ROW_SCHEMA_VERSION:
        errors.append(f"{context}.schema_version mismatch")
    if metadata.get("schema_version") != TAU3_GROUNDED_ROW_SCHEMA_VERSION:
        errors.append(f"{context}.metadata.schema_version mismatch")
    if metadata.get("lineage_id") != LINEAGE_ID:
        errors.append(f"{context}.metadata.lineage_id mismatch")
    if metadata.get("training_side_only") is not True:
        errors.append(f"{context} must be training_side_only")
    if metadata.get("domain") not in DOMAINS:
        errors.append(f"{context}.metadata.domain invalid")
    if metadata.get("split") not in SPLITS:
        errors.append(f"{context}.metadata.split invalid")
    if metadata.get("source_family") not in ALLOWED_SOURCE_FAMILIES:
        errors.append(f"{context}.metadata.source_family invalid")
    if not _is_replayable_runtime_family(str(metadata.get("runtime_family") or "")):
        errors.append(f"{context}.metadata.runtime_family invalid")
    if not _runtime_is_candidate_eligible(metadata.get("runtime_family")):
        errors.append(f"{context}.metadata.runtime_family is not candidate-eligible vendored Tau replay")
    if not HEX40_RE.fullmatch(str(metadata.get("tau_revision") or "")):
        errors.append(f"{context}.metadata.tau_revision must be immutable 40-hex")
    tau_repo = _object(metadata.get("tau_repo"), f"{context}.metadata.tau_repo", errors)
    tau_repo_path = tau_repo.get("path")
    if _runtime_is_vendored(str(metadata.get("runtime_family") or "")):
        if not isinstance(tau_repo_path, str) or not _safe_relative_path(tau_repo_path):
            errors.append(f"{context}.metadata.tau_repo.path must be a safe relative path")
        if tau_repo.get("revision") != metadata.get("tau_revision"):
            errors.append(f"{context}.metadata.tau_repo.revision must match tau_revision")
        if not SHA256_RE.fullmatch(str(tau_repo.get("tree_sha256") or "")):
            errors.append(f"{context}.metadata.tau_repo.tree_sha256 must be sha256")
        elif isinstance(tau_repo_path, str) and _safe_relative_path(tau_repo_path):
            try:
                expected_tree_sha = _tau_repo_tree_sha256(
                    _resolve_tau_repo(tau_repo_path),
                    str(metadata.get("tau_revision") or ""),
                )
            except Exception as exc:
                errors.append(f"{context}.metadata.tau_repo.tree_sha256 cannot be replayed: {exc}")
            else:
                if tau_repo.get("tree_sha256") != expected_tree_sha:
                    errors.append(f"{context}.metadata.tau_repo.tree_sha256 does not replay")
    for field in (
        "system_prompt_sha256",
        "policy_sha256",
        "tool_catalog_sha256",
        "initial_state_sha256",
        "final_state_sha256",
        "row_sha256",
    ):
        if not SHA256_RE.fullmatch(str(metadata.get(field) or "")):
            errors.append(f"{context}.metadata.{field} must be sha256")
    errors.extend(
        _contamination_errors(
            metadata.get("contamination"),
            str(metadata.get("split") or ""),
            f"{context}.metadata",
        )
    )
    candidate = _runtime_is_candidate_eligible(metadata.get("runtime_family"))
    selection_receipt = _dict(metadata.get("selection_receipt"))
    scaled_selection = (
        selection_receipt.get("schema_version")
        == SCALED_SELECTION_RECEIPT_SCHEMA_VERSION
    )
    if candidate:
        for field in (
            "full_environment_state",
            "runtime_prompt_derived",
            "openai_tool_catalog_derived",
        ):
            if metadata.get(field) is not True:
                errors.append(f"{context}.metadata.{field} must be true")
    errors.extend(
        _generation_provenance_errors(
            metadata.get("generation_provenance"),
            candidate,
            f"{context}.metadata.generation_provenance",
            scaled_generation=scaled_selection,
        )
    )
    errors.extend(
        _selection_receipt_errors(
            metadata.get("selection_receipt"),
            str(metadata.get("source_family_id") or ""),
            candidate,
            f"{context}.metadata.selection_receipt",
            domain=str(metadata.get("domain") or ""),
            split=str(metadata.get("split") or ""),
            tau_revision=str(metadata.get("tau_revision") or ""),
            expected_generation_stratum=_generation_stratum_definition(
                source_provenance=str(metadata.get("source_family") or ""),
                targets=(
                    row.get("training_targets")
                    if isinstance(row.get("training_targets"), list)
                    else []
                ),
                tool_history=(
                    row.get("tool_replay")
                    if isinstance(row.get("tool_replay"), list)
                    else []
                ),
                turns=(
                    _dict(row.get("trajectory")).get("turns")
                    if isinstance(_dict(row.get("trajectory")).get("turns"), list)
                    else []
                ),
            ),
            expected_recipe=(
                metadata.get("recipe")
                if isinstance(metadata.get("recipe"), dict)
                else None
            ),
        )
    )
    if selection_receipt.get("schema_version") == SCALED_SELECTION_RECEIPT_SCHEMA_VERSION:
        if metadata.get("generated_family_id") != selection_receipt.get(
            "generated_family_identifier"
        ):
            errors.append(
                f"{context}.metadata.generated_family_id does not bind scaled selection"
            )

    trajectory = _object(row.get("trajectory"), f"{context}.trajectory", errors)
    errors.extend(
        _decision_order_errors(
            trajectory.get("turns"),
            f"{context}.trajectory.turns",
        )
    )
    if trajectory.get("split") != metadata.get("split"):
        errors.append(f"{context}.trajectory split mismatch")
    if trajectory.get("domain") != metadata.get("domain"):
        errors.append(f"{context}.trajectory domain mismatch")
    if canonical_sha256(str(trajectory.get("system_prompt") or "")) != metadata.get("system_prompt_sha256"):
        errors.append(f"{context}.system_prompt_sha256 does not replay")
    tool_catalog = row.get("tool_catalog")
    if not isinstance(tool_catalog, list) or not tool_catalog:
        errors.append(f"{context}.tool_catalog must be non-empty")
        tool_catalog = []
    elif canonical_sha256(tool_catalog) != metadata.get("tool_catalog_sha256"):
        errors.append(f"{context}.tool_catalog_sha256 does not replay")
    errors.extend(
        _generation_provenance_binding_errors(
            metadata.get("generation_provenance"),
            tau_revision=str(metadata.get("tau_revision") or ""),
            system_prompt_sha256=str(metadata.get("system_prompt_sha256") or ""),
            tool_catalog_sha256=str(metadata.get("tool_catalog_sha256") or ""),
            source_task_sha256=str(
                _dict(metadata.get("selection_receipt")).get("task_sha256")
                or metadata.get("source_sha256")
                or ""
            ),
            selection_receipt_sha256=str(
                _dict(metadata.get("selection_receipt")).get("receipt_sha256") or ""
            ),
            context=f"{context}.metadata.generation_provenance",
            scaled_selection_receipt=(
                selection_receipt if scaled_selection else None
            ),
        )
    )
    errors.extend(
        _reviewer_generation_binding_errors(
            metadata.get("reviewer"),
            metadata.get("generation_provenance"),
            candidate=candidate,
            context=f"{context}.metadata.reviewer",
        )
    )
    if "initial_state" in row:
        errors.append(f"{context}.initial_state must not be embedded; use initial_state_ref")
    initial_state = _load_state_ref(bundle, row.get("initial_state_ref"), context, errors)
    if canonical_sha256(initial_state) != metadata.get("initial_state_sha256"):
        errors.append(f"{context}.initial_state_sha256 does not replay")
    if candidate and _dict(metadata.get("selection_receipt")).get(
        "task_initial_state_sha256"
    ) != canonical_sha256(_dict(initial_state).get("task_initialization")):
        errors.append(
            f"{context}.metadata.selection_receipt does not bind task initialization"
        )
    final_state = _load_state_ref(bundle, row.get("final_state_ref"), f"{context}.final", errors)
    if canonical_sha256(final_state) != metadata.get("final_state_sha256"):
        errors.append(f"{context}.final_state_sha256 does not replay")
    initial_sync = _expanded_initial_sync_evidence(
        bundle,
        row.get("initial_sync"),
        f"{context}.initial_sync",
        errors,
    )
    if candidate:
        sync_count = initial_sync.get("sync_count")
        if type(sync_count) is not int or sync_count < 1:
            errors.append(f"{context}.initial_sync must retain at least one physical sync step")
        if _dict(initial_state).get("pre_sync") != initial_sync.get("pre_state"):
            errors.append(f"{context}.initial_sync pre-state does not bind initial_state")
        if _dict(initial_state).get("post_sync") != initial_sync.get("post_state"):
            errors.append(f"{context}.initial_sync post-state does not bind initial_state")
        domain = str(metadata.get("domain") or "")
        for snapshot_context, snapshot in (
            (f"{context}.initial_state.pre_sync", _dict(initial_state).get("pre_sync")),
            (f"{context}.initial_state.post_sync", _dict(initial_state).get("post_sync")),
            (f"{context}.initial_sync.pre_state", initial_sync.get("pre_state")),
            (f"{context}.initial_sync.post_state", initial_sync.get("post_state")),
            (f"{context}.final_state", final_state),
        ):
            errors.extend(
                _full_environment_snapshot_errors(
                    snapshot,
                    domain=domain,
                    context=snapshot_context,
                )
            )
        for step_index, step in enumerate(initial_sync.get("steps", [])):
            if not isinstance(step, dict):
                continue
            for state_name in ("pre_state", "post_state"):
                snapshot = step.get(state_name)
                errors.extend(
                    _full_environment_snapshot_errors(
                        snapshot,
                        domain=domain,
                        context=(
                            f"{context}.initial_sync.steps[{step_index}].{state_name}"
                        ),
                    )
                )
    if canonical_sha256(_without_row_sha(row)) != metadata.get("row_sha256"):
        errors.append(f"{context}.row_sha256 does not replay")

    targets = row.get("training_targets")
    if not isinstance(targets, list) or not targets:
        errors.append(f"{context}.training_targets must be non-empty")
    else:
        for index, target in enumerate(targets):
            errors.extend(_validate_target(target, f"{context}.training_targets[{index}]"))
        errors.extend(_masked_correction_link_errors(targets, context))
        errors.extend(
            _policy_reviewer_record_errors(
                metadata.get("reviewer"),
                [target for target in targets if isinstance(target, dict)],
                policy_sha256=str(metadata.get("policy_sha256") or ""),
                candidate=candidate,
                context=f"{context}.metadata.reviewer",
            )
        )

    replay = row.get("tool_replay")
    if not isinstance(replay, list):
        errors.append(f"{context}.tool_replay must be a list")
        replay = []
    try:
        runtime = _runtime_for_scenario(
            {
                "runtime_family": metadata.get("runtime_family"),
                "tau_revision": metadata.get("tau_revision"),
                "domain": metadata.get("domain"),
                "initial_state": initial_state,
                "tau_repo": tau_repo_path,
            }
        )
        runtime_tool_catalog = runtime.tool_catalog()
    except Exception as exc:
        errors.append(f"{context} runtime cannot be instantiated for replay: {exc}")
        runtime = None
        runtime_tool_catalog = tool_catalog
    if runtime_tool_catalog != _canonical_value(tool_catalog):
        errors.append(f"{context}.tool_catalog is not the exact runtime-derived catalog")
    if runtime is not None and candidate:
        try:
            runtime_prompt = runtime.system_prompt()
            runtime_policy_sha = runtime.policy_sha256()
        except Exception as exc:
            errors.append(f"{context} runtime prompt/policy cannot be derived: {exc}")
        else:
            if trajectory.get("system_prompt") != runtime_prompt:
                errors.append(f"{context}.trajectory.system_prompt is not exact LLMAgent output")
            if metadata.get("policy_sha256") != runtime_policy_sha:
                errors.append(f"{context}.metadata.policy_sha256 is not runtime-derived")
        if initial_sync != _canonical_value(runtime.initial_sync_evidence):
            errors.append(f"{context}.initial_sync does not replay")
    errors.extend(
        _tool_exemption_errors(
            metadata.get("tool_exemptions"),
            tool_catalog,
            f"{context}.metadata.tool_exemptions",
        )
    )
    prior_calls: list[dict[str, Any]] = []
    per_turn_call_counts: dict[int, int] = {}
    for index, recorded in enumerate(replay):
        if not isinstance(recorded, dict):
            errors.append(f"{context}.tool_replay[{index}] must be an object")
            continue
        parent_turn = recorded.get("parent_turn_ordinal")
        parent_decision = recorded.get("parent_assistant_decision_ordinal")
        tool_call_ordinal = recorded.get("tool_call_ordinal")
        if type(parent_turn) is not int or parent_turn != parent_decision:
            errors.append(
                f"{context}.tool_replay[{index}] parent turn does not bind physical decision"
            )
        expected_call_ordinal = per_turn_call_counts.get(parent_turn, 0)
        if tool_call_ordinal != expected_call_ordinal:
            errors.append(
                f"{context}.tool_replay[{index}] tool_call_ordinal is not chronological"
            )
        if type(parent_turn) is int:
            per_turn_call_counts[parent_turn] = expected_call_ordinal + 1
        trajectory_turns = trajectory.get("turns")
        if (
            isinstance(trajectory_turns, list)
            and type(parent_turn) is int
            and 0 <= parent_turn < len(trajectory_turns)
            and isinstance(trajectory_turns[parent_turn], dict)
        ):
            trajectory_calls = _dict(trajectory_turns[parent_turn].get("assistant")).get(
                "tool_calls"
            )
            if (
                not isinstance(trajectory_calls, list)
                or type(tool_call_ordinal) is not int
                or not 0 <= tool_call_ordinal < len(trajectory_calls)
                or not isinstance(trajectory_calls[tool_call_ordinal], dict)
            ):
                errors.append(
                    f"{context}.tool_replay[{index}] is absent from its physical trajectory turn"
                )
            else:
                trajectory_call = trajectory_calls[tool_call_ordinal]
                if trajectory_call.get("tool_name") != recorded.get("tool_name"):
                    errors.append(
                        f"{context}.tool_replay[{index}] tool_name does not bind trajectory"
                    )
                if _canonical_value(trajectory_call.get("arguments") or {}) != recorded.get(
                    "canonical_arguments"
                ):
                    errors.append(
                        f"{context}.tool_replay[{index}] arguments do not bind trajectory"
                    )
                source_confirmation = _canonical_value(trajectory_call.get("confirmation"))
                recorded_confirmation = _canonical_value(recorded.get("confirmation"))
                if isinstance(recorded_confirmation, dict):
                    recorded_confirmation = {
                        key: value
                        for key, value in recorded_confirmation.items()
                        if key not in {"confirmed_arguments", "detail_grounding"}
                    }
                if source_confirmation != recorded_confirmation:
                    errors.append(
                        f"{context}.tool_replay[{index}] confirmation does not bind trajectory"
                    )
                for source_field, replay_field in (
                    ("expected_result_sha256", "source_expected_result_sha256"),
                    ("expected_result_class", "source_expected_result_class"),
                ):
                    if trajectory_call.get(source_field) != recorded.get(replay_field):
                        errors.append(
                            f"{context}.tool_replay[{index}] {source_field} does not bind trajectory"
                        )
        if recorded.get("arguments_sha256") != canonical_sha256(
            recorded.get("canonical_arguments")
        ):
            errors.append(f"{context}.tool_replay[{index}].arguments_sha256 does not replay")
        if recorded.get("result_sha256") != canonical_sha256(recorded.get("canonical_result")):
            errors.append(f"{context}.tool_replay[{index}].result_sha256 does not replay")
        try:
            if runtime is None:
                raise Tau3GroundedGenerationError("runtime unavailable")
            actual = _replay_call(
                runtime,
                {
                    "tool_name": recorded.get("tool_name"),
                    "arguments": recorded.get("canonical_arguments"),
                    "expected_result_sha256": recorded.get("source_expected_result_sha256"),
                    "expected_result_class": recorded.get("source_expected_result_class"),
                },
                tool_catalog=tool_catalog,
                tool_catalog_hash=canonical_sha256(tool_catalog),
                parent_turn_ordinal=int(recorded.get("parent_turn_ordinal", -1)),
                assistant_decision_ordinal=int(recorded.get("parent_assistant_decision_ordinal", -1)),
                tool_call_ordinal=int(recorded.get("tool_call_ordinal", -1)),
                prior_calls=prior_calls,
            )
            actual["argument_grounding"] = _argument_grounding_evidence(
                actual["canonical_arguments"],
                int(actual["parent_assistant_decision_ordinal"]),
                _list_required(trajectory.get("turns"), f"{context}.trajectory.turns"),
                prior_calls,
            )
            actual["confirmation"] = _canonical_value(recorded.get("confirmation"))
        except Exception as exc:
            errors.append(f"{context}.tool_replay[{index}] cannot replay: {exc}")
            continue
        for field in (
            "parent_turn_ordinal",
            "parent_assistant_decision_ordinal",
            "tool_call_ordinal",
            "tool_name",
            "tool_definition_sha256",
            "tool_catalog_sha256",
            "tool_mutates_state",
            "canonical_arguments",
            "arguments_sha256",
            "result_class",
            "canonical_result",
            "result_sha256",
            "source_expected_result_sha256",
            "source_expected_result_class",
            "source_expected_result_verified",
            "exception",
            "pre_state_sha256",
            "post_state_sha256",
            "state_diff",
            "argument_grounding",
            "confirmation",
            "context",
        ):
            if recorded.get(field) != actual.get(field):
                errors.append(f"{context}.tool_replay[{index}].{field} does not replay")
        for state_field, ref_field in (
            ("pre_state", "pre_state_ref"),
            ("pre_sync_state", "pre_sync_state_ref"),
            ("post_state", "post_state_ref"),
        ):
            recorded_state = _load_state_ref(
                bundle,
                recorded.get(ref_field),
                f"{context}.tool_replay[{index}].{ref_field}",
                errors,
            )
            if recorded_state != actual.get(state_field):
                errors.append(
                    f"{context}.tool_replay[{index}].{ref_field} does not replay full environment state"
                )
        recorded_sync = _expanded_sync_evidence(
            bundle,
            recorded.get("sync_evidence"),
            f"{context}.tool_replay[{index}].sync_evidence",
            errors,
        )
        if recorded_sync != _canonical_value(actual.get("sync_evidence")):
            errors.append(f"{context}.tool_replay[{index}].sync_evidence does not replay")
        if recorded.get("evidence_replayed") is not True:
            errors.append(f"{context}.tool_replay[{index}].evidence_replayed must be true")
        prior_calls.append(actual)
    if runtime is not None:
        if canonical_sha256(runtime.state) != metadata.get("final_state_sha256"):
            errors.append(f"{context}.metadata.final_state_sha256 does not replay")
        if _canonical_value(runtime.state) != final_state:
            errors.append(f"{context}.final_state_ref does not replay")
    errors.extend(_target_binding_errors(targets, tool_catalog, replay, context))
    if runtime is not None:
        errors.extend(
            _policy_call_errors(
                domain=str(metadata.get("domain") or ""),
                turns=trajectory.get("turns"),
                targets=targets,
                replay=replay,
                policy_sha256=runtime.policy_sha256(),
                reviewer_record=metadata.get("reviewer"),
                candidate=candidate,
                context=context,
            )
        )
    errors.extend(
        _recovery_context_errors(
            targets,
            replay,
            trajectory.get("turns"),
            context,
        )
    )
    errors.extend(_completion_claim_errors(row, context))
    return errors


def _recovery_context_errors(
    targets: Any,
    replay: Any,
    turns: Any,
    context: str,
) -> list[str]:
    """Require recovery targets to follow the replay condition they claim."""

    errors: list[str] = []
    if not isinstance(targets, list) or not isinstance(replay, list):
        return errors
    turn_list = turns if isinstance(turns, list) else []
    turns_by_decision = {
        assistant.get("decision_ordinal"): turn
        for turn in turn_list
        if isinstance(turn, dict)
        for assistant in [_dict(turn.get("assistant"))]
        if type(assistant.get("decision_ordinal")) is int
    }
    ordered_replay = sorted(
        (call for call in replay if isinstance(call, dict)),
        key=lambda call: (
            int(call.get("parent_assistant_decision_ordinal", -1)),
            int(call.get("tool_call_ordinal", -1)),
        ),
    )
    for index, target in enumerate(targets):
        if not isinstance(target, dict) or target.get("masked") is True:
            continue
        behavior = target.get("behavior")
        if behavior not in {
            "empty_result_recovery",
            "error_result_recovery",
            "repeated_call_recovery",
        }:
            continue
        decision = target.get("parent_assistant_decision_ordinal")
        label = f"{context}.training_targets[{index}]"
        if type(decision) is not int:
            continue
        turn = turns_by_decision.get(decision)
        if not isinstance(turn, dict):
            errors.append(f"{label} recovery decision is absent from the parent trajectory")
            continue
        user = turn.get("user")
        if isinstance(user, dict) and str(user.get("content") or "").strip():
            errors.append(
                f"{label} recovery must immediately follow tool evidence without an intervening user message"
            )
        prior = [
            call
            for call in ordered_replay
            if type(call.get("parent_assistant_decision_ordinal")) is int
            and call["parent_assistant_decision_ordinal"] < decision
        ]
        if not prior:
            errors.append(f"{label} recovery lacks preceding replayed tool evidence")
            continue
        evidence = prior[-1]
        if evidence.get("parent_assistant_decision_ordinal") != decision - 1:
            errors.append(f"{label} recovery does not immediately follow the evidence decision")
        result_class = evidence.get("result_class")
        if behavior == "empty_result_recovery" and result_class != "empty":
            errors.append(
                f"{label} claims empty-result recovery after result_class={result_class!r}"
            )
        elif behavior == "error_result_recovery" and result_class not in {
            "error",
            "exception",
        }:
            errors.append(
                f"{label} claims error-result recovery after result_class={result_class!r}"
            )
        elif behavior == "repeated_call_recovery":
            call_context = _dict(evidence.get("context"))
            if call_context.get("repeated_call") is not True:
                errors.append(
                    f"{label} claims repeated-call recovery without an identical prior call"
                )
    return errors


def _validate_target(target: Any, context: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(target, dict):
        return [f"{context} must be an object"]
    if target.get("behavior") not in BEHAVIORS:
        errors.append(f"{context}.behavior is not in rubric")
    if not isinstance(target.get("parent_assistant_decision_ordinal"), int):
        errors.append(f"{context}.parent_assistant_decision_ordinal must be an integer")
    if not isinstance(target.get("parent_turn_ordinal"), int):
        errors.append(f"{context}.parent_turn_ordinal must be an integer")
    elif target.get("parent_turn_ordinal") != target.get("parent_assistant_decision_ordinal"):
        errors.append(f"{context}.parent_turn_ordinal must bind the physical assistant decision")
    if target.get("masked") is True:
        if not target.get("mask_reason"):
            errors.append(f"{context}.mask_reason is required for masked targets")
        if target.get("reviewed") is not True:
            errors.append(f"{context}.reviewed must be true for masked targets")
        decision = target.get("parent_assistant_decision_ordinal")
        safe_decision = target.get("safe_correction_decision_ordinal")
        if (
            type(decision) is not int
            or type(safe_decision) is not int
            or safe_decision <= decision
        ):
            errors.append(
                f"{context}.safe_correction_decision_ordinal must identify a later decision"
            )
        negative_behavior = target.get("negative_behavior")
        expected_negative = {
            "hallucinated_tool_correction": "hallucinated_tool",
            "harmful_mutation_correction": "harmful_mutation",
            "premature_completion_correction": "premature_completion",
        }.get(target.get("behavior"))
        if negative_behavior != expected_negative:
            errors.append(f"{context}.negative_behavior does not match correction behavior")
        canonical = target.get("canonical_target")
        if not isinstance(canonical, dict):
            errors.append(f"{context}.canonical_target must retain the reviewed negative action")
        else:
            errors.extend(_target_shape_errors(canonical, f"{context}.canonical_target"))
            if (
                canonical.get("kind") == "assistant_message"
                and not str(canonical.get("text") or "")
            ):
                errors.append(
                    f"{context}.canonical_target masked assistant action must carry explicit text"
                )
            if canonical_sha256(canonical) != target.get("canonical_target_sha256"):
                errors.append(f"{context}.canonical_target_sha256 does not replay")
        return errors
    canonical = target.get("canonical_target")
    if not isinstance(canonical, dict):
        errors.append(f"{context}.canonical_target must be an object")
    else:
        errors.extend(_target_shape_errors(canonical, f"{context}.canonical_target"))
        if canonical_sha256(canonical) != target.get("canonical_target_sha256"):
            errors.append(f"{context}.canonical_target_sha256 does not replay")
    if canonical and _is_mutation_tool(str(canonical.get("tool_name") or "")) and target.get("behavior") in {
        "harmful_mutation_correction",
        "premature_completion_correction",
    }:
        errors.append(f"{context} unsafe corrected mutation target must be masked")
    return errors


def _masked_correction_link_errors(targets: list[Any], context: str) -> list[str]:
    errors: list[str] = []
    by_decision: dict[int, list[dict[str, Any]]] = {}
    for target in targets:
        if isinstance(target, dict) and type(target.get("parent_assistant_decision_ordinal")) is int:
            by_decision.setdefault(target["parent_assistant_decision_ordinal"], []).append(target)
    for index, target in enumerate(targets):
        if not isinstance(target, dict) or target.get("masked") is not True:
            continue
        safe_decision = target.get("safe_correction_decision_ordinal")
        linked = by_decision.get(safe_decision, []) if type(safe_decision) is int else []
        safe = [
            item
            for item in linked
            if item.get("masked") is not True
            and item.get("behavior") == target.get("behavior")
            and type(item.get("parent_turn_ordinal")) is int
            and type(target.get("parent_turn_ordinal")) is int
            and item["parent_turn_ordinal"] > target["parent_turn_ordinal"]
        ]
        if len(safe) != 1:
            errors.append(
                f"{context}.training_targets[{index}] must link to exactly one later unmasked correction"
            )
        elif safe[0].get("canonical_target_sha256") == target.get("canonical_target_sha256"):
            errors.append(
                f"{context}.training_targets[{index}] correction must differ from the masked action"
            )
    return errors


def _completion_claim_errors(row: dict[str, Any], context: str) -> list[str]:
    errors: list[str] = []
    replay = row.get("tool_replay") if isinstance(row.get("tool_replay"), list) else []
    targets = row.get("training_targets") if isinstance(row.get("training_targets"), list) else []
    for index, target in enumerate(targets):
        if not isinstance(target, dict) or target.get("masked") is True:
            continue
        canonical = target.get("canonical_target")
        text = str(_dict(canonical).get("text") or "").lower()
        if target.get("behavior") == "successful_completion" or "completed" in text or "success" in text:
            decision = target.get("parent_assistant_decision_ordinal")
            prior_mutation_replayed = any(
                isinstance(call, dict)
                and call.get("evidence_replayed") is True
                and type(decision) is int
                and type(call.get("parent_assistant_decision_ordinal")) is int
                and call.get("parent_assistant_decision_ordinal") < decision
                and call.get("pre_state_sha256") != call.get("post_state_sha256")
                and _dict(call.get("state_diff")).get("change_count", 0) > 0
                for call in replay
            )
            if not prior_mutation_replayed:
                errors.append(
                    f"{context}.training_targets[{index}] fabricates completion without prior replayed post-state mutation"
                )
    return errors


class _FakeTestTauRuntime:
    def __init__(self, state: dict[str, Any]) -> None:
        self.state = copy.deepcopy(state)
        self.state.setdefault("records", {})
        self.state.setdefault("notes", [])
        snapshot = copy.deepcopy(self.state)
        self.initial_sync_evidence = {
            "performed": False,
            "sync_count": 0,
            "steps": [],
            "sequence_sha256": canonical_sha256([]),
            "pre_state": snapshot,
            "post_state": copy.deepcopy(snapshot),
            "pre_state_sha256": canonical_sha256(snapshot),
            "post_state_sha256": canonical_sha256(snapshot),
            "state_diff": _state_diff(snapshot, snapshot),
        }
        self.last_sync_evidence: dict[str, Any] | None = None

    def call(self, tool_name: str, args: dict[str, Any]) -> Any:
        if tool_name == "get_record":
            record_id = _required_string(args, "id")
            return copy.deepcopy(self.state["records"].get(record_id))
        if tool_name == "update_record":
            record_id = _required_string(args, "id")
            patch = args.get("patch")
            if not isinstance(patch, dict):
                return {"error": "invalid_patch", "message": "patch must be an object"}
            record = self.state["records"].get(record_id)
            if not isinstance(record, dict):
                return {"error": "not_found", "message": f"record {record_id} not found"}
            record.update(copy.deepcopy(patch))
            return {"updated": True, "id": record_id, "record": copy.deepcopy(record)}
        if tool_name == "create_note":
            text = _required_string(args, "text")
            note = {"id": f"note-{len(self.state['notes']) + 1}", "text": text}
            self.state["notes"].append(note)
            return copy.deepcopy(note)
        if tool_name == "empty_search":
            return []
        if tool_name == "raise_tool_exception":
            raise RuntimeError(str(args.get("message") or "synthetic tool exception"))
        raise Tau3GroundedGenerationError(f"unknown replayable tool: {tool_name}")

    def tool_catalog(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_record",
                    "description": "TEST ONLY: read one fake record.",
                    "parameters": {
                        "type": "object",
                        "properties": {"id": {"type": "string"}},
                        "required": ["id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "empty_search",
                    "description": "TEST ONLY: return an empty result.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "update_record",
                    "description": "TEST ONLY: patch one fake record.",
                    "parameters": {
                        "type": "object",
                        "properties": {"id": {"type": "string"}, "patch": {"type": "object"}},
                        "required": ["id", "patch"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "raise_tool_exception",
                    "description": "TEST ONLY: raise an exception.",
                    "parameters": {
                        "type": "object",
                        "properties": {"message": {"type": "string"}},
                        "required": [],
                    },
                },
            },
        ]

    def system_prompt(self) -> None:
        return None

    def policy_sha256(self) -> str:
        return canonical_sha256("fake-test-policy")

    def tool_mutates_state(self, tool_name: str) -> bool:
        return tool_name in {"update_record", "create_note"}


def _full_environment_snapshot_errors(
    snapshot: Any,
    *,
    domain: str,
    context: str,
) -> list[str]:
    if not isinstance(snapshot, dict) or set(snapshot) != {"agent_db", "user_db"}:
        return [f"{context} must retain exactly agent_db and user_db"]
    errors: list[str] = []
    if not isinstance(snapshot.get("agent_db"), dict):
        errors.append(f"{context}.agent_db must be a complete object")
    user_state = snapshot.get("user_db")
    if domain == "telecom":
        if not isinstance(user_state, dict):
            errors.append(f"{context}.user_db must be a complete telecom object")
    elif user_state is not None and not isinstance(user_state, dict):
        errors.append(f"{context}.user_db must be an object or null")
    return errors


def _vendored_environment_state(environment: Any) -> dict[str, Any]:
    tools = environment.tools
    if tools is None or tools.db is None:
        raise Tau3GroundedGenerationError(
            "vendored Tau environment has no assistant database"
        )
    agent_state = _model_to_json(tools.db)
    user_tools = environment.user_tools
    user_state = None if user_tools is None else _model_to_json(user_tools.db)
    if not isinstance(agent_state, dict):
        raise Tau3GroundedGenerationError(
            "vendored Tau assistant DB is not an object"
        )
    if user_tools is not None and not isinstance(user_state, dict):
        raise Tau3GroundedGenerationError("vendored Tau user DB is not an object")
    return {"agent_db": agent_state, "user_db": user_state}


def _ordered_initial_sync_evidence(
    snapshots: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    if not snapshots:
        raise Tau3GroundedGenerationError(
            "task initialization must record at least one sync_tools call"
        )
    steps: list[dict[str, Any]] = []
    previous_step_sha256: str | None = None
    for ordinal, (before, after) in enumerate(snapshots):
        step = {
            "ordinal": ordinal,
            "performed": True,
            "previous_step_sha256": previous_step_sha256,
            "pre_state": copy.deepcopy(before),
            "post_state": copy.deepcopy(after),
            "pre_state_sha256": canonical_sha256(before),
            "post_state_sha256": canonical_sha256(after),
            "state_diff": _state_diff(before, after),
        }
        step["step_sha256"] = canonical_sha256(step)
        steps.append(step)
        previous_step_sha256 = step["step_sha256"]
    pre_state = copy.deepcopy(steps[0]["pre_state"])
    post_state = copy.deepcopy(steps[-1]["post_state"])
    return {
        "performed": True,
        "sync_count": len(steps),
        "steps": steps,
        "sequence_sha256": canonical_sha256(steps),
        "pre_state": pre_state,
        "post_state": post_state,
        "pre_state_sha256": canonical_sha256(pre_state),
        "post_state_sha256": canonical_sha256(post_state),
        "state_diff": _state_diff(pre_state, post_state),
    }


def _capture_physical_sync_sequence(
    environment: Any,
    operation: Any,
) -> dict[str, Any]:
    snapshots: list[tuple[dict[str, Any], dict[str, Any]]] = []
    original_sync_tools = environment.sync_tools

    def capture_sync_tools() -> Any:
        before_sync = copy.deepcopy(_vendored_environment_state(environment))
        try:
            return original_sync_tools()
        finally:
            snapshots.append(
                (
                    before_sync,
                    copy.deepcopy(_vendored_environment_state(environment)),
                )
            )

    environment.sync_tools = capture_sync_tools
    try:
        operation()
    finally:
        environment.sync_tools = original_sync_tools
    return _ordered_initial_sync_evidence(snapshots)


def _capture_official_factory_sync_sequence(
    environment_module: Any,
    environment_type: type[Any],
    factory_kwargs: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    snapshots: list[tuple[dict[str, Any], dict[str, Any]]] = []
    inherited_sync_tools = environment_type.sync_tools
    had_own_sync_tools = "sync_tools" in environment_type.__dict__
    own_sync_tools = environment_type.__dict__.get("sync_tools")

    def capture_sync_tools(instance: Any) -> Any:
        before_sync = copy.deepcopy(_vendored_environment_state(instance))
        try:
            return inherited_sync_tools(instance)
        finally:
            snapshots.append(
                (
                    before_sync,
                    copy.deepcopy(_vendored_environment_state(instance)),
                )
            )

    environment_type.sync_tools = capture_sync_tools
    try:
        environment = environment_module.get_environment(**factory_kwargs)
    finally:
        if had_own_sync_tools:
            environment_type.sync_tools = own_sync_tools
        else:
            delattr(environment_type, "sync_tools")
    if snapshots:
        evidence = _ordered_initial_sync_evidence(snapshots)
    else:
        snapshot = copy.deepcopy(_vendored_environment_state(environment))
        evidence = {
            "performed": False,
            "sync_count": 0,
            "steps": [],
            "sequence_sha256": canonical_sha256([]),
            "pre_state": snapshot,
            "post_state": copy.deepcopy(snapshot),
            "pre_state_sha256": canonical_sha256(snapshot),
            "post_state_sha256": canonical_sha256(snapshot),
            "state_diff": _state_diff(snapshot, snapshot),
        }
    return environment, evidence


def _official_runtime_contract(
    environment: Any,
    llm_agent_class: Any,
) -> tuple[tuple[Any, ...], list[dict[str, Any]], str, Any]:
    ordered_tools = tuple(environment.get_tools())
    if not ordered_tools:
        raise ValueError("environment ordered tool catalog is empty")
    catalog = [_canonical_value(tool.openai_schema) for tool in ordered_tools]
    policy = environment.get_policy()
    prompt = str(
        llm_agent_class(
            tools=list(ordered_tools),
            domain_policy=policy,
            llm="offline-contract-validation",
        ).system_prompt
    )
    return ordered_tools, catalog, prompt, policy


def _fixed_point_sequence_errors(
    evidence: dict[str, Any],
    terminal_state: dict[str, Any],
    context: str,
    *,
    require_physical_sync: bool = True,
) -> list[str]:
    errors: list[str] = []
    steps = evidence.get("steps")
    if not isinstance(steps, list):
        return [f"{context}.steps must be an ordered list"]
    if require_physical_sync and not steps:
        return [f"{context} must retain at least one physical sync step"]
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            errors.append(f"{context}.steps[{index}] must be an object")
            continue
        if step.get("pre_state") != terminal_state:
            errors.append(f"{context}.steps[{index}].pre_state is not the terminal fixed point")
        if step.get("post_state") != terminal_state:
            errors.append(f"{context}.steps[{index}].post_state is not the terminal fixed point")
    if evidence.get("pre_state") != terminal_state:
        errors.append(f"{context}.pre_state is not the terminal fixed point")
    if evidence.get("post_state") != terminal_state:
        errors.append(f"{context}.post_state is not the terminal fixed point")
    return errors


def _strict_set_state_preserving_independent_databases(
    environment: Any,
    *,
    domain: str,
    initialization_data: Any,
    initialization_actions: Any,
    message_history: list[Any],
    agent_db_type: type[Any],
    user_db_type: type[Any] | None,
) -> None:
    """Run official strict replay without permitting Tau's cross-schema DB alias."""

    if (
        domain != "telecom"
        or initialization_data is None
        or user_db_type is None
        or environment.user_tools is None
    ):
        environment.set_state(
            initialization_data,
            initialization_actions,
            message_history,
            strict=True,
        )
        return

    agent_tools = environment.tools
    user_tools = environment.user_tools
    original_agent_tools_type = type(agent_tools)
    original_user_tools_type = type(user_tools)
    blocked_cross_schema_assignments: list[str] = []

    class _IndependentAgentTools(original_agent_tools_type):  # type: ignore[misc, valid-type]
        def __setattr__(self, name: str, value: Any) -> None:
            if name == "db" and not isinstance(value, agent_db_type):
                blocked_cross_schema_assignments.append("agent")
                return
            super().__setattr__(name, value)

    class _IndependentUserTools(original_user_tools_type):  # type: ignore[misc, valid-type]
        def __setattr__(self, name: str, value: Any) -> None:
            if name == "db" and not isinstance(value, user_db_type):
                blocked_cross_schema_assignments.append("user")
                return
            super().__setattr__(name, value)

    agent_tools.__class__ = _IndependentAgentTools
    user_tools.__class__ = _IndependentUserTools
    try:
        environment.set_state(
            initialization_data,
            initialization_actions,
            message_history,
            strict=True,
        )
    finally:
        agent_tools.__class__ = original_agent_tools_type
        user_tools.__class__ = original_user_tools_type

    expected_blocks = []
    if initialization_data.agent_data is not None:
        expected_blocks.append("user")
    if initialization_data.user_data is not None:
        expected_blocks.append("agent")
    if blocked_cross_schema_assignments != expected_blocks:
        raise ValueError(
            "official telecom initialization-data alias guard did not replay exactly"
        )


def _assert_vendored_tau_module_origins(
    src: Path,
    modules: dict[str, Any],
) -> None:
    expected_root = src.resolve(strict=False)
    for name, module in modules.items():
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str) or not raw_path:
            raise Tau3GroundedGenerationError(
                f"vendored Tau module {name} has no inspectable filesystem origin"
            )
        actual_path = Path(raw_path).resolve(strict=False)
        if expected_root not in actual_path.parents:
            raise Tau3GroundedGenerationError(
                f"vendored Tau module {name} was imported outside the pinned checkout"
            )


class _VendoredTauRuntime:
    def __init__(
        self,
        *,
        domain: str,
        revision: str,
        state: dict[str, Any],
        repo: str | Path | None,
    ) -> None:
        repo_path = _resolve_tau_repo(repo)
        if not repo_path.is_dir():
            raise Tau3GroundedGenerationError(f"vendored Tau repository is unavailable: {repo_path}")
        actual_revision = _git(repo_path, "rev-parse", "HEAD")
        if actual_revision != revision:
            raise Tau3GroundedGenerationError(
                f"vendored Tau revision mismatch: expected {revision}, got {actual_revision}"
            )
        src = repo_path / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        _install_tau_import_shims()
        try:
            environment_module = importlib.import_module(f"tau2.domains.{domain}.environment")
            llm_agent_module = importlib.import_module("tau2.agent.llm_agent")
            tasks_module = importlib.import_module("tau2.data_model.tasks")
        except Exception as exc:
            raise Tau3GroundedGenerationError(
                f"cannot import vendored Tau {domain} environment: {exc}"
            ) from exc
        _assert_vendored_tau_module_origins(
            src,
            {
                f"tau2.domains.{domain}.environment": environment_module,
                "tau2.agent.llm_agent": llm_agent_module,
                "tau2.data_model.tasks": tasks_module,
            },
        )
        pre_sync_state = state.get("pre_sync")
        expected_post_sync_state = state.get("post_sync")
        task_initialization = state.get("task_initialization")
        if set(state) != {"task_initialization", "pre_sync", "post_sync"}:
            raise Tau3GroundedGenerationError(
                "candidate initial_state must contain exactly task_initialization, pre_sync, and post_sync"
            )
        snapshot_errors = []
        for label, snapshot in (
            ("pre_sync", pre_sync_state),
            ("post_sync", expected_post_sync_state),
        ):
            snapshot_errors.extend(
                _full_environment_snapshot_errors(
                    snapshot,
                    domain=domain,
                    context=f"candidate initial_state.{label}",
                )
            )
        if snapshot_errors:
            raise Tau3GroundedGenerationError("; ".join(snapshot_errors))

        try:
            self._llm_agent_class = llm_agent_module.LLMAgent
            self.last_sync_evidence: dict[str, Any] | None = None
            canonical_task_initialization = _canonical_value(task_initialization)
            task_initialization_sha256 = canonical_sha256(canonical_task_initialization)

            telecom_agent_db_type = None
            telecom_user_db_type = None
            if domain == "telecom":
                telecom_data_module = importlib.import_module(
                    "tau2.domains.telecom.data_model"
                )
                telecom_user_data_module = importlib.import_module(
                    "tau2.domains.telecom.user_data_model"
                )
                _assert_vendored_tau_module_origins(
                    src,
                    {
                        "tau2.domains.telecom.data_model": telecom_data_module,
                        "tau2.domains.telecom.user_data_model": telecom_user_data_module,
                    },
                )
                telecom_agent_db_type = telecom_data_module.TelecomDB
                telecom_user_db_type = telecom_user_data_module.TelecomUserDB

            def initialize_fresh_official_environment() -> dict[str, Any]:
                parsed_initialization = (
                    None
                    if task_initialization is None
                    else tasks_module.InitialState.model_validate(
                        copy.deepcopy(task_initialization)
                    )
                )
                if parsed_initialization is not None:
                    parsed_dump = _canonical_value(
                        parsed_initialization.model_dump(mode="json")
                    )
                    if parsed_dump != canonical_task_initialization:
                        raise ValueError(
                            "official InitialState does not round-trip canonically unchanged"
                        )
                environment = environment_module.get_environment()
                if environment.tools is None or environment.tools.db is None:
                    raise ValueError("environment has no assistant toolkit database")
                agent_db_type = type(environment.tools.db)
                user_db_type = (
                    None
                    if environment.user_tools is None
                    else type(environment.user_tools.db)
                )

                def apply_original_initial_state() -> None:
                    _strict_set_state_preserving_independent_databases(
                        environment,
                        domain=domain,
                        initialization_data=(
                            None
                            if parsed_initialization is None
                            else copy.deepcopy(parsed_initialization.initialization_data)
                        ),
                        initialization_actions=(
                            None
                            if parsed_initialization is None
                            else copy.deepcopy(parsed_initialization.initialization_actions)
                        ),
                        message_history=(
                            []
                            if parsed_initialization is None
                            or parsed_initialization.message_history is None
                            else copy.deepcopy(parsed_initialization.message_history)
                        ),
                        agent_db_type=agent_db_type,
                        user_db_type=user_db_type,
                    )

                sync_evidence = _capture_physical_sync_sequence(
                    environment,
                    apply_original_initial_state,
                )
                terminal_state = _vendored_environment_state(environment)
                ordered_tools, catalog, prompt, policy = _official_runtime_contract(
                    environment,
                    self._llm_agent_class,
                )
                if canonical_sha256(canonical_task_initialization) != task_initialization_sha256:
                    raise ValueError("official InitialState changed during strict replay")
                return {
                    "environment": environment,
                    "agent_db_type": agent_db_type,
                    "user_db_type": user_db_type,
                    "sync_evidence": sync_evidence,
                    "terminal_state": terminal_state,
                    "ordered_tools": ordered_tools,
                    "catalog": catalog,
                    "prompt": prompt,
                    "policy": policy,
                }

            authoritative = initialize_fresh_official_environment()
            replay = initialize_fresh_official_environment()
            for field in (
                "sync_evidence",
                "terminal_state",
                "catalog",
                "prompt",
                "policy",
            ):
                if _canonical_value(authoritative[field]) != _canonical_value(replay[field]):
                    raise ValueError(
                        f"second fresh official InitialState replay differs in {field}"
                    )

            environment = authoritative["environment"]
            terminal_state = _canonical_value(authoritative["terminal_state"])
            initial_sync_evidence = _canonical_value(authoritative["sync_evidence"])
            if _canonical_value(pre_sync_state) != initial_sync_evidence["pre_state"]:
                raise ValueError("task initialization pre-sync state does not replay")
            if _canonical_value(expected_post_sync_state) != terminal_state:
                raise ValueError("task initialization terminal state does not replay")
            if initial_sync_evidence["post_state"] != terminal_state:
                raise ValueError(
                    "task initialization final sync post-state is not the terminal state"
                )

            agent_db_type = authoritative["agent_db_type"]
            user_db_type = authoritative["user_db_type"]
            if domain == "telecom":
                if agent_db_type is not telecom_agent_db_type:
                    raise ValueError("official telecom assistant DB type changed before replay")
                if user_db_type is not telecom_user_db_type:
                    raise ValueError("official telecom user DB type changed before replay")
                if (
                    not isinstance(environment.tools.db, telecom_agent_db_type)
                    or environment.user_tools is None
                    or not isinstance(environment.user_tools.db, telecom_user_db_type)
                    or environment.tools.db is environment.user_tools.db
                ):
                    raise ValueError(
                        "official telecom initialization aliased or discarded typed databases"
                    )
                agent_db_type = telecom_agent_db_type
                user_db_type = telecom_user_db_type

            typed_agent_db = agent_db_type.model_validate(
                copy.deepcopy(terminal_state["agent_db"])
            )
            materialized_kwargs: dict[str, Any] = {"db": typed_agent_db}
            typed_user_db = None
            if user_db_type is None:
                if terminal_state["user_db"] is not None:
                    raise ValueError(
                        "terminal user state is present without an official user toolkit"
                    )
            else:
                if not isinstance(terminal_state["user_db"], dict):
                    raise ValueError("terminal user state must be an object")
                typed_user_db = user_db_type.model_validate(
                    copy.deepcopy(terminal_state["user_db"])
                )
                if typed_agent_db is typed_user_db:
                    raise ValueError("typed assistant and user databases must be independent")
                materialized_kwargs["user_db"] = typed_user_db

            if domain == "telecom":
                authoritative_domain = environment.get_domain_name()
                policy_type = {
                    "telecom": "manual",
                    "telecom-workflow": "workflow",
                }.get(authoritative_domain)
                if policy_type is None:
                    raise ValueError("official telecom policy type cannot be derived")
                materialized_kwargs["policy_type"] = policy_type

            materialized_environment, constructor_sync_evidence = (
                _capture_official_factory_sync_sequence(
                    environment_module,
                    type(environment),
                    materialized_kwargs,
                )
            )
            constructor_errors = _fixed_point_sequence_errors(
                constructor_sync_evidence,
                terminal_state,
                "typed factory constructor sync",
                require_physical_sync=False,
            )
            if constructor_errors:
                raise ValueError("; ".join(constructor_errors))
            if _vendored_environment_state(materialized_environment) != terminal_state:
                raise ValueError(
                    "typed assistant/user state does not replay through the official factory"
                )
            if typed_user_db is not None and (
                materialized_environment.user_tools is None
                or materialized_environment.tools.db
                is materialized_environment.user_tools.db
            ):
                raise ValueError("official typed factory aliased assistant and user databases")

            materialized_tools, materialized_catalog, materialized_prompt, materialized_policy = (
                _official_runtime_contract(
                    materialized_environment,
                    self._llm_agent_class,
                )
            )
            if materialized_catalog != authoritative["catalog"]:
                raise ValueError("typed state ordered tool catalog does not replay")
            if (
                type(materialized_policy) is not type(authoritative["policy"])
                or materialized_policy != authoritative["policy"]
            ):
                raise ValueError("typed state policy value or type does not replay")
            if materialized_prompt != authoritative["prompt"]:
                raise ValueError("typed state system prompt does not replay")

            materialized_sync_evidence = _capture_physical_sync_sequence(
                materialized_environment,
                lambda: materialized_environment.set_state(
                    None,
                    None,
                    [],
                    strict=True,
                ),
            )
            fixed_point_errors = _fixed_point_sequence_errors(
                materialized_sync_evidence,
                terminal_state,
                "typed final set_state sync",
            )
            if fixed_point_errors:
                raise ValueError("; ".join(fixed_point_errors))
            if _vendored_environment_state(materialized_environment) != terminal_state:
                raise ValueError("typed state final replay does not round-trip")
            if typed_user_db is not None and (
                materialized_environment.user_tools is None
                or materialized_environment.tools.db
                is materialized_environment.user_tools.db
            ):
                raise ValueError("typed final replay aliased assistant and user databases")

            self.environment = environment
            self._ordered_tools = authoritative["ordered_tools"]
            self.initial_sync_evidence = initial_sync_evidence
            self.original_initial_state_replay_evidence = _canonical_value(
                replay["sync_evidence"]
            )
            self.materialized_constructor_sync_evidence = constructor_sync_evidence
            self.materialized_state_replay_evidence = materialized_sync_evidence
            self.task_initialization_sha256 = task_initialization_sha256
        except Exception as exc:
            raise Tau3GroundedGenerationError(
                f"cannot instantiate vendored Tau {domain} full environment state: {exc}"
            ) from exc

    @property
    def state(self) -> dict[str, Any]:
        return _vendored_environment_state(self.environment)

    def call(self, tool_name: str, args: dict[str, Any]) -> Any:
        before = copy.deepcopy(self.state)
        self.last_sync_evidence = None
        pre_sync: dict[str, Any] | None = None
        try:
            result = self.environment.make_tool_call(
                tool_name,
                requestor="assistant",
                **copy.deepcopy(args),
            )
            pre_sync = copy.deepcopy(self.state)
            try:
                self.environment.sync_tools()
            except Exception:
                current = copy.deepcopy(self.state)
                self.last_sync_evidence = {
                    "performed": True,
                    "pre_state": pre_sync,
                    "post_state": current,
                    "pre_state_sha256": canonical_sha256(pre_sync),
                    "post_state_sha256": canonical_sha256(current),
                    "state_diff": _state_diff(pre_sync, current),
                    "call_pre_state_sha256": canonical_sha256(before),
                }
                raise
            post_sync = copy.deepcopy(self.state)
        except Exception:
            if self.last_sync_evidence is None:
                current = copy.deepcopy(self.state)
                self.last_sync_evidence = {
                    "performed": False,
                    "pre_state": current,
                    "post_state": current,
                    "pre_state_sha256": canonical_sha256(current),
                    "post_state_sha256": canonical_sha256(current),
                    "state_diff": _state_diff(current, current),
                }
            raise
        self.last_sync_evidence = {
            "performed": True,
            "pre_state": pre_sync,
            "post_state": post_sync,
            "pre_state_sha256": canonical_sha256(pre_sync),
            "post_state_sha256": canonical_sha256(post_sync),
            "state_diff": _state_diff(pre_sync, post_sync),
            "call_pre_state_sha256": canonical_sha256(before),
        }
        return _model_to_json(result)

    def tool_catalog(self) -> list[dict[str, Any]]:
        return [_canonical_value(tool.openai_schema) for tool in self._ordered_tools]

    def system_prompt(self) -> str:
        agent = self._llm_agent_class(
            tools=list(self._ordered_tools),
            domain_policy=self.environment.get_policy(),
            llm="offline-contract-validation",
        )
        return str(agent.system_prompt)

    def policy_sha256(self) -> str:
        return canonical_sha256(self.environment.get_policy())

    def tool_mutates_state(self, tool_name: str) -> bool:
        tools = self.environment.tools
        if tools is None:
            raise Tau3GroundedGenerationError("vendored Tau environment has no assistant tools")
        return bool(tools.tool_mutates_state(tool_name))


def _runtime_for_scenario(payload: dict[str, Any]) -> Any:
    runtime_family = str(payload.get("runtime_family") or "")
    if runtime_family == FAKE_TEST_RUNTIME_FAMILY:
        return _FakeTestTauRuntime(payload["initial_state"])
    if _runtime_is_vendored(runtime_family):
        return _VendoredTauRuntime(
            domain=str(payload["domain"]),
            revision=runtime_family.removeprefix(VENDORED_RUNTIME_PREFIX),
            state=payload["initial_state"],
            repo=payload.get("tau_repo"),
        )
    raise Tau3GroundedGenerationError(f"unsupported runtime_family: {runtime_family}")


def _is_replayable_runtime_family(runtime_family: str) -> bool:
    return runtime_family == FAKE_TEST_RUNTIME_FAMILY or _runtime_is_vendored(runtime_family)


def _runtime_is_vendored(runtime_family: str) -> bool:
    revision = runtime_family.removeprefix(VENDORED_RUNTIME_PREFIX)
    return runtime_family.startswith(VENDORED_RUNTIME_PREFIX) and HEX40_RE.fullmatch(revision) is not None


def _runtime_is_candidate_eligible(runtime_family: Any) -> bool:
    return isinstance(runtime_family, str) and _runtime_is_vendored(runtime_family)


def _is_mutation_tool(tool_name: str) -> bool:
    return tool_name.startswith(MUTATION_PREFIXES)


def _model_to_json(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (date, time)):
        return value.isoformat()
    if isinstance(value, Enum):
        return _model_to_json(value.value)
    if isinstance(value, dict):
        return {str(key): _model_to_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_model_to_json(item) for item in value]
    if isinstance(value, tuple):
        return [_model_to_json(item) for item in value]
    return value


def _install_tau_import_shims() -> None:
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    if os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP") != "True":
        raise Tau3GroundedGenerationError(
            "LITELLM_LOCAL_MODEL_COST_MAP must be True before Tau imports"
        )
    if importlib.util.find_spec("loguru") is not None or "loguru" in sys.modules:
        return

    class _NoopLogger:
        def __getattr__(self, _name: str) -> Any:
            return lambda *args, **kwargs: None

    module = types.ModuleType("loguru")
    module.logger = _NoopLogger()
    sys.modules["loguru"] = module


def _tau_repo_record(payload: dict[str, Any]) -> dict[str, Any]:
    runtime_family = str(payload.get("runtime_family") or "")
    if not _runtime_is_vendored(runtime_family):
        return {
            "path": None,
            "revision": None,
            "tree_sha256": None,
            "candidate_eligible": False,
        }
    repo = _resolve_tau_repo(payload.get("tau_repo"))
    revision = runtime_family.removeprefix(VENDORED_RUNTIME_PREFIX)
    return {
        "path": _repo_relative_path(repo),
        "revision": revision,
        "tree_sha256": _tau_repo_tree_sha256(repo, revision),
        "candidate_eligible": True,
    }


def _tau_repo_tree_sha256(repo: Path, revision: str) -> str:
    tree = _git(repo, "rev-parse", f"{revision}^{{tree}}")
    return hashlib.sha256(tree.encode("utf-8")).hexdigest()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_tau_repo(repo: str | Path | None) -> Path:
    root = _project_root()
    path = Path(repo or "local/tau3/repository")
    if not path.is_absolute():
        path = root / path
    try:
        resolved = path.resolve(strict=False)
        root_resolved = root.resolve(strict=True)
    except OSError as exc:
        raise Tau3GroundedGenerationError(f"cannot resolve Tau repository path: {exc}") from exc
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise Tau3GroundedGenerationError("Tau repository path must remain under the project root")
    return resolved


def _repo_relative_path(repo: Path) -> str:
    return repo.resolve(strict=False).relative_to(_project_root()).as_posix()


def _git(repo: Path, *args: str) -> str:
    import subprocess

    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "git command failed"
        raise Tau3GroundedGenerationError(detail)
    return completed.stdout.strip()


def _normalize_coverage_profile_id(profile_id: str) -> str:
    if profile_id not in {
        LEGACY_COVERAGE_PROFILE_ID,
        SCALED_COVERAGE_PROFILE_ID,
    }:
        raise Tau3GroundedGenerationError(
            f"unregistered coverage profile: {profile_id!r}"
        )
    return profile_id


def _coverage_profile_record(profile_id: str) -> dict[str, Any]:
    profile_id = _normalize_coverage_profile_id(profile_id)
    if profile_id == LEGACY_COVERAGE_PROFILE_ID:
        contract: dict[str, Any] = {
            "aggregate_contract": "source_family_and_behavior_presence_v1",
            "train_family_minimum": TRAIN_FAMILY_MIN,
            "validation_family_minimum": VALIDATION_FAMILY_MIN,
            "behavior_vocabulary": list(BEHAVIORS),
            "implicit_when_manifest_profile_absent": True,
        }
    else:
        contract = {
            "aggregate_contract": SCALED_COVERAGE_SCHEMA_VERSION,
            "selection_receipt_schema_version": SCALED_SELECTION_RECEIPT_SCHEMA_VERSION,
            "selection_algorithm": SCALED_SELECTION_ALGORITHM,
            "rank_tie_breaker_contract": SCALED_RANK_TIE_BREAKER_CONTRACT,
            "selection_stratum_schema_version": SELECTION_STRATUM_SCHEMA_VERSION,
            "generation_stratum_schema_version": GENERATION_STRATUM_SCHEMA_VERSION,
            "generated_family_schema_version": GENERATED_FAMILY_SCHEMA_VERSION,
            "tool_target_minimums": dict(SCALED_TOOL_TARGET_MINIMUMS),
            "tool_argument_minimums": dict(SCALED_TOOL_ARGUMENT_MINIMUMS),
            "tool_exemption_schema_version": TOOL_EXEMPTION_SCHEMA_VERSION,
            "tool_exemption_review_schema_version": (
                TOOL_EXEMPTION_REVIEW_SCHEMA_VERSION
            ),
            "behavior_target_minimums": dict(SCALED_BEHAVIOR_TARGET_MINIMUMS),
            "behavior_family_minimums": dict(SCALED_BEHAVIOR_FAMILY_MINIMUMS),
            "behavior_vocabulary": list(BEHAVIORS),
            "correction_behaviors": sorted(NEGATIVE_CORRECTION_BEHAVIORS),
            "telecom_train_example_minimum_fraction": {
                "numerator": 1,
                "denominator": 4,
            },
            "domain_token_fraction_bounds": {
                "minimum": {"numerator": 1, "denominator": 4},
                "maximum": {"numerator": 2, "denominator": 5},
            },
            "domain_token_ratio_maximum": {"numerator": 8, "denominator": 5},
            "positive_target_duplication_maximum": {
                "numerator": 1,
                "denominator": 5,
            },
            "training_example_duplication_maximum": {
                "numerator": 1,
                "denominator": 100,
            },
            "dominance_maximum": {"numerator": 1, "denominator": 5},
            "dominance_dimensions": [
                "generated_family_by_behavior",
                "target_tool",
                "canonical_argument_payload_by_tool",
                "synthetic_mutation_family",
            ],
            "hash_disjointness_dimensions": [
                "task_sha256",
                "source_family_sha256",
                "prompt_sha256",
            ],
            "hash_evidence_complete_per_row": True,
            "token_count_algorithm": TOKEN_COUNT_ALGORITHM,
            "token_count_canonicalization": "sorted_compact_ascii_json",
            "token_count_unit": "utf8_byte",
            "maximum_blocker_records": 4096,
            "dependency_free": True,
            "development_maps_only_to_internal_validation": True,
        }
    return {
        "schema_version": COVERAGE_PROFILE_SCHEMA_VERSION,
        "profile_id": profile_id,
        "contract_sha256": canonical_sha256(contract),
    }


def _coverage_profile_from_manifest(
    manifest: dict[str, Any],
    errors: list[str],
) -> str:
    value = manifest.get("coverage_profile")
    if value is None:
        return LEGACY_COVERAGE_PROFILE_ID
    if not isinstance(value, dict):
        errors.append("E_COVERAGE_PROFILE_INVALID")
        return SCALED_COVERAGE_PROFILE_ID
    profile_id = value.get("profile_id")
    if profile_id not in {
        LEGACY_COVERAGE_PROFILE_ID,
        SCALED_COVERAGE_PROFILE_ID,
    }:
        errors.append("E_COVERAGE_PROFILE_UNREGISTERED")
        return SCALED_COVERAGE_PROFILE_ID
    expected = _coverage_profile_record(str(profile_id))
    if value != expected:
        errors.append("E_COVERAGE_PROFILE_HASH_BINDING")
    if profile_id == LEGACY_COVERAGE_PROFILE_ID:
        errors.append("E_LEGACY_PROFILE_MUST_BE_IMPLICIT")
    return str(profile_id)


def _selection_profile_errors(
    rows_by_split: dict[str, list[dict[str, Any]]],
    profile_id: str,
) -> list[str]:
    versions = {
        str(
            _dict(_dict(row.get("metadata")).get("selection_receipt")).get(
                "schema_version"
            )
            or ""
        )
        for split in SPLITS
        for row in rows_by_split.get(split, [])
    }
    if (
        profile_id == LEGACY_COVERAGE_PROFILE_ID
        and SCALED_SELECTION_RECEIPT_SCHEMA_VERSION in versions
    ):
        return ["E_SCALE_COVERAGE_PROFILE_DOWNGRADE"]
    return []


def _training_handoff_errors(value: Any, context: str) -> list[str]:
    if not isinstance(value, dict):
        return ["E_SCALE_TRAINING_HANDOFF_MISSING"]
    errors: list[str] = []
    if set(value) != {
        "schema_version",
        "tokenizer",
        "chat_template",
        "context_budget",
        "handoff_sha256",
    }:
        errors.append("E_SCALE_TRAINING_HANDOFF_FIELDS")
    if value.get("schema_version") != TRAINING_HANDOFF_SCHEMA_VERSION:
        errors.append("E_SCALE_TRAINING_HANDOFF_VERSION")
    tokenizer = value.get("tokenizer")
    if not isinstance(tokenizer, dict):
        errors.append("E_SCALE_TOKENIZER_IDENTITY_MISSING")
        tokenizer = {}
    if set(tokenizer) != {"identifier", "revision", "identity_sha256"}:
        errors.append("E_SCALE_TOKENIZER_IDENTITY_FIELDS")
    for field in ("identifier", "revision"):
        if not isinstance(tokenizer.get(field), str) or not tokenizer[field]:
            errors.append("E_SCALE_TOKENIZER_IDENTITY_VALUE")
    expected_tokenizer_sha = canonical_sha256(
        {
            "identifier": tokenizer.get("identifier"),
            "revision": tokenizer.get("revision"),
        }
    )
    if tokenizer.get("identity_sha256") != expected_tokenizer_sha:
        errors.append("E_SCALE_TOKENIZER_IDENTITY_HASH")
    chat_template = value.get("chat_template")
    if not isinstance(chat_template, dict):
        errors.append("E_SCALE_CHAT_TEMPLATE_IDENTITY_MISSING")
        chat_template = {}
    if set(chat_template) != {"identifier", "sha256"}:
        errors.append("E_SCALE_CHAT_TEMPLATE_IDENTITY_FIELDS")
    if not isinstance(chat_template.get("identifier"), str) or not chat_template.get(
        "identifier"
    ):
        errors.append("E_SCALE_CHAT_TEMPLATE_IDENTITY_VALUE")
    if not SHA256_RE.fullmatch(str(chat_template.get("sha256") or "")):
        errors.append("E_SCALE_CHAT_TEMPLATE_IDENTITY_HASH")
    context_budget = value.get("context_budget")
    if type(context_budget) is not int or context_budget < 1:
        errors.append("E_SCALE_TRAINING_HANDOFF_CONTEXT_BUDGET")
    expected_handoff_sha = canonical_sha256(
        {key: item for key, item in value.items() if key != "handoff_sha256"}
    )
    if value.get("handoff_sha256") != expected_handoff_sha:
        errors.append("E_SCALE_TRAINING_HANDOFF_HASH")
    return sorted(set(errors))


def _split_hash_evidence(
    rows_by_split: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    fields = {
        "task_sha256": "task_sha256",
        "source_family_sha256": "source_family_sha256",
        "prompt_sha256": "prompt_sha256",
    }
    dimensions: dict[str, Any] = {}
    for label, receipt_field in fields.items():
        values: dict[str, set[str]] = {split: set() for split in SPLITS}
        valid_claim_counts = {split: 0 for split in SPLITS}
        row_counts = {
            split: len(rows_by_split.get(split, [])) for split in SPLITS
        }
        for split in SPLITS:
            for row in rows_by_split.get(split, []):
                metadata = _dict(row.get("metadata"))
                receipt = _dict(metadata.get("selection_receipt"))
                digest = receipt.get(receipt_field)
                if not SHA256_RE.fullmatch(str(digest or "")):
                    continue
                if (
                    label == "source_family_sha256"
                    and digest != metadata.get("source_family_id")
                ):
                    continue
                valid_claim_counts[split] += 1
                values[split].add(str(digest))
        intersection = values["train"] & values["validation"]
        dimensions[label] = {
            "train_row_count": row_counts["train"],
            "validation_row_count": row_counts["validation"],
            "train_valid_claim_count": valid_claim_counts["train"],
            "validation_valid_claim_count": valid_claim_counts["validation"],
            "train_unique_count": len(values["train"]),
            "validation_unique_count": len(values["validation"]),
            "train_set_sha256": canonical_sha256(sorted(values["train"])),
            "validation_set_sha256": canonical_sha256(sorted(values["validation"])),
            "intersection_count": len(intersection),
            "intersection_set_sha256": canonical_sha256(sorted(intersection)),
            "disjoint": not intersection,
            "complete": all(
                valid_claim_counts[split] == row_counts[split]
                for split in SPLITS
            ),
        }
    record: dict[str, Any] = {
        "schema_version": "hfr.tau3_split_hash_evidence.v1",
        "train_source_split": "train",
        "validation_source_split": "development",
        "development_maps_only_to_internal_validation": True,
        "dimensions": dimensions,
    }
    record["canonical_sha256"] = canonical_sha256(record)
    return record


def _scaled_hash_disjointness_errors(
    rows_by_split: dict[str, list[dict[str, Any]]],
) -> list[str]:
    evidence = _split_hash_evidence(rows_by_split)
    errors = []
    error_ids = {
        "task_sha256": "E_SCALE_TASK_HASH_OVERLAP",
        "source_family_sha256": "E_SCALE_SOURCE_FAMILY_HASH_OVERLAP",
        "prompt_sha256": "E_SCALE_PROMPT_HASH_OVERLAP",
    }
    incomplete_error_ids = {
        "task_sha256": "E_SCALE_TASK_HASH_EVIDENCE_INCOMPLETE",
        "source_family_sha256": "E_SCALE_SOURCE_FAMILY_HASH_EVIDENCE_INCOMPLETE",
        "prompt_sha256": "E_SCALE_PROMPT_HASH_EVIDENCE_INCOMPLETE",
    }
    for dimension, record in _dict(evidence.get("dimensions")).items():
        if _dict(record).get("complete") is not True:
            errors.append(
                incomplete_error_ids.get(
                    dimension,
                    "E_SCALE_SPLIT_HASH_EVIDENCE_INCOMPLETE",
                )
            )
        if _dict(record).get("disjoint") is not True:
            errors.append(error_ids.get(dimension, "E_SCALE_SPLIT_HASH_OVERLAP"))
    return sorted(set(errors))


def _scaled_selection_claim_errors(
    rows_by_split: dict[str, list[dict[str, Any]]],
) -> list[str]:
    errors: set[str] = set()
    rank_tasks: dict[tuple[str, int], set[str]] = {}
    rank_ordinals: dict[str, set[int]] = {}
    variant_ordinals: dict[tuple[str, int, str, str], list[int]] = {}
    claims: set[tuple[str, int, str, str, int]] = set()
    generated_bindings: dict[str, tuple[Any, ...]] = {}
    for split in SPLITS:
        for row in rows_by_split.get(split, []):
            metadata = _dict(row.get("metadata"))
            receipt = _dict(metadata.get("selection_receipt"))
            if receipt.get("schema_version") != SCALED_SELECTION_RECEIPT_SCHEMA_VERSION:
                errors.add("E_SCALE_SELECTION_RECEIPT_VERSION")
                continue
            expected_source = "train" if split == "train" else "development"
            if metadata.get("split") != split or receipt.get(
                "mapped_grounded_split"
            ) != split:
                errors.add("E_SCALE_SELECTION_ROW_SPLIT_BINDING")
            if receipt.get("source") != expected_source:
                errors.add("E_SCALE_SELECTION_SOURCE_SPLIT_BINDING")
            if metadata.get("domain") not in DOMAINS:
                errors.add("E_SCALE_SELECTION_DOMAIN_BINDING")
            stratum_sha = str(receipt.get("selection_stratum_sha256") or "")
            generation_stratum_sha = str(
                receipt.get("generation_stratum_sha256") or ""
            )
            recipe_sha = str(receipt.get("generation_recipe_sha256") or "")
            rank = receipt.get("rank_ordinal")
            variant = receipt.get("generation_variant_ordinal")
            if not SHA256_RE.fullmatch(stratum_sha) or type(rank) is not int or rank < 0:
                errors.add("E_SCALE_SELECTION_ORDINAL_INVALID")
                continue
            if type(variant) is not int or variant < 0:
                errors.add("E_SCALE_VARIANT_ORDINAL_INVALID")
                continue
            task_sha = str(receipt.get("task_sha256") or "")
            if not SHA256_RE.fullmatch(task_sha):
                errors.add("E_SCALE_SELECTION_TASK_HASH_INVALID")
            if not SHA256_RE.fullmatch(generation_stratum_sha) or not SHA256_RE.fullmatch(
                recipe_sha
            ):
                errors.add("E_SCALE_GENERATION_STRATUM_RECIPE_INVALID")
            rank_tasks.setdefault((stratum_sha, rank), set()).add(task_sha)
            rank_ordinals.setdefault(stratum_sha, set()).add(rank)
            variant_key = (stratum_sha, rank, generation_stratum_sha, recipe_sha)
            variant_ordinals.setdefault(variant_key, []).append(variant)
            claim = variant_key + (variant,)
            if claim in claims:
                errors.add("E_SCALE_SELECTION_DUPLICATE_ORDINAL")
            claims.add(claim)
            generated_id = str(receipt.get("generated_family_identifier") or "")
            if not SHA256_RE.fullmatch(generated_id):
                errors.add("E_SCALE_GENERATED_FAMILY_IDENTIFIER_INVALID")
            binding = (
                receipt.get("source_family_sha256"),
                receipt.get("source"),
                receipt.get("mapped_grounded_split"),
                metadata.get("domain"),
                generation_stratum_sha,
                recipe_sha,
                variant,
            )
            prior = generated_bindings.setdefault(generated_id, binding)
            if prior != binding:
                errors.add("E_SCALE_GENERATED_FAMILY_REUSE")
    for tasks in rank_tasks.values():
        if len(tasks) != 1:
            errors.add("E_SCALE_SELECTION_ORDINAL_TASK_COLLISION")
    for ordinals in rank_ordinals.values():
        if ordinals and ordinals != set(range(max(ordinals) + 1)):
            errors.add("E_SCALE_SELECTION_SKIPPED_ORDINAL")
    for variants in variant_ordinals.values():
        unique = set(variants)
        if len(unique) != len(variants):
            errors.add("E_SCALE_VARIANT_DUPLICATE_ORDINAL")
        if unique and unique != set(range(max(unique) + 1)):
            errors.add("E_SCALE_VARIANT_SKIPPED_ORDINAL")
    return sorted(errors)


def _coverage(
    rows_by_split: dict[str, list[dict[str, Any]]],
    *,
    profile_id: str = LEGACY_COVERAGE_PROFILE_ID,
    training_handoff: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile_id = _normalize_coverage_profile_id(profile_id)
    if profile_id == SCALED_COVERAGE_PROFILE_ID:
        return _scaled_coverage(rows_by_split, training_handoff=training_handoff)
    return _legacy_coverage(rows_by_split)


def _legacy_coverage(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    blockers: list[str] = []
    by_split: dict[str, Any] = {}
    for split in SPLITS:
        required_families = TRAIN_FAMILY_MIN if split == "train" else VALIDATION_FAMILY_MIN
        by_split[split] = {}
        for domain in DOMAINS:
            rows = [row for row in rows_by_split.get(split, []) if _dict(row.get("metadata")).get("domain") == domain]
            unsupported = [
                str(_dict(row.get("metadata")).get("runtime_family") or "")
                for row in rows
                if not _runtime_is_candidate_eligible(_dict(row.get("metadata")).get("runtime_family"))
            ]
            families = sorted({str(_dict(row.get("metadata")).get("source_family_id")) for row in rows})
            behaviors = sorted(
                {
                    behavior
                    for row in rows
                    for behavior in _dict(row.get("metadata")).get("behaviors", [])
                }
            )
            missing_behaviors = [behavior for behavior in BEHAVIORS if behavior not in behaviors]
            if len(families) < required_families:
                blockers.append(
                    f"{split}.{domain} has {len(families)} source families; requires at least {required_families}"
                )
            if missing_behaviors:
                blockers.append(f"{split}.{domain} missing rubric behaviors: {', '.join(missing_behaviors)}")
            if unsupported:
                blockers.append(
                    f"{split}.{domain} has {len(unsupported)} rows without candidate-eligible vendored Tau replay"
                )
            by_split[split][domain] = {
                "row_count": len(rows),
                "source_family_count": len(families),
                "behaviors": behaviors,
                "missing_behaviors": missing_behaviors,
                "candidate_eligible_runtime": not unsupported,
            }
    return {"passed": not blockers, "blockers": blockers, "by_split": by_split}


def _canonical_target_token_count(canonical_target: dict[str, Any]) -> int:
    canonical_json = json.dumps(
        _canonical_value(canonical_target),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return len(canonical_json)


def _coverage_target_is_admitted(target: Any) -> bool:
    if not isinstance(target, dict) or target.get("masked") is not False:
        return False
    if target.get("behavior") not in BEHAVIORS:
        return False
    canonical = target.get("canonical_target")
    if not isinstance(canonical, dict):
        return False
    if _target_shape_errors(canonical, "coverage"):
        return False
    return target.get("canonical_target_sha256") == canonical_sha256(canonical)


def _policy_review_is_bound(target: dict[str, Any], *, allowed: bool) -> bool:
    review = target.get("policy_review")
    if not isinstance(review, dict):
        return False
    expected_sha = canonical_sha256(
        {key: item for key, item in review.items() if key != "review_receipt_sha256"}
    )
    return (
        review.get("allowed") is allowed
        and review.get("canonical_target_sha256")
        == target.get("canonical_target_sha256")
        and review.get("parent_turn_ordinal") == target.get("parent_turn_ordinal")
        and review.get("parent_assistant_decision_ordinal")
        == target.get("parent_assistant_decision_ordinal")
        and review.get("review_receipt_sha256") == expected_sha
    )


def _has_reviewed_negative_context(
    row: dict[str, Any],
    safe_target: dict[str, Any],
) -> bool:
    behavior = safe_target.get("behavior")
    expected_negative = NEGATIVE_CORRECTION_BEHAVIORS.get(str(behavior))
    if expected_negative is None or not _policy_review_is_bound(safe_target, allowed=True):
        return False
    safe_decision = safe_target.get("parent_assistant_decision_ordinal")
    targets = row.get("training_targets")
    if not isinstance(targets, list):
        return False
    matches = []
    for target in targets:
        if not isinstance(target, dict) or target.get("masked") is not True:
            continue
        if (
            target.get("behavior") == behavior
            and target.get("negative_behavior") == expected_negative
            and target.get("reviewed") is True
            and target.get("safe_correction_decision_ordinal") == safe_decision
            and type(target.get("parent_assistant_decision_ordinal")) is int
            and type(safe_decision) is int
            and target["parent_assistant_decision_ordinal"] < safe_decision
            and target.get("canonical_target_sha256")
            != safe_target.get("canonical_target_sha256")
            and _policy_review_is_bound(target, allowed=False)
        ):
            matches.append(target)
    return len(matches) == 1


def _training_example_sha256(row: dict[str, Any]) -> str:
    trajectory = _dict(row.get("trajectory"))
    return canonical_sha256(
        {
            "system_prompt": trajectory.get("system_prompt"),
            "tool_catalog": row.get("tool_catalog"),
            "turns": trajectory.get("turns"),
            "training_targets": row.get("training_targets"),
        }
    )


def _fraction_record(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "decimal": 0.0 if denominator == 0 else round(numerator / denominator, 12),
    }


def _meets_minimum(observed: int, minimum: int) -> bool:
    return observed >= minimum


def _fraction_at_least(
    numerator: int,
    denominator: int,
    minimum_numerator: int,
    minimum_denominator: int,
) -> bool:
    return (
        denominator > 0
        and numerator * minimum_denominator
        >= denominator * minimum_numerator
    )


def _fraction_at_most(
    numerator: int,
    denominator: int,
    maximum_numerator: int,
    maximum_denominator: int,
) -> bool:
    return (
        denominator > 0
        and numerator * maximum_denominator
        <= denominator * maximum_numerator
    )


def _ratio_at_most(
    largest: int,
    smallest: int,
    maximum_numerator: int,
    maximum_denominator: int,
) -> bool:
    return (
        smallest > 0
        and largest * maximum_denominator
        <= smallest * maximum_numerator
    )


def _duplication_record(values: list[str]) -> dict[str, Any]:
    total = len(values)
    unique = len(set(values))
    duplicate_excess = total - unique
    return {
        "total": total,
        "unique": unique,
        "duplicate_excess": duplicate_excess,
        "fraction": _fraction_record(duplicate_excess, total),
    }


def _dominance_record(counter: Counter[str]) -> dict[str, Any]:
    total = sum(counter.values())
    if not counter:
        return {
            "applicable": False,
            "total": 0,
            "unique_identifier_count": 0,
            "maximum_count": 0,
            "maximum_share": _fraction_record(0, 0),
            "dominant_identifier_sha256": None,
            "passed": True,
        }
    dominant, maximum = min(
        counter.items(),
        key=lambda item: (-item[1], canonical_sha256(item[0])),
    )
    return {
        "applicable": True,
        "total": total,
        "unique_identifier_count": len(counter),
        "maximum_count": maximum,
        "maximum_share": _fraction_record(maximum, total),
        "dominant_identifier_sha256": (
            dominant if SHA256_RE.fullmatch(dominant) else canonical_sha256(dominant)
        ),
        "passed": _fraction_at_most(maximum, total, 1, 5),
    }


def _catalog_tool_names(tool_catalog: Any) -> list[str]:
    if not isinstance(tool_catalog, list):
        return []
    names: set[str] = set()
    for tool in tool_catalog:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if not isinstance(name, str):
            name = _dict(tool.get("function")).get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return sorted(names)


def _canonical_argument_template_sha256(arguments: dict[str, Any]) -> str:
    return canonical_sha256(arguments)


def _argument_shape(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _argument_shape(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_argument_shape(item) for item in value]
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return "string"


def _synthetic_mutation_family_sha256(
    domain: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> str:
    return canonical_sha256(
        {
            "schema_version": "hfr.tau3_synthetic_mutation_family.v1",
            "domain": domain,
            "tool_name": tool_name,
            "canonical_argument_shape": _argument_shape(arguments),
        }
    )


def _scaled_tool_exemption_errors(
    record: Any,
    *,
    row: dict[str, Any],
    tool_name: str,
) -> list[str]:
    if not isinstance(record, dict):
        return ["E_SCALE_TOOL_EXEMPTION_INVALID"]
    expected_keys = {
        "schema_version",
        "domain",
        "tool_name",
        "reason",
        "scope",
        "reviewer",
        "reviewer_artifact_sha256",
        "reviewer_inference_receipt_sha256",
        "reviewer_model",
        "reviewer_reasoning_effort",
        "independent_review",
        "review_artifact",
        "tool_catalog_sha256",
        "policy_hash",
        "evidence_sha256",
        "citation",
        "receipt_sha256",
    }
    errors: list[str] = []
    metadata = _dict(row.get("metadata"))
    domain = str(metadata.get("domain") or "")
    if set(record) != expected_keys:
        errors.append("E_SCALE_TOOL_EXEMPTION_FIELDS")
    if record.get("schema_version") != TOOL_EXEMPTION_SCHEMA_VERSION:
        errors.append("E_SCALE_TOOL_EXEMPTION_VERSION")
    if record.get("domain") != domain or record.get("tool_name") != tool_name:
        errors.append("E_SCALE_TOOL_EXEMPTION_SCOPE_BINDING")
    if record.get("scope") != [
        "canonical_argument_payload_count",
        "supervised_target_count",
    ]:
        errors.append("E_SCALE_TOOL_EXEMPTION_DIMENSIONS")
    if record.get("reviewer_model") != "gpt-5.6-sol":
        errors.append("E_SCALE_TOOL_EXEMPTION_REVIEWER_MODEL")
    if record.get("reviewer_reasoning_effort") != "ultra":
        errors.append("E_SCALE_TOOL_EXEMPTION_REVIEWER_EFFORT")
    if record.get("independent_review") is not True:
        errors.append("E_SCALE_TOOL_EXEMPTION_INDEPENDENCE")
    if not isinstance(record.get("reviewer"), str) or not record.get("reviewer"):
        errors.append("E_SCALE_TOOL_EXEMPTION_REVIEWER")
    for field in (
        "reviewer_artifact_sha256",
        "reviewer_inference_receipt_sha256",
        "evidence_sha256",
    ):
        if not SHA256_RE.fullmatch(str(record.get(field) or "")):
            errors.append("E_SCALE_TOOL_EXEMPTION_HASH")
    if record.get("tool_catalog_sha256") != metadata.get("tool_catalog_sha256"):
        errors.append("E_SCALE_TOOL_EXEMPTION_CATALOG_BINDING")
    if record.get("policy_hash") != metadata.get("policy_sha256"):
        errors.append("E_SCALE_TOOL_EXEMPTION_POLICY_BINDING")
    if not isinstance(record.get("citation"), str) or not record.get("citation"):
        errors.append("E_SCALE_TOOL_EXEMPTION_CITATION")

    review_artifact = record.get("review_artifact")
    if not isinstance(review_artifact, dict):
        errors.append("E_SCALE_TOOL_EXEMPTION_REVIEW_ARTIFACT")
        review_artifact = {}
    expected_review_artifact_keys = {
        "schema_version",
        "reviewer",
        "inference_receipt",
        "review_scope",
        "independent_review_pass",
    }
    if set(review_artifact) != expected_review_artifact_keys:
        errors.append("E_SCALE_TOOL_EXEMPTION_REVIEW_ARTIFACT_FIELDS")
    if (
        review_artifact.get("schema_version")
        != TOOL_EXEMPTION_REVIEW_SCHEMA_VERSION
    ):
        errors.append("E_SCALE_TOOL_EXEMPTION_REVIEW_ARTIFACT_VERSION")
    if review_artifact.get("reviewer") != record.get("reviewer"):
        errors.append("E_SCALE_TOOL_EXEMPTION_REVIEWER_BINDING")
    if review_artifact.get("independent_review_pass") is not True:
        errors.append("E_SCALE_TOOL_EXEMPTION_REVIEW_ARTIFACT_INDEPENDENCE")
    inference_receipt = review_artifact.get("inference_receipt")
    if not isinstance(inference_receipt, dict):
        errors.append("E_SCALE_TOOL_EXEMPTION_REVIEW_INFERENCE")
        inference_receipt = {}
    expected_inference_keys = {
        "schema_version",
        "inference_origin",
        "model",
        "reasoning_effort",
        "native_codex_inference_calls",
        "provider_accessed",
        "network_accessed",
        "prohibited_external_model_provider_calls",
        "prohibited_external_network_calls",
    }
    if set(inference_receipt) != expected_inference_keys:
        errors.append("E_SCALE_TOOL_EXEMPTION_REVIEW_INFERENCE_FIELDS")
    if (
        inference_receipt.get("schema_version")
        != TOOL_EXEMPTION_REVIEW_INFERENCE_SCHEMA_VERSION
    ):
        errors.append("E_SCALE_TOOL_EXEMPTION_REVIEW_INFERENCE_VERSION")
    if inference_receipt.get("inference_origin") != NATIVE_CODEX_ORIGIN:
        errors.append("E_SCALE_TOOL_EXEMPTION_REVIEWER_ORIGIN")
    if inference_receipt.get("model") != record.get("reviewer_model"):
        errors.append("E_SCALE_TOOL_EXEMPTION_REVIEWER_MODEL_BINDING")
    if inference_receipt.get("reasoning_effort") != record.get(
        "reviewer_reasoning_effort"
    ):
        errors.append("E_SCALE_TOOL_EXEMPTION_REVIEWER_EFFORT_BINDING")
    native_calls = inference_receipt.get("native_codex_inference_calls")
    if type(native_calls) is not int or native_calls < 1:
        errors.append("E_SCALE_TOOL_EXEMPTION_REVIEW_NATIVE_CALLS")
    for field in ("provider_accessed", "network_accessed"):
        if inference_receipt.get(field) is not True:
            errors.append("E_SCALE_TOOL_EXEMPTION_REVIEW_INFERENCE_ACCESS")
    for field in (
        "prohibited_external_model_provider_calls",
        "prohibited_external_network_calls",
    ):
        if inference_receipt.get(field) != 0:
            errors.append("E_SCALE_TOOL_EXEMPTION_REVIEW_PROHIBITED_CALLS")
    if record.get("reviewer_inference_receipt_sha256") != canonical_sha256(
        inference_receipt
    ):
        errors.append("E_SCALE_TOOL_EXEMPTION_REVIEW_INFERENCE_HASH")
    if record.get("reviewer_artifact_sha256") != canonical_sha256(review_artifact):
        errors.append("E_SCALE_TOOL_EXEMPTION_REVIEW_ARTIFACT_HASH")
    expected_review_scope = {
        "domain": record.get("domain"),
        "tool_name": record.get("tool_name"),
        "reason": record.get("reason"),
        "scope": record.get("scope"),
        "tool_catalog_sha256": record.get("tool_catalog_sha256"),
        "policy_hash": record.get("policy_hash"),
        "evidence_sha256": record.get("evidence_sha256"),
        "citation": record.get("citation"),
    }
    if review_artifact.get("review_scope") != expected_review_scope:
        errors.append("E_SCALE_TOOL_EXEMPTION_REVIEW_SCOPE_BINDING")
    try:
        tool_def = _find_tool(
            row.get("tool_catalog") if isinstance(row.get("tool_catalog"), list) else [],
            tool_name,
        )
    except Tau3GroundedGenerationError:
        errors.append("E_SCALE_TOOL_EXEMPTION_TOOL_BINDING")
        tool_def = {}
    reason = record.get("reason")
    if reason == "zero_arg":
        if not _is_zero_argument_tool(tool_def):
            errors.append("E_SCALE_TOOL_EXEMPTION_ZERO_ARG_FALSE")
    elif reason != "policy_forbidden":
        errors.append("E_SCALE_TOOL_EXEMPTION_REASON")
    expected_receipt_sha = canonical_sha256(
        {key: item for key, item in record.items() if key != "receipt_sha256"}
    )
    if record.get("receipt_sha256") != expected_receipt_sha:
        errors.append("E_SCALE_TOOL_EXEMPTION_RECEIPT_HASH")
    generator_sha = _dict(metadata.get("generation_provenance")).get(
        "inference_receipt_sha256"
    )
    if record.get("reviewer_inference_receipt_sha256") == generator_sha:
        errors.append("E_SCALE_TOOL_EXEMPTION_REVIEW_NOT_INDEPENDENT")
    return sorted(set(errors))


def _domain_tool_exemptions(
    domain: str,
    domain_rows: list[dict[str, Any]],
    tool_names: list[str],
) -> tuple[dict[str, str], list[str]]:
    accepted: dict[str, str] = {}
    errors: list[str] = []
    for tool_name in tool_names:
        records: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for row in domain_rows:
            exemptions = _dict(row.get("metadata")).get("tool_exemptions")
            if not isinstance(exemptions, list):
                exemptions = []
            matching = [
                record
                for record in exemptions
                if isinstance(record, dict) and record.get("tool_name") == tool_name
            ]
            if len(matching) > 1:
                errors.append("E_SCALE_TOOL_EXEMPTION_DUPLICATE")
            if len(matching) == 1:
                records.append((row, matching[0]))
        if not records:
            continue
        if len(records) != len(domain_rows):
            errors.append("E_SCALE_TOOL_EXEMPTION_INCONSISTENT")
            continue
        hashes = {canonical_sha256(record) for _, record in records}
        if len(hashes) != 1:
            errors.append("E_SCALE_TOOL_EXEMPTION_INCONSISTENT")
            continue
        record_errors = [
            error
            for row, record in records
            for error in _scaled_tool_exemption_errors(
                record,
                row=row,
                tool_name=tool_name,
            )
        ]
        if record_errors:
            errors.extend(record_errors)
            continue
        accepted[tool_name] = records[0][1]["receipt_sha256"]
    return accepted, sorted(set(errors))


def _scaled_coverage(
    rows_by_split: dict[str, list[dict[str, Any]]],
    *,
    training_handoff: dict[str, Any] | None,
) -> dict[str, Any]:
    blocker_records: list[dict[str, Any]] = []
    nonwaivable: set[str] = set()

    def block(
        error_id: str,
        *,
        split: str | None = None,
        domain: str | None = None,
        dimension: str | None = None,
        identifier: str | None = None,
        observed: int | None = None,
        required: int | None = None,
        nonwaivable_error: bool = False,
    ) -> None:
        scope = {
            key: value
            for key, value in {
                "split": split,
                "domain": domain,
                "dimension": dimension,
                "identifier_sha256": (
                    identifier
                    if isinstance(identifier, str) and SHA256_RE.fullmatch(identifier)
                    else canonical_sha256(identifier)
                    if isinstance(identifier, str)
                    else None
                ),
                "observed": observed,
                "required": required,
            }.items()
            if value is not None
        }
        record = {"error_id": error_id, **scope}
        record["record_sha256"] = canonical_sha256(record)
        if record not in blocker_records:
            blocker_records.append(record)
        if nonwaivable_error:
            nonwaivable.add(error_id)

    for error_id in _training_handoff_errors(
        training_handoff,
        "training_handoff",
    ):
        block(error_id, nonwaivable_error=True)
    for error_id in _scaled_selection_claim_errors(rows_by_split):
        block(error_id, nonwaivable_error=True)
    for error_id in _scaled_hash_disjointness_errors(rows_by_split):
        block(error_id, nonwaivable_error=True)

    all_domain_rows: dict[str, list[dict[str, Any]]] = {domain: [] for domain in DOMAINS}
    for split in SPLITS:
        for row in rows_by_split.get(split, []):
            domain = _dict(row.get("metadata")).get("domain")
            if domain in DOMAINS:
                all_domain_rows[str(domain)].append(row)

    catalog_tools: dict[str, list[str]] = {}
    exemptions: dict[str, dict[str, str]] = {}
    for domain in DOMAINS:
        domain_rows = all_domain_rows[domain]
        catalogs = {
            canonical_sha256(row.get("tool_catalog"))
            for row in domain_rows
            if isinstance(row.get("tool_catalog"), list)
        }
        if len(catalogs) > 1:
            block(
                "E_SCALE_TOOL_CATALOG_INCONSISTENT",
                domain=domain,
                nonwaivable_error=True,
            )
        names = sorted(
            {
                name
                for row in domain_rows
                for name in _catalog_tool_names(row.get("tool_catalog"))
            }
        )
        catalog_tools[domain] = names
        accepted, exemption_errors = _domain_tool_exemptions(
            domain,
            domain_rows,
            names,
        )
        exemptions[domain] = accepted
        for error_id in exemption_errors:
            block(
                error_id,
                domain=domain,
                nonwaivable_error=True,
            )

    by_split: dict[str, Any] = {}
    split_token_totals: dict[str, dict[str, int]] = {}
    split_row_counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        by_split[split] = {}
        split_token_totals[split] = {}
        split_row_counts[split] = {}
        for domain in DOMAINS:
            rows = [
                row
                for row in rows_by_split.get(split, [])
                if _dict(row.get("metadata")).get("domain") == domain
            ]
            split_row_counts[split][domain] = len(rows)
            if not rows:
                block("E_SCALE_DOMAIN_MISSING", split=split, domain=domain)
            unsupported_count = sum(
                1
                for row in rows
                if not _runtime_is_candidate_eligible(
                    _dict(row.get("metadata")).get("runtime_family")
                )
            )
            if unsupported_count:
                block(
                    "E_SCALE_RUNTIME_INELIGIBLE",
                    split=split,
                    domain=domain,
                    observed=unsupported_count,
                    required=0,
                    nonwaivable_error=True,
                )
            admitted: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for row in rows:
                targets = row.get("training_targets")
                if not isinstance(targets, list):
                    continue
                admitted.extend(
                    (row, target)
                    for target in targets
                    if _coverage_target_is_admitted(target)
                )
            token_total = sum(
                _canonical_target_token_count(_dict(target.get("canonical_target")))
                for _, target in admitted
            )
            split_token_totals[split][domain] = token_total

            behavior_metrics: dict[str, Any] = {}
            generated_family_dominance: dict[str, Any] = {}
            for behavior in BEHAVIORS:
                behavior_targets = [
                    (row, target)
                    for row, target in admitted
                    if target.get("behavior") == behavior
                ]
                family_counter: Counter[str] = Counter()
                for row, _ in behavior_targets:
                    generated_id = _dict(
                        _dict(row.get("metadata")).get("selection_receipt")
                    ).get("generated_family_identifier")
                    if SHA256_RE.fullmatch(str(generated_id or "")):
                        family_counter[str(generated_id)] += 1
                correction_count = (
                    sum(
                        1
                        for row, target in behavior_targets
                        if _has_reviewed_negative_context(row, target)
                    )
                    if behavior in NEGATIVE_CORRECTION_BEHAVIORS
                    else None
                )
                target_minimum = SCALED_BEHAVIOR_TARGET_MINIMUMS[split]
                family_minimum = SCALED_BEHAVIOR_FAMILY_MINIMUMS[split]
                if not _meets_minimum(len(behavior_targets), target_minimum):
                    block(
                        "E_SCALE_BEHAVIOR_TARGET_COUNT",
                        split=split,
                        domain=domain,
                        dimension=behavior,
                        observed=len(behavior_targets),
                        required=target_minimum,
                    )
                if not _meets_minimum(len(family_counter), family_minimum):
                    block(
                        "E_SCALE_BEHAVIOR_FAMILY_SPAN",
                        split=split,
                        domain=domain,
                        dimension=behavior,
                        observed=len(family_counter),
                        required=family_minimum,
                    )
                if correction_count is not None and not _meets_minimum(
                    correction_count,
                    target_minimum,
                ):
                    block(
                        "E_SCALE_CORRECTION_CONTEXT_COUNT",
                        split=split,
                        domain=domain,
                        dimension=behavior,
                        observed=correction_count,
                        required=target_minimum,
                    )
                family_dominance = _dominance_record(family_counter)
                if family_dominance["applicable"] and not family_dominance["passed"]:
                    block(
                        "E_SCALE_GENERATED_FAMILY_DOMINANCE",
                        split=split,
                        domain=domain,
                        dimension=behavior,
                        identifier=str(family_dominance["dominant_identifier_sha256"]),
                        observed=int(family_dominance["maximum_count"]),
                        required=int(family_dominance["total"]),
                    )
                generated_family_dominance[behavior] = family_dominance
                behavior_metrics[behavior] = {
                    "admitted_target_count": len(behavior_targets),
                    "generated_family_count": len(family_counter),
                    "reviewed_correction_pair_count": correction_count,
                }

            tool_targets = [
                (row, target, _dict(target.get("canonical_target")))
                for row, target in admitted
                if _dict(target.get("canonical_target")).get("kind") == "tool_call"
            ]
            tool_counter: Counter[str] = Counter(
                str(canonical.get("tool_name"))
                for _, _, canonical in tool_targets
                if isinstance(canonical.get("tool_name"), str)
                and canonical.get("tool_name")
            )
            argument_counters: dict[str, Counter[str]] = {
                tool_name: Counter() for tool_name in catalog_tools[domain]
            }
            mutation_counter: Counter[str] = Counter()
            for _, _, canonical in tool_targets:
                tool_name = str(canonical.get("tool_name") or "")
                arguments = canonical.get("arguments")
                if not isinstance(arguments, dict):
                    continue
                argument_counters.setdefault(tool_name, Counter())[
                    _canonical_argument_template_sha256(arguments)
                ] += 1
                if _is_mutation_tool(tool_name):
                    mutation_counter[
                        _synthetic_mutation_family_sha256(
                            domain,
                            tool_name,
                            arguments,
                        )
                    ] += 1
            tool_metrics: dict[str, Any] = {}
            argument_dominance: dict[str, Any] = {}
            for tool_name in catalog_tools[domain]:
                exempt_receipt = exemptions[domain].get(tool_name)
                target_count = tool_counter[tool_name]
                distinct_arguments = len(argument_counters.get(tool_name, Counter()))
                if exempt_receipt is None:
                    if not _meets_minimum(
                        target_count,
                        SCALED_TOOL_TARGET_MINIMUMS[split],
                    ):
                        block(
                            "E_SCALE_TOOL_TARGET_COUNT",
                            split=split,
                            domain=domain,
                            dimension=tool_name,
                            observed=target_count,
                            required=SCALED_TOOL_TARGET_MINIMUMS[split],
                        )
                    if not _meets_minimum(
                        distinct_arguments,
                        SCALED_TOOL_ARGUMENT_MINIMUMS[split],
                    ):
                        block(
                            "E_SCALE_TOOL_ARGUMENT_DIVERSITY",
                            split=split,
                            domain=domain,
                            dimension=tool_name,
                            observed=distinct_arguments,
                            required=SCALED_TOOL_ARGUMENT_MINIMUMS[split],
                        )
                dominance = _dominance_record(argument_counters.get(tool_name, Counter()))
                if dominance["applicable"] and not dominance["passed"]:
                    block(
                        "E_SCALE_ARGUMENT_TEMPLATE_DOMINANCE",
                        split=split,
                        domain=domain,
                        dimension=tool_name,
                        identifier=str(dominance["dominant_identifier_sha256"]),
                        observed=int(dominance["maximum_count"]),
                        required=int(dominance["total"]),
                    )
                argument_dominance[tool_name] = dominance
                tool_metrics[tool_name] = {
                    "admitted_target_count": target_count,
                    "distinct_canonical_argument_payload_count": distinct_arguments,
                    "exempt": exempt_receipt is not None,
                    "exemption_receipt_sha256": exempt_receipt,
                }
            tool_dominance = _dominance_record(tool_counter)
            if tool_dominance["applicable"] and not tool_dominance["passed"]:
                block(
                    "E_SCALE_TARGET_TOOL_DOMINANCE",
                    split=split,
                    domain=domain,
                    identifier=str(tool_dominance["dominant_identifier_sha256"]),
                    observed=int(tool_dominance["maximum_count"]),
                    required=int(tool_dominance["total"]),
                )
            mutation_dominance = _dominance_record(mutation_counter)
            if mutation_dominance["applicable"] and not mutation_dominance["passed"]:
                block(
                    "E_SCALE_SYNTHETIC_MUTATION_FAMILY_DOMINANCE",
                    split=split,
                    domain=domain,
                    identifier=str(mutation_dominance["dominant_identifier_sha256"]),
                    observed=int(mutation_dominance["maximum_count"]),
                    required=int(mutation_dominance["total"]),
                )
            positive_hashes = [
                str(target.get("canonical_target_sha256")) for _, target in admitted
            ]
            duplication = _duplication_record(positive_hashes)
            if duplication["total"] and not _fraction_at_most(
                int(duplication["duplicate_excess"]),
                int(duplication["total"]),
                1,
                5,
            ):
                block(
                    "E_SCALE_TARGET_DUPLICATION",
                    split=split,
                    domain=domain,
                    observed=int(duplication["duplicate_excess"]),
                    required=int(duplication["total"]),
                )
            by_split[split][domain] = {
                "row_count": len(rows),
                "admitted_unmasked_target_count": len(admitted),
                "masked_target_count": sum(
                    1
                    for row in rows
                    for target in (
                        row.get("training_targets")
                        if isinstance(row.get("training_targets"), list)
                        else []
                    )
                    if isinstance(target, dict) and target.get("masked") is True
                ),
                "supervised_target_token_count": token_total,
                "behaviors": behavior_metrics,
                "tools": tool_metrics,
                "positive_target_duplication": duplication,
                "dominance": {
                    "generated_family_by_behavior": generated_family_dominance,
                    "target_tool": tool_dominance,
                    "canonical_argument_payload_by_tool": argument_dominance,
                    "synthetic_mutation_family": mutation_dominance,
                },
            }

    token_balance: dict[str, Any] = {}
    for split in SPLITS:
        domain_tokens = split_token_totals[split]
        total_tokens = sum(domain_tokens.values())
        token_balance[split] = {
            "total_supervised_target_tokens": total_tokens,
            "domains": {},
        }
        for domain in DOMAINS:
            count = domain_tokens[domain]
            fraction = _fraction_record(count, total_tokens)
            token_balance[split]["domains"][domain] = {
                "supervised_target_token_count": count,
                "fraction": fraction,
            }
            if not _fraction_at_least(count, total_tokens, 1, 4) or not _fraction_at_most(
                count,
                total_tokens,
                2,
                5,
            ):
                block(
                    "E_SCALE_DOMAIN_TOKEN_SHARE",
                    split=split,
                    domain=domain,
                    observed=count,
                    required=total_tokens,
                )
        positive_domain_tokens = [domain_tokens[domain] for domain in DOMAINS]
        smallest = min(positive_domain_tokens)
        largest = max(positive_domain_tokens)
        ratio_passed = _ratio_at_most(largest, smallest, 8, 5)
        token_balance[split]["maximum_minimum_ratio"] = {
            "maximum": largest,
            "minimum": smallest,
            "fraction": _fraction_record(largest, smallest),
            "passed": ratio_passed,
        }
        if not ratio_passed:
            block(
                "E_SCALE_DOMAIN_TOKEN_RATIO",
                split=split,
                observed=largest,
                required=smallest,
            )

    train_total_rows = sum(split_row_counts["train"].values())
    telecom_train_rows = split_row_counts["train"]["telecom"]
    telecom_example_fraction = _fraction_record(telecom_train_rows, train_total_rows)
    if not _fraction_at_least(telecom_train_rows, train_total_rows, 1, 4):
        block(
            "E_SCALE_TELECOM_TRAIN_EXAMPLE_SHARE",
            split="train",
            domain="telecom",
            observed=telecom_train_rows,
            required=train_total_rows,
        )

    train_example_hashes = [
        _training_example_sha256(row) for row in rows_by_split.get("train", [])
    ]
    training_example_duplication = _duplication_record(train_example_hashes)
    if (
        training_example_duplication["total"]
        and not _fraction_at_most(
            int(training_example_duplication["duplicate_excess"]),
            int(training_example_duplication["total"]),
            1,
            100,
        )
    ):
        block(
            "E_SCALE_TRAINING_EXAMPLE_DUPLICATION",
            split="train",
            observed=int(training_example_duplication["duplicate_excess"]),
            required=int(training_example_duplication["total"]),
        )

    blocker_records.sort(key=canonical_sha256)
    if len(blocker_records) > 4096:
        blocker_records = blocker_records[:4095]
        block("E_SCALE_BLOCKER_RECORD_LIMIT", nonwaivable_error=True)
        blocker_records.sort(key=canonical_sha256)
    blockers = sorted({record["error_id"] for record in blocker_records})
    token_contract: dict[str, Any] = {
        "schema_version": TOKEN_COUNT_SCHEMA_VERSION,
        "algorithm": TOKEN_COUNT_ALGORITHM,
        "canonicalization": "sorted_compact_ascii_json",
        "counted_unit": "utf8_byte",
        "admitted_unmasked_canonical_target_only": True,
        "dependency_free": True,
    }
    token_contract["contract_sha256"] = canonical_sha256(token_contract)
    result: dict[str, Any] = {
        "schema_version": SCALED_COVERAGE_SCHEMA_VERSION,
        "coverage_profile": _coverage_profile_record(SCALED_COVERAGE_PROFILE_ID),
        "passed": not blockers,
        "blockers": blockers,
        "blocker_count": len(blocker_records),
        "blocker_error_id_count": len(blockers),
        "blocker_records": blocker_records,
        "nonwaivable_blockers": sorted(nonwaivable),
        "by_split": by_split,
        "token_count_contract": token_contract,
        "training_handoff_sha256": (
            training_handoff.get("handoff_sha256")
            if isinstance(training_handoff, dict)
            else None
        ),
        "token_balance": token_balance,
        "telecom_train_example_fraction": telecom_example_fraction,
        "training_example_duplication": training_example_duplication,
        "hash_disjointness": _split_hash_evidence(rows_by_split),
    }
    result["coverage_sha256"] = canonical_sha256(result)
    return result


def _manifest(
    source_path: Path,
    out: Path,
    rows_by_split: dict[str, list[dict[str, Any]]],
    coverage: dict[str, Any],
    *,
    profile_id: str = LEGACY_COVERAGE_PROFILE_ID,
    training_handoff: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": TAU3_GROUNDED_DATASET_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "passed": coverage["passed"],
        "status": "passed" if coverage["passed"] else "blocked",
        "blockers": coverage["blockers"],
        "source": {
            "path_leaf": source_path.name,
            "sha256": _sha256(source_path),
            "training_side_only": True,
            "accepted_source_families": sorted(ALLOWED_SOURCE_FAMILIES),
        },
        "files": {
            split: _file_record(out / f"{split}.jsonl", relative_to=out)
            for split in SPLITS
        },
        "counts": {split: len(rows_by_split[split]) for split in SPLITS},
        "derivation": {
            "algorithm": "deterministic_local_tau_tool_replay_v1",
            "dependency_free": True,
            "training_side_only": True,
            "replays_tool_methods": True,
            "candidate_eligible_runtime_family": f"{VENDORED_RUNTIME_PREFIX}<tau_revision>",
            "test_only_runtime_family": FAKE_TEST_RUNTIME_FAMILY,
            "fabricates_success_claims": False,
            "negative_or_unsafe_targets_masked": True,
            "runtime_derived_system_prompt": True,
            "ordered_openai_tool_schema_catalog": True,
            "full_environment_state_replayed": True,
            "chronological_argument_grounding": True,
            "policy_confirmation_replayed": True,
            "policy_mutation_review_replayed": True,
            "confirmation_detail_grounding_replayed": True,
            "task_initialization_replayed": True,
            "ordered_initial_sync_sequence_replayed": True,
            "initial_sync_sequence_sha256_bound": True,
            "single_ordered_tool_sequence": True,
            "scoped_codex_provenance": True,
            "registered_selection_algorithm": True,
            "selection_hash_semantics_replayed": True,
        },
        "artifact_security": {
            "classification": "sensitive",
            "owner_only_required": True,
            "directory_mode": "0700",
            "file_mode": "0600_or_stricter",
            "raw_session_publishable": False,
        },
        "row_contract": {
            "schema_version": TAU3_GROUNDED_ROW_SCHEMA_VERSION,
            "state_refs_owner_only": True,
            "initial_sync_evidence_required": True,
            "ordered_initial_sync_sequence_required": True,
            "initial_sync_sequence_sha256_required": True,
            "per_call_sync_evidence_required": True,
            "generation_provenance_required": True,
            "selection_receipt_required": True,
            "task_initialization_receipt_required": True,
            "policy_target_review_required": True,
            "confirmation_detail_grounding_required": True,
        },
        "coverage": coverage,
        "sealed_access": {
            "payload_accessed": False,
            "access_count": 0,
            "materialized_sealed_fields": [],
        },
        "contamination": {
            "raw_sealed_payload_read": False,
            "split_contamination_detected": False,
            "train_validation_source_hash_disjoint": True,
        },
    }
    if profile_id == SCALED_COVERAGE_PROFILE_ID:
        manifest["coverage_profile"] = _coverage_profile_record(profile_id)
        manifest["training_handoff"] = copy.deepcopy(training_handoff)
        manifest["hash_disjointness"] = _split_hash_evidence(rows_by_split)
        manifest["row_contract"]["scaled_selection_receipt_required"] = True
        manifest["row_contract"]["generated_family_binding_required"] = True
        manifest["derivation"]["deterministic_stratified_ordinal_selection"] = True
        manifest["derivation"]["full_rubric_aggregate_replayed"] = True
        manifest["derivation"]["supervised_target_token_count_dependency_free"] = True
    manifest["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    return manifest


def _externalize_state_snapshots(
    rows_by_split: dict[str, list[dict[str, Any]]],
    out: Path,
) -> None:
    states_dir = out / "states"
    _secure_mkdir(out)
    _secure_mkdir(states_dir)
    written: set[str] = set()

    def externalize(state: Any, label: str) -> dict[str, Any]:
        if not isinstance(state, dict):
            raise Tau3GroundedGenerationError(f"internal row missing {label} state")
        state_sha = canonical_sha256(state)
        state_path = states_dir / f"{state_sha}.json"
        if state_sha not in written:
            _write_json(state_path, state)
            written.add(state_sha)
        return _file_record(state_path, relative_to=out)

    for rows in rows_by_split.values():
        for row in rows:
            state = row.pop("initial_state", None)
            state_sha = canonical_sha256(state)
            if state_sha != _dict(row.get("metadata")).get("initial_state_sha256"):
                raise Tau3GroundedGenerationError("initial state hash changed before externalization")
            row["initial_state_ref"] = externalize(state, "initial")
            final_state = row.pop("final_state", None)
            if canonical_sha256(final_state) != _dict(row.get("metadata")).get("final_state_sha256"):
                raise Tau3GroundedGenerationError("final state hash changed before externalization")
            row["final_state_ref"] = externalize(final_state, "final")
            initial_sync = _dict(row.get("initial_sync"))
            steps = initial_sync.get("steps")
            if not isinstance(steps, list) or initial_sync.get("sync_count") != len(steps):
                raise Tau3GroundedGenerationError(
                    "initial sync sequence is missing or internally inconsistent"
                )
            for step_index, step in enumerate(steps):
                if not isinstance(step, dict) or step.get("ordinal") != step_index:
                    raise Tau3GroundedGenerationError(
                        "initial sync sequence ordinals are not physical order"
                    )
                step["pre_state_ref"] = externalize(
                    step.pop("pre_state", None),
                    f"initial sync step {step_index} pre",
                )
                step["post_state_ref"] = externalize(
                    step.pop("post_state", None),
                    f"initial sync step {step_index} post",
                )
            initial_sync["pre_state_ref"] = externalize(
                initial_sync.pop("pre_state", None),
                "initial sync pre",
            )
            initial_sync["post_state_ref"] = externalize(
                initial_sync.pop("post_state", None),
                "initial sync post",
            )
            for call in row.get("tool_replay", []):
                if not isinstance(call, dict):
                    continue
                call["pre_state_ref"] = externalize(call.pop("pre_state", None), "tool pre")
                call["pre_sync_state_ref"] = externalize(
                    call.pop("pre_sync_state", None),
                    "tool pre-sync",
                )
                call["post_state_ref"] = externalize(call.pop("post_state", None), "tool post")
                sync_evidence = _dict(call.get("sync_evidence"))
                sync_evidence["pre_state_ref"] = externalize(
                    sync_evidence.pop("pre_state", None),
                    "sync pre",
                )
                sync_evidence["post_state_ref"] = externalize(
                    sync_evidence.pop("post_state", None),
                    "sync post",
                )
            row["metadata"]["row_sha256"] = canonical_sha256(_without_row_sha(row))


def _load_state_ref(
    bundle: Path,
    value: Any,
    context: str,
    errors: list[str],
) -> dict[str, Any]:
    ref = _object(value, f"{context}.initial_state_ref", errors)
    rel = ref.get("path")
    if not isinstance(rel, str) or not _safe_relative_path(rel):
        errors.append(f"{context}.initial_state_ref.path must be a safe relative path")
        return {}
    path = bundle / rel
    permission_errors = _bundle_artifact_path_errors(
        bundle,
        path,
        f"{context}.state_ref",
        directory=False,
    )
    if permission_errors:
        errors.extend(permission_errors)
        return {}
    errors.extend(_file_record_errors(path, ref, f"{context}.state_ref"))
    if ref.get("sha256") != _sha256(path):
        errors.append(f"{context}.initial_state_ref.sha256 does not replay")
    if ref.get("bytes") != path.stat().st_size:
        errors.append(f"{context}.initial_state_ref.bytes does not replay")
    try:
        state = _read_json(path, f"{context}.initial_state_ref")
    except Tau3GroundedGenerationError as exc:
        errors.append(str(exc))
        return {}
    return state


def _expanded_sync_evidence(
    bundle: Path,
    value: Any,
    context: str,
    errors: list[str],
) -> dict[str, Any]:
    record = _object(value, context, errors)
    expanded = {
        key: copy.deepcopy(item)
        for key, item in record.items()
        if key not in {"pre_state_ref", "post_state_ref"}
    }
    pre_state = _load_state_ref(
        bundle,
        record.get("pre_state_ref"),
        f"{context}.pre_state_ref",
        errors,
    )
    post_state = _load_state_ref(
        bundle,
        record.get("post_state_ref"),
        f"{context}.post_state_ref",
        errors,
    )
    expanded["pre_state"] = pre_state
    expanded["post_state"] = post_state
    if record.get("performed") not in {True, False}:
        errors.append(f"{context}.performed must be boolean")
    if record.get("pre_state_sha256") != canonical_sha256(pre_state):
        errors.append(f"{context}.pre_state_sha256 does not replay")
    if record.get("post_state_sha256") != canonical_sha256(post_state):
        errors.append(f"{context}.post_state_sha256 does not replay")
    if record.get("state_diff") != _state_diff(pre_state, post_state):
        errors.append(f"{context}.state_diff does not replay")
    return expanded


def _expanded_initial_sync_evidence(
    bundle: Path,
    value: Any,
    context: str,
    errors: list[str],
) -> dict[str, Any]:
    record = _object(value, context, errors)
    expected_record_keys = {
        "performed",
        "sync_count",
        "steps",
        "sequence_sha256",
        "pre_state_ref",
        "post_state_ref",
        "pre_state_sha256",
        "post_state_sha256",
        "state_diff",
    }
    if set(record) != expected_record_keys:
        errors.append(f"{context} fields do not match the initial sync contract")
    expanded = _expanded_sync_evidence(bundle, record, context, errors)
    raw_steps = record.get("steps")
    if not isinstance(raw_steps, list):
        errors.append(f"{context}.steps must be an ordered list")
        raw_steps = []
    steps: list[dict[str, Any]] = []
    for step_index, raw_step in enumerate(raw_steps):
        step_context = f"{context}.steps[{step_index}]"
        expected_step_keys = {
            "ordinal",
            "performed",
            "previous_step_sha256",
            "step_sha256",
            "pre_state_ref",
            "post_state_ref",
            "pre_state_sha256",
            "post_state_sha256",
            "state_diff",
        }
        if not isinstance(raw_step, dict) or set(raw_step) != expected_step_keys:
            errors.append(f"{step_context} fields do not match the sync step contract")
        step = _expanded_sync_evidence(
            bundle,
            raw_step,
            step_context,
            errors,
        )
        if step.get("ordinal") != step_index:
            errors.append(f"{step_context}.ordinal must equal its physical order")
        if step.get("performed") is not True:
            errors.append(f"{step_context}.performed must be true")
        expected_previous = None if step_index == 0 else steps[-1].get("step_sha256")
        if step.get("previous_step_sha256") != expected_previous:
            errors.append(
                f"{step_context}.previous_step_sha256 breaks the physical hash chain"
            )
        expected_step_sha256 = canonical_sha256(
            {key: item for key, item in step.items() if key != "step_sha256"}
        )
        if step.get("step_sha256") != expected_step_sha256:
            errors.append(f"{step_context}.step_sha256 does not replay")
        steps.append(step)
    expanded["steps"] = steps
    sync_count = record.get("sync_count")
    if type(sync_count) is not int or sync_count < 0:
        errors.append(f"{context}.sync_count must be a non-negative integer")
    elif sync_count != len(steps):
        errors.append(f"{context}.sync_count does not match ordered steps")
    expected_sequence_sha256 = canonical_sha256(steps)
    if record.get("sequence_sha256") != expected_sequence_sha256:
        errors.append(f"{context}.sequence_sha256 does not replay ordered steps")
    if steps:
        if record.get("performed") is not True:
            errors.append(f"{context}.performed must be true when steps are retained")
        if expanded.get("pre_state") != steps[0].get("pre_state"):
            errors.append(f"{context}.pre_state must bind the first physical step")
        if expanded.get("post_state") != steps[-1].get("post_state"):
            errors.append(f"{context}.post_state must bind the final physical step")
    else:
        if record.get("performed") is not False:
            errors.append(f"{context}.performed must be false when steps are empty")
        if expanded.get("pre_state") != expanded.get("post_state"):
            errors.append(f"{context} empty sequence must preserve state")
    return expanded


def _split_contamination_errors(rows_by_split: dict[str, list[dict[str, Any]]]) -> list[str]:
    errors: list[str] = []
    seen: dict[str, str] = {}
    for split, rows in rows_by_split.items():
        for index, row in enumerate(rows):
            metadata = _dict(row.get("metadata"))
            source_hash = str(metadata.get("source_sha256") or "")
            source_id = str(metadata.get("source_id") or "")
            family_key = f"{metadata.get('domain')}:{metadata.get('source_family_id')}"
            for key, label in (
                (source_hash, "source_sha256"),
                (source_id, "source_id"),
                (family_key, "source_family_id"),
            ):
                if not key:
                    continue
                prior = seen.setdefault(key, split)
                if prior != split:
                    errors.append(f"{label} crosses splits at {split}[{index}]")
    return errors


def _find_tool(tool_catalog: list[dict[str, Any]], tool_name: str) -> dict[str, Any]:
    for tool in tool_catalog:
        if tool.get("name") == tool_name:
            return tool
        function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
        if function.get("name") == tool_name:
            return tool
    raise Tau3GroundedGenerationError(f"tool {tool_name!r} is not present in the exact tool catalog")


def _state_diff(before: Any, after: Any) -> dict[str, Any]:
    changes: list[dict[str, Any]] = []
    _diff_value(before, after, "", changes)
    changes.sort(key=lambda change: change["path"])
    return {
        "changed": bool(changes),
        "change_count": len(changes),
        "truncated": False,
        "changes": changes,
    }


def _diff_value(before: Any, after: Any, path: str, changes: list[dict[str, Any]]) -> None:
    if before == after:
        return
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(set(before) | set(after)):
            child_path = f"{path}/{_escape_json_pointer(str(key))}"
            if key not in before:
                changes.append({"kind": "added", "path": child_path, "before": None, "after": _canonical_value(after[key])})
            elif key not in after:
                changes.append({"kind": "removed", "path": child_path, "before": _canonical_value(before[key]), "after": None})
            else:
                _diff_value(before[key], after[key], child_path, changes)
        return
    changes.append({"kind": "changed", "path": path or "/", "before": _canonical_value(before), "after": _canonical_value(after)})


def _escape_json_pointer(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(_canonical_value(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _canonical_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonical_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonical_value(item) for item in value]
    if isinstance(value, tuple):
        return [_canonical_value(item) for item in value]
    return value


def _without_row_sha(row: dict[str, Any]) -> dict[str, Any]:
    clone = copy.deepcopy(row)
    metadata = clone.get("metadata")
    if isinstance(metadata, dict):
        metadata.pop("row_sha256", None)
    return clone


def _read_scenarios(path: Path) -> list[_Scenario]:
    errors: list[str] = []
    rows = _read_jsonl(path, "source", errors)
    if errors:
        raise Tau3GroundedGenerationError("; ".join(errors))
    return [
        _Scenario(index=index, payload=row, row_sha256=canonical_sha256(row))
        for index, row in enumerate(rows)
    ]


def _read_jsonl(path: Path, label: str, errors: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        errors.append(f"{label} cannot be read: {exc}")
        return rows
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{label}:{index}: invalid JSON: {exc}")
            continue
        if not isinstance(value, dict):
            errors.append(f"{label}:{index}: row must be an object")
            continue
        rows.append(value)
    return rows


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise Tau3GroundedGenerationError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Tau3GroundedGenerationError(f"{label} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise Tau3GroundedGenerationError(f"{label} must be an object")
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_owner_only_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _write_owner_only_text(
        path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )


def _write_new_owner_only_jsonl_atomically(
    path: Path,
    rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Write nonempty JSONL to a same-filesystem temp and publish create-only."""

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    row_count = 0
    published = False
    try:
        os.fchmod(descriptor, OWNER_FILE_MODE)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            for row_count, row in enumerate(rows, 1):
                if not isinstance(row, dict):
                    raise Tau3GroundedGenerationError(
                        f"candidate source row {row_count} must be an object"
                    )
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            if row_count == 0:
                raise Tau3GroundedGenerationError(
                    "candidate source must contain at least one row"
                )
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise Tau3GroundedGenerationError(
                f"candidate source already exists: {path}"
            ) from exc
        published = True
        temporary.unlink()
        _fsync_directory(path.parent)
        record = _file_record(path, relative_to=path.parent)
        record["row_count"] = row_count
        return record
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
        if published and not path.is_file():
            raise Tau3GroundedGenerationError(
                "candidate source publication did not produce a regular file"
            )


def _write_owner_only_text(path: Path, text: str) -> None:
    if not path.parent.is_dir():
        raise Tau3GroundedGenerationError(f"secure output parent is missing: {path.parent}")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        OWNER_FILE_MODE,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _publish_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically rename a completed directory while refusing any target."""

    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        rename_exclusive = getattr(libc, "renamex_np", None)
        if rename_exclusive is None:
            raise Tau3GroundedGenerationError(
                "atomic no-replace directory publication is unavailable"
            )
        rename_exclusive.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename_exclusive.restype = ctypes.c_int
        result = rename_exclusive(source_bytes, destination_bytes, 0x00000004)
    elif sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        rename_no_replace = getattr(libc, "renameat2", None)
        if rename_no_replace is None:
            raise Tau3GroundedGenerationError(
                "atomic no-replace directory publication is unavailable"
            )
        rename_no_replace.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename_no_replace.restype = ctypes.c_int
        result = rename_no_replace(
            -100,
            source_bytes,
            -100,
            destination_bytes,
            0x00000001,
        )
    elif os.name == "nt":
        try:
            os.rename(source, destination)
        except FileExistsError as exc:
            raise Tau3GroundedGenerationError(
                f"output directory already exists: {destination}"
            ) from exc
        return
    else:
        raise Tau3GroundedGenerationError(
            "atomic no-replace directory publication is unsupported on this platform"
        )

    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise Tau3GroundedGenerationError(
            f"output directory already exists: {destination}"
        )
    raise Tau3GroundedGenerationError(
        "atomic no-replace directory publication failed: "
        + os.strerror(error_number)
    )


def _secure_mkdir(path: Path) -> None:
    if path.exists():
        if not path.is_dir() or path.is_symlink():
            raise Tau3GroundedGenerationError(f"secure artifact path is not a directory: {path}")
    else:
        path.mkdir(mode=OWNER_DIRECTORY_MODE)
    os.chmod(path, OWNER_DIRECTORY_MODE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_record(path: Path, *, relative_to: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "mode": f"{stat.S_IMODE(path.stat().st_mode):04o}",
    }


def _owner_only_path_errors(path: Path, label: str, *, directory: bool) -> list[str]:
    errors: list[str] = []
    try:
        path_stat = path.lstat()
    except OSError as exc:
        return [f"{label} permissions cannot be inspected: {exc}"]
    if stat.S_ISLNK(path_stat.st_mode):
        return [f"{label} must not be a symbolic link"]
    mode = stat.S_IMODE(path_stat.st_mode)
    if directory:
        if not stat.S_ISDIR(path_stat.st_mode):
            errors.append(f"{label} must be a directory")
        if mode != OWNER_DIRECTORY_MODE:
            errors.append(f"{label} owner directory permissions must be 0700")
    else:
        if not stat.S_ISREG(path_stat.st_mode):
            errors.append(f"{label} must be a regular file")
        if mode & ~OWNER_FILE_MODE:
            errors.append(f"{label} permissions are broader than 0600")
        if mode & OWNER_FILE_MODE == 0:
            errors.append(f"{label} owner file permissions must include read or write")
    return errors


def _bundle_artifact_path_errors(
    bundle: Path,
    path: Path,
    label: str,
    *,
    directory: bool,
) -> list[str]:
    if _path_has_prohibited_basename(path):
        return [f"{label} path contains a prohibited basename"]
    if path_has_symlink_component(path, include_leaf=True):
        return [f"{label} path must not contain symbolic link components"]
    try:
        bundle_resolved = bundle.resolve(strict=True)
        resolved = path.resolve(strict=True)
    except OSError as exc:
        return [f"{label} cannot be resolved: {exc}"]
    if resolved != bundle_resolved and bundle_resolved not in resolved.parents:
        return [f"{label} escapes bundle root"]
    errors: list[str] = []
    try:
        relative = path.relative_to(bundle)
    except ValueError:
        return [f"{label} is not lexically contained by bundle root"]
    current = bundle
    for component in relative.parts[:-1]:
        current = current / component
        errors.extend(
            _owner_only_path_errors(
                current,
                f"{label} directory",
                directory=True,
            )
        )
    errors.extend(_owner_only_path_errors(path, label, directory=directory))
    return errors


def _file_record_errors(path: Path, record: dict[str, Any], label: str) -> list[str]:
    errors = _owner_only_path_errors(path, label, directory=False)
    try:
        actual_mode = f"{stat.S_IMODE(path.stat().st_mode):04o}"
    except OSError:
        return errors
    if record.get("mode") != actual_mode:
        errors.append(f"{label}.mode does not replay")
    if record.get("bytes") != path.stat().st_size:
        errors.append(f"{label}.bytes does not replay")
    return errors


def _require_input_file(path: Path, label: str) -> None:
    if path_has_symlink_component(path, include_leaf=True):
        raise Tau3GroundedGenerationError(f"{label} path must not contain symlink components")
    if not path.is_file():
        raise Tau3GroundedGenerationError(f"{label} must be a regular file: {path}")


def _require_new_output_file(path: Path, label: str) -> None:
    if path.exists() or path.is_symlink():
        raise Tau3GroundedGenerationError(f"{label} already exists: {path}")
    if _path_has_prohibited_basename(path):
        raise Tau3GroundedGenerationError(
            f"{label} path contains a prohibited basename"
        )
    if path_has_symlink_component(path, include_leaf=True):
        raise Tau3GroundedGenerationError(
            f"{label} path must not contain symlink components"
        )
    if not path.parent.is_dir():
        raise Tau3GroundedGenerationError(f"{label} parent does not exist: {path.parent}")


def _require_owner_only_directory(path: Path, label: str) -> None:
    errors = _owner_only_path_errors(path, label, directory=True)
    if errors:
        raise Tau3GroundedGenerationError("; ".join(errors))


def _require_new_output_dir(out: Path) -> None:
    if out.exists() or out.is_symlink():
        raise Tau3GroundedGenerationError(f"output directory already exists: {out}")
    if path_has_symlink_component(out, include_leaf=True):
        raise Tau3GroundedGenerationError("output path must not contain symlink components")
    if not out.parent.is_dir():
        raise Tau3GroundedGenerationError(f"output parent does not exist: {out.parent}")


def _staging_dir(out: Path) -> Path:
    return out.parent / f".{out.name}.tmp-{os.getpid()}"


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_relative_path(path: str) -> bool:
    pure = PurePosixPath(path)
    return (
        bool(path)
        and not pure.is_absolute()
        and ".." not in pure.parts
        and not any(re.search(r"blind|sealed", part, re.IGNORECASE) for part in pure.parts)
    )


def _path_has_prohibited_basename(path: Path) -> bool:
    return any(re.search(r"blind|sealed", part, re.IGNORECASE) for part in path.parts)


def _object(value: Any, label: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{label} must be an object")
        return {}
    return value


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _object_required(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise Tau3GroundedGenerationError(f"{label} must be an object")
    return value


def _list_required(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise Tau3GroundedGenerationError(f"{label} must be a list")
    return value


def _required_string(value: dict[str, Any], field: str) -> str:
    item = value.get(field)
    if not isinstance(item, str) or not item:
        raise Tau3GroundedGenerationError(f"{field} must be a non-empty string")
    return item


def _review_record(payload: dict[str, Any], field: str) -> dict[str, Any]:
    value = payload.get(field)
    if isinstance(value, dict):
        return copy.deepcopy(value)
    return {"id": "unspecified", "sha256": canonical_sha256("unspecified")}


def _redaction(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("redaction")
    if isinstance(value, dict):
        return copy.deepcopy(value)
    return {"passed": False}


def _contamination(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("contamination")
    if isinstance(value, dict):
        return copy.deepcopy(value)
    raise Tau3GroundedGenerationError("contamination metadata is required")


def _contamination_errors(value: Any, split: str, context: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, dict):
        return [f"{context}.contamination must be an explicit object"]
    if value.get("raw_sealed_payload_read") is not False:
        errors.append(f"{context}.contamination.raw_sealed_payload_read must be false")
    if value.get("sealed_hash_only") is not True:
        errors.append(f"{context}.contamination.sealed_hash_only must be true")
    if value.get("source_split") != split:
        errors.append(f"{context}.contamination.source_split must match row split")
    return errors


def _validation_result(
    bundle: Path,
    passed: bool,
    errors: list[str],
    coverage: dict[str, Any],
    strict: bool,
    *,
    profile_id: str = LEGACY_COVERAGE_PROFILE_ID,
) -> dict[str, Any]:
    result = {
        "schema_version": "hfr.validation.v1",
        "target": str(bundle),
        "passed": passed,
        "strict": strict,
        "errors": errors,
        "coverage": coverage,
        "coverage_profile": _coverage_profile_record(profile_id),
        "error_count": len(errors),
        "coverage_blocker_count": len(_dict(coverage).get("blockers", [])),
    }
    result["validation_sha256"] = canonical_sha256(
        {key: value for key, value in result.items() if key != "validation_sha256"}
    )
    return result
