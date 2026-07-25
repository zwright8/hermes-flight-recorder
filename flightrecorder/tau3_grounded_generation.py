"""Grounded Tau-3 trajectory generation for training-side evidence.

This module is intentionally dependency-free and fail-closed.  It accepts only
training-side source fixtures whose tool calls can be replayed against cloned
deterministic state, then exports full parent trajectories plus safe corrected
per-decision targets.  Validation replays the tool calls again; hashes recorded
by the builder are evidence, not authority.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
import os
import re
import shutil
import sys
import types
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
) -> dict[str, Any]:
    """Build a grounded JSONL bundle from replayable train-side scenarios."""

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

    coverage = _coverage(rows_by_split)
    if strict_coverage and not coverage["passed"]:
        raise Tau3GroundedGenerationError(
            "coverage is incomplete: " + "; ".join(coverage["blockers"])
        )

    staging = _staging_dir(out)
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
        manifest = _manifest(source_path, staging, rows_by_split, coverage)
        _write_json(staging / "manifest.json", manifest)
        os.replace(staging, out)
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
    manifest_path = bundle / "manifest.json"
    try:
        manifest = _read_json(manifest_path, "manifest")
    except Tau3GroundedGenerationError as exc:
        return _validation_result(bundle, False, [str(exc)], {}, strict)
    if manifest.get("schema_version") != TAU3_GROUNDED_DATASET_SCHEMA_VERSION:
        errors.append("manifest schema_version mismatch")
    if manifest.get("lineage_id") != LINEAGE_ID:
        errors.append("manifest lineage_id mismatch")
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
        if not path.exists():
            errors.append(f"{split} file missing: {rel}")
            continue
        if record.get("sha256") != _sha256(path):
            errors.append(f"{split} file hash does not replay")
        rows = _read_jsonl(path, split, errors)
        rows_by_split[split] = rows
        for index, row in enumerate(rows):
            errors.extend(_validate_row(row, f"{split}[{index}]", bundle))

    errors.extend(_split_contamination_errors(rows_by_split))
    coverage = _coverage(rows_by_split)
    errors.extend(coverage["blockers"])
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
    return _validation_result(bundle, not errors, errors, coverage, strict)


def build_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="training-side scenario JSONL")
    parser.add_argument("--out-dir", required=True, help="new output bundle directory")
    parser.add_argument(
        "--allow-incomplete-coverage",
        action="store_true",
        help="write a blocked manifest instead of failing on missing family/behavior coverage",
    )
    args = parser.parse_args(argv)
    try:
        manifest = build_tau3_grounded_generation_dataset(
            source=args.source,
            out_dir=args.out_dir,
            strict_coverage=not args.allow_incomplete_coverage,
        )
    except Tau3GroundedGenerationError as exc:
        parser.exit(1, f"error: {exc}\n")
    print(json.dumps({"out_dir": args.out_dir, "passed": manifest["passed"]}, sort_keys=True))
    return 0


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
    parent_turns = copy.deepcopy(payload["turns"])
    tool_history: list[dict[str, Any]] = []
    training_targets: list[dict[str, Any]] = []
    for turn_index, turn in enumerate(parent_turns):
        assistant = _object_required(turn.get("assistant"), f"turns[{turn_index}].assistant")
        decision_ordinal = int(assistant["decision_ordinal"])
        safe_target = _target_for_decision(payload, assistant, decision_ordinal, turn_index)
        if safe_target is not None:
            training_targets.append(safe_target)
        for call_ordinal, raw_call in enumerate(_list_required(assistant.get("tool_calls"), f"turns[{turn_index}].assistant.tool_calls")):
            tool_history.append(
                _replay_call(
                    runtime,
                    raw_call,
                    tool_catalog=tool_catalog,
                    tool_catalog_hash=tool_catalog_hash,
                    parent_turn_ordinal=turn_index,
                    assistant_decision_ordinal=decision_ordinal,
                    tool_call_ordinal=call_ordinal,
                    prior_calls=tool_history,
                )
            )
    _assert_training_targets_grounded(training_targets, tool_catalog, tool_history)
    tool_exemptions = _tool_exemptions(payload, tool_catalog)
    tau_repo = _tau_repo_record(payload)
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
        "tool_catalog_sha256": tool_catalog_hash,
        "initial_state_sha256": canonical_sha256(payload["initial_state"]),
        "final_state_sha256": canonical_sha256(runtime.state),
        "behaviors": sorted({target["behavior"] for target in training_targets}),
        "recipe": _review_record(payload, "recipe"),
        "teacher": _review_record(payload, "teacher"),
        "reviewer": _review_record(payload, "reviewer"),
        "redaction": _redaction(payload),
        "contamination": _contamination(payload),
        "tool_exemptions": tool_exemptions,
    }
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
            "parent_assistant_decision_ordinal": decision_ordinal,
            "behavior": behavior,
            "negative_behavior": str(target.get("negative_behavior") or ""),
            "masked": True,
            "mask_reason": str(target.get("mask_reason") or "unsafe_or_negative_action"),
            "reviewed": True,
            "safe_correction_decision_ordinal": safe_decision,
            "canonical_target": canonical,
            "canonical_target_sha256": canonical_sha256(canonical),
        }
    if isinstance(tool_name, str) and _is_mutation_tool(tool_name) and target.get("requires_confirmation") is True:
        raise Tau3GroundedGenerationError(
            f"turns[{turn_index}] unsafe mutation target was not masked"
        )
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
        "parent_assistant_decision_ordinal": decision_ordinal,
        "behavior": behavior,
        "masked": False,
        "mask_reason": None,
        "canonical_target": canonical,
        "canonical_target_sha256": canonical_sha256(canonical),
    }


def _assert_training_targets_grounded(
    targets: list[dict[str, Any]],
    tool_catalog: list[dict[str, Any]],
    tool_history: list[dict[str, Any]],
) -> None:
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
        bound = any(
            isinstance(call, dict)
            and call.get("tool_name") == tool_name
            and call.get("canonical_arguments") == args
            and call.get("evidence_replayed") is True
            for call in replay
        )
        if not bound:
            errors.append(
                f"{context}.training_targets[{index}] target tool call is not exactly bound to replayed evidence"
            )
    return errors


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
        if reason == "zero_arg" and _required_arg_count(tool_def) != 0:
            errors.append(f"{label}.reason zero_arg requires a runtime tool with zero required arguments")
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
    pre_state = copy.deepcopy(runtime.state)
    pre_hash = canonical_sha256(pre_state)
    result_class = "success"
    exception = None
    try:
        result = runtime.call(tool_name, copy.deepcopy(args))
        if result is None or result == [] or result == {}:
            result_class = "empty"
    except Exception as exc:  # deliberate: tool exceptions are evidence.
        result = {"error": exc.__class__.__name__, "message": str(exc)}
        result_class = "exception"
        exception = {
            "type": exc.__class__.__name__,
            "message": str(exc),
        }
    if isinstance(result, dict) and result.get("error") and result_class != "exception":
        result_class = "error"
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
    return {
        "parent_turn_ordinal": parent_turn_ordinal,
        "parent_assistant_decision_ordinal": assistant_decision_ordinal,
        "tool_call_ordinal": tool_call_ordinal,
        "tool_name": tool_name,
        "tool_definition_sha256": canonical_sha256(tool_def),
        "tool_catalog_sha256": tool_catalog_hash,
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
        "context": {
            "empty_result": result_class == "empty",
            "error_result": result_class in {"error", "exception"},
            "repeated_call_prior_count": repeated_prior,
            "repeated_call": repeated_prior > 0,
        },
        "evidence_replayed": True,
    }


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
    for field in ("system_prompt_sha256", "tool_catalog_sha256", "initial_state_sha256", "final_state_sha256", "row_sha256"):
        if not SHA256_RE.fullmatch(str(metadata.get(field) or "")):
            errors.append(f"{context}.metadata.{field} must be sha256")
    errors.extend(
        _contamination_errors(
            metadata.get("contamination"),
            str(metadata.get("split") or ""),
            f"{context}.metadata",
        )
    )

    trajectory = _object(row.get("trajectory"), f"{context}.trajectory", errors)
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
    if "initial_state" in row:
        errors.append(f"{context}.initial_state must not be embedded; use initial_state_ref")
    initial_state = _load_state_ref(bundle, row.get("initial_state_ref"), context, errors)
    if canonical_sha256(initial_state) != metadata.get("initial_state_sha256"):
        errors.append(f"{context}.initial_state_sha256 does not replay")
    if canonical_sha256(_without_row_sha(row)) != metadata.get("row_sha256"):
        errors.append(f"{context}.row_sha256 does not replay")

    targets = row.get("training_targets")
    if not isinstance(targets, list) or not targets:
        errors.append(f"{context}.training_targets must be non-empty")
    else:
        for index, target in enumerate(targets):
            errors.extend(_validate_target(target, f"{context}.training_targets[{index}]"))
        errors.extend(_masked_correction_link_errors(targets, context))

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
    errors.extend(
        _tool_exemption_errors(
            metadata.get("tool_exemptions"),
            tool_catalog,
            f"{context}.metadata.tool_exemptions",
        )
    )
    prior_calls: list[dict[str, Any]] = []
    for index, recorded in enumerate(replay):
        if not isinstance(recorded, dict):
            errors.append(f"{context}.tool_replay[{index}] must be an object")
            continue
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
        except Exception as exc:
            errors.append(f"{context}.tool_replay[{index}] cannot replay: {exc}")
            continue
        for field in (
            "tool_name",
            "tool_definition_sha256",
            "tool_catalog_sha256",
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
            "context",
        ):
            if recorded.get(field) != actual.get(field):
                errors.append(f"{context}.tool_replay[{index}].{field} does not replay")
        if recorded.get("evidence_replayed") is not True:
            errors.append(f"{context}.tool_replay[{index}].evidence_replayed must be true")
        prior_calls.append(actual)
    if runtime is not None and canonical_sha256(runtime.state) != metadata.get("final_state_sha256"):
        errors.append(f"{context}.metadata.final_state_sha256 does not replay")
    errors.extend(_target_binding_errors(targets, tool_catalog, replay, context))
    errors.extend(_completion_claim_errors(row, context))
    return errors


def _validate_target(target: Any, context: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(target, dict):
        return [f"{context} must be an object"]
    if target.get("behavior") not in BEHAVIORS:
        errors.append(f"{context}.behavior is not in rubric")
    if not isinstance(target.get("parent_assistant_decision_ordinal"), int):
        errors.append(f"{context}.parent_assistant_decision_ordinal must be an integer")
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
        ]
        if len(safe) != 1:
            errors.append(
                f"{context}.training_targets[{index}] must link to exactly one later unmasked correction"
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
                "name": "empty_search",
                "description": "TEST ONLY: return an empty result.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "get_record",
                "description": "TEST ONLY: read one fake record.",
                "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
            },
            {
                "name": "raise_tool_exception",
                "description": "TEST ONLY: raise an exception.",
                "parameters": {"type": "object", "properties": {"message": {"type": "string"}}, "required": []},
            },
            {
                "name": "update_record",
                "description": "TEST ONLY: patch one fake record.",
                "parameters": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}, "patch": {"type": "object"}},
                    "required": ["id", "patch"],
                },
            },
        ]


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
        if _git(repo_path, "status", "--porcelain=v1"):
            raise Tau3GroundedGenerationError("vendored Tau checkout must be clean for immutable replay")
        src = repo_path / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        _install_tau_import_shims()
        try:
            data_module = importlib.import_module(f"tau2.domains.{domain}.data_model")
            tools_module = importlib.import_module(f"tau2.domains.{domain}.tools")
        except Exception as exc:
            raise Tau3GroundedGenerationError(f"cannot import vendored Tau {domain} tools: {exc}") from exc
        db_class_name = {
            "airline": "FlightDB",
            "retail": "RetailDB",
            "telecom": "TelecomDB",
        }[domain]
        tools_class_name = {
            "airline": "AirlineTools",
            "retail": "RetailTools",
            "telecom": "TelecomTools",
        }[domain]
        try:
            db = getattr(data_module, db_class_name).model_validate(copy.deepcopy(state))
            self.toolkit = getattr(tools_module, tools_class_name)(db)
        except Exception as exc:
            raise Tau3GroundedGenerationError(
                f"cannot instantiate vendored Tau {domain} state/tools: {exc}"
            ) from exc

    @property
    def state(self) -> dict[str, Any]:
        db = self.toolkit.db
        if db is None:
            raise Tau3GroundedGenerationError("vendored Tau toolkit has no database")
        value = _model_to_json(db)
        if not isinstance(value, dict):
            raise Tau3GroundedGenerationError("vendored Tau DB did not serialize to an object")
        return value

    def call(self, tool_name: str, args: dict[str, Any]) -> Any:
        return _model_to_json(self.toolkit.use_tool(tool_name, **copy.deepcopy(args)))

    def tool_catalog(self) -> list[dict[str, Any]]:
        tools = [_model_to_json(tool) for tool in self.toolkit.get_tools().values()]
        return sorted(tools, key=lambda item: str(_dict(item).get("name") or ""))


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


def _coverage(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
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


def _manifest(
    source_path: Path,
    out: Path,
    rows_by_split: dict[str, list[dict[str, Any]]],
    coverage: dict[str, Any],
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
    manifest["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    return manifest


def _externalize_state_snapshots(
    rows_by_split: dict[str, list[dict[str, Any]]],
    out: Path,
) -> None:
    states_dir = out / "states"
    written: set[str] = set()
    for rows in rows_by_split.values():
        for row in rows:
            state = row.pop("initial_state", None)
            if not isinstance(state, dict):
                raise Tau3GroundedGenerationError("internal row missing initial_state before externalization")
            state_sha = canonical_sha256(state)
            if state_sha != _dict(row.get("metadata")).get("initial_state_sha256"):
                raise Tau3GroundedGenerationError("initial state hash changed before externalization")
            state_path = states_dir / f"{state_sha}.json"
            if state_sha not in written:
                _write_json(state_path, state)
                written.add(state_sha)
            row["initial_state_ref"] = _file_record(state_path, relative_to=out)
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
    try:
        resolved = path.resolve(strict=True)
        bundle_resolved = bundle.resolve(strict=True)
    except OSError as exc:
        errors.append(f"{context}.initial_state_ref cannot be resolved: {exc}")
        return {}
    if bundle_resolved not in resolved.parents:
        errors.append(f"{context}.initial_state_ref escapes bundle root")
        return {}
    if path_has_symlink_component(path, include_leaf=True):
        errors.append(f"{context}.initial_state_ref must not traverse symlinks")
        return {}
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_record(path: Path, *, relative_to: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _require_input_file(path: Path, label: str) -> None:
    if path_has_symlink_component(path, include_leaf=True):
        raise Tau3GroundedGenerationError(f"{label} path must not contain symlink components")
    if not path.is_file():
        raise Tau3GroundedGenerationError(f"{label} must be a regular file: {path}")


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
    return bool(path) and not pure.is_absolute() and ".." not in pure.parts


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
) -> dict[str, Any]:
    return {
        "schema_version": "hfr.validation.v1",
        "target": str(bundle),
        "passed": passed,
        "strict": strict,
        "errors": errors,
        "coverage": coverage,
    }
