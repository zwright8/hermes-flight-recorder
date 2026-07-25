"""Objective-validity artifacts for Tau-3 v3 training exports.

The validator is dependency-free and fail-closed. It proves that a training
JSONL export supervises exactly the eligible assistant decisions recorded in a
content-addressed parent trajectory JSONL file, while masking all non-target
and unsafe token classes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TAU3_OBJECTIVE_VALIDITY_SCHEMA_VERSION = "hfr.tau3_objective_validity.v1"
TAU3_DOMAINS = ("airline", "retail", "telecom")

PROTECTED_TOKEN_CLASSES = (
    "prompt",
    "tool_result",
    "user",
    "private_reference",
    "grader",
    "negative_action",
)

SOURCE_REF_KEYS = (
    "path",
    "size",
    "sha256",
    "row_count",
)


class Tau3ObjectiveValidityError(ValueError):
    """Raised when objective-validity evidence cannot be built or replayed."""


@dataclass
class _Context:
    training_export: Path
    parent_trajectories: Path
    checks: list[dict[str, Any]] = field(default_factory=list)

    def add(self, check_id: str, passed: bool, actual: Any = None, expected: Any = None, detail: str | None = None) -> None:
        item: dict[str, Any] = {
            "id": check_id,
            "passed": bool(passed),
            "actual": _json_safe(actual),
            "expected": _json_safe(expected),
        }
        if detail:
            item["detail"] = detail
        self.checks.append(item)


def build_tau3_objective_validity_report(
    *,
    training_export_path: str | Path,
    parent_trajectories_path: str | Path,
    source_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic objective-validity report from source JSONL files."""

    ctx = _Context(Path(training_export_path), Path(parent_trajectories_path))
    training_rows = _read_jsonl(ctx, ctx.training_export, "training export")
    parent_rows = _read_jsonl(ctx, ctx.parent_trajectories, "parent trajectories")
    report_root = Path(source_root) if source_root is not None else None

    parent_index: dict[str, dict[str, Any]] = {}
    parent_decisions: dict[tuple[str, int], dict[str, Any]] = {}
    eligible_keys: set[tuple[str, int]] = set()
    duplicate_parent_ids: list[str] = []
    duplicate_decisions: list[str] = []
    ordinal_gap_trajectories: list[str] = []
    parent_domains: set[str] = set()
    training_domains: set[str] = set()
    system_prompt_hashes_by_domain: dict[str, set[str]] = {domain: set() for domain in TAU3_DOMAINS}
    tool_catalog_hashes_by_domain: dict[str, set[str]] = {domain: set() for domain in TAU3_DOMAINS}

    for parent in parent_rows:
        trajectory_id = _require_non_empty_string(parent, "trajectory_id")
        domain = _domain_value(parent.get("domain"))
        if domain is None:
            ctx.add(f"parent.{trajectory_id}.domain_valid", False, parent.get("domain"), list(TAU3_DOMAINS))
            domain_key = "<invalid>"
        else:
            ctx.add(f"parent.{trajectory_id}.domain_valid", True, domain, list(TAU3_DOMAINS))
            domain_key = domain
            parent_domains.add(domain)
        parent_hash = parent_trajectory_sha256(parent)
        if trajectory_id in parent_index:
            duplicate_parent_ids.append(trajectory_id)
        parent_index[trajectory_id] = {"row": parent, "sha256": parent_hash, "domain": domain}
        _collect_hash_by_domain(system_prompt_hashes_by_domain, domain_key, parent, "system_prompt_sha256")
        _collect_hash_by_domain(tool_catalog_hashes_by_domain, domain_key, parent, "ordered_tool_catalog_sha256")
        decisions = parent.get("assistant_decisions")
        if not isinstance(decisions, list) or not decisions:
            ctx.add(f"parent.{trajectory_id}.assistant_decisions_present", False, _type_name(decisions), "non-empty list")
            continue
        ordinals: list[int] = []
        for decision in decisions:
            if not isinstance(decision, dict):
                ctx.add(f"parent.{trajectory_id}.decision_object", False, _type_name(decision), "object")
                continue
            ordinal = decision.get("decision_ordinal")
            if not isinstance(ordinal, int) or ordinal < 0:
                ctx.add(
                    f"parent.{trajectory_id}.decision_ordinal_valid",
                    False,
                    ordinal,
                    "non-negative integer",
                )
                continue
            ordinals.append(ordinal)
            key = (trajectory_id, ordinal)
            if key in parent_decisions:
                duplicate_decisions.append(f"{trajectory_id}:{ordinal}")
            parent_decisions[key] = decision
            if decision.get("eligible_for_supervision") is True:
                eligible_keys.add(key)
        if sorted(ordinals) != sorted(set(ordinals)):
            ordinal_gap_trajectories.append(trajectory_id)

    ctx.add("parent_trajectory_ids_unique", not duplicate_parent_ids, duplicate_parent_ids, [])
    ctx.add("parent_decision_ordinals_unique", not duplicate_decisions, duplicate_decisions, [])
    ctx.add("parent_decision_ordinals_preserved_unique", not ordinal_gap_trajectories, ordinal_gap_trajectories, [])

    row_index: dict[tuple[str, int], dict[str, Any]] = {}
    duplicate_rows: list[str] = []
    supervision_index: list[dict[str, Any]] = []
    negative_prefix_count = 0
    truncation_count = 0

    for line_no, row in enumerate(training_rows, start=1):
        row_id = _require_non_empty_string(row, "row_id")
        trajectory_id = _require_non_empty_string(row, "trajectory_id")
        domain = _domain_value(row.get("domain"))
        if domain is None:
            ctx.add(f"row.{row_id}.domain_valid", False, row.get("domain"), list(TAU3_DOMAINS))
            domain_key = "<invalid>"
        else:
            ctx.add(f"row.{row_id}.domain_valid", True, domain, list(TAU3_DOMAINS))
            domain_key = domain
            training_domains.add(domain)
        ordinal = row.get("decision_ordinal")
        key = (trajectory_id, ordinal) if isinstance(ordinal, int) else (trajectory_id, -1)
        if key in row_index:
            duplicate_rows.append(f"{trajectory_id}:{ordinal}")
        row_index[key] = row
        _collect_hash_by_domain(system_prompt_hashes_by_domain, domain_key, row, "system_prompt_sha256")
        _collect_hash_by_domain(tool_catalog_hashes_by_domain, domain_key, row, "ordered_tool_catalog_sha256")
        parent_record = parent_index.get(trajectory_id)
        parent_decision = parent_decisions.get(key)
        if row.get("negative_prefix") is True:
            negative_prefix_count += 1
        row_result = _validate_training_row(
            ctx,
            row=row,
            line_no=line_no,
            parent_record=parent_record,
            parent_decision=parent_decision,
        )
        truncation_count += 0 if row_result.get("complete_message") else 1
        supervision_index.append(
            {
                "row_id": row_id,
                "trajectory_id": trajectory_id,
                "domain": domain,
                "decision_ordinal": ordinal,
                "parent_trajectory_sha256": row.get("parent_trajectory_sha256"),
                "target_sha256": row.get("target_sha256"),
                "negative_prefix": bool(row.get("negative_prefix") is True),
            }
        )

    missing = sorted(f"{trajectory_id}:{ordinal}" for trajectory_id, ordinal in eligible_keys - set(row_index))
    extra = sorted(f"{trajectory_id}:{ordinal}" for trajectory_id, ordinal in set(row_index) - eligible_keys)
    ctx.add("training_rows_unique_per_decision", not duplicate_rows, duplicate_rows, [])
    ctx.add("every_eligible_assistant_decision_supervised", not missing, missing, [])
    ctx.add("no_ineligible_assistant_decision_supervised", not extra, extra, [])
    ctx.add("parent_domains_complete", parent_domains == set(TAU3_DOMAINS), sorted(parent_domains), list(TAU3_DOMAINS))
    ctx.add("training_domains_complete", training_domains == set(TAU3_DOMAINS), sorted(training_domains), list(TAU3_DOMAINS))
    system_prompt_sha256_by_domain = _stable_domain_hash_map(system_prompt_hashes_by_domain)
    ordered_tool_catalog_sha256_by_domain = _stable_domain_hash_map(tool_catalog_hashes_by_domain)
    ctx.add(
        "system_prompt_hash_stable_by_domain",
        system_prompt_sha256_by_domain is not None,
        _json_safe(system_prompt_hashes_by_domain),
        {domain: "exactly one sha256" for domain in TAU3_DOMAINS},
    )
    ctx.add(
        "ordered_tool_catalog_hash_stable_by_domain",
        ordered_tool_catalog_sha256_by_domain is not None,
        _json_safe(tool_catalog_hashes_by_domain),
        {domain: "exactly one sha256" for domain in TAU3_DOMAINS},
    )

    failed = [check for check in ctx.checks if not check["passed"]]
    return {
        "schema_version": TAU3_OBJECTIVE_VALIDITY_SCHEMA_VERSION,
        "passed": not failed,
        "status": "passed" if not failed else "failed",
        "summary": (
            "Tau-3 objective-validity report passed replayable supervision and masking checks."
            if not failed
            else f"Tau-3 objective-validity report failed {len(failed)} check(s)."
        ),
        "sources": {
            "training_export": _source_record(ctx.training_export, len(training_rows), report_root),
            "parent_trajectories": _source_record(ctx.parent_trajectories, len(parent_rows), report_root),
        },
        "system_prompt_sha256_by_domain": system_prompt_sha256_by_domain,
        "ordered_tool_catalog_sha256_by_domain": ordered_tool_catalog_sha256_by_domain,
        "eligible_decision_count": len(eligible_keys),
        "supervised_row_count": len(training_rows),
        "negative_prefix_count": negative_prefix_count,
        "complete_message_truncation_count": truncation_count,
        "check_count": len(ctx.checks),
        "failed_check_count": len(failed),
        "checks": ctx.checks,
        "supervision_index": sorted(
            supervision_index,
            key=lambda item: (str(item.get("trajectory_id")), int(item.get("decision_ordinal") or -1), str(item.get("row_id"))),
        ),
    }


def write_tau3_objective_validity_report(
    *,
    training_export_path: str | Path,
    parent_trajectories_path: str | Path,
    out_path: str | Path,
) -> dict[str, Any]:
    """Build and write a deterministic objective-validity JSON artifact."""

    target = Path(out_path)
    report = build_tau3_objective_validity_report(
        training_export_path=training_export_path,
        parent_trajectories_path=parent_trajectories_path,
        source_root=target.parent,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def validate_tau3_objective_validity_report(path: str | Path) -> dict[str, Any]:
    """Replay a saved objective-validity report and compare it to its sources."""

    report_path = Path(path)
    errors: list[str] = []
    try:
        saved = _read_json_object(report_path, "objective-validity report")
    except Tau3ObjectiveValidityError as exc:
        return _validation_result(False, str(report_path), [str(exc)])
    if saved.get("schema_version") != TAU3_OBJECTIVE_VALIDITY_SCHEMA_VERSION:
        errors.append(
            "schema_version must be "
            f"{TAU3_OBJECTIVE_VALIDITY_SCHEMA_VERSION!r}; got {saved.get('schema_version')!r}"
        )
    sources = saved.get("sources")
    if not isinstance(sources, dict):
        errors.append("sources must be an object")
        return _validation_result(False, str(report_path), errors)
    try:
        training_export = _resolve_source_ref(report_path.parent, sources.get("training_export"), "training_export")
        parent_trajectories = _resolve_source_ref(report_path.parent, sources.get("parent_trajectories"), "parent_trajectories")
        replayed = build_tau3_objective_validity_report(
            training_export_path=training_export,
            parent_trajectories_path=parent_trajectories,
            source_root=report_path.parent,
        )
    except Tau3ObjectiveValidityError as exc:
        errors.append(str(exc))
        return _validation_result(False, str(report_path), errors)

    comparable_saved = _strip_replay_volatile(saved)
    comparable_replayed = _strip_replay_volatile(replayed)
    if comparable_saved != comparable_replayed:
        errors.append("saved objective-validity report does not match deterministic replay")
    if replayed.get("passed") is not True:
        errors.append("replayed objective-validity checks failed")
    return {
        "schema_version": "hfr.tau3_objective_validity_validation.v1",
        "passed": not errors,
        "status": "passed" if not errors else "failed",
        "artifact_path": str(report_path),
        "replayed_passed": bool(replayed.get("passed")),
        "error_count": len(errors),
        "errors": errors,
        "replayed_failed_check_count": replayed.get("failed_check_count", 0),
    }


def parent_trajectory_sha256(parent_row: dict[str, Any]) -> str:
    """Return the content address for a parent trajectory row."""

    payload = {key: value for key, value in parent_row.items() if key not in {"sha256", "content_sha256"}}
    return _canonical_sha256(payload)


def _validate_training_row(
    ctx: _Context,
    *,
    row: dict[str, Any],
    line_no: int,
    parent_record: dict[str, Any] | None,
    parent_decision: dict[str, Any] | None,
) -> dict[str, Any]:
    row_id = str(row.get("row_id") or f"line:{line_no}")
    ordinal = row.get("decision_ordinal")
    target_text = row.get("target_text")
    target_sha256 = row.get("target_sha256")
    token_accounting = row.get("token_accounting")
    target_boundaries = row.get("target_boundaries")
    token_class_counts = row.get("token_class_counts")
    masked_token_class_counts = row.get("masked_token_class_counts")

    ctx.add(f"row.{row_id}.decision_ordinal_valid", isinstance(ordinal, int) and ordinal >= 0, ordinal, "non-negative integer")
    ctx.add(f"row.{row_id}.parent_exists", parent_record is not None, row.get("trajectory_id"), "known parent trajectory")
    if parent_record is not None:
        ctx.add(
            f"row.{row_id}.domain_matches_parent",
            row.get("domain") == parent_record.get("domain"),
            row.get("domain"),
            parent_record.get("domain"),
        )
        ctx.add(
            f"row.{row_id}.parent_hash_matches",
            row.get("parent_trajectory_sha256") == parent_record["sha256"],
            row.get("parent_trajectory_sha256"),
            parent_record["sha256"],
        )
    ctx.add(f"row.{row_id}.parent_decision_exists", parent_decision is not None, ordinal, "eligible parent decision")
    ctx.add(f"row.{row_id}.supervised_decision_true", row.get("supervised_decision") is True, row.get("supervised_decision"), True)
    ctx.add(f"row.{row_id}.target_text_present", isinstance(target_text, str) and bool(target_text), _type_name(target_text), "non-empty string")
    computed_target_sha = _canonical_sha256(target_text) if isinstance(target_text, str) else None
    ctx.add(f"row.{row_id}.target_sha256_matches_text", target_sha256 == computed_target_sha, target_sha256, computed_target_sha)
    if parent_decision is not None:
        ctx.add(
            f"row.{row_id}.target_sha256_matches_parent",
            target_sha256 == parent_decision.get("target_sha256"),
            target_sha256,
            parent_decision.get("target_sha256"),
        )
    ctx.add(
        f"row.{row_id}.system_prompt_hash_valid",
        _is_sha256(row.get("system_prompt_sha256")),
        row.get("system_prompt_sha256"),
        "sha256",
    )
    ctx.add(
        f"row.{row_id}.ordered_tool_catalog_hash_valid",
        _is_sha256(row.get("ordered_tool_catalog_sha256")),
        row.get("ordered_tool_catalog_sha256"),
        "sha256",
    )
    complete_message = False
    if isinstance(target_boundaries, dict):
        complete_message = target_boundaries.get("complete_message") is True and target_boundaries.get("truncated") is False
        ctx.add(f"row.{row_id}.target_boundaries_complete_message", complete_message, target_boundaries, "complete untruncated target")
    else:
        ctx.add(f"row.{row_id}.target_boundaries_present", False, _type_name(target_boundaries), "object")
    _validate_token_accounting(
        ctx,
        row_id=row_id,
        row=row,
        token_accounting=token_accounting,
        target_boundaries=target_boundaries,
        token_class_counts=token_class_counts,
        masked_token_class_counts=masked_token_class_counts,
        negative_prefix=row.get("negative_prefix") is True,
    )
    if row.get("negative_prefix") is True:
        ctx.add(
            f"row.{row_id}.negative_prefix_targets_safe_correction",
            row.get("target_kind") == "safe_correction",
            row.get("target_kind"),
            "safe_correction",
        )
        if parent_decision is not None:
            ctx.add(
                f"row.{row_id}.parent_requires_safe_correction",
                parent_decision.get("safe_correction_required") is True,
                parent_decision.get("safe_correction_required"),
                True,
            )
    return {"complete_message": complete_message}


def _validate_token_accounting(
    ctx: _Context,
    *,
    row_id: str,
    row: dict[str, Any],
    token_accounting: Any,
    target_boundaries: Any,
    token_class_counts: Any,
    masked_token_class_counts: Any,
    negative_prefix: bool,
) -> None:
    if not isinstance(token_accounting, dict):
        ctx.add(f"row.{row_id}.token_accounting_present", False, _type_name(token_accounting), "object")
        return
    if not isinstance(target_boundaries, dict):
        return
    if not isinstance(token_class_counts, dict):
        ctx.add(f"row.{row_id}.token_class_counts_present", False, _type_name(token_class_counts), "object")
        return
    if not isinstance(masked_token_class_counts, dict):
        ctx.add(f"row.{row_id}.masked_token_class_counts_present", False, _type_name(masked_token_class_counts), "object")
        return

    prompt_tokens = _int_value(token_accounting.get("prompt_tokens"))
    target_tokens = _int_value(token_accounting.get("target_tokens"))
    total_tokens = _int_value(token_accounting.get("total_tokens"))
    masked_tokens = _int_value(token_accounting.get("masked_tokens"))
    supervised_tokens = _int_value(token_accounting.get("supervised_tokens"))
    start_token = _int_value(target_boundaries.get("start_token"))
    end_token = _int_value(target_boundaries.get("end_token"))

    required_numbers = {
        "prompt_tokens": prompt_tokens,
        "target_tokens": target_tokens,
        "total_tokens": total_tokens,
        "masked_tokens": masked_tokens,
        "supervised_tokens": supervised_tokens,
        "start_token": start_token,
        "end_token": end_token,
    }
    invalid = {key: value for key, value in required_numbers.items() if value is None or value < 0}
    ctx.add(f"row.{row_id}.token_numbers_non_negative_integers", not invalid, invalid, {})
    if invalid:
        return
    assert prompt_tokens is not None
    assert target_tokens is not None
    assert total_tokens is not None
    assert masked_tokens is not None
    assert supervised_tokens is not None
    assert start_token is not None
    assert end_token is not None

    class_total = _sum_counts(token_class_counts)
    masked_class_total = _sum_counts(masked_token_class_counts)
    protected_total = sum(_count(token_class_counts, name) for name in PROTECTED_TOKEN_CLASSES)
    protected_masked_total = sum(_count(masked_token_class_counts, name) for name in PROTECTED_TOKEN_CLASSES)
    input_token_ids = row.get("input_token_ids")
    loss_mask = row.get("loss_mask")

    ctx.add(f"row.{row_id}.token_total_consistent", prompt_tokens + target_tokens == total_tokens, prompt_tokens + target_tokens, total_tokens)
    ctx.add(f"row.{row_id}.target_boundary_matches_tokens", start_token == prompt_tokens and end_token - start_token == target_tokens, {"start": start_token, "end": end_token}, {"start": prompt_tokens, "length": target_tokens})
    ctx.add(f"row.{row_id}.loss_tokens_match_target", supervised_tokens == target_tokens, supervised_tokens, target_tokens)
    ctx.add(f"row.{row_id}.prompt_tokens_masked", masked_tokens == prompt_tokens, masked_tokens, prompt_tokens)
    ctx.add(f"row.{row_id}.token_class_total_matches", class_total == total_tokens, class_total, total_tokens)
    ctx.add(f"row.{row_id}.masked_class_total_matches", masked_class_total == masked_tokens, masked_class_total, masked_tokens)
    ctx.add(f"row.{row_id}.protected_tokens_fully_masked", protected_masked_total == protected_total, protected_masked_total, protected_total)
    token_arrays_valid = (
        isinstance(input_token_ids, list)
        and isinstance(loss_mask, list)
        and all(isinstance(token, int) for token in input_token_ids)
        and all(mask in (0, 1) for mask in loss_mask)
        and len(input_token_ids) == total_tokens
        and len(loss_mask) == max(0, total_tokens - 1)
    )
    ctx.add(
        f"row.{row_id}.exact_token_ids_and_loss_mask_present",
        token_arrays_valid,
        {
            "input_token_ids": _type_name(input_token_ids),
            "loss_mask": _type_name(loss_mask),
            "input_len": len(input_token_ids) if isinstance(input_token_ids, list) else None,
            "mask_len": len(loss_mask) if isinstance(loss_mask, list) else None,
        },
        {"input_length": total_tokens, "loss_mask_length": max(0, total_tokens - 1), "mask_values": [0, 1]},
    )
    if token_arrays_valid:
        assert isinstance(input_token_ids, list)
        assert isinstance(loss_mask, list)
        prompt_mask = loss_mask[: max(0, prompt_tokens - 1)]
        target_mask = loss_mask[max(0, prompt_tokens - 1):]
        ctx.add(
            f"row.{row_id}.loss_mask_exact_mlx_shifted_prompt_target_boundary",
            all(mask == 0 for mask in prompt_mask) and all(mask == 1 for mask in target_mask),
            {"prompt": prompt_mask[:8], "target": target_mask[:8], "tail": loss_mask[-8:]},
            "shifted prompt zeros and target ones",
        )
        ctx.add(
            f"row.{row_id}.loss_mask_semantics_mlx_shifted",
            row.get("loss_mask_semantics") == "mlx_lm_shifted_targets_v1",
            row.get("loss_mask_semantics"),
            "mlx_lm_shifted_targets_v1",
        )
        ctx.add(
            f"row.{row_id}.loss_mask_supervised_count_matches_target",
            sum(loss_mask) == target_tokens,
            sum(loss_mask),
            target_tokens,
        )
        ctx.add(
            f"row.{row_id}.input_token_ids_sha256_matches",
            row.get("input_token_ids_sha256") == _canonical_sha256(input_token_ids),
            row.get("input_token_ids_sha256"),
            _canonical_sha256(input_token_ids),
        )
        ctx.add(
            f"row.{row_id}.loss_mask_sha256_matches",
            row.get("loss_mask_sha256") == _canonical_sha256(loss_mask),
            row.get("loss_mask_sha256"),
            _canonical_sha256(loss_mask),
        )
    for name in PROTECTED_TOKEN_CLASSES:
        ctx.add(
            f"row.{row_id}.masked_class.{name}",
            _count(masked_token_class_counts, name) == _count(token_class_counts, name),
            _count(masked_token_class_counts, name),
            _count(token_class_counts, name),
        )
    if negative_prefix:
        ctx.add(
            f"row.{row_id}.negative_action_tokens_present_and_masked",
            _count(token_class_counts, "negative_action") > 0
            and _count(masked_token_class_counts, "negative_action") == _count(token_class_counts, "negative_action"),
            {
                "negative_action": _count(token_class_counts, "negative_action"),
                "masked_negative_action": _count(masked_token_class_counts, "negative_action"),
            },
            "positive equal counts",
        )


def _source_record(path: Path, row_count: int, root: Path | None) -> dict[str, Any]:
    source_path = path.resolve()
    if root is None:
        ref = source_path.name
    else:
        root_path = root.resolve()
        try:
            ref = source_path.relative_to(root_path)
        except ValueError as exc:
            raise Tau3ObjectiveValidityError(
                f"source path must be inside objective-validity report directory: {source_path}"
            ) from exc
    return {
        "path": Path(ref).as_posix(),
        "size": path.stat().st_size,
        "sha256": _sha256(path),
        "row_count": row_count,
    }


def _resolve_source_ref(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, dict):
        raise Tau3ObjectiveValidityError(f"sources.{label} must be an object")
    missing = [key for key in SOURCE_REF_KEYS if key not in value]
    if missing:
        raise Tau3ObjectiveValidityError(f"sources.{label} is missing {missing[0]}")
    ref = value.get("path")
    if not isinstance(ref, str) or not ref:
        raise Tau3ObjectiveValidityError(f"sources.{label}.path must be a non-empty relative path")
    ref_path = Path(ref)
    if ref_path.is_absolute():
        raise Tau3ObjectiveValidityError(f"sources.{label}.path must not be absolute")
    if ".." in ref_path.parts:
        raise Tau3ObjectiveValidityError(f"sources.{label}.path must not contain '..'")
    root_path = root.resolve()
    path = (root_path / ref_path).resolve()
    try:
        path.relative_to(root_path)
    except ValueError as exc:
        raise Tau3ObjectiveValidityError(f"sources.{label}.path resolves outside report directory") from exc
    if not path.is_file():
        raise Tau3ObjectiveValidityError(f"sources.{label}.path not found: {ref}")
    size = path.stat().st_size
    digest = _sha256(path)
    rows = len(_read_jsonl(_Context(path, path), path, label))
    if value.get("sha256") != digest:
        raise Tau3ObjectiveValidityError(f"sources.{label}.sha256 mismatch")
    if value.get("size") != size:
        raise Tau3ObjectiveValidityError(f"sources.{label}.size mismatch")
    if value.get("row_count") != rows:
        raise Tau3ObjectiveValidityError(f"sources.{label}.row_count mismatch")
    return path


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise Tau3ObjectiveValidityError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise Tau3ObjectiveValidityError(f"invalid JSON in {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise Tau3ObjectiveValidityError(f"{label} must contain a JSON object: {path}")
    return payload


def _read_jsonl(ctx: _Context, path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise Tau3ObjectiveValidityError(f"{label} not found: {path}") from exc
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Tau3ObjectiveValidityError(f"invalid JSON in {label} {path}:{line_no}: {exc}") from exc
        if not isinstance(value, dict):
            raise Tau3ObjectiveValidityError(f"{label} row {line_no} must be an object")
        rows.append(value)
    if not rows:
        raise Tau3ObjectiveValidityError(f"{label} is empty: {path}")
    ctx.add(f"{label.replace(' ', '_')}_jsonl_readable", True, path.name, "non-empty JSONL objects")
    return rows


def _require_non_empty_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise Tau3ObjectiveValidityError(f"{key} must be a non-empty string")
    return value


def _domain_value(value: Any) -> str | None:
    return value if isinstance(value, str) and value in TAU3_DOMAINS else None


def _collect_hash_by_domain(values: dict[str, set[str]], domain: str, payload: dict[str, Any], key: str) -> None:
    value = payload.get(key)
    if domain not in values:
        values[domain] = set()
    if isinstance(value, str):
        values[domain].add(value)
    else:
        values[domain].add(f"<missing:{key}>")


def _stable_domain_hash_map(values: dict[str, set[str]]) -> dict[str, str] | None:
    result: dict[str, str] = {}
    for domain in TAU3_DOMAINS:
        hashes = values.get(domain, set())
        if len(hashes) != 1:
            return None
        value = next(iter(hashes))
        if not _is_sha256(value):
            return None
        result[domain] = value
    extra_domains = set(values) - set(TAU3_DOMAINS)
    if extra_domains:
        return None
    return result


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _int_value(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _count(counts: dict[str, Any], key: str) -> int:
    value = counts.get(key, 0)
    return value if isinstance(value, int) and value >= 0 else -1


def _sum_counts(counts: dict[str, Any]) -> int:
    total = 0
    for value in counts.values():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return -1
        total += value
    return total


def _type_name(value: Any) -> str:
    return type(value).__name__


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_json_safe(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    return value


def _strip_replay_volatile(report: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(report, sort_keys=True))


def _validation_result(passed: bool, path: str, errors: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "hfr.tau3_objective_validity_validation.v1",
        "passed": passed,
        "status": "passed" if passed else "failed",
        "artifact_path": path,
        "replayed_passed": False,
        "error_count": len(errors),
        "errors": errors,
        "replayed_failed_check_count": None,
    }
