"""Governed Tau-3 development candidate selection and public lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .atomic_json import atomic_write_json_cas
from .path_safety import path_has_symlink_component
from .repeated_eval import canonical_sha256
from .schema_registry import check_schema_contract
from .tau3_evaluation import (
    DOMAINS,
    NON_INFERIORITY_MARGIN,
    SAFETY_NON_INFERIORITY_MARGIN,
    _arm_effect,
    _domain_counts,
    _extract_result_rows,
    _macro_pass1,
    _per_domain_pass1,
    _public_key,
    _row_map,
    _sha256_file,
)

TAU3_CANDIDATE_SELECTION_SCHEMA_VERSION = "hfr.tau3_candidate_selection.v1"
TAU3_CANDIDATE_LOCK_SCHEMA_VERSION = "hfr.tau3_candidate_lock.v1"
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 8_675_309
RAW_LOCK_FORBIDDEN_KEYS = {
    "path",
    "paths",
    "result_path",
    "output_dir",
    "messages",
    "raw_data",
    "policy",
    "tool_defs",
    "tasks",
    "user_scenario",
    "initial_state",
    "evaluation_criteria",
}


class Tau3CandidateSelectionError(ValueError):
    """Raised when candidate selection cannot be proven safely."""


@dataclass(frozen=True)
class Tau3CandidateEntry:
    candidate_id: str
    development_manifest_path: Path
    training_receipt_path: Path
    candidate_identity_path: Path


def select_tau3_candidate(
    *,
    base_manifest_path: str | Path,
    candidates: Iterable[Tau3CandidateEntry | dict[str, Any]],
    report_path: str | Path,
    lock_path: str | Path,
    created_at: str | None = None,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    non_inferiority_margin: float = NON_INFERIORITY_MARGIN,
    safety_non_inferiority_margin: float = SAFETY_NON_INFERIORITY_MARGIN,
) -> dict[str, Any]:
    """Select exactly one development candidate and atomically write its lock."""

    report_out = Path(report_path)
    lock_out = Path(lock_path)
    if report_out.exists():
        raise Tau3CandidateSelectionError(f"selection report already exists: {report_out}")
    if lock_out.exists():
        raise Tau3CandidateSelectionError(f"candidate lock already exists: {lock_out}")
    if bootstrap_samples < 100:
        raise Tau3CandidateSelectionError("bootstrap_samples must be at least 100")
    if not 0.0 <= non_inferiority_margin <= 1.0:
        raise Tau3CandidateSelectionError("non_inferiority_margin must be in [0, 1]")
    if not 0.0 <= safety_non_inferiority_margin <= 1.0:
        raise Tau3CandidateSelectionError("safety_non_inferiority_margin must be in [0, 1]")

    created = created_at or _now_utc()
    entries = [_coerce_entry(entry) for entry in candidates]
    if not entries:
        raise Tau3CandidateSelectionError("at least one candidate entry is required")
    duplicate_ids = sorted(_duplicates(entry.candidate_id for entry in entries))
    if duplicate_ids:
        raise Tau3CandidateSelectionError("duplicate candidate_id(s): " + ", ".join(duplicate_ids))

    base = _load_benchmark_manifest(Path(base_manifest_path), expected_arm="base")
    candidate_reports = []
    eligible = []
    for entry in entries:
        candidate = _evaluate_candidate(
            entry,
            base=base,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            non_inferiority_margin=non_inferiority_margin,
            safety_non_inferiority_margin=safety_non_inferiority_margin,
        )
        candidate_reports.append(candidate)
        if candidate["eligible"]:
            eligible.append(candidate)

    if not eligible:
        raise Tau3CandidateSelectionError("no eligible Tau-3 candidate: " + json.dumps({c["candidate_id"]: c["blocking_reasons"] for c in candidate_reports}, sort_keys=True))

    eligible.sort(
        key=lambda item: (
            -float(item["metrics"]["macro_pass1"]["candidate"]),
            item["candidate_id"],
            item["candidate_identity"]["identity_sha256"],
        )
    )
    selected = eligible[0]
    selection_count = sum(1 for item in candidate_reports if item["candidate_id"] == selected["candidate_id"])
    if selection_count != 1:
        raise Tau3CandidateSelectionError("selection must resolve to exactly one candidate")

    report = {
        "schema_version": TAU3_CANDIDATE_SELECTION_SCHEMA_VERSION,
        "created_at": created,
        "passed": True,
        "selected_candidate_id": selected["candidate_id"],
        "selection_policy": {
            "primary_metric": "development_macro_pass1",
            "required_domains": list(DOMAINS),
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "confidence_level": 0.95,
            "non_inferiority_margin": non_inferiority_margin,
            "safety_non_inferiority_margin": safety_non_inferiority_margin,
            "tie_break": ["higher_macro_pass1", "candidate_id_lexicographic", "candidate_identity_canonical_sha256"],
            "dev_comparators_required": False,
            "sealed_inputs_allowed": False,
        },
        "base": _private_source_record(base),
        "candidates": candidate_reports,
        "eligible_candidate_count": len(eligible),
        "selection": {
            "candidate_id": selected["candidate_id"],
            "rank": 1,
            "macro_pass1": selected["metrics"]["macro_pass1"]["candidate"],
            "candidate_identity_sha256": selected["candidate_identity"]["sha256"],
            "candidate_identity_canonical_sha256": selected["candidate_identity"]["identity_sha256"],
        },
    }
    report["schema_checked"] = True
    report_check = check_schema_contract(report, name_or_id="tau3_candidate_selection")
    if report_check["passed"] is not True:
        raise Tau3CandidateSelectionError("selection report violates schema: " + "; ".join(report_check["errors"]))

    report_out.parent.mkdir(parents=True, exist_ok=True)
    with report_out.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    report_sha256 = _sha256_file(report_out)
    lock = _candidate_lock(selected, report_sha256=report_sha256, created_at=created)
    _assert_lock_public_safe(lock)
    lock_check = check_schema_contract(lock, name_or_id="tau3_candidate_lock")
    if lock_check["passed"] is not True:
        raise Tau3CandidateSelectionError("candidate lock violates schema: " + "; ".join(lock_check["errors"]))
    lock_digest = atomic_write_json_cas(lock_out, lock, expected_sha256=None, new_file_mode=0o444)
    return {
        "schema_version": "hfr.tau3_candidate_selection_result.v1",
        "selected_candidate_id": selected["candidate_id"],
        "report_path": str(report_out),
        "report_sha256": report_sha256,
        "lock_path": str(lock_out),
        "lock_sha256": lock_digest,
        "eligible_candidate_count": len(eligible),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--lock-out", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--non-inferiority-margin", type=float, default=NON_INFERIORITY_MARGIN)
    parser.add_argument("--safety-non-inferiority-margin", type=float, default=SAFETY_NON_INFERIORITY_MARGIN)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="candidate_id=development_manifest,training_receipt,candidate_identity",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        result = select_tau3_candidate(
            base_manifest_path=args.base_manifest,
            candidates=[_parse_candidate_arg(value) for value in args.candidate],
            report_path=args.report_out,
            lock_path=args.lock_out,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            non_inferiority_margin=args.non_inferiority_margin,
            safety_non_inferiority_margin=args.safety_non_inferiority_margin,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _evaluate_candidate(
    entry: Tau3CandidateEntry,
    *,
    base: dict[str, Any],
    bootstrap_samples: int,
    bootstrap_seed: int,
    non_inferiority_margin: float,
    safety_non_inferiority_margin: float,
) -> dict[str, Any]:
    errors: list[str] = []
    candidate = _load_benchmark_manifest(entry.development_manifest_path, expected_arm="adapter")
    training = _load_training_receipt(entry.training_receipt_path)
    identity = _load_identity(
        entry.candidate_identity_path,
        candidate_id=entry.candidate_id,
        expected_file_sha256=_nested(candidate["manifest"], "candidate_identity", "sha256"),
        training_receipt_sha256=training["record"]["sha256"],
        adapter_tree_sha256=_nested(training["payload"], "adapter", "tree_sha256"),
        endpoint_model_sha256=_nested(candidate["manifest"], "arm_identity", "endpoint_model_sha256"),
    )

    if candidate["manifest"].get("protocol_sha256") != base["manifest"].get("protocol_sha256"):
        errors.append("protocol_sha256_mismatch")
    if candidate["manifest"].get("tau_revision") != base["manifest"].get("tau_revision"):
        errors.append("tau_revision_mismatch")
    if _nested(training["payload"], "training_binding", "protocol", "sha256") != base["manifest"].get("protocol_sha256"):
        errors.append("training_protocol_sha256_mismatch")

    maps_errors: list[str] = []
    maps = {
        "base": _row_map("base", base["rows"], maps_errors),
        "candidate": _row_map(entry.candidate_id, candidate["rows"], maps_errors),
    }
    keys = sorted(maps["base"])
    paired = bool(keys) and sorted(maps["candidate"]) == keys and not maps_errors
    if not paired:
        errors.append("paired_development_rows_mismatch")
    domain_counts = _domain_counts(maps["base"].values()) if paired else {}
    if sorted(domain_counts) != list(DOMAINS):
        errors.append("required_development_domains_missing")
    identical_harness = candidate["harness_by_domain"] == base["harness_by_domain"]
    if not identical_harness:
        errors.append("harness_mismatch")

    completed_runs = candidate["manifest"].get("success_count") == candidate["manifest"].get("run_count") == len(candidate["run_receipts"])
    if not completed_runs:
        errors.append("candidate_runs_incomplete")
    training_ok = _training_receipt_eligible(training["payload"], training["path"])
    if not training_ok["passed"]:
        errors.extend(training_ok["blocking_reasons"])

    safety = _candidate_safety_summary({"base": maps["base"], "candidate": maps["candidate"]}) if paired else _blocked_safety()
    if safety.get("provable") is not True:
        errors.append("safety_not_provable")

    metrics = {
        "macro_pass1": {
            "base": _macro_pass1(maps["base"].values()) if paired else None,
            "candidate": _macro_pass1(maps["candidate"].values()) if paired else None,
        },
        "per_domain_pass1": {
            "base": _per_domain_pass1(maps["base"].values()) if paired else {},
            "candidate": _per_domain_pass1(maps["candidate"].values()) if paired else {},
        },
        "safety": safety,
    }
    effects = {}
    if paired:
        effects["base"] = _arm_effect(
            maps["candidate"],
            maps["base"],
            keys,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            non_inferiority_margin=non_inferiority_margin,
        )
        harmful_rates = safety.get("harmful_mutation_rates", {})
        policy_rates = safety.get("policy_violation_rates", {})
        if not (
            harmful_rates.get("candidate", 1.0) <= harmful_rates.get("base", 0.0) + safety_non_inferiority_margin
            and policy_rates.get("candidate", 1.0) <= policy_rates.get("base", 0.0) + safety_non_inferiority_margin
        ):
            errors.append("safety_non_inferiority_vs_base")
        if effects["base"].get("per_domain_non_inferiority_passed") is not True:
            errors.append("per_domain_non_inferiority_vs_base")

    blocking_reasons = sorted(set(errors))
    return {
        "candidate_id": entry.candidate_id,
        "eligible": not blocking_reasons,
        "blocking_reasons": blocking_reasons,
        "artifacts": {
            "development_manifest": _private_source_record(candidate),
            "training_receipt": training["record"],
        },
        "candidate_identity": identity["record"],
        "training_binding": _training_binding_summary(training["payload"]),
        "paired_development": {
            "passed": paired,
            "key_fields": ["domain", "task_sha256", "trial", "seed"],
            "paired_count": len(keys) if paired else 0,
            "domain_counts": domain_counts,
            "pair_set_sha256": canonical_sha256([_public_key(key) for key in keys]) if paired else None,
        },
        "harness": {
            "passed": identical_harness,
            "normalized_sha256": canonical_sha256(base["harness_by_domain"]) if identical_harness else None,
        },
        "endpoint_model_sha256": _nested(candidate["manifest"], "arm_identity", "endpoint_model_sha256"),
        "metrics": metrics,
        "effects": effects,
    }


def _load_benchmark_manifest(path: Path, *, expected_arm: str) -> dict[str, Any]:
    manifest = _load_json_object(path)
    if manifest.get("mode") == "sealed":
        raise Tau3CandidateSelectionError(f"{path}: sealed-mode benchmark input is forbidden")
    if _sealed_flags_present(manifest):
        raise Tau3CandidateSelectionError(f"{path}: sealed access flags are present")
    schema = check_schema_contract(manifest, name_or_id="tau3_benchmark_run")
    if schema["passed"] is not True:
        raise Tau3CandidateSelectionError(f"{path}: benchmark manifest schema failed: " + "; ".join(schema["errors"]))
    if manifest.get("phase") != "final":
        raise Tau3CandidateSelectionError(f"{path}: benchmark manifest must be final")
    if manifest.get("mode") != "development":
        raise Tau3CandidateSelectionError(f"{path}: benchmark manifest must be development mode")
    if manifest.get("arm_id") != expected_arm:
        raise Tau3CandidateSelectionError(f"{path}: expected arm_id {expected_arm!r}")
    run_refs = manifest.get("run_receipts")
    if not isinstance(run_refs, list) or not run_refs:
        raise Tau3CandidateSelectionError(f"{path}: missing run receipts")
    protocol = _load_protocol_record(path, manifest)
    source = _load_development_source_record(path, manifest, protocol=protocol)
    prelaunch = _load_prelaunch_record(path, manifest)
    run_receipts = []
    rows = []
    harness_by_domain: dict[str, dict[str, Any]] = {}
    source_errors: list[str] = []
    for ref in run_refs:
        if not isinstance(ref, dict):
            raise Tau3CandidateSelectionError(f"{path}: run receipt ref is not an object")
        rel = str(ref.get("path") or "")
        if not rel or "/" in rel or "\\" in rel or ".." in Path(rel).parts:
            raise Tau3CandidateSelectionError(f"{path}: unsafe run receipt reference {rel!r}")
        receipt_path = path.parent / rel
        receipt = _load_json_object(receipt_path)
        if receipt.get("schema_version") != manifest.get("schema_version") or receipt.get("phase") != "domain_seed":
            raise Tau3CandidateSelectionError(f"{receipt_path}: invalid run receipt phase/schema")
        if receipt.get("mode") != "development" or receipt.get("arm_id") != expected_arm:
            raise Tau3CandidateSelectionError(f"{receipt_path}: run receipt mode/arm mismatch")
        if receipt.get("protocol_sha256") != manifest.get("protocol_sha256"):
            raise Tau3CandidateSelectionError(f"{receipt_path}: run receipt protocol_sha256 mismatch")
        if ref.get("domain") != receipt.get("domain") or ref.get("seed") != receipt.get("seed"):
            raise Tau3CandidateSelectionError(f"{receipt_path}: run receipt ref domain/seed mismatch")
        if _sealed_flags_present(receipt):
            raise Tau3CandidateSelectionError(f"{receipt_path}: sealed access flags are present")
        if receipt.get("terminal_status") != "completed" or ref.get("terminal_status") != "completed":
            raise Tau3CandidateSelectionError(f"{receipt_path}: benchmark run is not completed")
        result_path_value = receipt.get("result_path")
        if not isinstance(result_path_value, str) or not result_path_value:
            raise Tau3CandidateSelectionError(f"{receipt_path}: missing raw result_path")
        result_path = Path(result_path_value)
        if not result_path.is_absolute():
            result_path = receipt_path.parent / result_path
        result_sha256 = _sha256_file(result_path)
        if receipt.get("result_sha256") != result_sha256 or ref.get("result_sha256") != result_sha256:
            raise Tau3CandidateSelectionError(f"{receipt_path}: raw result hash does not replay")
        copied_result_path_value = ref.get("result_path")
        if not isinstance(copied_result_path_value, str) or not copied_result_path_value:
            raise Tau3CandidateSelectionError(f"{path}: run receipt ref missing copied result_path")
        copied_result_path = path.parent / copied_result_path_value
        copied_result_sha256 = _sha256_file(copied_result_path)
        if copied_result_sha256 != result_sha256:
            raise Tau3CandidateSelectionError(f"{path}: copied result_path sha256 does not replay")
        raw = _load_json_object(result_path)
        extracted, harness, extraction_errors = _extract_result_rows(
            raw,
            path=result_path,
            arm=expected_arm,
            expected_tau_revision=str(manifest.get("tau_revision") or ""),
        )
        source_errors.extend(extraction_errors)
        domain = str(harness.get("domain_name") or "")
        if domain in harness_by_domain and harness_by_domain[domain] != harness:
            source_errors.append(f"{path}: duplicate differing harness for {domain}")
        harness_by_domain[domain] = harness
        rows.extend(extracted)
        receipt_sha256 = _sha256_file(receipt_path)
        expected_receipt_sha256 = ref.get("receipt_sha256") or ref.get("sha256")
        if expected_receipt_sha256 is not None and expected_receipt_sha256 != receipt_sha256:
            raise Tau3CandidateSelectionError(f"{receipt_path}: run receipt sha256 does not replay")
        run_receipts.append({"path": str(receipt_path), "sha256": receipt_sha256, "result_sha256": result_sha256})
    if source_errors:
        raise Tau3CandidateSelectionError(f"{path}: benchmark raw results invalid: " + "; ".join(source_errors))
    return {
        "path": path,
        "sha256": _sha256_file(path),
        "manifest": manifest,
        "protocol": protocol,
        "source": source,
        "prelaunch_receipt": prelaunch,
        "run_receipts": run_receipts,
        "rows": rows,
        "harness_by_domain": harness_by_domain,
    }


def _load_training_receipt(path: Path) -> dict[str, Any]:
    payload = _load_json_object(path)
    schema = check_schema_contract(payload, name_or_id="tau3_mlx_training_run")
    if schema["passed"] is not True:
        raise Tau3CandidateSelectionError(f"{path}: training receipt schema failed: " + "; ".join(schema["errors"]))
    return {"path": path, "payload": payload, "record": {"path": str(path), "sha256": _sha256_file(path), "terminal_status": payload.get("terminal_status")}}


def _load_prelaunch_record(manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    record = manifest.get("prelaunch_receipt")
    if not isinstance(record, dict):
        raise Tau3CandidateSelectionError(f"{manifest_path}: missing prelaunch_receipt")
    path = _resolve_manifest_file_ref(manifest_path, record, label="prelaunch_receipt")
    digest = _sha256_file(path)
    if record.get("sha256") != digest:
        raise Tau3CandidateSelectionError(f"{manifest_path}: prelaunch_receipt sha256 does not replay")
    payload = _load_json_object(path)
    if payload.get("schema_version") != manifest.get("schema_version") or payload.get("phase") != "prelaunch":
        raise Tau3CandidateSelectionError(f"{manifest_path}: prelaunch receipt phase/schema mismatch")
    for key in ("protocol_sha256", "mode", "arm_id", "config", "task_selection", "candidate_identity"):
        if payload.get(key) != manifest.get(key):
            raise Tau3CandidateSelectionError(f"{manifest_path}: prelaunch receipt {key} mismatch")
    if _sealed_flags_present(payload):
        raise Tau3CandidateSelectionError(f"{manifest_path}: prelaunch receipt sealed access flags are present")
    return {"path": str(path), "sha256": digest}


def _load_protocol_record(manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    record = manifest.get("protocol")
    if not isinstance(record, dict):
        raise Tau3CandidateSelectionError(f"{manifest_path}: missing protocol record")
    path = _resolve_manifest_file_ref(manifest_path, record, label="protocol")
    digest = _sha256_file(path)
    if record.get("sha256") != digest or manifest.get("protocol_sha256") != digest:
        raise Tau3CandidateSelectionError(f"{manifest_path}: protocol sha256 does not replay")
    payload = _load_json_object(path)
    schema = check_schema_contract(payload, name_or_id="tau3_protocol_config")
    if schema["passed"] is not True:
        raise Tau3CandidateSelectionError(f"{manifest_path}: protocol schema failed: " + "; ".join(schema["errors"]))
    return {"path": str(path), "sha256": digest, "payload": payload}


def _load_development_source_record(manifest_path: Path, manifest: dict[str, Any], *, protocol: dict[str, Any]) -> dict[str, Any]:
    record = manifest.get("source")
    if not isinstance(record, dict):
        raise Tau3CandidateSelectionError(f"{manifest_path}: missing development source record")
    path = _resolve_manifest_file_ref(manifest_path, record, label="development source")
    digest = _sha256_file(path)
    if record.get("sha256") != digest:
        raise Tau3CandidateSelectionError(f"{manifest_path}: development source sha256 does not replay")
    payload = _load_json_object(path)
    schema = check_schema_contract(payload, name_or_id="tau3_source_split")
    if schema["passed"] is not True:
        raise Tau3CandidateSelectionError(f"{manifest_path}: development source schema failed: " + "; ".join(schema["errors"]))
    if payload.get("schema_version") != "hfr.tau3_source_split.v1":
        raise Tau3CandidateSelectionError(f"{manifest_path}: development source schema_version mismatch")
    if payload.get("split") != "development":
        raise Tau3CandidateSelectionError(f"{manifest_path}: development source split mismatch")
    if payload.get("source_revision") != manifest.get("tau_revision"):
        raise Tau3CandidateSelectionError(f"{manifest_path}: development source revision mismatch")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or payload.get("task_count") != len(tasks):
        raise Tau3CandidateSelectionError(f"{manifest_path}: development source task_count mismatch")
    families = payload.get("family_ids")
    if not isinstance(families, list) or payload.get("family_count") != len(families):
        raise Tau3CandidateSelectionError(f"{manifest_path}: development source family_count mismatch")
    split_hash = _protocol_development_split_sha256(protocol["payload"])
    if split_hash is not None and split_hash != digest:
        raise Tau3CandidateSelectionError(f"{manifest_path}: development source sha256 does not match protocol split binding")
    return {"path": str(path), "sha256": digest}


def _protocol_development_split_sha256(protocol: dict[str, Any]) -> str | None:
    raw_tau_revision = protocol.get("tau_revision")
    tau_revision: dict[str, Any] = raw_tau_revision if isinstance(raw_tau_revision, dict) else {}
    raw_split_hashes = tau_revision.get("split_hashes")
    split_hashes: dict[str, Any] = raw_split_hashes if isinstance(raw_split_hashes, dict) else {}
    value = split_hashes.get("development")
    if isinstance(value, str) and value:
        return value
    raw_split_manifest = protocol.get("split_manifest")
    split_manifest: dict[str, Any] = raw_split_manifest if isinstance(raw_split_manifest, dict) else {}
    raw_splits = split_manifest.get("splits")
    splits: dict[str, Any] = raw_splits if isinstance(raw_splits, dict) else {}
    raw_development = splits.get("development")
    development: dict[str, Any] = raw_development if isinstance(raw_development, dict) else {}
    value = development.get("sha256")
    return value if isinstance(value, str) and value else None


def _resolve_manifest_file_ref(manifest_path: Path, record: dict[str, Any], *, label: str) -> Path:
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise Tau3CandidateSelectionError(f"{manifest_path}: {label} missing path")
    path = Path(raw_path)
    if not path.is_absolute():
        path = manifest_path.parent / path
    if not path.is_file():
        raise Tau3CandidateSelectionError(f"{manifest_path}: {label} file not found")
    return path


def _load_identity(
    path: Path,
    *,
    candidate_id: str,
    expected_file_sha256: Any,
    training_receipt_sha256: str,
    adapter_tree_sha256: Any,
    endpoint_model_sha256: Any,
) -> dict[str, Any]:
    payload = _load_json_object(path)
    file_sha256 = _sha256_file(path)
    if expected_file_sha256 != file_sha256:
        raise Tau3CandidateSelectionError(f"{path}: candidate manifest candidate_identity.sha256 does not match supplied identity file")
    if payload.get("schema_version") != "hfr.tau3_candidate_identity.v1":
        raise Tau3CandidateSelectionError(f"{path}: candidate identity schema_version must be hfr.tau3_candidate_identity.v1")
    schema = check_schema_contract(payload, name_or_id="tau3_candidate_identity")
    if schema["passed"] is not True:
        raise Tau3CandidateSelectionError(f"{path}: candidate identity schema failed: " + "; ".join(schema["errors"]))
    declared = payload.get("candidate_id") or payload.get("id")
    if declared != candidate_id:
        raise Tau3CandidateSelectionError(f"{path}: candidate identity does not bind candidate_id {candidate_id!r}")
    if _identity_field(payload, "training_receipt_sha256", "final_training_receipt_sha256") != training_receipt_sha256:
        raise Tau3CandidateSelectionError(f"{path}: candidate identity does not bind final training receipt sha256")
    if _identity_field(payload, "adapter_tree_sha256") != adapter_tree_sha256:
        raise Tau3CandidateSelectionError(f"{path}: candidate identity does not bind adapter_tree_sha256")
    if _identity_field(payload, "endpoint_model_sha256") != endpoint_model_sha256:
        raise Tau3CandidateSelectionError(f"{path}: candidate identity does not bind endpoint_model_sha256")
    return {
        "payload": payload,
        "record": {
            "path": str(path),
            "sha256": file_sha256,
            "identity_sha256": canonical_sha256(payload),
            "endpoint_model_sha256": endpoint_model_sha256,
        },
    }


def _training_receipt_eligible(receipt: dict[str, Any], receipt_path: Path | None = None) -> dict[str, Any]:
    reasons = []
    if receipt.get("phase") != "final" or receipt.get("terminal_status") != "success":
        reasons.append("training_not_final_success")
    if receipt.get("weights_updated") is not True or int(receipt.get("adapter_weight_file_count") or 0) <= 0:
        reasons.append("training_weights_not_updated")
    if receipt.get("schema_checked") is not True:
        reasons.append("training_schema_not_checked")
    raw_binding = receipt.get("training_binding")
    binding: dict[str, Any] = raw_binding if isinstance(raw_binding, dict) else {}
    if not binding:
        reasons.append("training_binding_missing")
    raw_adapter = receipt.get("adapter")
    adapter: dict[str, Any] = raw_adapter if isinstance(raw_adapter, dict) else {}
    if not adapter.get("tree_sha256"):
        reasons.append("adapter_tree_sha256_missing")
    if receipt_path is not None:
        reasons.extend(_replay_training_adapter(receipt_path, receipt, adapter))
        reasons.extend(_replay_training_telemetry(receipt_path, receipt))
    raw_bundle = receipt.get("bundle")
    bundle: dict[str, Any] = raw_bundle if isinstance(raw_bundle, dict) else {}
    bundle_kind = bundle.get("kind")
    raw_checks = receipt.get("checks")
    checks: list[Any] = raw_checks if isinstance(raw_checks, list) else []
    required_checks: tuple[str, ...]
    if bundle_kind == "bundle":
        required_checks = ("plan_uses_development_not_sealed",)
    elif bundle_kind == "mixture":
        required_checks = (
            "protocol_schema_passed",
            "recipe_within_protocol_recipe_space",
            "mixture_manifest_protocol_sha_matches",
            "mixture_no_sealed_or_test_rows",
        )
    else:
        required_checks = ()
    if not required_checks:
        reasons.append("training_bundle_kind_unknown")
    for check_id in required_checks:
        matches = [check for check in checks if isinstance(check, dict) and check.get("id") == check_id]
        if not matches:
            reasons.append(f"{check_id}_missing")
        elif any(check.get("passed") is not True for check in matches):
            reasons.append(check_id)
    required_proof_checks = _required_training_proof_checks(receipt)
    for check_id in required_proof_checks:
        matches = [check for check in checks if isinstance(check, dict) and check.get("id") == check_id]
        if not matches:
            reasons.append(f"{check_id}_missing")
        elif any(check.get("passed") is not True for check in matches):
            reasons.append(check_id)
    return {"passed": not reasons, "blocking_reasons": sorted(set(reasons))}


def _replay_training_adapter(receipt_path: Path, receipt: dict[str, Any], adapter: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    try:
        adapter_dir = _resolve_output_relative(receipt_path, adapter, "adapter")
    except Tau3CandidateSelectionError:
        return ["adapter_path_invalid"]
    if not adapter_dir.is_dir():
        return ["adapter_path_invalid"]
    try:
        tree = _fingerprint_adapter_tree(adapter_dir)
    except Tau3CandidateSelectionError:
        return ["adapter_tree_unsafe"]
    if tree["tree_sha256"] != adapter.get("tree_sha256"):
        reasons.append("adapter_tree_replay_mismatch")
    if tree["file_count"] != adapter.get("file_count"):
        reasons.append("adapter_file_count_replay_mismatch")
    declared_files = adapter.get("files")
    if isinstance(declared_files, list) and declared_files and declared_files != tree["files"]:
        reasons.append("adapter_files_replay_mismatch")
    if not any(file.get("kind") == "adapter" for file in tree["files"]):
        reasons.append("adapter_weight_files_missing")
    return reasons


def _replay_training_telemetry(receipt_path: Path, receipt: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    raw_telemetry = receipt.get("telemetry")
    telemetry: dict[str, Any] = raw_telemetry if isinstance(raw_telemetry, dict) else {}
    try:
        telemetry_path = _resolve_output_relative(receipt_path, telemetry, "telemetry")
    except Tau3CandidateSelectionError:
        return ["telemetry_path_invalid"]
    if not telemetry_path.is_file():
        return ["telemetry_path_invalid"]
    digest = _sha256_file(telemetry_path)
    if telemetry.get("sha256") != digest:
        reasons.append("telemetry_sha256_replay_mismatch")
    lines = telemetry_path.read_text(encoding="utf-8").splitlines()
    if telemetry.get("event_count") != len(lines):
        reasons.append("telemetry_event_count_replay_mismatch")
    raw_losses = receipt.get("losses")
    losses: dict[str, Any] = raw_losses if isinstance(raw_losses, dict) else {}
    train_losses = losses.get("train")
    if not isinstance(train_losses, list) or not train_losses:
        reasons.append("training_train_losses_missing")
        return reasons
    if not _telemetry_contains_train_losses(lines, train_losses):
        reasons.append("training_train_losses_not_in_telemetry")
    return reasons


def _telemetry_contains_train_losses(lines: list[str], train_losses: list[Any]) -> bool:
    texts: list[str] = []
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = event.get("text") if isinstance(event, dict) else None
        if isinstance(text, str):
            texts.append(text)
    joined = "\n".join(texts)
    for value in train_losses:
        if not isinstance(value, int | float):
            return False
        if _float_tokens(float(value)).isdisjoint(_numeric_tokens(joined)):
            return False
    return True


def _float_tokens(value: float) -> set[str]:
    return {format(value, "g"), str(value)}


def _numeric_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    current = []
    for char in text:
        if char.isdigit() or char in ".-+eE":
            current.append(char)
        elif current:
            tokens.add("".join(current))
            current = []
    if current:
        tokens.add("".join(current))
    return tokens


def _required_training_proof_checks(receipt: dict[str, Any]) -> tuple[str, ...]:
    required: list[str] = []
    if _proof_required(receipt, names={"smoke", "smoke_update", "instrumented_smoke"}):
        required.extend(("smoke_update_observed", "smoke_checkpoint_observed"))
    if _resume_required(receipt):
        required.extend(
            (
                "resume_receipt_schema_passed",
                "resume_adapter_tree_fingerprint_replays",
                "resume_adapter_file_bound_to_prior_fingerprint",
                "resume_protocol_model_dataset_match",
                "resume_hyperparameters_match",
            )
        )
    return tuple(required)


def _resume_required(value: Any) -> bool:
    if isinstance(value, dict):
        raw_resume = value.get("resume")
        if isinstance(raw_resume, dict) and raw_resume.get("enabled") is True:
            return True
        return any(_resume_required(child) for child in value.values())
    if isinstance(value, list):
        return any(_resume_required(child) for child in value)
    return False


def _proof_required(value: Any, *, names: set[str]) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            key_norm = str(key).lower()
            if key_norm in {f"{name}_required" for name in names} and child is True:
                return True
            if key_norm in names and isinstance(child, dict) and child.get("required") is True:
                return True
            if _proof_required(child, names=names):
                return True
    elif isinstance(value, list):
        return any(_proof_required(child, names=names) for child in value)
    return False


def _resolve_output_relative(receipt_path: Path, record: dict[str, Any], label: str) -> Path:
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise Tau3CandidateSelectionError(f"{receipt_path}: {label} path missing")
    rel = Path(raw_path)
    if rel.is_absolute() or ".." in rel.parts:
        raise Tau3CandidateSelectionError(f"{receipt_path}: {label} path must be output-relative")
    base = receipt_path.parent.resolve(strict=True)
    candidate = receipt_path.parent / rel
    if path_has_symlink_component(candidate, include_leaf=True):
        raise Tau3CandidateSelectionError(f"{receipt_path}: {label} path must not contain symlinks")
    path = candidate.resolve(strict=True)
    if base not in (path, *path.parents):
        raise Tau3CandidateSelectionError(f"{receipt_path}: {label} path escapes training output")
    return path


def _fingerprint_adapter_tree(root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path_has_symlink_component(path, include_leaf=True):
            raise Tau3CandidateSelectionError(f"adapter tree contains symlink: {path}")
        rel = path.relative_to(root).as_posix()
        files.append({"path": rel, "size": path.stat().st_size, "sha256": _sha256_file(path), "kind": _fingerprint_kind(rel)})
    digest = hashlib.sha256()
    for record in files:
        digest.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return {"file_count": len(files), "files": files, "tree_sha256": digest.hexdigest() if files else None}


def _fingerprint_kind(rel: str) -> str:
    name = Path(rel).name
    if name in {"adapter_config.json", "config.json"}:
        return "config"
    if "checkpoint" in rel.lower():
        return "checkpoint"
    if Path(rel).suffix in {".safetensors", ".npz", ".bin"}:
        return "adapter"
    return "artifact"


def _training_binding_summary(receipt: dict[str, Any]) -> dict[str, Any]:
    raw_binding = receipt.get("training_binding")
    binding: dict[str, Any] = raw_binding if isinstance(raw_binding, dict) else {}
    raw_adapter = receipt.get("adapter")
    adapter: dict[str, Any] = raw_adapter if isinstance(raw_adapter, dict) else {}
    return {
        "protocol_sha256": _nested(binding, "protocol", "sha256"),
        "protocol_signature": _nested(binding, "protocol", "protocol_signature"),
        "recipe_sha256": _nested(binding, "recipe", "recipe_sha256"),
        "base_identity_sha256": _nested(binding, "model", "identity_sha256"),
        "base_tree_sha256": _nested(binding, "model", "tree_sha256"),
        "dataset_manifest_sha256": _nested(binding, "dataset", "manifest_sha256"),
        "dataset_files_sha256": _nested(binding, "dataset", "files_sha256"),
        "source_binding_sha256": _nested(binding, "dataset", "source_binding_sha256"),
        "adapter_tree_sha256": adapter.get("tree_sha256"),
    }


def _candidate_lock(selected: dict[str, Any], *, report_sha256: str, created_at: str) -> dict[str, Any]:
    summary = selected["training_binding"]
    identity_file_hash = selected["candidate_identity"]["sha256"]
    lock = {
        "schema_version": TAU3_CANDIDATE_LOCK_SCHEMA_VERSION,
        "created_at": created_at,
        "selected_candidate_id_hash": canonical_sha256(selected["candidate_id"]),
        "candidate_identity_sha256": identity_file_hash,
        "development_selection_report_sha256": report_sha256,
        "development_benchmark_manifest_sha256": selected["artifacts"]["development_manifest"]["sha256"],
        "training_receipt_sha256": selected["artifacts"]["training_receipt"]["sha256"],
        "endpoint_model_sha256": selected["candidate_identity"]["endpoint_model_sha256"],
        "adapter_tree_sha256": summary.get("adapter_tree_sha256"),
        "recipe_sha256": summary.get("recipe_sha256"),
        "base_identity_sha256": summary.get("base_identity_sha256"),
        "base_tree_sha256": summary.get("base_tree_sha256"),
        "dataset_manifest_sha256": summary.get("dataset_manifest_sha256"),
        "dataset_files_sha256": summary.get("dataset_files_sha256"),
        "source_binding_sha256": summary.get("source_binding_sha256"),
        "protocol_sha256": summary.get("protocol_sha256"),
        "protocol_signature": summary.get("protocol_signature"),
        "hashes_only": True,
        "sealed_access_authorized": True,
        "local_paths_included": False,
        "raw_payload_included": False,
    }
    missing = sorted(key for key, value in lock.items() if key.endswith("sha256") and not value)
    if missing:
        raise Tau3CandidateSelectionError("candidate lock missing hash binding(s): " + ", ".join(missing))
    return lock


def _private_source_record(source: dict[str, Any]) -> dict[str, Any]:
    manifest = source["manifest"]
    return {
        "path": str(source["path"]),
        "sha256": source["sha256"],
        "mode": manifest.get("mode"),
        "arm_id": manifest.get("arm_id"),
        "protocol_sha256": manifest.get("protocol_sha256"),
        "tau_revision": manifest.get("tau_revision"),
        "run_count": manifest.get("run_count"),
        "success_count": manifest.get("success_count"),
        "run_receipts": source["run_receipts"],
    }


def _parse_candidate_arg(value: str) -> Tau3CandidateEntry:
    if "=" not in value:
        raise Tau3CandidateSelectionError("--candidate must use candidate_id=manifest,receipt,identity")
    candidate_id, paths = value.split("=", 1)
    parts = [part.strip() for part in paths.split(",")]
    if len(parts) != 3 or not all(parts):
        raise Tau3CandidateSelectionError("--candidate must use candidate_id=manifest,receipt,identity")
    return Tau3CandidateEntry(candidate_id=candidate_id.strip(), development_manifest_path=Path(parts[0]), training_receipt_path=Path(parts[1]), candidate_identity_path=Path(parts[2]))


def _coerce_entry(entry: Tau3CandidateEntry | dict[str, Any]) -> Tau3CandidateEntry:
    if isinstance(entry, Tau3CandidateEntry):
        return entry
    return Tau3CandidateEntry(
        candidate_id=str(entry["candidate_id"]),
        development_manifest_path=Path(entry["development_manifest_path"]),
        training_receipt_path=Path(entry["training_receipt_path"]),
        candidate_identity_path=Path(entry["candidate_identity_path"]),
    )


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Tau3CandidateSelectionError(f"expected JSON object: {path}")
    return payload


def _sealed_flags_present(value: dict[str, Any]) -> bool:
    return (
        value.get("mode") == "sealed"
        or value.get("sealed_payload_accessed") is not False
        or value.get("sealed_task_ids_materialized") is not False
    )


def _blocked_safety() -> dict[str, Any]:
    return {"provable": False, "blocking_reasons": ["unpaired_safety"]}


def _candidate_safety_summary(maps: dict[str, dict[tuple[str, str, int, int], dict[str, Any]]]) -> dict[str, Any]:
    missing_db = {
        arm: sum(1 for row in rows.values() if row.get("db_evaluated") is True and row.get("safety_provable") is not True)
        for arm, rows in maps.items()
    }
    missing_policy = {
        arm: sum(1 for row in rows.values() if row.get("policy_provable") is not True)
        for arm, rows in maps.items()
    }
    harmful_counts = {
        arm: sum(1 for row in rows.values() if row.get("harmful_mutation") is True)
        for arm, rows in maps.items()
    }
    policy_counts = {
        arm: sum(1 for row in rows.values() if row.get("policy_violation") is True)
        for arm, rows in maps.items()
    }
    totals = {arm: len(rows) for arm, rows in maps.items()}
    provable = all(count == 0 for count in missing_db.values()) and all(count == 0 for count in missing_policy.values())
    return {
        "provable": provable,
        "definition": "DB reward-basis mismatch is counted conservatively as harmful mutation; policy violation requires official Tau full-review evidence",
        "missing_db_evidence_counts": missing_db,
        "missing_policy_review_counts": missing_policy,
        "harmful_mutation_counts": harmful_counts,
        "harmful_mutation_rates": {arm: (harmful_counts[arm] / totals[arm] if totals[arm] else None) for arm in sorted(maps)},
        "policy_violation_counts": policy_counts,
        "policy_violation_rates": {arm: (policy_counts[arm] / totals[arm] if totals[arm] else None) for arm in sorted(maps)},
        "blocking_reasons": [] if provable else [
            reason
            for reason, present in (
                ("missing_db_check_for_db_reward_basis", any(missing_db.values())),
                ("missing_full_policy_review", any(missing_policy.values())),
            )
            if present
        ],
    }


def _duplicates(values: Iterable[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if not value:
            duplicates.add(value)
        elif value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


def _nested(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _identity_field(payload: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = payload.get(name)
        if value is not None:
            return value
    for key in ("candidate", "identity", "model_identity", "adapter_identity"):
        child = payload.get(key)
        if isinstance(child, dict):
            value = _identity_field(child, *names)
            if value is not None:
                return value
    return None


def _assert_lock_public_safe(lock: dict[str, Any]) -> None:
    forbidden = sorted(_find_forbidden_keys(lock))
    if forbidden:
        raise Tau3CandidateSelectionError("candidate lock contains forbidden public key(s): " + ", ".join(forbidden))
    encoded = json.dumps(lock, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    leaks = [pattern for pattern in ("/Users/", "localhost", "127.0.0.1", "local/tau3", "\\") if pattern in encoded]
    if leaks:
        raise Tau3CandidateSelectionError("candidate lock contains local/private text pattern(s): " + ", ".join(leaks))


def _find_forbidden_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in RAW_LOCK_FORBIDDEN_KEYS:
                found.add(key)
            found.update(_find_forbidden_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_find_forbidden_keys(item))
    return found


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
