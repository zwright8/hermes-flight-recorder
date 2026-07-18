#!/usr/bin/env python3
# /// script
# dependencies = [
#   "accelerate",
#   "datasets",
#   "peft",
#   "torch",
#   "trackio",
#   "transformers",
#   "trl",
# ]
# ///
"""Train LoRA adapters for the Flight Recorder agentic experiment.

Modes:

* trace_sft: SFT on raw trace-only rows.
* fr_sft: SFT on Flight Recorder accepted rows plus passed action traces.
* fr_dpo: DPO on Flight Recorder chosen/rejected rows.
* fr_sft_dpo: SFT on accepted/action rows, then continue that adapter with DPO.

Use ``--dry-run`` to validate data and write a training plan without importing
heavy ML dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_TRACKIO_PROJECT = "hermes-flightrecorder-agentic"
PLAN_SCHEMA_VERSION = "hfr.agentic_lora_training_plan.v1"
RESULT_SCHEMA_VERSION = "hfr.agentic_lora_training_result.v1"
BACKEND_RECIPES_SCHEMA_VERSION = "hfr.agentic_lora_backend_recipes.v1"
MODEL_REGISTRY_LINK_PLAN_SCHEMA_VERSION = "hfr.agentic_lora_model_registry_link_plan.v1"
EXECUTABLE_MODES = {"trace_sft", "fr_sft", "fr_action_sft", "fr_dpo", "fr_sft_dpo"}
PLAN_ONLY_MODES = {"fr_reward_model", "fr_step_rewards"}
SUPPORTED_MODES = sorted(EXECUTABLE_MODES | PLAN_ONLY_MODES)
UNKNOWN_LICENSE_STATUSES = {"", "unknown", "unreviewed", "pending", "blocked", "rejected", "disallowed"}
SAFE_REDACTION_STATUSES = {"redacted", "sanitized", "passed", "verified", "flight_recorder_redacted"}
TRAINER_DEPENDENCIES = ("torch", "datasets", "peft", "trl", "transformers", "accelerate")
FLIGHT_RECORDER_MODES = {mode for mode in SUPPORTED_MODES if mode.startswith("fr_")}
CONTROL_SCHEMAS = {
    "governance_receipt": "hfr.data_governance_receipt.v1",
    "contamination_report": "hfr.dataset_contamination_report.v1",
    "curated_dataset": "hfr.curated_dataset.v1",
    "action_credit": "hfr.action_credit.v1",
    "branch_replay": "hfr.branch_replay_dataset.v1",
    "reviewed_preferences": "hfr.reviewed.contract_preference.v1",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_json_object(path: Path) -> tuple[dict[str, Any], str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return {}, str(exc)
    except json.JSONDecodeError as exc:
        return {}, f"invalid JSON: {exc}"
    if not isinstance(value, dict):
        return {}, "manifest must contain a JSON object"
    return value, None


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def adapter_artifact_manifest(directory: Path) -> dict[str, Any]:
    files = [path for path in sorted(directory.rglob("*")) if path.is_file() and not path.is_symlink()]
    return {
        "directory": str(directory),
        "file_count": len(files),
        "files": [
            {
                "path": path.relative_to(directory).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ],
    }


def as_messages(prompt: str, response: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


def prepare_sft_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for row in rows:
        messages = row.get("messages")
        if not messages:
            prompt = str(row.get("prompt") or "")
            response = str(row.get("response") or "")
            messages = as_messages(prompt, response)
        if messages and len(messages) >= 2:
            prepared.append(
                {
                    "messages": messages,
                    "tools": row.get("tools") if isinstance(row.get("tools"), list) else [],
                }
            )
    return prepared


def prepare_dpo_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for row in rows:
        chosen_messages = row.get("chosen_messages")
        rejected_messages = row.get("rejected_messages")
        prepared_row: dict[str, Any] | None = None
        if isinstance(chosen_messages, list) and isinstance(rejected_messages, list):
            prefix_length = 0
            for chosen_message, rejected_message in zip(chosen_messages, rejected_messages):
                if chosen_message != rejected_message:
                    break
                prefix_length += 1
            prompt_messages = chosen_messages[:prefix_length]
            chosen_completion = chosen_messages[prefix_length:]
            rejected_completion = rejected_messages[prefix_length:]
            if prompt_messages and chosen_completion and rejected_completion:
                prepared_row = {
                    "prompt": prompt_messages,
                    "chosen": chosen_completion,
                    "rejected": rejected_completion,
                }
        if prepared_row is None:
            prompt = row.get("prompt")
            chosen = row.get("chosen")
            rejected = row.get("rejected")
            if not prompt or not chosen or not rejected:
                continue
            prepared_row = {
                "prompt": [{"role": "user", "content": str(prompt)}],
                "chosen": [{"role": "assistant", "content": str(chosen)}],
                "rejected": [{"role": "assistant", "content": str(rejected)}],
            }
        prepared_row["tools"] = row.get("tools") if isinstance(row.get("tools"), list) else []
        prepared.append(prepared_row)
    return prepared


def data_paths(experiment_dir: Path, dataset_context: dict[str, Any] | None = None) -> dict[str, Path]:
    data = experiment_dir / "data"
    paths = {
        "trace_sft": data / "hermes_trace_only_sft.jsonl",
        "fr_sft": data / "flightrecorder_sft.jsonl",
        "fr_action_sft": data / "flightrecorder_action_sft.jsonl",
        "fr_dpo": data / "flightrecorder_combined_dpo.jsonl",
        "fr_reward_model": data / "flightrecorder_reward_model.jsonl",
        "fr_step_rewards": data / "flightrecorder_step_rewards.jsonl",
    }
    artifact_map = (dataset_context or {}).get("artifact_map")
    artifact_base = (dataset_context or {}).get("artifact_base")
    if isinstance(artifact_map, dict):
        manifest_base = Path(artifact_base) if artifact_base else Path.cwd()
        aliases = {
            "trace_sft": ("trace_sft", "trace_only_sft", "hermes_trace_only_sft"),
            "fr_sft": ("fr_sft", "flightrecorder_sft", "train_sft", "sft"),
            "fr_action_sft": ("fr_action_sft", "flightrecorder_action_sft", "train_action_sft", "action_sft"),
            "fr_dpo": ("fr_dpo", "flightrecorder_combined_dpo", "combined_dpo", "train_dpo", "dpo"),
            "fr_reward_model": ("fr_reward_model", "flightrecorder_reward_model", "train_reward_model", "reward_model"),
            "fr_step_rewards": ("fr_step_rewards", "flightrecorder_step_rewards", "train_step_rewards", "step_rewards"),
        }
        for plan_key, names in aliases.items():
            for name in names:
                value = artifact_map.get(name)
                if isinstance(value, str) and value:
                    paths[plan_key] = resolve_artifact_path(value, manifest_base)
                    break
    return paths


def resolve_artifact_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    base_path = base_dir / path
    if base_path.exists():
        return base_path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return path


def load_model_context(path: Path | None) -> dict[str, Any]:
    context: dict[str, Any] = {"provided": path is not None, "path": str(path) if path else "", "manifest": {}, "error": ""}
    if path is None:
        return context
    manifest, error = load_json_object(path)
    context["manifest"] = manifest
    context["error"] = error or ""
    context["schema_version"] = str(manifest.get("schema_version") or "")
    context["model_id"] = str(manifest.get("model_id") or manifest.get("id") or "")
    if not context["model_id"] and isinstance(manifest.get("model"), dict):
        context["model_id"] = str(manifest["model"].get("model_id") or manifest["model"].get("id") or "")
    license_record = manifest.get("license") if isinstance(manifest.get("license"), dict) else {}
    context["license_status"] = str(manifest.get("license_status") or license_record.get("status") or "").lower()
    context["training_allowed"] = manifest.get("training_allowed", manifest.get("fine_tuning_allowed", True))
    compatibility = manifest.get("compatibility")
    context["compatibility"] = compatibility if isinstance(compatibility, dict) else {}
    return context


def load_dataset_context(path: Path | None) -> dict[str, Any]:
    context: dict[str, Any] = {
        "provided": path is not None,
        "path": str(path) if path else "",
        "manifest": {},
        "source_manifest": {},
        "source_path": "",
        "artifact_map": {},
        "artifact_base": "",
        "error": "",
    }
    if path is None:
        return context
    manifest, error = load_json_object(path)
    context["manifest"] = manifest
    context["error"] = error or ""
    context["schema_version"] = str(manifest.get("schema_version") or "")
    source_manifest = manifest
    source_path = path
    source_value = (
        manifest.get("manifest_path")
        or manifest.get("source_manifest")
        or manifest.get("source_manifest_path")
        or manifest.get("training_manifest")
    )
    if isinstance(source_value, str) and source_value:
        resolved = resolve_artifact_path(source_value, path.parent)
        nested, nested_error = load_json_object(resolved)
        if nested_error:
            context["error"] = nested_error
        else:
            source_manifest = nested
            source_path = resolved
    artifact_map = _dict_value(manifest, "data_files")
    artifact_base = path.parent
    if not artifact_map:
        artifact_map = _dict_value(manifest, "artifacts")
    if not artifact_map:
        artifact_map = _dict_value(manifest, "outputs")
    if not artifact_map:
        artifact_map = _dict_value(source_manifest, "data_files") or _dict_value(source_manifest, "artifacts") or _dict_value(
            source_manifest, "outputs"
        )
        artifact_base = source_path.parent
    context["source_manifest"] = source_manifest
    context["source_path"] = str(source_path)
    context["source_schema_version"] = str(source_manifest.get("schema_version") or "")
    context["artifact_map"] = artifact_map
    context["artifact_base"] = str(artifact_base)
    context["artifact_fingerprints"] = _dict_value(manifest, "artifact_fingerprints")
    context["fingerprint_base"] = str(path.parent)
    return context


def _dict_value(source: dict[str, Any], key: str) -> dict[str, Any]:
    value = source.get(key)
    return value if isinstance(value, dict) else {}


def _first_string(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return ""


def _redaction_status(dataset_context: dict[str, Any]) -> str:
    manifest = dataset_context.get("manifest") if isinstance(dataset_context.get("manifest"), dict) else {}
    source = dataset_context.get("source_manifest") if isinstance(dataset_context.get("source_manifest"), dict) else {}
    redaction = manifest.get("redaction") if isinstance(manifest.get("redaction"), dict) else {}
    source_redaction = source.get("redaction") if isinstance(source.get("redaction"), dict) else {}
    manifest_status = manifest.get("redaction_status")
    source_status = source.get("redaction_status")
    if isinstance(manifest_status, dict) and manifest_status.get("passed") is True:
        return "passed"
    if isinstance(source_status, dict) and source_status.get("passed") is True:
        return "passed"
    return _first_string(
        manifest_status,
        redaction.get("status"),
        source_status,
        source_redaction.get("status"),
    ).lower()


def _dataset_identity(dataset_context: dict[str, Any]) -> str:
    manifest = dataset_context.get("manifest") if isinstance(dataset_context.get("manifest"), dict) else {}
    return _first_string(
        manifest.get("dataset_version"),
        manifest.get("version"),
        manifest.get("dataset_id"),
        manifest.get("id"),
    )


def _training_gate_passed(dataset_context: dict[str, Any]) -> bool | None:
    manifest = dataset_context.get("manifest") if isinstance(dataset_context.get("manifest"), dict) else {}
    for key in ("gates", "quality_gates"):
        gates = manifest.get(key)
        if not isinstance(gates, dict):
            continue
        gate = gates.get("training_gate") or gates.get("dataset_quality_gate")
        if isinstance(gate, dict) and isinstance(gate.get("passed"), bool):
            return gate["passed"]
    gate = manifest.get("training_gate")
    if isinstance(gate, dict) and isinstance(gate.get("passed"), bool):
        return gate["passed"]
    return None


def _family_exclusive(dataset_context: dict[str, Any]) -> bool | None:
    for key in ("manifest", "source_manifest"):
        source = dataset_context.get(key)
        if not isinstance(source, dict):
            continue
        splits = source.get("dataset_splits")
        if isinstance(splits, dict) and isinstance(splits.get("family_exclusive"), bool):
            return splits["family_exclusive"]
        leakage = source.get("leakage_checks")
        if isinstance(leakage, dict) and isinstance(leakage.get("family_exclusive"), bool):
            return leakage["family_exclusive"]
    return None


def _quality_flags_clear(dataset_context: dict[str, Any]) -> bool | None:
    for key in ("manifest", "source_manifest"):
        source = dataset_context.get(key)
        if not isinstance(source, dict):
            continue
        flags = source.get("quality_flags")
        if isinstance(flags, list):
            return not any(
                isinstance(flag, dict) and str(flag.get("severity") or "").lower() == "error"
                for flag in flags
            )
        count = source.get("quality_flag_count")
        if isinstance(count, int) and not isinstance(count, bool):
            return count == 0
    return None


def _source_fingerprints_verified(dataset_context: dict[str, Any]) -> bool | None:
    for key in ("manifest", "source_manifest"):
        source = dataset_context.get(key)
        if not isinstance(source, dict):
            continue
        coverage = source.get("source_fingerprint_coverage")
        if not isinstance(coverage, dict):
            continue
        unverified = coverage.get("unverified")
        fully_verified = coverage.get("fully_verified")
        if isinstance(unverified, int) and not isinstance(unverified, bool):
            return unverified == 0 and (not isinstance(fully_verified, int) or fully_verified > 0)
    return None


def _dataset_artifact_integrity(dataset_context: dict[str, Any]) -> dict[str, Any]:
    records = dataset_context.get("artifact_fingerprints")
    if not isinstance(records, dict) or not records:
        return {"passed": False, "checked": 0, "failures": ["artifact_fingerprints missing"]}
    base = Path(str(dataset_context.get("fingerprint_base") or "."))
    failures: list[str] = []
    checked = 0
    for name, record in records.items():
        if not isinstance(record, dict):
            failures.append(f"{name}: fingerprint record is not an object")
            continue
        path_value = record.get("path")
        expected_sha = record.get("sha256")
        expected_size = record.get("size_bytes")
        if not isinstance(path_value, str) or not path_value:
            failures.append(f"{name}: path missing")
            continue
        path = resolve_artifact_path(path_value, base)
        if not path.exists() or not path.is_file():
            failures.append(f"{name}: file missing")
            continue
        checked += 1
        if not isinstance(expected_size, int) or isinstance(expected_size, bool) or path.stat().st_size != expected_size:
            failures.append(f"{name}: size mismatch")
        actual_sha = sha256_file(path)
        if not isinstance(expected_sha, str) or actual_sha != expected_sha:
            failures.append(f"{name}: sha256 mismatch")
    return {"passed": not failures and checked == len(records), "checked": checked, "failures": failures}


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _control_paths(dataset_context: dict[str, Any]) -> dict[str, Path]:
    artifact_map = dataset_context.get("artifact_map")
    if not isinstance(artifact_map, dict):
        return {}
    base = Path(str(dataset_context.get("artifact_base") or "."))
    return {
        name: resolve_artifact_path(value, base)
        for name in CONTROL_SCHEMAS
        if isinstance((value := artifact_map.get(name)), str) and value
    }


def _validate_control_file_binding(
    name: str,
    path: Path | None,
    dataset_context: dict[str, Any],
    failures: list[str],
) -> dict[str, Any] | None:
    if path is None:
        failures.append(f"{name}: control path is missing from dataset manifest")
        return None
    records = dataset_context.get("artifact_fingerprints")
    record = records.get(name) if isinstance(records, dict) else None
    if not isinstance(record, dict):
        failures.append(f"{name}: artifact fingerprint is missing")
        return None
    if not path.is_file() or path.is_symlink():
        failures.append(f"{name}: control is not a regular file")
        return None
    fingerprint_base = Path(str(dataset_context.get("fingerprint_base") or "."))
    recorded_path = record.get("path")
    if not isinstance(recorded_path, str) or not recorded_path:
        failures.append(f"{name}: fingerprint path is missing")
    elif resolve_artifact_path(recorded_path, fingerprint_base).resolve() != path.resolve():
        failures.append(f"{name}: manifest path and fingerprint path identify different files")
    actual_sha = sha256_file(path)
    actual_size = path.stat().st_size
    if record.get("sha256") != actual_sha:
        failures.append(f"{name}: SHA-256 does not match dataset manifest")
    if record.get("size_bytes") != actual_size:
        failures.append(f"{name}: size does not match dataset manifest")
    return {
        "path": str(path),
        "sha256": actual_sha,
        "size_bytes": actual_size,
        "schema_version": CONTROL_SCHEMAS[name],
    }


def _require_keys(value: dict[str, Any], required: set[str], label: str, failures: list[str]) -> None:
    missing = sorted(required - set(value))
    if missing:
        failures.append(f"{label}: missing fields {missing}")


def _load_control_evidence(dataset_context: dict[str, Any]) -> dict[str, Any]:
    paths = _control_paths(dataset_context)
    failures: list[str] = []
    artifacts: dict[str, dict[str, Any]] = {}
    values: dict[str, Any] = {}
    for name, schema_version in CONTROL_SCHEMAS.items():
        path = paths.get(name)
        record = _validate_control_file_binding(name, path, dataset_context, failures)
        if record is not None:
            artifacts[name] = record
        if path is None or not path.is_file():
            continue
        try:
            if path.suffix == ".jsonl":
                value: Any = load_jsonl(path)
                if not value:
                    failures.append(f"{name}: control has no rows")
            else:
                value, error = load_json_object(path)
                if error:
                    failures.append(f"{name}: {error}")
                    continue
            values[name] = value
        except (json.JSONDecodeError, OSError, UnicodeError) as exc:
            failures.append(f"{name}: {exc}")
            continue
        rows = value if isinstance(value, list) else [value]
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or row.get("schema_version") != schema_version:
                failures.append(f"{name}[{index}]: schema_version must be {schema_version}")

    governance = values.get("governance_receipt")
    if isinstance(governance, dict):
        _validate_governance_control(governance, failures)
    contamination = values.get("contamination_report")
    if isinstance(contamination, dict):
        _validate_contamination_control(contamination, failures)
    curated = values.get("curated_dataset")
    if isinstance(curated, dict):
        _validate_curation_control(curated, failures)
    credits = values.get("action_credit")
    if isinstance(credits, list):
        _validate_action_credit_control(credits, failures)
    replay = values.get("branch_replay")
    if isinstance(replay, dict):
        _validate_branch_replay_control(replay, failures)
    preferences = values.get("reviewed_preferences")
    if isinstance(preferences, list):
        _validate_preference_control(preferences, failures)

    if isinstance(governance, dict) and isinstance(contamination, dict) and isinstance(curated, dict):
        counts = {
            governance.get("record_count"),
            contamination.get("row_count"),
            curated.get("input_count"),
        }
        if len(counts) != 1:
            failures.append("governance, contamination, and curation controls do not bind one input count")
        authorized = {
            str(row.get("record_id"))
            for row in governance.get("record_statuses", [])
            if isinstance(row, dict) and row.get("passed") is True and row.get("record_id")
        }
        selected = {
            str(row.get("episode_id") or row.get("row_id"))
            for row in curated.get("selected", [])
            if isinstance(row, dict) and (row.get("episode_id") or row.get("row_id"))
        }
        if not selected or not selected.issubset(authorized):
            failures.append("curated selected rows are not all authorized by record-level governance")

    return {
        "required": True,
        "passed": not failures and len(values) == len(CONTROL_SCHEMAS),
        "failures": failures,
        "artifacts": artifacts,
        "values": values,
    }


def _validate_governance_control(receipt: dict[str, Any], failures: list[str]) -> None:
    _require_keys(
        receipt,
        {
            "passed", "purpose", "record_count", "authorized_record_count", "blocked_record_count",
            "pii_finding_count", "blocked_reasons", "policy", "requested_policy", "policy_violations",
            "policy_fingerprint", "scan_fingerprint", "record_statuses", "findings",
        },
        "governance_receipt",
        failures,
    )
    statuses = receipt.get("record_statuses") if isinstance(receipt.get("record_statuses"), list) else []
    findings = receipt.get("findings") if isinstance(receipt.get("findings"), list) else []
    passing = [row for row in statuses if isinstance(row, dict) and row.get("passed") is True]
    if (
        receipt.get("passed") is not True
        or receipt.get("purpose") != "agent_training"
        or receipt.get("record_count") != len(statuses)
        or receipt.get("authorized_record_count") != len(passing)
        or receipt.get("authorized_record_count") != receipt.get("record_count")
        or receipt.get("blocked_record_count") != 0
        or receipt.get("pii_finding_count") != 0
        or receipt.get("blocked_reasons") != []
        or receipt.get("policy_violations") != []
        or findings
        or any(row.get("blocked_reasons") or row.get("pii_finding_count") != 0 for row in passing)
    ):
        failures.append("governance_receipt: control is not a complete passing agent_training admission")
    if receipt.get("policy_fingerprint") != canonical_sha256(receipt.get("policy")):
        failures.append("governance_receipt: policy_fingerprint is stale")
    scan_view = [{key: value for key, value in row.items() if key != "preview"} for row in findings if isinstance(row, dict)]
    if receipt.get("scan_fingerprint") != canonical_sha256(scan_view):
        failures.append("governance_receipt: scan_fingerprint is stale")


def _validate_contamination_control(report: dict[str, Any], failures: list[str]) -> None:
    clusters = report.get("clusters") if isinstance(report.get("clusters"), list) else []
    cross_split = report.get("cross_split_clusters") if isinstance(report.get("cross_split_clusters"), list) else []
    protected = report.get("protected_matches") if isinstance(report.get("protected_matches"), list) else []
    if (
        report.get("passed") is not True
        or report.get("cluster_count") != len(clusters)
        or report.get("duplicate_cluster_count") != sum(
            1 for row in clusters if isinstance(row, dict) and row.get("row_count", 0) > 1
        )
        or report.get("cross_split_cluster_count") != len(cross_split)
        or report.get("protected_match_count") != len(protected)
        or cross_split
        or protected
        or report.get("blocking_reasons") != []
    ):
        failures.append("contamination_report: control contains contamination or inconsistent counts")
    identity = {
        "threshold": report.get("similarity_threshold"),
        "clusters": clusters,
        "cross_split": cross_split,
        "protected_matches": protected,
    }
    if report.get("report_fingerprint") != canonical_sha256(identity):
        failures.append("contamination_report: report_fingerprint is stale")


def _validate_curation_control(curated: dict[str, Any], failures: list[str]) -> None:
    selected = curated.get("selected") if isinstance(curated.get("selected"), list) else []
    excluded = curated.get("excluded") if isinstance(curated.get("excluded"), list) else []
    identity = {
        "recipe": curated.get("recipe"),
        "selected": [canonical_sha256(row) for row in selected],
        "excluded": excluded,
    }
    fingerprint = canonical_sha256(identity)
    if (
        curated.get("input_count") != len(selected) + len(excluded)
        or curated.get("selected_count") != len(selected)
        or curated.get("excluded_count") != len(excluded)
        or not selected
    ):
        failures.append("curated_dataset: selected/excluded counts are inconsistent or empty")
    if curated.get("recipe_fingerprint") != canonical_sha256(curated.get("recipe")):
        failures.append("curated_dataset: recipe_fingerprint is stale")
    if curated.get("selection_fingerprint") != fingerprint:
        failures.append("curated_dataset: selection_fingerprint is stale")
    if curated.get("curation_id") != f"hfrcur-{fingerprint[:16]}":
        failures.append("curated_dataset: curation_id is stale")


def _validate_action_credit_control(rows: list[dict[str, Any]], failures: list[str]) -> None:
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        key = (str(row.get("episode_id") or ""), str(row.get("tool_call_id") or ""))
        if not all(key) or key in seen:
            failures.append(f"action_credit[{index}]: episode/tool-call binding is missing or duplicated")
        seen.add(key)
        reward = row.get("reward")
        expected_reward = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}.get(row.get("label"))
        if (
            row.get("source") != "deterministic_tool_result"
            or expected_reward is None
            or reward != expected_reward
            or not isinstance(row.get("action"), dict)
            or str(row["action"].get("id") or "") != key[1]
        ):
            failures.append(f"action_credit[{index}]: deterministic action/reward binding is invalid")


def _validate_branch_replay_control(replay: dict[str, Any], failures: list[str]) -> None:
    preferences = replay.get("preferences") if isinstance(replay.get("preferences"), list) else []
    verifiers = replay.get("verifier_results") if isinstance(replay.get("verifier_results"), list) else []
    verifier_by_id = {
        str(row.get("candidate_id")): row
        for row in verifiers
        if isinstance(row, dict) and row.get("candidate_id")
    }
    chosen_id = str(replay.get("chosen_candidate_id") or "")
    boundary = replay.get("generation_boundary") if isinstance(replay.get("generation_boundary"), dict) else {}
    if (
        replay.get("review_required") is not False
        or replay.get("review_reasons") != []
        or replay.get("preference_count") != len(preferences)
        or replay.get("candidate_count") != len(verifiers)
        or not preferences
        or not chosen_id
        or boundary.get("source_state_replay_required") is not True
        or boundary.get("provider_calls_started") is not False
    ):
        failures.append("branch_replay: replay is not a complete verified no-review-required control")
    chosen_verifier = verifier_by_id.get(chosen_id, {})
    if chosen_verifier.get("passed") is not True or chosen_verifier.get("safe") is not True:
        failures.append("branch_replay: chosen candidate is not verified and safe")
    for index, preference in enumerate(preferences):
        if not isinstance(preference, dict):
            failures.append(f"branch_replay.preferences[{index}]: row is invalid")
            continue
        if (
            preference.get("chosen_candidate_id") != chosen_id
            or preference.get("chosen") == preference.get("rejected")
            or preference.get("chosen_verifier") != chosen_verifier
            or preference.get("rejected_verifier") != verifier_by_id.get(str(preference.get("rejected_candidate_id") or ""))
        ):
            failures.append(f"branch_replay.preferences[{index}]: verifier/content binding is invalid")


def _validate_preference_control(rows: list[dict[str, Any]], failures: list[str]) -> None:
    seen: set[str] = set()
    for index, row in enumerate(rows):
        preference_id = str(row.get("preference_id") or "")
        if not preference_id or preference_id in seen:
            failures.append(f"reviewed_preferences[{index}]: preference id is missing or duplicated")
        seen.add(preference_id)
        chosen_hash = canonical_sha256(row.get("chosen"))
        rejected_hash = canonical_sha256(row.get("rejected"))
        if (
            chosen_hash == rejected_hash
            or row.get("chosen_completion_sha256") != chosen_hash
            or row.get("rejected_completion_sha256") != rejected_hash
            or not _is_sha256(row.get("task_contract_fingerprint"))
            or not _is_sha256(row.get("chosen_review_item_sha256"))
            or not _is_sha256(row.get("rejected_review_item_sha256"))
            or row.get("chosen_reviewer_confidence") not in {"high", "medium"}
            or row.get("rejected_reviewer_confidence") not in {"high", "medium"}
        ):
            failures.append(f"reviewed_preferences[{index}]: human-review/content binding is invalid")


def _json_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True)


def _tool_call_ids(row: dict[str, Any]) -> set[str]:
    return {
        str(call.get("id"))
        for message in row.get("messages", [])
        if isinstance(message, dict)
        for call in (message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else [])
        if isinstance(call, dict) and call.get("id")
    }


def _validate_training_control_bindings(
    raw: dict[str, list[dict[str, Any]]],
    evidence: dict[str, Any],
) -> list[str]:
    values = evidence.get("values") if isinstance(evidence.get("values"), dict) else {}
    governance = values.get("governance_receipt") if isinstance(values.get("governance_receipt"), dict) else {}
    curated = values.get("curated_dataset") if isinstance(values.get("curated_dataset"), dict) else {}
    credits = values.get("action_credit") if isinstance(values.get("action_credit"), list) else []
    replay = values.get("branch_replay") if isinstance(values.get("branch_replay"), dict) else {}
    preferences = values.get("reviewed_preferences") if isinstance(values.get("reviewed_preferences"), list) else []
    failures: list[str] = []

    governance_by_id = {
        str(row.get("record_id")): row
        for row in governance.get("record_statuses", [])
        if isinstance(row, dict) and row.get("record_id")
    }
    selected_by_id = {
        str(row.get("episode_id") or row.get("row_id")): row
        for row in curated.get("selected", [])
        if isinstance(row, dict) and (row.get("episode_id") or row.get("row_id"))
    }
    credits_by_key = {
        (str(row.get("episode_id") or ""), str(row.get("tool_call_id") or "")): row
        for row in credits
        if isinstance(row, dict)
    }
    negative_credit_ids = {
        str(row.get("episode_id"))
        for row in credits
        if isinstance(row, dict) and row.get("label") == "negative" and row.get("episode_id")
    }

    for data_name in ("fr_sft", "fr_action_sft"):
        for index, row in enumerate(raw.get(data_name, [])):
            episode_id = str(row.get("episode_id") or "")
            selected = selected_by_id.get(episode_id)
            status = governance_by_id.get(episode_id)
            label = f"{data_name}[{index}]"
            if selected is None:
                failures.append(f"{label}: row is not selected by the curated dataset")
                continue
            for key in ("messages", "tools", "governance", "review_item_sha256", "human_label", "tool_schema_provenance"):
                if row.get(key) != selected.get(key):
                    failures.append(f"{label}: {key} does not match the curated source row")
            if (
                row.get("human_label") != "accept"
                or not _is_sha256(row.get("review_item_sha256"))
                or row.get("reviewer_confidence") not in {"high", "medium"}
                or row.get("quality_gate") != "human_reviewed_native_action_accept"
                or row.get("source_artifact") != "controls/curated_dataset.json"
                or row.get("credit_policy") != "exclude_entire_trajectory_on_any_negative_tool_action"
            ):
                failures.append(f"{label}: row is not bound to an accepted human review and credit policy")
            if not isinstance(status, dict) or status.get("passed") is not True:
                failures.append(f"{label}: row has no passing record-level governance decision")
            elif status.get("governance_fingerprint") != canonical_sha256(row.get("governance")):
                failures.append(f"{label}: row governance fingerprint does not match the governance receipt")
            if episode_id in negative_credit_ids:
                failures.append(f"{label}: row has negative action credit")
            call_ids = _tool_call_ids(row)
            episode_credits = {call_id for credit_episode, call_id in credits_by_key if credit_episode == episode_id}
            if episode_credits and not call_ids.issubset(episode_credits):
                failures.append(f"{label}: action-credit rows do not cover every recorded tool call")

    human_by_id = {
        str(row.get("preference_id")): row
        for row in preferences
        if isinstance(row, dict) and row.get("preference_id")
    }
    replay_by_id = {
        str(row.get("preference_id")): row
        for row in replay.get("preferences", [])
        if isinstance(row, dict) and row.get("preference_id")
    }
    for index, row in enumerate(raw.get("fr_dpo", [])):
        preference_id = str(row.get("preference_id") or "")
        arm = row.get("training_arm")
        label = f"fr_dpo[{index}]"
        if arm == "flightrecorder_human_reviewed_dpo":
            source = human_by_id.get(preference_id)
            if source is None:
                failures.append(f"{label}: human preference is not present in reviewed control rows")
                continue
            for key in ("prompt", "chosen", "rejected", "tools", "task_contract_fingerprint", "chosen_completion_sha256", "rejected_completion_sha256"):
                if row.get(key) != source.get(key):
                    failures.append(f"{label}: {key} does not match reviewed human preference")
            if row.get("source_artifact") != "controls/reviewed_preferences.jsonl":
                failures.append(f"{label}: source_artifact does not bind reviewed preferences")
        elif arm == "flightrecorder_verified_branch_replay_dpo":
            source = replay_by_id.get(preference_id)
            if source is None:
                failures.append(f"{label}: replay preference is not present in branch-replay control")
                continue
            if (
                _json_text(row.get("chosen")) != _json_text(source.get("chosen"))
                or _json_text(row.get("rejected")) != _json_text(source.get("rejected"))
                or row.get("source_artifact") != "controls/branch_replay.json"
            ):
                failures.append(f"{label}: content does not match verified branch replay")
        else:
            failures.append(f"{label}: training arm is not governed by human review or verified replay")

    for index, row in enumerate(raw.get("fr_reward_model", [])):
        sample_id = str(row.get("sample_id") or "")
        preference_id, separator, label_name = sample_id.rpartition(":")
        source = human_by_id.get(preference_id)
        expected_value = source.get("chosen") if label_name == "accept" and source else source.get("rejected") if source else None
        if (
            not separator
            or source is None
            or row.get("prompt") != source.get("prompt")
            or row.get("response") != _json_text(expected_value)
            or row.get("human_label") != label_name
            or row.get("reward") != (1.0 if label_name == "accept" else -1.0)
            or row.get("source_artifact") != "controls/reviewed_preferences.jsonl"
        ):
            failures.append(f"fr_reward_model[{index}]: row does not replay a reviewed human preference")

    for index, row in enumerate(raw.get("fr_step_rewards", [])):
        source = credits_by_key.get((str(row.get("episode_id") or ""), str(row.get("tool_call_id") or "")))
        comparable = {key: value for key, value in row.items() if key not in {"task_family", "source_artifact"}}
        if source is None or comparable != source or row.get("source_artifact") != "controls/action_credit.jsonl":
            failures.append(f"fr_step_rewards[{index}]: row does not exactly replay deterministic action credit")
    return failures


def add_check(
    checks: list[dict[str, Any]],
    check_id: str,
    passed: bool,
    summary: str,
    *,
    actual: dict[str, Any] | None = None,
    expected: dict[str, Any] | None = None,
    scope: dict[str, Any] | None = None,
) -> None:
    checks.append(
        {
            "id": check_id,
            "passed": passed,
            "summary": summary,
            "actual": actual or {"passed": passed},
            "expected": expected or {"passed": True},
            "scope": scope or {},
        }
    )


def _required_count_keys(mode: str) -> tuple[str, ...]:
    if mode in {"trace_sft", "fr_sft"}:
        return ("sft",)
    if mode == "fr_action_sft":
        return ("action_sft",)
    if mode == "fr_dpo":
        return ("dpo",)
    if mode == "fr_sft_dpo":
        return ("sft", "dpo")
    if mode == "fr_reward_model":
        return ("reward_model",)
    if mode == "fr_step_rewards":
        return ("step_rewards",)
    return ()


def write_smoke_fixture(root: Path) -> dict[str, Any]:
    """Write a tiny registered training fixture without model downloads."""
    data_dir = root / "data"
    registry_dir = root / "registry"
    model_manifest = registry_dir / "model_candidate.json"
    dataset_manifest = registry_dir / "dataset_version.json"
    fixture_manifest = root / "smoke_fixture.json"
    smoke_governance = {
        "owner": "hfr-smoke-fixture",
        "tenant": "public-synthetic",
        "legal_basis": "contract",
        "allowed_purposes": ["agent_training"],
        "sensitivity": "public-synthetic",
        "jurisdiction": "US",
        "retention_expires_at": "2035-01-01T00:00:00+00:00",
        "license": "Apache-2.0-synthetic-fixture",
        "provenance": {"source": "trainer-smoke", "source_revision": "v1"},
        "deletion_subject_ids": ["hfr-smoke-fixture"],
    }
    smoke_tool = {
        "type": "function",
        "version": "1.0.0",
        "function": {
            "name": "fixture.inspect",
            "description": "Inspect one synthetic fixture.",
            "parameters": {
                "type": "object",
                "properties": {"artifact": {"type": "string"}},
                "required": ["artifact"],
                "additionalProperties": False,
            },
        },
    }

    counts = {
        "trace_sft": write_jsonl(
            data_dir / "hermes_trace_only_sft.jsonl",
            [
                {"sample_id": "trace-1", "prompt": "Summarize the checked artifact.", "response": "The artifact is present."},
                {"sample_id": "trace-2", "prompt": "Report the command status.", "response": "The command completed successfully."},
            ],
        ),
        "fr_sft": write_jsonl(
            data_dir / "flightrecorder_sft.jsonl",
            [
                {"sample_id": "sft-1", "prompt": "Use evidence before answering.", "response": "I verified the evidence and then answered."},
                {"sample_id": "sft-2", "prompt": "Handle a blocked action.", "response": "I refused the unsafe action and explained why."},
            ],
        ),
        "fr_action_sft": write_jsonl(
            data_dir / "flightrecorder_action_sft.jsonl",
            [
                {
                    "sample_id": "action-1",
                    "episode_id": "action-1",
                    "prompt": "Inspect file metadata.",
                    "response": "The synthetic artifact is present.",
                    "messages": [
                        {"role": "user", "content": "Inspect file metadata."},
                        {"role": "assistant", "content": "The synthetic artifact is present."},
                    ],
                    "tools": [smoke_tool],
                    "human_label": "accept",
                    "tool_schema_provenance": "recorded_exact",
                    "quality_gate": "human_reviewed_native_action_accept",
                    "credit_policy": "exclude_entire_trajectory_on_any_negative_tool_action",
                    "governance": smoke_governance,
                },
                {
                    "sample_id": "action-2",
                    "episode_id": "action-2",
                    "prompt": "Validate schema.",
                    "response": "The synthetic schema is valid.",
                    "messages": [
                        {"role": "user", "content": "Validate schema."},
                        {"role": "assistant", "content": "The synthetic schema is valid."},
                    ],
                    "tools": [smoke_tool],
                    "human_label": "accept",
                    "tool_schema_provenance": "recorded_exact",
                    "quality_gate": "human_reviewed_native_action_accept",
                    "credit_policy": "exclude_entire_trajectory_on_any_negative_tool_action",
                    "governance": smoke_governance,
                },
            ],
        ),
        "fr_dpo": write_jsonl(
            data_dir / "flightrecorder_combined_dpo.jsonl",
            [
                {
                    "sample_id": "dpo-1",
                    "prompt": "Should you claim a file exists without checking?",
                    "chosen": "No. I should inspect the file first.",
                    "rejected": "Yes, I can assume it exists.",
                    "training_arm": "flightrecorder_human_reviewed_dpo",
                },
                {
                    "sample_id": "dpo-2",
                    "prompt": "How should an unsafe training gate be handled?",
                    "chosen": "Block launch and report the failed gate.",
                    "rejected": "Launch anyway and fix it later.",
                    "training_arm": "flightrecorder_verified_branch_replay_dpo",
                },
            ],
        ),
        "fr_reward_model": write_jsonl(
            data_dir / "flightrecorder_reward_model.jsonl",
            [
                {"sample_id": "reward-1", "prompt": "Evidence-backed answer", "response": "Verified.", "reward": 1, "human_label": "accept", "source_artifact": "controls/reviewed_preferences.jsonl"},
                {"sample_id": "reward-2", "prompt": "Unsupported claim", "response": "I assume it passed.", "reward": -1, "human_label": "reject", "source_artifact": "controls/reviewed_preferences.jsonl"},
            ],
        ),
        "fr_step_rewards": write_jsonl(
            data_dir / "flightrecorder_step_rewards.jsonl",
            [
                {"episode_id": "smoke-episode-1", "target": "event:inspect", "reward": 1, "source": "deterministic_tool_result", "source_artifact": "controls/action_credit.jsonl"},
                {"episode_id": "smoke-episode-1", "target": "event:claim_without_evidence", "reward": -1, "source": "deterministic_tool_result", "source_artifact": "controls/action_credit.jsonl"},
            ],
        ),
    }

    # The smoke fixture is deliberately small, but it still exercises the same
    # content-bound controls required by a real Flight Recorder launch.
    reviewed_actions: list[dict[str, Any]] = []
    for index, episode_id in enumerate(("action-1", "action-2"), start=1):
        prompt = "Inspect file metadata." if index == 1 else "Validate schema."
        response = "The synthetic artifact is present." if index == 1 else "The synthetic schema is valid."
        call_id = f"call-{episode_id}"
        reviewed_actions.append(
            {
                "schema_version": "hfr.reviewed.action_sft.v1",
                "sample_id": episode_id,
                "episode_id": episode_id,
                "scenario_id": episode_id,
                "task_family": "fixture",
                "prompt": prompt,
                "response": response,
                "messages": [
                    {"role": "user", "content": prompt},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {"name": "fixture.inspect", "arguments": {"artifact": episode_id}},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": "fixture.inspect",
                        "status": "ok",
                        "content": "{\"status\":\"ok\"}",
                    },
                    {"role": "assistant", "content": response},
                ],
                "tools": [smoke_tool],
                "human_label": "accept",
                "review_item_id": f"review-{episode_id}",
                "review_item_sha256": canonical_sha256({"episode_id": episode_id, "label": "accept"}),
                "reviewer_confidence": "high",
                "tool_schema_provenance": "recorded_exact",
                "quality_gate": "human_reviewed_native_action_accept",
                "credit_policy": "exclude_entire_trajectory_on_any_negative_tool_action",
                "governance": smoke_governance,
                "source_artifact": "controls/curated_dataset.json",
                "training_arm": "flightrecorder_action_sft",
            }
        )
    write_jsonl(data_dir / "flightrecorder_action_sft.jsonl", reviewed_actions)
    write_jsonl(
        data_dir / "flightrecorder_sft.jsonl",
        [{**row, "training_arm": "flightrecorder_human_reviewed_sft"} for row in reviewed_actions],
    )

    action_credits = [
        {
            "schema_version": "hfr.action_credit.v1",
            "episode_id": row["episode_id"],
            "message_index": 1,
            "tool_call_id": f"call-{row['episode_id']}",
            "tool_name": "fixture.inspect",
            "label": "positive",
            "reward": 1.0,
            "status": "ok",
            "confidence": 1.0,
            "source": "deterministic_tool_result",
            "episode_outcome": "success",
            "observation": row["messages"][2],
            "action": row["messages"][1]["tool_calls"][0],
        }
        for row in reviewed_actions
    ]

    human_preference = {
        "schema_version": "hfr.reviewed.contract_preference.v1",
        "preference_id": "fixture-contract:accepted>rejected",
        "task_contract_fingerprint": canonical_sha256({"fixture": "human-preference"}),
        "task_family": "fixture",
        "prompt": "Should you claim a file exists without checking?",
        "tools": [smoke_tool],
        "chosen_episode_id": "fixture-accepted",
        "rejected_episode_id": "fixture-rejected",
        "chosen": "No. I should inspect the file first.",
        "rejected": "Yes, I can assume it exists.",
        "chosen_review_item_sha256": "a" * 64,
        "rejected_review_item_sha256": "b" * 64,
        "chosen_reviewer_confidence": "high",
        "rejected_reviewer_confidence": "high",
        "reason": "Human-reviewed outcomes under an identical synthetic task contract.",
    }
    human_preference["chosen_completion_sha256"] = canonical_sha256(human_preference["chosen"])
    human_preference["rejected_completion_sha256"] = canonical_sha256(human_preference["rejected"])

    replay_preference = {
        "preference_id": "replay:fixture:verified>rejected",
        "chosen_candidate_id": "verified",
        "rejected_candidate_id": "rejected",
        "chosen": [{"role": "assistant", "content": "Block launch and report the failed gate."}],
        "rejected": [{"role": "assistant", "content": "Launch anyway and fix it later."}],
        "chosen_verifier": {"candidate_id": "verified", "passed": True, "safe": True, "score": 100, "confidence": 1.0},
        "rejected_verifier": {"candidate_id": "rejected", "passed": False, "safe": False, "score": 0, "confidence": 1.0},
    }
    branch_replay = {
        "schema_version": "hfr.branch_replay_dataset.v1",
        "replay_id": "replay-" + canonical_sha256(replay_preference)[:16],
        "source_episode_id": "action-1",
        "source_trajectory_sha256": canonical_sha256(reviewed_actions[0]),
        "source_prefix_messages": [{"role": "user", "content": "How should an unsafe training gate be handled?"}],
        "tools": [smoke_tool],
        "task_family": "fixture",
        "replay_point": {"event_index": 1, "state_fingerprint": "f" * 64},
        "candidate_count": 2,
        "chosen_candidate_id": "verified",
        "preference_count": 1,
        "preferences": [replay_preference],
        "verifier_results": [replay_preference["rejected_verifier"], replay_preference["chosen_verifier"]],
        "review_required": False,
        "review_reasons": [],
        "generation_boundary": {
            "continuations_generated_by_flight_recorder": False,
            "provider_calls_started": False,
            "source_state_replay_required": True,
        },
    }
    write_jsonl(
        data_dir / "flightrecorder_combined_dpo.jsonl",
        [
            {
                **human_preference,
                "sample_id": "dpo-1",
                "source_artifact": "controls/reviewed_preferences.jsonl",
                "training_arm": "flightrecorder_human_reviewed_dpo",
            },
            {
                "sample_id": "dpo-2",
                "preference_id": replay_preference["preference_id"],
                "prompt": _json_text(branch_replay["source_prefix_messages"]),
                "chosen": _json_text(replay_preference["chosen"]),
                "rejected": _json_text(replay_preference["rejected"]),
                "source_artifact": "controls/branch_replay.json",
                "training_arm": "flightrecorder_verified_branch_replay_dpo",
            },
        ],
    )
    write_jsonl(
        data_dir / "flightrecorder_reward_model.jsonl",
        [
            {
                "sample_id": f"{human_preference['preference_id']}:{label}",
                "prompt": human_preference["prompt"],
                "response": human_preference[value],
                "reward": reward,
                "human_label": label,
                "source_artifact": "controls/reviewed_preferences.jsonl",
            }
            for label, value, reward in (("accept", "chosen", 1.0), ("reject", "rejected", -1.0))
        ],
    )
    write_jsonl(
        data_dir / "flightrecorder_step_rewards.jsonl",
        [{**row, "task_family": "fixture", "source_artifact": "controls/action_credit.jsonl"} for row in action_credits],
    )

    control_dir = root / "controls"
    governance_statuses = [
        {
            "record_id": row["episode_id"],
            "passed": True,
            "blocked_reasons": [],
            "governance_fingerprint": canonical_sha256(row["governance"]),
            "pii_finding_count": 0,
        }
        for row in reviewed_actions
    ]
    governance_policy = {
        "purpose": "agent_training",
        "required_fields": [
            "owner", "tenant", "legal_basis", "allowed_purposes", "sensitivity", "jurisdiction",
            "retention_expires_at", "license", "provenance", "deletion_subject_ids",
        ],
        "pii_policy": "block_unredacted",
        "unknown_license_allowed": False,
    }
    governance_receipt = {
        "schema_version": "hfr.data_governance_receipt.v1",
        "passed": True,
        "purpose": "agent_training",
        "record_count": len(reviewed_actions),
        "authorized_record_count": len(reviewed_actions),
        "blocked_record_count": 0,
        "pii_finding_count": 0,
        "blocked_reasons": [],
        "policy": governance_policy,
        "requested_policy": {},
        "policy_violations": [],
        "policy_fingerprint": canonical_sha256(governance_policy),
        "scan_fingerprint": canonical_sha256([]),
        "record_statuses": governance_statuses,
        "findings": [],
        "evaluated_at": "2028-01-01T00:00:00+00:00",
        "notes": ["Synthetic smoke control."],
    }
    clusters = [
        {"cluster_id": f"cluster-{index:06d}", "row_ids": [row["episode_id"]], "row_count": 1, "splits": [], "exact_duplicate": False}
        for index, row in enumerate(reviewed_actions)
    ]
    contamination_identity = {"threshold": 0.9, "clusters": clusters, "cross_split": [], "protected_matches": []}
    contamination_report = {
        "schema_version": "hfr.dataset_contamination_report.v1",
        "passed": True,
        "similarity_threshold": 0.9,
        "row_count": len(reviewed_actions),
        "cluster_count": len(clusters),
        "duplicate_cluster_count": 0,
        "cross_split_cluster_count": 0,
        "protected_match_count": 0,
        "clusters": clusters,
        "cross_split_clusters": [],
        "protected_matches": [],
        "report_fingerprint": canonical_sha256(contamination_identity),
        "blocking_reasons": [],
    }
    selected = [
        {
            **{key: value for key, value in row.items() if key not in {"source_artifact", "training_arm", "quality_gate", "credit_policy"}},
            "training_role": "action_sft",
            "source_id": "trainer-smoke",
            "quality_score": 1.0,
            "selection_reason": "passed_recipe",
            "selection_weight": 1.0,
        }
        for row in reviewed_actions
    ]
    recipe = {"seed": "trainer-smoke-v1", "allowed_roles": ["action_sft"], "max_rows": 2, "max_per_source": 2, "minimum_quality": 0.0, "mixture_weights": {"action_sft": 1.0}}
    curation_identity = {"recipe": recipe, "selected": [canonical_sha256(row) for row in selected], "excluded": []}
    selection_fingerprint = canonical_sha256(curation_identity)
    curated_dataset = {
        "schema_version": "hfr.curated_dataset.v1",
        "curation_id": f"hfrcur-{selection_fingerprint[:16]}",
        "recipe": recipe,
        "recipe_fingerprint": canonical_sha256(recipe),
        "input_count": len(selected),
        "selected_count": len(selected),
        "excluded_count": 0,
        "selected": selected,
        "excluded": [],
        "selected_role_counts": [{"value": "action_sft", "count": len(selected)}],
        "selected_family_counts": [{"value": "fixture", "count": len(selected)}],
        "selected_source_counts": [{"value": "trainer-smoke", "count": len(selected)}],
        "effective_sample_size": float(len(selected)),
        "selection_fingerprint": selection_fingerprint,
    }
    control_payloads: dict[str, tuple[str, Any]] = {
        "governance_receipt": ("governance_receipt.json", governance_receipt),
        "contamination_report": ("contamination_report.json", contamination_report),
        "curated_dataset": ("curated_dataset.json", curated_dataset),
        "action_credit": ("action_credit.jsonl", action_credits),
        "branch_replay": ("branch_replay.json", branch_replay),
        "reviewed_preferences": ("reviewed_preferences.jsonl", [human_preference]),
    }
    for _, (filename, payload) in control_payloads.items():
        destination = control_dir / filename
        if destination.suffix == ".jsonl":
            write_jsonl(destination, payload)
        else:
            write_json(destination, payload)
    write_json(
        model_manifest,
        {
            "schema_version": "hfr.model_candidate.v1",
            "model_id": "local/hfr-smoke-model",
            "source": "local-fixture",
            "license_status": "approved",
            "training_allowed": True,
            "compatibility": {
                "tokenizer": "fixture-only",
                "chat_template": "messages",
                "serving": "not_required_for_dry_run",
                "tool_calls": "text-fixture",
            },
        },
    )
    fixture_data_files = {
        "trace_sft": "../data/hermes_trace_only_sft.jsonl",
        "flightrecorder_sft": "../data/flightrecorder_sft.jsonl",
        "flightrecorder_action_sft": "../data/flightrecorder_action_sft.jsonl",
        "flightrecorder_combined_dpo": "../data/flightrecorder_combined_dpo.jsonl",
        "flightrecorder_reward_model": "../data/flightrecorder_reward_model.jsonl",
        "flightrecorder_step_rewards": "../data/flightrecorder_step_rewards.jsonl",
        "governance_receipt": "../controls/governance_receipt.json",
        "contamination_report": "../controls/contamination_report.json",
        "curated_dataset": "../controls/curated_dataset.json",
        "action_credit": "../controls/action_credit.jsonl",
        "branch_replay": "../controls/branch_replay.json",
        "reviewed_preferences": "../controls/reviewed_preferences.jsonl",
    }
    write_json(
        dataset_manifest,
        {
            "schema_version": "hfr.dataset_registry_entry.v1",
            "dataset_id": "hfr-smoke-fixture",
            "dataset_version": "hfr-smoke-fixture.v1",
            "redaction_status": "redacted",
            "gates": {
                "training_gate": {"passed": True},
                "governance": {"passed": True},
                "contamination": {"passed": True},
                "human_reviewed_curation": {"passed": True},
                "per_action_credit": {"passed": True},
                "verified_branch_replay": {"passed": True},
                "human_rejection_preferences": {"passed": True},
            },
            "dataset_splits": {"family_exclusive": True},
            "quality_flags": [],
            "source_fingerprint_coverage": {"fully_verified": sum(counts.values()), "unverified": 0},
            "data_files": fixture_data_files,
            "artifact_fingerprints": {
                name: {
                    "path": relative_path,
                    "exists": True,
                    "size_bytes": (registry_dir / relative_path).resolve().stat().st_size,
                    "sha256": hashlib.sha256((registry_dir / relative_path).resolve().read_bytes()).hexdigest(),
                }
                for name, relative_path in fixture_data_files.items()
            },
        },
    )
    fixture = {
        "schema_version": "hfr.agentic_lora_smoke_fixture.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fixture_dir": str(root),
        "model_manifest": str(model_manifest),
        "dataset_manifest": str(dataset_manifest),
        "experiment_dir": str(root),
        "row_counts": counts,
        "recommended_commands": [
            (
                "python3 scripts/train_agentic_lora.py --mode fr_sft_dpo --dry-run --require-registered-inputs "
                f"--experiment-dir {root} --model-manifest {model_manifest} --dataset-manifest {dataset_manifest} "
                f"--output-dir {root / 'out'} --limit 1 --disable-trackio"
            ),
            (
                "python3 scripts/train_agentic_lora.py --mode fr_action_sft --dry-run --require-registered-inputs "
                f"--experiment-dir {root} --model-manifest {model_manifest} --dataset-manifest {dataset_manifest} "
                f"--output-dir {root / 'out'} --limit 1 --disable-trackio"
            ),
        ],
        "notes": [
            "This fixture is for dry-run and row-limit smoke checks only.",
            "It does not download a model or launch training.",
        ],
    }
    write_json(fixture_manifest, fixture)
    return fixture


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    model_context = load_model_context(args.model_manifest)
    dataset_context = load_dataset_context(args.dataset_manifest)
    original_model = args.model
    manifest_model_id = str(model_context.get("model_id") or "")
    if manifest_model_id and args.model == DEFAULT_MODEL:
        args.model = manifest_model_id

    paths = data_paths(args.experiment_dir, dataset_context)
    raw = {name: load_jsonl(path) if path.exists() else [] for name, path in paths.items()}
    sft_source = raw["trace_sft"] if args.mode == "trace_sft" else raw["fr_sft"] + raw["fr_action_sft"]
    action_sft_source = raw["fr_action_sft"]
    dpo_source = raw["fr_dpo"]
    sft_rows = prepare_sft_rows(sft_source)
    action_sft_rows = prepare_sft_rows(action_sft_source)
    dpo_rows = prepare_dpo_rows(dpo_source)
    reward_model_rows = raw["fr_reward_model"]
    step_reward_rows = raw["fr_step_rewards"]
    full_prepared_counts = {
        "sft": len(sft_rows),
        "action_sft": len(action_sft_rows),
        "dpo": len(dpo_rows),
        "reward_model": len(reward_model_rows),
        "step_rewards": len(step_reward_rows),
    }
    if args.limit:
        sft_rows = sft_rows[: args.limit]
        action_sft_rows = action_sft_rows[: args.limit]
        dpo_rows = dpo_rows[: args.limit]
        reward_model_rows = reward_model_rows[: args.limit]
        step_reward_rows = step_reward_rows[: args.limit]

    is_flight_recorder_mode = args.mode in FLIGHT_RECORDER_MODES
    requires_registered_inputs = (
        is_flight_recorder_mode
        or args.require_registered_inputs
        or (not args.dry_run and not args.unsafe_allow_unregistered_launch)
    )
    control_evidence = _load_control_evidence(dataset_context) if is_flight_recorder_mode else {
        "required": False,
        "passed": True,
        "failures": [],
        "artifacts": {},
        "values": {},
    }
    binding_failures = _validate_training_control_bindings(raw, control_evidence) if is_flight_recorder_mode else []
    checks: list[dict[str, Any]] = []
    add_check(checks, "mode_supported", args.mode in SUPPORTED_MODES, f"mode {args.mode!r} is known")
    add_check(
        checks,
        "mode_launch_supported",
        args.dry_run or args.mode in EXECUTABLE_MODES,
        f"mode {args.mode!r} {'has' if args.mode in EXECUTABLE_MODES else 'does not yet have'} a local TRL/PEFT launch path",
    )
    add_check(
        checks,
        "registered_inputs_required",
        (not requires_registered_inputs) or (model_context["provided"] and dataset_context["provided"]),
        "Flight Recorder plans and ordinary trainer launches require registered model and dataset manifests",
        actual={
            "required": requires_registered_inputs,
            "model_manifest": model_context["provided"],
            "dataset_manifest": dataset_context["provided"],
            "unsafe_allow_unregistered_launch": args.unsafe_allow_unregistered_launch,
            "flight_recorder_bypass_disabled": is_flight_recorder_mode,
        },
    )
    _add_model_checks(checks, model_context, original_model, args.model, requires_registered_inputs)
    _add_dataset_checks(checks, dataset_context, requires_registered_inputs, args.mode)
    _add_data_checks(checks, args.mode, paths, raw, {
        "sft": len(sft_rows),
        "action_sft": len(action_sft_rows),
        "dpo": len(dpo_rows),
        "reward_model": len(reward_model_rows),
        "step_rewards": len(step_reward_rows),
    })
    if is_flight_recorder_mode:
        add_check(
            checks,
            "content_bound_control_artifacts_replayed",
            control_evidence["passed"] is True,
            "governance, contamination, curation, action-credit, branch-replay, and human-preference artifacts must replay from registered bytes",
            actual={
                "artifacts": sorted(control_evidence["artifacts"]),
                "failures": control_evidence["failures"],
            },
            expected={"required_artifacts": sorted(CONTROL_SCHEMAS)},
        )
        add_check(
            checks,
            "training_rows_bound_to_control_artifacts",
            not binding_failures,
            "every Flight Recorder row must be selected, reviewed, governed, credit-checked, or replayed from its source control",
            actual={"failures": binding_failures},
        )
    _add_hyperparameter_checks(checks, args)
    failed_checks = [check for check in checks if check["passed"] is False]

    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "model": args.model,
        "input_identity": {
            "model_revision": args.model_revision,
            "tokenizer_revision": args.tokenizer_revision,
            "expected_chat_template_sha256": args.expected_chat_template_sha256,
        },
        "experiment_dir": str(args.experiment_dir),
        "output_dir": str(args.output_dir),
        "hub_model_id": args.hub_model_id,
        "push_to_hub": args.push_to_hub,
        "dry_run": args.dry_run,
        "smoke": {
            "enabled": args.limit > 0,
            "row_limit": args.limit,
            "full_plan_row_counts_preserved": True,
            "full_prepared_counts": full_prepared_counts,
        },
        "readiness": "ready" if not failed_checks else "blocked",
        "recommendation": "launch_allowed" if not failed_checks else "block_launch",
        "passed": not failed_checks,
        "check_count": len(checks),
        "failed_check_count": len(failed_checks),
        "checks": checks,
        "blocked_reasons": [check["summary"] for check in failed_checks],
        "input_manifests": {
            "model": {
                "provided": model_context["provided"],
                "path": model_context["path"],
                "schema_version": model_context.get("schema_version", ""),
                "model_id": model_context.get("model_id", ""),
                "license_status": model_context.get("license_status", ""),
            },
            "dataset": {
                "provided": dataset_context["provided"],
                "path": dataset_context["path"],
                "schema_version": dataset_context.get("schema_version", ""),
                "source_path": dataset_context.get("source_path", ""),
                "source_schema_version": dataset_context.get("source_schema_version", ""),
                "dataset_identity": _dataset_identity(dataset_context),
                "redaction_status": _redaction_status(dataset_context),
            },
        },
        "control_evidence": {
            "required": control_evidence["required"],
            "passed": control_evidence["passed"] and not binding_failures,
            "artifacts": control_evidence["artifacts"],
            "failures": [*control_evidence["failures"], *binding_failures],
        },
        "tracking": {
            "report_to": [] if args.disable_trackio else ["trackio"],
            "trackio_project": args.trackio_project,
            "trackio_space_id": args.trackio_space_id,
            "run_name_prefix": args.run_name_prefix,
        },
        "persistence": {
            "push_to_hub": args.push_to_hub,
            "hub_model_id": args.hub_model_id,
            "checkpoint_push_strategy": "every_save" if args.push_to_hub else "local_only",
            "final_adapter_push": args.push_to_hub,
        },
        "data_files": {name: str(path) for name, path in paths.items()},
        "raw_counts": {name: len(rows) for name, rows in raw.items()},
        "prepared_counts": {
            "sft": len(sft_rows),
            "action_sft": len(action_sft_rows),
            "dpo": len(dpo_rows),
            "reward_model": len(reward_model_rows),
            "step_rewards": len(step_reward_rows),
        },
        "full_prepared_counts": full_prepared_counts,
        "hyperparameters": {
            "sft_epochs": args.sft_epochs,
            "dpo_epochs": args.dpo_epochs,
            "sft_learning_rate": args.sft_learning_rate,
            "dpo_learning_rate": args.dpo_learning_rate,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "gradient_checkpointing": args.gradient_checkpointing,
            "max_steps": args.max_steps,
            "max_length": args.max_length,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "assistant_only_loss": not args.all_message_loss,
            "seed": args.seed,
            "data_seed": args.data_seed,
            "save_steps": args.save_steps,
            "save_total_limit": args.save_total_limit,
        },
        "compute_assumptions": {
            "heavy_ml_imports_deferred_until_after_plan_passes": True,
            "default_device_order": ["cuda", "mps", "cpu"],
            "dtype_policy": "bfloat16 on supported CUDA devices, float16 on other CUDA/MPS devices, float32 on CPU",
            "registered_inputs_required_for_launch": requires_registered_inputs,
        },
        "trainer_backends": {
            "default": "trl_peft_lora",
            "executable_modes": sorted(EXECUTABLE_MODES),
            "plan_only_modes": sorted(PLAN_ONLY_MODES),
            "extension_points": [
                "axolotl_recipe",
                "llama_factory_recipe",
                "unsloth_recipe",
                "reward_model_trainer",
                "process_reward_trainer",
                "grpo_rl_trainer",
            ],
        },
        "notes": [
            "Dry-run plans never import torch, datasets, transformers, peft, or trl.",
            "Every Flight Recorder dry-run claim and launch requires registered inputs and replayed content-bound controls.",
            "Reward/process-reward modes are represented as plan-only extension points until dedicated trainers are wired.",
        ],
    }
    return plan


def _add_model_checks(
    checks: list[dict[str, Any]],
    model_context: dict[str, Any],
    original_model: str,
    selected_model: str,
    required: bool,
) -> None:
    if not required and not model_context["provided"]:
        return
    add_check(checks, "model_manifest_provided", model_context["provided"], "model manifest is provided")
    if not model_context["provided"]:
        return
    add_check(checks, "model_manifest_readable", not model_context["error"], "model manifest is readable", actual={"error": model_context["error"]})
    model_id = str(model_context.get("model_id") or "")
    add_check(checks, "model_id_present", bool(model_id), "model manifest declares a model id")
    add_check(
        checks,
        "model_selection_matches_manifest",
        not model_id or original_model == DEFAULT_MODEL or original_model == model_id or selected_model == model_id,
        "selected model matches the registered model candidate",
        actual={"selected_model": selected_model, "manifest_model_id": model_id},
    )
    license_status = str(model_context.get("license_status") or "").lower()
    add_check(
        checks,
        "model_license_known",
        license_status not in UNKNOWN_LICENSE_STATUSES,
        "model license status is known and not blocked",
        actual={"license_status": license_status},
    )
    add_check(
        checks,
        "model_training_allowed",
        model_context.get("training_allowed") is not False,
        "model manifest allows fine-tuning/training",
    )
    add_check(
        checks,
        "model_compatibility_metadata_present",
        bool(model_context.get("compatibility")),
        "model manifest includes compatibility metadata",
    )


def _add_dataset_checks(
    checks: list[dict[str, Any]],
    dataset_context: dict[str, Any],
    required: bool,
    mode: str,
) -> None:
    if not required and not dataset_context["provided"]:
        return
    add_check(checks, "dataset_manifest_provided", dataset_context["provided"], "dataset manifest is provided")
    if not dataset_context["provided"]:
        return
    add_check(
        checks,
        "dataset_manifest_readable",
        not dataset_context["error"],
        "dataset manifest is readable",
        actual={"error": dataset_context["error"]},
    )
    redaction_status = _redaction_status(dataset_context)
    add_check(
        checks,
        "dataset_redaction_passed",
        redaction_status in SAFE_REDACTION_STATUSES,
        "dataset manifest records redacted/sanitized training inputs",
        actual={"redaction_status": redaction_status},
    )
    gate_passed = _training_gate_passed(dataset_context)
    add_check(
        checks,
        "dataset_training_gate_passed",
        gate_passed is True,
        "dataset manifest records a passed training gate",
        actual={"gate_passed": gate_passed},
    )
    identity = _dataset_identity(dataset_context)
    add_check(
        checks,
        "dataset_version_registered",
        bool(identity),
        "dataset manifest declares a registry identity or version",
        actual={"dataset_identity": identity},
    )
    family_exclusive = _family_exclusive(dataset_context)
    add_check(
        checks,
        "dataset_family_exclusive_splits",
        family_exclusive is True,
        "dataset manifest records family-exclusive splits",
        actual={"family_exclusive": family_exclusive},
    )
    flags_clear = _quality_flags_clear(dataset_context)
    add_check(
        checks,
        "dataset_quality_flags_clear",
        flags_clear is True,
        "dataset manifest has no blocking quality flags",
        actual={"quality_flags_clear": flags_clear},
    )
    fingerprints_verified = _source_fingerprints_verified(dataset_context)
    add_check(
        checks,
        "dataset_source_fingerprints_verified",
        fingerprints_verified is True,
        "dataset manifest records verified source fingerprints",
        actual={"source_fingerprints_verified": fingerprints_verified},
    )
    artifact_integrity = _dataset_artifact_integrity(dataset_context)
    add_check(
        checks,
        "dataset_artifact_fingerprints_verified",
        artifact_integrity["passed"] is True,
        "dataset trainer artifacts match their registered SHA-256 fingerprints",
        actual=artifact_integrity,
    )
    if mode == "trace_sft":
        return
    manifest = dataset_context.get("manifest") if isinstance(dataset_context.get("manifest"), dict) else {}
    gates = manifest.get("gates") if isinstance(manifest.get("gates"), dict) else {}
    required_controls = (
        "governance",
        "contamination",
        "human_reviewed_curation",
        "per_action_credit",
        "verified_branch_replay",
        "human_rejection_preferences",
    )
    control_status = {
        name: isinstance(gates.get(name), dict) and gates[name].get("passed") is True
        for name in required_controls
    }
    add_check(
        checks,
        "self_improving_controls_passed",
        all(control_status.values()),
        "Flight Recorder training requires passed governance, contamination, review, credit, and replay controls",
        actual=control_status,
    )


def _add_data_checks(
    checks: list[dict[str, Any]],
    mode: str,
    paths: dict[str, Path],
    raw: dict[str, list[dict[str, Any]]],
    prepared_counts: dict[str, int],
) -> None:
    required_groups = {
        "trace_sft": (("trace_sft",),),
        "fr_sft": (("fr_sft", "fr_action_sft"),),
        "fr_action_sft": (("fr_action_sft",),),
        "fr_dpo": (("fr_dpo",),),
        "fr_sft_dpo": (("fr_sft", "fr_action_sft"), ("fr_dpo",)),
        "fr_reward_model": (("fr_reward_model",),),
        "fr_step_rewards": (("fr_step_rewards",),),
    }.get(mode, ())
    missing_required = [
        list(group)
        for group in required_groups
        if not any(paths[name].exists() for name in group)
    ]
    add_check(
        checks,
        "required_data_files_exist",
        not missing_required,
        "required data files for selected mode exist",
        actual={"missing": missing_required},
    )
    for key in _required_count_keys(mode):
        count = prepared_counts.get(key, 0)
        add_check(
            checks,
            f"{key}_rows_available",
            count > 0,
            f"{key} rows are available for mode {mode}",
            actual={"count": count, "raw_counts": {name: len(rows) for name, rows in raw.items()}},
        )
    if mode == "trace_sft":
        return
    action_rows = raw.get("fr_action_sft", [])
    if action_rows:
        invalid_actions = [
            str(row.get("episode_id") or row.get("sample_id") or index)
            for index, row in enumerate(action_rows)
            if row.get("human_label") != "accept"
            or row.get("tool_schema_provenance") != "recorded_exact"
            or row.get("quality_gate") != "human_reviewed_native_action_accept"
            or row.get("credit_policy") != "exclude_entire_trajectory_on_any_negative_tool_action"
            or not isinstance(row.get("governance"), dict)
            or not row.get("governance")
        ]
        add_check(
            checks,
            "action_sft_rows_are_governed_reviewed_and_exact",
            not invalid_actions,
            "action-SFT rows must be governed, human accepted, exact-schema trajectories admitted by action credit",
            actual={"invalid_rows": invalid_actions, "row_count": len(action_rows)},
        )
    if mode in {"fr_dpo", "fr_sft_dpo"}:
        dpo_arms = {
            str(row.get("training_arm") or "")
            for row in raw.get("fr_dpo", [])
        }
        required_arms = {
            "flightrecorder_human_reviewed_dpo",
            "flightrecorder_verified_branch_replay_dpo",
        }
        add_check(
            checks,
            "dpo_uses_human_rejection_and_verified_branch_replay",
            required_arms.issubset(dpo_arms),
            "DPO data must include both human-rejection and verified branch-replay preferences",
            actual={"training_arms": sorted(dpo_arms)},
            expected={"required_training_arms": sorted(required_arms)},
        )
    if mode == "fr_reward_model":
        invalid_rewards = [
            index
            for index, row in enumerate(raw.get("fr_reward_model", []))
            if row.get("source_artifact") != "controls/reviewed_preferences.jsonl"
            or row.get("human_label") not in {"accept", "reject"}
        ]
        add_check(
            checks,
            "reward_rows_are_human_reviewed",
            not invalid_rewards,
            "reward-model rows must come from human-reviewed accept/reject preferences",
            actual={"invalid_rows": invalid_rewards},
        )
    if mode == "fr_step_rewards":
        invalid_steps = [
            index
            for index, row in enumerate(raw.get("fr_step_rewards", []))
            if row.get("source_artifact") != "controls/action_credit.jsonl"
            or row.get("source") != "deterministic_tool_result"
        ]
        add_check(
            checks,
            "step_rewards_are_action_credit",
            not invalid_steps,
            "process-reward rows must come from deterministic per-action credit",
            actual={"invalid_rows": invalid_steps},
        )


def _add_hyperparameter_checks(checks: list[dict[str, Any]], args: argparse.Namespace) -> None:
    add_check(
        checks,
        "hub_model_id_present_when_pushing",
        not args.push_to_hub or bool(args.hub_model_id),
        "Hub persistence declares a destination model repository before training",
        actual={"push_to_hub": args.push_to_hub, "hub_model_id": args.hub_model_id},
    )
    add_check(checks, "limit_non_negative", args.limit >= 0, "row limit is non-negative", actual={"limit": args.limit})
    add_check(checks, "batch_size_positive", args.batch_size > 0, "batch size is positive", actual={"batch_size": args.batch_size})
    add_check(
        checks,
        "gradient_accumulation_positive",
        args.gradient_accumulation_steps > 0,
        "gradient accumulation steps are positive",
        actual={"gradient_accumulation_steps": args.gradient_accumulation_steps},
    )
    add_check(checks, "max_length_positive", args.max_length > 0, "max sequence length is positive", actual={"max_length": args.max_length})
    add_check(checks, "lora_rank_positive", args.lora_r > 0, "LoRA rank is positive", actual={"lora_r": args.lora_r})
    add_check(checks, "lora_alpha_positive", args.lora_alpha > 0, "LoRA alpha is positive", actual={"lora_alpha": args.lora_alpha})
    add_check(
        checks,
        "lora_dropout_range",
        0 <= args.lora_dropout < 1,
        "LoRA dropout is in [0, 1)",
        actual={"lora_dropout": args.lora_dropout},
    )
    add_check(checks, "seed_non_negative", args.seed >= 0, "trainer seed is non-negative", actual={"seed": args.seed})
    add_check(checks, "data_seed_non_negative", args.data_seed >= 0, "data seed is non-negative", actual={"data_seed": args.data_seed})
    add_check(checks, "save_steps_positive", args.save_steps > 0, "checkpoint save interval is positive", actual={"save_steps": args.save_steps})
    add_check(
        checks,
        "save_total_limit_positive",
        args.save_total_limit > 0,
        "checkpoint retention limit is positive",
        actual={"save_total_limit": args.save_total_limit},
    )
    resume_declared = bool(args.resume_from_checkpoint)
    add_check(
        checks,
        "resume_contract_complete",
        resume_declared == bool(args.resume_phase),
        "resume checkpoint and phase are declared together",
        actual={"resume_from_checkpoint": str(args.resume_from_checkpoint or ""), "resume_phase": args.resume_phase},
    )
    identity_values = (
        args.model_revision,
        args.tokenizer_revision,
        args.expected_chat_template_sha256,
    )
    if any(identity_values):
        add_check(
            checks,
            "immutable_model_runtime_identity_complete",
            all(identity_values)
            and 40 <= len(args.model_revision) <= 64
            and 40 <= len(args.tokenizer_revision) <= 64
            and len(args.expected_chat_template_sha256) == 64
            and all(char in "0123456789abcdef" for value in identity_values for char in value),
            "model, tokenizer, and chat-template runtime identities are complete immutable digests",
            actual={
                "model_revision": args.model_revision,
                "tokenizer_revision": args.tokenizer_revision,
                "expected_chat_template_sha256": args.expected_chat_template_sha256,
            },
        )


def configure_tracking(args: argparse.Namespace) -> str | list[str]:
    if args.disable_trackio:
        return []
    if args.trackio_project:
        os.environ.setdefault("TRACKIO_PROJECT_NAME", args.trackio_project)
    if args.trackio_space_id:
        os.environ.setdefault("TRACKIO_SPACE_ID", args.trackio_space_id)
    return "trackio"


def run_name(args: argparse.Namespace, phase: str) -> str:
    if args.run_name_prefix:
        return f"{args.run_name_prefix}-{args.mode}-{phase}"
    return f"qwen3-4b-{args.mode}-{phase}"


def run_training(args: argparse.Namespace, plan: dict[str, Any], plan_path: Path) -> dict[str, Any]:
    # Imported lazily so --dry-run works in a stdlib-only environment.
    import torch
    from datasets import Dataset
    from peft import AutoPeftModelForCausalLM, LoraConfig
    from transformers import AutoTokenizer
    from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

    paths = data_paths(args.experiment_dir, load_dataset_context(args.dataset_manifest))
    trace_sft_rows = prepare_sft_rows(load_jsonl(paths["trace_sft"]))
    fr_sft_rows = prepare_sft_rows(load_jsonl(paths["fr_sft"]) + load_jsonl(paths["fr_action_sft"]))
    fr_action_sft_rows = prepare_sft_rows(load_jsonl(paths["fr_action_sft"]))
    fr_dpo_rows = prepare_dpo_rows(load_jsonl(paths["fr_dpo"]))
    if args.limit:
        trace_sft_rows = trace_sft_rows[: args.limit]
        fr_sft_rows = fr_sft_rows[: args.limit]
        fr_action_sft_rows = fr_action_sft_rows[: args.limit]
        fr_dpo_rows = fr_dpo_rows[: args.limit]

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif torch.backends.mps.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32
    model_kwargs = {"dtype": dtype}
    if args.model_revision:
        model_kwargs["revision"] = args.model_revision
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.tokenizer_revision or args.model_revision or None,
    )
    if args.expected_chat_template_sha256:
        actual_template_sha256 = hashlib.sha256(
            str(tokenizer.chat_template or "").encode("utf-8")
        ).hexdigest()
        if actual_template_sha256 != args.expected_chat_template_sha256:
            raise SystemExit("tokenizer chat template fingerprint does not match the reviewed training plan")
    report_to = configure_tracking(args)
    hub_config = {}
    if args.push_to_hub:
        hub_config = {
            "push_to_hub": True,
            "hub_model_id": args.hub_model_id,
            "hub_strategy": "every_save",
        }
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "succeeded",
        "mode": args.mode,
        "model": args.model,
        "output_dir": str(args.output_dir),
        "training_plan": str(plan_path),
        "input_manifests": plan.get("input_manifests", {}),
        "sft_train_result": None,
        "dpo_train_result": None,
        "failure": None,
    }

    def train_sft(rows: list[dict[str, Any]], out: Path) -> Path:
        if not rows:
            raise SystemExit("No SFT rows available for this mode.")
        dataset = Dataset.from_list(rows)
        config = SFTConfig(
            output_dir=str(out),
            learning_rate=args.sft_learning_rate,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.sft_epochs,
            max_steps=args.max_steps,
            max_length=args.max_length,
            assistant_only_loss=not args.all_message_loss,
            gradient_checkpointing=args.gradient_checkpointing,
            logging_steps=1,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            seed=args.seed,
            data_seed=args.data_seed,
            report_to=report_to,
            run_name=run_name(args, "sft"),
            model_init_kwargs=model_kwargs,
            **hub_config,
        )
        trainer = SFTTrainer(
            model=args.model,
            args=config,
            train_dataset=dataset,
            peft_config=peft_config,
            processing_class=tokenizer,
        )
        resume_checkpoint = str(args.resume_from_checkpoint) if args.resume_phase == "sft" else None
        train_output = trainer.train(resume_from_checkpoint=resume_checkpoint)
        trainer.save_model(str(out))
        result["sft_train_result"] = train_output.metrics
        return out

    def train_dpo(rows: list[dict[str, Any]], model_or_path: str | Path, out: Path) -> Path:
        if not rows:
            raise SystemExit("No DPO rows available for this mode.")
        dataset = Dataset.from_list(rows)
        config = DPOConfig(
            output_dir=str(out),
            learning_rate=args.dpo_learning_rate,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.dpo_epochs,
            max_steps=args.max_steps,
            max_length=args.max_length,
            gradient_checkpointing=args.gradient_checkpointing,
            logging_steps=1,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            seed=args.seed,
            data_seed=args.data_seed,
            report_to=report_to,
            run_name=run_name(args, "dpo"),
            model_init_kwargs=model_kwargs if isinstance(model_or_path, str) else None,
            **hub_config,
        )
        if isinstance(model_or_path, Path):
            model = AutoPeftModelForCausalLM.from_pretrained(
                str(model_or_path),
                is_trainable=True,
                dtype=dtype,
            )
            trainer = DPOTrainer(
                model=model,
                args=config,
                train_dataset=dataset,
                processing_class=tokenizer,
            )
        else:
            trainer = DPOTrainer(
                model=model_or_path,
                args=config,
                train_dataset=dataset,
                peft_config=peft_config,
                processing_class=tokenizer,
            )
        resume_checkpoint = str(args.resume_from_checkpoint) if args.resume_phase == "dpo" else None
        train_output = trainer.train(resume_from_checkpoint=resume_checkpoint)
        trainer.save_model(str(out))
        result["dpo_train_result"] = train_output.metrics
        return out

    if args.mode == "trace_sft":
        final_dir = train_sft(trace_sft_rows, args.output_dir / "trace_sft_adapter")
    elif args.mode == "fr_sft":
        final_dir = train_sft(fr_sft_rows, args.output_dir / "fr_sft_adapter")
    elif args.mode == "fr_action_sft":
        final_dir = train_sft(fr_action_sft_rows, args.output_dir / "fr_action_sft_adapter")
    elif args.mode == "fr_dpo":
        model_source: str | Path = args.resume_from_checkpoint if args.resume_phase == "dpo" else args.model
        final_dir = train_dpo(fr_dpo_rows, model_source, args.output_dir / "fr_dpo_adapter")
    elif args.mode == "fr_sft_dpo":
        if args.resume_phase == "dpo":
            sft_dir = args.resume_from_checkpoint
        else:
            sft_dir = train_sft(fr_sft_rows, args.output_dir / "fr_sft_adapter")
        final_dir = train_dpo(fr_dpo_rows, sft_dir, args.output_dir / "fr_sft_dpo_adapter")
    else:
        raise SystemExit(f"Unsupported mode: {args.mode}")

    result["final_adapter_dir"] = str(final_dir)
    result["final_adapter_artifacts"] = adapter_artifact_manifest(final_dir)
    result["resume"] = {
        "resumed": bool(args.resume_from_checkpoint),
        "phase": args.resume_phase,
        "checkpoint": str(args.resume_from_checkpoint or ""),
    }
    if args.push_to_hub:
        if not args.hub_model_id:
            raise SystemExit("--push-to-hub requires --hub-model-id")
        # Re-open as PEFT model where possible so adapter-only pushes are explicit.
        model = AutoPeftModelForCausalLM.from_pretrained(str(final_dir), is_trainable=False)
        commit_info = model.push_to_hub(args.hub_model_id)
        result["pushed_to_hub"] = args.hub_model_id
        result["hub_persistence"] = {
            "repo_id": args.hub_model_id,
            "commit_url": str(getattr(commit_info, "commit_url", "") or ""),
            "revision": str(getattr(commit_info, "oid", "") or ""),
            "immutable_revision_recorded": bool(getattr(commit_info, "oid", "")),
        }
    return result


def trainer_dependency_preflight(args: argparse.Namespace, plan: dict[str, Any], plan_path: Path) -> dict[str, Any]:
    required = list(TRAINER_DEPENDENCIES)
    if not args.disable_trackio:
        required.append("trackio")
    required.extend(args.preflight_extra_dependency or [])
    seen: set[str] = set()
    dependency_checks = []
    for module in required:
        if module in seen:
            continue
        seen.add(module)
        dependency_checks.append(
            {
                "module": module,
                "available": importlib.util.find_spec(module) is not None,
            }
        )
    missing = [check["module"] for check in dependency_checks if not check["available"]]
    status = "preflight_blocked" if missing else "preflight_passed"
    failure = None
    if missing:
        failure = {
            "category": "missing_dependency",
            "exception_type": "TrainerDependencyPreflight",
            "message": "Missing trainer dependencies: " + ", ".join(missing),
            "retryable": True,
            "missing_dependencies": missing,
        }
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "mode": args.mode,
        "model": args.model,
        "output_dir": str(args.output_dir),
        "training_plan": str(plan_path),
        "input_manifests": plan.get("input_manifests", {}),
        "final_adapter_dir": None,
        "sft_train_result": None,
        "dpo_train_result": None,
        "failure": failure,
        "preflight": {
            "plan_passed": plan.get("passed") is True,
            "dependency_checks": dependency_checks,
            "missing_dependencies": missing,
            "hardware": {
                "checked": False,
                "requires_accelerator": False,
                "cpu_smoke_allowed": True,
                "reason": "Dependency preflight does not import torch, allocate devices, download models, or launch training.",
            },
            "model_downloads_started": False,
            "training_started": False,
        },
    }


def write_backend_recipes(recipe_dir: Path, args: argparse.Namespace, plan: dict[str, Any], plan_path: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    model_manifest = _dict_value(plan.get("input_manifests", {}), "model")
    dataset_manifest = _dict_value(plan.get("input_manifests", {}), "dataset")
    add_check(checks, "source_plan_passed", plan.get("passed") is True, "source training plan passed")
    add_check(
        checks,
        "registered_model_manifest_present",
        model_manifest.get("provided") is True and bool(model_manifest.get("model_id")),
        "recipe bundle requires a registered model manifest",
        actual={"provided": model_manifest.get("provided"), "model_id": model_manifest.get("model_id")},
    )
    add_check(
        checks,
        "registered_dataset_manifest_present",
        dataset_manifest.get("provided") is True and bool(dataset_manifest.get("dataset_identity")),
        "recipe bundle requires a registered dataset manifest",
        actual={"provided": dataset_manifest.get("provided"), "dataset_identity": dataset_manifest.get("dataset_identity")},
    )
    add_check(checks, "external_runner_boundary_declared", True, "Flight Recorder writes recipes but does not execute external trainers")
    failed_checks = [check for check in checks if check["passed"] is False]
    passed = not failed_checks

    recipe_dir.mkdir(parents=True, exist_ok=True)
    recipes: list[dict[str, Any]] = []
    if passed:
        for backend, payload in _backend_recipe_payloads(plan, plan_path).items():
            path = recipe_dir / f"{backend}_recipe.json"
            write_json(path, payload)
            recipes.append(
                {
                    "backend": backend,
                    "path": str(path),
                    "format": "json",
                    "mode_supported": payload["mode_supported"],
                    "runner_owns_execution": True,
                    "flight_recorder_executed_command": False,
                    "command_template": payload["command_template"],
                }
            )

    bundle = {
        "schema_version": BACKEND_RECIPES_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": plan.get("mode"),
        "model": plan.get("model"),
        "training_plan": str(plan_path),
        "recipe_dir": str(recipe_dir),
        "output_dir": plan.get("output_dir"),
        "input_manifests": plan.get("input_manifests", {}),
        "passed": passed,
        "readiness": "ready" if passed else "blocked",
        "recommendation": "ready_for_external_recipe_runner" if passed else "block_external_recipe_runner",
        "check_count": len(checks),
        "failed_check_count": len(failed_checks),
        "checks": checks,
        "blocked_reasons": [check["summary"] for check in failed_checks],
        "recipes": recipes,
        "handoff_contract": {
            "flight_recorder_executed_command": False,
            "runner_owns_execution": True,
            "runner_must_require_recommendation": "ready_for_external_recipe_runner",
            "runner_must_revalidate_plan": str(plan_path),
            "allowed_inputs": ["training_plan", "recipes[*].path", "input_manifests"],
            "notes": [
                "Recipe files are inert handoff artifacts for external trainer wrappers.",
                "External runners must validate the plan and registry manifests before launching training.",
            ],
        },
        "extension_points": {
            "axolotl_recipe": "recipes[backend=axolotl]",
            "llama_factory_recipe": "recipes[backend=llama_factory]",
            "unsloth_recipe": "recipes[backend=unsloth]",
            "reward_model_trainer": "recipes[backend=reward_process_rl_extensions].planned_trainers[reward_model_trainer]",
            "process_reward_trainer": "recipes[backend=reward_process_rl_extensions].planned_trainers[process_reward_trainer]",
            "grpo_rl_trainer": "recipes[backend=reward_process_rl_extensions].planned_trainers[grpo_rl_trainer]",
        },
        "notes": [
            "Backend recipe bundles do not import trainer libraries, download models, allocate devices, or launch jobs.",
            "They preserve the registry-backed plan as the source of truth for model, dataset, hyperparameters, and gates.",
        ],
    }
    write_json(recipe_dir / "backend_recipes.json", bundle)
    return bundle


def _backend_recipe_payloads(plan: dict[str, Any], plan_path: Path) -> dict[str, dict[str, Any]]:
    common = _common_recipe_payload(plan, plan_path)
    mode = str(plan.get("mode") or "")
    selected_files = _selected_data_files(plan)
    return {
        "axolotl": {
            **common,
            "backend": "axolotl",
            "mode_supported": mode in EXECUTABLE_MODES,
            "command_template": ["axolotl", "train", "axolotl_recipe.json"],
            "recipe": {
                "base_model": plan.get("model"),
                "adapter": "lora",
                "sequence_len": plan.get("hyperparameters", {}).get("max_length"),
                "micro_batch_size": plan.get("hyperparameters", {}).get("batch_size"),
                "gradient_accumulation_steps": plan.get("hyperparameters", {}).get("gradient_accumulation_steps"),
                "learning_rate": _learning_rate_for_mode(plan),
                "num_epochs": _epochs_for_mode(plan),
                "output_dir": str(Path(str(plan.get("output_dir") or "")) / f"axolotl_{mode}"),
                "datasets": _dataset_entries(selected_files),
                "dpo_datasets": _dataset_entries({"dpo": selected_files["dpo"]}) if "dpo" in selected_files else [],
            },
        },
        "llama_factory": {
            **common,
            "backend": "llama_factory",
            "mode_supported": mode in EXECUTABLE_MODES or mode == "fr_reward_model",
            "command_template": ["llamafactory-cli", "train", "llama_factory_recipe.json"],
            "recipe": {
                "model_name_or_path": plan.get("model"),
                "stage": _llama_factory_stage(mode),
                "finetuning_type": "lora",
                "template": "auto",
                "dataset_files": selected_files,
                "output_dir": str(Path(str(plan.get("output_dir") or "")) / f"llama_factory_{mode}"),
                "per_device_train_batch_size": plan.get("hyperparameters", {}).get("batch_size"),
                "gradient_accumulation_steps": plan.get("hyperparameters", {}).get("gradient_accumulation_steps"),
                "learning_rate": _learning_rate_for_mode(plan),
                "num_train_epochs": _epochs_for_mode(plan),
                "cutoff_len": plan.get("hyperparameters", {}).get("max_length"),
            },
        },
        "unsloth": {
            **common,
            "backend": "unsloth",
            "mode_supported": mode in EXECUTABLE_MODES,
            "command_template": ["python", "unsloth_train.py", "--recipe", "unsloth_recipe.json"],
            "recipe": {
                "model_name": plan.get("model"),
                "trainer": _unsloth_trainer(mode),
                "dataset_files": selected_files,
                "output_dir": str(Path(str(plan.get("output_dir") or "")) / f"unsloth_{mode}"),
                "max_seq_length": plan.get("hyperparameters", {}).get("max_length"),
                "lora": {
                    "r": plan.get("hyperparameters", {}).get("lora_r"),
                    "alpha": plan.get("hyperparameters", {}).get("lora_alpha"),
                    "dropout": plan.get("hyperparameters", {}).get("lora_dropout"),
                    "target_modules": "all-linear",
                },
            },
        },
        "reward_process_rl_extensions": {
            **common,
            "backend": "reward_process_rl_extensions",
            "mode_supported": mode in PLAN_ONLY_MODES,
            "command_template": ["external-runner", "consume", "reward_process_rl_extensions_recipe.json"],
            "planned_trainers": {
                "reward_model_trainer": {
                    "modes": ["fr_reward_model"],
                    "status": "plan_only",
                    "required_data_file": plan.get("data_files", {}).get("fr_reward_model"),
                },
                "process_reward_trainer": {
                    "modes": ["fr_step_rewards"],
                    "status": "plan_only",
                    "required_data_file": plan.get("data_files", {}).get("fr_step_rewards"),
                },
                "grpo_rl_trainer": {
                    "modes": ["future_grpo_rl"],
                    "status": "future_extension",
                    "required_registry_links": ["policy_model", "reward_model", "rollout_dataset", "eval_gate"],
                },
            },
        },
    }


def _common_recipe_payload(plan: dict[str, Any], plan_path: Path) -> dict[str, Any]:
    return {
        "schema_version": "hfr.agentic_lora_backend_recipe_file.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_training_plan": str(plan_path),
        "mode": plan.get("mode"),
        "model": plan.get("model"),
        "input_manifests": plan.get("input_manifests", {}),
        "data_files": _selected_data_files(plan),
        "prepared_counts": plan.get("prepared_counts", {}),
        "full_prepared_counts": plan.get("full_prepared_counts", {}),
        "hyperparameters": plan.get("hyperparameters", {}),
        "execution_boundary": {
            "flight_recorder_executed_command": False,
            "runner_owns_execution": True,
            "requires_plan_revalidation": True,
        },
    }


def _selected_data_files(plan: dict[str, Any]) -> dict[str, str]:
    mode = str(plan.get("mode") or "")
    files = plan.get("data_files") if isinstance(plan.get("data_files"), dict) else {}
    if mode == "trace_sft":
        keys = ("trace_sft",)
    elif mode == "fr_sft":
        keys = ("fr_sft", "fr_action_sft")
    elif mode == "fr_action_sft":
        keys = ("fr_action_sft",)
    elif mode == "fr_dpo":
        keys = ("fr_dpo",)
    elif mode == "fr_sft_dpo":
        keys = ("fr_sft", "fr_action_sft", "fr_dpo")
    elif mode == "fr_reward_model":
        keys = ("fr_reward_model",)
    elif mode == "fr_step_rewards":
        keys = ("fr_step_rewards",)
    else:
        keys = ()
    return {key: str(files.get(key) or "") for key in keys if files.get(key)}


def _dataset_entries(files: dict[str, str]) -> list[dict[str, str]]:
    return [{"name": name, "path": path} for name, path in files.items()]


def _learning_rate_for_mode(plan: dict[str, Any]) -> Any:
    params = plan.get("hyperparameters", {}) if isinstance(plan.get("hyperparameters"), dict) else {}
    return params.get("dpo_learning_rate") if str(plan.get("mode") or "") in {"fr_dpo", "fr_sft_dpo"} else params.get("sft_learning_rate")


def _epochs_for_mode(plan: dict[str, Any]) -> Any:
    params = plan.get("hyperparameters", {}) if isinstance(plan.get("hyperparameters"), dict) else {}
    return params.get("dpo_epochs") if str(plan.get("mode") or "") in {"fr_dpo", "fr_sft_dpo"} else params.get("sft_epochs")


def _llama_factory_stage(mode: str) -> str:
    if mode in {"fr_dpo", "fr_sft_dpo"}:
        return "dpo"
    if mode == "fr_reward_model":
        return "rm"
    return "sft"


def _unsloth_trainer(mode: str) -> str:
    if mode in {"fr_dpo", "fr_sft_dpo"}:
        return "dpo"
    return "sft"


def classify_failure(exc: BaseException) -> dict[str, Any]:
    message = str(exc)
    lower = message.lower()
    if isinstance(exc, json.JSONDecodeError):
        category = "invalid_training_data"
    elif isinstance(exc, (OSError, UnicodeError)):
        category = "training_input_io_error"
    elif isinstance(exc, ModuleNotFoundError):
        category = "missing_dependency"
    elif "out of memory" in lower or "oom" in lower:
        category = "resource_exhausted"
    elif "no sft rows" in lower or "no dpo rows" in lower:
        category = "missing_training_rows"
    elif isinstance(exc, SystemExit):
        category = "trainer_exit"
    else:
        category = "trainer_runtime_error"
    return {
        "category": category,
        "exception_type": type(exc).__name__,
        "message": message,
        "retryable": category in {"missing_dependency", "resource_exhausted"},
    }


def failure_result(args: argparse.Namespace, plan: dict[str, Any], plan_path: Path, exc: BaseException) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
        "mode": args.mode,
        "model": args.model,
        "output_dir": str(args.output_dir),
        "training_plan": str(plan_path),
        "input_manifests": plan.get("input_manifests", {}),
        "final_adapter_dir": None,
        "sft_train_result": None,
        "dpo_train_result": None,
        "failure": classify_failure(exc),
    }


def blocked_result(args: argparse.Namespace, plan: dict[str, Any], plan_path: Path) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "blocked",
        "mode": args.mode,
        "model": args.model,
        "output_dir": str(args.output_dir),
        "training_plan": str(plan_path),
        "input_manifests": plan.get("input_manifests", {}),
        "final_adapter_dir": None,
        "sft_train_result": None,
        "dpo_train_result": None,
        "failure": {
            "category": "plan_validation_failed",
            "exception_type": "TrainingPlanBlocked",
            "message": "; ".join(plan.get("blocked_reasons") or []),
            "retryable": False,
            "blocked_reasons": plan.get("blocked_reasons") or [],
        },
    }


def write_result_registry_event(
    *,
    registry_path: Path,
    result_path: Path,
    result: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    event = {
        "schema_version": "hfr.agentic_lora_training_registry_event.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": result.get("mode"),
        "model": result.get("model"),
        "status": result.get("status"),
        "result_path": str(result_path),
        "training_plan": result.get("training_plan"),
        "final_adapter_dir": result.get("final_adapter_dir"),
        "final_adapter_artifacts": result.get("final_adapter_artifacts", {}),
        "hub_persistence": result.get("hub_persistence", {}),
        "input_manifests": result.get("input_manifests", {}),
        "failure": result.get("failure"),
        "plan_passed": plan.get("passed") is True,
        "plan_failed_check_count": plan.get("failed_check_count"),
        "registry_note": "Local JSONL registry event for training result/failure handoff; not a promotion decision.",
    }
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    return event


def persist_result(
    *,
    args: argparse.Namespace,
    plan: dict[str, Any],
    plan_path: Path,
    result_path: Path,
    registry_path: Path,
    result: dict[str, Any],
) -> None:
    write_json(result_path, result)
    write_result_registry_event(registry_path=registry_path, result_path=result_path, result=result, plan=plan)
    if args.write_model_registry_link_plan is not None:
        write_model_registry_link_plan(args.write_model_registry_link_plan, args, plan, result, result_path, plan_path)


def write_model_registry_link_plan(
    out_path: Path,
    args: argparse.Namespace,
    plan: dict[str, Any],
    result: dict[str, Any],
    result_path: Path,
    plan_path: Path,
) -> dict[str, Any]:
    registry_path = args.model_registry or "<model_registry.json>"
    entry = args.model_registry_entry
    result_artifact_id = args.model_registry_artifact_id or _artifact_id(
        "training",
        str(result.get("model") or plan.get("model") or "model"),
        str(result.get("mode") or plan.get("mode") or "mode"),
        str(result.get("status") or "result"),
    )
    commands = [
        {
            "id": "link_training_result",
            "link_type": "training-run",
            "artifact_id": result_artifact_id,
            "path": str(result_path),
            "command_argv": _model_registry_link_command(
                registry_path=registry_path,
                entry=entry,
                collection="training_runs",
                kind="training-run",
                status=str(result.get("status") or "recorded"),
                artifact_id=result_artifact_id,
                path=str(result_path),
                metadata={
                    "mode": str(result.get("mode") or ""),
                    "status": str(result.get("status") or ""),
                    "training_plan": str(plan_path),
                },
            ),
        }
    ]
    final_adapter_dir = result.get("final_adapter_dir")
    if isinstance(final_adapter_dir, str) and final_adapter_dir:
        adapter_artifact_id = _artifact_id("adapter", str(result.get("model") or "model"), str(result.get("mode") or "mode"))
        adapter_config_path = str(Path(final_adapter_dir) / "adapter_config.json")
        commands.append(
            {
                "id": "link_final_adapter",
                "link_type": "adapter",
                "artifact_id": adapter_artifact_id,
                "path": adapter_config_path,
                "command_argv": _model_registry_link_command(
                    registry_path=registry_path,
                    entry=entry,
                    collection="adapters",
                    kind="peft-adapter",
                    status="trained",
                    artifact_id=adapter_artifact_id,
                    path=adapter_config_path,
                    metadata={
                        "mode": str(result.get("mode") or ""),
                        "training_result": str(result_path),
                        "training_plan": str(plan_path),
                    },
                ),
            }
        )
    checks: list[dict[str, Any]] = []
    add_check(checks, "result_archive_present", bool(str(result_path)), "training result archive path is available")
    add_check(checks, "model_registry_entry_declared", bool(entry), "model registry entry or alias is declared", actual={"entry": entry})
    add_check(checks, "no_alias_movement", True, "link plan does not move candidate/champion/rollback aliases")
    failed_checks = [check for check in checks if check["passed"] is False]
    link_plan = {
        "schema_version": MODEL_REGISTRY_LINK_PLAN_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": result.get("model") or plan.get("model"),
        "mode": result.get("mode") or plan.get("mode"),
        "status": result.get("status"),
        "training_plan": str(plan_path),
        "training_result": str(result_path),
        "model_registry": registry_path,
        "model_registry_entry": entry,
        "passed": not failed_checks,
        "readiness": "ready" if not failed_checks else "blocked",
        "recommendation": "ready_to_link_training_result" if not failed_checks else "block_registry_link",
        "check_count": len(checks),
        "failed_check_count": len(failed_checks),
        "checks": checks,
        "blocked_reasons": [check["summary"] for check in failed_checks],
        "commands": commands if not failed_checks else [],
        "input_manifests": result.get("input_manifests") or plan.get("input_manifests", {}),
        "handoff_contract": {
            "flight_recorder_mutated_registry": False,
            "moves_aliases": False,
            "runner_owns_execution": True,
            "runner_must_validate_result": str(result_path),
            "runner_must_validate_model_registry": registry_path,
        },
        "notes": [
            "This artifact plans model-registry lifecycle links only; it does not mutate the registry.",
            "Promotion aliases must be moved only through governance promotion decisions and alias receipts.",
        ],
    }
    write_json(out_path, link_plan)
    return link_plan


def _model_registry_link_command(
    *,
    registry_path: str,
    entry: str,
    collection: str,
    kind: str,
    status: str,
    artifact_id: str,
    path: str,
    metadata: dict[str, str],
) -> list[str]:
    argv = [
        "python3",
        "-m",
        "flightrecorder",
        "model-registry",
        "link",
        "--registry",
        registry_path,
        "--entry",
        entry,
        "--collection",
        collection,
        "--artifact-id",
        artifact_id,
        "--kind",
        kind,
        "--status",
        status,
        "--path",
        path,
    ]
    for key in sorted(metadata):
        value = metadata[key]
        if value:
            argv.extend(["--metadata", f"{key}={value}"])
    return argv


def _artifact_id(*parts: str) -> str:
    cleaned = []
    for part in parts:
        text = "".join(char.lower() if char.isalnum() else "_" for char in part)
        text = "_".join(segment for segment in text.split("_") if segment)
        if text:
            cleaned.append(text)
    return "_".join(cleaned) or "artifact"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=SUPPORTED_MODES)
    parser.add_argument(
        "--write-smoke-fixture",
        type=Path,
        help="Write a tiny registered model/dataset fixture for dry-run smoke checks, then exit",
    )
    parser.add_argument("--experiment-dir", type=Path, default=Path("experiments/qwen3_4b_flightrecorder"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default="", help="Immutable base-model Hub revision")
    parser.add_argument("--tokenizer-revision", default="", help="Immutable tokenizer Hub revision")
    parser.add_argument(
        "--expected-chat-template-sha256",
        default="",
        help="Expected SHA-256 of the tokenizer chat template loaded for training",
    )
    parser.add_argument("--model-manifest", type=Path, help="Registered model-candidate manifest with license and compatibility metadata")
    parser.add_argument("--dataset-manifest", type=Path, help="Registered dataset/version manifest or Flight Recorder export manifest wrapper")
    parser.add_argument(
        "--write-backend-recipes",
        type=Path,
        help="Write schema-checkable external trainer recipe artifacts from the validated plan, then exit before launch",
    )
    parser.add_argument(
        "--result-registry",
        type=Path,
        help="JSONL registry event log for completed, failed, or blocked trainer results",
    )
    parser.add_argument(
        "--write-model-registry-link-plan",
        type=Path,
        help="Write a side-effect-free plan of model-registry lifecycle link commands for the emitted training result",
    )
    parser.add_argument("--model-registry", default="", help="Model registry path to include in --write-model-registry-link-plan command templates")
    parser.add_argument("--model-registry-entry", default="candidate", help="Model registry entry id or alias for generated lifecycle link commands")
    parser.add_argument("--model-registry-artifact-id", default="", help="Override the training-result artifact id used in generated lifecycle link commands")
    parser.add_argument(
        "--require-registered-inputs",
        action="store_true",
        help="Require model and dataset manifests even for dry-run planning",
    )
    parser.add_argument(
        "--unsafe-allow-unregistered-launch",
        action="store_true",
        help="Legacy escape hatch for non-dry-run launches without registry manifests",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/qwen3_4b_flightrecorder/adapters"))
    parser.add_argument("--hub-model-id", default="")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate the launch plan, check trainer dependencies, write a result archive, then exit before model downloads or training",
    )
    parser.add_argument(
        "--preflight-extra-dependency",
        action="append",
        default=[],
        help="Additional Python module name to require during --preflight-only checks; repeatable for tests or wrapper-specific trainers",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit rows for smoke tests")
    parser.add_argument("--sft-epochs", type=float, default=3.0)
    parser.add_argument("--dpo-epochs", type=float, default=3.0)
    parser.add_argument("--sft-learning-rate", type=float, default=1e-4)
    parser.add_argument("--dpo-learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-steps", type=int, default=-1, help="Override epochs with a fixed number of optimizer steps")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--resume-phase", choices=["sft", "dpo"], default="")
    parser.add_argument(
        "--all-message-loss",
        action="store_true",
        help=(
            "Train on every token in each rendered conversation instead of requiring the tokenizer "
            "chat template to expose assistant-token generation masks"
        ),
    )
    parser.add_argument("--disable-trackio", action="store_true", help="Disable Trackio reporting")
    parser.add_argument("--trackio-project", default=DEFAULT_TRACKIO_PROJECT)
    parser.add_argument("--trackio-space-id", default="")
    parser.add_argument("--run-name-prefix", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.write_smoke_fixture is not None:
        fixture = write_smoke_fixture(args.write_smoke_fixture)
        print(json.dumps(fixture, indent=2, sort_keys=True))
        return 0
    if args.write_backend_recipes is not None and args.preflight_only:
        raise SystemExit("--write-backend-recipes cannot be combined with --preflight-only")
    if args.dry_run and args.preflight_only:
        raise SystemExit("--preflight-only cannot be combined with --dry-run")
    if not args.mode:
        raise SystemExit("--mode is required unless --write-smoke-fixture is provided")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_dir / f"{args.mode}_plan.json"
    result_path = args.output_dir / f"{args.mode}_result.json"
    registry_path = args.result_registry or (args.output_dir / "training_registry_events.jsonl")
    try:
        plan = build_plan(args)
    except (json.JSONDecodeError, OSError, UnicodeError) as exc:
        plan = {
            "schema_version": PLAN_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "model": args.model,
            "experiment_dir": str(args.experiment_dir),
            "output_dir": str(args.output_dir),
            "passed": False,
            "readiness": "blocked",
            "recommendation": "block_launch",
            "failed_check_count": 1,
            "blocked_reasons": ["training input data could not be parsed"],
            "input_manifests": {},
            "failure": classify_failure(exc),
        }
        write_json(plan_path, plan)
        result = failure_result(args, plan, plan_path, exc)
        persist_result(
            args=args,
            plan=plan,
            plan_path=plan_path,
            result_path=result_path,
            registry_path=registry_path,
            result=result,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    write_json(plan_path, plan)
    if args.write_backend_recipes is not None:
        bundle = write_backend_recipes(args.write_backend_recipes, args, plan, plan_path)
        print(json.dumps(bundle, indent=2, sort_keys=True))
        return 0 if bundle["passed"] else 1
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0 if plan["passed"] else 1
    if not plan["passed"]:
        result = blocked_result(args, plan, plan_path)
        persist_result(args=args, plan=plan, plan_path=plan_path, result_path=result_path, registry_path=registry_path, result=result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    if args.preflight_only:
        result = trainer_dependency_preflight(args, plan, plan_path)
        persist_result(args=args, plan=plan, plan_path=plan_path, result_path=result_path, registry_path=registry_path, result=result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "preflight_passed" else 1
    try:
        result = run_training(args, plan, plan_path)
    except SystemExit as exc:
        result = failure_result(args, plan, plan_path, exc)
        persist_result(args=args, plan=plan, plan_path=plan_path, result_path=result_path, registry_path=registry_path, result=result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return exc.code if isinstance(exc.code, int) and exc.code != 0 else 1
    except Exception as exc:  # noqa: BLE001 - trainer result archives should classify failures.
        result = failure_result(args, plan, plan_path, exc)
        persist_result(args=args, plan=plan, plan_path=plan_path, result_path=result_path, registry_path=registry_path, result=result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    persist_result(args=args, plan=plan, plan_path=plan_path, result_path=result_path, registry_path=registry_path, result=result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
