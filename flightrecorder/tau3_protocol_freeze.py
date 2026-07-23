"""Fail-closed Tau-3 local training protocol freeze."""

from __future__ import annotations

import hashlib
import json
import re
import stat
import subprocess
from pathlib import Path
from typing import Any

from .atomic_json import atomic_write_json_cas
from .path_safety import path_has_symlink_component
from .schema_registry import check_schema_contract
from .tau3_capture import canonical_sha256, validate_tau3_capture
from .tau3_model_identity import validate_tau3_model_identity


DOMAINS = ("airline", "retail", "telecom")
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{16,}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)\b(api[_-]?key|access[_-]?token|secret)\s*[:=]\s*[A-Za-z0-9_./+=-]{12,}"),
)

SOURCE_SCHEMA_VERSION = "hfr.tau3_protocol_freeze_source.v1"
PROTOCOL_FREEZE_SCHEMA_VERSION = "hfr.tau3_protocol_freeze.v1"
PROTOCOL_CONFIG_SCHEMA_VERSION = "hfr.tau3_protocol_config.v1"
PARTITION_SCHEMA_VERSION = "hfr.tau3_source_partition.v1"
SPLIT_SCHEMA_VERSION = "hfr.tau3_source_split.v1"
TRAINING_SOURCE_SCHEMA_VERSION = "hfr.tau3_training_source.v1"
SEALED_SOURCE_SCHEMA_VERSION = "hfr.tau3_sealed_source_manifest.v1"
REQUIRED_BEHAVIORS = (
    "success",
    "correction",
    "clarification_refusal",
    "recovery",
    "policy_failure",
    "harmful_mutation",
    "hallucinated_tool",
    "premature_completion",
)
MODEL_SELECTION_RULE = (
    "official_upstream_public_ungated_immutable_open_weights_dense_7_to_9b_total_apache_2_"
    "pinned_mlx_conversion_local_load_documented_tool_or_function_use_identical_common_harness_no_per_model_prompt_tuning"
)
MODEL_SELECTED_AT = "2026-07-22T00:00:00+00:00"

MODEL_SPECS = {
    "base": {
        "name": "mlx-community/Qwen3.5-9B-4bit",
        "revision": "8b2b98c00a6b4d291155e4890773ca8f769aee53",
        "upstream_name": "Qwen/Qwen3.5-9B",
        "upstream_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "parameters_billion": 9.0,
        "architecture": "dense transformer",
        "license": "Apache-2.0",
        "quantization": "mlx-4bit",
        "tokenizer": "Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "chat_template": "qwen3.5-tool-use-chat-template",
        "model_card_url": "https://huggingface.co/mlx-community/Qwen3.5-9B-4bit",
        "evidence_urls": [
            "https://huggingface.co/mlx-community/Qwen3.5-9B-4bit",
            "https://huggingface.co/Qwen/Qwen3.5-9B",
            "https://qwen.readthedocs.io/",
        ],
    },
    "comparator-1": {
        "name": "mlx-community/Qwen3-8B-4bit",
        "revision": "545dc4251c05440727734bcd94334791f6ab0192",
        "upstream_name": "Qwen/Qwen3-8B",
        "upstream_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "parameters_billion": 8.2,
        "architecture": "dense transformer",
        "license": "Apache-2.0",
        "quantization": "mlx-4bit",
        "tokenizer": "Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218",
        "chat_template": "qwen3-tool-use-chat-template",
        "model_card_url": "https://huggingface.co/mlx-community/Qwen3-8B-4bit",
        "evidence_urls": [
            "https://huggingface.co/mlx-community/Qwen3-8B-4bit",
            "https://huggingface.co/Qwen/Qwen3-8B",
            "https://qwen.readthedocs.io/",
        ],
    },
    "comparator-2": {
        "name": "mlx-community/granite-3.3-8b-instruct-4bit",
        "revision": "0751a30ba2420ecd5c2f142707dd4fcfacc4486e",
        "upstream_name": "ibm-granite/granite-3.3-8b-instruct",
        "upstream_revision": "51dd4bc2ade4059a6bd87649d68aa11e4fb2529b",
        "parameters_billion": 8.0,
        "architecture": "dense transformer",
        "license": "Apache-2.0",
        "quantization": "mlx-4bit",
        "tokenizer": "ibm-granite/granite-3.3-8b-instruct@51dd4bc2ade4059a6bd87649d68aa11e4fb2529b",
        "chat_template": "granite-3.3-function-calling-chat-template",
        "model_card_url": "https://huggingface.co/mlx-community/granite-3.3-8b-instruct-4bit",
        "evidence_urls": [
            "https://huggingface.co/mlx-community/granite-3.3-8b-instruct-4bit",
            "https://huggingface.co/ibm-granite/granite-3.3-8b-instruct",
            "https://www.ibm.com/granite/docs/models/granite/",
        ],
    },
    "teacher": {
        "name": "mlx-community/Qwen3.6-35B-A3B-4bit",
        "revision": "38740b847e4cb78f352aba30aa41c76e08e6eb46",
        "upstream_name": "Qwen/Qwen3.6-35B-A3B",
        "upstream_revision": "7c787ca72eef6f25ba9a43a73219a461bf0b304d",
        "parameters_billion": 35.0,
        "architecture": "35B mixture-of-experts transformer",
        "license": "Apache-2.0",
        "quantization": "mlx-4bit",
        "tokenizer": "Qwen/Qwen3.6-35B-A3B@38740b847e4cb78f352aba30aa41c76e08e6eb46",
        "chat_template": "qwen3.6-tool-use-chat-template",
        "model_card_url": "https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit",
        "evidence_urls": [
            "https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit",
            "https://huggingface.co/Qwen/Qwen3.6-35B-A3B",
            "https://qwen.readthedocs.io/",
        ],
        "role": "teacher_generation_and_review_only",
    },
}


class Tau3ProtocolFreezeError(ValueError):
    """Raised when a local Tau-3 protocol cannot be frozen safely."""


def freeze_tau3_training_protocol(
    *,
    tau_repo: str | Path,
    tau_revision: str,
    source_manifest: str | Path,
    train_split: str | Path,
    development_split: str | Path,
    sealed_split: str | Path,
    train_tasks: str | Path,
    development_tasks: str | Path,
    base_identity: str | Path,
    base_model_path: str | Path,
    comparator1_identity: str | Path,
    comparator1_model_path: str | Path,
    comparator2_identity: str | Path,
    comparator2_model_path: str | Path,
    teacher_identity: str | Path,
    teacher_model_path: str | Path,
    captures: str | Path,
    out: str | Path,
    created_at: str = "2026-07-22T00:00:00+00:00",
    hardware_class: str = "local Apple M4 Max with 36 GB unified memory",
    memory_gib: int = 36,
) -> dict[str, Any]:
    """Freeze a production protocol config from local evidence only."""

    target = Path(out)
    _require_new_output_file(target)
    tau_repo_path = Path(tau_repo)
    _require_clean_tau_checkout(tau_repo_path, tau_revision)

    sources = _load_sources(
        source_manifest=Path(source_manifest),
        train_split=Path(train_split),
        development_split=Path(development_split),
        sealed_split=Path(sealed_split),
        train_tasks=Path(train_tasks),
        development_tasks=Path(development_tasks),
    )
    if sources["source_revision"] != tau_revision:
        raise Tau3ProtocolFreezeError("source manifest source_revision does not match pinned Tau revision")
    capture_rows = _load_captures(Path(captures))
    capture_source_bindings = _bind_captures_to_sources(capture_rows, sources["source_index"])
    _require_capture_coverage(capture_rows)
    models = _freeze_models(
        base_identity=Path(base_identity),
        base_model_path=Path(base_model_path),
        comparator1_identity=Path(comparator1_identity),
        comparator1_model_path=Path(comparator1_model_path),
        comparator2_identity=Path(comparator2_identity),
        comparator2_model_path=Path(comparator2_model_path),
        teacher_identity=Path(teacher_identity),
        teacher_model_path=Path(teacher_model_path),
    )
    contamination = _contamination_attestation(sources, capture_rows, capture_source_bindings)
    redaction = _redaction_attestation(sources, capture_rows)
    if contamination["passed"] is not True:
        raise Tau3ProtocolFreezeError("contamination attestation failed: " + json.dumps(contamination, sort_keys=True))
    if redaction["passed"] is not True:
        raise Tau3ProtocolFreezeError("redaction attestation failed: " + json.dumps(redaction, sort_keys=True))

    config = _build_protocol_config(
        tau_repo=tau_repo_path,
        tau_revision=tau_revision,
        sources=sources,
        models=models,
        contamination=contamination,
        redaction=redaction,
        captures=capture_rows,
        captures_path=Path(captures),
        created_at=created_at,
        hardware_class=hardware_class,
        memory_gib=memory_gib,
    )
    _assert_no_placeholders(config)
    schema_check = check_schema_contract(config, name_or_id="tau3_protocol_config")
    if schema_check["passed"] is not True:
        raise Tau3ProtocolFreezeError(
            "generated protocol violates the public schema: " + json.dumps(schema_check["errors"], sort_keys=True)
        )
    atomic_write_json_cas(target, config, expected_sha256=None, new_file_mode=0o600)
    return {
        "schema_version": PROTOCOL_FREEZE_SCHEMA_VERSION,
        "out": str(target),
        "protocol_sha256": _sha256_file(target),
        "tau_revision": tau_revision,
        "train_task_count": sources["train_split"]["task_count"],
        "development_task_count": sources["development_split"]["task_count"],
        "sealed_task_count": sources["sealed_split"]["task_count"],
        "capture_count": len(capture_rows),
        "models": [
            MODEL_SPECS["base"]["name"],
            MODEL_SPECS["comparator-1"]["name"],
            MODEL_SPECS["comparator-2"]["name"],
            MODEL_SPECS["teacher"]["name"],
        ],
    }


def _load_sources(
    *,
    source_manifest: Path,
    train_split: Path,
    development_split: Path,
    sealed_split: Path,
    train_tasks: Path,
    development_tasks: Path,
) -> dict[str, Any]:
    for path in (source_manifest, train_split, development_split, sealed_split, train_tasks, development_tasks):
        _reject_symlinked_file(path)
    manifest = _read_json(source_manifest)
    train = _read_json(train_split)
    development = _read_json(development_split)
    sealed = _read_json(sealed_split)
    train_payloads = _read_jsonl(train_tasks)
    development_payloads = _read_jsonl(development_tasks)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != PARTITION_SCHEMA_VERSION:
        raise Tau3ProtocolFreezeError(f"source manifest schema_version must be {PARTITION_SCHEMA_VERSION}")
    source_revision = str(manifest.get("source_revision") or "")
    if not HEX40_RE.fullmatch(source_revision):
        raise Tau3ProtocolFreezeError("source manifest source_revision must be an exact git commit")
    if not isinstance(train, dict) or train.get("schema_version") != SPLIT_SCHEMA_VERSION:
        raise Tau3ProtocolFreezeError(f"train split schema_version must be {SPLIT_SCHEMA_VERSION}")
    if not isinstance(development, dict) or development.get("schema_version") != SPLIT_SCHEMA_VERSION:
        raise Tau3ProtocolFreezeError(f"development split schema_version must be {SPLIT_SCHEMA_VERSION}")
    if not isinstance(sealed, dict) or sealed.get("schema_version") != SEALED_SOURCE_SCHEMA_VERSION:
        raise Tau3ProtocolFreezeError(f"sealed manifest schema_version must be {SEALED_SOURCE_SCHEMA_VERSION}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise Tau3ProtocolFreezeError("source manifest must include artifact records")
    for label, path in (
        ("train.json", train_split),
        ("development.json", development_split),
        ("sealed.json", sealed_split),
        ("training_source/train_tasks.jsonl", train_tasks),
        ("training_source/development_tasks.jsonl", development_tasks),
    ):
        record = artifacts.get(label)
        if not isinstance(record, dict):
            raise Tau3ProtocolFreezeError(f"source manifest missing artifact record for {label}")
        _assert_artifact_record(path, record, label)
    proofs = manifest.get("proofs") if isinstance(manifest.get("proofs"), dict) else {}
    if proofs.get("train_development_family_disjoint") is not True:
        raise Tau3ProtocolFreezeError("source manifest does not prove train/development family disjointness")
    if proofs.get("sealed_payload_non_materialization") is not True:
        raise Tau3ProtocolFreezeError("source manifest does not prove sealed payload non-materialization")
    if proofs.get("official_test_sealed") is not True:
        raise Tau3ProtocolFreezeError("source manifest does not prove official test is sealed")
    if proofs.get("sealed_payload_files") != []:
        raise Tau3ProtocolFreezeError("source manifest lists sealed payload files")
    for label, split in (("train", train), ("development", development)):
        if split.get("source_revision") != source_revision:
            raise Tau3ProtocolFreezeError(f"{label} split source_revision does not match source manifest")
        if split.get("split") != label:
            raise Tau3ProtocolFreezeError(f"{label} split name does not match source file")
    train_families = set(_required_string_list(train, "family_ids", "train split"))
    dev_families = set(_required_string_list(development, "family_ids", "development split"))
    if train_families & dev_families:
        raise Tau3ProtocolFreezeError("train/development families overlap")
    train_entries = _source_entries_from_split(train)
    development_entries = _source_entries_from_split(development)
    train_envelopes = _validate_training_source_envelopes(train_payloads, train_entries, source_revision, "train")
    development_envelopes = _validate_training_source_envelopes(
        development_payloads,
        development_entries,
        source_revision,
        "development",
    )
    _assert_hashes_only_sealed(sealed, source_revision)
    return {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "source_revision": source_revision,
        "source_manifest_path": str(source_manifest),
        "source_manifest": manifest,
        "train_split_path": str(train_split),
        "development_split_path": str(development_split),
        "sealed_split_path": str(sealed_split),
        "train_tasks_path": str(train_tasks),
        "development_tasks_path": str(development_tasks),
        "train_split": train,
        "development_split": development,
        "sealed_split": sealed,
        "train_payloads": train_envelopes,
        "development_payloads": development_envelopes,
        "source_index": _build_source_index([*train_envelopes, *development_envelopes]),
        "hashes": {
            "train": _sha256_file(train_split),
            "development": _sha256_file(development_split),
            "sealed": _sha256_file(sealed_split),
            "train_tasks": _sha256_file(train_tasks),
            "development_tasks": _sha256_file(development_tasks),
        },
    }


def _freeze_models(**paths: Path) -> dict[str, Any]:
    roles = (
        ("base", paths["base_identity"], paths["base_model_path"]),
        ("comparator-1", paths["comparator1_identity"], paths["comparator1_model_path"]),
        ("comparator-2", paths["comparator2_identity"], paths["comparator2_model_path"]),
        ("teacher", paths["teacher_identity"], paths["teacher_model_path"]),
    )
    frozen: dict[str, Any] = {}
    for role, identity_path, model_path in roles:
        _reject_symlinked_file(identity_path)
        if path_has_symlink_component(model_path, include_leaf=True):
            raise Tau3ProtocolFreezeError(f"{role} model path contains a symlink component")
        spec = MODEL_SPECS[role]
        identity = _read_json(identity_path)
        if not isinstance(identity, dict):
            raise Tau3ProtocolFreezeError(f"{role} model identity must be an object")
        errors = validate_tau3_model_identity(
            identity,
            model_path,
            expected_model_id=spec["name"],
            expected_revision=spec["revision"],
        )
        if errors:
            raise Tau3ProtocolFreezeError(f"{role} model identity does not replay: " + "; ".join(errors))
        frozen[role] = {
            **spec,
            "local_path": str(model_path),
            "local_identity_path": str(identity_path),
            "local_identity_sha256": _sha256_file(identity_path),
            "local_tree_sha256": identity.get("tree_sha256"),
            "local_file_count": identity.get("file_count"),
            "upstream": {
                "name": spec["upstream_name"],
                "revision": spec["upstream_revision"],
            },
            "pre_run_eligibility": _model_eligibility(role),
        }
    return frozen


def _model_eligibility(role: str) -> dict[str, Any]:
    if role == "teacher":
        return {
            "role": "teacher_generation_and_review_only",
            "comparator_eligible": False,
            "rule_applied": "teacher_identity_pin_only",
            "excluded_from_comparator_rule": True,
            "exclusion_reason": (
                "The teacher is a 35B mixture-of-experts model used only for generation and review. "
                "It is not a dense 7-9B comparator candidate and is never included in comparator superiority claims."
            ),
            "conversion_provider": "mlx-community",
            "conversion_identity_content_addressed": True,
            "public_ungated": True,
            "immutable_open_weights": True,
            "license": "Apache-2.0",
            "mlx_local_load_compatible": True,
            "identical_common_harness": False,
            "per_model_prompt_tuning": False,
        }
    return {
        "rule": MODEL_SELECTION_RULE,
        "upstream_official_or_developer_owned": True,
        "conversion_provider": "mlx-community",
        "conversion_identity_content_addressed": True,
        "public_ungated": True,
        "immutable_open_weights": True,
        "dense_total_parameters_7_to_9b": True,
        "license": "Apache-2.0",
        "mlx_local_load_compatible": True,
        "documented_tool_or_function_use": True,
        "identical_common_harness": True,
        "per_model_prompt_tuning": False,
    }


def _load_captures(path: Path) -> list[dict[str, Any]]:
    _reject_symlinked_file(path)
    rows = _read_jsonl(path)
    if not rows:
        raise Tau3ProtocolFreezeError("captures JSONL must contain at least one training-side capture")
    duplicate_ids: set[str] = set()
    seen: set[str] = set()
    errors: dict[str, list[str]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise Tau3ProtocolFreezeError(f"capture row {index} must be an object")
        capture_errors = validate_tau3_capture(row)
        if capture_errors:
            errors[str(row.get("trajectory_id") or index)] = capture_errors
        trajectory_id = str(row.get("trajectory_id") or "")
        if trajectory_id in seen:
            duplicate_ids.add(trajectory_id)
        seen.add(trajectory_id)
    if duplicate_ids:
        raise Tau3ProtocolFreezeError("duplicate capture trajectory IDs: " + ", ".join(sorted(duplicate_ids)))
    if errors:
        raise Tau3ProtocolFreezeError("invalid Tau captures: " + json.dumps(errors, sort_keys=True))
    return rows


def _bind_captures_to_sources(
    captures: list[dict[str, Any]],
    source_index: dict[tuple[str, str, str, str, str, str], dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    bindings: dict[str, dict[str, Any]] = {}
    for capture in captures:
        for key in ("source_task_sha256", "source_prompt_sha256"):
            if not isinstance(capture.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", capture[key]):
                raise Tau3ProtocolFreezeError(f"capture {capture['trajectory_id']} missing valid {key}")
        key = (
            str(capture["task_id"]),
            str(capture["domain"]),
            str(capture["split"]),
            str(capture["task_family"]),
            str(capture["source_task_sha256"]),
            str(capture["source_prompt_sha256"]),
        )
        envelope = source_index.get(key)
        if envelope is None:
            raise Tau3ProtocolFreezeError(
                f"capture {capture['trajectory_id']} does not match exactly one permitted source envelope"
            )
        bindings[str(capture["trajectory_id"])] = envelope
    return bindings


def _require_capture_coverage(captures: list[dict[str, Any]]) -> None:
    domains = {str(capture.get("domain")) for capture in captures}
    behaviors = {str(capture.get("behavior")) for capture in captures}
    missing_domains = sorted(set(DOMAINS) - domains)
    missing_behaviors = sorted(set(REQUIRED_BEHAVIORS) - behaviors)
    if missing_domains:
        raise Tau3ProtocolFreezeError("captures must cover all study domains; missing: " + ", ".join(missing_domains))
    if missing_behaviors:
        raise Tau3ProtocolFreezeError("captures must cover all required behaviors; missing: " + ", ".join(missing_behaviors))


def _contamination_attestation(
    sources: dict[str, Any],
    captures: list[dict[str, Any]],
    capture_source_bindings: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    train_tasks = sources["train_split"].get("tasks") if isinstance(sources["train_split"].get("tasks"), list) else []
    dev_tasks = sources["development_split"].get("tasks") if isinstance(sources["development_split"].get("tasks"), list) else []
    sealed_fields = _sealed_hashes_by_field(sources["sealed_split"]["entries"])
    nonsealed_fields: dict[str, set[str]] = {
        "task_id_sha256": set(),
        "prompt_sha256": set(),
        "task_sha256": set(),
    }
    for label, rows in (("train", train_tasks), ("development", dev_tasks)):
        for row in rows:
            if not isinstance(row, dict):
                continue
            mapping = {
                "task_id_sha256": row.get("raw_id_sha256"),
                "prompt_sha256": row.get("prompt_sha256"),
                "task_sha256": row.get("task_sha256"),
            }
            for key, value in mapping.items():
                if isinstance(value, str):
                    nonsealed_fields[key].add(value)
    identity_overlap = sorted(
        (nonsealed_fields["task_id_sha256"] & sealed_fields["task_id_sha256"])
        | (nonsealed_fields["task_sha256"] & sealed_fields["task_sha256"])
    )
    prompt_template_overlap = sorted(nonsealed_fields["prompt_sha256"] & sealed_fields["prompt_sha256"])
    exact_duplicates = sorted(
        key for key, count in _sequence_counts(canonical_sha256(capture) for capture in captures).items() if count > 1
    )
    duplicate_groups = {
        "near_duplicate": _collision_report(
            captures,
            capture_source_bindings,
            lambda row: _near_key(str(row.get("prompt") or "")),
            strict_cross_source=True,
        ),
        "task_template_overlap": _collision_report(
            captures,
            capture_source_bindings,
            lambda row: _template_key(str(row.get("prompt") or "")),
            strict_cross_source=True,
        ),
        "tool_sequence_overlap": _collision_report(
            captures,
            capture_source_bindings,
            _tool_sequence,
            strict_cross_source=False,
        ),
        "state_transition_overlap": _collision_report(
            captures,
            capture_source_bindings,
            _state_transition,
            strict_cross_source=False,
        ),
    }
    unresolved_counts = {name: report["unresolved_count"] for name, report in duplicate_groups.items()}
    passed = not (identity_overlap or exact_duplicates or any(unresolved_counts.values()))
    return {
        "passed": passed,
        "leakage_found": not passed,
        "unresolved_leakage": not passed,
        "checks": {
            "exact_duplicate": "passed" if not exact_duplicates else "failed",
            "sealed_task_identity_overlap": "passed" if not identity_overlap else "failed",
            "sealed_prompt_template_overlap": "resolved_shared_official_template" if prompt_template_overlap else "passed",
            "near_duplicate": "passed" if duplicate_groups["near_duplicate"]["unresolved_count"] == 0 else "failed",
            "task_template_overlap": "passed" if duplicate_groups["task_template_overlap"]["unresolved_count"] == 0 else "failed",
            "tool_sequence_overlap": "passed" if duplicate_groups["tool_sequence_overlap"]["unresolved_count"] == 0 else "failed",
            "state_transition_overlap": "passed" if duplicate_groups["state_transition_overlap"]["unresolved_count"] == 0 else "failed",
        },
        "evidence": {
            "sealed_hash_overlap_count": len(identity_overlap),
            "sealed_task_identity_overlap_count": len(identity_overlap),
            "sealed_prompt_template_overlap_count": len(prompt_template_overlap),
            "sealed_prompt_template_overlap_resolved": bool(prompt_template_overlap),
            "exact_duplicate_count": len(exact_duplicates),
            "collision_policy": (
                "Near-prompt and task-template collisions are resolved only within one source task, family, and split. "
                "Tool-sequence and state-transition reuse is reported but is not leakage by itself when every row "
                "is bound to a permitted source; exact capture duplicates, cross-source prompt/template collisions, "
                "or sealed task/id identity overlap fail closed. Exact prompt-template reuse is reported and resolved "
                "only when official task identities and complete task hashes remain disjoint; Tau telecom deliberately "
                "reuses user-visible templates across different hidden states."
            ),
            "collision_reports": duplicate_groups,
            "capture_count": len(captures),
            "sealed_hash_count": sum(len(values) for values in sealed_fields.values()),
            "sealed_task_identity_hash_count": len(sealed_fields["task_id_sha256"] | sealed_fields["task_sha256"]),
            "sealed_prompt_template_hash_count": len(sealed_fields["prompt_sha256"]),
        },
        "scope": "Hashes-only sealed manifests block task/id identity reuse. Shared official prompt templates are reported separately from sealed task exposure; non-sealed captures are checked for duplicate templates, tool sequences, and state transitions.",
    }


def _redaction_attestation(sources: dict[str, Any], captures: list[dict[str, Any]]) -> dict[str, Any]:
    findings: list[dict[str, str]] = []
    for label, value in (
        ("train_tasks", sources["train_payloads"]),
        ("development_tasks", sources["development_payloads"]),
        ("captures", captures),
    ):
        text = json.dumps(value, sort_keys=True, ensure_ascii=False)
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                findings.append({"source": label, "pattern": pattern.pattern})
    return {
        "passed": not findings,
        "secrets_found": bool(findings),
        "unredacted_sensitive_data": bool(findings),
        "findings": findings,
        "reviewed_sources": ["train_tasks", "development_tasks", "captures"],
    }


def _build_protocol_config(
    *,
    tau_repo: Path,
    tau_revision: str,
    sources: dict[str, Any],
    models: dict[str, Any],
    contamination: dict[str, Any],
    redaction: dict[str, Any],
    captures: list[dict[str, Any]],
    captures_path: Path,
    created_at: str,
    hardware_class: str,
    memory_gib: int,
) -> dict[str, Any]:
    base_system_instruction = (
        "You are a Tau text-mode customer-service agent. Use only the provided tools, "
        "respect policy, verify state before final answers, and stop at terminal success."
    )
    domain_contracts = _capture_domain_contracts(captures)
    base_model = _public_model(models["base"])
    comparators = [_public_model(models["comparator-1"]), _public_model(models["comparator-2"])]
    teachers = [_public_model(models["teacher"])]
    return {
        "schema_version": PROTOCOL_CONFIG_SCHEMA_VERSION,
        "protocol_manifest": {
            "title": "Tau-3 7-9B cross-domain local QLoRA study",
            "domains": list(DOMAINS),
            "primary_metric": "macro_pass_1",
            "secondary_metrics": [
                "per_domain_pass_1",
                "invalid_tool_call_rate",
                "invalid_argument_rate",
                "policy_violation_rate",
                "harmful_state_mutation_rate",
                "tool_error_recovery_rate",
                "turns_tokens_wall_time_peak_memory",
            ],
            "paired_confidence_procedure": "domain_stratified_paired_bootstrap_95pct",
            "claim_scope": "best frozen eligible 7-9B open model under the fixed harness",
            "promotion_predicates": [
                "beat_frozen_strongest_comparator",
                "paired_ci_excludes_zero",
                "beat_unmodified_base",
                "safety_non_inferior",
                "each_domain_non_inferior",
                "all_artifact_and_budget_gates_pass",
            ],
        },
        "tau_revision": {
            "schema_version": "hfr.tau3_revision.v1",
            "repository": "https://github.com/sierra-research/tau2-bench",
            "local_path": str(tau_repo),
            "revision": tau_revision,
            "local_git": True,
            "task_schema_version": "tau2.tasks.v1",
            "split_hashes": {
                "train": sources["hashes"]["train"],
                "development": sources["hashes"]["development"],
                "sealed": sources["hashes"]["sealed"],
            },
        },
        "split_manifest": {
            "schema_version": "hfr.tau3_split_manifest.v1",
            "domains": list(DOMAINS),
            "strategy": "task_family_before_generation",
            "splits": {
                "train": {"local_path": sources["train_split_path"], "sha256": sources["hashes"]["train"], "sealed": False},
                "development": {"local_path": sources["development_split_path"], "sha256": sources["hashes"]["development"], "sealed": False},
                "sealed": {"local_path": sources["sealed_split_path"], "sha256": sources["hashes"]["sealed"], "sealed": True},
            },
            "source_manifest": {
                "local_path": sources["source_manifest_path"],
                "sha256": _sha256_file(Path(sources["source_manifest_path"])),
            },
            "training_captures": {
                "local_path": str(captures_path),
                "sha256": _sha256_file(captures_path),
                "row_count": len(captures),
                "admitted_count": sum(
                    1 for capture in captures
                    if capture.get("review", {}).get("disposition") == "admit"
                    and capture.get("outcome", {}).get("success") is True
                ),
                "rejected_count": sum(
                    1 for capture in captures
                    if capture.get("review", {}).get("disposition") == "reject"
                    or capture.get("outcome", {}).get("success") is not True
                ),
                "trajectory_ids_sha256": canonical_sha256(
                    sorted(str(capture["trajectory_id"]) for capture in captures)
                ),
            },
        },
        "harness_contract": {
            "schema_version": "hfr.tau3_harness_contract.v1",
            "fixed": True,
            "domains": list(DOMAINS),
            "mode": "text_half_duplex",
            "system_prompt_sha256": canonical_sha256({
                "base_instruction": base_system_instruction,
                "domain_policy_hashes": {
                    domain: domain_contracts[domain]["policy_sha256"] for domain in DOMAINS
                },
            }),
            "tools_sha256": canonical_sha256({
                domain: domain_contracts[domain]["ordered_tool_schema_sha256"] for domain in DOMAINS
            }),
            "domain_contracts": domain_contracts,
            "tool_order": "frozen",
            "context_window": 16384,
            "decoding": {"temperature": 0.0, "top_p": 1.0, "max_output_tokens": 1024, "seeds": [101, 202, 303, 404]},
            "turn_limit": 30,
            "retry_policy": "none",
            "stop_conditions": ["tau_terminal", "turn_limit", "invalid_state"],
            "no_test_time_search": True,
            "test_time_search": False,
        },
        "model_freeze": {
            "schema_version": "hfr.tau3_model_freeze.v1",
            "base_model": base_model,
            "comparators": comparators,
            "teachers": teachers,
            "selected_at": MODEL_SELECTED_AT,
            "pre_run_eligibility_rule": MODEL_SELECTION_RULE,
            "selection_rule": (
                "Freeze before sealed evaluation using official upstream public ungated immutable Apache-2.0 "
                "7-9B dense models, pinned content-addressed MLX conversions, documented tool/function use, "
                "one identical common harness, and no per-model prompt tuning."
            ),
            "excluded_by_rule": [
                {
                    "family": "Hermes-3/community fine-tunes",
                    "reason": "community/custom-license or derivative fine-tune status did not satisfy the pre-run official/developer-owned Apache-2.0 comparator rule",
                },
                {
                    "family": "custom-license alternatives",
                    "reason": "custom or non-Apache license terms were excluded before evaluation; exclusion is not a benchmark superiority claim",
                },
            ],
            "benchmark_superiority_claimed": False,
            "teacher_policy": (
                "Teachers are pinned for local generation and review evidence only. They are excluded from "
                "the 7-9B dense comparator eligibility rule and from benchmark superiority claims."
            ),
        },
        "budget": {
            "schema_version": "hfr.tau3_budget.v1",
            "max_seconds": 604800,
            "reserved_final_eval": True,
            "reserved_final_eval_seconds": 86400,
            "local_only": True,
            "network": False,
            "stages": {"generation": 172800, "search": 172800, "final_training": 86400, "final_eval": 86400, "contingency": 86400},
            "deny_when_final_eval_cannot_complete": True,
        },
        "sealed_manifest": {
            "schema_version": "hfr.tau3_sealed_manifest.v1",
            "quarantined_at": created_at,
            "quarantine_predates_generation": True,
            "access_count": 0,
            "leakage_blocking_hashes": sorted({
                value
                for row in sources["sealed_split"]["entries"]
                for value in (row["task_id_sha256"], row["task_sha256"])
            }),
            "prompt_template_hashes": sorted({row["prompt_sha256"] for row in sources["sealed_split"]["entries"]}),
            "prompt_template_overlap_policy": "report_and_resolve_only_when_task_id_and_complete_task_hashes_are_disjoint",
            "manifest_sha256": sources["hashes"]["sealed"],
        },
        "mlx_qlora_plan": {
            "schema_version": "hfr.tau3_mlx_qlora_plan.v1",
            "passed": True,
            "backend": "mlx-lm",
            "mlx_lm_version": "0.31.3",
            "mlx_version": "0.32.0",
            "method": "4-bit QLoRA",
            "quantization": "4-bit",
            "local_only": True,
            "network": False,
            "data_layout": {
                "required_files": ["train.jsonl", "valid.jsonl"],
                "test_file_required": False,
                "sealed_or_test_rows": 0,
                "root": "input_export"
            },
            "command_argv": [
                "python",
                "-m",
                "mlx_lm",
                "lora",
                "--train",
                "--fine-tune-type",
                "lora",
                "--model",
                "model_input",
                "--data",
                "input_export",
                "--adapter-path",
                "adapter_output",
                "--batch-size",
                "1",
                "--grad-accumulation-steps",
                "8",
                "--max-seq-length",
                "12288",
                "--learning-rate",
                "5e-5",
                "--iters",
                "200",
                "--seed",
                "8675309",
                "--grad-checkpoint",
            ],
            "forbidden_flags": ["--test", "--push-to-hub", "--upload", "--allow-network", "--report-to", "--wandb"],
            "resume": {"enabled": True, "requires_recipe_and_dataset_hash_match": True},
            "stop_conditions": ["divergence", "budget_breach", "development_regression", "safety_regression", "artifact_validation_failure"],
            "output_contract": {"adapter_only": True, "base_revision_required": True, "quantization_identity_required": True},
        },
        "recipe_space": {
            "schema_version": "hfr.tau3_recipe_space.v1",
            "bounded": True,
            "max_trials": 12,
            "sealed_used": False,
            "development_only": True,
            "bounds": {
                "rank": [8, 16, 32],
                "alpha": [16, 32, 64],
                "learning_rate": [0.00001, 0.0002],
                "sequence_length": [4096, 16384],
                "steps": [100, 2000],
            },
        },
        "candidate_selection_contract": {
            "schema_version": "hfr.tau3_candidate_selection.v1",
            "passed": True,
            "development_only": True,
            "sealed_used": False,
            "one_untouched_checkpoint": True,
            "primary_metric": "macro_pass_1",
            "safety_non_inferiority_margin": 0.01,
            "per_domain_non_inferiority_margin": 0.03,
            "bootstrap": {"kind": "domain_stratified_paired", "confidence": 0.95, "resamples": 10000, "seed": 8675309},
        },
        "contamination_attestation": contamination,
        "redaction_attestation": redaction,
        "licenses": [
            {"id": "tau2-bench", "status": "approved", "training_allowed": True, "license": "MIT"},
            {"id": "mlx-community/Qwen3.5-9B-4bit", "status": "approved", "training_allowed": True, "license": "Apache-2.0"},
            {"id": "mlx-community/Qwen3-8B-4bit", "status": "approved", "training_allowed": True, "license": "Apache-2.0"},
            {"id": "mlx-community/granite-3.3-8b-instruct-4bit", "status": "approved", "training_allowed": True, "license": "Apache-2.0"},
            {
                "id": "mlx-community/Qwen3.6-35B-A3B-4bit",
                "status": "approved",
                "training_allowed": True,
                "license": "Apache-2.0",
                "usage": "teacher_generation_and_review_only",
            },
        ],
        "environment_manifest": {
            "schema_version": "hfr.tau3_environment.v1",
            "hardware_class": hardware_class,
            "memory_gib": memory_gib,
            "network_allowed": False,
            "device_identifiers_recorded": False,
        },
    }


def _public_model(model: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "name",
        "revision",
        "parameters_billion",
        "architecture",
        "license",
        "quantization",
        "tokenizer",
        "chat_template",
        "model_card_url",
        "local_path",
        "local_identity_path",
        "local_identity_sha256",
        "local_tree_sha256",
        "local_file_count",
        "upstream",
        "evidence_urls",
        "pre_run_eligibility",
    )
    public = {key: model[key] for key in keys}
    if "role" in model:
        public["role"] = model["role"]
    return public


def _require_clean_tau_checkout(repo: Path, expected_revision: str) -> None:
    if not HEX40_RE.fullmatch(expected_revision):
        raise Tau3ProtocolFreezeError("Tau revision must be an exact lowercase 40-hex git commit")
    if path_has_symlink_component(repo, include_leaf=True):
        raise Tau3ProtocolFreezeError("Tau repository path contains a symlink component")
    if not repo.is_dir() or not (repo / ".git").exists():
        raise Tau3ProtocolFreezeError("Tau repository must be a local git checkout")
    actual = _git(repo, "rev-parse", "HEAD")
    if actual != expected_revision:
        raise Tau3ProtocolFreezeError(f"Tau checkout revision mismatch: expected {expected_revision}, got {actual}")
    if _git(repo, "status", "--porcelain=v1"):
        raise Tau3ProtocolFreezeError("Tau checkout must be clean")


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(["git", "-C", str(repo), *args], check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise Tau3ProtocolFreezeError(completed.stderr.strip() or completed.stdout.strip() or "git failed")
    return completed.stdout.strip()


def _require_new_output_file(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise Tau3ProtocolFreezeError("protocol output must not already exist")
    if not path.parent.is_dir():
        raise Tau3ProtocolFreezeError("protocol output parent directory must exist")
    if path_has_symlink_component(path.parent, include_leaf=True):
        raise Tau3ProtocolFreezeError("protocol output parent path contains a symlink component")


def _reject_symlinked_file(path: Path) -> None:
    if path_has_symlink_component(path, include_leaf=True):
        raise Tau3ProtocolFreezeError(f"path must not contain symlink components: {path}")
    if not path.is_file():
        raise Tau3ProtocolFreezeError(f"path must be an existing regular file: {path}")
    mode = path.stat(follow_symlinks=False).st_mode
    if not stat.S_ISREG(mode):
        raise Tau3ProtocolFreezeError(f"path must be a regular file: {path}")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Tau3ProtocolFreezeError(f"invalid JSON in {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise Tau3ProtocolFreezeError(f"invalid JSONL in {path}:{line_number}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise Tau3ProtocolFreezeError(f"invalid UTF-8 in {path}: {exc}") from exc
    return rows


def _assert_artifact_record(path: Path, record: dict[str, Any], label: str) -> None:
    size = path.stat(follow_symlinks=False).st_size
    digest = _sha256_file(path)
    if record.get("size") != size or record.get("sha256") != digest:
        raise Tau3ProtocolFreezeError(f"source artifact record does not replay for {label}")


def _required_string_list(obj: dict[str, Any], key: str, label: str) -> list[str]:
    value = obj.get(key)
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item for item in value):
        raise Tau3ProtocolFreezeError(f"{label} must contain non-empty string list {key}")
    return value


def _assert_split_payload_count(split: dict[str, Any], payloads: list[Any], label: str) -> None:
    if not payloads:
        raise Tau3ProtocolFreezeError(f"{label} task payload JSONL must be non-empty")
    if split.get("task_count") != len(payloads):
        raise Tau3ProtocolFreezeError(f"{label} task payload count does not match split manifest")


def _source_entries_from_split(split: dict[str, Any]) -> dict[tuple[str, str, str, str, str], dict[str, Any]]:
    rows = split.get("tasks")
    if not isinstance(rows, list) or not rows:
        raise Tau3ProtocolFreezeError(f"{split.get('split')} split must contain task entries")
    entries: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise Tau3ProtocolFreezeError(f"{split.get('split')} split task {index} must be an object")
        for key in ("domain", "raw_id", "prompt_sha256", "task_sha256", "family_id"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise Tau3ProtocolFreezeError(f"{split.get('split')} split task {index} missing {key}")
        key = (row["domain"], str(split["split"]), row["family_id"], row["task_sha256"], row["prompt_sha256"])
        if key in entries:
            raise Tau3ProtocolFreezeError(f"{split.get('split')} split contains duplicate source entry")
        entries[key] = row
    return entries


def _validate_training_source_envelopes(
    payloads: list[Any],
    split_entries: dict[tuple[str, str, str, str, str], dict[str, Any]],
    source_revision: str,
    split: str,
) -> list[dict[str, Any]]:
    _assert_split_payload_count({"task_count": len(split_entries)}, payloads, split)
    envelopes: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for index, row in enumerate(payloads):
        if not isinstance(row, dict):
            raise Tau3ProtocolFreezeError(f"{split} source envelope {index} must be an object")
        if row.get("schema_version") != TRAINING_SOURCE_SCHEMA_VERSION:
            raise Tau3ProtocolFreezeError(f"{split} source envelope schema_version must be {TRAINING_SOURCE_SCHEMA_VERSION}")
        if row.get("source_revision") != source_revision:
            raise Tau3ProtocolFreezeError(f"{split} source envelope source_revision does not match manifest")
        if row.get("split") != split:
            raise Tau3ProtocolFreezeError(f"{split} source envelope split mismatch")
        task = row.get("task")
        if not isinstance(task, dict):
            raise Tau3ProtocolFreezeError(f"{split} source envelope task must be an object")
        task_id = task.get("id")
        if not isinstance(task_id, str) or not task_id:
            raise Tau3ProtocolFreezeError(f"{split} source envelope task.id must be a non-empty string")
        replayed_task_hash = canonical_sha256(task)
        replayed_prompt_hash = canonical_sha256(_prompt_material(task))
        if row.get("task_sha256") != replayed_task_hash:
            raise Tau3ProtocolFreezeError(f"{split} source envelope task_sha256 does not replay")
        if row.get("prompt_sha256") != replayed_prompt_hash:
            raise Tau3ProtocolFreezeError(f"{split} source envelope prompt_sha256 does not replay")
        for key_name in ("domain", "task_family", "task_sha256", "prompt_sha256"):
            if not isinstance(row.get(key_name), str) or not row[key_name]:
                raise Tau3ProtocolFreezeError(f"{split} source envelope {key_name} must be a non-empty string")
        source_key = (row["domain"], split, row["task_family"], row["task_sha256"], row["prompt_sha256"])
        entry = split_entries.get(source_key)
        if entry is None:
            raise Tau3ProtocolFreezeError(f"{split} source envelope does not match split entry")
        if entry["raw_id"] != task_id:
            raise Tau3ProtocolFreezeError(f"{split} source envelope task.id does not match split raw_id")
        if source_key in seen:
            raise Tau3ProtocolFreezeError(f"{split} source envelope duplicates a split entry")
        seen.add(source_key)
        envelopes.append(row)
    if seen != set(split_entries):
        raise Tau3ProtocolFreezeError(f"{split} source envelopes do not cover split entries exactly")
    return envelopes


def _build_source_index(envelopes: list[dict[str, Any]]) -> dict[tuple[str, str, str, str, str, str], dict[str, Any]]:
    index: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    for envelope in envelopes:
        task = envelope["task"]
        key = (
            str(task["id"]),
            str(envelope["domain"]),
            str(envelope["split"]),
            str(envelope["task_family"]),
            str(envelope["task_sha256"]),
            str(envelope["prompt_sha256"]),
        )
        if key in index:
            raise Tau3ProtocolFreezeError("permitted source index contains duplicate entries")
        index[key] = envelope
    return index


def _assert_hashes_only_sealed(sealed: dict[str, Any], source_revision: str) -> None:
    if sealed.get("source_revision") != source_revision:
        raise Tau3ProtocolFreezeError("sealed manifest source_revision does not match source manifest")
    if sealed.get("hashes_only") is not True:
        raise Tau3ProtocolFreezeError("sealed manifest must set hashes_only=true")
    rows = sealed.get("entries")
    if not isinstance(rows, list):
        raise Tau3ProtocolFreezeError("sealed manifest entries must be a list")
    if sealed.get("task_count") != len(rows):
        raise Tau3ProtocolFreezeError("sealed manifest task_count does not match entries")
    required = {"prompt_sha256", "task_id_sha256", "task_sha256"}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise Tau3ProtocolFreezeError(f"sealed row {index} must be an object")
        if set(row) != required:
            raise Tau3ProtocolFreezeError("sealed manifest must contain only prompt/task hash fields")
        for key, value in row.items():
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise Tau3ProtocolFreezeError(f"sealed manifest field {key} must be a SHA-256 digest")


def _sealed_hashes_by_field(rows: list[dict[str, str]]) -> dict[str, set[str]]:
    fields = {"task_id_sha256": set(), "prompt_sha256": set(), "task_sha256": set()}
    for row in rows:
        for key in fields:
            value = row.get(key)
            if isinstance(value, str):
                fields[key].add(value)
    return fields


def _prompt_material(task: dict[str, Any]) -> Any:
    scenario = task.get("user_scenario")
    if isinstance(scenario, dict):
        return scenario.get("instructions", scenario)
    return scenario


def _template_keys(prompts: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for prompt in prompts:
        key = re.sub(r"[0-9a-f]{8,}|[0-9]+", "<id>", prompt.lower())
        key = re.sub(r"\s+", " ", key).strip()
        if key:
            counts[key] = counts.get(key, 0) + 1
    return counts


def _tool_sequence(capture: dict[str, Any]) -> str:
    events = capture.get("events") if isinstance(capture.get("events"), list) else []
    sequence = [
        str(event.get("tool_name"))
        for event in events
        if isinstance(event, dict) and event.get("type") == "tool_call" and isinstance(event.get("tool_name"), str)
    ]
    return json.dumps(sequence, separators=(",", ":"))


def _state_transition(capture: dict[str, Any]) -> str:
    transition = capture.get("state_transition") if isinstance(capture.get("state_transition"), dict) else {}
    return canonical_sha256({
        "before": transition.get("before_hash"),
        "after": transition.get("after_hash"),
        "changes": transition.get("changes"),
    })


def _sequence_counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _collision_report(
    captures: list[dict[str, Any]],
    bindings: dict[str, dict[str, Any]],
    key_fn: Any,
    *,
    strict_cross_source: bool,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for capture in captures:
        key = key_fn(capture)
        if key:
            groups.setdefault(str(key), []).append(capture)
    detected = 0
    resolved = 0
    unresolved = 0
    cross_source = 0
    for rows in groups.values():
        if len(rows) < 2:
            continue
        detected += 1
        source_keys = {_source_collision_key(bindings[str(row["trajectory_id"])]) for row in rows}
        behaviors = {str(row.get("behavior")) for row in rows}
        if len(source_keys) > 1:
            cross_source += 1
        if not strict_cross_source:
            resolved += 1
        elif len(source_keys) == 1 and len(behaviors) > 1:
            resolved += 1
        else:
            unresolved += 1
    return {
        "detected_count": detected,
        "resolved_count": resolved,
        "unresolved_count": unresolved,
        "cross_source_count": cross_source,
    }


def _capture_domain_contracts(captures: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    contracts: dict[str, dict[str, str]] = {}
    for domain in DOMAINS:
        rows = [row for row in captures if row.get("domain") == domain]
        policy_hashes = {str(row.get("policy_hash") or "") for row in rows}
        tool_hashes = {str(row.get("tool_schema_revision") or "") for row in rows}
        if len(policy_hashes) != 1 or "" in policy_hashes:
            raise Tau3ProtocolFreezeError(f"captures do not freeze exactly one policy hash for {domain}")
        if len(tool_hashes) != 1 or "" in tool_hashes:
            raise Tau3ProtocolFreezeError(f"captures do not freeze exactly one ordered tool schema hash for {domain}")
        contracts[domain] = {
            "policy_sha256": next(iter(policy_hashes)),
            "ordered_tool_schema_sha256": next(iter(tool_hashes)),
        }
    return contracts


def _source_collision_key(envelope: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(envelope["domain"]),
        str(envelope["split"]),
        str(envelope["task_family"]),
        str(envelope["task_sha256"]),
    )


def _near_key(prompt: str) -> str:
    return _template_key(prompt).replace(" customer ", " ")


def _template_key(prompt: str) -> str:
    key = re.sub(r"[0-9a-f]{8,}|[0-9]+|customer [a-z]\b", "<id>", prompt.lower())
    key = re.sub(r"\s+", " ", key).strip()
    return key


def _assert_no_placeholders(value: Any) -> None:
    rendered = json.dumps(value, sort_keys=True, ensure_ascii=False)
    if "REPLACE_WITH_" in rendered or "TODO" in rendered:
        raise Tau3ProtocolFreezeError("protocol still contains unresolved placeholders")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
