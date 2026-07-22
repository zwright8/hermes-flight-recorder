"""Strict semantic validation for Tau-3 training-readiness bundles.

The validator is intentionally dependency-free and fail-closed.  It checks the
content-addressed bundle inventory first, then validates the study constraints
that make the produced artifacts admissible for the local-only 7-9B QLoRA run.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema_registry import SchemaRegistryError, check_schema_contract

TAU3_TRAINING_BUNDLE_SCHEMA_VERSION = "hfr.tau3_training_bundle.v1"
TAU3_TRAINING_VALIDATION_SCHEMA_VERSION = "hfr.tau3_training_bundle_validation.v1"

REQUIRED_DOMAINS = {"airline", "retail", "telecom"}
MAX_BUDGET_SECONDS = 604_800
MIN_MODEL_BILLIONS = 7.0
MAX_MODEL_BILLIONS = 9.0

REQUIRED_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("protocol_manifest", "protocol/protocol_manifest.json"),
    ("tau_revision", "protocol/tau_revision.json"),
    ("split_manifest", "protocol/split_manifest.json"),
    ("harness_contract", "protocol/harness_contract.json"),
    ("model_freeze", "protocol/model_freeze.json"),
    ("budget", "protocol/budget.json"),
    ("sealed_manifest", "sealed/sealed_manifest.json"),
    ("trajectories", "generation/trajectories.jsonl"),
    ("admission_ledger", "generation/admission_ledger.jsonl"),
    ("rejection_ledger", "generation/rejection_ledger.jsonl"),
    ("balance_report", "generation/balance_report.json"),
    ("contamination_report", "generation/contamination_report.json"),
    ("redaction_report", "generation/redaction_report.json"),
    ("license_report", "generation/license_report.json"),
    ("dataset_identity", "generation/dataset_identity.json"),
    ("sft_export", "exports/sft.jsonl"),
    ("action_sft_export", "exports/action_sft.jsonl"),
    ("dpo_export", "exports/dpo.jsonl"),
    ("export_manifest", "exports/manifest.json"),
    ("dataset_card", "exports/DATASET_CARD.md"),
    ("model_manifest", "training/model_manifest.json"),
    ("dataset_manifest", "training/dataset_manifest.json"),
    ("mlx_qlora_plan", "training/mlx_qlora_plan.json"),
    ("recipe_space", "training/recipe_space.json"),
    ("candidate_selection_contract", "training/candidate_selection_contract.json"),
    ("agentic_training_plan", "training/agentic_training_plan.json"),
    ("runtime_preflight", "training/runtime_preflight.json"),
    ("trainer_preflight", "training/trainer_preflight.json"),
    ("trainer_launch_check", "training/trainer_launch_check.json"),
    ("trainer_archive", "training/trainer_archive/trainer_archive.json"),
    ("trainer_archive_check", "training/trainer_archive_check.json"),
    ("trainer_consumer_plan", "training/trainer_consumer_plan.json"),
    ("rehearsal_result", "rehearsal/rehearsal_result.json"),
    ("evidence_bundle", "evidence/evidence_bundle.json"),
)

REQUIRED_ARTIFACT_MAP = dict(REQUIRED_ARTIFACTS)
FORBIDDEN_ENDPOINT_WORDS = (
    "api.openai.com",
    "anthropic.com",
    "together.ai",
    "replicate.com",
    "fireworks.ai",
    "modal.com",
    "runpod",
    "sagemaker",
    "vertex",
    "azure",
    "bedrock",
    "huggingface.co/inference",
)
PRIVATE_PATH_PATTERNS = (
    re.compile(r"/(?:Users|home)/[^\s\"']+"),
    re.compile(r"/(?:private/var|var/folders)/[^\s\"']+"),
    re.compile(r"[A-Za-z]:\\Users\\[^\s\"']+"),
)
CREDENTIAL_PATTERNS = (
    re.compile(r"\bhf_[A-Za-z0-9]{12,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(
        r'"(?:api[_-]?key|access[_-]?token|secret|password)"\s*:\s*"(?!<?redacted)[^\"]{8,}"',
        re.IGNORECASE,
    ),
)


def build_artifact_record(bundle_dir: str | Path, role: str) -> dict[str, Any]:
    """Return the canonical manifest record for one required artifact role."""

    if role not in REQUIRED_ARTIFACT_MAP:
        raise ValueError(f"unknown Tau-3 artifact role: {role}")
    root = Path(bundle_dir)
    rel_path = REQUIRED_ARTIFACT_MAP[role]
    path = root / rel_path
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "role": role,
        "path": rel_path,
        "size": path.stat().st_size,
        "sha256": _sha256(path),
    }


def build_bundle_manifest(
    bundle_dir: str | Path,
    *,
    bundle_mode: str,
    ready_for_training: bool,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic root manifest for all required Tau-3 artifacts."""

    if bundle_mode not in {"production", "rehearsal"}:
        raise ValueError("bundle_mode must be 'production' or 'rehearsal'")
    manifest: dict[str, Any] = {
        "schema_version": TAU3_TRAINING_BUNDLE_SCHEMA_VERSION,
        "bundle_mode": bundle_mode,
        "ready_for_training": bool(ready_for_training),
        "artifacts": [build_artifact_record(bundle_dir, role) for role, _ in REQUIRED_ARTIFACTS],
    }
    if created_at is not None:
        manifest["created_at"] = created_at
    return manifest


@dataclass
class _Context:
    bundle: Path
    strict: bool
    allow_rehearsal: bool
    checks: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, Path] = field(default_factory=dict)
    payloads: dict[str, Any] = field(default_factory=dict)
    jsonl_rows: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    text_payloads: dict[str, str] = field(default_factory=dict)

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


def _json_safe(value: Any) -> Any:
    """Normalize validator diagnostics so receipts are always valid JSON."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(_json_safe(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def validate_tau3_training_bundle(
    bundle_dir: str | Path,
    *,
    strict: bool = False,
    allow_rehearsal: bool = False,
) -> dict[str, Any]:
    """Validate a Tau-3 training-readiness bundle rooted at ``bundle_dir``."""

    ctx = _Context(Path(bundle_dir), strict=strict, allow_rehearsal=allow_rehearsal)
    _load_bundle(ctx)
    if ctx.payloads.get("manifest"):
        _validate_bundle_mode(ctx)
        _validate_schema_contracts(ctx)
        _validate_protocol(ctx)
        _validate_generation(ctx)
        _validate_exports(ctx)
        _validate_training(ctx)
        _validate_no_sealed_or_cloud_leaks(ctx)
    failed = [check for check in ctx.checks if not check["passed"]]
    manifest = ctx.payloads.get("manifest") if isinstance(ctx.payloads.get("manifest"), dict) else {}
    return {
        "schema_version": TAU3_TRAINING_VALIDATION_SCHEMA_VERSION,
        "passed": not failed,
        "status": "passed" if not failed else "failed",
        "summary": (
            "Tau-3 training-readiness bundle passed strict validation."
            if not failed
            else f"Tau-3 training-readiness bundle failed {len(failed)} check(s)."
        ),
        "bundle": str(ctx.bundle),
        "bundle_mode": manifest.get("bundle_mode"),
        "ready_for_training": bool(manifest.get("ready_for_training")) if isinstance(manifest, dict) else False,
        "strict": strict,
        "allow_rehearsal": allow_rehearsal,
        "check_count": len(ctx.checks),
        "failed_check_count": len(failed),
        "checks": ctx.checks,
    }


def _load_bundle(ctx: _Context) -> None:
    ctx.add("bundle_directory_exists", ctx.bundle.is_dir(), str(ctx.bundle), "existing directory")
    if not ctx.bundle.is_dir():
        return
    manifest_path = ctx.bundle / "manifest.json"
    ctx.add("root_manifest_exists", manifest_path.is_file(), "manifest.json", "present")
    if not manifest_path.is_file():
        return
    manifest = _read_json(ctx, "manifest", manifest_path)
    if not isinstance(manifest, dict):
        ctx.add("root_manifest_object", False, type(manifest).__name__, "object")
        return
    ctx.payloads["manifest"] = manifest
    ctx.add(
        "root_schema_version",
        manifest.get("schema_version") == TAU3_TRAINING_BUNDLE_SCHEMA_VERSION,
        manifest.get("schema_version"),
        TAU3_TRAINING_BUNDLE_SCHEMA_VERSION,
    )
    records = _artifact_records(manifest)
    by_path = {record.get("path"): record for record in records if isinstance(record.get("path"), str)}
    by_role = {record.get("role"): record for record in records if isinstance(record.get("role"), str)}
    ctx.add("manifest_artifact_records_present", bool(records), len(records), "non-empty artifact inventory")
    for role, rel_path in REQUIRED_ARTIFACTS:
        record = by_role.get(role) or by_path.get(rel_path)
        ctx.add(f"manifest_binds:{role}", isinstance(record, dict), record, f"{role} -> {rel_path}")
        if not isinstance(record, dict):
            continue
        record_path = record.get("path")
        safe_path = _safe_relative_path(record_path)
        ctx.add(f"artifact_path_safe:{role}", safe_path is not None, record_path, "relative path below bundle")
        if safe_path is None:
            continue
        ctx.add(f"artifact_path_expected:{role}", safe_path == Path(rel_path), str(safe_path), rel_path)
        path = ctx.bundle / safe_path
        ctx.artifacts[role] = path
        ctx.add(f"artifact_exists:{role}", path.is_file(), str(path), "file exists")
        if not path.is_file():
            continue
        size = path.stat().st_size
        digest = _sha256(path)
        ctx.add(f"artifact_size_replays:{role}", record.get("size") == size, record.get("size"), size)
        ctx.add(f"artifact_sha256_replays:{role}", record.get("sha256") == digest, record.get("sha256"), digest)
        if path.suffix == ".json":
            payload = _read_json(ctx, role, path)
            if isinstance(payload, dict):
                ctx.payloads[role] = payload
        elif path.suffix == ".jsonl":
            ctx.jsonl_rows[role] = _read_jsonl(ctx, role, path)
        else:
            try:
                ctx.text_payloads[role] = path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                ctx.add(f"artifact_utf8:{role}", False, str(exc), "UTF-8 text")


def _validate_bundle_mode(ctx: _Context) -> None:
    manifest = ctx.payloads["manifest"]
    mode = manifest.get("bundle_mode")
    ready = manifest.get("ready_for_training")
    ctx.add("bundle_mode_known", mode in {"production", "rehearsal"}, mode, "production or rehearsal")
    ctx.add("ready_for_training_boolean", isinstance(ready, bool), ready, "boolean")
    if ctx.strict and not ctx.allow_rehearsal:
        ctx.add("strict_requires_production_mode", mode == "production", mode, "production")
        ctx.add("strict_requires_ready_for_training", ready is True, ready, True)
    if mode == "rehearsal":
        ctx.add("rehearsal_not_ready_for_training", ready is False, ready, False)


def _validate_schema_contracts(ctx: _Context) -> None:
    for role, payload in sorted(ctx.payloads.items()):
        if role == "manifest":
            name = "tau3_training_bundle"
        elif not isinstance(payload, dict) or not isinstance(payload.get("schema_version"), str):
            continue
        else:
            name = str(payload["schema_version"])
        try:
            result = check_schema_contract(payload, name_or_id=name)
        except SchemaRegistryError:
            continue
        if role == "manifest" or payload.get("strict_schema_required") is True:
            ctx.add(f"schema_contract:{role}", bool(result.get("passed")), result.get("errors", []), [])
        else:
            ctx.add(
                f"schema_contract_checked:{role}",
                True,
                {"passed": bool(result.get("passed")), "errors": result.get("errors", [])},
                "informational nested schema check",
            )


def _validate_protocol(ctx: _Context) -> None:
    tau = _payload(ctx, "tau_revision")
    split = _payload(ctx, "split_manifest")
    harness = _payload(ctx, "harness_contract")
    models = _payload(ctx, "model_freeze")
    budget = _payload(ctx, "budget")
    protocol = _payload(ctx, "protocol_manifest")
    sealed = _payload(ctx, "sealed_manifest")

    protocol_body = dict(protocol)
    protocol_body.pop("signature", None)
    protocol_body.pop("signature_algorithm", None)
    expected_signature = _canonical_sha256({
        "protocol_manifest": protocol_body,
        "tau_revision": tau,
        "split_manifest": split,
        "harness_contract": harness,
        "model_freeze": models,
        "budget": budget,
    })
    ctx.add(
        "protocol_signature_replays",
        protocol.get("signature") == expected_signature,
        protocol.get("signature"),
        expected_signature,
    )

    ctx.add("protocol_domains_exact", _domains(protocol or harness or split) == REQUIRED_DOMAINS, sorted(_domains(protocol or harness or split)), sorted(REQUIRED_DOMAINS))
    ctx.add("tau_revision_local_git", _truthy(tau, "local_git", "local_git_revision", "is_local_git_revision"), _pick(tau, "local_git", "local_git_revision", "is_local_git_revision"), True)
    ctx.add("tau_repository_identity_present", bool(_pick(tau, "repository", "repository_url", "repo")), _pick(tau, "repository", "repository_url", "repo"), "repository identity")
    ctx.add("tau_revision_hash_present", bool(_pick(tau, "revision", "git_revision", "commit_sha", "commit")), _pick(tau, "revision", "git_revision", "commit_sha", "commit"), "git revision")
    ctx.add("tau_task_schema_version_present", bool(_pick(tau, "task_schema_version", "task_version", "schema_revision")), _pick(tau, "task_schema_version", "task_version", "schema_revision"), "task schema version")
    ctx.add("tau_revision_split_hashes_present", bool(_digests_from(tau) or _digests_from(split)), _digests_from(tau) or _digests_from(split), "split hashes")
    split_names = _split_names(split)
    ctx.add("split_manifest_required_splits", {"train", "development", "sealed"}.issubset(split_names), sorted(split_names), ["train", "development", "sealed"])
    ctx.add("split_hashes_present", all(_split_hash(split, name) for name in ("train", "development", "sealed")), [_split_hash(split, name) for name in ("train", "development", "sealed")], "hash per split")
    split_strategy = str(_pick(split, "strategy", "split_strategy") or "").lower()
    ctx.add("split_by_family_before_generation", "family" in split_strategy and "generation" in split_strategy, split_strategy, "task/scenario family before generation")

    ctx.add("harness_fixed", _truthy(harness, "fixed", "frozen", "deterministic"), _pick(harness, "fixed", "frozen", "deterministic"), True)
    ctx.add("harness_no_test_time_search", _falsey(harness, "test_time_search", "search_at_test_time") or _truthy(harness, "no_test_time_search"), _pick(harness, "test_time_search", "search_at_test_time", "no_test_time_search"), "disabled")
    ctx.add("harness_domains_exact", _domains(harness) == REQUIRED_DOMAINS, sorted(_domains(harness)), sorted(REQUIRED_DOMAINS))
    ctx.add("harness_decoding_seeds_fixed", bool(_pick(harness, "decoding", "seeds", "random_seeds")), _pick(harness, "decoding", "seeds", "random_seeds"), "fixed decoding/seeds")
    ctx.add(
        "harness_prompt_tools_context_limits_frozen",
        bool(_pick(harness, "system_prompt_sha256", "system_prompt_hash"))
        and bool(_pick(harness, "tool_order", "tool_schema_revision", "tools_sha256"))
        and _number(_pick(harness, "context_window")) is not None
        and _number(_pick(harness, "turn_limit", "step_limit")) is not None
        and bool(_pick(harness, "retry_policy")),
        _summary(harness),
        "prompt/tools/context/turn/retry contract",
    )

    base = _model(models, "base_model")
    comparators = _comparators(models)
    ctx.add("base_model_7_to_9b_dense", _eligible_model(base) and _dense_model(base), _model_summary(base), "dense 7-9B")
    ctx.add("base_model_quantization_4bit", _is_4bit(_pick(base, "quantization", "quantization_bits", "bits")), _pick(base, "quantization", "quantization_bits", "bits"), "4-bit")
    eligible = [model for model in comparators if _eligible_model(model)]
    ctx.add("at_least_two_eligible_comparators", len(eligible) >= 2, [_model_summary(model) for model in comparators], ">=2 eligible 7-9B comparators")
    ctx.add("model_revisions_and_licenses_present", _model_identity_complete(base) and all(_model_identity_complete(model) for model in comparators), [_model_summary(base), *[_model_summary(m) for m in comparators]], "name/revision/license")
    ctx.add(
        "all_frozen_models_have_quantization_identity",
        _is_4bit(_pick(base, "quantization", "quantization_bits", "bits"))
        and all(_is_4bit(_pick(model, "quantization", "quantization_bits", "bits")) for model in comparators),
        [_model_summary(base), *[_model_summary(model) for model in comparators]],
        "4-bit identity for base and comparators",
    )
    ctx.add(
        "base_tokenizer_chat_template_model_card_present",
        bool(_pick(base, "tokenizer", "tokenizer_id"))
        and bool(_pick(base, "chat_template", "chat_template_sha256"))
        and bool(_pick(base, "model_card_url", "original_model_card_url")),
        _model_summary(base),
        "tokenizer/chat template/original model card",
    )

    max_seconds = _number(_pick(budget, "max_seconds", "wallclock_seconds", "total_seconds"))
    ctx.add("budget_max_seven_days", max_seconds is not None and max_seconds <= MAX_BUDGET_SECONDS, max_seconds, f"<= {MAX_BUDGET_SECONDS}")
    ctx.add("budget_reserves_final_eval", _truthy(budget, "reserved_final_eval", "final_eval_reserved", "sealed_eval_reserved"), _pick(budget, "reserved_final_eval", "final_eval_reserved", "sealed_eval_reserved"), True)
    ctx.add("budget_final_eval_seconds_positive", (_number(_pick(budget, "reserved_final_eval_seconds", "final_eval_seconds")) or 0) > 0, _pick(budget, "reserved_final_eval_seconds", "final_eval_seconds"), ">0")
    ctx.add("budget_stage_allocations_present", isinstance(budget.get("stages"), dict) and bool(budget["stages"]), budget.get("stages"), "per-stage allocations")
    ctx.add("budget_local_only", _local_only(budget), _pick(budget, "local_only", "network", "allow_network"), "local only")

    ctx.add("sealed_quarantine_predates_generation", _truthy(sealed, "quarantine_predates_generation", "predates_generation"), _pick(sealed, "quarantine_predates_generation", "predates_generation"), True)
    ctx.add("sealed_prompt_hashes_present", bool(_sealed_prompt_hashes(ctx)), sorted(_sealed_prompt_hashes(ctx)), "sealed prompt hashes")


def _validate_generation(ctx: _Context) -> None:
    trajectories = ctx.jsonl_rows.get("trajectories", [])
    admissions = ctx.jsonl_rows.get("admission_ledger", [])
    rejections = ctx.jsonl_rows.get("rejection_ledger", [])
    balance = _payload(ctx, "balance_report")
    contamination = _payload(ctx, "contamination_report")
    redaction = _payload(ctx, "redaction_report")
    license_report = _payload(ctx, "license_report")
    dataset_identity = _payload(ctx, "dataset_identity")

    ctx.add("trajectories_non_empty", bool(trajectories), len(trajectories), "at least one row")
    ctx.add("trajectories_domains_present", REQUIRED_DOMAINS.issubset(_row_domains(trajectories)), sorted(_row_domains(trajectories)), sorted(REQUIRED_DOMAINS))
    ctx.add("trajectories_behavior_coverage", _behavior_coverage(trajectories), sorted(_behaviors(trajectories)), "success/correction/clarification/refusal/recovery/policy/harmful/hallucinated_tool/premature_completion")
    ctx.add("balance_domains_present", REQUIRED_DOMAINS.issubset(_domains(balance) or set((balance.get("token_share_by_domain") or {}).keys() if isinstance(balance.get("token_share_by_domain"), dict) else [])), _domains(balance) or list((balance.get("token_share_by_domain") or {}).keys()) if isinstance(balance, dict) else [], sorted(REQUIRED_DOMAINS))
    token_shares = _token_shares(balance, trajectories)
    ctx.add("balance_token_share_max_45pct", bool(token_shares) and all(value <= 0.45 for value in token_shares.values()), token_shares, "each domain <= 0.45")

    admitted_ids = {_row_id(row) for row in admissions if _row_id(row)}
    rejected_ids = {_row_id(row) for row in rejections if _row_id(row)}
    ctx.add("admission_ledger_non_empty", bool(admitted_ids), sorted(admitted_ids), "admitted rows")
    ctx.add("rejected_ids_disjoint_from_admitted", admitted_ids.isdisjoint(rejected_ids), sorted(admitted_ids & rejected_ids), [])
    ctx.add("admitted_rows_have_required_evidence", all(_admitted_evidence(row) for row in admissions), [_row_id(row) for row in admissions if not _admitted_evidence(row)], [])
    trajectory_ids = {_row_id(row) for row in trajectories if _row_id(row)}
    ctx.add("admitted_ids_have_trajectories", admitted_ids.issubset(trajectory_ids), sorted(admitted_ids - trajectory_ids), [])

    ctx.add("contamination_report_passed", _passed(contamination) and not _truthy(contamination, "leakage_found", "unresolved_leakage"), _summary(contamination), "passed with no unresolved leakage")
    ctx.add("redaction_report_passed", _passed(redaction) and not _truthy(redaction, "secrets_found", "unredacted_sensitive_data"), _summary(redaction), "passed with no unredacted sensitive data")
    ctx.add("license_report_passed", _passed(license_report), _summary(license_report), "passed")
    ctx.add("dataset_identity_hash_present", bool(_pick(dataset_identity, "dataset_sha256", "identity_hash", "sha256")), _pick(dataset_identity, "dataset_sha256", "identity_hash", "sha256"), "dataset identity hash")
    ctx.add("dataset_deletion_lineage_present", bool(_pick(dataset_identity, "deletion_lineage", "deletion_key")), _pick(dataset_identity, "deletion_lineage", "deletion_key"), "deletion lineage")


def _validate_exports(ctx: _Context) -> None:
    sft = ctx.jsonl_rows.get("sft_export", [])
    action_sft = ctx.jsonl_rows.get("action_sft_export", [])
    dpo = ctx.jsonl_rows.get("dpo_export", [])
    export_manifest = _payload(ctx, "export_manifest")
    rejection_ids = {_row_id(row) for row in ctx.jsonl_rows.get("rejection_ledger", []) if _row_id(row)}
    unsafe_ids = {_row_id(row) for row in ctx.jsonl_rows.get("trajectories", []) if _row_id(row) and _unsafe(row)}

    export_ids = {_row_id(row) for row in [*sft, *action_sft, *dpo] if _row_id(row)}
    positive_sft_ids = {_row_id(row) for row in [*sft, *action_sft] if _row_id(row) and _positive(row)}
    ctx.add("exports_non_empty", bool(sft and action_sft and dpo), {"sft": len(sft), "action_sft": len(action_sft), "dpo": len(dpo)}, "all export views non-empty")
    ctx.add("rejected_ids_absent_from_exports", export_ids.isdisjoint(rejection_ids), sorted(export_ids & rejection_ids), [])
    ctx.add("unsafe_rows_not_positive_sft", unsafe_ids.isdisjoint(positive_sft_ids), sorted(unsafe_ids & positive_sft_ids), [])
    ctx.add("dpo_rows_have_preference_evidence", all(_preference_evidence(row) for row in dpo), [_row_id(row) for row in dpo if not _preference_evidence(row)], [])
    ctx.add("export_counts_replay", _export_count_ok(export_manifest, sft, action_sft, dpo), _pick(export_manifest, "counts", "row_counts"), {"sft": len(sft), "action_sft": len(action_sft), "dpo": len(dpo)})
    ctx.add("export_hashes_replay", _export_hashes_ok(ctx, export_manifest), _pick(export_manifest, "hashes", "artifact_fingerprints", "files"), "current file hashes")
    ctx.add("dataset_card_present", bool(ctx.text_payloads.get("dataset_card", "").strip()), "non-empty" if ctx.text_payloads.get("dataset_card", "").strip() else "empty", "non-empty")


def _validate_training(ctx: _Context) -> None:
    mlx = _payload(ctx, "mlx_qlora_plan")
    recipe = _payload(ctx, "recipe_space")
    training_plan = _payload(ctx, "agentic_training_plan")
    runtime = _payload(ctx, "runtime_preflight")
    preflight = _payload(ctx, "trainer_preflight")
    launch = _payload(ctx, "trainer_launch_check")
    archive = _payload(ctx, "trainer_archive")
    archive_check = _payload(ctx, "trainer_archive_check")
    consumer = _payload(ctx, "trainer_consumer_plan")
    rehearsal = _payload(ctx, "rehearsal_result")
    evidence = _payload(ctx, "evidence_bundle")
    candidate = _payload(ctx, "candidate_selection_contract")

    ctx.add("mlx_qlora_4bit_lora", _is_4bit(_pick(mlx, "quantization", "quantization_bits", "bits")) and _lora(_pick(mlx, "method", "adapter_type", "training_method")), _pick(mlx, "method", "adapter_type", "training_method", "quantization", "quantization_bits"), "4-bit QLoRA/LoRA")
    ctx.add("recipe_search_bounded", _truthy(recipe, "bounded", "bounded_search") or _number(_pick(recipe, "max_trials", "candidate_count")) is not None, _summary(recipe), "bounded recipe search")
    ctx.add("recipe_search_development_only", _truthy(recipe, "development_only") and not _truthy(recipe, "sealed_used"), _summary(recipe), "development only, sealed unused")
    ctx.add(
        "candidate_selection_development_only",
        _truthy(candidate, "development_only")
        and not _truthy(candidate, "sealed_used")
        and _truthy(candidate, "one_untouched_checkpoint"),
        _summary(candidate),
        "development-only selection and one untouched checkpoint",
    )
    ctx.add(
        "candidate_metrics_margins_bootstrap_frozen",
        bool(_pick(candidate, "primary_metric"))
        and _number(_pick(candidate, "safety_non_inferiority_margin")) is not None
        and _number(_pick(candidate, "per_domain_non_inferiority_margin")) is not None
        and bool(_pick(candidate, "bootstrap")),
        _summary(candidate),
        "primary metric, safety/per-domain margins, paired bootstrap",
    )
    ctx.add(
        "mlx_resume_stop_output_contract_present",
        bool(_pick(mlx, "resume")) and bool(_pick(mlx, "stop_conditions")) and bool(_pick(mlx, "output_contract")),
        _summary(mlx),
        "resume, stop conditions, adapter output contract",
    )
    ctx.add("runtime_preflight_local_only", _local_only(runtime), _summary(runtime), "local only")
    ctx.add("runtime_preflight_passed", _passed(runtime), _summary(runtime), "passed")
    ctx.add("trainer_preflight_passed", _passed(preflight), _summary(preflight), "passed")
    ctx.add("archive_check_passed", _passed(archive_check), _summary(archive_check), "passed")
    ctx.add("consumer_plan_present", bool(consumer), _summary(consumer), "present")
    ctx.add("consumer_plan_passed", _passed(consumer), _summary(consumer), "passed")
    ctx.add("trainer_launch_check_passed", _passed(launch), _summary(launch), "passed")
    ctx.add("trainer_archive_passed", _passed(archive), _summary(archive), "passed")
    ctx.add("evidence_bundle_hash_checked", _passed(evidence) or _truthy(evidence, "hash_checked", "hashes_verified"), _summary(evidence), "hash checked")
    ctx.add(
        "evidence_bundle_artifact_hashes_replay",
        _evidence_artifacts_replay(ctx, evidence),
        len(evidence.get("artifacts", [])) if isinstance(evidence.get("artifacts"), list) else 0,
        len(REQUIRED_ARTIFACTS) - 1,
    )
    ctx.add("rehearsal_tiny_non_sealed", _truthy(rehearsal, "tiny", "non_sealed") and not _truthy(rehearsal, "sealed_evaluation_started"), _summary(rehearsal), "tiny non-sealed rehearsal")
    ctx.add("rehearsal_passed", _passed(rehearsal), _summary(rehearsal), "passed")

    for role, payload in (
        ("training_plan", training_plan),
        ("runtime_preflight", runtime),
        ("trainer_preflight", preflight),
        ("trainer_launch_check", launch),
        ("trainer_archive", archive),
        ("trainer_archive_check", archive_check),
    ):
        ctx.add(f"{role}_did_not_start_training", not _truthy(payload, "training_started", "execution_started", "weights_updated", "sealed_evaluation_started", "promotion_applied"), _summary(payload), "no long training/sealed eval/promotion")

    command, command_tokens = _launch_command(launch)
    planned_command = mlx.get("command_argv")
    production_mode = _payload(ctx, "manifest").get("bundle_mode") == "production"
    lowered = command.lower()
    ctx.add("launch_command_present", bool(command), command, "approved local command")
    ctx.add("launch_command_mlx", "mlx" in lowered, command, "MLX command")
    ctx.add("launch_command_lora", "lora" in lowered or "qlora" in lowered, command, "LoRA/QLoRA")
    ctx.add(
        "launch_command_required_inputs_and_output",
        all(flag in command_tokens for flag in ("--train", "--model", "--data", "--adapter-path")),
        command_tokens,
        ["--train", "--model", "--data", "--adapter-path"],
    )
    if production_mode:
        ctx.add(
            "production_launch_command_frozen_in_qlora_plan",
            isinstance(planned_command, list)
            and bool(planned_command)
            and all(isinstance(value, str) and bool(value) for value in planned_command),
            planned_command,
            "non-empty command_argv string array",
        )
        ctx.add(
            "production_launch_command_matches_qlora_plan",
            command_tokens == planned_command,
            command_tokens,
            planned_command,
        )
        ctx.add(
            "production_launch_command_uses_local_bundle_bindings",
            _production_command_uses_local_bundle_bindings(command_tokens),
            command_tokens,
            {"--model": "model_input", "--data": "input_export", "--adapter-path": "adapter_output"},
        )
    ctx.add("launch_command_no_execute_flag", not _truthy(launch, "executed", "training_started", "weights_updated"), _summary(launch), "not executed")
    ctx.add("launch_command_no_push_cloud", not _has_forbidden_launch_tokens(command_tokens, command), command, "no push/cloud/network flags")


def _validate_no_sealed_or_cloud_leaks(ctx: _Context) -> None:
    sealed_hashes = _sealed_prompt_hashes(ctx)
    all_roles = sorted(ctx.payloads.keys() | ctx.jsonl_rows.keys() | ctx.text_payloads.keys())
    searchable_roles = [
        role
        for role in (
            "trajectories",
            "admission_ledger",
            "sft_export",
            "action_sft_export",
            "dpo_export",
            "export_manifest",
            "dataset_card",
            "candidate_selection_contract",
            "recipe_space",
        )
        if role in ctx.payloads or role in ctx.jsonl_rows or role in ctx.text_payloads
    ]
    leaks: list[str] = []
    if sealed_hashes:
        for role in searchable_roles:
            text = _role_text(ctx, role)
            for digest in sealed_hashes:
                if digest and digest in text:
                    leaks.append(f"{role}:{digest}")
    ctx.add("sealed_prompt_hashes_absent_from_training_artifacts", not leaks, leaks, [])

    cloud_hits = []
    private_path_hits = []
    credential_hits = []
    for role in all_roles:
        raw_text = _role_text(ctx, role)
        for pattern in PRIVATE_PATH_PATTERNS:
            if match := pattern.search(raw_text):
                private_path_hits.append(f"{role}:{match.group(0)}")
        for pattern in CREDENTIAL_PATTERNS:
            if pattern.search(raw_text):
                credential_hits.append(f"{role}:{pattern.pattern}")
        text = raw_text.lower()
        for word in FORBIDDEN_ENDPOINT_WORDS:
            if word in text:
                cloud_hits.append(f"{role}:{word}")
    ctx.add("hosted_cloud_endpoints_absent", not cloud_hits, cloud_hits, [])
    ctx.add("private_local_paths_absent", not private_path_hits, private_path_hits, [])
    ctx.add("credential_like_values_absent", not credential_hits, credential_hits, [])
    ctx.add("all_local_only_flags_hold", _all_local_only_flags_hold(ctx), "scanned artifacts", "no network/cloud/hosted true")


def _artifact_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = manifest.get("artifacts")
    if isinstance(artifacts, list):
        return [item for item in artifacts if isinstance(item, dict)]
    if isinstance(artifacts, dict):
        records = []
        for role, value in artifacts.items():
            if isinstance(value, dict):
                records.append({"role": role, **value})
        return records
    return []


def _safe_relative_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        return None
    return path


def _read_json(ctx: _Context, role: str, path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        ctx.add(f"json_parse:{role}", False, str(exc), "valid JSON")
        return None


def _read_jsonl(ctx: _Context, role: str, path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            ctx.add(f"jsonl_parse:{role}:{line_no}", False, exc.msg, "valid JSON object")
            continue
        if not isinstance(payload, dict):
            ctx.add(f"jsonl_object:{role}:{line_no}", False, type(payload).__name__, "object")
            continue
        rows.append(payload)
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _payload(ctx: _Context, role: str) -> dict[str, Any]:
    payload = ctx.payloads.get(role)
    return payload if isinstance(payload, dict) else {}


def _pick(payload: Any, *keys: str) -> Any:
    if not isinstance(payload, dict):
        return None
    for key in keys:
        if key in payload:
            return payload[key]
    for key, value in payload.items():
        if isinstance(value, dict):
            found = _pick(value, *keys)
            if found is not None:
                return found
    return None


def _truthy(payload: Any, *keys: str) -> bool:
    value = _pick(payload, *keys)
    return value is True or (isinstance(value, str) and value.lower() in {"true", "yes", "passed", "ready"})


def _falsey(payload: Any, *keys: str) -> bool:
    value = _pick(payload, *keys)
    return value is False or (isinstance(value, str) and value.lower() in {"false", "no", "disabled", "none"})


def _passed(payload: Any) -> bool:
    return _truthy(payload, "passed", "ok", "valid", "ready")


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _domains(payload: Any) -> set[str]:
    if not isinstance(payload, dict):
        return set()
    values = _pick(payload, "domains", "tau_domains", "evaluation_domains")
    if isinstance(values, list):
        return {str(item).lower() for item in values}
    if isinstance(values, dict):
        return {str(key).lower() for key in values}
    return set()


def _split_names(payload: dict[str, Any]) -> set[str]:
    splits = payload.get("splits")
    if isinstance(splits, dict):
        return {str(key).replace("_final", "").lower() for key in splits}
    if isinstance(splits, list):
        return {str(_pick(row, "name", "split")).replace("_final", "").lower() for row in splits if isinstance(row, dict)}
    return set()


def _split_hash(payload: dict[str, Any], name: str) -> str | None:
    names = {name, "sealed_final" if name == "sealed" else name}
    splits = payload.get("splits")
    candidates: list[Any] = []
    if isinstance(splits, dict):
        candidates.extend(splits.get(item) for item in names)
    elif isinstance(splits, list):
        candidates.extend(row for row in splits if isinstance(row, dict) and str(_pick(row, "name", "split")).lower() in names)
    for candidate in candidates:
        value = _pick(candidate, "sha256", "hash", "split_hash", "prompt_hash")
        if isinstance(value, str) and len(value) >= 16:
            return value
    return None


def _digests_from(payload: Any) -> list[str]:
    text = json.dumps(payload, sort_keys=True) if payload is not None else ""
    return re.findall(r"\b[0-9a-f]{32,64}\b", text)


def _model(payload: dict[str, Any], key: str) -> dict[str, Any]:
    model = payload.get(key)
    return model if isinstance(model, dict) else {}


def _comparators(payload: dict[str, Any]) -> list[dict[str, Any]]:
    value = _pick(payload, "comparators", "baseline_comparators", "eligible_comparators")
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _params_billion(model: dict[str, Any]) -> float | None:
    value = _pick(model, "parameters_billion", "parameter_billion", "params_billion", "billion_parameters")
    number = _number(value)
    if number is not None:
        return number
    params = _number(_pick(model, "parameters", "parameter_count"))
    if params is not None and params > 1_000_000:
        return params / 1_000_000_000
    return None


def _eligible_model(model: dict[str, Any]) -> bool:
    params = _params_billion(model)
    return params is not None and MIN_MODEL_BILLIONS <= params <= MAX_MODEL_BILLIONS


def _dense_model(model: dict[str, Any]) -> bool:
    arch = str(_pick(model, "architecture", "model_type") or "").lower()
    return "dense" in arch and "moe" not in arch


def _model_identity_complete(model: dict[str, Any]) -> bool:
    return bool(_pick(model, "name", "model_id")) and bool(_pick(model, "revision", "commit_sha")) and bool(_pick(model, "license", "license_id"))


def _model_summary(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": _pick(model, "name", "model_id"),
        "parameters_billion": _params_billion(model),
        "architecture": _pick(model, "architecture", "model_type"),
        "revision": _pick(model, "revision", "commit_sha"),
        "license": _pick(model, "license", "license_id"),
        "quantization": _pick(model, "quantization", "quantization_bits", "bits"),
    }


def _is_4bit(value: Any) -> bool:
    if isinstance(value, int):
        return value == 4
    text = str(value).lower()
    return "4" in text and "8" not in text and "16" not in text


def _lora(value: Any) -> bool:
    text = str(value).lower()
    return "lora" in text


def _local_only(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    if _truthy(payload, "local_only"):
        return True
    if _falsey(payload, "network", "allow_network", "cloud", "hosted"):
        return True
    boundary = payload.get("execution_boundary")
    if isinstance(boundary, dict):
        blocked = ("cloud_jobs_started", "model_downloads_started", "training_started", "weights_updated")
        if all(boundary.get(key) is False for key in blocked):
            return True
    return False


def _sealed_prompt_hashes(ctx: _Context) -> set[str]:
    sealed = _payload(ctx, "sealed_manifest")
    values = _pick(sealed, "prompt_hashes", "sealed_prompt_hashes", "hashes")
    if isinstance(values, list):
        return {str(item) for item in values if isinstance(item, str)}
    if isinstance(values, dict):
        return {str(item) for item in values.values() if isinstance(item, str)}
    return set(_digests_from(sealed))


def _row_id(row: dict[str, Any]) -> str | None:
    value = _pick(row, "id", "row_id", "trajectory_id", "episode_id", "source_id")
    return str(value) if value is not None else None


def _row_domains(rows: list[dict[str, Any]]) -> set[str]:
    domains = set()
    for row in rows:
        value = _pick(row, "domain", "task_domain")
        if isinstance(value, str):
            domains.add(value.lower())
        elif isinstance(value, list):
            domains.update(str(item).lower() for item in value)
    return domains


def _behaviors(rows: list[dict[str, Any]]) -> set[str]:
    behaviors = set()
    for row in rows:
        values = _pick(row, "behavior", "behavior_tag", "behavior_tags", "category")
        if isinstance(values, str):
            behaviors.add(values.lower())
        elif isinstance(values, list):
            behaviors.update(str(item).lower() for item in values)
    return behaviors


def _behavior_coverage(rows: list[dict[str, Any]]) -> bool:
    behaviors = " ".join(sorted(_behaviors(rows)))
    required_groups = (
        ("success",),
        ("correction",),
        ("clarification", "refusal"),
        ("recovery",),
        ("policy", "policy_failure"),
        ("harmful", "unnecessary_mutation"),
        ("hallucinated_tool", "hallucinated tools"),
        ("premature_completion", "premature completion"),
    )
    return all(any(item in behaviors for item in group) for group in required_groups)


def _token_shares(balance: dict[str, Any], trajectories: list[dict[str, Any]]) -> dict[str, float]:
    shares = _pick(balance, "token_share_by_domain", "domain_token_share", "shares")
    if isinstance(shares, dict):
        return {str(key).lower(): float(value) for key, value in shares.items() if _number(value) is not None}
    totals: dict[str, float] = {}
    for row in trajectories:
        domain = str(_pick(row, "domain", "task_domain") or "").lower()
        tokens = _number(_pick(row, "tokens", "token_count")) or 1.0
        if domain:
            totals[domain] = totals.get(domain, 0.0) + tokens
    total = sum(totals.values())
    return {key: value / total for key, value in totals.items()} if total else {}


def _admitted_evidence(row: dict[str, Any]) -> bool:
    evidence = _pick(row, "evidence") if isinstance(_pick(row, "evidence"), dict) else row
    return all(bool(_pick(evidence, key)) for key in ("lineage", "state_transition", "executable", "safety", "reviewer"))


def _unsafe(row: dict[str, Any]) -> bool:
    return _truthy(row, "unsafe", "policy_violation", "harmful_mutation") or str(_pick(row, "safety_label", "label") or "").lower() in {"unsafe", "policy_failure", "harmful"}


def _positive(row: dict[str, Any]) -> bool:
    value = str(_pick(row, "label", "target", "polarity", "preference") or "positive").lower()
    return value in {"positive", "chosen", "accepted", "safe_success", "success"}


def _preference_evidence(row: dict[str, Any]) -> bool:
    if (
        bool(_pick(row, "preference_evidence", "preference_judgment", "chosen_evidence"))
        and bool(_pick(row, "chosen", "chosen_episode_id"))
        and bool(_pick(row, "rejected", "rejected_episode_id"))
    ):
        return True
    provenance = _pick(row, "label_provenance")
    score_gap = _number(_pick(row, "score_gap", "preference_score_gap", "margin"))
    return (
        bool(provenance)
        and bool(_pick(row, "chosen", "chosen_episode_id"))
        and bool(_pick(row, "rejected", "rejected_episode_id"))
        and (score_gap is None or score_gap > 0)
        and bool(_pick(row, "reason", "rationale"))
    )


def _export_count_ok(manifest: dict[str, Any], sft: list[dict[str, Any]], action_sft: list[dict[str, Any]], dpo: list[dict[str, Any]]) -> bool:
    counts = _pick(manifest, "counts", "row_counts")
    expected = {"sft": len(sft), "action_sft": len(action_sft), "dpo": len(dpo)}
    if isinstance(counts, dict) and all(_number(counts.get(key)) == value for key, value in expected.items()):
        return True
    native = {
        "sft": _number(manifest.get("sft_count")),
        "action_sft": _number(manifest.get("action_sft_count")),
        "dpo": _number(manifest.get("dpo_count")),
    }
    return all(native[key] == value for key, value in expected.items())


def _export_hashes_ok(ctx: _Context, manifest: dict[str, Any]) -> bool:
    text = json.dumps(manifest, sort_keys=True)
    for role in ("sft_export", "action_sft_export", "dpo_export"):
        path = ctx.artifacts.get(role)
        if not path or not path.is_file() or _sha256(path) not in text:
            return False
    return True


def _evidence_artifacts_replay(ctx: _Context, evidence: dict[str, Any]) -> bool:
    rows = evidence.get("artifacts")
    if not isinstance(rows, list):
        return False
    expected_roles = {role for role, _ in REQUIRED_ARTIFACTS if role != "evidence_bundle"}
    records = {
        str(row.get("role")): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("role"), str)
    }
    if set(records) != expected_roles:
        return False
    for role in sorted(expected_roles):
        row = records[role]
        safe_path = _safe_relative_path(row.get("path"))
        expected_path = Path(REQUIRED_ARTIFACT_MAP[role])
        if safe_path != expected_path:
            return False
        path = ctx.bundle / expected_path
        if not path.is_file() or row.get("size") != path.stat().st_size or row.get("sha256") != _sha256(path):
            return False
    return True


def _has_forbidden_launch_tokens(tokens: list[str], command: str) -> bool:
    lowered_tokens = {token.lower() for token in tokens}
    if {
        "--push-to-hub",
        "--upload",
        "--allow-network",
        "--cloud",
        "--hosted",
        "--trust-remote-code",
    } & lowered_tokens:
        return True
    lowered = command.lower()
    return "://" in lowered or any(word in lowered for word in FORBIDDEN_ENDPOINT_WORDS)


def _production_command_uses_local_bundle_bindings(tokens: list[str]) -> bool:
    if tokens[:4] == ["python", "-m", "mlx_lm", "lora"]:
        valid_prefix = True
    else:
        valid_prefix = tokens[:3] == ["python", "-m", "mlx_lm.lora"]
    if not valid_prefix or "--train" not in tokens:
        return False
    for flag, expected in (
        ("--model", "model_input"),
        ("--data", "input_export"),
        ("--adapter-path", "adapter_output"),
    ):
        values = [tokens[index + 1] for index, token in enumerate(tokens[:-1]) if token == flag]
        if values != [expected]:
            return False
    return not _has_forbidden_launch_tokens(tokens, " ".join(tokens))


def _launch_command(launch: dict[str, Any]) -> tuple[str, list[str]]:
    value = _pick(launch, "launch_command", "approved_command", "command")
    if isinstance(value, dict):
        shell = value.get("shell")
        if isinstance(shell, str) and shell.strip():
            return shell, shlex.split(shell)
        argv = value.get("argv")
        if isinstance(argv, list) and all(isinstance(item, str) for item in argv):
            return " ".join(shlex.quote(item) for item in argv), list(argv)
        return json.dumps(value, sort_keys=True), []
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return " ".join(shlex.quote(item) for item in value), list(value)
    command = str(value or "")
    try:
        return command, shlex.split(command) if command else []
    except ValueError:
        return command, []


def _role_text(ctx: _Context, role: str) -> str:
    if role in ctx.payloads:
        return json.dumps(ctx.payloads[role], sort_keys=True)
    if role in ctx.jsonl_rows:
        return "\n".join(json.dumps(row, sort_keys=True) for row in ctx.jsonl_rows[role])
    return ctx.text_payloads.get(role, "")


def _all_local_only_flags_hold(ctx: _Context) -> bool:
    text = "\n".join(_role_text(ctx, role).lower() for role in sorted(ctx.payloads.keys() | ctx.jsonl_rows.keys() | ctx.text_payloads.keys()))
    forbidden_true = (
        '"allow_network": true',
        '"network": true',
        '"cloud": true',
        '"hosted": true',
        '"training_started": true',
        '"weights_updated": true',
        '"sealed_evaluation_started": true',
        '"promotion_applied": true',
    )
    return not any(item in text for item in forbidden_true)


def _summary(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    return {key: payload.get(key) for key in ("schema_version", "passed", "ready", "status", "training_started", "sealed_evaluation_started", "promotion_applied") if key in payload}
