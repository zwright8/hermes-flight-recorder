"""Fail-closed cloud training provider contracts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

from .schema_registry import SchemaRegistryError, check_schema_file

CLOUD_TRAINING_PROVIDER_REGISTRY_SCHEMA_VERSION = "hfr.cloud_training_provider_registry.v1"
CLOUD_TRAINING_PREFLIGHT_SCHEMA_VERSION = "hfr.cloud_training_preflight.v1"
CLOUD_TRAINING_ARTIFACT_MANIFEST_SCHEMA_VERSION = "hfr.cloud_training_artifact_manifest.v1"
CLOUD_TRAINING_LAUNCH_PLAN_SCHEMA_VERSION = "hfr.cloud_training_launch_plan.v1"
CLOUD_TRAINING_LAUNCH_RECEIPT_SCHEMA_VERSION = "hfr.cloud_training_launch_receipt.v1"
CLOUD_TRAINING_STATUS_RECEIPT_SCHEMA_VERSION = "hfr.cloud_training_status_receipt.v1"
CLOUD_TRAINING_PROVIDER_ADAPTER_CONTRACT_VERSION = "hfr.cloud_training_provider_adapter.v1"

PROVIDER_ADAPTER_RECEIPT_TYPES: tuple[str, ...] = (
    CLOUD_TRAINING_PROVIDER_REGISTRY_SCHEMA_VERSION,
    CLOUD_TRAINING_PREFLIGHT_SCHEMA_VERSION,
    CLOUD_TRAINING_ARTIFACT_MANIFEST_SCHEMA_VERSION,
    CLOUD_TRAINING_LAUNCH_PLAN_SCHEMA_VERSION,
    CLOUD_TRAINING_LAUNCH_RECEIPT_SCHEMA_VERSION,
    CLOUD_TRAINING_STATUS_RECEIPT_SCHEMA_VERSION,
)

PROVIDERS: dict[str, dict[str, Any]] = {
    "huggingface_jobs": {
        "display_name": "Hugging Face Jobs / AutoTrain",
        "credential_env_vars": ["HF_TOKEN"],
        "regions": ["provider_default"],
        "gpu_classes": ["cpu", "t4", "l4", "a10g", "a100"],
        "job_modes": ["sft", "action_sft", "dpo", "sft_then_dpo"],
        "artifact_protocols": ["hub", "http", "archive"],
        "live_status": "preflight_only",
    },
    "modal": {
        "display_name": "Modal",
        "credential_env_vars": ["MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"],
        "regions": ["provider_default"],
        "gpu_classes": ["t4", "l4", "a10g", "a100", "h100"],
        "job_modes": ["sft", "action_sft", "dpo", "reward_model", "process_rewards"],
        "artifact_protocols": ["archive", "volume", "object_store"],
        "live_status": "preflight_only",
    },
    "runpod": {
        "display_name": "RunPod",
        "credential_env_vars": ["RUNPOD_API_KEY"],
        "regions": ["provider_default"],
        "gpu_classes": ["rtx_4090", "a40", "a100", "h100"],
        "job_modes": ["sft", "action_sft", "dpo"],
        "artifact_protocols": ["archive", "object_store"],
        "live_status": "preflight_only",
    },
    "lambda_labs": {
        "display_name": "Lambda Labs",
        "credential_env_vars": ["LAMBDA_API_KEY"],
        "regions": ["us"],
        "gpu_classes": ["a10", "a100", "h100"],
        "job_modes": ["sft", "action_sft", "dpo"],
        "artifact_protocols": ["archive", "object_store"],
        "live_status": "preflight_only",
    },
    "coreweave": {
        "display_name": "CoreWeave",
        "credential_env_vars": ["COREWEAVE_API_TOKEN"],
        "regions": ["us", "eu"],
        "gpu_classes": ["a40", "a100", "h100"],
        "job_modes": ["sft", "action_sft", "dpo", "reward_model", "process_rewards"],
        "artifact_protocols": ["archive", "s3_compatible"],
        "live_status": "preflight_only",
    },
    "together": {
        "display_name": "Together AI",
        "credential_env_vars": ["TOGETHER_API_KEY"],
        "regions": ["provider_default"],
        "gpu_classes": ["provider_managed"],
        "job_modes": ["sft", "dpo"],
        "artifact_protocols": ["http", "provider_managed"],
        "live_status": "preflight_only",
    },
    "fireworks": {
        "display_name": "Fireworks AI",
        "credential_env_vars": ["FIREWORKS_API_KEY"],
        "regions": ["provider_default"],
        "gpu_classes": ["provider_managed"],
        "job_modes": ["sft"],
        "artifact_protocols": ["http", "provider_managed"],
        "live_status": "preflight_only",
    },
    "replicate": {
        "display_name": "Replicate",
        "credential_env_vars": ["REPLICATE_API_TOKEN"],
        "regions": ["provider_default"],
        "gpu_classes": ["provider_managed"],
        "job_modes": ["sft"],
        "artifact_protocols": ["http", "provider_managed"],
        "live_status": "preflight_only",
    },
    "aws_sagemaker": {
        "display_name": "AWS SageMaker",
        "credential_env_vars": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"],
        "regions": ["us-east-1", "us-west-2", "eu-west-1"],
        "gpu_classes": ["ml.g5", "ml.p4d", "ml.p5"],
        "job_modes": ["sft", "action_sft", "dpo", "reward_model", "process_rewards"],
        "artifact_protocols": ["s3", "ecr", "archive"],
        "live_status": "preflight_only",
    },
    "gcp_vertex_ai": {
        "display_name": "GCP Vertex AI",
        "credential_env_vars": ["GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_PROJECT"],
        "regions": ["us-central1", "us-east4", "europe-west4"],
        "gpu_classes": ["nvidia-tesla-t4", "nvidia-l4", "nvidia-tesla-a100", "nvidia-h100"],
        "job_modes": ["sft", "action_sft", "dpo", "reward_model"],
        "artifact_protocols": ["gcs", "artifact_registry", "archive"],
        "live_status": "preflight_only",
    },
    "azure_ml": {
        "display_name": "Azure ML",
        "credential_env_vars": ["AZURE_CLIENT_ID", "AZURE_TENANT_ID", "AZURE_CLIENT_SECRET"],
        "regions": ["eastus", "westus3", "westeurope"],
        "gpu_classes": ["standard_nc", "standard_nd"],
        "job_modes": ["sft", "action_sft", "dpo", "reward_model"],
        "artifact_protocols": ["azure_blob", "acr", "archive"],
        "live_status": "preflight_only",
    },
    "databricks_mosaic": {
        "display_name": "Databricks / Mosaic AI",
        "credential_env_vars": ["DATABRICKS_HOST", "DATABRICKS_TOKEN"],
        "regions": ["workspace_default"],
        "gpu_classes": ["workspace_cluster"],
        "job_modes": ["sft", "dpo", "reward_model"],
        "artifact_protocols": ["dbfs", "mlflow", "unity_catalog"],
        "live_status": "preflight_only",
    },
    "nvidia_dgx_cloud": {
        "display_name": "NVIDIA DGX Cloud",
        "credential_env_vars": ["NVIDIA_API_KEY"],
        "regions": ["provider_default"],
        "gpu_classes": ["a100", "h100", "gb200"],
        "job_modes": ["sft", "action_sft", "dpo", "reward_model", "process_rewards", "grpo"],
        "artifact_protocols": ["archive", "object_store"],
        "live_status": "preflight_only",
    },
    "brev": {
        "display_name": "NVIDIA Brev",
        "credential_env_vars": ["BREV_API_KEY"],
        "regions": ["provider_default"],
        "gpu_classes": ["l4", "a10g", "a100", "h100"],
        "job_modes": ["sft", "action_sft", "dpo", "reward_model"],
        "artifact_protocols": ["archive", "object_store"],
        "live_status": "preflight_only",
    },
}

PROVIDER_CLIENT_MODULES: dict[str, list[str]] = {
    "huggingface_jobs": ["huggingface_hub"],
    "modal": ["modal"],
    "runpod": ["runpod"],
    "lambda_labs": ["requests"],
    "coreweave": ["kubernetes"],
    "together": ["together"],
    "fireworks": ["fireworks"],
    "replicate": ["replicate"],
    "aws_sagemaker": ["boto3", "sagemaker"],
    "gcp_vertex_ai": ["google.cloud.aiplatform"],
    "azure_ml": ["azure.ai.ml"],
    "databricks_mosaic": ["databricks.sdk", "mlflow"],
    "nvidia_dgx_cloud": ["requests"],
    "brev": ["requests"],
}


class CloudTrainingError(ValueError):
    """Raised when cloud training contracts cannot be built."""


def provider_choices() -> list[str]:
    """Return stable provider ids for CLI choices."""
    return sorted(PROVIDERS)


def build_cloud_training_provider_registry(provider_ids: list[str] | None = None, *, created_at: str | None = None) -> dict[str, Any]:
    """Build the provider-neutral capability registry."""
    selected = _select_providers(provider_ids)
    providers = [_provider_record(provider_id) for provider_id in selected]
    return {
        "schema_version": CLOUD_TRAINING_PROVIDER_REGISTRY_SCHEMA_VERSION,
        "created_at": created_at or _now(),
        "provider_count": len(providers),
        "providers": providers,
        "execution_boundary": _boundary(),
        "notes": [
            "Provider records describe contracts and capability probes only; they do not create cloud jobs.",
            "Credential values are never recorded, only environment variable names and presence checks in preflight artifacts.",
        ],
    }


def build_cloud_training_preflight(
    *,
    provider_id: str,
    agentic_training_plan_path: str | Path,
    trainer_preflight_path: str | Path | None = None,
    trainer_launch_check_path: str | Path | None = None,
    region: str | None = None,
    gpu_class: str | None = None,
    max_cost_usd: float | None = None,
    live_preflight: bool = False,
    live_requested: bool = False,
    allow_live: bool = False,
    preserve_paths: bool = False,
    output_base_dir: str | Path | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a fail-closed readiness preflight for a cloud trainer handoff."""
    provider = _provider(provider_id)
    checks: list[dict[str, Any]] = []
    display_base_dir = Path(output_base_dir) if output_base_dir is not None else None
    plan_ref = _json_artifact_ref(
        "agentic_training_plan",
        Path(agentic_training_plan_path),
        "agentic_training_plan",
        preserve_paths,
        display_base_dir,
    )
    trainer_preflight_ref = (
        _json_artifact_ref("trainer_preflight", Path(trainer_preflight_path), "trainer_preflight", preserve_paths, display_base_dir)
        if trainer_preflight_path
        else _missing_ref("trainer_preflight")
    )
    trainer_launch_ref = (
        _json_artifact_ref("trainer_launch_check", Path(trainer_launch_check_path), "trainer_launch_check", preserve_paths, display_base_dir)
        if trainer_launch_check_path
        else _missing_ref("trainer_launch_check")
    )

    _add_check(checks, "agentic_training_plan_ready", _artifact_ready(plan_ref), {"artifact": plan_ref}, {"schema": "agentic_training_plan", "passed": True})
    _add_check(
        checks,
        "trainer_preflight_ready",
        _artifact_ready(trainer_preflight_ref),
        {"artifact": trainer_preflight_ref},
        {"schema": "trainer_preflight", "passed": True},
    )
    _add_check(
        checks,
        "trainer_launch_check_ready",
        _artifact_ready(trainer_launch_ref),
        {"artifact": trainer_launch_ref},
        {"schema": "trainer_launch_check", "passed": True},
    )
    _add_check(
        checks,
        "region_allowed",
        region is None or region in provider["regions"],
        {"region": region, "allowed_regions": provider["regions"]},
        {"region_in_allowed_regions": True},
    )
    _add_check(
        checks,
        "gpu_class_allowed",
        gpu_class is None or gpu_class in provider["gpu_classes"],
        {"gpu_class": gpu_class, "allowed_gpu_classes": provider["gpu_classes"]},
        {"gpu_class_in_allowed_gpu_classes": True},
    )
    _add_check(
        checks,
        "cost_limit_declared",
        max_cost_usd is not None and max_cost_usd >= 0,
        {"max_cost_usd": max_cost_usd},
        {"max_cost_usd": "non_negative_number"},
    )
    credential_checks = _credential_checks(provider)
    live_preflight_probe = _live_preflight_probe(provider_id, provider, credential_checks, live_preflight)
    _add_check(
        checks,
        "live_launch_explicitly_enabled",
        not live_requested or allow_live,
        {"live_requested": live_requested, "allow_live": allow_live},
        {"allow_live": True},
    )
    _add_check(
        checks,
        "live_credentials_available_when_requested",
        not live_requested or all(check["present"] for check in credential_checks),
        {"live_requested": live_requested, "credential_checks": credential_checks},
        {"all_provider_credentials_present": True},
    )
    _add_check(
        checks,
        "live_preflight_credentials_available_when_requested",
        not live_preflight or all(check["present"] for check in credential_checks),
        {"live_preflight": live_preflight, "credential_checks": credential_checks},
        {"all_provider_credentials_present": True},
    )
    _add_check(
        checks,
        "live_preflight_client_dependencies_available_when_requested",
        not live_preflight or all(check["available"] for check in live_preflight_probe["client_dependency_checks"]),
        {"live_preflight": live_preflight, "client_dependency_checks": live_preflight_probe["client_dependency_checks"]},
        {"all_client_dependencies_available": True},
    )
    _add_check(
        checks,
        "flight_recorder_did_not_launch_cloud_job",
        True,
        {"cloud_job_started": False, "provider_api_called": False},
        {"cloud_job_started": False, "provider_api_called": False},
    )

    failed = [check for check in checks if not check["passed"]]
    passed = not failed
    return {
        "schema_version": CLOUD_TRAINING_PREFLIGHT_SCHEMA_VERSION,
        "created_at": created_at or _now(),
        "provider": _provider_record(provider_id),
        "passed": passed,
        "readiness": "ready_for_dry_run_launch_plan" if passed else "blocked",
        "recommendation": "build_dry_run_launch_plan" if passed else "block_cloud_training_launch",
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "checks": checks,
        "blocked_reasons": [check["summary"] for check in failed],
        "constraints": _constraints(region=region, gpu_class=gpu_class, max_cost_usd=max_cost_usd),
        "credential_checks": credential_checks,
        "live_preflight": live_preflight_probe,
        "source_artifacts": {
            "agentic_training_plan": plan_ref,
            "trainer_preflight": trainer_preflight_ref,
            "trainer_launch_check": trainer_launch_ref,
        },
        "execution_boundary": _boundary(live_requested=live_requested, allow_live=allow_live, live_preflight=live_preflight),
        "handoff_contract": _handoff_contract(),
        "notes": [
            "Preflight checks local receipts, constraints, and credential variable presence only.",
            "Live preflight probes use importlib metadata and environment-variable presence only; no provider SDK is imported and no provider API call is made.",
        ],
    }


def build_cloud_training_artifact_manifest(
    *,
    provider_id: str,
    upload_paths: list[str | Path] | None = None,
    expected_downloads: list[str] | None = None,
    preserve_paths: bool = False,
    output_base_dir: str | Path | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build upload/download artifact manifest for a cloud trainer handoff."""
    provider = _provider(provider_id)
    display_base_dir = Path(output_base_dir) if output_base_dir is not None else None
    uploads = [_file_ref("upload", Path(path), preserve_paths, display_base_dir) for path in upload_paths or []]
    downloads = [{"role": "download", "path": path, "exists": False, "sha256": None, "size_bytes": None} for path in expected_downloads or []]
    transfer_plan = _artifact_transfer_plan(provider, uploads, downloads)
    checks: list[dict[str, Any]] = []
    _add_check(checks, "upload_artifacts_declared", bool(uploads), {"upload_count": len(uploads)}, {"min_upload_count": 1})
    _add_check(checks, "upload_artifacts_exist", all(item["exists"] for item in uploads), {"uploads": uploads}, {"all_uploads_exist": True})
    _add_check(checks, "download_artifacts_not_assumed", True, {"downloads_exist": False}, {"downloads_exist": False})
    _add_check(
        checks,
        "transfer_plan_matches_artifacts",
        transfer_plan["upload_count"] == len(uploads) and transfer_plan["expected_download_count"] == len(downloads),
        {"transfer_plan": transfer_plan},
        {"upload_count": len(uploads), "expected_download_count": len(downloads)},
    )
    _add_check(
        checks,
        "flight_recorder_did_not_transfer_artifacts",
        not transfer_plan["flight_recorder_uploaded_artifacts"]
        and not transfer_plan["flight_recorder_downloaded_artifacts"]
        and not transfer_plan["provider_api_called"],
        {
            "flight_recorder_uploaded_artifacts": transfer_plan["flight_recorder_uploaded_artifacts"],
            "flight_recorder_downloaded_artifacts": transfer_plan["flight_recorder_downloaded_artifacts"],
            "provider_api_called": transfer_plan["provider_api_called"],
        },
        {
            "flight_recorder_uploaded_artifacts": False,
            "flight_recorder_downloaded_artifacts": False,
            "provider_api_called": False,
        },
    )
    failed = [check for check in checks if not check["passed"]]
    return {
        "schema_version": CLOUD_TRAINING_ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "created_at": created_at or _now(),
        "provider": _provider_record(provider_id),
        "passed": not failed,
        "readiness": "ready" if not failed else "blocked",
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "checks": checks,
        "blocked_reasons": [check["summary"] for check in failed],
        "upload_artifacts": uploads,
        "expected_download_artifacts": downloads,
        "artifact_protocols": list(provider["artifact_protocols"]),
        "transfer_plan": transfer_plan,
        "execution_boundary": _boundary(),
    }


def build_cloud_training_launch_plan(
    *,
    preflight_path: str | Path,
    artifact_manifest_path: str | Path | None = None,
    preserve_paths: bool = False,
    output_base_dir: str | Path | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a dry-run launch plan from a cloud-training preflight."""
    display_base_dir = Path(output_base_dir) if output_base_dir is not None else None
    preflight_source_path = Path(preflight_path)
    artifact_manifest_source_path = Path(artifact_manifest_path) if artifact_manifest_path else None
    preflight_ref = _json_artifact_ref(
        "cloud_training_preflight",
        preflight_source_path,
        "cloud_training_preflight",
        preserve_paths,
        display_base_dir,
    )
    artifact_ref = (
        _json_artifact_ref(
            "cloud_training_artifact_manifest",
            artifact_manifest_source_path,
            "cloud_training_artifact_manifest",
            preserve_paths,
            display_base_dir,
        )
        if artifact_manifest_source_path
        else _missing_ref("cloud_training_artifact_manifest")
    )
    preflight = _read_json(preflight_source_path) if preflight_ref["exists"] else {}
    artifact_manifest = _read_json(artifact_manifest_source_path) if artifact_ref["exists"] and artifact_manifest_source_path else {}
    preflight_provider_id = _payload_provider_id(preflight)
    artifact_manifest_provider_id = _payload_provider_id(artifact_manifest)
    provider_chain = _launch_plan_provider_chain(
        preflight_provider_id,
        artifact_manifest_provider_id,
        artifact_manifest_required=True,
    )
    checks: list[dict[str, Any]] = []
    _add_check(checks, "preflight_ready", _artifact_ready(preflight_ref), {"artifact": preflight_ref}, {"schema": "cloud_training_preflight", "passed": True})
    _add_check(
        checks,
        "artifact_manifest_ready",
        _artifact_ready(artifact_ref),
        {"artifact": artifact_ref},
        {"schema": "cloud_training_artifact_manifest", "passed": True},
    )
    _add_check(
        checks,
        "provider_chain_consistent",
        provider_chain["provider_consistent"],
        {"provider_chain": provider_chain},
        {"provider_consistent": True},
    )
    _add_check(checks, "dry_run_launch_only", True, {"dry_run": True, "cloud_job_started": False}, {"dry_run": True})
    failed = [check for check in checks if not check["passed"]]
    provider_id = preflight_provider_id
    provider = _provider_record(provider_id) if provider_id in PROVIDERS else {"id": provider_id, "display_name": provider_id}
    return {
        "schema_version": CLOUD_TRAINING_LAUNCH_PLAN_SCHEMA_VERSION,
        "created_at": created_at or _now(),
        "provider": provider,
        "passed": not failed,
        "readiness": "ready_for_dry_run_launch" if not failed else "blocked",
        "recommendation": "emit_dry_run_launch_receipt" if not failed else "block_launch_receipt",
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "checks": checks,
        "blocked_reasons": [check["summary"] for check in failed],
        "source_artifacts": {"preflight": preflight_ref, "artifact_manifest": artifact_ref},
        "provider_chain": provider_chain,
        "launch": {
            "mode": "dry_run",
            "live_launch_supported": False,
            "provider_api_call_planned": False,
            "command": ["<external-cloud-training-runner>", "--provider", provider_id or "<provider>", "--dry-run"],
        },
        "execution_boundary": _boundary(),
        "handoff_contract": _handoff_contract(),
    }


def build_cloud_training_launch_receipt(
    *,
    launch_plan_path: str | Path,
    live: bool = False,
    preserve_paths: bool = False,
    output_base_dir: str | Path | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a dry-run launch receipt or blocked live-launch receipt."""
    display_base_dir = Path(output_base_dir) if output_base_dir is not None else None
    plan_ref = _json_artifact_ref("cloud_training_launch_plan", Path(launch_plan_path), "cloud_training_launch_plan", preserve_paths, display_base_dir)
    checks: list[dict[str, Any]] = []
    _add_check(checks, "launch_plan_ready", _artifact_ready(plan_ref), {"artifact": plan_ref}, {"schema": "cloud_training_launch_plan", "passed": True})
    _add_check(checks, "live_launch_not_implemented", not live, {"live": live}, {"live": False})
    _add_check(checks, "cloud_job_not_started", True, {"cloud_job_started": False, "provider_api_called": False}, {"cloud_job_started": False})
    failed = [check for check in checks if not check["passed"]]
    return {
        "schema_version": CLOUD_TRAINING_LAUNCH_RECEIPT_SCHEMA_VERSION,
        "created_at": created_at or _now(),
        "passed": not failed,
        "readiness": "dry_run_recorded" if not failed else "blocked",
        "recommendation": "safe_to_archive_dry_run_receipt" if not failed else "block_live_cloud_launch",
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "checks": checks,
        "blocked_reasons": [check["summary"] for check in failed],
        "source_artifacts": {"launch_plan": plan_ref},
        "launch": {
            "mode": "live" if live else "dry_run",
            "cloud_job_started": False,
            "provider_job_id": None,
            "provider_api_called": False,
            "cost_incurred_usd": 0,
        },
        "execution_boundary": _boundary(live_requested=live, allow_live=False),
    }


def build_cloud_training_status_receipt(
    *,
    launch_receipt_path: str | Path,
    cancel_requested: bool = False,
    preserve_paths: bool = False,
    output_base_dir: str | Path | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a dry-run status/cancellation receipt for a cloud training job."""
    display_base_dir = Path(output_base_dir) if output_base_dir is not None else None
    launch_ref = _json_artifact_ref(
        "cloud_training_launch_receipt",
        Path(launch_receipt_path),
        "cloud_training_launch_receipt",
        preserve_paths,
        display_base_dir,
    )
    checks: list[dict[str, Any]] = []
    _add_check(checks, "launch_receipt_readable", launch_ref["exists"], {"artifact": launch_ref}, {"exists": True})
    _add_check(checks, "status_check_did_not_call_provider", True, {"provider_api_called": False}, {"provider_api_called": False})
    _add_check(checks, "cancel_is_dry_run", True, {"cancel_requested": cancel_requested, "provider_cancel_called": False}, {"provider_cancel_called": False})
    failed = [check for check in checks if not check["passed"]]
    return {
        "schema_version": CLOUD_TRAINING_STATUS_RECEIPT_SCHEMA_VERSION,
        "created_at": created_at or _now(),
        "passed": not failed,
        "readiness": "status_recorded" if not failed else "blocked",
        "recommendation": "archive_status_receipt" if not failed else "inspect_launch_receipt",
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "checks": checks,
        "blocked_reasons": [check["summary"] for check in failed],
        "source_artifacts": {"launch_receipt": launch_ref},
        "status": {
            "provider_status": "not_started",
            "terminal": True,
            "cancel_requested": cancel_requested,
            "provider_cancel_called": False,
            "provider_api_called": False,
            "cost_incurred_usd": 0,
        },
        "execution_boundary": _boundary(),
    }


def write_cloud_training_artifact(path: str | Path, payload: dict[str, Any]) -> None:
    """Write stable JSON for any cloud-training contract."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _select_providers(provider_ids: list[str] | None) -> list[str]:
    selected = sorted(set(provider_ids or PROVIDERS))
    unknown = sorted(set(selected) - set(PROVIDERS))
    if unknown:
        raise CloudTrainingError(f"Unknown cloud training provider(s): {', '.join(unknown)}")
    if not selected:
        raise CloudTrainingError("At least one provider must be selected")
    return selected


def _provider(provider_id: str) -> dict[str, Any]:
    try:
        return PROVIDERS[provider_id]
    except KeyError as exc:
        raise CloudTrainingError(f"Unknown cloud training provider: {provider_id}") from exc


def _provider_record(provider_id: str) -> dict[str, Any]:
    provider = _provider(provider_id)
    return {
        "id": provider_id,
        "display_name": provider["display_name"],
        "credential_env_vars": list(provider["credential_env_vars"]),
        "regions": list(provider["regions"]),
        "gpu_classes": list(provider["gpu_classes"]),
        "job_modes": list(provider["job_modes"]),
        "artifact_protocols": list(provider["artifact_protocols"]),
        "client_import_names": list(PROVIDER_CLIENT_MODULES.get(provider_id, ())),
        "live_status": provider["live_status"],
        "default_live_execution_allowed": False,
        "adapter_contract": _provider_adapter_contract(provider_id),
    }


def _provider_adapter_contract(provider_id: str) -> dict[str, Any]:
    return {
        "schema_version": CLOUD_TRAINING_PROVIDER_ADAPTER_CONTRACT_VERSION,
        "adapter_id": f"cloud_training.{provider_id}.fail_closed.v1",
        "provider_id": provider_id,
        "receipt_types": list(PROVIDER_ADAPTER_RECEIPT_TYPES),
        "dry_run_transport": "mock_receipts",
        "live_preflight_transport": "metadata_only",
        "live_preflight_supported": True,
        "live_launch_supported": False,
        "status_cancel_receipts_supported": True,
        "provider_api_called_by_flight_recorder": False,
        "client_modules_imported_by_flight_recorder": False,
        "credential_values_recorded": False,
        "cloud_cost_incurred_usd": 0,
        "model_downloads_started": False,
        "weights_updated_by_flight_recorder": False,
        "requires_explicit_live_opt_in": True,
        "requires_environment_credentials_for_live": True,
        "requires_cost_limit": True,
        "requires_region_and_gpu_constraints": True,
    }


def _credential_checks(provider: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "env_var": name,
            "present": name in os.environ and bool(os.environ.get(name)),
            "value_recorded": False,
        }
        for name in provider["credential_env_vars"]
    ]


def _live_preflight_probe(
    provider_id: str,
    provider: dict[str, Any],
    credential_checks: list[dict[str, Any]],
    requested: bool,
) -> dict[str, Any]:
    client_dependency_checks = [
        {
            "module": module,
            "available": _module_available(module),
            "module_imported": False,
        }
        for module in PROVIDER_CLIENT_MODULES.get(provider_id, [])
    ]
    return {
        "requested": requested,
        "transport": "metadata_only",
        "provider_api_called": False,
        "client_modules_imported": False,
        "credential_values_recorded": False,
        "credential_env_vars": list(provider["credential_env_vars"]),
        "credential_present_count": sum(1 for check in credential_checks if check["present"]),
        "credential_required_count": len(credential_checks),
        "client_dependency_checks": client_dependency_checks,
        "client_dependency_available_count": sum(1 for check in client_dependency_checks if check["available"]),
        "client_dependency_required_count": len(client_dependency_checks),
    }


def _artifact_transfer_plan(provider: dict[str, Any], uploads: list[dict[str, Any]], downloads: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "mode": "dry_run_manifest_only",
        "upload_count": len(uploads),
        "expected_download_count": len(downloads),
        "upload_size_bytes": sum(_int_value(item.get("size_bytes")) for item in uploads if item.get("exists") is True),
        "artifact_protocols": list(provider["artifact_protocols"]),
        "requires_external_runner_upload": bool(uploads),
        "requires_external_runner_download": bool(downloads),
        "download_artifacts_expected_to_exist_before_launch": False,
        "flight_recorder_uploaded_artifacts": False,
        "flight_recorder_downloaded_artifacts": False,
        "provider_api_called": False,
        "credential_values_recorded": False,
    }


def _launch_plan_provider_chain(
    preflight_provider_id: str,
    artifact_manifest_provider_id: str,
    artifact_manifest_required: bool,
) -> dict[str, Any]:
    provider_consistent = (
        bool(preflight_provider_id)
        and (
            not artifact_manifest_required
            or (bool(artifact_manifest_provider_id) and preflight_provider_id == artifact_manifest_provider_id)
        )
    )
    return {
        "preflight_provider_id": preflight_provider_id,
        "artifact_manifest_provider_id": artifact_manifest_provider_id,
        "artifact_manifest_required": artifact_manifest_required,
        "provider_consistent": provider_consistent,
        "mismatched_provider_ids": (
            []
            if provider_consistent
            else [provider_id for provider_id in (preflight_provider_id, artifact_manifest_provider_id) if provider_id]
        ),
    }


def _payload_provider_id(payload: dict[str, Any]) -> str:
    provider = payload.get("provider") if isinstance(payload, dict) else {}
    return str(provider.get("id") or "") if isinstance(provider, dict) else ""


def _module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _json_artifact_ref(
    role: str,
    path: Path,
    schema_name: str,
    preserve_paths: bool,
    display_base_dir: Path | None,
) -> dict[str, Any]:
    ref = _file_ref(role, path, preserve_paths, display_base_dir)
    schema = _schema_check(path, schema_name) if ref["exists"] else {"passed": False, "error_count": 1, "errors": ["artifact not found"]}
    payload = _read_json(path) if ref["exists"] else {}
    ref.update(
        {
            "schema_name": schema_name,
            "schema_passed": schema["passed"],
            "schema_error_count": schema["error_count"],
            "schema_errors": schema["errors"],
            "source_passed": payload.get("passed") if isinstance(payload.get("passed"), bool) else None,
            "source_recommendation": str(payload.get("recommendation") or ""),
        }
    )
    return ref


def _file_ref(role: str, path: Path, preserve_paths: bool, display_base_dir: Path | None = None) -> dict[str, Any]:
    exists = path.exists() and path.is_file()
    display_path, replayable = _source_display_path(path, preserve_paths, display_base_dir)
    public_exists = exists and replayable
    return {
        "role": role,
        "path": display_path,
        "exists": public_exists,
        "sha256": _sha256(path) if public_exists else None,
        "size_bytes": path.stat().st_size if public_exists else None,
    }


def _missing_ref(role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": "",
        "exists": False,
        "sha256": None,
        "size_bytes": None,
        "schema_name": role,
        "schema_passed": False,
        "schema_error_count": 1,
        "schema_errors": ["artifact not provided"],
        "source_passed": None,
        "source_recommendation": "",
    }


def _artifact_ready(ref: dict[str, Any]) -> bool:
    return ref.get("exists") is True and ref.get("schema_passed") is True and ref.get("source_passed") is True


def _schema_check(path: Path, schema_name: str) -> dict[str, Any]:
    try:
        result = check_schema_file(path, schema_name)
    except (OSError, json.JSONDecodeError, SchemaRegistryError) as exc:
        return {"passed": False, "error_count": 1, "errors": [str(exc)]}
    return {
        "passed": result.get("passed") is True,
        "error_count": _int_value(result.get("error_count")),
        "errors": [str(error) for error in result.get("errors", []) if isinstance(error, str)][:20],
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _constraints(*, region: str | None, gpu_class: str | None, max_cost_usd: float | None) -> dict[str, Any]:
    return {
        "region": region or "",
        "gpu_class": gpu_class or "",
        "max_cost_usd": max_cost_usd,
        "requires_region_allowlist": True,
        "requires_gpu_class_allowlist": True,
        "requires_cost_estimate": True,
        "live_spend_allowed": False,
    }


def _boundary(*, live_requested: bool = False, allow_live: bool = False, live_preflight: bool = False) -> dict[str, Any]:
    return {
        "dry_run_only": True,
        "live_preflight_requested": live_preflight,
        "live_requested": live_requested,
        "allow_live": allow_live,
        "provider_api_called": False,
        "cloud_job_started": False,
        "cloud_cost_incurred_usd": 0,
        "model_downloads_started": False,
        "weights_updated_by_flight_recorder": False,
        "credential_values_recorded": False,
    }


def _handoff_contract() -> dict[str, Any]:
    return {
        "default_live_execution_allowed": False,
        "requires_explicit_live_opt_in": True,
        "requires_environment_credentials_for_live": True,
        "requires_cost_limit": True,
        "requires_region_and_gpu_constraints": True,
        "requires_artifact_upload_manifest": True,
        "requires_status_and_cancel_receipts": True,
        "flight_recorder_controls_preflight_only": True,
        "external_provider_owns_execution": True,
    }


def _display_path(path: Path, preserve_paths: bool, display_base_dir: Path | None = None) -> str:
    raw = str(path)
    if display_base_dir is not None:
        relative = _output_relative_path(path, display_base_dir)
        return relative if relative is not None else f"<redacted:{_basename(raw)}>"
    if preserve_paths:
        return raw if _is_safe_public_path(raw) else f"<redacted:{_basename(raw)}>"
    if _is_windows_absolute(raw):
        return f"<redacted:{_basename(raw)}>"
    if not path.is_absolute():
        return raw if _is_safe_public_path(raw) else f"<redacted:{_basename(raw)}>"
    try:
        relative = str(path.resolve().relative_to(Path.cwd().resolve()))
    except (OSError, ValueError):
        return f"<redacted:{path.name}>"
    return relative if _is_safe_public_path(relative) else f"<redacted:{path.name}>"


def _source_display_path(path: Path, preserve_paths: bool, display_base_dir: Path | None = None) -> tuple[str, bool]:
    displayed = _display_path(path, preserve_paths, display_base_dir)
    return displayed, _is_safe_public_path(displayed)


def _output_relative_path(path: Path, display_base_dir: Path) -> str | None:
    try:
        relative = os.path.relpath(path.resolve(), display_base_dir.resolve())
    except OSError:
        return None
    return relative if _is_safe_public_path(relative) else None


def _basename(value: str) -> str:
    return value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] or "path"


def _is_windows_absolute(value: str) -> bool:
    normalized = value.replace("/", "\\")
    return (len(normalized) >= 3 and normalized[1:3] == ":\\" and normalized[0].isalpha()) or normalized.startswith("\\\\")


def _is_safe_public_path(value: str) -> bool:
    if not value or value.startswith("<redacted:"):
        return False
    path = Path(value)
    windows_path = PureWindowsPath(value)
    return (
        not path.is_absolute()
        and not windows_path.is_absolute()
        and not windows_path.drive
        and "\\" not in value
        and "~" not in path.parts
        and ".." not in path.parts
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _int_value(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _add_check(checks: list[dict[str, Any]], check_id: str, passed: bool, actual: dict[str, Any], expected: dict[str, Any]) -> None:
    checks.append(
        {
            "id": check_id,
            "passed": bool(passed),
            "actual": actual,
            "expected": expected,
            "summary": f"{check_id}: passed={bool(passed)}",
        }
    )
