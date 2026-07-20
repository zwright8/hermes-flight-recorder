#!/usr/bin/env python3
"""Import a downloaded Hugging Face completion into canonical loop receipts.

This command is intentionally offline. The operator must first download the
model repository at the exact immutable completion revision and provide the
existing Hugging Face cloud launch/status evidence from the loop workspace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.agentic_training_result import (  # noqa: E402
    build_agentic_training_result,
    write_agentic_training_result,
)
from flightrecorder.cloud_training_completion import (  # noqa: E402
    build_cloud_training_completion_receipt,
    write_cloud_training_completion_receipt,
)
from flightrecorder.huggingface_lifecycle import (  # noqa: E402
    HF_CANONICAL_IMPORT_SCHEMA_VERSION,
    HuggingFaceLifecycleError,
    attach_identity,
    sha256_file,
    validate_job_completion,
)
from flightrecorder.validation import validate_agentic_training_result  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise HuggingFaceLifecycleError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _immutable_revision(value: str) -> bool:
    return 40 <= len(value) <= 64 and all(char in "0123456789abcdef" for char in value)


def verify_hub_snapshot_identity(
    snapshot: Path,
    completion_revision: str,
    publication: dict[str, Any],
) -> dict[str, Any]:
    """Verify an immutable clean Git checkout and its Hugging Face origin."""

    repo_id = str(publication.get("repo_id") or "")
    repo_type = str(publication.get("repo_type") or "")
    if repo_type not in {"model", "dataset"} or "/" not in repo_id:
        raise HuggingFaceLifecycleError(
            "publication receipt must identify an owner/repository Hub source"
        )
    commands = {
        "root": ["git", "-C", str(snapshot), "rev-parse", "--show-toplevel"],
        "revision": ["git", "-C", str(snapshot), "rev-parse", "HEAD"],
        "origin": ["git", "-C", str(snapshot), "config", "--get", "remote.origin.url"],
        "status": ["git", "-C", str(snapshot), "status", "--porcelain", "--untracked-files=all"],
    }
    observed: dict[str, str] = {}
    for label, command in commands.items():
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if completed.returncode != 0:
            raise HuggingFaceLifecycleError(
                f"--snapshot-dir is not a verifiable Git checkout ({label})"
            )
        observed[label] = completed.stdout.strip()
    if Path(observed["root"]).resolve() != snapshot:
        raise HuggingFaceLifecycleError("--snapshot-dir must be the Git checkout root")
    if observed["revision"] != completion_revision:
        raise HuggingFaceLifecycleError(
            "--completion-revision does not match the checked-out Git commit"
        )
    if observed["status"]:
        raise HuggingFaceLifecycleError("Hugging Face completion checkout must be clean")
    normalized_origin = observed["origin"].removesuffix(".git").rstrip("/")
    allowed_origins = {
        f"https://huggingface.co/{repo_id}",
        f"ssh://git@hf.co/{repo_id}",
        f"git@hf.co:{repo_id}",
    }
    if normalized_origin not in allowed_origins:
        raise HuggingFaceLifecycleError(
            "Git checkout origin does not match the publication receipt repository"
        )
    return {
        "source": "huggingface_git_checkout",
        "repo_id": repo_id,
        "repo_type": repo_type,
        "revision": completion_revision,
        "git_commit_verified": True,
        "remote_origin_sha256": hashlib.sha256(observed["origin"].encode("utf-8")).hexdigest(),
    }


def _copy_control(source: Path, target: Path) -> Path:
    if not source.is_file() or source.is_symlink():
        raise HuggingFaceLifecycleError(f"canonical import control source is not a regular file: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if sha256_file(source) != sha256_file(target):
        raise HuggingFaceLifecycleError(f"canonical import copy changed content: {source}")
    return target


def _materialize_flow_closure(
    *,
    flow_source: Path,
    plan_source: Path,
    runtime_source: Path,
    output: Path,
) -> tuple[Path, Path, Path]:
    flow = _read_json(flow_source)
    refs = flow.get("source_artifacts") if isinstance(flow.get("source_artifacts"), dict) else {}
    role_sources = {
        "agentic_training_plan": plan_source,
        "agentic_training_runtime_preflight": runtime_source,
    }
    copied: dict[str, Path] = {}
    for role, row in refs.items():
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise HuggingFaceLifecycleError(f"agentic training flow has no replayable {role} source")
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts or "\\" in row["path"]:
            raise HuggingFaceLifecycleError(f"agentic training flow source path is unsafe: {row['path']}")
        source = role_sources.get(role, flow_source.parent / relative)
        target = _copy_control(source.resolve(), output / relative)
        if target.stat().st_size != row.get("size_bytes") or sha256_file(target) != row.get("sha256"):
            raise HuggingFaceLifecycleError(f"agentic training flow source fingerprint mismatch: {role}")
        copied[role] = target
    for required in ("agentic_training_plan", "agentic_training_runtime_preflight"):
        if required not in copied:
            raise HuggingFaceLifecycleError(f"agentic training flow is missing {required}")
    flow_path = _copy_control(flow_source, output / "agentic_training_flow.json")
    return copied["agentic_training_plan"], copied["agentic_training_runtime_preflight"], flow_path


def _materialize_artifacts(
    completion: dict[str, Any], snapshot: Path, destination: Path
) -> dict[str, list[Path]]:
    role_mapping = {
        "adapter": "adapter",
        "checkpoint": "checkpoint",
        "training_metrics": "metrics",
        "trainer_log": "log",
        "trainer_error_log": "log",
        "training_plan": "config",
        "training_result": "config",
        "trainer_argv": "config",
        "adapter_manifest": "config",
    }
    mapped: dict[str, list[Path]] = {}
    for row in completion.get("artifacts", []):
        if not isinstance(row, dict):
            continue
        source = snapshot / str(row["path"])
        target = destination / str(row["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if target.stat().st_size != row["size_bytes"] or sha256_file(target) != row["sha256"]:
            raise HuggingFaceLifecycleError(f"materialized artifact fingerprint mismatch: {row['path']}")
        canonical_role = role_mapping.get(str(row.get("role") or ""), "config")
        mapped.setdefault(canonical_role, []).append(target)
    return mapped


def _provider_id(launch_plan: dict[str, Any]) -> str:
    provider = launch_plan.get("provider") if isinstance(launch_plan.get("provider"), dict) else {}
    return str(provider.get("id") or "")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_completion(args: argparse.Namespace) -> dict[str, Any]:
    snapshot = args.snapshot_dir.resolve()
    completion_path = args.completion_receipt.resolve()
    output = args.out_dir.resolve()
    if not _immutable_revision(args.completion_revision):
        raise HuggingFaceLifecycleError("--completion-revision must be an immutable Hub commit")
    if not snapshot.is_dir() or completion_path.parent != snapshot and snapshot not in completion_path.parents:
        raise HuggingFaceLifecycleError("completion receipt must be contained by --snapshot-dir")
    completion = _read_json(completion_path)
    publication = completion.get("publication_receipt")
    if not isinstance(publication, dict) or publication.get("private") is not True:
        raise HuggingFaceLifecycleError("canonical import requires a private immutable publication receipt")
    snapshot_provenance = verify_hub_snapshot_identity(
        snapshot,
        args.completion_revision,
        publication,
    )
    errors = validate_job_completion(completion, snapshot)
    if errors:
        raise HuggingFaceLifecycleError("Hugging Face completion validation failed: " + "; ".join(errors))

    launch_plan = _read_json(args.cloud_launch_plan)
    if _provider_id(launch_plan) != "huggingface_jobs":
        raise HuggingFaceLifecycleError("cloud launch plan provider must be huggingface_jobs")
    for source in (args.cloud_launch_plan, args.cloud_launch_receipt, args.cloud_status_receipt):
        if source.resolve().parent != output:
            raise HuggingFaceLifecycleError(
                "cloud launch plan, launch receipt, and status receipt must be direct children of --out-dir"
            )

    output.mkdir(parents=True, exist_ok=True)
    plan_path, runtime_path, flow_path = _materialize_flow_closure(
        flow_source=args.agentic_training_flow.resolve(),
        plan_source=args.agentic_training_plan.resolve(),
        runtime_source=args.runtime_preflight.resolve(),
        output=output,
    )
    artifacts = _materialize_artifacts(completion, snapshot, output / "imported_hf_artifacts")
    artifacts.setdefault("config", []).append(_copy_control(completion_path, output / "hf_job_completion.json"))

    hf_status = str(completion["status"])
    result_status = {"completed": "completed", "failed": "failed", "interrupted": "aborted"}[hf_status]
    failure_class = "none" if hf_status == "completed" else ("interrupted" if hf_status == "interrupted" else "trainer_crash")
    failure_message = "" if hf_status == "completed" else f"Hugging Face job reported {hf_status}"
    result_path = output / "agentic_training_result.json"
    training_result = build_agentic_training_result(
        plan_path=plan_path,
        runtime_preflight_path=runtime_path,
        agentic_training_flow_path=flow_path,
        out_path=result_path,
        status=result_status,
        failure_class=failure_class,
        failure_message=failure_message,
        runner_id="huggingface_jobs",
        run_id=str(completion["job"]["job_id"]),
        output_dir=output / "imported_hf_artifacts",
        artifacts=artifacts,
        created_at=str(completion["created_at"]),
    )
    if training_result.get("passed") is not True:
        reasons = training_result.get("blocked_reasons") or []
        raise HuggingFaceLifecycleError("canonical training-result import was blocked: " + "; ".join(map(str, reasons)))
    write_agentic_training_result(result_path, training_result)
    result_validation = validate_agentic_training_result(result_path)
    result_validation_errors = [*result_validation.errors, *result_validation.warnings]
    if result_validation_errors:
        raise HuggingFaceLifecycleError(
            "canonical training-result receipt did not replay: " + "; ".join(result_validation_errors)
        )

    raw_result_path = output / "huggingface_raw_provider_result.json"
    _write_json(
        raw_result_path,
        {
            "schema_version": "hfr.huggingface_completion_import.v1",
            "completion_revision": args.completion_revision,
            "snapshot_provenance": snapshot_provenance,
            "completion_receipt_sha256": sha256_file(completion_path),
            "completion_identity_sha256": completion["identity"]["sha256"],
            "artifact_revision": publication["revision"],
            "artifact_set_sha256": completion["artifact_set_sha256"],
            "runtime_declaration_identity_sha256": completion["runtime_declaration"]["declaration_identity_sha256"],
            "runtime_observation_identity_sha256": completion["runtime_observation"]["runtime_identity_sha256"],
        },
    )
    candidate_id = str(training_result["registry_update"]["target_model_id"])
    if not candidate_id:
        raise HuggingFaceLifecycleError("canonical training plan does not identify a target model")
    timestamp = str(completion["created_at"])
    runner_path = output / "huggingface_runner_metadata.json"
    cloud_status = "completed" if hf_status == "completed" else "failed"
    cloud_failure = "none" if hf_status == "completed" else ("cancelled" if hf_status == "interrupted" else "runner")
    metadata = {
        "schema_version": "hfr.external_cloud_training_runner.v1",
        "provider_id": "huggingface_jobs",
        "provider_job_id": str(completion["job"]["job_id"]),
        "execution_id": "hf-" + str(completion["identity"]["sha256"])[:24],
        "result_run_id": str(completion["job"]["job_id"]),
        "candidate_model_id": candidate_id,
        "status": cloud_status,
        "terminal": True,
        "failure": {"class": cloud_failure, "message": failure_message},
        "runner": {"id": "flightrecorder-huggingface-import", "version": "1"},
        "started_at": timestamp,
        "observed_at": timestamp,
        "finished_at": timestamp,
        "exit_code": int(completion["job"]["exit_code"]),
        "provider_constraints": {
            "region": args.region,
            "gpu_class": args.gpu_class,
            "reported_cost_usd": args.reported_cost_usd,
        },
        "source_sha256": {
            "launch_plan": _sha256(args.cloud_launch_plan),
            "launch_receipt": _sha256(args.cloud_launch_receipt),
            "status_receipt": _sha256(args.cloud_status_receipt),
            "raw_provider_result": _sha256(raw_result_path),
            "output_artifact_manifest": _sha256(result_path),
        },
        "side_effects": {
            "external_provider_api_called": True,
            "external_cloud_job_started": True,
            "external_artifacts_uploaded": True,
            "external_artifacts_downloaded": True,
            "credential_values_recorded": "not_observed",
            "provider_api_called_by_flight_recorder": False,
            "cloud_job_started_by_flight_recorder": False,
            "provider_status_polled_by_flight_recorder": False,
            "artifacts_uploaded_by_flight_recorder": False,
            "artifacts_downloaded_by_flight_recorder": False,
            "model_downloads_started_by_flight_recorder": False,
            "weights_updated_by_flight_recorder": False,
            "provider_modules_imported_by_flight_recorder": False,
        },
    }
    _write_json(runner_path, metadata)
    cloud_path = output / "cloud_training_completion_receipt.json"
    cloud_receipt = build_cloud_training_completion_receipt(
        launch_plan_path=args.cloud_launch_plan,
        launch_receipt_path=args.cloud_launch_receipt,
        status_receipt_path=args.cloud_status_receipt,
        runner_metadata_path=runner_path,
        raw_provider_result_path=raw_result_path,
        output_artifact_manifest_path=result_path,
        out_path=cloud_path,
        created_at=timestamp,
    )
    if cloud_receipt.get("passed") is not True:
        integrity = cloud_receipt.get("integrity") if isinstance(cloud_receipt.get("integrity"), dict) else {}
        reasons = integrity.get("blocking_reasons") or [
            check.get("summary") or check.get("id")
            for check in cloud_receipt.get("checks", [])
            if isinstance(check, dict) and check.get("passed") is False
        ]
        raise HuggingFaceLifecycleError("canonical cloud completion import was blocked: " + "; ".join(map(str, reasons)))
    write_cloud_training_completion_receipt(cloud_receipt, cloud_path)
    import_receipt = attach_identity(
        {
            "schema_version": HF_CANONICAL_IMPORT_SCHEMA_VERSION,
            "completion_revision": args.completion_revision,
            "completion_receipt_sha256": sha256_file(completion_path),
            "source_completion_identity_sha256": completion["identity"]["sha256"],
            "snapshot_provenance": snapshot_provenance,
            "agentic_training_result": {"path": result_path.name, "sha256": sha256_file(result_path)},
            "cloud_training_completion_receipt": {"path": cloud_path.name, "sha256": sha256_file(cloud_path)},
        }
    )
    _write_json(output / "huggingface_canonical_import.json", import_receipt)
    return import_receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--completion-receipt", type=Path, required=True)
    parser.add_argument("--completion-revision", required=True)
    parser.add_argument("--agentic-training-plan", type=Path, required=True)
    parser.add_argument("--runtime-preflight", type=Path, required=True)
    parser.add_argument("--agentic-training-flow", type=Path, required=True)
    parser.add_argument("--cloud-launch-plan", type=Path, required=True)
    parser.add_argument("--cloud-launch-receipt", type=Path, required=True)
    parser.add_argument("--cloud-status-receipt", type=Path, required=True)
    parser.add_argument("--region", default="provider_default")
    parser.add_argument("--gpu-class", required=True)
    parser.add_argument("--reported-cost-usd", type=float, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    try:
        receipt = import_completion(parse_args())
    except (HuggingFaceLifecycleError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
