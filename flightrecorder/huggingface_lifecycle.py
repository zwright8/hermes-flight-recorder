"""Dependency-free contracts for immutable Hugging Face training lifecycles.

The module is intentionally standalone: the handoff builder copies it beside
the remote Jobs entry point so the same validation and argument reconstruction
code runs before local submission and inside the ephemeral worker.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import platform
import re
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


HF_REVIEWED_TRAINING_PLAN_SCHEMA_VERSION = "hfr.huggingface_reviewed_training_plan.v1"
HF_JOBS_HANDOFF_SCHEMA_VERSION = "hfr.huggingface_jobs_handoff.v1"
HF_PUBLICATION_RECEIPT_SCHEMA_VERSION = "hfr.huggingface_publication_receipt.v1"
HF_JOB_COMPLETION_SCHEMA_VERSION = "hfr.huggingface_job_completion.v1"
HF_CANONICAL_IMPORT_SCHEMA_VERSION = "hfr.huggingface_canonical_import.v1"

EXECUTABLE_MODES = {"trace_sft", "fr_sft", "fr_action_sft", "fr_dpo", "fr_sft_dpo"}
COMPLETION_STATUSES = {"completed", "failed", "interrupted"}
RESUME_PHASES = {"sft", "dpo"}

EXACT_DEPENDENCIES = (
    "accelerate==1.12.0",
    "datasets==4.4.1",
    "huggingface-hub==1.1.5",
    "peft==0.19.1",
    "torch==2.9.1",
    "trackio==0.11.0",
    "transformers==4.57.6",
    "trl==0.26.2",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HUB_COMMIT_RE = re.compile(r"^[0-9a-f]{40,64}$")
_PINNED_IMAGE_RE = re.compile(r"^\S+@sha256:[0-9a-f]{64}$")
_EXACT_DEPENDENCY_RE = re.compile(r"^[A-Za-z0-9_.-]+==[^=<>!~\s]+$")


class HuggingFaceLifecycleError(ValueError):
    """Raised when immutable Hugging Face lifecycle evidence is invalid."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_record(path: str | Path, *, base_dir: str | Path | None = None, role: str = "artifact") -> dict[str, Any]:
    source = Path(path)
    if not source.exists() or not source.is_file() or source.is_symlink():
        raise HuggingFaceLifecycleError(f"{role} must be an existing regular non-symlink file: {source}")
    display = source.name
    if base_dir is not None:
        try:
            display = source.resolve().relative_to(Path(base_dir).resolve()).as_posix()
        except ValueError as exc:
            raise HuggingFaceLifecycleError(f"{role} must be contained by {Path(base_dir)}: {source}") from exc
    return {
        "role": role,
        "path": display,
        "sha256": sha256_file(source),
        "size_bytes": source.stat().st_size,
    }


def artifact_set_sha256(records: Iterable[dict[str, Any]]) -> str:
    normalized = [
        {
            "role": str(row.get("role") or ""),
            "path": str(row.get("path") or ""),
            "sha256": str(row.get("sha256") or ""),
            "size_bytes": row.get("size_bytes"),
        }
        for row in records
    ]
    return canonical_sha256(sorted(normalized, key=lambda row: (row["role"], row["path"])))


def payload_identity(value: dict[str, Any]) -> str:
    identity_view = copy.deepcopy(value)
    identity_view.pop("identity", None)
    return canonical_sha256(identity_view)


def attach_identity(value: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result["identity"] = {"algorithm": "sha256", "sha256": payload_identity(result)}
    return result


def dependency_lock_sha256(dependencies: Iterable[str]) -> str:
    return canonical_sha256(list(dependencies))


def attach_runtime_identity(observation: dict[str, Any]) -> dict[str, Any]:
    """Bind only runtime-probed libraries, container, hardware, and CUDA facts."""
    result = copy.deepcopy(observation)
    result["library_identity_sha256"] = canonical_sha256(_dict(result.get("dependencies")))
    result["cuda_identity_sha256"] = canonical_sha256(_dict(result.get("cuda")))
    result["container_identity_sha256"] = canonical_sha256(_dict(result.get("container")))
    result["hardware_identity_sha256"] = canonical_sha256(_dict(result.get("hardware")))
    result["runtime_identity_sha256"] = canonical_sha256(
        {
            "context": result.get("context"),
            "python": result.get("python"),
            "python_implementation": result.get("python_implementation"),
            "platform": result.get("platform"),
            "library_identity_sha256": result["library_identity_sha256"],
            "cuda_identity_sha256": result["cuda_identity_sha256"],
            "container_identity_sha256": result["container_identity_sha256"],
            "hardware_identity_sha256": result["hardware_identity_sha256"],
            "measured_artifacts": _dict(result.get("measured_artifacts")),
        }
    )
    return result


def runtime_declaration_from_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Copy reviewed intent into a field explicitly labelled as declaration."""
    runtime = _dict(plan.get("runtime"))
    inputs = _dict(plan.get("inputs"))
    declaration = {
        "python": runtime.get("python"),
        "container_image": runtime.get("container_image"),
        "hardware_flavor": runtime.get("hardware_flavor"),
        "timeout": runtime.get("timeout"),
        "dependencies": copy.deepcopy(runtime.get("dependencies")),
        "dependency_lock_sha256": runtime.get("dependency_lock_sha256"),
        "trainer_script_sha256": runtime.get("trainer_script_sha256"),
        "model_identity": {
            "base_model_revision": _dict(inputs.get("base_model")).get("revision"),
            "tokenizer_revision": _dict(inputs.get("tokenizer")).get("revision"),
            "chat_template_sha256": _dict(inputs.get("chat_template")).get("sha256"),
        },
    }
    declaration["declaration_identity_sha256"] = canonical_sha256(declaration)
    return declaration


def _read_probe(path: Path) -> tuple[str, str]:
    try:
        content = path.read_bytes()
    except OSError:
        return "unavailable", ""
    return hashlib.sha256(content).hexdigest(), content.decode("utf-8", errors="replace")


def _container_observation() -> dict[str, Any]:
    markers = [path for path in (Path("/.dockerenv"), Path("/run/.containerenv")) if path.exists()]
    cgroup_sha256, cgroup_text = _read_probe(Path("/proc/self/cgroup"))
    os_release_sha256, _ = _read_probe(Path("/etc/os-release"))
    container_tokens = ("docker", "containerd", "kubepods", "podman", "libpod")
    detected = bool(markers) or any(token in cgroup_text.casefold() for token in container_tokens)
    image_digest = "unavailable"
    image_digest_source = "unavailable"
    for name in ("HF_JOB_IMAGE_DIGEST", "CONTAINER_IMAGE_DIGEST", "OCI_IMAGE_DIGEST"):
        value = os.environ.get(name, "").strip()
        match = re.search(r"sha256:[0-9a-f]{64}", value)
        if match:
            image_digest = match.group(0)
            image_digest_source = f"environment:{name}"
            break
    return {
        "measurement_source": "runtime_probe",
        "detected": detected,
        "isolation": "container" if detected else "host_or_unknown",
        "marker_paths": [path.as_posix() for path in markers],
        "cgroup_sha256": cgroup_sha256,
        "os_release_sha256": os_release_sha256,
        "image_digest": image_digest,
        "image_digest_source": image_digest_source,
    }


def observe_runtime_identity(
    dependencies: Iterable[str],
    *,
    context: str = "training_job",
    trainer_script_path: str | Path | None = None,
) -> dict[str, Any]:
    """Measure worker libraries, container probes, hardware, and accelerator state."""
    observed_dependencies: dict[str, dict[str, Any]] = {}
    for requirement in dependencies:
        name, _expected = str(requirement).split("==", 1)
        try:
            actual = version(name)
        except PackageNotFoundError:
            actual = "missing"
        observed_dependencies[name] = {"actual": actual, "measurement_source": "importlib.metadata"}

    cuda: dict[str, Any] = {
        "available": False,
        "runtime_version": "unavailable",
        "driver_version": "unavailable",
        "cudnn_version": "unavailable",
        "device_count": 0,
        "devices": [],
    }
    try:
        torch = importlib.import_module("torch")
    except (ImportError, OSError, RuntimeError):
        torch = None
    if torch is not None:
        cuda_api = getattr(torch, "cuda", None)
        cuda["runtime_version"] = str(getattr(getattr(torch, "version", None), "cuda", None) or "unavailable")
        cudnn_version = None
        try:
            cudnn_version = torch.backends.cudnn.version()
        except (AttributeError, RuntimeError):
            pass
        cuda["cudnn_version"] = str(cudnn_version or "unavailable")
        try:
            cuda["available"] = bool(cuda_api is not None and cuda_api.is_available())
            cuda["device_count"] = int(cuda_api.device_count()) if cuda["available"] else 0
            if cuda["available"]:
                cuda["devices"] = [
                    {
                        "index": index,
                        "name": str(cuda_api.get_device_name(index)),
                        "compute_capability": ".".join(str(part) for part in cuda_api.get_device_capability(index)),
                    }
                    for index in range(cuda["device_count"])
                ]
        except (AttributeError, RuntimeError, ValueError):
            cuda["available"] = False
            cuda["device_count"] = 0
            cuda["devices"] = []
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        driver = None
    if driver is not None and driver.returncode == 0:
        versions = sorted({row.strip() for row in driver.stdout.splitlines() if row.strip()})
        if versions:
            cuda["driver_version"] = ",".join(versions)

    hardware = {
        "measurement_source": "runtime_probe",
        "machine": platform.machine() or "unavailable",
        "processor": platform.processor() or "unavailable",
        "logical_cpu_count": int(os.cpu_count() or 0),
        "accelerator_kind": "cuda" if cuda["available"] else "none",
        "accelerator_count": cuda["device_count"],
    }
    measured_artifacts = {
        "trainer_script_sha256": sha256_file(trainer_script_path)
        if trainer_script_path is not None
        else "unavailable"
    }
    return attach_runtime_identity(
        {
            "context": context,
            "observed_at": utc_now(),
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "dependencies": observed_dependencies,
            "container": _container_observation(),
            "hardware": hardware,
            "cuda": cuda,
            "measured_artifacts": measured_artifacts,
        }
    )


def build_reviewed_training_plan(
    *,
    source_plan: dict[str, Any],
    source_plan_record: dict[str, Any],
    model_manifest: dict[str, Any],
    model_manifest_record: dict[str, Any],
    dataset_manifest_record: dict[str, Any],
    trainer_script_record: dict[str, Any],
    data_artifacts: list[dict[str, Any]],
    reviewer: str,
    reviewed_at: str,
    dataset_repo: str,
    model_repo: str,
    flavor: str,
    timeout: str,
    container_image: str,
    dependencies: Iterable[str] = EXACT_DEPENDENCIES,
    private: bool = True,
    resume: dict[str, Any] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Create the sole behavior-affecting configuration for an HF job."""
    mode = str(source_plan.get("mode") or "")
    hyperparameters = copy.deepcopy(source_plan.get("hyperparameters"))
    tracking = copy.deepcopy(source_plan.get("tracking"))
    if not isinstance(hyperparameters, dict):
        raise HuggingFaceLifecycleError("source training plan must contain hyperparameters")
    if not isinstance(tracking, dict):
        raise HuggingFaceLifecycleError("source training plan must contain tracking")
    model_id = str(source_plan.get("model") or model_manifest.get("model_id") or "")
    source = model_manifest.get("source") if isinstance(model_manifest.get("source"), dict) else {}
    compatibility = model_manifest.get("compatibility") if isinstance(model_manifest.get("compatibility"), dict) else {}
    tokenizer = compatibility.get("tokenizer") if isinstance(compatibility.get("tokenizer"), dict) else {}
    chat_template = compatibility.get("chat_template") if isinstance(compatibility.get("chat_template"), dict) else {}
    base_revision = str(source.get("revision") or "")
    tokenizer_revision = str(tokenizer.get("revision") or base_revision)
    chat_template_sha256 = str(chat_template.get("sha256") or "")
    dependency_rows = tuple(str(item) for item in dependencies)
    smoke = source_plan.get("smoke") if isinstance(source_plan.get("smoke"), dict) else {}
    normalized_resume = _normalized_resume(resume)
    plan = {
        "schema_version": HF_REVIEWED_TRAINING_PLAN_SCHEMA_VERSION,
        "created_at": created_at or utc_now(),
        "approved": True,
        "review": {
            "reviewer": reviewer,
            "reviewed_at": reviewed_at,
            "source_plan_passed": source_plan.get("passed") is True,
            "source_plan_recommendation": source_plan.get("recommendation"),
        },
        "source_plan": copy.deepcopy(source_plan_record),
        "trainer": {
            "mode": mode,
            "model_id": model_id,
            "loss_scope": "assistant_only" if hyperparameters.get("assistant_only_loss") is True else "all_messages",
            "hyperparameters": hyperparameters,
            "tracking": tracking,
            "row_limit": int(smoke.get("row_limit") or 0),
        },
        "inputs": {
            "model_manifest": copy.deepcopy(model_manifest_record),
            "dataset_manifest": copy.deepcopy(dataset_manifest_record),
            "trainer_script": copy.deepcopy(trainer_script_record),
            "data_artifacts": copy.deepcopy(sorted(data_artifacts, key=lambda row: str(row.get("path") or ""))),
            "artifact_set_sha256": artifact_set_sha256(data_artifacts),
            "base_model": {"repo_id": model_id, "revision": base_revision},
            "tokenizer": {"repo_id": str(tokenizer.get("repo_id") or model_id), "revision": tokenizer_revision},
            "chat_template": {"sha256": chat_template_sha256},
        },
        "runtime": {
            "python": "3.11",
            "container_image": container_image,
            "hardware_flavor": flavor,
            "timeout": timeout,
            "dependencies": list(dependency_rows),
            "dependency_lock_sha256": dependency_lock_sha256(dependency_rows),
            "trainer_script_sha256": str(trainer_script_record.get("sha256") or ""),
        },
        "output": {
            "dataset_repo": dataset_repo,
            "model_repo": model_repo,
            "dataset_private": bool(private),
            "model_private": bool(private),
            "checkpoint_push_strategy": "every_save",
            "final_adapter_push": True,
            "result_path": "lifecycle/completion_receipt.json",
            "publication_receipt_path": "lifecycle/publication_receipt.json",
        },
        "resume": normalized_resume,
    }
    plan = attach_identity(plan)
    errors = validate_reviewed_training_plan(plan)
    if errors:
        raise HuggingFaceLifecycleError("; ".join(errors))
    return plan


def validate_reviewed_training_plan(plan: Any, base_dir: str | Path | None = None) -> list[str]:
    errors: list[str] = []
    if not isinstance(plan, dict):
        return ["reviewed training plan must be a JSON object"]
    if plan.get("schema_version") != HF_REVIEWED_TRAINING_PLAN_SCHEMA_VERSION:
        errors.append(f"schema_version must be {HF_REVIEWED_TRAINING_PLAN_SCHEMA_VERSION}")
    if plan.get("approved") is not True:
        errors.append("approved must be true")
    review = _dict(plan.get("review"))
    for key in ("reviewer", "reviewed_at"):
        if not _nonempty(review.get(key)):
            errors.append(f"review.{key} must be non-empty")
    if review.get("source_plan_passed") is not True or review.get("source_plan_recommendation") != "launch_allowed":
        errors.append("review must bind a passing launch_allowed source plan")

    trainer = _dict(plan.get("trainer"))
    if trainer.get("mode") not in EXECUTABLE_MODES:
        errors.append("trainer.mode must be executable")
    if not _nonempty(trainer.get("model_id")):
        errors.append("trainer.model_id must be non-empty")
    if trainer.get("loss_scope") not in {"assistant_only", "all_messages"}:
        errors.append("trainer.loss_scope must be assistant_only or all_messages")
    hyperparameters = _dict(trainer.get("hyperparameters"))
    _validate_hyperparameters(hyperparameters, trainer, errors)
    tracking = _dict(trainer.get("tracking"))
    if tracking.get("report_to") != ["trackio"]:
        errors.append("trainer.tracking.report_to must be ['trackio']")
    if not _nonempty(tracking.get("trackio_project")):
        errors.append("trainer.tracking.trackio_project must be non-empty")
    if not isinstance(trainer.get("row_limit"), int) or trainer.get("row_limit", -1) < 0:
        errors.append("trainer.row_limit must be a non-negative integer")

    inputs = _dict(plan.get("inputs"))
    for name in ("source_plan",):
        _validate_artifact_ref(plan.get(name), name, base_dir, errors)
    for name in ("model_manifest", "dataset_manifest", "trainer_script"):
        _validate_artifact_ref(inputs.get(name), f"inputs.{name}", base_dir, errors)
    data_artifacts = inputs.get("data_artifacts")
    if not isinstance(data_artifacts, list) or not data_artifacts:
        errors.append("inputs.data_artifacts must be a non-empty list")
        data_artifacts = []
    for index, row in enumerate(data_artifacts):
        _validate_artifact_ref(row, f"inputs.data_artifacts[{index}]", base_dir, errors)
    if inputs.get("artifact_set_sha256") != artifact_set_sha256(row for row in data_artifacts if isinstance(row, dict)):
        errors.append("inputs.artifact_set_sha256 does not match data_artifacts")
    base_model = _dict(inputs.get("base_model"))
    tokenizer = _dict(inputs.get("tokenizer"))
    chat_template = _dict(inputs.get("chat_template"))
    if base_model.get("repo_id") != trainer.get("model_id"):
        errors.append("inputs.base_model.repo_id must match trainer.model_id")
    for label, value in (("base_model", base_model), ("tokenizer", tokenizer)):
        if not _nonempty(value.get("repo_id")) or not _immutable_revision(value.get("revision")):
            errors.append(f"inputs.{label} must contain a repo_id and immutable revision")
    if not _is_sha256(chat_template.get("sha256")):
        errors.append("inputs.chat_template.sha256 must be a SHA-256 digest")

    runtime = _dict(plan.get("runtime"))
    dependencies = runtime.get("dependencies")
    if not isinstance(dependencies, list) or not dependencies or not all(
        isinstance(item, str) and _EXACT_DEPENDENCY_RE.fullmatch(item) for item in dependencies
    ):
        errors.append("runtime.dependencies must contain exact name==version pins")
        dependencies = []
    dependency_names = [str(item).split("==", 1)[0] for item in dependencies]
    if len(dependency_names) != len(set(dependency_names)):
        errors.append("runtime.dependencies must not repeat package names")
    if runtime.get("dependency_lock_sha256") != dependency_lock_sha256(dependencies):
        errors.append("runtime.dependency_lock_sha256 does not match dependencies")
    if not _PINNED_IMAGE_RE.fullmatch(str(runtime.get("container_image") or "")):
        errors.append("runtime.container_image must use an immutable @sha256 digest")
    if runtime.get("python") != "3.11":
        errors.append("runtime.python must be 3.11")
    for key in ("hardware_flavor", "timeout"):
        if not _nonempty(runtime.get(key)):
            errors.append(f"runtime.{key} must be non-empty")
    trainer_script = _dict(inputs.get("trainer_script"))
    if runtime.get("trainer_script_sha256") != trainer_script.get("sha256"):
        errors.append("runtime.trainer_script_sha256 must match inputs.trainer_script")

    output = _dict(plan.get("output"))
    for key in ("dataset_repo", "model_repo", "result_path", "publication_receipt_path"):
        if not _nonempty(output.get(key)):
            errors.append(f"output.{key} must be non-empty")
    if output.get("dataset_private") is not True or output.get("model_private") is not True:
        errors.append("output repositories must be private")
    if output.get("checkpoint_push_strategy") != "every_save" or output.get("final_adapter_push") is not True:
        errors.append("output must persist every saved checkpoint and the final adapter")
    for key in ("result_path", "publication_receipt_path"):
        if _nonempty(output.get(key)) and not _safe_repo_path(str(output[key])):
            errors.append(f"output.{key} must be a safe relative repository path")

    _validate_resume(_dict(plan.get("resume")), errors)
    identity = _dict(plan.get("identity"))
    if identity.get("algorithm") != "sha256" or identity.get("sha256") != payload_identity(plan):
        errors.append("identity.sha256 does not match the reviewed plan")
    return errors


def trainer_configuration(plan: dict[str, Any]) -> dict[str, Any]:
    """Return the exact behavior-affecting trainer configuration."""
    errors = validate_reviewed_training_plan(plan)
    if errors:
        raise HuggingFaceLifecycleError("; ".join(errors))
    trainer = _dict(plan["trainer"])
    runtime = _dict(plan["runtime"])
    output = _dict(plan["output"])
    return {
        "mode": trainer["mode"],
        "model": trainer["model_id"],
        "loss_scope": trainer["loss_scope"],
        "hyperparameters": copy.deepcopy(trainer["hyperparameters"]),
        "tracking": copy.deepcopy(trainer["tracking"]),
        "row_limit": trainer["row_limit"],
        "hub_model_id": output["model_repo"],
        "push_to_hub": True,
        "runtime_dependencies": list(runtime["dependencies"]),
        "resume": copy.deepcopy(plan["resume"]),
    }


def trainer_argv_from_plan(
    plan: dict[str, Any],
    *,
    python: str,
    trainer_script: str | Path,
    experiment_dir: str | Path,
    output_dir: str | Path,
    model_manifest: str | Path,
    dataset_manifest: str | Path,
    result_registry: str | Path,
    model_registry_link_plan: str | Path,
    resume_checkpoint: str | Path | None = None,
) -> list[str]:
    """Reconstruct the full trainer command from the reviewed plan only."""
    config = trainer_configuration(plan)
    params = _dict(config["hyperparameters"])
    tracking = _dict(config["tracking"])
    command = [
        str(python),
        str(trainer_script),
        "--mode",
        str(config["mode"]),
        "--experiment-dir",
        str(experiment_dir),
        "--output-dir",
        str(output_dir),
        "--model",
        str(config["model"]),
        "--model-revision",
        str(plan["inputs"]["base_model"]["revision"]),
        "--tokenizer-revision",
        str(plan["inputs"]["tokenizer"]["revision"]),
        "--expected-chat-template-sha256",
        str(plan["inputs"]["chat_template"]["sha256"]),
        "--model-manifest",
        str(model_manifest),
        "--dataset-manifest",
        str(dataset_manifest),
        "--require-registered-inputs",
        "--push-to-hub",
        "--hub-model-id",
        str(config["hub_model_id"]),
        "--result-registry",
        str(result_registry),
        "--write-model-registry-link-plan",
        str(model_registry_link_plan),
        "--sft-epochs",
        _number_text(params["sft_epochs"]),
        "--dpo-epochs",
        _number_text(params["dpo_epochs"]),
        "--sft-learning-rate",
        _number_text(params["sft_learning_rate"]),
        "--dpo-learning-rate",
        _number_text(params["dpo_learning_rate"]),
        "--batch-size",
        str(params["batch_size"]),
        "--gradient-accumulation-steps",
        str(params["gradient_accumulation_steps"]),
        "--max-steps",
        str(params["max_steps"]),
        "--max-length",
        str(params["max_length"]),
        "--lora-r",
        str(params["lora_r"]),
        "--lora-alpha",
        str(params["lora_alpha"]),
        "--lora-dropout",
        _number_text(params["lora_dropout"]),
        "--seed",
        str(params["seed"]),
        "--data-seed",
        str(params["data_seed"]),
        "--save-steps",
        str(params["save_steps"]),
        "--save-total-limit",
        str(params["save_total_limit"]),
        "--trackio-project",
        str(tracking["trackio_project"]),
    ]
    if tracking.get("trackio_space_id"):
        command.extend(["--trackio-space-id", str(tracking["trackio_space_id"])])
    if tracking.get("run_name_prefix"):
        command.extend(["--run-name-prefix", str(tracking["run_name_prefix"])])
    if config["row_limit"]:
        command.extend(["--limit", str(config["row_limit"])])
    if params.get("gradient_checkpointing") is True:
        command.append("--gradient-checkpointing")
    if config["loss_scope"] == "all_messages":
        command.append("--all-message-loss")
    resume = _dict(config.get("resume"))
    if resume.get("enabled") is True:
        if resume_checkpoint is None:
            raise HuggingFaceLifecycleError("enabled resume requires a materialized checkpoint")
        command.extend(
            [
                "--resume-from-checkpoint",
                str(resume_checkpoint),
                "--resume-phase",
                str(resume["phase"]),
            ]
        )
    return command


def build_publication_receipt(
    *,
    repo_id: str,
    repo_type: str,
    revision: str,
    artifacts: list[dict[str, Any]],
    private: bool,
    source_plan_sha256: str,
    reviewed_plan_record: dict[str, Any],
    runtime_observation: dict[str, Any],
    base_dir: str | Path,
    runtime_declaration: dict[str, Any] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    if runtime_declaration is None:
        runtime_declaration = runtime_declaration_from_plan(
            _read_recorded_json(reviewed_plan_record, base_dir, "reviewed_plan")
        )
    receipt = {
        "schema_version": HF_PUBLICATION_RECEIPT_SCHEMA_VERSION,
        "created_at": created_at or utc_now(),
        "repo_id": repo_id,
        "repo_type": repo_type,
        "private": private,
        "revision": revision,
        "source_plan_sha256": source_plan_sha256,
        "reviewed_plan": copy.deepcopy(reviewed_plan_record),
        "runtime_declaration": copy.deepcopy(runtime_declaration),
        "runtime_observation": copy.deepcopy(runtime_observation),
        "artifact_count": len(artifacts),
        "artifact_set_sha256": artifact_set_sha256(artifacts),
        "artifacts": copy.deepcopy(artifacts),
    }
    receipt = attach_identity(receipt)
    errors = validate_publication_receipt(receipt, base_dir)
    if errors:
        raise HuggingFaceLifecycleError("; ".join(errors))
    return receipt


def validate_publication_receipt(receipt: Any, base_dir: str | Path | None = None) -> list[str]:
    errors: list[str] = []
    if not isinstance(receipt, dict):
        return ["publication receipt must be a JSON object"]
    if receipt.get("schema_version") != HF_PUBLICATION_RECEIPT_SCHEMA_VERSION:
        errors.append(f"schema_version must be {HF_PUBLICATION_RECEIPT_SCHEMA_VERSION}")
    if receipt.get("private") is not True:
        errors.append("publication receipt must record a private repository")
    if receipt.get("repo_type") not in {"dataset", "model"} or not _nonempty(receipt.get("repo_id")):
        errors.append("publication receipt must identify a dataset or model repository")
    if not _immutable_revision(receipt.get("revision")):
        errors.append("publication receipt revision must be immutable")
    if not _is_sha256(receipt.get("source_plan_sha256")):
        errors.append("publication receipt source_plan_sha256 must be a SHA-256 digest")
    reviewed_plan = _load_reviewed_plan(
        receipt.get("reviewed_plan"),
        base_dir,
        "reviewed_plan",
        errors,
    )
    runtime_declaration = _dict(receipt.get("runtime_declaration"))
    runtime_observation = _dict(receipt.get("runtime_observation"))
    _validate_runtime_declaration(runtime_declaration, errors)
    _validate_runtime_observation(runtime_observation, errors)
    expected_context = "payload_publisher" if receipt.get("repo_type") == "dataset" else "training_job"
    if runtime_observation.get("context") != expected_context:
        errors.append(f"publication runtime_observation.context must be {expected_context}")
    if reviewed_plan is not None:
        if receipt.get("source_plan_sha256") != _dict(reviewed_plan.get("identity")).get("sha256"):
            errors.append("publication receipt source_plan_sha256 must match the reviewed plan identity")
        _validate_completion_runtime_identity(
            reviewed_plan,
            runtime_declaration,
            runtime_observation,
            None,
            errors,
            require_execution_match=False,
        )
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append("publication receipt artifacts must be non-empty")
        artifacts = []
    for index, row in enumerate(artifacts):
        _validate_artifact_ref(row, f"artifacts[{index}]", None, errors)
    plan_ref = _dict(receipt.get("reviewed_plan"))
    if not any(
        isinstance(row, dict)
        and row.get("path") == plan_ref.get("path")
        and row.get("sha256") == plan_ref.get("sha256")
        and row.get("size_bytes") == plan_ref.get("size_bytes")
        for row in artifacts
    ):
        errors.append("publication receipt reviewed_plan path/hash must match a published artifact")
    if receipt.get("artifact_count") != len(artifacts):
        errors.append("publication receipt artifact_count does not match artifacts")
    if receipt.get("artifact_set_sha256") != artifact_set_sha256(row for row in artifacts if isinstance(row, dict)):
        errors.append("publication receipt artifact_set_sha256 does not match artifacts")
    identity = _dict(receipt.get("identity"))
    if identity.get("algorithm") != "sha256" or identity.get("sha256") != payload_identity(receipt):
        errors.append("publication receipt identity does not match")
    return errors


def build_job_completion(
    *,
    status: str,
    plan_record: dict[str, Any],
    dataset_revision: str,
    job_id: str,
    exit_code: int,
    artifacts: list[dict[str, Any]],
    publication_receipt: dict[str, Any] | None,
    runtime_observation: dict[str, Any],
    resume: dict[str, Any],
    base_dir: str | Path,
    runtime_declaration: dict[str, Any] | None = None,
    failure: dict[str, Any] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    if runtime_declaration is None:
        runtime_declaration = runtime_declaration_from_plan(
            _read_recorded_json(plan_record, base_dir, "reviewed_plan")
        )
    completion = {
        "schema_version": HF_JOB_COMPLETION_SCHEMA_VERSION,
        "created_at": created_at or utc_now(),
        "status": status,
        "terminal": True,
        "job": {
            "provider": "huggingface_jobs",
            "job_id": job_id,
            "dataset_revision": dataset_revision,
            "exit_code": exit_code,
        },
        "reviewed_plan": copy.deepcopy(plan_record),
        "runtime_declaration": copy.deepcopy(runtime_declaration),
        "runtime_observation": copy.deepcopy(runtime_observation),
        "artifacts": copy.deepcopy(artifacts),
        "artifact_count": len(artifacts),
        "artifact_set_sha256": artifact_set_sha256(artifacts),
        "publication_receipt": copy.deepcopy(publication_receipt),
        "resume": copy.deepcopy(resume),
        "failure": copy.deepcopy(failure),
        "canonical_import": {
            "required": True,
            "status": "ready" if status == "completed" and publication_receipt else "blocked",
            "required_outputs": ["hfr.agentic_training_result.v1", "hfr.cloud_training_completion_receipt.v1"],
        },
    }
    completion = attach_identity(completion)
    errors = validate_job_completion(completion, base_dir)
    if errors:
        raise HuggingFaceLifecycleError("; ".join(errors))
    return completion


def validate_job_completion(completion: Any, base_dir: str | Path | None = None) -> list[str]:
    errors: list[str] = []
    if not isinstance(completion, dict):
        return ["job completion must be a JSON object"]
    if completion.get("schema_version") != HF_JOB_COMPLETION_SCHEMA_VERSION:
        errors.append(f"schema_version must be {HF_JOB_COMPLETION_SCHEMA_VERSION}")
    status = completion.get("status")
    if status not in COMPLETION_STATUSES or completion.get("terminal") is not True:
        errors.append("job completion must have a supported terminal status")
    job = _dict(completion.get("job"))
    if job.get("provider") != "huggingface_jobs" or not _nonempty(job.get("job_id")):
        errors.append("job completion must identify the Hugging Face job")
    if not _immutable_revision(job.get("dataset_revision")):
        errors.append("job.dataset_revision must be immutable")
    if not isinstance(job.get("exit_code"), int):
        errors.append("job.exit_code must be an integer")
    reviewed_plan = _load_reviewed_plan(
        completion.get("reviewed_plan"),
        base_dir,
        "reviewed_plan",
        errors,
    )
    runtime_declaration = _dict(completion.get("runtime_declaration"))
    runtime_observation = _dict(completion.get("runtime_observation"))
    _validate_runtime_declaration(runtime_declaration, errors)
    _validate_runtime_observation(runtime_observation, errors)
    if runtime_observation.get("context") != "training_job":
        errors.append("job completion runtime_observation.context must be training_job")
    artifacts = completion.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append("job completion artifacts must be non-empty")
        artifacts = []
    for index, row in enumerate(artifacts):
        _validate_artifact_ref(row, f"artifacts[{index}]", base_dir, errors)
    if completion.get("artifact_count") != len(artifacts):
        errors.append("job completion artifact_count does not match artifacts")
    if completion.get("artifact_set_sha256") != artifact_set_sha256(row for row in artifacts if isinstance(row, dict)):
        errors.append("job completion artifact_set_sha256 does not match artifacts")
    training_plan_artifacts = [
        row
        for row in artifacts
        if isinstance(row, dict) and str(row.get("role") or "") in {"training_plan", "reviewed_training_plan"}
    ]
    if len(training_plan_artifacts) != 1 or training_plan_artifacts[0] != completion.get("reviewed_plan"):
        errors.append("reviewed_plan path/hash must exactly match the durable training-plan artifact")
    publication = completion.get("publication_receipt")
    if isinstance(publication, dict):
        errors.extend(
            f"publication_receipt: {error}"
            for error in validate_publication_receipt(publication, base_dir)
        )
    if status == "completed":
        if job.get("exit_code") != 0:
            errors.append("completed job must have exit_code 0")
        if not isinstance(publication, dict):
            errors.append("completed job must include a publication receipt")
        roles = {str(row.get("role") or "") for row in artifacts if isinstance(row, dict)}
        for role in ("training_plan", "trainer_log", "training_metrics", "adapter_manifest", "adapter", "training_result"):
            if role not in roles:
                errors.append(f"completed job is missing durable {role}")
        canonical_import = _dict(completion.get("canonical_import"))
        if canonical_import.get("status") != "ready":
            errors.append("completed job must be ready for canonical import")
    elif completion.get("failure") is None:
        errors.append("failed or interrupted job must include failure metadata")
    if isinstance(publication, dict):
        if publication.get("artifact_set_sha256") != completion.get("artifact_set_sha256"):
            errors.append("publication_receipt must bind the completion artifact set")
        if publication.get("artifacts") != artifacts:
            errors.append("publication_receipt artifacts must exactly match completion artifacts")
        if publication.get("reviewed_plan") != completion.get("reviewed_plan"):
            errors.append("publication_receipt reviewed_plan path/hash must match job completion")
        if publication.get("runtime_observation") != runtime_observation:
            errors.append("publication_receipt runtime_observation must match job completion")
        if publication.get("runtime_declaration") != runtime_declaration:
            errors.append("publication_receipt runtime_declaration must match job completion")
    if reviewed_plan is not None:
        _validate_completion_runtime_identity(
            reviewed_plan,
            runtime_declaration,
            runtime_observation,
            publication,
            errors,
            require_execution_match=status == "completed",
        )
    _validate_resume(_dict(completion.get("resume")), errors, completion=True)
    identity = _dict(completion.get("identity"))
    if identity.get("algorithm") != "sha256" or identity.get("sha256") != payload_identity(completion):
        errors.append("job completion identity does not match")
    return errors


def _validate_runtime_declaration(runtime_declaration: dict[str, Any], errors: list[str]) -> None:
    expected_fields = {
        "python", "container_image", "hardware_flavor", "timeout", "dependencies",
        "dependency_lock_sha256", "trainer_script_sha256", "model_identity", "declaration_identity_sha256",
    }
    if set(runtime_declaration) != expected_fields:
        errors.append("runtime_declaration fields do not match the declared-runtime contract")
    if runtime_declaration.get("python") != "3.11":
        errors.append("runtime_declaration.python must be 3.11")
    if not _PINNED_IMAGE_RE.fullmatch(str(runtime_declaration.get("container_image") or "")):
        errors.append("runtime_declaration.container_image must use an immutable @sha256 digest")
    for key in ("hardware_flavor", "timeout"):
        if not _nonempty(runtime_declaration.get(key)):
            errors.append(f"runtime_declaration.{key} must be non-empty")
    dependencies = runtime_declaration.get("dependencies")
    if not isinstance(dependencies, list) or not dependencies:
        errors.append("runtime_declaration.dependencies must be a non-empty exact lock")
        dependencies = []
    for index, requirement in enumerate(dependencies):
        if not isinstance(requirement, str) or _EXACT_DEPENDENCY_RE.fullmatch(requirement) is None:
            errors.append(f"runtime_declaration.dependencies[{index}] must be exactly pinned")
    dependency_names = [str(item).split("==", 1)[0] for item in dependencies]
    if len(dependency_names) != len(set(dependency_names)):
        errors.append("runtime_declaration.dependencies must not repeat package names")
    if runtime_declaration.get("dependency_lock_sha256") != dependency_lock_sha256(dependencies):
        errors.append("runtime_declaration.dependency_lock_sha256 does not match dependencies")
    if not _is_sha256(runtime_declaration.get("trainer_script_sha256")):
        errors.append("runtime_declaration.trainer_script_sha256 must be a SHA-256 digest")
    model_identity = _dict(runtime_declaration.get("model_identity"))
    for key in ("base_model_revision", "tokenizer_revision"):
        if not _immutable_revision(model_identity.get(key)):
            errors.append(f"runtime_declaration.model_identity.{key} must be immutable")
    if not _is_sha256(model_identity.get("chat_template_sha256")):
        errors.append("runtime_declaration.model_identity.chat_template_sha256 must be a SHA-256 digest")
    expected_identity = canonical_sha256(
        {key: value for key, value in runtime_declaration.items() if key != "declaration_identity_sha256"}
    )
    if runtime_declaration.get("declaration_identity_sha256") != expected_identity:
        errors.append("runtime_declaration.declaration_identity_sha256 does not match declaration")


def _validate_runtime_observation(runtime_observation: dict[str, Any], errors: list[str]) -> None:
    expected_fields = {
        "context", "observed_at", "python", "python_implementation", "platform", "dependencies",
        "container", "hardware", "cuda", "measured_artifacts", "library_identity_sha256",
        "container_identity_sha256", "hardware_identity_sha256", "cuda_identity_sha256",
        "runtime_identity_sha256",
    }
    if set(runtime_observation) != expected_fields:
        errors.append("runtime_observation fields do not match the measured-runtime contract")
    if runtime_observation.get("context") not in {"training_job", "payload_publisher"}:
        errors.append("runtime_observation.context must identify the measuring process")
    if not _nonempty(runtime_observation.get("observed_at")):
        errors.append("runtime_observation.observed_at must be non-empty")
    for key in ("python", "python_implementation", "platform"):
        if not _nonempty(runtime_observation.get(key)):
            errors.append(f"runtime_observation.{key} must be non-empty")

    dependencies = runtime_observation.get("dependencies")
    if not isinstance(dependencies, dict) or not dependencies:
        errors.append("runtime_observation.dependencies must be a non-empty observed library map")
        dependencies = {}
    for name, value in dependencies.items():
        row = _dict(value)
        if (
            not _nonempty(name)
            or not _nonempty(row.get("actual"))
            or row.get("measurement_source") != "importlib.metadata"
            or set(row) != {"actual", "measurement_source"}
        ):
            errors.append(f"runtime_observation.dependencies.{name} must contain only measured actual version evidence")
    if runtime_observation.get("library_identity_sha256") != canonical_sha256(dependencies):
        errors.append("runtime_observation.library_identity_sha256 does not match dependencies")

    cuda = runtime_observation.get("cuda")
    if not isinstance(cuda, dict):
        errors.append("runtime_observation.cuda must be an observed CUDA identity object")
        cuda = {}
    if not isinstance(cuda.get("available"), bool):
        errors.append("runtime_observation.cuda.available must be a boolean")
    for key in ("runtime_version", "driver_version", "cudnn_version"):
        if not _nonempty(cuda.get(key)):
            errors.append(f"runtime_observation.cuda.{key} must be non-empty")
    device_count = cuda.get("device_count")
    devices = cuda.get("devices")
    if not isinstance(device_count, int) or isinstance(device_count, bool) or device_count < 0:
        errors.append("runtime_observation.cuda.device_count must be a non-negative integer")
        device_count = 0
    if not isinstance(devices, list):
        errors.append("runtime_observation.cuda.devices must be a list")
        devices = []
    if len(devices) != device_count:
        errors.append("runtime_observation.cuda.device_count must match devices")
    for index, value in enumerate(devices):
        row = _dict(value)
        if row.get("index") != index or not _nonempty(row.get("name")) or not _nonempty(row.get("compute_capability")):
            errors.append(f"runtime_observation.cuda.devices[{index}] is invalid")
    if cuda.get("available") is False and (device_count != 0 or devices):
        errors.append("runtime_observation.cuda unavailable state cannot list devices")
    if cuda.get("available") is True and device_count < 1:
        errors.append("runtime_observation.cuda available state must list a device")
    if runtime_observation.get("cuda_identity_sha256") != canonical_sha256(cuda):
        errors.append("runtime_observation.cuda_identity_sha256 does not match cuda")

    container = _dict(runtime_observation.get("container"))
    if (
        container.get("measurement_source") != "runtime_probe"
        or not isinstance(container.get("detected"), bool)
        or container.get("isolation") not in {"container", "host_or_unknown"}
        or not isinstance(container.get("marker_paths"), list)
        or not _nonempty(container.get("cgroup_sha256"))
        or not _nonempty(container.get("os_release_sha256"))
        or not _nonempty(container.get("image_digest"))
        or not _nonempty(container.get("image_digest_source"))
    ):
        errors.append("runtime_observation.container must contain runtime-probed container evidence")
    if (container.get("isolation") == "container") != (container.get("detected") is True):
        errors.append("runtime_observation.container isolation must match probe detection")
    if any(not isinstance(path, str) or not path.startswith("/") for path in container.get("marker_paths", [])):
        errors.append("runtime_observation.container marker paths must be absolute probe paths")
    image_digest = str(container.get("image_digest") or "")
    if image_digest != "unavailable" and re.fullmatch(r"sha256:[0-9a-f]{64}", image_digest) is None:
        errors.append("runtime_observation.container.image_digest must be measured or unavailable")
    if (image_digest == "unavailable") != (container.get("image_digest_source") == "unavailable"):
        errors.append("runtime_observation.container image digest and source availability must match")
    if runtime_observation.get("container_identity_sha256") != canonical_sha256(container):
        errors.append("runtime_observation.container_identity_sha256 does not match container")

    hardware = _dict(runtime_observation.get("hardware"))
    if (
        hardware.get("measurement_source") != "runtime_probe"
        or not _nonempty(hardware.get("machine"))
        or not _nonempty(hardware.get("processor"))
        or not isinstance(hardware.get("logical_cpu_count"), int)
        or isinstance(hardware.get("logical_cpu_count"), bool)
        or hardware.get("logical_cpu_count", -1) < 0
        or hardware.get("accelerator_kind") not in {"none", "cuda"}
        or not isinstance(hardware.get("accelerator_count"), int)
        or isinstance(hardware.get("accelerator_count"), bool)
        or hardware.get("accelerator_count", -1) < 0
    ):
        errors.append("runtime_observation.hardware must contain runtime-probed hardware evidence")
    if hardware.get("accelerator_count") != device_count:
        errors.append("runtime_observation.hardware accelerator count must match CUDA devices")
    if (hardware.get("accelerator_kind") == "cuda") != (cuda.get("available") is True):
        errors.append("runtime_observation.hardware accelerator kind must match CUDA availability")
    if runtime_observation.get("hardware_identity_sha256") != canonical_sha256(hardware):
        errors.append("runtime_observation.hardware_identity_sha256 does not match hardware")

    measured_artifacts = _dict(runtime_observation.get("measured_artifacts"))
    trainer_script_sha256 = measured_artifacts.get("trainer_script_sha256")
    if trainer_script_sha256 != "unavailable" and not _is_sha256(trainer_script_sha256):
        errors.append("runtime_observation.measured_artifacts.trainer_script_sha256 must be measured or unavailable")
    expected_runtime_identity = canonical_sha256(
        {
            "context": runtime_observation.get("context"),
            "python": runtime_observation.get("python"),
            "python_implementation": runtime_observation.get("python_implementation"),
            "platform": runtime_observation.get("platform"),
            "library_identity_sha256": runtime_observation.get("library_identity_sha256"),
            "cuda_identity_sha256": runtime_observation.get("cuda_identity_sha256"),
            "container_identity_sha256": runtime_observation.get("container_identity_sha256"),
            "hardware_identity_sha256": runtime_observation.get("hardware_identity_sha256"),
            "measured_artifacts": measured_artifacts,
        }
    )
    if runtime_observation.get("runtime_identity_sha256") != expected_runtime_identity:
        errors.append("runtime_observation.runtime_identity_sha256 does not match observed runtime")


def _validate_completion_runtime_identity(
    plan: dict[str, Any],
    runtime_declaration: dict[str, Any],
    runtime_observation: dict[str, Any],
    publication: Any,
    errors: list[str],
    *,
    require_execution_match: bool,
) -> None:
    runtime = _dict(plan.get("runtime"))
    expected_declaration = runtime_declaration_from_plan(plan)
    if runtime_declaration != expected_declaration:
        errors.append("runtime_declaration must exactly match the reviewed plan")
    observed_dependencies = _dict(runtime_observation.get("dependencies"))
    expected_dependencies = {
        str(requirement).split("==", 1)[0]: str(requirement).split("==", 1)[1]
        for requirement in runtime.get("dependencies", [])
    }
    if require_execution_match and set(observed_dependencies) != set(expected_dependencies):
        errors.append("runtime_observation dependencies must exactly match the reviewed dependency names")
    for name, expected in expected_dependencies.items():
        row = _dict(observed_dependencies.get(name))
        if require_execution_match and row.get("actual") != expected:
            errors.append(f"runtime_observation dependency drift: {name}")
    measured_trainer_sha = _dict(runtime_observation.get("measured_artifacts")).get("trainer_script_sha256")
    if require_execution_match and measured_trainer_sha != runtime.get("trainer_script_sha256"):
        errors.append("runtime_observation measured trainer script must match the reviewed plan")
    observed_python = str(runtime_observation.get("python") or "")
    if require_execution_match and not observed_python.startswith(str(runtime.get("python") or "") + "."):
        errors.append("runtime_observation Python version must match the reviewed declaration")
    container = _dict(runtime_observation.get("container"))
    if require_execution_match and container.get("detected") is not True:
        errors.append("completed Hugging Face job must observe container isolation")
    observed_digest = container.get("image_digest")
    declared_digest = str(runtime.get("container_image") or "").rsplit("@", 1)[-1]
    if observed_digest != "unavailable" and observed_digest != declared_digest:
        errors.append("runtime_observation container image digest conflicts with the reviewed declaration")
    cuda = _dict(runtime_observation.get("cuda"))
    if require_execution_match and not _cpu_or_tpu_flavor(str(runtime.get("hardware_flavor") or "")):
        if cuda.get("available") is not True:
            errors.append("completed GPU job must observe CUDA as available")
        if cuda.get("driver_version") == "unavailable" or cuda.get("runtime_version") == "unavailable":
            errors.append("completed GPU job must observe CUDA runtime and driver versions")
        if not isinstance(cuda.get("device_count"), int) or cuda.get("device_count", 0) < 1:
            errors.append("completed GPU job must observe at least one CUDA device")
    if (
        isinstance(publication, dict)
        and publication.get("source_plan_sha256") != _dict(plan.get("identity")).get("sha256")
    ):
        errors.append("publication_receipt.source_plan_sha256 must match the reviewed plan identity")


def _validate_hyperparameters(params: dict[str, Any], trainer: dict[str, Any], errors: list[str]) -> None:
    required = {
        "sft_epochs",
        "dpo_epochs",
        "sft_learning_rate",
        "dpo_learning_rate",
        "batch_size",
        "gradient_accumulation_steps",
        "gradient_checkpointing",
        "max_steps",
        "max_length",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "assistant_only_loss",
        "seed",
        "data_seed",
        "save_steps",
        "save_total_limit",
    }
    missing = sorted(required - set(params))
    if missing:
        errors.append(f"trainer.hyperparameters missing {missing}")
        return
    positive_numbers = ("sft_epochs", "dpo_epochs", "sft_learning_rate", "dpo_learning_rate")
    if any(not _positive_number(params.get(key)) for key in positive_numbers):
        errors.append("trainer epoch and learning-rate values must be positive")
    for key in ("batch_size", "gradient_accumulation_steps", "max_length", "lora_r", "lora_alpha", "save_steps", "save_total_limit"):
        if not isinstance(params.get(key), int) or params[key] <= 0:
            errors.append(f"trainer.hyperparameters.{key} must be a positive integer")
    for key in ("seed", "data_seed"):
        if not isinstance(params.get(key), int) or params[key] < 0:
            errors.append(f"trainer.hyperparameters.{key} must be a non-negative integer")
    if not isinstance(params.get("max_steps"), int) or params["max_steps"] == 0 or params["max_steps"] < -1:
        errors.append("trainer.hyperparameters.max_steps must be -1 or a positive integer")
    if not _number(params.get("lora_dropout")) or not 0 <= float(params["lora_dropout"]) < 1:
        errors.append("trainer.hyperparameters.lora_dropout must be in [0, 1)")
    if not isinstance(params.get("gradient_checkpointing"), bool) or not isinstance(params.get("assistant_only_loss"), bool):
        errors.append("trainer boolean hyperparameters must be booleans")
    expected_loss = "assistant_only" if params.get("assistant_only_loss") is True else "all_messages"
    if trainer.get("loss_scope") != expected_loss:
        errors.append("trainer.loss_scope must match hyperparameters.assistant_only_loss")


def _validate_resume(resume: dict[str, Any], errors: list[str], *, completion: bool = False) -> None:
    if not isinstance(resume.get("enabled"), bool):
        errors.append("resume.enabled must be a boolean")
        return
    if resume.get("enabled") is True:
        if resume.get("phase") not in RESUME_PHASES:
            errors.append("resume.phase must be sft or dpo")
        if not _immutable_revision(resume.get("revision")):
            errors.append("resume.revision must be immutable")
        if not _safe_repo_path(str(resume.get("checkpoint_path") or "")):
            errors.append("resume.checkpoint_path must be a safe relative repository path")
        if not _is_sha256(resume.get("checkpoint_manifest_sha256")):
            errors.append("resume.checkpoint_manifest_sha256 must be a SHA-256 digest")
    elif not completion:
        expected = {"enabled": False, "phase": "", "revision": "", "checkpoint_path": "", "checkpoint_manifest_sha256": ""}
        if resume != expected:
            errors.append("disabled resume metadata must use empty immutable-source fields")


def _normalized_resume(value: dict[str, Any] | None) -> dict[str, Any]:
    if not value or value.get("enabled") is not True:
        return {"enabled": False, "phase": "", "revision": "", "checkpoint_path": "", "checkpoint_manifest_sha256": ""}
    return {
        "enabled": True,
        "phase": str(value.get("phase") or ""),
        "revision": str(value.get("revision") or ""),
        "checkpoint_path": str(value.get("checkpoint_path") or ""),
        "checkpoint_manifest_sha256": str(value.get("checkpoint_manifest_sha256") or ""),
    }


def _read_recorded_json(record: dict[str, Any], base_dir: str | Path, label: str) -> dict[str, Any]:
    errors: list[str] = []
    _validate_artifact_ref(record, label, base_dir, errors)
    if errors:
        raise HuggingFaceLifecycleError("; ".join(errors))
    path = Path(base_dir) / str(record["path"])
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HuggingFaceLifecycleError(f"{label} could not be replayed: {exc}") from exc
    if not isinstance(value, dict):
        raise HuggingFaceLifecycleError(f"{label} must contain a JSON object")
    return value


def _load_reviewed_plan(
    value: Any,
    base_dir: str | Path | None,
    label: str,
    errors: list[str],
) -> dict[str, Any] | None:
    _validate_artifact_ref(value, label, base_dir, errors)
    if base_dir is None:
        errors.append(f"{label} path/hash binding requires base_dir")
        return None
    record = _dict(value)
    path_value = record.get("path")
    if not isinstance(path_value, str) or not _safe_repo_path(path_value):
        return None
    path = Path(base_dir) / path_value
    if not path.is_file() or path.is_symlink():
        return None
    try:
        reviewed_plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"reviewed plan could not be replayed: {exc}")
        return None
    plan_errors = validate_reviewed_training_plan(reviewed_plan)
    errors.extend(f"reviewed plan: {error}" for error in plan_errors)
    return reviewed_plan if isinstance(reviewed_plan, dict) else None


def _validate_artifact_ref(value: Any, label: str, base_dir: str | Path | None, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{label} must be an artifact object")
        return
    path_value = value.get("path")
    if not isinstance(path_value, str) or not _safe_repo_path(path_value):
        errors.append(f"{label}.path must be a safe relative path")
    if not _is_sha256(value.get("sha256")):
        errors.append(f"{label}.sha256 must be a SHA-256 digest")
    if not isinstance(value.get("size_bytes"), int) or value["size_bytes"] < 0:
        errors.append(f"{label}.size_bytes must be a non-negative integer")
    if base_dir is None or not isinstance(path_value, str) or not _safe_repo_path(path_value):
        return
    path = Path(base_dir) / path_value
    if not path.exists() or not path.is_file() or path.is_symlink():
        errors.append(f"{label}.path does not resolve to a regular file")
        return
    if path.stat().st_size != value.get("size_bytes") or sha256_file(path) != value.get("sha256"):
        errors.append(f"{label} fingerprint is stale")


def _safe_repo_path(value: str) -> bool:
    if not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and all(part not in {"", ".", ".."} for part in path.parts)


def _immutable_revision(value: Any) -> bool:
    return isinstance(value, str) and _HUB_COMMIT_RE.fullmatch(value) is not None


def _cpu_or_tpu_flavor(value: str) -> bool:
    return value.startswith("cpu-") or value.startswith("v5e-")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _positive_number(value: Any) -> bool:
    return _number(value) and float(value) > 0


def _number_text(value: Any) -> str:
    return str(value)
