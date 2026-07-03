"""Runtime preflight checks for agentic training handoffs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schema_registry import SchemaRegistryError, check_schema_file, check_schema_jsonl_file

AGENTIC_TRAINING_RUNTIME_PREFLIGHT_SCHEMA_VERSION = "hfr.agentic_training_runtime_preflight.v1"

PLAN_READY_RECOMMENDATION = "ready_for_external_trainer_plan"
RUNTIME_READY_RECOMMENDATION = "ready_for_tiny_smoke_launch"
RUNTIME_BLOCK_RECOMMENDATION = "block_tiny_smoke_launch"

BACKEND_MODULE_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "external": (),
    "axolotl": ("axolotl", "torch", "transformers", "datasets", "peft", "trl", "accelerate", "yaml"),
    "llama_factory": ("llamafactory", "torch", "transformers", "datasets", "peft", "accelerate"),
    "llamafactory": ("llamafactory", "torch", "transformers", "datasets", "peft", "accelerate"),
    "unsloth": ("unsloth", "torch", "transformers", "datasets", "peft", "trl"),
    "process_reward_trainer": ("torch", "transformers", "datasets"),
    "process_reward_wrapper": ("torch", "transformers", "datasets"),
}

VIEW_SCHEMA_DEFAULTS: dict[str, str] = {
    "action_sft": "rl_sft",
    "dpo": "rl_dpo",
    "episodes": "rl_episode",
    "preferences": "rl_preference",
    "process_rewards": "rl_step_reward",
    "reward_model": "rl_reward_model",
    "rollouts": "rl_episode",
    "sft": "rl_sft",
    "step_rewards": "rl_step_reward",
}


class AgenticTrainingRuntimePreflightError(ValueError):
    """Raised when a runtime preflight artifact cannot be written."""


def build_agentic_training_runtime_preflight(
    *,
    plan_path: str | Path,
    out_path: str | Path | None = None,
    require_modules: list[str] | tuple[str, ...] | None = None,
    skip_default_modules: bool = False,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a side-effect-free runtime readiness artifact for a training plan."""
    plan_file = Path(plan_path)
    plan_payload, plan_read_errors = _read_json_object(plan_file)
    plan_schema_check = _json_schema_record(plan_file, "agentic_training_plan")
    trainer_plan = plan_payload.get("trainer_plan") if isinstance(plan_payload.get("trainer_plan"), dict) else {}
    mode = str(plan_payload.get("mode") or "")
    backend = str(trainer_plan.get("backend") or "external")

    plan_ready = (
        not plan_read_errors
        and plan_schema_check["passed"]
        and plan_payload.get("passed") is True
        and plan_payload.get("recommendation") == PLAN_READY_RECOMMENDATION
    )
    dependency_checks = _dependency_checks(backend, require_modules or (), skip_default_modules)
    view_checks = _view_checks(plan_payload, plan_file)

    checks: list[dict[str, Any]] = []
    _add_check(
        checks,
        "plan_json_readable",
        not plan_read_errors,
        {"errors": plan_read_errors},
        {"errors": []},
    )
    _add_check(
        checks,
        "plan_schema_passed",
        plan_schema_check["passed"],
        {"error_count": plan_schema_check["error_count"], "errors": plan_schema_check["errors"]},
        {"schema_name": "agentic_training_plan", "error_count": 0},
    )
    _add_check(
        checks,
        "plan_recommendation_ready",
        plan_ready,
        {"passed": plan_payload.get("passed"), "recommendation": plan_payload.get("recommendation")},
        {"passed": True, "recommendation": PLAN_READY_RECOMMENDATION},
    )
    _add_check(
        checks,
        "selected_views_schema_passed",
        bool(view_checks) and all(view["passed"] for view in view_checks),
        {
            "view_count": len(view_checks),
            "failed_views": [view["name"] for view in view_checks if not view["passed"]],
        },
        {"all_selected_views_pass": True},
    )
    _add_check(
        checks,
        "runtime_dependencies_available",
        all(check["passed"] for check in dependency_checks),
        {
            "backend": backend,
            "missing_modules": [check["module"] for check in dependency_checks if not check["passed"]],
        },
        {"all_required_modules_available": True},
    )
    _add_check(
        checks,
        "flight_recorder_did_not_launch_training",
        True,
        {"training_started": False},
        {"training_started": False},
    )
    _add_check(
        checks,
        "model_downloads_not_started",
        True,
        {"model_downloads_started": False},
        {"model_downloads_started": False},
    )

    failed_checks = [check for check in checks if check["passed"] is False]
    passed = not failed_checks
    preflight = {
        "schema_version": AGENTIC_TRAINING_RUNTIME_PREFLIGHT_SCHEMA_VERSION,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "artifact_path": str(out_path or ""),
        "plan_path": str(plan_file),
        "plan_sha256": _sha256_or_none(plan_file),
        "plan_mode": mode,
        "backend": backend,
        "passed": passed,
        "readiness": "ready" if passed else "blocked",
        "recommendation": RUNTIME_READY_RECOMMENDATION if passed else RUNTIME_BLOCK_RECOMMENDATION,
        "check_count": len(checks),
        "failed_check_count": len(failed_checks),
        "checks": checks,
        "blocked_reasons": _blocked_reasons(failed_checks, dependency_checks, view_checks),
        "plan_check": {
            "path": str(plan_file),
            "exists": plan_file.exists(),
            "regular_file": plan_file.is_file(),
            "schema_name": "agentic_training_plan",
            "schema_passed": plan_schema_check["passed"],
            "error_count": plan_schema_check["error_count"],
            "errors": plan_schema_check["errors"],
            "required_recommendation": PLAN_READY_RECOMMENDATION,
            "observed_recommendation": plan_payload.get("recommendation"),
            "observed_passed": plan_payload.get("passed"),
        },
        "dependency_checks": dependency_checks,
        "view_checks": view_checks,
        "execution_boundary": {
            "dry_run_preflight_only": True,
            "flight_recorder_launched_training": False,
            "training_started": False,
            "model_downloads_started": False,
            "trainer_modules_imported": False,
            "dependency_resolution_method": "importlib.util.find_spec",
        },
        "handoff_contract": {
            "runner_owns_execution": True,
            "runner_must_require_recommendation": RUNTIME_READY_RECOMMENDATION,
            "requires_registered_inputs": True,
            "requires_registered_model": True,
            "requires_registered_dataset": True,
            "requires_known_license_status": True,
            "requires_redacted_dataset": True,
            "disallow_unredacted_traces": True,
            "flight_recorder_launched_training": False,
            "model_downloads_started": False,
        },
        "notes": [
            "This artifact checks local readiness for a bounded tiny smoke launch only.",
            "Flight Recorder did not import trainer packages, download models, mutate weights, or launch training.",
            "External runners must revalidate manifests, redaction, license status, file hashes, and dependency versions immediately before execution.",
        ],
    }
    return preflight


def write_agentic_training_runtime_preflight(path: str | Path, preflight: dict[str, Any]) -> None:
    """Write a deterministic JSON runtime preflight artifact."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json_object(path: Path) -> tuple[dict[str, Any], list[str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}, [f"file not found: {path}"]
    except json.JSONDecodeError as exc:
        return {}, [f"invalid JSON: {exc.msg}"]
    except OSError as exc:
        return {}, [str(exc)]
    if not isinstance(payload, dict):
        return {}, ["plan must contain a JSON object"]
    return payload, []


def _json_schema_record(path: Path, schema_name: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "kind": "json",
        "schema_name": schema_name,
        "passed": False,
        "error_count": 0,
        "errors": [],
    }
    try:
        result = check_schema_file(path, schema_name)
    except (OSError, json.JSONDecodeError, SchemaRegistryError) as exc:
        record["errors"] = [str(exc)]
        record["error_count"] = 1
        return record
    record["passed"] = result.get("passed") is True
    record["error_count"] = _int_value(result.get("error_count"))
    record["errors"] = [str(error) for error in result.get("errors", []) if isinstance(error, str)][:20]
    return record


def _jsonl_schema_record(path: Path, schema_name: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "kind": "jsonl",
        "schema_name": schema_name,
        "passed": False,
        "error_count": 0,
        "errors": [],
        "row_count": 0,
        "row_schema_counts": [],
    }
    try:
        result = check_schema_jsonl_file(path, schema_name)
    except (OSError, json.JSONDecodeError, SchemaRegistryError) as exc:
        record["errors"] = [str(exc)]
        record["error_count"] = 1
        return record
    record["passed"] = result.get("passed") is True
    record["error_count"] = _int_value(result.get("error_count"))
    record["errors"] = [str(error) for error in result.get("errors", []) if isinstance(error, str)][:20]
    record["row_count"] = _int_value(result.get("row_count"))
    record["row_schema_counts"] = result.get("row_schema_counts") if isinstance(result.get("row_schema_counts"), list) else []
    return record


def _dependency_checks(
    backend: str,
    require_modules: list[str] | tuple[str, ...],
    skip_default_modules: bool,
) -> list[dict[str, Any]]:
    modules: dict[str, set[str]] = {}
    if not skip_default_modules:
        for module in BACKEND_MODULE_REQUIREMENTS.get(_normalize_backend(backend), ()):
            modules.setdefault(module, set()).add("backend_default")
    for module in require_modules:
        cleaned = str(module).strip()
        if cleaned:
            modules.setdefault(cleaned, set()).add("override")

    checks: list[dict[str, Any]] = []
    for module in sorted(modules):
        available = _module_available(module)
        checks.append(
            {
                "module": module,
                "sources": sorted(modules[module]),
                "required": True,
                "available": available,
                "passed": available,
                "summary": f"{module}: available={available}",
            }
        )
    return checks


def _view_checks(plan: dict[str, Any], plan_path: Path) -> list[dict[str, Any]]:
    selected_views = plan.get("selected_views") if isinstance(plan.get("selected_views"), list) else []
    checks: list[dict[str, Any]] = []
    for index, raw_view in enumerate(selected_views):
        if not isinstance(raw_view, dict):
            continue
        name = str(raw_view.get("name") or f"view_{index}")
        raw_path = str(raw_view.get("path") or "")
        expected_rows = _int_value(raw_view.get("row_count"))
        declared_schema = str(raw_view.get("schema_version") or "")
        schema_name = declared_schema or VIEW_SCHEMA_DEFAULTS.get(name, "")
        resolved_path = _resolve_view_path(raw_path, plan, plan_path)
        regular_file = resolved_path.is_file()
        schema_record = _jsonl_schema_record(resolved_path, schema_name) if regular_file and schema_name else {}
        observed_rows = _int_value(schema_record.get("row_count")) if schema_record else 0
        row_count_matches = expected_rows == 0 or observed_rows == expected_rows
        errors: list[str] = []
        if not raw_path:
            errors.append("view path is empty")
        if not schema_name:
            errors.append(f"no bundled schema mapping for view {name!r}")
        if not regular_file:
            errors.append("view path is not a regular file")
        errors.extend(str(error) for error in schema_record.get("errors", []) if isinstance(error, str))
        if schema_record and not row_count_matches:
            errors.append(f"row_count mismatch: manifest={expected_rows}, observed={observed_rows}")

        passed = bool(raw_path) and bool(schema_name) and regular_file and schema_record.get("passed") is True and row_count_matches
        checks.append(
            {
                "name": name,
                "path": raw_path,
                "resolved_path": _display_path(resolved_path),
                "exists": resolved_path.exists(),
                "regular_file": regular_file,
                "schema_name": schema_name,
                "schema_version_declared": declared_schema,
                "expected_row_count": expected_rows,
                "observed_row_count": observed_rows,
                "row_count_matches_manifest": row_count_matches,
                "sha256": _sha256_or_none(resolved_path),
                "passed": passed,
                "error_count": len(errors),
                "errors": errors[:20],
                "row_schema_counts": schema_record.get("row_schema_counts", []),
            }
        )
    return checks


def _resolve_view_path(raw_path: str, plan: dict[str, Any], plan_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidates: list[Path] = [plan_path.parent / path]
    candidates.extend(parent / path for parent in plan_path.parents)
    dataset_manifest = _manifest_path(plan, "dataset", plan_path)
    if dataset_manifest is not None:
        candidates.append(dataset_manifest.parent / path)
    deduped: list[Path] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return next((candidate for candidate in deduped if candidate.exists()), deduped[0] if deduped else path)


def _display_path(path: Path) -> str:
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name


def _manifest_path(plan: dict[str, Any], key: str, plan_path: Path) -> Path | None:
    manifests = plan.get("input_manifests")
    if not isinstance(manifests, dict):
        return None
    record = manifests.get(key)
    if not isinstance(record, dict):
        return None
    raw_path = str(record.get("path") or "")
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidates = [plan_path.parent / path]
    candidates.extend(parent / path for parent in plan_path.parents)
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def _module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _normalize_backend(backend: str) -> str:
    return backend.strip().lower().replace("-", "_")


def _blocked_reasons(
    failed_checks: list[dict[str, Any]],
    dependency_checks: list[dict[str, Any]],
    view_checks: list[dict[str, Any]],
) -> list[str]:
    reasons = [str(check["summary"]) for check in failed_checks]
    reasons.extend(f"missing dependency module: {check['module']}" for check in dependency_checks if not check["passed"])
    for view in view_checks:
        if not view["passed"]:
            detail = "; ".join(view["errors"]) if view["errors"] else "view did not pass"
            reasons.append(f"view {view['name']} blocked: {detail}")
    return reasons


def _add_check(
    checks: list[dict[str, Any]],
    check_id: str,
    passed: bool,
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    checks.append(
        {
            "id": check_id,
            "passed": bool(passed),
            "actual": actual,
            "expected": expected,
            "summary": f"{check_id}: passed={bool(passed)}",
        }
    )


def _int_value(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _sha256_or_none(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
