"""Fail-closed plans for external eval harness adapters."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

from .dependency_probe import module_available_without_import

EXTERNAL_EVAL_PLAN_SCHEMA_VERSION = "hfr.external_eval_adapters.v1"
EXTERNAL_EVAL_RECEIPT_SCHEMA_VERSION = "hfr.external_eval_receipt.v1"
EXTERNAL_EVAL_ADAPTER_CONTRACT_VERSION = "hfr.external_eval_adapter_contract.v1"

EXTERNAL_EVAL_ADAPTER_RECEIPT_TYPES: tuple[str, ...] = (
    EXTERNAL_EVAL_PLAN_SCHEMA_VERSION,
    EXTERNAL_EVAL_RECEIPT_SCHEMA_VERSION,
)

ADAPTERS: dict[str, dict[str, Any]] = {
    "bfcl": {
        "name": "BFCL",
        "full_name": "Berkeley Function Calling Leaderboard",
        "domain": "tool_calling",
        "suite_tags": ["function_calling", "tool_calling"],
        "import_names": ["bfcl_eval"],
        "command_names": ["bfcl"],
        "required_inputs": ["scenario_manifest", "model_endpoint", "tool_schema_set"],
        "execution_boundary": "Function/tool-calling suites must use the same held-out scenario manifest and declared tool schema set.",
    },
    "inspect_ai": {
        "name": "Inspect AI",
        "full_name": "Inspect AI",
        "domain": "agentic_tasks",
        "suite_tags": ["agentic", "red_team", "inspect"],
        "import_names": ["inspect_ai"],
        "command_names": ["inspect"],
        "required_inputs": ["scenario_manifest", "model_endpoint", "inspect_task_set", "sandbox_policy"],
        "execution_boundary": "Inspect tasks must be mapped to held-out scenario IDs and run under an explicit sandbox policy.",
    },
    "lm_eval_harness": {
        "name": "lm-evaluation-harness",
        "full_name": "EleutherAI lm-evaluation-harness",
        "domain": "language_model_capability",
        "suite_tags": ["lm_eval", "capability"],
        "import_names": ["lm_eval"],
        "command_names": ["lm_eval"],
        "required_inputs": ["scenario_manifest", "model_endpoint", "lm_eval_task_list"],
        "execution_boundary": "Language-model harness tasks must be tied back to the held-out scenario manifest before comparison.",
    },
    "swe_bench": {
        "name": "SWE-bench",
        "full_name": "SWE-bench",
        "domain": "software_engineering",
        "suite_tags": ["software_engineering", "repository_tasks"],
        "import_names": ["swebench"],
        "command_names": ["swebench"],
        "required_inputs": ["scenario_manifest", "model_endpoint", "swe_bench_task_set", "sandbox_policy"],
        "execution_boundary": "Repository tasks must be selected from a declared held-out task set and run under an explicit sandbox policy.",
    },
    "local_mock": {
        "name": "Local mock eval",
        "full_name": "Flight Recorder local mock external eval",
        "domain": "offline_agentic_tasks",
        "suite_tags": ["agentic", "mock", "offline"],
        "import_names": [],
        "command_names": [],
        "built_in_dependency": True,
        "required_inputs": ["scenario_manifest", "model_endpoint"],
        "execution_boundary": "Local mock eval receipts replay committed held-out fixtures only and never start live benchmarks.",
    },
}


class ExternalEvalPlanError(ValueError):
    """Raised when an external eval adapter plan cannot be built."""


def build_external_eval_plan(
    *,
    adapters: list[str] | None = None,
    scenario_manifest: str | Path | None = None,
    model_endpoint: str | None = None,
    model: str | None = None,
    tool_schema_set: str | None = None,
    inspect_task_set: str | None = None,
    lm_eval_task_list: list[str] | None = None,
    swe_bench_task_set: str | None = None,
    sandbox_policy: str | None = None,
    allow_installed: bool = False,
    preserve_paths: bool = False,
    output_base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build a readiness plan for optional external eval harness adapters."""
    selected = _selected_adapters(adapters)
    display_base_dir = Path(output_base_dir) if output_base_dir is not None else None
    inputs = _inputs(
        scenario_manifest=scenario_manifest,
        model_endpoint=model_endpoint,
        model=model,
        tool_schema_set=tool_schema_set,
        inspect_task_set=inspect_task_set,
        lm_eval_task_list=lm_eval_task_list or [],
        swe_bench_task_set=swe_bench_task_set,
        sandbox_policy=sandbox_policy,
        preserve_paths=preserve_paths,
        output_base_dir=display_base_dir,
    )
    adapter_rows = [_adapter_plan(adapter_id, inputs, allow_installed) for adapter_id in selected]
    blocking_reasons = _blocking_reasons(adapter_rows)
    ready = bool(adapter_rows) and not blocking_reasons
    return {
        "schema_version": EXTERNAL_EVAL_PLAN_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ready": ready,
        "adapter_count": len(adapter_rows),
        "ready_adapter_count": sum(1 for row in adapter_rows if row["ready"]),
        "selected_adapters": selected,
        "allow_installed": allow_installed,
        "inputs": inputs,
        "adapters": adapter_rows,
        "blocking_reasons": blocking_reasons,
        "governance_handoff": {
            "external_eval_claims_allowed": ready,
            "requires_identical_heldout_scenarios": True,
            "recommendation": (
                "External eval adapters are ready to execute against the declared held-out scenario manifest."
                if ready
                else "Keep external eval claims disabled until every selected adapter is ready."
            ),
        },
    }


def adapter_choices() -> list[str]:
    """Return supported adapter IDs for argparse choices and docs."""
    return sorted(ADAPTERS)


def build_external_eval_receipt(
    *,
    plan_path: str | Path,
    adapters: list[str] | None = None,
    live: bool = False,
    created_at: str | None = None,
    preserve_paths: bool = False,
    output_base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build a dry-run or blocked live external-eval receipt."""
    path = Path(plan_path)
    display_base_dir = Path(output_base_dir) if output_base_dir is not None else None
    source_plan = _plan_ref(path, preserve_paths, display_base_dir)
    plan = _read_plan(path) if source_plan["exists"] else {}
    selected = sorted(set(adapters or plan.get("selected_adapters") or ADAPTERS))
    unknown = sorted(set(selected) - set(ADAPTERS))
    if unknown:
        raise ExternalEvalPlanError(f"Unknown external eval adapter(s): {', '.join(unknown)}")
    plan_adapters = {row.get("id"): row for row in plan.get("adapters", []) if isinstance(row, dict)}
    adapter_receipts = [_adapter_receipt(adapter_id, plan_adapters.get(adapter_id), live) for adapter_id in selected]
    checks: list[dict[str, Any]] = []
    _add_receipt_check(
        checks,
        "plan_schema_supported",
        plan.get("schema_version") == EXTERNAL_EVAL_PLAN_SCHEMA_VERSION,
        {"schema_version": plan.get("schema_version")},
        {"schema_version": EXTERNAL_EVAL_PLAN_SCHEMA_VERSION},
    )
    _add_receipt_check(
        checks,
        "plan_ready_for_live_execution",
        bool(plan.get("ready")) and all(row["ready"] for row in plan.get("adapters", []) if isinstance(row, dict)),
        {"plan_ready": plan.get("ready"), "ready_adapter_count": plan.get("ready_adapter_count")},
        {"plan_ready": True, "all_selected_adapters_ready": True},
    )
    _add_receipt_check(checks, "live_benchmark_not_requested", not live, {"live": live}, {"live": False})
    _add_receipt_check(
        checks,
        "external_benchmark_not_started",
        True,
        {"live_benchmarks_started": False, "provider_api_called": False},
        {"live_benchmarks_started": False, "provider_api_called": False},
    )
    failed = [check for check in checks if not check["passed"]]
    return {
        "schema_version": EXTERNAL_EVAL_RECEIPT_SCHEMA_VERSION,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "passed": not failed,
        "readiness": "dry_run_recorded" if not failed else "blocked",
        "recommendation": "archive_external_eval_dry_run" if not failed else "keep_external_eval_claims_disabled",
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "checks": checks,
        "blocked_reasons": [check["summary"] for check in failed],
        "source_plan": source_plan,
        "adapter_count": len(adapter_receipts),
        "ready_adapter_count": sum(1 for row in adapter_receipts if row["ready"]),
        "adapter_receipts": adapter_receipts,
        "launch": {
            "mode": "live" if live else "dry_run",
            "live_benchmarks_started": False,
            "provider_api_called": False,
            "model_downloads_started": False,
            "cost_incurred_usd": 0,
        },
        "execution_boundary": {
            "dry_run_only": True,
            "live_benchmarks_started": False,
            "provider_api_called": False,
            "model_downloads_started": False,
            "cloud_cost_incurred_usd": 0,
            "credential_values_recorded": False,
            "weights_updated_by_flight_recorder": False,
        },
        "notes": [
            "External eval receipts archive readiness and dry-run intent only; they do not run BFCL, Inspect AI, lm-eval, or SWE-bench.",
            "Live benchmark execution remains blocked unless an external runner separately opts in and archives its own receipt.",
        ],
    }


def write_external_eval_receipt(receipt: dict[str, Any], out_path: str | Path, *, preserve_paths: bool = False) -> None:
    """Write an external eval receipt as stable JSON."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _receipt_for_write(receipt, path, preserve_paths)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _selected_adapters(adapters: list[str] | None) -> list[str]:
    selected = sorted(set(adapters or ADAPTERS))
    unknown = sorted(set(selected) - set(ADAPTERS))
    if unknown:
        raise ExternalEvalPlanError(f"Unknown external eval adapter(s): {', '.join(unknown)}")
    if not selected:
        raise ExternalEvalPlanError("At least one external eval adapter must be selected")
    return selected


def _inputs(
    *,
    scenario_manifest: str | Path | None,
    model_endpoint: str | None,
    model: str | None,
    tool_schema_set: str | None,
    inspect_task_set: str | None,
    lm_eval_task_list: list[str],
    swe_bench_task_set: str | None,
    sandbox_policy: str | None,
    preserve_paths: bool,
    output_base_dir: Path | None,
) -> dict[str, Any]:
    manifest = _manifest_record(scenario_manifest, preserve_paths, output_base_dir)
    return {
        "scenario_manifest": manifest,
        "model_endpoint": model_endpoint or None,
        "model": model or None,
        "tool_schema_set": tool_schema_set or None,
        "inspect_task_set": inspect_task_set or None,
        "lm_eval_task_list": [task for task in lm_eval_task_list if task],
        "swe_bench_task_set": swe_bench_task_set or None,
        "sandbox_policy": sandbox_policy or None,
    }


def _manifest_record(path: str | Path | None, preserve_paths: bool, output_base_dir: Path | None) -> dict[str, Any]:
    if path is None:
        return _missing_manifest_record(None)
    manifest_path = Path(path)
    if output_base_dir is not None:
        display_path, replayable = _source_display_path(manifest_path, preserve_paths, output_base_dir)
    else:
        display_path, replayable = _display_path(manifest_path, preserve_paths), True
    exists = manifest_path.exists() and manifest_path.is_file()
    if not exists or not replayable:
        return _missing_manifest_record(display_path)
    return {
        "path": display_path,
        "exists": True,
        "sha256": _sha256(manifest_path) if exists else None,
        "size_bytes": manifest_path.stat().st_size if exists else None,
        **(_manifest_metadata(manifest_path) if exists else {"schema_version": None, "ready": None, "scenario_count": None}),
    }


def _adapter_plan(adapter_id: str, inputs: dict[str, Any], allow_installed: bool) -> dict[str, Any]:
    spec = ADAPTERS[adapter_id]
    dependency = _dependency_status(spec)
    input_blockers = [reason for name in spec["required_inputs"] for reason in _input_blockers(inputs, name)]
    blocking_reasons: list[str] = []
    if not dependency["available"]:
        blocking_reasons.append("dependencies_missing")
    if not allow_installed:
        blocking_reasons.append("adapter_disabled_until_allow_installed")
    blocking_reasons.extend(input_blockers)
    ready = not blocking_reasons
    return {
        "id": adapter_id,
        "name": spec["name"],
        "full_name": spec["full_name"],
        "domain": spec["domain"],
        "suite_tags": spec["suite_tags"],
        "required_inputs": spec["required_inputs"],
        "provided_inputs": [name for name in spec["required_inputs"] if _input_present(inputs, name)],
        "dependency_status": dependency,
        "execution_contract": {
            "requires_identical_heldout_scenarios": True,
            "scenario_manifest_sha256": inputs["scenario_manifest"]["sha256"],
            "boundary": spec["execution_boundary"],
        },
        "adapter_contract": _adapter_contract(adapter_id),
        "ready": ready,
        "blocking_reasons": blocking_reasons,
    }


def _dependency_status(spec: dict[str, Any]) -> dict[str, Any]:
    if spec.get("built_in_dependency") is True:
        return {
            "available": True,
            "imports": {},
            "commands": {},
        }
    imports = {name: module_available_without_import(name) for name in spec["import_names"]}
    commands = {name: shutil.which(name) is not None for name in spec["command_names"]}
    return {
        "available": any(imports.values()) or any(commands.values()),
        "imports": imports,
        "commands": commands,
    }


def _manifest_metadata(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": None, "ready": False, "scenario_count": None}
    if not isinstance(payload, dict):
        return {"schema_version": None, "ready": False, "scenario_count": None}
    ready = payload.get("ready") if isinstance(payload.get("ready"), bool) else None
    scenario_count = payload.get("scenario_count") if isinstance(payload.get("scenario_count"), int) else None
    return {
        "schema_version": payload.get("schema_version") if isinstance(payload.get("schema_version"), str) else None,
        "ready": ready,
        "scenario_count": scenario_count,
    }


def _missing_manifest_record(path: str | None) -> dict[str, Any]:
    return {
        "path": path,
        "exists": False,
        "sha256": None,
        "size_bytes": None,
        "schema_version": None,
        "ready": None,
        "scenario_count": None,
    }


def _input_blockers(inputs: dict[str, Any], name: str) -> list[str]:
    if _input_present(inputs, name):
        return []
    if name == "scenario_manifest":
        manifest = inputs.get("scenario_manifest")
        if isinstance(manifest, dict) and manifest.get("exists") is True and manifest.get("sha256") and manifest.get("ready") is False:
            return ["scenario_manifest_not_ready"]
    return [f"missing_{name}"]


def _input_present(inputs: dict[str, Any], name: str) -> bool:
    if name == "scenario_manifest":
        manifest = inputs.get("scenario_manifest")
        return (
            isinstance(manifest, dict)
            and manifest.get("exists") is True
            and isinstance(manifest.get("sha256"), str)
            and manifest.get("ready") is not False
        )
    value = inputs.get(name)
    if isinstance(value, list):
        return bool(value)
    return isinstance(value, str) and bool(value)


def _blocking_reasons(adapter_rows: list[dict[str, Any]]) -> list[str]:
    reasons = sorted({reason for row in adapter_rows for reason in row["blocking_reasons"]})
    if adapter_rows and all(not row["ready"] for row in adapter_rows):
        reasons.insert(0, "no_ready_external_adapters")
    return reasons


def _read_plan(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        raise ExternalEvalPlanError(f"External eval plan not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ExternalEvalPlanError(f"External eval plan must contain a JSON object: {path}")
    return payload


def _adapter_receipt(adapter_id: str, adapter: dict[str, Any] | None, live: bool) -> dict[str, Any]:
    ready = bool(adapter and adapter.get("ready") is True and not live)
    blocking = list(adapter.get("blocking_reasons", [])) if isinstance(adapter, dict) and isinstance(adapter.get("blocking_reasons"), list) else []
    if adapter is None:
        blocking.append("adapter_missing_from_plan")
    if live:
        blocking.append("live_external_eval_blocked_by_default")
    return {
        "id": adapter_id,
        "name": ADAPTERS[adapter_id]["name"],
        "domain": ADAPTERS[adapter_id]["domain"],
        "ready": ready,
        "planned_ready": bool(adapter and adapter.get("ready") is True),
        "blocking_reasons": sorted(set(str(reason) for reason in blocking)),
        "dependency_status": adapter.get("dependency_status") if isinstance(adapter, dict) else _missing_dependency_status(adapter_id),
        "required_inputs": list(ADAPTERS[adapter_id]["required_inputs"]),
        "provided_inputs": adapter.get("provided_inputs") if isinstance(adapter, dict) and isinstance(adapter.get("provided_inputs"), list) else [],
        "adapter_contract": _adapter_contract(adapter_id),
        "live_benchmark_started": False,
        "provider_api_called": False,
        "model_downloads_started": False,
        "credential_values_recorded": False,
        "cost_incurred_usd": 0,
    }


def _missing_dependency_status(adapter_id: str) -> dict[str, Any]:
    spec = ADAPTERS[adapter_id]
    return {
        "available": False,
        "imports": {name: False for name in spec["import_names"]},
        "commands": {name: False for name in spec["command_names"]},
    }


def _adapter_contract(adapter_id: str) -> dict[str, Any]:
    return {
        "schema_version": EXTERNAL_EVAL_ADAPTER_CONTRACT_VERSION,
        "adapter_id": f"external_eval.{adapter_id}.fail_closed.v1",
        "external_adapter_id": adapter_id,
        "receipt_types": list(EXTERNAL_EVAL_ADAPTER_RECEIPT_TYPES),
        "dry_run_transport": "plan_and_receipt_only",
        "live_benchmark_supported": False,
        "provider_api_called_by_flight_recorder": False,
        "model_downloads_started_by_flight_recorder": False,
        "credential_values_recorded": False,
        "cost_incurred_usd": 0,
        "requires_identical_heldout_scenarios": True,
        "requires_external_runner_receipt_for_live": True,
        "requires_dependency_probe_before_live": True,
        "requires_explicit_live_opt_in": True,
    }


def _plan_ref(path: Path, preserve_paths: bool, display_base_dir: Path | None) -> dict[str, Any]:
    exists = path.exists() and path.is_file()
    display_path, replayable = _source_display_path(path, preserve_paths, display_base_dir)
    public_exists = exists and replayable
    plan = _read_plan(path) if public_exists else {}
    return {
        "path": display_path,
        "exists": public_exists,
        "sha256": _sha256(path) if public_exists else None,
        "size_bytes": path.stat().st_size if public_exists else None,
        "schema_version": plan.get("schema_version") if public_exists else None,
        "ready": plan.get("ready") if public_exists and isinstance(plan.get("ready"), bool) else None,
        "adapter_count": plan.get("adapter_count") if public_exists and isinstance(plan.get("adapter_count"), int) else None,
    }


def _add_receipt_check(
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
            "summary": f"{check_id}: {'passed' if passed else 'failed'}",
        }
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path, preserve_paths: bool, display_base_dir: Path | None = None) -> str:
    if preserve_paths:
        return str(path)
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except (OSError, ValueError):
        return str(path)


def _source_display_path(path: Path, preserve_paths: bool, display_base_dir: Path | None = None) -> tuple[str, bool]:
    displayed = _source_ref_display_path(path, preserve_paths, display_base_dir)
    return displayed, _is_safe_public_path(displayed)


def _source_ref_display_path(path: Path, preserve_paths: bool, display_base_dir: Path | None = None) -> str:
    raw = str(path)
    if display_base_dir is not None:
        relative = _safe_output_relative_path(path, display_base_dir)
        return relative if relative is not None else f"<redacted:{_basename(raw)}>"
    if preserve_paths:
        return raw if _is_safe_public_path(raw) else f"<redacted:{_basename(raw)}>"
    try:
        relative = str(path.resolve().relative_to(Path.cwd().resolve()))
    except (OSError, ValueError):
        return f"<redacted:{_basename(raw)}>"
    return relative if _is_safe_public_path(relative) else f"<redacted:{_basename(raw)}>"


def _safe_output_relative_path(path: Path, display_base_dir: Path) -> str | None:
    try:
        relative = os.path.relpath(path.resolve(), display_base_dir.resolve())
    except OSError:
        return None
    return relative if _is_safe_public_path(relative) else None


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


def _basename(value: str) -> str:
    return value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] or "path"


def write_external_eval_plan(plan: dict[str, Any], out_path: str | Path, *, preserve_paths: bool = False) -> None:
    """Write an external eval plan as stable JSON."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _plan_for_write(plan, path, preserve_paths)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _plan_for_write(plan: dict[str, Any], out_path: Path, preserve_paths: bool) -> dict[str, Any]:
    payload = copy.deepcopy(plan)
    manifest = payload.get("inputs", {}).get("scenario_manifest") if isinstance(payload.get("inputs"), dict) else None
    if isinstance(manifest, dict):
        if _rewrite_manifest_ref_for_output(manifest, out_path.parent):
            _refresh_plan_readiness(payload)
    return payload


def _receipt_for_write(receipt: dict[str, Any], out_path: Path, preserve_paths: bool) -> dict[str, Any]:
    if preserve_paths:
        return receipt
    payload = copy.deepcopy(receipt)
    source_plan = payload.get("source_plan") if isinstance(payload.get("source_plan"), dict) else None
    if isinstance(source_plan, dict):
        source_plan["path"] = _output_relative_path(source_plan.get("path"), out_path.parent)
    return payload


def _output_relative_path(value: Any, output_dir: Path) -> Any:
    if not isinstance(value, str) or not value:
        return value
    path = Path(value)
    if not path.is_absolute():
        if not path.exists():
            return value
        path = path.resolve()
    return os.path.relpath(path.resolve(), output_dir.resolve())


def _rewrite_manifest_ref_for_output(manifest: dict[str, Any], output_dir: Path) -> bool:
    value = manifest.get("path")
    if not isinstance(value, str) or not value:
        return False
    if value.startswith("<redacted:"):
        return False
    path = Path(value)
    if _is_safe_public_path(value):
        if (output_dir / path).is_file():
            return False
        if path.exists():
            relative = _safe_output_relative_path(path, output_dir)
            if relative is not None:
                manifest["path"] = relative
                return False
            _redact_manifest_ref(manifest, value)
            return True
        if manifest.get("exists") is True:
            _redact_manifest_ref(manifest, value)
            return True
        return False
    relative = _safe_output_relative_path(path, output_dir) if path.is_absolute() or path.exists() else None
    if relative is not None:
        manifest["path"] = relative
        return False
    _redact_manifest_ref(manifest, value)
    return True


def _redact_manifest_ref(manifest: dict[str, Any], raw: str) -> None:
    manifest.update(_missing_manifest_record(f"<redacted:{_basename(raw)}>"))


def _refresh_plan_readiness(payload: dict[str, Any]) -> None:
    inputs = payload.get("inputs") if isinstance(payload.get("inputs"), dict) else {}
    adapters = payload.get("adapters") if isinstance(payload.get("adapters"), list) else []
    for adapter in adapters:
        if not isinstance(adapter, dict):
            continue
        adapter_id = adapter.get("id")
        spec = ADAPTERS.get(adapter_id) if isinstance(adapter_id, str) else None
        if spec is None:
            continue
        old_blockers = adapter.get("blocking_reasons") if isinstance(adapter.get("blocking_reasons"), list) else []
        input_blocker_reasons = _input_blocker_reasons(spec)
        preserved_blockers = [reason for reason in old_blockers if isinstance(reason, str) and reason not in input_blocker_reasons]
        new_input_blockers = [reason for name in spec["required_inputs"] for reason in _input_blockers(inputs, name)]
        blockers = sorted(set(preserved_blockers + new_input_blockers))
        adapter["provided_inputs"] = [name for name in spec["required_inputs"] if _input_present(inputs, name)]
        contract = adapter.get("execution_contract")
        if isinstance(contract, dict):
            manifest = inputs.get("scenario_manifest") if isinstance(inputs, dict) else {}
            contract["scenario_manifest_sha256"] = manifest.get("sha256") if isinstance(manifest, dict) else None
        adapter["blocking_reasons"] = blockers
        adapter["ready"] = not blockers
    adapter_rows = [row for row in adapters if isinstance(row, dict) and "ready" in row and "blocking_reasons" in row]
    blocking_reasons = _blocking_reasons(adapter_rows)
    payload["ready_adapter_count"] = sum(1 for row in adapter_rows if row.get("ready") is True)
    payload["blocking_reasons"] = blocking_reasons
    payload["ready"] = bool(adapter_rows) and not blocking_reasons
    governance = payload.get("governance_handoff")
    if isinstance(governance, dict):
        governance["external_eval_claims_allowed"] = payload["ready"]
        governance["recommendation"] = (
            "External eval adapters are ready to execute against the declared held-out scenario manifest."
            if payload["ready"]
            else "Keep external eval claims disabled until every selected adapter is ready."
        )


def _input_blocker_reasons(spec: dict[str, Any]) -> set[str]:
    return {f"missing_{name}" for name in spec["required_inputs"]} | {"scenario_manifest_not_ready"}
