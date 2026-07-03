"""Fail-closed plans for external eval harness adapters."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXTERNAL_EVAL_PLAN_SCHEMA_VERSION = "hfr.external_eval_adapters.v1"

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
) -> dict[str, Any]:
    """Build a readiness plan for optional external eval harness adapters."""
    selected = _selected_adapters(adapters)
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
) -> dict[str, Any]:
    manifest = _manifest_record(scenario_manifest, preserve_paths)
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


def _manifest_record(path: str | Path | None, preserve_paths: bool) -> dict[str, Any]:
    if path is None:
        return {
            "path": None,
            "exists": False,
            "sha256": None,
            "size_bytes": None,
            "schema_version": None,
            "ready": None,
            "scenario_count": None,
        }
    manifest_path = Path(path)
    exists = manifest_path.exists() and manifest_path.is_file()
    return {
        "path": _display_path(manifest_path, preserve_paths),
        "exists": exists,
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
        "ready": ready,
        "blocking_reasons": blocking_reasons,
    }


def _dependency_status(spec: dict[str, Any]) -> dict[str, Any]:
    imports = {name: importlib.util.find_spec(name) is not None for name in spec["import_names"]}
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path, preserve_paths: bool) -> str:
    if preserve_paths:
        return str(path)
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except (OSError, ValueError):
        return str(path)


def write_external_eval_plan(plan: dict[str, Any], out_path: str | Path, *, preserve_paths: bool = False) -> None:
    """Write an external eval plan as stable JSON."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _plan_for_write(plan, path, preserve_paths)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _plan_for_write(plan: dict[str, Any], out_path: Path, preserve_paths: bool) -> dict[str, Any]:
    if preserve_paths:
        return plan
    payload = copy.deepcopy(plan)
    manifest = payload.get("inputs", {}).get("scenario_manifest") if isinstance(payload.get("inputs"), dict) else None
    if isinstance(manifest, dict):
        manifest["path"] = _output_relative_path(manifest.get("path"), out_path.parent)
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
