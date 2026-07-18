#!/usr/bin/env python3
"""Write fail-closed plans for optional external eval adapters.

The Eval layer can advertise BFCL, Inspect AI, lm-evaluation-harness, and
SWE-bench readiness without importing those projects or pretending they are
available. This script emits a deterministic artifact Governance can inspect
before any external benchmark runner is wired in.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "hfr.external_eval_adapters.v1"

ADAPTERS: dict[str, dict[str, Any]] = {
    "bfcl": {
        "name": "Berkeley Function Calling Leaderboard",
        "domain": "tool_calling",
        "import_names": ["bfcl_eval"],
        "default_suite_tags": ["external", "tool_calling", "function_calling"],
        "required_inputs": ["model_endpoint", "tool_schema_set", "identical_scenario_manifest"],
        "reference": "https://gorilla.cs.berkeley.edu/leaderboard.html",
    },
    "inspect_ai": {
        "name": "Inspect AI",
        "domain": "agentic_safety",
        "import_names": ["inspect_ai"],
        "default_suite_tags": ["external", "agentic_safety", "custom_eval"],
        "required_inputs": ["model_endpoint", "task_set", "sandbox_policy", "identical_scenario_manifest"],
        "reference": "https://inspect.aisi.org.uk/",
    },
    "lm_eval_harness": {
        "name": "lm-evaluation-harness",
        "domain": "language_benchmark",
        "import_names": ["lm_eval"],
        "default_suite_tags": ["external", "language_benchmark", "sanity_check"],
        "required_inputs": ["model_endpoint_or_model_args", "task_list", "identical_scenario_manifest"],
        "reference": "https://github.com/EleutherAI/lm-evaluation-harness",
    },
    "swe_bench": {
        "name": "SWE-bench",
        "domain": "coding_agent",
        "import_names": ["swebench"],
        "default_suite_tags": ["external", "coding_agent", "repository_repair"],
        "required_inputs": ["model_endpoint", "repository_task_set", "sandbox_policy", "identical_scenario_manifest"],
        "reference": "https://www.swebench.com/",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", action="append", choices=sorted(ADAPTERS), help="Adapter id to plan; repeatable. Defaults to all.")
    parser.add_argument("--out", type=Path, required=True, help="Write external eval adapter plan JSON here.")
    parser.add_argument("--suite-id", default="agentic_finetune_external_eval_adapters")
    parser.add_argument("--model", default="", help="Optional model or served endpoint identity.")
    parser.add_argument("--base-url", default="", help="Optional OpenAI-compatible endpoint for adapter plans.")
    parser.add_argument("--scenario-manifest", default="", help="Held-out scenario manifest shared by all compared arms.")
    parser.add_argument("--allow-installed", action="store_true", help="Mark adapters ready when optional dependency imports and required inputs are present.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    adapter_ids = args.adapter or sorted(ADAPTERS)
    artifact = build_external_eval_adapters(
        adapter_ids=adapter_ids,
        suite_id=args.suite_id,
        model=args.model,
        base_url=args.base_url,
        scenario_manifest=args.scenario_manifest,
        allow_installed=args.allow_installed,
    )
    write_json(args.out, artifact)
    print(
        json.dumps(
            {
                "ready": artifact["ready"],
                "adapter_count": artifact["adapter_count"],
                "ready_adapter_count": artifact["ready_adapter_count"],
                "out": str(args.out),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if artifact["ready"] else 1


def build_external_eval_adapters(
    *,
    adapter_ids: list[str],
    suite_id: str,
    model: str,
    base_url: str,
    scenario_manifest: str,
    allow_installed: bool,
) -> dict[str, Any]:
    adapters = [
        _adapter_plan(
            adapter_id=adapter_id,
            model=model,
            base_url=base_url,
            scenario_manifest=scenario_manifest,
            allow_installed=allow_installed,
        )
        for adapter_id in adapter_ids
    ]
    ready_count = sum(1 for adapter in adapters if adapter["ready"])
    blocking_reasons = sorted({reason for adapter in adapters for reason in adapter["blocking_reasons"]})
    return {
        "schema_version": SCHEMA_VERSION,
        "suite_id": suite_id,
        "ready": bool(adapters) and ready_count == len(adapters),
        "adapter_count": len(adapters),
        "ready_adapter_count": ready_count,
        "blocking_reasons": blocking_reasons,
        "adapters": adapters,
        "governance_handoff": {
            "ready": bool(adapters) and ready_count == len(adapters),
            "status": "ready" if bool(adapters) and ready_count == len(adapters) else "blocked",
            "recommendation": "run_external_eval_adapters" if bool(adapters) and ready_count == len(adapters) else "keep_external_adapters_disabled",
            "requires_identical_scenario_ids_for_cross_arm_claims": True,
            "next_actions": _next_actions(adapters),
        },
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "inputs": {
            "model": model or None,
            "base_url": base_url or None,
            "scenario_manifest": _input_record(scenario_manifest),
            "allow_installed": allow_installed,
        },
    }


def _adapter_plan(
    *,
    adapter_id: str,
    model: str,
    base_url: str,
    scenario_manifest: str,
    allow_installed: bool,
) -> dict[str, Any]:
    if adapter_id not in ADAPTERS:
        raise ValueError(f"Unknown external eval adapter: {adapter_id}")
    spec = ADAPTERS[adapter_id]
    imports = {name: importlib.util.find_spec(name) is not None for name in spec["import_names"]}
    missing_imports = [name for name, available in imports.items() if not available]
    missing_inputs = _missing_inputs(spec["required_inputs"], model=model, base_url=base_url, scenario_manifest=scenario_manifest)
    blocking_reasons: list[str] = []
    if missing_imports:
        blocking_reasons.append("missing_optional_dependency")
    if missing_inputs:
        blocking_reasons.append("missing_required_inputs")
    if not allow_installed:
        blocking_reasons.append("adapter_disabled_until_explicitly_enabled")
    ready = not blocking_reasons
    return {
        "id": adapter_id,
        "name": spec["name"],
        "domain": spec["domain"],
        "ready": ready,
        "status": "ready" if ready else "blocked",
        "blocking_reasons": blocking_reasons,
        "missing_imports": missing_imports,
        "missing_inputs": missing_inputs,
        "import_names": spec["import_names"],
        "suite_tags": spec["default_suite_tags"],
        "required_inputs": spec["required_inputs"],
        "reference": spec["reference"],
        "runner_contract": {
            "mode": "external_eval_adapter",
            "must_use_identical_scenario_manifest": True,
            "must_emit_eval_summary": True,
            "must_emit_repair_work_items_on_failure": True,
            "must_not_claim_success_without_identical_scenarios": True,
        },
        "next_actions": _adapter_next_actions(adapter_id, missing_imports, missing_inputs, allow_installed),
    }


def _missing_inputs(required: list[str], *, model: str, base_url: str, scenario_manifest: str) -> list[str]:
    missing: list[str] = []
    for item in required:
        if item in {"model_endpoint", "model_endpoint_or_model_args"} and not (model or base_url):
            missing.append(item)
        elif item == "identical_scenario_manifest" and not scenario_manifest:
            missing.append(item)
        elif item in {"tool_schema_set", "task_set", "sandbox_policy", "task_list", "repository_task_set"}:
            missing.append(item)
    return missing


def _adapter_next_actions(adapter_id: str, missing_imports: list[str], missing_inputs: list[str], allow_installed: bool) -> list[str]:
    actions: list[str] = []
    if missing_imports:
        actions.append(f"Install and pin optional dependency import(s) for {adapter_id}: {', '.join(missing_imports)}.")
    if missing_inputs:
        actions.append(f"Provide required adapter input(s): {', '.join(missing_inputs)}.")
    if not allow_installed:
        actions.append("Rerun with --allow-installed only after dependency and input artifacts are intentionally available.")
    if not actions:
        actions.append("Run the adapter and convert results into hfr.hermes_heldout_eval_summary.v1-compatible summaries.")
    return actions


def _next_actions(adapters: list[dict[str, Any]]) -> list[str]:
    blocked = [adapter for adapter in adapters if not adapter["ready"]]
    if not blocked:
        return ["Run ready adapters and attach their eval summaries to the Governance packet."]
    return [
        "Keep blocked external adapters out of promotion evidence.",
        "Resolve each adapter's missing optional dependency and required input artifacts before enabling it.",
        "Use the same held-out scenario manifest for every baseline, trace-only, champion, and candidate external eval arm.",
    ]


def _input_record(path_value: str) -> dict[str, Any]:
    if not path_value:
        return {"path": None, "exists": False, "sha256": None}
    path = Path(path_value)
    return {"path": path_value, "exists": path.exists(), "sha256": _sha256(path) if path.is_file() else None}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
