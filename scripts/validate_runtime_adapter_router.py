#!/usr/bin/env python3
"""Validate the runtime adapter router case study without external calls."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder import cli  # noqa: E402
from flightrecorder.realistic_tool_corpus import CHAT_TEMPLATE_SHA256, MODEL_REVISION  # noqa: E402
from flightrecorder.schema_registry import check_schema_file  # noqa: E402
from flightrecorder.runtime_adapter_router import (  # noqa: E402
    ApprovalStore,
    build_write_approval,
    canonical_sha256,
    dispatch_tool_call,
)
from flightrecorder.validation import (  # noqa: E402
    validate_adapter_route_decision_artifact,
    validate_tool_capability_selection_artifact,
)


MODEL_ID = "Qwen/Qwen3-0.6B"
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def file_ref(path: Path, base: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": path.relative_to(base).as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def run_cli(argv: list[str]) -> None:
    stdout = StringIO()
    stderr = StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = cli.main(argv)
        except SystemExit as exc:
            code = int(exc.code) if isinstance(exc.code, int) else 1
    if code:
        raise RuntimeError(
            f"flightrecorder {' '.join(argv)} failed: {stderr.getvalue() or stdout.getvalue()}"
        )


def promotion_fixture(candidate_id: str) -> dict[str, Any]:
    source = (
        ROOT
        / "examples"
        / "agentic_training"
        / "promotion_governance"
        / "promotion_decision.json"
    )
    promotion = copy.deepcopy(json.loads(source.read_text(encoding="utf-8")))
    promotion["models"]["candidate"]["id"] = candidate_id
    for alias in promotion["alias_update"]["aliases"]:
        if alias.get("alias") in {"candidate", "champion"}:
            alias["target"] = candidate_id
    return promotion


def make_candidate(
    root: Path,
    *,
    candidate_id: str,
    kind: str,
    domains: list[str],
    capabilities: list[str],
    adapter_sha256: str,
) -> dict[str, Any]:
    safe_id = candidate_id.replace("/", "_")
    promotion_path = root / "evidence" / f"{safe_id}.promotion.json"
    training_path = root / "evidence" / f"{safe_id}.training.json"
    evaluation_path = root / "evidence" / f"{safe_id}.evaluation.json"
    write_json(promotion_path, promotion_fixture(candidate_id))
    write_json(
        training_path,
        {
            "schema_version": "hfr.router_demo_training.v1",
            "candidate_id": candidate_id,
            "passed": True,
        },
    )
    write_json(
        evaluation_path,
        {
            "schema_version": "hfr.router_demo_evaluation.v1",
            "candidate_id": candidate_id,
            "passed": True,
        },
    )
    adapter_id = f"adapter-{safe_id}"
    registry_id = f"registry-{safe_id}"
    binding = {
        "independent_evidence": True,
        "candidate_id": candidate_id,
        "registry_entry_id": registry_id,
        "adapter_id": adapter_id,
        "adapter_revision": f"revision-{safe_id}",
        "adapter_sha256": adapter_sha256,
        "base_model_id": MODEL_ID,
        "base_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "chat_template_sha256": CHAT_TEMPLATE_SHA256,
        "training_result_ref": file_ref(training_path, root),
        "evaluation_result_ref": file_ref(evaluation_path, root),
    }
    return {
        "candidate_id": candidate_id,
        "candidate_kind": kind,
        "adapter_id": adapter_id,
        "active_adapter_ids": [adapter_id],
        "adapter_revision": f"revision-{safe_id}",
        "adapter_sha256": adapter_sha256,
        "base_model_id": MODEL_ID,
        "base_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "chat_template_sha256": CHAT_TEMPLATE_SHA256,
        "domains": domains,
        "capabilities": capabilities,
        "registry_entry_id": registry_id,
        "composition_members": [],
        "promotion_decision": json.loads(promotion_path.read_text(encoding="utf-8")),
        "promotion_evidence_ref": file_ref(promotion_path, root),
        "promotion_binding": binding,
    }


def task(task_id: str, domains: list[str], required: list[str]) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "contract_fingerprint": canonical_sha256(
            {"task_id": task_id, "domains": domains, "required": required}
        ),
        "capabilities": {"required": required, "optional": []},
        "domains": domains,
        "requires_tools": True,
        "allow_no_tools": False,
    }


def tool(
    name: str, capability: str, sha256: str, *, write_capable: bool = False
) -> dict[str, Any]:
    properties: dict[str, Any]
    required: list[str]
    if write_capable:
        properties = {
            "path": {"type": "string", "minLength": 1},
            "content": {"type": "string"},
        }
        required = ["path", "content"]
    else:
        properties = {"query": {"type": "string", "minLength": 1}}
        required = ["query"]
    return {
        "name": name,
        "version": "2026-07-01.immutable",
        "definition_sha256": sha256,
        "capabilities": [capability],
        "risk_class": "write" if write_capable else "read",
        "write_capable": write_capable,
        "parameters_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


def validate_case_study() -> dict[str, Any]:
    corpus = json.loads(
        (
            ROOT
            / "examples"
            / "case_studies"
            / "runtime_adapter_router"
            / "corpus_manifest.json"
        ).read_text(encoding="utf-8")
    )
    if corpus.get("split_counts") != {
        "train": 288,
        "development": 36,
        "sealed_final": 36,
    }:
        raise RuntimeError(
            "runtime adapter corpus split counts do not match the reviewed case study"
        )
    evaluation_path = (
        ROOT
        / "examples"
        / "case_studies"
        / "runtime_adapter_router"
        / "evaluation"
        / "actual_local_evaluation.json"
    )
    evaluation_check = check_schema_file(
        evaluation_path,
        "runtime_adapter_candidate_evaluation",
    )
    if not evaluation_check["passed"]:
        raise RuntimeError(
            "actual local runtime-adapter evaluation failed schema validation: "
            + "; ".join(evaluation_check["errors"])
        )
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation_fingerprint_input = copy.deepcopy(evaluation)
    declared_evaluation_fingerprint = evaluation_fingerprint_input.pop(
        "evaluation_fingerprint", None
    )
    computed_evaluation_fingerprint = hashlib.sha256(
        json.dumps(
            evaluation_fingerprint_input,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if declared_evaluation_fingerprint != computed_evaluation_fingerprint:
        raise RuntimeError(
            "actual local evaluation fingerprint does not match its contents"
        )
    expected_candidates = {
        "base",
        "generalist_lora_v1",
        "browser_lora_v1",
        "database_lora_v1",
        "code_terminal_lora_v1",
    }
    actual_candidates = {
        str(report.get("candidate_id") or "")
        for report in evaluation.get("candidate_reports", [])
        if isinstance(report, dict)
    }
    if (
        evaluation.get("passed") is not False
        or evaluation.get("promotion_eligible_candidates") != []
    ):
        raise RuntimeError(
            "actual local evaluation must preserve the fail-closed no-promotion result"
        )
    if actual_candidates != expected_candidates:
        raise RuntimeError(
            "actual local evaluation candidate set does not match the reviewed run"
        )

    with tempfile.TemporaryDirectory(prefix="hfr-runtime-router-") as directory:
        root = Path(directory)
        environment = {
            "base_model_id": MODEL_ID,
            "base_revision": MODEL_REVISION,
            "tokenizer_revision": MODEL_REVISION,
            "chat_template_sha256": CHAT_TEMPLATE_SHA256,
            "runtime": "local-offline-fixture",
        }
        routing_policy = {
            "policy_id": "runtime-router-demo",
            "allow_generalist_fallback": True,
            "required_base_model_id": MODEL_ID,
            "required_base_revision": MODEL_REVISION,
            "required_tokenizer_revision": MODEL_REVISION,
            "required_chat_template_sha256": CHAT_TEMPLATE_SHA256,
        }
        candidates = [
            make_candidate(
                root,
                candidate_id="demo/browser-specialist",
                kind="specialist",
                domains=["browser"],
                capabilities=["browser.search"],
                adapter_sha256=SHA_A,
            ),
            make_candidate(
                root,
                candidate_id="demo/generalist",
                kind="generalist",
                domains=["browser", "database", "code_terminal"],
                capabilities=["browser.search", "code.write"],
                adapter_sha256=SHA_B,
            ),
        ]
        write_json(root / "environment.json", environment)
        write_json(root / "routing_policy.json", routing_policy)
        write_json(root / "candidate_catalog.json", {"candidates": candidates})

        browser_task = task("demo-browser", ["browser"], ["browser.search"])
        write_json(root / "browser_task.json", browser_task)
        write_json(
            root / "browser_tools.json",
            {"tools": [tool("browser.search", "browser.search", SHA_C)]},
        )
        write_json(
            root / "browser_tool_policy.json",
            {
                "policy_id": "browser-tools",
                "known_capabilities": ["browser.search"],
                "allow_write_tools": False,
                "allowed_risk_classes": ["read"],
            },
        )
        run_cli(
            [
                "runtime-router",
                "tool-capabilities",
                "--task-contract",
                str(root / "browser_task.json"),
                "--tool-catalog",
                str(root / "browser_tools.json"),
                "--policy",
                str(root / "browser_tool_policy.json"),
                "--environment",
                str(root / "environment.json"),
                "--out",
                str(root / "browser_selection.json"),
            ]
        )
        run_cli(
            [
                "runtime-router",
                "adapter",
                "--task-contract",
                str(root / "browser_task.json"),
                "--capability-selection",
                str(root / "browser_selection.json"),
                "--candidate-catalog",
                str(root / "candidate_catalog.json"),
                "--routing-policy",
                str(root / "routing_policy.json"),
                "--runtime-environment",
                str(root / "environment.json"),
                "--out",
                str(root / "browser_route.json"),
            ]
        )
        browser_selection = json.loads(
            (root / "browser_selection.json").read_text(encoding="utf-8")
        )
        browser_route = json.loads(
            (root / "browser_route.json").read_text(encoding="utf-8")
        )
        if (
            browser_route["selected_candidate"]["candidate_id"]
            != "demo/browser-specialist"
        ):
            raise RuntimeError(
                "single-domain task did not select the promoted specialist"
            )

        fallback_candidates = copy.deepcopy(candidates)
        fallback_candidates[0]["adapter_sha256"] = SHA_C
        write_json(
            root / "fallback_candidates.json", {"candidates": fallback_candidates}
        )
        run_cli(
            [
                "runtime-router",
                "adapter",
                "--task-contract",
                str(root / "browser_task.json"),
                "--capability-selection",
                str(root / "browser_selection.json"),
                "--candidate-catalog",
                str(root / "fallback_candidates.json"),
                "--routing-policy",
                str(root / "routing_policy.json"),
                "--runtime-environment",
                str(root / "environment.json"),
                "--out",
                str(root / "fallback_route.json"),
            ]
        )
        fallback_route = json.loads(
            (root / "fallback_route.json").read_text(encoding="utf-8")
        )
        if fallback_route["selected_candidate"]["candidate_id"] != "demo/generalist":
            raise RuntimeError(
                "stale specialist did not fall back to the promoted generalist"
            )

        write_task = task("demo-write", ["code_terminal"], ["code.write"])
        write_json(root / "write_task.json", write_task)
        write_json(
            root / "write_tools.json",
            {"tools": [tool("code.patch", "code.write", SHA_A, write_capable=True)]},
        )
        write_json(
            root / "write_tool_policy.json",
            {
                "policy_id": "write-tools",
                "known_capabilities": ["code.write"],
                "allow_write_tools": True,
                "allowed_risk_classes": ["write"],
            },
        )
        run_cli(
            [
                "runtime-router",
                "tool-capabilities",
                "--task-contract",
                str(root / "write_task.json"),
                "--tool-catalog",
                str(root / "write_tools.json"),
                "--policy",
                str(root / "write_tool_policy.json"),
                "--environment",
                str(root / "environment.json"),
                "--out",
                str(root / "write_selection.json"),
            ]
        )
        run_cli(
            [
                "runtime-router",
                "adapter",
                "--task-contract",
                str(root / "write_task.json"),
                "--capability-selection",
                str(root / "write_selection.json"),
                "--candidate-catalog",
                str(root / "candidate_catalog.json"),
                "--routing-policy",
                str(root / "routing_policy.json"),
                "--runtime-environment",
                str(root / "environment.json"),
                "--out",
                str(root / "write_route.json"),
            ]
        )
        write_selection = json.loads(
            (root / "write_selection.json").read_text(encoding="utf-8")
        )
        write_route = json.loads(
            (root / "write_route.json").read_text(encoding="utf-8")
        )

        for artifact_path, validator in (
            (
                root / "browser_selection.json",
                validate_tool_capability_selection_artifact,
            ),
            (root / "browser_route.json", validate_adapter_route_decision_artifact),
            (root / "fallback_route.json", validate_adapter_route_decision_artifact),
            (
                root / "write_selection.json",
                validate_tool_capability_selection_artifact,
            ),
            (root / "write_route.json", validate_adapter_route_decision_artifact),
        ):
            result = validator(artifact_path)
            if result.errors:
                raise RuntimeError(
                    f"{artifact_path.name} failed validation: {result.errors}"
                )

        read_tool = browser_selection["selected_tools"][0]
        read_key = f"{read_tool['name']}@{read_tool['version']}#{read_tool['definition_sha256']}"
        read_dispatch = dispatch_tool_call(
            capability_selection=browser_selection,
            route_decision=browser_route,
            tool_call={
                "call_id": "read-1",
                "tool": read_tool,
                "arguments": {"query": "public synthetic status"},
            },
            handlers={
                read_key: lambda arguments: {
                    "query": arguments["query"],
                    "status": "ok",
                }
            },
            policy={"policy_id": "dispatch", "allow_readonly_dispatch": True},
        )

        write_tool = write_selection["selected_tools"][0]
        write_key = f"{write_tool['name']}@{write_tool['version']}#{write_tool['definition_sha256']}"
        calls = {"count": 0}

        def write_handler(arguments: dict[str, Any]) -> dict[str, Any]:
            calls["count"] += 1
            return {"path": arguments["path"], "status": "synthetic-write-recorded"}

        write_arguments = {"path": "synthetic/policy.json", "content": "{}"}
        dispatch_policy = {"policy_id": "dispatch", "allow_readonly_dispatch": True}
        denied_write = dispatch_tool_call(
            capability_selection=write_selection,
            route_decision=write_route,
            tool_call={
                "call_id": "write-1",
                "tool": write_tool,
                "arguments": write_arguments,
            },
            handlers={write_key: write_handler},
            policy=dispatch_policy,
            now=100,
        )
        if calls["count"] != 0 or denied_write.get("handler_called") is not False:
            raise RuntimeError("denied write reached its handler")

        store = ApprovalStore()
        store.issue(
            build_write_approval(
                approval_id="approval-write-2",
                task_fingerprint=write_selection["task_contract_fingerprint"],
                route_fingerprint=write_route["route_fingerprint"],
                policy_fingerprint=canonical_sha256(dispatch_policy),
                tool_identity=write_tool,
                call_id="write-2",
                arguments=write_arguments,
                issued_at=100,
                expires_at=200,
            )
        )
        authorized_write = dispatch_tool_call(
            capability_selection=write_selection,
            route_decision=write_route,
            tool_call={
                "call_id": "write-2",
                "tool": write_tool,
                "arguments": write_arguments,
                "approval_id": "approval-write-2",
            },
            handlers={write_key: write_handler},
            policy=dispatch_policy,
            approval_store=store,
            now=150,
        )
        if calls["count"] != 1 or authorized_write.get("status") != "dispatched":
            raise RuntimeError(
                "content-bound approved write was not dispatched exactly once"
            )

        return {
            "passed": True,
            "corpus_split_counts": corpus["split_counts"],
            "actual_candidate_evaluation_passed": evaluation["passed"],
            "actual_promotion_eligible_candidate_count": len(
                evaluation["promotion_eligible_candidates"]
            ),
            "actual_evaluation_schema_passed": evaluation_check["passed"],
            "specialist_candidate": browser_route["selected_candidate"]["candidate_id"],
            "fallback_candidate": fallback_route["selected_candidate"]["candidate_id"],
            "read_dispatch": read_dispatch["status"],
            "denied_write": denied_write["reason_code"],
            "denied_write_handler_called": denied_write["handler_called"],
            "authorized_write": authorized_write["status"],
            "authorized_write_handler_count": calls["count"],
        }


def main() -> int:
    result = validate_case_study()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
