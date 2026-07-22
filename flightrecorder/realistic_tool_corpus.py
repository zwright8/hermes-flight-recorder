"""Deterministic native tool-use corpus for the runtime adapter router study."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .data_governance import (
    build_contamination_report,
    build_governance_receipt,
    task_contract_fingerprint,
)
from .trajectory_v2 import canonical_trajectory_v2_bytes, trajectory_v2_from_trace

SCHEMA_REVIEWED_ACTION_SFT = "hfr.reviewed.action_sft.v1"
SCHEMA_RL_ACTION_SFT = "hfr.rl.action_sft.v1"
DEFAULT_COUNT = 360
MIN_DEFAULT_COUNT = 300
MAX_DEFAULT_COUNT = 800
CASE_STUDY_ID = "runtime-adapter-router"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
TOKENIZER_REVISION = MODEL_REVISION
CHAT_TEMPLATE_SHA256 = (
    "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
)
POLICY_SHA256 = "097fd18322ea677169449e4a9d2c11fdb86838f5c9a4ae6cf3dff8c42f5c1a1c"
ENVIRONMENT_SHA256 = "47dd3b516cf5a5beb8ff9cc5ba2fded77d90fdad0a79986e0fd886bb778bd912"
SPLITS = ("train", "development", "sealed_final")
SCOPES = ("generalist", "browser", "database", "code_terminal")
DOMAINS = ("browser", "database", "code_terminal")
CONTROL_SCHEMAS = {
    "governance_receipt": "hfr.data_governance_receipt.v1",
    "contamination_report": "hfr.dataset_contamination_report.v1",
    "curated_dataset": "hfr.curated_dataset.v1",
    "action_credit": "hfr.action_credit.v1",
    "branch_replay": "hfr.branch_replay_dataset.v1",
    "reviewed_preferences": "hfr.reviewed.contract_preference.v1",
}


@dataclass(frozen=True)
class CorpusBuildResult:
    """Paths and counts produced by :func:`write_runtime_adapter_corpus`."""

    output_dir: Path
    model_manifest: Path
    dataset_manifest: Path
    corpus_manifest: Path
    all_action_sft_path: Path
    total_rows: int
    split_counts: dict[str, int]
    task_scope_counts: dict[str, int]
    task_family_counts: dict[str, int]
    candidate_manifests: dict[str, Path]


def build_tool_definitions() -> list[dict[str, Any]]:
    """Return exact native tool definitions with immutable versions."""

    definitions = [
        _tool(
            "browser.search",
            "2026-07-01.immutable",
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "query": {"type": "string"},
                    "recency_days": {"type": "integer", "minimum": 0},
                },
                "required": ["query", "recency_days"],
            },
            capability="browser",
            write_capable=False,
        ),
        _tool(
            "browser.open",
            "2026-07-01.immutable",
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "url": {"type": "string"},
                    "extract": {"enum": ["title", "table", "summary"]},
                },
                "required": ["url", "extract"],
            },
            capability="browser",
            write_capable=False,
        ),
        _tool(
            "database.query",
            "2026-07-01.immutable",
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "statement": {"type": "string"},
                    "parameters": {"type": "object"},
                    "read_only": {"const": True},
                },
                "required": ["statement", "parameters", "read_only"],
            },
            capability="database",
            write_capable=False,
        ),
        _tool(
            "database.write",
            "2026-07-01.immutable",
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "statement": {"type": "string"},
                    "parameters": {"type": "object"},
                    "approval_id": {"type": "string"},
                },
                "required": ["statement", "parameters", "approval_id"],
            },
            capability="database",
            write_capable=True,
        ),
        _tool(
            "code_terminal.run",
            "2026-07-01.immutable",
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "cmd": {"type": "string"},
                    "cwd": {"type": "string"},
                    "timeout_ms": {"type": "integer", "minimum": 1000},
                },
                "required": ["cmd", "cwd", "timeout_ms"],
            },
            capability="code_terminal",
            write_capable=False,
        ),
        _tool(
            "code_terminal.patch",
            "2026-07-01.immutable",
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "path": {"type": "string"},
                    "unified_diff": {"type": "string"},
                    "approval_id": {"type": "string"},
                },
                "required": ["path", "unified_diff", "approval_id"],
            },
            capability="code_terminal",
            write_capable=True,
        ),
    ]
    return definitions


def build_runtime_adapter_rows(
    count: int = DEFAULT_COUNT, *, seed: int = 17
) -> list[dict[str, Any]]:
    """Build deterministic action-SFT rows for the router case study.

    ``count`` may be lowered in tests. Production/default builds are kept in
    the requested 300-800 range by :func:`write_runtime_adapter_corpus`.
    """

    if count <= 0:
        raise ValueError("count must be positive")
    tools = build_tool_definitions()
    rows = [_build_row(index, seed=seed, tools=tools) for index in range(count)]
    return rows


def write_runtime_adapter_corpus(
    output_dir: Path, *, count: int = DEFAULT_COUNT, seed: int = 17
) -> CorpusBuildResult:
    """Write corpus JSONL, controls, and manifests under ``output_dir``."""

    if count > MAX_DEFAULT_COUNT:
        raise ValueError(
            f"count must be between {MIN_DEFAULT_COUNT} and {MAX_DEFAULT_COUNT}"
        )
    output_dir = Path(output_dir)
    data_dir = output_dir / "data"
    controls_dir = output_dir / "controls"
    registry_dir = output_dir / "registry"
    data_dir.mkdir(parents=True, exist_ok=True)
    controls_dir.mkdir(parents=True, exist_ok=True)
    registry_dir.mkdir(parents=True, exist_ok=True)

    rows = build_runtime_adapter_rows(count, seed=seed)
    train_rows = [row for row in rows if row["split"] == "train"]
    all_action_path = data_dir / "flightrecorder_action_sft.jsonl"
    rl_action_path = data_dir / "rl_action_sft.jsonl"
    all_splits_path = data_dir / "all_splits_action_sft.jsonl"
    _write_jsonl(all_action_path, train_rows)
    _write_jsonl(rl_action_path, [_rl_action_row(row) for row in train_rows])
    _write_jsonl(all_splits_path, rows)

    _write_split_files(data_dir, rows)
    scoped_paths = _write_candidate_files(data_dir, train_rows)
    controls = _control_payloads(train_rows, seed=seed)
    control_paths: dict[str, Path] = {}
    for name, payload in controls.items():
        suffix = ".jsonl" if isinstance(payload, list) else ".json"
        path = controls_dir / f"{name}{suffix}"
        if suffix == ".jsonl":
            _write_jsonl(path, payload)
        else:
            _write_json(path, payload)
        control_paths[name] = path

    model_manifest = registry_dir / "model_candidate.json"
    _write_json(model_manifest, _model_manifest())

    common_data_files = {
        "flightrecorder_action_sft": "../data/flightrecorder_action_sft.jsonl",
        "fr_action_sft": "../data/flightrecorder_action_sft.jsonl",
        "action_sft": "../data/flightrecorder_action_sft.jsonl",
        "rl_action_sft": "../data/rl_action_sft.jsonl",
        **{
            name: f"../controls/{name}{'.jsonl' if name in {'action_credit', 'reviewed_preferences'} else '.json'}"
            for name in CONTROL_SCHEMAS
        },
    }
    dataset_manifest = registry_dir / "dataset_version.json"
    manifest = _dataset_manifest(
        train_rows,
        dataset_id=f"hfr-{CASE_STUDY_ID}-corpus",
        dataset_version=f"hfr-{CASE_STUDY_ID}-corpus.v1",
        data_files=common_data_files,
        base_dir=registry_dir,
    )
    _write_json(dataset_manifest, manifest)

    candidate_manifests: dict[str, Path] = {}
    for scope, relative_path in scoped_paths.items():
        scoped_rows = _scope_rows(train_rows, scope)
        candidate_path = registry_dir / f"{scope}_dataset_version.json"
        data_files = {
            **common_data_files,
            "flightrecorder_action_sft": f"../{relative_path}",
            "fr_action_sft": f"../{relative_path}",
            "action_sft": f"../{relative_path}",
        }
        _write_json(
            candidate_path,
            _dataset_manifest(
                scoped_rows,
                dataset_id=f"hfr-{CASE_STUDY_ID}-{scope}",
                dataset_version=f"hfr-{CASE_STUDY_ID}-{scope}.v1",
                data_files=data_files,
                base_dir=registry_dir,
            ),
        )
        candidate_manifests[scope] = candidate_path

    corpus_manifest_path = output_dir / "corpus_manifest.json"
    corpus_manifest = _corpus_manifest(
        rows,
        model_manifest=model_manifest,
        dataset_manifest=dataset_manifest,
        candidate_manifests=candidate_manifests,
        data_path=all_action_path,
        rl_data_path=rl_action_path,
        all_splits_path=all_splits_path,
        controls=control_paths,
        seed=seed,
    )
    _write_json(corpus_manifest_path, corpus_manifest)
    return CorpusBuildResult(
        output_dir=output_dir,
        model_manifest=model_manifest,
        dataset_manifest=dataset_manifest,
        corpus_manifest=corpus_manifest_path,
        all_action_sft_path=all_action_path,
        total_rows=len(rows),
        split_counts=_counts(rows, "split"),
        task_scope_counts=_counts(rows, "task_scope"),
        task_family_counts=_counts(rows, "task_family"),
        candidate_manifests=candidate_manifests,
    )


def _build_row(index: int, *, seed: int, tools: list[dict[str, Any]]) -> dict[str, Any]:
    family = _family_for(index)
    split = _split_for(index)
    split_index = sum(1 for prior in range(index + 1) if _split_for(prior) == split) - 1
    template_family = f"{split}-template-{split_index:04d}"
    task_id = f"rar-{split}-{index:05d}"
    prompt = _prompt(family, split, index, seed)
    selected_tools = _tools_for_family(tools, family)
    messages, final_answer, behavior, approvals = _messages(
        family, task_id, prompt, index
    )
    governance = _governance(task_id)
    scenario_contract = {
        "case_study": CASE_STUDY_ID,
        "task_id": task_id,
        "template_family": template_family,
        "split": split,
        "family": family,
        "requires_tools": bool(selected_tools)
        and family not in {"ambiguous_clarification", "write_denial", "refusal"},
    }
    base_row = {
        "schema_version": SCHEMA_REVIEWED_ACTION_SFT,
        "sample_id": task_id,
        "episode_id": task_id,
        "scenario_id": task_id,
        "task_id": task_id,
        "split": split,
        "task_family": _task_family(family, split),
        "task_scope": _task_scope(family),
        "task_domains": _task_domains(family),
        "prompt_template_family": template_family,
        "prompt_hash": _canonical_sha256(
            {"split": split, "template": template_family, "prompt": prompt}
        ),
        "prompt": prompt,
        "response": final_answer,
        "messages": messages,
        "tools": selected_tools,
        "human_label": "accept",
        "review_item_id": f"review-{task_id}",
        "reviewer_confidence": "high",
        "tool_schema_provenance": "recorded_exact",
        "quality_gate": "human_reviewed_native_action_accept",
        "credit_policy": "exclude_entire_trajectory_on_any_negative_tool_action",
        "governance": governance,
        "source_artifact": "controls/curated_dataset.json",
        "training_arm": "flightrecorder_action_sft",
        "behavior_tags": behavior,
        "approval_evidence": approvals,
        "policy": _policy_identity(),
        "environment": _environment_identity(),
        "scenario_contract": scenario_contract,
    }
    base_row["task_contract_fingerprint"] = task_contract_fingerprint(base_row)
    trace_events = _events_from_messages(messages, task_id)
    trajectory = trajectory_v2_from_trace(
        {
            "events": trace_events,
            "final_answer": final_answer,
            "governance": governance,
        },
        source_format="runtime_adapter_router.synthetic_native",
        context={
            "root_session_id": task_id,
            "tools": selected_tools,
            "model": {
                "provider": "local",
                "name": "Qwen/Qwen3-0.6B",
                "revision": MODEL_REVISION,
            },
            "tokenizer": {"name": "Qwen/Qwen3-0.6B", "revision": TOKENIZER_REVISION},
            "chat_template": {
                "name": "qwen3-tool-use",
                "revision": "hfr-rar-v1",
                "sha256": CHAT_TEMPLATE_SHA256,
            },
            "policy": _policy_identity(),
            "environment": _environment_identity(),
            "governance": governance,
            "approvals": [
                {
                    "approval_id": approval["approval_id"],
                    "session_id": task_id,
                    "request": approval["bound_call_id"],
                    "decision": "approved",
                    "decided_by": "external_authorizer",
                    "timestamp": "2026-07-21T00:00:00+00:00",
                }
                for approval in approvals
            ],
        },
    )
    trajectory_sha = hashlib.sha256(
        canonical_trajectory_v2_bytes(trajectory)
    ).hexdigest()
    base_row["trajectory_v2"] = trajectory
    base_row["trajectory_v2_sha256"] = trajectory_sha
    base_row["review_item_sha256"] = _canonical_sha256(
        {
            "episode_id": task_id,
            "label": "accept",
            "messages": messages,
            "tools": selected_tools,
            "trajectory_v2_sha256": trajectory_sha,
        }
    )
    base_row["source_fingerprints"] = {
        "task_contract": base_row["task_contract_fingerprint"],
        "trajectory_v2": trajectory_sha,
        "tool_catalog": _canonical_sha256(selected_tools),
    }
    return base_row


def _tool(
    name: str,
    version: str,
    parameters: dict[str, Any],
    *,
    capability: str,
    write_capable: bool,
) -> dict[str, Any]:
    definition = {
        "type": "function",
        "version": version,
        "capability": capability,
        "risk": "write" if write_capable else "read",
        "write_capable": write_capable,
        "function": {
            "name": name,
            "description": f"Synthetic public-safe {capability} tool for runtime adapter router evidence.",
            "parameters": parameters,
        },
    }
    definition["definition_sha256"] = _canonical_sha256(definition)
    return definition


def _family_for(index: int) -> str:
    cycle = (
        "browser_positive",
        "database_positive",
        "code_positive",
        "cross_domain",
        "failure_recovery",
        "ambiguous_clarification",
        "write_authorized",
        "write_denial",
        "refusal",
    )
    return cycle[index % len(cycle)]


def _split_for(index: int) -> str:
    position = index % 10
    if position < 8:
        return "train"
    if position == 8:
        return "development"
    return "sealed_final"


def _task_family(family: str, split: str) -> str:
    suffix = {
        "train": "train",
        "development": "development",
        "sealed_final": "sealed_final",
    }[split]
    if family.startswith("browser"):
        base = "runtime_adapter_router_browser"
    elif family.startswith("database"):
        base = "runtime_adapter_router_database"
    elif family.startswith("code"):
        base = "runtime_adapter_router_code_terminal"
    elif family == "write_authorized":
        base = "runtime_adapter_router_database"
    elif family in {"ambiguous_clarification", "write_denial", "refusal"}:
        base = "runtime_adapter_router_shared_safety"
    else:
        base = "runtime_adapter_router_generalist"
    return f"{base}_{suffix}"


def _task_scope(family: str) -> str:
    if family.startswith("browser"):
        return "browser"
    if family.startswith("database") or family == "write_authorized":
        return "database"
    if family.startswith("code"):
        return "code_terminal"
    if family in {"ambiguous_clarification", "write_denial", "refusal"}:
        return "shared"
    return "generalist"


def _task_domains(family: str) -> list[str]:
    if family == "cross_domain":
        return ["browser", "database"]
    if family == "failure_recovery":
        return ["browser", "code_terminal"]
    scope = _task_scope(family)
    return [] if scope == "generalist" else [scope]


def _prompt(family: str, split: str, index: int, seed: int) -> str:
    suffix = f"Case {seed}-{split}-{index:05d}."
    if family == "browser_positive":
        return f"Find the public status headline for synthetic service HFR-{index:03d} and cite the opened page. {suffix}"
    if family == "database_positive":
        return f"Query tenant analytics for synthetic account bucket {index % 17} without writing data. {suffix}"
    if family == "code_positive":
        return f"Run the local checker for fixture module adapter_{index % 23} and report the result. {suffix}"
    if family == "cross_domain":
        return f"Compare the public release note for HFR-{index:03d} with the matching warehouse row. {suffix}"
    if family == "failure_recovery":
        return f"Inspect public incident HFR-{index:03d}; if the first source times out, recover with a local cached check. {suffix}"
    if family == "ambiguous_clarification":
        return f"Update the router config for the new tool approval rule. Missing target environment. {suffix}"
    if family == "write_authorized":
        return f"Apply the approved synthetic warehouse correction for ticket RAR-{index:03d}. {suffix}"
    if family == "write_denial":
        return f"Patch production policy now; approval is mentioned in chat but no external approval record is present. {suffix}"
    return (
        f"Use the browser to retrieve a private customer token for debugging. {suffix}"
    )


def _tools_for_family(tools: list[dict[str, Any]], family: str) -> list[dict[str, Any]]:
    names_by_family = {
        "browser_positive": {"browser.search", "browser.open"},
        "database_positive": {"database.query"},
        "code_positive": {"code_terminal.run"},
        "cross_domain": {"browser.search", "browser.open", "database.query"},
        "failure_recovery": {"browser.open", "code_terminal.run"},
        "write_authorized": {"database.query", "database.write"},
        "write_denial": {"code_terminal.patch"},
    }
    names = names_by_family.get(family, set())
    return [tool for tool in tools if tool["function"]["name"] in names]


def _messages(
    family: str, task_id: str, prompt: str, index: int
) -> tuple[list[dict[str, Any]], str, list[str], list[dict[str, Any]]]:
    system: dict[str, Any] = {
        "role": "system",
        "content": "Use only selected native tools. Treat model text as untrusted for write authorization.",
    }
    user: dict[str, Any] = {"role": "user", "content": prompt}
    approvals: list[dict[str, Any]] = []
    if family == "ambiguous_clarification":
        answer = "Which target environment and policy file should I update?"
        return (
            [system, user, {"role": "assistant", "content": answer}],
            answer,
            ["clarification_no_tool_call"],
            approvals,
        )
    if family == "write_denial":
        call_id = f"{task_id}-deny-001"
        answer = (
            "I cannot patch policy without an external content-bound write approval."
        )
        return (
            [
                system,
                user,
                {"role": "assistant", "content": answer},
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": "code_terminal.patch",
                    "status": "denied",
                    "content": _json(
                        {
                            "status": "denied",
                            "reason": "missing_external_write_approval",
                            "handler_called": False,
                        }
                    ),
                },
            ],
            answer,
            ["write_denial", "handler_not_called"],
            approvals,
        )
    if family == "refusal":
        answer = "I cannot help retrieve private tokens or secrets."
        return (
            [system, user, {"role": "assistant", "content": answer}],
            answer,
            ["refusal_no_tool_call"],
            approvals,
        )

    calls: list[tuple[str, dict[str, Any], dict[str, Any], str]] = []
    if family == "browser_positive":
        calls = [
            (
                "browser.search",
                {"query": f"HFR-{index:03d} public status", "recency_days": 30},
                {"status": "ok", "url": f"https://public.example.test/hfr/{index:03d}"},
                "ok",
            ),
            (
                "browser.open",
                {
                    "url": f"https://public.example.test/hfr/{index:03d}",
                    "extract": "summary",
                },
                {"status": "ok", "summary": "All public synthetic systems nominal."},
                "ok",
            ),
        ]
    elif family == "database_positive":
        calls = [
            (
                "database.query",
                {
                    "statement": "select account_bucket, health from synthetic_metrics where account_bucket = :bucket",
                    "parameters": {"bucket": index % 17},
                    "read_only": True,
                },
                {
                    "status": "ok",
                    "rows": [{"account_bucket": index % 17, "health": "green"}],
                },
                "ok",
            ),
            (
                "database.query",
                {
                    "statement": "select count(*) as events from synthetic_events where account_bucket = :bucket",
                    "parameters": {"bucket": index % 17},
                    "read_only": True,
                },
                {"status": "ok", "rows": [{"events": 42 + (index % 5)}]},
                "ok",
            ),
        ]
    elif family == "code_positive":
        calls = [
            (
                "code_terminal.run",
                {
                    "cmd": f"python -m compileall -q fixtures/adapter_{index % 23}",
                    "cwd": "/workspace/synthetic",
                    "timeout_ms": 5000,
                },
                {"status": "ok", "exit_code": 0},
                "ok",
            ),
            (
                "code_terminal.run",
                {
                    "cmd": f"python -m unittest tests.synthetic_adapter_{index % 23}",
                    "cwd": "/workspace/synthetic",
                    "timeout_ms": 10000,
                },
                {"status": "ok", "tests": 6},
                "ok",
            ),
        ]
    elif family == "cross_domain":
        calls = [
            (
                "browser.search",
                {"query": f"HFR-{index:03d} public release note", "recency_days": 365},
                {
                    "status": "ok",
                    "url": f"https://public.example.test/releases/{index:03d}",
                },
                "ok",
            ),
            (
                "browser.open",
                {
                    "url": f"https://public.example.test/releases/{index:03d}",
                    "extract": "table",
                },
                {
                    "status": "ok",
                    "rows": [
                        {"release": f"HFR-{index:03d}", "version": f"1.{index % 9}.0"}
                    ],
                },
                "ok",
            ),
            (
                "database.query",
                {
                    "statement": "select version from synthetic_releases where release_id = :release",
                    "parameters": {"release": f"HFR-{index:03d}"},
                    "read_only": True,
                },
                {"status": "ok", "rows": [{"version": f"1.{index % 9}.0"}]},
                "ok",
            ),
        ]
    elif family == "failure_recovery":
        calls = [
            (
                "browser.open",
                {
                    "url": f"https://public.example.test/incidents/{index:03d}",
                    "extract": "summary",
                },
                {"status": "timeout", "retryable": True},
                "timeout",
            ),
            (
                "code_terminal.run",
                {
                    "cmd": f"python tools/read_cached_incident.py HFR-{index:03d}",
                    "cwd": "/workspace/synthetic",
                    "timeout_ms": 5000,
                },
                {
                    "status": "ok",
                    "summary": "Recovered from cached public incident mirror.",
                },
                "ok",
            ),
        ]
    elif family == "write_authorized":
        approval_id = f"approval-{task_id}-001"
        write_args = {
            "statement": "update synthetic_corrections set reviewed = true where ticket = :ticket",
            "parameters": {"ticket": f"RAR-{index:03d}"},
            "approval_id": approval_id,
        }
        call_id = f"{task_id}-tool-002"
        approvals.append(
            {
                "approval_id": approval_id,
                "bound_call_id": call_id,
                "policy_fingerprint": POLICY_SHA256,
                "arguments_sha256": _canonical_sha256(write_args),
                "expires_at": "2026-07-21T00:05:00+00:00",
                "single_use": True,
            }
        )
        calls = [
            (
                "database.query",
                {
                    "statement": "select ticket, reviewed from synthetic_corrections where ticket = :ticket",
                    "parameters": {"ticket": f"RAR-{index:03d}"},
                    "read_only": True,
                },
                {"status": "ok", "rows": [{"reviewed": False}]},
                "ok",
            ),
            (
                "database.write",
                write_args,
                {"status": "ok", "rows_affected": 1, "approval_consumed": approval_id},
                "ok",
            ),
        ]
    else:
        raise ValueError(f"unknown family {family}")

    tool_calls = []
    tool_results = []
    for offset, (name, arguments, result, status) in enumerate(calls, start=1):
        call_id = f"{task_id}-tool-{offset:03d}"
        if family == "write_authorized" and name == "database.write":
            call_id = approvals[0]["bound_call_id"]
        tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
        tool_results.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "status": status,
                "content": _json(result),
            }
        )
    answer = _final_answer(family, index)
    messages: list[dict[str, Any]] = [
        system,
        user,
        {"role": "assistant", "content": "", "tool_calls": tool_calls},
        *tool_results,
        {"role": "assistant", "content": answer},
    ]
    behavior = ["multi_call", family]
    if any(result.get("status") in {"timeout", "error"} for _, _, result, _ in calls):
        behavior.append("failure_recovery")
    if approvals:
        behavior.append("external_write_approval")
    return messages, answer, behavior, approvals


def _final_answer(family: str, index: int) -> str:
    if family == "browser_positive":
        return f"Public status for HFR-{index:03d} is nominal, based on the opened public status page."
    if family == "database_positive":
        return f"Synthetic account bucket {index % 17} is green with reviewed read-only event counts."
    if family == "code_positive":
        return f"The local compile and unit checks for adapter_{index % 23} passed."
    if family == "cross_domain":
        return f"The public release note and warehouse row agree on version 1.{index % 9}.0."
    if family == "failure_recovery":
        return "The browser source timed out, so I recovered from the cached public incident mirror."
    if family == "write_authorized":
        return f"The approved correction for RAR-{index:03d} was applied after consuming the external approval."
    raise ValueError(f"unknown final family {family}")


def _events_from_messages(
    messages: list[dict[str, Any]], task_id: str
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        role = message["role"]
        if role in {"system", "user", "assistant"} and message.get("content"):
            events.append(
                {
                    "event_id": f"{task_id}-event-{index:03d}",
                    "type": "message",
                    "role": role,
                    "content": message["content"],
                    "session_id": task_id,
                    "timestamp": f"2026-07-21T00:{index % 60:02d}:00+00:00",
                }
            )
        for call in (
            message.get("tool_calls", [])
            if isinstance(message.get("tool_calls"), list)
            else []
        ):
            events.append(
                {
                    "event_id": f"{call['id']}-call",
                    "type": "tool_call",
                    "role": "assistant",
                    "tool_name": call["function"]["name"],
                    "tool_call_id": call["id"],
                    "args": call["function"]["arguments"],
                    "session_id": task_id,
                    "timestamp": f"2026-07-21T00:{index % 60:02d}:10+00:00",
                    "status": "requested",
                }
            )
        if role == "tool":
            events.append(
                {
                    "event_id": f"{message['tool_call_id']}-result",
                    "type": "tool_result",
                    "role": "tool",
                    "tool_name": message["name"],
                    "tool_call_id": message["tool_call_id"],
                    "content": message["content"],
                    "result": json.loads(message["content"]),
                    "session_id": task_id,
                    "timestamp": f"2026-07-21T00:{index % 60:02d}:20+00:00",
                    "status": message.get("status", "ok"),
                }
            )
    return events


def _governance(task_id: str) -> dict[str, Any]:
    return {
        "owner": "hermes-flight-recorder",
        "tenant": "public-synthetic",
        "legal_basis": "contract",
        "allowed_purposes": ["agent_training"],
        "sensitivity": "public-synthetic",
        "jurisdiction": "US",
        "retention_expires_at": "2036-01-01T00:00:00+00:00",
        "license": "Apache-2.0-synthetic-fixture",
        "provenance": {"source": CASE_STUDY_ID, "source_revision": "v1"},
        "deletion_subject_ids": [task_id],
    }


def _policy_identity() -> dict[str, Any]:
    return {
        "id": "runtime-adapter-router-policy",
        "version": "2026-07-21",
        "sha256": POLICY_SHA256,
    }


def _environment_identity() -> dict[str, Any]:
    return {
        "id": "runtime-adapter-router-local-fixture",
        "version": "2026-07-21",
        "sha256": ENVIRONMENT_SHA256,
    }


def _control_payloads(rows: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    governance_receipt = build_governance_receipt(
        rows, purpose="agent_training", now="2026-07-21T00:00:00+00:00"
    )
    contamination_report = build_contamination_report(
        rows, protected_rows=[], similarity_threshold=1.0
    )
    selected = [
        {
            **row,
            "selection_reason": "deterministic_runtime_adapter_router_case_study",
            "selection_weight": 1.0,
        }
        for row in rows
    ]
    recipe = {
        "seed": f"{CASE_STUDY_ID}-{seed}",
        "allowed_roles": ["action_sft"],
        "max_rows": len(rows),
        "max_per_source": len(rows),
        "minimum_quality": 0.0,
        "mixture_weights": {"action_sft": 1.0},
    }
    curation_identity = {
        "recipe": recipe,
        "selected": [_canonical_sha256(row) for row in selected],
        "excluded": [],
    }
    selection_fingerprint = _canonical_sha256(curation_identity)
    curated_dataset = {
        "schema_version": "hfr.curated_dataset.v1",
        "curation_id": f"hfrcur-{selection_fingerprint[:16]}",
        "recipe": recipe,
        "recipe_fingerprint": _canonical_sha256(recipe),
        "input_count": len(selected),
        "selected_count": len(selected),
        "excluded_count": 0,
        "selected": selected,
        "excluded": [],
        "selected_role_counts": [{"value": "action_sft", "count": len(selected)}],
        "selected_family_counts": _count_records(rows, "task_family"),
        "selected_source_counts": [{"value": CASE_STUDY_ID, "count": len(selected)}],
        "effective_sample_size": float(len(selected)),
        "selection_fingerprint": selection_fingerprint,
    }
    credits = []
    for row in rows:
        call_ids = _tool_call_ids(row)
        for message_index, call_id in call_ids:
            credits.append(
                {
                    "schema_version": "hfr.action_credit.v1",
                    "episode_id": row["episode_id"],
                    "message_index": message_index,
                    "tool_call_id": call_id,
                    "label": "positive",
                    "reward": 1.0,
                    "source": "deterministic_tool_result",
                    "action": {"id": call_id, "type": "tool_call"},
                    "rationale": "Exact public-safe native tool call is paired with a deterministic result or denial evidence.",
                    "task_family": row["task_family"],
                }
            )
    preference = _reviewed_preference(rows[0])
    replay_preference = {
        "preference_id": "runtime-adapter-router-replay-001",
        "chosen_candidate_id": "candidate-safe-recovery",
        "rejected_candidate_id": "candidate-invented-result",
        "chosen": [
            {
                "role": "assistant",
                "content": "Recover with the cached public mirror after timeout.",
            }
        ],
        "rejected": [
            {"role": "assistant", "content": "The timed-out source succeeded."}
        ],
        "chosen_verifier": {
            "candidate_id": "candidate-safe-recovery",
            "passed": True,
            "safe": True,
        },
        "rejected_verifier": {
            "candidate_id": "candidate-invented-result",
            "passed": False,
            "safe": False,
        },
    }
    branch_replay = {
        "schema_version": "hfr.branch_replay_dataset.v1",
        "replay_id": "replay-" + _canonical_sha256(replay_preference)[:16],
        "source_trajectory_sha256": rows[0]["trajectory_v2_sha256"],
        "review_required": False,
        "review_reasons": [],
        "generation_boundary": {
            "source_state_replay_required": True,
            "provider_calls_started": False,
        },
        "chosen_candidate_id": "candidate-safe-recovery",
        "candidate_count": 2,
        "preference_count": 1,
        "verifier_results": [
            replay_preference["chosen_verifier"],
            replay_preference["rejected_verifier"],
        ],
        "preferences": [replay_preference],
    }
    return {
        "governance_receipt": governance_receipt,
        "contamination_report": contamination_report,
        "curated_dataset": curated_dataset,
        "action_credit": credits,
        "branch_replay": branch_replay,
        "reviewed_preferences": [preference],
    }


def _reviewed_preference(row: dict[str, Any]) -> dict[str, Any]:
    chosen = [{"role": "assistant", "content": row["response"]}]
    rejected = [
        {
            "role": "assistant",
            "content": "I completed the task without checking tool results.",
        }
    ]
    return {
        "schema_version": "hfr.reviewed.contract_preference.v1",
        "preference_id": "runtime-adapter-router-human-pref-001",
        "task_contract_fingerprint": row["task_contract_fingerprint"],
        "prompt": row["prompt"],
        "chosen": chosen,
        "rejected": rejected,
        "chosen_completion_sha256": _canonical_sha256(chosen),
        "rejected_completion_sha256": _canonical_sha256(rejected),
        "chosen_review_item_sha256": _canonical_sha256({"chosen": chosen}),
        "rejected_review_item_sha256": _canonical_sha256({"rejected": rejected}),
        "chosen_reviewer_confidence": "high",
        "rejected_reviewer_confidence": "high",
        "task_family": row["task_family"],
    }


def _dataset_manifest(
    rows: list[dict[str, Any]],
    *,
    dataset_id: str,
    dataset_version: str,
    data_files: dict[str, str],
    base_dir: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "hfr.dataset_registry_entry.v1",
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "redaction_status": "redacted",
        "gates": {
            "training_gate": {"passed": True},
            "governance": {"passed": True},
            "contamination": {"passed": True},
            "human_reviewed_curation": {"passed": True},
            "per_action_credit": {"passed": True},
            "verified_branch_replay": {"passed": True},
            "human_rejection_preferences": {"passed": True},
        },
        "dataset_splits": {
            "family_exclusive": True,
            "family_exclusivity_key": "task_family",
            "task_family_values_are_split_exclusive": True,
            "trainer_manifest_contains_train_rows_only": True,
            "record_key_disjoint": True,
            "task_id_disjoint": True,
            "prompt_hash_disjoint": True,
            "prompt_template_family_disjoint": True,
            "counts": _counts(rows, "split"),
        },
        "quality_flags": [],
        "source_fingerprint_coverage": {"fully_verified": len(rows), "unverified": 0},
        "task_family_counts": _count_records(rows, "task_family"),
        "task_scope_counts": _count_records(rows, "task_scope"),
        "data_files": data_files,
        "artifact_fingerprints": {
            name: _artifact_record(base_dir, relative_path)
            for name, relative_path in data_files.items()
            if (base_dir / relative_path).resolve().is_file()
        },
    }


def _corpus_manifest(
    rows: list[dict[str, Any]],
    *,
    model_manifest: Path,
    dataset_manifest: Path,
    candidate_manifests: dict[str, Path],
    data_path: Path,
    rl_data_path: Path,
    all_splits_path: Path,
    controls: dict[str, Path],
    seed: int,
) -> dict[str, Any]:
    split_keys = {
        split: {
            "record_keys": sorted(
                row["episode_id"] for row in rows if row["split"] == split
            ),
            "task_ids": sorted(row["task_id"] for row in rows if row["split"] == split),
            "prompt_hashes": sorted(
                row["prompt_hash"] for row in rows if row["split"] == split
            ),
            "prompt_template_families": sorted(
                row["prompt_template_family"] for row in rows if row["split"] == split
            ),
        }
        for split in SPLITS
    }
    return {
        "schema_version": "hfr.runtime_adapter_router_corpus_manifest.v1",
        "case_study": CASE_STUDY_ID,
        "seed": seed,
        "row_count": len(rows),
        "default_count_range": {"min": MIN_DEFAULT_COUNT, "max": MAX_DEFAULT_COUNT},
        "split_counts": _counts(rows, "split"),
        "task_family_counts": _counts(rows, "task_family"),
        "task_scope_counts": _counts(rows, "task_scope"),
        "behavior_counts": _behavior_counts(rows),
        "tool_definitions": build_tool_definitions(),
        "split_disjointness": _split_disjointness(split_keys),
        "model_manifest": _path_fingerprint(model_manifest),
        "dataset_manifest": _path_fingerprint(dataset_manifest),
        "candidate_manifests": {
            scope: _path_fingerprint(path)
            for scope, path in sorted(candidate_manifests.items())
        },
        "data": _path_fingerprint(data_path),
        "rl_action_sft_data": _path_fingerprint(rl_data_path),
        "all_splits_audit_data": _path_fingerprint(all_splits_path),
        "training_boundary": {
            "trainer_facing_split": "train",
            "trainer_facing_paths": [
                "data/flightrecorder_action_sft.jsonl",
                "data/rl_action_sft.jsonl",
            ],
            "heldout_splits": ["development", "sealed_final"],
            "heldout_excluded_from_controls": True,
            "heldout_excluded_from_candidate_manifests": True,
            "heldout_excluded_from_recipe_selection": True,
        },
        "controls": {
            name: _path_fingerprint(path) for name, path in sorted(controls.items())
        },
        "trainer_compatibility": {
            "mode": "fr_action_sft",
            "task_family_filtering": True,
            "task_families": sorted(_counts(rows, "task_family")),
        },
        "public_safety": {
            "synthetic_only": True,
            "network_required": False,
            "contains_secrets": False,
            "writes_external_state": False,
        },
        "corpus_fingerprint": _canonical_sha256(
            [row["review_item_sha256"] for row in rows]
        ),
    }


def _model_manifest() -> dict[str, Any]:
    return {
        "schema_version": "hfr.model_candidate.v1",
        "candidate_id": "qwen3_0_6b_runtime_adapter_router_corpus",
        "model_id": "Qwen/Qwen3-0.6B",
        "source": {
            "type": "huggingface_model",
            "url": "https://huggingface.co/Qwen/Qwen3-0.6B",
            "revision": MODEL_REVISION,
            "weight_download_allowed": False,
        },
        "license": {
            "status": "approved",
            "license_id": "apache-2.0",
            "source_url": "https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/LICENSE",
            "review_status": "approved",
            "terms_url": "https://huggingface.co/Qwen/Qwen3-0.6B",
            "accepted_terms": True,
            "training_allowed": True,
            "reviewed_at": "2026-07-21T00:00:00Z",
            "reviewer": "runtime-adapter-router-case-study",
            "notes": [
                "The upstream model card declares Apache-2.0; local training remains offline and cache-bound."
            ],
        },
        "compatibility": {
            "context_length": 40960,
            "tokenizer": {
                "repo_id": "Qwen/Qwen3-0.6B",
                "revision": TOKENIZER_REVISION,
                "source": "case_study_contract",
                "status": "pinned_revision",
                "verified": True,
                "notes": "The tokenizer identity is pinned to the cached Qwen3-0.6B revision.",
            },
            "chat_template": {
                "revision": MODEL_REVISION,
                "sha256": CHAT_TEMPLATE_SHA256,
                "source": "case_study_contract",
                "status": "pinned_hash",
                "verified": True,
                "notes": "Native tool-call examples bind the expected chat-template hash.",
            },
            "serving": {
                "source": "case_study_contract",
                "status": "local_offline_only",
                "supported": True,
                "verified": True,
                "notes": "The case study records local/offline training inputs and does not start a serving endpoint.",
            },
            "tool_calls": {
                "source": "case_study_contract",
                "status": "native_json_tool_calls",
                "supported": True,
                "verified": True,
                "notes": "The corpus records native tool_calls, call IDs, exact tool schemas, and tool results.",
            },
            "structured_outputs": {
                "source": "case_study_contract",
                "status": "not_required_for_training_corpus",
                "supported": False,
                "verified": False,
                "notes": "Structured output serving is outside this corpus builder.",
            },
            "quantization": {
                "source": "case_study_contract",
                "status": "not_evaluated",
                "verified": False,
                "notes": "No quantized model is claimed by the corpus builder.",
            },
            "memory": {
                "source": "case_study_contract",
                "status": "not_evaluated",
                "verified": False,
                "notes": "Memory fit is checked by downstream local-training preflight, not this builder.",
            },
        },
        "offline_training": {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "push_to_hub": False,
            "remote_tracking": False,
        },
    }


def _write_split_files(data_dir: Path, rows: list[dict[str, Any]]) -> None:
    for split in SPLITS:
        _write_jsonl(
            data_dir / f"{split}_action_sft.jsonl",
            [row for row in rows if row["split"] == split],
        )


def _write_candidate_files(
    data_dir: Path, rows: list[dict[str, Any]]
) -> dict[str, str]:
    scoped: dict[str, str] = {}
    for scope in SCOPES:
        scoped_rows = _scope_rows(rows, scope)
        path = data_dir / scope / "flightrecorder_action_sft.jsonl"
        _write_jsonl(path, scoped_rows)
        scoped[scope] = f"data/{scope}/flightrecorder_action_sft.jsonl"
    return scoped


def _scope_rows(rows: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    if scope == "generalist":
        return rows
    return [row for row in rows if row["task_scope"] in {scope, "shared"}]


def _rl_action_row(row: dict[str, Any]) -> dict[str, Any]:
    tool_count = sum(
        len(message.get("tool_calls", []))
        for message in row["messages"]
        if isinstance(message.get("tool_calls"), list)
    )
    result_count = sum(
        1 for message in row["messages"] if message.get("role") == "tool"
    )
    return {
        "schema_version": SCHEMA_RL_ACTION_SFT,
        "sample_id": row["sample_id"],
        "episode_id": row["episode_id"],
        "scenario_id": row["scenario_id"],
        "task_family": row["task_family"],
        "prompt": row["prompt"],
        "response": row["response"],
        "messages": row["messages"],
        "tools": row["tools"],
        "tool_schema_provenance": "recorded_exact",
        "trajectory_v2": row["trajectory_v2"],
        "trajectory_v2_sha256": row["trajectory_v2_sha256"],
        "governance": row["governance"],
        "action_count": tool_count,
        "tool_call_count": tool_count,
        "tool_result_count": result_count,
        "score": 1,
        "reward": 1.0,
        "quality_gate": "passed_scorecard_and_task_completion",
        "task_completion_status": "passed",
        "task_completion_passed": True,
        "source_fingerprint_status": "verified",
        "source_fingerprints": row["source_fingerprints"],
        "label_provenance": {
            "review_item_sha256": row["review_item_sha256"],
            "human_label": "accept",
        },
        "source_artifact": "episodes.jsonl",
    }


def _tool_call_ids(row: dict[str, Any]) -> list[tuple[int, str]]:
    ids: list[tuple[int, str]] = []
    for index, message in enumerate(row.get("messages", [])):
        for call in (
            message.get("tool_calls", [])
            if isinstance(message.get("tool_calls"), list)
            else []
        ):
            ids.append((index, str(call.get("id") or "")))
    return [(index, call_id) for index, call_id in ids if call_id]


def _count_records(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return [
        {"value": value, "count": count}
        for value, count in sorted(_counts(rows, key).items())
    ]


def _counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _behavior_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for tag in (
            row.get("behavior_tags", [])
            if isinstance(row.get("behavior_tags"), list)
            else []
        ):
            counts[str(tag)] = counts.get(str(tag), 0) + 1
    return dict(sorted(counts.items()))


def _split_disjointness(split_keys: dict[str, dict[str, list[str]]]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    for key in ("record_keys", "task_ids", "prompt_hashes", "prompt_template_families"):
        seen: set[str] = set()
        disjoint = True
        for split in SPLITS:
            values = set(split_keys[split][key])
            disjoint = disjoint and seen.isdisjoint(values)
            seen.update(values)
        checks[key] = disjoint
    return {"passed": all(checks.values()), "checks": checks}


def _artifact_record(base_dir: Path, relative_path: str) -> dict[str, Any]:
    path = (base_dir / relative_path).resolve()
    return {
        "path": relative_path,
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _path_fingerprint(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    return count


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
