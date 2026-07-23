"""Fail-closed official-Tau benchmark arm runner for the Tau-3 study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

from .path_safety import path_has_symlink_component
from .schema_registry import check_schema_contract
from .tau3_sealed_authorization import Tau3SealedAuthorizationError, validate_tau3_sealed_authorization

TAU3_BENCHMARK_RUN_SCHEMA_VERSION = "hfr.tau3_benchmark_run.v1"
TAU3_PROTOCOL_CONFIG_SCHEMA_VERSION = "hfr.tau3_protocol_config.v1"
DOMAINS = ("airline", "retail", "telecom")
DEFAULT_SEEDS = (101, 202, 303, 404)
REQUIRED_ARMS = ("adapter", "base", "comparator_1", "comparator_2")
CONTEXT_WINDOW = 16384
PROMPT_TOKEN_CEILING = CONTEXT_WINDOW


class Tau3BenchmarkRunError(ValueError):
    """Raised when a benchmark arm launch would violate study governance."""


@dataclass(frozen=True)
class Tau3BenchmarkEndpoint:
    model: str
    api_base: str
    adapter_path: Path | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1024
    context_window: int = CONTEXT_WINDOW


@dataclass(frozen=True)
class Tau3BenchmarkConfig:
    mode: str
    arm_id: str
    protocol_path: Path
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    timeout_seconds: int = 600
    max_steps: int = 30
    max_errors: int = 10
    domains: tuple[str, ...] = DOMAINS
    source_split: Path | None = None
    sealed_task_count_manifest: Path | None = None
    sealed_authorization: Path | None = None
    sealed_authorization_sha256: str | None = None
    candidate_identity: Path | None = None
    candidate_identity_sha256: str | None = None
    candidate_lock: Path | None = None
    candidate_lock_sha256: str | None = None
    save_prefix: str = "hfr-benchmark"
    command_timeout_padding_seconds: int = 30
    extra_binding: dict[str, Any] = field(default_factory=dict)


def run_tau3_benchmark_arm(
    *,
    out_dir: str | Path,
    tau_repo: str | Path,
    tau_venv_bin: str | Path,
    expected_tau_revision: str,
    agent: Tau3BenchmarkEndpoint,
    user: Tau3BenchmarkEndpoint,
    reviewer: Tau3BenchmarkEndpoint,
    config: Tau3BenchmarkConfig,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Run one official Tau benchmark arm and write private receipts."""

    out = Path(out_dir)
    repo = Path(tau_repo).resolve(strict=True)
    tau2 = Path(tau_venv_bin)
    if tau2.is_dir():
        tau2 = tau2 / "tau2"
    if not tau2.is_absolute():
        tau2 = Path.cwd() / tau2
    if out.exists() and not out.is_dir():
        raise Tau3BenchmarkRunError(f"output exists and is not a directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    _validate_config(config)
    _require_loopback(agent.api_base, "agent")
    _require_loopback(user.api_base, "user")
    _require_loopback(reviewer.api_base, "reviewer")
    if user.adapter_path is not None:
        raise Tau3BenchmarkRunError("user simulator endpoint must not receive an adapter path")
    if reviewer.adapter_path is not None:
        raise Tau3BenchmarkRunError("reviewer endpoint must not receive an adapter path")
    _validate_endpoint(agent, "agent")
    _validate_endpoint(user, "user")
    _validate_endpoint(reviewer, "reviewer")
    protocol = _load_protocol(config.protocol_path)
    protocol_hash = _sha256(config.protocol_path)
    _validate_protocol_binding(
        protocol,
        protocol_hash=protocol_hash,
        expected_tau_revision=expected_tau_revision,
        agent=agent,
        user=user,
        reviewer=reviewer,
        config=config,
    )
    _require_revision(repo, expected_tau_revision)
    tasks_by_domain = _development_tasks_by_domain(config.source_split, expected_tau_revision) if config.mode == "development" else None
    sealed_task_count_manifest = (
        _sealed_task_count_manifest_record(config.sealed_task_count_manifest, protocol=protocol, expected_revision=expected_tau_revision)
        if config.mode == "sealed"
        else None
    )
    protocol_ref = _stage_input_file(config.protocol_path, out, "inputs/protocol.json", "protocol")
    source_ref = (
        _stage_input_file(config.source_split, out, "inputs/development_source.json", "development source")
        if config.mode == "development" and config.source_split is not None
        else None
    )
    sealed_task_count_ref = (
        _stage_input_file(config.sealed_task_count_manifest, out, "inputs/sealed_task_count_manifest.json", "sealed task-count manifest")
        if config.mode == "sealed" and config.sealed_task_count_manifest is not None
        else None
    )
    sealed_authorization_ref = (
        _stage_input_file(config.sealed_authorization, out, "inputs/sealed_authorization.json", "sealed authorization")
        if config.mode == "sealed" and config.sealed_authorization is not None
        else None
    )
    candidate_identity_ref = (
        _stage_input_file(config.candidate_identity, out, "inputs/candidate_identity.json", "candidate identity")
        if config.mode == "development" and config.arm_id == "adapter" and config.candidate_identity is not None
        else None
    )
    candidate_lock_ref = (
        _stage_input_file(config.candidate_lock, out, "inputs/candidate_lock.json", "candidate lock")
        if config.mode == "sealed" and config.candidate_lock is not None
        else None
    )
    candidate_lock = _candidate_lock_record(config, candidate_lock_ref) if config.mode == "sealed" else None
    sealed_authorization = (
        _sealed_authorization_binding(config, sealed_authorization_ref, out=out, expected_tau_revision=expected_tau_revision)
        if config.mode == "sealed"
        else None
    )
    candidate_identity = _candidate_identity_record(config, candidate_identity_ref) if config.mode == "development" and config.arm_id == "adapter" else None
    adapter_binding = _adapter_binding(agent, config)
    arm_identity = _arm_identity_record(
        protocol,
        agent=agent,
        config=config,
        candidate_lock=candidate_lock,
        candidate_identity=candidate_identity,
        adapter_binding=adapter_binding,
    )
    _validate_mode_source_binding(protocol, config=config, source_tasks=tasks_by_domain, candidate_lock=candidate_lock)

    expected_binding = {
        "protocol": protocol_ref,
        "protocol_sha256": protocol_hash,
        "tau_revision": expected_tau_revision,
        "mode": config.mode,
        "arm_id": config.arm_id,
        "arm_identity": arm_identity,
        "agent": _endpoint_record(agent),
        "user_simulator": _endpoint_record(user),
        "reviewer": _endpoint_record(reviewer),
        "config": _config_record(config),
        "source": source_ref,
        "sealed_task_count_manifest": _sealed_task_count_binding(sealed_task_count_ref, sealed_task_count_manifest),
        "sealed_authorization": sealed_authorization,
        "candidate_identity": candidate_identity,
        "candidate_lock": candidate_lock,
        "task_selection": _task_selection_record(
            config,
            tasks_by_domain,
            sealed_task_count=sealed_task_count_manifest["task_count"] if sealed_task_count_manifest is not None else None,
        ),
        "extra_binding": config.extra_binding,
    }
    final_manifest_path = out / "manifest.json"
    if final_manifest_path.exists():
        manifest = _read_json(final_manifest_path)
        _validate_resumable_manifest(manifest, out=out, expected_binding=expected_binding)
        return manifest

    prelaunch = {
        "schema_version": TAU3_BENCHMARK_RUN_SCHEMA_VERSION,
        "phase": "prelaunch",
        "created_at": created_at or _now_utc(),
        **expected_binding,
        "out_dir": ".",
        "loopback_only": True,
        "training_started": False,
        "sealed_payload_accessed": False,
        "sealed_task_ids_materialized": False,
    }
    prelaunch_path = out / "prelaunch_receipt.json"
    if prelaunch_path.exists():
        existing = _read_json(prelaunch_path)
        _require_matching_binding(existing, expected_binding, label="existing prelaunch receipt")
    else:
        _write_json_new(prelaunch_path, prelaunch)

    receipts: list[dict[str, Any]] = []
    for domain in config.domains:
        for seed in config.seeds:
            receipt_path = out / f"run-{domain}-seed{seed}.json"
            save_to = f"{config.save_prefix}/{out.name}/{config.mode}-{config.arm_id}-{domain}-seed{seed}"
            argv = _tau2_argv(
                tau2=tau2,
                domain=domain,
                seed=seed,
                save_to=save_to,
                agent=agent,
                user=user,
                reviewer=reviewer,
                config=config,
                task_ids=tasks_by_domain[domain] if tasks_by_domain is not None else None,
            )
            if receipt_path.exists():
                receipt = _read_json(receipt_path)
                _validate_task_receipt(receipt, receipt_path=receipt_path, command=_redact_argv(argv))
                receipts.append(receipt)
                continue
            result_path = repo / "data" / "simulations" / save_to / "results.json"
            if result_path.exists():
                raise Tau3BenchmarkRunError(f"refusing existing raw Tau output without receipt: {result_path}")
            command_timeout_seconds = _command_timeout_seconds(
                protocol=protocol,
                config=config,
                domain=domain,
                tasks_by_domain=tasks_by_domain,
                sealed_task_count=sealed_task_count_manifest["task_count"] if sealed_task_count_manifest is not None else None,
            )
            start = time.monotonic()
            try:
                proc = subprocess.run(
                    argv,
                    cwd=repo,
                    env=_reviewer_environment(reviewer),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=command_timeout_seconds,
                )
                exit_code = proc.returncode
                stdout = proc.stdout
                stderr = proc.stderr
                timed_out = False
            except subprocess.TimeoutExpired as exc:
                exit_code = None
                stdout = str(exc.stdout or "")
                stderr = str(exc.stderr or "")
                timed_out = True
            duration = time.monotonic() - start
            has_result = result_path.is_file()
            try:
                result_summary = _result_summary(result_path) if has_result else None
            except (OSError, Tau3BenchmarkRunError) as exc:
                result_summary = None
                stderr = f"{stderr}\nresult validation failed: {exc}"
            copied_result_rel = _copied_result_relative_path(domain, seed)
            copied_result_path = out / copied_result_rel
            if copied_result_path.exists():
                raise Tau3BenchmarkRunError(f"refusing existing copied Tau output without receipt: {copied_result_path}")
            if has_result:
                _copy_file_new(result_path, copied_result_path)
            valid_result = isinstance(result_summary, dict) and _valid_result_summary(result_summary)
            status = "completed" if exit_code == 0 and has_result and valid_result else ("timeout" if timed_out else "failed")
            receipt = {
                "schema_version": TAU3_BENCHMARK_RUN_SCHEMA_VERSION,
                "phase": "domain_seed",
                "created_at": created_at or _now_utc(),
                "protocol_sha256": protocol_hash,
                "arm_identity": arm_identity,
                "mode": config.mode,
                "arm_id": config.arm_id,
                "domain": domain,
                "seed": seed,
                "command": _redact_argv(argv),
                "result_path": copied_result_rel if has_result else None,
                "result_sha256": _sha256(copied_result_path) if has_result else None,
                "result_summary": result_summary,
                "exit_code": exit_code,
                "timed_out": timed_out,
                "duration_seconds": round(duration, 6),
                "terminal_status": status,
                "stdout_tail": _redact(stdout)[-1000:],
                "stderr_tail": _redact(stderr)[-1000:],
                "training_started": False,
                "sealed_payload_accessed": False,
                "sealed_task_ids_materialized": False,
            }
            _write_json_new(receipt_path, receipt)
            receipt_sha256 = _sha256(receipt_path)
            receipt["receipt_sha256"] = receipt_sha256
            receipts.append(receipt)

    success_count = sum(1 for receipt in receipts if receipt.get("terminal_status") == "completed")
    failure_count = len(receipts) - success_count
    manifest = {
        "schema_version": TAU3_BENCHMARK_RUN_SCHEMA_VERSION,
        "phase": "final",
        "created_at": created_at or _now_utc(),
        **expected_binding,
        "prelaunch_receipt": _output_file_record(prelaunch_path, out),
        "run_count": len(receipts),
        "success_count": success_count,
        "failure_count": failure_count,
        "run_receipts": [
            {
                "path": f"run-{receipt['domain']}-seed{receipt['seed']}.json",
                "receipt_sha256": receipt.get("receipt_sha256") or _sha256(out / f"run-{receipt['domain']}-seed{receipt['seed']}.json"),
                "domain": receipt["domain"],
                "seed": receipt["seed"],
                "terminal_status": receipt["terminal_status"],
                "result_path": receipt.get("result_path"),
                "result_sha256": receipt.get("result_sha256"),
                "result_summary": receipt.get("result_summary"),
            }
            for receipt in receipts
        ],
        "loopback_only": True,
        "training_started": False,
        "sealed_payload_accessed": False,
        "sealed_task_ids_materialized": False,
    }
    check = check_schema_contract(manifest, name_or_id="tau3_benchmark_run")
    if check["passed"] is not True:
        raise Tau3BenchmarkRunError("manifest schema failed: " + json.dumps(check["errors"], sort_keys=True))
    _write_json_new(final_manifest_path, manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("development", "sealed"), required=True)
    parser.add_argument("--arm-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", dest="protocol_path", type=Path, required=True)
    parser.add_argument("--tau-repo", type=Path, required=True)
    parser.add_argument("--tau-venv-bin", type=Path, required=True)
    parser.add_argument("--expected-tau-revision", required=True)
    parser.add_argument("--source-split", type=Path)
    parser.add_argument("--sealed-task-count-manifest", type=Path)
    parser.add_argument("--sealed-authorization", type=Path)
    parser.add_argument("--sealed-authorization-sha256")
    parser.add_argument("--candidate-identity", type=Path)
    parser.add_argument("--candidate-identity-sha256")
    parser.add_argument("--candidate-lock", type=Path)
    parser.add_argument("--candidate-lock-sha256")
    parser.add_argument("--agent-model", required=True)
    parser.add_argument("--agent-api-base", required=True)
    parser.add_argument("--agent-adapter-path", type=Path)
    parser.add_argument("--user-model", required=True)
    parser.add_argument("--user-api-base", required=True)
    parser.add_argument("--reviewer-model", required=True)
    parser.add_argument("--reviewer-api-base", required=True)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--domains", default=",".join(DOMAINS))
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--save-prefix", default="hfr-benchmark")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        manifest = run_tau3_benchmark_arm(
            out_dir=args.out,
            tau_repo=args.tau_repo,
            tau_venv_bin=args.tau_venv_bin,
            expected_tau_revision=args.expected_tau_revision,
            agent=Tau3BenchmarkEndpoint(model=args.agent_model, api_base=args.agent_api_base, adapter_path=args.agent_adapter_path),
            user=Tau3BenchmarkEndpoint(model=args.user_model, api_base=args.user_api_base),
            reviewer=Tau3BenchmarkEndpoint(model=args.reviewer_model, api_base=args.reviewer_api_base),
            config=Tau3BenchmarkConfig(
                mode=args.mode,
                arm_id=args.arm_id,
                protocol_path=args.protocol_path,
                source_split=args.source_split,
                sealed_task_count_manifest=args.sealed_task_count_manifest,
                sealed_authorization=args.sealed_authorization,
                sealed_authorization_sha256=args.sealed_authorization_sha256,
                candidate_identity=args.candidate_identity,
                candidate_identity_sha256=args.candidate_identity_sha256,
                candidate_lock=args.candidate_lock,
                candidate_lock_sha256=args.candidate_lock_sha256,
                seeds=_parse_int_csv(args.seeds, "seeds"),
                domains=_parse_str_csv(args.domains, "domains"),
                timeout_seconds=args.timeout_seconds,
                max_steps=args.max_steps,
                max_errors=args.max_errors,
                save_prefix=args.save_prefix,
            ),
        )
    except (OSError, Tau3BenchmarkRunError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {"manifest": str(Path(args.out) / "manifest.json"), "success_count": manifest["success_count"], "failure_count": manifest["failure_count"]},
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if manifest["failure_count"] == 0 else 1


def _tau2_argv(
    *,
    tau2: Path,
    domain: str,
    seed: int,
    save_to: str,
    agent: Tau3BenchmarkEndpoint,
    user: Tau3BenchmarkEndpoint,
    reviewer: Tau3BenchmarkEndpoint,
    config: Tau3BenchmarkConfig,
    task_ids: list[str] | None,
) -> list[str]:
    split = "train" if config.mode == "development" else "test"
    argv = [
        str(tau2),
        "run",
        "--domain",
        domain,
        "--task-set-name",
        domain,
        "--task-split-name",
        split,
    ]
    if task_ids is not None:
        argv.extend(["--task-ids", *task_ids])
    argv.extend(
        [
            "--num-trials",
            "1",
            "--agent",
            "llm_agent",
            "--agent-llm",
            agent.model,
            "--agent-llm-args",
            json.dumps(_endpoint_args(agent), separators=(",", ":")),
            "--user",
            "user_simulator",
            "--user-llm",
            user.model,
            "--user-llm-args",
            json.dumps(_endpoint_args(user), separators=(",", ":")),
            "--max-steps",
            str(config.max_steps),
            "--max-errors",
            str(config.max_errors),
            "--timeout",
            str(config.timeout_seconds),
            "--save-to",
            save_to,
            "--max-concurrency",
            "1",
            "--seed",
            str(seed),
            "--max-retries",
            "0",
            "--hallucination-retries",
            "0",
            "--auto-review",
            "--review-mode",
            "full",
            "--review-model",
            reviewer.model,
            "--enforce-communication-protocol",
            "--log-level",
            "ERROR",
        ]
    )
    return argv


def _endpoint_args(endpoint: Tau3BenchmarkEndpoint) -> dict[str, Any]:
    args: dict[str, Any] = {
        "api_base": endpoint.api_base,
        "api_key": "local",
        "temperature": endpoint.temperature,
        "top_p": endpoint.top_p,
        "max_tokens": endpoint.max_tokens,
        "num_retries": 0,
    }
    if endpoint.adapter_path is not None:
        args["extra_body"] = {"adapters": str(endpoint.adapter_path.resolve(strict=True))}
    return args


def _development_tasks_by_domain(path: Path | None, expected_revision: str) -> dict[str, list[str]]:
    if path is None:
        raise Tau3BenchmarkRunError("development mode requires --source-split")
    payload = _read_json(path)
    schema = check_schema_contract(payload, name_or_id="tau3_source_split")
    if schema["passed"] is not True:
        raise Tau3BenchmarkRunError("development source schema failed: " + "; ".join(schema["errors"]))
    if payload.get("schema_version") != "hfr.tau3_source_split.v1":
        raise Tau3BenchmarkRunError("development source is not a tau3_source_split manifest")
    if payload.get("split") != "development":
        raise Tau3BenchmarkRunError("development source split must be development")
    if payload.get("source_revision") != expected_revision:
        raise Tau3BenchmarkRunError("development source revision mismatch")
    rows = payload.get("tasks")
    if not isinstance(rows, list):
        raise Tau3BenchmarkRunError("development source manifest must contain a tasks list")
    if payload.get("task_count") != len(rows):
        raise Tau3BenchmarkRunError("development source task_count mismatch")
    family_ids = payload.get("family_ids")
    if not isinstance(family_ids, list) or payload.get("family_count") != len(family_ids):
        raise Tau3BenchmarkRunError("development source family_count mismatch")
    tasks_by_domain: dict[str, list[str]] = {domain: [] for domain in DOMAINS}
    seen = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise Tau3BenchmarkRunError(f"development source row {index} is not an object")
        domain = str(row.get("domain") or "")
        if domain not in DOMAINS:
            raise Tau3BenchmarkRunError(f"development source row {index} has unsupported domain")
        raw_id = row.get("raw_id") or row.get("task_id")
        task = row.get("task") if isinstance(row.get("task"), dict) else None
        if raw_id is None and task is not None:
            raw_id = task.get("id")
        if raw_id is None:
            raise Tau3BenchmarkRunError(f"development source row {index} is missing raw task id")
        raw_id = str(raw_id)
        global_id = f"{domain}:{raw_id}"
        if row.get("raw_id_sha256") != hashlib.sha256(global_id.encode("utf-8")).hexdigest():
            raise Tau3BenchmarkRunError(f"development source row {index} raw task id hash mismatch")
        if row.get("family_id") not in family_ids:
            raise Tau3BenchmarkRunError(f"development source row {index} family is not declared")
        key = (domain, raw_id)
        if key in seen:
            raise Tau3BenchmarkRunError(f"duplicate development task id in {domain}: {raw_id}")
        seen.add(key)
        tasks_by_domain[domain].append(raw_id)
    for domain, ids in tasks_by_domain.items():
        if not ids:
            raise Tau3BenchmarkRunError(f"development source has no {domain} tasks")
    return tasks_by_domain


def _load_protocol(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Tau3BenchmarkRunError(f"protocol does not exist: {path}")
    payload = _read_json(path)
    schema = check_schema_contract(payload, name_or_id="tau3_protocol_config")
    if schema["passed"] is not True:
        raise Tau3BenchmarkRunError("protocol schema failed: " + "; ".join(schema["errors"]))
    if payload.get("schema_version") != TAU3_PROTOCOL_CONFIG_SCHEMA_VERSION:
        raise Tau3BenchmarkRunError("protocol is not a tau3_protocol_config manifest")
    return payload


def _validate_protocol_binding(
    protocol: dict[str, Any],
    *,
    protocol_hash: str,
    expected_tau_revision: str,
    agent: Tau3BenchmarkEndpoint,
    user: Tau3BenchmarkEndpoint,
    reviewer: Tau3BenchmarkEndpoint,
    config: Tau3BenchmarkConfig,
) -> None:
    revision = _dict(protocol.get("tau_revision")).get("revision")
    if revision != expected_tau_revision:
        raise Tau3BenchmarkRunError("protocol Tau revision mismatch")
    harness = _dict(protocol.get("harness_contract"))
    decoding = _dict(harness.get("decoding"))
    expected_harness = {
        "domains": list(DOMAINS),
        "context_window": CONTEXT_WINDOW,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_output_tokens": 1024,
        "seeds": list(DEFAULT_SEEDS),
        "turn_limit": 30,
        "retry_policy": "none",
        "test_time_search": False,
        "no_test_time_search": True,
    }
    actual_harness = {
        "domains": harness.get("domains"),
        "context_window": harness.get("context_window"),
        "temperature": decoding.get("temperature"),
        "top_p": decoding.get("top_p"),
        "max_output_tokens": decoding.get("max_output_tokens"),
        "seeds": decoding.get("seeds"),
        "turn_limit": harness.get("turn_limit"),
        "retry_policy": harness.get("retry_policy"),
        "test_time_search": harness.get("test_time_search"),
        "no_test_time_search": harness.get("no_test_time_search"),
    }
    if actual_harness != expected_harness:
        raise Tau3BenchmarkRunError("protocol harness contract does not match frozen benchmark values")
    for endpoint, label in ((agent, "agent"), (user, "user_simulator"), (reviewer, "reviewer")):
        _validate_endpoint(endpoint, label)
    if config.seeds != tuple(decoding["seeds"]):
        raise Tau3BenchmarkRunError("config seeds do not match protocol")
    if config.domains != tuple(harness["domains"]):
        raise Tau3BenchmarkRunError("config domains do not match protocol")
    if config.max_steps != harness["turn_limit"]:
        raise Tau3BenchmarkRunError("config max_steps does not match protocol turn_limit")
    if protocol_hash != _sha256(config.protocol_path):
        raise Tau3BenchmarkRunError("protocol hash changed during validation")


def _validate_mode_source_binding(
    protocol: dict[str, Any],
    *,
    config: Tau3BenchmarkConfig,
    source_tasks: dict[str, list[str]] | None,
    candidate_lock: dict[str, Any] | None,
) -> None:
    split_hashes = _protocol_split_hashes(protocol)
    if config.mode == "development":
        if config.source_split is None or source_tasks is None:
            raise Tau3BenchmarkRunError("development mode requires a protocol-bound source split")
        if _sha256(config.source_split) != split_hashes.get("development"):
            raise Tau3BenchmarkRunError("development source sha256 does not match protocol split hash")
        return
    sealed_manifest = _dict(protocol.get("sealed_manifest"))
    sealed_sha = sealed_manifest.get("manifest_sha256")
    if sealed_sha != split_hashes.get("sealed"):
        raise Tau3BenchmarkRunError("protocol sealed manifest hash does not match sealed split hash")
    if sealed_manifest.get("access_count") not in (0, None):
        raise Tau3BenchmarkRunError("protocol sealed manifest must declare zero sealed access")
    if candidate_lock is None:
        raise Tau3BenchmarkRunError("sealed mode requires a protocol-bound candidate lock")
    expected_lock = _protocol_candidate_lock_sha256(protocol)
    if expected_lock is not None and candidate_lock.get("sha256") != expected_lock:
        raise Tau3BenchmarkRunError("candidate lock sha256 does not match protocol")


def _arm_identity_record(
    protocol: dict[str, Any],
    *,
    agent: Tau3BenchmarkEndpoint,
    config: Tau3BenchmarkConfig,
    candidate_lock: dict[str, Any] | None,
    candidate_identity: dict[str, Any] | None,
    adapter_binding: dict[str, Any] | None,
) -> dict[str, Any]:
    if config.arm_id == "adapter":
        if adapter_binding is None:
            raise Tau3BenchmarkRunError("adapter arm requires --agent-adapter-path")
        if config.mode == "sealed":
            if candidate_lock is None:
                raise Tau3BenchmarkRunError("sealed adapter arm requires a candidate lock")
            lock = _read_json(config.candidate_lock) if config.candidate_lock is not None else {}
            allowed = _identity_strings_from_candidate_record(lock)
            if not allowed or not _endpoint_model_matches(agent.model, allowed, protocol_path=config.protocol_path):
                raise Tau3BenchmarkRunError("sealed adapter model identity does not match candidate lock")
            _require_candidate_adapter_tree_hash(lock, adapter_binding["tree_sha256"], "candidate lock")
            return {
                "arm_id": "adapter",
                "source": "candidate_lock",
                "candidate_lock_sha256": candidate_lock["sha256"],
                "candidate_identity_sha256": _candidate_identity_sha256_from_lock(lock),
                "endpoint_model_sha256": hashlib.sha256(agent.model.encode("utf-8")).hexdigest(),
                "adapter": adapter_binding,
            }
        if candidate_identity is None:
            raise Tau3BenchmarkRunError("development adapter arm requires --candidate-identity")
        identity = _read_json(config.candidate_identity) if config.candidate_identity is not None else {}
        allowed = _identity_strings_from_candidate_record(identity)
        if not allowed or not _endpoint_model_matches(agent.model, allowed, protocol_path=config.protocol_path):
            raise Tau3BenchmarkRunError("development adapter model identity does not match candidate identity")
        _require_candidate_adapter_tree_hash(identity, adapter_binding["tree_sha256"], "candidate identity")
        return {
            "arm_id": "adapter",
            "source": "candidate_identity",
            "candidate_identity_sha256": candidate_identity["sha256"],
            "candidate_record_sha256": _public_identity_hash(identity),
            "endpoint_model_sha256": hashlib.sha256(agent.model.encode("utf-8")).hexdigest(),
            "adapter": adapter_binding,
        }
    if adapter_binding is not None:
        raise Tau3BenchmarkRunError("agent adapter path is only allowed for the adapter arm")
    model_record = _protocol_model_for_arm(protocol, config.arm_id)
    allowed = _identity_strings_from_model_record(model_record)
    if not _endpoint_model_matches(agent.model, allowed, protocol_path=config.protocol_path):
        raise Tau3BenchmarkRunError(f"{config.arm_id} model identity does not match protocol")
    return {
        "arm_id": config.arm_id,
        "source": "protocol_model_freeze",
        "model_identity_sha256": str(model_record.get("local_identity_sha256") or ""),
        "model_record_sha256": _canonical_sha256(_public_model_record(model_record)),
        "endpoint_model_sha256": hashlib.sha256(agent.model.encode("utf-8")).hexdigest(),
    }


def _adapter_binding(agent: Tau3BenchmarkEndpoint, config: Tau3BenchmarkConfig) -> dict[str, Any] | None:
    if agent.adapter_path is None:
        if config.arm_id == "adapter":
            raise Tau3BenchmarkRunError("adapter arm requires --agent-adapter-path")
        return None
    if config.arm_id != "adapter":
        raise Tau3BenchmarkRunError("agent adapter path is only allowed for the adapter arm")
    if path_has_symlink_component(agent.adapter_path, include_leaf=True):
        raise Tau3BenchmarkRunError(f"agent adapter path must not contain symlink components: {agent.adapter_path}")
    adapter_path = agent.adapter_path.resolve(strict=True)
    if not adapter_path.is_dir():
        raise Tau3BenchmarkRunError(f"agent adapter path must be a directory: {agent.adapter_path}")
    tree = _fingerprint_tree(adapter_path)
    if tree["file_count"] <= 0 or tree["tree_sha256"] is None:
        raise Tau3BenchmarkRunError("agent adapter path must contain adapter artifacts")
    return tree


def _fingerprint_tree(root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rel = path.relative_to(root).as_posix()
        files.append({"path": rel, "size": path.stat().st_size, "sha256": _sha256(path), "kind": _fingerprint_kind(rel)})
    digest = hashlib.sha256()
    for record in files:
        digest.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return {"path": str(root), "file_count": len(files), "files": files, "tree_sha256": digest.hexdigest() if files else None}


def _fingerprint_kind(rel: str) -> str:
    name = Path(rel).name
    if name in {"adapter_config.json", "config.json"}:
        return "config"
    if "checkpoint" in rel.lower():
        return "checkpoint"
    if Path(rel).suffix in {".safetensors", ".npz", ".bin"}:
        return "adapter"
    return "artifact"


def _require_candidate_adapter_tree_hash(record: dict[str, Any], tree_sha256: str, label: str) -> None:
    allowed = _adapter_tree_hashes_from_candidate_record(record)
    if tree_sha256 not in allowed:
        raise Tau3BenchmarkRunError(f"{label} adapter tree sha256 does not match agent adapter path")


def _candidate_identity_sha256_from_lock(lock: dict[str, Any]) -> str:
    value = lock.get("candidate_identity_sha256")
    if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
        return value
    candidate = lock.get("candidate")
    if isinstance(candidate, dict):
        value = candidate.get("candidate_identity_sha256")
        if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
            return value
    raise Tau3BenchmarkRunError("candidate lock is missing candidate_identity_sha256")


def _adapter_tree_hashes_from_candidate_record(record: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in ("adapter_tree_sha256", "tree_sha256"):
        value = record.get(key)
        if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
            values.add(value)
    for key in ("candidate", "identity", "model_identity", "adapter_identity", "adapter"):
        value = record.get(key)
        if isinstance(value, dict):
            values.update(_adapter_tree_hashes_from_candidate_record(value))
    return values


def _protocol_model_for_arm(protocol: dict[str, Any], arm_id: str) -> dict[str, Any]:
    freeze = _dict(protocol.get("model_freeze"))
    if arm_id == "base":
        return _dict(freeze.get("base_model"))
    index_by_arm = {"comparator_1": 0, "comparator_2": 1}
    comparators = freeze.get("comparators")
    if arm_id in index_by_arm and isinstance(comparators, list) and len(comparators) > index_by_arm[arm_id]:
        return _dict(comparators[index_by_arm[arm_id]])
    raise Tau3BenchmarkRunError(f"protocol is missing model identity for {arm_id}")


def _identity_strings_from_model_record(record: dict[str, Any]) -> set[str]:
    values = {record.get("name"), record.get("local_path"), record.get("local_identity_sha256")}
    if record.get("name") and record.get("revision"):
        values.add(f"{record['name']}@{record['revision']}")
    return {str(value) for value in values if isinstance(value, str) and value}


def _identity_strings_from_candidate_record(record: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in (
        "model",
        "model_id",
        "served_model_id",
        "candidate_model",
        "adapter",
        "adapter_id",
        "adapter_path",
        "local_path",
        "adapter_sha256",
        "adapter_identity_sha256",
        "endpoint_model_sha256",
        "local_identity_sha256",
    ):
        value = record.get(key)
        if isinstance(value, str) and value:
            values.add(value)
    for key in ("candidate", "identity", "model_identity", "adapter_identity"):
        value = record.get(key)
        if isinstance(value, dict):
            values.update(_identity_strings_from_candidate_record(value))
    return values


def _endpoint_model_matches(model: str, allowed: set[str], *, protocol_path: Path) -> bool:
    model_hash = hashlib.sha256(model.encode("utf-8")).hexdigest()
    normalized_model = _normalize_model_identifier(model, protocol_path=protocol_path)
    normalized_allowed = {_normalize_model_identifier(value, protocol_path=protocol_path) for value in allowed}
    return model_hash in allowed or normalized_model in normalized_allowed


def _normalize_model_identifier(value: str, *, protocol_path: Path) -> str:
    normalized = value.removeprefix("openai//")
    if _looks_like_local_path(normalized):
        path = Path(normalized)
        if path.is_absolute():
            return str(path.resolve(strict=False))
        workspace_relative = Path.cwd() / path
        if workspace_relative.exists():
            return str(workspace_relative.resolve(strict=False))
        return str((protocol_path.parent / path).resolve(strict=False))
    return normalized


def _looks_like_local_path(value: str) -> bool:
    return value.startswith(("/", "./", "../")) or "/" in value or "\\" in value


def _protocol_split_hashes(protocol: dict[str, Any]) -> dict[str, str]:
    hashes = _dict(_dict(protocol.get("tau_revision")).get("split_hashes"))
    splits = _dict(_dict(protocol.get("split_manifest")).get("splits"))
    for name, record in splits.items():
        if isinstance(record, dict) and isinstance(record.get("sha256"), str):
            hashes.setdefault(str(name), record["sha256"])
    return {str(key): str(value) for key, value in hashes.items() if isinstance(value, str)}


def _protocol_candidate_lock_sha256(protocol: dict[str, Any]) -> str | None:
    candidates = (
        _dict(protocol.get("candidate_selection_contract")),
        _dict(protocol.get("sealed_manifest")),
        _dict(protocol.get("protocol_manifest")),
    )
    for section in candidates:
        for key in ("candidate_lock_sha256", "candidate_lock_manifest_sha256", "adapter_candidate_lock_sha256"):
            value = section.get(key)
            if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
                return value
    return None


def _public_model_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "name_sha256": hashlib.sha256(str(record.get("name") or "").encode("utf-8")).hexdigest(),
        "revision_sha256": hashlib.sha256(str(record.get("revision") or "").encode("utf-8")).hexdigest(),
        "local_identity_sha256": record.get("local_identity_sha256"),
    }


def _public_identity_hash(value: Any) -> str:
    return _canonical_sha256(value)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _candidate_lock_record(config: Tau3BenchmarkConfig, staged_ref: dict[str, Any] | None) -> dict[str, Any]:
    if config.candidate_lock is None:
        raise Tau3BenchmarkRunError("sealed mode requires --candidate-lock")
    path = config.candidate_lock
    if not path.is_file():
        raise Tau3BenchmarkRunError(f"candidate lock does not exist: {path}")
    digest = _sha256(path)
    if config.candidate_lock_sha256 is not None and config.candidate_lock_sha256 != digest:
        raise Tau3BenchmarkRunError("candidate lock sha256 mismatch")
    if staged_ref is None:
        raise Tau3BenchmarkRunError("sealed mode requires a staged candidate lock")
    return dict(staged_ref)


def _sealed_authorization_binding(config: Tau3BenchmarkConfig, staged_ref: dict[str, Any] | None, *, out: Path, expected_tau_revision: str) -> dict[str, Any]:
    if config.sealed_authorization is None:
        raise Tau3BenchmarkRunError("sealed mode requires --sealed-authorization")
    if config.candidate_lock is None or config.sealed_task_count_manifest is None:
        raise Tau3BenchmarkRunError("sealed authorization requires candidate lock and sealed task-count manifest")
    if staged_ref is None:
        raise Tau3BenchmarkRunError("sealed mode requires a staged sealed authorization")
    staged_path_value = staged_ref.get("path")
    if not isinstance(staged_path_value, str):
        raise Tau3BenchmarkRunError("staged sealed authorization path is missing")
    staged_authorization_path = _resolve_output_relative_path(staged_path_value, out)
    if not staged_authorization_path.is_file():
        raise Tau3BenchmarkRunError("staged sealed authorization is missing")
    if staged_ref.get("sha256") != _sha256(staged_authorization_path) or staged_ref.get("size") != staged_authorization_path.stat().st_size:
        raise Tau3BenchmarkRunError("staged sealed authorization drifted before replay")
    if config.sealed_authorization_sha256 is not None and config.sealed_authorization_sha256 != staged_ref.get("sha256"):
        raise Tau3BenchmarkRunError("sealed authorization sha256 mismatch")
    try:
        record = validate_tau3_sealed_authorization(
            authorization_path=staged_authorization_path,
            candidate_lock_path=config.candidate_lock,
            protocol_path=config.protocol_path,
            sealed_source_manifest_path=config.sealed_task_count_manifest,
            arm_id=config.arm_id,
            seeds=config.seeds,
            expected_tau_revision=expected_tau_revision,
            expected_authorization_sha256=str(staged_ref["sha256"]),
        )
    except Tau3SealedAuthorizationError as exc:
        raise Tau3BenchmarkRunError(str(exc)) from exc
    result = dict(staged_ref)
    for key in ("authorized", "candidate_lock_sha256", "protocol_sha256", "sealed_source_sha256", "task_count", "arms", "seeds"):
        result[key] = record[key]
    return result


def _candidate_identity_record(config: Tau3BenchmarkConfig, staged_ref: dict[str, Any] | None) -> dict[str, Any]:
    if config.candidate_identity is None:
        raise Tau3BenchmarkRunError("development adapter mode requires --candidate-identity")
    path = config.candidate_identity
    if not path.is_file():
        raise Tau3BenchmarkRunError(f"candidate identity does not exist: {path}")
    payload = _read_json(path)
    if payload.get("schema_version") != "hfr.tau3_candidate_identity.v1":
        raise Tau3BenchmarkRunError("candidate identity must use hfr.tau3_candidate_identity.v1")
    schema = check_schema_contract(payload, name_or_id="tau3_candidate_identity")
    if schema.get("passed") is not True:
        errors = "; ".join(str(error) for error in schema.get("errors", []))
        raise Tau3BenchmarkRunError("candidate identity violates registered schema: " + errors)
    if not _identity_strings_from_candidate_record(payload):
        raise Tau3BenchmarkRunError("candidate identity does not declare a usable model or adapter identity")
    digest = _sha256(path)
    if config.candidate_identity_sha256 is not None and config.candidate_identity_sha256 != digest:
        raise Tau3BenchmarkRunError("candidate identity sha256 mismatch")
    if staged_ref is None:
        raise Tau3BenchmarkRunError("development adapter mode requires a staged candidate identity")
    record = dict(staged_ref)
    candidate_id = _schema_validated_candidate_id(payload)
    if candidate_id is not None:
        record["candidate_id"] = candidate_id
    return record


def _sealed_task_count_manifest_record(path: Path | None, *, protocol: dict[str, Any], expected_revision: str) -> dict[str, Any]:
    if path is None:
        raise Tau3BenchmarkRunError("sealed mode requires --sealed-task-count-manifest")
    if path_has_symlink_component(path, include_leaf=True):
        raise Tau3BenchmarkRunError(f"sealed task-count manifest must not contain symlink components: {path}")
    payload = _read_json(path)
    schema = check_schema_contract(payload, name_or_id="tau3_sealed_source_manifest")
    if schema.get("passed") is not True:
        errors = "; ".join(str(error) for error in schema.get("errors", []))
        raise Tau3BenchmarkRunError("sealed task-count manifest schema failed: " + errors)
    if payload.get("schema_version") != "hfr.tau3_sealed_source_manifest.v1":
        raise Tau3BenchmarkRunError("sealed task-count manifest must use hfr.tau3_sealed_source_manifest.v1")
    if payload.get("source_revision") != expected_revision:
        raise Tau3BenchmarkRunError("sealed task-count manifest revision mismatch")
    if payload.get("hashes_only") is not True:
        raise Tau3BenchmarkRunError("sealed task-count manifest must be hashes-only")
    task_count = payload.get("task_count")
    if not isinstance(task_count, int) or isinstance(task_count, bool) or task_count < 1:
        raise Tau3BenchmarkRunError("sealed task-count manifest task_count must be a positive integer")
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != task_count:
        raise Tau3BenchmarkRunError("sealed task-count manifest task_count mismatch")
    allowed_entry_keys = {"task_id_sha256", "prompt_sha256", "task_sha256"}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != allowed_entry_keys:
            raise Tau3BenchmarkRunError(f"sealed task-count manifest entry {index} is not hash-only")
    digest = _sha256(path)
    split_hashes = _protocol_split_hashes(protocol)
    sealed_manifest = _dict(protocol.get("sealed_manifest"))
    if digest != split_hashes.get("sealed") or digest != sealed_manifest.get("manifest_sha256"):
        raise Tau3BenchmarkRunError("sealed task-count manifest sha256 does not match protocol sealed split")
    return {"task_count": task_count, "sha256": digest}


def _sealed_task_count_binding(staged_ref: dict[str, Any] | None, count_record: dict[str, Any] | None) -> dict[str, Any] | None:
    if count_record is None:
        return None
    if staged_ref is None:
        raise Tau3BenchmarkRunError("sealed mode requires a staged sealed task-count manifest")
    record = dict(staged_ref)
    record["task_count"] = count_record["task_count"]
    record["hashes_only"] = True
    return record


def _schema_validated_candidate_id(payload: dict[str, Any]) -> str | None:
    candidate_id = payload.get("candidate_id")
    return candidate_id if isinstance(candidate_id, str) and candidate_id else None


def _task_selection_record(
    config: Tau3BenchmarkConfig,
    tasks_by_domain: dict[str, list[str]] | None,
    *,
    sealed_task_count: int | None = None,
) -> dict[str, Any]:
    if config.mode == "sealed":
        if sealed_task_count is None:
            raise Tau3BenchmarkRunError("sealed mode requires a validated task count")
        return {
            "official_split": "test",
            "task_ids_in_command": False,
            "task_payload_accessed": False,
            "domains": list(config.domains),
            "sealed_task_count": sealed_task_count,
            "task_count_by_domain": None,
        }
    assert tasks_by_domain is not None
    return {
        "official_split": "train",
        "logical_split": "development",
        "task_ids_in_command": True,
        "task_payload_accessed": False,
        "domains": list(config.domains),
        "sealed_task_count": None,
        "task_count_by_domain": {domain: len(tasks_by_domain[domain]) for domain in config.domains},
        "task_id_sha256_by_domain": {
            domain: [hashlib.sha256(task_id.encode("utf-8")).hexdigest() for task_id in tasks_by_domain[domain]]
            for domain in config.domains
        },
    }


def _result_summary(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    prompt_tokens = list(_iter_prompt_tokens(payload))
    sims = payload.get("simulations")
    if not isinstance(sims, list):
        return {
            "simulation_count": 0,
            "reward_sum": None,
            "success_count": None,
            "prompt_token_ceiling": PROMPT_TOKEN_CEILING,
            "prompt_token_observation_count": len(prompt_tokens),
            "prompt_token_ceiling_checked": bool(prompt_tokens),
            "prompt_token_ceiling_exceeded": _prompt_token_ceiling_exceeded(prompt_tokens),
        }
    rewards = []
    for sim in sims:
        reward_info = sim.get("reward_info") if isinstance(sim, dict) else None
        reward = reward_info.get("reward") if isinstance(reward_info, dict) else None
        if isinstance(reward, (int, float)) and not isinstance(reward, bool):
            rewards.append(float(reward))
    return {
        "simulation_count": len(sims),
        "reward_sum": round(sum(rewards), 6) if rewards else None,
        "success_count": sum(1 for reward in rewards if reward == 1.0) if rewards else None,
        "prompt_token_ceiling": PROMPT_TOKEN_CEILING,
        "prompt_token_observation_count": len(prompt_tokens),
        "prompt_token_ceiling_checked": bool(prompt_tokens),
        "prompt_token_ceiling_exceeded": _prompt_token_ceiling_exceeded(prompt_tokens),
    }


def _valid_result_summary(summary: dict[str, Any]) -> bool:
    return (
        isinstance(summary.get("simulation_count"), int)
        and summary["simulation_count"] > 0
        and isinstance(summary.get("reward_sum"), (int, float))
        and not isinstance(summary.get("reward_sum"), bool)
        and isinstance(summary.get("success_count"), int)
        and summary.get("prompt_token_ceiling") == PROMPT_TOKEN_CEILING
        and isinstance(summary.get("prompt_token_observation_count"), int)
        and summary["prompt_token_observation_count"] > 0
        and summary.get("prompt_token_ceiling_checked") is True
        and summary.get("prompt_token_ceiling_exceeded") is False
    )


def _prompt_token_ceiling_exceeded(values: list[int]) -> bool:
    for prompt_tokens in values:
        if prompt_tokens > PROMPT_TOKEN_CEILING:
            return True
    return False


def _iter_prompt_tokens(value: Any) -> Iterator[int]:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key == "prompt_tokens" and isinstance(nested, int) and not isinstance(nested, bool):
                yield nested
            else:
                yield from _iter_prompt_tokens(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_prompt_tokens(item)


def _validate_config(config: Tau3BenchmarkConfig) -> None:
    if config.mode not in {"development", "sealed"}:
        raise Tau3BenchmarkRunError("mode must be development or sealed")
    if config.arm_id not in REQUIRED_ARMS:
        raise Tau3BenchmarkRunError("arm_id must be adapter, base, comparator_1, or comparator_2")
    if config.seeds != DEFAULT_SEEDS:
        raise Tau3BenchmarkRunError("seeds must be exactly 101,202,303,404 in frozen order")
    if config.domains != DOMAINS:
        raise Tau3BenchmarkRunError("domains must be exactly airline,retail,telecom in frozen order")
    if config.max_steps != 30:
        raise Tau3BenchmarkRunError("max_steps must be exactly 30")
    if config.max_errors != 10:
        raise Tau3BenchmarkRunError("max_errors must be exactly 10")
    if not 1 <= config.timeout_seconds <= 7200:
        raise Tau3BenchmarkRunError("timeout_seconds must be between 1 and 7200")
    if config.mode == "development" and config.candidate_lock is not None:
        raise Tau3BenchmarkRunError("development mode must not receive a candidate lock")
    if config.mode == "development" and config.sealed_task_count_manifest is not None:
        raise Tau3BenchmarkRunError("development mode must not receive a sealed task-count manifest")
    if config.mode == "development" and config.sealed_authorization is not None:
        raise Tau3BenchmarkRunError("development mode must not receive a sealed authorization")
    if config.mode == "sealed" and config.source_split is not None:
        raise Tau3BenchmarkRunError("sealed mode must not receive a source split")
    if config.mode == "sealed" and config.sealed_task_count_manifest is None:
        raise Tau3BenchmarkRunError("sealed mode requires --sealed-task-count-manifest")
    if config.mode == "sealed" and config.sealed_authorization is None:
        raise Tau3BenchmarkRunError("sealed mode requires --sealed-authorization")
    if config.mode == "sealed" and config.candidate_identity is not None:
        raise Tau3BenchmarkRunError("sealed mode must not receive a candidate identity")
    if config.mode == "development" and config.arm_id != "adapter" and config.candidate_identity is not None:
        raise Tau3BenchmarkRunError("candidate identity is only allowed for the development adapter arm")


def _require_loopback(api_base: str, label: str) -> None:
    parsed = urlparse(api_base)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise Tau3BenchmarkRunError(f"{label} endpoint must be loopback: {api_base}")


def _validate_endpoint(endpoint: Tau3BenchmarkEndpoint, label: str) -> None:
    if endpoint.temperature != 0.0:
        raise Tau3BenchmarkRunError(f"{label} temperature must be exactly 0.0")
    if endpoint.top_p != 1.0:
        raise Tau3BenchmarkRunError(f"{label} top_p must be exactly 1.0")
    if endpoint.max_tokens != 1024:
        raise Tau3BenchmarkRunError(f"{label} max_tokens must be exactly 1024")
    if endpoint.context_window != CONTEXT_WINDOW:
        raise Tau3BenchmarkRunError(f"{label} context_window must be exactly {CONTEXT_WINDOW}")


def _reviewer_environment(reviewer: Tau3BenchmarkEndpoint) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().endswith("_API_KEY") and "TOKEN" not in key.upper()
    }
    env["OPENAI_API_BASE"] = reviewer.api_base
    env["OPENAI_API_KEY"] = "local"
    return env


def _require_revision(repo: Path, expected_revision: str) -> None:
    revision = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()
    if revision != expected_revision:
        raise Tau3BenchmarkRunError(f"Tau repository revision mismatch: {revision!r}")
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    if dirty:
        raise Tau3BenchmarkRunError("Tau repository has tracked modifications")


def _config_record(config: Tau3BenchmarkConfig) -> dict[str, Any]:
    return {
        "seeds": list(config.seeds),
        "timeout_seconds": config.timeout_seconds,
        "max_steps": config.max_steps,
        "max_errors": config.max_errors,
        "domains": list(config.domains),
        "agent": "llm_agent",
        "user": "user_simulator",
        "num_trials": 1,
        "max_concurrency": 1,
        "max_retries": 0,
        "hallucination_retries": 0,
        "auto_review": True,
        "review_mode": "full",
        "communication_protocol_enforced": True,
        "context_window": CONTEXT_WINDOW,
        "test_time_search": False,
        "resume": False,
    }


def _command_timeout_seconds(
    *,
    protocol: dict[str, Any],
    config: Tau3BenchmarkConfig,
    domain: str,
    tasks_by_domain: dict[str, list[str]] | None,
    sealed_task_count: int | None = None,
) -> int:
    """Bound the whole Tau command without confusing it with the per-task timeout."""

    task_count: Any
    if tasks_by_domain is not None:
        task_count = len(tasks_by_domain.get(domain, ()))
    else:
        task_count = sealed_task_count
    if not isinstance(task_count, int) or isinstance(task_count, bool) or task_count < 1:
        raise Tau3BenchmarkRunError(f"cannot derive a positive command task count for {domain}")
    return config.timeout_seconds * task_count + config.command_timeout_padding_seconds


def _endpoint_record(endpoint: Tau3BenchmarkEndpoint) -> dict[str, Any]:
    parsed = urlparse(endpoint.api_base)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return {
        "model_sha256": hashlib.sha256(endpoint.model.encode("utf-8")).hexdigest(),
        "endpoint_hash": hashlib.sha256(f"{parsed.scheme}://{parsed.hostname}:{port}{parsed.path}".encode("utf-8")).hexdigest(),
        "loopback": parsed.hostname in {"127.0.0.1", "localhost"},
        "temperature": endpoint.temperature,
        "top_p": endpoint.top_p,
        "max_tokens": endpoint.max_tokens,
        "context_window": endpoint.context_window,
    }


def _validate_resumable_manifest(manifest: dict[str, Any], *, out: Path, expected_binding: dict[str, Any]) -> None:
    schema = check_schema_contract(manifest, name_or_id="tau3_benchmark_run")
    errors = list(schema.get("errors") or [])
    if manifest.get("phase") != "final":
        errors.append("existing manifest is not final")
    _collect_binding_errors(manifest, expected_binding, errors, label="existing manifest")
    prelaunch_ref = manifest.get("prelaunch_receipt")
    if isinstance(prelaunch_ref, dict):
        prelaunch_path_value = prelaunch_ref.get("path")
        try:
            prelaunch_path = _resolve_output_relative_path(prelaunch_path_value, out) if isinstance(prelaunch_path_value, str) else None
        except Tau3BenchmarkRunError as exc:
            errors.append(str(exc))
            prelaunch_path = None
        if prelaunch_path is None or not prelaunch_path.is_file():
            errors.append("existing prelaunch receipt is missing")
        elif prelaunch_ref.get("sha256") != _sha256(prelaunch_path):
            errors.append("existing prelaunch receipt hash mismatch")
    else:
        errors.append("existing manifest prelaunch_receipt is missing or invalid")
    receipts = manifest.get("run_receipts")
    if not isinstance(receipts, list):
        errors.append("existing manifest run_receipts is not a list")
        receipts = []
    success_count = 0
    failure_count = 0
    for ref in receipts:
        if not isinstance(ref, dict):
            errors.append("existing run receipt reference is not an object")
            continue
        rel = ref.get("path")
        if not isinstance(rel, str) or "/" in rel:
            errors.append("existing run receipt path is invalid")
            continue
        receipt_path = out / rel
        if not receipt_path.is_file():
            errors.append(f"existing run receipt is missing: {rel}")
            continue
        receipt_sha256 = _sha256(receipt_path)
        if ref.get("receipt_sha256") != receipt_sha256:
            errors.append(f"existing run receipt sha mismatch: {rel}")
        receipt = _read_json(receipt_path)
        _collect_completed_receipt_errors(receipt, errors, label=f"existing run receipt {rel}")
        if receipt.get("terminal_status") != ref.get("terminal_status"):
            errors.append(f"existing run receipt status mismatch: {rel}")
        if receipt.get("result_path") != ref.get("result_path"):
            errors.append(f"existing run receipt result path mismatch: {rel}")
        if receipt.get("result_sha256") != ref.get("result_sha256"):
            errors.append(f"existing run receipt result hash mismatch: {rel}")
        result_sha256 = receipt.get("result_sha256")
        if result_sha256 is not None:
            result_path_value = receipt.get("result_path")
            result_path = _resolve_output_relative_path(result_path_value, out) if isinstance(result_path_value, str) else None
            if result_path is None or not result_path.is_file() or _sha256(result_path) != result_sha256:
                errors.append(f"existing generated result drifted: {rel}")
        if ref.get("terminal_status") == "completed":
            success_count += 1
        else:
            failure_count += 1
    if manifest.get("success_count") != success_count or manifest.get("failure_count") != failure_count:
        errors.append("existing manifest success/failure counts do not replay run receipts")
    if errors:
        raise Tau3BenchmarkRunError("existing final manifest is stale or invalid: " + "; ".join(errors))


def _validate_task_receipt(receipt: dict[str, Any], *, receipt_path: Path, command: list[str]) -> None:
    errors = []
    if receipt.get("phase") != "domain_seed":
        errors.append("phase mismatch")
    if receipt.get("command") != command:
        errors.append("command mismatch")
    _collect_completed_receipt_errors(receipt, errors, label="completed receipt")
    result_sha256 = receipt.get("result_sha256")
    if result_sha256 is not None:
        result_path_value = receipt.get("result_path")
        result_path = _resolve_output_relative_path(result_path_value, receipt_path.parent) if isinstance(result_path_value, str) else None
        if result_path is None or not result_path.is_file() or _sha256(result_path) != result_sha256:
            errors.append("generated result drifted")
    if errors:
        raise Tau3BenchmarkRunError(f"existing run receipt is stale or invalid: {receipt_path}: {'; '.join(errors)}")


def _collect_completed_receipt_errors(receipt: dict[str, Any], errors: list[str], *, label: str) -> None:
    if receipt.get("terminal_status") != "completed":
        return
    if receipt.get("exit_code") != 0:
        errors.append(f"{label} exit_code must be 0")
    if receipt.get("timed_out") is not False:
        errors.append(f"{label} timed_out must be false")
    digest = receipt.get("result_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        errors.append(f"{label} result_sha256 is missing or invalid")
    summary = receipt.get("result_summary")
    if not isinstance(summary, dict) or not _valid_result_summary(summary):
        errors.append(f"{label} result_summary is missing or invalid")


def _require_matching_binding(payload: dict[str, Any], expected_binding: dict[str, Any], *, label: str) -> None:
    errors: list[str] = []
    _collect_binding_errors(payload, expected_binding, errors, label=label)
    if errors:
        raise Tau3BenchmarkRunError("; ".join(errors))


def _collect_binding_errors(payload: dict[str, Any], expected_binding: dict[str, Any], errors: list[str], *, label: str) -> None:
    for key, expected in expected_binding.items():
        if payload.get(key) != expected:
            errors.append(f"{label} {key} does not match this benchmark arm")


def _write_json_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise Tau3BenchmarkRunError(f"refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o600)


def _copy_file_new(source: Path, destination: Path) -> None:
    if destination.exists():
        raise Tau3BenchmarkRunError(f"refusing to overwrite existing file: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    destination.chmod(0o600)


def _stage_input_file(source: Path, output_root: Path, rel: str, label: str) -> dict[str, Any]:
    if not source.is_file():
        raise Tau3BenchmarkRunError(f"{label} does not exist: {source}")
    if path_has_symlink_component(source, include_leaf=True):
        raise Tau3BenchmarkRunError(f"{label} must not contain symlink components: {source}")
    destination = _resolve_output_relative_path(rel, output_root)
    if path_has_symlink_component(destination, include_leaf=True):
        raise Tau3BenchmarkRunError(f"staged {label} destination must not contain symlink components: {rel}")
    output_root_resolved = output_root.resolve(strict=True)
    destination_resolved = destination.resolve(strict=False)
    if not destination_resolved.is_relative_to(output_root_resolved):
        raise Tau3BenchmarkRunError(f"staged {label} destination escapes output directory: {rel}")
    source_sha256 = _sha256(source)
    if destination.exists():
        if not destination.is_file() or _sha256(destination) != source_sha256:
            raise Tau3BenchmarkRunError(f"staged {label} drifted from supplied original: {rel}")
    else:
        _copy_file_new(source, destination)
    return {"path": rel, "size": source.stat().st_size, "sha256": source_sha256}


def _output_file_record(path: Path, output_root: Path) -> dict[str, Any]:
    return {"path": _relative_output_path(path, output_root), "size": path.stat().st_size, "sha256": _sha256(path)}


def _relative_output_path(path: Path, output_root: Path) -> str:
    try:
        rel = path.relative_to(output_root).as_posix()
    except ValueError as exc:
        raise Tau3BenchmarkRunError(f"output artifact is outside output directory: {path}") from exc
    _resolve_output_relative_path(rel, output_root)
    return rel


def _copied_result_relative_path(domain: str, seed: int) -> str:
    if domain not in DOMAINS or seed not in DEFAULT_SEEDS:
        raise Tau3BenchmarkRunError(f"unsafe Tau result coordinates: {domain} seed {seed}")
    return f"results/{domain}/seed-{seed}/results.json"


def _resolve_output_relative_path(value: str, output_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute() or value in {"", "."} or ".." in path.parts:
        raise Tau3BenchmarkRunError(f"output-relative path is unsafe: {value}")
    return output_root / path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Tau3BenchmarkRunError(f"invalid JSON file: {path}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise Tau3BenchmarkRunError(f"JSON file must contain an object: {path}")
    return payload


def _file_record(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return {"path": str(path), "size": path.stat().st_size, "sha256": _sha256(path)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _redact_argv(argv: list[str]) -> list[str]:
    return [_redact(item) for item in argv]


def _redact(text: str) -> str:
    return text.replace('"api_key":"local"', '"api_key":"[REDACTED]"').replace('"api_key": "local"', '"api_key": "[REDACTED]"')


def _parse_int_csv(value: str, label: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise Tau3BenchmarkRunError(f"{label} must be comma-separated integers") from exc
    return parsed


def _parse_str_csv(value: str, label: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if not parsed:
        raise Tau3BenchmarkRunError(f"{label} must not be empty")
    return parsed


def _now_utc() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
