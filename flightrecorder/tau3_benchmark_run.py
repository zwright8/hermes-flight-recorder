"""Fail-closed official-Tau benchmark arm runner for the Tau-3 study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .schema_registry import check_schema_contract

TAU3_BENCHMARK_RUN_SCHEMA_VERSION = "hfr.tau3_benchmark_run.v1"
DOMAINS = ("airline", "retail", "telecom")
DEFAULT_SEEDS = (101, 202, 303, 404)
REQUIRED_ARMS = ("adapter", "base", "comparator_1", "comparator_2")
CONTEXT_WINDOW = 16384


class Tau3BenchmarkRunError(ValueError):
    """Raised when a benchmark arm launch would violate study governance."""


@dataclass(frozen=True)
class Tau3BenchmarkEndpoint:
    model: str
    api_base: str
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1024
    context_window: int = CONTEXT_WINDOW


@dataclass(frozen=True)
class Tau3BenchmarkConfig:
    mode: str
    arm_id: str
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    timeout_seconds: int = 600
    max_steps: int = 30
    max_errors: int = 10
    domains: tuple[str, ...] = DOMAINS
    source_split: Path | None = None
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
    _validate_endpoint(agent, "agent")
    _validate_endpoint(user, "user")
    _validate_endpoint(reviewer, "reviewer")
    _require_revision(repo, expected_tau_revision)
    tasks_by_domain = _development_tasks_by_domain(config.source_split, expected_tau_revision) if config.mode == "development" else None
    candidate_lock = _candidate_lock_record(config) if config.mode == "sealed" else None

    expected_binding = {
        "tau_revision": expected_tau_revision,
        "mode": config.mode,
        "arm_id": config.arm_id,
        "agent": _endpoint_record(agent),
        "user_simulator": _endpoint_record(user),
        "reviewer": _endpoint_record(reviewer),
        "config": _config_record(config),
        "source": _file_record(config.source_split) if config.mode == "development" and config.source_split is not None else None,
        "candidate_lock": candidate_lock,
        "task_selection": _task_selection_record(config, tasks_by_domain),
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
        "out_dir": str(out),
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
            start = time.monotonic()
            try:
                proc = subprocess.run(
                    argv,
                    cwd=repo,
                    env=_reviewer_environment(reviewer),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=config.timeout_seconds + config.command_timeout_padding_seconds,
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
            valid_result = isinstance(result_summary, dict) and _valid_result_summary(result_summary)
            status = "completed" if exit_code == 0 and has_result and valid_result else ("timeout" if timed_out else "failed")
            receipt = {
                "schema_version": TAU3_BENCHMARK_RUN_SCHEMA_VERSION,
                "phase": "domain_seed",
                "created_at": created_at or _now_utc(),
                "mode": config.mode,
                "arm_id": config.arm_id,
                "domain": domain,
                "seed": seed,
                "command": _redact_argv(argv),
                "result_path": str(result_path),
                "result_sha256": _sha256(result_path) if has_result else None,
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
            receipts.append(receipt)

    success_count = sum(1 for receipt in receipts if receipt.get("terminal_status") == "completed")
    failure_count = len(receipts) - success_count
    manifest = {
        "schema_version": TAU3_BENCHMARK_RUN_SCHEMA_VERSION,
        "phase": "final",
        "created_at": created_at or _now_utc(),
        **expected_binding,
        "prelaunch_receipt": _file_record(prelaunch_path),
        "run_count": len(receipts),
        "success_count": success_count,
        "failure_count": failure_count,
        "run_receipts": [
            {
                "path": f"run-{receipt['domain']}-seed{receipt['seed']}.json",
                "domain": receipt["domain"],
                "seed": receipt["seed"],
                "terminal_status": receipt["terminal_status"],
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
    parser.add_argument("--tau-repo", type=Path, required=True)
    parser.add_argument("--tau-venv-bin", type=Path, required=True)
    parser.add_argument("--expected-tau-revision", required=True)
    parser.add_argument("--source-split", type=Path)
    parser.add_argument("--candidate-lock", type=Path)
    parser.add_argument("--candidate-lock-sha256")
    parser.add_argument("--agent-model", required=True)
    parser.add_argument("--agent-api-base", required=True)
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
            agent=Tau3BenchmarkEndpoint(model=args.agent_model, api_base=args.agent_api_base),
            user=Tau3BenchmarkEndpoint(model=args.user_model, api_base=args.user_api_base),
            reviewer=Tau3BenchmarkEndpoint(model=args.reviewer_model, api_base=args.reviewer_api_base),
            config=Tau3BenchmarkConfig(
                mode=args.mode,
                arm_id=args.arm_id,
                source_split=args.source_split,
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
    return {
        "api_base": endpoint.api_base,
        "api_key": "local",
        "temperature": endpoint.temperature,
        "top_p": endpoint.top_p,
        "max_tokens": endpoint.max_tokens,
        "num_retries": 0,
    }


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
    tasks_by_domain = {domain: [] for domain in DOMAINS}
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
        if row.get("raw_id_sha256") != hashlib.sha256(raw_id.encode("utf-8")).hexdigest():
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


def _candidate_lock_record(config: Tau3BenchmarkConfig) -> dict[str, Any]:
    if config.candidate_lock is None:
        raise Tau3BenchmarkRunError("sealed mode requires --candidate-lock")
    path = config.candidate_lock
    if not path.is_file():
        raise Tau3BenchmarkRunError(f"candidate lock does not exist: {path}")
    digest = _sha256(path)
    if config.candidate_lock_sha256 is not None and config.candidate_lock_sha256 != digest:
        raise Tau3BenchmarkRunError("candidate lock sha256 mismatch")
    return {"path": str(path), "size": path.stat().st_size, "sha256": digest}


def _task_selection_record(config: Tau3BenchmarkConfig, tasks_by_domain: dict[str, list[str]] | None) -> dict[str, Any]:
    if config.mode == "sealed":
        return {
            "official_split": "test",
            "task_ids_in_command": False,
            "task_payload_accessed": False,
            "domains": list(config.domains),
            "task_count_by_domain": None,
        }
    assert tasks_by_domain is not None
    return {
        "official_split": "train",
        "logical_split": "development",
        "task_ids_in_command": True,
        "task_payload_accessed": False,
        "domains": list(config.domains),
        "task_count_by_domain": {domain: len(tasks_by_domain[domain]) for domain in config.domains},
        "task_id_sha256_by_domain": {
            domain: [hashlib.sha256(task_id.encode("utf-8")).hexdigest() for task_id in tasks_by_domain[domain]]
            for domain in config.domains
        },
    }


def _result_summary(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    sims = payload.get("simulations")
    if not isinstance(sims, list):
        return {"simulation_count": 0, "reward_sum": None, "success_count": None}
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
    }


def _valid_result_summary(summary: dict[str, Any]) -> bool:
    return (
        isinstance(summary.get("simulation_count"), int)
        and summary["simulation_count"] > 0
        and isinstance(summary.get("reward_sum"), (int, float))
        and not isinstance(summary.get("reward_sum"), bool)
        and isinstance(summary.get("success_count"), int)
    )


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
    if config.mode == "sealed" and config.source_split is not None:
        raise Tau3BenchmarkRunError("sealed mode must not receive a source split")


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
        receipt = _read_json(receipt_path)
        _collect_completed_receipt_errors(receipt, errors, label=f"existing run receipt {rel}")
        if receipt.get("terminal_status") != ref.get("terminal_status"):
            errors.append(f"existing run receipt status mismatch: {rel}")
        if receipt.get("result_sha256") != ref.get("result_sha256"):
            errors.append(f"existing run receipt result hash mismatch: {rel}")
        result_sha256 = receipt.get("result_sha256")
        if result_sha256 is not None:
            result_path_value = receipt.get("result_path")
            result_path = Path(result_path_value) if isinstance(result_path_value, str) else None
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
        result_path = Path(result_path_value) if isinstance(result_path_value, str) else None
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
