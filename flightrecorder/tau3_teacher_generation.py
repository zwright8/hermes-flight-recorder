"""Governed official-Tau teacher trajectory generation runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .schema_registry import check_schema_contract

TAU3_TEACHER_GENERATION_RUN_SCHEMA_VERSION = "hfr.tau3_teacher_generation_run.v1"
DOMAINS = {"airline", "retail", "telecom"}
TRAIN_SPLITS = {"train", "development"}


class Tau3TeacherGenerationError(ValueError):
    """Raised when official-Tau teacher generation would violate governance."""


@dataclass(frozen=True)
class Tau3Endpoint:
    model: str
    api_base: str
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1024


@dataclass(frozen=True)
class Tau3TeacherGenerationConfig:
    max_steps: int = 30
    max_errors: int = 10
    timeout_seconds: int = 600
    seed: int = 101
    max_tasks: int = 1
    agent: str = "auto"
    allow_teacher_protocol_normalization: bool = False


def run_tau3_teacher_generation(
    *,
    source_jsonl: str | Path,
    out_dir: str | Path,
    tau_repo: str | Path,
    tau_venv_bin: str | Path,
    expected_tau_revision: str,
    teacher: Tau3Endpoint,
    user: Tau3Endpoint,
    config: Tau3TeacherGenerationConfig | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Run official Tau generation over deterministic train/development tasks."""

    cfg = config or Tau3TeacherGenerationConfig()
    source = Path(source_jsonl)
    out = Path(out_dir)
    repo = Path(tau_repo).resolve(strict=True)
    tau2 = Path(tau_venv_bin)
    if tau2.is_dir():
        tau2 = tau2 / "tau2"
    if not tau2.is_absolute():
        tau2 = Path.cwd() / tau2
    if out.exists() and not out.is_dir():
        raise Tau3TeacherGenerationError(f"output exists and is not a directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    final_manifest_path = out / "manifest.json"
    _validate_config(cfg)
    _require_loopback(teacher.api_base, "teacher")
    _require_loopback(user.api_base, "user")
    _require_revision(repo, expected_tau_revision)
    tasks = _load_tasks(source, expected_tau_revision)[: cfg.max_tasks]
    if not tasks:
        raise Tau3TeacherGenerationError("no eligible train/development tasks")
    expected_binding = {
        "tau_revision": expected_tau_revision,
        "source": _file_record(source),
        "teacher": _endpoint_record(teacher),
        "user_simulator": _endpoint_record(user),
        "config": _config_record(cfg),
        "task_count": len(tasks),
    }
    if final_manifest_path.exists():
        manifest = _read_json(final_manifest_path)
        _validate_resumable_manifest(manifest, out=out, tasks=tasks, expected_binding=expected_binding)
        return manifest

    prelaunch = {
        "schema_version": TAU3_TEACHER_GENERATION_RUN_SCHEMA_VERSION,
        "phase": "prelaunch",
        "created_at": created_at or _now_utc(),
        **expected_binding,
        "out_dir": str(out),
        "sealed_rows": 0,
        "test_rows": 0,
        "loopback_only": True,
        "training_started": False,
        "sealed_payload_accessed": False,
    }
    prelaunch_path = out / "prelaunch_receipt.json"
    if prelaunch_path.exists():
        existing_prelaunch = _read_json(prelaunch_path)
        if any(
            existing_prelaunch.get(key) != prelaunch.get(key)
            for key in ("tau_revision", "source", "teacher", "user_simulator", "config", "task_count")
        ):
            raise Tau3TeacherGenerationError("existing prelaunch receipt does not match this generation run")
    else:
        _write_json_new(prelaunch_path, prelaunch)
    receipts = []
    for index, task in enumerate(tasks):
        task_key = _task_key(task["task_id"])
        receipt_path = out / f"task-{index:04d}-{task['domain']}-{task_key}.json"
        if receipt_path.exists():
            receipt = _read_json(receipt_path)
            if receipt.get("task") != task:
                raise Tau3TeacherGenerationError(f"existing task receipt identity mismatch: {receipt_path}")
            receipts.append(receipt)
            continue
        save_to = f"hfr-generation/{out.name}/{index:04d}-{task['domain']}-{task_key}"
        argv = _tau2_argv(
            tau2=tau2,
            task=task,
            save_to=save_to,
            teacher=teacher,
            user=user,
            cfg=cfg,
        )
        start = time.monotonic()
        try:
            proc = subprocess.run(
                argv,
                cwd=repo,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=cfg.timeout_seconds + 30,
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
        result_path = repo / "data" / "simulations" / save_to / "results.json"
        reward = _result_reward(result_path) if result_path.exists() else None
        status = "success" if exit_code == 0 and reward == 1.0 else ("timeout" if timed_out else "failed")
        receipt = {
            "schema_version": TAU3_TEACHER_GENERATION_RUN_SCHEMA_VERSION,
            "phase": "task",
            "created_at": created_at or _now_utc(),
            "task": task,
            "command": _redact_argv(argv),
            "result_path": str(result_path),
            "result_sha256": _sha256(result_path) if result_path.exists() else None,
            "reward": reward,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "duration_seconds": round(duration, 6),
            "terminal_status": status,
            "stdout_tail": _redact(stdout)[-1000:],
            "stderr_tail": _redact(stderr)[-1000:],
            "training_started": False,
            "sealed_payload_accessed": False,
        }
        _write_json_new(receipt_path, receipt)
        receipts.append(receipt)
    manifest = {
        "schema_version": TAU3_TEACHER_GENERATION_RUN_SCHEMA_VERSION,
        "phase": "final",
        "created_at": created_at or _now_utc(),
        **expected_binding,
        "prelaunch_receipt": _file_record(prelaunch_path),
        "success_count": sum(1 for receipt in receipts if receipt.get("terminal_status") == "success"),
        "failure_count": sum(1 for receipt in receipts if receipt.get("terminal_status") != "success"),
        "task_receipts": [
            {
                "path": f"task-{index:04d}-{receipt['task']['domain']}-{_task_key(receipt['task']['task_id'])}.json",
                "terminal_status": receipt["terminal_status"],
                "result_sha256": receipt.get("result_sha256"),
            }
            for index, receipt in enumerate(receipts)
        ],
        "sealed_rows": 0,
        "test_rows": 0,
        "loopback_only": True,
        "training_started": False,
        "sealed_payload_accessed": False,
    }
    check = check_schema_contract(manifest, name_or_id="tau3_teacher_generation_run")
    if check["passed"] is not True:
        raise Tau3TeacherGenerationError("manifest schema failed: " + json.dumps(check["errors"], sort_keys=True))
    _write_json_new(final_manifest_path, manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tau-repo", type=Path, required=True)
    parser.add_argument("--tau-venv-bin", type=Path, required=True)
    parser.add_argument("--expected-tau-revision", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--teacher-api-base", required=True)
    parser.add_argument("--user-model", required=True)
    parser.add_argument("--user-api-base", required=True)
    parser.add_argument("--max-tasks", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--agent", choices=("auto", "llm_agent", "llm_agent_gt"), default="auto")
    parser.add_argument("--allow-teacher-protocol-normalization", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        manifest = run_tau3_teacher_generation(
            source_jsonl=args.source_jsonl,
            out_dir=args.out,
            tau_repo=args.tau_repo,
            tau_venv_bin=args.tau_venv_bin,
            expected_tau_revision=args.expected_tau_revision,
            teacher=Tau3Endpoint(model=args.teacher_model, api_base=args.teacher_api_base),
            user=Tau3Endpoint(model=args.user_model, api_base=args.user_api_base),
            config=Tau3TeacherGenerationConfig(
                max_steps=args.max_steps,
                timeout_seconds=args.timeout_seconds,
                seed=args.seed,
                max_tasks=args.max_tasks,
                agent=args.agent,
                allow_teacher_protocol_normalization=args.allow_teacher_protocol_normalization,
            ),
        )
    except (OSError, Tau3TeacherGenerationError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps({"manifest": str(Path(args.out) / "manifest.json"), "success_count": manifest["success_count"]}, indent=2, sort_keys=True))
    return 0 if manifest["failure_count"] == 0 else 1


def _tau2_argv(*, tau2: Path, task: dict[str, str], save_to: str, teacher: Tau3Endpoint, user: Tau3Endpoint, cfg: Tau3TeacherGenerationConfig) -> list[str]:
    agent = cfg.agent
    if agent == "auto":
        agent = "llm_agent_gt" if task["has_reference_actions"] == "true" else "llm_agent"
    argv = [
        str(tau2),
        "run",
        "--domain",
        task["domain"],
        "--task-set-name",
        task["domain"],
        "--task-split-name",
        task["official_task_split"],
        "--task-ids",
        task["task_id"],
        "--num-trials",
        "1",
        "--agent",
        agent,
        "--agent-llm",
        teacher.model,
        "--agent-llm-args",
        json.dumps(_endpoint_args(teacher), separators=(",", ":")),
        "--user",
        "user_simulator",
        "--user-llm",
        user.model,
        "--user-llm-args",
        json.dumps(_endpoint_args(user), separators=(",", ":")),
        "--max-steps",
        str(cfg.max_steps),
        "--max-errors",
        str(cfg.max_errors),
        "--timeout",
        str(cfg.timeout_seconds),
        "--save-to",
        save_to,
        "--max-concurrency",
        "1",
        "--seed",
        str(cfg.seed),
        "--max-retries",
        "0",
        "--log-level",
        "ERROR",
    ]
    if not cfg.allow_teacher_protocol_normalization:
        argv.append("--enforce-communication-protocol")
    return argv


def _endpoint_args(endpoint: Tau3Endpoint) -> dict[str, Any]:
    return {
        "api_base": endpoint.api_base,
        "api_key": "local",
        "temperature": endpoint.temperature,
        "top_p": endpoint.top_p,
        "max_tokens": endpoint.max_tokens,
        "num_retries": 0,
    }


def _load_tasks(path: Path, expected_revision: str) -> list[dict[str, str]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        domain = str(row.get("domain") or "")
        split = str(row.get("split") or "")
        task = row.get("task") if isinstance(row.get("task"), dict) else {}
        if domain not in DOMAINS:
            raise Tau3TeacherGenerationError(f"{path}:{line_number}: unsupported domain")
        if split not in TRAIN_SPLITS:
            raise Tau3TeacherGenerationError(f"{path}:{line_number}: sealed/test split rejected")
        if row.get("source_revision") != expected_revision:
            raise Tau3TeacherGenerationError(f"{path}:{line_number}: source revision mismatch")
        task_id = str(task.get("id") or "")
        if not task_id:
            raise Tau3TeacherGenerationError(f"{path}:{line_number}: missing task id")
        evaluation = task.get("evaluation_criteria") if isinstance(task.get("evaluation_criteria"), dict) else {}
        actions = evaluation.get("actions") if isinstance(evaluation.get("actions"), list) else []
        rows.append({
            "domain": domain,
            "split": split,
            "official_task_split": "train",
            "task_id": task_id,
            "task_family": str(row.get("task_family") or ""),
            "task_sha256": str(row.get("task_sha256") or ""),
            "prompt_sha256": str(row.get("prompt_sha256") or ""),
            "has_reference_actions": "true" if actions else "false",
        })
    return rows


def _result_reward(path: Path) -> float | None:
    payload = _read_json(path)
    sims = payload.get("simulations")
    if not isinstance(sims, list) or not sims:
        return None
    reward_info = sims[0].get("reward_info") if isinstance(sims[0], dict) else None
    if not isinstance(reward_info, dict):
        return None
    reward = reward_info.get("reward")
    return float(reward) if isinstance(reward, (int, float)) and not isinstance(reward, bool) else None


def _require_loopback(api_base: str, label: str) -> None:
    parsed = urlparse(api_base)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise Tau3TeacherGenerationError(f"{label} endpoint must be loopback: {api_base}")


def _require_revision(repo: Path, expected_revision: str) -> None:
    revision = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()
    if revision != expected_revision:
        raise Tau3TeacherGenerationError(f"Tau repository revision mismatch: {revision!r}")
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    if dirty:
        raise Tau3TeacherGenerationError("Tau repository has tracked modifications")


def _validate_config(config: Tau3TeacherGenerationConfig) -> None:
    if config.max_tasks < 1:
        raise Tau3TeacherGenerationError("max_tasks must be positive")
    if not 1 <= config.max_steps <= 100:
        raise Tau3TeacherGenerationError("max_steps must be between 1 and 100")
    if not 1 <= config.max_errors <= 100:
        raise Tau3TeacherGenerationError("max_errors must be between 1 and 100")
    if not 1 <= config.timeout_seconds <= 3600:
        raise Tau3TeacherGenerationError("timeout_seconds must be between 1 and 3600")
    if config.agent not in {"auto", "llm_agent", "llm_agent_gt"}:
        raise Tau3TeacherGenerationError("agent must be auto, llm_agent, or llm_agent_gt")


def _config_record(config: Tau3TeacherGenerationConfig) -> dict[str, Any]:
    return {
        "max_steps": config.max_steps,
        "max_errors": config.max_errors,
        "timeout_seconds": config.timeout_seconds,
        "seed": config.seed,
        "max_tasks": config.max_tasks,
        "agent": config.agent,
        "communication_protocol_enforced": not config.allow_teacher_protocol_normalization,
    }


def _validate_resumable_manifest(
    manifest: dict[str, Any],
    *,
    out: Path,
    tasks: list[dict[str, str]],
    expected_binding: dict[str, Any],
) -> None:
    schema = check_schema_contract(manifest, name_or_id="tau3_teacher_generation_run")
    errors = list(schema.get("errors") or [])
    if manifest.get("phase") != "final":
        errors.append("existing manifest is not final")
    for key, expected in expected_binding.items():
        if manifest.get(key) != expected:
            errors.append(f"existing manifest {key} does not match this generation run")
    receipts = manifest.get("task_receipts")
    if not isinstance(receipts, list) or len(receipts) != len(tasks):
        errors.append("existing manifest task receipt count does not match selected tasks")
        receipts = []
    success_count = 0
    failure_count = 0
    for index, (task, ref) in enumerate(zip(tasks, receipts)):
        if not isinstance(ref, dict):
            errors.append(f"existing manifest task receipt {index} is not an object")
            continue
        expected_name = f"task-{index:04d}-{task['domain']}-{_task_key(task['task_id'])}.json"
        if ref.get("path") != expected_name:
            errors.append(f"existing manifest task receipt {index} path mismatch")
            continue
        receipt_path = out / expected_name
        if not receipt_path.is_file():
            errors.append(f"existing task receipt is missing: {expected_name}")
            continue
        receipt = _read_json(receipt_path)
        if receipt.get("task") != task:
            errors.append(f"existing task receipt identity mismatch: {expected_name}")
        if receipt.get("terminal_status") != ref.get("terminal_status"):
            errors.append(f"existing task receipt status mismatch: {expected_name}")
        if receipt.get("result_sha256") != ref.get("result_sha256"):
            errors.append(f"existing task receipt result hash mismatch: {expected_name}")
        result_path_value = receipt.get("result_path")
        result_sha256 = receipt.get("result_sha256")
        if result_sha256 is not None:
            result_path = Path(result_path_value) if isinstance(result_path_value, str) else None
            if result_path is None or not result_path.is_file() or _sha256(result_path) != result_sha256:
                errors.append(f"existing generated result drifted: {expected_name}")
        if ref.get("terminal_status") == "success":
            success_count += 1
        else:
            failure_count += 1
    if manifest.get("success_count") != success_count or manifest.get("failure_count") != failure_count:
        errors.append("existing manifest success/failure counts do not replay task receipts")
    if errors:
        raise Tau3TeacherGenerationError("existing final manifest is stale or invalid: " + "; ".join(errors))


def _write_json_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise Tau3TeacherGenerationError(f"refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o600)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Tau3TeacherGenerationError(f"JSON file must contain an object: {path}")
    return payload


def _file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size": path.stat().st_size, "sha256": _sha256(path)}


def _endpoint_record(endpoint: Tau3Endpoint) -> dict[str, Any]:
    parsed = urlparse(endpoint.api_base)
    return {
        "model": endpoint.model,
        "endpoint": f"{parsed.scheme}://{parsed.hostname}:{parsed.port or 80}{parsed.path}",
        "loopback": parsed.hostname in {"127.0.0.1", "localhost"},
        "temperature": endpoint.temperature,
        "top_p": endpoint.top_p,
        "max_tokens": endpoint.max_tokens,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _task_key(task_id: str) -> str:
    return hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:16]


def _redact_argv(argv: list[str]) -> list[str]:
    return [_redact(item) for item in argv]


def _redact(text: str) -> str:
    return text.replace('"api_key":"local"', '"api_key":"[REDACTED]"').replace('"api_key": "local"', '"api_key": "[REDACTED]"')


def _now_utc() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
