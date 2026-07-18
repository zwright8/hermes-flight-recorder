#!/usr/bin/env python3
"""Start a serving process, preflight it, optionally run Eval, then stop it."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.check_openai_serving import check_endpoint  # noqa: E402


SCHEMA_VERSION = "hfr.serving_lifecycle_run.v1"
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-command", required=True, help="Command to start the serving process; parsed with shlex")
    parser.add_argument("--server-cwd", type=Path, default=Path.cwd(), help="Working directory for the serving process")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="Extra environment variable for the server and eval command")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL to preflight")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--provider", default="custom")
    parser.add_argument("--arm", default="candidate")
    parser.add_argument("--engine", choices=["openai_compatible", "sglang", "transformers", "vllm"], default="transformers")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--profile-id", default="")
    parser.add_argument("--out", type=Path, required=True, help="Lifecycle artifact directory")
    parser.add_argument("--api-key-env", default="HERMES_EVAL_API_KEY")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--request-timeout", type=float, default=15.0)
    parser.add_argument("--readiness-timeout", type=float, default=120.0)
    parser.add_argument("--probe-interval", type=float, default=2.0)
    parser.add_argument("--terminate-timeout", type=float, default=20.0)
    parser.add_argument("--require-tool-call", action="store_true")
    parser.add_argument("--require-structured-output", action="store_true")
    parser.add_argument(
        "--eval-command",
        default="",
        help=(
            "Optional Eval command; parsed with shlex after formatting placeholders "
            "{base_url}, {serving_profile}, {serving_check}, {compatibility_report}, {model}, and {out}"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    server_stdout_path = out_dir / "server_stdout.txt"
    server_stderr_path = out_dir / "server_stderr.txt"
    eval_stdout_path = out_dir / "eval_stdout.txt"
    eval_stderr_path = out_dir / "eval_stderr.txt"
    lifecycle_path = out_dir / "serving_lifecycle_run.json"
    server_command = shlex.split(args.server_command)
    if not server_command:
        raise SystemExit("--server-command parsed to an empty command")

    env = _environment(args.env)
    started_at = _utc_now()
    process: subprocess.Popen[str] | None = None
    readiness: dict[str, Any] = {"passed": False, "attempt_count": 0, "attempts": []}
    eval_result: dict[str, Any] = {"status": "skipped", "reason": "no_eval_command"}
    cleanup: dict[str, Any] = {"attempted": False, "terminated": False, "killed": False}

    with server_stdout_path.open("w", encoding="utf-8") as server_stdout, server_stderr_path.open("w", encoding="utf-8") as server_stderr:
        try:
            process = subprocess.Popen(
                server_command,
                cwd=args.server_cwd.expanduser().resolve(),
                env=env,
                text=True,
                stdout=server_stdout,
                stderr=server_stderr,
                start_new_session=True,
            )
            readiness = _wait_for_readiness(args, out_dir, process)
            if readiness["passed"] and args.eval_command:
                eval_result = _run_eval_command(
                    args=args,
                    out_dir=out_dir,
                    env=env,
                    stdout_path=eval_stdout_path,
                    stderr_path=eval_stderr_path,
                )
        finally:
            if process is not None:
                cleanup = _terminate_process(process, process_group_id=process.pid, timeout=float(args.terminate_timeout))

    lifecycle = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "started_at": started_at,
        "completed_at": _utc_now(),
        "passed": bool(readiness.get("passed") and eval_result.get("status") in {"passed", "skipped"}),
        "server": {
            "command": server_command,
            "command_text": args.server_command,
            "cwd": str(args.server_cwd.expanduser().resolve()),
            "pid": process.pid if process is not None else None,
            "stdout": str(server_stdout_path),
            "stderr": str(server_stderr_path),
            "exit_code_after_cleanup": process.poll() if process is not None else None,
        },
        "preflight": readiness,
        "eval": eval_result,
        "cleanup": cleanup,
        "artifacts": {
            "serving_profile": str(out_dir / "serving_profile.json"),
            "compatibility_report": str(out_dir / "compatibility_report.json"),
            "serving_check": str(out_dir / "serving_check.json"),
            "lifecycle_run": str(lifecycle_path),
            "server_stdout": str(server_stdout_path),
            "server_stderr": str(server_stderr_path),
            "eval_stdout": str(eval_stdout_path) if eval_result.get("status") != "skipped" else None,
            "eval_stderr": str(eval_stderr_path) if eval_result.get("status") != "skipped" else None,
        },
    }
    _write_json(lifecycle_path, lifecycle)
    print(json.dumps({"passed": lifecycle["passed"], "out": str(out_dir), "preflight": readiness.get("readiness"), "eval": eval_result.get("status")}, indent=2))
    return 0 if lifecycle["passed"] else 1


def _wait_for_readiness(args: argparse.Namespace, out_dir: Path, process: subprocess.Popen[str]) -> dict[str, Any]:
    deadline = time.time() + float(args.readiness_timeout)
    attempts: list[dict[str, Any]] = []
    last_profile: dict[str, Any] | None = None
    last_compatibility: dict[str, Any] | None = None
    last_report: dict[str, Any] | None = None
    api_key = _api_key(args, args.base_url)

    while time.time() <= deadline:
        if process.poll() is not None:
            attempts.append({"at": _utc_now(), "ready": False, "reason": "server_exited", "exit_code": process.returncode})
            break
        profile, compatibility, report = check_endpoint(
            base_url=str(args.base_url),
            model=str(args.model),
            provider=str(args.provider),
            arm=str(args.arm),
            engine=str(args.engine),
            adapter=str(args.adapter),
            profile_id=str(args.profile_id),
            api_key=api_key,
            timeout=float(args.request_timeout),
            out_dir=out_dir,
            require_tool_call=bool(args.require_tool_call),
            require_structured_output=bool(args.require_structured_output),
        )
        last_profile = profile
        last_compatibility = compatibility
        last_report = report
        attempts.append({"at": _utc_now(), "ready": bool(report["passed"]), "failed_checks": report.get("failed_checks", [])})
        if report["passed"]:
            _write_serving_artifacts(out_dir, profile, compatibility, report)
            return {
                "passed": True,
                "readiness": "ready",
                "attempt_count": len(attempts),
                "attempts": attempts,
                "failed_checks": [],
                "serving_profile": str(out_dir / "serving_profile.json"),
                "compatibility_report": str(out_dir / "compatibility_report.json"),
                "serving_check": str(out_dir / "serving_check.json"),
            }
        time.sleep(float(args.probe_interval))

    if last_profile and last_compatibility and last_report:
        _write_serving_artifacts(out_dir, last_profile, last_compatibility, last_report)
    return {
        "passed": False,
        "readiness": "blocked",
        "attempt_count": len(attempts),
        "attempts": attempts,
        "failed_checks": (last_report or {}).get("failed_checks", ["endpoint_not_ready"]),
        "serving_profile": str(out_dir / "serving_profile.json") if last_profile else None,
        "compatibility_report": str(out_dir / "compatibility_report.json") if last_compatibility else None,
        "serving_check": str(out_dir / "serving_check.json") if last_report else None,
    }


def _run_eval_command(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> dict[str, Any]:
    formatted = args.eval_command.format(
        base_url=args.base_url,
        serving_profile=str(out_dir / "serving_profile.json"),
        serving_check=str(out_dir / "serving_check.json"),
        compatibility_report=str(out_dir / "compatibility_report.json"),
        model=args.model,
        out=str(out_dir),
    )
    command = shlex.split(formatted)
    eval_env = {
        **env,
        "HFR_SERVING_BASE_URL": str(args.base_url),
        "HFR_SERVING_PROFILE": str(out_dir / "serving_profile.json"),
        "HFR_SERVING_CHECK": str(out_dir / "serving_check.json"),
        "HFR_SERVING_COMPATIBILITY_REPORT": str(out_dir / "compatibility_report.json"),
    }
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            command,
            cwd=Path.cwd(),
            env=eval_env,
            text=True,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    return {
        "status": "passed" if completed.returncode == 0 else "failed",
        "command": command,
        "command_text": formatted,
        "exit_code": completed.returncode,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def _terminate_process(process: subprocess.Popen[str], *, process_group_id: int, timeout: float) -> dict[str, Any]:
    cleanup = {
        "attempted": True,
        "terminated": False,
        "killed": False,
        "process_group_id": process_group_id,
        "exit_code_before_cleanup": process.poll(),
    }
    if _process_group_exists(process_group_id):
        os.killpg(process_group_id, signal.SIGTERM)
        cleanup["terminated_signal_sent"] = True
        deadline = time.time() + timeout
        while time.time() < deadline and _process_group_exists(process_group_id):
            if process.poll() is None:
                try:
                    process.wait(timeout=min(0.2, max(0.01, deadline - time.time())))
                except subprocess.TimeoutExpired:
                    pass
            else:
                time.sleep(0.05)
        if _process_group_exists(process_group_id):
            os.killpg(process_group_id, signal.SIGKILL)
            cleanup["killed"] = True
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass
        else:
            cleanup["terminated"] = True
    elif process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=timeout)
            cleanup["terminated"] = True
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=timeout)
            cleanup["killed"] = True
    cleanup["exit_code_after_cleanup"] = process.poll()
    return cleanup


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _environment(extra: list[str]) -> dict[str, str]:
    env = os.environ.copy()
    for raw in extra:
        if "=" not in raw:
            raise SystemExit(f"--env must use KEY=VALUE format: {raw}")
        key, value = raw.split("=", 1)
        if not key:
            raise SystemExit(f"--env has empty key: {raw}")
        env[key] = value
    return env


def _api_key(args: argparse.Namespace, base_url: str) -> str:
    if args.api_key:
        return str(args.api_key)
    if args.api_key_env and os.environ.get(args.api_key_env):
        return str(os.environ[args.api_key_env])
    host = urlparse(base_url).hostname or ""
    if host in {"127.0.0.1", "localhost", "::1"}:
        return "hfr-local-key"
    return ""


def _write_serving_artifacts(out_dir: Path, profile: dict[str, Any], compatibility: dict[str, Any], report: dict[str, Any]) -> None:
    _write_json(out_dir / "serving_profile.json", profile)
    _write_json(out_dir / "compatibility_report.json", compatibility)
    _write_json(out_dir / "serving_check.json", report)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


if __name__ == "__main__":
    raise SystemExit(main())
