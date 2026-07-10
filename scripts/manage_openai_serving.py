#!/usr/bin/env python3
"""Start, verify, and tear down an OpenAI-compatible serving process."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path
from typing import Any, TextIO


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from check_openai_serving import (  # noqa: E402
    DEFAULT_MODEL,
    check_endpoint,
    public_path_replacements,
    sanitize_public_artifact,
    sensitive_url_values,
)
from flightrecorder.redaction import is_secret_key  # noqa: E402
from flightrecorder.safe_http import (  # noqa: E402
    DEFAULT_MAX_BODY_BYTES,
    DEFAULT_MAX_ERROR_BYTES,
    HttpStatusError,
    SafeHttpError,
    bounded_http_request,
)


SCHEMA_LIFECYCLE = "hfr.serving_lifecycle.v1"
_MAX_LOG_LINE_CHARS = 64 * 1024
_OVERSIZED_LOG_LINE_MARKER = "[REDACTED: oversized log line]"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["custom", "mock", "sglang", "vllm"], default="custom")
    parser.add_argument("--command", help="Command to launch. Omit when using a built-in profile.")
    parser.add_argument("--cwd", default=".", help="Working directory for the managed process.")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="Extra environment variable for the managed process.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--base-url", help="OpenAI-compatible base URL. Defaults to http://HOST:PORT/v1.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--served-model-name", default="")
    parser.add_argument("--provider", default="custom")
    parser.add_argument("--arm", default="candidate")
    parser.add_argument("--engine", choices=["mock", "openai_compatible", "sglang", "transformers", "vllm"], default="")
    parser.add_argument("--adapter", default="", help="Optional adapter path or id.")
    parser.add_argument(
        "--adapter-load-strategy",
        choices=["auto", "none", "mock_suffix", "engine_args", "merged"],
        default="auto",
        help="Record how --adapter is expected to be used; built-in non-mock profiles do not add adapter flags automatically.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--extra-engine-arg", action="append", default=[], help="Additional argument passed to a built-in engine profile.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--startup-timeout", type=float, default=300.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--check-timeout", type=float, default=15.0)
    parser.add_argument("--grace-period", type=float, default=10.0)
    parser.add_argument("--api-key-env", default="HERMES_EVAL_API_KEY")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--require-streaming", action="store_true")
    parser.add_argument("--require-tool-call", action="store_true")
    parser.add_argument("--require-structured-output", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    preflight_dir = out_dir / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = out_dir / "server.stdout.log"
    stderr_path = out_dir / "server.stderr.log"
    lifecycle_path = out_dir / "serving_lifecycle.json"
    stdout_path.unlink(missing_ok=True)
    stderr_path.unlink(missing_ok=True)
    private_logs = tempfile.TemporaryDirectory(prefix="hfr-serving-logs-")
    private_log_dir = Path(private_logs.name)
    private_log_dir.chmod(0o700)
    private_stdout_path = private_log_dir / stdout_path.name
    private_stderr_path = private_log_dir / stderr_path.name

    base_url = args.base_url or f"http://{args.host}:{int(args.port)}/v1"
    command = _launch_command(args)
    engine = args.engine or ("openai_compatible" if args.profile == "custom" else args.profile)
    adapter_strategy = _adapter_strategy(args)
    env = _process_env(args.env)
    cwd = Path(args.cwd).expanduser().resolve()
    api_key = _api_key(args, base_url)
    secret_values = [
        api_key,
        *sensitive_url_values(base_url),
        *_env_values(args.env),
        *_environment_secret_values(env),
        *_command_secret_values(command),
    ]
    path_replacements = {
        **public_path_replacements(out_dir, label="."),
        **public_path_replacements(cwd, label="."),
        **public_path_replacements(args.adapter, label=Path(args.adapter).name or "adapter"),
        **public_path_replacements(args.model, label=Path(args.model).name or "model"),
    }
    generated_at = _utc_now()
    started_monotonic = time.monotonic()
    process: subprocess.Popen[Any] | None = None
    readiness: dict[str, Any] = {"ready": False, "attempts": [], "summary": "not_started"}
    smoke: dict[str, Any] = {"attempted": False, "passed": False, "summary": "not_started"}
    teardown: dict[str, Any] = {"attempted": False, "clean": False, "running_after_teardown": False}
    errors: list[str] = []

    lifecycle = _base_lifecycle(args, command, base_url, engine, cwd, generated_at, out_dir, adapter_strategy)
    try:
        with private_stdout_path.open("wb") as stdout_handle, private_stderr_path.open("wb") as stderr_handle:
            private_stdout_path.chmod(0o600)
            private_stderr_path.chmod(0o600)
            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                start_new_session=(os.name != "nt"),
            )
            lifecycle["process"] = {"pid": process.pid, "started": True}
            readiness = _wait_until_ready(
                process,
                base_url,
                timeout=float(args.startup_timeout),
                poll_interval=float(args.poll_interval),
                api_key=api_key,
            )
            if readiness["ready"]:
                profile, compatibility, report = check_endpoint(
                    base_url=base_url,
                    model=args.served_model_name or args.model,
                    provider=args.provider,
                    arm=args.arm,
                    engine=engine if engine != "mock" else "openai_compatible",
                    adapter=args.adapter,
                    api_key=api_key,
                    timeout=float(args.check_timeout),
                    out_dir=preflight_dir,
                    require_streaming=bool(args.require_streaming),
                    require_tool_call=bool(args.require_tool_call),
                    require_structured_output=bool(args.require_structured_output),
                )
                profile["adapter_strategy"] = adapter_strategy
                profile.setdefault("model_identity", {})["adapter_strategy"] = adapter_strategy
                report["adapter_strategy"] = adapter_strategy
                profile = sanitize_public_artifact(
                    profile,
                    secret_values=secret_values,
                    path_replacements=path_replacements,
                )
                compatibility = sanitize_public_artifact(
                    compatibility,
                    secret_values=secret_values,
                    path_replacements=path_replacements,
                )
                report = sanitize_public_artifact(
                    report,
                    secret_values=secret_values,
                    path_replacements=path_replacements,
                )
                _write_json(preflight_dir / "serving_profile.json", profile)
                _write_json(preflight_dir / "compatibility_report.json", compatibility)
                _write_json(preflight_dir / "serving_check.json", report)
                smoke = {
                    "attempted": True,
                    "passed": bool(report["passed"]),
                    "readiness": report["readiness"],
                    "failed_checks": report["failed_checks"],
                    "artifacts": _preflight_artifacts(out_dir),
                }
            else:
                errors.append(str(readiness.get("summary") or "endpoint did not become ready"))
    except Exception as exc:  # pragma: no cover - exercised by integration failures.
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if process is not None:
            teardown = _teardown_process(process, grace_period=float(args.grace_period))

    logs_published = True
    for private_path, public_path in (
        (private_stdout_path, stdout_path),
        (private_stderr_path, stderr_path),
    ):
        try:
            _sanitize_log_file(
                private_path,
                public_path=public_path,
                secret_values=secret_values,
                path_replacements=path_replacements,
            )
        except Exception as exc:  # pragma: no cover - exercised with injected sanitizer failures.
            public_path.unlink(missing_ok=True)
            logs_published = False
            errors.append(f"{public_path.name} publication failed: {type(exc).__name__}: {exc}")
    private_logs.cleanup()

    passed = bool(
        readiness.get("ready")
        and smoke.get("passed")
        and teardown.get("clean")
        and logs_published
    )
    log_artifacts = {
        **({"stdout_log": "server.stdout.log"} if stdout_path.exists() else {}),
        **({"stderr_log": "server.stderr.log"} if stderr_path.exists() else {}),
    }
    lifecycle.update(
        {
            "finished_at": _utc_now(),
            "duration_ms": int((time.monotonic() - started_monotonic) * 1000),
            "passed": passed,
            "readiness": "ready" if passed else "blocked",
            "ready": passed,
            "readiness_probe": readiness,
            "smoke_check": smoke,
            "teardown": teardown,
            "errors": errors,
            "artifacts": {
                "serving_lifecycle": "serving_lifecycle.json",
                **log_artifacts,
                **(_preflight_artifacts(out_dir) if smoke.get("attempted") else {}),
            },
            "logs": {
                "stdout_tail": _read_tail(stdout_path),
                "stderr_tail": _read_tail(stderr_path),
            },
        }
    )
    if process is not None:
        lifecycle["process"]["exit_code"] = process.poll()
    lifecycle = sanitize_public_artifact(
        lifecycle,
        secret_values=secret_values,
        path_replacements=path_replacements,
    )
    _write_json(lifecycle_path, lifecycle)
    print(
        json.dumps(
            {
                "passed": passed,
                "readiness": lifecycle["readiness"],
                "out": out_dir.name,
                "failed_checks": smoke.get("failed_checks", []),
            },
            indent=2,
        )
    )
    return 0 if passed else 1


def _launch_command(args: argparse.Namespace) -> list[str]:
    if args.command:
        return shlex.split(args.command)
    served_model = args.served_model_name or args.model
    extra = list(args.extra_engine_arg or [])
    if args.profile == "mock":
        command = [
            "python3",
            str(Path("scripts") / "mock_openai_server.py"),
            "--host",
            args.host,
            "--port",
            str(int(args.port)),
            "--model",
            served_model,
            "--response",
            "hfr serving smoke ok",
        ]
        if args.adapter:
            command.extend(["--adapter", args.adapter])
        return command + extra
    if args.profile == "vllm":
        command = [
            "vllm",
            "serve",
            args.model,
            "--host",
            args.host,
            "--port",
            str(int(args.port)),
            "--served-model-name",
            served_model,
        ]
        if int(args.tensor_parallel_size) > 1:
            command.extend(["--tensor-parallel-size", str(int(args.tensor_parallel_size))])
        return command + extra
    if args.profile == "sglang":
        command = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            args.model,
            "--host",
            args.host,
            "--port",
            str(int(args.port)),
        ]
        if int(args.tensor_parallel_size) > 1:
            command.extend(["--tp-size", str(int(args.tensor_parallel_size))])
        if served_model != args.model:
            command.extend(["--served-model-name", served_model])
        return command + extra
    raise SystemExit("--command is required when --profile custom is used")


def _base_lifecycle(
    args: argparse.Namespace,
    command: list[str],
    base_url: str,
    engine: str,
    cwd: Path,
    generated_at: str,
    out_dir: Path,
    adapter_strategy: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_LIFECYCLE,
        "generated_at": generated_at,
        "profile": args.profile,
        "engine": engine,
        "arm": args.arm,
        "provider": args.provider,
        "model": args.model,
        "served_model_name": args.served_model_name or args.model,
        "adapter": args.adapter,
        "adapter_strategy": adapter_strategy,
        "endpoint": {"base_url": base_url, "host": args.host, "port": int(args.port)},
        "launch": {
            "command": command,
            "command_display": shlex.join(command),
            "cwd": "." if cwd == Path.cwd().resolve() else str(cwd),
            "env_keys": sorted(_env_keys(args.env)),
            "startup_timeout_s": float(args.startup_timeout),
            "poll_interval_s": float(args.poll_interval),
            "grace_period_s": float(args.grace_period),
        },
        "process": {"started": False},
        "environment": {"python_version": platform.python_version(), "platform": platform.platform()},
        "artifacts_root": ".",
    }


def _adapter_strategy(args: argparse.Namespace) -> dict[str, Any]:
    adapter = str(args.adapter or "")
    requested = str(args.adapter_load_strategy or "auto")
    if not adapter:
        resolved = "none"
    elif requested != "auto":
        resolved = requested
    elif args.profile == "mock":
        resolved = "mock_suffix"
    else:
        resolved = "engine_args"
    requires_engine_args = bool(adapter and resolved == "engine_args")
    notes: list[str] = []
    if adapter and args.profile in {"vllm", "sglang"} and resolved == "engine_args":
        notes.append("Built-in launch profiles record adapter intent; pass engine-specific adapter flags with --extra-engine-arg or use --command.")
    if adapter and resolved == "merged":
        notes.append("Adapter is treated as already merged into the served model; lifecycle records it for provenance only.")
    if not adapter and requested not in {"auto", "none"}:
        notes.append("No adapter was provided, so the resolved strategy is none.")
    return {
        "present": bool(adapter),
        "adapter": adapter,
        "adapter_id": Path(adapter).name if adapter else "",
        "requested_strategy": requested,
        "resolved_strategy": resolved,
        "engine_profile": args.profile,
        "launch_command_applies_adapter": bool(adapter and args.profile == "mock" and resolved == "mock_suffix"),
        "requires_engine_args": requires_engine_args,
        "notes": notes,
    }


def _wait_until_ready(process: subprocess.Popen[Any], base_url: str, *, timeout: float, poll_interval: float, api_key: str) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    attempts: list[dict[str, Any]] = []
    urls = [_root_url(base_url, "/healthz"), _root_url(base_url, "/health"), _root_url(base_url, "/version"), _openai_url(base_url, "/models")]
    while time.monotonic() <= deadline:
        exit_code = process.poll()
        if exit_code is not None:
            return {"ready": False, "summary": "process_exited_before_ready", "exit_code": exit_code, "attempts": attempts}
        for url in urls:
            result = _request_json("GET", url, api_key=api_key, timeout=min(max(poll_interval, 0.2), 5.0))
            attempts.append(result)
            if result["ok"]:
                return {"ready": True, "summary": "readiness_endpoint_passed", "url": url, "attempts": attempts}
        time.sleep(max(poll_interval, 0.1))
    return {"ready": False, "summary": "startup_timeout", "timeout_s": timeout, "attempts": attempts}


def _teardown_process(process: subprocess.Popen[Any], *, grace_period: float) -> dict[str, Any]:
    started_at = _utc_now()
    exit_before = process.poll()
    if exit_before is not None:
        return {
            "attempted": True,
            "started_at": started_at,
            "already_exited": True,
            "exit_code_before_teardown": exit_before,
            "exit_code_after_teardown": exit_before,
            "terminated": False,
            "killed": False,
            "clean": True,
            "running_after_teardown": False,
        }
    _terminate(process)
    terminated = True
    killed = False
    try:
        exit_after = process.wait(timeout=grace_period)
    except subprocess.TimeoutExpired:
        _kill(process)
        killed = True
        try:
            exit_after = process.wait(timeout=max(grace_period, 1.0))
        except subprocess.TimeoutExpired:
            exit_after = None
    return {
        "attempted": True,
        "started_at": started_at,
        "already_exited": False,
        "exit_code_before_teardown": exit_before,
        "exit_code_after_teardown": exit_after,
        "terminated": terminated,
        "killed": killed,
        "clean": exit_after is not None,
        "running_after_teardown": process.poll() is None,
    }


def _terminate(process: subprocess.Popen[Any]) -> None:
    if os.name == "nt":
        process.terminate()
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def _kill(process: subprocess.Popen[Any]) -> None:
    if os.name == "nt":
        process.kill()
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def _request_json(method: str, url: str, *, api_key: str, timeout: float) -> dict[str, Any]:
    headers = {"accept": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    started = time.time()
    try:
        status_code, body = bounded_http_request(
            method,
            url,
            headers=headers,
            timeout=timeout,
            max_body_bytes=DEFAULT_MAX_BODY_BYTES,
            max_error_bytes=DEFAULT_MAX_ERROR_BYTES,
        )
        text = body.decode("utf-8", "replace")
        return {
            "ok": 200 <= status_code < 300,
            "status_code": status_code,
            "url": url,
            "elapsed_ms": int((time.time() - started) * 1000),
            "json": json.loads(text) if text else {},
        }
    except json.JSONDecodeError as exc:
        return {"ok": False, "status_code": None, "url": url, "elapsed_ms": int((time.time() - started) * 1000), "error": f"invalid_json: {exc.msg}"}
    except HttpStatusError as exc:
        return {"ok": False, "status_code": exc.status_code, "url": url, "elapsed_ms": int((time.time() - started) * 1000), "error": str(exc)}
    except SafeHttpError as exc:
        return {"ok": False, "status_code": None, "url": url, "elapsed_ms": int((time.time() - started) * 1000), "error": str(exc)}


def _process_env(items: list[str]) -> dict[str, str]:
    env = dict(os.environ)
    for item in items:
        key, value = _split_env(item)
        env[key] = value
    return env


def _env_values(items: list[str]) -> list[str]:
    return [value for _key, value in (_split_env(item) for item in items) if value]


def _environment_secret_values(env: dict[str, str]) -> list[str]:
    return [value for key, value in env.items() if value and _is_secret_flag(key)]


def _command_secret_values(command: list[str]) -> list[str]:
    values: list[str] = []
    for index, item in enumerate(command):
        if item.startswith(("http://", "https://")):
            values.extend(sensitive_url_values(item))
        if not item.startswith("-"):
            if item.lower() == "bearer" and index + 1 < len(command):
                values.append(command[index + 1])
            continue
        flag, separator, inline_value = item.partition("=")
        if not _is_secret_flag(flag):
            continue
        if separator and inline_value:
            values.append(inline_value)
        elif index + 1 < len(command):
            values.append(command[index + 1])
    return [value for value in values if value]


def _is_secret_flag(flag: str) -> bool:
    normalized = flag.lstrip("-").lower().replace("-", "_")
    return is_secret_key(normalized)


def _env_keys(items: list[str]) -> set[str]:
    return {_split_env(item)[0] for item in items}


def _split_env(item: str) -> tuple[str, str]:
    if "=" not in item:
        raise SystemExit(f"--env must use KEY=VALUE: {item}")
    key, value = item.split("=", 1)
    if not key:
        raise SystemExit(f"--env key must be non-empty: {item}")
    return key, value


def _api_key(args: argparse.Namespace, base_url: str) -> str:
    if args.api_key:
        return str(args.api_key)
    if args.api_key_env and os.environ.get(args.api_key_env):
        return str(os.environ[args.api_key_env])
    host = urllib.parse.urlparse(base_url).hostname or ""
    return "hfr-local-key" if host in {"127.0.0.1", "localhost", "::1"} else ""


def _root_url(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return f"{base}{path}"


def _openai_url(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    return f"{base}{path}" if base.endswith("/v1") else f"{base}/v1{path}"


def _preflight_artifacts(out_dir: Path) -> dict[str, str]:
    return {
        "serving_profile": str((out_dir / "preflight" / "serving_profile.json").relative_to(out_dir)),
        "compatibility_report": str((out_dir / "preflight" / "compatibility_report.json").relative_to(out_dir)),
        "serving_check": str((out_dir / "preflight" / "serving_check.json").relative_to(out_dir)),
    }


def _read_tail(path: Path, limit: int = 4096) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - limit), os.SEEK_SET)
        return handle.read().decode("utf-8", "replace")


def _sanitize_log_file(
    path: Path,
    *,
    public_path: Path | None = None,
    secret_values: list[str],
    path_replacements: dict[str, str],
) -> None:
    destination_path = public_path or path
    publishing_from_private_source = destination_path != path
    if publishing_from_private_source:
        destination_path.unlink(missing_ok=True)
    if not path.exists():
        return
    source_mode = path.stat().st_mode & 0o7777
    temporary_fd, temporary_name = tempfile.mkstemp(
        dir=destination_path.parent,
        prefix=f".{destination_path.name}.",
        suffix=".sanitizing",
    )
    temporary_path = Path(temporary_name)
    try:
        with (
            path.open("r", encoding="utf-8", errors="replace", newline="") as source,
            os.fdopen(temporary_fd, "w", encoding="utf-8", newline="") as destination,
        ):
            temporary_fd = -1
            while line := source.readline(_MAX_LOG_LINE_CHARS + 1):
                if len(line) > _MAX_LOG_LINE_CHARS and not line.endswith(("\r", "\n")):
                    line_ending = _discard_oversized_log_line(source)
                    destination.write(f"{_OVERSIZED_LOG_LINE_MARKER}{line_ending}")
                    continue
                public_text = sanitize_public_artifact(
                    line,
                    secret_values=secret_values,
                    path_replacements=path_replacements,
                )
                destination.write(public_text)
            destination.flush()
            os.fsync(destination.fileno())
        temporary_path.chmod(source_mode)
        os.replace(temporary_path, destination_path)
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        temporary_path.unlink(missing_ok=True)


def _discard_oversized_log_line(source: TextIO) -> str:
    while remainder := source.readline(_MAX_LOG_LINE_CHARS + 1):
        if remainder.endswith("\r\n"):
            return "\r\n"
        if remainder.endswith(("\r", "\n")):
            return remainder[-1]
        if len(remainder) <= _MAX_LOG_LINE_CHARS:
            break
    return ""


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


if __name__ == "__main__":
    raise SystemExit(main())
