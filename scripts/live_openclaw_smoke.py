#!/usr/bin/env python3
"""Run a live OpenClaw smoke test against a local mock model server.

The smoke uses a real installed `openclaw` CLI, an isolated OpenClaw config and
state directory, the read-only Flight Recorder OpenClaw plugin, and a local
OpenAI-compatible chat endpoint. It requires no external API keys.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flightrecorder.cli import _run_scenario_artifacts
from scripts.hermes_harness import publish_harness_artifacts, write_fake_secret_canaries


SUMMARY_SCHEMA_VERSION = "hfr.openclaw.live_smoke.summary.v1"
PROMPT = "Reply exactly: flight recorder openclaw live smoke ok"
EXPECTED = "flight recorder openclaw live smoke ok"
MODEL_ID = "hfr-openclaw-smoke"
MODEL_REF = f"hfrmock/{MODEL_ID}"
SESSION_KEY = "agent:main:hfr-openclaw-live-smoke"


class MockOpenAIHandler(BaseHTTPRequestHandler):
    """Small OpenAI-compatible chat/completions endpoint."""

    protocol_version = "HTTP/1.1"
    requests: list[dict[str, Any]] = []

    def log_message(self, _fmt: str, *_args: Any) -> None:
        return None

    def do_GET(self) -> None:
        self.requests.append(_request_summary(self.path, {}))
        path = self.path.split("?", 1)[0].rstrip("/")
        if path in {"/api/v1/models", "/v1/models", "/models"}:
            self._send_json({"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]})
            return
        if path == f"/v1/models/{MODEL_ID}":
            self._send_json({"id": MODEL_ID, "object": "model"})
            return
        if path in {"/v1/props", "/props"}:
            self._send_json({"model": MODEL_ID, "context_length": 32768})
            return
        self.send_response(404)
        self.send_header("connection", "close")
        self.end_headers()

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            payload = {"_raw": body.decode("utf-8", "replace")}
        self.requests.append(_request_summary(self.path, payload))

        path = self.path.split("?", 1)[0].rstrip("/")
        if path == "/v1/chat/completions":
            if payload.get("stream"):
                self._send_stream()
            else:
                self._send_json(_completion_payload())
            return
        if path == "/v1/responses":
            self._send_json(
                {
                    "id": "resp-hfr-openclaw",
                    "object": "response",
                    "created_at": int(time.time()),
                    "model": MODEL_ID,
                    "output_text": EXPECTED,
                    "status": "completed",
                    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                }
            )
            return
        self.send_response(404)
        self.send_header("connection", "close")
        self.end_headers()

    def _send_stream(self) -> None:
        created = int(time.time())
        chunks = [
            {
                "id": "chatcmpl-hfr-openclaw",
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": EXPECTED},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-hfr-openclaw",
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        ]
        text = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        text += "data: [DONE]\n\n"
        self._send_raw(text.encode("utf-8"), "text/event-stream")

    def _send_json(self, payload: dict[str, Any]) -> None:
        self._send_raw(json.dumps(payload).encode("utf-8"), "application/json")

    def _send_raw(self, raw: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("content-type", content_type)
        self.send_header("connection", "close")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a live OpenClaw Flight Recorder smoke test")
    parser.add_argument("--out", default="live_openclaw_smoke_artifacts/latest", help="Directory for smoke artifacts")
    parser.add_argument("--keep-temp", action="store_true", help="Keep the isolated temporary OpenClaw state")
    args = parser.parse_args(argv)

    if shutil.which("openclaw") is None:
        raise SystemExit("openclaw is required for the live OpenClaw smoke")

    out_dir = Path(args.out).expanduser().resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    MockOpenAIHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockOpenAIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    temp_root = Path(tempfile.mkdtemp(prefix="hfr-live-openclaw-"))
    try:
        result = _run_live_session(Path(__file__).resolve().parents[1], out_dir, temp_root, server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()
        if args.keep_temp:
            print(f"kept temp root: {temp_root}")
        else:
            shutil.rmtree(temp_root, ignore_errors=True)

    summary = _write_smoke_summary(out_dir, result)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


def _run_live_session(flight_root: Path, out_dir: Path, temp_root: Path, port: int) -> dict[str, Any]:
    config_path = temp_root / "openclaw.json"
    state_dir = temp_root / "state"
    workspace = temp_root / "workspace"
    events_dir = temp_root / "events"
    home_dir = temp_root / "home"
    gateway_port = _free_port()
    events_dir.mkdir(parents=True)
    home_dir.mkdir(parents=True)
    fake_secret_files = write_fake_secret_canaries(home_dir)

    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home_dir),
            "OPENCLAW_CONFIG_PATH": str(config_path),
            "OPENCLAW_STATE_DIR": str(state_dir),
            "OPENCLAW_FLIGHT_RECORDER_OUTPUT_DIR": str(events_dir),
            "OPENCLAW_FLIGHT_RECORDER_MAX_FIELD_CHARS": "20000",
            "NO_COLOR": "1",
        }
    )

    setup = _run_openclaw(
        ["setup", "--non-interactive", "--accept-risk", "--mode", "local", "--workspace", str(workspace)],
        env,
        out_dir,
        "openclaw_setup",
        timeout=60,
    )
    if not config_path.exists():
        raise SystemExit(f"OpenClaw setup did not create {config_path}; see {out_dir}")

    _patch_config(env, out_dir, workspace, port, gateway_port)
    _run_openclaw(["config", "validate", "--json"], env, out_dir, "openclaw_config_validate", timeout=30, check=True)
    _run_openclaw(
        ["plugins", "install", str(flight_root / "plugins" / "openclaw" / "flight_recorder"), "--link"],
        env,
        out_dir,
        "openclaw_plugin_install",
        timeout=60,
        check=True,
    )
    _run_openclaw(["plugins", "enable", "flight-recorder"], env, out_dir, "openclaw_plugin_enable", timeout=30, check=True)
    inspect = _run_openclaw(
        ["plugins", "inspect", "flight-recorder", "--json", "--runtime"],
        env,
        out_dir,
        "openclaw_plugin_inspect",
        timeout=60,
        check=True,
    )
    gateway = _start_gateway(env, out_dir, gateway_port)
    try:
        _wait_for_gateway(env, out_dir, timeout_seconds=45)
        plugins = _run_openclaw(["plugins", "list", "--json"], env, out_dir, "openclaw_plugins_list", timeout=60, check=True)
        agent = _run_openclaw(
            [
                "agent",
                "--json",
                "--message",
                PROMPT,
                "--model",
                MODEL_REF,
                "--session-key",
                SESSION_KEY,
                "--timeout",
                "60",
            ],
            env,
            out_dir,
            "openclaw_agent",
            timeout=90,
        )
    finally:
        _stop_gateway(gateway)
    (out_dir / "mock_requests.json").write_text(
        json.dumps(MockOpenAIHandler.requests, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    event_files = sorted(events_dir.glob("*.openclaw.jsonl"))
    if agent.returncode != 0:
        raise SystemExit(f"OpenClaw live smoke failed with exit {agent.returncode}; see {out_dir}")
    if not event_files:
        raise SystemExit(f"OpenClaw live smoke produced no OpenClaw JSONL; see {out_dir}")

    openclaw_trace = out_dir / "live_openclaw.openclaw.jsonl"
    _combine_jsonl(event_files, openclaw_trace)
    run_result = _write_smoke_artifacts(openclaw_trace, out_dir)
    harness_result = publish_harness_artifacts(
        scenario_path=out_dir / "live_openclaw_scenario.json",
        run_dir=out_dir,
        artifact_result=run_result,
        trace_path=openclaw_trace,
        trace_format="openclaw_jsonl",
        runner="openclaw_live_smoke",
        provider="hfrmock",
        model=MODEL_REF,
        base_url=f"http://127.0.0.1:{port}/v1",
        sandbox={
            "root": temp_root,
            "home": home_dir,
            "workspace": workspace,
            "events": events_dir,
            "state_dir": str(state_dir),
            "config": str(config_path),
            "ephemeral": True,
            "audit_artifacts_kept": True,
        },
        fake_secret_files=fake_secret_files,
        process={
            "setup_exit_code": setup.returncode,
            "agent_exit_code": agent.returncode,
            "setup_stdout": str(out_dir / "openclaw_setup_stdout.txt"),
            "setup_stderr": str(out_dir / "openclaw_setup_stderr.txt"),
            "agent_stdout": str(out_dir / "openclaw_agent_stdout.txt"),
            "agent_stderr": str(out_dir / "openclaw_agent_stderr.txt"),
            "gateway_port": gateway_port,
        },
        metadata={"source": "scripts/live_openclaw_smoke.py", "mock_endpoint": True, "session_key": SESSION_KEY},
    )
    scorecard = run_result["scorecard"]
    report_path = run_result["paths"]["report"]

    hooks = _read_hooks(openclaw_trace)
    required_hooks = {"agent_end"}
    missing_hooks = sorted(required_hooks - set(hooks))
    chat_requests = [request for request in MockOpenAIHandler.requests if request["path"].rstrip("/") == "/v1/chat/completions"]
    plugin_loaded = _plugin_loaded(plugins.stdout)
    runtime_hook_count = _runtime_hook_count(inspect.stdout)
    passed = (
        setup.returncode in {0, 1}
        and agent.returncode == 0
        and scorecard["passed"]
        and not missing_hooks
        and bool(chat_requests)
        and plugin_loaded
        and runtime_hook_count > 0
    )
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "passed": passed,
        "openclaw_exit_code": agent.returncode,
        "openclaw_setup_exit_code": setup.returncode,
        "mock_request_count": len(MockOpenAIHandler.requests),
        "chat_completion_request_count": len(chat_requests),
        "plugin_loaded": plugin_loaded,
        "plugin_runtime_hook_count": runtime_hook_count,
        "agent_mode": "gateway",
        "gateway_port": gateway_port,
        "openclaw_event_file": str(openclaw_trace),
        "hooks": hooks,
        "missing_hooks": missing_hooks,
        "score": scorecard["score"],
        "report": str(report_path),
        "lineage": str(run_result["paths"]["lineage"]),
        "task_completion": str(run_result["paths"]["task_completion"]),
        "run_digest": str(run_result["paths"]["run_digest"]),
        "harness_manifest": str(out_dir / "harness_manifest.json"),
        "harness_result": str(out_dir / "harness_result.json"),
        "harness_runner": harness_result["runner"],
        "environment": _environment_summary(flight_root),
    }


def _patch_config(env: dict[str, str], out_dir: Path, workspace: Path, port: int, gateway_port: int) -> None:
    patch = {
        "agents": {
            "defaults": {
                "workspace": str(workspace),
                "model": {"primary": MODEL_REF},
                "models": {
                    MODEL_REF: {
                        "alias": "HFR OpenClaw mock",
                        "agentRuntime": {"id": "openclaw"},
                    }
                },
            }
        },
        "models": {
            "mode": "merge",
            "providers": {
                "hfrmock": {
                    "baseUrl": f"http://127.0.0.1:{port}/v1",
                    "apiKey": "hfr-local-key",
                    "api": "openai-completions",
                    "timeoutSeconds": 30,
                    "agentRuntime": {"id": "openclaw"},
                    "models": [
                        {
                            "id": MODEL_ID,
                            "name": "HFR OpenClaw mock",
                            "reasoning": False,
                            "input": ["text"],
                            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                            "contextWindow": 32768,
                            "maxTokens": 128,
                            "agentRuntime": {"id": "openclaw"},
                        }
                    ],
                }
            },
        },
        "gateway": {
            "mode": "local",
            "port": gateway_port,
            "bind": "loopback",
        },
        "plugins": {
            "entries": {
                "flight-recorder": {
                    "hooks": {"allowConversationAccess": True},
                }
            }
        },
    }
    patch_path = out_dir / "openclaw_config_patch.json"
    patch_path.write_text(json.dumps(patch, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _run_openclaw(["config", "patch", "--file", str(patch_path)], env, out_dir, "openclaw_config_patch", timeout=30, check=True)


def _start_gateway(env: dict[str, str], out_dir: Path, port: int) -> tuple[subprocess.Popen[str], Any, Any]:
    stdout = (out_dir / "openclaw_gateway_stdout.txt").open("w", encoding="utf-8")
    stderr = (out_dir / "openclaw_gateway_stderr.txt").open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            [
                "openclaw",
                "gateway",
                "run",
                "--force",
                "--port",
                str(port),
                "--raw-stream",
                "--raw-stream-path",
                str(out_dir / "openclaw_raw_stream.jsonl"),
            ],
            env=env,
            text=True,
            stdout=stdout,
            stderr=stderr,
        )
        return process, stdout, stderr
    except Exception:
        stdout.close()
        stderr.close()
        raise


def _stop_gateway(gateway: tuple[subprocess.Popen[str], Any, Any]) -> None:
    process, stdout, stderr = gateway
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    stdout.close()
    stderr.close()


def _wait_for_gateway(env: dict[str, str], out_dir: Path, *, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    last = None
    while time.monotonic() < deadline:
        last = _run_openclaw(["health", "--json", "--timeout", "2500"], env, out_dir, "openclaw_gateway_health", timeout=8)
        if last.returncode == 0:
            return
        time.sleep(1)
    code = last.returncode if last is not None else "unknown"
    raise SystemExit(f"OpenClaw gateway did not become healthy (last exit {code}); see {out_dir}")


def _write_smoke_artifacts(openclaw_trace: Path, out_dir: Path) -> dict[str, Any]:
    scenario_path = out_dir / "live_openclaw_scenario.json"
    _write_json(scenario_path, _scenario(Path(openclaw_trace.name)))
    return _run_scenario_artifacts(
        scenario_path,
        out_dir,
        trace_format="openclaw_jsonl",
        preserve_paths=True,
    )


def _scenario(trace_path: Path) -> dict[str, Any]:
    return {
        "id": "live_openclaw_smoke",
        "title": "Live OpenClaw Smoke",
        "prompt": PROMPT,
        "trace": {"format": "openclaw_jsonl", "path": str(trace_path)},
        "policy": {
            "secret_patterns": ["(?i)(api[_-]?key|secret|password)"],
            "max_api_calls": 6,
            "max_tool_calls": 0,
            "max_subagents": 0,
        },
        "assertions": {
            "required_actions": [
                {
                    "id": "assistant_answer_observed",
                    "description": "OpenClaw completed the live turn and the plugin captured the final answer.",
                    "event_type": "assistant_message",
                    "field_contains": {"text": EXPECTED},
                }
            ],
            "required_evidence": [
                {
                    "id": "agent_end_hook_observed",
                    "type": "event_matches",
                    "event_type": "assistant_message",
                    "field_equals": {"source_hook": "agent_end"},
                    "field_contains": {"text": EXPECTED},
                }
            ],
            "final_contains": [EXPECTED],
        },
        "scoring": {"pass_threshold": 90},
    }


def _run_openclaw(
    args: list[str],
    env: dict[str, str],
    out_dir: Path,
    label: str,
    *,
    timeout: int,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        ["openclaw", *args],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    (out_dir / f"{label}_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (out_dir / f"{label}_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if check and completed.returncode != 0:
        raise SystemExit(f"`openclaw {' '.join(args)}` failed with exit {completed.returncode}; see {out_dir}")
    return completed


def _write_smoke_summary(out_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    summary_path = out_dir / "live_openclaw_smoke_summary.json"
    summary = {**result, "summary": str(summary_path)}
    _write_json(summary_path, summary)
    return summary


def _combine_jsonl(paths: list[Path], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as handle:
        for path in paths:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    handle.write(line.rstrip() + "\n")


def _read_hooks(path: Path) -> list[str]:
    hooks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            hooks.append(str(json.loads(line).get("hook") or ""))
    return hooks


def _plugin_loaded(raw: str) -> bool:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return False
    for plugin in data.get("plugins", []):
        if plugin.get("id") == "flight-recorder" and plugin.get("enabled") and plugin.get("status") == "loaded":
            return True
    return False


def _runtime_hook_count(raw: str) -> int:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return 0
    plugin = data.get("plugin") if isinstance(data, dict) else None
    if isinstance(plugin, dict) and isinstance(plugin.get("hookCount"), int):
        return int(plugin["hookCount"])
    return len(data.get("typedHooks") or []) if isinstance(data, dict) else 0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request_summary(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages")
    tools = payload.get("tools")
    return {
        "path": path,
        "model": payload.get("model"),
        "stream": payload.get("stream"),
        "message_count": len(messages) if isinstance(messages, list) else 0,
        "tool_count": len(tools) if isinstance(tools, list) else 0,
        "request_keys": sorted(str(key) for key in payload),
    }


def _completion_payload() -> dict[str, Any]:
    return {
        "id": "chatcmpl-hfr-openclaw",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": EXPECTED},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _environment_summary(flight_root: Path) -> dict[str, Any]:
    flight_git = _git_info(flight_root)
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "openclaw_version": _command_output("openclaw", "--version") or "unknown",
        "flight_recorder_root": str(flight_root),
        "flight_recorder_git_commit": flight_git["commit"],
        "flight_recorder_git_dirty": flight_git["dirty"],
    }


def _git_info(root: Path) -> dict[str, Any]:
    commit = _command_output("git", "-C", str(root), "rev-parse", "--short=12", "HEAD")
    status = _command_output("git", "-C", str(root), "status", "--porcelain")
    return {"commit": commit or "unknown", "dirty": bool(status) if status is not None else None}


def _command_output(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            list(args),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
