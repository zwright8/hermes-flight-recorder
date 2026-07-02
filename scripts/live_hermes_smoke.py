#!/usr/bin/env python3
"""Run a live Hermes observer smoke test against a local mock model server.

This script proves the Flight Recorder observer adapter can be loaded by a real
Hermes runtime session without requiring external API keys. It creates an
isolated HERMES_HOME, installs a temporary user plugin wrapper, starts a local
OpenAI-compatible streaming endpoint, runs `uv run hermes chat`, then normalizes,
scores, reports, and records lineage for the captured observer JSONL.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
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
from flightrecorder.hermes_plugin import HOOKS, LIVE_SMOKE_SUMMARY_SCHEMA_VERSION
from flightrecorder.schema import ScenarioError
from scripts.hermes_harness import publish_harness_artifacts, write_fake_secret_canaries


PROMPT = "Reply exactly: flight recorder live smoke ok"
EXPECTED = "flight recorder live smoke ok"


class MockChatHandler(BaseHTTPRequestHandler):
    """Tiny OpenAI-compatible streaming chat/completions endpoint."""

    protocol_version = "HTTP/1.1"
    requests: list[dict[str, Any]] = []

    def log_message(self, _fmt: str, *_args: Any) -> None:
        return None

    def do_GET(self) -> None:
        self.requests.append(_request_summary(self.path, {}))
        path = self.path.split("?", 1)[0].rstrip("/")
        if path == "/api/show":
            self._send_json(_model_details())
            return
        if path in {"/api/v1/models", "/v1/models", "/models"}:
            self._send_json({"object": "list", "data": [{"id": "hfr-mock", "object": "model"}]})
            return
        if path == "/v1/models/hfr-mock":
            self._send_json({"id": "hfr-mock", "object": "model"})
            return
        if path == "/api/tags":
            self._send_json({"models": [{"name": "hfr-mock", "model": "hfr-mock"}]})
            return
        if path in {"/v1/props", "/props"}:
            self._send_json({"model": "hfr-mock", "context_length": 32768})
            return
        if path == "/version":
            self._send_json({"version": "hfr-mock"})
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
        if path == "/api/show":
            self._send_json(_model_details())
            return

        if path != "/v1/chat/completions":
            self.send_response(404)
            self.send_header("connection", "close")
            self.end_headers()
            return

        created = int(time.time())
        chunks = [
            {
                "id": "chatcmpl-hfr-live",
                "object": "chat.completion.chunk",
                "created": created,
                "model": "hfr-mock",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": EXPECTED},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-hfr-live",
                "object": "chat.completion.chunk",
                "created": created,
                "model": "hfr-mock",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        ]
        text = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        text += "data: [DONE]\n\n"
        raw = text.encode("utf-8")

        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "close")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_json(self, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("connection", "close")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a live Hermes Flight Recorder observer smoke test")
    parser.add_argument(
        "--hermes-root",
        default=os.environ.get("HERMES_AGENT_ROOT") or _default_hermes_root(),
        help="Path to a hermes-agent source checkout",
    )
    parser.add_argument(
        "--out",
        default="live_smoke_artifacts/latest",
        help="Directory for smoke artifacts",
    )
    parser.add_argument("--keep-temp", action="store_true", help="Keep the isolated temporary HERMES_HOME")
    args = parser.parse_args(argv)

    hermes_root = Path(args.hermes_root).expanduser().resolve()
    if not (hermes_root / "pyproject.toml").exists():
        raise SystemExit(f"Hermes checkout not found: {hermes_root}")
    if shutil.which("uv") is None:
        raise SystemExit("uv is required to run the Hermes checkout")

    out_dir = Path(args.out).expanduser().resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    MockChatHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockChatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    temp_root_obj = tempfile.TemporaryDirectory(prefix="hfr-live-hermes-")
    temp_root = Path(temp_root_obj.name)
    try:
        result = _run_live_session(
            hermes_root,
            Path(__file__).resolve().parents[1],
            out_dir,
            temp_root,
            server.server_address[1],
        )
    finally:
        server.shutdown()
        server.server_close()
        if args.keep_temp:
            print(f"kept temp root: {temp_root}")
        else:
            temp_root_obj.cleanup()

    result = _write_smoke_summary(out_dir, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


def _run_live_session(hermes_root: Path, flight_root: Path, out_dir: Path, temp_root: Path, port: int) -> dict[str, Any]:
    hermes_home = temp_root / "hermes-home"
    events_dir = temp_root / "events"
    home_dir = temp_root / "home"
    plugin_dir = hermes_home / "plugins" / "flight_recorder_live"
    plugin_dir.mkdir(parents=True)
    events_dir.mkdir(parents=True)
    home_dir.mkdir(parents=True)
    fake_secret_files = write_fake_secret_canaries(home_dir)

    _write_plugin(plugin_dir)
    _write_config(hermes_home / "config.yaml", port)

    env = os.environ.copy()
    env.update(
        {
            "HERMES_HOME": str(hermes_home),
            "HOME": str(home_dir),
            "HERMES_FLIGHT_RECORDER_OUTPUT_DIR": str(events_dir),
            "HERMES_FLIGHT_RECORDER_MAX_FIELD_CHARS": "20000",
            "HERMES_API_TIMEOUT": "30",
            "HERMES_STREAM_READ_TIMEOUT": "30",
            "HERMES_STREAM_RETRIES": "0",
            "HERMES_DISABLE_UPDATE_CHECK": "1",
            "PYTHONPATH": f"{flight_root}:{hermes_root}:{env.get('PYTHONPATH', '')}",
        }
    )
    cmd = [
        "uv",
        "run",
        "hermes",
        "chat",
        "--query",
        PROMPT,
        "--provider",
        "custom",
        "--model",
        "hfr-mock",
        "--quiet",
        "--ignore-rules",
        "--max-turns",
        "2",
    ]
    completed = subprocess.run(
        cmd,
        cwd=hermes_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=90,
    )
    (out_dir / "hermes_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (out_dir / "hermes_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    (out_dir / "mock_requests.json").write_text(
        json.dumps(MockChatHandler.requests, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    observer_files = sorted(events_dir.glob("*.observer.jsonl"))
    if completed.returncode != 0:
        raise SystemExit(f"Hermes live smoke failed with exit {completed.returncode}; see {out_dir}")
    if not observer_files:
        raise SystemExit(f"Hermes live smoke produced no observer JSONL; see {out_dir}")

    observer_path = out_dir / "live_observer.jsonl"
    shutil.copyfile(observer_files[0], observer_path)
    run_result = _write_smoke_artifacts(observer_path, out_dir)
    harness_result = publish_harness_artifacts(
        scenario_path=out_dir / "live_scenario.json",
        run_dir=out_dir,
        artifact_result=run_result,
        trace_path=observer_path,
        trace_format="observer_jsonl",
        runner="hermes_live_smoke",
        provider="custom",
        model="hfr-mock",
        base_url=f"http://127.0.0.1:{port}/v1",
        sandbox={
            "root": temp_root,
            "home": home_dir,
            "workspace": hermes_root,
            "events": events_dir,
            "ephemeral": True,
            "audit_artifacts_kept": True,
        },
        fake_secret_files=fake_secret_files,
        process={
            "exit_code": completed.returncode,
            "stdout": str(out_dir / "hermes_stdout.txt"),
            "stderr": str(out_dir / "hermes_stderr.txt"),
        },
        metadata={"source": "scripts/live_hermes_smoke.py", "mock_endpoint": True},
    )
    scorecard = run_result["scorecard"]
    report_path = run_result["paths"]["report"]

    hook_names = []
    for line in observer_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            hook_names.append(json.loads(line)["hook"])
    required_hooks = {"on_session_start", "pre_llm_call", "pre_api_request", "post_api_request", "post_llm_call"}
    missing_hooks = sorted(required_hooks - set(hook_names))
    chat_requests = [request for request in MockChatHandler.requests if request["path"].rstrip("/") == "/v1/chat/completions"]
    passed = completed.returncode == 0 and scorecard["passed"] and not missing_hooks and bool(chat_requests)
    return {
        "passed": passed,
        "hermes_exit_code": completed.returncode,
        "mock_request_count": len(MockChatHandler.requests),
        "chat_completion_request_count": len(chat_requests),
        "observer_file": str(observer_path),
        "hooks": hook_names,
        "missing_hooks": missing_hooks,
        "score": scorecard["score"],
        "report": str(report_path),
        "lineage": str(run_result["paths"]["lineage"]),
        "task_completion": str(run_result["paths"]["task_completion"]),
        "run_digest": str(run_result["paths"]["run_digest"]),
        "harness_manifest": str(out_dir / "harness_manifest.json"),
        "harness_result": str(out_dir / "harness_result.json"),
        "harness_runner": harness_result["runner"],
        "environment": _environment_summary(hermes_root, flight_root),
    }


def _write_smoke_artifacts(observer_path: Path, out_dir: Path) -> dict[str, Any]:
    """Write normal Flight Recorder artifacts for a captured live observer JSONL."""
    scenario_path = out_dir / "live_scenario.json"
    trace_ref = Path(observer_path.name) if observer_path.parent == out_dir else observer_path
    _write_json(scenario_path, _scenario(trace_ref))
    return _run_scenario_artifacts(
        scenario_path,
        out_dir,
        trace_format="observer_jsonl",
        preserve_paths=True,
    )


def _write_smoke_summary(out_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    """Write a stable machine-readable live smoke summary."""
    summary_path = out_dir / "live_smoke_summary.json"
    summary = {
        "schema_version": LIVE_SMOKE_SUMMARY_SCHEMA_VERSION,
        **result,
        "summary": str(summary_path),
    }
    _write_json(summary_path, summary)
    return summary


def _write_plugin(plugin_dir: Path) -> None:
    (plugin_dir / "plugin.yaml").write_text(
        "\n".join(
            [
                "name: flight_recorder_live",
                'version: "0.1"',
                "kind: standalone",
                "description: Flight Recorder live smoke plugin",
                "provides_hooks:",
                *[f"  - {hook}" for hook in HOOKS],
                "",
            ]
        ),
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        "from flightrecorder.hermes_plugin import register as register_flight_recorder\n\n"
        "def register(ctx):\n"
        "    return register_flight_recorder(ctx)\n",
        encoding="utf-8",
    )


def _write_config(path: Path, port: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""model:
  provider: custom
  default: hfr-mock
  base_url: http://127.0.0.1:{port}/v1
  api_key: hfr-local-key
  api_mode: chat_completions
plugins:
  enabled:
    - flight_recorder_live
agent:
  max_turns: 2
""",
        encoding="utf-8",
    )


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


def _model_details() -> dict[str, Any]:
    return {
        "model": "hfr-mock",
        "details": {"family": "mock"},
        "model_info": {},
        "capabilities": ["completion"],
    }


def _scenario(observer_path: Path) -> dict[str, Any]:
    return {
        "id": "live_hermes_observer_smoke",
        "title": "Live Hermes Observer Smoke",
        "prompt": PROMPT,
        "trace": {"format": "observer_jsonl", "path": str(observer_path)},
        "policy": {
            "secret_patterns": ["(?i)(api[_-]?key|secret|token|password)"],
            "max_api_calls": 3,
        },
        "assertions": {
            "required_actions": [
                {
                    "id": "assistant_answer_observed",
                    "description": "Hermes completed the live turn and observer captured the final answer",
                    "event_type": "assistant_message",
                    "contains": EXPECTED,
                }
            ],
            "required_evidence": [
                {
                    "id": "api_response_hook_observed",
                    "type": "event_matches",
                    "event_type": "api_call",
                    "field": "text",
                    "contains": "post_api_request",
                }
            ],
            "final_contains": [EXPECTED],
        },
        "scoring": {"pass_threshold": 90},
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _environment_summary(hermes_root: Path, flight_root: Path) -> dict[str, Any]:
    """Return compact provenance for comparing smoke results across checkouts."""
    hermes_git = _git_info(hermes_root)
    flight_git = _git_info(flight_root)
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "hermes_root": str(hermes_root),
        "hermes_git_commit": hermes_git["commit"],
        "hermes_git_dirty": hermes_git["dirty"],
        "flight_recorder_root": str(flight_root),
        "flight_recorder_git_commit": flight_git["commit"],
        "flight_recorder_git_dirty": flight_git["dirty"],
    }


def _git_info(root: Path) -> dict[str, Any]:
    """Return best-effort git provenance without failing the live smoke."""
    commit = _git_output(root, "rev-parse", "--short=12", "HEAD")
    status = _git_output(root, "status", "--porcelain")
    return {
        "commit": commit or "unknown",
        "dirty": bool(status) if status is not None else None,
    }


def _git_output(root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
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


def _default_hermes_root() -> str:
    candidate = Path(__file__).resolve().parents[2] / "upstream-hermes-agent"
    return str(candidate)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ScenarioError as exc:
        raise SystemExit(str(exc)) from exc
