#!/usr/bin/env python3
"""Run a live Hermes observer smoke test against a local mock model server.

This script proves the Flight Recorder observer adapter can be loaded by a real
Hermes runtime session without requiring external API keys. It creates an
isolated HERMES_HOME, installs a temporary user plugin wrapper, starts a local
OpenAI-compatible streaming endpoint, runs `uv run hermes chat`, then normalizes,
scores, and reports the captured observer JSONL.
"""

from __future__ import annotations

import argparse
import json
import os
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

from flightrecorder.adapters import normalize_trace
from flightrecorder.hermes_plugin import HOOKS
from flightrecorder.report import write_report
from flightrecorder.schema import ScenarioError
from flightrecorder.scorers import score_trace


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
        result = _run_live_session(hermes_root, Path.cwd().resolve(), out_dir, temp_root, server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()
        if args.keep_temp:
            print(f"kept temp root: {temp_root}")
        else:
            temp_root_obj.cleanup()

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


def _run_live_session(hermes_root: Path, flight_root: Path, out_dir: Path, temp_root: Path, port: int) -> dict[str, Any]:
    hermes_home = temp_root / "hermes-home"
    events_dir = temp_root / "events"
    plugin_dir = hermes_home / "plugins" / "flight_recorder_live"
    plugin_dir.mkdir(parents=True)
    events_dir.mkdir(parents=True)

    _write_plugin(plugin_dir)
    _write_config(hermes_home / "config.yaml", port)

    env = os.environ.copy()
    env.update(
        {
            "HERMES_HOME": str(hermes_home),
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
    trace = normalize_trace(observer_path, "observer_jsonl")
    scenario = _scenario(observer_path)
    scorecard = score_trace(scenario, trace)
    _write_json(out_dir / "normalized_trace.json", trace)
    _write_json(out_dir / "scorecard.json", scorecard)
    write_report(scenario, trace, scorecard, out_dir / "report.html")

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
        "report": str(out_dir / "report.html"),
    }


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


def _default_hermes_root() -> str:
    candidate = Path(__file__).resolve().parents[2] / "upstream-hermes-agent"
    return str(candidate)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ScenarioError as exc:
        raise SystemExit(str(exc)) from exc
