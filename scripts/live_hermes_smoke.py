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
from flightrecorder.hermes_plugin import LIVE_SMOKE_SUMMARY_SCHEMA_VERSION
from flightrecorder.schema import ScenarioError
from scripts.hermes_harness import (
    default_hermes_root,
    environment_summary,
    hermes_chat_command,
    hermes_run_env,
    model_probe_payload,
    publish_harness_artifacts,
    require_hermes_checkout,
    send_json,
    send_stream,
    write_fake_secret_canaries,
    write_observer_plugin,
    write_runtime_config,
)


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
        payload = model_probe_payload(self.path, "hfr-mock", version="hfr-mock")
        if payload is not None:
            self._send_json(payload)
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
            self._send_json(model_probe_payload(path, "hfr-mock", version="hfr-mock") or {"id": "hfr-mock"})
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
        send_stream(self, chunks)

    def _send_json(self, payload: dict[str, Any]) -> None:
        send_json(self, payload)


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
    require_hermes_checkout(hermes_root)

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
    events_dir.mkdir(parents=True)
    home_dir.mkdir(parents=True)
    fake_secret_files = write_fake_secret_canaries(home_dir)

    write_observer_plugin(plugin_dir, description="Flight Recorder live smoke plugin")
    write_runtime_config(
        hermes_home / "config.yaml",
        provider="custom",
        model="hfr-mock",
        base_url=f"http://127.0.0.1:{port}/v1",
        api_key="hfr-local-key",
        max_turns=2,
    )

    env = hermes_run_env(
        flight_root=flight_root,
        hermes_root=hermes_root,
        hermes_home=hermes_home,
        home_dir=home_dir,
        events_dir=events_dir,
        timeout=30,
        max_field_chars=20000,
    )
    cmd = hermes_chat_command(
        hermes_root=hermes_root,
        prompt=PROMPT,
        provider="custom",
        model="hfr-mock",
        max_turns=2,
        source="flightrecorder-live-smoke",
    )
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
    return environment_summary(hermes_root, flight_root)


def _default_hermes_root() -> str:
    return default_hermes_root(__file__)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ScenarioError as exc:
        raise SystemExit(str(exc)) from exc
