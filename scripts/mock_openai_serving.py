#!/usr/bin/env python3
"""Run a tiny OpenAI-compatible mock server for serving lifecycle smoke tests."""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--model", default="hfr-mock-model")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--response", default="hfr serving smoke ok")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    served_model = f"{args.model}+{Path(args.adapter).name}" if args.adapter else args.model
    server = ThreadingHTTPServer((args.host, args.port), _handler(served_model, args.response))
    print(
        json.dumps(
            {
                "schema_version": "hfr.mock_openai_serving.v1",
                "base_url": f"http://{args.host}:{server.server_address[1]}/v1",
                "model": served_model,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


def _handler(served_model: str, response_text: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, _fmt: str, *_args: Any) -> None:
            return None

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path in {"/healthz", "/health"}:
                _send_json(self, {"ok": True, "model": served_model})
                return
            if path == "/version":
                _send_json(self, {"version": "hfr-mock-openai-serving"})
                return
            if path in {"/v1/models", "/api/v1/models", "/models"}:
                _send_json(self, {"object": "list", "data": [{"id": served_model, "object": "model"}]})
                return
            if path.startswith("/v1/models/") or path.startswith("/models/") or path.startswith("/api/v1/models/"):
                _send_json(self, {"id": served_model, "object": "model"})
                return
            if path in {"/v1/props", "/props", "/api/v1/props"}:
                _send_json(self, _model_details(served_model))
                return
            _send_json(self, {"error": {"message": f"not found: {path}"}}, status=404)

        def do_POST(self) -> None:
            length = int(self.headers.get("content-length") or 0)
            body = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                payload = {}
            path = self.path.split("?", 1)[0].rstrip("/")
            if path == "/api/show":
                _send_json(self, _model_details(served_model))
                return
            if path != "/v1/chat/completions":
                _send_json(self, {"error": {"message": f"not found: {path}"}}, status=404)
                return
            _send_json(self, _chat_response(payload, served_model, response_text))

    return Handler


def _chat_response(payload: dict[str, Any], served_model: str, response_text: str) -> dict[str, Any]:
    finish_reason = "stop"
    if payload.get("tools"):
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_hfr_mock_openai",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": json.dumps({"path": "demo.txt"}, sort_keys=True)},
                }
            ],
        }
        finish_reason = "tool_calls"
    elif payload.get("response_format"):
        message = {"role": "assistant", "content": json.dumps({"status": "ok", "evidence": "hfr-structured-smoke"}, sort_keys=True)}
    else:
        message = {"role": "assistant", "content": response_text}
    return {
        "id": "chatcmpl-hfr-mock-openai",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": served_model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
    }


def _model_details(served_model: str) -> dict[str, Any]:
    return {
        "id": served_model,
        "model": served_model,
        "object": "model",
        "context_length": 32768,
        "capabilities": ["chat_completions", "tools", "json_object"],
    }


def _send_json(handler: BaseHTTPRequestHandler, payload: dict[str, Any], *, status: int = 200) -> None:
    raw = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("content-type", "application/json")
    handler.send_header("connection", "close")
    handler.send_header("content-length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


if __name__ == "__main__":
    raise SystemExit(main())
