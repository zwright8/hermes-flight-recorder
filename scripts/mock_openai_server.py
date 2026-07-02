#!/usr/bin/env python3
"""Run a tiny OpenAI-compatible mock server for serving lifecycle checks."""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model", default="hfr-mock-model")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--response", default="hfr serving smoke ok")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    server = ThreadingHTTPServer((args.host, int(args.port)), _handler(args.response, args.model, args.adapter))
    base_url = f"http://{server.server_address[0]}:{server.server_address[1]}/v1"
    print(json.dumps({"event": "mock_openai_server_ready", "base_url": base_url, "model": _served_model(args.model, args.adapter)}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


def _handler(response: str, model: str, adapter: str) -> type[BaseHTTPRequestHandler]:
    served_model = _served_model(model, adapter)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, _fmt: str, *_args: Any) -> None:
            return None

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path in {"/healthz", "/health"}:
                _send_json(self, {"ok": True, "model": served_model})
            elif path == "/version":
                _send_json(self, {"version": "hfr-mock", "model": served_model})
            elif path in {"/v1/models", "/models"}:
                _send_json(self, {"object": "list", "data": [{"id": served_model, "object": "model"}]})
            elif path.startswith("/v1/models/") or path in {"/v1/props", "/props"}:
                _send_json(self, {"id": served_model, "model": served_model, "object": "model"})
            else:
                _send_json(self, {"error": {"message": "not found"}}, status=404)

        def do_POST(self) -> None:
            length = int(self.headers.get("content-length") or 0)
            body = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                payload = {}
            if self.path.split("?", 1)[0].rstrip("/") != "/v1/chat/completions":
                _send_json(self, {"error": {"message": "not found"}}, status=404)
                return
            if payload.get("tools"):
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_hfr_mock",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": json.dumps({"path": "demo.txt"})},
                        }
                    ],
                }
                finish_reason = "tool_calls"
            elif payload.get("response_format"):
                message = {
                    "role": "assistant",
                    "content": json.dumps({"status": "ok", "evidence": "hfr-structured-smoke"}, sort_keys=True),
                }
                finish_reason = "stop"
            else:
                message = {"role": "assistant", "content": response}
                finish_reason = "stop"
            _send_json(
                self,
                {
                    "id": "chatcmpl-hfr-mock",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": served_model,
                    "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                    "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
                },
            )

    return Handler


def _served_model(model: str, adapter: str) -> str:
    return f"{model}+{Path(adapter).name}" if adapter else model


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
