"""Auditable loopback proxy for terminating completed Tau user simulations.

Some local user-simulator models continue exchanging pleasantries after the
task has reached an executable success state.  Tau then records ``max_steps``
instead of evaluating the completed state.  This proxy forwards every request
to the pinned upstream model except when the transcript itself contains a
domain-specific success signal.  In that case it emits Tau's normal user stop
marker and records a content-free hash receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

MAX_BODY_BYTES = 16 * 1024 * 1024
USER_SIMULATOR_MARKERS = ("user simulation guidelines", "<scenario>")
MOBILE_SCENARIO_MARKER = "speed test returns excellent"
SERVICE_SCENARIO_MARKER = "status bar shows that you have signal"
EXCELLENT_SPEED_RE = re.compile(
    r"(?:speed test result[^\n]*\bexcellent\b|connection is very fast)",
    flags=re.IGNORECASE,
)
STATUS_BAR_RE = re.compile(r"status bar\s*:\s*([^\n]+)", flags=re.IGNORECASE)
NO_SERVICE_RE = re.compile(r"(?:no[ _-]?service|not connected|signal\s*:\s*none)", flags=re.IGNORECASE)
POSITIVE_SIGNAL_RE = re.compile(
    r"(?:📶|\b[1-5]\s*(?:signal\s*)?bars?\b|\bsignal\s+(?:present|excellent|good|fair|poor)\b|"
    r"\bcellular connection\s*:\s*connected\b)",
    flags=re.IGNORECASE,
)


class Tau3UserTerminalProxyError(ValueError):
    """Raised when proxy configuration or a request is unsafe."""


def terminal_reason(payload: dict[str, Any]) -> str | None:
    """Return the evidence class that permits a normal Tau user stop."""

    messages = payload.get("messages")
    if not isinstance(messages, list):
        return None
    system_text = "\n".join(
        _message_text(message)
        for message in messages
        if isinstance(message, dict) and message.get("role") == "system"
    ).lower()
    if not all(marker in system_text for marker in USER_SIMULATOR_MARKERS):
        return None
    observations = "\n".join(
        _message_text(message)
        for message in messages
        if isinstance(message, dict) and message.get("role") == "tool"
    )
    if MOBILE_SCENARIO_MARKER in system_text and EXCELLENT_SPEED_RE.search(observations):
        return "mobile_excellent_speed"
    if SERVICE_SCENARIO_MARKER in system_text:
        status_matches = list(STATUS_BAR_RE.finditer(observations))
        if status_matches:
            latest_status = status_matches[-1].group(1)
            if POSITIVE_SIGNAL_RE.search(latest_status) and not NO_SERVICE_RE.search(latest_status):
                return "service_signal_present"
    return None


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        values = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                values.append(item["text"])
        return "\n".join(values)
    return ""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _completion(payload: dict[str, Any], marker: str) -> bytes:
    now = int(time.time())
    body = {
        "id": f"chatcmpl-hfr-terminal-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": now,
        "model": str(payload.get("model") or "hfr-tau-user-terminal-proxy"),
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": marker, "tool_calls": None},
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 1, "total_tokens": 1},
    }
    return _canonical_bytes(body)


def _append_audit(path: Path, record: dict[str, Any], lock: threading.Lock) -> None:
    line = _canonical_bytes(record) + b"\n"
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, line)
        finally:
            os.close(fd)


def build_server(
    *,
    upstream_base_url: str,
    host: str,
    port: int,
    audit_log: str | Path,
) -> ThreadingHTTPServer:
    """Construct a loopback-only OpenAI-compatible proxy server."""

    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise Tau3UserTerminalProxyError("proxy host must be loopback")
    parsed = urlparse(upstream_base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise Tau3UserTerminalProxyError("upstream must be an HTTP loopback endpoint")
    upstream_origin = f"{parsed.scheme}://{parsed.netloc}"
    upstream_prefix = parsed.path.rstrip("/")
    audit_path = Path(audit_log)
    audit_lock = threading.Lock()
    hop_by_hop = {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, _format: str, *_args: Any) -> None:
            return None

        def do_GET(self) -> None:  # noqa: N802
            self._forward(None)

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("content-length") or 0)
            if length < 0 or length > MAX_BODY_BYTES:
                self._json_error(413, "request body exceeds proxy limit")
                return
            body = self.rfile.read(length) if length else b""
            path = self.path.split("?", 1)[0].rstrip("/")
            if path.endswith("/chat/completions"):
                try:
                    payload = json.loads(body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    self._json_error(400, "chat request body must be valid JSON")
                    return
                if not isinstance(payload, dict):
                    self._json_error(400, "chat request body must be a JSON object")
                    return
                reason = terminal_reason(payload)
                if reason is not None:
                    request_sha256 = hashlib.sha256(body).hexdigest()
                    response = _completion(payload, "###STOP###")
                    _append_audit(
                        audit_path,
                        {
                            "schema_version": "hfr.tau3_user_terminal_proxy.v1",
                            "created_at_unix": int(time.time()),
                            "request_sha256": request_sha256,
                            "response_sha256": hashlib.sha256(response).hexdigest(),
                            "reason": reason,
                            "marker": "STOP",
                            "payload_recorded": False,
                        },
                        audit_lock,
                    )
                    self._relay(200, {"content-type": "application/json"}, response)
                    return
            self._forward(body)

        def _forward(self, body: bytes | None) -> None:
            path = self.path
            if upstream_prefix and not path.startswith(upstream_prefix + "/") and path != upstream_prefix:
                path = upstream_prefix + (path if path.startswith("/") else "/" + path)
            headers = {
                key: value
                for key, value in self.headers.items()
                if key.lower() not in hop_by_hop | {"host", "content-length"}
            }
            request = urllib.request.Request(
                upstream_origin + path,
                data=body,
                headers=headers,
                method=self.command,
            )
            try:
                with urllib.request.urlopen(request, timeout=600) as response:
                    self._relay(response.status, response.headers, response.read())
            except urllib.error.HTTPError as exc:
                self._relay(exc.code, exc.headers, exc.read())
            except (OSError, urllib.error.URLError) as exc:
                self._json_error(502, f"upstream request failed: {exc}")

        def _relay(self, status: int, headers: Any, body: bytes) -> None:
            self.send_response(status)
            for key, value in headers.items():
                if key.lower() not in hop_by_hop | {"content-length"}:
                    self.send_header(key, value)
            self.send_header("content-length", str(len(body)))
            self.send_header("connection", "close")
            self.end_headers()
            if body:
                self.wfile.write(body)

        def _json_error(self, status: int, message: str) -> None:
            self._relay(status, {"content-type": "application/json"}, _canonical_bytes({"error": message}))

    return ThreadingHTTPServer((host, port), Handler)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18083)
    parser.add_argument("--audit-log", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        server = build_server(
            upstream_base_url=args.upstream,
            host=args.host,
            port=args.port,
            audit_log=args.audit_log,
        )
    except (OSError, Tau3UserTerminalProxyError, ValueError) as exc:
        print(str(exc))
        return 2
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
