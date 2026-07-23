"""Auditable loopback alias proxy for Tau natural-language assertion judging.

Tau pins the NL-assertion evaluator model name in its own config.  Local
OpenAI-compatible servers such as MLX can route by the request model, so the
pinned OpenAI model name must be rewritten to the locally served teacher model
without modifying Tau's repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

MAX_BODY_BYTES = 16 * 1024 * 1024
DEFAULT_REQUEST_MODEL = "gpt-4.1-2025-04-14"


class Tau3NLJudgeAliasProxyError(ValueError):
    """Raised when proxy configuration or a request is unsafe."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _append_audit(path: Path, record: dict[str, Any], lock: threading.Lock) -> None:
    line = _canonical_bytes(record) + b"\n"
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, line)
        finally:
            os.close(fd)


def _model_sha256(model: str) -> str:
    return hashlib.sha256(model.encode("utf-8")).hexdigest()


def build_server(
    *,
    upstream_base_url: str,
    host: str,
    port: int,
    audit_log: str | Path,
    request_model: str = DEFAULT_REQUEST_MODEL,
    served_model: str,
) -> ThreadingHTTPServer:
    """Construct a loopback-only OpenAI-compatible model-alias proxy."""

    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise Tau3NLJudgeAliasProxyError("proxy host must be loopback")
    parsed = urlparse(upstream_base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise Tau3NLJudgeAliasProxyError("upstream must be an HTTP loopback endpoint")
    if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
        raise Tau3NLJudgeAliasProxyError("upstream endpoint must not contain credentials, query, or fragment")
    if not request_model:
        raise Tau3NLJudgeAliasProxyError("request model must be non-empty")
    if not served_model:
        raise Tau3NLJudgeAliasProxyError("served model must be non-empty")

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
            path = self.path.split("?", 1)[0].rstrip("/")
            if path.endswith("/models"):
                body = _canonical_bytes(
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": request_model,
                                "object": "model",
                                "created": int(time.time()),
                            }
                        ],
                    }
                )
                self._relay(200, {"content-type": "application/json"}, body)
                return
            self._json_error(404, "unsupported judge proxy route")

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("content-length") or 0)
            if length < 0 or length > MAX_BODY_BYTES:
                self._json_error(413, "request body exceeds proxy limit")
                return
            body = self.rfile.read(length) if length else b""
            path = self.path.split("?", 1)[0].rstrip("/")
            if not path.endswith("/chat/completions"):
                self._json_error(404, "unsupported judge proxy route")
                return
            try:
                payload = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._json_error(400, "chat request body must be valid JSON")
                return
            if not isinstance(payload, dict):
                self._json_error(400, "chat request body must be a JSON object")
                return
            if payload.get("model") != request_model:
                self._json_error(400, "unexpected judge request model")
                return
            if payload.get("stream") is True:
                self._json_error(400, "streaming judge requests are not supported")
                return
            request_sha256 = hashlib.sha256(body).hexdigest()
            rewritten_payload = dict(payload)
            rewritten_payload["model"] = served_model
            rewritten_body = _canonical_bytes(rewritten_payload)
            status, headers, response_body = self._forward_chat(rewritten_body)
            _append_audit(
                audit_path,
                {
                    "schema_version": "hfr.tau3_nl_judge_alias_proxy.v1",
                    "created_at_unix": int(time.time()),
                    "request_sha256": request_sha256,
                    "rewritten_request_sha256": hashlib.sha256(rewritten_body).hexdigest(),
                    "response_sha256": hashlib.sha256(response_body).hexdigest(),
                    "status": status,
                    "request_model_sha256": _model_sha256(request_model),
                    "served_model_sha256": _model_sha256(served_model),
                    "payload_recorded": False,
                },
                audit_lock,
            )
            self._relay(status, headers, response_body)

        def _forward_chat(self, body: bytes) -> tuple[int, Any, bytes]:
            path = self.path
            if upstream_prefix and not path.startswith(upstream_prefix + "/") and path != upstream_prefix:
                path = upstream_prefix + (path if path.startswith("/") else "/" + path)
            request = urllib.request.Request(
                upstream_origin + path,
                data=body,
                headers={
                    "content-type": "application/json",
                    "accept": "application/json",
                    "authorization": "Bearer local",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=600) as response:
                    return response.status, response.headers, response.read()
            except urllib.error.HTTPError as exc:
                return exc.code, exc.headers, exc.read()
            except (OSError, urllib.error.URLError) as exc:
                return 502, {"content-type": "application/json"}, _canonical_bytes({"error": f"upstream request failed: {exc}"})

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
    parser.add_argument("--port", type=int, default=18085)
    parser.add_argument("--audit-log", type=Path, required=True)
    parser.add_argument("--request-model", default=DEFAULT_REQUEST_MODEL)
    parser.add_argument("--served-model", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        server = build_server(
            upstream_base_url=args.upstream,
            host=args.host,
            port=args.port,
            audit_log=args.audit_log,
            request_model=args.request_model,
            served_model=args.served_model,
        )
    except (OSError, Tau3NLJudgeAliasProxyError, ValueError) as exc:
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
