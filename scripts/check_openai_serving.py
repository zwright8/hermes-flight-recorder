#!/usr/bin/env python3
"""Check an OpenAI-compatible serving endpoint and write serving artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


SCHEMA_PROFILE = "hfr.serving_profile.v1"
SCHEMA_COMPATIBILITY = "hfr.serving_compatibility_report.v1"
SCHEMA_CHECK = "hfr.serving_endpoint_check.v1"
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--provider", default="custom")
    parser.add_argument("--arm", default="candidate")
    parser.add_argument("--engine", choices=["openai_compatible", "sglang", "transformers", "vllm"], default="transformers")
    parser.add_argument("--adapter", default="", help="Optional adapter path or id")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--api-key-env", default="HERMES_EVAL_API_KEY")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--mock-response", help="Start a managed mock endpoint returning this text")
    parser.add_argument("--require-tool-call", action="store_true")
    parser.add_argument("--require-structured-output", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mock_server: ThreadingHTTPServer | None = None
    mock_requests: list[dict[str, Any]] = []
    base_url = args.base_url
    if args.mock_response is not None:
        mock_server, mock_requests, base_url = _start_mock_server(args.mock_response, args.model, args.adapter)
    if not base_url:
        raise SystemExit("--base-url is required unless --mock-response is used")

    try:
        profile, compatibility, report = check_endpoint(
            base_url=str(base_url),
            model=args.model,
            provider=args.provider,
            arm=args.arm,
            engine=args.engine,
            adapter=args.adapter,
            api_key=_api_key(args, str(base_url)),
            timeout=float(args.timeout),
            out_dir=out_dir,
            require_tool_call=bool(args.require_tool_call),
            require_structured_output=bool(args.require_structured_output),
        )
    finally:
        if mock_server is not None:
            mock_server.shutdown()
            mock_server.server_close()

    _write_json(out_dir / "serving_profile.json", profile)
    _write_json(out_dir / "compatibility_report.json", compatibility)
    _write_json(out_dir / "serving_check.json", report)
    if mock_requests:
        _write_json(out_dir / "mock_requests.json", {"requests": mock_requests})
    print(json.dumps({"passed": report["passed"], "failed_checks": report["failed_checks"], "out": str(out_dir)}, indent=2))
    return 0 if report["passed"] else 1


def check_endpoint(
    *,
    base_url: str,
    model: str,
    provider: str,
    arm: str,
    engine: str,
    adapter: str,
    api_key: str,
    timeout: float,
    out_dir: Path,
    require_tool_call: bool,
    require_structured_output: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    generated_at = _utc_now()
    checks: list[dict[str, Any]] = []

    health = _get_first_json([_root_url(base_url, "/healthz"), _root_url(base_url, "/health"), _root_url(base_url, "/version")], api_key=api_key, timeout=timeout)
    checks.append(_check("health", health["ok"], health))

    models = _request_json("GET", _openai_url(base_url, "/models"), api_key=api_key, timeout=timeout)
    observed_model_ids = _model_ids(models.get("json")) if models["ok"] else []
    checks.append(_check("models", bool(models["ok"] and observed_model_ids), models))

    metadata = _get_first_json([_openai_url(base_url, f"/models/{urllib.parse.quote(model, safe='')}"), _openai_url(base_url, "/props"), _root_url(base_url, "/api/show")], api_key=api_key, timeout=timeout)
    metadata_model = _metadata_model_id(metadata.get("json"))
    checks.append(_check("model_metadata", bool(metadata["ok"] and metadata_model), metadata))

    chat = _request_json(
        "POST",
        _openai_url(base_url, "/chat/completions"),
        payload={"model": model, "messages": [{"role": "user", "content": "Reply with: hfr serving smoke ok"}], "temperature": 0, "max_tokens": 32, "stream": False},
        api_key=api_key,
        timeout=timeout,
    )
    chat_text = _chat_text(chat.get("json"))
    chat_model = _chat_model(chat.get("json"))
    checks.append(_check("chat_completion", bool(chat["ok"] and chat_text), {**chat, "text": chat_text, "response_model": chat_model}))

    tool_check = _tool_call_check(base_url, model, api_key=api_key, timeout=timeout)
    structured_check = _structured_output_check(base_url, model, api_key=api_key, timeout=timeout)
    if require_tool_call:
        checks.append(_check("tool_call_required", tool_check["status"] == "supported", tool_check))
    if require_structured_output:
        checks.append(_check("structured_output_required", structured_check["status"] == "supported", structured_check))

    served_model_id = _first_non_empty(chat_model, metadata_model, observed_model_ids[0] if observed_model_ids else "", model)
    identity_match = _identity_matches(model, adapter, [served_model_id, metadata_model, chat_model, *observed_model_ids])
    checks.append(_check("model_identity", identity_match, {"expected": model, "served_model_id": served_model_id, "observed_model_ids": observed_model_ids, "metadata_model": metadata_model, "chat_response_model": chat_model}))

    failed_checks = [item["id"] for item in checks if not item["passed"]]
    passed = not failed_checks
    artifacts = _artifact_refs(out_dir)
    compatibility = {
        "schema_version": SCHEMA_COMPATIBILITY,
        "generated_at": generated_at,
        "profile_id": _profile_id(arm, engine, model, adapter),
        "model": model,
        "served_model_id": served_model_id,
        "engine": engine,
        "checks": {
            "openai_core": {item["id"]: item["passed"] for item in checks if item["id"] in {"health", "models", "model_metadata", "chat_completion"}},
            "tool_calls": tool_check,
            "structured_outputs": structured_check,
        },
    }
    profile = {
        "schema_version": SCHEMA_PROFILE,
        "generated_at": generated_at,
        "profile_id": compatibility["profile_id"],
        "arm": arm,
        "provider": provider,
        "engine": engine,
        "endpoint": {"base_url": _normalize_base_url(base_url), "models_url": _openai_url(base_url, "/models"), "chat_completions_url": _openai_url(base_url, "/chat/completions")},
        "model_identity": {"requested_model": model, "served_model_id": served_model_id, "observed_model_ids": observed_model_ids, "metadata_model": metadata_model, "chat_response_model": chat_model, "adapter": _adapter_identity(adapter)},
        "capabilities": {
            "health": _check_passed(checks, "health"),
            "models": _check_passed(checks, "models"),
            "model_metadata": _check_passed(checks, "model_metadata"),
            "chat_completions": _check_passed(checks, "chat_completion"),
            "streaming": "not_checked",
            "tool_calls": tool_check["status"],
            "structured_outputs": structured_check["status"],
        },
        "artifacts": artifacts,
        "eval_preflight": {"ready": passed, "readiness": "ready" if passed else "blocked", "failed_checks": failed_checks},
        "environment": {"python_version": platform.python_version(), "platform": platform.platform()},
    }
    report = {
        "schema_version": SCHEMA_CHECK,
        "generated_at": generated_at,
        "passed": passed,
        "readiness": "ready" if passed else "blocked",
        "profile_id": compatibility["profile_id"],
        "arm": arm,
        "model": model,
        "served_model_id": served_model_id,
        "base_url": _normalize_base_url(base_url),
        "checks": checks,
        "failed_checks": failed_checks,
        "artifacts": artifacts,
    }
    return profile, compatibility, report


def _tool_call_check(base_url: str, model: str, *, api_key: str, timeout: float) -> dict[str, Any]:
    response = _request_json(
        "POST",
        _openai_url(base_url, "/chat/completions"),
        payload={"model": model, "messages": [{"role": "user", "content": "Call read_file for demo.txt."}], "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}], "tool_choice": "auto", "temperature": 0, "max_tokens": 64},
        api_key=api_key,
        timeout=timeout,
    )
    tool_calls = _tool_calls(response.get("json"))
    return {"status": "supported" if response["ok"] and tool_calls else "not_verified", "response_ok": response["ok"], "tool_call_count": len(tool_calls), "tool_calls": tool_calls, "error": response.get("error")}


def _structured_output_check(base_url: str, model: str, *, api_key: str, timeout: float) -> dict[str, Any]:
    response = _request_json(
        "POST",
        _openai_url(base_url, "/chat/completions"),
        payload={"model": model, "messages": [{"role": "user", "content": "Return JSON with keys status and evidence."}], "response_format": {"type": "json_object"}, "temperature": 0, "max_tokens": 64},
        api_key=api_key,
        timeout=timeout,
    )
    text = _chat_text(response.get("json"))
    parsed = None
    if text:
        try:
            value = json.loads(text)
            parsed = value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            parsed = None
    return {"status": "supported" if response["ok"] and isinstance(parsed, dict) else "not_verified", "response_ok": response["ok"], "json_parse_passed": isinstance(parsed, dict), "parsed": parsed, "text": text, "error": response.get("error")}


def _request_json(method: str, url: str, *, payload: dict[str, Any] | None = None, api_key: str, timeout: float) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, method=method, data=data)
    request.add_header("accept", "application/json")
    if data is not None:
        request.add_header("content-type", "application/json")
    if api_key:
        request.add_header("authorization", f"Bearer {api_key}")
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8", "replace")
            try:
                parsed = json.loads(text) if text else {}
            except json.JSONDecodeError:
                parsed = {"_raw": text}
            return {"ok": 200 <= int(response.status) < 300, "status_code": int(response.status), "url": url, "elapsed_ms": int((time.time() - started) * 1000), "json": parsed}
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status_code": int(exc.code), "url": url, "elapsed_ms": int((time.time() - started) * 1000), "error": exc.read().decode("utf-8", "replace")}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {"ok": False, "status_code": None, "url": url, "elapsed_ms": int((time.time() - started) * 1000), "error": str(exc)}


def _get_first_json(urls: list[str], *, api_key: str, timeout: float) -> dict[str, Any]:
    attempts = []
    for url in urls:
        result = _request_json("GET", url, api_key=api_key, timeout=timeout)
        attempts.append(result)
        if result["ok"]:
            return {**result, "attempts": attempts}
    return {**attempts[-1], "attempts": attempts}


def _start_mock_server(response: str, model: str, adapter: str) -> tuple[ThreadingHTTPServer, list[dict[str, Any]], str]:
    requests: list[dict[str, Any]] = []
    served_model = f"{model}+{Path(adapter).name}" if adapter else model

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, _fmt: str, *_args: Any) -> None:
            return None

        def do_GET(self) -> None:
            requests.append({"method": "GET", "path": self.path})
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path in {"/healthz", "/health"}:
                _send_handler_json(self, {"ok": True, "model": served_model})
            elif path == "/version":
                _send_handler_json(self, {"version": "hfr-mock"})
            elif path in {"/v1/models", "/models"}:
                _send_handler_json(self, {"object": "list", "data": [{"id": served_model, "object": "model"}]})
            elif path.startswith("/v1/models/") or path in {"/v1/props", "/props"}:
                _send_handler_json(self, {"id": served_model, "model": served_model, "object": "model"})
            else:
                _send_handler_json(self, {"error": {"message": "not found"}}, status=404)

        def do_POST(self) -> None:
            length = int(self.headers.get("content-length") or 0)
            body = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                payload = {}
            requests.append({"method": "POST", "path": self.path, "payload": payload})
            if self.path.split("?", 1)[0].rstrip("/") != "/v1/chat/completions":
                _send_handler_json(self, {"error": {"message": "not found"}}, status=404)
                return
            if payload.get("tools"):
                message = {"role": "assistant", "content": None, "tool_calls": [{"id": "call_hfr_mock", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": "demo.txt"})}}]}
                finish_reason = "tool_calls"
            elif payload.get("response_format"):
                message = {"role": "assistant", "content": json.dumps({"status": "ok", "evidence": "hfr-structured-smoke"}, sort_keys=True)}
                finish_reason = "stop"
            else:
                message = {"role": "assistant", "content": response}
                finish_reason = "stop"
            _send_handler_json(self, {"id": "chatcmpl-hfr-mock", "object": "chat.completion", "created": int(time.time()), "model": served_model, "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}], "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12}})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, requests, f"http://127.0.0.1:{server.server_address[1]}/v1"


def _adapter_identity(adapter: str) -> dict[str, Any]:
    if not adapter:
        return {"present": False, "id": "", "path": "", "local": False}
    path = Path(adapter).expanduser()
    if not path.exists():
        return {"present": True, "id": adapter, "path": adapter, "local": False}
    files = []
    for name in ("adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json", "chat_template.jinja"):
        candidate = path / name
        if candidate.exists() and candidate.is_file():
            files.append({"path": str(candidate), "sha256": _sha256_file(candidate), "bytes": candidate.stat().st_size})
    return {"present": True, "id": path.name, "path": str(path.resolve()), "local": True, "files": files}


def _identity_matches(expected_model: str, adapter: str, observed_values: list[str]) -> bool:
    values = [str(value) for value in observed_values if value]
    if expected_model in values:
        return True
    if adapter:
        return f"{expected_model}+{Path(adapter).name}" in values
    return False


def _model_ids(payload: Any) -> list[str]:
    data = payload.get("data") if isinstance(payload, dict) else None
    return [str(item["id"]) for item in data if isinstance(item, dict) and item.get("id")] if isinstance(data, list) else []


def _metadata_model_id(payload: Any) -> str:
    return str((payload or {}).get("id") or (payload or {}).get("model") or "") if isinstance(payload, dict) else ""


def _chat_text(payload: Any) -> str:
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else {}
    return str(message.get("content") or "") if isinstance(message, dict) else ""


def _chat_model(payload: Any) -> str:
    return str(payload.get("model") or "") if isinstance(payload, dict) else ""


def _tool_calls(payload: Any) -> list[dict[str, Any]]:
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list) or not choices:
        return []
    message = choices[0].get("message") if isinstance(choices[0], dict) else {}
    calls = message.get("tool_calls") if isinstance(message, dict) else []
    return calls if isinstance(calls, list) else []


def _check(check_id: str, passed: bool, details: dict[str, Any]) -> dict[str, Any]:
    return {"id": check_id, "passed": bool(passed), "details": details}


def _check_passed(checks: list[dict[str, Any]], check_id: str) -> bool:
    return any(check["id"] == check_id and check["passed"] for check in checks)


def _api_key(args: argparse.Namespace, base_url: str) -> str:
    if args.api_key:
        return str(args.api_key)
    if args.api_key_env and os.environ.get(args.api_key_env):
        return str(os.environ[args.api_key_env])
    host = urllib.parse.urlparse(base_url).hostname or ""
    return "hfr-local-key" if host in {"127.0.0.1", "localhost", "::1"} else ""


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _root_url(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return f"{base}{path}"


def _openai_url(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    return f"{base}{path}" if base.endswith("/v1") else f"{base}/v1{path}"


def _first_non_empty(*values: str) -> str:
    return next((value for value in values if value), "")


def _profile_id(arm: str, engine: str, model: str, adapter: str) -> str:
    parts = [arm, engine, Path(adapter).name if adapter else model.rsplit("/", 1)[-1]]
    return "-".join("".join(char.lower() if char.isalnum() else "-" for char in part).strip("-") for part in parts if part)


def _artifact_refs(out_dir: Path) -> dict[str, str]:
    return {
        "serving_profile": str((out_dir / "serving_profile.json").relative_to(out_dir)),
        "compatibility_report": str((out_dir / "compatibility_report.json").relative_to(out_dir)),
        "serving_check": str((out_dir / "serving_check.json").relative_to(out_dir)),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _send_handler_json(handler: BaseHTTPRequestHandler, payload: dict[str, Any], *, status: int = 200) -> None:
    raw = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("content-type", "application/json")
    handler.send_header("connection", "close")
    handler.send_header("content-length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


if __name__ == "__main__":
    raise SystemExit(main())
