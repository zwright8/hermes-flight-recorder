#!/usr/bin/env python3
# /// script
# dependencies = [
#   "accelerate",
#   "peft",
#   "torch",
#   "transformers",
# ]
# ///
"""Serve a local Transformers causal LM through a tiny OpenAI-compatible API.

This is intentionally small and single-process. It exists to run the Flight
Recorder held-out evaluator against the exact same local model family for the
baseline, trace-only adapter, and Flight Recorder adapter arms.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


class ModelBackend:
    def __init__(
        self,
        *,
        model_id: str,
        adapter: str,
        device: str,
        dtype_name: str,
        max_new_tokens: int,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.adapter = adapter
        self.device_name = _resolve_device(device)
        self.dtype = _resolve_dtype(dtype_name, self.device_name)
        self.max_new_tokens = max_new_tokens
        self._generate_lock = threading.Lock()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs: dict[str, Any] = {
            "dtype": self.dtype,
            "low_cpu_mem_usage": True,
        }
        if self.device_name == "cuda":
            kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)

        if self.device_name != "cuda":
            self.model = self.model.to(self.device_name)
        self.model.eval()
        self.torch = torch

    @property
    def served_model_id(self) -> str:
        if self.adapter:
            return f"{self.model_id}+{Path(self.adapter).name}"
        return self.model_id

    @property
    def model_device(self) -> Any:
        return next(self.model.parameters()).device

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages") or []
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")

        requested_max_tokens = int(payload.get("max_tokens") or payload.get("max_completion_tokens") or self.max_new_tokens)
        max_new_tokens = max(1, min(requested_max_tokens, self.max_new_tokens))
        temperature = float(payload.get("temperature", 0.0) or 0.0)
        top_p = float(payload.get("top_p", 1.0) or 1.0)
        stop = payload.get("stop")
        if isinstance(stop, str):
            stop_sequences = [stop]
        elif isinstance(stop, list):
            stop_sequences = [str(item) for item in stop]
        else:
            stop_sequences = []

        prompt = _render_chat_prompt(self.tokenizer, messages, payload.get("tools"))
        inputs = None
        output = None
        new_tokens = None
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model_device)
        input_tokens = int(inputs.input_ids.shape[-1])
        generate_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p

        try:
            with self._generate_lock, self.torch.no_grad():
                output = self.model.generate(**generate_kwargs)
            new_tokens = output[:, input_tokens:]
            text = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
            text = _apply_stop_sequences(text, stop_sequences)
            completion_tokens = int(new_tokens.shape[-1])
            tool_calls = _extract_tool_calls(text, payload)
        finally:
            del inputs, output, new_tokens
            self._clear_device_cache()
        return {
            "text": text,
            "tool_calls": tool_calls,
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": input_tokens + completion_tokens,
            },
        }

    def _clear_device_cache(self) -> None:
        gc.collect()
        if self.device_name == "cuda" and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        if self.device_name == "mps" and hasattr(self.torch, "mps"):
            self.torch.mps.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default="", help="Optional PEFT adapter directory or Hub id")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    backend = ModelBackend(
        model_id=args.model,
        adapter=args.adapter,
        device=args.device,
        dtype_name=args.dtype,
        max_new_tokens=args.max_new_tokens,
    )
    handler = _handler(backend)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{server.server_address[1]}/v1"
    print(
        json.dumps(
            {
                "schema_version": "hfr.transformers_openai_server.v1",
                "base_url": url,
                "model": backend.served_model_id,
                "device": str(backend.model_device),
                "dtype": str(backend.dtype),
                "pid": os.getpid(),
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


def _handler(backend: ModelBackend) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"{self.address_string()} - {fmt % args}", flush=True)

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0].rstrip("/")
            if path in {"", "/healthz"}:
                self._send_json({"ok": True, "model": backend.served_model_id})
                return
            if path in {"/api/v1/models", "/v1/models", "/models"}:
                self._send_json({"object": "list", "data": [{"id": backend.served_model_id, "object": "model"}]})
                return
            if path in {f"/v1/models/{backend.served_model_id}", f"/models/{backend.served_model_id}"}:
                self._send_json({"id": backend.served_model_id, "object": "model"})
                return
            if path == "/api/tags":
                self._send_json({"models": [{"name": backend.served_model_id, "model": backend.served_model_id}]})
                return
            if path in {"/v1/props", "/props"}:
                self._send_json(_model_details(backend))
                return
            if path == "/version":
                self._send_json({"version": "hfr-transformers-openai"})
                return
            self._send_json({"error": {"message": f"Not found: {path}"}}, status=404)

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0].rstrip("/")
            length = int(self.headers.get("content-length") or 0)
            body = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError as exc:
                self._send_json({"error": {"message": f"Invalid JSON: {exc}"}}, status=400)
                return

            if path == "/api/show":
                self._send_json(_model_details(backend))
                return

            if path != "/v1/chat/completions":
                self._send_json({"error": {"message": f"Not found: {path}"}}, status=404)
                return

            try:
                completion = backend.complete(payload)
            except Exception as exc:  # pragma: no cover - exercised only by live server failures.
                self._send_json({"error": {"message": str(exc), "type": type(exc).__name__}}, status=500)
                return

            if payload.get("stream"):
                self._send_stream(completion["text"], completion["usage"], completion.get("tool_calls") or [])
                return

            self._send_json(_completion_payload(backend, completion["text"], completion["usage"], completion.get("tool_calls") or []))

        def _send_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("connection", "close")
            self.send_header("content-length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _send_stream(self, text: str, usage: dict[str, int], tool_calls: list[dict[str, Any]]) -> None:
            created = int(time.time())
            if tool_calls:
                chunk = {
                    "id": "chatcmpl-hfr-local",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": backend.served_model_id,
                    "choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": tool_calls}, "finish_reason": None}],
                }
                finish_reason = "tool_calls"
            else:
                chunk = {
                    "id": "chatcmpl-hfr-local",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": backend.served_model_id,
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}],
                }
                finish_reason = "stop"
            final = {
                "id": "chatcmpl-hfr-local",
                "object": "chat.completion.chunk",
                "created": created,
                "model": backend.served_model_id,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                "usage": usage,
            }
            raw = f"data: {json.dumps(chunk)}\n\ndata: {json.dumps(final)}\n\ndata: [DONE]\n\n".encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("cache-control", "no-cache")
            self.send_header("connection", "close")
            self.send_header("content-length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    return Handler


def _completion_payload(
    backend: ModelBackend,
    text: str,
    usage: dict[str, int],
    tool_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": None if tool_calls else text}
    finish_reason = "stop"
    if tool_calls:
        message["tool_calls"] = tool_calls
        finish_reason = "tool_calls"
    return {
        "id": "chatcmpl-hfr-local",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": backend.served_model_id,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": usage,
    }


def _model_details(backend: ModelBackend) -> dict[str, Any]:
    max_context = getattr(getattr(backend.model, "config", None), "max_position_embeddings", None)
    return {
        "id": backend.served_model_id,
        "model": backend.served_model_id,
        "object": "model",
        "context_length": max_context,
        "device": str(backend.model_device),
        "dtype": str(backend.dtype),
    }


def _resolve_device(device: str) -> str:
    import torch

    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(dtype_name: str, device: str) -> Any:
    import torch

    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def _apply_stop_sequences(text: str, stop_sequences: list[str]) -> str:
    first_index: int | None = None
    for sequence in stop_sequences:
        if not sequence:
            continue
        index = text.find(sequence)
        if index >= 0:
            first_index = index if first_index is None else min(first_index, index)
    if first_index is None:
        return text
    return text[:first_index]


def _render_chat_prompt(tokenizer: Any, messages: list[dict[str, Any]], tools: Any) -> str:
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if isinstance(tools, list) and tools:
        kwargs["tools"] = tools
    try:
        return str(tokenizer.apply_chat_template(messages, **kwargs))
    except TypeError as exc:
        if "tools" in kwargs:
            raise ValueError("Tokenizer chat template does not accept structured tools for agentic serving") from exc
        raise


def _extract_tool_calls(text: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    available = _available_tool_names(payload)
    if not available:
        return []
    parsed = _parse_json_tool_call(text, available) or _parse_read_file_call(text, available) or _parse_terminal_call(text, available)
    if parsed is None:
        return []
    name, args = parsed
    return [
        {
            "index": 0,
            "id": f"call_hfr_{int(time.time() * 1000)}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args, sort_keys=True),
            },
        }
    ]


def _available_tool_names(payload: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for tool in payload.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") or {}
        name = function.get("name")
        if name:
            names.add(str(name))
    return names


def _parse_json_tool_call(text: str, available: set[str]) -> tuple[str, dict[str, Any]] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        parsed = _tool_call_from_json_object(obj, available)
        if parsed is not None:
            return parsed
    return None


def _tool_call_from_json_object(obj: Any, available: set[str]) -> tuple[str, dict[str, Any]] | None:
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or obj.get("tool_name")
    args = obj.get("arguments", obj.get("args", {}))
    function = obj.get("function")
    if isinstance(function, dict):
        name = name or function.get("name")
        args = function.get("arguments", args)
    if not name or str(name) not in available:
        return None
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    return str(name), args


def _parse_read_file_call(text: str, available: set[str]) -> tuple[str, dict[str, Any]] | None:
    if "read_file" not in available:
        return None
    patterns = [
        r"read_file\s*\(\s*path\s*=\s*['\"]([^'\"]+)['\"]",
        r"read_file\s*\(\s*['\"]([^'\"]+)['\"]",
        r"read_file\s*\(\s*\{\s*['\"]path['\"]\s*:\s*['\"]([^'\"]+)['\"]",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return "read_file", {"path": match.group(1)}
    return None


def _parse_terminal_call(text: str, available: set[str]) -> tuple[str, dict[str, Any]] | None:
    if "terminal" not in available:
        return None
    for block in re.findall(r"```(?:bash|sh|shell)?\s*\n(.*?)```", text, flags=re.IGNORECASE | re.DOTALL):
        command = _first_shell_command(block)
        if command:
            return "terminal", {"command": command}
    return None


def _first_shell_command(block: str) -> str:
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("$ ", "> ")):
            line = line[2:].strip()
        if line.startswith(("ls ", "cp ", "mkdir ", "test ", "cat ", "python ", "python3 ")):
            return line
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
