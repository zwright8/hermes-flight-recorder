from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from flightrecorder.schema_registry import check_schema_file
from flightrecorder.tau3_behavior_probes import (
    EndpointConfig,
    build_tau3_behavior_probes,
    validate_tau3_behavior_probes,
)


class Tau3BehaviorProbesTests(unittest.TestCase):
    def test_builds_and_validates_hash_bound_probe_artifacts(self) -> None:
        with _mock_openai_server() as server:
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                build_tau3_behavior_probes(out, endpoint=EndpointConfig(base_url=server.base_url, model="local-agent"), bindings=_bindings())

                aggregate_path = out / "behavior-probes.json"
                aggregate = _read_json(aggregate_path)
                self.assertTrue(aggregate["passed"], json.dumps(aggregate, indent=2))
                self.assertEqual(aggregate["aggregate"]["total_probe_count"], 8)
                self.assertEqual(aggregate["aggregate"]["failed_probe_count"], 0)
                self.assertEqual({request["model"] for request in server.requests}, {"local-agent"})
                self.assertEqual(len(server.requests), 8)

                schema_result = check_schema_file(aggregate_path, "tau3_behavior_probes")
                self.assertTrue(schema_result["passed"], schema_result["errors"])
                self.assertEqual(aggregate_path.stat().st_mode & 0o777, 0o600)
                for ref in aggregate["probe_results"]:
                    result_path = out / ref["path"]
                    result = check_schema_file(result_path, "tau3_behavior_probe_result")
                    self.assertTrue(result["passed"], result["errors"])
                    self.assertEqual(result_path.stat().st_mode & 0o777, 0o600)

                validation = validate_tau3_behavior_probes(out)
                self.assertTrue(validation["passed"], json.dumps(validation, indent=2))

    def test_validator_rejects_forged_probe_summary(self) -> None:
        with _mock_openai_server() as server:
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                build_tau3_behavior_probes(out, endpoint=EndpointConfig(base_url=server.base_url, model="local-agent"), bindings=_bindings())
                aggregate_path = out / "behavior-probes.json"
                aggregate = _read_json(aggregate_path)
                first_result = out / aggregate["probe_results"][0]["path"]
                result = _read_json(first_result)
                result["observation"]["content"]["redacted"] = "{\"status\":\"bad\"}"
                _write_json(first_result, result)
                aggregate["probe_results"][0]["sha256"] = _sha256_file(first_result)
                aggregate["passed"] = True
                aggregate["aggregate"]["failed_probe_count"] = 0
                _write_json(aggregate_path, aggregate)

                validation = validate_tau3_behavior_probes(out)

                self.assertFalse(validation["passed"])
                self.assertIn("actual_outcome does not replay", json.dumps(validation))
                self.assertIn("aggregate.failed_probe_count does not replay", json.dumps(validation))

    def test_missing_required_family_fails_before_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            specs = [
                {
                    "probe_id": "format-only",
                    "family": "formatting",
                    "prompt": "Return JSON",
                    "assertions": [{"type": "json_field_equals", "field": "status", "value": "ok"}],
                }
            ]
            with self.assertRaisesRegex(ValueError, "missing required probe families"):
                build_tau3_behavior_probes(
                    out,
                    endpoint=EndpointConfig(base_url="http://127.0.0.1:9/v1", model="local-agent"),
                    bindings=_bindings(),
                    probe_specs=specs,
                )

    def test_builder_refuses_to_overwrite_existing_probe_evidence(self) -> None:
        with _mock_openai_server() as server:
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                build_tau3_behavior_probes(
                    out,
                    endpoint=EndpointConfig(
                        base_url=server.base_url,
                        model="local-agent",
                    ),
                    bindings=_bindings(),
                )
                aggregate_path = out / "behavior-probes.json"
                original = aggregate_path.read_bytes()

                with self.assertRaisesRegex(
                    ValueError,
                    "output directory must be empty",
                ):
                    build_tau3_behavior_probes(
                        out,
                        endpoint=EndpointConfig(
                            base_url=server.base_url,
                            model="local-agent",
                        ),
                        bindings=_bindings(),
                    )

                self.assertEqual(aggregate_path.read_bytes(), original)
                self.assertEqual(len(server.requests), 8)


class _Server:
    def __init__(self, httpd: ThreadingHTTPServer, requests: list[dict[str, Any]]) -> None:
        self.httpd = httpd
        self.requests = requests
        self.base_url = f"http://127.0.0.1:{httpd.server_address[1]}/v1"
        self._thread = threading.Thread(target=httpd.serve_forever, daemon=True)

    def __enter__(self) -> "_Server":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self._thread.join(timeout=5)


def _mock_openai_server() -> _Server:
    requests: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            requests.append(payload)
            prompt = payload["messages"][0]["content"]
            message = _message_for_prompt(prompt)
            body = json.dumps({"choices": [{"message": message}]}, sort_keys=True).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_: object) -> None:
            return

    return _Server(ThreadingHTTPServer(("127.0.0.1", 0), Handler), requests)


def _message_for_prompt(prompt: str) -> dict[str, Any]:
    if "JSON object" in prompt:
        return {"role": "assistant", "content": "{\"status\":\"ok\"}"}
    if "unspecified" in prompt:
        return {"role": "assistant", "content": "Which client and what account change should I use?"}
    if "account lookup tool" in prompt:
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "lookup_account", "arguments": "{\"account\":\"A-100\"}"}}],
        }
    if "no rows" in prompt:
        return {"role": "assistant", "content": "No rows were found. Verify the account id or broaden the lookup."}
    if "HTTP 503" in prompt:
        return {"role": "assistant", "content": "The error looks temporary; retry later and report the failed check."}
    if "failed three times" in prompt:
        return {"role": "assistant", "content": "Stop and escalate; do not repeat the same lookup again."}
    if "Delete the customer" in prompt:
        return {"role": "assistant", "content": "Please confirm before deleting the customer record."}
    if "Stop." in prompt:
        return {"role": "assistant", "content": "Stopping now; I will not continue or call tools."}
    return {"role": "assistant", "content": ""}


def _bindings() -> dict[str, str]:
    return {
        "training_receipt_sha256": "1" * 64,
        "adapter_tree_sha256": "2" * 64,
        "harness_sha256": "3" * 64,
        "protocol_sha256": "4" * 64,
        "grid_sha256": "5" * 64,
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
