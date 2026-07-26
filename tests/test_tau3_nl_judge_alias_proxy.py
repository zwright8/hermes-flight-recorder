from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from flightrecorder.tau3_nl_judge_alias_proxy import (
    DEFAULT_REQUEST_MODEL,
    Tau3NLJudgeAliasProxyError,
    build_arg_parser,
    build_server,
    main,
)


class Tau3NLJudgeAliasProxyTests(unittest.TestCase):
    def test_proxy_rewrites_only_the_model_and_records_hash_audit(self) -> None:
        upstream_calls: list[dict[str, object]] = []

        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args: object) -> None:
                return None

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("content-length") or 0)
                payload = json.loads(self.rfile.read(length))
                upstream_calls.append(payload)
                body = json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": json.dumps(
                                        {
                                            "results": [
                                                {
                                                    "expectedOutcome": "Agent resolves the issue.",
                                                    "reasoning": "The transcript matches.",
                                                    "metExpectation": True,
                                                }
                                            ]
                                        }
                                    ),
                                }
                            }
                        ]
                    }
                ).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        upstream_thread.start()
        with tempfile.TemporaryDirectory() as tmp:
            audit = Path(tmp) / "audit.jsonl"
            served_model = "/models/local-teacher"
            proxy = build_server(
                upstream_base_url=f"http://127.0.0.1:{upstream.server_address[1]}/v1",
                host="127.0.0.1",
                port=0,
                audit_log=audit,
                served_model=served_model,
            )
            proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
            proxy_thread.start()
            try:
                payload = {
                    "model": DEFAULT_REQUEST_MODEL,
                    "messages": [{"role": "user", "content": "private transcript"}],
                    "temperature": 0,
                }
                request = urllib.request.Request(
                    f"http://127.0.0.1:{proxy.server_address[1]}/v1/chat/completions",
                    data=json.dumps(payload).encode(),
                    headers={"content-type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request) as response:
                    self.assertEqual(response.status, 200)
                    json.loads(response.read())
                self.assertEqual(upstream_calls[0]["model"], served_model)
                self.assertEqual(upstream_calls[0]["messages"], payload["messages"])

                record = json.loads(audit.read_text(encoding="utf-8"))
                self.assertEqual(record["schema_version"], "hfr.tau3_nl_judge_alias_proxy.v1")
                self.assertEqual(record["status"], 200)
                self.assertFalse(record["payload_recorded"])
                self.assertIn("request_sha256", record)
                self.assertIn("rewritten_request_sha256", record)
                self.assertIn("response_sha256", record)
                self.assertEqual(record["allowed_request_model_count"], 1)
                self.assertNotIn("messages", record)
                self.assertNotIn("private transcript", json.dumps(record))
                self.assertNotIn(served_model, json.dumps(record))
            finally:
                proxy.shutdown()
                proxy.server_close()
        upstream.shutdown()
        upstream.server_close()

    def test_proxy_accepts_all_aliases_and_lists_them_deterministically(self) -> None:
        upstream_calls: list[dict[str, object]] = []

        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args: object) -> None:
                return None

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("content-length") or 0)
                upstream_calls.append(json.loads(self.rfile.read(length)))
                body = b'{"choices":[]}'
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        upstream_thread.start()
        with tempfile.TemporaryDirectory() as tmp:
            audit = Path(tmp) / "audit.jsonl"
            served_model = "/models/local-teacher"
            allowed_models = sorted([DEFAULT_REQUEST_MODEL, "frozen-teacher"])
            proxy = build_server(
                upstream_base_url=f"http://127.0.0.1:{upstream.server_address[1]}/v1",
                host="127.0.0.1",
                port=0,
                audit_log=audit,
                request_model_aliases=["frozen-teacher"],
                served_model=served_model,
            )
            proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
            proxy_thread.start()
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{proxy.server_address[1]}/v1/models"
                ) as response:
                    model_listing = json.loads(response.read())
                self.assertEqual([model["id"] for model in model_listing["data"]], allowed_models)

                for model in allowed_models:
                    request = urllib.request.Request(
                        f"http://127.0.0.1:{proxy.server_address[1]}/v1/chat/completions",
                        data=json.dumps({"model": model, "messages": []}).encode(),
                        headers={"content-type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(request) as response:
                        self.assertEqual(response.status, 200)

                self.assertEqual([call["model"] for call in upstream_calls], [served_model, served_model])
                records = [
                    json.loads(line)
                    for line in audit.read_text(encoding="utf-8").splitlines()
                ]
                self.assertEqual(
                    [record["request_model_sha256"] for record in records],
                    [hashlib.sha256(model.encode()).hexdigest() for model in allowed_models],
                )
                self.assertEqual({record["allowed_request_model_count"] for record in records}, {2})
                self.assertEqual(len({record["allowed_request_models_sha256"] for record in records}), 1)
                self.assertNotIn("frozen-teacher", json.dumps(records))
                self.assertNotIn(served_model, json.dumps(records))
            finally:
                proxy.shutdown()
                proxy.server_close()
        upstream.shutdown()
        upstream.server_close()

    def test_request_model_aliases_reject_empty_and_duplicate_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            common: dict[str, Any] = {
                "upstream_base_url": "http://127.0.0.1:18084/v1",
                "host": "127.0.0.1",
                "port": 0,
                "audit_log": Path(tmp) / "audit.jsonl",
                "served_model": "/models/local-teacher",
            }
            for aliases in ([""], ["   "], [DEFAULT_REQUEST_MODEL], ["alias", " alias "]):
                with self.subTest(aliases=aliases):
                    with self.assertRaises(Tau3NLJudgeAliasProxyError):
                        build_server(request_model_aliases=aliases, **common)

    def test_cli_preserves_default_model_and_collects_repeatable_aliases(self) -> None:
        parser = build_arg_parser()
        defaults = parser.parse_args(
            [
                "--upstream",
                "http://127.0.0.1:18084/v1",
                "--audit-log",
                "audit.jsonl",
                "--served-model",
                "/models/local-teacher",
            ]
        )
        self.assertEqual(defaults.request_model, DEFAULT_REQUEST_MODEL)
        self.assertEqual(defaults.request_model_aliases, [])

        aliases = parser.parse_args(
            [
                "--upstream",
                "http://127.0.0.1:18084/v1",
                "--audit-log",
                "audit.jsonl",
                "--served-model",
                "/models/local-teacher",
                "--request-model-alias",
                "frozen-teacher",
                "--request-model-alias",
                "reviewer-alias",
            ]
        )
        self.assertEqual(aliases.request_model_aliases, ["frozen-teacher", "reviewer-alias"])

    def test_cli_rejects_empty_and_duplicate_aliases(self) -> None:
        common = [
            "--upstream",
            "http://127.0.0.1:18084/v1",
            "--audit-log",
            "audit.jsonl",
            "--served-model",
            "/models/local-teacher",
        ]
        for extra in (
            ["--request-model-alias", ""],
            ["--request-model-alias", DEFAULT_REQUEST_MODEL],
            ["--request-model-alias", "alias", "--request-model-alias", " alias "],
        ):
            with self.subTest(extra=extra), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(common + extra), 2)

    def test_proxy_rejects_unexpected_models_without_forwarding(self) -> None:
        upstream_calls: list[bytes] = []

        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args: object) -> None:
                return None

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("content-length") or 0)
                upstream_calls.append(self.rfile.read(length))
                self.send_response(500)
                self.end_headers()

        upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        upstream_thread.start()
        with tempfile.TemporaryDirectory() as tmp:
            proxy = build_server(
                upstream_base_url=f"http://127.0.0.1:{upstream.server_address[1]}/v1",
                host="127.0.0.1",
                port=0,
                audit_log=Path(tmp) / "audit.jsonl",
                request_model_aliases=["frozen-teacher"],
                served_model="/models/local-teacher",
            )
            proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
            proxy_thread.start()
            try:
                request = urllib.request.Request(
                    f"http://127.0.0.1:{proxy.server_address[1]}/v1/chat/completions",
                    data=json.dumps({"model": "wrong-model", "messages": []}).encode(),
                    headers={"content-type": "application/json"},
                    method="POST",
                )
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(request)
                self.assertEqual(caught.exception.code, 400)
                self.assertEqual(upstream_calls, [])
            finally:
                proxy.shutdown()
                proxy.server_close()
        upstream.shutdown()
        upstream.server_close()


if __name__ == "__main__":
    unittest.main()
