from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from flightrecorder.tau3_user_terminal_proxy import build_server, terminal_reason


def _payload(system: str, *messages: dict[str, object]) -> dict[str, object]:
    return {
        "model": "local-user",
        "messages": [{"role": "system", "content": f"User Simulation Guidelines\n<scenario>{system}</scenario>"}, *messages],
    }


class Tau3UserTerminalProxyTests(unittest.TestCase):
    def test_mobile_stops_only_on_excellent_speed_evidence(self) -> None:
        system = "You will consider the issue resolved only when speed test returns excellent internet speed."
        self.assertIsNone(terminal_reason(_payload(system, {"role": "tool", "content": "Speed Test Result: 80 Mbps (Good)."})))
        self.assertIsNone(
            terminal_reason(
                _payload(system, {"role": "assistant", "content": "Speed Test Result: 275 Mbps (Excellent)."})
            )
        )
        self.assertEqual(
            terminal_reason(_payload(system, {"role": "tool", "content": "Speed Test Result: 275.00 Mbps (Excellent). Connection is very fast."})),
            "mobile_excellent_speed",
        )

    def test_service_uses_latest_status_bar_evidence(self) -> None:
        system = "You will consider the issue resolved when the status bar shows that you have signal."
        payload = _payload(
            system,
            {"role": "tool", "content": "Status Bar: signal present | 5G"},
            {"role": "tool", "content": "Status Bar: No Service"},
        )
        self.assertIsNone(terminal_reason(payload))
        payload["messages"].append({"role": "tool", "content": "Status Bar: 4 bars | 5G | Data Enabled"})  # type: ignore[index,union-attr]
        self.assertEqual(terminal_reason(payload), "service_signal_present")

    def test_service_rejects_claimed_or_ambiguous_status(self) -> None:
        system = "You will consider the issue resolved when the status bar shows that you have signal."
        for role, status in (
            ("assistant", "4 bars | 5G"),
            ("user", "signal present"),
            ("tool", "Unknown"),
            ("tool", "Searching"),
            ("tool", "Loading"),
            ("tool", "0 bars"),
        ):
            with self.subTest(role=role, status=status):
                self.assertIsNone(
                    terminal_reason(_payload(system, {"role": role, "content": f"Status Bar: {status}"}))
                )

    def test_non_user_simulator_requests_are_not_intercepted(self) -> None:
        payload = {
            "messages": [
                {"role": "system", "content": "customer service agent"},
                {"role": "tool", "content": "Speed Test Result: 275 Mbps (Excellent)."},
            ]
        }
        self.assertIsNone(terminal_reason(payload))

    def test_transfer_requires_exact_trusted_tool_observation(self) -> None:
        system = "Wait for a transfer when the agent cannot resolve the issue."
        self.assertIsNone(terminal_reason(_payload(system, {"role": "assistant", "content": "Transfer successful"})))
        self.assertIsNone(terminal_reason(_payload(system, {"role": "tool", "content": "Transfer pending"})))
        self.assertEqual(
            terminal_reason(_payload(system, {"role": "tool", "content": "Transfer successful"})),
            "transfer_successful",
        )

    def test_proxy_intercepts_and_hash_logs_without_forwarding(self) -> None:
        upstream_calls: list[bytes] = []

        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args: object) -> None:
                return None

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("content-length") or 0)
                upstream_calls.append(self.rfile.read(length))
                body = json.dumps({"choices": [{"message": {"role": "assistant", "content": "forwarded"}}]}).encode()
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
            proxy = build_server(
                upstream_base_url=f"http://127.0.0.1:{upstream.server_address[1]}/v1",
                host="127.0.0.1",
                port=0,
                audit_log=audit,
            )
            proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
            proxy_thread.start()
            try:
                payload = _payload(
                    "speed test returns excellent",
                    {"role": "tool", "content": "Speed Test Result: 300 Mbps (Excellent)."},
                )
                request = urllib.request.Request(
                    f"http://127.0.0.1:{proxy.server_address[1]}/v1/chat/completions",
                    data=json.dumps(payload).encode(),
                    headers={"content-type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request) as response:
                    result = json.loads(response.read())
                self.assertEqual(result["choices"][0]["message"]["content"], "###STOP###")
                self.assertEqual(upstream_calls, [])
                record = json.loads(audit.read_text())
                self.assertEqual(record["reason"], "mobile_excellent_speed")
                self.assertFalse(record["payload_recorded"])
                self.assertNotIn("messages", record)
            finally:
                proxy.shutdown()
                proxy.server_close()
        upstream.shutdown()
        upstream.server_close()

    def test_proxy_emits_transfer_marker_from_trusted_tool_evidence(self) -> None:
        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args: object) -> None:
                return None

        upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        with tempfile.TemporaryDirectory() as tmp:
            audit = Path(tmp) / "audit.jsonl"
            proxy = build_server(
                upstream_base_url=f"http://127.0.0.1:{upstream.server_address[1]}/v1",
                host="127.0.0.1",
                port=0,
                audit_log=audit,
            )
            proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
            proxy_thread.start()
            try:
                payload = _payload(
                    "Wait for a transfer when required.",
                    {"role": "tool", "content": "Transfer successful"},
                )
                request = urllib.request.Request(
                    f"http://127.0.0.1:{proxy.server_address[1]}/v1/chat/completions",
                    data=json.dumps(payload).encode(),
                    headers={"content-type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request) as response:
                    result = json.loads(response.read())
                self.assertEqual(result["choices"][0]["message"]["content"], "###TRANSFER###")
                record = json.loads(audit.read_text())
                self.assertEqual(record["reason"], "transfer_successful")
                self.assertEqual(record["marker"], "TRANSFER")
            finally:
                proxy.shutdown()
                proxy.server_close()
        upstream.server_close()


if __name__ == "__main__":
    unittest.main()
