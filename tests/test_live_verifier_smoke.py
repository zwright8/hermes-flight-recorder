import importlib.util
import json
import os
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "live_verifier_smoke.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("live_verifier_smoke", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class LiveVerifierSmokeTests(unittest.TestCase):
    def test_no_network_mode_reports_skipped_provider_without_credentials(self):
        live_smoke = _load_script()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            with redirect_stdout(StringIO()):
                code = live_smoke.main(["--out", tmp, "--provider", "slack"])

            self.assertEqual(code, 0)
            summary = _read_json(Path(tmp) / "live_verifier_smoke_summary.json")
            self.assertTrue(summary["passed"])
            self.assertEqual(summary["selected_provider_count"], 1)
            self.assertEqual(summary["skipped_provider_count"], 1)
            self.assertEqual(summary["providers"][0]["status"], "skipped")
            self.assertEqual(summary["providers"][0]["reason"], "network_disabled")
            schema_check = _read_json(Path(tmp) / "live_verifier_smoke_summary.schema_check.json")
            self.assertTrue(schema_check["passed"], schema_check["errors"])

    def test_strict_live_fails_when_requested_provider_is_missing_credentials(self):
        live_smoke = _load_script()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            with redirect_stdout(StringIO()):
                code = live_smoke.main(["--out", tmp, "--provider", "slack", "--allow-network", "--strict-live"])

            self.assertEqual(code, 1)
            summary = _read_json(Path(tmp) / "live_verifier_smoke_summary.json")
            self.assertFalse(summary["passed"])
            self.assertEqual(summary["providers"][0]["status"], "skipped")
            self.assertEqual(summary["providers"][0]["reason"], "missing_configuration")
            self.assertEqual(summary["providers"][0]["missing_env"], ["SLACK_BOT_TOKEN", "HFR_SLACK_CHANNEL_ID"])

    def test_malformed_provider_configuration_is_reported_without_traceback(self):
        live_smoke = _load_script()
        env = {
            "SLACK_BOT_TOKEN": "slack-token",
            "HFR_SLACK_CHANNEL_ID": "C123",
            "HFR_SLACK_LIMIT": "not-an-int",
        }
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, env, clear=True):
            with redirect_stdout(StringIO()):
                code = live_smoke.main(["--out", tmp, "--provider", "slack", "--allow-network"])

            self.assertEqual(code, 1)
            summary = _read_json(Path(tmp) / "live_verifier_smoke_summary.json")
            self.assertFalse(summary["passed"])
            self.assertEqual(summary["providers"][0]["status"], "failed")
            self.assertEqual(summary["providers"][0]["reason"], "capture_failed")
            self.assertIn("HFR_SLACK_LIMIT must be an integer", summary["providers"][0]["error"])
            self.assertNotIn("slack-token", json.dumps(summary))

    def test_all_http_provider_smokes_pass_against_readonly_local_endpoints(self):
        live_smoke = _load_script()
        server = _JsonServer(_provider_routes())
        providers = [
            "discord",
            "github",
            "gitlab",
            "gmail",
            "google_calendar",
            "google_drive",
            "imap",
            "jira",
            "kubernetes",
            "linear",
            "microsoft_graph_events",
            "microsoft_graph_messages",
            "notion",
            "pagerduty",
            "s3",
            "slack",
            "stripe",
            "zendesk",
        ]
        env = _provider_env(server.url)
        try:
            with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, env, clear=True):
                with patch("flightrecorder.verifiers.imaplib.IMAP4_SSL", FakeIMAP):
                    args = ["--out", tmp, "--allow-network", "--strict-live", "--require-live-provider"]
                    for provider in providers:
                        args.extend(["--provider", provider])
                    with redirect_stdout(StringIO()):
                        code = live_smoke.main(args)

                self.assertEqual(code, 0)
                out = Path(tmp)
                summary = _read_json(out / "live_verifier_smoke_summary.json")
                self.assertTrue(summary["passed"], summary)
                self.assertEqual(summary["passed_provider_count"], len(providers))
                self.assertEqual(summary["failed_provider_count"], 0)
                self.assertEqual(summary["skipped_provider_count"], 0)
                self.assertEqual({record["provider"] for record in summary["providers"]}, set(providers))
                self.assertTrue(_read_json(out / "live_verifier_smoke_summary.schema_check.json")["passed"])

                for record in summary["providers"]:
                    self.assertEqual(record["status"], "passed", record)
                    self.assertTrue(record["validation_passed"], record)
                    artifacts = record["artifacts"]
                    self.assertTrue(Path(artifacts["state_snapshot"]).exists(), artifacts)
                    validation = _read_json(Path(artifacts["validation"]))
                    self.assertTrue(validation["passed"], validation)

                rendered = "\n".join(path.read_text(encoding="utf-8") for path in out.rglob("*.json"))
                for secret in _secret_values(env):
                    self.assertNotIn(secret, rendered)
        finally:
            server.close()


class FakeIMAP:
    def __init__(self, host, port, timeout=None):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.selected_readonly = None

    def login(self, username, password):
        self.username = username
        self.password = password
        return "OK", [b"authenticated"]

    def select(self, mailbox, readonly=False):
        self.mailbox = mailbox
        self.selected_readonly = readonly
        return "OK", [b"1"]

    def search(self, charset, *criteria):
        self.charset = charset
        self.criteria = criteria
        return "OK", [b"1"]

    def fetch(self, message_id, query):
        return "OK", [(b"1 (UID 42 FLAGS (\\Seen) RFC822 {100}", _email_bytes())]

    def logout(self):
        return "OK", [b"bye"]


class _JsonServer:
    def __init__(self, routes):
        self.routes = routes
        self.requests = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def _handler(self):
        routes = self.routes
        requests = self.requests

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path, _, query = self.path.partition("?")
                requests.append({"method": "GET", "path": path, "query": query})
                if path not in routes:
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(b'{"error":"not found"}')
                    return
                self._send_route(routes[path])

            def do_POST(self):
                path, _, query = self.path.partition("?")
                requests.append({"method": "POST", "path": path, "query": query})
                if path not in routes:
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(b'{"error":"not found"}')
                    return
                length = int(self.headers.get("Content-Length") or "0")
                if length:
                    self.rfile.read(length)
                self._send_route(routes[path])

            def _send_route(self, value):
                if isinstance(value, str):
                    payload = value.encode("utf-8")
                    content_type = "application/xml"
                else:
                    payload = json.dumps(value).encode("utf-8")
                    content_type = "application/json"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, _format, *args):
                return

        return Handler

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def _provider_routes():
    return {
        "/discord/api/v10/channels/C123/messages": [
            {"id": "discord-msg-1", "content": "deployment finished", "author": {"id": "U1", "username": "agent"}}
        ],
        "/repos/octo/repo/issues/7": {"number": 7, "state": "closed", "title": "Support reply", "comments": 1},
        "/repos/octo/repo/issues/7/comments": [{"id": 11, "body": "Reply sent.", "user": {"login": "agent"}}],
        "/gitlab/api/v4/projects/group%2Fproject/issues": [
            {"id": 1, "iid": 7, "title": "Fix deploy", "state": "closed", "labels": ["ops"]}
        ],
        "/gmail/v1/users/me/threads": {"threads": [{"id": "thread-1"}]},
        "/gmail/v1/users/me/threads/thread-1": {
            "id": "thread-1",
            "messages": [
                {
                    "id": "msg-1",
                    "threadId": "thread-1",
                    "labelIds": ["SENT"],
                    "snippet": "Reply sent",
                    "payload": {"headers": [{"name": "Subject", "value": "Re: customer"}]},
                }
            ],
        },
        "/calendar/v3/calendars/primary/events": {
            "items": [{"id": "evt-1", "summary": "Customer follow-up", "status": "confirmed"}]
        },
        "/drive/v3/files": {"files": [{"id": "file-1", "name": "handoff notes", "mimeType": "text/plain"}]},
        "/jira/rest/api/3/search": {
            "issues": [{"id": "10001", "key": "OPS-7", "fields": {"summary": "Close incident", "status": {"name": "Done"}}}],
            "total": 1,
        },
        "/k8s/apis/apps/v1/namespaces/prod/deployments": {
            "items": [{"kind": "Deployment", "metadata": {"name": "api"}, "spec": {"replicas": 1}, "status": {"readyReplicas": 1}}]
        },
        "/linear/graphql": {
            "data": {"issues": {"nodes": [{"id": "lin-1", "identifier": "ENG-123", "title": "Fix deploy", "state": {"name": "Done"}}]}}
        },
        "/graph/v1.0/me/events": {"value": [{"id": "event-1", "subject": "Customer follow-up", "showAs": "busy"}]},
        "/graph/v1.0/me/mailFolders/SentItems/messages": {
            "value": [{"id": "msg-graph-1", "subject": "Re: customer", "bodyPreview": "Sent update"}]
        },
        "/notion/v1/databases/db1/query": {
            "results": [
                {
                    "id": "page-1",
                    "properties": {"Name": {"type": "title", "title": [{"plain_text": "Runbook update"}]}},
                }
            ]
        },
        "/pagerduty/incidents": {"incidents": [{"id": "PD123", "title": "API outage", "status": "resolved"}]},
        "/s3": (
            "<ListBucketResult xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\">"
            "<Contents><Key>reports/out.json</Key><LastModified>2026-06-29T00:00:00Z</LastModified>"
            "<ETag>\"abc123\"</ETag><Size>42</Size><StorageClass>STANDARD</StorageClass></Contents>"
            "</ListBucketResult>"
        ),
        "/slack/conversations.history": {"ok": True, "messages": [{"type": "message", "text": "deployment finished"}]},
        "/stripe/v1/payment_intents/pi_123": {
            "id": "pi_123",
            "object": "payment_intent",
            "status": "succeeded",
            "amount": 4200,
            "currency": "usd",
        },
        "/zendesk/api/v2/tickets/42.json": {"ticket": {"id": 42, "subject": "Customer request", "status": "solved"}},
    }


def _provider_env(base_url: str) -> dict[str, str]:
    return {
        "DISCORD_BOT_TOKEN": "discord-token",
        "HFR_DISCORD_CHANNEL_ID": "C123",
        "HFR_DISCORD_BASE_URL": f"{base_url}/discord/api/v10",
        "GITHUB_TOKEN": "github-token",
        "HFR_GITHUB_OWNER": "octo",
        "HFR_GITHUB_REPO": "repo",
        "HFR_GITHUB_ISSUE_NUMBER": "7",
        "HFR_GITHUB_BASE_URL": base_url,
        "GITLAB_TOKEN": "gitlab-token",
        "HFR_GITLAB_PROJECT_ID": "group/project",
        "HFR_GITLAB_BASE_URL": f"{base_url}/gitlab/api/v4",
        "GMAIL_ACCESS_TOKEN": "gmail-token",
        "HFR_GMAIL_BASE_URL": f"{base_url}/gmail/v1",
        "GOOGLE_CALENDAR_ACCESS_TOKEN": "calendar-token",
        "HFR_GOOGLE_CALENDAR_BASE_URL": f"{base_url}/calendar/v3",
        "GOOGLE_DRIVE_ACCESS_TOKEN": "drive-token",
        "HFR_GOOGLE_DRIVE_BASE_URL": f"{base_url}/drive/v3",
        "IMAP_HOST": "imap.example.test",
        "IMAP_USERNAME": "agent@example.test",
        "IMAP_PASSWORD": "imap-password",
        "JIRA_API_TOKEN": "jira-token",
        "JIRA_EMAIL": "agent@example.test",
        "HFR_JIRA_BASE_URL": f"{base_url}/jira",
        "HFR_K8S_RESOURCE_URL": f"{base_url}/k8s/apis/apps/v1/namespaces/prod/deployments",
        "LINEAR_API_KEY": "linear-token",
        "HFR_LINEAR_BASE_URL": f"{base_url}/linear/graphql",
        "MICROSOFT_GRAPH_TOKEN": "graph-token",
        "HFR_MICROSOFT_GRAPH_BASE_URL": f"{base_url}/graph/v1.0",
        "HFR_GRAPH_MAIL_FOLDER_ID": "SentItems",
        "NOTION_TOKEN": "notion-token",
        "HFR_NOTION_DATABASE_ID": "db1",
        "HFR_NOTION_BASE_URL": f"{base_url}/notion/v1",
        "PAGERDUTY_API_TOKEN": "pagerduty-token",
        "HFR_PAGERDUTY_BASE_URL": f"{base_url}/pagerduty",
        "HFR_S3_BUCKET": "demo",
        "HFR_S3_URL": f"{base_url}/s3",
        "HFR_S3_UNSIGNED": "true",
        "SLACK_BOT_TOKEN": "slack-token",
        "HFR_SLACK_CHANNEL_ID": "C123",
        "HFR_SLACK_BASE_URL": f"{base_url}/slack",
        "STRIPE_SECRET_KEY": "stripe-token",
        "HFR_STRIPE_BASE_URL": f"{base_url}/stripe/v1",
        "HFR_STRIPE_OBJECT_ID": "pi_123",
        "ZENDESK_API_TOKEN": "zendesk-token",
        "HFR_ZENDESK_BASE_URL": f"{base_url}/zendesk/api/v2",
        "HFR_ZENDESK_TICKET_ID": "42",
    }


def _secret_values(env: dict[str, str]) -> list[str]:
    return [value for key, value in env.items() if "TOKEN" in key or "PASSWORD" in key or "SECRET" in key]


def _email_bytes() -> bytes:
    return (
        "From: agent@example.test\n"
        "To: customer@example.test\n"
        "Subject: Re: customer\n"
        "Message-ID: <imap-msg@example.test>\n"
        "\n"
        "Reply sent.\n"
    ).encode("utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
