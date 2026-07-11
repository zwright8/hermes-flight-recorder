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
CUSTOM_ORIGIN_FLAGS = {
    "discord": "HFR_DISCORD_ALLOW_CUSTOM_ORIGIN",
    "github": "HFR_GITHUB_ALLOW_CUSTOM_ORIGIN",
    "gitlab": "HFR_GITLAB_ALLOW_CUSTOM_ORIGIN",
    "gmail": "HFR_GMAIL_ALLOW_CUSTOM_ORIGIN",
    "google_calendar": "HFR_GOOGLE_CALENDAR_ALLOW_CUSTOM_ORIGIN",
    "google_drive": "HFR_GOOGLE_DRIVE_ALLOW_CUSTOM_ORIGIN",
    "imap": "HFR_IMAP_ALLOW_CUSTOM_ORIGIN",
    "jira": "HFR_JIRA_ALLOW_CUSTOM_ORIGIN",
    "kubernetes": "HFR_K8S_ALLOW_CUSTOM_ORIGIN",
    "linear": "HFR_LINEAR_ALLOW_CUSTOM_ORIGIN",
    "microsoft_graph_events": "HFR_MICROSOFT_GRAPH_ALLOW_CUSTOM_ORIGIN",
    "microsoft_graph_messages": "HFR_MICROSOFT_GRAPH_ALLOW_CUSTOM_ORIGIN",
    "notion": "HFR_NOTION_ALLOW_CUSTOM_ORIGIN",
    "pagerduty": "HFR_PAGERDUTY_ALLOW_CUSTOM_ORIGIN",
    "s3": "HFR_S3_ALLOW_CUSTOM_ORIGIN",
    "slack": "HFR_SLACK_ALLOW_CUSTOM_ORIGIN",
    "stripe": "HFR_STRIPE_ALLOW_CUSTOM_ORIGIN",
    "zendesk": "HFR_ZENDESK_ALLOW_CUSTOM_ORIGIN",
}


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

    def test_each_custom_base_requires_its_provider_opt_in(self):
        live_smoke = _load_script()
        specs = live_smoke._provider_specs_by_id()
        env_without_opt_ins = _provider_env("https://provider.example.test")
        for flag in set(CUSTOM_ORIGIN_FLAGS.values()):
            env_without_opt_ins.pop(flag)

        for provider, flag in CUSTOM_ORIGIN_FLAGS.items():
            with self.subTest(provider=provider):
                source = specs[provider].build_source(env_without_opt_ins)
                self.assertFalse(source["allow_custom_origin"])

                opted_in_env = {**env_without_opt_ins, flag: "yes"}
                opted_in_source = specs[provider].build_source(opted_in_env)
                self.assertTrue(opted_in_source["allow_custom_origin"])
                if provider == "imap":
                    self.assertEqual(opted_in_source["username_env"], "IMAP_USERNAME")
                    self.assertEqual(opted_in_source["password_env"], "IMAP_PASSWORD")
                elif provider == "kubernetes":
                    self.assertEqual(opted_in_source["bearer_token_env"], "KUBERNETES_BEARER_TOKEN")
                elif provider == "s3":
                    self.assertEqual(opted_in_source["access_key_env"], "AWS_ACCESS_KEY_ID")
                    self.assertEqual(opted_in_source["secret_key_env"], "AWS_SECRET_ACCESS_KEY")
                    self.assertEqual(opted_in_source["session_token_env"], "AWS_SESSION_TOKEN")

    def test_custom_base_url_alone_does_not_authorize_credentials(self):
        live_smoke = _load_script()
        server = _JsonServer(_provider_routes())
        env = {
            "SLACK_BOT_TOKEN": "slack-token",
            "HFR_SLACK_CHANNEL_ID": "C123",
            "HFR_SLACK_BASE_URL": f"{server.url}/slack",
        }
        try:
            with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, env, clear=True):
                with redirect_stdout(StringIO()):
                    code = live_smoke.main(["--out", tmp, "--provider", "slack", "--allow-network"])

                self.assertEqual(code, 1)
                self.assertEqual(server.requests, [])
                summary = _read_json(Path(tmp) / "live_verifier_smoke_summary.json")
                self.assertFalse(summary["passed"])
                self.assertEqual(summary["providers"][0]["status"], "failed")
                self.assertIn("allow_custom_origin=true", summary["providers"][0]["error"])
                config = _read_json(Path(tmp) / "slack" / "verifier_config.json")
                self.assertFalse(config["sources"][0]["allow_custom_origin"])
        finally:
            server.close()

    def test_signed_s3_custom_url_alone_does_not_authorize_aws_credentials(self):
        live_smoke = _load_script()
        server = _JsonServer(_provider_routes())
        env = {
            "AWS_ACCESS_KEY_ID": "default-access-key",
            "AWS_SECRET_ACCESS_KEY": "default-secret-key",
            "AWS_SESSION_TOKEN": "default-session-token",
            "HFR_S3_BUCKET": "demo",
            "HFR_S3_URL": f"{server.url}/s3",
        }
        try:
            with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, env, clear=True):
                with redirect_stdout(StringIO()):
                    code = live_smoke.main(["--out", tmp, "--provider", "s3", "--allow-network"])

                self.assertEqual(code, 1)
                self.assertEqual(server.requests, [])
                summary = _read_json(Path(tmp) / "live_verifier_smoke_summary.json")
                self.assertFalse(summary["passed"])
                self.assertEqual(summary["providers"][0]["status"], "failed")
                self.assertIn("allow_custom_origin=true", summary["providers"][0]["error"])
                config = _read_json(Path(tmp) / "s3" / "verifier_config.json")
                self.assertFalse(config["sources"][0]["allow_custom_origin"])
        finally:
            server.close()

    def test_imap_host_alone_does_not_authorize_mailbox_credentials(self):
        live_smoke = _load_script()
        env = {
            "IMAP_HOST": "imap.attacker.example.test",
            "IMAP_USERNAME": "agent@example.test",
            "IMAP_PASSWORD": "private-password",
        }
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, env, clear=True):
            with patch("flightrecorder.verifiers.imaplib.IMAP4_SSL") as client:
                with redirect_stdout(StringIO()):
                    code = live_smoke.main(["--out", tmp, "--provider", "imap", "--allow-network"])

            self.assertEqual(code, 1)
            client.assert_not_called()
            summary = _read_json(Path(tmp) / "live_verifier_smoke_summary.json")
            self.assertFalse(summary["passed"])
            self.assertEqual(summary["providers"][0]["status"], "failed")
            self.assertIn("allow_custom_origin=true", summary["providers"][0]["error"])
            config = _read_json(Path(tmp) / "imap" / "verifier_config.json")
            self.assertFalse(config["sources"][0]["allow_custom_origin"])

    def test_kubernetes_url_alone_does_not_authorize_bearer_credentials(self):
        live_smoke = _load_script()
        server = _JsonServer(_provider_routes())
        env = {
            "HFR_K8S_RESOURCE_URL": f"{server.url}/k8s/apis/apps/v1/namespaces/prod/deployments",
            "KUBERNETES_BEARER_TOKEN": "private-token",
        }
        try:
            with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, env, clear=True):
                with redirect_stdout(StringIO()):
                    code = live_smoke.main(["--out", tmp, "--provider", "kubernetes", "--allow-network"])

                self.assertEqual(code, 1)
                self.assertEqual(server.requests, [])
                summary = _read_json(Path(tmp) / "live_verifier_smoke_summary.json")
                self.assertFalse(summary["passed"])
                self.assertEqual(summary["providers"][0]["status"], "failed")
                self.assertIn("allow_custom_origin=true", summary["providers"][0]["error"])
                config = _read_json(Path(tmp) / "kubernetes" / "verifier_config.json")
                self.assertFalse(config["sources"][0]["allow_custom_origin"])
        finally:
            server.close()

    def test_all_http_provider_smokes_pass_against_readonly_local_endpoints(self):
        live_smoke = _load_script()
        server = _JsonServer(_provider_routes())
        FakeIMAP.instances.clear()
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
                    config = _read_json(Path(artifacts["verifier_config"]))
                    if record["provider"] in CUSTOM_ORIGIN_FLAGS:
                        self.assertTrue(config["sources"][0]["allow_custom_origin"], record)
                    if record["provider"] == "s3":
                        self.assertEqual(config["sources"][0]["access_key_env"], "AWS_ACCESS_KEY_ID")
                        self.assertIn("secret_key_env", config["sources"][0])
                        self.assertIn("session_token_env", config["sources"][0])
                    validation = _read_json(Path(artifacts["validation"]))
                    self.assertTrue(validation["passed"], validation)

                s3_request = next(request for request in server.requests if request["path"] == "/s3")
                self.assertIn("Credential=aws-access/", s3_request["authorization"])
                self.assertEqual(s3_request["session_token"], "aws-session-token")
                kubernetes_request = next(request for request in server.requests if request["path"].startswith("/k8s/"))
                self.assertEqual(kubernetes_request["authorization"], "Bearer k8s-token")
                self.assertEqual(len(FakeIMAP.instances), 1)
                self.assertEqual(FakeIMAP.instances[0].username, "agent@example.test")
                self.assertEqual(FakeIMAP.instances[0].password, "imap-password")

                rendered = "\n".join(path.read_text(encoding="utf-8") for path in out.rglob("*.json"))
                for secret in _secret_values(env):
                    self.assertNotIn(secret, rendered)
        finally:
            server.close()


class FakeIMAP:
    instances = []

    def __init__(self, host, port, timeout=None):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.selected_readonly = None
        self.instances.append(self)

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
                requests.append(
                    {
                        "method": "GET",
                        "path": path,
                        "query": query,
                        "authorization": self.headers.get("Authorization"),
                        "session_token": self.headers.get("x-amz-security-token"),
                    }
                )
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
    env = {
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
        "KUBERNETES_BEARER_TOKEN": "k8s-token",
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
        "AWS_ACCESS_KEY_ID": "aws-access",
        "AWS_SECRET_ACCESS_KEY": "aws-secret-key",
        "AWS_SESSION_TOKEN": "aws-session-token",
        "HFR_S3_BUCKET": "demo",
        "HFR_S3_URL": f"{base_url}/s3",
        "HFR_S3_UNSIGNED": "false",
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
    env.update({flag: "1" for flag in set(CUSTOM_ORIGIN_FLAGS.values())})
    return env


def _secret_values(env: dict[str, str]) -> list[str]:
    return [
        value
        for key, value in env.items()
        if "TOKEN" in key or "PASSWORD" in key or "SECRET" in key or "ACCESS_KEY" in key
    ]


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
