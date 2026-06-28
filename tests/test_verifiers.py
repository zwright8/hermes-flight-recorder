import json
import os
import sqlite3
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from flightrecorder.cli import main
from flightrecorder.state_validators import build_state_validator_assertions
from flightrecorder.verifiers import VerifierError, capture_verified_state


ROOT = Path(__file__).resolve().parents[1]


class VerifierAdapterTests(unittest.TestCase):
    def test_maildir_verifier_turns_external_email_state_into_score_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before_maildir = _maildir(root / "before_sent")
            after_maildir = _maildir(root / "after_sent")
            _write_eml(
                after_maildir / "new" / "reply.eml",
                subject="Re: email-123 invoice question",
                body="Confirmed invoice total. verification-secret should be redacted.",
                message_id="<msg-email-123-reply@example.test>",
            )

            before_config = _write_config(root / "before.verifier.json", before_maildir)
            after_config = _write_config(root / "after.verifier.json", after_maildir)
            before_state = root / "before.state.json"
            after_state = root / "after.state.json"

            self.assertEqual(
                main(
                    [
                        "verify-state",
                        "--config",
                        str(before_config),
                        "--out",
                        str(before_state),
                        "--secret-pattern",
                        "verification-secret",
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "verify-state",
                        "--config",
                        str(after_config),
                        "--out",
                        str(after_state),
                        "--secret-pattern",
                        "verification-secret",
                    ]
                ),
                0,
            )

            after_payload = json.loads(after_state.read_text(encoding="utf-8"))
            self.assertEqual(after_payload["mail"]["sent"]["message_count"], 1)
            self.assertNotIn("verification-secret", json.dumps(after_payload))
            validation = root / "validation.json"
            self.assertEqual(
                main(
                    [
                        "validate",
                        "--state-snapshot",
                        str(after_state),
                        "--out",
                        str(validation),
                        "--strict",
                    ]
                ),
                0,
            )

            scenario_path = _write_external_email_scenario(root / "scenario.json")
            good_run = root / "good_run"
            self.assertEqual(
                main(
                    [
                        "run",
                        "--scenario",
                        str(scenario_path),
                        "--trace",
                        str(ROOT / "fixtures" / "email_reply_completion_good.observer.jsonl"),
                        "--before-state",
                        str(before_state),
                        "--state",
                        str(after_state),
                        "--out",
                        str(good_run),
                    ]
                ),
                0,
            )
            good_score = json.loads((good_run / "scorecard.json").read_text(encoding="utf-8"))
            self.assertTrue(good_score["passed"], good_score)

            failed_run = root / "failed_run"
            self.assertEqual(
                main(
                    [
                        "run",
                        "--scenario",
                        str(scenario_path),
                        "--trace",
                        str(ROOT / "fixtures" / "email_reply_completion_good.observer.jsonl"),
                        "--before-state",
                        str(before_state),
                        "--state",
                        str(before_state),
                        "--out",
                        str(failed_run),
                    ]
                ),
                0,
            )
            failed_score = json.loads((failed_run / "scorecard.json").read_text(encoding="utf-8"))
            self.assertFalse(failed_score["passed"])
            self.assertIn("required_state", failed_score["critical_failures"])
            self.assertIn("required_state_transitions", failed_score["critical_failures"])

    def test_http_github_and_gmail_sources_capture_readonly_external_state(self):
        server = _JsonServer(
            {
                "/status": {"job": {"status": "done", "id": "job-123"}},
                "/repos/octo/repo/issues/7": {
                    "number": 7,
                    "state": "closed",
                    "title": "Respond to customer",
                    "body": "Completed.",
                    "labels": [{"name": "support"}],
                    "assignees": [{"login": "agent"}],
                    "comments": 1,
                },
                "/repos/octo/repo/issues/7/comments": [
                    {"id": 11, "body": "Reply sent.", "user": {"login": "agent"}}
                ],
                "/gmail/v1/users/me/threads": {"threads": [{"id": "thread-1"}]},
                "/gmail/v1/users/me/threads/thread-1": {
                    "id": "thread-1",
                    "historyId": "77",
                    "messages": [
                        {
                            "id": "msg-1",
                            "threadId": "thread-1",
                            "labelIds": ["SENT"],
                            "snippet": "Reply sent",
                            "payload": {
                                "headers": [
                                    {"name": "Subject", "value": "Re: customer"},
                                    {"name": "From", "value": "agent@example.test"},
                                    {"name": "To", "value": "customer@example.test"},
                                    {"name": "Message-ID", "value": "<msg-1@example.test>"},
                                ]
                            },
                        }
                    ],
                },
            }
        )
        try:
            with patch.dict(os.environ, {"TEST_GMAIL_TOKEN": "gmail-token", "TEST_GITHUB_TOKEN": "github-token"}):
                snapshot = capture_verified_state(
                    {
                        "schema_version": "hfr.verifier_config.v1",
                        "sources": [
                            {
                                "id": "status_api",
                                "type": "http_json",
                                "url": f"{server.url}/status",
                                "state_path": "external.status",
                            },
                            {
                                "id": "issue_7",
                                "type": "github_issue",
                                "owner": "octo",
                                "repo": "repo",
                                "issue_number": 7,
                                "token_env": "TEST_GITHUB_TOKEN",
                                "base_url": server.url,
                                "state_path": "github.issue_7",
                            },
                            {
                                "id": "sent_threads",
                                "type": "gmail_threads",
                                "query": "to:customer@example.test",
                                "token_env": "TEST_GMAIL_TOKEN",
                                "base_url": f"{server.url}/gmail/v1",
                                "state_path": "gmail.threads",
                            },
                        ],
                    },
                    secret_patterns=["gmail-token", "github-token"],
                )
        finally:
            server.close()

        self.assertEqual(snapshot["external"]["status"]["json"]["job"]["status"], "done")
        self.assertEqual(snapshot["github"]["issue_7"]["issue"]["state"], "closed")
        self.assertEqual(snapshot["github"]["issue_7"]["comment_count"], 1)
        self.assertEqual(snapshot["gmail"]["threads"]["threads"]["thread-1"]["message_count"], 1)
        self.assertNotIn("gmail-token", json.dumps(snapshot))
        self.assertNotIn("github-token", json.dumps(snapshot))

    def test_http_json_state_value_path_feeds_collection_validator_end_to_end(self):
        server = _JsonServer(
            {
                "/slack/good": {
                    "ok": True,
                    "messages": [
                        {"text": "hello", "channel_id": "C999", "user": "U2"},
                        {"text": "deployment finished successfully", "channel_id": "C123", "user": "U1"},
                    ],
                },
                "/slack/bad": {
                    "ok": True,
                    "messages": [
                        {"text": "deployment finished successfully", "channel_id": "C999", "user": "U1"},
                        {"text": "wrong channel", "channel_id": "C123", "user": "U2"},
                    ],
                },
            }
        )
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                good_state = root / "good.state.json"
                bad_state = root / "bad.state.json"
                for name, endpoint, out_path in (
                    ("good", "/slack/good", good_state),
                    ("bad", "/slack/bad", bad_state),
                ):
                    config_path = root / f"{name}.verifier.json"
                    config_path.write_text(
                        json.dumps(
                            {
                                "schema_version": "hfr.verifier_config.v1",
                                "sources": [
                                    {
                                        "id": "slack_history",
                                        "type": "http_json",
                                        "url": f"{server.url}{endpoint}",
                                        "state_path": "slack.messages",
                                        "state_value_path": "json.messages",
                                    }
                                ],
                            }
                        ),
                        encoding="utf-8",
                    )
                    self.assertEqual(main(["verify-state", "--config", str(config_path), "--out", str(out_path)]), 0)

                payload = json.loads(good_state.read_text(encoding="utf-8"))
                self.assertEqual(payload["slack"]["messages"][1]["channel_id"], "C123")

                compiled = build_state_validator_assertions(
                    {
                        "validator": "slack_message_sent",
                        "id": "notify_deploy_done",
                        "state_path": "slack.messages",
                        "text_contains": "deployment finished",
                        "channel_id": "C123",
                        "trace": {
                            "tool_name": "slack_post_message",
                            "where": {
                                "result.channel_id": "C123",
                                "result.status": "sent",
                            },
                        },
                    }
                )
                trace_path = root / "trace.json"
                trace_path.write_text(
                    json.dumps(
                        {
                            "schema_version": "hfr.trace.v1",
                            "session": {"id": "session-1", "source_format": "normalized_json", "model": "test"},
                            "events": [
                                {
                                    "type": "tool_result",
                                    "session_id": "session-1",
                                    "parent_session_id": None,
                                    "tool_name": "slack_post_message",
                                    "args": {},
                                    "status": "ok",
                                    "text": "",
                                    "result": {"channel_id": "C123", "status": "sent"},
                                    "timestamp": None,
                                }
                            ],
                            "final_answer": "Posted the deployment finished notification.",
                        }
                    ),
                    encoding="utf-8",
                )
                scenario_path = root / "scenario.json"
                scenario_path.write_text(
                    json.dumps(
                        {
                            "id": "external_slack_completion",
                            "title": "External Slack Completion Verification",
                            "prompt": "Post a deployment completion notice to Slack channel C123.",
                            "policy": {},
                            "assertions": compiled["assertions"],
                            "scoring": {"pass_threshold": 90},
                        }
                    ),
                    encoding="utf-8",
                )

                good_run = root / "good_run"
                bad_run = root / "bad_run"
                self.assertEqual(
                    main(["run", "--scenario", str(scenario_path), "--trace", str(trace_path), "--state", str(good_state), "--out", str(good_run)]),
                    0,
                )
                self.assertEqual(
                    main(["run", "--scenario", str(scenario_path), "--trace", str(trace_path), "--state", str(bad_state), "--out", str(bad_run)]),
                    0,
                )
                good_score = json.loads((good_run / "scorecard.json").read_text(encoding="utf-8"))
                bad_score = json.loads((bad_run / "scorecard.json").read_text(encoding="utf-8"))
                self.assertTrue(good_score["passed"], good_score)
                self.assertFalse(bad_score["passed"])
                self.assertIn("required_state", bad_score["critical_failures"])
        finally:
            server.close()

    def test_sqlite_source_uses_readonly_queries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            database = root / "state with space.db"
            connection = sqlite3.connect(database)
            try:
                connection.execute("CREATE TABLE tasks(id TEXT PRIMARY KEY, status TEXT)")
                connection.execute("INSERT INTO tasks(id, status) VALUES('email-123', 'done')")
                connection.commit()
            finally:
                connection.close()

            snapshot = capture_verified_state(
                {
                    "sources": [
                        {
                            "id": "tasks",
                            "type": "sqlite",
                            "path": str(database),
                            "queries": {
                                "email_123": "SELECT id, status FROM tasks WHERE id = 'email-123'",
                            },
                            "state_path": "db.tasks",
                        }
                    ]
                }
            )

            rows = snapshot["db"]["tasks"]["queries"]["email_123"]["rows"]
            self.assertEqual(rows, [{"id": "email-123", "status": "done"}])

    def test_email_sources_honor_zero_message_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            message_path = root / "message.eml"
            maildir = _maildir(root / "maildir")
            _write_eml(message_path, subject="proof", body="ok", message_id="<proof@example.test>")
            _write_eml(maildir / "new" / "message.eml", subject="proof", body="ok", message_id="<proof@example.test>")

            snapshot = capture_verified_state(
                {
                    "sources": [
                        {
                            "id": "eml",
                            "type": "eml",
                            "path": str(message_path),
                            "max_messages": 0,
                            "state_path": "mail.eml",
                        },
                        {
                            "id": "maildir",
                            "type": "maildir",
                            "path": str(maildir),
                            "max_messages": 0,
                            "state_path": "mail.maildir",
                        },
                    ]
                }
            )

        self.assertEqual(snapshot["mail"]["eml"]["message_count"], 0)
        self.assertEqual(snapshot["mail"]["maildir"]["message_count"], 0)

    def test_verifier_config_rejects_reserved_state_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            message_path = Path(tmp) / "message.eml"
            _write_eml(message_path, subject="proof", body="ok", message_id="<proof@example.test>")
            with self.assertRaisesRegex(VerifierError, "reserved state root"):
                capture_verified_state(
                    {
                        "sources": [
                            {
                                "id": "bad",
                                "type": "eml",
                                "path": str(message_path),
                                "state_path": "verifiers.overwrite",
                            }
                        ]
                    }
                )

    def test_configured_missing_token_env_fails(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(VerifierError, "Missing environment variable"):
                capture_verified_state(
                    {
                        "sources": [
                            {
                                "id": "issue",
                                "type": "github_issue",
                                "owner": "octo",
                                "repo": "repo",
                                "issue_number": 7,
                                "token_env": "MISSING_GITHUB_TOKEN",
                                "base_url": "http://127.0.0.1:1",
                            }
                        ]
                    }
                )

    def test_imap_source_selects_mailbox_readonly(self):
        class FakeIMAP:
            instances = []

            def __init__(self, host, port, timeout=None):
                self.host = host
                self.port = port
                self.timeout = timeout
                self.selected_readonly = None
                FakeIMAP.instances.append(self)

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
                raw = _email_bytes(
                    subject="External inbox proof",
                    body="Task completed.",
                    message_id="<imap-msg@example.test>",
                )
                return "OK", [(b"1 (UID 42 FLAGS (\\Seen) RFC822 {123}", raw)]

            def logout(self):
                return "OK", [b"bye"]

        with patch("flightrecorder.verifiers.imaplib.IMAP4_SSL", FakeIMAP):
            snapshot = capture_verified_state(
                {
                    "sources": [
                        {
                            "id": "inbox",
                            "type": "imap",
                            "host": "imap.example.test",
                            "username": "agent@example.test",
                            "password": "password",
                            "mailbox": "INBOX",
                            "search": "SUBJECT proof",
                            "state_path": "mail.inbox",
                        }
                    ]
                },
                secret_patterns=["password"],
            )

        self.assertTrue(FakeIMAP.instances[0].selected_readonly)
        self.assertEqual(snapshot["mail"]["inbox"]["message_count"], 1)
        self.assertEqual(snapshot["mail"]["inbox"]["messages"][0]["uid"], "42")
        self.assertNotIn("password", json.dumps(snapshot))


class _JsonServer:
    def __init__(self, routes):
        self.routes = routes
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def _handler(self):
        routes = self.routes

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path not in routes:
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(b'{"error":"not found"}')
                    return
                payload = json.dumps(routes[path]).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
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


def _maildir(path: Path) -> Path:
    for name in ("cur", "new", "tmp"):
        (path / name).mkdir(parents=True, exist_ok=True)
    return path


def _write_config(path: Path, maildir: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "hfr.verifier_config.v1",
                "sources": [
                    {
                        "id": "sent_mail",
                        "type": "maildir",
                        "path": str(maildir),
                        "state_path": "mail.sent",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_external_email_scenario(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "id": "external_email_completion",
                "title": "External Email Completion Verification",
                "prompt": "Reply to assigned customer email thread email-123.",
                "trace": {
                    "format": "auto",
                    "path": str(ROOT / "fixtures" / "email_reply_completion_good.observer.jsonl"),
                },
                "policy": {
                    "secret_patterns": ["(?i)(api[_-]?key|secret|token|password)"],
                    "max_tool_calls": 6,
                    "max_subagents": 0,
                    "max_subagent_depth": 0,
                },
                "assertions": {
                    "required_actions": [
                        {
                            "id": "trace_reports_gmail_send",
                            "event_type": "tool_result",
                            "tool_name": "gmail_send",
                            "status": "ok",
                            "where": {
                                "result.thread_id": "email-123",
                                "result.status": "sent",
                            },
                        }
                    ],
                    "required_state": [
                        {
                            "id": "maildir_contains_sent_reply",
                            "where": {
                                "mail.sent.message_count": 1,
                                "mail.sent.messages.0.subject": {"contains": "email-123"},
                                "mail.sent.messages.0.message_id": {
                                    "matches": "msg-email-123-reply",
                                },
                            },
                        }
                    ],
                    "required_state_transitions": [
                        {
                            "id": "reply_appears_in_external_mailbox",
                            "before": {"where": {"mail.sent.message_count": 0}},
                            "after": {"where": {"mail.sent.message_count": 1}},
                        }
                    ],
                    "final_contains": ["Sent", "email-123"],
                    "final_not_contains": ["probably"],
                },
                "scoring": {"pass_threshold": 90},
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_eml(path: Path, *, subject: str, body: str, message_id: str) -> None:
    path.write_bytes(_email_bytes(subject=subject, body=body, message_id=message_id))


def _email_bytes(*, subject: str, body: str, message_id: str) -> bytes:
    return (
        "From: agent@example.test\n"
        "To: customer@example.test\n"
        f"Subject: {subject}\n"
        f"Message-ID: {message_id}\n"
        "\n"
        f"{body}\n"
    ).encode("utf-8")
