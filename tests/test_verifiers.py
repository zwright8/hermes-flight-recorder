import io
import json
import os
import sqlite3
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from flightrecorder.cli import main
from flightrecorder.safe_http import (
    HttpStatusError,
    SafeHttpError,
    SafeRedirectHandler,
    bounded_http_request,
    read_bounded_sse,
)
from flightrecorder.state_validators import build_state_validator_assertions
from flightrecorder.verifiers import VerifierError, capture_verified_state


ROOT = Path(__file__).resolve().parents[1]


class VerifierAdapterTests(unittest.TestCase):
    def test_http_policy_rejects_https_downgrade_redirects(self):
        request = urllib.request.Request(
            "https://example.test/private",
            headers={"Authorization": "Bearer top-secret"},
        )

        with self.assertRaisesRegex(SafeHttpError, "HTTPS downgrade redirect"):
            SafeRedirectHandler().redirect_request(
                request,
                None,
                302,
                "Found",
                {},
                "http://example.test/public",
            )

    def test_verifier_rejects_oversized_success_body(self):
        server = _JsonServer({"/oversized": b"x" * 65})
        try:
            with self.assertRaisesRegex(VerifierError, "exceeded max_bytes=64"):
                capture_verified_state(
                    {
                        "sources": [
                            {
                                "id": "oversized",
                                "type": "http_json",
                                "url": f"{server.url}/oversized",
                                "max_bytes": 64,
                            }
                        ]
                    }
                )
        finally:
            server.close()

    def test_required_http_verifier_does_not_render_error_response_body(self):
        error = HttpStatusError(
            503,
            b'{"api_key":"private-error-value"}',
            truncated=False,
        )
        with patch("flightrecorder.verifiers.bounded_http_request", side_effect=error):
            with self.assertRaises(VerifierError) as raised:
                capture_verified_state(
                    {
                        "sources": [
                            {
                                "id": "remote",
                                "type": "http_json",
                                "url": "https://example.test/state",
                            }
                        ]
                    }
                )

        rendered = str(raised.exception)
        self.assertIn("status 503", rendered)
        self.assertNotIn("private-error-value", rendered)

    def test_invalid_json_verifier_error_omits_url_credentials_and_query(self):
        url = "https://user:private-password@example.test/state?api_key=private-query-value"
        with patch(
            "flightrecorder.verifiers.bounded_http_request",
            return_value=(200, b"not-json"),
        ):
            with self.assertRaises(VerifierError) as raised:
                capture_verified_state(
                    {
                        "sources": [
                            {
                                "id": "remote",
                                "type": "http_json",
                                "url": url,
                            }
                        ]
                    }
                )

        rendered = str(raised.exception)
        self.assertIn("did not return valid JSON", rendered)
        self.assertNotIn("private-password", rendered)
        self.assertNotIn("private-query-value", rendered)

    def test_verifier_rejects_duplicate_source_ids_before_capture(self):
        config = {
            "sources": [
                {"id": "duplicate", "type": "http_json", "url": "https://example.test/one"},
                {"id": "duplicate", "type": "http_json", "url": "https://example.test/two"},
            ]
        }

        with patch("flightrecorder.verifiers._capture_source") as capture:
            with self.assertRaisesRegex(VerifierError, "duplicate verifier source id"):
                capture_verified_state(config)

        capture.assert_not_called()

    def test_verifier_rejects_non_http_url_before_opening_it(self):
        with patch("flightrecorder.safe_http.urllib.request.build_opener") as build_opener:
            with self.assertRaisesRegex(VerifierError, "must use http or https"):
                capture_verified_state(
                    {
                        "sources": [
                            {
                                "id": "local_file",
                                "type": "http_json",
                                "url": "file:///tmp/private.json",
                            }
                        ]
                    }
                )

        build_opener.assert_not_called()

    def test_verifier_blocks_credentialed_cross_origin_redirects(self):
        sink = _JsonServer({"/sink": {"ok": True}})
        redirector = _JsonServer({}, redirects={"/redirect": f"{sink.url}/sink"})
        try:
            with patch.dict(os.environ, {"TEST_REDIRECT_TOKEN": "top-secret"}):
                with self.assertRaisesRegex(VerifierError, "credentialed cross-origin redirect"):
                    capture_verified_state(
                        {
                            "sources": [
                                {
                                    "id": "redirect",
                                    "type": "http_json",
                                    "url": f"{redirector.url}/redirect",
                                    "bearer_token_env": "TEST_REDIRECT_TOKEN",
                                }
                            ]
                        }
                    )
        finally:
            redirector.close()
            sink.close()

        self.assertEqual(sink.requests, [])

    def test_slack_default_token_is_not_sent_to_custom_origin(self):
        attacker = _JsonServer({"/slack/conversations.history": {"ok": True, "messages": []}})
        try:
            with patch.dict(os.environ, {"SLACK_BOT_TOKEN": "private-default-token"}):
                with self.assertRaisesRegex(VerifierError, "allow_custom_origin=true"):
                    capture_verified_state(
                        {
                            "sources": [
                                {
                                    "id": "slack",
                                    "type": "slack_history",
                                    "base_url": f"{attacker.url}/slack",
                                    "channel_id": "C123",
                                }
                            ]
                        }
                    )
        finally:
            attacker.close()

        self.assertEqual(attacker.requests, [])

    def test_custom_provider_origin_requires_explicit_credential_config(self):
        with patch.dict(os.environ, {"SLACK_BOT_TOKEN": "private-default-token"}):
            with patch("flightrecorder.verifiers.bounded_http_request") as request:
                with self.assertRaisesRegex(VerifierError, "explicit credential configuration"):
                    capture_verified_state(
                        {
                            "sources": [
                                {
                                    "id": "slack",
                                    "type": "slack_history",
                                    "base_url": "https://attacker.example.test/slack",
                                    "allow_custom_origin": True,
                                    "channel_id": "C123",
                                }
                            ]
                        }
                    )

        request.assert_not_called()

    def test_provider_adapters_reject_custom_origins_without_opt_in(self):
        custom_base = "https://attacker.example.test"
        cases = [
            ("slack", {"type": "slack_history", "channel_id": "C123"}),
            ("calendar", {"type": "google_calendar_events"}),
            ("drive", {"type": "google_drive_files"}),
            ("gmail", {"type": "gmail_threads"}),
            ("stripe", {"type": "stripe_objects", "resource": "payment_intents"}),
            ("notion", {"type": "notion_database", "database_id": "db1"}),
            ("linear", {"type": "linear_issues"}),
            ("graph_messages", {"type": "microsoft_graph_messages"}),
            ("graph_events", {"type": "microsoft_graph_events"}),
            (
                "github",
                {
                    "type": "github_issue",
                    "owner": "octo",
                    "repo": "repo",
                    "issue_number": 7,
                    "token_env": "TEST_GITHUB_TOKEN",
                },
            ),
            ("gitlab", {"type": "gitlab_issues", "project_id": "group/project"}),
            ("discord", {"type": "discord_messages", "channel_id": "C123"}),
            ("pagerduty", {"type": "pagerduty_incidents"}),
            ("jira", {"type": "jira_issues"}),
            ("zendesk", {"type": "zendesk_tickets"}),
        ]

        with patch("flightrecorder.verifiers.bounded_http_request") as request:
            for source_id, source in cases:
                configured_source = {"id": source_id, "base_url": custom_base, **source}
                with self.subTest(source_type=source["type"]):
                    with self.assertRaisesRegex(VerifierError, "allow_custom_origin=true"):
                        capture_verified_state({"sources": [configured_source]})

        request.assert_not_called()

    def test_signed_s3_custom_origin_rejects_default_aws_credentials_before_request(self):
        server = _JsonServer({"/s3": _s3_listing_xml()})
        try:
            with patch.dict(
                os.environ,
                {
                    "AWS_ACCESS_KEY_ID": "default-access-key",
                    "AWS_SECRET_ACCESS_KEY": "default-secret-key",
                    "AWS_SESSION_TOKEN": "default-session-token",
                },
                clear=True,
            ):
                for location in ({"url": f"{server.url}/s3"}, {"endpoint_url": server.url}):
                    with self.subTest(location=next(iter(location))):
                        with self.assertRaisesRegex(VerifierError, "allow_custom_origin=true"):
                            capture_verified_state(
                                {
                                    "sources": [
                                        {
                                            "id": "s3",
                                            "type": "s3_objects",
                                            "bucket": "demo",
                                            **location,
                                        }
                                    ]
                                }
                            )
        finally:
            server.close()

        self.assertEqual(server.requests, [])

    def test_signed_s3_custom_origin_requires_explicit_access_and_secret_env_names(self):
        with patch.dict(
            os.environ,
            {
                "AWS_ACCESS_KEY_ID": "default-access-key",
                "AWS_SECRET_ACCESS_KEY": "default-secret-key",
                "AWS_SESSION_TOKEN": "default-session-token",
            },
            clear=True,
        ):
            with patch("flightrecorder.verifiers.bounded_http_request") as request:
                with self.assertRaisesRegex(VerifierError, "explicit access_key_env and secret_key_env"):
                    capture_verified_state(
                        {
                            "sources": [
                                {
                                    "id": "s3",
                                    "type": "s3_objects",
                                    "bucket": "demo",
                                    "url": "https://storage.example.test/demo",
                                    "allow_custom_origin": True,
                                }
                            ]
                        }
                    )

        request.assert_not_called()

    def test_signed_s3_custom_origin_uses_only_explicitly_named_credentials(self):
        with patch.dict(
            os.environ,
            {
                "TEST_S3_ACCESS_KEY": "explicit-access-key",
                "TEST_S3_SECRET_KEY": "explicit-secret-key",
                "TEST_S3_SESSION_TOKEN": "explicit-session-token",
                "AWS_SESSION_TOKEN": "default-session-token",
            },
            clear=True,
        ):
            with patch(
                "flightrecorder.verifiers.bounded_http_request",
                return_value=(200, _s3_listing_xml().encode("utf-8")),
            ) as request:
                snapshot = capture_verified_state(
                    {
                        "sources": [
                            {
                                "id": "s3",
                                "type": "s3_objects",
                                "bucket": "demo",
                                "url": "https://storage.example.test/demo",
                                "allow_custom_origin": True,
                                "access_key_env": "TEST_S3_ACCESS_KEY",
                                "secret_key_env": "TEST_S3_SECRET_KEY",
                                "session_token_env": "TEST_S3_SESSION_TOKEN",
                            }
                        ]
                    }
                )

        self.assertEqual(snapshot["verifiers"]["sources"]["s3"]["status"], "ok")
        headers = request.call_args.kwargs["headers"]
        self.assertIn("Credential=explicit-access-key/", headers["Authorization"])
        self.assertEqual(headers["x-amz-security-token"], "explicit-session-token")
        self.assertNotIn("default-session-token", json.dumps(headers))

    def test_signed_s3_custom_origin_suppresses_implicit_default_session_token(self):
        with patch.dict(
            os.environ,
            {
                "TEST_S3_ACCESS_KEY": "explicit-access-key",
                "TEST_S3_SECRET_KEY": "explicit-secret-key",
                "AWS_SESSION_TOKEN": "default-session-token",
            },
            clear=True,
        ):
            with patch(
                "flightrecorder.verifiers.bounded_http_request",
                return_value=(200, _s3_listing_xml().encode("utf-8")),
            ) as request:
                capture_verified_state(
                    {
                        "sources": [
                            {
                                "id": "s3",
                                "type": "s3_objects",
                                "bucket": "demo",
                                "url": "https://storage.example.test/demo",
                                "allow_custom_origin": True,
                                "access_key_env": "TEST_S3_ACCESS_KEY",
                                "secret_key_env": "TEST_S3_SECRET_KEY",
                            }
                        ]
                    }
                )

        self.assertNotIn("x-amz-security-token", request.call_args.kwargs["headers"])

    def test_unsigned_s3_custom_endpoint_needs_no_credential_consent(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "flightrecorder.verifiers.bounded_http_request",
                return_value=(200, _s3_listing_xml().encode("utf-8")),
            ) as request:
                snapshot = capture_verified_state(
                    {
                        "sources": [
                            {
                                "id": "s3",
                                "type": "s3_objects",
                                "bucket": "demo",
                                "endpoint_url": "https://storage.example.test",
                                "unsigned": True,
                            }
                        ]
                    }
                )

        self.assertEqual(snapshot["verifiers"]["sources"]["s3"]["status"], "ok")
        headers = request.call_args.kwargs["headers"]
        self.assertNotIn("Authorization", headers)
        self.assertNotIn("x-amz-security-token", headers)

    def test_s3_region_cannot_inject_a_non_aws_default_origin(self):
        with patch.dict(
            os.environ,
            {
                "AWS_ACCESS_KEY_ID": "default-access-key",
                "AWS_SECRET_ACCESS_KEY": "default-secret-key",
            },
            clear=True,
        ):
            with patch("flightrecorder.verifiers.bounded_http_request") as request:
                with self.assertRaisesRegex(VerifierError, "S3 region must contain"):
                    capture_verified_state(
                        {
                            "sources": [
                                {
                                    "id": "s3",
                                    "type": "s3_objects",
                                    "bucket": "demo",
                                    "region": "attacker.example/path",
                                }
                            ]
                        }
                    )

        request.assert_not_called()

    def test_imap_host_rejects_existing_credentials_without_explicit_consent(self):
        with patch.dict(
            os.environ,
            {
                "TEST_IMAP_USERNAME": "agent@example.test",
                "TEST_IMAP_PASSWORD": "private-password",
            },
            clear=True,
        ):
            with patch("flightrecorder.verifiers.imaplib.IMAP4_SSL") as client:
                with self.assertRaisesRegex(VerifierError, "allow_custom_origin=true"):
                    capture_verified_state(
                        {
                            "sources": [
                                {
                                    "id": "imap",
                                    "type": "imap",
                                    "host": "imap.attacker.example.test",
                                    "username_env": "TEST_IMAP_USERNAME",
                                    "password_env": "TEST_IMAP_PASSWORD",
                                }
                            ]
                        }
                    )

        client.assert_not_called()

    def test_kubernetes_credentialed_url_rejects_missing_explicit_consent_before_request(self):
        server = _JsonServer({"/api/v1/pods": {"items": []}})
        try:
            with patch.dict(os.environ, {"TEST_K8S_TOKEN": "private-token"}, clear=True):
                credential_cases = (
                    {"token_env": "TEST_K8S_TOKEN"},
                    {"bearer_token_env": "TEST_K8S_TOKEN"},
                    {"headers_from_env": {"Authorization": "TEST_K8S_TOKEN"}},
                    {"headers": {"authorization": "Bearer private-token"}},
                )
                for credentials in credential_cases:
                    with self.subTest(credentials=next(iter(credentials))):
                        with self.assertRaisesRegex(VerifierError, "allow_custom_origin=true"):
                            capture_verified_state(
                                {
                                    "sources": [
                                        {
                                            "id": "kubernetes",
                                            "type": "kubernetes_resources",
                                            "url": f"{server.url}/api/v1/pods",
                                            **credentials,
                                        }
                                    ]
                                }
                            )
        finally:
            server.close()

        self.assertEqual(server.requests, [])

    def test_kubernetes_unauthenticated_custom_url_needs_no_credential_consent(self):
        server = _JsonServer({"/api/v1/pods": {"items": [{"kind": "Pod", "metadata": {"name": "demo"}}]}})
        try:
            with patch.dict(os.environ, {}, clear=True):
                snapshot = capture_verified_state(
                    {
                        "sources": [
                            {
                                "id": "kubernetes",
                                "type": "kubernetes_resources",
                                "url": f"{server.url}/api/v1/pods",
                            }
                        ]
                    }
                )
        finally:
            server.close()

        self.assertEqual(snapshot["verifiers"]["sources"]["kubernetes"]["status"], "ok")
        self.assertEqual(len(server.requests), 1)

    def test_slack_default_token_remains_available_at_official_origin(self):
        with patch.dict(os.environ, {"SLACK_BOT_TOKEN": "official-origin-token"}):
            with patch(
                "flightrecorder.verifiers._http_get_json",
                return_value=(200, {"ok": True, "messages": []}),
            ) as request:
                snapshot = capture_verified_state(
                    {
                        "sources": [
                            {
                                "id": "slack",
                                "type": "slack_history",
                                "channel_id": "C123",
                            }
                        ]
                    },
                    secret_patterns=["official-origin-token"],
                )

        self.assertEqual(snapshot["verifiers"]["sources"]["slack"]["status"], "ok")
        self.assertEqual(request.call_args.args[0].split("?", 1)[0], "https://slack.com/api/conversations.history")
        self.assertEqual(request.call_args.kwargs["headers"]["Authorization"], "Bearer official-origin-token")

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
                                "allow_custom_origin": True,
                                "state_path": "github.issue_7",
                            },
                            {
                                "id": "sent_threads",
                                "type": "gmail_threads",
                                "query": "to:customer@example.test",
                                "token_env": "TEST_GMAIL_TOKEN",
                                "base_url": f"{server.url}/gmail/v1",
                                "allow_custom_origin": True,
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

    def test_slack_history_adapter_feeds_scorecard_end_to_end(self):
        server = _JsonServer(
            {
                "/slack/conversations.history": {
                    "ok": True,
                    "messages": [
                        {"text": "deployment finished successfully", "channel_id": "ignored", "user": "U1"}
                    ],
                }
            }
        )
        try:
            with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"TEST_SLACK_TOKEN": "slack-token"}):
                root = Path(tmp)
                state_path = root / "slack.state.json"
                verifier_config = root / "slack.verifier.json"
                verifier_config.write_text(
                    json.dumps(
                        {
                            "schema_version": "hfr.verifier_config.v1",
                            "sources": [
                                {
                                    "id": "slack_history",
                                    "type": "slack_history",
                                    "base_url": f"{server.url}/slack",
                                    "allow_custom_origin": True,
                                    "channel_id": "C123",
                                    "token_env": "TEST_SLACK_TOKEN",
                                    "state_path": "slack",
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                self.assertEqual(
                    main(["verify-state", "--config", str(verifier_config), "--out", str(state_path)]),
                    0,
                )
                compiled = build_state_validator_assertions(
                    {
                        "validator": "slack_message_sent",
                        "id": "notify_deploy_done",
                        "state_path": "slack.messages",
                        "text_contains": "deployment finished",
                        "channel_id": "C123",
                        "trace": {
                            "tool_name": "slack_post_message",
                            "where": {"result.channel_id": "C123", "result.status": "sent"},
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
                            "id": "slack_adapter_completion",
                            "title": "Slack Adapter Completion Verification",
                            "prompt": "Post a deployment completion notice to Slack channel C123.",
                            "policy": {},
                            "assertions": compiled["assertions"],
                            "scoring": {"pass_threshold": 90},
                        }
                    ),
                    encoding="utf-8",
                )
                run_dir = root / "run"
                self.assertEqual(
                    main(
                        [
                            "run",
                            "--scenario",
                            str(scenario_path),
                            "--trace",
                            str(trace_path),
                            "--state",
                            str(state_path),
                            "--out",
                            str(run_dir),
                        ]
                    ),
                    0,
                )
                score = json.loads((run_dir / "scorecard.json").read_text(encoding="utf-8"))
                self.assertTrue(score["passed"], score)
                self.assertTrue((run_dir / "report.html").exists())
        finally:
            server.close()

    def test_calendar_adapter_feeds_scorecard_end_to_end_and_honors_orderby(self):
        server = _JsonServer(
            {
                "/calendar/v3/calendars/primary/events": {
                    "items": [
                        {
                            "id": "evt-1",
                            "summary": "Customer follow-up",
                            "status": "confirmed",
                            "attendees": [{"email": "customer@example.test"}],
                        }
                    ]
                }
            }
        )
        try:
            with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"TEST_CALENDAR_TOKEN": "calendar-token"}):
                root = Path(tmp)
                state_path = root / "calendar.state.json"
                verifier_config = root / "calendar.verifier.json"
                verifier_config.write_text(
                    json.dumps(
                        {
                            "schema_version": "hfr.verifier_config.v1",
                            "sources": [
                                {
                                    "id": "calendar_events",
                                    "type": "google_calendar_events",
                                    "base_url": f"{server.url}/calendar/v3",
                                    "allow_custom_origin": True,
                                    "calendar_id": "primary",
                                    "orderby": "startTime",
                                    "token_env": "TEST_CALENDAR_TOKEN",
                                    "state_path": "calendar",
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                self.assertEqual(
                    main(["verify-state", "--config", str(verifier_config), "--out", str(state_path)]),
                    0,
                )
                self.assertTrue(
                    any("orderBy=startTime" in request["query"] for request in server.requests),
                    server.requests,
                )
                compiled = build_state_validator_assertions(
                    {
                        "validator": "calendar_event_created",
                        "id": "calendar_followup_created",
                        "state_path": "calendar.events",
                        "summary_contains": "Customer follow-up",
                        "attendee_contains": "customer@example.test",
                        "status": "confirmed",
                        "trace": {"tool_name": "calendar_create_event"},
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
                                    "tool_name": "calendar_create_event",
                                    "args": {},
                                    "status": "ok",
                                    "text": "",
                                    "result": {"id": "evt-1", "status": "confirmed"},
                                    "timestamp": None,
                                }
                            ],
                            "final_answer": "Created the customer follow-up event.",
                        }
                    ),
                    encoding="utf-8",
                )
                scenario_path = root / "scenario.json"
                scenario_path.write_text(
                    json.dumps(
                        {
                            "id": "calendar_adapter_completion",
                            "title": "Calendar Adapter Completion Verification",
                            "prompt": "Create a customer follow-up calendar event.",
                            "policy": {},
                            "assertions": compiled["assertions"],
                            "scoring": {"pass_threshold": 90},
                        }
                    ),
                    encoding="utf-8",
                )
                run_dir = root / "run"
                self.assertEqual(
                    main(
                        [
                            "run",
                            "--scenario",
                            str(scenario_path),
                            "--trace",
                            str(trace_path),
                            "--state",
                            str(state_path),
                            "--out",
                            str(run_dir),
                        ]
                    ),
                    0,
                )
                score = json.loads((run_dir / "scorecard.json").read_text(encoding="utf-8"))
                self.assertTrue(score["passed"], score)
        finally:
            server.close()

    def test_provider_verifier_adapters_capture_normalized_external_state(self):
        server = _JsonServer(
            {
                "/slack/conversations.history": {
                    "ok": True,
                    "messages": [
                        {
                            "type": "message",
                            "ts": "1710000000.000100",
                            "user": "U1",
                            "text": "deployment finished successfully",
                        }
                    ],
                },
                "/calendar/v3/calendars/primary/events": {
                    "items": [
                        {
                            "id": "evt-1",
                            "summary": "Customer follow-up",
                            "status": "confirmed",
                            "attendees": [{"email": "customer@example.test"}],
                            "start": {"dateTime": "2026-06-29T10:00:00Z"},
                            "end": {"dateTime": "2026-06-29T10:30:00Z"},
                        }
                    ]
                },
                "/drive/v3/files": {
                    "files": [
                        {
                            "id": "file-1",
                            "name": "handoff notes",
                            "mimeType": "text/plain",
                            "owners": [{"emailAddress": "agent@example.test"}],
                        }
                    ]
                },
                "/k8s/apis/apps/v1/namespaces/prod/deployments": {
                    "items": [
                        {
                            "kind": "Deployment",
                            "metadata": {"name": "api", "namespace": "prod", "labels": {"app": "api"}},
                            "spec": {"replicas": 2},
                            "status": {"readyReplicas": 2},
                        }
                    ]
                },
                "/stripe/v1/payment_intents/pi_123": {
                    "id": "pi_123",
                    "object": "payment_intent",
                    "status": "succeeded",
                    "amount": 4200,
                    "currency": "usd",
                },
                "/notion/v1/databases/db1/query": {
                    "results": [
                        {
                            "id": "page-1",
                            "url": "https://notion.example/page-1",
                            "last_edited_time": "2026-06-29T00:00:00Z",
                            "properties": {
                                "Name": {
                                    "type": "title",
                                    "title": [{"plain_text": "Runbook update"}],
                                },
                                "Notes": {
                                    "type": "rich_text",
                                    "rich_text": [{"plain_text": "Added rollback steps"}],
                                },
                            },
                        }
                    ],
                    "has_more": False,
                },
                "/linear/graphql": {
                    "data": {
                        "issues": {
                            "nodes": [
                                {
                                    "id": "lin-1",
                                    "identifier": "ENG-123",
                                    "title": "Fix deploy",
                                    "state": {"name": "Done"},
                                    "team": {"key": "ENG"},
                                }
                            ]
                        }
                    }
                },
                "/jira/rest/api/3/search": {
                    "issues": [
                        {
                            "id": "10001",
                            "key": "OPS-7",
                            "fields": {
                                "summary": "Close incident",
                                "status": {"name": "Done"},
                                "issuetype": {"name": "Task"},
                                "labels": ["incident"],
                            },
                        }
                    ],
                    "total": 1,
                },
                "/s3": (
                    "<ListBucketResult xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\">"
                    "<Contents><Key>reports/out.json</Key><LastModified>2026-06-29T00:00:00Z</LastModified>"
                    "<ETag>\"abc123\"</ETag><Size>42</Size><StorageClass>STANDARD</StorageClass></Contents>"
                    "</ListBucketResult>"
                ),
                "/graph/v1.0/me/mailFolders/SentItems/messages": {
                    "value": [
                        {
                            "id": "msg-graph-1",
                            "subject": "Re: customer",
                            "bodyPreview": "Sent the requested update",
                            "from": {"emailAddress": {"address": "agent@example.test"}},
                            "toRecipients": [{"emailAddress": {"address": "customer@example.test"}}],
                        }
                    ]
                },
                "/graph/v1.0/me/events": {
                    "value": [
                        {
                            "id": "graph-event-1",
                            "subject": "Customer follow-up",
                            "showAs": "busy",
                            "attendees": [{"emailAddress": {"address": "customer@example.test"}}],
                        }
                    ]
                },
                "/gitlab/api/v4/projects/group%2Fproject/issues": [
                    {
                        "id": 1,
                        "iid": 7,
                        "title": "Fix deploy",
                        "state": "closed",
                        "labels": ["ops"],
                        "assignees": [{"username": "agent"}],
                    }
                ],
                "/discord/api/v10/channels/C123/messages": [
                    {
                        "id": "discord-msg-1",
                        "content": "deployment finished successfully",
                        "author": {"id": "U1", "username": "agent"},
                    }
                ],
                "/zendesk/api/v2/tickets/42.json": {
                    "ticket": {
                        "id": 42,
                        "subject": "Customer request",
                        "status": "solved",
                        "priority": "normal",
                    }
                },
                "/pagerduty/incidents": {
                    "incidents": [
                        {
                            "id": "PD123",
                            "incident_number": 99,
                            "title": "API outage",
                            "status": "resolved",
                            "service": {"summary": "api"},
                            "assignments": [{"assignee": {"summary": "agent"}}],
                        }
                    ],
                    "more": False,
                },
            }
        )
        try:
            with patch.dict(
                os.environ,
                {
                    "TEST_SLACK_TOKEN": "slack-token",
                    "TEST_CALENDAR_TOKEN": "calendar-token",
                    "TEST_DRIVE_TOKEN": "drive-token",
                    "TEST_STRIPE_TOKEN": "stripe-token",
                    "TEST_NOTION_TOKEN": "notion-token",
                    "TEST_LINEAR_TOKEN": "linear-token",
                    "TEST_JIRA_TOKEN": "jira-token",
                    "TEST_GRAPH_TOKEN": "graph-token",
                    "TEST_GITLAB_TOKEN": "gitlab-token",
                    "TEST_DISCORD_TOKEN": "discord-token",
                    "TEST_ZENDESK_TOKEN": "zendesk-token",
                    "TEST_PAGERDUTY_TOKEN": "pagerduty-token",
                },
            ):
                snapshot = capture_verified_state(
                    {
                        "schema_version": "hfr.verifier_config.v1",
                        "sources": [
                            {
                                "id": "slack",
                                "type": "slack_history",
                                "base_url": f"{server.url}/slack",
                                "allow_custom_origin": True,
                                "channel_id": "C123",
                                "token_env": "TEST_SLACK_TOKEN",
                                "state_path": "slack",
                            },
                            {
                                "id": "calendar",
                                "type": "google_calendar_events",
                                "base_url": f"{server.url}/calendar/v3",
                                "allow_custom_origin": True,
                                "calendar_id": "primary",
                                "token_env": "TEST_CALENDAR_TOKEN",
                                "state_path": "calendar",
                            },
                            {
                                "id": "drive",
                                "type": "google_drive_files",
                                "base_url": f"{server.url}/drive/v3",
                                "allow_custom_origin": True,
                                "token_env": "TEST_DRIVE_TOKEN",
                                "state_path": "drive",
                            },
                            {
                                "id": "kubernetes",
                                "type": "kubernetes_resources",
                                "url": f"{server.url}/k8s/apis/apps/v1/namespaces/prod/deployments",
                                "state_path": "kubernetes",
                            },
                            {
                                "id": "stripe",
                                "type": "stripe_objects",
                                "base_url": f"{server.url}/stripe/v1",
                                "allow_custom_origin": True,
                                "resource": "payment_intents",
                                "object_id": "pi_123",
                                "token_env": "TEST_STRIPE_TOKEN",
                                "state_path": "payments.payment",
                            },
                            {
                                "id": "notion",
                                "type": "notion_database",
                                "base_url": f"{server.url}/notion/v1",
                                "allow_custom_origin": True,
                                "database_id": "db1",
                                "token_env": "TEST_NOTION_TOKEN",
                                "state_path": "notion",
                            },
                            {
                                "id": "linear",
                                "type": "linear_issues",
                                "base_url": f"{server.url}/linear/graphql",
                                "allow_custom_origin": True,
                                "token_env": "TEST_LINEAR_TOKEN",
                                "state_path": "linear.issue",
                                "state_value_path": "issues.0",
                            },
                            {
                                "id": "jira",
                                "type": "jira_issues",
                                "base_url": f"{server.url}/jira",
                                "allow_custom_origin": True,
                                "jql": "project = OPS",
                                "bearer_token_env": "TEST_JIRA_TOKEN",
                                "state_path": "jira.issue",
                                "state_value_path": "issues.0",
                            },
                            {
                                "id": "s3",
                                "type": "s3_objects",
                                "url": f"{server.url}/s3",
                                "bucket": "demo",
                                "prefix": "reports/",
                                "unsigned": True,
                                "state_path": "s3.objects",
                                "state_value_path": "objects",
                            },
                            {
                                "id": "graph_messages",
                                "type": "microsoft_graph_messages",
                                "base_url": f"{server.url}/graph/v1.0",
                                "allow_custom_origin": True,
                                "folder_id": "SentItems",
                                "token_env": "TEST_GRAPH_TOKEN",
                                "state_path": "graph.mail",
                            },
                            {
                                "id": "graph_events",
                                "type": "microsoft_graph_events",
                                "base_url": f"{server.url}/graph/v1.0",
                                "allow_custom_origin": True,
                                "token_env": "TEST_GRAPH_TOKEN",
                                "state_path": "graph.calendar",
                            },
                            {
                                "id": "gitlab",
                                "type": "gitlab_issues",
                                "base_url": f"{server.url}/gitlab/api/v4",
                                "allow_custom_origin": True,
                                "project_id": "group/project",
                                "token_env": "TEST_GITLAB_TOKEN",
                                "state_path": "gitlab",
                            },
                            {
                                "id": "discord",
                                "type": "discord_messages",
                                "base_url": f"{server.url}/discord/api/v10",
                                "allow_custom_origin": True,
                                "channel_id": "C123",
                                "token_env": "TEST_DISCORD_TOKEN",
                                "state_path": "discord",
                            },
                            {
                                "id": "zendesk",
                                "type": "zendesk_tickets",
                                "base_url": f"{server.url}/zendesk/api/v2",
                                "allow_custom_origin": True,
                                "ticket_id": "42",
                                "bearer_token_env": "TEST_ZENDESK_TOKEN",
                                "state_path": "zendesk.ticket",
                                "state_value_path": "tickets.0",
                            },
                            {
                                "id": "pagerduty",
                                "type": "pagerduty_incidents",
                                "base_url": f"{server.url}/pagerduty",
                                "allow_custom_origin": True,
                                "token_env": "TEST_PAGERDUTY_TOKEN",
                                "state_path": "pagerduty.incident",
                                "state_value_path": "incidents.0",
                            },
                        ],
                    },
                    secret_patterns=[
                        "slack-token",
                        "calendar-token",
                        "drive-token",
                        "stripe-token",
                        "notion-token",
                        "linear-token",
                        "jira-token",
                        "graph-token",
                        "gitlab-token",
                        "discord-token",
                        "zendesk-token",
                        "pagerduty-token",
                    ],
                )
        finally:
            server.close()

        self.assertEqual(snapshot["verifiers"]["source_count"], 15)
        self.assertEqual(snapshot["slack"]["messages"][0]["channel_id"], "C123")
        self.assertEqual(snapshot["calendar"]["events"][0]["attendees"], ["customer@example.test"])
        self.assertEqual(snapshot["drive"]["files"][0]["name"], "handoff notes")
        self.assertTrue(snapshot["kubernetes"]["resources"][0]["ready"])
        self.assertEqual(snapshot["payments"]["payment"]["status"], "succeeded")
        self.assertEqual(snapshot["notion"]["pages"][0]["title"], "Runbook update")
        self.assertEqual(snapshot["linear"]["issue"]["status"], "Done")
        self.assertEqual(snapshot["jira"]["issue"]["key"], "OPS-7")
        self.assertEqual(snapshot["s3"]["objects"][0]["key"], "reports/out.json")
        self.assertEqual(snapshot["graph"]["mail"]["messages"][0]["to"], ["customer@example.test"])
        self.assertEqual(snapshot["graph"]["calendar"]["events"][0]["summary"], "Customer follow-up")
        self.assertEqual(snapshot["gitlab"]["issues"][0]["state"], "closed")
        self.assertEqual(snapshot["discord"]["messages"][0]["text"], "deployment finished successfully")
        self.assertEqual(snapshot["zendesk"]["ticket"]["status"], "solved")
        self.assertEqual(snapshot["pagerduty"]["incident"]["status"], "resolved")
        self.assertNotIn("stripe-token", json.dumps(snapshot))

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
                                "allow_custom_origin": True,
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
                            "allow_custom_origin": True,
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


class SafeHttpPolicyTests(unittest.TestCase):
    def test_error_bodies_are_clipped_at_the_configured_limit(self):
        error = urllib.error.HTTPError(
            "https://example.test/failure",
            500,
            "failure",
            {},
            io.BytesIO(b"sensitive-error-body"),
        )
        with patch("flightrecorder.safe_http.urllib.request.build_opener") as build_opener:
            build_opener.return_value.open.side_effect = error
            with self.assertRaises(HttpStatusError) as raised:
                bounded_http_request(
                    "GET",
                    "https://example.test/failure",
                    timeout=1,
                    max_body_bytes=64,
                    max_error_bytes=9,
                )

        self.assertEqual(raised.exception.body, b"sensitive")
        self.assertTrue(raised.exception.truncated)

    def test_http_status_errors_do_not_render_response_bodies(self):
        error = HttpStatusError(
            500,
            b'{"access_token":"private-response-value"}',
            truncated=True,
        )

        rendered = str(error)
        self.assertIn("HTTP status 500", rendered)
        self.assertIn("truncated", rendered)
        self.assertNotIn("private-response-value", rendered)

    def test_sse_reader_bounds_lines_events_and_aggregate_bytes(self):
        cases = [
            (
                b"data: 123456789\n\n",
                {"max_line_bytes": 8, "max_event_bytes": 64, "max_total_bytes": 128},
                "line exceeded",
            ),
            (
                b"data: 1234\ndata: 5678\n\n",
                {"max_line_bytes": 32, "max_event_bytes": 12, "max_total_bytes": 128},
                "event exceeded",
            ),
            (
                b": comment-one\n: comment-two\n",
                {"max_line_bytes": 32, "max_event_bytes": 64, "max_total_bytes": 20},
                "aggregate exceeded",
            ),
        ]

        for payload, limits, expected in cases:
            with self.subTest(expected=expected), self.assertRaisesRegex(SafeHttpError, expected):
                read_bounded_sse(io.BytesIO(payload), max_events=8, **limits)


class _JsonServer:
    def __init__(self, routes, *, redirects=None):
        self.routes = routes
        self.redirects = redirects or {}
        self.requests = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def _handler(self):
        routes = self.routes
        redirects = self.redirects
        requests = self.requests

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path, _, query = self.path.partition("?")
                if path in redirects:
                    self.send_response(302)
                    self.send_header("Location", redirects[path])
                    self.end_headers()
                    return
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
                if isinstance(value, bytes):
                    payload = value
                    content_type = "application/octet-stream"
                elif isinstance(value, str):
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


def _s3_listing_xml() -> str:
    return (
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        "<Contents><Key>reports/out.json</Key><LastModified>2026-06-29T00:00:00Z</LastModified>"
        '<ETag>"abc123"</ETag><Size>42</Size><StorageClass>STANDARD</StorageClass></Contents>'
        "</ListBucketResult>"
    )


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
