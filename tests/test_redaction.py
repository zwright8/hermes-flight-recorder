import copy
import unittest

from flightrecorder.redaction import (
    RedactionError,
    contains_unredacted_secret_assignment,
    is_secret_key,
    redact_text,
    sanitize_trace,
)


class RedactionTests(unittest.TestCase):
    def test_sanitize_trace_redacts_nested_values_under_secret_keys(self):
        trace = {
            "metadata": {
                "credentials": {
                    "api_key": "alpha-secret-value",
                    "password": {"primary": "bravo-secret-value"},
                    "token_count": 42,
                    "api_key_env": "HFR_API_KEY",
                }
            }
        }
        original = copy.deepcopy(trace)

        sanitized = sanitize_trace(trace)

        credentials = sanitized["metadata"]["credentials"]
        self.assertEqual(credentials["api_key"], "[REDACTED]")
        self.assertEqual(credentials["password"], "[REDACTED]")
        self.assertEqual(credentials["token_count"], 42)
        self.assertEqual(credentials["api_key_env"], "HFR_API_KEY")
        self.assertEqual(trace, original)

    def test_sanitize_trace_recognizes_common_camel_case_secret_keys(self):
        sanitized = sanitize_trace(
            {
                "accessToken": "access-token-value",
                "refreshToken": "refresh-token-value",
                "clientSecret": "client-secret-value",
                "authorizationUsage": 4,
            }
        )

        self.assertEqual(sanitized["accessToken"], "[REDACTED]")
        self.assertEqual(sanitized["refreshToken"], "[REDACTED]")
        self.assertEqual(sanitized["clientSecret"], "[REDACTED]")
        self.assertEqual(sanitized["authorizationUsage"], 4)

    def test_redact_text_handles_json_quotes_bearer_headers_and_quoted_spaces(self):
        text = (
            '{"api_key": "alpha beta gamma", "safe": "keep me"}\n'
            "Authorization: Bearer header-token-value\n"
            'Bearer="delta epsilon zeta"'
        )

        redacted = redact_text(text)

        for secret in ("alpha beta gamma", "Bearer header-token-value", "delta epsilon zeta"):
            self.assertNotIn(secret, redacted)
        self.assertIn('"api_key": "[REDACTED]"', redacted)
        self.assertIn("Authorization: [REDACTED]", redacted)
        self.assertIn('Bearer="[REDACTED]"', redacted)
        self.assertIn('"safe": "keep me"', redacted)

    def test_redact_text_does_not_over_redact_safe_metadata_keys(self):
        text = 'token_count=42 api_key_env="HFR_API_KEY" token_budget=100'

        self.assertEqual(redact_text(text), text)

    def test_redact_text_consumes_complete_authorization_header_values(self):
        text = (
            "Authorization: Basic QWxhZGRpbjpvcGVuIHNlc2FtZQ==\n"
            "Authorization: ApiKey alpha key with spaces\n"
            "Authorization: Bearer bearer-token-value\n"
            "safe: visible"
        )

        redacted = redact_text(text)

        self.assertEqual(
            redacted,
            "Authorization: [REDACTED]\n"
            "Authorization: [REDACTED]\n"
            "Authorization: [REDACTED]\n"
            "safe: visible",
        )

    def test_redact_text_preserves_delimiters_and_adjacent_safe_fields(self):
        text = (
            "password=correct horse battery staple; safe=keep\n"
            "auth: client credential phrase, mode: demo\n"
            "cookie=session-cookie-value adjacent=visible\n"
            "session-token: token with spaces\n"
            "This prose remains visible."
        )

        redacted = redact_text(text)

        self.assertEqual(
            redacted,
            "password=[REDACTED]; safe=keep\n"
            "auth: [REDACTED], mode: demo\n"
            "cookie=[REDACTED] adjacent=visible\n"
            "session-token: [REDACTED]\n"
            "This prose remains visible.",
        )

    def test_sanitize_trace_recognizes_auth_credential_cookie_and_session_token_keys(self):
        sanitized = sanitize_trace(
            {
                "auth": "auth-value",
                "credential": "credential-value",
                "credentials": {"user": "alice", "password": "secret-value"},
                "cookie": "session-cookie",
                "sessionToken": "session-token-value",
                "auth_count": 1,
                "credential_type": "oauth",
                "cookie_name": "session",
                "session_token_env": "SESSION_TOKEN",
            }
        )

        for key in ("auth", "credential", "cookie", "sessionToken"):
            with self.subTest(key=key):
                self.assertEqual(sanitized[key], "[REDACTED]")
        self.assertEqual(sanitized["credentials"], {"user": "[REDACTED]", "password": "[REDACTED]"})
        self.assertEqual(sanitized["auth_count"], 1)
        self.assertEqual(sanitized["credential_type"], "oauth")
        self.assertEqual(sanitized["cookie_name"], "session")
        self.assertEqual(sanitized["session_token_env"], "SESSION_TOKEN")

    def test_credential_containers_fail_closed_for_unknown_descendants_and_lists(self):
        sanitized = sanitize_trace(
            {
                "credentials": {
                    "username": "alice",
                    "value": "supersecret",
                    "nested": {"primary": "hunter2"},
                    "aliases": ["alice", "backup-secret"],
                    "credential_type": "oauth",
                    "api_key_env": "HFR_API_KEY",
                    "token_count": 2,
                },
                "credentialList": ["first-secret", {"secondary": "second-secret"}],
            }
        )

        self.assertEqual(
            sanitized["credentials"],
            {
                "username": "[REDACTED]",
                "value": "[REDACTED]",
                "nested": {"primary": "[REDACTED]"},
                "aliases": ["[REDACTED]", "[REDACTED]"],
                "credential_type": "oauth",
                "api_key_env": "HFR_API_KEY",
                "token_count": 2,
            },
        )
        self.assertEqual(sanitized["credentialList"], "[REDACTED]")

    def test_composite_credential_keys_are_secret_bearing_but_metadata_flags_are_not(self):
        for key in (
            "credential_value",
            "cookie_value",
            "access_key",
            "private_key",
            "auth_header",
            "signing_key",
        ):
            with self.subTest(key=key):
                self.assertTrue(is_secret_key(key))

        for key in (
            "credential_type",
            "credential_values_recorded",
            "credentials_available",
            "auth_count",
            "cookie_name",
            "access_key_env",
        ):
            with self.subTest(key=key):
                self.assertFalse(is_secret_key(key))

        sanitized = sanitize_trace(
            {
                "credential_value": "credential-secret",
                "cookie_value": "session=secret",
                "access_key": "AKIAEXAMPLEONLY0000",
                "private_key": "-----BEGIN PRIVATE KEY-----\nfixture\n-----END PRIVATE KEY-----",
                "auth_header": "Bearer header-secret",
            }
        )
        self.assertEqual(set(sanitized.values()), {"[REDACTED]"})

    def test_redact_text_preserves_quotes_and_recovers_from_unterminated_values(self):
        text = (
            'credential="quoted credential with spaces" safe="visible"\n'
            "cookie='quoted cookie with spaces'; mode=demo\n"
            'password="unterminated secret value\n'
            "safe prose on the next line"
        )

        redacted = redact_text(text)

        self.assertEqual(
            redacted,
            'credential="[REDACTED]" safe="visible"\n'
            "cookie='[REDACTED]'; mode=demo\n"
            'password="[REDACTED]\n'
            "safe prose on the next line",
        )

    def test_secret_assignment_scanner_agrees_with_redactor(self):
        secret_texts = (
            "Authorization: Basic QWxhZGRpbjpvcGVuIHNlc2FtZQ==",
            "Authorization: ApiKey alpha key with spaces",
            "Authorization: Bearer bearer-token-value",
            "password=correct horse battery staple; safe=keep",
            'credential="quoted credential with spaces" safe=visible',
            "cookie=session-cookie-value adjacent=visible",
            "session-token: token with spaces",
            'password="unterminated secret value',
        )
        for text in secret_texts:
            with self.subTest(text=text):
                self.assertTrue(contains_unredacted_secret_assignment(text))
                self.assertFalse(contains_unredacted_secret_assignment(redact_text(text)))

        safe_texts = (
            "authorization_count=3 credential_type=oauth cookie_name=session",
            "session_token_env=SESSION_TOKEN safe=visible",
            "The authorization process and credential policy are documented.",
            "prompt_injection_bad:secret_exposure:3",
            'token="[REDACTED]" api_key="<redacted:environment>"',
        )
        for text in safe_texts:
            with self.subTest(text=text):
                self.assertFalse(contains_unredacted_secret_assignment(text))
                self.assertEqual(redact_text(text), text)

    def test_invalid_custom_regex_fails_closed_with_domain_error(self):
        operations = (
            lambda: redact_text("safe text", ["["]),
            lambda: sanitize_trace({"final_answer": "safe text"}, ["["]),
        )
        for operation in operations:
            with self.subTest(operation=operation):
                with self.assertRaisesRegex(RedactionError, "Invalid redaction regex at index 0"):
                    operation()


if __name__ == "__main__":
    unittest.main()
