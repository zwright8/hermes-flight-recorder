import json
import tempfile
import unittest
from pathlib import Path

from flightrecorder.schema import ScenarioError, load_scenario


ROOT = Path(__file__).resolve().parents[1]


class ScenarioSchemaTests(unittest.TestCase):
    def test_load_scenario_rejects_non_object_sections_cleanly(self):
        for field in ("assertions", "scoring"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "bad.json"
                payload = {"id": "bad", "title": "Bad", "prompt": "x", "policy": {}, field: []}
                path.write_text(json.dumps(payload), encoding="utf-8")

                with self.assertRaisesRegex(ScenarioError, rf"Scenario {field} must be an object"):
                    load_scenario(path)

    def test_load_scenario_rejects_boolean_numeric_limits(self):
        cases = (
            ({"policy": {"max_tool_calls": True}}, "max_tool_calls"),
            ({"policy": {}, "scoring": {"pass_threshold": False}}, "pass_threshold"),
        )
        for overrides, expected in cases:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "bad.json"
                payload = {"id": "bad", "title": "Bad", "prompt": "x", **overrides}
                path.write_text(json.dumps(payload), encoding="utf-8")

                with self.assertRaisesRegex(ScenarioError, expected):
                    load_scenario(path)

    def test_load_scenario_rejects_non_string_title_and_prompt(self):
        for field, value in (("title", {}), ("prompt", [])):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "bad.json"
                payload = {"id": "bad", "title": "Bad", "prompt": "x", "policy": {}}
                payload[field] = value
                path.write_text(json.dumps(payload), encoding="utf-8")

                with self.assertRaisesRegex(ScenarioError, rf"Scenario {field} must be a non-empty string"):
                    load_scenario(path)

    def test_load_valid_scenario_applies_defaults(self):
        scenario = load_scenario(ROOT / "scenarios" / "prompt_injection_good.json")

        self.assertEqual(scenario["id"], "prompt_injection_good")
        self.assertEqual(scenario["scoring"]["pass_threshold"], 90)
        self.assertIn("max_tool_calls", scenario["policy"])
        self.assertIn("required_actions", scenario["assertions"])

    def test_load_scenario_accepts_required_actions(self):
        scenario = load_scenario(ROOT / "scenarios" / "email_reply_completion_good.json")

        action = scenario["assertions"]["required_actions"][0]
        self.assertEqual(action["id"], "reply_email_123")
        self.assertEqual(action["where"]["result.thread_id"], "email-123")
        sequence = scenario["assertions"]["required_action_sequences"][0]
        count = scenario["assertions"]["required_event_counts"][0]
        state_check = scenario["assertions"]["required_state"][0]
        state_transition = scenario["assertions"]["required_state_transitions"][0]
        self.assertEqual(sequence["steps"][0]["tool_name"], "gmail_read")
        self.assertEqual(count["exact_count"], 1)
        self.assertEqual(state_check["id"], "state_has_sent_reply_email_123")
        self.assertEqual(state_transition["id"], "reply_added_to_thread_email_123")
        self.assertEqual(scenario["state"]["before_path"], "../fixtures/email_reply_completion_before.state.json")
        self.assertEqual(scenario["state"]["format"], "json")

    def test_required_state_rejects_unbounded_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(
                json.dumps(
                    {
                        "id": "bad",
                        "title": "Bad",
                        "prompt": "x",
                        "policy": {},
                        "assertions": {
                            "required_state": [
                                {"id": "state_anything"}
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ScenarioError, "missing matcher"):
                load_scenario(path)

    def test_required_state_accepts_where_any_collection_matcher(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "good.json"
            path.write_text(
                json.dumps(
                    {
                        "id": "good",
                        "title": "Good",
                        "prompt": "x",
                        "policy": {},
                        "assertions": {
                            "required_state": [
                                {
                                    "id": "message_exists",
                                    "where_any": {
                                        "path": "slack.messages",
                                        "where": {
                                            "text": {"contains": "done"},
                                            "channel_id": "C123",
                                        },
                                    },
                                }
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )

            scenario = load_scenario(path)

            self.assertEqual(
                scenario["assertions"]["required_state"][0]["where_any"]["path"],
                "slack.messages",
            )

    def test_required_state_rejects_malformed_where_any(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(
                json.dumps(
                    {
                        "id": "bad",
                        "title": "Bad",
                        "prompt": "x",
                        "policy": {},
                        "assertions": {
                            "required_state": [
                                {
                                    "id": "message_exists",
                                    "where_any": {"path": "slack.messages"},
                                }
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ScenarioError, "where must be a non-empty object"):
                load_scenario(path)

    def test_required_state_transitions_require_before_and_after_matchers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(
                json.dumps(
                    {
                        "id": "bad",
                        "title": "Bad",
                        "prompt": "x",
                        "policy": {},
                        "assertions": {
                            "required_state_transitions": [
                                {
                                    "id": "missing_after",
                                    "before": {"where": {"ticket.status": "open"}},
                                }
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ScenarioError, "after must be an object"):
                load_scenario(path)

    def test_required_state_transitions_reject_event_matchers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(
                json.dumps(
                    {
                        "id": "bad",
                        "title": "Bad",
                        "prompt": "x",
                        "policy": {},
                        "assertions": {
                            "required_state_transitions": [
                                {
                                    "id": "bad_phase",
                                    "before": {"where": {"ticket.status": "open"}},
                                    "after": {"event_type": "tool_result", "where": {"ticket.status": "closed"}},
                                }
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ScenarioError, "unsupported matcher fields"):
                load_scenario(path)

    def test_missing_required_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps({"title": "Bad", "prompt": "x", "policy": {}}), encoding="utf-8")
            with self.assertRaisesRegex(ScenarioError, "missing required field: id"):
                load_scenario(path)

    def test_malformed_regex_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(
                json.dumps(
                    {
                        "id": "bad",
                        "title": "Bad",
                        "prompt": "x",
                        "policy": {"forbidden_command_patterns": ["["]},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ScenarioError, "Invalid regex"):
                load_scenario(path)

    def test_required_evidence_rejects_missing_matcher(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(
                json.dumps(
                    {
                        "id": "bad",
                        "title": "Bad",
                        "prompt": "x",
                        "policy": {},
                        "assertions": {
                            "required_evidence": [
                                {"id": "needs_pattern", "type": "event_matches"}
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ScenarioError, "missing matcher"):
                load_scenario(path)

    def test_required_evidence_rejects_unknown_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(
                json.dumps(
                    {
                        "id": "bad",
                        "title": "Bad",
                        "prompt": "x",
                        "policy": {},
                        "assertions": {
                            "required_evidence": [
                                {"id": "unknown", "type": "sometimes_matches", "pattern": "x"}
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ScenarioError, "unsupported type"):
                load_scenario(path)

    def test_required_actions_rejects_unbounded_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(
                json.dumps(
                    {
                        "id": "bad",
                        "title": "Bad",
                        "prompt": "x",
                        "policy": {},
                        "assertions": {
                            "required_actions": [
                                {"id": "does_anything"}
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ScenarioError, "must define an event selector or field matcher"):
                load_scenario(path)

    def test_required_action_sequences_reject_empty_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(
                json.dumps(
                    {
                        "id": "bad",
                        "title": "Bad",
                        "prompt": "x",
                        "policy": {},
                        "assertions": {
                            "required_action_sequences": [
                                {"id": "empty_sequence", "steps": []}
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ScenarioError, "steps must be a non-empty list"):
                load_scenario(path)

    def test_required_event_counts_require_a_count_constraint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(
                json.dumps(
                    {
                        "id": "bad",
                        "title": "Bad",
                        "prompt": "x",
                        "policy": {},
                        "assertions": {
                            "required_event_counts": [
                                {"id": "unbounded_count", "event_type": "tool_result"}
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ScenarioError, "must define exact_count"):
                load_scenario(path)

    def test_structured_evidence_rejects_bad_regex(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(
                json.dumps(
                    {
                        "id": "bad",
                        "title": "Bad",
                        "prompt": "x",
                        "policy": {},
                        "assertions": {
                            "required_evidence": [
                                {
                                    "id": "bad_regex",
                                    "type": "event_matches",
                                    "where": {"args.command": {"matches": "["}},
                                }
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ScenarioError, "Invalid regex"):
                load_scenario(path)

    def test_final_evidence_rejects_event_only_matchers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(
                json.dumps(
                    {
                        "id": "bad",
                        "title": "Bad",
                        "prompt": "x",
                        "policy": {},
                        "assertions": {
                            "required_evidence": [
                                {
                                    "id": "bad_final",
                                    "type": "final_matches",
                                    "where": {"args.command": "printf ok"},
                                }
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ScenarioError, "event-only matcher fields"):
                load_scenario(path)


if __name__ == "__main__":
    unittest.main()
