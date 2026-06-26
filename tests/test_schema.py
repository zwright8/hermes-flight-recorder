import json
import tempfile
import unittest
from pathlib import Path

from flightrecorder.schema import ScenarioError, load_scenario


ROOT = Path(__file__).resolve().parents[1]


class ScenarioSchemaTests(unittest.TestCase):
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
        self.assertEqual(sequence["steps"][0]["tool_name"], "gmail_read")
        self.assertEqual(count["exact_count"], 1)

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
