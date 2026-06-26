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

    def test_required_evidence_rejects_missing_pattern(self):
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
            with self.assertRaisesRegex(ScenarioError, "missing field: pattern"):
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


if __name__ == "__main__":
    unittest.main()
