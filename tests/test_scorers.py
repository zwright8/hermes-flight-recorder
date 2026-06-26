import unittest
from pathlib import Path

from flightrecorder.adapters import normalize_trace
from flightrecorder.schema import load_scenario
from flightrecorder.scorers import score_trace


ROOT = Path(__file__).resolve().parents[1]


class ScorerTests(unittest.TestCase):
    def _score(self, scenario_name):
        scenario = load_scenario(ROOT / "scenarios" / scenario_name)
        trace = normalize_trace(ROOT / scenario["trace"]["path"].replace("../", ""))
        return score_trace(scenario, trace)

    def test_prompt_injection_good_passes(self):
        scorecard = self._score("prompt_injection_good.json")

        self.assertTrue(scorecard["passed"])
        self.assertEqual(scorecard["score"], 100)

    def test_prompt_injection_bad_fails_for_forbidden_action_and_secret(self):
        scorecard = self._score("prompt_injection_bad.json")

        self.assertFalse(scorecard["passed"])
        self.assertIn("forbidden_actions", scorecard["critical_failures"])
        self.assertIn("secret_exposure", scorecard["critical_failures"])
        self.assertIn("final_answer", scorecard["critical_failures"])

    def test_subagent_claim_fails_for_missing_evidence(self):
        scorecard = self._score("subagent_claim_bad.json")

        self.assertFalse(scorecard["passed"])
        self.assertIn("required_evidence", scorecard["critical_failures"])

    def test_budget_runaway_fails_for_budget(self):
        scorecard = self._score("budget_runaway_bad.json")

        self.assertFalse(scorecard["passed"])
        self.assertIn("budget", scorecard["critical_failures"])

    def test_required_actions_can_prove_email_task_completion(self):
        scorecard = self._score("email_reply_completion_good.json")

        self.assertTrue(scorecard["passed"])
        action_rule = next(rule for rule in scorecard["rules"] if rule["id"] == "required_actions")
        self.assertTrue(action_rule["passed"])
        self.assertEqual(action_rule["items"][0]["id"], "reply_email_123")
        self.assertIn("matched event", action_rule["items"][0]["evidence"])

    def test_required_actions_fail_when_observable_action_is_missing(self):
        scenario = load_scenario(ROOT / "scenarios" / "email_reply_completion_good.json")
        scenario["assertions"]["required_actions"][0]["where"]["result.thread_id"] = "email-999"
        trace = normalize_trace(ROOT / "fixtures" / "email_reply_completion_good.observer.jsonl")

        scorecard = score_trace(scenario, trace)

        self.assertFalse(scorecard["passed"])
        self.assertIn("required_actions", scorecard["critical_failures"])

    def test_required_evidence_supports_top_level_contains_matcher(self):
        scenario = load_scenario(ROOT / "scenarios" / "email_reply_completion_good.json")
        scenario["assertions"]["required_evidence"].append(
            {"id": "sent_tool_seen", "type": "event_matches", "contains": "gmail_send"}
        )
        trace = normalize_trace(ROOT / "fixtures" / "email_reply_completion_good.observer.jsonl")

        scorecard = score_trace(scenario, trace)

        self.assertTrue(scorecard["passed"])

    def test_final_evidence_supports_contains_matcher(self):
        scenario = load_scenario(ROOT / "scenarios" / "email_reply_completion_good.json")
        scenario["assertions"]["required_evidence"].append(
            {"id": "final_mentions_thread", "type": "final_matches", "contains": "email-123"}
        )
        trace = normalize_trace(ROOT / "fixtures" / "email_reply_completion_good.observer.jsonl")

        scorecard = score_trace(scenario, trace)

        self.assertTrue(scorecard["passed"])


if __name__ == "__main__":
    unittest.main()
