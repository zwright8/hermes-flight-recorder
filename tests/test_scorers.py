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
        forbidden_rule = next(rule for rule in scorecard["rules"] if rule["id"] == "forbidden_actions")
        self.assertTrue(any(ref["target"] == "event" for ref in forbidden_rule["evidence_refs"]))
        final_answer_rule = next(rule for rule in scorecard["rules"] if rule["id"] == "final_answer")
        self.assertTrue(any(ref["target"] == "final_answer" for ref in final_answer_rule["evidence_refs"]))

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
        self.assertEqual(action_rule["items"][0]["evidence_ref"]["target"], "event")
        self.assertEqual(action_rule["items"][0]["evidence_ref"]["event_index"], action_rule["items"][0]["event_index"])
        sequence_rule = next(rule for rule in scorecard["rules"] if rule["id"] == "required_action_sequences")
        count_rule = next(rule for rule in scorecard["rules"] if rule["id"] == "required_event_counts")
        self.assertTrue(sequence_rule["passed"])
        self.assertEqual(sequence_rule["items"][0]["event_indices"], [2, 4])
        self.assertTrue(count_rule["passed"])
        self.assertEqual(count_rule["items"][0]["actual_count"], 1)

    def test_required_actions_reject_final_answer_completion_claim_without_send_evidence(self):
        scorecard = self._score("email_reply_completion_bad.json")

        self.assertFalse(scorecard["passed"])
        self.assertEqual(scorecard["score"], 10)
        self.assertIn("required_actions", scorecard["critical_failures"])
        self.assertIn("required_action_sequences", scorecard["critical_failures"])
        self.assertIn("required_event_counts", scorecard["critical_failures"])
        final_rule = next(rule for rule in scorecard["rules"] if rule["id"] == "final_answer")
        self.assertTrue(final_rule["passed"])
        action_rule = next(rule for rule in scorecard["rules"] if rule["id"] == "required_actions")
        self.assertIn("missing required action", action_rule["evidence"][0])

    def test_required_actions_fail_when_observable_action_is_missing(self):
        scenario = load_scenario(ROOT / "scenarios" / "email_reply_completion_good.json")
        scenario["assertions"]["required_actions"][0]["where"]["result.thread_id"] = "email-999"
        trace = normalize_trace(ROOT / "fixtures" / "email_reply_completion_good.observer.jsonl")

        scorecard = score_trace(scenario, trace)

        self.assertFalse(scorecard["passed"])
        self.assertIn("required_actions", scorecard["critical_failures"])

    def test_required_action_sequences_fail_when_events_happen_out_of_order(self):
        scenario = load_scenario(ROOT / "scenarios" / "email_reply_completion_good.json")
        trace = normalize_trace(ROOT / "fixtures" / "email_reply_completion_good.observer.jsonl")
        send_result = trace["events"].pop(4)
        trace["events"].insert(2, send_result)

        scorecard = score_trace(scenario, trace)

        self.assertFalse(scorecard["passed"])
        self.assertIn("required_action_sequences", scorecard["critical_failures"])
        sequence_rule = next(rule for rule in scorecard["rules"] if rule["id"] == "required_action_sequences")
        self.assertFalse(sequence_rule["items"][0]["passed"])
        self.assertEqual(sequence_rule["items"][0]["event_indices"], [3])

    def test_required_event_counts_fail_when_action_happens_too_many_times(self):
        scenario = load_scenario(ROOT / "scenarios" / "email_reply_completion_good.json")
        trace = normalize_trace(ROOT / "fixtures" / "email_reply_completion_good.observer.jsonl")
        duplicate_send = dict(trace["events"][4])
        trace["events"].insert(5, duplicate_send)

        scorecard = score_trace(scenario, trace)

        self.assertFalse(scorecard["passed"])
        self.assertIn("required_event_counts", scorecard["critical_failures"])
        count_rule = next(rule for rule in scorecard["rules"] if rule["id"] == "required_event_counts")
        self.assertFalse(count_rule["items"][0]["passed"])
        self.assertEqual(count_rule["items"][0]["actual_count"], 2)

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
