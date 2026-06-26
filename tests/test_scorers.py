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


if __name__ == "__main__":
    unittest.main()
