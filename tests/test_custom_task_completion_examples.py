import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main
from flightrecorder.schema import load_scenario


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = ROOT / "examples" / "custom_task_completion"


def run_cli(args):
    with redirect_stdout(StringIO()):
        return main(args)


class CustomTaskCompletionExampleTests(unittest.TestCase):
    def test_support_ticket_scenarios_load_from_example_directory(self):
        good = load_scenario(EXAMPLE_DIR / "support_ticket_completion_good.json")
        bad = load_scenario(EXAMPLE_DIR / "support_ticket_completion_bad.json")

        self.assertEqual(good["trace"]["path"], "fixtures/support_ticket_completion_good.observer.jsonl")
        self.assertEqual(bad["trace"]["path"], "fixtures/support_ticket_completion_bad.observer.jsonl")
        self.assertEqual(good["assertions"]["required_actions"][0]["tool_name"], "support_ticket_create")
        self.assertEqual(good["assertions"]["required_state"][0]["where"]["support.tickets.SUP-42.status"], "open")

    def test_support_ticket_good_trace_passes_with_action_and_state_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "good"

            code = run_cli(["run", "--scenario", str(EXAMPLE_DIR / "support_ticket_completion_good.json"), "--out", str(out)])

            self.assertEqual(code, 0)
            scorecard = json.loads((out / "scorecard.json").read_text(encoding="utf-8"))
            task_completion = json.loads((out / "task_completion.json").read_text(encoding="utf-8"))
            state_diff = json.loads((out / "state_diff.json").read_text(encoding="utf-8"))

            self.assertTrue(scorecard["passed"])
            self.assertEqual(scorecard["score"], 100)
            self.assertEqual(task_completion["status"], "complete")
            self.assertEqual(task_completion["passed_check_count"], 6)
            self.assertEqual(task_completion["failed_check_count"], 0)
            self.assertTrue(state_diff["changed"])
            self.assertTrue(any(change["path"] == "support.tickets.SUP-42" for change in state_diff["changes"]))
            self.assertEqual(run_cli(["validate", "--run", str(out), "--strict"]), 0)
            self.assertEqual(run_cli(["schemas", "--check", str(out / "scorecard.json")]), 0)
            self.assertEqual(run_cli(["schemas", "--check", str(out / "task_completion.json")]), 0)

    def test_support_ticket_bad_trace_fails_despite_final_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "bad"

            code = run_cli(["run", "--scenario", str(EXAMPLE_DIR / "support_ticket_completion_bad.json"), "--out", str(out)])

            self.assertEqual(code, 0)
            scorecard = json.loads((out / "scorecard.json").read_text(encoding="utf-8"))
            task_completion = json.loads((out / "task_completion.json").read_text(encoding="utf-8"))
            final_rule = next(rule for rule in scorecard["rules"] if rule["id"] == "final_answer")

            self.assertFalse(scorecard["passed"])
            self.assertEqual(scorecard["score"], 0)
            self.assertTrue(final_rule["passed"])
            self.assertIn("required_actions", scorecard["critical_failures"])
            self.assertIn("required_action_sequences", scorecard["critical_failures"])
            self.assertIn("required_event_counts", scorecard["critical_failures"])
            self.assertIn("required_state", scorecard["critical_failures"])
            self.assertIn("required_state_transitions", scorecard["critical_failures"])
            self.assertEqual(task_completion["status"], "incomplete")
            self.assertEqual(task_completion["passed_check_count"], 1)
            self.assertEqual(task_completion["failed_check_count"], 5)
            self.assertTrue((out / "regression_scenario.json").exists())
            self.assertEqual(run_cli(["validate", "--run", str(out), "--strict"]), 0)


if __name__ == "__main__":
    unittest.main()
