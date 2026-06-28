import unittest
from pathlib import Path

from flightrecorder.adapters import normalize_trace
from flightrecorder.schema import load_scenario
from flightrecorder.scorers import score_trace
from flightrecorder.state import load_state_snapshot, resolve_before_state_snapshot_path, resolve_state_snapshot_path


ROOT = Path(__file__).resolve().parents[1]


class ScorerTests(unittest.TestCase):
    def _score(self, scenario_name):
        scenario = load_scenario(ROOT / "scenarios" / scenario_name)
        trace = normalize_trace(ROOT / scenario["trace"]["path"].replace("../", ""))
        before_state_path = resolve_before_state_snapshot_path(scenario)
        state_path = resolve_state_snapshot_path(scenario)
        before_state = load_state_snapshot(before_state_path) if before_state_path is not None else None
        state = load_state_snapshot(state_path) if state_path is not None else None
        return score_trace(scenario, trace, state, before_state)

    def test_prompt_injection_good_passes(self):
        scorecard = self._score("prompt_injection_good.json")

        self.assertTrue(scorecard["passed"])
        self.assertEqual(scorecard["score"], 100)
        self.assertEqual(scorecard["task_completion"]["status"], "complete")

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

    def test_cron_async_delegation_fails_when_batch_completion_is_missing(self):
        scorecard = self._score("cron_async_delegation_bad.json")

        self.assertFalse(scorecard["passed"])
        self.assertEqual(scorecard["score"], 10)
        self.assertEqual(
            scorecard["critical_failures"],
            ["required_evidence", "required_action_sequences", "required_event_counts"],
        )
        action_rule = next(rule for rule in scorecard["rules"] if rule["id"] == "required_actions")
        self.assertTrue(action_rule["passed"])
        self.assertEqual([item["id"] for item in action_rule["items"]], [
            "dispatch_background_delegate_batch",
            "inbox_child_completed",
            "calendar_child_completed",
        ])
        sequence_rule = next(rule for rule in scorecard["rules"] if rule["id"] == "required_action_sequences")
        self.assertFalse(sequence_rule["passed"])
        self.assertEqual(sequence_rule["items"][0]["event_indices"], [2, 5, 6])
        count_rule = next(rule for rule in scorecard["rules"] if rule["id"] == "required_event_counts")
        self.assertFalse(count_rule["passed"])
        self.assertEqual(count_rule["items"][0]["actual_count"], 0)
        final_rule = next(rule for rule in scorecard["rules"] if rule["id"] == "final_answer")
        self.assertTrue(final_rule["passed"])
        task = scorecard["task_completion"]
        self.assertEqual(task["status"], "incomplete")
        self.assertEqual(task["required_check_count"], 6)
        self.assertEqual(task["passed_check_count"], 3)
        self.assertEqual(task["failed_check_count"], 3)
        self.assertEqual(
            task["blocking_rule_ids"],
            ["required_action_sequences", "required_event_counts", "required_evidence"],
        )

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
        task = scorecard["task_completion"]
        self.assertEqual(task["schema_version"], "hfr.task_completion.v1")
        self.assertEqual(task["status"], "complete")
        self.assertTrue(task["passed"])
        self.assertEqual(task["required_check_count"], 6)
        self.assertEqual(task["passed_check_count"], 6)
        self.assertEqual(task["failed_check_count"], 0)
        self.assertEqual(
            {check["rule_id"] for check in task["checks"]},
            {
                "required_actions",
                "required_action_sequences",
                "required_event_counts",
                "required_evidence",
                "required_state",
                "required_state_transitions",
            },
        )

    def test_required_actions_reject_final_answer_completion_claim_without_send_evidence(self):
        scorecard = self._score("email_reply_completion_bad.json")

        self.assertFalse(scorecard["passed"])
        self.assertEqual(scorecard["score"], 0)
        self.assertIn("required_actions", scorecard["critical_failures"])
        self.assertIn("required_action_sequences", scorecard["critical_failures"])
        self.assertIn("required_event_counts", scorecard["critical_failures"])
        self.assertIn("required_state", scorecard["critical_failures"])
        self.assertIn("required_state_transitions", scorecard["critical_failures"])
        final_rule = next(rule for rule in scorecard["rules"] if rule["id"] == "final_answer")
        self.assertTrue(final_rule["passed"])
        action_rule = next(rule for rule in scorecard["rules"] if rule["id"] == "required_actions")
        self.assertIn("missing required action", action_rule["evidence"][0])
        task = scorecard["task_completion"]
        self.assertEqual(task["status"], "incomplete")
        self.assertFalse(task["passed"])
        self.assertEqual(task["required_check_count"], 6)
        self.assertEqual(task["passed_check_count"], 1)
        self.assertEqual(task["failed_check_count"], 5)
        self.assertEqual(
            task["blocking_rule_ids"],
            [
                "required_action_sequences",
                "required_actions",
                "required_event_counts",
                "required_state",
                "required_state_transitions",
            ],
        )
        self.assertTrue(task["missing_evidence_refs"])

    def test_required_state_rejects_missing_post_run_snapshot(self):
        scenario = load_scenario(ROOT / "scenarios" / "email_reply_completion_good.json")
        trace = normalize_trace(ROOT / "fixtures" / "email_reply_completion_good.observer.jsonl")

        scorecard = score_trace(scenario, trace)

        self.assertFalse(scorecard["passed"])
        self.assertIn("required_state", scorecard["critical_failures"])
        state_rule = next(rule for rule in scorecard["rules"] if rule["id"] == "required_state")
        self.assertFalse(state_rule["passed"])
        self.assertEqual(state_rule["items"][0]["evidence_refs"][0]["target"], "state_snapshot")

    def test_required_state_transitions_reject_missing_before_snapshot(self):
        scenario = load_scenario(ROOT / "scenarios" / "email_reply_completion_good.json")
        trace = normalize_trace(ROOT / "fixtures" / "email_reply_completion_good.observer.jsonl")
        state = load_state_snapshot(resolve_state_snapshot_path(scenario))

        scorecard = score_trace(scenario, trace, state)

        self.assertFalse(scorecard["passed"])
        self.assertIn("required_state_transitions", scorecard["critical_failures"])
        transition_rule = next(rule for rule in scorecard["rules"] if rule["id"] == "required_state_transitions")
        self.assertFalse(transition_rule["items"][0]["passed"])
        self.assertFalse(transition_rule["items"][0]["before_passed"])
        self.assertTrue(transition_rule["items"][0]["after_passed"])
        self.assertEqual(transition_rule["items"][0]["evidence_refs"][0]["phase"], "before")

    def test_required_state_supports_wildcard_collection_paths(self):
        scenario = _state_only_scenario(
            {
                "id": "slack_message_text_exists",
                "where": {
                    "slack.messages.*.text": {"contains": "deployment finished"},
                },
            }
        )
        trace = {"schema_version": "hfr.trace.v1", "events": [], "final_answer": ""}
        state = {
            "slack": {
                "messages": [
                    {"text": "hello", "user": "U1"},
                    {"text": "deployment finished successfully", "user": "U2"},
                ]
            }
        }

        scorecard = score_trace(scenario, trace, state)

        self.assertTrue(scorecard["passed"], scorecard)

    def test_required_state_where_any_requires_same_collection_item(self):
        scenario = _state_only_scenario(
            {
                "id": "same_slack_message_text_and_user",
                "where_any": {
                    "path": "slack.messages",
                    "where": {
                        "text": {"contains": "deployment finished"},
                        "user": "U1",
                    },
                },
            }
        )
        trace = {"schema_version": "hfr.trace.v1", "events": [], "final_answer": ""}
        split_state = {
            "slack": {
                "messages": [
                    {"text": "deployment finished successfully", "user": "U2"},
                    {"text": "different message", "user": "U1"},
                ]
            }
        }
        same_item_state = {
            "slack": {
                "messages": [
                    {"text": "deployment finished successfully", "user": "U1"},
                ]
            }
        }

        split_score = score_trace(scenario, trace, split_state)
        same_item_score = score_trace(scenario, trace, same_item_state)

        self.assertFalse(split_score["passed"])
        self.assertIn("required_state", split_score["critical_failures"])
        self.assertTrue(same_item_score["passed"], same_item_score)

    def test_required_actions_fail_when_observable_action_is_missing(self):
        scenario = load_scenario(ROOT / "scenarios" / "email_reply_completion_good.json")
        scenario["assertions"]["required_actions"][0]["where"]["result.thread_id"] = "email-999"
        trace = normalize_trace(ROOT / "fixtures" / "email_reply_completion_good.observer.jsonl")
        before_state = load_state_snapshot(resolve_before_state_snapshot_path(scenario))
        state = load_state_snapshot(resolve_state_snapshot_path(scenario))

        scorecard = score_trace(scenario, trace, state, before_state)

        self.assertFalse(scorecard["passed"])
        self.assertIn("required_actions", scorecard["critical_failures"])

    def test_required_action_sequences_fail_when_events_happen_out_of_order(self):
        scenario = load_scenario(ROOT / "scenarios" / "email_reply_completion_good.json")
        trace = normalize_trace(ROOT / "fixtures" / "email_reply_completion_good.observer.jsonl")
        before_state = load_state_snapshot(resolve_before_state_snapshot_path(scenario))
        state = load_state_snapshot(resolve_state_snapshot_path(scenario))
        send_result = trace["events"].pop(4)
        trace["events"].insert(2, send_result)

        scorecard = score_trace(scenario, trace, state, before_state)

        self.assertFalse(scorecard["passed"])
        self.assertIn("required_action_sequences", scorecard["critical_failures"])
        sequence_rule = next(rule for rule in scorecard["rules"] if rule["id"] == "required_action_sequences")
        self.assertFalse(sequence_rule["items"][0]["passed"])
        self.assertEqual(sequence_rule["items"][0]["event_indices"], [3])

    def test_required_event_counts_fail_when_action_happens_too_many_times(self):
        scenario = load_scenario(ROOT / "scenarios" / "email_reply_completion_good.json")
        trace = normalize_trace(ROOT / "fixtures" / "email_reply_completion_good.observer.jsonl")
        before_state = load_state_snapshot(resolve_before_state_snapshot_path(scenario))
        state = load_state_snapshot(resolve_state_snapshot_path(scenario))
        duplicate_send = dict(trace["events"][4])
        trace["events"].insert(5, duplicate_send)

        scorecard = score_trace(scenario, trace, state, before_state)

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
        before_state = load_state_snapshot(resolve_before_state_snapshot_path(scenario))
        state = load_state_snapshot(resolve_state_snapshot_path(scenario))

        scorecard = score_trace(scenario, trace, state, before_state)

        self.assertTrue(scorecard["passed"])

    def test_final_evidence_supports_contains_matcher(self):
        scenario = load_scenario(ROOT / "scenarios" / "email_reply_completion_good.json")
        scenario["assertions"]["required_evidence"].append(
            {"id": "final_mentions_thread", "type": "final_matches", "contains": "email-123"}
        )
        trace = normalize_trace(ROOT / "fixtures" / "email_reply_completion_good.observer.jsonl")
        before_state = load_state_snapshot(resolve_before_state_snapshot_path(scenario))
        state = load_state_snapshot(resolve_state_snapshot_path(scenario))

        scorecard = score_trace(scenario, trace, state, before_state)

        self.assertTrue(scorecard["passed"])


def _state_only_scenario(state_assertion):
    return {
        "id": "state_only",
        "title": "State Only",
        "policy": {},
        "assertions": {
            "required_state": [state_assertion],
        },
        "scoring": {"pass_threshold": 90},
    }


if __name__ == "__main__":
    unittest.main()
