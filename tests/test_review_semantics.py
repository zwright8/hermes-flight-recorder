from __future__ import annotations

import unittest

from flightrecorder.data_governance import task_contract_fingerprint
from flightrecorder.review_semantics import (
    ReviewSemanticsError,
    branch_replay_state_fingerprint,
    build_action_credit,
    build_branch_replay_dataset,
    build_contract_preferences,
    build_reviewed_action_rows,
    curate_training_rows,
)
from flightrecorder.review import _reviewed_preferences
from flightrecorder.schema_registry import check_schema_contract


def _action_row(episode_id: str, prompt: str, tool: str = "mail.search") -> dict[str, object]:
    messages = [
        {"role": "system", "content": "Use tools safely."},
        {"role": "user", "content": prompt},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"call-{episode_id}",
                    "type": "function",
                    "function": {"name": tool, "arguments": {"query": prompt}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": f"call-{episode_id}", "name": tool, "content": '{"status":"ok"}'},
        {"role": "assistant", "content": "Done."},
    ]
    return {
        "schema_version": "hfr.rl.action_sft.v1",
        "episode_id": episode_id,
        "task_family": "mail",
        "prompt": prompt,
        "messages": messages,
        "tools": [{"type": "function", "function": {"name": tool, "description": "Search", "parameters": {"type": "object"}}}],
        "environment": {"fixture": "v1"},
        "policy": {"name": "safe"},
        "scenario_contract": {"id": prompt},
        "source_trace_sha256": "a" * 64,
    }


def _label(episode_id: str, human_label: str) -> dict[str, object]:
    return {
        "episode_id": episode_id,
        "review_item_id": f"review-{episode_id}",
        "review_item_sha256": (episode_id[-1] if episode_id else "a") * 64,
        "human_label": human_label,
        "reviewer_confidence": "high",
        "score": 100 if human_label == "accept" else 0,
    }


class ReviewSemanticsTests(unittest.TestCase):
    def test_reviewed_action_rows_preserve_native_structure_and_exclude_rejections(self) -> None:
        accepted = _action_row("episode-a", "Find invoice A")
        rejected = _action_row("episode-b", "Find invoice B")
        rows = build_reviewed_action_rows(
            [accepted, rejected],
            [_label("episode-a", "accept"), _label("episode-b", "reject")],
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["episode_id"], "episode-a")
        self.assertEqual(rows[0]["messages"], accepted["messages"])
        self.assertEqual(rows[0]["tools"], accepted["tools"])
        self.assertEqual(rows[0]["human_label"], "accept")
        self.assertRegex(rows[0]["task_contract_fingerprint"], r"^[0-9a-f]{64}$")
        self.assertTrue(check_schema_contract(rows[0], name_or_id="reviewed_action_sft")["passed"])

    def test_preferences_require_identical_task_contract_and_distinct_completions(self) -> None:
        chosen = {**_action_row("episode-a", "Same task"), **_label("episode-a", "accept"), "response": "correct"}
        rejected = {**_action_row("episode-b", "Same task"), **_label("episode-b", "reject"), "response": "wrong"}
        for row in (chosen, rejected):
            row["task_contract_fingerprint"] = task_contract_fingerprint(row)
        pairs = build_contract_preferences([chosen, rejected])
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["task_contract_fingerprint"], chosen["task_contract_fingerprint"])
        self.assertTrue(check_schema_contract(pairs[0], name_or_id="reviewed_contract_preference")["passed"])

        different = {**rejected, "prompt": "Different task"}
        different["task_contract_fingerprint"] = task_contract_fingerprint(different)
        self.assertEqual(build_contract_preferences([chosen, different]), [])

        identical = {**rejected, "response": "correct"}
        with self.assertRaises(ReviewSemanticsError):
            build_contract_preferences([chosen, identical])

    def test_legacy_review_export_no_longer_pairs_different_task_contracts(self) -> None:
        chosen = {
            **_label("episode-a", "accept"),
            "task_family": "mail",
            "prompt": "Send report A",
            "response": "sent",
            "scenario_id": "mail-a",
            "task_contract_fingerprint": "a" * 64,
        }
        rejected = {
            **_label("episode-b", "reject"),
            "task_family": "mail",
            "prompt": "Send report B",
            "response": "failed",
            "scenario_id": "mail-b",
            "task_contract_fingerprint": "b" * 64,
        }
        self.assertEqual(_reviewed_preferences([chosen, rejected], max_pairs_per_family=0), [])
        rejected["task_contract_fingerprint"] = chosen["task_contract_fingerprint"]
        self.assertEqual(len(_reviewed_preferences([chosen, rejected], max_pairs_per_family=0)), 1)

    def test_action_credit_does_not_make_recovered_failure_positive(self) -> None:
        row = _action_row("episode-a", "Recover")
        messages = list(row["messages"])
        messages[3] = {
            "role": "tool",
            "tool_call_id": "call-episode-a",
            "name": "mail.search",
            "status": "failed",
            "content": '{"status":"failed","error":"timeout"}',
        }
        messages.insert(
            4,
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call-retry", "type": "function", "function": {"name": "mail.search", "arguments": {"query": "Recover"}}}
                ],
            },
        )
        messages.insert(5, {"role": "tool", "tool_call_id": "call-retry", "name": "mail.search", "status": "ok", "content": "found"})
        credits = build_action_credit({**row, "messages": messages, "episode_outcome": "success"})
        by_id = {credit["tool_call_id"]: credit for credit in credits}
        self.assertEqual(by_id["call-episode-a"]["label"], "negative")
        self.assertEqual(by_id["call-retry"]["label"], "positive")
        self.assertEqual(sum(1 for credit in credits if credit["label"] == "negative"), 1)
        self.assertTrue(all(check_schema_contract(row, name_or_id="action_credit")["passed"] for row in credits))

    def test_branch_replay_selects_verified_candidate_and_routes_uncertainty(self) -> None:
        source = _action_row("episode-a", "Recover")
        state_fingerprint = branch_replay_state_fingerprint(source, 2)
        result = build_branch_replay_dataset(
            source_trajectory=source,
            replay_point={"event_index": 2, "state_fingerprint": state_fingerprint},
            candidates=[
                {"candidate_id": "bad", "continuation": [{"role": "assistant", "content": "guess"}]},
                {"candidate_id": "good", "continuation": [{"role": "assistant", "content": "verified"}]},
            ],
            verifier_results=[
                {"candidate_id": "bad", "passed": False, "score": 10, "safe": True, "confidence": 0.95},
                {"candidate_id": "good", "passed": True, "score": 95, "safe": True, "confidence": 0.92},
            ],
            high_impact=True,
            novel_behavior=False,
            grader_disagreement=False,
        )
        self.assertEqual(result["chosen_candidate_id"], "good")
        self.assertEqual(result["source_prefix_messages"], source["messages"][:2])
        self.assertEqual(result["preference_count"], 1)
        self.assertTrue(result["review_required"])
        self.assertIn("high_impact", result["review_reasons"])
        self.assertTrue(check_schema_contract(result, name_or_id="branch_replay_dataset")["passed"])

    def test_branch_replay_rejects_unbound_or_out_of_range_replay_points(self) -> None:
        source = _action_row("episode-a", "Recover")
        common = {
            "source_trajectory": source,
            "candidates": [{"candidate_id": "candidate", "continuation": [{"role": "assistant", "content": "ok"}]}],
            "verifier_results": [{"candidate_id": "candidate", "passed": True, "safe": True, "score": 100, "confidence": 1.0}],
            "high_impact": False,
            "novel_behavior": False,
            "grader_disagreement": False,
        }
        with self.assertRaisesRegex(ReviewSemanticsError, "between 1 and"):
            build_branch_replay_dataset(
                **common,
                replay_point={"event_index": 999, "state_fingerprint": "f" * 64},
            )
        with self.assertRaisesRegex(ReviewSemanticsError, "does not match"):
            build_branch_replay_dataset(
                **common,
                replay_point={"event_index": 2, "state_fingerprint": "f" * 64},
            )

    def test_curation_is_deterministic_and_records_reasons_and_mixture(self) -> None:
        rows = [
            {"row_id": "a", "task_family": "mail", "source_id": "s1", "training_role": "action_sft", "quality_score": 0.9},
            {"row_id": "b", "task_family": "mail", "source_id": "s1", "training_role": "action_sft", "quality_score": 0.8},
            {"row_id": "c", "task_family": "code", "source_id": "s2", "training_role": "sft", "quality_score": 0.7},
        ]
        recipe = {"seed": "fixed", "max_per_source": 1, "max_rows": 2, "mixture_weights": {"action_sft": 0.7, "sft": 0.3}}
        first = curate_training_rows(rows, recipe=recipe)
        second = curate_training_rows(rows, recipe=recipe)
        self.assertEqual(first, second)
        self.assertEqual(first["selected_count"], 2)
        self.assertEqual(first["excluded_count"], 1)
        self.assertIn("source_cap", {row["reason"] for row in first["excluded"]})
        self.assertEqual(first["recipe"]["mixture_weights"], recipe["mixture_weights"])
        self.assertTrue(check_schema_contract(first, name_or_id="curated_dataset")["passed"])


if __name__ == "__main__":
    unittest.main()
