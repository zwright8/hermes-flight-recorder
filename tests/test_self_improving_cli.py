from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main
from flightrecorder.review_semantics import branch_replay_state_fingerprint


def _run(args: list[str]) -> int:
    with redirect_stdout(StringIO()):
        return main(args)


class SelfImprovingCliTests(unittest.TestCase):
    def test_controller_plan_and_disposable_execute_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan_path = root / "plan.json"
            state_path = root / "state.json"
            self.assertEqual(
                _run(
                    [
                        "agentic-loop",
                        "controller-plan",
                        "--controller-id",
                        "cli-loop",
                        "--artifact-dir",
                        str(root / "artifacts"),
                        "--candidate-model",
                        "candidate-v2",
                        "--champion-model",
                        "champion-v1",
                        "--canary-percentage",
                        "1",
                        "--canary-percentage",
                        "100",
                        "--max-cost-usd",
                        "1",
                        "--max-duration-seconds",
                        "60",
                        "--max-attempts",
                        "30",
                        "--deadline-at",
                        "2099-01-01T00:00:00+00:00",
                        "--out",
                        str(plan_path),
                    ]
                ),
                0,
            )
            self.assertEqual(
                _run(
                    [
                        "agentic-loop",
                        "execute",
                        "--plan",
                        str(plan_path),
                        "--state",
                        str(state_path),
                        "--owner-id",
                        "cli-test",
                        "--approve-all",
                    ]
                ),
                0,
            )
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "complete")
            self.assertEqual(state["active_model"], "candidate-v2")

    def test_governance_and_intervention_commands_write_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records = root / "records.jsonl"
            records.write_text(
                json.dumps(
                    {
                        "episode_id": "e1",
                        "governance": {
                            "owner": "owner",
                            "tenant": "tenant",
                            "legal_basis": "contract",
                            "allowed_purposes": ["agent_training"],
                            "sensitivity": "internal",
                            "jurisdiction": "US",
                            "retention_expires_at": "2030-01-01T00:00:00+00:00",
                            "license": "internal-training",
                            "provenance": {"source": "fixture"},
                            "deletion_subject_ids": ["subject-1"],
                        },
                        "prompt": "Summarize the approved fixture.",
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            governance = root / "governance.json"
            self.assertEqual(
                _run(
                    [
                        "data-governance",
                        "check",
                        "--input",
                        str(records),
                        "--purpose",
                        "agent_training",
                        "--now",
                        "2028-01-01T00:00:00+00:00",
                        "--out",
                        str(governance),
                    ]
                ),
                0,
            )
            self.assertTrue(json.loads(governance.read_text(encoding="utf-8"))["passed"])

            cluster = root / "cluster.json"
            cluster.write_text(
                json.dumps(
                    {
                        "cluster_id": "c1",
                        "failure_modes": ["invalid_tool_arguments"],
                        "severity": "high",
                        "confidence": 0.9,
                        "frequency": 3,
                        "evidence_refs": [],
                    }
                ),
                encoding="utf-8",
            )
            route = root / "route.json"
            self.assertEqual(
                _run(["intervention-route", "--cluster", str(cluster), "--out", str(route)]),
                0,
            )
            self.assertEqual(json.loads(route.read_text(encoding="utf-8"))["selected_intervention"], "tool_schema")

    def test_review_semantics_commands_emit_native_credit_replay_and_curation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trajectory = {
                "episode_id": "e1",
                "messages": [
                    {"role": "user", "content": "Find the approved fixture."},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "fixture.search", "arguments": {"query": "approved"}},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call-1", "name": "fixture.search", "status": "ok", "content": "found"},
                ],
            }
            trajectory_path = root / "trajectory.json"
            trajectory_path.write_text(json.dumps(trajectory), encoding="utf-8")
            credit_path = root / "credit.jsonl"
            self.assertEqual(
                _run(["review-semantics", "action-credit", "--trajectory", str(trajectory_path), "--out", str(credit_path)]),
                0,
            )
            self.assertEqual(json.loads(credit_path.read_text(encoding="utf-8"))["label"], "positive")

            replay_request = {
                "source_trajectory": trajectory,
                "replay_point": {
                    "event_index": 1,
                    "state_fingerprint": branch_replay_state_fingerprint(trajectory, 1),
                },
                "candidates": [
                    {"candidate_id": "good", "continuation": [{"role": "assistant", "content": "verified"}]},
                    {"candidate_id": "bad", "continuation": [{"role": "assistant", "content": "guess"}]},
                ],
                "verifier_results": [
                    {"candidate_id": "good", "passed": True, "safe": True, "score": 1, "confidence": 0.95},
                    {"candidate_id": "bad", "passed": False, "safe": True, "score": 0, "confidence": 0.95},
                ],
                "high_impact": True,
            }
            replay_request_path = root / "replay_request.json"
            replay_request_path.write_text(json.dumps(replay_request), encoding="utf-8")
            replay_path = root / "replay.json"
            self.assertEqual(
                _run(["review-semantics", "branch-replay", "--request", str(replay_request_path), "--out", str(replay_path)]),
                0,
            )
            self.assertEqual(json.loads(replay_path.read_text(encoding="utf-8"))["chosen_candidate_id"], "good")

            rows_path = root / "rows.jsonl"
            rows_path.write_text(
                json.dumps({"row_id": "r1", "source_id": "s1", "training_role": "action_sft", "quality_score": 1.0}) + "\n",
                encoding="utf-8",
            )
            recipe_path = root / "recipe.json"
            recipe_path.write_text(json.dumps({"seed": "fixed", "max_rows": 1}), encoding="utf-8")
            curated_path = root / "curated.json"
            self.assertEqual(
                _run(["review-semantics", "curate", "--input", str(rows_path), "--recipe", str(recipe_path), "--out", str(curated_path)]),
                0,
            )
            self.assertEqual(json.loads(curated_path.read_text(encoding="utf-8"))["selected_count"], 1)


if __name__ == "__main__":
    unittest.main()
