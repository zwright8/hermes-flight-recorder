from __future__ import annotations

import unittest

from flightrecorder.intervention_router import route_failure_cluster
from flightrecorder.schema_registry import check_schema_contract


class InterventionRouterTests(unittest.TestCase):
    def test_routes_known_failure_classes_to_least_cost_adequate_intervention(self) -> None:
        cases = {
            "prompt_policy": ["ambiguous_instruction"],
            "tool_schema": ["invalid_tool_arguments"],
            "parser_runtime": ["tool_result_parse_error"],
            "planner_routing": ["planning_loop"],
            "memory_retrieval": ["retrieval_miss"],
            "guardrail_sandbox": ["prompt_injection"],
            "dataset": ["training_data_contamination"],
            "evaluation": ["insufficient_eval_repeats"],
            "model_training": ["model_capability_shortfall"],
        }
        for expected, failure_modes in cases.items():
            with self.subTest(expected=expected):
                result = route_failure_cluster(
                    {
                        "cluster_id": f"cluster-{expected}",
                        "failure_modes": failure_modes,
                        "severity": "high",
                        "confidence": 0.95,
                        "affected_task_families": ["mail"],
                        "affected_tools": ["mail.search"],
                        "affected_policies": ["safe-v1"],
                        "evidence_refs": [{"artifact": "scorecard.json", "sha256": "a" * 64}],
                        "frequency": 5,
                    }
                )
                self.assertEqual(result["selected_intervention"], expected)
                self.assertTrue(result["work_item"]["acceptance_metrics"])
                self.assertTrue(result["rejected_alternatives"])
                self.assertTrue(check_schema_contract(result, name_or_id="intervention_route")["passed"])

    def test_unknown_or_low_confidence_routes_to_review_not_training(self) -> None:
        result = route_failure_cluster(
            {
                "cluster_id": "cluster-unknown",
                "failure_modes": ["unexpected_behavior"],
                "severity": "critical",
                "confidence": 0.3,
                "evidence_refs": [],
                "frequency": 1,
            }
        )
        self.assertEqual(result["selected_intervention"], "human_review")
        self.assertNotEqual(result["selected_intervention"], "model_training")
        self.assertIn("low_confidence", result["routing_reasons"])

    def test_critical_and_grader_disagreement_clusters_require_human_review(self) -> None:
        for failure_modes, severity, expected_reason in (
            (["invalid_tool_arguments"], "critical", "high_impact_requires_human_review"),
            (["grader_disagreement"], "high", "grader_disagreement_requires_human_review"),
        ):
            with self.subTest(failure_modes=failure_modes):
                result = route_failure_cluster(
                    {
                        "cluster_id": "cluster-review",
                        "failure_modes": failure_modes,
                        "severity": severity,
                        "confidence": 0.95,
                        "evidence_refs": [{"artifact": "review.json", "sha256": "d" * 64}],
                        "frequency": 3,
                    }
                )
                self.assertEqual(result["selected_intervention"], "human_review")
                self.assertIn(expected_reason, result["routing_reasons"])

    def test_model_training_without_capability_evidence_requires_review(self) -> None:
        result = route_failure_cluster(
            {
                "cluster_id": "cluster-capability",
                "failure_modes": ["model_capability_shortfall"],
                "severity": "high",
                "confidence": 0.95,
                "evidence_refs": [],
                "frequency": 4,
            }
        )
        self.assertEqual(result["selected_intervention"], "human_review")
        self.assertIn("model_training_requires_capability_evidence", result["routing_reasons"])

    def test_model_training_requires_capability_evidence(self) -> None:
        result = route_failure_cluster(
            {
                "cluster_id": "cluster-quality",
                "failure_modes": ["low_final_answer_quality"],
                "severity": "medium",
                "confidence": 0.9,
                "evidence_refs": [{"artifact": "eval.json", "sha256": "b" * 64}],
                "frequency": 10,
            }
        )
        self.assertEqual(result["selected_intervention"], "prompt_policy")
        self.assertIn("model_training", {row["intervention"] for row in result["rejected_alternatives"]})

    def test_route_is_deterministic_and_evidence_bound(self) -> None:
        cluster = {
            "cluster_id": "cluster-parser",
            "failure_modes": ["tool_result_parse_error"],
            "severity": "high",
            "confidence": 0.91,
            "evidence_refs": [{"artifact": "trace.json", "sha256": "c" * 64}],
            "frequency": 2,
        }
        first = route_failure_cluster(cluster)
        second = route_failure_cluster(cluster)
        self.assertEqual(first, second)
        changed = route_failure_cluster({**cluster, "frequency": 3})
        self.assertNotEqual(first["routing_fingerprint"], changed["routing_fingerprint"])


if __name__ == "__main__":
    unittest.main()
