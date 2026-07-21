import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from flightrecorder.lora_recipe_search import (
    LoraRecipeSearchError,
    build_search_plan,
    run_search,
    validate_promotion_handoff,
    validate_search_plan,
    validate_search_result,
    write_json,
)
from flightrecorder.repeated_eval import validate_promotion_evidence
from flightrecorder.schema_registry import check_schema_file
from scripts.run_lora_recipe_autoresearch_demo import run_demo


class _QueueProposer:
    def __init__(self, proposals):
        self.proposals = list(proposals)

    def propose(self, state):
        del state
        return self.proposals.pop(0) if self.proposals else None


class _Evaluator:
    def __init__(self, *, cost=0.01, duration=1.0):
        self.cost = cost
        self.duration = duration
        self.calls = []

    def evaluate(self, recipe, *, trial_id, development_suite_path):
        self.calls.append((trial_id, development_suite_path))
        metric = 0.5 + (0.1 if recipe["lora_r"] == 32 else 0.0)
        return {
            "status": "completed",
            "primary_metric": metric,
            "critical_failures": 0,
            "cost_usd": self.cost,
            "duration_seconds": self.duration,
            "candidate_identity_sha256": _digest(json.dumps(recipe, sort_keys=True)),
            "diagnostics": [],
            "execution_mode": "simulation",
            "external_side_effects_observed": False,
            "model_weights_updated_externally": False,
        }


class LoraRecipeSearchTests(unittest.TestCase):
    def test_demo_replays_search_and_promotion_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "demo"

            summary = run_demo(root)

            self.assertTrue(summary["passed"])
            self.assertEqual(summary["trial_count"], 6)
            self.assertGreaterEqual(summary["kept_trial_count"], 2)
            self.assertGreaterEqual(summary["discarded_trial_count"], 1)
            search_path = root / "search_result.json"
            handoff_path = root / "promotion_handoff.json"
            evidence_path = root / "promotion_evidence.json"
            self.assertTrue(validate_search_result(search_path)["passed"])
            self.assertTrue(validate_promotion_evidence(evidence_path)["passed"])
            self.assertTrue(validate_promotion_handoff(handoff_path)["passed"])
            for path in (
                root / "search_plan.json",
                search_path,
                handoff_path,
                evidence_path,
                *sorted(root.glob("trial-*.json")),
            ):
                result = check_schema_file(path)
                self.assertTrue(result["passed"], (path, result["errors"]))
            search = json.loads(search_path.read_text(encoding="utf-8"))
            self.assertEqual(search["heldout_access"], {"used_during_search": False, "artifact_count": 0, "artifacts": []})
            handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
            self.assertTrue(handoff["candidate_binding"]["matched"])
            self.assertEqual(handoff["readiness"], "ready_for_governance_review")
            self.assertFalse(handoff["execution_boundary"]["promotion_applied"])
            self.assertIn("Synthetic offline contract demonstration", (root / "REPORT.md").read_text(encoding="utf-8"))

    def test_search_plan_rejects_heldout_tagged_selector(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite_path = _write_development_suite(root, tags=["development", "frozen"])

            with self.assertRaisesRegex(LoraRecipeSearchError, "held-out tags"):
                _build_plan(root, suite_path)

    def test_unauthorized_mutation_is_blocked_without_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite_path = _write_development_suite(root)
            plan_path = _build_plan(root, suite_path)
            evaluator = _Evaluator()
            proposer = _QueueProposer(
                [
                    {
                        "proposal_id": "change-seed",
                        "hypothesis": "Changing the seed might create a favorable comparison.",
                        "mutations": {"seed": 99},
                        "estimated_cost_usd": 0.01,
                        "estimated_duration_seconds": 1.0,
                    }
                ]
            )

            result = run_search(
                plan_path=plan_path,
                out_path=root / "search_result.json",
                proposer=proposer,
                evaluator=evaluator,
                created_at="2026-07-20T00:00:00+00:00",
            )

            self.assertTrue(result["passed"])
            self.assertEqual(result["blocked_trial_count"], 1)
            self.assertEqual(len(evaluator.calls), 1, "only the baseline may reach the evaluator")
            blocked = json.loads((root / "trial-001-change-seed.json").read_text(encoding="utf-8"))
            self.assertEqual(blocked["decision"]["outcome"], "blocked")
            self.assertIn("outside the plan allowlist", blocked["proposal"]["validation_errors"][0])
            self.assertTrue(validate_search_result(root / "search_result.json")["passed"])

    def test_reported_budget_overrun_is_valid_evidence_but_blocks_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite_path = _write_development_suite(root)
            plan_path = _build_plan(root, suite_path)

            result = run_search(
                plan_path=plan_path,
                out_path=root / "search_result.json",
                proposer=_QueueProposer([]),
                evaluator=_Evaluator(cost=0.5),
                created_at="2026-07-20T00:00:00+00:00",
            )

            self.assertFalse(result["passed"])
            self.assertEqual(result["readiness"], "blocked")
            self.assertEqual(result["stop_reason"], "actual_budget_violation")
            baseline = json.loads((root / "trial-000-baseline.json").read_text(encoding="utf-8"))
            self.assertEqual(baseline["status"], "budget_violation")
            self.assertEqual(baseline["decision"]["outcome"], "blocked")
            self.assertTrue(validate_search_result(root / "search_result.json")["passed"])

    def test_trial_decision_tampering_is_detected_even_after_refingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite_path = _write_development_suite(root)
            plan_path = _build_plan(root, suite_path)
            result_path = root / "search_result.json"
            run_search(
                plan_path=plan_path,
                out_path=result_path,
                proposer=_QueueProposer(
                    [
                        {
                            "proposal_id": "rank-32",
                            "hypothesis": "Increase adapter rank.",
                            "mutations": {"lora_r": 32},
                            "estimated_cost_usd": 0.01,
                            "estimated_duration_seconds": 1.0,
                        }
                    ]
                ),
                evaluator=_Evaluator(),
                created_at="2026-07-20T00:00:00+00:00",
            )
            trial_path = root / "trial-001-rank-32.json"
            trial = json.loads(trial_path.read_text(encoding="utf-8"))
            trial["decision"]["outcome"] = "discard"
            trial["decision"]["reason"] = "tampered"
            write_json(trial_path, trial)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["trials"][1]["sha256"] = _file_sha256(trial_path)
            result["trials"][1]["size_bytes"] = trial_path.stat().st_size
            result["trials"][1]["outcome"] = "discard"
            write_json(result_path, result)

            validation = validate_search_result(result_path)

            self.assertFalse(validation["passed"])
            self.assertIn("decision does not replay", "\n".join(validation["errors"]))


def _build_plan(root: Path, suite_path: Path) -> Path:
    plan_path = root / "search_plan.json"
    plan = build_search_plan(
        campaign_id="test-search",
        objective="Test bounded recipe search.",
        development_suite_path=suite_path,
        base_recipe=_recipe(),
        mutable_fields=["lora_r", "lora_dropout", "max_steps"],
        budget={
            "max_trials": 3,
            "max_cost_usd": 0.1,
            "max_duration_seconds": 10.0,
            "per_trial_cost_ceiling_usd": 0.02,
            "per_trial_duration_ceiling_seconds": 2.0,
        },
        out_path=plan_path,
        minimum_delta=0.01,
        created_at="2026-07-20T00:00:00+00:00",
    )
    write_json(plan_path, plan)
    validation = validate_search_plan(plan_path)
    if not validation["passed"]:
        raise AssertionError(validation["errors"])
    return plan_path


def _write_development_suite(root: Path, tags=None) -> Path:
    path = root / "development_suite.json"
    write_json(
        path,
        {
            "schema_version": "hfr.eval_suite_manifest.v1",
            "suite_id": "test-development",
            "description": "Development-only selector for tests.",
            "tags": tags or ["development", "synthetic"],
            "scenario_ids": ["prompt_injection_good"],
        },
    )
    return path


def _recipe():
    return {
        "mode": "fr_sft",
        "sft_learning_rate": 0.0001,
        "dpo_learning_rate": 0.00001,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_steps": 80,
        "max_length": 640,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "seed": 42,
        "data_seed": 42,
    }


def _digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
