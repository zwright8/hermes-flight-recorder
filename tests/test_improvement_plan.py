import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main


ROOT = Path(__file__).resolve().parents[1]


def run_cli(args):
    with redirect_stdout(StringIO()):
        return main(args)


class ImprovementPlanTests(unittest.TestCase):
    def test_improvement_plan_joins_repair_curriculum_digest_and_bundle_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            plan_path = runs / "improvement_plan.json"
            self.assertEqual(
                run_cli(
                    [
                        "run-suite",
                        "--scenarios",
                        str(ROOT / "scenarios"),
                        "--out",
                        str(runs),
                        "--export-rl",
                        "--validate",
                        "--strict",
                        "--evidence-handoff",
                    ]
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    [
                        "improvement-plan",
                        "--evidence-bundle",
                        str(runs / "evidence_bundle.json"),
                        "--repair-queue",
                        str(runs / "repair_queue.json"),
                        "--training-export",
                        str(runs / "training_export"),
                        "--runs",
                        str(runs),
                        "--out",
                        str(plan_path),
                    ]
                ),
                0,
            )
            self.assertEqual(run_cli(["validate", "--improvement-plan", str(plan_path), "--strict"]), 0)
            self.assertEqual(run_cli(["schemas", "--check", str(plan_path)]), 0)

            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            repair_queue = json.loads((runs / "repair_queue.json").read_text(encoding="utf-8"))
            bundle = json.loads((runs / "evidence_bundle.json").read_text(encoding="utf-8"))
            self.assertEqual(plan["schema_version"], "hfr.improvement_plan.v1")
            self.assertEqual(plan["decision"]["recommendation"], "run_improvement_iteration")
            self.assertEqual(plan["readiness"], "ready")
            self.assertEqual(plan["metrics"]["repair_backed_count"], repair_queue["item_count"])
            self.assertEqual(plan["metrics"]["curriculum_backed_count"], repair_queue["item_count"])
            self.assertEqual(plan["metrics"]["digest_backed_count"], repair_queue["item_count"])
            self.assertEqual(plan["metrics"]["bundle_action_count"], bundle["decision"]["next_action_count"])
            self.assertGreater(plan["metrics"]["evidence_ref_count"], 0)
            self.assertIn("repair", {row["id"] for row in plan["metrics"]["category_counts"]})
            self.assertIn("bundle_action", {row["id"] for row in plan["metrics"]["category_counts"]})
            self.assertEqual(plan["work_item_count"], len(plan["work_items"]))
            self.assertTrue(all(len(item["fingerprint"]) == 64 for item in plan["work_items"]))
            repair_items = [item for item in plan["work_items"] if item["category"] == "repair"]
            self.assertEqual(len(repair_items), repair_queue["item_count"])
            self.assertTrue(all(item["sources"]["curriculum_priorities"] for item in repair_items))
            self.assertTrue(all(item["sources"]["run_digest"] for item in repair_items))
            self.assertTrue(any(item["scenario_id"] == "prompt_injection_bad" for item in repair_items))

    def test_validate_rejects_stale_improvement_plan_metrics_and_fingerprints(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            plan_path = runs / "improvement_plan.json"
            summary_path = runs / "validation.json"
            self.assertEqual(
                run_cli(
                    [
                        "run-suite",
                        "--scenarios",
                        str(ROOT / "scenarios"),
                        "--out",
                        str(runs),
                        "--export-rl",
                        "--evidence-handoff",
                    ]
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    [
                        "improvement-plan",
                        "--evidence-bundle",
                        str(runs / "evidence_bundle.json"),
                        "--repair-queue",
                        str(runs / "repair_queue.json"),
                        "--training-export",
                        str(runs / "training_export"),
                        "--runs",
                        str(runs),
                        "--out",
                        str(plan_path),
                    ]
                ),
                0,
            )
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan["metrics"]["work_item_count"] += 1
            plan["work_items"][0]["fingerprint"] = "0" * 64
            plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--improvement-plan", str(plan_path), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("improvement_plan.metrics.work_item_count", errors)
            self.assertIn("fingerprint does not match item content", errors)


if __name__ == "__main__":
    unittest.main()
