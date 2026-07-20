import csv
import json
import unittest
from pathlib import Path

from flightrecorder.schema_registry import check_schema_file, check_schema_jsonl_file


ROOT = Path(__file__).resolve().parents[1]
CASE_STUDY = ROOT / "examples" / "case_studies" / "qwen3_0_6b_flightrecorder_lora"
SELF_IMPROVING_README = ROOT / "examples" / "self_improving_loop" / "README.md"


class AgenticLoraCaseStudyTests(unittest.TestCase):
    def test_redacted_action_trajectory_matches_public_schema(self):
        data_path = CASE_STUDY / "data" / "action_sft.jsonl"
        result = check_schema_jsonl_file(data_path, "rl_action_sft")

        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["row_count"], 1)

        row = json.loads(data_path.read_text(encoding="utf-8"))
        self.assertEqual([message["role"] for message in row["messages"]], [
            "user",
            "assistant",
            "tool",
            "assistant",
            "tool",
            "assistant",
        ])
        self.assertEqual(row["tool_call_count"], 2)
        self.assertEqual(row["tool_result_count"], 2)
        self.assertEqual(row["source_fingerprint_status"], "verified")

    def test_model_decision_matches_public_schema(self):
        result = check_schema_file(CASE_STUDY / "model_manifest.json", "model_candidate")
        self.assertTrue(result["passed"], result["errors"])

    def test_evaluation_keeps_the_claim_narrow(self):
        evaluation_path = CASE_STUDY / "evaluation.json"
        schema = check_schema_file(evaluation_path, "finetune_demo_evaluation")
        self.assertTrue(schema["passed"], schema["errors"])
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))

        self.assertTrue(evaluation["sequence_loss_improved"])
        self.assertLess(evaluation["tuned"]["sequence_loss"], evaluation["base"]["sequence_loss"])
        self.assertIn("memorization", evaluation["claim_scope"].lower())
        self.assertIn("not a held-out", evaluation["claim_scope"].lower())

    def test_training_curve_and_repository_artifacts_are_portable(self):
        with (CASE_STUDY / "training_curve.csv").open(newline="", encoding="utf-8") as handle:
            curve = list(csv.DictReader(handle))

        self.assertEqual(len(curve), 20)
        self.assertGreater(float(curve[0]["loss"]), float(curve[-1]["loss"]))
        self.assertEqual(float(curve[-1]["mean_token_accuracy"]), 1.0)
        self.assertEqual(list(CASE_STUDY.rglob("*.safetensors")), [])

        for path in CASE_STUDY.rglob("*"):
            if path.is_file():
                self.assertNotIn("/Users/", path.read_text(encoding="utf-8"), path)

    def test_self_improving_dry_run_registers_both_inputs(self):
        readme = SELF_IMPROVING_README.read_text(encoding="utf-8")
        self.assertIn(
            "--model-manifest examples/case_studies/"
            "qwen3_0_6b_flightrecorder_lora/model_manifest.json",
            readme,
        )
        self.assertIn("--dataset-manifest runs/self_improving_loop/", readme)


if __name__ == "__main__":
    unittest.main()
