import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / "scripts" / "build_agentic_finetune_experiment.py"
TRAIN_SCRIPT = ROOT / "scripts" / "train_agentic_lora.py"


class AgenticFinetuneExperimentTests(unittest.TestCase):
    def test_standard_export_supplies_dpo_rows_to_sft_then_dpo(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiment = Path(tmp) / "experiment"
            built = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    str(BUILD_SCRIPT),
                    "--runs-dir",
                    str(ROOT / "examples" / "agentic_training"),
                    "--out",
                    str(experiment),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(built.returncode, 0, built.stderr + built.stdout)

            stats = json.loads((experiment / "stats.json").read_text(encoding="utf-8"))
            counts = stats["dataset_counts"]
            self.assertGreater(counts["flightrecorder_scorecard_dpo"], 0)
            self.assertEqual(
                counts["flightrecorder_combined_dpo"],
                counts["flightrecorder_scorecard_dpo"],
            )
            action_rows = [
                json.loads(line)
                for line in (experiment / "data" / "flightrecorder_action_sft.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertTrue(action_rows)
            self.assertTrue(
                any(
                    message.get("role") == "assistant" and message.get("tool_calls")
                    for row in action_rows
                    for message in row["messages"]
                )
            )
            self.assertTrue(
                any(message.get("role") == "tool" for row in action_rows for message in row["messages"])
            )
            dataset_manifest = json.loads(
                (experiment / "dataset_training_manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(dataset_manifest["gates"]["training_gate"]["passed"])
            self.assertTrue(dataset_manifest["leakage_checks"]["heldout_scenario_exclusive"])
            self.assertIn("flightrecorder_action_sft", dataset_manifest["artifact_fingerprints"])
            self.assertEqual(
                dataset_manifest["data_files"]["train_action_sft"],
                "data/flightrecorder_action_sft.jsonl",
            )

            planned = subprocess.run(
                [
                    sys.executable,
                    str(TRAIN_SCRIPT),
                    "--mode",
                    "fr_sft_dpo",
                    "--dry-run",
                    "--experiment-dir",
                    str(experiment),
                    "--output-dir",
                    str(Path(tmp) / "out"),
                    "--disable-trackio",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(planned.returncode, 0, planned.stderr + planned.stdout)
            plan = json.loads(planned.stdout)
            self.assertTrue(plan["passed"])
            self.assertGreater(plan["prepared_counts"]["dpo"], 0)

    def test_builder_fails_closed_without_split_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            training = root / "runs" / "training_export"
            training.mkdir(parents=True)
            (training / "dataset_metrics.json").write_text(
                json.dumps({"redaction_status": {"passed": True}}),
                encoding="utf-8",
            )

            built = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    str(BUILD_SCRIPT),
                    "--runs-dir",
                    str(root / "runs"),
                    "--out",
                    str(root / "experiment"),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(built.returncode, 0)
            self.assertIn("deterministic split assignments", built.stderr)


if __name__ == "__main__":
    unittest.main()
