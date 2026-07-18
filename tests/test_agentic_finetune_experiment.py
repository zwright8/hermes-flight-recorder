import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / "scripts" / "build_agentic_finetune_experiment.py"
TRAIN_SCRIPT = ROOT / "scripts" / "train_agentic_lora.py"
PREP_SCRIPT = ROOT / "scripts" / "prepare_self_improving_case_study.py"


class AgenticFinetuneExperimentTests(unittest.TestCase):
    def test_governed_curated_export_supplies_review_and_replay_dpo_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            experiment = Path(tmp) / "experiment"
            prepared = subprocess.run(
                [sys.executable, "-I", "-S", str(PREP_SCRIPT), "--out", str(runs)],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(prepared.returncode, 0, prepared.stderr + prepared.stdout)
            built = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    str(BUILD_SCRIPT),
                    "--runs-dir",
                    str(runs),
                    "--controls-dir",
                    str(runs),
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
            self.assertEqual(counts["flightrecorder_scorecard_dpo"], 0)
            self.assertEqual(counts["flightrecorder_combined_dpo"], 2)
            action_rows = [
                json.loads(line)
                for line in (experiment / "data" / "flightrecorder_action_sft.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertTrue(action_rows)
            self.assertEqual({row["human_label"] for row in action_rows}, {"accept"})
            self.assertEqual({row["tool_schema_provenance"] for row in action_rows}, {"recorded_exact"})
            self.assertNotIn("inventory-recovery", {row["episode_id"] for row in action_rows})
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
            self.assertTrue(dataset_manifest["gates"]["governance"]["passed"])
            self.assertTrue(dataset_manifest["gates"]["contamination"]["passed"])
            self.assertTrue(dataset_manifest["gates"]["per_action_credit"]["passed"])
            self.assertTrue(dataset_manifest["gates"]["verified_branch_replay"]["passed"])
            self.assertTrue(dataset_manifest["leakage_checks"]["heldout_scenario_exclusive"])
            self.assertIn("flightrecorder_action_sft", dataset_manifest["artifact_fingerprints"])
            self.assertEqual(
                dataset_manifest["data_files"]["train_action_sft"],
                "data/flightrecorder_action_sft.jsonl",
            )
            model_manifest = Path(tmp) / "model.json"
            model_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.model_candidate.v1",
                        "model_id": "local/test-model",
                        "license_status": "approved",
                        "training_allowed": True,
                        "compatibility": {"tokenizer": "fixture", "chat_template": "messages"},
                    }
                ),
                encoding="utf-8",
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
                    "--model",
                    "local/test-model",
                    "--model-manifest",
                    str(model_manifest),
                    "--dataset-manifest",
                    str(experiment / "dataset_training_manifest.json"),
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
            checks = {check["id"]: check for check in plan["checks"]}
            self.assertTrue(checks["self_improving_controls_passed"]["passed"])
            self.assertTrue(checks["action_sft_rows_are_governed_reviewed_and_exact"]["passed"])
            self.assertTrue(checks["dpo_uses_human_rejection_and_verified_branch_replay"]["passed"])

    def test_builder_fails_closed_without_split_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            prepared = subprocess.run(
                [sys.executable, "-I", "-S", str(PREP_SCRIPT), "--out", str(runs)],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(prepared.returncode, 0, prepared.stderr + prepared.stdout)
            (runs / "training_export" / "dataset_splits.json").unlink()

            built = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    str(BUILD_SCRIPT),
                    "--runs-dir",
                    str(runs),
                    "--controls-dir",
                    str(runs),
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
