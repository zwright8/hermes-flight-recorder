import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from flightrecorder.agentic_training_plan import build_agentic_training_plan
from flightrecorder.schema_registry import check_schema_contract, check_schema_file, list_schema_records


ROOT = Path(__file__).resolve().parents[1]


class AgenticTrainingPlanTests(unittest.TestCase):
    def write_model_manifest(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "schema_version": "hfr.model_candidate.test.v1",
                    "model_id": "local/test-agentic-model",
                    "candidate_id": "candidate",
                    "base_model": "local/base",
                    "license": {"status": "approved", "allow_training": True},
                    "compatibility": {"passed": True},
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def write_dataset_manifest(self, path: Path, *, redacted: bool = True) -> None:
        path.write_text(
            json.dumps(
                {
                    "schema_version": "hfr.dataset_manifest.test.v1",
                    "dataset_id": "agentic-smoke-dataset",
                    "dataset_version": "v1",
                    "license": {"status": "approved", "allow_training": True},
                    "redaction": {
                        "status": "redacted" if redacted else "raw",
                        "passed": redacted,
                        "contains_unredacted_traces": not redacted,
                    },
                    "views": {
                        "sft": {"path": "sft.jsonl", "row_count": 3, "schema_version": "hfr.rl.sft.v1"},
                        "dpo": {"path": "dpo.jsonl", "row_count": 2, "schema_version": "hfr.rl.dpo.v1"},
                        "reward_model": {
                            "path": "reward_model.jsonl",
                            "row_count": 2,
                            "schema_version": "hfr.rl.reward_model.v1",
                        },
                        "step_rewards": {
                            "path": "step_rewards.jsonl",
                            "row_count": 4,
                            "schema_version": "hfr.rl.step_reward.v1",
                        },
                        "episodes": {"path": "episodes.jsonl", "row_count": 5, "schema_version": "hfr.rl.episode.v1"},
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def test_sft_then_dpo_plan_requires_registered_redacted_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model.json"
            dataset = root / "dataset.json"
            out = root / "plan.json"
            self.write_model_manifest(model)
            self.write_dataset_manifest(dataset)

            plan = build_agentic_training_plan(
                out_path=out,
                mode="sft_then_dpo",
                model_manifest_path=model,
                dataset_manifest_path=dataset,
                trainer_backend="axolotl",
                output_dir=root / "adapters",
                limit=2,
            )

            self.assertTrue(plan["passed"], plan["blocked_reasons"])
            self.assertEqual(plan["recommendation"], "ready_for_external_trainer_plan")
            self.assertEqual(plan["trainer_plan"]["stage_sequence"], ["sft", "dpo"])
            self.assertEqual({view["name"] for view in plan["selected_views"]}, {"sft", "dpo"})
            self.assertFalse(plan["execution"]["training_started"])
            self.assertFalse(plan["handoff_contract"]["flight_recorder_executed_training"])
            schema = check_schema_contract(plan)
            self.assertTrue(schema["passed"], schema["errors"])

    def test_unredacted_dataset_blocks_training_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model.json"
            dataset = root / "dataset.json"
            self.write_model_manifest(model)
            self.write_dataset_manifest(dataset, redacted=False)

            plan = build_agentic_training_plan(
                out_path=root / "plan.json",
                mode="sft",
                model_manifest_path=model,
                dataset_manifest_path=dataset,
            )

            self.assertFalse(plan["passed"])
            self.assertEqual(plan["recommendation"], "block_external_training")
            self.assertIn("dataset_redaction_passed", {check["id"] for check in plan["checks"] if not check["passed"]})
            schema = check_schema_contract(plan)
            self.assertTrue(schema["passed"], schema["errors"])

    def test_future_rl_modes_require_explicit_enablement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model.json"
            dataset = root / "dataset.json"
            self.write_model_manifest(model)
            self.write_dataset_manifest(dataset)

            blocked = build_agentic_training_plan(
                out_path=root / "blocked.json",
                mode="grpo",
                model_manifest_path=model,
                dataset_manifest_path=dataset,
            )
            allowed = build_agentic_training_plan(
                out_path=root / "allowed.json",
                mode="grpo",
                model_manifest_path=model,
                dataset_manifest_path=dataset,
                allow_future_rl=True,
            )

            self.assertFalse(blocked["passed"])
            self.assertIn("future_rl_explicitly_enabled", {check["id"] for check in blocked["checks"] if not check["passed"]})
            self.assertTrue(allowed["passed"], allowed["blocked_reasons"])

    def test_cli_writes_schema_checkable_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model.json"
            dataset = root / "dataset.json"
            out = root / "plan.json"
            self.write_model_manifest(model)
            self.write_dataset_manifest(dataset)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "plan_agentic_training.py"),
                    "--mode",
                    "process_rewards",
                    "--model-manifest",
                    str(model),
                    "--dataset-manifest",
                    str(dataset),
                    "--trainer-backend",
                    "process-reward-wrapper",
                    "--out",
                    str(out),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            plan = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(plan["mode"], "process_rewards")
            self.assertEqual(plan["selected_views"][0]["name"], "step_rewards")
            schema = check_schema_file(out)
            self.assertTrue(schema["passed"], schema["errors"])

    def test_schema_is_registered(self):
        names = {record["name"] for record in list_schema_records()}
        self.assertIn("agentic_training_plan", names)
