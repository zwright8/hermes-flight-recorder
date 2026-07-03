import hashlib
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
            self.assertEqual(plan["input_manifests"]["model"]["size_bytes"], model.stat().st_size)
            self.assertEqual(plan["input_manifests"]["dataset"]["size_bytes"], dataset.stat().st_size)
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

    def test_reward_and_process_modes_are_blocked_without_advanced_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model.json"
            dataset = root / "dataset.json"
            self.write_model_manifest(model)
            self.write_dataset_manifest(dataset)

            reward_blocked = build_agentic_training_plan(
                out_path=root / "reward_blocked.json",
                mode="reward_model",
                model_manifest_path=model,
                dataset_manifest_path=dataset,
            )
            process_blocked = build_agentic_training_plan(
                out_path=root / "process_blocked.json",
                mode="process_rewards",
                model_manifest_path=model,
                dataset_manifest_path=dataset,
            )
            process_allowed = build_agentic_training_plan(
                out_path=root / "process_allowed.json",
                mode="process_rewards",
                model_manifest_path=model,
                dataset_manifest_path=dataset,
                allow_advanced_training=True,
            )

            self.assertFalse(reward_blocked["passed"])
            self.assertFalse(process_blocked["passed"])
            self.assertIn(
                "advanced_reward_mode_explicitly_enabled",
                {check["id"] for check in process_blocked["checks"] if not check["passed"]},
            )
            self.assertTrue(process_allowed["passed"], process_allowed["blocked_reasons"])

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
                    "dpo",
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
            self.assertEqual(plan["mode"], "dpo")
            self.assertEqual(plan["selected_views"][0]["name"], "dpo")
            schema = check_schema_file(out)
            self.assertTrue(schema["passed"], schema["errors"])

    def test_schema_is_registered(self):
        names = {record["name"] for record in list_schema_records()}
        self.assertIn("agentic_training_plan", names)

    def test_schema_rejects_manifest_ref_without_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model.json"
            dataset = root / "dataset.json"
            self.write_model_manifest(model)
            self.write_dataset_manifest(dataset)
            plan = build_agentic_training_plan(
                out_path=root / "plan.json",
                mode="sft",
                model_manifest_path=model,
                dataset_manifest_path=dataset,
            )
            plan["input_manifests"]["model"].pop("size_bytes")

            schema = check_schema_contract(plan)

            self.assertFalse(schema["passed"])
            self.assertTrue(any("size_bytes" in error for error in schema["errors"]))

    def test_committed_example_plan_matches_registered_manifests(self):
        plan_path = ROOT / "examples" / "agentic_training" / "plans" / "sft_then_dpo_plan.json"
        model_path = ROOT / "examples" / "agentic_training" / "model_manifest.json"
        dataset_path = ROOT / "examples" / "agentic_training" / "dataset_manifest.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))

        self.assertTrue(plan["passed"], plan["blocked_reasons"])
        self.assertEqual(plan["created_at"], "2026-07-02T00:00:00+00:00")
        self.assertEqual(plan["trainer_plan"]["stage_sequence"], ["sft", "dpo"])
        self.assertEqual(plan["input_manifests"]["model"]["sha256"], sha256(model_path))
        self.assertEqual(plan["input_manifests"]["dataset"]["sha256"], sha256(dataset_path))
        self.assertEqual(plan["input_manifests"]["model"]["size_bytes"], model_path.stat().st_size)
        self.assertEqual(plan["input_manifests"]["dataset"]["size_bytes"], dataset_path.stat().st_size)
        self.assertEqual({view["name"] for view in plan["selected_views"]}, {"sft", "dpo"})
        schema = check_schema_file(plan_path)
        self.assertTrue(schema["passed"], schema["errors"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
