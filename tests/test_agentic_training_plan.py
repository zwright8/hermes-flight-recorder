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
            self.assertEqual(plan["mode_contract"]["category"], "default_executable")
            self.assertTrue(plan["mode_contract"]["planning_gate"]["open"])
            self.assertEqual(plan["mode_contract"]["reward_contract"]["kind"], "preference_pairs")
            self.assertTrue(all(requirement["satisfied"] for requirement in plan["mode_contract"]["data_requirements"]))
            self.assertFalse(plan["mode_contract"]["side_effect_boundary"]["cloud_jobs_started"])
            self.assertFalse(plan["execution"]["cloud_jobs_started"])
            self.assertFalse(plan["execution"]["paid_model_grader_calls_started"])
            self.assertFalse(plan["execution"]["weights_updated"])
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
            self.assertIn("mode_contract_planning_gate_open", {check["id"] for check in blocked["checks"] if not check["passed"]})
            self.assertEqual(blocked["mode_contract"]["category"], "future_rl")
            self.assertFalse(blocked["mode_contract"]["planning_gate"]["open"])
            self.assertEqual(blocked["mode_contract"]["planning_gate"]["required_flag"], "--allow-future-rl")
            self.assertEqual(blocked["mode_contract"]["reward_contract"]["kind"], "trl_grpo_reward_function")
            self.assertTrue(blocked["mode_contract"]["reward_contract"]["external_runner_must_supply"])
            self.assertIn("reward_fn(prompts, completions", blocked["mode_contract"]["reward_contract"]["callable_signature"])
            self.assertTrue(allowed["passed"], allowed["blocked_reasons"])
            self.assertTrue(allowed["mode_contract"]["planning_gate"]["open"])
            self.assertEqual(allowed["mode_contract"]["data_requirements"][1]["id"], "external_reward_function_contract")
            self.assertTrue(all(requirement["satisfied"] for requirement in allowed["mode_contract"]["data_requirements"]))

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
            self.assertFalse(process_blocked["mode_contract"]["planning_gate"]["open"])
            self.assertEqual(process_blocked["mode_contract"]["reward_contract"]["kind"], "step_rewards")
            self.assertTrue(process_allowed["passed"], process_allowed["blocked_reasons"])
            self.assertEqual(process_allowed["mode_contract"]["category"], "advanced_reward")
            self.assertEqual(process_allowed["mode_contract"]["data_requirements"][0]["id"], "step_reward_rows")
            self.assertTrue(process_allowed["mode_contract"]["reward_contract"]["requires_calibration_or_human_review_gate"])

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

    def test_cli_writes_schema_checkable_blocked_grpo_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model.json"
            dataset = root / "dataset.json"
            out = root / "grpo_plan.json"
            self.write_model_manifest(model)
            self.write_dataset_manifest(dataset)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "plan_agentic_training.py"),
                    "--mode",
                    "grpo",
                    "--model-manifest",
                    str(model),
                    "--dataset-manifest",
                    str(dataset),
                    "--out",
                    str(out),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(completed.returncode, 1, completed.stderr + completed.stdout)
            plan = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse(plan["passed"])
            self.assertEqual(plan["recommendation"], "block_external_training")
            self.assertEqual(plan["mode_contract"]["planning_gate"]["required_flag"], "--allow-future-rl")
            self.assertEqual(plan["mode_contract"]["reward_contract"]["kind"], "trl_grpo_reward_function")
            self.assertFalse(plan["mode_contract"]["side_effect_boundary"]["provider_credentials_required_by_flight_recorder"])
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

    def test_schema_rejects_tampered_mode_contract_side_effects(self):
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
            plan["mode_contract"]["reward_contract"]["flight_recorder_supplies_callable"] = True
            plan["mode_contract"]["side_effect_boundary"]["cloud_jobs_started"] = True

            schema = check_schema_contract(plan)

            self.assertFalse(schema["passed"])
            errors = "\n".join(schema["errors"])
            self.assertIn("flight_recorder_supplies_callable", errors)
            self.assertIn("cloud_jobs_started", errors)

    def test_schema_rejects_mode_specific_contract_tampering(self):
        cases = [
            (
                "reward_model",
                {"allow_advanced_training": True},
                "--allow-advanced-training",
                "scalar_or_preference_rewards",
                "",
                "rl_reward_model",
            ),
            (
                "process_rewards",
                {"allow_advanced_training": True},
                "--allow-advanced-training",
                "step_rewards",
                "",
                "rl_step_reward",
            ),
            (
                "grpo",
                {"allow_future_rl": True},
                "--allow-future-rl",
                "trl_grpo_reward_function",
                "reward_fn(prompts, completions, **kwargs) -> list[float]",
                "rl_episode",
            ),
            (
                "rl",
                {"allow_future_rl": True},
                "--allow-future-rl",
                "external_rl_reward_function",
                "reward_fn(episodes, actions, **kwargs) -> list[float]",
                "rl_episode",
            ),
        ]
        for mode, flags, required_flag, kind, callable_signature, schema_name in cases:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                model = root / "model.json"
                dataset = root / "dataset.json"
                self.write_model_manifest(model)
                self.write_dataset_manifest(dataset)
                plan = build_agentic_training_plan(
                    out_path=root / f"{mode}.json",
                    mode=mode,
                    model_manifest_path=model,
                    dataset_manifest_path=dataset,
                    **flags,
                )
                self.assertTrue(check_schema_contract(plan)["passed"])
                self.assertEqual(plan["mode_contract"]["planning_gate"]["required_flag"], required_flag)
                self.assertEqual(plan["mode_contract"]["reward_contract"]["kind"], kind)
                self.assertEqual(plan["mode_contract"]["reward_contract"]["callable_signature"], callable_signature)
                self.assertIn(schema_name, plan["mode_contract"]["data_requirements"][0]["required_schema_names"])

                tampered_kind = json.loads(json.dumps(plan))
                tampered_kind["mode_contract"]["reward_contract"]["kind"] = "not_applicable"
                self.assertFalse(check_schema_contract(tampered_kind)["passed"])

                tampered_signature = json.loads(json.dumps(plan))
                tampered_signature["mode_contract"]["reward_contract"]["callable_signature"] = "reward_fn(*args) -> float"
                self.assertFalse(check_schema_contract(tampered_signature)["passed"])

                tampered_flag = json.loads(json.dumps(plan))
                tampered_flag["mode_contract"]["planning_gate"]["required_flag"] = "--allow-training"
                self.assertFalse(check_schema_contract(tampered_flag)["passed"])

                tampered_schema = json.loads(json.dumps(plan))
                tampered_schema["mode_contract"]["data_requirements"][0]["required_schema_names"] = ["rl_sft"]
                self.assertFalse(check_schema_contract(tampered_schema)["passed"])

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
        self.assertEqual(plan["mode_contract"]["reward_contract"]["kind"], "preference_pairs")
        self.assertTrue(all(requirement["satisfied"] for requirement in plan["mode_contract"]["data_requirements"]))
        schema = check_schema_file(plan_path)
        self.assertTrue(schema["passed"], schema["errors"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
