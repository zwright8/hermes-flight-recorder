import copy
import json
import py_compile
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from flightrecorder.huggingface_lifecycle import (
    HuggingFaceLifecycleError,
    payload_identity,
    trainer_argv_from_plan,
    validate_reviewed_training_plan,
)
from flightrecorder.schema_registry import check_schema_contract


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "scripts" / "build_agentic_finetune_experiment.py"
PREPARE = ROOT / "scripts" / "prepare_huggingface_jobs_handoff.py"
TRAIN = ROOT / "scripts" / "train_agentic_lora.py"
PREPARE_CASE = ROOT / "scripts" / "prepare_self_improving_case_study.py"
PINNED_IMAGE = "registry.example/hfr-trainer@sha256:" + "c" * 64


class HuggingFaceJobsHandoffTests(unittest.TestCase):
    def _prepare_fixture(self, root: Path, *, extra_prepare: list[str] | None = None) -> tuple[Path, dict, dict]:
        experiment = root / "experiment"
        handoff_dir = root / "hf_jobs"
        model_manifest = root / "model.json"
        model_manifest.write_text(
            json.dumps(
                {
                    "schema_version": "hfr.model_candidate.v1",
                    "candidate_id": "qwen-agentic-candidate",
                    "model_id": "Qwen/Qwen3-4B-Instruct-2507",
                    "source": {"type": "huggingface", "revision": "a" * 40},
                    "license": {
                        "status": "approved",
                        "review_status": "approved",
                        "accepted_terms": True,
                        "training_allowed": True,
                    },
                    "training_allowed": True,
                    "compatibility": {
                        "tokenizer": {"repo_id": "Qwen/Qwen3-4B-Instruct-2507", "revision": "b" * 40},
                        "chat_template": {"format": "messages_and_tools", "sha256": "d" * 64},
                    },
                }
            ),
            encoding="utf-8",
        )
        case_runs = root / "case_runs"
        prepared_case = subprocess.run(
            [sys.executable, "-I", "-S", str(PREPARE_CASE), "--out", str(case_runs)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(prepared_case.returncode, 0, prepared_case.stderr + prepared_case.stdout)
        built = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                str(BUILD),
                "--runs-dir",
                str(case_runs),
                "--controls-dir",
                str(case_runs),
                "--out",
                str(experiment),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(built.returncode, 0, built.stderr + built.stdout)
        source_plan_dir = root / "source_plan"
        planned = subprocess.run(
            [
                sys.executable,
                str(TRAIN),
                "--mode",
                "fr_sft_dpo",
                "--dry-run",
                "--require-registered-inputs",
                "--experiment-dir",
                str(experiment),
                "--model",
                "Qwen/Qwen3-4B-Instruct-2507",
                "--model-manifest",
                str(model_manifest),
                "--dataset-manifest",
                str(experiment / "dataset_training_manifest.json"),
                "--output-dir",
                str(source_plan_dir),
                "--max-steps",
                "7",
                "--lora-r",
                "8",
                "--lora-alpha",
                "16",
                "--lora-dropout",
                "0.1",
                "--all-message-loss",
                "--seed",
                "19",
                "--data-seed",
                "23",
                "--save-steps",
                "5",
                "--save-total-limit",
                "2",
                "--trackio-project",
                "hfr-test",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(planned.returncode, 0, planned.stderr + planned.stdout)
        source_plan = source_plan_dir / "fr_sft_dpo_plan.json"
        command = [
            sys.executable,
            str(PREPARE),
            "--experiment-dir",
            str(experiment),
            "--training-plan",
            str(source_plan),
            "--model-manifest",
            str(model_manifest),
            "--reviewer",
            "test-reviewer",
            "--reviewed-at",
            "2026-07-18T00:00:00+00:00",
            "--dataset-repo",
            "test-user/hermes-agentic-data",
            "--hub-model-id",
            "test-user/hermes-agentic-adapter",
            "--container-image",
            PINNED_IMAGE,
            "--out",
            str(handoff_dir),
        ]
        command.extend(extra_prepare or [])
        prepared = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
        self.assertEqual(prepared.returncode, 0, prepared.stderr + prepared.stdout)
        handoff = json.loads((handoff_dir / "handoff.json").read_text(encoding="utf-8"))
        reviewed = json.loads((handoff_dir / "payload" / "reviewed_training_plan.json").read_text(encoding="utf-8"))
        return handoff_dir, handoff, reviewed

    def test_handoff_reconstructs_every_trainer_setting_from_fingerprinted_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            handoff_dir, handoff, reviewed = self._prepare_fixture(Path(tmp))
            request = json.loads((handoff_dir / "job_request.template.json").read_text(encoding="utf-8"))
            dataset_manifest = json.loads(
                (handoff_dir / "payload" / "dataset_training_manifest.json").read_text(encoding="utf-8")
            )
            argv = trainer_argv_from_plan(
                reviewed,
                python="python",
                trainer_script="train.py",
                experiment_dir="dataset",
                output_dir="output",
                model_manifest="model.json",
                dataset_manifest="dataset.json",
                result_registry="registry.jsonl",
                model_registry_link_plan="registry-plan.json",
            )

            self.assertFalse(handoff["submitted"])
            self.assertFalse(handoff["network_writes_performed"])
            self.assertTrue(handoff["submission"]["requires_explicit_paid_network_approval"])
            self.assertEqual(request["secrets"], {"HF_TOKEN": "$HF_TOKEN"})
            self.assertEqual(request["image"], PINNED_IMAGE)
            self.assertIn("REPLACE_WITH_DATASET_COMMIT", request["script"])
            self.assertEqual(request["script_args"][:2], ["--dataset-revision", "REPLACE_WITH_DATASET_COMMIT"])
            self.assertNotIn("--lora-r", request["script_args"])
            self.assertNotIn("source_manifest", dataset_manifest)
            self.assertEqual(reviewed["trainer"]["loss_scope"], "all_messages")
            self.assertEqual(reviewed["trainer"]["hyperparameters"]["lora_r"], 8)
            self.assertEqual(reviewed["trainer"]["hyperparameters"]["seed"], 19)
            self.assertEqual(reviewed["inputs"]["base_model"]["revision"], "a" * 40)
            self.assertEqual(reviewed["inputs"]["tokenizer"]["revision"], "b" * 40)
            self.assertTrue(any(row["path"].startswith("controls/") for row in reviewed["inputs"]["data_artifacts"]))
            self.assertTrue((handoff_dir / "payload" / "controls" / "governance_receipt.json").is_file())
            job_source = (handoff_dir / "payload" / "runtime" / "hf_job.py").read_text(encoding="utf-8")
            self.assertIn("runtime_declaration_from_plan(plan)", job_source)
            self.assertIn('context="training_job"', job_source)
            self.assertNotIn('"container_image": runtime["container_image"]', job_source)
            self.assertNotIn('"hardware_flavor": runtime["hardware_flavor"]', job_source)
            self.assertIn("--all-message-loss", argv)
            for flag, expected in {
                "--lora-r": "8",
                "--lora-alpha": "16",
                "--lora-dropout": "0.1",
                "--seed": "19",
                "--data-seed": "23",
                "--save-steps": "5",
                "--save-total-limit": "2",
                "--max-steps": "7",
                "--model-revision": "a" * 40,
                "--tokenizer-revision": "b" * 40,
                "--expected-chat-template-sha256": "d" * 64,
            }.items():
                self.assertEqual(argv[argv.index(flag) + 1], expected)
            self.assertEqual(validate_reviewed_training_plan(reviewed, handoff_dir / "payload"), [])
            payload_inventory = json.loads(
                (handoff_dir / "payload" / "payload_manifest.json").read_text(encoding="utf-8")
            )
            for schema_name, payload in (
                ("huggingface_reviewed_training_plan", reviewed),
                ("huggingface_payload_manifest", payload_inventory),
                ("huggingface_jobs_handoff", handoff),
            ):
                checked = check_schema_contract(payload, name_or_id=schema_name)
                self.assertTrue(checked["passed"], checked["errors"])
            py_compile.compile(str(handoff_dir / "payload" / "runtime" / "hf_job.py"), doraise=True)
            py_compile.compile(str(handoff_dir / "upload_to_hub.py"), doraise=True)

    def test_reviewed_plan_identity_rejects_argument_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, reviewed = self._prepare_fixture(Path(tmp))
            drifted = copy.deepcopy(reviewed)
            drifted["trainer"]["hyperparameters"]["lora_r"] = 64
            errors = validate_reviewed_training_plan(drifted)
            self.assertIn("identity.sha256 does not match the reviewed plan", errors)

            drifted["identity"]["sha256"] = payload_identity(drifted)
            argv = trainer_argv_from_plan(
                drifted,
                python="python",
                trainer_script="train.py",
                experiment_dir="dataset",
                output_dir="output",
                model_manifest="model.json",
                dataset_manifest="dataset.json",
                result_registry="registry.jsonl",
                model_registry_link_plan="registry-plan.json",
            )
            self.assertEqual(argv[argv.index("--lora-r") + 1], "64")

    def test_resume_contract_is_immutable_and_replayed(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, reviewed = self._prepare_fixture(
                Path(tmp),
                extra_prepare=[
                    "--resume-revision",
                    "e" * 40,
                    "--resume-phase",
                    "dpo",
                    "--resume-checkpoint-path",
                    "lifecycle/checkpoint-100",
                    "--resume-checkpoint-manifest-sha256",
                    "f" * 64,
                ],
            )
            with self.assertRaises(HuggingFaceLifecycleError):
                trainer_argv_from_plan(
                    reviewed,
                    python="python",
                    trainer_script="train.py",
                    experiment_dir="dataset",
                    output_dir="output",
                    model_manifest="model.json",
                    dataset_manifest="dataset.json",
                    result_registry="registry.jsonl",
                    model_registry_link_plan="registry-plan.json",
                )
            argv = trainer_argv_from_plan(
                reviewed,
                python="python",
                trainer_script="train.py",
                experiment_dir="dataset",
                output_dir="output",
                model_manifest="model.json",
                dataset_manifest="dataset.json",
                result_registry="registry.jsonl",
                model_registry_link_plan="registry-plan.json",
                resume_checkpoint="materialized-checkpoint",
            )
            self.assertEqual(argv[argv.index("--resume-from-checkpoint") + 1], "materialized-checkpoint")
            self.assertEqual(argv[argv.index("--resume-phase") + 1], "dpo")

    def test_private_and_immutable_runtime_identity_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, reviewed = self._prepare_fixture(root)
            unpinned = copy.deepcopy(reviewed)
            unpinned["runtime"]["container_image"] = "registry.example/hfr:latest"
            unpinned["identity"]["sha256"] = payload_identity(unpinned)
            self.assertIn("runtime.container_image must use an immutable @sha256 digest", validate_reviewed_training_plan(unpinned))

            public = copy.deepcopy(reviewed)
            public["output"]["model_private"] = False
            public["identity"]["sha256"] = payload_identity(public)
            self.assertIn("output repositories must be private", validate_reviewed_training_plan(public))


if __name__ == "__main__":
    unittest.main()
