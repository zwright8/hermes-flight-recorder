import json
import hashlib
import importlib.util
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from flightrecorder.schema_registry import check_schema_contract, check_schema_file, check_schema_jsonl_file, list_schema_records


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_agentic_lora.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("train_agentic_lora_test_module", SCRIPT)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
TRAIN_MODULE = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(TRAIN_MODULE)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def run_train(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def stdout_json(completed: subprocess.CompletedProcess[str]) -> dict:
    return json.loads(completed.stdout)


class AgenticLoraTrainingPlanTests(unittest.TestCase):
    def test_local_offline_environment_and_device_selection_are_explicit(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            TRAIN_MODULE._configure_local_offline_environment()
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
            self.assertEqual(os.environ["HF_HUB_DISABLE_TELEMETRY"], "1")
            self.assertEqual(os.environ["DO_NOT_TRACK"], "1")

        class FakeMps:
            @staticmethod
            def is_available():
                return True

        class FakeCuda:
            @staticmethod
            def is_available():
                return False

            @staticmethod
            def is_bf16_supported():
                return False

        class FakeTorch:
            backends = type("Backends", (), {"mps": FakeMps()})()
            cuda = FakeCuda()
            bfloat16 = "bf16"
            float16 = "fp16"
            float32 = "fp32"

        self.assertEqual(TRAIN_MODULE._select_training_device(FakeTorch(), "auto"), ("mps", "fp16"))
        self.assertEqual(TRAIN_MODULE._select_training_device(FakeTorch(), "cpu"), ("cpu", "fp32"))
        with self.assertRaisesRegex(SystemExit, "CUDA is unavailable"):
            TRAIN_MODULE._select_training_device(FakeTorch(), "cuda")

    def test_fixed_training_time_budget_accumulates_across_phases(self):
        now = [100.0]
        budget = TRAIN_MODULE.FixedTrainingTimeBudget(5.0, clock=lambda: now[0])

        budget.begin_phase()
        now[0] = 102.0
        self.assertFalse(budget.should_stop())
        budget.end_phase()
        now[0] = 110.0
        budget.begin_phase()
        now[0] = 113.0

        self.assertTrue(budget.should_stop())
        budget.end_phase()
        self.assertEqual(budget.elapsed_seconds, 5.0)
        self.assertTrue(budget.stop_requested)

    def test_task_family_filter_is_exact_and_preserves_native_tools(self):
        tool_row = {
            "task_family": "tool_calling",
            "messages": [{"role": "assistant", "tool_calls": [{"id": "call-1"}]}],
            "tools": [{"type": "function", "function": {"name": "read_file"}}],
        }
        rows = [tool_row, {"task_family": "email", "messages": []}]

        selected = TRAIN_MODULE.filter_task_rows(rows, ["tool_calling"])

        self.assertEqual(selected, [tool_row])
        self.assertEqual(selected[0]["messages"][0]["tool_calls"][0]["id"], "call-1")
        self.assertEqual(selected[0]["tools"][0]["function"]["name"], "read_file")

    def test_fake_local_sft_launch_binds_offline_model_budget_and_runtime_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = self.make_experiment(root)
            local_model = root / "local-model"
            local_model.mkdir()
            model_manifest = experiment / "registry" / "model_candidate.json"
            model = json.loads(model_manifest.read_text(encoding="utf-8"))
            model["model_id"] = str(local_model)
            write_json(model_manifest, model)
            output = root / "out"
            args = TRAIN_MODULE.parse_args(
                [
                    "--mode", "fr_action_sft",
                    "--local-training",
                    "--execute-local-training",
                    "--model", str(local_model),
                    "--model-manifest", str(model_manifest),
                    "--dataset-manifest", str(experiment / "registry" / "dataset_version.json"),
                    "--experiment-dir", str(experiment),
                    "--output-dir", str(output),
                    "--task-family", "fixture",
                    "--device", "mps",
                    "--max-training-seconds", "1",
                    "--disable-trackio",
                ]
            )
            plan = TRAIN_MODULE.build_plan(args)
            self.assertTrue(plan["passed"])
            plan_path = output / "fr_action_sft_plan.json"
            write_json(plan_path, plan)

            budget_clock = [0.0]
            tokenizer_calls = []
            model_calls = []
            trainer_instances = []

            class TestBudget(TRAIN_MODULE.FixedTrainingTimeBudget):
                def __init__(self, max_seconds):
                    super().__init__(max_seconds, clock=lambda: budget_clock[0])

            class FakeMpsBackend:
                @staticmethod
                def is_available():
                    return True

            fake_torch = types.ModuleType("torch")
            fake_torch.backends = types.SimpleNamespace(mps=FakeMpsBackend())
            fake_torch.cuda = types.SimpleNamespace(
                is_available=lambda: False,
                is_bf16_supported=lambda: False,
            )
            fake_torch.mps = types.SimpleNamespace(
                current_allocated_memory=lambda: 2 * 1024 * 1024,
                driver_allocated_memory=lambda: 3 * 1024 * 1024,
            )
            fake_torch.bfloat16 = "bf16"
            fake_torch.float16 = "fp16"
            fake_torch.float32 = "fp32"

            class FakeModel:
                @classmethod
                def from_pretrained(cls, model_path, **kwargs):
                    model_calls.append((model_path, kwargs))
                    return cls()

                def to(self, **kwargs):
                    self.to_kwargs = kwargs
                    return self

            class FakeDataset:
                @staticmethod
                def from_list(rows):
                    return rows

            class FakeTokenizer:
                chat_template = "fixture-template"

                @classmethod
                def from_pretrained(cls, model_path, **kwargs):
                    tokenizer_calls.append((model_path, kwargs))
                    return cls()

            class FakeTrainerCallback:
                pass

            class FakeConfig:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs

            class FakeLoraConfig(FakeConfig):
                pass

            class FakeTrainer:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs
                    trainer_instances.append(self)

                def train(self, resume_from_checkpoint=None):
                    self.resume_from_checkpoint = resume_from_checkpoint
                    control = types.SimpleNamespace(should_training_stop=False)
                    for callback in self.kwargs["callbacks"]:
                        callback.on_train_begin(None, None, control)
                    budget_clock[0] = 2.0
                    for callback in self.kwargs["callbacks"]:
                        callback.on_step_end(None, None, control)
                        callback.on_train_end(None, None, control)
                    self.control = control
                    return types.SimpleNamespace(metrics={"train_loss": 0.25})

                def save_model(self, destination):
                    adapter = Path(destination)
                    adapter.mkdir(parents=True, exist_ok=True)
                    (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
                    (adapter / "adapter_model.safetensors").write_bytes(b"fixture-adapter")

            fake_datasets = types.ModuleType("datasets")
            fake_datasets.Dataset = FakeDataset
            fake_peft = types.ModuleType("peft")
            fake_peft.AutoPeftModelForCausalLM = type("UnusedAutoPeft", (), {})
            fake_peft.LoraConfig = FakeLoraConfig
            fake_transformers = types.ModuleType("transformers")
            fake_transformers.AutoModelForCausalLM = FakeModel
            fake_transformers.AutoTokenizer = FakeTokenizer
            fake_transformers.TrainerCallback = FakeTrainerCallback
            fake_trl = types.ModuleType("trl")
            fake_trl.DPOConfig = FakeConfig
            fake_trl.DPOTrainer = FakeTrainer
            fake_trl.SFTConfig = FakeConfig
            fake_trl.SFTTrainer = FakeTrainer
            fake_modules = {
                "torch": fake_torch,
                "datasets": fake_datasets,
                "peft": fake_peft,
                "transformers": fake_transformers,
                "trl": fake_trl,
            }

            with mock.patch.dict(sys.modules, fake_modules), mock.patch.object(
                TRAIN_MODULE, "FixedTrainingTimeBudget", TestBudget
            ), mock.patch.dict(os.environ, {}, clear=True):
                result = TRAIN_MODULE.run_training(args, plan, plan_path)

            self.assertEqual(tokenizer_calls, [(str(local_model), {"revision": None, "local_files_only": True})])
            self.assertEqual(
                model_calls,
                [(str(local_model), {"revision": None, "dtype": "fp32", "local_files_only": True})],
            )
            trainer = trainer_instances[0]
            self.assertIsInstance(trainer.kwargs["model"], FakeModel)
            self.assertEqual(trainer.kwargs["model"].to_kwargs, {"device": "mps", "dtype": "fp16"})
            self.assertNotIn("model_init_kwargs", trainer.kwargs["args"].kwargs)
            self.assertFalse(trainer.kwargs["args"].kwargs["use_cpu"])
            self.assertEqual(len(trainer.kwargs["train_dataset"]), 2)
            self.assertTrue(trainer.kwargs["train_dataset"][0]["messages"][1]["tool_calls"])
            self.assertTrue(trainer.control.should_training_stop)
            self.assertEqual(result["runtime_observation"]["device"], "mps")
            self.assertEqual(result["runtime_observation"]["trainer_active_seconds"], 2.0)
            self.assertTrue(result["runtime_observation"]["stopped_by_time_budget"])
            self.assertEqual(result["runtime_observation"]["accelerator_memory_mb"]["current_allocated_mb"], 2.0)
            self.assertEqual(result["runtime_observation"]["task_families"], ["fixture"])
            self.assertEqual(result["final_adapter_artifacts"]["file_count"], 2)
            self.assertTrue(check_schema_contract(result)["passed"])

            missing_runtime = dict(result)
            missing_runtime.pop("runtime_observation")
            self.assertFalse(check_schema_contract(missing_runtime)["passed"])
            missing_artifacts = dict(result)
            missing_artifacts.pop("final_adapter_artifacts")
            self.assertFalse(check_schema_contract(missing_artifacts)["passed"])

    def test_preparation_preserves_native_tools_for_sft_and_dpo(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
            }
        ]
        prompt = {"role": "user", "content": "Read the file."}
        chosen_call = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{\"path\":\"safe.txt\"}"},
                }
            ],
        }
        rejected_call = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-2",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{\"path\":\"secret.txt\"}"},
                }
            ],
        }

        sft = TRAIN_MODULE.prepare_sft_rows([{"messages": [prompt, chosen_call], "tools": tools}])
        dpo = TRAIN_MODULE.prepare_dpo_rows(
            [
                {
                    "chosen_messages": [prompt, chosen_call],
                    "rejected_messages": [prompt, rejected_call],
                    "tools": tools,
                }
            ]
        )

        self.assertEqual(sft[0]["messages"][1]["tool_calls"][0]["id"], "call-1")
        self.assertEqual(sft[0]["tools"], tools)
        self.assertEqual(dpo[0]["prompt"], [prompt])
        self.assertEqual(dpo[0]["chosen"], [chosen_call])
        self.assertEqual(dpo[0]["rejected"], [rejected_call])
        self.assertEqual(dpo[0]["tools"], tools)

    def make_experiment(self, root: Path) -> Path:
        experiment = root / "experiment"
        TRAIN_MODULE.write_smoke_fixture(experiment)
        return experiment

    def write_model_manifest(self, path: Path, *, license_status: str = "approved") -> None:
        write_json(
            path,
            {
                "schema_version": "hfr.model_candidate.v1",
                "model_id": "local/test-model",
                "source": "local-test",
                "license_status": license_status,
                "training_allowed": True,
                "compatibility": {
                    "tokenizer": "available",
                    "chat_template": "messages",
                    "serving": "local-test",
                },
            },
        )

    def write_dataset_manifest(self, path: Path, experiment: Path) -> None:
        source_path = experiment / "registry" / "dataset_version.json"
        source = json.loads(source_path.read_text(encoding="utf-8"))
        files = {
            name: (source_path.parent / relative).resolve()
            for name, relative in source["data_files"].items()
        }
        source["dataset_id"] = "flightrecorder-test"
        source["dataset_version"] = "2026-07-02.test"
        source["data_files"] = {name: str(file_path) for name, file_path in files.items()}
        source["artifact_fingerprints"] = {
            name: {
                "path": str(file_path),
                "exists": True,
                "size_bytes": file_path.stat().st_size,
                "sha256": hashlib.sha256(file_path.read_bytes()).hexdigest(),
            }
            for name, file_path in files.items()
        }
        write_json(path, source)

    def test_registry_backed_action_sft_dry_run_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = self.make_experiment(root)
            model_manifest = root / "model.json"
            dataset_manifest = root / "dataset.json"
            out = root / "out"
            self.write_model_manifest(model_manifest)
            self.write_dataset_manifest(dataset_manifest, experiment)

            completed = run_train(
                [
                    "--mode",
                    "fr_action_sft",
                    "--dry-run",
                    "--require-registered-inputs",
                    "--experiment-dir",
                    str(experiment),
                    "--output-dir",
                    str(out),
                    "--model-manifest",
                    str(model_manifest),
                    "--dataset-manifest",
                    str(dataset_manifest),
                    "--disable-trackio",
                ]
            )

            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            plan = stdout_json(completed)
            self.assertTrue(plan["passed"])
            schema_result = check_schema_contract(plan)
            self.assertTrue(schema_result["passed"], schema_result["errors"])
            missing_task_scope = dict(plan)
            missing_task_scope.pop("task_scope")
            self.assertFalse(check_schema_contract(missing_task_scope)["passed"])
            missing_local_training = dict(plan)
            missing_local_training.pop("local_training")
            self.assertFalse(check_schema_contract(missing_local_training)["passed"])
            self.assertEqual(schema_result["schema"]["name"], "agentic_lora_training_plan")
            self.assertEqual(plan["model"], "local/test-model")
            self.assertEqual(plan["prepared_counts"]["action_sft"], 2)
            self.assertTrue(plan["hyperparameters"]["assistant_only_loss"])
            self.assertEqual(plan["input_manifests"]["dataset"]["dataset_identity"], "2026-07-02.test")
            self.assertIn("fr_action_sft", plan["trainer_backends"]["executable_modes"])
            self.assertEqual(json.loads((out / "fr_action_sft_plan.json").read_text(encoding="utf-8"))["passed"], True)

    def test_local_task_scoped_dry_run_is_offline_fixed_time_and_schema_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = self.make_experiment(root)
            local_model = root / "local-model"
            local_model.mkdir()
            model_manifest = experiment / "registry" / "model_candidate.json"
            model = json.loads(model_manifest.read_text(encoding="utf-8"))
            model["model_id"] = str(local_model)
            write_json(model_manifest, model)
            dataset_manifest = experiment / "registry" / "dataset_version.json"
            out = root / "out"

            completed = run_train(
                [
                    "--mode", "fr_action_sft",
                    "--dry-run",
                    "--local-training",
                    "--model", str(local_model),
                    "--model-manifest", str(model_manifest),
                    "--dataset-manifest", str(dataset_manifest),
                    "--experiment-dir", str(experiment),
                    "--output-dir", str(out),
                    "--task-family", "fixture",
                    "--device", "mps",
                    "--max-training-seconds", "12.5",
                    "--disable-trackio",
                ]
            )

            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            plan = stdout_json(completed)
            self.assertTrue(plan["passed"])
            self.assertEqual(plan["task_scope"]["requested_task_families"], ["fixture"])
            self.assertEqual(plan["task_scope"]["selected_task_families"], ["fixture"])
            self.assertTrue(plan["local_training"]["enabled"])
            self.assertTrue(plan["local_training"]["local_files_only"])
            self.assertFalse(plan["local_training"]["network_allowed"])
            self.assertFalse(plan["local_training"]["hub_push_allowed"])
            self.assertFalse(plan["local_training"]["remote_tracking_allowed"])
            self.assertEqual(plan["local_training"]["device_order"], ["mps"])
            self.assertEqual(plan["local_training"]["fixed_training_time_budget_seconds"], 12.5)
            self.assertEqual(plan["prepared_counts"]["action_sft"], 2)
            schema_result = check_schema_contract(plan)
            self.assertTrue(schema_result["passed"], schema_result["errors"])

    def test_unknown_task_family_and_nonlocal_model_block_local_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = self.make_experiment(root)
            fixture = json.loads((experiment / "smoke_fixture.json").read_text(encoding="utf-8"))

            completed = run_train(
                [
                    "--mode", "fr_action_sft",
                    "--dry-run",
                    "--local-training",
                    "--model-manifest", fixture["model_manifest"],
                    "--dataset-manifest", fixture["dataset_manifest"],
                    "--experiment-dir", str(experiment),
                    "--output-dir", str(root / "out"),
                    "--task-family", "missing-family",
                    "--disable-trackio",
                ]
            )

            self.assertEqual(completed.returncode, 1)
            plan = stdout_json(completed)
            failed = {check["id"] for check in plan["checks"] if not check["passed"]}
            self.assertIn("requested_task_families_available", failed)
            self.assertIn("action_sft_rows_available", failed)
            self.assertIn("local_model_directory_present", failed)

    def test_local_model_path_is_separate_from_registered_model_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = self.make_experiment(root)
            local_model = root / "cached-model-snapshot"
            local_model.mkdir()
            model_manifest = root / "model.json"
            dataset_manifest = root / "dataset.json"
            self.write_model_manifest(model_manifest)
            self.write_dataset_manifest(dataset_manifest, experiment)

            completed = run_train(
                [
                    "--mode", "fr_action_sft",
                    "--dry-run",
                    "--local-training",
                    "--model", "local/test-model",
                    "--local-model-path", str(local_model),
                    "--model-manifest", str(model_manifest),
                    "--dataset-manifest", str(dataset_manifest),
                    "--experiment-dir", str(experiment),
                    "--output-dir", str(root / "out"),
                    "--task-family", "fixture",
                    "--disable-trackio",
                ]
            )

            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            plan = stdout_json(completed)
            self.assertEqual(plan["model"], "local/test-model")
            self.assertEqual(plan["local_training"]["model_path"], str(local_model))
            self.assertTrue(plan["passed"])

    def test_local_weight_update_requires_explicit_execution_acknowledgement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = self.make_experiment(root)
            local_model = root / "local-model"
            local_model.mkdir()
            model_manifest = experiment / "registry" / "model_candidate.json"
            model = json.loads(model_manifest.read_text(encoding="utf-8"))
            model["model_id"] = str(local_model)
            write_json(model_manifest, model)
            dataset_manifest = experiment / "registry" / "dataset_version.json"

            completed = run_train(
                [
                    "--mode", "fr_action_sft",
                    "--local-training",
                    "--model", str(local_model),
                    "--model-manifest", str(model_manifest),
                    "--dataset-manifest", str(dataset_manifest),
                    "--experiment-dir", str(experiment),
                    "--output-dir", str(root / "out"),
                    "--task-family", "fixture",
                    "--disable-trackio",
                ]
            )

            self.assertEqual(completed.returncode, 1)
            self.assertNotIn("ModuleNotFoundError", completed.stderr)
            result = stdout_json(completed)
            self.assertEqual(result["status"], "blocked")
            plan = json.loads((root / "out" / "fr_action_sft_plan.json").read_text(encoding="utf-8"))
            failed = {check["id"] for check in plan["checks"] if not check["passed"]}
            self.assertEqual(failed, {"local_training_execution_acknowledged"})

    def test_local_training_refuses_base_model_overlap_and_populated_adapter_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = self.make_experiment(root)
            local_model = root / "local-model"
            local_model.mkdir()
            model_manifest = experiment / "registry" / "model_candidate.json"
            model = json.loads(model_manifest.read_text(encoding="utf-8"))
            model["model_id"] = str(local_model)
            write_json(model_manifest, model)
            dataset_manifest = experiment / "registry" / "dataset_version.json"
            output = local_model / "training-output"
            occupied = output / "fr_action_sft_adapter"
            occupied.mkdir(parents=True)
            (occupied / "adapter_config.json").write_text("{}\n", encoding="utf-8")

            completed = run_train(
                [
                    "--mode", "fr_action_sft",
                    "--dry-run",
                    "--local-training",
                    "--model", str(local_model),
                    "--model-manifest", str(model_manifest),
                    "--dataset-manifest", str(dataset_manifest),
                    "--experiment-dir", str(experiment),
                    "--output-dir", str(output),
                    "--task-family", "fixture",
                    "--disable-trackio",
                ]
            )

            self.assertEqual(completed.returncode, 1)
            plan = stdout_json(completed)
            failed = {check["id"] for check in plan["checks"] if not check["passed"]}
            self.assertIn("local_model_output_separate", failed)
            self.assertIn("local_adapter_targets_empty", failed)

    def test_all_message_loss_is_recorded_for_templates_without_assistant_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = self.make_experiment(root)
            model_manifest = root / "model.json"
            dataset_manifest = root / "dataset.json"
            out = root / "out"
            self.write_model_manifest(model_manifest)
            self.write_dataset_manifest(dataset_manifest, experiment)

            completed = run_train(
                [
                    "--mode",
                    "fr_action_sft",
                    "--dry-run",
                    "--require-registered-inputs",
                    "--experiment-dir",
                    str(experiment),
                    "--output-dir",
                    str(out),
                    "--model-manifest",
                    str(model_manifest),
                    "--dataset-manifest",
                    str(dataset_manifest),
                    "--all-message-loss",
                    "--disable-trackio",
                ]
            )

            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            plan = stdout_json(completed)
            self.assertTrue(plan["passed"])
            self.assertFalse(plan["hyperparameters"]["assistant_only_loss"])

    def test_unknown_model_license_blocks_required_registry_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = self.make_experiment(root)
            model_manifest = root / "model.json"
            dataset_manifest = root / "dataset.json"
            self.write_model_manifest(model_manifest, license_status="unknown")
            self.write_dataset_manifest(dataset_manifest, experiment)

            completed = run_train(
                [
                    "--mode",
                    "trace_sft",
                    "--dry-run",
                    "--require-registered-inputs",
                    "--experiment-dir",
                    str(experiment),
                    "--output-dir",
                    str(root / "out"),
                    "--model-manifest",
                    str(model_manifest),
                    "--dataset-manifest",
                    str(dataset_manifest),
                    "--disable-trackio",
                ]
            )

            self.assertEqual(completed.returncode, 1)
            plan = stdout_json(completed)
            self.assertFalse(plan["passed"])
            failed = {check["id"] for check in plan["checks"] if not check["passed"]}
            self.assertIn("model_license_known", failed)

    def test_push_to_hub_without_repository_blocks_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = self.make_experiment(root)
            completed = run_train(
                [
                    "--mode",
                    "fr_sft",
                    "--dry-run",
                    "--push-to-hub",
                    "--experiment-dir",
                    str(experiment),
                    "--output-dir",
                    str(root / "out"),
                    "--disable-trackio",
                ]
            )

            self.assertEqual(completed.returncode, 1)
            plan = stdout_json(completed)
            failed = {check["id"] for check in plan["checks"] if not check["passed"]}
            self.assertIn("hub_model_id_present_when_pushing", failed)

    def test_non_dry_run_without_registry_blocks_before_heavy_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = self.make_experiment(root)
            out = root / "out"
            registry = root / "registry" / "training_events.jsonl"

            completed = run_train(
                [
                    "--mode",
                    "trace_sft",
                    "--experiment-dir",
                    str(experiment),
                    "--output-dir",
                    str(out),
                    "--limit",
                    "1",
                    "--result-registry",
                    str(registry),
                    "--disable-trackio",
                ]
            )

            self.assertEqual(completed.returncode, 1)
            self.assertNotIn("ModuleNotFoundError", completed.stderr)
            result = stdout_json(completed)
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["failure"]["category"], "plan_validation_failed")
            result_schema = check_schema_contract(result)
            self.assertTrue(result_schema["passed"], result_schema["errors"])
            self.assertEqual(result_schema["schema"]["name"], "agentic_lora_training_result")
            plan = json.loads((out / "trace_sft_plan.json").read_text(encoding="utf-8"))
            self.assertFalse(plan["passed"])
            self.assertTrue(plan["compute_assumptions"]["registered_inputs_required_for_launch"])
            failed = {check["id"] for check in plan["checks"] if not check["passed"]}
            self.assertIn("model_manifest_provided", failed)
            self.assertIn("dataset_manifest_provided", failed)
            self.assertTrue((out / "trace_sft_plan.json").exists())
            self.assertTrue((out / "trace_sft_result.json").exists())
            events = [json.loads(line) for line in registry.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["schema_version"], "hfr.agentic_lora_training_registry_event.v1")
            self.assertEqual(events[0]["status"], "blocked")
            self.assertEqual(events[0]["failure"]["category"], "plan_validation_failed")
            event_schema = check_schema_jsonl_file(registry, "agentic_lora_training_registry_event")
            self.assertTrue(event_schema["passed"], event_schema["errors"])

    def test_malformed_training_jsonl_is_archived_before_trainer_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = self.make_experiment(root)
            out = root / "out"
            registry = root / "registry" / "training_events.jsonl"
            (experiment / "data" / "flightrecorder_action_sft.jsonl").write_text(
                '{"messages": [}\n',
                encoding="utf-8",
            )

            completed = run_train(
                [
                    "--mode",
                    "fr_action_sft",
                    "--experiment-dir",
                    str(experiment),
                    "--output-dir",
                    str(out),
                    "--result-registry",
                    str(registry),
                    "--disable-trackio",
                ]
            )

            self.assertEqual(completed.returncode, 1)
            self.assertNotIn("Traceback", completed.stderr)
            result = stdout_json(completed)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["failure"]["category"], "invalid_training_data")
            self.assertTrue((out / "fr_action_sft_plan.json").exists())
            self.assertEqual(
                json.loads((out / "fr_action_sft_result.json").read_text(encoding="utf-8"))["failure"]["category"],
                "invalid_training_data",
            )
            events = [json.loads(line) for line in registry.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(events[0]["failure"]["category"], "invalid_training_data")
            result_schema = check_schema_contract(result)
            self.assertTrue(result_schema["passed"], result_schema["errors"])

    def test_preflight_only_archives_missing_dependency_without_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = self.make_experiment(root)
            model_manifest = root / "model.json"
            dataset_manifest = root / "dataset.json"
            out = root / "out"
            registry = root / "registry" / "training_events.jsonl"
            link_plan = root / "registry" / "training_link_plan.json"
            missing_module = "hfr_missing_trainer_dependency_for_test"
            self.write_model_manifest(model_manifest)
            self.write_dataset_manifest(dataset_manifest, experiment)

            completed = run_train(
                [
                    "--mode",
                    "fr_action_sft",
                    "--preflight-only",
                    "--require-registered-inputs",
                    "--experiment-dir",
                    str(experiment),
                    "--output-dir",
                    str(out),
                    "--model-manifest",
                    str(model_manifest),
                    "--dataset-manifest",
                    str(dataset_manifest),
                    "--result-registry",
                    str(registry),
                    "--write-model-registry-link-plan",
                    str(link_plan),
                    "--model-registry",
                    str(root / "registry" / "model_registry.json"),
                    "--model-registry-entry",
                    "candidate",
                    "--preflight-extra-dependency",
                    missing_module,
                    "--disable-trackio",
                ]
            )

            self.assertEqual(completed.returncode, 1)
            result = stdout_json(completed)
            self.assertEqual(result["status"], "preflight_blocked")
            self.assertEqual(result["failure"]["category"], "missing_dependency")
            self.assertIn(missing_module, result["failure"]["missing_dependencies"])
            self.assertFalse(result["preflight"]["model_downloads_started"])
            self.assertFalse(result["preflight"]["training_started"])
            result_schema = check_schema_contract(result)
            self.assertTrue(result_schema["passed"], result_schema["errors"])
            plan = json.loads((out / "fr_action_sft_plan.json").read_text(encoding="utf-8"))
            self.assertTrue(plan["passed"])
            archived = json.loads((out / "fr_action_sft_result.json").read_text(encoding="utf-8"))
            self.assertEqual(archived["status"], "preflight_blocked")
            events = [json.loads(line) for line in registry.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["status"], "preflight_blocked")
            event_schema = check_schema_jsonl_file(registry, "agentic_lora_training_registry_event")
            self.assertTrue(event_schema["passed"], event_schema["errors"])
            link = json.loads(link_plan.read_text(encoding="utf-8"))
            self.assertEqual(link["schema_version"], "hfr.agentic_lora_model_registry_link_plan.v1")
            self.assertEqual(link["recommendation"], "ready_to_link_training_result")
            self.assertFalse(link["handoff_contract"]["moves_aliases"])
            self.assertFalse(link["handoff_contract"]["flight_recorder_mutated_registry"])
            self.assertEqual(link["commands"][0]["link_type"], "training-run")
            self.assertIn("--collection", link["commands"][0]["command_argv"])
            self.assertIn("training_runs", link["commands"][0]["command_argv"])
            self.assertIn("--kind", link["commands"][0]["command_argv"])
            self.assertNotIn("--type", link["commands"][0]["command_argv"])
            link_schema = check_schema_file(link_plan)
            self.assertTrue(link_schema["passed"], link_schema["errors"])

    def test_reward_modes_are_dry_run_extension_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = self.make_experiment(root)
            model_manifest = root / "model.json"
            dataset_manifest = root / "dataset.json"
            self.write_model_manifest(model_manifest)
            self.write_dataset_manifest(dataset_manifest, experiment)
            completed = run_train(
                [
                    "--mode",
                    "fr_step_rewards",
                    "--dry-run",
                    "--experiment-dir",
                    str(experiment),
                    "--model-manifest",
                    str(model_manifest),
                    "--dataset-manifest",
                    str(dataset_manifest),
                    "--output-dir",
                    str(root / "out"),
                    "--disable-trackio",
                ]
            )

            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            plan = stdout_json(completed)
            self.assertTrue(plan["passed"])
            self.assertEqual(plan["prepared_counts"]["step_rewards"], 2)
            self.assertIn("fr_step_rewards", plan["trainer_backends"]["plan_only_modes"])

    def test_flight_recorder_dry_run_rejects_unsafe_unregistered_bypass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = self.make_experiment(root)
            completed = run_train(
                [
                    "--mode", "fr_sft", "--dry-run", "--unsafe-allow-unregistered-launch",
                    "--experiment-dir", str(experiment), "--output-dir", str(root / "out"),
                    "--disable-trackio",
                ]
            )

            self.assertEqual(completed.returncode, 1)
            plan = stdout_json(completed)
            failed = {check["id"] for check in plan["checks"] if not check["passed"]}
            self.assertIn("registered_inputs_required", failed)
            self.assertTrue(plan["compute_assumptions"]["registered_inputs_required_for_launch"])

    def test_manifest_booleans_and_refingerprinted_row_cannot_forge_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = self.make_experiment(root)
            model_manifest = root / "model.json"
            dataset_manifest = root / "dataset.json"
            self.write_model_manifest(model_manifest)
            self.write_dataset_manifest(dataset_manifest, experiment)
            manifest = json.loads(dataset_manifest.read_text(encoding="utf-8"))

            control_path = Path(manifest["data_files"].pop("governance_receipt"))
            manifest["artifact_fingerprints"].pop("governance_receipt")
            self.assertTrue(control_path.is_file())
            write_json(dataset_manifest, manifest)
            blocked_control = run_train(
                [
                    "--mode", "fr_sft", "--dry-run", "--model-manifest", str(model_manifest),
                    "--dataset-manifest", str(dataset_manifest), "--experiment-dir", str(experiment),
                    "--output-dir", str(root / "missing-control"), "--disable-trackio",
                ]
            )
            self.assertEqual(blocked_control.returncode, 1)
            checks = {row["id"]: row for row in stdout_json(blocked_control)["checks"]}
            self.assertFalse(checks["content_bound_control_artifacts_replayed"]["passed"])

            self.write_dataset_manifest(dataset_manifest, experiment)
            manifest = json.loads(dataset_manifest.read_text(encoding="utf-8"))
            sft_path = Path(manifest["data_files"]["flightrecorder_sft"])
            rows = [json.loads(line) for line in sft_path.read_text(encoding="utf-8").splitlines() if line]
            rows[0]["review_item_sha256"] = "0" * 64
            write_jsonl(sft_path, rows)
            record = manifest["artifact_fingerprints"]["flightrecorder_sft"]
            record["size_bytes"] = sft_path.stat().st_size
            record["sha256"] = hashlib.sha256(sft_path.read_bytes()).hexdigest()
            write_json(dataset_manifest, manifest)
            forged_row = run_train(
                [
                    "--mode", "fr_sft", "--dry-run", "--model-manifest", str(model_manifest),
                    "--dataset-manifest", str(dataset_manifest), "--experiment-dir", str(experiment),
                    "--output-dir", str(root / "forged-row"), "--disable-trackio",
                ]
            )
            self.assertEqual(forged_row.returncode, 1)
            checks = {row["id"]: row for row in stdout_json(forged_row)["checks"]}
            self.assertTrue(checks["dataset_artifact_fingerprints_verified"]["passed"])
            self.assertFalse(checks["training_rows_bound_to_control_artifacts"]["passed"])

    def test_training_schemas_are_registered(self):
        names = {record["name"] for record in list_schema_records()}
        self.assertIn("agentic_lora_training_plan", names)
        self.assertIn("agentic_lora_training_result", names)
        self.assertIn("agentic_lora_training_registry_event", names)
        self.assertIn("agentic_lora_smoke_fixture", names)
        self.assertIn("agentic_lora_backend_recipes", names)
        self.assertIn("agentic_lora_model_registry_link_plan", names)

    def test_write_smoke_fixture_supports_registry_backed_row_limit_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture_dir = root / "fixture"
            completed = run_train(["--write-smoke-fixture", str(fixture_dir)])

            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            fixture = stdout_json(completed)
            self.assertEqual(fixture["schema_version"], "hfr.agentic_lora_smoke_fixture.v1")
            fixture_schema = check_schema_contract(fixture)
            self.assertTrue(fixture_schema["passed"], fixture_schema["errors"])
            self.assertEqual(fixture_schema["schema"]["name"], "agentic_lora_smoke_fixture")
            fixture_file_schema = check_schema_file(fixture_dir / "smoke_fixture.json")
            self.assertTrue(fixture_file_schema["passed"], fixture_file_schema["errors"])
            model_manifest = Path(fixture["model_manifest"])
            dataset_manifest = Path(fixture["dataset_manifest"])
            self.assertTrue(model_manifest.exists())
            self.assertTrue(dataset_manifest.exists())

            out = root / "out"
            plan_completed = run_train(
                [
                    "--mode",
                    "fr_sft_dpo",
                    "--dry-run",
                    "--require-registered-inputs",
                    "--experiment-dir",
                    str(fixture_dir),
                    "--model-manifest",
                    str(model_manifest),
                    "--dataset-manifest",
                    str(dataset_manifest),
                    "--output-dir",
                    str(out),
                    "--limit",
                    "1",
                    "--disable-trackio",
                ]
            )

            self.assertEqual(plan_completed.returncode, 0, plan_completed.stderr + plan_completed.stdout)
            plan = stdout_json(plan_completed)
            self.assertTrue(plan["passed"])
            self.assertEqual(plan["model"], "local/hfr-smoke-model")
            self.assertTrue(plan["smoke"]["enabled"])
            self.assertEqual(plan["smoke"]["row_limit"], 1)
            self.assertEqual(plan["prepared_counts"]["sft"], 1)
            self.assertEqual(plan["prepared_counts"]["dpo"], 1)
            self.assertGreater(plan["full_prepared_counts"]["sft"], plan["prepared_counts"]["sft"])
            self.assertGreater(plan["full_prepared_counts"]["dpo"], plan["prepared_counts"]["dpo"])
            self.assertEqual(plan["input_manifests"]["dataset"]["dataset_identity"], "hfr-smoke-fixture.v1")
            schema_result = check_schema_contract(plan)
            self.assertTrue(schema_result["passed"], schema_result["errors"])

    def test_write_backend_recipes_from_registered_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture_dir = root / "fixture"
            fixture_completed = run_train(["--write-smoke-fixture", str(fixture_dir)])
            self.assertEqual(fixture_completed.returncode, 0, fixture_completed.stderr + fixture_completed.stdout)
            fixture = stdout_json(fixture_completed)
            out = root / "out"
            recipe_dir = root / "recipes"

            completed = run_train(
                [
                    "--mode",
                    "fr_sft_dpo",
                    "--write-backend-recipes",
                    str(recipe_dir),
                    "--require-registered-inputs",
                    "--experiment-dir",
                    str(fixture_dir),
                    "--model-manifest",
                    fixture["model_manifest"],
                    "--dataset-manifest",
                    fixture["dataset_manifest"],
                    "--output-dir",
                    str(out),
                    "--limit",
                    "1",
                    "--disable-trackio",
                ]
            )

            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            bundle = stdout_json(completed)
            self.assertEqual(bundle["schema_version"], "hfr.agentic_lora_backend_recipes.v1")
            self.assertTrue(bundle["passed"])
            self.assertEqual(bundle["recommendation"], "ready_for_external_recipe_runner")
            self.assertFalse(bundle["handoff_contract"]["flight_recorder_executed_command"])
            self.assertTrue(bundle["handoff_contract"]["runner_owns_execution"])
            self.assertEqual({recipe["backend"] for recipe in bundle["recipes"]}, {"axolotl", "llama_factory", "unsloth", "reward_process_rl_extensions"})
            bundle_schema = check_schema_contract(bundle)
            self.assertTrue(bundle_schema["passed"], bundle_schema["errors"])
            file_schema = check_schema_file(recipe_dir / "backend_recipes.json")
            self.assertTrue(file_schema["passed"], file_schema["errors"])
            axolotl = json.loads((recipe_dir / "axolotl_recipe.json").read_text(encoding="utf-8"))
            self.assertEqual(axolotl["source_training_plan"], str(out / "fr_sft_dpo_plan.json"))
            self.assertFalse(axolotl["execution_boundary"]["flight_recorder_executed_command"])
            self.assertIn("fr_dpo", axolotl["data_files"])
            reward = json.loads((recipe_dir / "reward_process_rl_extensions_recipe.json").read_text(encoding="utf-8"))
            self.assertIn("grpo_rl_trainer", reward["planned_trainers"])
            plan = json.loads((out / "fr_sft_dpo_plan.json").read_text(encoding="utf-8"))
            self.assertTrue(plan["passed"])


if __name__ == "__main__":
    unittest.main()
