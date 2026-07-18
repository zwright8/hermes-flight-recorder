import json
import hashlib
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

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
        data = experiment / "data"
        write_jsonl(
            data / "hermes_trace_only_sft.jsonl",
            [{"prompt": "Trace prompt", "response": "Trace response"}],
        )
        write_jsonl(
            data / "flightrecorder_sft.jsonl",
            [{"prompt": "Curated prompt", "response": "Curated response"}],
        )
        write_jsonl(
            data / "flightrecorder_action_sft.jsonl",
            [{"prompt": "Action prompt", "response": "tool_name({})"}],
        )
        write_jsonl(
            data / "flightrecorder_combined_dpo.jsonl",
            [{"prompt": "DPO prompt", "chosen": "Safe answer", "rejected": "Unsafe answer"}],
        )
        write_jsonl(
            data / "flightrecorder_reward_model.jsonl",
            [{"prompt": "Reward prompt", "response": "Reward response", "reward": 1}],
        )
        write_jsonl(
            data / "flightrecorder_step_rewards.jsonl",
            [{"episode_id": "ep-1", "target": "event", "reward": 1}],
        )
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
        data = experiment / "data"
        files = {
            "trace_sft": data / "hermes_trace_only_sft.jsonl",
            "flightrecorder_sft": data / "flightrecorder_sft.jsonl",
            "flightrecorder_action_sft": data / "flightrecorder_action_sft.jsonl",
            "flightrecorder_combined_dpo": data / "flightrecorder_combined_dpo.jsonl",
            "flightrecorder_reward_model": data / "flightrecorder_reward_model.jsonl",
            "flightrecorder_step_rewards": data / "flightrecorder_step_rewards.jsonl",
        }
        write_json(
            path,
            {
                "schema_version": "hfr.dataset_registry_entry.v1",
                "dataset_id": "flightrecorder-test",
                "dataset_version": "2026-07-02.test",
                "redaction_status": "redacted",
                "gates": {"training_gate": {"passed": True}},
                "dataset_splits": {"family_exclusive": True},
                "quality_flags": [],
                "source_fingerprint_coverage": {"fully_verified": 6, "unverified": 0},
                "data_files": {name: str(file_path) for name, file_path in files.items()},
                "artifact_fingerprints": {
                    name: {
                        "path": str(file_path),
                        "exists": True,
                        "size_bytes": file_path.stat().st_size,
                        "sha256": hashlib.sha256(file_path.read_bytes()).hexdigest(),
                    }
                    for name, file_path in files.items()
                },
            },
        )

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
            self.assertEqual(schema_result["schema"]["name"], "agentic_lora_training_plan")
            self.assertEqual(plan["model"], "local/test-model")
            self.assertEqual(plan["prepared_counts"]["action_sft"], 1)
            self.assertEqual(plan["input_manifests"]["dataset"]["dataset_identity"], "2026-07-02.test")
            self.assertIn("fr_action_sft", plan["trainer_backends"]["executable_modes"])
            self.assertEqual(json.loads((out / "fr_action_sft_plan.json").read_text(encoding="utf-8"))["passed"], True)

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
            completed = run_train(
                [
                    "--mode",
                    "fr_step_rewards",
                    "--dry-run",
                    "--experiment-dir",
                    str(experiment),
                    "--output-dir",
                    str(root / "out"),
                    "--disable-trackio",
                ]
            )

            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            plan = stdout_json(completed)
            self.assertTrue(plan["passed"])
            self.assertEqual(plan["prepared_counts"]["step_rewards"], 1)
            self.assertIn("fr_step_rewards", plan["trainer_backends"]["plan_only_modes"])

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
