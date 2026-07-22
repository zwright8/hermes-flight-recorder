import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from flightrecorder.lora_recipe_search import write_json


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_runtime_adapter_autoresearch.py"
VALIDATOR_PATH = ROOT / "scripts" / "validate_runtime_adapter_autoresearch.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("runtime_adapter_autoresearch_test_module", RUNNER_PATH)
runner = importlib.util.module_from_spec(RUNNER_SPEC)
assert RUNNER_SPEC.loader is not None
sys.modules[RUNNER_SPEC.name] = runner
RUNNER_SPEC.loader.exec_module(runner)
VALIDATOR_SPEC = importlib.util.spec_from_file_location("runtime_adapter_autoresearch_validator_test_module", VALIDATOR_PATH)
validator = importlib.util.module_from_spec(VALIDATOR_SPEC)
assert VALIDATOR_SPEC.loader is not None
sys.modules[VALIDATOR_SPEC.name] = validator
VALIDATOR_SPEC.loader.exec_module(validator)


class RuntimeAdapterAutoresearchTests(unittest.TestCase):
    def test_default_records_no_training_and_launches_no_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _fixture(Path(tmp))

            with mock.patch.object(runner.subprocess, "run") as subprocess_run:
                record = runner.run_campaign(
                    runner.parse_args(
                        [
                            "--campaign-dir",
                            str(fixture["campaign"]),
                            "--development-suite",
                            str(fixture["development_suite"]),
                            "--development-jsonl",
                            str(fixture["development_jsonl"]),
                            "--max-trials",
                            "1",
                        ]
                    )
                )

            subprocess_run.assert_not_called()
            self.assertFalse(record["search_result_summary"]["passed"])
            self.assertEqual(len(record["candidate_records"]), 1)
            blocked = record["candidate_records"][0]
            self.assertEqual(blocked["status"], "blocked")
            self.assertFalse(blocked["sealed_inputs_accessed"])
            self.assertFalse(blocked["execute_local_training"])
            launch_path = Path(blocked["proposal_launch_record"])
            self.assertTrue(launch_path.is_file())

    def test_execute_training_propagates_offline_constraints_and_budgets(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _fixture(Path(tmp))
            seen_commands: list[list[str]] = []

            def fake_run(command, cwd, check):
                del cwd, check
                seen_commands.append(command)
                if command[1].endswith("train_agentic_lora.py"):
                    _fake_training_output(command)
                else:
                    _fake_development_report(command)
                return _Completed(0)

            with mock.patch.object(runner.subprocess, "run", side_effect=fake_run):
                record = runner.run_campaign(
                    runner.parse_args(
                        [
                            "--campaign-dir",
                            str(fixture["campaign"]),
                            "--development-suite",
                            str(fixture["development_suite"]),
                            "--development-jsonl",
                            str(fixture["development_jsonl"]),
                            "--model-manifest",
                            str(fixture["model_manifest"]),
                            "--dataset-manifest",
                            str(fixture["dataset_manifest"]),
                            "--local-model-path",
                            str(fixture["local_model"]),
                            "--execute-local-training",
                            "--max-trials",
                            "1",
                            "--trial-training-seconds",
                            "7",
                            "--per-trial-duration-ceiling-seconds",
                            "11",
                            "--task-family",
                            "runtime_adapter_router_browser_train",
                            "--evaluation-scope",
                            "browser",
                        ]
                    )
                )

            trainer_command = seen_commands[0]
            self.assertIn("--local-training", trainer_command)
            self.assertIn("--execute-local-training", trainer_command)
            self.assertIn("--disable-trackio", trainer_command)
            self.assertIn("--require-registered-inputs", trainer_command)
            self.assertEqual(
                trainer_command[trainer_command.index("--action-turn-repeats") + 1],
                "0",
            )
            self.assertEqual(
                trainer_command[trainer_command.index("--task-family") + 1],
                "runtime_adapter_router_browser_train",
            )
            candidate = record["candidate_records"][0]["candidate"]
            self.assertEqual(
                candidate["training_task_families"],
                ["runtime_adapter_router_browser_train"],
            )
            self.assertEqual(candidate["evaluation_scopes"], ["browser"])
            self.assertNotIn("--push-to-hub", trainer_command)
            self.assertEqual(trainer_command[trainer_command.index("--max-training-seconds") + 1], "7.0")
            self.assertEqual(record["search_result_summary"]["passed"], True)
            outcome = record["candidate_records"][0]["development_outcome"]
            self.assertEqual(outcome["development_quality_score"], 0.63)
            self.assertEqual(outcome["critical_unsafe_call_count"], 0)

    def test_development_score_preserves_zero_pass_rates(self):
        report = {
            "candidate_reports": [
                {
                    "candidate_id": "candidate",
                    "promotion_eligible": False,
                    "blocking_reasons": ["write_denial_min_pass_rate"],
                    "metrics": {
                        "overall": {"pass_rate": 0.5},
                        "safety": {"pass_rate": 1.0},
                        "write_denial": {"pass_rate": 0.0},
                        "failure_recovery": {"pass_rate": 0.0},
                        "check_pass_rates": {
                            "tool_calls_exact_order": {"pass_rate": 0.5},
                            "final_answer_exact": {"pass_rate": 0.5},
                        },
                        "critical_safety_failures": 0,
                    },
                }
            ]
        }

        outcome = runner.development_outcome(report, "candidate")

        self.assertEqual(outcome["write_denial_pass_rate"], 0.0)
        self.assertEqual(outcome["failure_recovery_pass_rate"], 0.0)
        self.assertEqual(outcome["tool_calls_functional_rate"], 0.5)
        self.assertEqual(outcome["development_quality_score"], 0.475)

    def test_search_refuses_sealed_or_heldout_development_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _fixture(Path(tmp), dev_jsonl_name="sealed_final.jsonl")

            with self.assertRaisesRegex(SystemExit, "held-out/sealed"):
                runner.run_campaign(
                    runner.parse_args(
                        [
                            "--campaign-dir",
                            str(fixture["campaign"]),
                            "--development-suite",
                            str(fixture["development_suite"]),
                            "--development-jsonl",
                            str(fixture["development_jsonl"]),
                        ]
                    )
                )

    def test_search_refuses_nontraining_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _fixture(Path(tmp))
            args = runner.parse_args(
                [
                    "--campaign-dir",
                    str(fixture["campaign"]),
                    "--development-suite",
                    str(fixture["development_suite"]),
                    "--development-jsonl",
                    str(fixture["development_jsonl"]),
                    "--task-family",
                    "runtime_adapter_router_browser_development",
                ]
            )
            with self.assertRaisesRegex(SystemExit, "ending in _train"):
                runner.run_campaign(args)

    def test_resume_reuses_immutable_plan_and_candidate_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _fixture(Path(tmp))
            first_args = runner.parse_args(
                [
                    "--campaign-dir",
                    str(fixture["campaign"]),
                    "--development-suite",
                    str(fixture["development_suite"]),
                    "--development-jsonl",
                    str(fixture["development_jsonl"]),
                    "--max-trials",
                    "1",
                ]
            )
            first = runner.run_campaign(first_args)
            plan_before = (fixture["campaign"] / "search_plan.json").read_bytes()

            resumed = runner.run_campaign(
                runner.parse_args(
                    [
                        "--campaign-dir",
                        str(fixture["campaign"]),
                        "--development-suite",
                        str(fixture["development_suite"]),
                        "--development-jsonl",
                        str(fixture["development_jsonl"]),
                        "--max-trials",
                        "1",
                        "--resume",
                    ]
                )
            )

            self.assertEqual((fixture["campaign"] / "search_plan.json").read_bytes(), plan_before)
            self.assertEqual(resumed["candidate_records"], first["candidate_records"])

    def test_non_resume_refuses_existing_campaign_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _fixture(Path(tmp))
            args = runner.parse_args(
                [
                    "--campaign-dir",
                    str(fixture["campaign"]),
                    "--development-suite",
                    str(fixture["development_suite"]),
                    "--development-jsonl",
                    str(fixture["development_jsonl"]),
                    "--max-trials",
                    "1",
                ]
            )
            runner.run_campaign(args)

            with self.assertRaisesRegex(SystemExit, "already contains autoresearch state"):
                runner.run_campaign(args)

    def test_resume_flag_is_forwarded_only_when_run_search_supports_it(self):
        captured: dict[str, object] = {}

        def fake_run_search(**kwargs):
            captured.update(kwargs)
            return {"passed": False}

        with mock.patch.object(runner, "run_search", new=fake_run_search):
            runner.run_search_maybe_resume(
                plan_path=Path("plan.json"),
                out_path=Path("result.json"),
                proposer=runner.DeterministicQueueProposer([]),
                evaluator=mock.Mock(),
                resume=True,
            )

        self.assertNotIn("resume", captured)

        def fake_run_search_with_resume(**kwargs):
            captured.clear()
            captured.update(kwargs)
            return {"passed": False}

        fake_run_search_with_resume.__signature__ = inspect_signature_with_resume()
        with mock.patch.object(runner, "run_search", new=fake_run_search_with_resume):
            runner.run_search_maybe_resume(
                plan_path=Path("plan.json"),
                out_path=Path("result.json"),
                proposer=runner.DeterministicQueueProposer([]),
                evaluator=mock.Mock(),
                resume=True,
            )

        self.assertIs(captured["resume"], True)

    def test_finalize_runs_champion_only_once_and_validator_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _fixture(Path(tmp))
            seen_commands: list[list[str]] = []

            def fake_run(command, cwd, check):
                del cwd, check
                seen_commands.append(command)
                if command[1].endswith("train_agentic_lora.py"):
                    _fake_training_output(command)
                else:
                    _fake_development_report(command)
                return _Completed(0)

            args = runner.parse_args(
                [
                    "--campaign-dir",
                    str(fixture["campaign"]),
                    "--campaign-id",
                    "seal-test",
                    "--development-suite",
                    str(fixture["development_suite"]),
                    "--development-jsonl",
                    str(fixture["development_jsonl"]),
                    "--model-manifest",
                    str(fixture["model_manifest"]),
                    "--dataset-manifest",
                    str(fixture["dataset_manifest"]),
                    "--local-model-path",
                    str(fixture["local_model"]),
                    "--execute-local-training",
                    "--max-trials",
                    "1",
                ]
            )
            with mock.patch.object(runner.subprocess, "run", side_effect=fake_run):
                record = runner.run_campaign(args)
                finalize_args = runner.parse_args(
                    [
                        "--campaign-dir",
                        str(fixture["campaign"]),
                        "--campaign-id",
                        "seal-test",
                        "--development-suite",
                        str(fixture["development_suite"]),
                        "--development-jsonl",
                        str(fixture["development_jsonl"]),
                        "--sealed-jsonl",
                        str(fixture["sealed_jsonl"]),
                        "--finalize-sealed",
                    ]
                )
                tampered = json.loads(json.dumps(record))
                tampered["candidate_records"][0]["development_outcome"][
                    "overall_pass_rate"
                ] = 0.0
                campaign_path = fixture["campaign"] / runner.CAMPAIGN_RECORD
                runner.atomic_write_json(campaign_path, tampered)
                with self.assertRaisesRegex(SystemExit, "summary does not replay"):
                    runner.finalize_sealed(finalize_args)
                self.assertFalse((fixture["campaign"] / runner.SEALED_RECEIPT).exists())
                runner.atomic_write_json(campaign_path, record)
                receipt = runner.finalize_sealed(finalize_args)

            sealed_candidate_payload = json.loads(
                (fixture["campaign"] / "sealed_champion_candidate.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(sealed_candidate_payload["candidates"]), 1)
            self.assertEqual(
                receipt["candidate_identity_sha256"],
                record["candidate_records"][0]["candidate"]["candidate_identity_sha256"],
            )
            self.assertEqual(receipt["status"], "completed")
            self.assertEqual(receipt["campaign_id"], "seal-test")
            self.assertEqual(
                receipt["model_identity"],
                runner.model_identity_from_candidate(
                    record["candidate_records"][0]["candidate"]
                ),
            )
            with self.assertRaisesRegex(SystemExit, "already exists"):
                runner.finalize_sealed(
                    runner.parse_args(
                        [
                            "--campaign-dir",
                            str(fixture["campaign"]),
                            "--sealed-jsonl",
                            str(fixture["sealed_jsonl"]),
                            "--finalize-sealed",
                        ]
                    )
                )

            validation = validator.validate_campaign(campaign_dir=fixture["campaign"])
            self.assertTrue(validation["passed"], validation["errors"])

            sealed_report = json.loads((fixture["campaign"] / runner.SEALED_REPORT).read_text(encoding="utf-8"))
            sealed_report["candidate_reports"][0]["metrics"]["critical_safety_failures"] = 1
            write_json(fixture["campaign"] / runner.SEALED_REPORT, sealed_report)
            validation = validator.validate_campaign(campaign_dir=fixture["campaign"])
            self.assertFalse(validation["passed"])
            self.assertIn("critical safety failures", "\n".join(validation["errors"]))

    def test_attempt_artifact_resolver_accepts_repo_relative_campaign_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = Path("runs/campaign")
            report = root / campaign / "attempts/trial-000/development.json"
            report.parent.mkdir(parents=True)
            report.write_text("{}\n", encoding="utf-8")

            with mock.patch.object(runner, "ROOT", root):
                resolved = runner.resolve_campaign_attempt_artifact(
                    report.relative_to(root).as_posix(),
                    paths=runner.runner_paths(campaign),
                )

            self.assertEqual(resolved, report.resolve())

    def test_sealed_access_remains_closed_after_evaluator_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _fixture(Path(tmp))

            def fake_run(command, cwd, check):
                del cwd, check
                if command[1].endswith("train_agentic_lora.py"):
                    _fake_training_output(command)
                else:
                    _fake_development_report(command)
                return _Completed(0)

            args = runner.parse_args(
                [
                    "--campaign-dir",
                    str(fixture["campaign"]),
                    "--campaign-id",
                    "crash-closed-test",
                    "--development-suite",
                    str(fixture["development_suite"]),
                    "--development-jsonl",
                    str(fixture["development_jsonl"]),
                    "--model-manifest",
                    str(fixture["model_manifest"]),
                    "--dataset-manifest",
                    str(fixture["dataset_manifest"]),
                    "--local-model-path",
                    str(fixture["local_model"]),
                    "--execute-local-training",
                    "--max-trials",
                    "1",
                ]
            )
            with mock.patch.object(runner.subprocess, "run", side_effect=fake_run):
                runner.run_campaign(args)

            finalize_args = runner.parse_args(
                [
                    "--campaign-dir",
                    str(fixture["campaign"]),
                    "--sealed-jsonl",
                    str(fixture["sealed_jsonl"]),
                    "--finalize-sealed",
                ]
            )
            with mock.patch.object(
                runner, "_run_subprocess", side_effect=RuntimeError("evaluator crashed")
            ):
                with self.assertRaisesRegex(RuntimeError, "evaluator crashed"):
                    runner.finalize_sealed(finalize_args)

            started = json.loads(
                (fixture["campaign"] / runner.SEALED_RECEIPT).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(started["status"], "started")
            with self.assertRaisesRegex(SystemExit, "already exists"):
                runner.finalize_sealed(finalize_args)

    def test_finalize_refuses_champion_that_did_not_pass_development(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _fixture(Path(tmp))

            def fake_run(command, cwd, check):
                del cwd, check
                if command[1].endswith("train_agentic_lora.py"):
                    _fake_training_output(command)
                else:
                    _fake_development_report(command, promotion_eligible=False, critical_failures=1)
                return _Completed(0)

            args = runner.parse_args(
                [
                    "--campaign-dir",
                    str(fixture["campaign"]),
                    "--development-suite",
                    str(fixture["development_suite"]),
                    "--development-jsonl",
                    str(fixture["development_jsonl"]),
                    "--model-manifest",
                    str(fixture["model_manifest"]),
                    "--dataset-manifest",
                    str(fixture["dataset_manifest"]),
                    "--local-model-path",
                    str(fixture["local_model"]),
                    "--execute-local-training",
                    "--max-trials",
                    "1",
                ]
            )
            with mock.patch.object(runner.subprocess, "run", side_effect=fake_run):
                runner.run_campaign(args)

            with self.assertRaisesRegex(SystemExit, "critical unsafe calls on development"):
                runner.finalize_sealed(
                    runner.parse_args(
                        [
                            "--campaign-dir",
                            str(fixture["campaign"]),
                            "--sealed-jsonl",
                            str(fixture["sealed_jsonl"]),
                            "--finalize-sealed",
                        ]
                    )
                )

    def test_finalize_rejects_tampered_champion_before_sealed_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _fixture(Path(tmp))

            def fake_run(command, cwd, check):
                del cwd, check
                if command[1].endswith("train_agentic_lora.py"):
                    _fake_training_output(command)
                else:
                    _fake_development_report(command)
                return _Completed(0)

            args = runner.parse_args(
                [
                    "--campaign-dir",
                    str(fixture["campaign"]),
                    "--development-suite",
                    str(fixture["development_suite"]),
                    "--development-jsonl",
                    str(fixture["development_jsonl"]),
                    "--model-manifest",
                    str(fixture["model_manifest"]),
                    "--dataset-manifest",
                    str(fixture["dataset_manifest"]),
                    "--local-model-path",
                    str(fixture["local_model"]),
                    "--execute-local-training",
                    "--max-trials",
                    "1",
                ]
            )
            with mock.patch.object(runner.subprocess, "run", side_effect=fake_run):
                runner.run_campaign(args)

            campaign_path = fixture["campaign"] / runner.CAMPAIGN_RECORD
            campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
            campaign["candidate_records"][0]["candidate"]["adapter_dir"] = str(
                fixture["campaign"] / "different-adapter"
            )
            runner.atomic_write_json(campaign_path, campaign)
            finalize_args = runner.parse_args(
                [
                    "--campaign-dir",
                    str(fixture["campaign"]),
                    "--sealed-jsonl",
                    str(fixture["sealed_jsonl"]),
                    "--finalize-sealed",
                ]
            )

            with self.assertRaisesRegex(SystemExit, "content hash mismatch"):
                runner.finalize_sealed(finalize_args)

            self.assertFalse((fixture["campaign"] / runner.SEALED_RECEIPT).exists())


class _Completed:
    def __init__(self, returncode: int) -> None:
        self.returncode = returncode


def inspect_signature_with_resume():
    import inspect

    def sample(*, plan_path, out_path, proposer, evaluator, resume=False):
        del plan_path, out_path, proposer, evaluator, resume

    return inspect.signature(sample)


def _fixture(root: Path, dev_jsonl_name: str = "development_rows.jsonl") -> dict[str, Path]:
    development_suite = root / "development_suite.json"
    development_jsonl = root / dev_jsonl_name
    sealed_jsonl = root / "sealed_rows.jsonl"
    campaign = root / "campaign"
    model_manifest = root / "model_manifest.json"
    dataset_manifest = root / "dataset_manifest.json"
    local_model = root / "local_model"
    local_model.mkdir()
    write_json(
        development_suite,
        {
            "schema_version": "hfr.eval_suite_manifest.v1",
            "suite_id": "runtime_adapter_development",
            "description": "Development-only runtime adapter selector.",
            "tags": ["development", "runtime_adapter"],
            "scenario_ids": ["dev-001"],
            "notes": ["No sealed inputs."],
        },
    )
    development_jsonl.write_text(json.dumps({"task_id": "dev-001"}) + "\n", encoding="utf-8")
    sealed_jsonl.write_text(json.dumps({"task_id": "sealed-001"}) + "\n", encoding="utf-8")
    write_json(
        model_manifest,
        {
            "schema_version": "hfr.model_candidate.v1",
            "model_id": runner.MODEL_ID,
            "source": {"revision": runner.MODEL_REVISION},
            "compatibility": {
                "tokenizer": {"revision": runner.TOKENIZER_REVISION},
                "chat_template": {"sha256": runner.CHAT_TEMPLATE_SHA256},
            },
        },
    )
    write_json(dataset_manifest, {"schema_version": "hfr.dataset_registry_entry.v1", "dataset_id": "fixture"})
    return {
        "development_suite": development_suite,
        "development_jsonl": development_jsonl,
        "sealed_jsonl": sealed_jsonl,
        "campaign": campaign,
        "model_manifest": model_manifest,
        "dataset_manifest": dataset_manifest,
        "local_model": local_model,
    }


def _fake_training_output(command: list[str]) -> None:
    output_dir = Path(command[command.index("--output-dir") + 1])
    mode = command[command.index("--mode") + 1]
    adapter_dir = runner.adapter_dir_for_recipe(output_dir, mode)
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_model.safetensors").write_text("fake adapter", encoding="utf-8")
    adapter_fingerprint = runner.adapter_directory_fingerprint(adapter_dir)
    write_json(
        output_dir / f"{mode}_result.json",
        {
            "status": "succeeded",
            "base_model": runner.MODEL_ID,
            "base_model_revision": runner.MODEL_REVISION,
            "adapter_artifacts": {"sha256": adapter_fingerprint["sha256"]},
        },
    )


def _fake_development_report(
    command: list[str],
    *,
    promotion_eligible: bool = True,
    critical_failures: int = 0,
) -> None:
    candidates = json.loads(Path(command[command.index("--candidates") + 1]).read_text(encoding="utf-8"))["candidates"]
    out = Path(command[command.index("--out") + 1])
    observations = Path(command[command.index("--observations-out") + 1])
    observations.write_text("", encoding="utf-8")
    evaluation_split = command[command.index("--evaluation-split") + 1]
    candidate = candidates[0]
    adapter = runner.adapter_directory_fingerprint(Path(candidate["adapter_dir"]))
    report = {
        "schema_version": "hfr.runtime_adapter_candidate_evaluation.v1",
        "created_at": "2026-07-21T00:00:00+00:00",
        "base_model": {"id": runner.MODEL_ID, "revision": runner.MODEL_REVISION},
        "tokenizer": {"id": runner.MODEL_ID, "revision": runner.TOKENIZER_REVISION},
        "chat_template": {"sha256": runner.CHAT_TEMPLATE_SHA256},
        "heldout": {"row_count": 1, "sha256": "dev", "split": evaluation_split},
        "passed": True,
        "candidate_count": 1,
        "promotion_eligible_candidates": [candidate["candidate_id"]] if promotion_eligible else [],
        "candidate_reports": [
            {
                "candidate_id": candidate["candidate_id"],
                "status": "evaluated",
                "passed": True,
                "promotion_eligible": promotion_eligible,
                "blocking_reasons": [] if promotion_eligible else ["development gate failed"],
                "identity": {
                    "candidate_id": candidate["candidate_id"],
                    "scope": "runtime_adapter_router",
                    "base_model": runner.MODEL_ID,
                    "base_revision": runner.MODEL_REVISION,
                    "tokenizer_revision": runner.TOKENIZER_REVISION,
                    "chat_template_sha256": runner.CHAT_TEMPLATE_SHA256,
                    "adapter": adapter,
                    "training_result": {
                        "path": candidate["training_result_path"],
                        "sha256": candidate["training_result_sha256"],
                    },
                },
                "heldout_subset": {"row_count": 1, "task_ids_sha256": "fake", "evaluation_scopes": ["*"]},
                "metrics": {
                    "overall": {"passed": 4, "total": 5, "pass_rate": 0.8},
                    "single_domain": {"passed": 4, "total": 5, "pass_rate": 0.8},
                    "cross_domain": {"passed": 0, "total": 0, "pass_rate": 0.0},
                    "safety": {"passed": 1, "total": 1, "pass_rate": 1.0},
                    "write_denial": {"passed": 1, "total": 1, "pass_rate": 1.0},
                    "failure_recovery": {"passed": 1, "total": 1, "pass_rate": 1.0},
                    "critical_safety_failures": critical_failures,
                },
                "thresholds": runner.DEFAULT_THRESHOLDS,
                "scores": [],
            }
        ],
    }
    report["evaluation_fingerprint"] = runner.canonical_sha256(report)
    write_json(out, report)


if __name__ == "__main__":
    unittest.main()
